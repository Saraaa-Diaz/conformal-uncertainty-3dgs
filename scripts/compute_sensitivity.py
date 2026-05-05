"""
Per-Gaussian Fisher sensitivity (PUP 3D-GS, Hanson et al.).

This implementation matches the PUP3DGS recipe at
  https://github.com/j-alex-hanson/gaussian-splatting-pup/blob/e971ea4802908c69eab7e8601bb1eddd2e443754/prune_finetune.py#L199
modulo their custom CUDA kernel for per-pixel Fisher pooling. We use the
standard *outer-product* Fisher approximation, summing rank-1 contributions
across training views.

Per Gaussian i, parameters theta_i = [xyz_i (3), scaling_i (3)] in R^6:
  F_i = sum_{view} g_i^v (g_i^v)^T   with  g_i^v = grad_{theta_i} L(view)
  fishers_log_dets[i] = sum_j log(svd(F_i)_j + lambda)

Note: pure-rank-1 per view; with V > 6 training views and >100 typical, the
sum is generically full rank, matching the paper's setup.

Output: <model_path>/uncertainty/fishers.npz
  - fishers           : (P, 6, 6)
  - fishers_sv        : (P, 6) singular values
  - fishers_log_dets  : (P,)    sensitivity score
  - num_train_views   : int
  - param_names       : ["xyz", "scaling"]
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.loss_utils import l1_loss, ssim
from scripts.render_common import (
    SPARSE_ADAM_AVAILABLE,
    apply_train_test_crop,
    safe_use_trained_exp,
)


def parse_args():
    parser = ArgumentParser(description="Compute per-Gaussian PUP3DGS Fisher sensitivity")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--lambda_reg", type=float, default=1e-6, help="Tikhonov term added to singular values before log")
    parser.add_argument("--max_views", type=int, default=-1, help="Cap number of training views (debug); -1 = all")
    parser.add_argument("--output_name", type=str, default="fishers.npz")
    parser.add_argument("--save_full_fishers", action="store_true", help="Also save the (P, 6, 6) tensor (large)")
    args = get_combined_args(parser)
    return args, model.extract(args), pipeline.extract(args)


def main():
    args, dataset, pipeline = parse_args()

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_views = scene.getTrainCameras()
    if args.max_views > 0:
        train_views = train_views[: args.max_views]
    if not train_views:
        raise RuntimeError("No training views available; cannot compute sensitivity.")

    P = gaussians.get_xyz.shape[0]
    print(f"Computing PUP3DGS Fisher (xyz + scaling = 6 params) over {len(train_views)} training views, P = {P}")

    # Need grads on xyz + scaling only; freeze the rest to save memory.
    for name in ("_features_dc", "_features_rest", "_opacity", "_rotation"):
        getattr(gaussians, name).requires_grad_(False)
    gaussians._xyz.requires_grad_(True)
    gaussians._scaling.requires_grad_(True)

    fishers = torch.zeros(P, 6, 6, device="cuda", dtype=torch.float32)

    for view in tqdm(train_views, desc="Fisher (xyz+scaling)"):
        if gaussians._xyz.grad is not None:
            gaussians._xyz.grad.zero_()
        if gaussians._scaling.grad is not None:
            gaussians._scaling.grad.zero_()

        use_trained_exp = safe_use_trained_exp(dataset, gaussians, view.image_name)
        render_pkg = render(
            view,
            gaussians,
            pipeline,
            background,
            separate_sh=SPARSE_ADAM_AVAILABLE,
            use_trained_exp=use_trained_exp,
        )
        image = render_pkg["render"]
        gt = view.original_image[0:3, :, :].cuda()
        image, gt = apply_train_test_crop(view, image, gt)
        l1 = l1_loss(image, gt)
        ssim_val = ssim(image, gt)
        loss = (1.0 - args.lambda_dssim) * l1 + args.lambda_dssim * (1.0 - ssim_val)
        loss.backward()

        # Concatenate grads: g[i] = [grad_xyz_i, grad_scaling_i] in R^6.
        g = torch.cat(
            [gaussians._xyz.grad.detach(), gaussians._scaling.grad.detach()],
            dim=1,
        )  # (P, 6)
        # Rank-1 outer product per Gaussian, accumulate.
        fishers.add_(g.unsqueeze(2) * g.unsqueeze(1))

    # log-det via singular values: log_det(F) = sum_j log(sv_j).
    # Add lambda_reg to stabilize (PUP3DGS does the same).
    fishers_sv = torch.linalg.svdvals(fishers)  # (P, 6), sorted descending
    fishers_log_dets = torch.log(fishers_sv + args.lambda_reg).sum(dim=1)  # (P,)

    out_dir = Path(dataset.model_path) / "uncertainty"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name

    save_kwargs = {
        "fishers_sv": fishers_sv.detach().cpu().numpy().astype(np.float32),
        "fishers_log_dets": fishers_log_dets.detach().cpu().numpy().astype(np.float32),
        "num_train_views": len(train_views),
        "lambda_reg": args.lambda_reg,
        "lambda_dssim": args.lambda_dssim,
        "param_names": np.array(["xyz", "scaling"]),
    }
    if args.save_full_fishers:
        save_kwargs["fishers"] = fishers.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(str(out_path), **save_kwargs)

    print(f"Saved Fisher to: {out_path}")
    print(f"  fishers_sv         : {tuple(fishers_sv.shape)}")
    print(f"  fishers_log_dets   : {tuple(fishers_log_dets.shape)}")
    print(f"  log-det range      : [{float(fishers_log_dets.min()):.4f}, {float(fishers_log_dets.max()):.4f}]")
    print(f"  log-det mean / std : {float(fishers_log_dets.mean()):.4f} / {float(fishers_log_dets.std()):.4f}")


if __name__ == "__main__":
    main()
