"""
Per-Gaussian visibility / informativeness, inspired by

    "4D Gaussian Splatting in the Wild with Uncertainty-Aware Regularization"
    (arXiv 2411.08879)

The score for Gaussian i is its total contribution mass across all training
pixels and views:

    visibility[i] = sum_{view v, pixel p} w_i(v, p) = sum_{v, p} alpha_i(v, p) * T_i(v, p)

Computed without any CUDA modification using an autograd trick:
  - Set override_color = s_i (a leaf tensor of ones with requires_grad).
  - Render. The image equals sum_i w_i(p) * s_i, by the rasterizer's
    contract.
  - Sum of pixel intensities s_view = sum_p sum_i w_i(p) * s_i = sum_i s_i * sum_p w_i(p).
  - ds_view / ds_i = sum_p w_i(p) -> exactly the per-pixel sum of weights for
    Gaussian i in this view.
  - Accumulate across views.

Output: <model_path>/uncertainty/visibility.npz
  - visibility       : (P,) total sum of w_i across all training pixels and views
  - visibility_log   : (P,) log(visibility + lambda)  (often more useful as a sigma)
  - visibility_norm  : (P,) per-Gaussian visibility / num_train_views (per-view avg)
  - uncertainty      : (P,) 1 - sigmoid(C_k), low visibility => high uncertainty
  - uncertainty_log  : (P,) log-compressed uncertainty for downstream rendering
  - num_train_views  : int
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
from scripts.render_common import (
    SPARSE_ADAM_AVAILABLE,
    apply_train_test_crop,
    safe_use_trained_exp,
)


def parse_args():
    parser = ArgumentParser(description="Compute per-Gaussian visibility (4DGS-W style)")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--lambda_reg", type=float, default=1e-6, help="Tikhonov term added before log")
    parser.add_argument("--log_compression_eps", type=float, default=1e-3, help="Scale used by log1p(uncertainty / eps)")
    parser.add_argument("--max_views", type=int, default=-1)
    parser.add_argument("--output_name", type=str, default="visibility.npz")
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
        raise RuntimeError("No training views available; cannot compute visibility.")

    P = gaussians.get_xyz.shape[0]
    print(f"Computing visibility over {len(train_views)} training views, P = {P}")

    # We freeze the Gaussian parameters and use a single per-Gaussian scalar
    # parameter as the autograd hook to read sum of weights.
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        getattr(gaussians, name).requires_grad_(False)

    visibility = torch.zeros(P, device="cuda", dtype=torch.float32)

    # We render with an *identity-style* override_color = ones, exposing one
    # scalar per Gaussian to autograd. sum-of-pixels has gradient = sum of
    # blending weights for that Gaussian in this view.
    for view in tqdm(train_views, desc="Visibility"):
        score = torch.ones(P, 1, device="cuda", dtype=torch.float32, requires_grad=True)
        score_rgb = score.expand(-1, 3).contiguous()
        use_trained_exp = safe_use_trained_exp(dataset, gaussians, view.image_name)
        render_pkg = render(
            view,
            gaussians,
            pipeline,
            torch.zeros_like(background),  # black bg so bg doesn't contaminate the sum
            override_color=score_rgb,
            separate_sh=SPARSE_ADAM_AVAILABLE,
            use_trained_exp=use_trained_exp,
            clamp_output=False,
        )
        rendered = render_pkg["render"]
        # rendered[ch, p] = sum_i w_i(p) * score[i] ; all 3 channels identical because score is broadcast.
        pixel_sum = rendered[0].sum()
        pixel_sum.backward()
        visibility += score.grad.detach().reshape(-1)

    visibility_norm = visibility / float(len(train_views))
    visibility_log = torch.log(visibility + args.lambda_reg)

    # Paper's uncertainty (4DGS-W eq. 8): U_k = 1 - Sigmoid(C_k; c0, c1)
    # so that low visibility => high uncertainty. Fit c0 to the median, c1 to a
    # robust spread (IQR/2) of the visibility distribution.
    vis_np = visibility.detach().cpu().numpy()
    c0 = float(np.median(vis_np))
    c1 = max(float(np.subtract(*np.percentile(vis_np, [75, 25])) / 2.0), args.lambda_reg)
    sig = 1.0 / (1.0 + np.exp(-(vis_np - c0) / c1))
    uncertainty = (1.0 - sig).astype(np.float32)
    uncertainty_log = np.log1p(uncertainty / args.log_compression_eps).astype(np.float32)
    inv_visibility = (1.0 / (vis_np + args.lambda_reg)).astype(np.float32)

    out_dir = Path(dataset.model_path) / "uncertainty"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name

    np.savez_compressed(
        str(out_path),
        visibility=visibility.detach().cpu().numpy().astype(np.float32),
        visibility_log=visibility_log.detach().cpu().numpy().astype(np.float32),
        visibility_norm=visibility_norm.detach().cpu().numpy().astype(np.float32),
        uncertainty=uncertainty,
        uncertainty_log=uncertainty_log,
        visibility_uncertainty_log=uncertainty_log,
        inv_visibility=inv_visibility,
        sigmoid_c0=np.array(c0, dtype=np.float32),
        sigmoid_c1=np.array(c1, dtype=np.float32),
        log_compression="log1p(uncertainty / log_compression_eps)",
        log_compression_eps=np.array(args.log_compression_eps, dtype=np.float32),
        num_train_views=len(train_views),
        lambda_reg=args.lambda_reg,
    )

    print(f"Saved visibility to: {out_path}")
    print(f"  visibility range : [{float(visibility.min()):.4f}, {float(visibility.max()):.4f}]")
    print(f"  log range        : [{float(visibility_log.min()):.4f}, {float(visibility_log.max()):.4f}]")
    print(f"  uncertainty_log  : [{float(uncertainty_log.min()):.4f}, {float(uncertainty_log.max()):.4f}]")
    print(f"  mean / std       : {float(visibility.mean()):.4f} / {float(visibility.std()):.4f}")


if __name__ == "__main__":
    main()
