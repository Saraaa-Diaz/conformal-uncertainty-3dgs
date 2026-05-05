"""
Entropy of top-K alpha-blending weights (k=4 by default).

For each pixel, the rasterizer's alpha-blending traversal yields a sequence of
weights w_i = alpha_i * T_i. Most contribute very little; a few dominate. We
keep the K largest, normalize them to a discrete probability p_k = w_k / sum(w_k),
and report the Shannon entropy H = -sum(p_k log p_k). Low H = one Gaussian
explains the pixel (confident); high H = multiple Gaussians compete (uncertain).

Requires the rasterizer extension built with the top-K patch (provides
_C.rasterize_gaussians_topk via GaussianRasterizer.rasterize_topk_weights).
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_renderer import render
from scripts.render_common import (
    SPARSE_ADAM_AVAILABLE,
    apply_train_test_crop,
    build_output_dirs,
    load_scene,
    load_split_manifest,
    parse_args,
    run_name_from_model_path,
    safe_use_trained_exp,
    save_npz,
    save_preview_png,
    save_rgb_png_if_missing,
    select_views,
    write_metadata,
)


def build_rasterizer(view, gaussians, pipeline, background):
    tanfovx = math.tan(view.FoVx * 0.5)
    tanfovy = math.tan(view.FoVy * 0.5)
    settings = GaussianRasterizationSettings(
        image_height=int(view.image_height),
        image_width=int(view.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=1.0,
        viewmatrix=view.world_view_transform,
        projmatrix=view.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=view.camera_center,
        prefiltered=False,
        debug=pipeline.debug,
        antialiasing=pipeline.antialiasing,
    )
    return GaussianRasterizer(raster_settings=settings)


def render_topk_passes(view, gaussians, pipeline, background, top_k, weight_threshold):
    rasterizer = build_rasterizer(view, gaussians, pipeline, background)
    top_weights, _ = rasterizer.rasterize_topk_weights(
        means3D=gaussians.get_xyz,
        opacities=gaussians.get_opacity,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
        top_k=top_k,
    )
    # top_weights: [K, H, W] sorted descending
    weight_sum = top_weights.sum(dim=0)
    mask = weight_sum > weight_threshold

    eps = 1e-8
    probs = top_weights / weight_sum.clamp_min(eps).unsqueeze(0)
    log_probs = torch.where(probs > 0, torch.log(probs.clamp_min(eps)), torch.zeros_like(probs))
    entropy = -(probs * log_probs).sum(dim=0)
    entropy = torch.where(mask, entropy, torch.zeros_like(entropy))

    n_eff = torch.where(mask, torch.exp(entropy), torch.zeros_like(entropy))

    return {
        "top_weights": top_weights,
        "weight_sum": weight_sum,
        "entropy": entropy,
        "n_eff": n_eff,
        "mask": mask,
    }


def main():
    args, dataset, pipeline = parse_args("Render entropy of top-K blending weights")
    top_k = int(args.top_k)
    split_map = load_split_manifest(args.split_dir or dataset.split_dir)

    with torch.no_grad():
        gaussians, scene, background = load_scene(dataset, args.iteration)
        selected_views = select_views(scene, split_map, args)
        run_name = run_name_from_model_path(dataset.model_path)

        for split_name, views in selected_views.items():
            out_dirs = build_output_dirs(args.output_root, run_name, split_name, scene.loaded_iter, "entropy")
            metadata = {
                "run_name": run_name,
                "split": split_name,
                "iteration": scene.loaded_iter,
                "model_path": dataset.model_path,
                "source_path": dataset.source_path,
                "split_dir": args.split_dir or dataset.split_dir,
                "weight_threshold": args.weight_threshold,
                "top_k": top_k,
                "modality": "entropy",
                "view_names": [view.image_name for view in views],
            }

            for idx, view in enumerate(tqdm(views, desc=f"Entropy rendering [{split_name}]")):
                use_trained_exp = safe_use_trained_exp(dataset, gaussians, view.image_name)
                render_pkg = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    separate_sh=SPARSE_ADAM_AVAILABLE,
                    use_trained_exp=use_trained_exp,
                )
                gt = view.original_image[0:3, :, :]
                topk_outputs = render_topk_passes(view, gaussians, pipeline, background, top_k, args.weight_threshold)

                rendering, gt = apply_train_test_crop(view, render_pkg["render"], gt)
                cropped = apply_train_test_crop(
                    view,
                    topk_outputs["top_weights"],
                    topk_outputs["weight_sum"],
                    topk_outputs["entropy"],
                    topk_outputs["n_eff"],
                    topk_outputs["mask"].float(),
                )
                top_weights, weight_sum, entropy, n_eff, mask = cropped
                mask_bool = mask > 0.5

                stem = f"{idx:05d}"
                save_rgb_png_if_missing(gt, os.path.join(out_dirs["gt"], f"{stem}.png"))
                save_rgb_png_if_missing(rendering, os.path.join(out_dirs["render"], f"{stem}.png"))
                save_npz(
                    os.path.join(out_dirs["raw_sigma"], f"{stem}.npz"),
                    top_weights=top_weights.detach().cpu().numpy().astype(np.float32),
                    weight_sum=weight_sum.detach().cpu().numpy().astype(np.float32),
                    entropy=entropy.detach().cpu().numpy().astype(np.float32),
                    n_eff=n_eff.detach().cpu().numpy().astype(np.float32),
                    mask=mask_bool.detach().cpu().numpy().astype(np.bool_),
                )
                save_preview_png(
                    entropy.detach().cpu().numpy(),
                    os.path.join(out_dirs["preview"], f"{stem}.png"),
                    mask=mask_bool.detach().cpu().numpy(),
                )

            metadata["num_views"] = len(views)
            metadata["raw_sigma_keys"] = ["top_weights", "weight_sum", "entropy", "n_eff", "mask"]
            metadata["preview_key"] = "entropy"
            write_metadata(os.path.join(out_dirs["base"], "metadata.json"), metadata)


if __name__ == "__main__":
    main()
