import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from utils.sh_utils import eval_sh


def gaussian_rgb(view, gaussians):
    shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
    dirs = gaussians.get_xyz - view.camera_center.repeat(gaussians.get_features.shape[0], 1)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-6)
    rgb = eval_sh(gaussians.active_sh_degree, shs_view, dirs)
    return torch.clamp_min(rgb + 0.5, 0.0)


def render_color_passes(view, gaussians, pipeline, background, separate_sh, use_trained_exp, weight_threshold):
    rgb = gaussian_rgb(view, gaussians)
    black_background = torch.zeros_like(background)

    def feature_render(feature_n1):
        feature_rgb = feature_n1.expand(-1, 3).contiguous()
        return render(
            view,
            gaussians,
            pipeline,
            black_background,
            override_color=feature_rgb,
            separate_sh=separate_sh,
            use_trained_exp=use_trained_exp,
            clamp_output=False,
        )["render"][0]

    weight_map = feature_render(torch.ones((rgb.shape[0], 1), device=rgb.device, dtype=rgb.dtype))
    mask = weight_map > weight_threshold

    color_mean = []
    color_var = []
    color_std = []
    for channel in range(3):
        channel_values = rgb[:, channel : channel + 1]
        s1 = feature_render(channel_values)
        s2 = feature_render(channel_values ** 2)
        mean = torch.zeros_like(s1)
        second_moment = torch.zeros_like(s2)
        mean[mask] = s1[mask] / weight_map[mask]
        second_moment[mask] = s2[mask] / weight_map[mask]
        var = torch.zeros_like(s1)
        var[mask] = (second_moment[mask] - mean[mask] ** 2).clamp(min=0.0)
        std = var.sqrt()
        color_mean.append(mean)
        color_var.append(var)
        color_std.append(std)

    color_mean_rgb = torch.stack(color_mean, dim=0)
    color_var_rgb = torch.stack(color_var, dim=0)
    color_std_rgb = torch.stack(color_std, dim=0)
    scalar_color_var = color_var_rgb.mean(dim=0)
    scalar_color_std = scalar_color_var.sqrt()

    return {
        "color_mean_rgb": color_mean_rgb,
        "color_var_rgb": color_var_rgb,
        "color_std_rgb": color_std_rgb,
        "color_var": scalar_color_var,
        "color_std": scalar_color_std,
        "W": weight_map,
        "mask": mask,
    }


def main():
    args, dataset, pipeline = parse_args("Render color uncertainty quantities")
    split_map = load_split_manifest(args.split_dir or dataset.split_dir)

    with torch.no_grad():
        gaussians, scene, background = load_scene(dataset, args.iteration)
        selected_views = select_views(scene, split_map, args)
        run_name = run_name_from_model_path(dataset.model_path)

        for split_name, views in selected_views.items():
            out_dirs = build_output_dirs(args.output_root, run_name, split_name, scene.loaded_iter, "color")
            metadata = {
                "run_name": run_name,
                "split": split_name,
                "iteration": scene.loaded_iter,
                "model_path": dataset.model_path,
                "source_path": dataset.source_path,
                "split_dir": args.split_dir or dataset.split_dir,
                "weight_threshold": args.weight_threshold,
                "modality": "color",
                "view_names": [view.image_name for view in views],
            }

            for idx, view in enumerate(tqdm(views, desc=f"Color rendering [{split_name}]")):
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
                color_outputs = render_color_passes(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    SPARSE_ADAM_AVAILABLE,
                    use_trained_exp,
                    args.weight_threshold,
                )

                rendering, gt = apply_train_test_crop(view, render_pkg["render"], gt)
                cropped = apply_train_test_crop(
                    view,
                    color_outputs["color_mean_rgb"],
                    color_outputs["color_var_rgb"],
                    color_outputs["color_std_rgb"],
                    color_outputs["color_var"],
                    color_outputs["color_std"],
                    color_outputs["W"],
                    color_outputs["mask"].float(),
                )
                color_mean_rgb, color_var_rgb, color_std_rgb, color_var, color_std, weight_map, mask = cropped
                mask_bool = mask > 0.5

                stem = f"{idx:05d}"
                save_rgb_png_if_missing(gt, os.path.join(out_dirs["gt"], f"{stem}.png"))
                save_rgb_png_if_missing(rendering, os.path.join(out_dirs["render"], f"{stem}.png"))
                save_npz(
                    os.path.join(out_dirs["raw_sigma"], f"{stem}.npz"),
                    color_mean_rgb=color_mean_rgb.detach().cpu().numpy().astype(np.float32),
                    color_var_rgb=color_var_rgb.detach().cpu().numpy().astype(np.float32),
                    color_std_rgb=color_std_rgb.detach().cpu().numpy().astype(np.float32),
                    color_var=color_var.detach().cpu().numpy().astype(np.float32),
                    color_std=color_std.detach().cpu().numpy().astype(np.float32),
                    W=weight_map.detach().cpu().numpy().astype(np.float32),
                    mask=mask_bool.detach().cpu().numpy().astype(np.bool_),
                )
                save_preview_png(
                    color_std.detach().cpu().numpy(),
                    os.path.join(out_dirs["preview"], f"{stem}.png"),
                    mask=mask_bool.detach().cpu().numpy(),
                )

            metadata["num_views"] = len(views)
            metadata["raw_sigma_keys"] = ["color_mean_rgb", "color_var_rgb", "color_std_rgb", "color_var", "color_std", "W", "mask"]
            metadata["preview_key"] = "color_std"
            write_metadata(os.path.join(out_dirs["base"], "metadata.json"), metadata)


if __name__ == "__main__":
    main()
