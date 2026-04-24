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


def render_depth_passes(view, gaussians, pipeline, background, separate_sh, use_trained_exp, weight_threshold):
    xyz_world = gaussians.get_xyz
    wvt = view.world_view_transform
    z = (xyz_world @ wvt[:3, :3] + wvt[3, :3])[..., 2:3]
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

    s1 = feature_render(z)
    s2 = feature_render(z ** 2)
    inv_z = 1.0 / z.clamp_min(1e-6)
    s1_inv = feature_render(inv_z)
    s2_inv = feature_render(inv_z ** 2)
    weight_map = feature_render(torch.ones_like(z))

    mask = weight_map > weight_threshold

    ez = torch.zeros_like(s1)
    ez2 = torch.zeros_like(s2)
    ez[mask] = s1[mask] / weight_map[mask]
    ez2[mask] = s2[mask] / weight_map[mask]
    depth_var = torch.zeros_like(s1)
    depth_var[mask] = (ez2[mask] - ez[mask] ** 2).clamp(min=0.0)
    depth_std = depth_var.sqrt()

    e_inv = torch.zeros_like(s1_inv)
    e_inv2 = torch.zeros_like(s2_inv)
    e_inv[mask] = s1_inv[mask] / weight_map[mask]
    e_inv2[mask] = s2_inv[mask] / weight_map[mask]
    invdepth_var = torch.zeros_like(s1_inv)
    invdepth_var[mask] = (e_inv2[mask] - e_inv[mask] ** 2).clamp(min=0.0)
    invdepth_std = invdepth_var.sqrt()

    return {
        "Ez": ez,
        "depth_var": depth_var,
        "depth_std": depth_std,
        "E_inv": e_inv,
        "invdepth_var": invdepth_var,
        "invdepth_std": invdepth_std,
        "W": weight_map,
        "mask": mask,
    }


def main():
    args, dataset, pipeline = parse_args("Render depth/disparity uncertainty quantities")
    split_map = load_split_manifest(args.split_dir or dataset.split_dir)

    with torch.no_grad():
        gaussians, scene, background = load_scene(dataset, args.iteration)
        selected_views = select_views(scene, split_map, args)
        run_name = run_name_from_model_path(dataset.model_path)

        for split_name, views in selected_views.items():
            out_dirs = build_output_dirs(args.output_root, run_name, split_name, scene.loaded_iter, "depth")
            metadata = {
                "run_name": run_name,
                "split": split_name,
                "iteration": scene.loaded_iter,
                "model_path": dataset.model_path,
                "source_path": dataset.source_path,
                "split_dir": args.split_dir or dataset.split_dir,
                "weight_threshold": args.weight_threshold,
                "modality": "depth",
                "view_names": [view.image_name for view in views],
            }

            for idx, view in enumerate(tqdm(views, desc=f"Depth rendering [{split_name}]")):
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
                depth_outputs = render_depth_passes(
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
                    depth_outputs["Ez"],
                    depth_outputs["depth_var"],
                    depth_outputs["depth_std"],
                    depth_outputs["E_inv"],
                    depth_outputs["invdepth_var"],
                    depth_outputs["invdepth_std"],
                    depth_outputs["W"],
                    depth_outputs["mask"].float(),
                )
                ez, depth_var, depth_std, e_inv, invdepth_var, invdepth_std, weight_map, mask = cropped
                mask_bool = mask > 0.5

                stem = f"{idx:05d}"
                save_rgb_png_if_missing(gt, os.path.join(out_dirs["gt"], f"{stem}.png"))
                save_rgb_png_if_missing(rendering, os.path.join(out_dirs["render"], f"{stem}.png"))
                save_npz(
                    os.path.join(out_dirs["raw_sigma"], f"{stem}.npz"),
                    Ez=ez.detach().cpu().numpy().astype(np.float32),
                    depth_var=depth_var.detach().cpu().numpy().astype(np.float32),
                    depth_std=depth_std.detach().cpu().numpy().astype(np.float32),
                    E_inv=e_inv.detach().cpu().numpy().astype(np.float32),
                    invdepth_var=invdepth_var.detach().cpu().numpy().astype(np.float32),
                    invdepth_std=invdepth_std.detach().cpu().numpy().astype(np.float32),
                    W=weight_map.detach().cpu().numpy().astype(np.float32),
                    mask=mask_bool.detach().cpu().numpy().astype(np.bool_),
                )
                save_preview_png(
                    depth_std.detach().cpu().numpy(),
                    os.path.join(out_dirs["preview"], f"{stem}.png"),
                    mask=mask_bool.detach().cpu().numpy(),
                )

            metadata["num_views"] = len(views)
            metadata["raw_sigma_keys"] = ["Ez", "depth_var", "depth_std", "E_inv", "invdepth_var", "invdepth_std", "W", "mask"]
            metadata["preview_key"] = "depth_std"
            write_metadata(os.path.join(out_dirs["base"], "metadata.json"), metadata)


if __name__ == "__main__":
    main()
