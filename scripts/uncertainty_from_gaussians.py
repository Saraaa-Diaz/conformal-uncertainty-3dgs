"""
Conformal Uncertainty from 3D Gaussian Point Cloud

Extract per-pixel uncertainty from the 3D Gaussian Splatting point cloud by computing
Gaussian visibility, opacity, and depth variance, then apply conformal prediction.

Usage:
    python scripts/uncertainty_from_gaussians.py --scene <scene_name>

Supported scenes: drjohnson, playroom, train, truck
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from scipy.special import expit

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate conformal uncertainty maps from 3D Gaussian projections"
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        choices=["drjohnson", "playroom", "train", "truck"],
        help="Name of the scene to process"
    )
    parser.add_argument(
        "--calib-ratio",
        type=float,
        default=0.5,
        help="Fraction of images to use for calibration (default: 0.5 for 50-50 split)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Significance level; coverage = 1 - alpha (default: 0.1 for 90%% coverage)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="uncertainty_maps",
        help="Base output directory (default: uncertainty_maps)"
    )
    return parser.parse_args()


def load_ply(filepath):
    """Load PLY file with Gaussian properties."""
    with open(filepath, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('ascii')
            header_lines.append(line)
            if line.startswith('end_header'):
                break
        
        num_vertices = 0
        properties = []
        for line in header_lines:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.strip().split()
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_name, prop_type))
        
        dtype_list = [(name, '<f4' if ptype == 'float' else '<f8') for name, ptype in properties]
        data = np.frombuffer(f.read(np.dtype(dtype_list).itemsize * num_vertices), 
                             dtype=dtype_list, count=num_vertices)
    
    return data, properties


def extract_gaussian_properties(data):
    """Extract xyz, opacity, and scale information from point cloud."""
    positions = np.stack([data['x'], data['y'], data['z']], axis=1)
    opacities = expit(data['opacity'])
    
    if 'scale_0' in data.dtype.names:
        scales = np.stack([data['scale_0'], data['scale_1'], data['scale_2']], axis=1)
        scales = np.exp(scales)
    else:
        scales = None
    
    print(f"Loaded {len(positions)} Gaussians")
    print(f"Opacities: [{opacities.min():.4f}, {opacities.max():.4f}]")
    
    return positions, opacities, scales


def parse_camera_params(camera_dict):
    """Parse camera parameters from JSON."""
    fx = camera_dict['fx']
    fy = camera_dict['fy']
    cx = camera_dict['width'] / 2.0
    cy = camera_dict['height'] / 2.0
    
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float32)
    
    R = np.array(camera_dict['rotation'], dtype=np.float32)
    t = np.array(camera_dict['position'], dtype=np.float32)
    
    return K, R, t


def project_gaussians_to_image(positions, opacities, scales, K, R, t, image_shape):
    """Project 3D Gaussians to 2D image space and compute per-pixel properties."""
    h, w = image_shape[:2]
    
    per_pixel_opacity = np.zeros((h, w), dtype=np.float32)
    per_pixel_depth_variance = np.zeros((h, w), dtype=np.float32)
    per_pixel_count = np.zeros((h, w), dtype=np.float32)
    per_pixel_scale = np.zeros((h, w), dtype=np.float32)
    per_pixel_depth_mean = np.zeros((h, w), dtype=np.float32)
    per_pixel_depth_sq = np.zeros((h, w), dtype=np.float32)
    
    # Transform to camera space
    positions_cam = (positions - t[np.newaxis, :]) @ R.T
    z_cam = positions_cam[:, 2]
    
    valid_depth = z_cam > 0.1
    valid_indices = np.where(valid_depth)[0]
    
    print(f"Valid Gaussians in view: {len(valid_indices)} / {len(positions)}")
    
    fx = K[0, 0]
    
    for idx in valid_indices:
        P_cam = positions_cam[idx]
        z = P_cam[2]
        
        x_img = (K[0, 0] * P_cam[0] + K[0, 2] * z) / z
        y_img = (K[1, 1] * P_cam[1] + K[1, 2] * z) / z
        
        opacity = opacities[idx]
        
        if scales is not None:
            scale_magnitude = np.linalg.norm(scales[idx])
        else:
            scale_magnitude = 0.01
        
        sigma_pixel = max(scale_magnitude * K[0, 0] / z, 0.5)
        radius = max(int(np.ceil(3 * sigma_pixel)), 1)
        
        x_min = max(0, int(np.floor(x_img - radius)))
        x_max = min(w, int(np.ceil(x_img + radius)) + 1)
        y_min = max(0, int(np.floor(y_img - radius)))
        y_max = min(h, int(np.ceil(y_img + radius)) + 1)
        
        if x_min >= x_max or y_min >= y_max:
            continue
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        dist_sq = (xx - x_img) ** 2 + (yy - y_img) ** 2
        weight = np.exp(-0.5 * dist_sq / (sigma_pixel ** 2 + 1e-6))
        
        per_pixel_count[y_min:y_max, x_min:x_max] += weight
        per_pixel_opacity[y_min:y_max, x_min:x_max] += opacity * weight
        per_pixel_depth_mean[y_min:y_max, x_min:x_max] += z * weight
        per_pixel_depth_sq[y_min:y_max, x_min:x_max] += (z ** 2) * weight
        per_pixel_scale[y_min:y_max, x_min:x_max] += scale_magnitude * weight
    
    valid_pixels = per_pixel_count > 0
    per_pixel_opacity[valid_pixels] /= per_pixel_count[valid_pixels]
    per_pixel_scale[valid_pixels] /= per_pixel_count[valid_pixels]
    per_pixel_depth_mean[valid_pixels] /= per_pixel_count[valid_pixels]
    per_pixel_depth_sq[valid_pixels] /= per_pixel_count[valid_pixels]
    
    per_pixel_depth_variance[valid_pixels] = (
        per_pixel_depth_sq[valid_pixels] - per_pixel_depth_mean[valid_pixels] ** 2
    )
    per_pixel_depth_variance = np.maximum(per_pixel_depth_variance, 0)
    
    max_variance = np.max(per_pixel_depth_variance[valid_pixels]) if np.any(valid_pixels) else 1.0
    if max_variance > 0:
        per_pixel_depth_variance[valid_pixels] /= max_variance
    
    print(f"Per-pixel statistics:")
    print(f"  Pixels with Gaussians: {np.sum(valid_pixels)}")
    if np.any(valid_pixels):
        print(f"  Opacity - mean: {per_pixel_opacity[valid_pixels].mean():.4f}")
        print(f"  Count - mean: {per_pixel_count[valid_pixels].mean():.2f}")
    
    return per_pixel_opacity, per_pixel_depth_variance, per_pixel_count, per_pixel_scale, per_pixel_depth_mean


def compute_sigma_from_projections(per_pixel_opacity, per_pixel_depth_variance, per_pixel_count):
    """Compute per-pixel uncertainty sigma from Gaussian projections."""
    sigma = np.zeros_like(per_pixel_opacity)
    valid_mask = per_pixel_count > 0
    
    opacity_uncertainty = 1.0 - per_pixel_opacity
    depth_uncertainty = per_pixel_depth_variance
    
    max_count = np.max(per_pixel_count[valid_mask]) if np.any(valid_mask) else 1.0
    count_uncertainty = 1.0 - (per_pixel_count / (max_count + 1e-6))
    count_uncertainty = np.clip(count_uncertainty, 0, 1)
    
    sigma[valid_mask] = (
        opacity_uncertainty[valid_mask] * 0.4 +
        depth_uncertainty[valid_mask] * 0.4 +
        count_uncertainty[valid_mask] * 0.2
    )
    sigma[~valid_mask] = 0.5
    sigma = np.maximum(sigma, 0.01)
    
    return sigma


def compute_calibration_scores(test_dir, calibration_images, sigma_map):
    """Compute non-conformity scores for calibration set."""
    calib_scores = []
    
    for img in calibration_images:
        render_path = os.path.join(test_dir, "renders", img)
        gt_path = os.path.join(test_dir, "gt", img)
        
        render = np.array(Image.open(render_path)).astype(np.float32) / 255.0
        gt = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
        
        if len(render.shape) == 3:
            errors = np.mean(np.abs(render - gt), axis=2)
        else:
            errors = np.abs(render - gt)
        
        sigma_map_safe = np.maximum(sigma_map, 1e-6)
        scores = errors / sigma_map_safe
        calib_scores.extend(scores.flatten())
    
    return np.array(calib_scores)


def visualize_test_images(test_dir, test_images, sigma_map, threshold, output_path):
    """Visualize uncertainty maps for test images."""
    print(f"Generating visualizations for {len(test_images)} test images...")
    
    fig, axes = plt.subplots(len(test_images), 6, figsize=(24, 4 * len(test_images)))
    if len(test_images) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img in enumerate(test_images):
        render_path = os.path.join(test_dir, "renders", img)
        gt_path = os.path.join(test_dir, "gt", img)
        
        render = np.array(Image.open(render_path)).astype(np.float32) / 255.0
        gt = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
        
        if len(render.shape) == 3:
            errors = np.mean(np.abs(render - gt), axis=2)
        else:
            errors = np.abs(render - gt)
        
        sigma_map_safe = np.maximum(sigma_map, 1e-6)
        interval_width = threshold * sigma_map_safe
        coverage = (errors <= interval_width).astype(float)
        
        # Render
        axes[idx, 0].imshow(np.clip(render, 0, 1))
        axes[idx, 0].set_title(f"Render ({img})")
        axes[idx, 0].axis('off')
        
        # Ground Truth
        axes[idx, 1].imshow(np.clip(gt, 0, 1))
        axes[idx, 1].set_title("Ground Truth")
        axes[idx, 1].axis('off')
        
        # Absolute Error
        im = axes[idx, 2].imshow(errors, cmap='hot')
        axes[idx, 2].set_title("Absolute Error")
        axes[idx, 2].axis('off')
        plt.colorbar(im, ax=axes[idx, 2])
        
        # Opacity (placeholder - could use actual projection map)
        im = axes[idx, 3].imshow(sigma_map, cmap='viridis')
        axes[idx, 3].set_title("Per-pixel Sigma\n(from Gaussians)")
        axes[idx, 3].axis('off')
        plt.colorbar(im, ax=axes[idx, 3])
        
        # Depth Variance (placeholder)
        im = axes[idx, 4].imshow(errors / np.maximum(sigma_map, 1e-6), cmap='plasma')
        axes[idx, 4].set_title("Normalized Error\n(error/sigma)")
        axes[idx, 4].axis('off')
        plt.colorbar(im, ax=axes[idx, 4])
        
        # Coverage Map
        im = axes[idx, 5].imshow(coverage, cmap='RdYlGn')
        axes[idx, 5].set_title(f"Conformal Coverage\n(threshold={threshold:.3f})")
        axes[idx, 5].axis('off')
        plt.colorbar(im, ax=axes[idx, 5], ticks=[0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    scene = args.scene
    cwd = os.getcwd()
    if not os.path.exists(os.path.join(cwd, "output")):
        cwd = os.path.dirname(cwd)
    
    project_root = cwd
    test_dir = os.path.join(project_root, f"output/{scene}/test/ours_30000")
    point_cloud_path = os.path.join(project_root, f"output/{scene}/point_cloud/iteration_30000/point_cloud.ply")
    cameras_path = os.path.join(project_root, f"output/{scene}/cameras.json")
    
    # Create output directories
    scene_output_dir = os.path.join(args.output_dir, scene)
    os.makedirs(scene_output_dir, exist_ok=True)
    
    # Validate paths
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        sys.exit(1)
    
    renders_dir = os.path.join(test_dir, "renders")
    if not os.path.exists(renders_dir):
        print(f"Error: Renders directory not found: {renders_dir}")
        sys.exit(1)
    
    # Load data
    print(f"Processing scene: {scene}")
    gaussian_data, _ = load_ply(point_cloud_path)
    positions, opacities, scales = extract_gaussian_properties(gaussian_data)
    
    with open(cameras_path) as f:
        cameras_data = json.load(f)
    
    # Split images
    all_images = sorted([f for f in os.listdir(renders_dir) if f.endswith(".png")])
    num_images = len(all_images)
    num_calib = int(num_images * args.calib_ratio)
    
    calibration_images = all_images[:num_calib]
    test_images = all_images[num_calib:]
    
    print(f"Total images: {num_images}")
    print(f"Calibration images: {len(calibration_images)} ({args.calib_ratio * 100:.0f}%)")
    print(f"Test images: {len(test_images)} ({(1 - args.calib_ratio) * 100:.0f}%)")
    
    # Project Gaussians
    K, R, t = parse_camera_params(cameras_data[0])
    sample_render = np.array(Image.open(os.path.join(test_dir, "renders", calibration_images[0])))
    sample_opacity, sample_depth_var, sample_count, sample_scale, _ = project_gaussians_to_image(
        positions, opacities, scales, K, R, t, sample_render.shape
    )
    
    # Compute sigma
    sample_sigma_map = compute_sigma_from_projections(sample_opacity, sample_depth_var, sample_count)
    
    # Compute calibration scores
    calib_scores = compute_calibration_scores(test_dir, calibration_images, sample_sigma_map)
    
    # Compute threshold
    q_level = 1 - args.alpha
    threshold = np.quantile(calib_scores, q_level)
    
    # Compute coverage
    coverage_rates = []
    for img in test_images:
        render_path = os.path.join(test_dir, "renders", img)
        gt_path = os.path.join(test_dir, "gt", img)
        
        render = np.array(Image.open(render_path)).astype(np.float32) / 255.0
        gt = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
        
        if len(render.shape) == 3:
            errors = np.mean(np.abs(render - gt), axis=2)
        else:
            errors = np.abs(render - gt)
        
        sigma_map_safe = np.maximum(sample_sigma_map, 1e-6)
        within_interval = errors <= (threshold * sigma_map_safe)
        coverage_rates.append(np.mean(within_interval))
    
    avg_coverage = np.mean(coverage_rates) if coverage_rates else 0.0
    
    # Visualize
    viz_path = os.path.join(scene_output_dir, f"maps_from_gaussians_{scene}.png")
    visualize_test_images(test_dir, test_images, sample_sigma_map, threshold, viz_path)
    
    # Save metrics
    metrics = {
        "scene": scene,
        "target_coverage": (1 - args.alpha) * 100,
        "actual_coverage": avg_coverage * 100,
        "calibration_threshold": float(threshold),
        "calibration_ratio": args.calib_ratio,
        "alpha": args.alpha,
        "num_calibration_images": len(calibration_images),
        "num_test_images": len(test_images),
        "calibration_score_stats": {
            "mean": float(calib_scores.mean()),
            "std": float(calib_scores.std()),
            "min": float(calib_scores.min()),
            "max": float(calib_scores.max())
        },
        "sigma_map_stats": {
            "mean": float(sample_sigma_map.mean()),
            "std": float(sample_sigma_map.std()),
            "min": float(sample_sigma_map.min()),
            "max": float(sample_sigma_map.max())
        },
        "coverage_per_test_image": {test_images[i]: float(coverage_rates[i]) for i in range(len(test_images))}
    }
    
    # Save as JSON
    metrics_json_path = os.path.join(scene_output_dir, "metrics_from_gaussians.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_json_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONFORMAL UNCERTAINTY FROM 3D GAUSSIANS")
    print("=" * 60)
    print(f"Scene: {scene}")
    print(f"Target coverage: {(1 - args.alpha) * 100:.1f}%")
    print(f"Actual coverage: {avg_coverage * 100:.1f}%")
    print(f"Calibration threshold: {threshold:.4f}")
    print(f"Output directory: {scene_output_dir}")
    print(f"Map saved to: {viz_path}")
    print(f"Metrics saved to: {metrics_json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
