"""
Conformal Uncertainty Maps Generator

Converts rendered images into conformal uncertainty maps using calibration-based
quantile thresholding. This script generates visualization of prediction intervals,
coverage maps, and normalized error maps.

Usage:
    python scripts/uncertainty_maps.py --scene <scene_name>

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

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate conformal uncertainty maps for a given scene"
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        choices=["drjohnson", "playroom", "train", "truck"],
        help="Name of the scene to process"
    )
    parser.add_argument(
        "--num-pixels",
        type=int,
        default=5000,
        help="Number of pixels to sample per calibration image (default: 5000)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Significance level; coverage = 1 - alpha (default: 0.1 for 90%% coverage)"
    )
    parser.add_argument(
        "--calib-ratio",
        type=float,
        default=0.5,
        help="Fraction of images to use for calibration (default: 0.5 for 50-50 split)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="uncertainty_maps",
        help="Output directory for visualizations (default: uncertainty_maps)"
    )
    return parser.parse_args()


def compute_non_conformity_score(error, sigma):
    """
    Compute non-conformity score: s_i = |error_i| / sigma_i
    
    Args:
        error: Per-pixel error values
        sigma: Estimated uncertainty (standard deviation)
        
    Returns:
        Non-conformity score
    """
    sigma = np.maximum(sigma, 1e-6)
    return error / sigma


def calibrate_threshold(test_dir, calibration_images, num_pixels_per_image, alpha):
    """
    Step 4 & 5: Calibrate the conformal prediction threshold
    
    Args:
        test_dir: Path to test directory
        calibration_images: List of calibration image filenames
        num_pixels_per_image: Number of pixels to sample per image
        alpha: Significance level
        
    Returns:
        threshold: Quantile threshold
        image_sigmas: Dictionary of sigma estimates per image
        calib_scores: Array of calibration non-conformity scores
    """
    calib_scores = []
    image_sigmas = {}
    
    print("\nCalibrating threshold using calibration set...")
    for img in calibration_images:
        render_path = os.path.join(test_dir, "renders", img)
        gt_path = os.path.join(test_dir, "gt", img)
        
        if not os.path.exists(render_path) or not os.path.exists(gt_path):
            print(f"  Warning: Skipping {img} - files not found")
            continue
        
        render = np.array(Image.open(render_path)).astype(np.float32) / 255.0
        gt = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
        
        # Handle grayscale images
        if len(render.shape) == 2:
            render = np.stack([render] * 3, axis=-1)
            gt = np.stack([gt] * 3, axis=-1)
        
        # Compute per-pixel errors
        errors = np.abs(render - gt)
        
        # Estimate sigma as the standard deviation of errors
        sigma_estimate = np.std(errors) if np.std(errors) > 0 else 0.01
        image_sigmas[img] = sigma_estimate
        
        # Sample pixels and compute non-conformity scores
        h, w = render.shape[:2]
        sampled_x = np.random.randint(0, w, num_pixels_per_image)
        sampled_y = np.random.randint(0, h, num_pixels_per_image)
        
        for x, y in zip(sampled_x, sampled_y):
            error = errors[y, x]
            if len(error.shape) > 0:
                error = np.mean(error)
            
            score = compute_non_conformity_score(error, sigma_estimate)
            calib_scores.append(score)
    
    calib_scores = np.array(calib_scores)
    if len(calib_scores) == 0:
        raise ValueError("No calibration data found. Check file paths.")
    
    print(f"  Total calibration samples: {len(calib_scores)}")
    print(f"  Non-conformity scores: min={calib_scores.min():.4f}, max={calib_scores.max():.4f}, mean={calib_scores.mean():.4f}")
    
    # Compute the quantile threshold
    q_level = 1 - alpha
    threshold = np.quantile(calib_scores, q_level)
    print(f"  Quantile threshold (q_{q_level}): {threshold:.4f}")
    
    return threshold, image_sigmas, calib_scores


def build_uncertainty_map(render_path, gt_path, sigma, threshold):
    """
    Build uncertainty map using conformal prediction.
    
    Args:
        render_path: Path to rendered image
        gt_path: Path to ground truth image
        sigma: Estimated uncertainty (standard deviation)
        threshold: Quantile threshold from calibration
        
    Returns:
        Tuple of (normalized_error, coverage_map, errors, interval_width)
    """
    render = np.array(Image.open(render_path)).astype(np.float32) / 255.0
    gt = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
    
    # Handle grayscale images
    if len(render.shape) == 2:
        render = np.stack([render] * 3, axis=-1)
        gt = np.stack([gt] * 3, axis=-1)
    
    # Compute per-pixel errors
    errors = np.abs(render - gt)
    if len(errors.shape) == 3:
        errors = np.mean(errors, axis=2)
    
    # Conformal prediction interval half-width
    interval_width = threshold * sigma
    
    # Normalized error: how much each pixel exceeds the interval
    normalized_error = errors / interval_width
    
    # Coverage: 1 if within interval, 0 if outside
    coverage_map = (errors <= interval_width).astype(float)
    
    return normalized_error, coverage_map, errors, interval_width


def visualize_test_images(test_dir, test_images, image_sigmas, threshold, output_path):
    """
    Visualize uncertainty maps for test images.
    
    Args:
        test_dir: Path to test directory
        test_images: List of test image filenames
        image_sigmas: Dictionary of sigma estimates
        threshold: Quantile threshold from calibration
        output_path: Path to save visualization
    """
    print(f"\nGenerating visualizations for {len(test_images)} test images...")
    
    fig, axes = plt.subplots(len(test_images), 5, figsize=(20, 4 * len(test_images)))
    if len(test_images) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img in enumerate(test_images):
        render_path = os.path.join(test_dir, "renders", img)
        gt_path = os.path.join(test_dir, "gt", img)
        
        if not os.path.exists(render_path) or not os.path.exists(gt_path):
            print(f"  Warning: Skipping {img} - files not found")
            continue
        
        # Compute sigma for this test image
        render = np.array(Image.open(render_path)).astype(np.float32) / 255.0
        gt = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
        
        if len(render.shape) == 2:
            render = np.stack([render] * 3, axis=-1)
        if len(gt.shape) == 2:
            gt = np.stack([gt] * 3, axis=-1)
        
        errors = np.abs(render - gt)
        if len(errors.shape) == 3:
            errors = np.mean(errors, axis=2)
        
        sigma = np.std(errors) if np.std(errors) > 0 else 0.01
        image_sigmas[img] = sigma
        
        normalized_error, coverage_map, error_map, interval_width = build_uncertainty_map(
            render_path, gt_path, sigma, threshold
        )
        
        # Display renders (already loaded above)
        
        # Render
        axes[idx, 0].imshow(np.clip(render, 0, 1))
        axes[idx, 0].set_title(f"Render ({img})")
        axes[idx, 0].axis('off')
        
        # Ground Truth
        axes[idx, 1].imshow(np.clip(gt, 0, 1))
        axes[idx, 1].set_title("Ground Truth")
        axes[idx, 1].axis('off')
        
        # Absolute Error Map
        im = axes[idx, 2].imshow(error_map, cmap='hot')
        axes[idx, 2].set_title("Absolute Error")
        axes[idx, 2].axis('off')
        plt.colorbar(im, ax=axes[idx, 2])
        
        # Coverage Map: green = within prediction interval, red = outside
        im = axes[idx, 3].imshow(coverage_map, cmap='RdYlGn')
        axes[idx, 3].set_title(f"Conformal Coverage\n(interval width={interval_width:.4f})")
        axes[idx, 3].axis('off')
        plt.colorbar(im, ax=axes[idx, 3], ticks=[0, 1], label='Outside|Inside')
        
        # Normalized Error: pixel error relative to prediction interval
        im = axes[idx, 4].imshow(np.clip(normalized_error, 0, 2), cmap='viridis')
        axes[idx, 4].set_title(f"Normalized Error\n(error/interval)")
        axes[idx, 4].axis('off')
        plt.colorbar(im, ax=axes[idx, 4])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"  Visualization saved to {output_path}")
    plt.close()


def compute_coverage_rate(test_dir, test_images, image_sigmas, threshold):
    """
    Compute actual coverage rate on test set.
    
    Args:
        test_dir: Path to test directory
        test_images: List of test image filenames
        image_sigmas: Dictionary of sigma estimates (populated during visualization)
        threshold: Quantile threshold from calibration
        
    Returns:
        Coverage rate as a percentage
    """
    coverage_vals = []
    
    for img in test_images:
        render_path = os.path.join(test_dir, "renders", img)
        gt_path = os.path.join(test_dir, "gt", img)
        
        if not os.path.exists(render_path) or not os.path.exists(gt_path):
            continue
        
        render = np.array(Image.open(render_path)).astype(np.float32) / 255.0
        gt = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
        
        if len(render.shape) == 3:
            errors = np.mean(np.abs(render - gt), axis=2)
        else:
            errors = np.abs(render - gt)
        
        sigma = image_sigmas.get(img)
        if sigma is None:
            sigma = np.std(errors) if np.std(errors) > 0 else 0.01
            image_sigmas[img] = sigma
        
        within_interval = errors <= (threshold * sigma)
        coverage_vals.append(np.mean(within_interval))
    
    return np.mean(coverage_vals) if coverage_vals else 0.0


def main():
    """Main entry point for the uncertainty maps generator."""
    args = parse_arguments()
    
    # Construct paths based on scene
    scene = args.scene
    test_dir = f"output/{scene}/test/ours_30000"
    
    # Validate that test directory exists
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        sys.exit(1)
    
    # Create scene-specific output directory
    scene_output_dir = os.path.join(args.output_dir, scene)
    os.makedirs(scene_output_dir, exist_ok=True)
    
    # Load available test images
    renders_dir = os.path.join(test_dir, "renders")
    if not os.path.exists(renders_dir):
        print(f"Error: Renders directory not found: {renders_dir}")
        sys.exit(1)
    
    all_images = sorted([f for f in os.listdir(renders_dir) if f.endswith(".png")])
    if not all_images:
        print(f"Error: No PNG images found in {renders_dir}")
        sys.exit(1)
    
    num_images = len(all_images)
    num_calib = int(num_images * args.calib_ratio)
    
    calibration_images = all_images[:num_calib]
    test_images = all_images[num_calib:]
    
    print(f"Processing scene: {scene}")
    print(f"Total images: {num_images}")
    print(f"Calibration images: {len(calibration_images)} ({args.calib_ratio * 100:.0f}%)")
    print(f"Test images: {len(test_images)} ({(1 - args.calib_ratio) * 100:.0f}%)")
    print(f"Target coverage: {(1 - args.alpha) * 100}%")
    
    # Calibrate threshold
    threshold, image_sigmas, calib_scores = calibrate_threshold(
        test_dir, calibration_images, args.num_pixels, args.alpha
    )
    
    # Visualize results
    output_filename = f"map_{scene}.png"
    output_path = os.path.join(scene_output_dir, output_filename)
    visualize_test_images(test_dir, test_images, image_sigmas, threshold, output_path)
    
    # Compute coverage rate
    coverage_rate = compute_coverage_rate(test_dir, test_images, image_sigmas, threshold)
    
    # Save metrics as JSON
    # Compute average sigma across all images (calibration + test)
    avg_sigma = np.mean(list(image_sigmas.values())) if image_sigmas else 0.01
    metrics = {
        "scene": scene,
        "target_coverage": (1 - args.alpha) * 100,
        "actual_coverage": coverage_rate * 100,
        "calibration_threshold": float(threshold),
        "prediction_interval_width": float(threshold * avg_sigma),
        "calibration_ratio": args.calib_ratio,
        "alpha": args.alpha,
        "num_calibration_images": len(calibration_images),
        "num_test_images": len(test_images),
        "num_pixels_per_image": args.num_pixels,
        "calibration_score_stats": {
            "mean": float(calib_scores.mean()),
            "std": float(calib_scores.std()),
            "min": float(calib_scores.min()),
            "max": float(calib_scores.max())
        }
    }
    
    metrics_json_path = os.path.join(scene_output_dir, "metrics.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    avg_sigma = np.mean(list(image_sigmas.values())) if image_sigmas else 0.01
    print("\n" + "=" * 60)
    print("Conformal Uncertainty Quantification Complete!")
    print("=" * 60)
    print(f"Scene: {scene}")
    print(f"Coverage target: {(1 - args.alpha) * 100}%")
    print(f"Actual coverage on test set: {coverage_rate * 100:.1f}%")
    print(f"Calibration threshold (q_{1 - args.alpha}): {threshold:.4f}")
    print(f"Average prediction interval width: {threshold * avg_sigma:.4f}")
    print(f"Output directory: {scene_output_dir}")
    print(f"Map saved to: {output_path}")
    print(f"Metrics saved to: {metrics_json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
