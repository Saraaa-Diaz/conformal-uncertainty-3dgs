"""
Per-view quality prediction from conformal uncertainty.

Research question: Can the mean conformal uncertainty of a view predict
how bad the render will be — before having ground truth?

For each test view and each sigma modality, loads the saved conformal arrays,
computes per-view mean absolute error and mean uncertainty width, then checks
whether uncertainty correlates with error across views (rather than within them).

This is different from the per-pixel Pearson r already computed in conformal_prediction.py:
  - Per-pixel r: within one view, do uncertain pixels have higher error?
  - Per-view r (this script): across views, do more uncertain views have higher mean error?

Usage:
  python scripts/predict_quality.py --run_dir output/my_run [--iteration -1] [--alpha 0.1]

Outputs (written to <run_dir>/quality_prediction/):
  quality_prediction.json   per-view stats + per-modality Pearson r
  quality_prediction.png    scatter plots, one panel per modality
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


MODALITIES = [
    ("color", "color_std"),
    ("depth", "invdepth_std"),
]


def find_iteration(run_dir: Path, iteration: int) -> int:
    if iteration != -1:
        return iteration
    candidates = []
    for split_name in ("calib", "test"):
        split_dir = run_dir / split_name
        if not split_dir.exists():
            continue
        for child in split_dir.iterdir():
            if child.is_dir() and child.name.startswith("ours_"):
                try:
                    candidates.append(int(child.name.split("_", 1)[1]))
                except ValueError:
                    continue
    if not candidates:
        raise FileNotFoundError(f"No iteration directories found under {run_dir}")
    return max(candidates)


def load_per_view_stats(arrays_dir: Path) -> list[dict]:
    """Load per-view mean absolute error and mean uncertainty from saved .npz files."""
    records = []
    for npz_path in sorted(arrays_dir.glob("*.npz")):
        d = np.load(npz_path)
        # abs_error is (H,W) mean or max across channels
        abs_error = d["abs_error"].astype(np.float32)
        uncertainty = d["uncertainty_full_width"].astype(np.float32)
        mask = d["mask"].astype(bool) if "mask" in d else np.ones_like(abs_error, dtype=bool)

        valid = mask & np.isfinite(abs_error) & np.isfinite(uncertainty)
        if not np.any(valid):
            continue

        records.append({
            "frame": npz_path.stem,
            "mean_abs_error": float(abs_error[valid].mean()),
            "mean_uncertainty": float(uncertainty[valid].mean()),
            "mean_coverage": float(d["within_interval"].astype(float)[valid].mean()) if "within_interval" in d else None,
            "n_pixels": int(valid.sum()),
        })
    return records


def compute_pearson(records: list[dict]) -> tuple[float, float]:
    errors = np.array([r["mean_abs_error"] for r in records])
    uncertainties = np.array([r["mean_uncertainty"] for r in records])
    if len(errors) < 3 or errors.std() < 1e-9 or uncertainties.std() < 1e-9:
        return float("nan"), float("nan")
    r, p = pearsonr(errors, uncertainties)
    return float(r), float(p)


def plot_scatter(ax, records: list[dict], modality: str, sigma_key: str, r: float, p: float):
    errors = np.array([r_["mean_abs_error"] for r_ in records])
    uncertainties = np.array([r_["mean_uncertainty"] for r_ in records])

    ax.scatter(uncertainties, errors, s=60, alpha=0.8, edgecolors="white", linewidths=0.5)

    # Fit line
    if len(errors) >= 3 and not math.isnan(r):
        coeffs = np.polyfit(uncertainties, errors, 1)
        x_line = np.linspace(uncertainties.min(), uncertainties.max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), "r--", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Mean uncertainty width (2u)", fontsize=10)
    ax.set_ylabel("Mean absolute error", fontsize=10)
    p_str = f"{p:.3f}" if not math.isnan(p) else "n/a"
    r_str = f"{r:.3f}" if not math.isnan(r) else "n/a"
    ax.set_title(f"{modality} / {sigma_key}\nPearson r={r_str}  p={p_str}  n={len(records)} views", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate each dot with frame index
    for rec in records:
        ax.annotate(
            rec["frame"],
            xy=(rec["mean_uncertainty"], rec["mean_abs_error"]),
            fontsize=6,
            alpha=0.6,
            textcoords="offset points",
            xytext=(3, 3),
        )


def main():
    parser = argparse.ArgumentParser(description="Per-view quality prediction from conformal uncertainty")
    parser.add_argument("--run_dir", required=True, type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--modalities", nargs="+", default=None,
                        help="Override modalities, e.g. --modalities color depth. Default: color + depth.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    iteration = find_iteration(run_dir, args.iteration)

    modalities = args.modalities or None
    if modalities is not None:
        pairs = [(m, None) for m in modalities]
    else:
        pairs = MODALITIES

    out_dir = run_dir / "quality_prediction"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover available analysis names from the conformal output dirs
    test_base = run_dir / "test" / f"ours_{iteration}" / "conformal"
    if not test_base.exists():
        raise FileNotFoundError(
            f"No conformal outputs found at {test_base}.\n"
            "Run conformal_prediction.py first."
        )

    analysis_dirs = sorted(d for d in test_base.iterdir() if d.is_dir())
    if not analysis_dirs:
        raise FileNotFoundError(f"No analysis subdirectories found under {test_base}")

    print(f"Run dir   : {run_dir}")
    print(f"Iteration : {iteration}")
    print(f"Analyses  : {[d.name for d in analysis_dirs]}")

    results = {}
    valid_analyses = []

    for analysis_dir in analysis_dirs:
        arrays_dir = analysis_dir / "arrays"
        if not arrays_dir.exists():
            print(f"  [SKIP] {analysis_dir.name} — no arrays/ subdir")
            continue

        records = load_per_view_stats(arrays_dir)
        if len(records) < 2:
            print(f"  [SKIP] {analysis_dir.name} — fewer than 2 test views")
            continue

        r, p = compute_pearson(records)
        print(f"  {analysis_dir.name:40s}  n={len(records):3d}  Pearson r={r:+.3f}  p={p:.3f}")

        results[analysis_dir.name] = {
            "analysis": analysis_dir.name,
            "n_views": len(records),
            "pearson_r": r,
            "pearson_p": p,
            "per_view": records,
        }
        valid_analyses.append((analysis_dir.name, records, r, p))

    # Save JSON
    json_path = out_dir / "quality_prediction.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved JSON → {json_path}")

    # Plot
    n = len(valid_analyses)
    if n == 0:
        print("Nothing to plot.")
        return

    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle("Per-view quality prediction: mean uncertainty vs mean absolute error", fontsize=13)

    for idx, (analysis_name, records, r, p) in enumerate(valid_analyses):
        row, col = divmod(idx, ncols)
        modality, sigma_key = analysis_name.split("_", 1) if "_" in analysis_name else (analysis_name, "")
        plot_scatter(axes[row][col], records, modality, sigma_key, r, p)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    fig_path = out_dir / "quality_prediction.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {fig_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Modality':<40} {'r':>8} {'p':>8} {'n':>5}")
    print("-" * 60)
    for analysis_name, _, r, p in sorted(valid_analyses, key=lambda x: -abs(x[2]) if not math.isnan(x[2]) else -1):
        r_str = f"{r:+.3f}" if not math.isnan(r) else "  n/a"
        p_str = f"{p:.3f}" if not math.isnan(p) else "  n/a"
        n_views = results[analysis_name]["n_views"]
        print(f"{analysis_name:<40} {r_str:>8} {p_str:>8} {n_views:>5}")
    print("=" * 60)


if __name__ == "__main__":
    main()
