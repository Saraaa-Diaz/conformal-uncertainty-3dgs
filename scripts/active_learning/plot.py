"""
Plot active-learning curves from active_learning_summary.csv.

Produces one figure per metric under:
  <al_root>/figures/
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_METRICS = {
    "psnr": ("PSNR", "PSNR (dB)", "higher"),
    "ssim": ("SSIM", "SSIM", "higher"),
    "lpips": ("LPIPS", "LPIPS", "lower"),
}

UNCERTAINTY_METRICS = {
    "coverage": ("Coverage", "Coverage", "target"),
    "mean_2u": ("Mean Interval Width", "mean 2u", "lower"),
    "ae_corr": ("AE-Uncertainty Correlation", "Pearson r", "higher"),
    "ause": ("AUSE", "AUSE", "lower"),
}


def parse_float(value):
    if value is None or value == "":
        return None
    return float(value)


def read_rows(path):
    with open(path, "r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    for row in rows:
        out = dict(row)
        out["round"] = int(row["round"])
        out["train_views"] = int(row["train_views"])
        for key in ("psnr", "ssim", "lpips", "coverage", "mean_2u", "ae_corr", "ause"):
            out[key] = parse_float(row.get(key))
        parsed.append(out)
    return parsed


def group_image_rows(rows):
    # Image metrics are repeated once per conformal modality in the summary CSV.
    # Collapse to one point per method/seed/round.
    by_key = {}
    for row in rows:
        key = (row["method"], row["seed"], row["round"], row["train_views"])
        by_key.setdefault(key, row)
    return list(by_key.values())


def group_uncertainty_rows(rows, modality):
    return [row for row in rows if row.get("modality") == modality]


def series_by_method(rows, metric):
    grouped = defaultdict(lambda: defaultdict(list))
    x_by_round = {}
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        method = row["method"]
        round_idx = row["round"]
        grouped[method][round_idx].append(value)
        x_by_round[round_idx] = row["train_views"]

    series = {}
    for method, by_round in grouped.items():
        xs, ys, yerr = [], [], []
        for round_idx in sorted(by_round):
            vals = by_round[round_idx]
            mean = sum(vals) / len(vals)
            var = sum((val - mean) ** 2 for val in vals) / len(vals)
            xs.append(x_by_round[round_idx])
            ys.append(mean)
            yerr.append(var ** 0.5)
        series[method] = (xs, ys, yerr)
    return series


def plot_metric(rows, metric, title, ylabel, out_path, target=None):
    series = series_by_method(rows, metric)
    if not series:
        return False

    plt.figure(figsize=(7.5, 5.0))
    for method, (xs, ys, yerr) in sorted(series.items()):
        plt.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=2, capsize=3, label=method)
    if target is not None:
        plt.axhline(target, color="black", linestyle="--", linewidth=1, label=f"target {target:g}")
    plt.xlabel("Training views")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Plot active-learning learning curves")
    parser.add_argument("--al_root", required=True)
    parser.add_argument("--summary_csv", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--modalities", nargs="+", default=["color", "sensitivity", "visibility"])
    parser.add_argument("--coverage_target", type=float, default=0.9)
    args = parser.parse_args()

    al_root = Path(args.al_root).resolve()
    summary_csv = Path(args.summary_csv) if args.summary_csv else al_root / "active_learning_summary.csv"
    out_dir = Path(args.out_dir) if args.out_dir else al_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(summary_csv)
    image_rows = group_image_rows(rows)

    written = []
    for metric, (title, ylabel, _) in IMAGE_METRICS.items():
        out_path = out_dir / f"{metric}_vs_train_views.png"
        if plot_metric(image_rows, metric, f"{title} vs Training Views", ylabel, out_path):
            written.append(out_path)

    for modality in args.modalities:
        modality_rows = group_uncertainty_rows(rows, modality)
        if not modality_rows:
            continue
        for metric, (title, ylabel, _) in UNCERTAINTY_METRICS.items():
            target = args.coverage_target if metric == "coverage" else None
            out_path = out_dir / f"{modality}_{metric}_vs_train_views.png"
            if plot_metric(modality_rows, metric, f"{modality}: {title} vs Training Views", ylabel, out_path, target=target):
                written.append(out_path)

    print(f"Wrote {len(written)} figures to {out_dir}")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
