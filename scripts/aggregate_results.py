"""
Aggregate conformal prediction metrics across all sigma modalities into a
single results.md table.

Reads each modality's metrics.json from
    <run_dir>/conformal/<modality>_<sigma_key>/ours_<iter>/metrics.json
and produces:
    <run_dir>/results.md

Prints the table to stdout as well.
"""

import argparse
import json
from pathlib import Path


def find_metrics(run_dir: Path):
    conformal_dir = run_dir / "conformal"
    if not conformal_dir.exists():
        return []
    rows = []
    for analysis_dir in sorted(conformal_dir.iterdir()):
        if not analysis_dir.is_dir():
            continue
        for iter_dir in sorted(analysis_dir.iterdir()):
            if not iter_dir.is_dir() or not iter_dir.name.startswith("ours_"):
                continue
            metrics_path = iter_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            with open(metrics_path) as fh:
                rows.append((analysis_dir.name, iter_dir.name, json.load(fh)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=str)
    parser.add_argument("--out", default=None, type=str)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    rows = find_metrics(run_dir)
    if not rows:
        raise SystemExit(f"No metrics.json found under {run_dir / 'conformal'}")

    out_path = Path(args.out) if args.out else run_dir / "results.md"

    lines = []
    lines.append(f"# Conformal results — {run_dir.name}\n")
    lines.append(f"Run dir: `{run_dir}`\n")
    lines.append("")
    lines.append("| modality | sigma_key | iter | α | target cov | per-view cov | per-view mean full width 2q*u | per-view std full width | per-view AE corr | per-view AUSE | calib frames | test frames |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for analysis_name, iter_name, metrics in rows:
        modality = metrics.get("modality", "?")
        sigma_key = metrics.get("sigma_key", "?")
        alpha = metrics.get("alpha")
        target = metrics.get("target_coverage")
        cov = metrics.get("test_pixel_coverage")
        cov_std = metrics.get("test_pixel_coverage_std", 0.0)
        mean_w = metrics.get("test_mean_full_width")
        mean_w_std = metrics.get("test_mean_full_width_std_per_view", 0.0)
        std_w = metrics.get("test_full_width_std")
        std_w_std = metrics.get("test_full_width_std_std_per_view", 0.0)
        corr = metrics.get("mean_per_view_ae_uncertainty_corr")
        corr_std = metrics.get("std_per_view_ae_uncertainty_corr", 0.0)
        ause = metrics.get("mean_per_view_ause", 0.0)
        ause_std = metrics.get("std_per_view_ause", 0.0)
        nc = metrics.get("num_calib_frames")
        nt = metrics.get("num_test_frames")
        lines.append(
            "| {modality} | {sigma_key} | {it} | {alpha:.2f} | {target:.3f} | {cov:.4f}±{cov_std:.4f} | {mean_w:.2f}±{mean_w_std:.2f} | {std_w:.2f}±{std_w_std:.2f} | {corr:.4f}±{corr_std:.4f} | {ause:.4f}±{ause_std:.4f} | {nc} | {nt} |".format(
                modality=modality,
                sigma_key=sigma_key,
                it=iter_name.replace("ours_", ""),
                alpha=alpha,
                target=target,
                cov=cov,
                cov_std=cov_std,
                mean_w=mean_w,
                mean_w_std=mean_w_std,
                std_w=std_w,
                std_w_std=std_w_std,
                corr=corr,
                corr_std=corr_std,
                ause=ause,
                ause_std=ause_std,
                nc=nc,
                nt=nt,
            )
        )

    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append("- **per-view cov**: empirical coverage computed per test view, then reported as mean±std over views. Should be near `target cov = 1−α`.")
    lines.append("- **per-view mean full width 2q*u**: calibrated conformal interval width `2·q̂·u_norm` computed per view, then reported as mean±std over views. Smaller = tighter intervals at the same coverage.")
    lines.append("- **per-view AE corr**: per-view Pearson correlation between `|render − gt|` and calibrated full width `2·q̂·u_norm`, reported as mean±std over views. Higher positive values mean uncertainty tracks error.")
    lines.append("- **per-view AUSE**: area under the sparsification error curve per view, reported as mean±std. Lower is better.")
    lines.append("")
    lines.append("**The most useful sigma is the one with the highest AE correlation at the lowest mean width while hitting target coverage.**")

    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    print(text)
    print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    main()
