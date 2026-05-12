"""
Aggregate active-learning round outputs across methods/seeds.

Expected layout:
  <al_root>/<method>/seed_<seed>/round_<rr>/

Writes:
  <al_root>/active_learning_summary.csv
  <al_root>/active_learning_summary.md
"""

import argparse
import csv
import json
from pathlib import Path


def read_split_count(round_dir, split_name):
    path = round_dir / "splits" / f"{split_name}.txt"
    if not path.exists():
        return 0
    with open(path, "r") as handle:
        return sum(1 for line in handle if line.strip())


def read_image_metrics(round_dir):
    path = round_dir / "results.json"
    if not path.exists():
        return {}
    with open(path, "r") as handle:
        data = json.load(handle)
    for payload in data.values():
        if isinstance(payload, dict):
            return {
                "psnr": payload.get("PSNR"),
                "ssim": payload.get("SSIM"),
                "lpips": payload.get("LPIPS"),
            }
    return {}


def iter_conformal_metrics(round_dir):
    conformal_dir = round_dir / "conformal"
    if not conformal_dir.exists():
        return
    for metrics_path in sorted(conformal_dir.glob("*/ours_*/metrics.json")):
        with open(metrics_path, "r") as handle:
            yield json.load(handle)


def collect_rows(al_root):
    rows = []
    for method_dir in sorted(path for path in al_root.iterdir() if path.is_dir()):
        method = method_dir.name
        for seed_dir in sorted(path for path in method_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
            seed = seed_dir.name.replace("seed_", "")
            for round_dir in sorted(path for path in seed_dir.iterdir() if path.is_dir() and path.name.startswith("round_")):
                round_idx = int(round_dir.name.replace("round_", ""))
                base = {
                    "method": method,
                    "seed": seed,
                    "round": round_idx,
                    "train_views": read_split_count(round_dir, "train"),
                    "candidate_views": read_split_count(round_dir, "candidate"),
                    **read_image_metrics(round_dir),
                }
                conformal_seen = False
                for metrics in iter_conformal_metrics(round_dir):
                    conformal_seen = True
                    row = dict(base)
                    row.update(
                        {
                            "modality": metrics.get("modality"),
                            "sigma_key": metrics.get("sigma_key"),
                            "coverage": metrics.get("test_pixel_coverage"),
                            "coverage_std": metrics.get("test_pixel_coverage_std"),
                            "mean_2u": metrics.get("test_mean_interval_size"),
                            "mean_2u_std": metrics.get("test_mean_interval_size_std_per_view"),
                            "ae_corr": metrics.get("mean_per_view_ae_uncertainty_corr"),
                            "ae_corr_std": metrics.get("std_per_view_ae_uncertainty_corr"),
                            "ause": metrics.get("mean_per_view_ause"),
                            "ause_std": metrics.get("std_per_view_ause"),
                        }
                    )
                    rows.append(row)
                if not conformal_seen:
                    rows.append({**base, "modality": ""})
    return rows


def write_csv(path, rows):
    fields = [
        "method",
        "seed",
        "round",
        "train_views",
        "candidate_views",
        "psnr",
        "ssim",
        "lpips",
        "modality",
        "sigma_key",
        "coverage",
        "coverage_std",
        "mean_2u",
        "mean_2u_std",
        "ae_corr",
        "ae_corr_std",
        "ause",
        "ause_std",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def write_markdown(path, rows):
    lines = ["# Active Learning Summary", ""]
    lines.append("| method | seed | round | train views | PSNR | SSIM | LPIPS | modality | coverage | mean 2u | AE corr | AUSE |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {method} | {seed} | {round} | {train_views} | {psnr} | {ssim} | {lpips} | {modality} | {coverage} | {mean_2u} | {ae_corr} | {ause} |".format(
                method=row.get("method", ""),
                seed=row.get("seed", ""),
                round=row.get("round", ""),
                train_views=row.get("train_views", ""),
                psnr=fmt(row.get("psnr"), 3),
                ssim=fmt(row.get("ssim"), 4),
                lpips=fmt(row.get("lpips"), 4),
                modality=row.get("modality", ""),
                coverage=fmt(row.get("coverage"), 4),
                mean_2u=fmt(row.get("mean_2u"), 2),
                ae_corr=fmt(row.get("ae_corr"), 4),
                ause=fmt(row.get("ause"), 4),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate active-learning experiment outputs")
    parser.add_argument("--al_root", required=True)
    args = parser.parse_args()

    al_root = Path(args.al_root).resolve()
    rows = collect_rows(al_root)
    csv_path = al_root / "active_learning_summary.csv"
    md_path = al_root / "active_learning_summary.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote markdown summary to {md_path}")


if __name__ == "__main__":
    main()
