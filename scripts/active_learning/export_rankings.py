"""
Export per-view signal rankings for active-learning experiments.

Reads rendered raw_sigma maps from a run directory and produces view-level
acquisition scores for color, Fisher sensitivity, visibility, and simple
combined scores. This is intentionally a hook: it ranks candidate views but does
not run a retraining loop.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_SIGMA_KEYS = {
    "color": "color_std",
    "sensitivity": "sigma_mean",
    "visibility": "sigma_mean",
}


def find_iteration(run_dir, iteration):
    if iteration != -1:
        return iteration
    candidates = []
    for split in ("train", "calib", "test", "candidate"):
        split_dir = run_dir / split
        if not split_dir.exists():
            continue
        for child in split_dir.iterdir():
            if child.is_dir() and child.name.startswith("ours_"):
                try:
                    candidates.append(int(child.name.split("_", 1)[1]))
                except ValueError:
                    pass
    if not candidates:
        raise FileNotFoundError(f"No iteration directories found under {run_dir}")
    return max(candidates)


def load_signal(path, key):
    data = np.load(path)
    if key not in data.files:
        raise KeyError(f"{key} not found in {path}. Available keys: {data.files}")
    signal = np.asarray(data[key], dtype=np.float32)
    if signal.ndim == 3:
        if signal.shape[0] in (1, 3):
            signal = signal.mean(axis=0)
        elif signal.shape[-1] in (1, 3):
            signal = signal.mean(axis=-1)
        else:
            raise ValueError(f"Unsupported signal shape for {path}: {signal.shape}")
    mask = data["mask"].astype(bool) if "mask" in data.files else np.ones_like(signal, dtype=bool)
    valid = np.isfinite(signal) & mask
    if not np.any(valid):
        values = np.array([0.0], dtype=np.float32)
    else:
        values = signal[valid]
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "valid_pixels": int(valid.sum()),
    }


def normalize_scores(rows, fields):
    normalized = {}
    for field in fields:
        values = np.array([row.get(field, 0.0) for row in rows], dtype=np.float32)
        lo = float(np.min(values)) if values.size else 0.0
        hi = float(np.max(values)) if values.size else 0.0
        scale = hi - lo
        if scale <= 1e-12:
            normalized[field] = np.zeros_like(values)
        else:
            normalized[field] = (values - lo) / scale
    return normalized


def collect_rows(run_dir, iteration, splits, sigma_keys):
    rows_by_name = {}
    for split in splits:
        split_dir = run_dir / split / f"ours_{iteration}"
        if not split_dir.exists():
            continue
        render_dir = split_dir / "render"
        if not render_dir.exists():
            continue
        for render_path in sorted(render_dir.glob("*.png")):
            frame = render_path.name
            row = rows_by_name.setdefault(frame, {"frame": frame, "split": split})
            for modality, key in sigma_keys.items():
                sigma_path = split_dir / "raw_sigma" / modality / frame.replace(".png", ".npz")
                if not sigma_path.exists():
                    continue
                stats = load_signal(sigma_path, key)
                for stat_name, value in stats.items():
                    row[f"{modality}_{stat_name}"] = value
    return list(rows_by_name.values())


def add_combined_scores(rows):
    fields = ["color_mean", "sensitivity_mean", "visibility_mean"]
    normalized = normalize_scores(rows, fields)
    for idx, row in enumerate(rows):
        present = [field for field in fields if field in row]
        norm_values = [float(normalized[field][idx]) for field in present]
        row["combined_mean"] = float(np.mean(norm_values)) if norm_values else 0.0
        row["combined_max"] = float(np.max(norm_values)) if norm_values else 0.0
    return rows


def ranked(rows, score_key):
    return sorted(
        (
            {
                "rank": rank,
                "frame": row["frame"],
                "split": row["split"],
                "score": float(row.get(score_key, 0.0)),
            }
            for rank, row in enumerate(sorted(rows, key=lambda item: item.get(score_key, 0.0), reverse=True), start=1)
        ),
        key=lambda item: item["rank"],
    )


def main():
    parser = argparse.ArgumentParser(description="Export active-learning per-view signal rankings")
    parser.add_argument("--run_dir", required=True, type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--splits", nargs="+", default=["calib", "test"], choices=["train", "calib", "test", "candidate"])
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--color_key", default=DEFAULT_SIGMA_KEYS["color"], type=str)
    parser.add_argument("--sensitivity_key", default=DEFAULT_SIGMA_KEYS["sensitivity"], type=str)
    parser.add_argument("--visibility_key", default=DEFAULT_SIGMA_KEYS["visibility"], type=str)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    iteration = find_iteration(run_dir, args.iteration)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "active_learning" / f"ours_{iteration}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sigma_keys = {
        "color": args.color_key,
        "sensitivity": args.sensitivity_key,
        "visibility": args.visibility_key,
    }
    rows = collect_rows(run_dir, iteration, args.splits, sigma_keys)
    rows = add_combined_scores(rows)
    rows = sorted(rows, key=lambda row: row["frame"])

    csv_path = out_dir / "view_signal_scores.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rankings = {
        "color_mean": ranked(rows, "color_mean"),
        "sensitivity_mean": ranked(rows, "sensitivity_mean"),
        "visibility_mean": ranked(rows, "visibility_mean"),
        "combined_mean": ranked(rows, "combined_mean"),
        "combined_max": ranked(rows, "combined_max"),
    }
    payload = {
        "run_dir": str(run_dir),
        "iteration": iteration,
        "splits": args.splits,
        "sigma_keys": sigma_keys,
        "num_views": len(rows),
        "rankings": rankings,
    }
    json_path = out_dir / "view_rankings.json"
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Exported {len(rows)} view scores to {csv_path}")
    print(f"Exported rankings to {json_path}")


if __name__ == "__main__":
    main()
