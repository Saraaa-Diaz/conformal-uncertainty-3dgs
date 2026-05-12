"""
Export per-view acquisition rankings for active-learning experiments.

By default this computes both:
  - raw acquisition scores: mean rendered signal over valid candidate pixels
  - conformal acquisition scores: mean 2*q_hat*sigma over valid candidate pixels

The conformal scores use fixed calibration views to estimate q_hat and then
apply that scalar to candidate-view sigma maps. Candidate GT is not used.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


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


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def load_sigma(path, key):
    data = np.load(path)
    if key not in data.files:
        raise KeyError(f"{key} not found in {path}. Available keys: {data.files}")
    sigma = np.asarray(data[key], dtype=np.float32)
    if sigma.ndim == 3:
        if sigma.shape[0] in (1, 3):
            sigma = sigma.mean(axis=0)
        elif sigma.shape[-1] in (1, 3):
            sigma = sigma.mean(axis=-1)
        else:
            raise ValueError(f"Unsupported sigma shape for {path}: {sigma.shape}")
    if sigma.ndim != 2:
        raise ValueError(f"Expected 2D sigma map in {path}, got {sigma.shape}")
    mask = data["mask"].astype(bool) if "mask" in data.files else np.ones_like(sigma, dtype=bool)
    return sigma, mask


def conformal_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=np.float32)
    n = scores.size
    if n == 0:
        raise ValueError("No calibration scores collected.")
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    level = float(np.clip(level, 0.0, 1.0))
    return float(np.quantile(scores, level, method="higher")), level


def raw_stats(values, mask):
    valid = np.isfinite(values) & mask
    if not np.any(valid):
        vals = np.array([0.0], dtype=np.float32)
    else:
        vals = values[valid]
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(vals.max()),
        "valid_pixels": int(valid.sum()),
    }


def compute_qhat(run_dir, iteration, modality, sigma_key, alpha, sample_ratio, seed, eps):
    calib_dir = run_dir / "calib" / f"ours_{iteration}"
    render_dir = calib_dir / "render"
    gt_dir = calib_dir / "gt"
    sigma_dir = calib_dir / "raw_sigma" / modality
    if not render_dir.exists() or not gt_dir.exists() or not sigma_dir.exists():
        return None

    rng = np.random.default_rng(seed)
    scores = []
    for render_path in sorted(render_dir.glob("*.png")):
        frame = render_path.name
        sigma_path = sigma_dir / frame.replace(".png", ".npz")
        if not sigma_path.exists():
            continue
        render = load_rgb(render_path)
        gt = load_rgb(gt_dir / frame)
        sigma, mask = load_sigma(sigma_path, sigma_key)
        err = np.abs(render - gt).mean(axis=2)
        sigma_safe = np.maximum(np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0), eps)
        valid = (np.isfinite(err) & np.isfinite(sigma_safe) & mask).reshape(-1)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            continue
        n_sample = max(1, min(int(math.ceil(sample_ratio * err.size)), valid_idx.size))
        chosen = rng.choice(valid_idx, size=n_sample, replace=False)
        scores.extend((err.reshape(-1)[chosen] / sigma_safe.reshape(-1)[chosen]).tolist())
    if not scores:
        return None
    q_hat, level = conformal_quantile(scores, alpha)
    return {
        "q_hat": q_hat,
        "quantile_level": level,
        "num_scores": len(scores),
        "alpha": alpha,
        "sample_ratio": sample_ratio,
    }


def normalize_scores(rows, fields, prefix):
    values_by_field = {}
    for field in fields:
        values = np.array([row.get(field, 0.0) for row in rows], dtype=np.float32)
        lo = float(values.min()) if values.size else 0.0
        hi = float(values.max()) if values.size else 0.0
        scale = hi - lo
        if scale <= 1e-12:
            values_by_field[field] = np.zeros_like(values)
        else:
            values_by_field[field] = (values - lo) / scale

    for idx, row in enumerate(rows):
        present = [field for field in fields if field in row]
        vals = [float(values_by_field[field][idx]) for field in present]
        row[f"{prefix}_combined_mean"] = float(np.mean(vals)) if vals else 0.0
        row[f"{prefix}_combined_max"] = float(np.max(vals)) if vals else 0.0


def collect_rows(run_dir, iteration, splits, sigma_keys, qhats, eps):
    rows_by_name = {}
    for split in splits:
        split_dir = run_dir / split / f"ours_{iteration}"
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
                sigma, mask = load_sigma(sigma_path, key)
                stats = raw_stats(sigma, mask)
                for stat_name, value in stats.items():
                    row[f"{modality}_raw_{stat_name}"] = value

                qhat = qhats.get(modality)
                if qhat is not None:
                    sigma_safe = np.maximum(np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0), eps)
                    full_width = 2.0 * qhat["q_hat"] * sigma_safe
                    conf_stats = raw_stats(full_width, mask)
                    for stat_name, value in conf_stats.items():
                        row[f"{modality}_conformal_{stat_name}_2u"] = value
    return list(rows_by_name.values())


def ranked(rows, score_key):
    return [
        {
            "rank": rank,
            "frame": row["frame"],
            "split": row["split"],
            "score": float(row.get(score_key, 0.0)),
        }
        for rank, row in enumerate(sorted(rows, key=lambda item: item.get(score_key, 0.0), reverse=True), start=1)
    ]


def main():
    parser = argparse.ArgumentParser(description="Export active-learning per-view acquisition rankings")
    parser.add_argument("--run_dir", required=True, type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--splits", nargs="+", default=["calib", "test"], choices=["train", "calib", "test", "candidate"])
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--calib_sample_ratio", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--acquisition_mode", choices=["both", "conformal", "raw"], default="both")
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
    qhats = {}
    if args.acquisition_mode in ("both", "conformal"):
        for modality, key in sigma_keys.items():
            qhats[modality] = compute_qhat(
                run_dir,
                iteration,
                modality,
                key,
                args.alpha,
                args.calib_sample_ratio,
                args.seed,
                args.eps,
            )
    rows = collect_rows(run_dir, iteration, args.splits, sigma_keys, qhats, args.eps)
    rows = sorted(rows, key=lambda row: row["frame"])

    raw_fields = ["color_raw_mean", "sensitivity_raw_mean", "visibility_raw_mean"]
    conformal_fields = [
        "color_conformal_mean_2u",
        "sensitivity_conformal_mean_2u",
        "visibility_conformal_mean_2u",
    ]
    if args.acquisition_mode in ("both", "raw"):
        normalize_scores(rows, raw_fields, "raw")
    if args.acquisition_mode in ("both", "conformal"):
        normalize_scores(rows, conformal_fields, "conformal")

    csv_path = out_dir / "view_signal_scores.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rankings = {}
    ranking_keys = {
        "raw_color": "color_raw_mean",
        "raw_sensitivity": "sensitivity_raw_mean",
        "raw_visibility": "visibility_raw_mean",
        "raw_combined": "raw_combined_mean",
        "conformal_color": "color_conformal_mean_2u",
        "conformal_sensitivity": "sensitivity_conformal_mean_2u",
        "conformal_visibility": "visibility_conformal_mean_2u",
        "conformal_combined": "conformal_combined_mean",
    }
    for name, key in ranking_keys.items():
        if any(key in row for row in rows):
            rankings[name] = ranked(rows, key)

    payload = {
        "run_dir": str(run_dir),
        "iteration": iteration,
        "splits": args.splits,
        "sigma_keys": sigma_keys,
        "acquisition_mode": args.acquisition_mode,
        "conformal_qhats": qhats,
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
