"""
Create and update active-learning split manifests.

The split layout is:
  train.txt      views currently used for 3DGS training
  calib.txt      fixed calibration views for conformal evaluation
  test.txt       fixed held-out test views, never selected
  candidate.txt  selectable pool for the next AL round

Selection methods:
  random      random candidate views
  uniform     evenly spaced candidate views in image order
  conformal_color       color conformal mean interval width
  conformal_visibility  visibility conformal mean interval width
  conformal_sensitivity sensitivity conformal mean interval width
  raw_sensitivity       PUP-style raw Fisher/sensitivity baseline
"""

import argparse
import json
from pathlib import Path

import numpy as np


ACQUISITION_KEYS = {
    "fisher": "raw_sensitivity",
    "pup": "raw_sensitivity",
    "raw_sensitivity": "raw_sensitivity",
    "raw_color": "raw_color",
    "raw_visibility": "raw_visibility",
    "raw_combined": "raw_combined",
    "sensitivity": "conformal_sensitivity",
    "color": "conformal_color",
    "visibility": "conformal_visibility",
    "combined": "conformal_combined",
    "conformal_sensitivity": "conformal_sensitivity",
    "conformal_color": "conformal_color",
    "conformal_visibility": "conformal_visibility",
    "conformal_combined": "conformal_combined",
}


def read_names(path):
    with open(path, "r") as handle:
        return [line.strip() for line in handle if line.strip()]


def write_names(path, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(names))
        if names:
            handle.write("\n")


def list_images(source_path, images):
    image_dir = Path(source_path).resolve() / images
    if not image_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {image_dir}")
    return sorted(path.name for path in image_dir.iterdir() if path.is_file())


def count_from_arg(value, total, name):
    if isinstance(value, str) and value.endswith("%"):
        frac = float(value[:-1]) / 100.0
        return max(1, int(round(frac * total)))
    parsed = float(value)
    if 0.0 < parsed < 1.0:
        return max(1, int(round(parsed * total)))
    count = int(parsed)
    if count < 0:
        raise ValueError(f"{name} must be non-negative")
    return count


def write_summary(output_dir, payload):
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    with open(output_dir / "summary.txt", "w") as handle:
        for key in ("train", "calib", "test", "candidate"):
            names = payload.get(key, [])
            handle.write(f"{key}: {len(names)}\n")


def init_splits(args):
    images = list_images(args.source_path, args.images)
    total = len(images)
    if total == 0:
        raise ValueError("No images found.")

    init_train_n = count_from_arg(args.init_train, total, "init_train")
    calib_n = count_from_arg(args.calib, total, "calib")
    test_n = count_from_arg(args.test, total, "test")
    if args.test_mode == "llffhold":
        test_idx = [idx for idx in range(total) if idx % args.llffhold == 0]
        train_pool = [idx for idx in range(total) if idx not in set(test_idx)]
        if init_train_n + calib_n >= len(train_pool):
            raise ValueError(
                f"init_train+calib must leave candidates: "
                f"{init_train_n}+{calib_n} >= {len(train_pool)}"
            )
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(train_pool).tolist()
        train_idx = sorted(perm[:init_train_n])
        calib_idx = sorted(perm[init_train_n : init_train_n + calib_n])
        used = set(train_idx) | set(calib_idx) | set(test_idx)
        candidate_idx = [idx for idx in range(total) if idx not in used]
    elif init_train_n + calib_n + test_n >= total:
        raise ValueError(
            f"init_train+calib+test must leave candidates: "
            f"{init_train_n}+{calib_n}+{test_n} >= {total}"
        )
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(total).tolist()
        train_idx = sorted(perm[:init_train_n])
        calib_idx = sorted(perm[init_train_n : init_train_n + calib_n])
        test_idx = sorted(perm[init_train_n + calib_n : init_train_n + calib_n + test_n])
        used = set(train_idx) | set(calib_idx) | set(test_idx)
        candidate_idx = [idx for idx in range(total) if idx not in used]

    split_map = {
        "train": [images[idx] for idx in train_idx],
        "calib": [images[idx] for idx in calib_idx],
        "test": [images[idx] for idx in test_idx],
        "candidate": [images[idx] for idx in candidate_idx],
    }
    output_dir = Path(args.output_dir).resolve()
    for split_name, names in split_map.items():
        write_names(output_dir / f"{split_name}.txt", names)

    payload = {
        "mode": "init",
        "source_path": str(Path(args.source_path).resolve()),
        "images": args.images,
        "seed": args.seed,
        "init_train": args.init_train,
        "calib": args.calib,
        "test": args.test,
        "test_mode": args.test_mode,
        "llffhold": args.llffhold,
        **split_map,
    }
    write_summary(output_dir, payload)
    print(f"Wrote initial AL splits to {output_dir}")


def load_scores(rankings_path, method):
    key = ACQUISITION_KEYS[method]
    with open(rankings_path, "r") as handle:
        payload = json.load(handle)
    rankings = payload.get("rankings", {})
    if key not in rankings:
        raise KeyError(f"{key} not found in {rankings_path}. Available: {sorted(rankings)}")
    return [entry["frame"] for entry in rankings[key]]


def choose_candidates(args, candidate):
    k = min(args.add_k, len(candidate))
    if k <= 0:
        return []
    method = args.method
    if method == "random":
        rng = np.random.default_rng(args.seed + args.round)
        return [candidate[idx] for idx in rng.choice(len(candidate), size=k, replace=False).tolist()]
    if method == "uniform":
        if k == 1:
            return [candidate[len(candidate) // 2]]
        indices = np.linspace(0, len(candidate) - 1, num=k)
        return [candidate[int(round(idx))] for idx in indices]

    ranked_frames = load_scores(Path(args.rankings_path), method)
    candidate_set = set(candidate)
    selected = []
    for frame in ranked_frames:
        if frame in candidate_set:
            selected.append(frame)
        if len(selected) == k:
            break
    if len(selected) < k:
        selected_set = set(selected)
        selected.extend([name for name in candidate if name not in selected_set][: k - len(selected)])
    return selected


def update_splits(args):
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    train = read_names(input_dir / "train.txt")
    calib = read_names(input_dir / "calib.txt")
    test = read_names(input_dir / "test.txt")
    candidate = read_names(input_dir / "candidate.txt")

    selected = choose_candidates(args, candidate)
    selected_set = set(selected)
    next_train = train + selected
    next_candidate = [name for name in candidate if name not in selected_set]

    split_map = {
        "train": next_train,
        "calib": calib,
        "test": test,
        "candidate": next_candidate,
    }
    for split_name, names in split_map.items():
        write_names(output_dir / f"{split_name}.txt", names)

    payload = {
        "mode": "update",
        "input_dir": str(input_dir),
        "method": args.method,
        "round": args.round,
        "seed": args.seed,
        "add_k": args.add_k,
        "rankings_path": str(Path(args.rankings_path).resolve()) if args.rankings_path else "",
        "selected": selected,
        **split_map,
    }
    write_summary(output_dir, payload)
    print(f"Selected {len(selected)} views with method={args.method}:")
    for name in selected:
        print(f"  {name}")
    print(f"Wrote next AL splits to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare/update active-learning split manifests")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--source_path", required=True)
    init.add_argument("--images", default="images")
    init.add_argument("--output_dir", required=True)
    init.add_argument("--seed", type=int, default=0)
    init.add_argument("--init_train", default="10%")
    init.add_argument("--calib", default="10%")
    init.add_argument("--test", default="20%")
    init.add_argument("--test_mode", default="random", choices=["random", "llffhold"])
    init.add_argument("--llffhold", type=int, default=8)
    init.set_defaults(func=init_splits)

    update = sub.add_parser("update")
    update.add_argument("--input_dir", required=True)
    update.add_argument("--output_dir", required=True)
    update.add_argument(
        "--method",
        required=True,
        choices=[
            "random",
            "uniform",
            "fisher",
            "pup",
            "sensitivity",
            "color",
            "visibility",
            "combined",
            "raw_sensitivity",
            "raw_color",
            "raw_visibility",
            "raw_combined",
            "conformal_sensitivity",
            "conformal_color",
            "conformal_visibility",
            "conformal_combined",
        ],
    )
    update.add_argument("--rankings_path", default="")
    update.add_argument("--add_k", type=int, default=5)
    update.add_argument("--round", type=int, required=True)
    update.add_argument("--seed", type=int, default=0)
    update.set_defaults(func=update_splits)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
