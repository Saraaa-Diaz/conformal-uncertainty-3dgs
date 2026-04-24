import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare round-robin 7/1/2 train-calib-test splits")
    parser.add_argument("--source_path", required=True, type=str)
    parser.add_argument("--images", default="images", type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    images_dir = Path(args.source_path).resolve() / args.images
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_names = sorted(path.name for path in images_dir.iterdir() if path.is_file())
    split_map = {"train": [], "calib": [], "test": []}
    for idx, image_name in enumerate(image_names):
        offset = idx % 10
        if offset <= 6:
            split_map["train"].append(image_name)
        elif offset == 7:
            split_map["calib"].append(image_name)
        else:
            split_map["test"].append(image_name)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, names in split_map.items():
        with open(output_dir / f"{split_name}.txt", "w") as handle:
            handle.write("\n".join(names))
            handle.write("\n")

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as handle:
        for split_name in ("train", "calib", "test"):
            count = len(split_map[split_name])
            pct = (count / max(len(image_names), 1)) * 100.0
            handle.write(f"{split_name}: {count} ({pct:.2f}%)\n")


if __name__ == "__main__":
    main()
