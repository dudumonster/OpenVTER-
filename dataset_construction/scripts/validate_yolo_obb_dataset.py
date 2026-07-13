#!/usr/bin/env python3
"""Validate a YOLO-OBB dataset exported by the pseudo-label pipeline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


def validate(args: argparse.Namespace) -> None:
    root = Path(args.yolo_root)
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, list):
        valid_class_ids = set(range(len(names)))
    else:
        valid_class_ids = {int(k) for k in names}

    errors = []
    label_files = sorted((root / "labels").glob("*/*.txt"))
    class_counts: Counter[int] = Counter()
    nonempty_files = 0
    rows = 0

    for label_file in label_files:
        split = label_file.parent.name
        image_dir = root / "images" / split
        image_candidates = [
            image_dir / f"{label_file.stem}.jpg",
            image_dir / f"{label_file.stem}.png",
            image_dir / f"{label_file.stem}.jpeg",
        ]
        if not any(p.exists() for p in image_candidates):
            errors.append(f"missing image for {label_file}")

        lines = [line.strip() for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            nonempty_files += 1
        for line_no, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 9:
                errors.append(f"{label_file}:{line_no}: expected 9 columns, got {len(parts)}")
                continue
            try:
                class_id = int(parts[0])
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                errors.append(f"{label_file}:{line_no}: non-numeric value")
                continue
            if class_id not in valid_class_ids:
                errors.append(f"{label_file}:{line_no}: class_id {class_id} not in {sorted(valid_class_ids)}")
            bad_coords = [v for v in coords if v < 0.0 or v > 1.0]
            if bad_coords:
                errors.append(f"{label_file}:{line_no}: coordinates outside [0,1]")
            class_counts[class_id] += 1
            rows += 1

    summary = {
        "yolo_root": str(root),
        "label_files": len(label_files),
        "nonempty_label_files": nonempty_files,
        "rows": rows,
        "class_counts": dict(sorted(class_counts.items())),
        "errors": len(errors),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        for err in errors[: args.max_errors]:
            print(err)
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yolo_root", nargs="?", default="dataset_construction/derived/visdrone_yolo_obb_v1")
    parser.add_argument("--max-errors", type=int, default=50)
    return parser


def main() -> None:
    validate(build_parser().parse_args())


if __name__ == "__main__":
    main()
