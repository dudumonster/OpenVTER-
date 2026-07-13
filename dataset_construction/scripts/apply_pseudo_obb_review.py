#!/usr/bin/env python3
"""Apply review decisions to a generated pseudo YOLO-OBB dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml


GLOBAL_NAMES = {
    0: "motor_vehicle",
    1: "bicycle",
    2: "motor",
    3: "tricycle",
    4: "awning_tricycle",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_points(points: list[list[float]], image_w: int, image_h: int) -> list[float]:
    values: list[float] = []
    for x, y in points:
        values.append(round(min(max(float(x), 0.0), image_w - 1) / max(image_w, 1), 6))
        values.append(round(min(max(float(y), 0.0), image_h - 1) / max(image_h, 1), 6))
    return values


def save_data_yaml(yolo_root: Path) -> None:
    data = {
        "path": str(yolo_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": GLOBAL_NAMES,
    }
    with (yolo_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def apply_review(args: argparse.Namespace) -> None:
    pseudo_root = Path(args.pseudo_root)
    yolo_root = Path(args.yolo_root)
    quality_path = Path(args.quality_jsonl or pseudo_root / "quality.jsonl")
    decisions_path = Path(args.review_decisions or pseudo_root / "review_decisions.jsonl")
    records = read_jsonl(quality_path)
    decisions = {}
    for row in read_jsonl(decisions_path):
        sample_id = row.get("sample_id")
        if sample_id:
            decisions[sample_id] = row

    labels_by_image: dict[tuple[str, str], list[str]] = defaultdict(list)
    stats = {
        "records": len(records),
        "decisions": len(decisions),
        "exported": 0,
        "rejected": 0,
        "pending_low_quality": 0,
        "edited": 0,
    }

    for record in records:
        sample_id = record["sample_id"]
        split = record["split"]
        image_stem = Path(record["image_name"]).stem
        decision = decisions.get(sample_id)

        if decision and decision.get("decision") == "reject":
            stats["rejected"] += 1
            continue

        if decision and decision.get("decision") in {"accept", "edit"}:
            class_id = int(decision.get("class_id", record["class_id"]))
            points = decision.get("obb_points") or record.get("obb_points") or []
            if len(points) != 4:
                stats["rejected"] += 1
                continue
            if decision.get("decision") == "edit":
                stats["edited"] += 1
            values = normalize_points(points, int(record["image_width"]), int(record["image_height"]))
        else:
            if record.get("quality", {}).get("quality_status") != "auto_accept":
                stats["pending_low_quality"] += 1
                continue
            class_id = int(record["class_id"])
            yolo_obb = record.get("yolo_obb") or []
            if len(yolo_obb) != 9:
                stats["pending_low_quality"] += 1
                continue
            values = [float(v) for v in yolo_obb[1:]]

        if len(values) != 8:
            continue
        labels_by_image[(split, image_stem)].append(
            " ".join([str(class_id), *[f"{float(v):.6f}" for v in values]])
        )
        stats["exported"] += 1

    label_root = yolo_root / "labels"
    for label_file in label_root.glob("*/*.txt"):
        label_file.write_text("", encoding="utf-8")
    for (split, image_stem), rows in labels_by_image.items():
        label_path = label_root / split / f"{image_stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    save_data_yaml(yolo_root)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pseudo-root",
        default="dataset_construction/derived/visdrone_pseudo_obb_v1",
    )
    parser.add_argument(
        "--yolo-root",
        default="dataset_construction/derived/visdrone_yolo_obb_v1",
    )
    parser.add_argument("--quality-jsonl", default=None)
    parser.add_argument("--review-decisions", default=None)
    return parser


def main() -> None:
    apply_review(build_parser().parse_args())


if __name__ == "__main__":
    main()
