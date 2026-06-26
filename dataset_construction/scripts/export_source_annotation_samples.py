#!/usr/bin/env python3
"""Export raw/annotated sample previews for source datasets.

This script is intentionally read-only for dataset sources. It writes a compact
review package under dataset_construction/derived so we can inspect how each
source dataset looks before conversion.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


VISDRONE_NAMES = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

VISDRONE_ZH = {
    "ignored": "忽略区域",
    "pedestrian": "行人",
    "people": "人群/多人",
    "bicycle": "自行车",
    "car": "小汽车",
    "van": "面包车",
    "truck": "卡车",
    "tricycle": "三轮车",
    "awning-tricycle": "带篷三轮车",
    "bus": "公交车",
    "motor": "摩托车/电动车",
    "others": "其他",
}

UAV_OBB_NAMES = {
    0: "bike",
    1: "bus",
    2: "car",
    3: "other_vehicle",
    4: "taxi",
    5: "truck",
}

UAV_OBB_ZH = {
    "bike": "自行车/两轮车",
    "bus": "公交车",
    "car": "小汽车",
    "other_vehicle": "其他车辆",
    "taxi": "出租车",
    "truck": "卡车",
}

VSAI_ZH = {
    "small-vehicle": "小型车辆",
    "large-vehicle": "大型车辆",
}

PALETTE = [
    (255, 80, 80),
    (80, 200, 255),
    (80, 255, 120),
    (255, 180, 80),
    (190, 120, 255),
    (255, 80, 190),
    (160, 255, 80),
    (80, 140, 255),
    (255, 240, 80),
    (80, 255, 230),
    (230, 230, 230),
    (120, 120, 255),
]


@dataclass
class ExportedSample:
    dataset: str
    split: str
    stem: str
    raw_rel: str
    annotated_rel: str
    classes: dict[str, int]
    annotation_type: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_image(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image)


def label_color(label: str) -> tuple[int, int, int]:
    return PALETTE[abs(hash(label)) % len(PALETTE)]


def put_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    y = max(y, 14)
    scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(image, (x, y - th - baseline - 3), (x + tw + 4, y + 2), color, -1)
    cv2.putText(
        image,
        text,
        (x + 2, y - baseline - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def resize_for_preview(image: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = image.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return image
    scale = max_side / side
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def make_pair(raw: np.ndarray, annotated: np.ndarray, title: str) -> np.ndarray:
    raw = resize_for_preview(raw.copy(), 900)
    annotated = resize_for_preview(annotated.copy(), 900)
    h = max(raw.shape[0], annotated.shape[0])
    w1, w2 = raw.shape[1], annotated.shape[1]
    canvas = np.full((h + 44, w1 + w2 + 8, 3), 245, dtype=np.uint8)
    canvas[44 : 44 + raw.shape[0], :w1] = raw
    canvas[44 : 44 + annotated.shape[0], w1 + 8 : w1 + 8 + w2] = annotated
    cv2.putText(canvas, "RAW / original image", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, title[:72], (w1 + 18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
    return canvas


def draw_visdrone(image: np.ndarray, ann_path: Path) -> tuple[np.ndarray, Counter]:
    out = image.copy()
    counts: Counter = Counter()
    if not ann_path.exists():
        return out, counts
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        x, y, w, h = [int(float(v)) for v in parts[:4]]
        cls_id = int(float(parts[5]))
        label = VISDRONE_NAMES.get(cls_id, f"class_{cls_id}")
        counts[label] += 1
        color = label_color(label)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        put_label(out, label, x, y, color)
    return out, counts


def parse_visdrone_counts(ann_path: Path) -> Counter:
    counts: Counter = Counter()
    if not ann_path.exists():
        return counts
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            label = VISDRONE_NAMES.get(int(float(parts[5])), f"class_{parts[5]}")
            counts[label] += 1
    return counts


def draw_vsai(image: np.ndarray, ann_path: Path) -> tuple[np.ndarray, Counter]:
    out = image.copy()
    counts: Counter = Counter()
    if not ann_path.exists():
        return out, counts
    obj = json.loads(ann_path.read_text(encoding="utf-8"))
    for item in obj.get("objects", []):
        label = item.get("classTitle", "unknown")
        pts = item.get("points", {}).get("exterior", [])
        if len(pts) < 3:
            continue
        counts[label] += 1
        color = label_color(label)
        arr = np.asarray(pts, dtype=np.int32)
        cv2.polylines(out, [arr], True, color, 2)
        x, y = arr[:, 0].min(), arr[:, 1].min()
        put_label(out, label, int(x), int(y), color)
    return out, counts


def parse_vsai_counts(ann_path: Path) -> Counter:
    counts: Counter = Counter()
    obj = json.loads(ann_path.read_text(encoding="utf-8"))
    for item in obj.get("objects", []):
        counts[item.get("classTitle", "unknown")] += 1
    return counts


def draw_yolo_obb(image: np.ndarray, label_text: str, names: dict[int, str]) -> tuple[np.ndarray, Counter]:
    out = image.copy()
    h, w = out.shape[:2]
    counts: Counter = Counter()
    for line in label_text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 9:
            continue
        cls_id = int(float(parts[0]))
        label = names.get(cls_id, f"class_{cls_id}")
        coords = [float(v) for v in parts[1:]]
        pts = np.asarray([[coords[i] * w, coords[i + 1] * h] for i in range(0, 8, 2)], dtype=np.int32)
        counts[label] += 1
        color = label_color(label)
        cv2.polylines(out, [pts], True, color, 2)
        x, y = pts[:, 0].min(), pts[:, 1].min()
        put_label(out, label, int(x), int(y), color)
    return out, counts


def select_class_balanced(items: list[tuple[Path, Counter]], limit: int) -> list[Path]:
    selected: list[Path] = []
    seen: set[Path] = set()
    by_class: dict[str, list[Path]] = defaultdict(list)
    for path, counts in items:
        for label, count in counts.items():
            if count > 0:
                by_class[label].append(path)
    for label in sorted(by_class):
        for path in by_class[label]:
            if path not in seen:
                selected.append(path)
                seen.add(path)
                break
    for path, counts in sorted(items, key=lambda x: (-sum(x[1].values()), str(x[0]))):
        if len(selected) >= limit:
            break
        if path not in seen and sum(counts.values()) > 0:
            selected.append(path)
            seen.add(path)
    return selected[:limit]


def export_visdrone(root: Path, out_root: Path, limit: int) -> tuple[list[ExportedSample], dict]:
    dataset_out = out_root / "visdrone"
    raw_out = dataset_out / "raw"
    ann_out = dataset_out / "annotated"
    pairs_out = dataset_out / "pairs"
    samples: list[ExportedSample] = []
    class_totals: Counter = Counter()
    candidates: list[tuple[Path, Counter]] = []

    for split_name in ["VisDrone2019-DET-train", "VisDrone2019-DET-val"]:
        ann_dir = root / split_name / "annotations"
        for ann_path in sorted(ann_dir.glob("*.txt")):
            counts = parse_visdrone_counts(ann_path)
            if counts:
                candidates.append((ann_path, counts))
                class_totals.update(counts)

    for ann_path in select_class_balanced(candidates, limit):
        split = ann_path.parents[1].name.replace("VisDrone2019-DET-", "")
        img_path = ann_path.parents[1] / "images" / f"{ann_path.stem}.jpg"
        image = read_image(img_path)
        if image is None:
            continue
        annotated, counts = draw_visdrone(image, ann_path)
        raw_dest = raw_out / split / img_path.name
        ann_dest = ann_out / split / f"{ann_path.stem}_visdrone_hbb.jpg"
        pair_dest = pairs_out / split / f"{ann_path.stem}_pair.jpg"
        ensure_dir(raw_dest.parent)
        shutil.copy2(img_path, raw_dest)
        write_image(ann_dest, resize_for_preview(annotated))
        write_image(pair_dest, make_pair(image, annotated, "VisDrone original HBB annotation"))
        samples.append(
            ExportedSample(
                dataset="VisDrone",
                split=split,
                stem=ann_path.stem,
                raw_rel=str(raw_dest.relative_to(out_root)),
                annotated_rel=str(ann_dest.relative_to(out_root)),
                classes=dict(counts),
                annotation_type="HBB: x,y,w,h + class_id",
            )
        )
    return samples, {"classes": dict(class_totals), "annotation_type": "HBB"}


def export_vsai(root: Path, out_root: Path, limit: int) -> tuple[list[ExportedSample], dict]:
    dataset_out = out_root / "vsai"
    raw_out = dataset_out / "raw"
    ann_out = dataset_out / "annotated"
    pairs_out = dataset_out / "pairs"
    samples: list[ExportedSample] = []
    class_totals: Counter = Counter()
    candidates: list[tuple[Path, Counter]] = []

    for split in ["train", "val", "test"]:
        for ann_path in sorted((root / split / "ann").glob("*.json")):
            counts = parse_vsai_counts(ann_path)
            if counts:
                candidates.append((ann_path, counts))
                class_totals.update(counts)

    for ann_path in select_class_balanced(candidates, limit):
        split = ann_path.parents[1].name
        img_path = ann_path.parents[1] / "img" / ann_path.name.removesuffix(".json")
        image = read_image(img_path)
        if image is None:
            continue
        annotated, counts = draw_vsai(image, ann_path)
        raw_dest = raw_out / split / img_path.name
        ann_dest = ann_out / split / f"{img_path.stem}_vsai_polygon.jpg"
        pair_dest = pairs_out / split / f"{img_path.stem}_pair.jpg"
        ensure_dir(raw_dest.parent)
        shutil.copy2(img_path, raw_dest)
        write_image(ann_dest, resize_for_preview(annotated))
        write_image(pair_dest, make_pair(image, annotated, "VSAI original polygon / OBB-like annotation"))
        samples.append(
            ExportedSample(
                dataset="VSAI",
                split=split,
                stem=img_path.stem,
                raw_rel=str(raw_dest.relative_to(out_root)),
                annotated_rel=str(ann_dest.relative_to(out_root)),
                classes=dict(counts),
                annotation_type="polygon: exterior points",
            )
        )
    return samples, {"classes": dict(class_totals), "annotation_type": "polygon"}


def zip_read_image(zf: zipfile.ZipFile, name: str) -> np.ndarray | None:
    data = np.frombuffer(zf.read(name), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def export_uav_obb(zip_path: Path, out_root: Path, limit: int) -> tuple[list[ExportedSample], dict]:
    dataset_out = out_root / "uav_obb"
    raw_out = dataset_out / "raw"
    ann_out = dataset_out / "annotated"
    pairs_out = dataset_out / "pairs"
    samples: list[ExportedSample] = []
    class_totals: Counter = Counter()
    candidates: list[tuple[str, Counter]] = []

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        label_names = sorted(n for n in names if n.startswith("UAV-OBB/") and "/labels/" in n and n.endswith(".txt"))
        for label_name in label_names:
            counts = draw_yolo_obb(np.zeros((1, 1, 3), dtype=np.uint8), zf.read(label_name).decode("utf-8"), UAV_OBB_NAMES)[1]
            if counts:
                candidates.append((label_name, counts))
                class_totals.update(counts)

        selected_labels = [p for p in select_class_balanced([(Path(p), c) for p, c in candidates], limit)]
        for label_path in selected_labels:
            label_name = str(label_path)
            split = label_name.split("/")[1]
            stem = Path(label_name).stem
            image_name = label_name.replace("/labels/", "/images/").removesuffix(".txt") + ".jpg"
            if image_name not in names:
                continue
            image = zip_read_image(zf, image_name)
            if image is None:
                continue
            label_text = zf.read(label_name).decode("utf-8")
            annotated, counts = draw_yolo_obb(image, label_text, UAV_OBB_NAMES)
            raw_dest = raw_out / split / Path(image_name).name
            ann_dest = ann_out / split / f"{stem}_uav_obb.jpg"
            pair_dest = pairs_out / split / f"{stem}_pair.jpg"
            ensure_dir(raw_dest.parent)
            raw_dest.write_bytes(zf.read(image_name))
            write_image(ann_dest, resize_for_preview(annotated))
            write_image(pair_dest, make_pair(image, annotated, "UAV-OBB original YOLO-OBB annotation"))
            samples.append(
                ExportedSample(
                    dataset="UAV-OBB",
                    split=split,
                    stem=stem,
                    raw_rel=str(raw_dest.relative_to(out_root)),
                    annotated_rel=str(ann_dest.relative_to(out_root)),
                    classes=dict(counts),
                    annotation_type="YOLO-OBB: class + normalized 4-point polygon",
                )
            )
    return samples, {"classes": dict(class_totals), "annotation_type": "YOLO-OBB"}


def make_contact_sheet(samples: list[ExportedSample], out_root: Path, name: str, max_items: int = 30) -> Path | None:
    selected = samples[:max_items]
    thumbs: list[np.ndarray] = []
    for sample in selected:
        image = read_image(out_root / sample.annotated_rel)
        if image is None:
            continue
        h, w = image.shape[:2]
        scale = min(1.0, 360 / max(h, w))
        thumb = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        canvas = np.full((300, 380, 3), 248, dtype=np.uint8)
        y = 36
        x = max(0, (380 - thumb.shape[1]) // 2)
        canvas[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb[: 300 - y, :380]
        cv2.putText(canvas, sample.stem[:42], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (35, 35, 35), 1, cv2.LINE_AA)
        thumbs.append(canvas)
    if not thumbs:
        return None
    cols = 3
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.full((rows * 300, cols * 380, 3), 255, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        sheet[r * 300 : (r + 1) * 300, c * 380 : (c + 1) * 380] = thumb
    dest = out_root / f"{name}_contact_sheet.jpg"
    write_image(dest, sheet)
    return dest


def make_pair_contact_sheet(samples: list[ExportedSample], out_root: Path, name: str, max_items: int = 18) -> Path | None:
    selected = samples[:max_items]
    thumbs: list[np.ndarray] = []
    for sample in selected:
        pair_path = out_root / sample.raw_rel.replace("/raw/", "/pairs/").rsplit(".", 1)[0]
        pair_candidates = sorted(pair_path.parent.glob(f"{sample.stem}*_pair.jpg")) if pair_path.parent.exists() else []
        if not pair_candidates:
            pair_candidates = sorted((out_root / sample.dataset.lower().replace("-", "_") / "pairs").glob(f"**/{sample.stem}*_pair.jpg"))
        if not pair_candidates:
            continue
        image = read_image(pair_candidates[0])
        if image is None:
            continue
        h, w = image.shape[:2]
        scale = min(1.0, 560 / max(h, w))
        thumb = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        canvas = np.full((360, 600, 3), 248, dtype=np.uint8)
        y = 34
        x = max(0, (600 - thumb.shape[1]) // 2)
        canvas[y : y + min(thumb.shape[0], 326), x : x + min(thumb.shape[1], 600)] = thumb[:326, :600]
        cv2.putText(canvas, sample.stem[:56], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (35, 35, 35), 1, cv2.LINE_AA)
        thumbs.append(canvas)
    if not thumbs:
        return None
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.full((rows * 360, cols * 600, 3), 255, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        sheet[r * 360 : (r + 1) * 360, c * 600 : (c + 1) * 600] = thumb
    dest = out_root / f"{name}_pair_contact_sheet.jpg"
    write_image(dest, sheet)
    return dest


def format_classes(classes: dict[str, int], zh: dict[str, str]) -> str:
    parts = []
    for name, count in sorted(classes.items(), key=lambda x: (-x[1], x[0])):
        parts.append(f"{html.escape(name)} / {html.escape(zh.get(name, ''))}: {count}")
    return "<br>".join(parts)


def write_index(out_root: Path, all_samples: list[ExportedSample], summaries: dict[str, dict]) -> None:
    by_dataset: dict[str, list[ExportedSample]] = defaultdict(list)
    for sample in all_samples:
        by_dataset[sample.dataset].append(sample)

    zh_maps = {"VisDrone": VISDRONE_ZH, "VSAI": VSAI_ZH, "UAV-OBB": UAV_OBB_ZH}
    rows = []
    sections = []
    for dataset, samples in by_dataset.items():
        summary = summaries[dataset]
        rows.append(
            f"<tr><td>{dataset}</td><td>{summary['annotation_type']}</td>"
            f"<td>{len(samples)}</td><td>{format_classes(summary['classes'], zh_maps.get(dataset, {}))}</td></tr>"
        )
        slug = dataset.lower().replace("-", "_")
        contact = f"{slug}_contact_sheet.jpg"
        pair_contact = f"{slug}_pair_contact_sheet.jpg"
        sections.append(
            f"<h2>{dataset}</h2><p><a href='{contact}'>打开标注拼图</a> | "
            f"<a href='{pair_contact}'>打开原图/标注对比拼图</a></p><div class='grid'>"
        )
        for sample in samples:
            classes = ", ".join(f"{k}:{v}" for k, v in sorted(sample.classes.items()))
            sections.append(
                "<div class='card'>"
                f"<a href='{html.escape(sample.raw_rel)}'><img src='{html.escape(sample.raw_rel)}'></a>"
                f"<a href='{html.escape(sample.annotated_rel)}'><img src='{html.escape(sample.annotated_rel)}'></a>"
                f"<p><b>{html.escape(sample.split)} / {html.escape(sample.stem)}</b><br>"
                f"{html.escape(sample.annotation_type)}<br>{html.escape(classes)}</p>"
                "</div>"
            )
        sections.append("</div>")

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>源数据集原图与原始标注抽样</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #20222b; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d8dbe3; padding: 8px 10px; vertical-align: top; }}
    th {{ background: #f3f5f8; text-align: left; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #d8dbe3; border-radius: 8px; padding: 10px; background: #fff; }}
    .card img {{ width: 49%; max-height: 220px; object-fit: contain; background: #f7f7f7; border: 1px solid #eceff3; }}
    .card p {{ font-size: 13px; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>源数据集原图与原始标注抽样</h1>
  <p>每个卡片左图是原始未标注图片，右图是按该数据集原始标注格式画出来的可视化结果。</p>
  <table>
    <thead><tr><th>数据集</th><th>原始标注格式</th><th>本次抽样数</th><th>原始类别统计</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  {''.join(sections)}
</body>
</html>
"""
    (out_root / "index.html").write_text(html_text, encoding="utf-8")


def write_markdown_summary(out_root: Path, summaries: dict[str, dict], counts: dict[str, int]) -> None:
    lines = ["# 源数据集原图与原始标注抽样\n"]
    lines.append("每个数据集均输出 `raw/` 原图、`annotated/` 原始标注可视化图、`pairs/` 左右对比图，以及 contact sheet。\n")
    for dataset, summary in summaries.items():
        lines.append(f"## {dataset}\n")
        lines.append(f"- 抽样数量: {counts.get(dataset, 0)}\n")
        lines.append(f"- 原始标注格式: {summary['annotation_type']}\n")
        lines.append("- 类别统计:\n")
        for name, count in sorted(summary["classes"].items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"  - {name}: {count}\n")
        lines.append("\n")
    (out_root / "README.md").write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("dataset_construction/derived/source_annotation_samples_v1"))
    parser.add_argument("--samples-per-dataset", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260615)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    ensure_dir(args.out_root)

    all_samples: list[ExportedSample] = []
    summaries: dict[str, dict] = {}

    vis_root = Path("dataset_construction/data_sources/visdrone/raw")
    if vis_root.exists():
        samples, summary = export_visdrone(vis_root, args.out_root, args.samples_per_dataset)
        all_samples.extend(samples)
        summaries["VisDrone"] = summary
        make_contact_sheet(samples, args.out_root, "visdrone")
        make_pair_contact_sheet(samples, args.out_root, "visdrone")

    vsai_root = Path("dataset_construction/data_sources/vsai/downloads/vsai-DatasetNinja")
    if vsai_root.exists():
        samples, summary = export_vsai(vsai_root, args.out_root, args.samples_per_dataset)
        all_samples.extend(samples)
        summaries["VSAI"] = summary
        make_contact_sheet(samples, args.out_root, "vsai")
        make_pair_contact_sheet(samples, args.out_root, "vsai")

    uav_zip = Path("dataset_construction/data_sources/uav_obb/downloads/UAV-OBB-dlaCi7.zip")
    if uav_zip.exists():
        samples, summary = export_uav_obb(uav_zip, args.out_root, args.samples_per_dataset)
        all_samples.extend(samples)
        summaries["UAV-OBB"] = summary
        make_contact_sheet(samples, args.out_root, "uav_obb")
        make_pair_contact_sheet(samples, args.out_root, "uav_obb")

    write_index(args.out_root, all_samples, summaries)
    write_markdown_summary(args.out_root, summaries, Counter(s.dataset for s in all_samples))
    manifest = {
        "out_root": str(args.out_root),
        "samples": [sample.__dict__ for sample in all_samples],
        "summaries": summaries,
    }
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_root": str(args.out_root), "samples": len(all_samples), "datasets": sorted(summaries)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
