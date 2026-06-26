#!/usr/bin/env python3
"""Generate pseudo YOLO-OBB labels from VisDrone HBB annotations.

The default path uses OpenCV GrabCut so the pipeline is runnable without SAM
weights. If Segment Anything is installed and a checkpoint is provided, use
``--segmenter sam`` or ``--segmenter sam2`` for box-prompt segmentation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


VISDRONE_TO_GLOBAL = {
    3: (1, "bicycle"),
    7: (3, "tricycle"),
    8: (4, "awning_tricycle"),
    10: (2, "motor"),
}

GLOBAL_NAMES = {
    0: "motor_vehicle",
    1: "bicycle",
    2: "motor",
    3: "tricycle",
    4: "awning_tricycle",
}

ASPECT_PRIORS = {
    "bicycle": (1.2, 8.0),
    "motor": (1.1, 8.0),
    "tricycle": (1.0, 6.0),
    "awning_tricycle": (1.0, 5.0),
}


@dataclass(frozen=True)
class VisDroneObject:
    x: float
    y: float
    w: float
    h: float
    score: int
    class_id: int
    truncation: int
    occlusion: int
    line_index: int

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.x + self.w, self.y + self.h


class Segmenter:
    def segment(
        self,
        image_bgr: np.ndarray,
        hbb_xyxy: tuple[int, int, int, int],
        expanded_xyxy: tuple[int, int, int, int],
    ) -> np.ndarray:
        raise NotImplementedError


class GrabCutSegmenter(Segmenter):
    def __init__(self, iterations: int = 3) -> None:
        self.iterations = iterations

    def segment(
        self,
        image_bgr: np.ndarray,
        hbb_xyxy: tuple[int, int, int, int],
        expanded_xyxy: tuple[int, int, int, int],
    ) -> np.ndarray:
        ex1, ey1, ex2, ey2 = expanded_xyxy
        x1, y1, x2, y2 = hbb_xyxy
        crop = image_bgr[ey1:ey2, ex1:ex2]
        if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
            return np.zeros((max(ey2 - ey1, 1), max(ex2 - ex1, 1)), dtype=np.uint8)

        rx = max(int(round(x1 - ex1)), 0)
        ry = max(int(round(y1 - ey1)), 0)
        rw = max(int(round(x2 - x1)), 1)
        rh = max(int(round(y2 - y1)), 1)
        rw = min(rw, crop.shape[1] - rx)
        rh = min(rh, crop.shape[0] - ry)
        if rw < 2 or rh < 2:
            mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            mask[ry : ry + max(rh, 1), rx : rx + max(rw, 1)] = 255
            return mask

        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        bg_model = np.zeros((1, 65), dtype=np.float64)
        fg_model = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.grabCut(
                crop,
                mask,
                (rx, ry, rw, rh),
                bg_model,
                fg_model,
                self.iterations,
                cv2.GC_INIT_WITH_RECT,
            )
            fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(
                np.uint8
            )
            return postprocess_mask(fg)
        except cv2.error:
            fallback = np.zeros(crop.shape[:2], dtype=np.uint8)
            fallback[ry : ry + rh, rx : rx + rw] = 255
            return fallback


class SamSegmenter(Segmenter):
    def __init__(self, checkpoint: str, model_type: str, device: str) -> None:
        from segment_anything import SamPredictor, sam_model_registry  # type: ignore

        model = sam_model_registry[model_type](checkpoint=checkpoint)
        model.to(device=device)
        self.predictor = SamPredictor(model)
        self._last_image_id: int | None = None

    def segment(
        self,
        image_bgr: np.ndarray,
        hbb_xyxy: tuple[int, int, int, int],
        expanded_xyxy: tuple[int, int, int, int],
    ) -> np.ndarray:
        image_id = id(image_bgr)
        if image_id != self._last_image_id:
            self.predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            self._last_image_id = image_id
        masks, scores, _ = self.predictor.predict(
            box=np.asarray(expanded_xyxy, dtype=np.float32),
            multimask_output=True,
        )
        best = masks[int(np.argmax(scores))].astype(np.uint8) * 255
        ex1, ey1, ex2, ey2 = expanded_xyxy
        return postprocess_mask(best[ey1:ey2, ex1:ex2])


class Sam2Segmenter(Segmenter):
    def __init__(self, checkpoint: str, config: str, device: str) -> None:
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        model = build_sam2(config, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(model)
        self._last_image_id: int | None = None

    def segment(
        self,
        image_bgr: np.ndarray,
        hbb_xyxy: tuple[int, int, int, int],
        expanded_xyxy: tuple[int, int, int, int],
    ) -> np.ndarray:
        image_id = id(image_bgr)
        if image_id != self._last_image_id:
            self.predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            self._last_image_id = image_id
        masks, scores, _ = self.predictor.predict(
            box=np.asarray(expanded_xyxy, dtype=np.float32),
            multimask_output=True,
        )
        best = masks[int(np.argmax(scores))].astype(np.uint8) * 255
        ex1, ey1, ex2, ey2 = expanded_xyxy
        return postprocess_mask(best[ey1:ey2, ex1:ex2])


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(np.uint8)


def make_segmenter(args: argparse.Namespace) -> Segmenter:
    if args.segmenter == "grabcut":
        return GrabCutSegmenter(iterations=args.grabcut_iterations)
    if args.segmenter == "sam":
        if not args.sam_checkpoint:
            raise ValueError("--sam-checkpoint is required for --segmenter sam")
        return SamSegmenter(args.sam_checkpoint, args.sam_model_type, args.device)
    if args.segmenter == "sam2":
        if not args.sam_checkpoint or not args.sam2_config:
            raise ValueError("--sam-checkpoint and --sam2-config are required for --segmenter sam2")
        return Sam2Segmenter(args.sam_checkpoint, args.sam2_config, args.device)
    if args.segmenter == "auto":
        if args.sam_checkpoint and args.sam2_config:
            return Sam2Segmenter(args.sam_checkpoint, args.sam2_config, args.device)
        if args.sam_checkpoint:
            return SamSegmenter(args.sam_checkpoint, args.sam_model_type, args.device)
        return GrabCutSegmenter(iterations=args.grabcut_iterations)
    raise ValueError(f"Unknown segmenter: {args.segmenter}")


def parse_annotation(path: Path) -> list[VisDroneObject]:
    objects: list[VisDroneObject] = []
    for idx, line in enumerate(path.read_text(errors="ignore").splitlines()):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 8:
            continue
        try:
            x, y, w, h = map(float, parts[:4])
            score, class_id, truncation, occlusion = map(int, parts[4:8])
        except ValueError:
            continue
        objects.append(VisDroneObject(x, y, w, h, score, class_id, truncation, occlusion, idx))
    return objects


def resolve_split_dir(raw_root: Path, split: str) -> Path:
    direct = raw_root / f"VisDrone2019-DET-{split}"
    nested = direct / f"VisDrone2019-DET-{split}"
    for candidate in (direct, nested):
        if (candidate / "images").is_dir() and (candidate / "annotations").is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find VisDrone split directory for {split} under {raw_root}")


def expand_box(
    box: tuple[float, float, float, float],
    image_w: int,
    image_h: int,
    ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    dx = bw * ratio
    dy = bh * ratio
    ex1 = int(max(math.floor(x1 - dx), 0))
    ey1 = int(max(math.floor(y1 - dy), 0))
    ex2 = int(min(math.ceil(x2 + dx), image_w))
    ey2 = int(min(math.ceil(y2 + dy), image_h))
    return ex1, ey1, max(ex2, ex1 + 1), max(ey2, ey1 + 1)


def select_component(mask: np.ndarray, hbb_center_local: tuple[float, float]) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return np.zeros_like(binary, dtype=np.uint8)
    hx, hy = hbb_center_local
    best_idx = None
    best_score = -1.0
    for idx in range(1, num):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        cx, cy = centroids[idx]
        dist = math.hypot(cx - hx, cy - hy)
        score = area / (1.0 + dist)
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is None:
        return np.zeros_like(binary, dtype=np.uint8)
    return (labels == best_idx).astype(np.uint8) * 255


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def order_points_clockwise(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = np.argmin(ordered[:, 0] + ordered[:, 1])
    return np.roll(ordered, -start, axis=0)


def obb_from_mask(
    component_mask: np.ndarray,
    expanded_xyxy: tuple[int, int, int, int],
) -> tuple[np.ndarray | None, tuple[float, float, float] | None]:
    ys, xs = np.where(component_mask > 0)
    if len(xs) < 4:
        return None, None
    ex1, ey1, _, _ = expanded_xyxy
    pts = np.column_stack([xs + ex1, ys + ey1]).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect).astype(np.float32)
    box = order_points_clockwise(box)
    width, height = rect[1]
    short = max(min(width, height), 1e-6)
    aspect = max(width, height) / short
    return box, (float(width), float(height), float(aspect))


def normalize_points(points: np.ndarray, image_w: int, image_h: int) -> list[float]:
    norm = points.astype(np.float32).copy()
    norm[:, 0] = np.clip(norm[:, 0], 0, image_w - 1) / max(image_w, 1)
    norm[:, 1] = np.clip(norm[:, 1], 0, image_h - 1) / max(image_h, 1)
    return [round(float(v), 6) for v in norm.reshape(-1)]


def score_quality(
    obj: VisDroneObject,
    class_name: str,
    obb_points: np.ndarray | None,
    rect_info: tuple[float, float, float] | None,
    component_mask: np.ndarray,
    expanded_xyxy: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
) -> dict:
    flags: list[str] = []
    hbb_area = max(obj.w * obj.h, 1.0)
    hbb_diag = max(math.hypot(obj.w, obj.h), 1.0)
    mask_area = float(np.count_nonzero(component_mask))
    if obb_points is None or rect_info is None or mask_area <= 0:
        return {
            "quality_score": 0.0,
            "quality_status": "needs_review",
            "flags": ["mask_empty"],
            "metrics": {
                "hbb_area": hbb_area,
                "mask_area": mask_area,
                "obb_area": 0.0,
                "area_ratio": 0.0,
                "center_shift": 1.0,
                "foreground_ratio": 0.0,
                "aspect_ratio": 0.0,
            },
        }

    clipped = obb_points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, image_w - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, image_h - 1)
    clip_delta = float(np.abs(clipped - obb_points).sum())
    obb_area = max(polygon_area(clipped), 1.0)
    area_ratio = obb_area / hbb_area
    obb_center = clipped.mean(axis=0)
    hbb_center = np.asarray([obj.x + obj.w / 2.0, obj.y + obj.h / 2.0])
    center_shift = float(np.linalg.norm(obb_center - hbb_center) / hbb_diag)
    foreground_ratio = float(mask_area / obb_area)
    aspect_ratio = rect_info[2]
    min_aspect, max_aspect = ASPECT_PRIORS[class_name]

    if area_ratio < 0.25:
        flags.append("area_ratio_too_small")
    if area_ratio > 1.50:
        flags.append("area_ratio_too_large")
    if center_shift > 0.35:
        flags.append("center_shift_too_large")
    if foreground_ratio < 0.20:
        flags.append("foreground_ratio_too_low")
    if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
        flags.append("aspect_ratio_out_of_prior")
    if clip_delta > 1e-3:
        flags.append("obb_clipped")
    if obj.occlusion >= 2:
        flags.append("heavy_occlusion")
    if obj.truncation >= 2:
        flags.append("heavy_truncation")

    area_score = max(0.0, 1.0 - min(abs(area_ratio - 0.8), 1.0))
    center_score = max(0.0, 1.0 - min(center_shift / 0.5, 1.0))
    fg_score = min(max(foreground_ratio / 0.55, 0.0), 1.0)
    if min_aspect <= aspect_ratio <= max_aspect:
        aspect_score = 1.0
    else:
        aspect_score = 0.25
    quality_score = 0.35 * area_score + 0.25 * center_score + 0.25 * fg_score + 0.15 * aspect_score
    critical_flags = [f for f in flags if f not in {"heavy_occlusion", "heavy_truncation"}]
    status = "auto_accept" if quality_score >= 0.55 and not critical_flags else "needs_review"

    return {
        "quality_score": round(float(quality_score), 4),
        "quality_status": status,
        "flags": flags,
        "metrics": {
            "hbb_area": round(float(hbb_area), 3),
            "mask_area": round(float(mask_area), 3),
            "obb_area": round(float(obb_area), 3),
            "area_ratio": round(float(area_ratio), 4),
            "center_shift": round(float(center_shift), 4),
            "foreground_ratio": round(float(foreground_ratio), 4),
            "aspect_ratio": round(float(aspect_ratio), 4),
        },
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def link_or_copy_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "none":
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
        return
    raise ValueError(f"Unknown copy mode: {mode}")


def draw_preview(
    image_bgr: np.ndarray,
    hbb_xyxy: tuple[int, int, int, int],
    obb_points: np.ndarray | None,
    component_mask: np.ndarray,
    expanded_xyxy: tuple[int, int, int, int],
    label: str,
    quality_score: float,
) -> np.ndarray:
    canvas = image_bgr.copy()
    ex1, ey1, ex2, ey2 = expanded_xyxy
    if component_mask.size:
        overlay = np.zeros_like(canvas)
        crop_overlay = overlay[ey1:ey2, ex1:ex2]
        crop_overlay[component_mask > 0] = (0, 180, 255)
        canvas = cv2.addWeighted(canvas, 1.0, overlay, 0.35, 0)
    x1, y1, x2, y2 = hbb_xyxy
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
    if obb_points is not None:
        pts = np.round(obb_points).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(
        canvas,
        f"{label} q={quality_score:.2f}",
        (max(x1, 0), max(y1 - 6, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def save_data_yaml(yolo_root: Path) -> None:
    data = {
        "path": str(yolo_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": GLOBAL_NAMES,
    }
    with (yolo_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def generate(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    pseudo_root = Path(args.pseudo_root)
    yolo_root = Path(args.yolo_root)
    pseudo_root.mkdir(parents=True, exist_ok=True)
    yolo_root.mkdir(parents=True, exist_ok=True)

    segmenter = make_segmenter(args)
    rng = random.Random(args.seed)
    all_records: list[dict] = []
    review_candidates: list[dict] = []
    labels_by_image: dict[tuple[str, str], list[str]] = defaultdict(list)
    counters: Counter[str] = Counter()

    for split in args.splits:
        split_dir = resolve_split_dir(raw_root, split)
        image_dir = split_dir / "images"
        ann_dir = split_dir / "annotations"
        images = sorted(image_dir.glob("*.jpg"))
        if args.max_images:
            images = images[: args.max_images]
        missing = [p.stem for p in images if not (ann_dir / f"{p.stem}.txt").exists()]
        if missing:
            raise RuntimeError(f"{split}: {len(missing)} images have no annotation, first={missing[:3]}")

        for image_index, image_path in enumerate(images, start=1):
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                counters["image_read_failed"] += 1
                continue
            image_h, image_w = image_bgr.shape[:2]
            yolo_image_path = yolo_root / "images" / split / image_path.name
            link_or_copy_image(image_path, yolo_image_path, args.copy_mode)
            objects = parse_annotation(ann_dir / f"{image_path.stem}.txt")

            for obj in objects:
                if obj.class_id not in VISDRONE_TO_GLOBAL:
                    continue
                global_id, class_name = VISDRONE_TO_GLOBAL[obj.class_id]
                hbb_xyxy_float = obj.xyxy
                hbb_xyxy = tuple(int(round(v)) for v in hbb_xyxy_float)
                expanded = expand_box(hbb_xyxy_float, image_w, image_h, args.expand_ratio)
                ex1, ey1, _, _ = expanded
                mask = segmenter.segment(image_bgr, hbb_xyxy, expanded)
                hbb_center_local = (obj.x + obj.w / 2.0 - ex1, obj.y + obj.h / 2.0 - ey1)
                component = select_component(mask, hbb_center_local)
                obb_points, rect_info = obb_from_mask(component, expanded)
                quality = score_quality(
                    obj,
                    class_name,
                    obb_points,
                    rect_info,
                    component,
                    expanded,
                    image_w,
                    image_h,
                )
                if obb_points is not None:
                    obb_points[:, 0] = np.clip(obb_points[:, 0], 0, image_w - 1)
                    obb_points[:, 1] = np.clip(obb_points[:, 1], 0, image_h - 1)
                    yolo_values = normalize_points(obb_points, image_w, image_h)
                else:
                    yolo_values = []

                sample_id = f"{split}__{image_path.stem}__{obj.line_index:04d}"
                mask_rel = None
                if args.save_mask_crops and component.size:
                    mask_path = pseudo_root / "masks" / split / f"{sample_id}.png"
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(mask_path), component)
                    mask_rel = str(mask_path.relative_to(pseudo_root))

                record = {
                    "sample_id": sample_id,
                    "split": split,
                    "source_dataset": "VisDrone2019-DET",
                    "source_image": str(image_path),
                    "image_name": image_path.name,
                    "image_width": image_w,
                    "image_height": image_h,
                    "annotation_line_index": obj.line_index,
                    "visdrone_class_id": obj.class_id,
                    "class_id": global_id,
                    "class_name": class_name,
                    "hbb_xywh": [round(obj.x, 3), round(obj.y, 3), round(obj.w, 3), round(obj.h, 3)],
                    "hbb_xyxy": [round(float(v), 3) for v in hbb_xyxy_float],
                    "expanded_xyxy": list(expanded),
                    "obb_points": (
                        [[round(float(x), 3), round(float(y), 3)] for x, y in obb_points]
                        if obb_points is not None
                        else []
                    ),
                    "yolo_obb": [global_id, *yolo_values] if yolo_values else [],
                    "mask_crop": mask_rel,
                    "segmenter": args.segmenter,
                    "quality": quality,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                all_records.append(record)
                counters[f"{split}_{class_name}"] += 1

                queue_reason = None
                if quality["quality_status"] == "needs_review":
                    queue_reason = "low_quality"
                elif rng.random() < args.review_sample_rate:
                    queue_reason = "random_sample"

                if quality["quality_status"] == "auto_accept" and yolo_values:
                    labels_by_image[(split, image_path.stem)].append(
                        " ".join([str(global_id), *[f"{v:.6f}" for v in yolo_values]])
                    )

                if queue_reason:
                    record_for_queue = dict(record)
                    record_for_queue["queue_reason"] = queue_reason
                    review_candidates.append(record_for_queue)
                    preview_path = pseudo_root / "previews" / split / f"{sample_id}.jpg"
                    preview_path.parent.mkdir(parents=True, exist_ok=True)
                    preview = draw_preview(
                        image_bgr,
                        hbb_xyxy,
                        obb_points,
                        component,
                        expanded,
                        class_name,
                        float(quality["quality_score"]),
                    )
                    cv2.imwrite(str(preview_path), preview)
                    record_for_queue["preview"] = str(preview_path.relative_to(pseudo_root))

            if args.progress_every and image_index % args.progress_every == 0:
                print(f"{split}: processed {image_index}/{len(images)} images", flush=True)

    labels_root = yolo_root / "labels"
    for split in args.splits:
        split_dir = resolve_split_dir(raw_root, split)
        image_paths = sorted((split_dir / "images").glob("*.jpg"))
        if args.max_images:
            image_paths = image_paths[: args.max_images]
        for image_path in image_paths:
            label_path = labels_root / split / f"{image_path.stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            rows = labels_by_image.get((split, image_path.stem), [])
            label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    write_jsonl(pseudo_root / "quality.jsonl", all_records)
    write_jsonl(pseudo_root / "review_queue.jsonl", review_candidates)
    save_data_yaml(yolo_root)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root),
        "pseudo_root": str(pseudo_root),
        "yolo_root": str(yolo_root),
        "splits": args.splits,
        "segmenter": args.segmenter,
        "expand_ratio": args.expand_ratio,
        "copy_mode": args.copy_mode,
        "records": len(all_records),
        "review_queue": len(review_candidates),
        "counters": dict(counters),
        "global_names": GLOBAL_NAMES,
    }
    (pseudo_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def validate(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    for split in args.splits:
        split_dir = resolve_split_dir(raw_root, split)
        images = sorted((split_dir / "images").glob("*.jpg"))
        anns = sorted((split_dir / "annotations").glob("*.txt"))
        image_stems = {p.stem for p in images}
        ann_stems = {p.stem for p in anns}
        counts: Counter[int] = Counter()
        for ann in anns:
            for obj in parse_annotation(ann):
                counts[obj.class_id] += 1
        print(
            json.dumps(
                {
                    "split": split,
                    "dir": str(split_dir),
                    "images": len(images),
                    "annotations": len(anns),
                    "missing_annotations": len(image_stems - ann_stems),
                    "missing_images": len(ann_stems - image_stems),
                    "target_counts": {
                        VISDRONE_TO_GLOBAL[cid][1]: counts[cid]
                        for cid in sorted(VISDRONE_TO_GLOBAL)
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    default_raw = "dataset_construction/data_sources/visdrone/raw"
    default_pseudo = "dataset_construction/derived/visdrone_pseudo_obb_v1"
    default_yolo = "dataset_construction/derived/visdrone_yolo_obb_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate_parser = sub.add_parser("validate", help="Validate VisDrone split structure")
    validate_parser.add_argument("--raw-root", default=default_raw)
    validate_parser.add_argument("--splits", nargs="+", default=["train", "val"])
    validate_parser.set_defaults(func=validate)

    gen = sub.add_parser("generate", help="Generate pseudo OBB labels")
    gen.add_argument("--raw-root", default=default_raw)
    gen.add_argument("--pseudo-root", default=default_pseudo)
    gen.add_argument("--yolo-root", default=default_yolo)
    gen.add_argument("--splits", nargs="+", default=["train", "val"])
    gen.add_argument("--segmenter", choices=["auto", "grabcut", "sam", "sam2"], default="auto")
    gen.add_argument("--sam-checkpoint", default=None)
    gen.add_argument("--sam-model-type", default="vit_h")
    gen.add_argument("--sam2-config", default=None)
    gen.add_argument("--device", default="cuda")
    gen.add_argument("--grabcut-iterations", type=int, default=3)
    gen.add_argument("--expand-ratio", type=float, default=0.10)
    gen.add_argument("--review-sample-rate", type=float, default=0.02)
    gen.add_argument("--save-mask-crops", action=argparse.BooleanOptionalAction, default=True)
    gen.add_argument("--copy-mode", choices=["symlink", "copy", "none"], default="symlink")
    gen.add_argument("--max-images", type=int, default=None)
    gen.add_argument("--seed", type=int, default=20260614)
    gen.add_argument("--progress-every", type=int, default=100)
    gen.set_defaults(func=generate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
