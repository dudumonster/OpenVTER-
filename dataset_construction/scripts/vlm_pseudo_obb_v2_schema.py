#!/usr/bin/env python3
"""VLM/grounding pseudo OBB v2: JSONL schema, quality scoring, and export utilities.

This module defines the v2 data structures used across the VLM-assisted pseudo
OBB pipeline.  It does *not* contain model inference code — those live in
separate backend modules under ``dataset_construction/scripts/vlm_backends/``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np


# ---------------------------------------------------------------------------
# Constants (shared with v1, kept here for v2 independence)
# ---------------------------------------------------------------------------

VISDRONE_TO_GLOBAL: dict[int, tuple[int, str]] = {
    3: (1, "bicycle"),
    7: (3, "tricycle"),
    8: (4, "awning_tricycle"),
    10: (2, "motor"),
}

GLOBAL_NAMES: dict[int, str] = {
    0: "motor_vehicle",
    1: "bicycle",
    2: "motor",
    3: "tricycle",
    4: "awning_tricycle",
}

ASPECT_PRIORS: dict[str, tuple[float, float]] = {
    "bicycle": (1.2, 8.0),
    "motor": (1.1, 8.0),
    "tricycle": (1.0, 6.0),
    "awning_tricycle": (1.0, 5.0),
}

VLM_BACKENDS = (
    "grounded_sam2",
    "groundingdino_sam2",
    "florence2_sam2",
    "yoloworld_sam2",
    "qwen",
    "gemini",
    "wan",
)

ReviewDecision = Literal["auto_accept", "review", "reject"]
ReviewStatus = Literal["auto_accept", "review", "reject",
                       "accepted_on_review", "edited_on_review",
                       "rejected_on_review"]


# ---------------------------------------------------------------------------
# V2 JSONL record schema
# ---------------------------------------------------------------------------

@dataclass
class VLMQualityMetrics:
    """Quality metrics for a single pseudo OBB sample (v2 expanded schema).

    All fields are serialised to ``quality.jsonl``.
    """

    # Geometry (inherited from v1)
    hbb_area: float = 0.0
    mask_area: float = 0.0
    obb_area: float = 0.0
    area_ratio: float = 0.0
    center_shift: float = 0.0
    foreground_ratio: float = 0.0
    aspect_ratio: float = 0.0

    # v2 additions
    boundary_clip_ratio: float = 0.0      # fraction of OBB clipped by image border
    mask_solidity: float = 0.0            # mask_area / convex_hull_area (proxy for shape quality)

    # VLM / semantic metrics
    class_confidence: float = 0.0         # VLM class confidence
    vlm_box_confidence: float = 0.0       # grounding box confidence
    vlm_box_iou: float = 0.0             # IoU between VLM box and SAM2 box
    vlm_class_name: str = ""             # VLM-predicted class name
    vlm_class_agrees_with_hbb: bool = False

    # Composite scores
    geometry_score: float = 0.0
    semantic_score: float = 0.0
    final_score: float = 0.0

    # Flags
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hbb_area": round(self.hbb_area, 3),
            "mask_area": round(self.mask_area, 3),
            "obb_area": round(self.obb_area, 3),
            "area_ratio": round(self.area_ratio, 4),
            "center_shift": round(self.center_shift, 4),
            "foreground_ratio": round(self.foreground_ratio, 4),
            "aspect_ratio": round(self.aspect_ratio, 4),
            "boundary_clip_ratio": round(self.boundary_clip_ratio, 4),
            "mask_solidity": round(self.mask_solidity, 4),
            "class_confidence": round(self.class_confidence, 4),
            "vlm_box_confidence": round(self.vlm_box_confidence, 4),
            "vlm_box_iou": round(self.vlm_box_iou, 4),
            "vlm_class_name": self.vlm_class_name,
            "vlm_class_agrees_with_hbb": self.vlm_class_agrees_with_hbb,
            "geometry_score": round(self.geometry_score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "final_score": round(self.final_score, 4),
            "flags": self.flags,
        }


@dataclass
class VLMPseudoObbRecord:
    """Complete v2 record for a single VisDrone HBB → OBB conversion.

    Serialised one-per-line in ``quality.jsonl``.
    """

    # Identity
    sample_id: str = ""                     # e.g. "val__0000001_0012__0003"
    source_dataset: str = "VisDrone2019-DET"
    split: str = ""                         # train | val | test-dev

    # Source image & annotation
    image_path: str = ""
    annotation_path: str = ""
    image_width: int = 0
    image_height: int = 0

    # Source HBB (VisDrone)
    source_hbb_xywh: list[float] = field(default_factory=list)   # [x, y, w, h]
    source_hbb_xyxy: list[float] = field(default_factory=list)   # [x1, y1, x2, y2]
    source_class_id: int = -1                                     # VisDrone class_id
    source_class_name: str = ""
    source_occlusion: int = 0
    source_truncation: int = 0
    annotation_line_index: int = -1

    # Target class (OBB training)
    target_class_id: int = -1               # 1-4
    target_class_name: str = ""

    # Crop
    crop_box_xyxy: list[int] = field(default_factory=list)       # expanded crop in global coords
    crop_scale: float = 1.0                 # resize scale if crop was resized for VLM

    # VLM / grounding outputs
    vlm_backend: str = ""                   # grounded_sam2 | florence2_sam2 | qwen | gemini | ...
    text_prompt: str = ""
    vlm_box_xyxy: list[float] = field(default_factory=list)      # VLM refined box in global coords
    vlm_box_confidence: float = 0.0
    vlm_class_name: str = ""
    vlm_class_confidence: float = 0.0

    # SAM2 mask
    mask_path: str = ""                     # relative path to mask crop PNG

    # Output OBB
    obb_points: list[list[float]] = field(default_factory=list)  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    yolo_obb: list[float] = field(default_factory=list)          # [class_id, x1_norm, y1_norm, ...]

    # Quality
    quality: VLMQualityMetrics = field(default_factory=VLMQualityMetrics)
    review_status: str = ""                 # auto_accept | review | reject
    queue_reason: str = ""                  # why this sample entered review queue
    failure_reasons: list[str] = field(default_factory=list)

    # Metadata
    preview_path: str = ""                  # relative path to review preview image
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_dataset": self.source_dataset,
            "split": self.split,
            "image_path": self.image_path,
            "annotation_path": self.annotation_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "source_hbb_xywh": [round(v, 3) for v in self.source_hbb_xywh],
            "source_hbb_xyxy": [round(v, 3) for v in self.source_hbb_xyxy],
            "source_class_id": self.source_class_id,
            "source_class_name": self.source_class_name,
            "source_occlusion": self.source_occlusion,
            "source_truncation": self.source_truncation,
            "annotation_line_index": self.annotation_line_index,
            "target_class_id": self.target_class_id,
            "target_class_name": self.target_class_name,
            "crop_box_xyxy": self.crop_box_xyxy,
            "crop_scale": self.crop_scale,
            "vlm_backend": self.vlm_backend,
            "text_prompt": self.text_prompt,
            "vlm_box_xyxy": [round(v, 3) for v in self.vlm_box_xyxy],
            "vlm_box_confidence": round(self.vlm_box_confidence, 4),
            "vlm_class_name": self.vlm_class_name,
            "vlm_class_confidence": round(self.vlm_class_confidence, 4),
            "mask_path": self.mask_path,
            "obb_points": [[round(x, 3), round(y, 3)] for x, y in self.obb_points] if self.obb_points else [],
            "yolo_obb": self.yolo_obb,
            "quality": self.quality.to_dict(),
            "review_status": self.review_status,
            "queue_reason": self.queue_reason,
            "failure_reasons": self.failure_reasons,
            "preview_path": self.preview_path,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Quality scoring (v2 — geometry + semantics)
# ---------------------------------------------------------------------------

def score_v2_quality(
    class_name: str,
    obb_points: np.ndarray | None,
    rect_info: tuple[float, float, float] | None,
    component_mask: np.ndarray,
    expanded_xyxy: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    hbb_xywh: tuple[float, float, float, float],
    occlusion: int,
    truncation: int,
    vlm_class_name: str = "",
    vlm_class_confidence: float = 0.0,
    vlm_box_confidence: float = 0.0,
    vlm_box_xyxy: list[float] | None = None,
) -> VLMQualityMetrics:
    """Compute v2 quality metrics combining geometry and VLM semantic signals.

    Parameters
    ----------
    class_name : str
        HBB class name (bicycle, motor, tricycle, awning_tricycle).
    obb_points : np.ndarray or None
        (4,2) OBB corner points in global image coords, or None if no OBB.
    rect_info : tuple or None
        (width, height, aspect_ratio) from cv2.minAreaRect, or None.
    component_mask : np.ndarray
        Binary mask of the segmented foreground (crop-relative).
    expanded_xyxy : tuple
        (ex1, ey1, ex2, ey2) crop box in global coords.
    image_w, image_h : int
        Full image dimensions.
    hbb_xywh : tuple
        (x, y, w, h) of the original VisDrone HBB.
    occlusion, truncation : int
        VisDrone occlusion/truncation scores (0-2).
    vlm_class_name : str
        VLM-predicted class name (empty if no VLM used).
    vlm_class_confidence : float
        VLM class confidence [0,1].
    vlm_box_confidence : float
        Grounding box confidence [0,1].
    vlm_box_xyxy : list[float] or None
        VLM predicted box [x1, y1, x2, y2] in global coords.

    Returns
    -------
    VLMQualityMetrics
    """
    metrics = VLMQualityMetrics()
    flags: list[str] = []

    hbb_w, hbb_h = hbb_xywh[2], hbb_xywh[3]
    hbb_area = max(hbb_w * hbb_h, 1.0)
    hbb_diag = max(math.hypot(hbb_w, hbb_h), 1.0)
    metrics.hbb_area = float(hbb_area)

    mask_area = float(np.count_nonzero(component_mask))
    metrics.mask_area = float(mask_area)

    if obb_points is None or rect_info is None or mask_area <= 0:
        flags.append("mask_empty")
        metrics.flags = flags
        return metrics  # all scores stay at 0.0

    # --- Geometry metrics ---
    clipped = obb_points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, image_w - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, image_h - 1)
    clip_delta = float(np.abs(clipped - obb_points).sum())
    metrics.boundary_clip_ratio = round(clip_delta / max(np.sum(np.abs(obb_points)), 1e-6), 4)

    obb_area = max(_polygon_area(clipped), 1.0)
    metrics.obb_area = float(obb_area)

    area_ratio = obb_area / hbb_area
    metrics.area_ratio = round(float(area_ratio), 4)

    obb_center = clipped.mean(axis=0)
    hbb_center = np.asarray([hbb_xywh[0] + hbb_w / 2.0, hbb_xywh[1] + hbb_h / 2.0])
    center_shift = float(np.linalg.norm(obb_center - hbb_center) / hbb_diag)
    metrics.center_shift = round(center_shift, 4)

    foreground_ratio = float(mask_area / obb_area)
    metrics.foreground_ratio = round(foreground_ratio, 4)

    aspect_ratio = rect_info[2]
    metrics.aspect_ratio = round(float(aspect_ratio), 4)

    # mask solidity (convex hull area ratio)
    hull_area = _convex_hull_area(component_mask)
    metrics.mask_solidity = round(mask_area / max(hull_area, 1e-6), 4)

    # --- VLM semantic metrics ---
    metrics.vlm_class_name = vlm_class_name
    metrics.vlm_class_confidence = round(vlm_class_confidence, 4)
    metrics.vlm_box_confidence = round(vlm_box_confidence, 4)
    metrics.vlm_class_agrees_with_hbb = (vlm_class_name == class_name)

    # VLM box vs SAM2 OBB IoU (rough: compare VLM box area with OBB area)
    if vlm_box_xyxy and len(vlm_box_xyxy) == 4:
        vlm_box_area = max((vlm_box_xyxy[2] - vlm_box_xyxy[0]) * (vlm_box_xyxy[3] - vlm_box_xyxy[1]), 1.0)
        iou_area = min(obb_area, vlm_box_area) / max(obb_area, vlm_box_area, 1e-6)
        metrics.vlm_box_iou = round(float(iou_area), 4)

    # --- Geometry flags ---
    min_aspect, max_aspect = ASPECT_PRIORS.get(class_name, (1.0, 8.0))
    if area_ratio < 0.15:
        flags.append("area_ratio_too_small")
    if area_ratio > 2.00:
        flags.append("area_ratio_too_large")
    if area_ratio > 3.00:
        flags.append("area_ratio_extreme")
    if center_shift > 0.40:
        flags.append("center_shift_too_large")
    if foreground_ratio < 0.15:
        flags.append("foreground_ratio_too_low")
    if foreground_ratio > 0.95:
        flags.append("foreground_ratio_suspicious")
    if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
        flags.append("aspect_ratio_out_of_prior")
    if clip_delta > 1e-3:
        flags.append("obb_clipped")
    if occlusion >= 2:
        flags.append("heavy_occlusion")
    if truncation >= 2:
        flags.append("heavy_truncation")
    if metrics.mask_solidity < 0.30:
        flags.append("mask_fragmented")

    # --- VLM / semantic flags ---
    if vlm_class_name and vlm_class_name != class_name and vlm_class_confidence > 0.8:
        flags.append("vlm_class_conflict_high_conf")
    if vlm_class_name and vlm_class_name != class_name and vlm_class_confidence > 0.5:
        flags.append("vlm_class_conflict")

    # --- Composite scores ---
    area_score = max(0.0, 1.0 - min(abs(area_ratio - 0.8), 1.0))
    center_score = max(0.0, 1.0 - min(center_shift / 0.5, 1.0))
    fg_score = min(max(foreground_ratio / 0.55, 0.0), 1.0)
    if min_aspect <= aspect_ratio <= max_aspect:
        aspect_score = 1.0
    else:
        aspect_score = 0.25
    boundary_score = max(0.0, 1.0 - metrics.boundary_clip_ratio / 0.10)

    metrics.geometry_score = round(
        0.35 * area_score + 0.25 * center_score + 0.20 * fg_score
        + 0.10 * aspect_score + 0.10 * boundary_score, 4
    )
    has_vlm_signal = bool(vlm_class_name or vlm_box_xyxy or vlm_class_confidence or vlm_box_confidence)
    if has_vlm_signal:
        metrics.semantic_score = round(
            0.60 * vlm_class_confidence + 0.40 * metrics.vlm_box_iou, 4
        )
        metrics.final_score = round(
            0.55 * metrics.geometry_score + 0.45 * metrics.semantic_score, 4
        )
    else:
        metrics.semantic_score = 1.0
        metrics.final_score = metrics.geometry_score
    metrics.flags = flags

    return metrics


def classify_review_status(
    metrics: VLMQualityMetrics,
) -> tuple[ReviewDecision, str]:
    """Determine review status and reason from v2 quality metrics."""
    if metrics.final_score >= 0.65 and metrics.geometry_score >= 0.50 and metrics.semantic_score >= 0.50:
        critical_flags = [
            f for f in metrics.flags
            if f not in {"heavy_occlusion", "heavy_truncation"}
        ]
        if not critical_flags:
            return "auto_accept", ""
        return "review", "critical_flags_after_high_score"
    if metrics.final_score < 0.35:
        return "reject", "low_final_score"
    return "review", "medium_quality"


# ---------------------------------------------------------------------------
# Geometry helpers (reused from v1, kept here for v2 independence)
# ---------------------------------------------------------------------------

def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _convex_hull_area(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return float(len(xs))
    pts = np.column_stack([xs, ys]).astype(np.float32)
    try:
        hull = cv2.convexHull(pts.reshape(-1, 1, 2).astype(np.int32))
        return float(cv2.contourArea(hull))
    except Exception:
        return float(len(xs))


# Import cv2 locally to avoid top-level dependency for schema-only usage
import cv2  # noqa: E402


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


def normalize_points(points: np.ndarray, image_w: int, image_h: int) -> list[float]:
    norm = points.astype(np.float32).copy()
    norm[:, 0] = np.clip(norm[:, 0], 0, image_w - 1) / max(image_w, 1)
    norm[:, 1] = np.clip(norm[:, 1], 0, image_h - 1) / max(image_h, 1)
    return [round(float(v), 6) for v in norm.reshape(-1)]


def expand_box(
    box: tuple[float, float, float, float],
    image_w: int,
    image_h: int,
    ratio: float,
    min_size: int = 40,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    dx = bw * ratio
    dy = bh * ratio
    ex1 = max(math.floor(x1 - dx), 0)
    ey1 = max(math.floor(y1 - dy), 0)
    ex2 = min(math.ceil(x2 + dx), image_w)
    ey2 = min(math.ceil(y2 + dy), image_h)

    # Ensure minimum size
    if ex2 - ex1 < min_size:
        mid_x = (ex1 + ex2) / 2
        ex1 = max(int(mid_x - min_size / 2), 0)
        ex2 = min(int(mid_x + min_size / 2), image_w)
    if ey2 - ey1 < min_size:
        mid_y = (ey1 + ey2) / 2
        ey1 = max(int(mid_y - min_size / 2), 0)
        ey2 = min(int(mid_y + min_size / 2), image_h)

    return ex1, ey1, max(ex2, ex1 + 1), max(ey2, ey1 + 1)


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed.astype(np.uint8)


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
