#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert OpenVTER det_bbox_result pkl files to the required trajectory CSV schema.

The conversion source of truth is det_bbox_result_*.pkl -> traj_info.
raw_det is not used as final trajectory data, and *_stab.pkl is only inspected.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - fallback for lean environments.
    PchipInterpolator = None
    savgol_filter = None


VIS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INITIAL_ROOT = VIS_ROOT / "Initial results"
DEFAULT_ADJUSTED_ROOT = VIS_ROOT / "Adjusted results"
DEFAULT_LOG_PATH = VIS_ROOT / "logs" / "conversion.log"

CATEGORY_ID_TO_CLASS = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "freight_car",
    4: "van",
    5: "pedestrian",
    6: "people",
    7: "bicycle",
    8: "tricycle",
    9: "awning-tricycle",
    10: "motor",
}

ALL_CLASSES = [
    "car",
    "truck",
    "bus",
    "freight_car",
    "van",
    "pedestrian",
    "people",
    "bicycle",
    "tricycle",
    "awning-tricycle",
    "motor",
]

VEHICLE_CLASSES = {"car", "truck", "bus", "freight_car", "van", "motor", "tricycle", "awning-tricycle"}
VRU_CLASSES = {"pedestrian", "people", "bicycle", "tricycle", "awning-tricycle", "motor"}

SHORT_GAP_MAX = 5
MEDIUM_GAP_MAX = 15
MAX_MISSING_RATIO = 0.40
HEADING_SMOOTH_WINDOW = 5
MIN_DISPLACEMENT_FOR_HEADING = 0.05
SHORT_TRACK_MIN_FRAMES = {
    "pedestrian": 90,
    "people": 90,
    "default": 150,
}
CONSECUTIVE_OUTLIER_SPLIT = 15

PHYSICAL_LIMITS = {
    "car": {"max_speed": 25.0, "max_acc": 8.0},
    "van": {"max_speed": 25.0, "max_acc": 8.0},
    "truck": {"max_speed": 25.0, "max_acc": 8.0},
    "bus": {"max_speed": 25.0, "max_acc": 8.0},
    "freight_car": {"max_speed": 25.0, "max_acc": 8.0},
    "motor": {"max_speed": 20.0, "max_acc": 8.0},
    "bicycle": {"max_speed": 12.0, "max_acc": 5.0},
    "tricycle": {"max_speed": 12.0, "max_acc": 5.0},
    "awning-tricycle": {"max_speed": 12.0, "max_acc": 5.0},
    "pedestrian": {"max_speed": 6.0, "max_acc": 4.0},
    "people": {"max_speed": 6.0, "max_acc": 4.0},
}

CONFUSABLE_CLASS_GROUPS = [
    {"car", "van"},
    {"truck", "freight_car"},
    {"tricycle", "awning-tricycle"},
    {"pedestrian", "people"},
]

FRAGMENTATION_CLASS_GROUPS = [
    {"car", "van"},
    {"truck", "freight_car", "bus"},
    {"motor", "tricycle", "awning-tricycle"},
    {"pedestrian", "people"},
    {"bicycle"},
]

FRAGMENTATION_FILTER = {
    "enabled": True,
    "min_gap_frames": 5,
    "max_gap_frames": 100,
    "min_track_frames": 5,
    "score_threshold": 0.70,
    "max_position_distance_base": 30.0,
    "max_position_distance_gap_factor": 2.0,
    "min_bbox_size_ratio": 0.4,
    "max_bbox_size_ratio": 2.5,
    "max_heading_diff_deg": 60.0,
    "border_margin_px": 20.0,
    "velocity_window_frames": 5,
    "filter_strategy": "drop_all_suspected_fragments",
}

# The converter writes two dataset versions:
# full keeps every cleaned track, while moving_filtered removes long-lived,
# nearly stationary motorized tracks. Values are in the current SI trajectory
# units, so displacement is meters, mean_speed is m/s, and per-frame motion is m.
STATIC_GATE = {
    "min_track_length": 30,
    "max_displacement": 1.0,
    "max_path_length": 2.0,
    "max_stationary_extent": 2.0,
    "max_mean_speed": 0.2,
    "static_ratio_threshold": 0.8,
    "per_frame_motion_threshold": 0.05,
    "filter_classes": sorted(VEHICLE_CLASSES),
}

RECORDING_META_FIELDS = [
    "recordingId",
    "locationId",
    "frameRate",
    "numFrames",
    "duration",
    "numTracks",
    "numVehicles",
    "numVRUs",
    "classTrackCounts",
    "orthoPxToMeter",
]

TRACKS_META_FIELDS = [
    "recordingId",
    "trackId",
    "raw_object_id",
    "initialFrame",
    "finalFrame",
    "numFrames",
    "startXCenter",
    "startYCenter",
    "endXCenter",
    "endYCenter",
    "startLaneId",
    "endLaneId",
    "width",
    "length",
    "raw_mean_width",
    "raw_mean_height",
    "corrected_width",
    "corrected_height",
    "box_orientation_source",
    "missing_ratio",
    "class",
]

TRACKS_FIELDS = [
    "recordingId",
    "trackId",
    "raw_object_id",
    "lane_id",
    "frame",
    "trackLifetime",
    "xCenter",
    "yCenter",
    "heading",
    "width",
    "length",
    "raw_mean_width",
    "raw_mean_height",
    "corrected_width",
    "corrected_height",
    "box_orientation_source",
    "is_interpolated",
    "missing_ratio",
    "xVelocity",
    "yVelocity",
    "xAcceleration",
    "yAcceleration",
    "lonVelocity",
    "latVelocity",
    "lonAcceleration",
    "latAcceleration",
    "centerX",
    "centerY",
]

ID_MAPPING_FIELDS = [
    "dataset_id",
    "version",
    "raw_object_id",
    "final_object_id",
    "class_name_mode",
    "start_frame",
    "end_frame",
    "total_frames",
    "mean_confidence",
    "is_kept",
    "is_filtered",
    "filter_type",
    "filter_reason",
    "fragmentation_group_id",
    "quality_score",
]

FILTER_REPORT_FIELDS = [
    "dataset_id",
    "version",
    "raw_object_id",
    "filter_type",
    "filter_reason",
    "fragmentation_group_id",
    "related_raw_object_ids",
    "fragmentation_score",
    "quality_score",
    "start_frame",
    "end_frame",
    "total_frames",
    "class_name_mode",
]


class ConversionError(RuntimeError):
    """Raised when source data cannot be converted."""


def configure_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("standard_trajectory_converter")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _finite(value: Any) -> bool:
    return _safe_float(value) is not None


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _format_value(field: str, value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        if field == "duration":
            return f"{value:.3f}"
        if field == "orthoPxToMeter":
            return f"{value:.6f}"
        if field == "heading":
            return f"{value:.2f}"
        return f"{value:.4f}"
    return value


def _write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_value(field, row.get(field, "")) for field in fieldnames})


def _find_detection_pkl(dataset_dir: Path) -> Optional[Path]:
    for pattern in ("det_bbox_result_*.pkl", "stitch_bbox_result_*.pkl", "*.detpkl", "*.pkl"):
        matches = sorted(dataset_dir.glob(pattern))
        matches = [p for p in matches if "stab" not in p.stem.lower()]
        if matches:
            return matches[0]
    return None


def _find_stabilization_pkl(dataset_dir: Path) -> Optional[Path]:
    matches = sorted(dataset_dir.glob("*_stab.pkl")) + sorted(dataset_dir.glob("*stab*.pkl"))
    return matches[0] if matches else None


def _parse_folder_identity(folder_name: str, warnings: List[str]) -> Tuple[str, str]:
    parts = folder_name.rsplit("_", 1)
    if len(parts) == 2 and re.fullmatch(r"\d+", parts[1]):
        return parts[1], parts[0]
    warnings.append(
        f"Folder name '{folder_name}' does not end with an underscore numeric id; "
        "recordingId and locationId both use folderName."
    )
    return folder_name, folder_name


def _video_info(data: Dict[str, Any]) -> Dict[str, Any]:
    video_info = data.get("video_info")
    if isinstance(video_info, list) and video_info and isinstance(video_info[0], dict):
        return video_info[0]
    if isinstance(video_info, dict):
        return video_info
    return {}


def _format_entry(entry: Tuple[Any, ...]) -> Tuple[int, int, Any, Optional[Any]]:
    if not isinstance(entry, tuple) or len(entry) not in (3, 4):
        raise ConversionError(f"traj_info entry must be tuple(frame, output_frame, array[, time]), got {type(entry)}")
    frame_id, output_frame, arr = entry[:3]
    frame_time = entry[3] if len(entry) == 4 else None
    return int(frame_id), int(output_frame), arr, frame_time


def _mode_class(rows: List[Dict[str, Any]], logger: logging.Logger, label: str, quality: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, int]]:
    real_rows = [row for row in rows if not row.get("is_interpolated")]
    if not real_rows:
        quality["warnings"].append(f"{label}: no real detection rows for class majority; fallback to first raw_class.")
        first = rows[0].get("raw_class") or "car"
        return first, 0.0, True, {first: 0}

    counts: Counter[str] = Counter(row["raw_class"] for row in real_rows)
    conf_sums: Dict[str, float] = defaultdict(float)
    first_frame: Dict[str, int] = {}
    for row in real_rows:
        cls = row["raw_class"]
        conf_sums[cls] += float(row.get("confidence") or 0.0)
        first_frame.setdefault(cls, int(row["frame"]))

    max_count = max(counts.values())
    candidates = [cls for cls, count in counts.items() if count == max_count]
    max_conf = max(conf_sums[cls] for cls in candidates)
    candidates = [cls for cls in candidates if abs(conf_sums[cls] - max_conf) < 1e-9]
    if len(candidates) > 1:
        chosen = min(candidates, key=lambda cls: first_frame[cls])
        msg = f"{label}: class majority tie {dict(counts)}, confidence tie; choose earliest class '{chosen}'."
        logger.warning(msg)
        quality["warnings"].append(msg)
    else:
        chosen = candidates[0]

    ratio = counts[chosen] / float(len(real_rows))
    unstable = ratio < 0.7
    if len(counts) > 1:
        quality["category_jump_tracks"].append(
            {"track": label, "class_counts": dict(counts), "final_class": chosen, "final_class_ratio": ratio}
        )
    if unstable:
        quality["category_unstable_tracks"].append(
            {"track": label, "class_counts": dict(counts), "final_class": chosen, "final_class_ratio": ratio}
        )
    return chosen, ratio, unstable, dict(counts)


def _edge_sizes(world: np.ndarray) -> Tuple[float, float]:
    e12 = _dist(tuple(world[0]), tuple(world[1]))
    e23 = _dist(tuple(world[1]), tuple(world[2]))
    e34 = _dist(tuple(world[2]), tuple(world[3]))
    e41 = _dist(tuple(world[3]), tuple(world[0]))
    edge_a = (e12 + e34) / 2.0
    edge_b = (e23 + e41) / 2.0
    return min(edge_a, edge_b), max(edge_a, edge_b)


def _long_edge_heading(row: Dict[str, Any]) -> Optional[float]:
    pts = [
        (row.get("world_q1_x"), row.get("world_q1_y")),
        (row.get("world_q2_x"), row.get("world_q2_y")),
        (row.get("world_q3_x"), row.get("world_q3_y")),
        (row.get("world_q4_x"), row.get("world_q4_y")),
    ]
    if not all(_finite(x) and _finite(y) for x, y in pts):
        return None
    edges = [
        (pts[0], pts[1]),
        (pts[1], pts[2]),
        (pts[2], pts[3]),
        (pts[3], pts[0]),
    ]
    a, b = max(edges, key=lambda edge: _dist(edge[0], edge[1]))
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    if math.hypot(dx, dy) < 1e-9:
        return None
    return math.degrees(math.atan2(dx, dy)) % 360.0


def _expand_traj_info(data: Dict[str, Any], logger: logging.Logger, quality: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]], Counter[int]]:
    traj_info = data.get("traj_info")
    if not isinstance(traj_info, list) or not traj_info:
        raise ConversionError("det pkl missing non-empty list field 'traj_info'.")

    rows: List[Dict[str, Any]] = []
    frame_meta: Dict[int, Dict[str, Any]] = {}
    col_counts: Counter[int] = Counter()
    unknown_category_ids: set[int] = set()
    invalid_world_rows = 0
    lane_minus_one = 0

    for entry in traj_info:
        frame, output_frame, arr, frame_time = _format_entry(entry)
        frame_meta.setdefault(frame, {"output_frame": output_frame, "frame_time": frame_time})
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        if arr.ndim != 2 or arr.shape[1] < 19:
            raise ConversionError(f"Frame {frame} array must be 2D with at least 19 columns, got {arr.shape}.")
        col_counts[arr.shape[1]] += 1

        for row_index, raw in enumerate(arr):
            raw = np.asarray(raw, dtype=float)
            category_id = int(raw[9]) if _finite(raw[9]) else -1
            raw_class = CATEGORY_ID_TO_CLASS.get(category_id)
            if raw_class is None:
                raw_class = f"unknown_{category_id}"
                unknown_category_ids.add(category_id)
            object_id = int(raw[10]) if _finite(raw[10]) else None
            if object_id is None:
                quality["warnings"].append(f"Frame {frame} row {row_index}: missing object_id; row skipped.")
                continue

            pixel = raw[0:8].reshape(4, 2)
            world = raw[11:19].reshape(4, 2)
            if not np.isfinite(world).all():
                invalid_world_rows += 1
                continue

            raw_width, raw_length = _edge_sizes(world)
            x_center = float(world[:, 0].mean())
            y_center = float(world[:, 1].mean())
            pixel_center_x = float(pixel[:, 0].mean())
            pixel_center_y = float(pixel[:, 1].mean())
            lane_id = int(raw[19]) if raw.shape[0] >= 20 and _finite(raw[19]) else -1
            if lane_id == -1:
                lane_minus_one += 1

            rows.append(
                {
                    "frame": frame,
                    "output_frame": output_frame,
                    "object_id": object_id,
                    "category_id": category_id,
                    "raw_class": raw_class,
                    "confidence": float(raw[8]) if _finite(raw[8]) else math.nan,
                    "lane_id": lane_id,
                    "q1_x": float(pixel[0, 0]),
                    "q1_y": float(pixel[0, 1]),
                    "q2_x": float(pixel[1, 0]),
                    "q2_y": float(pixel[1, 1]),
                    "q3_x": float(pixel[2, 0]),
                    "q3_y": float(pixel[2, 1]),
                    "q4_x": float(pixel[3, 0]),
                    "q4_y": float(pixel[3, 1]),
                    "pixel_cx": pixel_center_x,
                    "pixel_cy": pixel_center_y,
                    "world_q1_x": float(world[0, 0]),
                    "world_q1_y": float(world[0, 1]),
                    "world_q2_x": float(world[1, 0]),
                    "world_q2_y": float(world[1, 1]),
                    "world_q3_x": float(world[2, 0]),
                    "world_q3_y": float(world[2, 1]),
                    "world_q4_x": float(world[3, 0]),
                    "world_q4_y": float(world[3, 1]),
                    "xCenter_raw": x_center,
                    "yCenter_raw": y_center,
                    "raw_width": raw_width,
                    "raw_length": raw_length,
                    "is_interpolated": False,
                    "is_outlier": False,
                    "source_row_index": row_index,
                }
            )

    if invalid_world_rows:
        logger.warning("%s rows have invalid world coordinates and were skipped.", invalid_world_rows)
    if unknown_category_ids:
        logger.warning("Unknown category_id values: %s", sorted(unknown_category_ids))
    quality["invalid_world_rows"] = invalid_world_rows
    quality["lane_id_minus_one_records"] = lane_minus_one
    quality["unknown_category_ids"] = sorted(unknown_category_ids)
    return rows, frame_meta, col_counts


def _confidence_key(row: Dict[str, Any]) -> Tuple[float, int]:
    confidence = row.get("confidence")
    if not _finite(confidence):
        confidence = -math.inf
    # Earlier rows win exact confidence ties to keep de-duplication stable.
    return float(confidence), -int(row.get("source_row_index", 0))


def _dedupe_track_frames(rows: List[Dict[str, Any]], object_id: int, logger: logging.Logger, quality: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[int(row["frame"])].append(row)

    deduped: List[Dict[str, Any]] = []
    for frame, frame_rows in sorted(by_frame.items()):
        if len(frame_rows) > 1:
            kept = max(frame_rows, key=_confidence_key)
            msg = (
                f"object_id={object_id} frame={frame}: duplicate detections={len(frame_rows)}; "
                "kept highest confidence row for missing-ratio statistics."
            )
            logger.warning(msg)
            quality["warnings"].append(msg)
            quality["duplicate_frame_record_count"] += len(frame_rows) - 1
            quality["duplicate_frame_tracks"].add(object_id)
        else:
            kept = frame_rows[0]
        deduped.append(dict(kept))
    return deduped


def _track_missing_stats(object_id: int, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    frames = sorted({int(row["frame"]) for row in rows})
    min_frame = frames[0]
    max_frame = frames[-1]
    expected = max_frame - min_frame + 1
    observed = len(frames)
    missing = max(expected - observed, 0)
    ratio = missing / float(expected) if expected > 0 else 0.0
    return {
        "trackId": object_id,
        "min_frame": min_frame,
        "max_frame": max_frame,
        "expected_frame_count": expected,
        "observed_frame_count": observed,
        "missing_frame_count": missing,
        "missing_ratio": ratio,
        "is_dropped": ratio > MAX_MISSING_RATIO,
        "drop_reason": "missing_ratio_exceeded" if ratio > MAX_MISSING_RATIO else "",
        "num_interpolated_frames": 0 if ratio > MAX_MISSING_RATIO else missing,
    }


def _class_group_key(class_name: str) -> Optional[int]:
    for idx, group in enumerate(FRAGMENTATION_CLASS_GROUPS):
        if class_name in group:
            return idx
    return None


def _angle_diff_deg(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return diff


def _dedupe_rows_for_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[int(row["frame"])].append(row)
    return [dict(max(frame_rows, key=_confidence_key)) for _, frame_rows in sorted(by_frame.items())]


def _velocity_from_points(points: List[Tuple[int, float, float]], use_tail: bool, window: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(points) < 2:
        return None, None, None
    sample = points[-window:] if use_tail else points[:window]
    if len(sample) < 2:
        sample = points
    f0, x0, y0 = sample[0]
    f1, x1, y1 = sample[-1]
    dt = max(int(f1) - int(f0), 0)
    if dt <= 0:
        return None, None, None
    vx = (float(x1) - float(x0)) / dt
    vy = (float(y1) - float(y0)) / dt
    heading = math.degrees(math.atan2(vy, vx)) % 360.0 if math.hypot(vx, vy) > 1e-9 else None
    return vx, vy, heading


def _raw_tracklet_stats(raw_rows: List[Dict[str, Any]], dataset_id: str, frame_rate: float) -> Dict[int, Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[int(row["object_id"])].append(row)

    stats_by_raw: Dict[int, Dict[str, Any]] = {}
    for raw_id, grouped_rows in sorted(grouped.items()):
        rows = _dedupe_rows_for_stats(grouped_rows)
        frames = [int(row["frame"]) for row in rows]
        points = [(int(row["frame"]), float(row["xCenter_raw"]), float(row["yCenter_raw"])) for row in rows]
        pixel_points = [(float(row.get("pixel_cx", math.nan)), float(row.get("pixel_cy", math.nan))) for row in rows]
        classes = [row.get("raw_class") or "unknown" for row in rows]
        class_counts = Counter(classes)
        class_name_mode = class_counts.most_common(1)[0][0] if class_counts else "unknown"
        confidences = [float(row["confidence"]) for row in rows if _finite(row.get("confidence"))]
        widths = [float(row["raw_width"]) for row in rows if _finite(row.get("raw_width"))]
        heights = [float(row["raw_length"]) for row in rows if _finite(row.get("raw_length"))]
        segment_distances = [
            _dist((a[1], a[2]), (b[1], b[2]))
            for a, b in zip(points[:-1], points[1:])
        ]
        segment_speeds = []
        for dist, f0, f1 in zip(segment_distances, frames[:-1], frames[1:]):
            dt = max((f1 - f0) / frame_rate, 1.0 / frame_rate)
            segment_speeds.append(dist / dt)
        elapsed = max((frames[-1] - frames[0]) / frame_rate, 1.0 / frame_rate) if len(frames) >= 2 else 1.0 / frame_rate
        path_length = float(sum(segment_distances)) if segment_distances else 0.0
        displacement = _dist((points[0][1], points[0][2]), (points[-1][1], points[-1][2])) if len(points) >= 2 else 0.0
        mean_confidence = float(np.mean(confidences)) if confidences else math.nan
        mean_width = float(np.mean(widths)) if widths else math.nan
        mean_height = float(np.mean(heights)) if heights else math.nan
        mean_bbox_diag = math.hypot(mean_width, mean_height) if _finite(mean_width) and _finite(mean_height) else 0.0
        missing = max(frames[-1] - frames[0] + 1 - len(set(frames)), 0)
        quality_score = len(frames) * mean_confidence if _finite(mean_confidence) else float(len(frames))
        tail_vx, tail_vy, tail_heading = _velocity_from_points(points, True, int(FRAGMENTATION_FILTER["velocity_window_frames"]))
        head_vx, head_vy, head_heading = _velocity_from_points(points, False, int(FRAGMENTATION_FILTER["velocity_window_frames"]))
        stats_by_raw[raw_id] = {
            "dataset_id": dataset_id,
            "raw_object_id": raw_id,
            "class_name_mode": class_name_mode,
            "class_group": _class_group_key(class_name_mode),
            "start_frame": frames[0],
            "end_frame": frames[-1],
            "total_frames": len(frames),
            "frame_span": frames[-1] - frames[0] + 1,
            "missing_frames_inside_track": missing,
            "mean_confidence": mean_confidence,
            "start_cx": points[0][1],
            "start_cy": points[0][2],
            "end_cx": points[-1][1],
            "end_cy": points[-1][2],
            "start_pixel_cx": pixel_points[0][0],
            "start_pixel_cy": pixel_points[0][1],
            "end_pixel_cx": pixel_points[-1][0],
            "end_pixel_cy": pixel_points[-1][1],
            "start_width": float(rows[0]["raw_width"]) if _finite(rows[0].get("raw_width")) else math.nan,
            "start_height": float(rows[0]["raw_length"]) if _finite(rows[0].get("raw_length")) else math.nan,
            "end_width": float(rows[-1]["raw_width"]) if _finite(rows[-1].get("raw_width")) else math.nan,
            "end_height": float(rows[-1]["raw_length"]) if _finite(rows[-1].get("raw_length")) else math.nan,
            "mean_width": mean_width,
            "mean_height": mean_height,
            "mean_bbox_diag": mean_bbox_diag,
            "displacement": displacement,
            "path_length": path_length,
            "mean_speed": path_length / elapsed,
            "max_speed": max(segment_speeds) if segment_speeds else 0.0,
            "quality_score": quality_score,
            "_points": points,
            "_tail_vx": tail_vx,
            "_tail_vy": tail_vy,
            "_tail_heading": tail_heading,
            "_head_heading": head_heading,
        }
    return stats_by_raw


def _near_image_border(stat: Dict[str, Any], prefix: str, video_info: Dict[str, Any]) -> bool:
    width = _safe_float(video_info.get("width"))
    height = _safe_float(video_info.get("height"))
    if width is None or height is None or width <= 0 or height <= 0:
        return False
    margin = float(FRAGMENTATION_FILTER["border_margin_px"])
    x = _safe_float(stat.get(f"{prefix}_pixel_cx"))
    y = _safe_float(stat.get(f"{prefix}_pixel_cy"))
    if x is None or y is None:
        return False
    return x <= margin or y <= margin or x >= width - margin or y >= height - margin


def _score_fragmentation_pair(a: Dict[str, Any], b: Dict[str, Any], video_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cfg = FRAGMENTATION_FILTER
    if int(b["start_frame"]) <= int(a["end_frame"]):
        return None
    gap = int(b["start_frame"]) - int(a["end_frame"]) - 1
    if gap < int(cfg["min_gap_frames"]) or gap > int(cfg["max_gap_frames"]):
        return None
    if int(a["total_frames"]) < int(cfg["min_track_frames"]) or int(b["total_frames"]) < int(cfg["min_track_frames"]):
        return None
    if a.get("class_group") is None or a.get("class_group") != b.get("class_group"):
        return None

    ratios = []
    for key in ("mean_width", "mean_height"):
        av = _safe_float(a.get(key))
        bv = _safe_float(b.get(key))
        if av is None or bv is None or av <= 0 or bv <= 0:
            continue
        ratio = bv / av
        if ratio < float(cfg["min_bbox_size_ratio"]) or ratio > float(cfg["max_bbox_size_ratio"]):
            return None
        ratios.append(min(ratio, 1.0 / ratio))
    bbox_score = float(np.mean(ratios)) if ratios else 0.7

    vx = a.get("_tail_vx")
    vy = a.get("_tail_vy")
    if vx is not None and vy is not None:
        pred_x = float(a["end_cx"]) + float(vx) * gap
        pred_y = float(a["end_cy"]) + float(vy) * gap
    else:
        pred_x = float(a["end_cx"])
        pred_y = float(a["end_cy"])
    distance = _dist((pred_x, pred_y), (float(b["start_cx"]), float(b["start_cy"])))
    allowed = float(cfg["max_position_distance_base"]) + float(cfg["max_position_distance_gap_factor"]) * gap
    allowed = max(allowed, max(float(a.get("mean_bbox_diag") or 0.0), float(b.get("mean_bbox_diag") or 0.0)) * 2.0)
    if allowed <= 0 or distance > allowed:
        return None
    position_score = max(0.0, 1.0 - distance / allowed)

    heading_diff = _angle_diff_deg(a.get("_tail_heading"), b.get("_head_heading"))
    if heading_diff is not None:
        if heading_diff > float(cfg["max_heading_diff_deg"]):
            return None
        heading_score = max(0.0, 1.0 - heading_diff / float(cfg["max_heading_diff_deg"]))
    else:
        heading_score = 0.7

    gap_range = max(int(cfg["max_gap_frames"]) - int(cfg["min_gap_frames"]), 1)
    gap_score = max(0.0, 1.0 - (gap - int(cfg["min_gap_frames"])) / gap_range)
    class_score = 1.0 if a["class_name_mode"] == b["class_name_mode"] else 0.85
    near_border = _near_image_border(a, "end", video_info) or _near_image_border(b, "start", video_info)
    border_penalty = 0.20 if near_border else 0.0
    score = (
        0.20 * gap_score
        + 0.15 * class_score
        + 0.35 * position_score
        + 0.15 * bbox_score
        + 0.15 * heading_score
        - border_penalty
    )
    if near_border and score < 0.95:
        return None
    if score < float(cfg["score_threshold"]):
        return None
    return {
        "raw_a": int(a["raw_object_id"]),
        "raw_b": int(b["raw_object_id"]),
        "gap_frames": gap,
        "distance": distance,
        "max_allowed_distance": allowed,
        "gap_score": gap_score,
        "class_score": class_score,
        "position_score": position_score,
        "bbox_score": bbox_score,
        "heading_score": heading_score,
        "border_penalty": border_penalty,
        "fragmentation_score": score,
    }


def _detect_fragmentation_groups(
    stats_by_raw: Dict[int, Dict[str, Any]],
    candidate_raw_ids: Iterable[int],
    video_info: Dict[str, Any],
) -> Dict[str, Any]:
    if not FRAGMENTATION_FILTER["enabled"]:
        return {"parameters": FRAGMENTATION_FILTER, "strategy": FRAGMENTATION_FILTER["filter_strategy"], "groups": [], "pairs": [], "filtered_raw_object_ids": []}
    candidates = [stats_by_raw[raw_id] for raw_id in sorted(set(candidate_raw_ids)) if raw_id in stats_by_raw]
    candidates.sort(key=lambda item: (int(item["start_frame"]), int(item["raw_object_id"])))
    pairs = []
    edges: Dict[int, set[int]] = defaultdict(set)
    for a in candidates:
        for b in candidates:
            if int(b["start_frame"]) <= int(a["end_frame"]):
                continue
            pair = _score_fragmentation_pair(a, b, video_info)
            if pair is None:
                continue
            pairs.append(pair)
            edges[pair["raw_a"]].add(pair["raw_b"])
            edges[pair["raw_b"]].add(pair["raw_a"])

    visited: set[int] = set()
    groups = []
    for raw_id in sorted(edges):
        if raw_id in visited:
            continue
        stack = [raw_id]
        component = set()
        while stack:
            cur = stack.pop()
            if cur in component:
                continue
            component.add(cur)
            stack.extend(sorted(edges.get(cur, set()) - component))
        visited.update(component)
        if len(component) < 2:
            continue
        group_pairs = [p for p in pairs if p["raw_a"] in component and p["raw_b"] in component]
        group_score = max(float(p["fragmentation_score"]) for p in group_pairs) if group_pairs else 0.0
        groups.append(
            {
                "fragmentation_group_id": f"fg_{len(groups) + 1:04d}",
                "raw_object_ids": sorted(component),
                "fragmentation_score": group_score,
                "pair_count": len(group_pairs),
                "pairs": group_pairs,
                "filter_reason": "suspected_id_fragmentation_drop_all_related_tracklets",
            }
        )
    filtered_raw_ids = sorted({raw_id for group in groups for raw_id in group["raw_object_ids"]})
    return {
        "parameters": dict(FRAGMENTATION_FILTER),
        "strategy": FRAGMENTATION_FILTER["filter_strategy"],
        "groups": groups,
        "pairs": pairs,
        "filtered_raw_object_ids": filtered_raw_ids,
        "filtered_track_count": len(filtered_raw_ids),
    }


def _split_raw_tracks(raw_rows: List[Dict[str, Any]], frame_rate: float, logger: logging.Logger, quality: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[int(row["object_id"])].append(row)

    tracks: List[List[Dict[str, Any]]] = []
    for object_id, rows in sorted(grouped.items()):
        rows = _dedupe_track_frames(rows, object_id, logger, quality)
        rows = sorted(rows, key=lambda item: int(item["frame"]))
        stats = _track_missing_stats(object_id, rows)
        quality["track_missing_stats"].append(dict(stats))
        if stats["missing_frame_count"] > 0:
            quality["gap_tracks"].add(object_id)
            quality["total_missing_frame_count"] += int(stats["missing_frame_count"])
        if stats["is_dropped"]:
            quality["dropped_track_count"] += 1
            logger.info(
                "object_id=%s dropped by missing_ratio %.4f > %.2f",
                object_id,
                stats["missing_ratio"],
                MAX_MISSING_RATIO,
            )
            continue

        final_class, _, _, class_counts = _mode_class(rows, logger, f"object_id={object_id}", quality)
        if len(class_counts) > 1:
            quality["raw_object_category_jump_count"] += 1
        for prev, cur in zip(rows[:-1], rows[1:]):
            gap = int(cur["frame"]) - int(prev["frame"]) - 1
            if gap > 0:
                if gap <= SHORT_GAP_MAX:
                    quality["short_gap_count"] += gap
                elif gap <= MEDIUM_GAP_MAX:
                    quality["medium_gap_count"] += gap
                else:
                    quality["long_gap_count"] += gap
        tracks.append(rows)
    return tracks


def _interpolate_values(frames: List[int], values: List[float], new_frames: List[int]) -> List[float]:
    if len(frames) >= 3 and PchipInterpolator is not None:
        try:
            f = PchipInterpolator(np.asarray(frames, dtype=float), np.asarray(values, dtype=float))
            return [float(v) for v in f(np.asarray(new_frames, dtype=float))]
        except Exception:
            pass
    return [float(v) for v in np.interp(new_frames, frames, values)]


def _interpolate_between(prev: Dict[str, Any], cur: Dict[str, Any], final_class: str, frame_rate: float) -> List[Dict[str, Any]]:
    gap = int(cur["frame"]) - int(prev["frame"]) - 1
    if gap <= 0:
        return []

    frames = [int(prev["frame"]), int(cur["frame"])]
    new_frames = list(range(frames[0] + 1, frames[1]))
    xs = _interpolate_values(frames, [prev["xCenter_raw"], cur["xCenter_raw"]], new_frames)
    ys = _interpolate_values(frames, [prev["yCenter_raw"], cur["yCenter_raw"]], new_frames)
    widths = _interpolate_values(frames, [prev["raw_width"], cur["raw_width"]], new_frames)
    lengths = _interpolate_values(frames, [prev["raw_length"], cur["raw_length"]], new_frames)
    rows = []
    for frame, x, y, width, length in zip(new_frames, xs, ys, widths, lengths):
        lane = prev["lane_id"] if prev["lane_id"] == cur["lane_id"] else (prev["lane_id"] if frame - frames[0] <= frames[1] - frame else cur["lane_id"])
        item = dict(prev)
        item.update(
            {
                "frame": frame,
                "output_frame": frame,
                "lane_id": lane if lane is not None else -1,
                "raw_class": final_class,
                "category_id": None,
                "confidence": math.nan,
                "xCenter_raw": x,
                "yCenter_raw": y,
                "raw_width": min(width, length),
                "raw_length": max(width, length),
                "is_interpolated": True,
                "is_outlier": False,
                "source_row_index": -1,
            }
        )
        rows.append(item)
    return rows


def _mark_isolated_outliers(rows: List[Dict[str, Any]], final_class: str, frame_rate: float, quality: Dict[str, Any]) -> None:
    if len(rows) < 3:
        return
    limit = PHYSICAL_LIMITS.get(final_class, PHYSICAL_LIMITS["car"])
    for i in range(1, len(rows) - 1):
        prev, cur, nxt = rows[i - 1], rows[i], rows[i + 1]
        if cur.get("is_interpolated"):
            continue
        dt_prev = max((cur["frame"] - prev["frame"]) / frame_rate, 1.0 / frame_rate)
        dt_next = max((nxt["frame"] - cur["frame"]) / frame_rate, 1.0 / frame_rate)
        d_prev = _dist((cur["xCenter_raw"], cur["yCenter_raw"]), (prev["xCenter_raw"], prev["yCenter_raw"]))
        d_next = _dist((nxt["xCenter_raw"], nxt["yCenter_raw"]), (cur["xCenter_raw"], cur["yCenter_raw"]))
        d_bridge = _dist((nxt["xCenter_raw"], nxt["yCenter_raw"]), (prev["xCenter_raw"], prev["yCenter_raw"]))
        speed_bad = d_prev / dt_prev > limit["max_speed"] or d_next / dt_next > limit["max_speed"]
        bridge_ok = d_bridge / max((nxt["frame"] - prev["frame"]) / frame_rate, 1.0 / frame_rate) <= limit["max_speed"]
        if speed_bad and bridge_ok:
            cur["is_outlier"] = True
            cur["xCenter_raw"] = math.nan
            cur["yCenter_raw"] = math.nan
            quality["outlier_frame_count"] += 1


def _fill_nan_centers(rows: List[Dict[str, Any]]) -> None:
    frames = [int(row["frame"]) for row in rows]
    for key in ("xCenter_raw", "yCenter_raw"):
        valid_frames = [frame for frame, row in zip(frames, rows) if _finite(row.get(key))]
        valid_values = [float(row[key]) for row in rows if _finite(row.get(key))]
        if len(valid_frames) < 2:
            continue
        new_values = _interpolate_values(valid_frames, valid_values, frames)
        for row, value in zip(rows, new_values):
            row[key] = value


def _smooth_series(values: List[float], quality: Dict[str, Any], label: str) -> List[float]:
    n = len(values)
    if n < 5 or savgol_filter is None:
        if n < 5:
            quality["short_tracks"].append(label)
        return [float(v) for v in values]
    window = 15 if n >= 15 else (n if n % 2 == 1 else n - 1)
    if window < 5:
        return [float(v) for v in values]
    try:
        return [float(v) for v in savgol_filter(np.asarray(values, dtype=float), window_length=window, polyorder=2, mode="interp")]
    except Exception:
        return [float(v) for v in values]


def _valid_dimensions(rows: List[Dict[str, Any]], quality: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    widths = [float(row["raw_width"]) for row in rows if _finite(row.get("raw_width")) and row["raw_width"] > 0]
    lengths = [float(row["raw_length"]) for row in rows if _finite(row.get("raw_length")) and row["raw_length"] > 0]
    if not widths or not lengths:
        return [], []
    med_w = float(np.median(widths))
    med_l = float(np.median(lengths))
    valid_w, valid_l = [], []
    for row in rows:
        w, l = row.get("raw_width"), row.get("raw_length")
        if not (_finite(w) and _finite(l) and w > 0 and l > 0):
            continue
        if w > 2.5 * med_w or w < 0.4 * med_w or l > 2.5 * med_l or l < 0.4 * med_l:
            quality["size_outlier_frame_count"] += 1
            row["size_outlier"] = True
            continue
        valid_w.append(float(min(w, l)))
        valid_l.append(float(max(w, l)))
    return valid_w, valid_l


def _differentiate(values: List[float], frames: List[int], frame_rate: float) -> List[float]:
    n = len(values)
    if n == 1:
        return [0.0]
    out = []
    times = [frame / frame_rate for frame in frames]
    for i in range(n):
        if i == 0:
            j0, j1 = 0, 1
        elif i == n - 1:
            j0, j1 = n - 2, n - 1
        else:
            j0, j1 = i - 1, i + 1
        dt = times[j1] - times[j0]
        out.append(0.0 if abs(dt) < 1e-12 else (values[j1] - values[j0]) / dt)
    return out


def _smooth_angles_degrees(values: List[float], window: int) -> List[float]:
    if len(values) < 3 or window <= 1:
        return [float(v) % 360.0 for v in values]
    half = max(1, window // 2)
    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        radians = np.radians(np.asarray(values[start:end], dtype=float))
        sin_mean = float(np.mean(np.sin(radians)))
        cos_mean = float(np.mean(np.cos(radians)))
        if abs(sin_mean) < 1e-12 and abs(cos_mean) < 1e-12:
            out.append(float(values[i]) % 360.0)
        else:
            out.append(math.degrees(math.atan2(sin_mean, cos_mean)) % 360.0)
    return out


def _compute_heading(xs: List[float], ys: List[float], rows: List[Dict[str, Any]], frame_rate: float, logger: logging.Logger, label: str) -> List[float]:
    n = len(xs)
    half_window = max(1, HEADING_SMOOTH_WINDOW // 2)
    headings: List[Optional[float]] = []
    last_valid: Optional[float] = None
    for i in range(n):
        j0 = max(0, i - half_window)
        j1 = min(n - 1, i + half_window)
        dx = xs[j1] - xs[j0]
        dy = ys[j1] - ys[j0]
        if math.hypot(dx, dy) >= MIN_DISPLACEMENT_FOR_HEADING:
            last_valid = math.degrees(math.atan2(dx, dy)) % 360.0
            headings.append(last_valid)
        elif last_valid is not None:
            headings.append(last_valid)
        else:
            headings.append(None)

    fallback = next((_long_edge_heading(row) for row in rows if _long_edge_heading(row) is not None), None)
    if fallback is None:
        fallback = 0.0
        logger.warning("%s has no valid motion or bbox heading; heading fallback is 0.0.", label)
    filled = [float(h if h is not None else fallback) for h in headings]
    return _smooth_angles_degrees(filled, HEADING_SMOOTH_WINDOW)


def _min_track_frames(final_class: str) -> int:
    return int(SHORT_TRACK_MIN_FRAMES.get(final_class, SHORT_TRACK_MIN_FRAMES["default"]))


def _estimate_ortho_px_to_meter(rows: List[Dict[str, Any]], logger: logging.Logger) -> Optional[float]:
    scales = []
    for row in rows:
        pixel_pts = [
            (row["q1_x"], row["q1_y"]),
            (row["q2_x"], row["q2_y"]),
            (row["q3_x"], row["q3_y"]),
            (row["q4_x"], row["q4_y"]),
        ]
        world_pts = [
            (row["world_q1_x"], row["world_q1_y"]),
            (row["world_q2_x"], row["world_q2_y"]),
            (row["world_q3_x"], row["world_q3_y"]),
            (row["world_q4_x"], row["world_q4_y"]),
        ]
        for i, j in ((0, 1), (1, 2), (2, 3), (3, 0)):
            px_len = _dist(pixel_pts[i], pixel_pts[j])
            w_len = _dist(world_pts[i], world_pts[j])
            if px_len > 1e-6 and w_len > 1e-6:
                scales.append(w_len / px_len)
    if len(scales) < 30:
        logger.warning("Not enough valid pixel/world edge pairs to estimate orthoPxToMeter.")
        return None
    arr = np.asarray(scales, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        arr = arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]
    return float(np.median(arr)) if arr.size else None


def _build_final_tracks(fragments: List[List[Dict[str, Any]]], frame_rate: float, logger: logging.Logger, quality: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prepared = []
    for idx, fragment in enumerate(fragments, start=1):
        fragment = [dict(row) for row in sorted(fragment, key=lambda item: int(item["frame"]))]
        label = f"fragment={idx},object_id={fragment[0]['object_id']}"
        final_class, ratio, unstable, class_counts = _mode_class(fragment, logger, label, quality)
        _mark_isolated_outliers(fragment, final_class, frame_rate, quality)
        _fill_nan_centers(fragment)

        with_gaps: List[Dict[str, Any]] = []
        interpolated_in_track = 0
        for prev, cur in zip(fragment[:-1], fragment[1:]):
            with_gaps.append(prev)
            inserts = _interpolate_between(prev, cur, final_class, frame_rate)
            interpolated_in_track += len(inserts)
            with_gaps.extend(inserts)
        with_gaps.append(fragment[-1])
        with_gaps = sorted(with_gaps, key=lambda item: int(item["frame"]))
        min_frames = _min_track_frames(final_class)
        if len(with_gaps) < min_frames:
            quality["short_duration_dropped_track_count"] += 1
            quality["short_duration_dropped_tracks"].append(
                {
                    "object_id": int(fragment[0]["object_id"]),
                    "class": final_class,
                    "numFrames": len(with_gaps),
                    "min_required_frames": min_frames,
                    "initialFrame": int(with_gaps[0]["frame"]),
                    "finalFrame": int(with_gaps[-1]["frame"]),
                    "drop_reason": "track_duration_too_short",
                }
            )
            continue
        quality["interpolated_frame_count"] += interpolated_in_track
        prepared.append(
            {
                "rows": with_gaps,
                "raw_rows": fragment,
                "final_class": final_class,
                "final_class_ratio": ratio,
                "category_unstable": unstable,
                "class_counts": class_counts,
                "original_object_id": int(fragment[0]["object_id"]),
                "missing_stats": _track_missing_stats(int(fragment[0]["object_id"]), fragment),
            }
        )

    # Conservative stitching is intentionally disabled unless a single unambiguous candidate is found.
    # Current implementation records zero merges rather than making aggressive ID merges.
    quality["stitched_track_count"] = 0
    prepared.sort(key=lambda item: (item["rows"][0]["frame"], item["original_object_id"]))

    class_dim_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"widths": [], "lengths": []})
    track_payloads = []
    for item in prepared:
        valid_w, valid_l = _valid_dimensions(item["raw_rows"], quality)
        if valid_w and valid_l:
            class_dim_values[item["final_class"]]["widths"].extend(valid_w)
            class_dim_values[item["final_class"]]["lengths"].extend(valid_l)
        track_payloads.append((item, valid_w, valid_l))

    class_dim_mean = {}
    for cls, vals in class_dim_values.items():
        if vals["widths"] and vals["lengths"]:
            class_dim_mean[cls] = (float(np.mean(vals["widths"])), float(np.mean(vals["lengths"])))

    tracks_meta: List[Dict[str, Any]] = []
    tracks_rows: List[Dict[str, Any]] = []
    for track_id, (item, valid_w, valid_l) in enumerate(track_payloads, start=1):
        rows = item["rows"]
        final_class = item["final_class"]
        if valid_w and valid_l:
            mean_width = float(np.mean(valid_w))
            mean_length = float(np.mean(valid_l))
        else:
            med_w = [row["raw_width"] for row in item["raw_rows"] if _finite(row.get("raw_width"))]
            med_l = [row["raw_length"] for row in item["raw_rows"] if _finite(row.get("raw_length"))]
            if med_w and med_l:
                mean_width = float(np.median(med_w))
                mean_length = float(np.median(med_l))
            elif final_class in class_dim_mean:
                mean_width, mean_length = class_dim_mean[final_class]
            else:
                mean_width, mean_length = math.nan, math.nan
                logger.warning("trackId %s has insufficient size data and no class fallback.", track_id)
        if _finite(mean_width) and _finite(mean_length) and mean_width > mean_length:
            mean_width, mean_length = mean_length, mean_width
        raw_mean_width = mean_width
        raw_mean_height = mean_length
        if final_class == "pedestrian" and _finite(mean_width) and _finite(mean_length):
            pedestrian_side = max(mean_width, mean_length)
            corrected_width = pedestrian_side
            corrected_height = pedestrian_side
        else:
            corrected_width = mean_width
            corrected_height = mean_length
        box_orientation_source = "heading_corrected_from_hbb"
        missing_ratio = float(item["missing_stats"]["missing_ratio"])

        xs = _smooth_series([float(row["xCenter_raw"]) for row in rows], quality, f"trackId={track_id}")
        ys = _smooth_series([float(row["yCenter_raw"]) for row in rows], quality, f"trackId={track_id}")
        frames = [int(row["frame"]) for row in rows]
        headings = _compute_heading(xs, ys, rows, frame_rate, logger, f"trackId={track_id}")
        x_vel = _differentiate(xs, frames, frame_rate)
        y_vel = _differentiate(ys, frames, frame_rate)
        x_acc = _differentiate(x_vel, frames, frame_rate)
        y_acc = _differentiate(y_vel, frames, frame_rate)

        for lifetime, (row, x, y, heading, vx, vy, ax, ay) in enumerate(zip(rows, xs, ys, headings, x_vel, y_vel, x_acc, y_acc), start=1):
            theta = math.radians(heading)
            lon_v = vx * math.sin(theta) + vy * math.cos(theta)
            lat_v = vx * (-math.cos(theta)) + vy * math.sin(theta)
            lon_a = ax * math.sin(theta) + ay * math.cos(theta)
            lat_a = ax * (-math.cos(theta)) + ay * math.sin(theta)
            tracks_rows.append(
                {
                    "trackId": track_id,
                    "raw_object_id": item["original_object_id"],
                    "lane_id": int(row.get("lane_id", -1)) if _finite(row.get("lane_id", -1)) else -1,
                    "frame": int(row["frame"]),
                    "trackLifetime": lifetime,
                    "xCenter": x,
                    "yCenter": y,
                    "heading": heading,
                    "width": corrected_width,
                    "length": corrected_height,
                    "raw_mean_width": raw_mean_width,
                    "raw_mean_height": raw_mean_height,
                    "corrected_width": corrected_width,
                    "corrected_height": corrected_height,
                    "box_orientation_source": box_orientation_source,
                    "is_interpolated": bool(row.get("is_interpolated", False)),
                    "missing_ratio": missing_ratio,
                    "xVelocity": vx,
                    "yVelocity": vy,
                    "xAcceleration": ax,
                    "yAcceleration": ay,
                    "lonVelocity": lon_v,
                    "latVelocity": lat_v,
                    "lonAcceleration": lon_a,
                    "latAcceleration": lat_a,
                    "centerX": x,
                    "centerY": y,
                }
            )

        tracks_meta.append(
            {
                "trackId": track_id,
                "raw_object_id": item["original_object_id"],
                "initialFrame": frames[0],
                "finalFrame": frames[-1],
                "numFrames": len(rows),
                "startXCenter": xs[0],
                "startYCenter": ys[0],
                "endXCenter": xs[-1],
                "endYCenter": ys[-1],
                "startLaneId": int(rows[0].get("lane_id", -1)) if _finite(rows[0].get("lane_id", -1)) else -1,
                "endLaneId": int(rows[-1].get("lane_id", -1)) if _finite(rows[-1].get("lane_id", -1)) else -1,
                "width": corrected_width,
                "length": corrected_height,
                "raw_mean_width": raw_mean_width,
                "raw_mean_height": raw_mean_height,
                "corrected_width": corrected_width,
                "corrected_height": corrected_height,
                "box_orientation_source": box_orientation_source,
                "missing_ratio": missing_ratio,
                "class": final_class,
                "_class_counts": item["class_counts"],
                "_final_class_ratio": item["final_class_ratio"],
                "_category_unstable": item["category_unstable"],
            }
        )

    return tracks_meta, tracks_rows


def _num_frames(video_info: Dict[str, Any], frame_meta: Dict[int, Dict[str, Any]], logger: logging.Logger) -> int:
    total = _safe_float(video_info.get("total_frames"))
    if total is not None:
        return int(round(total))
    frames = sorted(frame_meta)
    if not frames:
        return 0
    if frames[0] == 0:
        return frames[-1] + 1
    if frames[0] == 1:
        return frames[-1]
    logger.warning("Frame start is neither 0 nor 1; numFrames inferred as max(frame)+1.")
    return frames[-1] + 1


def _quality_template() -> Dict[str, Any]:
    return {
        "warnings": [],
        "raw_object_category_jump_count": 0,
        "category_jump_tracks": [],
        "category_unstable_tracks": [],
        "gap_tracks": set(),
        "short_gap_count": 0,
        "medium_gap_count": 0,
        "long_gap_count": 0,
        "total_missing_frame_count": 0,
        "interpolated_frame_count": 0,
        "dropped_track_count": 0,
        "duplicate_frame_record_count": 0,
        "duplicate_frame_tracks": set(),
        "track_missing_stats": [],
        "missing_ratio_parameters": {"max_missing_ratio": MAX_MISSING_RATIO},
        "short_track_filter_parameters": dict(SHORT_TRACK_MIN_FRAMES),
        "short_duration_dropped_track_count": 0,
        "short_duration_dropped_tracks": [],
        "heading_parameters": {
            "heading_smooth_window": HEADING_SMOOTH_WINDOW,
            "min_displacement_for_heading": MIN_DISPLACEMENT_FOR_HEADING,
        },
        "outlier_frame_count": 0,
        "split_track_count": 0,
        "stitched_track_count": 0,
        "size_outlier_frame_count": 0,
        "short_tracks": [],
    }


def _summarize_track_set(tracks_meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    class_counts = {cls: 0 for cls in ALL_CLASSES}
    for row in tracks_meta:
        if row["class"] in class_counts:
            class_counts[row["class"]] += 1
    return {
        "numTracks": len(tracks_meta),
        "numVehicles": sum(1 for row in tracks_meta if row["class"] in VEHICLE_CLASSES),
        "numVRUs": sum(1 for row in tracks_meta if row["class"] in VRU_CLASSES),
        "classTrackCounts": class_counts,
    }


def _build_recording_meta(
    recording_id: str,
    location_id: str,
    frame_rate: float,
    num_frames: int,
    ortho: Optional[float],
    summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    duration = num_frames / frame_rate if frame_rate else math.nan
    return [
        {
            "recordingId": recording_id,
            "locationId": location_id,
            "frameRate": frame_rate,
            "numFrames": num_frames,
            "duration": duration,
            "numTracks": summary["numTracks"],
            "numVehicles": summary["numVehicles"],
            "numVRUs": summary["numVRUs"],
            "classTrackCounts": json.dumps(summary["classTrackCounts"], ensure_ascii=False, separators=(",", ":")),
            "orthoPxToMeter": ortho,
        }
    ]


def _track_motion_metrics(track_meta: Dict[str, Any], rows: List[Dict[str, Any]], frame_rate: float) -> Dict[str, Any]:
    sorted_rows = sorted(rows, key=lambda item: int(item["frame"]))
    points = [(float(row["xCenter"]), float(row["yCenter"])) for row in sorted_rows]
    frames = [int(row["frame"]) for row in sorted_rows]
    if len(points) < 2:
        displacement = 0.0
        path_length = 0.0
        mean_speed = 0.0
        max_speed = 0.0
        static_ratio = 1.0
        stationary_extent = 0.0
    else:
        displacement = _dist(points[0], points[-1])
        segment_distances = [_dist(a, b) for a, b in zip(points[:-1], points[1:])]
        path_length = float(sum(segment_distances))
        xs = np.asarray([point[0] for point in points], dtype=float)
        ys = np.asarray([point[1] for point in points], dtype=float)
        median_x = float(np.median(xs))
        median_y = float(np.median(ys))
        radii = np.hypot(xs - median_x, ys - median_y)
        stationary_extent = float(np.percentile(radii, 95) * 2.0) if radii.size else 0.0
        segment_speeds = []
        for dist, f0, f1 in zip(segment_distances, frames[:-1], frames[1:]):
            dt = max((f1 - f0) / frame_rate, 1.0 / frame_rate)
            segment_speeds.append(dist / dt)
        elapsed = max((frames[-1] - frames[0]) / frame_rate, 1.0 / frame_rate)
        mean_speed = path_length / elapsed
        max_speed = max(segment_speeds) if segment_speeds else 0.0
        threshold = float(STATIC_GATE["per_frame_motion_threshold"])
        static_ratio = sum(1 for dist in segment_distances if dist <= threshold) / float(len(segment_distances))

    signals = []
    if displacement <= float(STATIC_GATE["max_displacement"]):
        signals.append("low_displacement")
    if path_length <= float(STATIC_GATE["max_path_length"]):
        signals.append("low_path_length")
    if stationary_extent <= float(STATIC_GATE["max_stationary_extent"]):
        signals.append("low_stationary_extent")
    if mean_speed <= float(STATIC_GATE["max_mean_speed"]):
        signals.append("low_mean_speed")
    if static_ratio >= float(STATIC_GATE["static_ratio_threshold"]):
        signals.append("high_static_ratio")

    cls = track_meta["class"]
    is_static = (
        cls in set(STATIC_GATE["filter_classes"])
        and int(track_meta["numFrames"]) >= int(STATIC_GATE["min_track_length"])
        and "low_stationary_extent" in signals
    )
    return {
        "trackId": int(track_meta["trackId"]),
        "raw_object_id": int(track_meta.get("raw_object_id", track_meta["trackId"])),
        "class": cls,
        "start_frame": int(track_meta["initialFrame"]),
        "end_frame": int(track_meta["finalFrame"]),
        "total_frames": int(track_meta["numFrames"]),
        "start_x": float(track_meta["startXCenter"]),
        "start_y": float(track_meta["startYCenter"]),
        "end_x": float(track_meta["endXCenter"]),
        "end_y": float(track_meta["endYCenter"]),
        "displacement": displacement,
        "path_length": path_length,
        "stationary_extent": stationary_extent,
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "static_ratio": static_ratio,
        "is_static": is_static,
        "filter_reason": ",".join(signals) if is_static else "",
    }


def _moving_filtered_tracks(
    tracks_meta: List[Dict[str, Any]],
    tracks_rows: List[Dict[str, Any]],
    frame_rate: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rows_by_track: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in tracks_rows:
        rows_by_track[int(row["trackId"])].append(row)

    metrics_by_track = {
        int(meta["trackId"]): _track_motion_metrics(meta, rows_by_track.get(int(meta["trackId"]), []), frame_rate)
        for meta in tracks_meta
    }
    filtered_ids = {track_id for track_id, metrics in metrics_by_track.items() if metrics["is_static"]}
    kept_meta_old = [meta for meta in tracks_meta if int(meta["trackId"]) not in filtered_ids]
    kept_meta_old.sort(key=lambda item: (int(item["initialFrame"]), int(item.get("raw_object_id", item["trackId"]))))
    id_map = {int(meta["trackId"]): new_id for new_id, meta in enumerate(kept_meta_old, start=1)}

    filtered_meta = []
    for meta in kept_meta_old:
        item = dict(meta)
        item["trackId"] = id_map[int(meta["trackId"])]
        filtered_meta.append(item)

    filtered_rows = []
    for row in tracks_rows:
        old_id = int(row["trackId"])
        if old_id in filtered_ids:
            continue
        item = dict(row)
        item["trackId"] = id_map[old_id]
        filtered_rows.append(item)
    filtered_rows.sort(key=lambda item: (int(item["trackId"]), int(item["frame"])))

    gate_report = {
        "parameters": STATIC_GATE,
        "original_track_count": len(tracks_meta),
        "filtered_track_count": len(filtered_ids),
        "kept_track_count": len(filtered_meta),
        "filtered_tracks": [metrics_by_track[track_id] for track_id in sorted(filtered_ids)],
        "all_track_metrics": [metrics_by_track[track_id] for track_id in sorted(metrics_by_track)],
    }
    return filtered_meta, filtered_rows, gate_report


def _apply_fragmentation_filter(
    tracks_meta: List[Dict[str, Any]],
    tracks_rows: List[Dict[str, Any]],
    fragmentation_report: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    filtered_raw_ids = {int(raw_id) for raw_id in fragmentation_report.get("filtered_raw_object_ids", [])}
    if not filtered_raw_ids:
        return [dict(row) for row in tracks_meta], [dict(row) for row in tracks_rows]
    kept_meta = [dict(meta) for meta in tracks_meta if int(meta.get("raw_object_id", meta["trackId"])) not in filtered_raw_ids]
    kept_ids = {int(meta["trackId"]) for meta in kept_meta}
    kept_rows = [dict(row) for row in tracks_rows if int(row["trackId"]) in kept_ids]
    return kept_meta, kept_rows


def _sanitized_tracklet_stats(stats_by_raw: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    private_keys = {"_points", "_tail_vx", "_tail_vy", "_tail_heading", "_head_heading"}
    return [{key: value for key, value in stats.items() if key not in private_keys} for _, stats in sorted(stats_by_raw.items())]


def _base_filter_info_by_raw(quality: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    info: Dict[int, Dict[str, Any]] = {}
    for item in quality.get("track_missing_stats", []):
        if item.get("is_dropped"):
            raw_id = int(item["trackId"])
            info[raw_id] = {
                "filter_type": "missing_ratio_filter",
                "filter_reason": item.get("drop_reason") or "missing_ratio_exceeded",
                "fragmentation_group_id": "",
                "related_raw_object_ids": "",
                "fragmentation_score": "",
            }
    for item in quality.get("short_duration_dropped_tracks", []):
        raw_id = int(item["object_id"])
        info[raw_id] = {
            "filter_type": "short_track_filter",
            "filter_reason": item.get("drop_reason") or "track_duration_too_short",
            "fragmentation_group_id": "",
            "related_raw_object_ids": "",
            "fragmentation_score": "",
        }
    return info


def _fragmentation_filter_info_by_raw(fragmentation_report: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    info: Dict[int, Dict[str, Any]] = {}
    for group in fragmentation_report.get("groups", []):
        raw_ids = [int(raw_id) for raw_id in group.get("raw_object_ids", [])]
        group_id = group.get("fragmentation_group_id", "")
        score = group.get("fragmentation_score", "")
        for raw_id in raw_ids:
            related = [str(other) for other in raw_ids if other != raw_id]
            info[raw_id] = {
                "filter_type": "fragmentation_filter",
                "filter_reason": "suspected_id_fragmentation_drop_all_related_tracklets",
                "fragmentation_group_id": group_id,
                "related_raw_object_ids": "|".join(related),
                "fragmentation_score": score,
            }
    return info


def _static_filter_info_by_raw(static_gate_report: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    info: Dict[int, Dict[str, Any]] = {}
    for item in static_gate_report.get("filtered_tracks", []):
        raw_id = int(item.get("raw_object_id", item.get("trackId")))
        info[raw_id] = {
            "filter_type": "static_gate",
            "filter_reason": item.get("filter_reason") or "stationary_track",
            "fragmentation_group_id": "",
            "related_raw_object_ids": "",
            "fragmentation_score": "",
        }
    return info


def _build_id_mapping_rows(
    dataset_id: str,
    version: str,
    stats_by_raw: Dict[int, Dict[str, Any]],
    tracks_meta: List[Dict[str, Any]],
    filter_info_by_raw: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    final_by_raw = {
        int(meta.get("raw_object_id", meta["trackId"])): int(meta["trackId"])
        for meta in tracks_meta
    }
    rows = []
    for raw_id, stats in sorted(stats_by_raw.items()):
        kept = raw_id in final_by_raw
        filter_info = filter_info_by_raw.get(raw_id, {})
        rows.append(
            {
                "dataset_id": dataset_id,
                "version": version,
                "raw_object_id": raw_id,
                "final_object_id": final_by_raw.get(raw_id, ""),
                "class_name_mode": stats.get("class_name_mode", ""),
                "start_frame": stats.get("start_frame", ""),
                "end_frame": stats.get("end_frame", ""),
                "total_frames": stats.get("total_frames", ""),
                "mean_confidence": stats.get("mean_confidence", ""),
                "is_kept": kept,
                "is_filtered": not kept,
                "filter_type": "" if kept else filter_info.get("filter_type", "not_in_version"),
                "filter_reason": "" if kept else filter_info.get("filter_reason", "not_kept_after_conversion"),
                "fragmentation_group_id": "" if kept else filter_info.get("fragmentation_group_id", ""),
                "quality_score": stats.get("quality_score", ""),
            }
        )
    return rows


def _build_filter_report_rows(
    dataset_id: str,
    version: str,
    stats_by_raw: Dict[int, Dict[str, Any]],
    tracks_meta: List[Dict[str, Any]],
    filter_info_by_raw: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    kept_raw_ids = {int(meta.get("raw_object_id", meta["trackId"])) for meta in tracks_meta}
    rows = []
    for raw_id, stats in sorted(stats_by_raw.items()):
        if raw_id in kept_raw_ids:
            continue
        filter_info = filter_info_by_raw.get(raw_id, {})
        rows.append(
            {
                "dataset_id": dataset_id,
                "version": version,
                "raw_object_id": raw_id,
                "filter_type": filter_info.get("filter_type", "not_in_version"),
                "filter_reason": filter_info.get("filter_reason", "not_kept_after_conversion"),
                "fragmentation_group_id": filter_info.get("fragmentation_group_id", ""),
                "related_raw_object_ids": filter_info.get("related_raw_object_ids", ""),
                "fragmentation_score": filter_info.get("fragmentation_score", ""),
                "quality_score": stats.get("quality_score", ""),
                "start_frame": stats.get("start_frame", ""),
                "end_frame": stats.get("end_frame", ""),
                "total_frames": stats.get("total_frames", ""),
                "class_name_mode": stats.get("class_name_mode", ""),
            }
        )
    return rows


def _write_dataset_version(
    version_dir: Path,
    folder_name: str,
    recording_meta: List[Dict[str, Any]],
    tracks_meta: List[Dict[str, Any]],
    tracks_rows: List[Dict[str, Any]],
    report: Dict[str, Any],
    id_mapping_rows: List[Dict[str, Any]],
    filter_report_rows: List[Dict[str, Any]],
    log_lines: List[str],
) -> Dict[str, str]:
    version_dir.mkdir(parents=True, exist_ok=True)
    rec_path = version_dir / f"{folder_name}_recordingMeta.csv"
    meta_path = version_dir / f"{folder_name}_tracksMeta.csv"
    tracks_path = version_dir / f"{folder_name}_tracks.csv"
    id_mapping_path = version_dir / "id_mapping.csv"
    filter_report_path = version_dir / "filter_report.csv"
    _write_csv(rec_path, RECORDING_META_FIELDS, recording_meta)
    _write_csv(meta_path, TRACKS_META_FIELDS, tracks_meta)
    _write_csv(tracks_path, TRACKS_FIELDS, tracks_rows)
    _write_csv(id_mapping_path, ID_MAPPING_FIELDS, id_mapping_rows)
    _write_csv(filter_report_path, FILTER_REPORT_FIELDS, filter_report_rows)
    with (version_dir / "quality_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    metadata = {
        "dataset_id": folder_name,
        "version": report.get("version"),
        "use_fragmentation_filter": bool(report.get("fragmentationFilter", {}).get("use_fragmentation_filter", False)),
        "fragmentation_filter_config": report.get("fragmentationFilter", {}).get("parameters", {}),
        "fragmentation_filter_strategy": report.get("fragmentationFilter", {}).get("strategy", FRAGMENTATION_FILTER["filter_strategy"]),
        "raw_track_count": report.get("rawObjectCount", 0),
        "kept_track_count": len(tracks_meta),
        "filtered_track_count": max(len(id_mapping_rows) - len(tracks_meta), 0),
        "fragmentation_filtered_count": report.get("fragmentationFilter", {}).get("filtered_track_count", 0),
        "static_filtered_count": report.get("staticGate", {}).get("filtered_track_count", 0) if report.get("version") == "moving_filtered" else 0,
        "id_mapping_file": "id_mapping.csv",
        "filter_report_file": "filter_report.csv",
    }
    with (version_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    with (version_dir / "conversion_log.txt").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")
    return {
        "recordingMeta": str(rec_path),
        "tracksMeta": str(meta_path),
        "tracks": str(tracks_path),
        "idMapping": str(id_mapping_path),
        "filterReport": str(filter_report_path),
        "metadata": str(version_dir / "metadata.json"),
        "qualityReport": str(version_dir / "quality_report.json"),
        "conversionLog": str(version_dir / "conversion_log.txt"),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def convert_dataset(
    dataset_dir: Path,
    output_root: Path = DEFAULT_ADJUSTED_ROOT,
    initial_root: Path = DEFAULT_INITIAL_ROOT,
    force: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    logger = logger or configure_logger()
    dataset_dir = Path(dataset_dir).resolve()
    folder_name = dataset_dir.name
    output_dir = Path(output_root) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_log_lines: List[str] = []
    quality = _quality_template()

    def log_line(level: str, message: str) -> None:
        dataset_log_lines.append(f"[{level}] {message}")
        if level == "WARNING":
            logger.warning(message)
            quality["warnings"].append(message)
        else:
            logger.info(message)

    recording_id, location_id = _parse_folder_identity(folder_name, quality["warnings"])
    log_line("INFO", f"input_folder={dataset_dir}")
    log_line("INFO", f"recordingId={recording_id}, locationId={location_id}")

    det_pkl = _find_detection_pkl(dataset_dir)
    if det_pkl is None:
        raise ConversionError(f"No det_bbox_result_*.pkl found in {dataset_dir}.")
    stab_pkl = _find_stabilization_pkl(dataset_dir)
    log_line("INFO", f"detection_pkl={det_pkl.name}")
    if stab_pkl:
        log_line("INFO", f"stabilization_pkl={stab_pkl.name} (not used for dynamics)")

    with det_pkl.open("rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict):
        raise ConversionError(f"{det_pkl.name} must contain a dict.")

    video_info = _video_info(data)
    output_info = data.get("output_info", {}) if isinstance(data.get("output_info"), dict) else {}
    frame_rate = _safe_float(output_info.get("output_fps")) or 29.97
    log_line(
        "INFO",
        "video_info width=%s height=%s fps=%s total_frames=%s"
        % (video_info.get("width"), video_info.get("height"), video_info.get("fps"), video_info.get("total_frames")),
    )
    log_line("INFO", f"frameRate={frame_rate}")

    raw_rows, frame_meta, col_counts = _expand_traj_info(data, logger, quality)
    if not raw_rows:
        raise ConversionError("No valid world-coordinate trajectory rows found.")
    raw_object_count = len({row["object_id"] for row in raw_rows})
    raw_stats_by_id = _raw_tracklet_stats(raw_rows, folder_name, frame_rate)
    quality["raw_tracklet_stats"] = _sanitized_tracklet_stats(raw_stats_by_id)
    log_line("INFO", f"raw_object_id_count={raw_object_count}")

    fragments = _split_raw_tracks(raw_rows, frame_rate, logger, quality)
    tracks_meta, tracks_rows = _build_final_tracks(fragments, frame_rate, logger, quality)
    missing_ratios = [float(item["missing_ratio"]) for item in quality["track_missing_stats"]]
    quality["missing_ratio_summary"] = {
        "raw_track_count": raw_object_count,
        "kept_track_count": len(tracks_meta),
        "missing_ratio_dropped_track_count": quality["dropped_track_count"],
        "short_duration_dropped_track_count": quality["short_duration_dropped_track_count"],
        "dropped_track_count": quality["dropped_track_count"] + quality["short_duration_dropped_track_count"],
        "total_interpolated_frames": quality["interpolated_frame_count"],
        "mean_missing_ratio": float(np.mean(missing_ratios)) if missing_ratios else 0.0,
        "max_missing_ratio": float(np.max(missing_ratios)) if missing_ratios else 0.0,
    }
    for row in tracks_meta:
        row["recordingId"] = recording_id
    for row in tracks_rows:
        row["recordingId"] = recording_id

    num_frames = _num_frames(video_info, frame_meta, logger)
    ortho = _estimate_ortho_px_to_meter(raw_rows, logger)
    if ortho is None:
        log_line("WARNING", "orthoPxToMeter could not be estimated reliably; output is empty.")
    else:
        log_line("INFO", f"orthoPxToMeter={ortho:.6f}")

    full_summary = _summarize_track_set(tracks_meta)
    full_recording_meta = _build_recording_meta(recording_id, location_id, frame_rate, num_frames, ortho, full_summary)
    full_raw_ids = {int(meta.get("raw_object_id", meta["trackId"])) for meta in tracks_meta}
    fragmentation_report = _detect_fragmentation_groups(raw_stats_by_id, full_raw_ids, video_info)
    fragmentation_meta, fragmentation_rows = _apply_fragmentation_filter(tracks_meta, tracks_rows, fragmentation_report)
    moving_meta, moving_rows, static_gate_report = _moving_filtered_tracks(fragmentation_meta, fragmentation_rows, frame_rate)
    moving_summary = _summarize_track_set(moving_meta)
    moving_recording_meta = _build_recording_meta(recording_id, location_id, frame_rate, num_frames, ortho, moving_summary)

    base_filter_info = _base_filter_info_by_raw(quality)
    full_filter_info = dict(base_filter_info)
    moving_filter_info = dict(base_filter_info)
    moving_filter_info.update(_fragmentation_filter_info_by_raw(fragmentation_report))
    moving_filter_info.update(_static_filter_info_by_raw(static_gate_report))
    full_id_mapping_rows = _build_id_mapping_rows(folder_name, "full", raw_stats_by_id, tracks_meta, full_filter_info)
    moving_id_mapping_rows = _build_id_mapping_rows(folder_name, "moving_filtered", raw_stats_by_id, moving_meta, moving_filter_info)
    full_filter_report_rows = _build_filter_report_rows(folder_name, "full", raw_stats_by_id, tracks_meta, full_filter_info)
    moving_filter_report_rows = _build_filter_report_rows(folder_name, "moving_filtered", raw_stats_by_id, moving_meta, moving_filter_info)

    log_line("INFO", f"final_track_count={full_summary['numTracks']}")
    log_line("INFO", f"classTrackCounts={full_summary['classTrackCounts']}")
    log_line("INFO", "output_versions=full,moving_filtered")
    log_line("INFO", f"fragmentation_filter_parameters={FRAGMENTATION_FILTER}")
    log_line("INFO", f"fragmentation_filter_strategy={FRAGMENTATION_FILTER['filter_strategy']}")
    log_line("INFO", f"full fragmentation_filtered_count=0 (fragmentation filter disabled for full)")
    log_line(
        "INFO",
        "moving_filtered fragmentation_groups=%s fragmentation_filtered_tracks=%s"
        % (len(fragmentation_report.get("groups", [])), fragmentation_report.get("filtered_track_count", 0)),
    )
    for group in fragmentation_report.get("groups", []):
        log_line(
            "INFO",
            "fragmentation_group %s raw_object_ids=%s score=%.4f strategy=%s reason=%s"
            % (
                group["fragmentation_group_id"],
                group["raw_object_ids"],
                float(group.get("fragmentation_score", 0.0)),
                FRAGMENTATION_FILTER["filter_strategy"],
                group["filter_reason"],
            ),
        )
    log_line("INFO", f"static_gate_parameters={STATIC_GATE}")
    log_line(
        "INFO",
        "moving_filtered original_tracks=%s filtered_tracks=%s kept_tracks=%s"
        % (
            static_gate_report["original_track_count"],
            static_gate_report["filtered_track_count"],
            static_gate_report["kept_track_count"],
        ),
    )
    for item in static_gate_report["filtered_tracks"]:
        log_line(
            "INFO",
            "static_filtered trackId=%s raw_object_id=%s class=%s frames=%s displacement=%.4f path_length=%.4f "
            "stationary_extent=%.4f mean_speed=%.4f static_ratio=%.4f reason=%s"
            % (
                item["trackId"],
                item.get("raw_object_id", item["trackId"]),
                item["class"],
                item["total_frames"],
                item["displacement"],
                item["path_length"],
                item.get("stationary_extent", 0.0),
                item["mean_speed"],
                item["static_ratio"],
                item["filter_reason"],
            ),
        )
    log_line("INFO", f"category_jump_object_count={quality['raw_object_category_jump_count']}")
    log_line("INFO", f"category_jump_final_track_count={len(quality['category_jump_tracks'])}")
    for item in quality["category_jump_tracks"]:
        log_line(
            "INFO",
            "category_jump detail track=%s class_counts=%s final_class=%s final_class_ratio=%.4f"
            % (item["track"], item["class_counts"], item["final_class"], item["final_class_ratio"]),
        )
    log_line("INFO", f"category_unstable_track_count={len(quality['category_unstable_tracks'])}")
    for item in quality["category_unstable_tracks"]:
        log_line(
            "WARNING",
            "category_unstable detail track=%s class_counts=%s final_class=%s final_class_ratio=%.4f"
            % (item["track"], item["class_counts"], item["final_class"], item["final_class_ratio"]),
        )
    log_line("INFO", f"lane_id_minus_one_records={quality.get('lane_id_minus_one_records', 0)}")
    log_line("INFO", f"missing_ratio_parameters={quality['missing_ratio_parameters']}")
    log_line("INFO", f"heading_parameters={quality['heading_parameters']}")
    log_line("INFO", f"short_track_filter_parameters={quality['short_track_filter_parameters']}")
    log_line("INFO", f"raw_track_count={quality['missing_ratio_summary']['raw_track_count']}")
    log_line("INFO", f"kept_track_count={quality['missing_ratio_summary']['kept_track_count']}")
    log_line("INFO", f"missing_ratio_dropped_track_count={quality['missing_ratio_summary']['missing_ratio_dropped_track_count']}")
    log_line("INFO", f"short_duration_dropped_track_count={quality['short_duration_dropped_track_count']}")
    log_line("INFO", f"dropped_track_count={quality['missing_ratio_summary']['dropped_track_count']}")
    log_line("INFO", f"duplicate_frame_record_count={quality['duplicate_frame_record_count']}")
    log_line("INFO", f"gap_track_count={len(quality['gap_tracks'])}")
    log_line("INFO", f"total_missing_frames_before_interpolation={quality['total_missing_frame_count']}")
    log_line("INFO", f"short_gap_missing_frames={quality['short_gap_count']}")
    log_line("INFO", f"medium_gap_missing_frames={quality['medium_gap_count']}")
    log_line("INFO", f"long_gap_missing_frames={quality['long_gap_count']}")
    log_line("INFO", f"interpolated_frame_count={quality['interpolated_frame_count']}")
    log_line("INFO", f"mean_missing_ratio={quality['missing_ratio_summary']['mean_missing_ratio']:.4f}")
    log_line("INFO", f"max_missing_ratio={quality['missing_ratio_summary']['max_missing_ratio']:.4f}")
    log_line("INFO", f"outlier_frame_count={quality['outlier_frame_count']}")
    log_line("INFO", f"split_track_count={quality['split_track_count']}")
    log_line("INFO", f"stitched_track_count={quality['stitched_track_count']}")
    log_line("INFO", f"size_outlier_frame_count={quality['size_outlier_frame_count']}")
    log_line("INFO", f"valid_size_rows_used≈{sum(row['numFrames'] for row in tracks_meta) - quality['size_outlier_frame_count']}")
    log_line("INFO", f"short_track_count={len(quality['short_tracks'])}")
    if quality["short_tracks"]:
        log_line("WARNING", f"short_tracks={quality['short_tracks'][:50]}")
    log_line("INFO", f"unknown_category_ids={quality.get('unknown_category_ids', [])}")

    base_report = {
        "folderName": folder_name,
        "recordingId": recording_id,
        "locationId": location_id,
        "detectionPkl": str(det_pkl),
        "stabilizationPkl": str(stab_pkl) if stab_pkl else None,
        "videoInfo": video_info,
        "outputInfo": output_info,
        "frameRate": frame_rate,
        "arrayColumnCounts": dict(col_counts),
        "rawObjectCount": raw_object_count,
        "orthoPxToMeter": ortho,
        "quality": _json_safe(quality),
        "staticGate": _json_safe(static_gate_report),
    }

    full_report = dict(base_report)
    full_report.update(
        {
            "version": "full",
            "fragmentationFilter": {
                "use_fragmentation_filter": False,
                "parameters": dict(FRAGMENTATION_FILTER),
                "strategy": FRAGMENTATION_FILTER["filter_strategy"],
                "groups": [],
                "filtered_track_count": 0,
            },
            "finalTrackCount": full_summary["numTracks"],
            "classTrackCounts": full_summary["classTrackCounts"],
            "numVehicles": full_summary["numVehicles"],
            "numVRUs": full_summary["numVRUs"],
        }
    )
    moving_report = dict(base_report)
    moving_report.update(
        {
            "version": "moving_filtered",
            "fragmentationFilter": {
                **_json_safe(fragmentation_report),
                "use_fragmentation_filter": True,
            },
            "finalTrackCount": moving_summary["numTracks"],
            "classTrackCounts": moving_summary["classTrackCounts"],
            "numVehicles": moving_summary["numVehicles"],
            "numVRUs": moving_summary["numVRUs"],
        }
    )

    version_outputs = {
        "full": _write_dataset_version(
            output_dir / "full",
            folder_name,
            full_recording_meta,
            tracks_meta,
            tracks_rows,
            full_report,
            full_id_mapping_rows,
            full_filter_report_rows,
            dataset_log_lines,
        ),
        "moving_filtered": _write_dataset_version(
            output_dir / "moving_filtered",
            folder_name,
            moving_recording_meta,
            moving_meta,
            moving_rows,
            moving_report,
            moving_id_mapping_rows,
            moving_filter_report_rows,
            dataset_log_lines,
        ),
    }

    return {
        "dataset_id": folder_name,
        "status": "converted",
        "recordingId": recording_id,
        "locationId": location_id,
        "numTracks": full_summary["numTracks"],
        "versions": {
            "full": {
                "numTracks": full_summary["numTracks"],
                "outputs": version_outputs["full"],
            },
            "moving_filtered": {
                "numTracks": moving_summary["numTracks"],
                "filteredTracks": static_gate_report["filtered_track_count"] + fragmentation_report.get("filtered_track_count", 0),
                "staticFilteredTracks": static_gate_report["filtered_track_count"],
                "fragmentationFilteredTracks": fragmentation_report.get("filtered_track_count", 0),
                "outputs": version_outputs["moving_filtered"],
            },
        },
    }


def find_dataset_dirs(source_root: Path) -> List[Path]:
    source_root = Path(source_root)
    if not source_root.exists():
        return []
    return [child for child in sorted(source_root.iterdir()) if child.is_dir() and any(child.glob("*.pkl"))]


def convert_all(
    source_root: Path = DEFAULT_INITIAL_ROOT,
    output_root: Path = DEFAULT_ADJUSTED_ROOT,
    force: bool = False,
    datasets: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    logger = logger or configure_logger()
    source_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    selected = set(datasets or [])
    results = []
    for dataset_dir in find_dataset_dirs(source_root):
        if selected and dataset_dir.name not in selected:
            continue
        try:
            results.append(convert_dataset(dataset_dir, output_root, source_root, force, logger))
        except Exception as exc:
            logger.exception("Failed to convert %s", dataset_dir.name)
            results.append({"dataset_id": dataset_dir.name, "status": "failed", "error": str(exc)})
    return {"source_root": str(source_root), "output_root": str(output_root), "force": force, "results": results}


def inspect_pkl_structure(pkl_path: Path) -> Dict[str, Any]:
    with pkl_path.open("rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict):
        return {"path": str(pkl_path), "type": type(data).__name__}
    summary = {"path": str(pkl_path), "top_level_keys": list(data.keys())}
    traj = data.get("traj_info")
    if isinstance(traj, list):
        summary["traj_info_length"] = len(traj)
        col_counts = Counter()
        for entry in traj:
            try:
                _, _, arr, _ = _format_entry(entry)
            except Exception:
                continue
            if isinstance(arr, np.ndarray):
                col_counts[arr.shape[1] if arr.ndim == 2 else arr.shape[0]] += 1
        summary["array_column_count_frequencies"] = dict(col_counts)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OpenVTER pkl result folders to standardized trajectory CSV files.")
    parser.add_argument("--source-root", default=str(DEFAULT_INITIAL_ROOT), help="Folder containing raw result subfolders.")
    parser.add_argument("--output-root", default=str(DEFAULT_ADJUSTED_ROOT), help="Folder for standardized CSV datasets.")
    parser.add_argument("--force", action="store_true", help="Re-run conversion.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Only convert these dataset folder names.")
    parser.add_argument("--inspect", default=None, help="Only inspect a detection pkl and print structure JSON.")
    args = parser.parse_args()

    logger = configure_logger()
    if args.inspect:
        print(json.dumps(inspect_pkl_structure(Path(args.inspect)), ensure_ascii=False, indent=2))
        return
    result = convert_all(Path(args.source_root), Path(args.output_root), args.force, args.datasets, logger)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
