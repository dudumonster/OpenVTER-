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
LONG_GAP_SPLIT = 30
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

# The converter writes two dataset versions:
# full keeps every cleaned track, while moving_filtered removes long-lived,
# nearly stationary motorized tracks. Values are in the current SI trajectory
# units, so displacement is meters, mean_speed is m/s, and per-frame motion is m.
STATIC_GATE = {
    "min_track_length": 30,
    "max_displacement": 1.0,
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
    "class",
]

TRACKS_FIELDS = [
    "recordingId",
    "trackId",
    "lane_id",
    "frame",
    "trackLifetime",
    "xCenter",
    "yCenter",
    "heading",
    "width",
    "length",
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


def _should_split(prev: Dict[str, Any], cur: Dict[str, Any], final_class: str, frame_rate: float) -> bool:
    gap = int(cur["frame"]) - int(prev["frame"]) - 1
    if gap > LONG_GAP_SPLIT:
        return True
    dt = max((int(cur["frame"]) - int(prev["frame"])) / frame_rate, 1.0 / frame_rate)
    speed = _dist((prev["xCenter_raw"], prev["yCenter_raw"]), (cur["xCenter_raw"], cur["yCenter_raw"])) / dt
    limit = PHYSICAL_LIMITS.get(final_class, PHYSICAL_LIMITS["car"])["max_speed"]
    return speed > limit * 1.5


def _split_raw_tracks(raw_rows: List[Dict[str, Any]], frame_rate: float, logger: logging.Logger, quality: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[int(row["object_id"])].append(row)

    fragments: List[List[Dict[str, Any]]] = []
    for object_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: int(item["frame"]))
        final_class, _, _, class_counts = _mode_class(rows, logger, f"object_id={object_id}", quality)
        if len(class_counts) > 1:
            quality["raw_object_category_jump_count"] += 1
        current: List[Dict[str, Any]] = [rows[0]]
        for prev, cur in zip(rows[:-1], rows[1:]):
            gap = int(cur["frame"]) - int(prev["frame"]) - 1
            if gap > 0:
                quality["gap_tracks"].add(object_id)
                if gap <= SHORT_GAP_MAX:
                    quality["short_gap_count"] += gap
                elif gap <= MEDIUM_GAP_MAX:
                    quality["medium_gap_count"] += gap
                else:
                    quality["long_gap_count"] += gap
            if _should_split(prev, cur, final_class, frame_rate):
                fragments.append(current)
                current = [cur]
                quality["split_track_count"] += 1
            else:
                current.append(cur)
        fragments.append(current)
    return fragments


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
    if gap > SHORT_GAP_MAX:
        dt = (int(cur["frame"]) - int(prev["frame"])) / frame_rate
        speed = _dist((prev["xCenter_raw"], prev["yCenter_raw"]), (cur["xCenter_raw"], cur["yCenter_raw"])) / max(dt, 1.0 / frame_rate)
        limit = PHYSICAL_LIMITS.get(final_class, PHYSICAL_LIMITS["car"])["max_speed"]
        same_or_known_lane = prev["lane_id"] == cur["lane_id"] or prev["lane_id"] == -1 or cur["lane_id"] == -1
        if gap > MEDIUM_GAP_MAX or speed > limit or not same_or_known_lane:
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


def _compute_heading(xs: List[float], ys: List[float], rows: List[Dict[str, Any]], frame_rate: float, logger: logging.Logger, label: str) -> List[float]:
    n = len(xs)
    half_window = max(1, int(round(0.25 * frame_rate)))
    headings: List[Optional[float]] = []
    last_valid: Optional[float] = None
    for i in range(n):
        j0 = max(0, i - half_window)
        j1 = min(n - 1, i + half_window)
        dx = xs[j1] - xs[j0]
        dy = ys[j1] - ys[j0]
        if math.hypot(dx, dy) >= 0.2:
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
    return [float(h if h is not None else fallback) for h in headings]


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
        for prev, cur in zip(fragment[:-1], fragment[1:]):
            with_gaps.append(prev)
            inserts = _interpolate_between(prev, cur, final_class, frame_rate)
            if inserts:
                quality["interpolated_frame_count"] += len(inserts)
            with_gaps.extend(inserts)
        with_gaps.append(fragment[-1])
        with_gaps = sorted(with_gaps, key=lambda item: int(item["frame"]))
        prepared.append(
            {
                "rows": with_gaps,
                "final_class": final_class,
                "final_class_ratio": ratio,
                "category_unstable": unstable,
                "class_counts": class_counts,
                "original_object_id": int(fragment[0]["object_id"]),
            }
        )

    # Conservative stitching is intentionally disabled unless a single unambiguous candidate is found.
    # Current implementation records zero merges rather than making aggressive ID merges.
    quality["stitched_track_count"] = 0
    prepared.sort(key=lambda item: (item["rows"][0]["frame"], item["original_object_id"]))

    class_dim_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"widths": [], "lengths": []})
    track_payloads = []
    for item in prepared:
        valid_w, valid_l = _valid_dimensions(item["rows"], quality)
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
            med_w = [row["raw_width"] for row in rows if _finite(row.get("raw_width"))]
            med_l = [row["raw_length"] for row in rows if _finite(row.get("raw_length"))]
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
                    "lane_id": int(row.get("lane_id", -1)) if _finite(row.get("lane_id", -1)) else -1,
                    "frame": int(row["frame"]),
                    "trackLifetime": lifetime,
                    "xCenter": x,
                    "yCenter": y,
                    "heading": heading,
                    "width": mean_width,
                    "length": mean_length,
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
                "initialFrame": frames[0],
                "finalFrame": frames[-1],
                "numFrames": len(rows),
                "startXCenter": xs[0],
                "startYCenter": ys[0],
                "endXCenter": xs[-1],
                "endYCenter": ys[-1],
                "startLaneId": int(rows[0].get("lane_id", -1)) if _finite(rows[0].get("lane_id", -1)) else -1,
                "endLaneId": int(rows[-1].get("lane_id", -1)) if _finite(rows[-1].get("lane_id", -1)) else -1,
                "width": mean_width,
                "length": mean_length,
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
        "interpolated_frame_count": 0,
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
    else:
        displacement = _dist(points[0], points[-1])
        segment_distances = [_dist(a, b) for a, b in zip(points[:-1], points[1:])]
        path_length = float(sum(segment_distances))
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
    if mean_speed <= float(STATIC_GATE["max_mean_speed"]):
        signals.append("low_mean_speed")
    if static_ratio >= float(STATIC_GATE["static_ratio_threshold"]):
        signals.append("high_static_ratio")

    cls = track_meta["class"]
    is_static = (
        cls in set(STATIC_GATE["filter_classes"])
        and int(track_meta["numFrames"]) >= int(STATIC_GATE["min_track_length"])
        and len(signals) >= 2
    )
    return {
        "trackId": int(track_meta["trackId"]),
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
    kept_meta_old.sort(key=lambda item: (int(item["initialFrame"]), int(item["trackId"])))
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


def _write_dataset_version(
    version_dir: Path,
    folder_name: str,
    recording_meta: List[Dict[str, Any]],
    tracks_meta: List[Dict[str, Any]],
    tracks_rows: List[Dict[str, Any]],
    report: Dict[str, Any],
    log_lines: List[str],
) -> Dict[str, str]:
    version_dir.mkdir(parents=True, exist_ok=True)
    rec_path = version_dir / f"{folder_name}_recordingMeta.csv"
    meta_path = version_dir / f"{folder_name}_tracksMeta.csv"
    tracks_path = version_dir / f"{folder_name}_tracks.csv"
    _write_csv(rec_path, RECORDING_META_FIELDS, recording_meta)
    _write_csv(meta_path, TRACKS_META_FIELDS, tracks_meta)
    _write_csv(tracks_path, TRACKS_FIELDS, tracks_rows)
    with (version_dir / "quality_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    with (version_dir / "conversion_log.txt").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")
    return {
        "recordingMeta": str(rec_path),
        "tracksMeta": str(meta_path),
        "tracks": str(tracks_path),
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
    log_line("INFO", f"raw_object_id_count={raw_object_count}")

    fragments = _split_raw_tracks(raw_rows, frame_rate, logger, quality)
    tracks_meta, tracks_rows = _build_final_tracks(fragments, frame_rate, logger, quality)
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
    moving_meta, moving_rows, static_gate_report = _moving_filtered_tracks(tracks_meta, tracks_rows, frame_rate)
    moving_summary = _summarize_track_set(moving_meta)
    moving_recording_meta = _build_recording_meta(recording_id, location_id, frame_rate, num_frames, ortho, moving_summary)

    log_line("INFO", f"final_track_count={full_summary['numTracks']}")
    log_line("INFO", f"classTrackCounts={full_summary['classTrackCounts']}")
    log_line("INFO", "output_versions=full,moving_filtered")
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
            "static_filtered trackId=%s class=%s frames=%s displacement=%.4f path_length=%.4f "
            "mean_speed=%.4f static_ratio=%.4f reason=%s"
            % (
                item["trackId"],
                item["class"],
                item["total_frames"],
                item["displacement"],
                item["path_length"],
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
    log_line("INFO", f"gap_track_count={len(quality['gap_tracks'])}")
    log_line("INFO", f"short_gap_missing_frames={quality['short_gap_count']}")
    log_line("INFO", f"medium_gap_missing_frames={quality['medium_gap_count']}")
    log_line("INFO", f"long_gap_missing_frames={quality['long_gap_count']}")
    log_line("INFO", f"interpolated_frame_count={quality['interpolated_frame_count']}")
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
            "finalTrackCount": moving_summary["numTracks"],
            "classTrackCounts": moving_summary["classTrackCounts"],
            "numVehicles": moving_summary["numVehicles"],
            "numVRUs": moving_summary["numVRUs"],
        }
    )

    version_outputs = {
        "full": _write_dataset_version(output_dir / "full", folder_name, full_recording_meta, tracks_meta, tracks_rows, full_report, dataset_log_lines),
        "moving_filtered": _write_dataset_version(
            output_dir / "moving_filtered",
            folder_name,
            moving_recording_meta,
            moving_meta,
            moving_rows,
            moving_report,
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
                "filteredTracks": static_gate_report["filtered_track_count"],
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
