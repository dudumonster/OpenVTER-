#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert OpenVTER pkl result folders into CSV datasets for visualization."""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import logging
import math
import pickle
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


VIS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INITIAL_ROOT = VIS_ROOT / "Initial results"
DEFAULT_ADJUSTED_ROOT = VIS_ROOT / "Adjusted results"
DEFAULT_LOG_PATH = VIS_ROOT / "logs" / "conversion.log"

CATEGORY_NAMES = [
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

STATIC_GATE_CONFIG = {
    "min_track_length": 30,
    "max_displacement": 10.0,
    "max_mean_speed": 0.5,
    "static_ratio_threshold": 0.8,
    "per_frame_motion_threshold": 1.0,
}

STATIC_GATE_VEHICLE_CLASSES = {"car", "truck", "bus", "freight_car", "van"}

CORE_TRACK_FIELDS = [
    "dataset_id",
    "frame_id",
    "object_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "width",
    "height",
]

EXTRA_TRACK_FIELDS = [
    "category_id",
    "output_frame",
    "timestamp",
    "angle_deg",
    "q1_x",
    "q1_y",
    "q2_x",
    "q2_y",
    "q3_x",
    "q3_y",
    "q4_x",
    "q4_y",
    "world_q1_x",
    "world_q1_y",
    "world_q2_x",
    "world_q2_y",
    "world_q3_x",
    "world_q3_y",
    "world_q4_x",
    "world_q4_y",
    "lane_id",
    "source_row_index",
]


class ConversionError(RuntimeError):
    """Raised when a dataset cannot be converted into the expected CSV shape."""


def configure_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("visualization_converter")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def category_name(category_id: Optional[int]) -> str:
    if category_id is None:
        return ""
    if 0 <= category_id < len(CATEGORY_NAMES):
        return CATEGORY_NAMES[category_id]
    return f"class_{category_id}"


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return value


def _format_entry(entry: Tuple[Any, ...]) -> Tuple[int, int, Any, Optional[Any]]:
    if not isinstance(entry, tuple) or len(entry) not in (3, 4):
        raise ConversionError(f"traj_info entry must be tuple(frame_id, output_frame, array[, timestamp]), got {type(entry)}")
    frame_id, output_frame, arr = entry[:3]
    timestamp = entry[3] if len(entry) == 4 else None
    return int(frame_id), int(output_frame), arr, timestamp


def infer_detection_columns(num_cols: int) -> List[str]:
    base = [
        "q1_x",
        "q1_y",
        "q2_x",
        "q2_y",
        "q3_x",
        "q3_y",
        "q4_x",
        "q4_y",
        "confidence",
        "category_id",
        "object_id",
    ]
    if num_cols == 11:
        return base
    if num_cols == 19:
        return base + [
            "world_q1_x",
            "world_q1_y",
            "world_q2_x",
            "world_q2_y",
            "world_q3_x",
            "world_q3_y",
            "world_q4_x",
            "world_q4_y",
        ]
    if num_cols == 20:
        return base + [
            "world_q1_x",
            "world_q1_y",
            "world_q2_x",
            "world_q2_y",
            "world_q3_x",
            "world_q3_y",
            "world_q4_x",
            "world_q4_y",
            "lane_id",
        ]
    raise ConversionError(
        f"Unexpected traj_info array column count {num_cols}; expected 11, 19, or 20."
    )


def inspect_pkl_structure(pkl_path: Path) -> Dict[str, Any]:
    with pkl_path.open("rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict):
        raise ConversionError(f"{pkl_path.name} is {type(data).__name__}, expected dict.")

    summary: Dict[str, Any] = {
        "path": str(pkl_path),
        "top_level_type": "dict",
        "top_level_keys": list(data.keys()),
    }
    traj_info = data.get("traj_info")
    if isinstance(traj_info, list):
        summary["traj_info_length"] = len(traj_info)
        frame_ids: List[int] = []
        output_frames: List[int] = []
        column_counts: Counter[int] = Counter()
        non_empty_shapes: List[List[int]] = []
        object_ids: set[int] = set()
        category_ids: set[int] = set()
        first_non_empty: Optional[np.ndarray] = None

        for entry in traj_info:
            try:
                frame_id, output_frame, arr, _ = _format_entry(entry)
            except Exception:
                continue
            frame_ids.append(frame_id)
            output_frames.append(output_frame)
            if isinstance(arr, np.ndarray):
                cols = arr.shape[1] if arr.ndim == 2 else arr.shape[0]
                column_counts[cols] += 1
                if arr.ndim == 2 and len(arr) > 0:
                    non_empty_shapes.append(list(arr.shape))
                    first_non_empty = arr if first_non_empty is None else first_non_empty
                    if arr.shape[1] > 10:
                        object_ids.update(int(v) for v in arr[:, 10])
                    if arr.shape[1] > 9:
                        category_ids.update(int(v) for v in arr[:, 9])

        summary.update(
            {
                "frame_id_min": min(frame_ids) if frame_ids else None,
                "frame_id_max": max(frame_ids) if frame_ids else None,
                "output_frame_min": min(output_frames) if output_frames else None,
                "output_frame_max": max(output_frames) if output_frames else None,
                "array_column_count_frequencies": dict(column_counts),
                "non_empty_frame_count": len(non_empty_shapes),
                "first_non_empty_shape": non_empty_shapes[0] if non_empty_shapes else None,
                "object_id_count": len(object_ids),
                "object_id_min": min(object_ids) if object_ids else None,
                "object_id_max": max(object_ids) if object_ids else None,
                "category_ids": sorted(category_ids),
            }
        )
        if first_non_empty is not None:
            cols = infer_detection_columns(first_non_empty.shape[1])
            summary["traj_info_column_mapping"] = cols
            summary["first_rows"] = first_non_empty[: min(5, len(first_non_empty))].tolist()
    return summary


def inspect_stabilization_pkl(stab_path: Path) -> Dict[str, Any]:
    with stab_path.open("rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict):
        return {"path": str(stab_path), "type": type(data).__name__}
    keys = list(data.keys())
    first_value = data[keys[0]] if keys else None
    return {
        "path": str(stab_path),
        "type": "dict",
        "frame_transform_count": len(data),
        "sample_keys": [str(k) for k in keys[:5]],
        "value_type": type(first_value).__name__ if first_value is not None else None,
        "value_shape": list(first_value.shape) if isinstance(first_value, np.ndarray) else None,
        "meaning": "per-frame 2x3 affine stabilization transform matrix",
    }


def _find_detection_pkl(dataset_dir: Path) -> Optional[Path]:
    patterns = ["det_bbox_result_*.pkl", "stitch_bbox_result_*.pkl", "*.detpkl", "*.pkl"]
    for pattern in patterns:
        matches = sorted(dataset_dir.glob(pattern))
        matches = [p for p in matches if "_stab" not in p.stem.lower() and "stab" not in p.stem.lower()]
        if matches:
            return matches[0]
    return None


def _find_stabilization_pkl(dataset_dir: Path) -> Optional[Path]:
    matches = sorted(dataset_dir.glob("*_stab.pkl")) + sorted(dataset_dir.glob("*stab*.pkl"))
    return matches[0] if matches else None


def _find_background(dataset_dir: Path) -> Tuple[Optional[Path], List[str]]:
    warnings: List[str] = []
    image_patterns = [
        "background_*.jpg",
        "background_*.jpeg",
        "background_*.png",
        "background*.jpg",
        "first_frame_*.jpg",
        "first_frame_*.jpeg",
        "first_frame_*.png",
    ]
    for pattern in image_patterns:
        matches = sorted(dataset_dir.glob(pattern))
        if matches:
            if not matches[0].name.lower().startswith("background"):
                warnings.append("background_*.jpg not found; using first_frame image as background.")
            return matches[0], warnings
    warnings.append("No background_*.jpg or first_frame_*.jpg found; frontend will use a blank canvas.")
    return None, warnings


def _image_size(image_path: Optional[Path]) -> Tuple[Optional[int], Optional[int]]:
    if image_path is None:
        return None, None
    with Image.open(image_path) as image:
        width, height = image.size
    return int(width), int(height)


def _video_info(data: Dict[str, Any]) -> Dict[str, Any]:
    info = {}
    video_info = data.get("video_info")
    if isinstance(video_info, list) and video_info:
        first = video_info[0]
        if isinstance(first, dict):
            info = first
    elif isinstance(video_info, dict):
        info = video_info
    return info


def _timestamp(frame_id: int, frame_time: Any, fps: Optional[float]) -> Optional[float]:
    value = _safe_float(frame_time)
    if value is not None:
        return value
    if fps and fps > 0:
        return frame_id / fps
    return None


def _angle_deg(points: np.ndarray) -> Optional[float]:
    if points.shape != (4, 2):
        return None
    dx = points[1, 0] - points[0, 0]
    dy = points[1, 1] - points[0, 1]
    if dx == 0 and dy == 0:
        return None
    return float(math.degrees(math.atan2(dy, dx)))


def _make_track_record(
    dataset_id: str,
    frame_id: int,
    output_frame: int,
    timestamp: Optional[float],
    row: np.ndarray,
    row_index: int,
) -> Dict[str, Any]:
    row = np.asarray(row, dtype=float)
    if row.shape[0] < 11:
        raise ConversionError(
            f"Frame {frame_id} row {row_index} has {row.shape[0]} columns; object_id requires at least 11."
        )
    points = row[:8].reshape(4, 2)
    xs = points[:, 0]
    ys = points[:, 1]
    x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    category_id = int(row[9]) if not math.isnan(float(row[9])) else None
    object_id = int(row[10]) if not math.isnan(float(row[10])) else None
    if object_id is None:
        raise ConversionError(f"Frame {frame_id} row {row_index} is missing object_id.")

    record: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "frame_id": frame_id,
        "object_id": object_id,
        "class_name": category_name(category_id),
        "confidence": _safe_float(row[8]),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "cx": float(points[:, 0].mean()),
        "cy": float(points[:, 1].mean()),
        "width": x2 - x1,
        "height": y2 - y1,
        "category_id": category_id,
        "output_frame": output_frame,
        "timestamp": timestamp,
        "angle_deg": _angle_deg(points),
        "q1_x": float(points[0, 0]),
        "q1_y": float(points[0, 1]),
        "q2_x": float(points[1, 0]),
        "q2_y": float(points[1, 1]),
        "q3_x": float(points[2, 0]),
        "q3_y": float(points[2, 1]),
        "q4_x": float(points[3, 0]),
        "q4_y": float(points[3, 1]),
        "source_row_index": row_index,
    }

    if row.shape[0] >= 19:
        world = row[11:19].reshape(4, 2)
        for idx in range(4):
            record[f"world_q{idx + 1}_x"] = float(world[idx, 0])
            record[f"world_q{idx + 1}_y"] = float(world[idx, 1])
    if row.shape[0] >= 20:
        lane_value = _safe_float(row[19])
        record["lane_id"] = int(lane_value) if lane_value is not None else ""
    return record


def _write_csv(path: Path, fieldnames: List[str], records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({field: _csv_value(record.get(field, "")) for field in fieldnames})


def _compute_object_metrics(records: List[Dict[str, Any]], static_gate: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["object_id"])].append(record)

    metrics: Dict[int, Dict[str, Any]] = {}
    for object_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: int(item["frame_id"]))
        frames = sorted({int(row["frame_id"]) for row in rows})
        class_counts = Counter(row.get("class_name", "") for row in rows)
        class_name = class_counts.most_common(1)[0][0] if class_counts else ""
        confs = [float(row["confidence"]) for row in rows if row.get("confidence") not in (None, "")]

        start = rows[0]
        end = rows[-1]
        start_cx, start_cy = float(start["cx"]), float(start["cy"])
        end_cx, end_cy = float(end["cx"]), float(end["cy"])
        displacement = math.hypot(end_cx - start_cx, end_cy - start_cy)

        path_length = 0.0
        speeds: List[float] = []
        static_steps = 0
        for prev, cur in zip(rows[:-1], rows[1:]):
            frame_gap = max(1, int(cur["frame_id"]) - int(prev["frame_id"]))
            dist = math.hypot(float(cur["cx"]) - float(prev["cx"]), float(cur["cy"]) - float(prev["cy"]))
            speed = dist / frame_gap
            path_length += dist
            speeds.append(speed)
            if speed < float(static_gate["per_frame_motion_threshold"]):
                static_steps += 1

        mean_speed = float(sum(speeds) / len(speeds)) if speeds else 0.0
        max_speed = float(max(speeds)) if speeds else 0.0
        static_ratio = float(static_steps / len(speeds)) if speeds else 1.0

        filter_reason = ""
        is_static = False
        if class_name in STATIC_GATE_VEHICLE_CLASSES and len(frames) >= int(static_gate["min_track_length"]):
            low_total_motion = (
                displacement < float(static_gate["max_displacement"])
                and mean_speed < float(static_gate["max_mean_speed"])
            )
            mostly_static = static_ratio > float(static_gate["static_ratio_threshold"])
            if low_total_motion or mostly_static:
                is_static = True
                reasons = []
                if low_total_motion:
                    reasons.append(
                        "displacement<%.2f and mean_speed<%.2f"
                        % (float(static_gate["max_displacement"]), float(static_gate["max_mean_speed"]))
                    )
                if mostly_static:
                    reasons.append("static_ratio>%.2f" % float(static_gate["static_ratio_threshold"]))
                filter_reason = "; ".join(reasons)

        metrics[object_id] = {
            "object_id": object_id,
            "class_name": class_name,
            "start_frame": frames[0],
            "end_frame": frames[-1],
            "total_frames": len(frames),
            "mean_confidence": float(sum(confs) / len(confs)) if confs else None,
            "start_cx": start_cx,
            "start_cy": start_cy,
            "end_cx": end_cx,
            "end_cy": end_cy,
            "displacement": displacement,
            "path_length": path_length,
            "mean_speed": mean_speed,
            "max_speed": max_speed,
            "static_ratio": static_ratio,
            "is_static": is_static,
            "filter_reason": filter_reason,
        }
    return metrics


def _object_records(dataset_id: str, records: List[Dict[str, Any]], metrics: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    object_ids = sorted({int(record["object_id"]) for record in records})
    objects = []
    for object_id in object_ids:
        metric = metrics[object_id]
        row = {"dataset_id": dataset_id}
        row.update(metric)
        objects.append(row)
    return objects


def _frame_records(
    dataset_id: str,
    frame_meta: Dict[int, Dict[str, Any]],
    records: List[Dict[str, Any]],
    width: Optional[int],
    height: Optional[int],
) -> List[Dict[str, Any]]:
    counts = Counter(int(row["frame_id"]) for row in records)
    frames = []
    for frame_id in sorted(frame_meta.keys()):
        meta = frame_meta[frame_id]
        frames.append(
            {
                "dataset_id": dataset_id,
                "frame_id": frame_id,
                "timestamp": meta.get("timestamp"),
                "width": width,
                "height": height,
                "num_objects": counts.get(frame_id, 0),
                "output_frame": meta.get("output_frame"),
            }
        )
    return frames


def _version_metadata(
    dataset_id: str,
    version: str,
    records: List[Dict[str, Any]],
    frame_meta: Dict[int, Dict[str, Any]],
    image_width: Optional[int],
    image_height: Optional[int],
    background_image_name: Optional[str],
    source_folder: str,
    fps: float,
    det_pkl: Path,
    stab_pkl: Optional[Path],
    column_counts: Counter[int],
    warnings: List[str],
    pkl_structure: Dict[str, Any],
    object_metrics: Dict[int, Dict[str, Any]],
    static_gate: Dict[str, Any],
    filtered_object_ids: List[int],
) -> Dict[str, Any]:
    object_ids = {int(record["object_id"]) for record in records}
    return {
        "dataset_id": dataset_id,
        "version": version,
        "display_name": f"{dataset_id} / {version}",
        "fps": fps,
        "total_frames": len(frame_meta),
        "image_width": image_width,
        "image_height": image_height,
        "background_image": background_image_name,
        "source_folder": source_folder,
        "converted_time": _dt.datetime.now().isoformat(timespec="seconds"),
        "detection_pkl": det_pkl.name,
        "stabilization_pkl": stab_pkl.name if stab_pkl else None,
        "row_count": len(records),
        "object_count": len(object_ids),
        "full_object_count": len(object_metrics),
        "filtered_object_count": len(filtered_object_ids),
        "filtered_object_ids": filtered_object_ids,
        "class_names": sorted({record["class_name"] for record in records if record.get("class_name")}),
        "traj_info_array_columns": {str(k): v for k, v in sorted(column_counts.items())},
        "coordinate_system": (
            "Pixel columns are copied from det_bbox_result traj_info and scaled to the displayed canvas. "
            "q1..q4 preserve the oriented bounding box; x1..y2 are the horizontal envelope."
        ),
        "column_mapping": {
            "0:8": "q1_x,q1_y,q2_x,q2_y,q3_x,q3_y,q4_x,q4_y pixel oriented bbox",
            "8": "confidence",
            "9": "category_id mapped to class_name",
            "10": "object_id / track_id",
            "11:19": "world_q1..world_q4 coordinates when present",
            "19": "lane_id when present",
        },
        "static_gate": static_gate,
        "pkl_structure": pkl_structure,
        "stabilization_structure": inspect_stabilization_pkl(stab_pkl) if stab_pkl else None,
        "warnings": warnings,
    }


def _write_dataset_version(
    version_dir: Path,
    dataset_id: str,
    version: str,
    records: List[Dict[str, Any]],
    frame_meta: Dict[int, Dict[str, Any]],
    image_width: Optional[int],
    image_height: Optional[int],
    background_src: Optional[Path],
    background_image_name: Optional[str],
    source_folder: str,
    fps: float,
    det_pkl: Path,
    stab_pkl: Optional[Path],
    column_counts: Counter[int],
    warnings: List[str],
    pkl_structure: Dict[str, Any],
    object_metrics: Dict[int, Dict[str, Any]],
    static_gate: Dict[str, Any],
    filtered_object_ids: List[int],
) -> Dict[str, Any]:
    version_dir.mkdir(parents=True, exist_ok=True)
    if background_src and background_image_name:
        shutil.copy2(background_src, version_dir / background_image_name)

    track_fields = CORE_TRACK_FIELDS + EXTRA_TRACK_FIELDS
    object_fields = [
        "dataset_id",
        "object_id",
        "class_name",
        "start_frame",
        "end_frame",
        "total_frames",
        "mean_confidence",
        "start_cx",
        "start_cy",
        "end_cx",
        "end_cy",
        "displacement",
        "path_length",
        "mean_speed",
        "max_speed",
        "static_ratio",
        "is_static",
        "filter_reason",
    ]

    _write_csv(version_dir / "tracks.csv", track_fields, records)
    _write_csv(version_dir / "objects.csv", object_fields, _object_records(dataset_id, records, object_metrics))
    _write_csv(
        version_dir / "frames.csv",
        ["dataset_id", "frame_id", "timestamp", "width", "height", "num_objects", "output_frame"],
        _frame_records(dataset_id, frame_meta, records, image_width, image_height),
    )

    metadata = _version_metadata(
        dataset_id,
        version,
        records,
        frame_meta,
        image_width,
        image_height,
        background_image_name,
        source_folder,
        fps,
        det_pkl,
        stab_pkl,
        column_counts,
        warnings,
        pkl_structure,
        object_metrics,
        static_gate,
        filtered_object_ids,
    )
    with (version_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    return metadata


def convert_dataset(
    dataset_dir: Path,
    output_root: Path = DEFAULT_ADJUSTED_ROOT,
    initial_root: Path = DEFAULT_INITIAL_ROOT,
    force: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    logger = logger or configure_logger()
    dataset_dir = Path(dataset_dir).resolve()
    dataset_id = dataset_dir.name
    output_dir = output_root / dataset_id
    full_dir = output_dir / "full"
    moving_dir = output_dir / "moving_filtered"

    if (
        (full_dir / "metadata.json").exists()
        and (full_dir / "tracks.csv").exists()
        and (moving_dir / "metadata.json").exists()
        and (moving_dir / "tracks.csv").exists()
        and not force
    ):
        logger.info("Skip %s because full and moving_filtered versions already exist.", dataset_id)
        return {"dataset_id": dataset_id, "status": "skipped", "warnings": []}

    warnings: List[str] = []
    det_pkl = _find_detection_pkl(dataset_dir)
    if det_pkl is None:
        raise ConversionError(f"No det_bbox_result_*.pkl or equivalent detection pkl found in {dataset_dir}.")

    stab_pkl = _find_stabilization_pkl(dataset_dir)
    background_src, bg_warnings = _find_background(dataset_dir)
    warnings.extend(bg_warnings)

    logger.info("Converting dataset %s from %s", dataset_id, dataset_dir)
    logger.info("Using detection pkl: %s", det_pkl.name)
    if stab_pkl:
        logger.info("Found stabilization pkl: %s", stab_pkl.name)
    for warning in warnings:
        logger.warning("%s: %s", dataset_id, warning)

    with det_pkl.open("rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict):
        raise ConversionError(f"{det_pkl.name} must contain a dict, got {type(data).__name__}.")
    traj_info = data.get("traj_info")
    if not isinstance(traj_info, list):
        raise ConversionError(f"{det_pkl.name} missing list field 'traj_info'.")
    if not traj_info:
        raise ConversionError(f"{det_pkl.name} has empty 'traj_info'.")

    video_info = _video_info(data)
    output_info = data.get("output_info", {}) if isinstance(data.get("output_info"), dict) else {}
    fps = _safe_float(output_info.get("output_fps")) or _safe_float(video_info.get("fps")) or 10.0

    image_width, image_height = _image_size(background_src)
    image_width = image_width or int(video_info.get("width") or 0) or None
    image_height = image_height or int(video_info.get("height") or 0) or None

    records: List[Dict[str, Any]] = []
    frame_meta: Dict[int, Dict[str, Any]] = {}
    column_counts: Counter[int] = Counter()
    row_errors: List[str] = []
    for entry in traj_info:
        frame_id, output_frame, arr, frame_time = _format_entry(entry)
        ts = _timestamp(frame_id, frame_time, fps)
        frame_meta.setdefault(frame_id, {"output_frame": output_frame, "timestamp": ts})
        if arr is None:
            warnings.append(f"Frame {frame_id} has None detection array.")
            continue
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        if arr.ndim != 2:
            raise ConversionError(f"Frame {frame_id} detection array must be 2D, got shape {arr.shape}.")
        column_counts[arr.shape[1]] += 1
        infer_detection_columns(arr.shape[1])
        for row_index, row in enumerate(arr):
            try:
                records.append(_make_track_record(dataset_id, frame_id, output_frame, ts, row, row_index))
            except ConversionError as exc:
                row_errors.append(str(exc))
                if len(row_errors) > 20:
                    raise ConversionError("Too many row conversion errors; first errors: " + "; ".join(row_errors[:20]))

    if row_errors:
        warnings.extend(row_errors[:20])
        logger.warning("%s row warnings: %s", dataset_id, "; ".join(row_errors[:5]))
    if not records:
        raise ConversionError(f"No valid track records found in {det_pkl.name}.")

    if force:
        for version_dir in (full_dir, moving_dir):
            if version_dir.exists():
                shutil.rmtree(version_dir)
        for legacy_name in ("tracks.csv", "objects.csv", "frames.csv", "metadata.json", "background.jpg"):
            legacy_path = output_dir / legacy_name
            if legacy_path.exists() and legacy_path.is_file():
                legacy_path.unlink()

    background_image_name = None
    if background_src:
        background_image_name = "background.jpg"

    try:
        source_rel = dataset_dir.relative_to(initial_root.resolve())
        if initial_root.resolve() == DEFAULT_INITIAL_ROOT.resolve():
            source_folder = str(Path("Visualization") / "Initial results" / source_rel)
        else:
            source_folder = str(dataset_dir)
    except ValueError:
        source_folder = str(dataset_dir)

    static_gate = dict(STATIC_GATE_CONFIG)
    object_metrics = _compute_object_metrics(records, static_gate)
    filtered_object_ids = sorted(
        object_id for object_id, metrics in object_metrics.items() if metrics.get("is_static")
    )
    filtered_set = set(filtered_object_ids)
    moving_records = [record for record in records if int(record["object_id"]) not in filtered_set]

    for object_id in filtered_object_ids:
        metrics = object_metrics[object_id]
        logger.info(
            "%s moving_filtered removes object_id=%s class=%s frames=%s displacement=%.3f "
            "path_length=%.3f mean_speed=%.3f max_speed=%.3f static_ratio=%.3f reason=%s",
            dataset_id,
            object_id,
            metrics["class_name"],
            metrics["total_frames"],
            metrics["displacement"],
            metrics["path_length"],
            metrics["mean_speed"],
            metrics["max_speed"],
            metrics["static_ratio"],
            metrics["filter_reason"],
        )

    pkl_structure = inspect_pkl_structure(det_pkl)
    full_metadata = _write_dataset_version(
        full_dir,
        dataset_id,
        "full",
        records,
        frame_meta,
        image_width,
        image_height,
        background_src,
        background_image_name,
        source_folder,
        fps,
        det_pkl,
        stab_pkl,
        column_counts,
        warnings,
        pkl_structure,
        object_metrics,
        static_gate,
        [],
    )
    moving_metadata = _write_dataset_version(
        moving_dir,
        dataset_id,
        "moving_filtered",
        moving_records,
        frame_meta,
        image_width,
        image_height,
        background_src,
        background_image_name,
        source_folder,
        fps,
        det_pkl,
        stab_pkl,
        column_counts,
        warnings,
        pkl_structure,
        object_metrics,
        static_gate,
        filtered_object_ids,
    )

    logger.info(
        "Converted %s full: %s rows, %s objects, %s frames",
        dataset_id,
        full_metadata["row_count"],
        full_metadata["object_count"],
        len(frame_meta),
    )
    logger.info(
        "Converted %s moving_filtered: %s rows, %s kept objects, %s filtered objects",
        dataset_id,
        moving_metadata["row_count"],
        moving_metadata["object_count"],
        moving_metadata["filtered_object_count"],
    )
    return {
        "dataset_id": dataset_id,
        "status": "converted",
        "warnings": warnings,
        "versions": {"full": full_metadata, "moving_filtered": moving_metadata},
    }


def find_dataset_dirs(source_root: Path) -> List[Path]:
    source_root = Path(source_root)
    if not source_root.exists():
        return []
    dirs = []
    for child in sorted(source_root.iterdir()):
        if child.is_dir() and any(child.glob("*.pkl")):
            dirs.append(child)
    return dirs


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
    return {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "force": force,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OpenVTER pkl result folders to visualization CSV datasets.")
    parser.add_argument("--source-root", default=str(DEFAULT_INITIAL_ROOT), help="Folder containing raw result subfolders.")
    parser.add_argument("--output-root", default=str(DEFAULT_ADJUSTED_ROOT), help="Folder for standardized CSV datasets.")
    parser.add_argument("--force", action="store_true", help="Overwrite converted datasets.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Only convert these dataset folder names.")
    parser.add_argument("--inspect", default=None, help="Only inspect a detection pkl and print its structure JSON.")
    args = parser.parse_args()

    logger = configure_logger()
    if args.inspect:
        print(json.dumps(inspect_pkl_structure(Path(args.inspect)), ensure_ascii=False, indent=2))
        return
    result = convert_all(Path(args.source_root), Path(args.output_root), args.force, args.datasets, logger)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
