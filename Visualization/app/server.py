#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dependency-light local backend for the OpenVTER trajectory visualizer."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import mimetypes
import pickle
from collections import Counter, defaultdict
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

try:
    from .converter import DEFAULT_ADJUSTED_ROOT, DEFAULT_FINAL_ROOT, DEFAULT_INITIAL_ROOT, convert_all
except ImportError:
    from converter import DEFAULT_ADJUSTED_ROOT, DEFAULT_FINAL_ROOT, DEFAULT_INITIAL_ROOT, convert_all


APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"
VIS_ROOT = APP_ROOT.parent
DEFAULT_VISUALIZER_LOG_PATH = VIS_ROOT / "logs" / "visualization_server.log"
VISUALIZER_LOGGER = logging.getLogger("openvter_visualizer")
ACTIVE_INITIAL_ROOT = DEFAULT_INITIAL_ROOT
ACTIVE_ADJUSTED_ROOT = DEFAULT_ADJUSTED_ROOT
ACTIVE_FINAL_ROOT = DEFAULT_FINAL_ROOT


def _set_runtime_roots(initial_root: Path, adjusted_root: Path, final_root: Path) -> None:
    global ACTIVE_INITIAL_ROOT, ACTIVE_ADJUSTED_ROOT, ACTIVE_FINAL_ROOT
    ACTIVE_INITIAL_ROOT = Path(initial_root).expanduser().resolve()
    ACTIVE_ADJUSTED_ROOT = Path(adjusted_root).expanduser().resolve()
    ACTIVE_FINAL_ROOT = Path(final_root).expanduser().resolve()
    _affine_from_pkl.cache_clear()


def configure_visualizer_logger(log_path: Path = DEFAULT_VISUALIZER_LOG_PATH) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("openvter_visualizer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _safe_dataset_path(dataset_id: str, version: str) -> Path:
    if (
        not dataset_id
        or "/" in dataset_id
        or "\\" in dataset_id
        or ".." in dataset_id
    ):
        raise ValueError("Invalid dataset id.")
    final_root = ACTIVE_FINAL_ROOT.resolve()
    path = (final_root / dataset_id).resolve()
    if final_root not in path.parents and path != final_root:
        raise ValueError("Dataset path escapes Final Data.")
    return path


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _csv_records(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _standard_paths(dataset_dir: Path):
    dataset_id = dataset_dir.name
    return {
        "recording": dataset_dir / f"{dataset_id}_recordingMeta.csv",
        "tracks_meta": dataset_dir / f"{dataset_id}_tracksMeta.csv",
        "tracks": dataset_dir / f"{dataset_id}_tracks.csv",
        "quality": dataset_dir / "quality_report.json",
    }


def _background_path(dataset_id: str, dataset_dir: Path = None):
    candidates = []
    if dataset_dir is not None:
        candidates.extend(sorted(dataset_dir.glob("background*.jpg")))
        candidates.extend(sorted(dataset_dir.glob("background*.jpeg")))
        candidates.extend(sorted(dataset_dir.glob("background*.png")))
        candidates.extend(sorted(dataset_dir.glob("first_frame*.jpg")))
        candidates.extend(sorted(dataset_dir.glob("first_frame*.jpeg")))
        candidates.extend(sorted(dataset_dir.glob("first_frame*.png")))
    source_dir = ACTIVE_INITIAL_ROOT / dataset_id
    if source_dir.exists():
        candidates.extend(sorted(source_dir.glob("background*.jpg")))
        candidates.extend(sorted(source_dir.glob("background*.jpeg")))
        candidates.extend(sorted(source_dir.glob("background*.png")))
        candidates.extend(sorted(source_dir.glob("first_frame*.jpg")))
        candidates.extend(sorted(source_dir.glob("first_frame*.jpeg")))
        candidates.extend(sorted(source_dir.glob("first_frame*.png")))
    return candidates[0] if candidates else None


@lru_cache(maxsize=64)
def _image_dimensions(path: Path):
    if not path:
        return None
    try:
        from PIL import Image

        with Image.open(path) as img:
            return {"width": img.width, "height": img.height}
    except Exception:
        return None


def _is_standard_dataset(dataset_dir: Path) -> bool:
    paths = _standard_paths(dataset_dir)
    return paths["recording"].exists() and paths["tracks_meta"].exists() and paths["tracks"].exists()


def _missing_standard_files(dataset_dir: Path):
    paths = _standard_paths(dataset_dir)
    return [path.name for key, path in paths.items() if key != "quality" and not path.exists()]


def _standard_dataset_summary_light(dataset_dir: Path):
    """Lightweight dataset-list item: only check folder/name and required files."""
    missing_files = _missing_standard_files(dataset_dir)
    is_available = not missing_files
    return {
        "dataset_id": dataset_dir.name,
        "version": "final",
        "display_name": dataset_dir.name,
        "row_count": "",
        "object_count": None,
        "full_object_count": None,
        "filtered_object_count": None,
        "total_frames": None,
        "fps": None,
        "class_names": [],
        "converted_time": None,
        "warning_count": len(missing_files),
        "source_type": "final_data" if is_available else "missing_final_data",
        "is_available": is_available,
        "missing_files": missing_files,
    }


SCENE_VEHICLE_CLASSES = ["car", "van", "truck", "bus", "freight_car", "motor", "tricycle", "bicycle", "awning-tricycle"]
SCENE_DIMENSION_CLASSES = ["car", "van", "truck", "bus", "freight_car", "motor", "tricycle", "bicycle", "awning-tricycle"]
SCENE_DEFAULT_LENGTH_BINS = [0.0, 2.0, 3.5, 5.0, 6.8, 8.0, 9.5, 12.0, math.inf]
SCENE_LENGTH_BINS_BY_CLASS = {
    "car": [0.0, 3.5, 3.8, 4.1, 4.4, 4.7, 5.0, 5.2, 5.4, 5.8, math.inf],
    "van": [0.0, 4.5, 5.0, 5.4, 5.8, 6.2, 6.8, 7.5, 8.5, math.inf],
    "truck": [0.0, 5.0, 6.0, 6.8, 7.5, 8.0, 8.5, 9.0, 9.5, 10.5, 12.0, math.inf],
    "bus": [0.0, 7.0, 8.0, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0, 14.0, math.inf],
    "freight_car": [0.0, 5.0, 6.0, 6.8, 7.5, 8.0, 8.5, 9.0, 9.5, 10.5, 12.0, math.inf],
    "motor": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, math.inf],
    "tricycle": [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, math.inf],
    "bicycle": [0.0, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, math.inf],
    "awning-tricycle": [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, math.inf],
}
SCENE_DEFAULT_WIDTH_BINS = [0.0, 0.5, 1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0, math.inf]
SCENE_WIDTH_BINS_BY_CLASS = {
    "car": [0.0, 1.4, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, math.inf],
    "van": [0.0, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0, math.inf],
    "truck": [0.0, 1.8, 2.0, 2.2, 2.5, 2.8, 3.2, 3.6, math.inf],
    "bus": [0.0, 2.0, 2.2, 2.4, 2.6, 2.8, 3.2, 3.6, math.inf],
    "freight_car": [0.0, 1.8, 2.0, 2.2, 2.5, 2.8, 3.2, 3.6, math.inf],
    "motor": [0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, math.inf],
    "tricycle": [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, math.inf],
    "bicycle": [0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, math.inf],
    "awning-tricycle": [0.0, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, math.inf],
}


def _scene_bin_label(lo: float, hi: float) -> str:
    if math.isinf(hi):
        return f">={lo:g}"
    return f"{lo:g}-{hi:g}"


def _scene_bins_for(class_name: str, metric: str):
    if metric == "width":
        return SCENE_WIDTH_BINS_BY_CLASS.get(class_name, SCENE_DEFAULT_WIDTH_BINS)
    return SCENE_LENGTH_BINS_BY_CLASS.get(class_name, SCENE_DEFAULT_LENGTH_BINS)


def _scene_empty_histogram(bins):
    return [
        {"label": _scene_bin_label(lo, hi), "min": lo, "max": None if math.isinf(hi) else hi, "count": 0}
        for lo, hi in zip(bins[:-1], bins[1:])
    ]


def _scene_add_value(histogram, bins, value: float) -> None:
    for item, lo, hi in zip(histogram, bins[:-1], bins[1:]):
        if lo <= value < hi:
            item["count"] += 1
            return


def _scene_percentile(values, pct: float):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - rank) + values[hi] * (rank - lo)


def _nice_step(raw_step: float) -> float:
    if raw_step <= 0:
        return 0.5
    base = 10 ** math.floor(math.log10(raw_step))
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = base * mult
        if raw_step <= step:
            return step
    return base * 10.0


def _scene_dynamic_bins(values, metric: str):
    finite = sorted(v for v in values if v is not None and math.isfinite(v) and v > 0)
    if not finite:
        return SCENE_DEFAULT_WIDTH_BINS if metric == "width" else SCENE_DEFAULT_LENGTH_BINS
    lower = 0.0
    upper = _scene_percentile(finite, 98) or finite[-1]
    upper = max(upper, finite[-1] if len(finite) <= 3 else upper)
    if metric == "width":
        upper = min(max(upper * 1.12, 1.0), 4.0)
        preferred_step = 0.2
    else:
        upper = min(max(upper * 1.12, 2.0), 18.0)
        preferred_step = 0.5
    step = max(preferred_step, _nice_step(upper / 8.0))
    upper = math.ceil(upper / step) * step
    bins = [lower]
    cur = lower + step
    while cur < upper - 1e-9:
        bins.append(round(cur, 4))
        cur += step
    bins.append(round(upper, 4))
    bins.append(math.inf)
    return bins


def _scene_length_stats(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}

    def q(pct):
        if len(values) == 1:
            return values[0]
        rank = (len(values) - 1) * pct / 100.0
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        if lo == hi:
            return values[lo]
        return values[lo] * (hi - rank) + values[hi] * (rank - lo)

    return {
        "count": len(values),
        "min": values[0],
        "median": q(50),
        "mean": sum(values) / len(values),
        "p95": q(95),
        "max": values[-1],
    }


def _scene_class_summary(class_name: str, rows):
    lengths = []
    widths = []
    for row in rows:
        length = _float(row.get("corrected_height"), _float(row.get("length")))
        width = _float(row.get("corrected_width"), _float(row.get("width")))
        if length is not None and length > 0:
            lengths.append(length)
        if width is not None and width > 0:
            widths.append(width)
    length_bins = _scene_bins_for(class_name, "length") if class_name in SCENE_LENGTH_BINS_BY_CLASS else _scene_dynamic_bins(lengths, "length")
    width_bins = _scene_bins_for(class_name, "width") if class_name in SCENE_WIDTH_BINS_BY_CLASS else _scene_dynamic_bins(widths, "width")
    length_histogram = _scene_empty_histogram(length_bins)
    width_histogram = _scene_empty_histogram(width_bins)
    for length in lengths:
        _scene_add_value(length_histogram, length_bins, length)
    for width in widths:
        _scene_add_value(width_histogram, width_bins, width)
    stats = _scene_length_stats(lengths)
    width_stats = _scene_length_stats(widths)
    length_peak = max(length_histogram, key=lambda item: item["count"], default={"label": "", "count": 0})
    width_peak = max(width_histogram, key=lambda item: item["count"], default={"label": "", "count": 0})
    return {
        "class_name": class_name,
        "count": len(rows),
        "length": stats,
        "width": width_stats,
        "histogram": length_histogram,
        "length_histogram": length_histogram,
        "width_histogram": width_histogram,
        "peak_label": length_peak["label"],
        "peak_count": length_peak["count"],
        "length_peak_label": length_peak["label"],
        "length_peak_count": length_peak["count"],
        "width_peak_label": width_peak["label"],
        "width_peak_count": width_peak["count"],
    }


def _scene_summary() -> dict:
    ACTIVE_INITIAL_ROOT.mkdir(parents=True, exist_ok=True)
    ACTIVE_FINAL_ROOT.mkdir(parents=True, exist_ok=True)

    initial_names = {path.name for path in ACTIVE_INITIAL_ROOT.iterdir() if path.is_dir()}
    final_names = {path.name for path in ACTIVE_FINAL_ROOT.iterdir() if path.is_dir()}
    names = sorted(initial_names | final_names)

    videos = []
    all_rows = []
    total_class_counts = Counter()
    issue_count = 0
    car_long_count = 0
    van_short_count = 0

    for name in names:
        final_dir = ACTIVE_FINAL_ROOT / name
        status = "converted" if _is_standard_dataset(final_dir) else "missing_final_data"
        missing_files = [] if status == "converted" else _missing_standard_files(final_dir)
        class_counts = Counter()
        vehicle_count = 0
        car_count = 0
        van_count = 0
        car_ge_5_4 = 0
        van_lt_5_4 = 0
        tracks_meta = []

        if status == "converted":
            try:
                _, tracks_meta, _ = _standard_header_source(final_dir)
            except Exception as exc:
                status = "read_failed"
                missing_files = [str(exc)]

        if status == "converted":
            for row in tracks_meta:
                class_name = row.get("class") or "unknown"
                class_counts[class_name] += 1
                total_class_counts[class_name] += 1
                all_rows.append(row)
                if class_name in SCENE_VEHICLE_CLASSES:
                    vehicle_count += 1
                length = _float(row.get("corrected_height"), _float(row.get("length")))
                if class_name == "car":
                    car_count += 1
                    if length is not None and length >= 5.4:
                        car_ge_5_4 += 1
                        car_long_count += 1
                elif class_name == "van":
                    van_count += 1
                    if length is not None and length < 5.4:
                        van_lt_5_4 += 1
                        van_short_count += 1
        else:
            issue_count += 1

        videos.append(
            {
                "dataset_id": name,
                "status": status,
                "missing_files": missing_files,
                "track_count": len(tracks_meta),
                "vehicle_count": vehicle_count,
                "class_counts": dict(sorted(class_counts.items())),
                "car_count": car_count,
                "van_count": van_count,
                "car_ge_5_4": car_ge_5_4,
                "car_ge_5_4_ratio": car_ge_5_4 / car_count if car_count else 0.0,
                "van_lt_5_4": van_lt_5_4,
                "van_lt_5_4_ratio": van_lt_5_4 / van_count if van_count else 0.0,
                "has_initial": name in initial_names,
                "has_final": name in final_names,
            }
        )

    by_class = defaultdict(list)
    for row in all_rows:
        by_class[row.get("class") or "unknown"].append(row)

    class_summaries = [
        _scene_class_summary(class_name, by_class[class_name])
        for class_name in SCENE_DIMENSION_CLASSES
        if by_class.get(class_name)
    ]

    vehicle_total = sum(row["vehicle_count"] for row in videos)
    car_total = total_class_counts.get("car", 0)
    van_total = total_class_counts.get("van", 0)
    return {
        "initial_root": str(ACTIVE_INITIAL_ROOT),
        "final_root": str(ACTIVE_FINAL_ROOT),
        "video_count": len(videos),
        "converted_count": sum(1 for row in videos if row["status"] == "converted"),
        "issue_count": issue_count,
        "track_count": len(all_rows),
        "vehicle_count": vehicle_total,
        "class_counts": dict(sorted(total_class_counts.items())),
        "car_count": car_total,
        "van_count": van_total,
        "car_ge_5_4": car_long_count,
        "car_ge_5_4_ratio": car_long_count / car_total if car_total else 0.0,
        "van_lt_5_4": van_short_count,
        "van_lt_5_4_ratio": van_short_count / van_total if van_total else 0.0,
        "length_bins": [_scene_bin_label(lo, hi) for lo, hi in zip(SCENE_DEFAULT_LENGTH_BINS[:-1], SCENE_DEFAULT_LENGTH_BINS[1:])],
        "class_summaries": class_summaries,
        "videos": videos,
    }


def _is_legacy_dataset(dataset_dir: Path) -> bool:
    return (dataset_dir / "metadata.json").exists()


def _float(value, default=None):
    try:
        if value in (None, ""):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _int(value, default=None):
    out = _float(value, default)
    if out is None:
        return default
    return int(round(out))


def _standard_source(dataset_dir: Path):
    paths = _standard_paths(dataset_dir)
    recording = _csv_records(paths["recording"])
    tracks_meta = _csv_records(paths["tracks_meta"])
    tracks = _csv_records(paths["tracks"])
    quality = _read_json(paths["quality"]) if paths["quality"].exists() else {}
    return recording[0] if recording else {}, tracks_meta, tracks, quality


def _standard_header_source(dataset_dir: Path):
    """Read the small standard CSV files used for dataset listing/metadata."""
    paths = _standard_paths(dataset_dir)
    recording = _csv_records(paths["recording"])
    tracks_meta = _csv_records(paths["tracks_meta"])
    quality = _read_json(paths["quality"]) if paths["quality"].exists() else {}
    return recording[0] if recording else {}, tracks_meta, quality


def _standard_bounds(tracks):
    xs, ys = [], []
    for row in tracks:
        x = _float(row.get("xCenter"))
        y = _float(row.get("yCenter"))
        width = _float(row.get("corrected_width"), _float(row.get("width"), 0.0) or 0.0) or 0.0
        length = _float(row.get("corrected_height"), _float(row.get("length"), 0.0) or 0.0) or 0.0
        pad = max(width, length, 1.0)
        if x is not None and y is not None:
            xs.extend([x - pad, x + pad])
            ys.extend([y - pad, y + pad])
    if not xs or not ys:
        return {"min_x": 0.0, "max_x": 100.0, "min_y": 0.0, "max_y": 100.0, "pad": 10.0}
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = max(span * 0.04, 2.0)
    return {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys), "pad": pad}


def _standard_bounds_from_meta(tracks_meta):
    xs, ys = [], []
    for row in tracks_meta:
        width = _float(row.get("corrected_width"), _float(row.get("width"), 0.0) or 0.0) or 0.0
        length = _float(row.get("corrected_height"), _float(row.get("length"), 0.0) or 0.0) or 0.0
        pad = max(width, length, 1.0)
        for x_key, y_key in (("startXCenter", "startYCenter"), ("endXCenter", "endYCenter")):
            x = _float(row.get(x_key))
            y = _float(row.get(y_key))
            if x is not None and y is not None:
                xs.extend([x - pad, x + pad])
                ys.extend([y - pad, y + pad])
    if not xs or not ys:
        return {"min_x": 0.0, "max_x": 100.0, "min_y": 0.0, "max_y": 100.0, "pad": 10.0}
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = max(span * 0.04, 2.0)
    return {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys), "pad": pad}


def _view_point(x, y, bounds):
    # Source local coordinates use +Y upward. Canvas/image coordinates use +Y downward,
    # so y is flipped while x remains right-positive.
    return {
        "x": x - bounds["min_x"] + bounds["pad"],
        "y": bounds["max_y"] - y + bounds["pad"],
    }


def _view_size(bounds):
    return {
        "width": bounds["max_x"] - bounds["min_x"] + bounds["pad"] * 2,
        "height": bounds["max_y"] - bounds["min_y"] + bounds["pad"] * 2,
    }


def _oriented_box(cx, cy, width, length, heading_deg, bounds):
    width = max(_float(width, 1.0) or 1.0, 0.1)
    length = max(_float(length, width) or width, width)
    heading = math.radians(_float(heading_deg, 0.0) or 0.0)
    ux = math.sin(heading)
    uy = math.cos(heading)
    lx = -math.cos(heading)
    ly = math.sin(heading)
    hl = length / 2.0
    hw = width / 2.0
    world_points = [
        (cx + ux * hl + lx * hw, cy + uy * hl + ly * hw),
        (cx + ux * hl - lx * hw, cy + uy * hl - ly * hw),
        (cx - ux * hl - lx * hw, cy - uy * hl - ly * hw),
        (cx - ux * hl + lx * hw, cy - uy * hl + ly * hw),
    ]
    return [_view_point(x, y, bounds) for x, y in world_points]


def _standard_metadata(dataset_id: str, version: str, dataset_dir: Path):
    recording, tracks_meta, quality = _standard_header_source(dataset_dir)
    bounds = _standard_bounds_from_meta(tracks_meta)
    size = _view_size(bounds)
    background = _background_path(dataset_id, dataset_dir)
    background_size = _image_dimensions(background)
    if background_size and _world_to_pixel_affine(dataset_id) is not None:
        size = background_size
    classes = sorted({row.get("class") or "unknown" for row in tracks_meta})
    static_gate = quality.get("staticGate", {})
    fragmentation = quality.get("fragmentationFilter", {})
    filtered_count = 0
    return {
        "dataset_id": dataset_id,
        "version": version,
        "display_name": dataset_id,
        "fps": _float(recording.get("frameRate"), 29.97),
        "total_frames": _int(recording.get("numFrames"), 0),
        "image_width": size["width"],
        "image_height": size["height"],
        "background_image": background.name if background else None,
        "row_count": "",
        "object_count": len(tracks_meta),
        "full_object_count": quality.get("staticGate", {}).get("original_track_count", len(tracks_meta)),
        "filtered_object_count": filtered_count,
        "fragmentation_filtered_count": fragmentation.get("filtered_track_count", 0),
        "static_filtered_count": static_gate.get("filtered_track_count", 0),
        "class_names": classes,
        "warnings": quality.get("quality", {}).get("warnings", []),
        "coordinate_system": "standard_pixel_background" if background_size else "standard_world_meter_view",
    }


def _standard_tracks(dataset_dir: Path):
    _, tracks_meta, tracks, _ = _standard_source(dataset_dir)
    dataset_id = dataset_dir.name
    class_by_track = {row.get("trackId"): row.get("class") or "unknown" for row in tracks_meta}
    raw_by_track = {row.get("trackId"): row.get("raw_object_id") or row.get("trackId") for row in tracks_meta}
    bounds = _standard_bounds(tracks)
    world_to_pixel = _world_to_pixel_affine(dataset_id) if _background_path(dataset_id, dataset_dir) else None
    out = []
    for row in tracks:
        cx_src = _float(row.get("xCenter"))
        cy_src = _float(row.get("yCenter"))
        if cx_src is None or cy_src is None:
            continue
        width = _float(row.get("corrected_width"), _float(row.get("width"), 1.0) or 1.0) or 1.0
        length = _float(row.get("corrected_height"), _float(row.get("length"), width) or width) or width
        if world_to_pixel is not None:
            center = _world_to_pixel_point(cx_src, cy_src, world_to_pixel)
            quad = _oriented_box_pixel(cx_src, cy_src, width, length, row.get("heading"), world_to_pixel)
            heading_screen_rad = _heading_screen_angle(cx_src, cy_src, row.get("heading"), transform=world_to_pixel)
        else:
            center = _view_point(cx_src, cy_src, bounds)
            quad = _oriented_box(cx_src, cy_src, width, length, row.get("heading"), bounds)
            heading_screen_rad = _heading_screen_angle(cx_src, cy_src, row.get("heading"), bounds=bounds)
        xs = [p["x"] for p in quad]
        ys = [p["y"] for p in quad]
        item = {
            "dataset_id": dataset_id,
            "frame_id": row.get("frame"),
            "object_id": row.get("trackId"),
            "raw_object_id": row.get("raw_object_id") or raw_by_track.get(row.get("trackId"), row.get("trackId")),
            "class_name": class_by_track.get(row.get("trackId"), "unknown"),
            "confidence": "",
            "x1": min(xs),
            "y1": min(ys),
            "x2": max(xs),
            "y2": max(ys),
            "cx": center["x"],
            "cy": center["y"],
            "width": width,
            "height": length,
            "raw_mean_width": row.get("raw_mean_width"),
            "raw_mean_height": row.get("raw_mean_height"),
            "corrected_width": row.get("corrected_width") or width,
            "corrected_height": row.get("corrected_height") or length,
            "box_orientation_source": row.get("box_orientation_source", ""),
            "is_interpolated": row.get("is_interpolated", ""),
            "missing_ratio": row.get("missing_ratio", ""),
            "angle_deg": row.get("heading"),
            "heading_screen_rad": heading_screen_rad,
            "lane_id": row.get("lane_id"),
            "source_xCenter": row.get("xCenter"),
            "source_yCenter": row.get("yCenter"),
        }
        for idx, point in enumerate(quad, start=1):
            item[f"q{idx}_x"] = point["x"]
            item[f"q{idx}_y"] = point["y"]
        out.append(item)
    return out


def _standard_objects(dataset_dir: Path):
    _, tracks_meta, quality = _standard_header_source(dataset_dir)
    metrics = {}
    for item in quality.get("staticGate", {}).get("all_track_metrics", []):
        metrics[str(item.get("trackId"))] = item
    out = []
    for row in tracks_meta:
        track_id = row.get("trackId")
        metric = metrics.get(str(track_id), {})
        width = _float(row.get("corrected_width"), _float(row.get("width"), None))
        length = _float(row.get("corrected_height"), _float(row.get("length"), None))
        out.append(
            {
                "dataset_id": dataset_dir.name,
                "object_id": track_id,
                "raw_object_id": row.get("raw_object_id") or track_id,
                "class_name": row.get("class"),
                "width": width,
                "length": length,
                "start_frame": row.get("initialFrame"),
                "end_frame": row.get("finalFrame"),
                "total_frames": row.get("numFrames"),
                "mean_confidence": "",
                "lane_id": row.get("startLaneId", -1),
                "startLaneId": row.get("startLaneId"),
                "endLaneId": row.get("endLaneId"),
                "displacement": metric.get("displacement", ""),
                "path_length": metric.get("path_length", ""),
                "stationary_extent": metric.get("stationary_extent", ""),
                "mean_speed": metric.get("mean_speed", ""),
                "max_speed": metric.get("max_speed", ""),
                "static_ratio": metric.get("static_ratio", ""),
                "is_static": metric.get("is_static", False),
                "filter_reason": metric.get("filter_reason", ""),
            }
        )
    return out


def _standard_frames(dataset_dir: Path):
    recording, _, tracks, _ = _standard_source(dataset_dir)
    bounds = _standard_bounds(tracks)
    size = _view_size(bounds)
    background_size = _image_dimensions(_background_path(dataset_dir.name, dataset_dir))
    if background_size and _world_to_pixel_affine(dataset_dir.name) is not None:
        size = background_size
    counts = Counter(row.get("frame") for row in tracks)
    fps = _float(recording.get("frameRate"), 29.97) or 29.97
    frames = sorted((_int(frame) for frame in counts if _int(frame) is not None))
    return [
        {
            "dataset_id": dataset_dir.name,
            "frame_id": frame,
            "timestamp": frame / fps,
            "width": size["width"],
            "height": size["height"],
            "num_objects": counts[str(frame)] + counts[frame],
        }
        for frame in frames
    ]


def _source_dataset_dir(dataset_id: str) -> Path:
    return ACTIVE_INITIAL_ROOT / dataset_id


def _looks_like_road_config(path: Path) -> bool:
    try:
        data = _read_json(path)
    except Exception:
        return False
    shapes = data.get("shapes") if isinstance(data, dict) else None
    if not isinstance(shapes, list):
        return False
    for shape in shapes:
        label = str(shape.get("label", "")).lower()
        if label == "road" or label.startswith(("lane_", "laneline_", "drivingline")):
            return True
    return False


def _find_road_config(dataset_id: str):
    source_dir = _source_dataset_dir(dataset_id)
    candidates = []
    if source_dir.exists():
        candidates.extend(source_dir.rglob("*.json"))
    config_root = VIS_ROOT.parent / "config"
    if config_root.exists():
        candidates.extend(p for p in config_root.rglob("*.json") if dataset_id.lower() in p.stem.lower())

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _looks_like_road_config(path):
            return path
    return None


def _det_pkl_path(dataset_id: str):
    source_dir = _source_dataset_dir(dataset_id)
    if not source_dir.exists():
        return None
    matches = sorted(source_dir.glob("det_bbox_result_*.pkl"))
    if matches:
        return matches[0]
    matches = [p for p in sorted(source_dir.glob("*.pkl")) if not p.name.endswith("_stab.pkl")]
    return matches[0] if matches else None


def _pixel_to_world_affine(dataset_id: str):
    return _affine_from_pkl(dataset_id, "pixel_to_world")


def _world_to_pixel_affine(dataset_id: str):
    return _affine_from_pkl(dataset_id, "world_to_pixel")


@lru_cache(maxsize=32)
def _affine_from_pkl(dataset_id: str, direction: str):
    pkl_path = _det_pkl_path(dataset_id)
    if not pkl_path:
        return None
    try:
        import numpy as np
    except Exception:
        return None
    try:
        with pkl_path.open("rb") as fh:
            data = pickle.load(fh)
    except Exception:
        return None

    sources = []
    targets = []
    for entry in data.get("traj_info", []):
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        array = entry[2]
        for row in array:
            pairs = (
                (0, 1, 11, 12),
                (2, 3, 13, 14),
                (4, 5, 15, 16),
                (6, 7, 17, 18),
            )
            for px_i, py_i, wx_i, wy_i in pairs:
                try:
                    px, py, wx, wy = float(row[px_i]), float(row[py_i]), float(row[wx_i]), float(row[wy_i])
                except Exception:
                    continue
                if all(math.isfinite(v) for v in (px, py, wx, wy)):
                    if direction == "world_to_pixel":
                        sources.append([wx, wy, 1.0])
                        targets.append([px, py])
                    else:
                        sources.append([px, py, 1.0])
                        targets.append([wx, wy])
            if len(sources) >= 5000:
                break
        if len(sources) >= 5000:
            break

    if len(sources) < 6:
        return None
    try:
        a = np.asarray(sources, dtype=float)
        b = np.asarray(targets, dtype=float)
        params, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
        return params
    except Exception:
        return None


def _transform_road_point(point, transform, bounds):
    px, py = float(point[0]), float(point[1])
    wx = px * transform[0][0] + py * transform[1][0] + transform[2][0]
    wy = px * transform[0][1] + py * transform[1][1] + transform[2][1]
    return _view_point(wx, wy, bounds)


def _world_to_pixel_point(x, y, transform):
    return {
        "x": x * transform[0][0] + y * transform[1][0] + transform[2][0],
        "y": x * transform[0][1] + y * transform[1][1] + transform[2][1],
    }


def _oriented_box_pixel(cx, cy, width, length, heading_deg, transform):
    width = max(_float(width, 1.0) or 1.0, 0.1)
    length = max(_float(length, width) or width, width)
    heading = math.radians(_float(heading_deg, 0.0) or 0.0)
    ux = math.sin(heading)
    uy = math.cos(heading)
    lx = -math.cos(heading)
    ly = math.sin(heading)
    hl = length / 2.0
    hw = width / 2.0
    world_points = [
        (cx + ux * hl + lx * hw, cy + uy * hl + ly * hw),
        (cx + ux * hl - lx * hw, cy + uy * hl - ly * hw),
        (cx - ux * hl - lx * hw, cy - uy * hl - ly * hw),
        (cx - ux * hl + lx * hw, cy - uy * hl + ly * hw),
    ]
    return [_world_to_pixel_point(x, y, transform) for x, y in world_points]


def _heading_screen_angle(cx, cy, heading_deg, bounds=None, transform=None):
    heading_value = _float(heading_deg)
    if heading_value is None:
        return None
    heading = math.radians(heading_value)
    wx = cx + math.sin(heading)
    wy = cy + math.cos(heading)
    if transform is not None:
        start = _world_to_pixel_point(cx, cy, transform)
        end = _world_to_pixel_point(wx, wy, transform)
    else:
        start = _view_point(cx, cy, bounds)
        end = _view_point(wx, wy, bounds)
    dx = end["x"] - start["x"]
    dy = end["y"] - start["y"]
    if math.hypot(dx, dy) < 1e-12:
        return None
    return math.atan2(dy, dx)


def _lane_geometry(dataset_id: str, version: str, dataset_dir: Path):
    config_path = _find_road_config(dataset_id)
    if not config_path:
        return {
            "available": False,
            "reason": "未找到 road_config json。请把对应地点的 road_config json 放到 Initial results/<dataset_id>/ 下。",
            "shapes": [],
        }
    background_size = _image_dimensions(_background_path(dataset_id, dataset_dir))
    use_raw_pixel = background_size is not None
    transform = None if use_raw_pixel else _pixel_to_world_affine(dataset_id)
    if transform is None and _is_standard_dataset(dataset_dir) and not use_raw_pixel:
        return {
            "available": False,
            "reason": "找到了 road_config，但无法从 pkl 中估计 pixel -> world/view 的转换关系，暂不绘制以避免错位。",
            "source_config": str(config_path),
            "shapes": [],
        }

    data = _read_json(config_path)
    recording, tracks_meta, _ = _standard_header_source(dataset_dir) if _is_standard_dataset(dataset_dir) else ({}, [], {})
    bounds = _standard_bounds_from_meta(tracks_meta) if tracks_meta else None
    shapes = []
    for shape in data.get("shapes", []):
        label = str(shape.get("label", ""))
        if label == "road":
            role = "road"
        elif label.startswith("lane_"):
            role = "lane"
        elif label.startswith("laneline_"):
            role = "laneline"
        elif label.startswith("drivingline"):
            role = "drivingline"
        else:
            continue
        raw_points = shape.get("points") or []
        points = []
        for point in raw_points:
            try:
                if use_raw_pixel:
                    points.append({"x": float(point[0]), "y": float(point[1])})
                elif transform is not None and bounds is not None:
                    points.append(_transform_road_point(point, transform, bounds))
                else:
                    points.append({"x": float(point[0]), "y": float(point[1])})
            except Exception:
                continue
        if len(points) < 2:
            continue
        item = {
            "label": label,
            "role": role,
            "shape_type": shape.get("shape_type", "polygon" if role in {"road", "lane"} else "line"),
            "points": points,
        }
        if label.startswith("lane_"):
            item["lane_id"] = label.split("_")[-1]
        shapes.append(item)

    size = background_size or (_view_size(bounds) if bounds else {"width": data.get("imageWidth"), "height": data.get("imageHeight")})
    return {
        "available": bool(shapes),
        "reason": "" if shapes else "road_config 中没有可绘制的 road/lane/laneline/drivingline 形状。",
        "source_config": str(config_path),
        "coordinate_system": "viewer",
        "image_width": size["width"],
        "image_height": size["height"],
        "shapes": shapes,
    }


class VisualizerHandler(BaseHTTPRequestHandler):
    server_version = "OpenVTERVisualizer/1.0"

    def log_message(self, fmt, *args):  # noqa: D401 - keep BaseHTTPRequestHandler signature.
        message = "%s - - %s" % (self.client_address[0], fmt % args)
        print(message)
        VISUALIZER_LOGGER.info(message)

    def _send_bytes(self, data: bytes, status=HTTPStatus.OK, content_type="application/octet-stream") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload, status=HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(data, status, "application/json; charset=utf-8")

    def _send_error(self, status, message: str) -> None:
        self._send_json({"error": message}, status)

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "File not found.")
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send_bytes(path.read_bytes(), HTTPStatus.OK, content_type)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler.
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/" or path == "/index.html":
                self._serve_file(STATIC_ROOT / "index.html")
                return
            if path == "/scene.html":
                self._serve_file(STATIC_ROOT / "scene.html")
                return
            if path.startswith("/static/"):
                rel = path[len("/static/") :]
                file_path = (STATIC_ROOT / rel).resolve()
                if STATIC_ROOT.resolve() not in file_path.parents:
                    self._send_error(HTTPStatus.BAD_REQUEST, "Invalid static path.")
                    return
                self._serve_file(file_path)
                return
            if path == "/api/datasets":
                self._send_json(self._datasets())
                return
            if path == "/api/scene-summary":
                self._send_json(_scene_summary())
                return
            if path.startswith("/api/datasets/"):
                self._dataset_endpoint(path)
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint.")
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler.
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path != "/api/scan":
            self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint.")
            return
        try:
            query = parse_qs(parsed.query)
            force = query.get("force", ["false"])[0].lower() in {"1", "true", "yes"}
            result = convert_all(
                ACTIVE_INITIAL_ROOT,
                ACTIVE_ADJUSTED_ROOT,
                force=force,
                final_root=ACTIVE_FINAL_ROOT,
            )
            self._send_json(result)
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _datasets(self):
        ACTIVE_INITIAL_ROOT.mkdir(parents=True, exist_ok=True)
        ACTIVE_FINAL_ROOT.mkdir(parents=True, exist_ok=True)
        converted = []
        for dataset_dir in sorted(ACTIVE_FINAL_ROOT.iterdir()):
            if not dataset_dir.is_dir():
                continue
            converted.append(_standard_dataset_summary_light(dataset_dir))

        initial = []
        for source_dir in sorted(ACTIVE_INITIAL_ROOT.iterdir()):
            if source_dir.is_dir():
                initial.append(
                    {
                        "dataset_id": source_dir.name,
                        "has_pkl": None,
                        "converted": _is_standard_dataset(ACTIVE_FINAL_ROOT / source_dir.name),
                    }
                )
        return {"converted": converted, "initial": initial}

    def _dataset_endpoint(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) < 4:
            self._send_error(HTTPStatus.BAD_REQUEST, "Dataset endpoint missing dataset id.")
            return
        dataset_id = parts[2]
        version = parts[3]
        action = parts[4] if len(parts) > 4 else "metadata"
        dataset_dir = _safe_dataset_path(dataset_id, version)
        if not dataset_dir.exists():
            self._send_error(HTTPStatus.NOT_FOUND, f"Dataset {dataset_id} not found.")
            return

        is_standard = _is_standard_dataset(dataset_dir)
        if not is_standard:
            missing_files = _missing_standard_files(dataset_dir)
            message = "Final Data is missing required CSV files: %s" % ", ".join(missing_files)
            self._send_error(HTTPStatus.NOT_FOUND, message)
            return
        if action == "metadata":
            self._send_json(_standard_metadata(dataset_id, version, dataset_dir))
            return
        if action == "tracks":
            self._send_json(_standard_tracks(dataset_dir))
            return
        if action == "objects":
            self._send_json(_standard_objects(dataset_dir))
            return
        if action == "frames":
            self._send_json(_standard_frames(dataset_dir))
            return
        if action == "lanes":
            self._send_json(_lane_geometry(dataset_id, version, dataset_dir))
            return
        if action == "background":
            image_path = _background_path(dataset_id, dataset_dir)
            if not image_path:
                self._send_error(
                    HTTPStatus.NOT_FOUND,
                    "Dataset has no background_*.jpg or first_frame_*.jpg in Initial results.",
                )
                return
            self._serve_file(image_path)
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Unknown dataset action.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local OpenVTER trajectory visualizer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--initial-root", default=str(DEFAULT_INITIAL_ROOT), help="Folder containing raw result subfolders.")
    parser.add_argument("--adjusted-root", default=str(DEFAULT_ADJUSTED_ROOT), help="Folder for converted intermediate CSV datasets.")
    parser.add_argument("--final-root", default=str(DEFAULT_FINAL_ROOT), help="Folder containing final visualization CSV datasets.")
    args = parser.parse_args()

    _set_runtime_roots(Path(args.initial_root), Path(args.adjusted_root), Path(args.final_root))
    logger = configure_visualizer_logger()
    ACTIVE_INITIAL_ROOT.mkdir(parents=True, exist_ok=True)
    ACTIVE_ADJUSTED_ROOT.mkdir(parents=True, exist_ok=True)
    ACTIVE_FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), VisualizerHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"OpenVTER trajectory visualizer running at {url}")
    print(f"Initial root: {ACTIVE_INITIAL_ROOT}")
    print(f"Adjusted root: {ACTIVE_ADJUSTED_ROOT}")
    print(f"Final root: {ACTIVE_FINAL_ROOT}")
    print("Press Ctrl+C to stop.")
    logger.info("OpenVTER trajectory visualizer running at %s", url)
    logger.info("Initial root: %s", ACTIVE_INITIAL_ROOT)
    logger.info("Adjusted root: %s", ACTIVE_ADJUSTED_ROOT)
    logger.info("Final root: %s", ACTIVE_FINAL_ROOT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        logger.info("Stopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
