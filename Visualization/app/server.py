#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dependency-light local backend for the OpenVTER trajectory visualizer."""
from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import pickle
from collections import Counter, defaultdict
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from converter import DEFAULT_ADJUSTED_ROOT, DEFAULT_INITIAL_ROOT, convert_all


APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"
VIS_ROOT = APP_ROOT.parent


def _safe_dataset_path(dataset_id: str, version: str) -> Path:
    if (
        not dataset_id
        or not version
        or "/" in dataset_id
        or "\\" in dataset_id
        or ".." in dataset_id
        or "/" in version
        or "\\" in version
        or ".." in version
    ):
        raise ValueError("Invalid dataset id.")
    path = (DEFAULT_ADJUSTED_ROOT / dataset_id / version).resolve()
    if DEFAULT_ADJUSTED_ROOT.resolve() not in path.parents and path != DEFAULT_ADJUSTED_ROOT.resolve():
        raise ValueError("Dataset path escapes Adjusted results.")
    return path


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _csv_records(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _standard_paths(dataset_dir: Path):
    dataset_id = dataset_dir.parent.name
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
    source_dir = DEFAULT_INITIAL_ROOT / dataset_id
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
        width = _float(row.get("width"), 0.0) or 0.0
        length = _float(row.get("length"), 0.0) or 0.0
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
        width = _float(row.get("width"), 0.0) or 0.0
        length = _float(row.get("length"), 0.0) or 0.0
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
    filtered_count = static_gate.get("filtered_track_count", 0) if version == "moving_filtered" else 0
    return {
        "dataset_id": dataset_id,
        "version": version,
        "display_name": f"{dataset_id} / {version}",
        "fps": _float(recording.get("frameRate"), 29.97),
        "total_frames": _int(recording.get("numFrames"), 0),
        "image_width": size["width"],
        "image_height": size["height"],
        "background_image": background.name if background else None,
        "row_count": "",
        "object_count": len(tracks_meta),
        "full_object_count": quality.get("staticGate", {}).get("original_track_count", len(tracks_meta)),
        "filtered_object_count": filtered_count,
        "class_names": classes,
        "warnings": quality.get("quality", {}).get("warnings", []),
        "coordinate_system": "standard_pixel_background" if background_size else "standard_world_meter_view",
    }


def _standard_tracks(dataset_dir: Path):
    _, tracks_meta, tracks, _ = _standard_source(dataset_dir)
    dataset_id = dataset_dir.parent.name
    class_by_track = {row.get("trackId"): row.get("class") or "unknown" for row in tracks_meta}
    bounds = _standard_bounds(tracks)
    world_to_pixel = _world_to_pixel_affine(dataset_id) if _background_path(dataset_id, dataset_dir) else None
    out = []
    for row in tracks:
        cx_src = _float(row.get("xCenter"))
        cy_src = _float(row.get("yCenter"))
        if cx_src is None or cy_src is None:
            continue
        width = _float(row.get("width"), 1.0) or 1.0
        length = _float(row.get("length"), width) or width
        if world_to_pixel is not None:
            center = _world_to_pixel_point(cx_src, cy_src, world_to_pixel)
            quad = _oriented_box_pixel(cx_src, cy_src, width, length, row.get("heading"), world_to_pixel)
        else:
            center = _view_point(cx_src, cy_src, bounds)
            quad = _oriented_box(cx_src, cy_src, width, length, row.get("heading"), bounds)
        xs = [p["x"] for p in quad]
        ys = [p["y"] for p in quad]
        item = {
            "dataset_id": dataset_id,
            "frame_id": row.get("frame"),
            "object_id": row.get("trackId"),
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
            "angle_deg": row.get("heading"),
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
        out.append(
            {
                "dataset_id": dataset_dir.parent.name,
                "object_id": track_id,
                "class_name": row.get("class"),
                "start_frame": row.get("initialFrame"),
                "end_frame": row.get("finalFrame"),
                "total_frames": row.get("numFrames"),
                "mean_confidence": "",
                "lane_id": row.get("startLaneId", -1),
                "startLaneId": row.get("startLaneId"),
                "endLaneId": row.get("endLaneId"),
                "displacement": metric.get("displacement", ""),
                "path_length": metric.get("path_length", ""),
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
    background_size = _image_dimensions(_background_path(dataset_dir.parent.name, dataset_dir))
    if background_size and _world_to_pixel_affine(dataset_dir.parent.name) is not None:
        size = background_size
    counts = Counter(row.get("frame") for row in tracks)
    fps = _float(recording.get("frameRate"), 29.97) or 29.97
    frames = sorted((_int(frame) for frame in counts if _int(frame) is not None))
    return [
        {
            "dataset_id": dataset_dir.parent.name,
            "frame_id": frame,
            "timestamp": frame / fps,
            "width": size["width"],
            "height": size["height"],
            "num_objects": counts[str(frame)] + counts[frame],
        }
        for frame in frames
    ]


def _source_dataset_dir(dataset_id: str) -> Path:
    return DEFAULT_INITIAL_ROOT / dataset_id


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
        print("%s - - %s" % (self.client_address[0], fmt % args))

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
            result = convert_all(DEFAULT_INITIAL_ROOT, DEFAULT_ADJUSTED_ROOT, force=force)
            self._send_json(result)
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _datasets(self):
        DEFAULT_INITIAL_ROOT.mkdir(parents=True, exist_ok=True)
        DEFAULT_ADJUSTED_ROOT.mkdir(parents=True, exist_ok=True)
        converted = []
        for dataset_dir in sorted(DEFAULT_ADJUSTED_ROOT.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for version_dir in sorted(dataset_dir.iterdir()):
                if not version_dir.is_dir():
                    continue
                if _is_standard_dataset(version_dir):
                    metadata = _standard_metadata(dataset_dir.name, version_dir.name, version_dir)
                    source_type = "standard"
                elif _is_legacy_dataset(version_dir):
                    metadata = _read_json(version_dir / "metadata.json")
                    source_type = "legacy"
                else:
                    continue
                converted.append(
                    {
                        "dataset_id": dataset_dir.name,
                        "version": version_dir.name,
                        "display_name": metadata.get("display_name", f"{dataset_dir.name} / {version_dir.name}"),
                        "row_count": metadata.get("row_count"),
                        "object_count": metadata.get("object_count"),
                        "full_object_count": metadata.get("full_object_count"),
                        "filtered_object_count": metadata.get("filtered_object_count"),
                        "total_frames": metadata.get("total_frames"),
                        "fps": metadata.get("fps"),
                        "class_names": metadata.get("class_names", []),
                        "converted_time": metadata.get("converted_time"),
                        "warning_count": len(metadata.get("warnings", [])),
                        "source_type": source_type,
                    }
                )

        initial = []
        for source_dir in sorted(DEFAULT_INITIAL_ROOT.iterdir()):
            if source_dir.is_dir():
                initial.append(
                    {
                        "dataset_id": source_dir.name,
                        "has_pkl": any(source_dir.glob("*.pkl")),
                        "converted": (
                            (
                                _is_standard_dataset(DEFAULT_ADJUSTED_ROOT / source_dir.name / "full")
                                or _is_legacy_dataset(DEFAULT_ADJUSTED_ROOT / source_dir.name / "full")
                            )
                            and (
                                _is_standard_dataset(DEFAULT_ADJUSTED_ROOT / source_dir.name / "moving_filtered")
                                or _is_legacy_dataset(DEFAULT_ADJUSTED_ROOT / source_dir.name / "moving_filtered")
                            )
                        ),
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
        if action == "metadata":
            if is_standard:
                self._send_json(_standard_metadata(dataset_id, version, dataset_dir))
            else:
                self._send_json(_read_json(dataset_dir / "metadata.json"))
            return
        if action == "tracks":
            self._send_json(_standard_tracks(dataset_dir) if is_standard else _csv_records(dataset_dir / "tracks.csv"))
            return
        if action == "objects":
            self._send_json(_standard_objects(dataset_dir) if is_standard else _csv_records(dataset_dir / "objects.csv"))
            return
        if action == "frames":
            self._send_json(_standard_frames(dataset_dir) if is_standard else _csv_records(dataset_dir / "frames.csv"))
            return
        if action == "lanes":
            self._send_json(_lane_geometry(dataset_id, version, dataset_dir))
            return
        if action == "background":
            if is_standard:
                image_path = _background_path(dataset_id, dataset_dir)
                if not image_path:
                    self._send_error(
                        HTTPStatus.NOT_FOUND,
                        "Dataset has no background_*.jpg or first_frame_*.jpg in Initial results.",
                    )
                    return
                self._serve_file(image_path)
                return
            metadata = _read_json(dataset_dir / "metadata.json")
            image_name = metadata.get("background_image")
            if not image_name:
                self._send_error(HTTPStatus.NOT_FOUND, "Dataset has no background image.")
                return
            self._serve_file(dataset_dir / image_name)
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Unknown dataset action.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local OpenVTER trajectory visualizer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), VisualizerHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"OpenVTER trajectory visualizer running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
