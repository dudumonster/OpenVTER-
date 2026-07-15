#!/usr/bin/env python
"""Overlay processed OBB tracks from Visualization outputs onto source video.

The Visualization standard CSV stores OBBs in world-meter coordinates. This
tool reuses the affine relation embedded in det_bbox_result_*.pkl to convert
world points back to image pixels, then optionally maps stabilized/background
pixels back to the original frame using the inverse per-frame stabilization
matrix from *_stab.pkl.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Iterable

import cv2
import numpy as np


CLASS_COLORS = {
    "car": (255, 110, 35),
    "motor": (35, 135, 255),
    "pedestrian": (45, 55, 235),
    "people": (45, 45, 45),
    "truck": (40, 40, 190),
    "van": (190, 150, 30),
    "bus": (55, 180, 55),
    "bicycle": (70, 220, 220),
    "tricycle": (170, 80, 210),
    "awning-tricycle": (170, 80, 210),
    "unknown": (220, 220, 220),
}

DEFAULT_CATEGORY_NAMES = [
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

TRAIL_MIN_ALPHA = 0.12
TRAIL_MAX_ALPHA = 0.82
PERMANENT_TRAIL_ALPHA = 0.58


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay processed Visualization OBB tracks onto source video."
    )
    parser.add_argument("--dataset-id", default="xiang_shi_zhong_xue_019")
    parser.add_argument(
        "--video-source",
        choices=("original", "tracking"),
        default="original",
        help=(
            "original renders processed tracks on the source MP4; tracking adds "
            "raw-PKL trails to tracking_output_*.mp4 without redrawing boxes."
        ),
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Fallback directory containing <dataset-id>.MP4 for original mode.",
    )
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=Path("Visualization"),
    )
    parser.add_argument(
        "--version",
        choices=("moving_filtered", "full", "final"),
        default="moving_filtered",
        help="Processed result version to render.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "overlay_validation",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help=(
            "Persistent root for validation_frames/ and reports/. When omitted, "
            "the legacy --output-dir/<dataset>/<version> layout is used."
        ),
    )
    parser.add_argument(
        "--frames",
        default="0,1000,5000,9000",
        help="Comma-separated frame indices to export as JPG validation frames.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=0)
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Optional MP4 path for a short validation clip.",
    )
    parser.add_argument(
        "--draw-space",
        choices=("original", "stabilized"),
        default="original",
        help=(
            "original maps boxes back through inverse stabilization; stabilized "
            "draws in stabilized/background coordinates."
        ),
    )
    parser.add_argument("--line-width", type=int, default=2)
    label_group = parser.add_mutually_exclusive_group()
    label_group.add_argument("--show-labels", action="store_true", dest="show_labels")
    label_group.add_argument("--hide-labels", action="store_false", dest="show_labels")
    parser.set_defaults(show_labels=None)
    box_group = parser.add_mutually_exclusive_group()
    box_group.add_argument("--draw-boxes", action="store_true", dest="draw_boxes")
    box_group.add_argument("--no-draw-boxes", action="store_false", dest="draw_boxes")
    parser.set_defaults(draw_boxes=None)
    legend_group = parser.add_mutually_exclusive_group()
    legend_group.add_argument("--show-legend", action="store_true", dest="show_legend")
    legend_group.add_argument("--hide-legend", action="store_false", dest="show_legend")
    parser.set_defaults(show_legend=None)
    parser.add_argument(
        "--trail-mode",
        choices=("none", "finite", "permanent"),
        default="none",
    )
    parser.add_argument(
        "--trail-seconds",
        type=float,
        default=17.0,
        help="Lifetime of each trail segment in finite mode.",
    )
    parser.add_argument("--trail-width", type=int, default=4)
    parser.add_argument(
        "--max-link-gap-frames",
        type=int,
        default=30,
        help="Do not connect two observations farther apart than this many frames.",
    )
    parser.add_argument("--max-objects-per-frame", type=int, default=0)
    args = parser.parse_args()
    if args.trail_seconds <= 0:
        parser.error("--trail-seconds must be greater than zero")
    if args.trail_width <= 0:
        parser.error("--trail-width must be greater than zero")
    if args.max_link_gap_frames <= 0:
        parser.error("--max-link-gap-frames must be greater than zero")
    if args.draw_boxes is None:
        args.draw_boxes = args.video_source == "original"
    if args.show_labels is None:
        args.show_labels = args.trail_mode == "none" and args.video_source == "original"
    if args.show_legend is None:
        args.show_legend = args.trail_mode != "none"
    return args


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def to_int(value: Any, default: int | None = None) -> int | None:
    number = to_float(value)
    if number is None:
        return default
    return int(round(number))


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    dataset_id = args.dataset_id
    initial_dir = args.visualization_dir / "Initial results" / dataset_id
    if args.version == "final":
        tracks_dir = args.visualization_dir / "Final Data" / dataset_id
    else:
        tracks_dir = args.visualization_dir / "Adjusted results" / dataset_id / args.version

    paths = {
        "initial_dir": initial_dir,
        "tracks": tracks_dir / f"{dataset_id}_tracks.csv",
        "tracks_meta": tracks_dir / f"{dataset_id}_tracksMeta.csv",
        "recording_meta": tracks_dir / f"{dataset_id}_recordingMeta.csv",
    }
    det_matches = sorted(initial_dir.glob("det_bbox_result_*.pkl"))
    stab_matches = sorted(initial_dir.glob("*_stab.pkl"))
    if det_matches:
        paths["det_pkl"] = det_matches[0]
    if stab_matches:
        paths["stab_pkl"] = stab_matches[0]

    if args.video_path is not None:
        paths["video"] = args.video_path
    elif args.video_source == "tracking":
        preferred = initial_dir / f"tracking_output_stab_det_{dataset_id}.mp4"
        tracking_matches = sorted(initial_dir.glob("tracking_output_*.mp4"))
        if preferred.exists():
            paths["video"] = preferred
        elif tracking_matches:
            paths["video"] = tracking_matches[0]
    else:
        runtime_path = initial_dir / "runtime_config.json"
        if runtime_path.exists():
            try:
                runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                video_files = runtime.get("video_file")
                if isinstance(video_files, str):
                    video_files = [video_files]
                if isinstance(video_files, list) and video_files:
                    candidate = Path(str(video_files[0]))
                    if candidate.exists():
                        paths["video"] = candidate
            except (OSError, ValueError, TypeError):
                pass
        if "video" not in paths and args.video_dir is not None:
            paths["video"] = args.video_dir / f"{dataset_id}.MP4"
    return paths


def require_existing(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + "; ".join(missing))


def load_detection_data(det_pkl: Path) -> dict[str, Any]:
    with det_pkl.open("rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Detection result must contain a dict: {det_pkl}")
    return data


def estimate_world_to_pixel(
    det_pkl: Path,
    max_pairs: int = 5000,
    data: dict[str, Any] | None = None,
) -> np.ndarray:
    data = data if data is not None else load_detection_data(det_pkl)

    sources: list[list[float]] = []
    targets: list[list[float]] = []
    for entry in data.get("traj_info", []):
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        arr = np.asarray(entry[2])
        if arr.ndim != 2 or arr.shape[1] < 19:
            continue
        for row in arr:
            pairs = ((0, 1, 11, 12), (2, 3, 13, 14), (4, 5, 15, 16), (6, 7, 17, 18))
            for px_i, py_i, wx_i, wy_i in pairs:
                values = [row[px_i], row[py_i], row[wx_i], row[wy_i]]
                if np.isfinite(values).all():
                    sources.append([float(row[wx_i]), float(row[wy_i]), 1.0])
                    targets.append([float(row[px_i]), float(row[py_i])])
            if len(sources) >= max_pairs:
                break
        if len(sources) >= max_pairs:
            break

    if len(sources) < 6:
        raise ValueError(f"Not enough point pairs to estimate world_to_pixel from {det_pkl}")

    a = np.asarray(sources, dtype=float)
    b = np.asarray(targets, dtype=float)
    params, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    return params


def load_category_names(initial_dir: Path) -> list[str]:
    runtime_path = initial_dir / "runtime_config.json"
    if runtime_path.exists():
        try:
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            names = runtime.get("global_category")
            if isinstance(names, list) and names:
                return [str(name) for name in names]
        except (OSError, ValueError, TypeError):
            pass
    return list(DEFAULT_CATEGORY_NAMES)


def _track_id_from_value(value: Any) -> str | None:
    number = to_float(value)
    if number is None:
        return None
    return str(int(round(number)))


def build_raw_frame_tracks(
    data: dict[str, Any],
    category_names: list[str],
) -> tuple[
    dict[int, list[dict[str, Any]]],
    dict[int, int],
    dict[int, int],
    float,
]:
    """Build raw stabilized-pixel tracks and source/output frame mappings."""

    category_counts: dict[str, Counter[int]] = defaultdict(Counter)
    entries: list[tuple[int, int, np.ndarray]] = []
    output_to_source: dict[int, int] = {}
    source_to_output: dict[int, int] = {}

    for entry in data.get("traj_info", []):
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        source_frame = to_int(entry[0])
        output_frame = to_int(entry[1])
        arr = np.asarray(entry[2])
        if source_frame is None or output_frame is None or arr.ndim != 2:
            continue
        output_to_source[output_frame] = source_frame
        source_to_output[source_frame] = output_frame
        entries.append((source_frame, output_frame, arr))
        if arr.shape[1] < 11:
            continue
        for row in arr:
            track_id = _track_id_from_value(row[10])
            category_id = to_int(row[9])
            if track_id is not None and category_id is not None:
                category_counts[track_id][category_id] += 1

    category_by_track = {
        track_id: counts.most_common(1)[0][0]
        for track_id, counts in category_counts.items()
        if counts
    }
    frame_tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source_frame, _, arr in entries:
        if arr.shape[1] < 11:
            continue
        for row in arr:
            track_id = _track_id_from_value(row[10])
            if track_id is None:
                continue
            points = np.asarray(row[:8], dtype=float).reshape(4, 2)
            if not np.isfinite(points).all():
                continue
            category_id = category_by_track.get(track_id, to_int(row[9], -1))
            if category_id is not None and 0 <= category_id < len(category_names):
                class_name = category_names[category_id]
            else:
                class_name = "unknown"
            frame_tracks[source_frame].append(
                {
                    "track_id": track_id,
                    "raw_object_id": track_id,
                    "class_name": class_name,
                    "is_interpolated": False,
                    "points": points,
                    "center": points.mean(axis=0),
                }
            )

    output_info = data.get("output_info", {})
    if not isinstance(output_info, dict):
        output_info = {}
    output_fps = to_float(output_info.get("output_fps"), 29.97) or 29.97
    return frame_tracks, output_to_source, source_to_output, output_fps


def frame_mappings_from_data(
    data: dict[str, Any],
) -> tuple[dict[int, int], dict[int, int], float]:
    output_to_source: dict[int, int] = {}
    source_to_output: dict[int, int] = {}
    for entry in data.get("traj_info", []):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        source_frame = to_int(entry[0])
        output_frame = to_int(entry[1])
        if source_frame is None or output_frame is None:
            continue
        output_to_source[output_frame] = source_frame
        source_to_output[source_frame] = output_frame
    output_info = data.get("output_info", {})
    if not isinstance(output_info, dict):
        output_info = {}
    output_fps = to_float(output_info.get("output_fps"), 29.97) or 29.97
    return output_to_source, source_to_output, output_fps


def world_to_pixel_point(x: float, y: float, transform: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            x * transform[0, 0] + y * transform[1, 0] + transform[2, 0],
            x * transform[0, 1] + y * transform[1, 1] + transform[2, 1],
        ],
        dtype=float,
    )


def obb_world_to_pixel(row: dict[str, Any], transform: np.ndarray) -> np.ndarray | None:
    cx = to_float(row.get("xCenter"))
    cy = to_float(row.get("yCenter"))
    heading_deg = to_float(row.get("heading"), 0.0)
    width = to_float(row.get("corrected_width"), to_float(row.get("width"), 1.0))
    length = to_float(row.get("corrected_height"), to_float(row.get("length"), width or 1.0))
    if cx is None or cy is None or width is None or length is None or heading_deg is None:
        return None

    width = max(width, 0.1)
    length = max(length, width)
    heading = math.radians(heading_deg)
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
    return np.vstack([world_to_pixel_point(x, y, transform) for x, y in world_points])


def invert_affine_2x3(matrix: np.ndarray) -> np.ndarray:
    matrix_3x3 = np.vstack([matrix, [0.0, 0.0, 1.0]])
    inv = np.linalg.inv(matrix_3x3)
    return inv[:2, :]


def apply_affine(points: np.ndarray, matrix_2x3: np.ndarray) -> np.ndarray:
    hom = np.hstack([points, np.ones((points.shape[0], 1), dtype=float)])
    return hom @ matrix_2x3.T


def load_stabilization(stab_pkl: Path | None) -> dict[int, np.ndarray]:
    if stab_pkl is None or not stab_pkl.exists():
        return {}
    with stab_pkl.open("rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict):
        return {}
    out: dict[int, np.ndarray] = {}
    for key, value in data.items():
        try:
            frame_idx = int(key)
            matrix = np.asarray(value, dtype=float)
            if matrix.shape == (2, 3):
                out[frame_idx] = matrix
        except Exception:
            continue
    return out


def build_frame_tracks(
    tracks_path: Path,
    tracks_meta_path: Path,
    transform: np.ndarray,
) -> dict[int, list[dict[str, Any]]]:
    meta_rows = read_csv_rows(tracks_meta_path)
    class_by_track = {
        str(row.get("trackId")): row.get("class") or "unknown" for row in meta_rows
    }
    frame_tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with tracks_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            frame_idx = to_int(row.get("frame"))
            track_id = row.get("trackId")
            if frame_idx is None or track_id is None:
                continue
            points = obb_world_to_pixel(row, transform)
            if points is None:
                continue
            item = {
                "track_id": str(track_id),
                "raw_object_id": row.get("raw_object_id") or str(track_id),
                "class_name": class_by_track.get(str(track_id), "unknown"),
                "is_interpolated": str(row.get("is_interpolated", "")).lower()
                in {"true", "1", "yes"},
                "points": points,
                "center": points.mean(axis=0),
            }
            frame_tracks[frame_idx].append(item)
    return frame_tracks


def color_for_class(class_name: str, interpolated: bool) -> tuple[int, int, int]:
    color = CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"])
    if not interpolated:
        return color
    return tuple(int(0.55 * channel + 0.45 * 255) for channel in color)


@dataclass(frozen=True)
class TrailPoint:
    frame_idx: int
    center: tuple[float, float]
    class_name: str


class TrailRenderer:
    """Maintain finite or permanent trails in stabilized pixel coordinates."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        mode: str,
        seconds: float,
        line_width: int,
        max_link_gap_frames: int,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.mode = mode
        self.ttl_frames = max(1, int(round(float(seconds) * self.fps)))
        self.line_width = int(line_width)
        self.max_link_gap_frames = int(max_link_gap_frames)
        self.history: dict[str, Deque[TrailPoint]] = defaultdict(deque)
        self.last_point: dict[str, TrailPoint] = {}
        self.permanent_color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.permanent_alpha = np.zeros((self.height, self.width), dtype=np.uint8)

    @staticmethod
    def _point(item: dict[str, Any], frame_idx: int) -> TrailPoint | None:
        center = np.asarray(item.get("center"), dtype=float).reshape(-1)
        if center.size < 2 or not np.isfinite(center[:2]).all():
            return None
        return TrailPoint(
            frame_idx=int(frame_idx),
            center=(float(center[0]), float(center[1])),
            class_name=str(item.get("class_name") or "unknown"),
        )

    def update(self, frame_idx: int, tracks: list[dict[str, Any]]) -> None:
        if self.mode == "none":
            return

        unique_tracks: dict[str, dict[str, Any]] = {}
        for item in tracks:
            unique_tracks[str(item.get("track_id"))] = item

        for track_id, item in unique_tracks.items():
            point = self._point(item, frame_idx)
            if point is None:
                continue
            if self.mode == "permanent":
                previous = self.last_point.get(track_id)
                if previous is not None:
                    gap = point.frame_idx - previous.frame_idx
                    if 0 < gap <= self.max_link_gap_frames:
                        color = CLASS_COLORS.get(point.class_name, CLASS_COLORS["unknown"])
                        p1 = tuple(np.round(previous.center).astype(int))
                        p2 = tuple(np.round(point.center).astype(int))
                        cv2.line(
                            self.permanent_color,
                            p1,
                            p2,
                            color,
                            self.line_width,
                            cv2.LINE_AA,
                        )
                        cv2.line(
                            self.permanent_alpha,
                            p1,
                            p2,
                            int(round(PERMANENT_TRAIL_ALPHA * 255)),
                            self.line_width,
                            cv2.LINE_AA,
                        )
                self.last_point[track_id] = point
            else:
                history = self.history[track_id]
                if history and history[-1].frame_idx == point.frame_idx:
                    history[-1] = point
                else:
                    history.append(point)

        if self.mode == "finite":
            oldest = int(frame_idx) - self.ttl_frames
            empty_tracks = []
            for track_id, history in self.history.items():
                while history and history[0].frame_idx < oldest:
                    history.popleft()
                if not history:
                    empty_tracks.append(track_id)
            for track_id in empty_tracks:
                del self.history[track_id]

    def _finite_layer(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        color_layer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        alpha_layer = np.zeros((self.height, self.width), dtype=np.uint8)
        for history in self.history.values():
            points = list(history)
            for previous, current in zip(points, points[1:]):
                gap = current.frame_idx - previous.frame_idx
                if gap <= 0 or gap > self.max_link_gap_frames:
                    continue
                age = max(0, int(frame_idx) - current.frame_idx)
                life = max(0.0, 1.0 - age / self.ttl_frames)
                alpha = TRAIL_MIN_ALPHA + (TRAIL_MAX_ALPHA - TRAIL_MIN_ALPHA) * life
                color = CLASS_COLORS.get(current.class_name, CLASS_COLORS["unknown"])
                p1 = tuple(np.round(previous.center).astype(int))
                p2 = tuple(np.round(current.center).astype(int))
                cv2.line(color_layer, p1, p2, color, self.line_width, cv2.LINE_AA)
                cv2.line(
                    alpha_layer,
                    p1,
                    p2,
                    int(round(alpha * 255)),
                    self.line_width,
                    cv2.LINE_AA,
                )
        return color_layer, alpha_layer

    def layer(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if self.mode == "finite":
            return self._finite_layer(frame_idx)
        if self.mode == "permanent":
            return self.permanent_color, self.permanent_alpha
        return (
            np.zeros((self.height, self.width, 3), dtype=np.uint8),
            np.zeros((self.height, self.width), dtype=np.uint8),
        )

    def active_segment_count(self) -> int:
        if self.mode == "permanent":
            return int(np.count_nonzero(self.permanent_alpha))
        return sum(max(0, len(history) - 1) for history in self.history.values())


def blend_trail_layer(
    frame: np.ndarray,
    color_layer: np.ndarray,
    alpha_layer: np.ndarray,
) -> None:
    if alpha_layer.size == 0 or not np.any(alpha_layer):
        return
    alpha = alpha_layer.astype(np.float32)[:, :, None] / 255.0
    blended = frame.astype(np.float32) * (1.0 - alpha) + color_layer.astype(np.float32) * alpha
    frame[:] = np.clip(blended, 0, 255).astype(np.uint8)


def render_trails(
    frame: np.ndarray,
    renderer: TrailRenderer,
    frame_idx: int,
    stab_transforms: dict[int, np.ndarray],
    draw_space: str,
) -> None:
    color_layer, alpha_layer = renderer.layer(frame_idx)
    if draw_space == "original" and frame_idx in stab_transforms:
        inverse = invert_affine_2x3(stab_transforms[frame_idx])
        size = (frame.shape[1], frame.shape[0])
        color_layer = cv2.warpAffine(
            color_layer,
            inverse,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        alpha_layer = cv2.warpAffine(
            alpha_layer,
            inverse,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
    blend_trail_layer(frame, color_layer, alpha_layer)


def classes_in_tracks(frame_tracks: dict[int, list[dict[str, Any]]]) -> list[str]:
    classes = {
        str(item.get("class_name") or "unknown")
        for items in frame_tracks.values()
        for item in items
    }
    ordered = [name for name in DEFAULT_CATEGORY_NAMES if name in classes]
    return ordered + sorted(classes - set(ordered))


def draw_class_legend(frame: np.ndarray, class_names: list[str]) -> None:
    if not class_names:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(frame.shape[1], frame.shape[0]) / 2400.0)
    thickness = max(1, int(round(font_scale * 2)))
    row_height = max(26, int(round(34 * font_scale)))
    swatch = max(12, int(round(18 * font_scale)))
    margin = max(14, int(round(20 * font_scale)))
    text_width = max(cv2.getTextSize(name, font, font_scale, thickness)[0][0] for name in class_names)
    panel_width = margin * 3 + swatch + text_width
    panel_height = margin * 2 + row_height * len(class_names)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    for index, class_name in enumerate(class_names):
        y = 8 + margin + index * row_height
        color = CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"])
        cv2.rectangle(frame, (8 + margin, y), (8 + margin + swatch, y + swatch), color, -1)
        cv2.putText(
            frame,
            class_name,
            (8 + margin * 2 + swatch, y + swatch),
            font,
            font_scale,
            (245, 245, 245),
            thickness,
            cv2.LINE_AA,
        )


def draw_tracks(
    frame: np.ndarray,
    tracks: list[dict[str, Any]],
    frame_idx: int,
    stab_transforms: dict[int, np.ndarray],
    draw_space: str,
    line_width: int,
    show_labels: bool,
    max_objects: int,
) -> int:
    drawn = 0
    inverse = None
    if draw_space == "original" and frame_idx in stab_transforms:
        inverse = invert_affine_2x3(stab_transforms[frame_idx])

    for item in tracks:
        if max_objects and drawn >= max_objects:
            break
        points = np.asarray(item["points"], dtype=float)
        if inverse is not None:
            points = apply_affine(points, inverse)
        if not np.isfinite(points).all():
            continue
        int_points = np.round(points).astype(np.int32)
        color = color_for_class(item["class_name"], item["is_interpolated"])
        cv2.polylines(frame, [int_points], isClosed=True, color=color, thickness=line_width)
        drawn += 1

        if show_labels:
            label = f'{item["track_id"]} {item["class_name"]}'
            if item["is_interpolated"]:
                label += " I"
            anchor = int_points[np.argmin(int_points[:, 1])]
            x = int(anchor[0])
            y = max(18, int(anchor[1]) - 5)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x, y - th - 5), (x + tw + 6, y + 3), (255, 255, 255), -1)
            cv2.putText(
                frame,
                label,
                (x + 3, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )
    return drawn


def parse_frame_list(value: str) -> list[int]:
    frames: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        frames.append(int(part))
    return sorted(set(frames))


def seek_and_read(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def source_frame_for_base(
    base_frame: int,
    video_source: str,
    output_to_source: dict[int, int],
) -> int | None:
    if video_source == "original":
        return int(base_frame)
    return output_to_source.get(int(base_frame))


def make_trail_renderer(
    width: int,
    height: int,
    fps: float,
    args: argparse.Namespace,
) -> TrailRenderer:
    return TrailRenderer(
        width=width,
        height=height,
        fps=fps,
        mode=args.trail_mode,
        seconds=args.trail_seconds,
        line_width=args.trail_width,
        max_link_gap_frames=args.max_link_gap_frames,
    )


def warm_trail_renderer(
    renderer: TrailRenderer,
    start_base_frame: int,
    frame_tracks: dict[int, list[dict[str, Any]]],
    output_to_source: dict[int, int],
    video_source: str,
) -> None:
    if renderer.mode == "none" or start_base_frame <= 0:
        return
    if renderer.mode == "permanent":
        warm_start = 0
    else:
        warm_start = max(
            0,
            start_base_frame - renderer.ttl_frames - renderer.max_link_gap_frames,
        )
    for base_frame in range(warm_start, start_base_frame):
        source_frame = source_frame_for_base(base_frame, video_source, output_to_source)
        if source_frame is None:
            continue
        renderer.update(source_frame, frame_tracks.get(source_frame, []))


def annotate_frame(
    frame: np.ndarray,
    source_frame: int,
    tracks: list[dict[str, Any]],
    renderer: TrailRenderer,
    stab_transforms: dict[int, np.ndarray],
    class_names: list[str],
    args: argparse.Namespace,
) -> int:
    renderer.update(source_frame, tracks)
    if args.trail_mode != "none":
        render_trails(frame, renderer, source_frame, stab_transforms, args.draw_space)
    drawn = 0
    if args.draw_boxes:
        drawn = draw_tracks(
            frame,
            tracks,
            source_frame,
            stab_transforms,
            args.draw_space,
            args.line_width,
            args.show_labels,
            args.max_objects_per_frame,
        )
    if args.show_legend:
        draw_class_legend(frame, class_names)
    return drawn


def write_sample_frames(
    video_path: Path,
    output_dir: Path,
    frame_indices: list[int],
    frame_tracks: dict[int, list[dict[str, Any]]],
    output_to_source: dict[int, int],
    stab_transforms: dict[int, np.ndarray],
    class_names: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    reports: list[dict[str, Any]] = []
    for base_frame in frame_indices:
        frame = seek_and_read(cap, base_frame)
        if frame is None:
            reports.append({"frame": base_frame, "ok": False, "reason": "read_failed"})
            continue
        source_frame = source_frame_for_base(base_frame, args.video_source, output_to_source)
        if source_frame is None:
            reports.append({"frame": base_frame, "ok": False, "reason": "frame_mapping_missing"})
            continue
        renderer = make_trail_renderer(width, height, fps, args)
        warm_trail_renderer(
            renderer,
            base_frame,
            frame_tracks,
            output_to_source,
            args.video_source,
        )
        drawn = annotate_frame(
            frame,
            source_frame,
            frame_tracks.get(source_frame, []),
            renderer,
            stab_transforms,
            class_names,
            args,
        )
        out_path = output_dir / (
            f"{args.dataset_id}_{args.video_source}_{args.trail_mode}_"
            f"frame_{base_frame:06d}.jpg"
        )
        cv2.imwrite(str(out_path), frame)
        reports.append(
            {
                "frame": base_frame,
                "source_frame": source_frame,
                "ok": True,
                "boxes_drawn": drawn,
                "path": str(out_path),
            }
        )
    cap.release()
    return reports


def write_video_clip(
    video_path: Path,
    output_video: Path,
    start_frame: int,
    num_frames: int,
    frame_tracks: dict[int, list[dict[str, Any]]],
    output_to_source: dict[int, int],
    stab_transforms: dict[int, np.ndarray],
    class_names: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    temp_video = output_video.with_name(
        f".{output_video.stem}.part{output_video.suffix}"
    )
    if temp_video.exists():
        temp_video.unlink()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame < 0 or start_frame >= input_frame_count:
        cap.release()
        raise ValueError(
            f"--start-frame {start_frame} is outside input range 0..{max(0, input_frame_count - 1)}"
        )
    requested_frames = num_frames if num_frames > 0 else input_frame_count - start_frame
    writer = cv2.VideoWriter(
        str(temp_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer: {temp_video}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    renderer = make_trail_renderer(width, height, fps, args)
    warm_trail_renderer(
        renderer,
        start_frame,
        frame_tracks,
        output_to_source,
        args.video_source,
    )
    frames_written = 0
    boxes_drawn = 0
    missing_frame_mappings = 0
    try:
        for offset in range(requested_frames):
            base_frame = start_frame + offset
            ok, frame = cap.read()
            if not ok:
                break
            source_frame = source_frame_for_base(base_frame, args.video_source, output_to_source)
            if source_frame is None:
                missing_frame_mappings += 1
            else:
                boxes_drawn += annotate_frame(
                    frame,
                    source_frame,
                    frame_tracks.get(source_frame, []),
                    renderer,
                    stab_transforms,
                    class_names,
                    args,
                )
            writer.write(frame)
            frames_written += 1
            if (
                frames_written == 1
                or frames_written % 100 == 0
                or frames_written == requested_frames
            ):
                print(
                    f"[render] {output_video.name}: "
                    f"{frames_written}/{requested_frames} frames",
                    flush=True,
                )
    finally:
        cap.release()
        writer.release()

    if frames_written != requested_frames:
        raise RuntimeError(
            f"Video ended early: wrote {frames_written}/{requested_frames} frames; "
            f"incomplete file kept at {temp_video}"
        )
    validation_cap = cv2.VideoCapture(str(temp_video))
    validated_frames = int(validation_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if validation_cap.isOpened() else 0
    validation_cap.release()
    if validated_frames != frames_written:
        raise RuntimeError(
            f"Encoded video validation failed: expected {frames_written} frames, "
            f"found {validated_frames}; incomplete file kept at {temp_video}"
        )
    temp_video.replace(output_video)
    return {
        "enabled": True,
        "path": str(output_video),
        "start_frame": start_frame,
        "requested_frames": requested_frames,
        "frames_written": frames_written,
        "boxes_drawn": boxes_drawn,
        "missing_frame_mappings": missing_frame_mappings,
        "fps": fps,
        "size": [width, height],
    }


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    required_keys = ["video", "det_pkl"]
    if args.video_source == "original":
        required_keys.extend(["tracks", "tracks_meta"])
        if args.draw_space == "original":
            required_keys.append("stab_pkl")
    unresolved = [key for key in required_keys if key not in paths]
    if unresolved:
        raise FileNotFoundError(
            "Could not resolve required input path(s): " + ", ".join(unresolved)
        )
    require_existing([paths[key] for key in required_keys])

    legacy_output_dir = args.output_dir / args.dataset_id / args.version
    artifact_root = args.artifact_root or legacy_output_dir
    validation_dir = artifact_root / "validation_frames"
    reports_dir = artifact_root / "reports"
    validation_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    detection_data = load_detection_data(paths["det_pkl"])
    category_names = load_category_names(paths["initial_dir"])
    transform: np.ndarray | None = None
    if args.video_source == "tracking":
        args.draw_space = "stabilized"
        frame_tracks, output_to_source, source_to_output, result_fps = build_raw_frame_tracks(
            detection_data,
            category_names,
        )
        stab_transforms: dict[int, np.ndarray] = {}
    else:
        transform = estimate_world_to_pixel(paths["det_pkl"], data=detection_data)
        frame_tracks = build_frame_tracks(paths["tracks"], paths["tracks_meta"], transform)
        output_to_source, source_to_output, result_fps = frame_mappings_from_data(detection_data)
        stab_transforms = load_stabilization(paths.get("stab_pkl"))
        if args.draw_space == "stabilized":
            stab_transforms = {}
        else:
            # Frame 0 is the stabilization reference frame. The original
            # stabilizer treats a missing transform as identity and therefore
            # normally stores matrices starting at frame 1.
            if 0 in source_to_output and 0 not in stab_transforms:
                stab_transforms[0] = np.asarray(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=float,
                )
            expected_frames = set(source_to_output)
            missing_stabilization = sorted(expected_frames - set(stab_transforms))
            if missing_stabilization:
                preview = ", ".join(str(frame) for frame in missing_stabilization[:10])
                raise ValueError(
                    "Stabilization transforms are missing for "
                    f"{len(missing_stabilization)} source frame(s), starting with: {preview}"
                )

    class_names = classes_in_tracks(frame_tracks)

    frame_indices = parse_frame_list(args.frames)
    sample_report = write_sample_frames(
        paths["video"],
        validation_dir,
        frame_indices,
        frame_tracks,
        output_to_source,
        stab_transforms,
        class_names,
        args,
    )

    output_video = args.output_video
    if output_video is None and args.num_frames > 0:
        output_video = artifact_root / (
            f"{args.dataset_id}_{args.video_source}_{args.version}_{args.trail_mode}_"
            f"{args.start_frame:06d}_{args.num_frames}f.mp4"
        )
    clip_report = (
        write_video_clip(
            paths["video"],
            output_video,
            args.start_frame,
            args.num_frames,
            frame_tracks,
            output_to_source,
            stab_transforms,
            class_names,
            args,
        )
        if output_video is not None
        else {"enabled": False}
    )

    report = {
        "dataset_id": args.dataset_id,
        "video_source": args.video_source,
        "version": args.version,
        "draw_space": args.draw_space,
        "trail_mode": args.trail_mode,
        "trail_seconds": args.trail_seconds,
        "trail_frames_at_result_fps": int(round(args.trail_seconds * result_fps)),
        "trail_width": args.trail_width,
        "max_link_gap_frames": args.max_link_gap_frames,
        "draw_boxes": args.draw_boxes,
        "show_labels": args.show_labels,
        "show_legend": args.show_legend,
        "video": str(paths["video"]),
        "tracks": str(paths.get("tracks", "")),
        "tracks_meta": str(paths.get("tracks_meta", "")),
        "det_pkl": str(paths["det_pkl"]),
        "stab_pkl": str(paths.get("stab_pkl", "")),
        "result_fps": result_fps,
        "source_to_output_frame_count": len(source_to_output),
        "output_to_source_frame_count": len(output_to_source),
        "frame_count_with_tracks": len(frame_tracks),
        "track_rows": sum(len(items) for items in frame_tracks.values()),
        "track_count": len(
            {
                str(item.get("track_id"))
                for items in frame_tracks.values()
                for item in items
            }
        ),
        "classes": class_names,
        "world_to_pixel": np.round(transform, 6).tolist() if transform is not None else None,
        "stabilization_transforms": len(stab_transforms),
        "artifact_root": str(artifact_root),
        "sample_frames": sample_report,
        "clip": clip_report,
    }
    report_stem = output_video.stem if output_video is not None else (
        f"{args.dataset_id}_{args.video_source}_{args.version}_{args.trail_mode}"
    )
    report_path = reports_dir / f"{report_stem}_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps({"report": str(report_path), **report}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
