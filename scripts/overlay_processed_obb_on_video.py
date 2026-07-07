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
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay processed Visualization OBB tracks onto source video."
    )
    parser.add_argument("--dataset-id", default="xiang_shi_zhong_xue_019")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path(r"E:\drone_data\dong_guan\xiang_shi_zhong_xue"),
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
    parser.add_argument("--show-labels", action="store_true", default=True)
    parser.add_argument("--hide-labels", action="store_false", dest="show_labels")
    parser.add_argument("--max-objects-per-frame", type=int, default=0)
    return parser.parse_args()


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
    video_path = args.video_path or (args.video_dir / f"{dataset_id}.MP4")
    initial_dir = args.visualization_dir / "Initial results" / dataset_id
    if args.version == "final":
        tracks_dir = args.visualization_dir / "Final Data" / dataset_id
    else:
        tracks_dir = args.visualization_dir / "Adjusted results" / dataset_id / args.version

    paths = {
        "video": video_path,
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
    return paths


def require_existing(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + "; ".join(missing))


def estimate_world_to_pixel(det_pkl: Path, max_pairs: int = 5000) -> np.ndarray:
    with det_pkl.open("rb") as fh:
        data = pickle.load(fh)

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
            }
            frame_tracks[frame_idx].append(item)
    return frame_tracks


def color_for_class(class_name: str, interpolated: bool) -> tuple[int, int, int]:
    color = CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"])
    if not interpolated:
        return color
    return tuple(int(0.55 * channel + 0.45 * 255) for channel in color)


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


def write_sample_frames(
    video_path: Path,
    output_dir: Path,
    frame_indices: list[int],
    frame_tracks: dict[int, list[dict[str, Any]]],
    stab_transforms: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    reports: list[dict[str, Any]] = []
    for frame_idx in frame_indices:
        frame = seek_and_read(cap, frame_idx)
        if frame is None:
            reports.append({"frame": frame_idx, "ok": False, "reason": "read_failed"})
            continue
        drawn = draw_tracks(
            frame,
            frame_tracks.get(frame_idx, []),
            frame_idx,
            stab_transforms,
            args.draw_space,
            args.line_width,
            args.show_labels,
            args.max_objects_per_frame,
        )
        out_path = output_dir / f"{args.dataset_id}_frame_{frame_idx:06d}_{args.draw_space}.jpg"
        cv2.imwrite(str(out_path), frame)
        reports.append({"frame": frame_idx, "ok": True, "drawn": drawn, "path": str(out_path)})
    cap.release()
    return reports


def write_video_clip(
    video_path: Path,
    output_video: Path,
    start_frame: int,
    num_frames: int,
    frame_tracks: dict[int, list[dict[str, Any]]],
    stab_transforms: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if num_frames <= 0:
        return {"enabled": False}

    output_video.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer: {output_video}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_written = 0
    boxes_drawn = 0
    for offset in range(num_frames):
        frame_idx = start_frame + offset
        ok, frame = cap.read()
        if not ok:
            break
        boxes_drawn += draw_tracks(
            frame,
            frame_tracks.get(frame_idx, []),
            frame_idx,
            stab_transforms,
            args.draw_space,
            args.line_width,
            args.show_labels,
            args.max_objects_per_frame,
        )
        writer.write(frame)
        frames_written += 1

    cap.release()
    writer.release()
    return {
        "enabled": True,
        "path": str(output_video),
        "start_frame": start_frame,
        "frames_written": frames_written,
        "boxes_drawn": boxes_drawn,
        "fps": fps,
        "size": [width, height],
    }


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    require_existing([paths["video"], paths["tracks"], paths["tracks_meta"], paths["det_pkl"]])

    output_dir = args.output_dir / args.dataset_id / args.version
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = estimate_world_to_pixel(paths["det_pkl"])
    frame_tracks = build_frame_tracks(paths["tracks"], paths["tracks_meta"], transform)
    stab_transforms = load_stabilization(paths.get("stab_pkl"))
    if args.draw_space == "stabilized":
        stab_transforms = {}

    frame_indices = parse_frame_list(args.frames)
    sample_report = write_sample_frames(
        paths["video"],
        output_dir / "frames",
        frame_indices,
        frame_tracks,
        stab_transforms,
        args,
    )

    output_video = args.output_video
    if output_video is None and args.num_frames > 0:
        output_video = output_dir / (
            f"{args.dataset_id}_{args.version}_{args.draw_space}_"
            f"{args.start_frame:06d}_{args.num_frames}f.mp4"
        )
    clip_report = (
        write_video_clip(
            paths["video"],
            output_video,
            args.start_frame,
            args.num_frames,
            frame_tracks,
            stab_transforms,
            args,
        )
        if output_video is not None
        else {"enabled": False}
    )

    report = {
        "dataset_id": args.dataset_id,
        "version": args.version,
        "draw_space": args.draw_space,
        "video": str(paths["video"]),
        "tracks": str(paths["tracks"]),
        "tracks_meta": str(paths["tracks_meta"]),
        "det_pkl": str(paths["det_pkl"]),
        "stab_pkl": str(paths.get("stab_pkl", "")),
        "frame_count_with_tracks": len(frame_tracks),
        "track_rows": sum(len(items) for items in frame_tracks.values()),
        "world_to_pixel": np.round(transform, 6).tolist(),
        "stabilization_transforms": len(stab_transforms),
        "sample_frames": sample_report,
        "clip": clip_report,
    }
    report_path = output_dir / f"{args.dataset_id}_{args.version}_{args.draw_space}_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps({"report": str(report_path), **report}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
