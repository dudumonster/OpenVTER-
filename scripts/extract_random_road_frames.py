#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mask_road_from_labelme import (
    _load_labelme_json,
    _parse_ellipse,
    _parse_rgb,
    apply_mask,
    build_label_mask,
    detect_center_island_mask,
    ellipse_mask,
    fill_border_connected_black,
)


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")


def _resolve_video(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        videos = sorted(item for item in path.iterdir() if item.is_file() and item.suffix in VIDEO_EXTENSIONS)
        if len(videos) == 1:
            return videos[0]
        if not videos:
            raise FileNotFoundError(f"No video files found under: {path}")
        raise ValueError(
            "Multiple video files found. Please pass one video file explicitly:\n"
            + "\n".join(str(video) for video in videos[:20])
        )

    for suffix in VIDEO_EXTENSIONS:
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Video path does not exist: {path}")


def _reference_image_path(json_path: Path, data: dict[str, Any]) -> Path | None:
    image_path = data.get("imagePath")
    if not isinstance(image_path, str) or not image_path:
        return None
    candidate = Path(image_path)
    if not candidate.is_absolute():
        candidate = json_path.parent / candidate
    return candidate if candidate.is_file() else None


def _build_mask(
    data: dict[str, Any],
    frame_size: tuple[int, int],
    labels: set[str],
) -> np.ndarray:
    json_height = data.get("imageHeight")
    json_width = data.get("imageWidth")
    if isinstance(json_height, int) and isinstance(json_width, int) and json_height > 0 and json_width > 0:
        mask_size = (json_height, json_width)
    else:
        mask_size = frame_size

    mask = build_label_mask(data, mask_size, labels)
    if mask.shape[:2] != frame_size:
        mask = cv2.resize(mask, (frame_size[1], frame_size[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def _remove_center_island(
    mask: np.ndarray,
    image: np.ndarray | None,
    center_island_pad: int,
    center_island_min_area: int,
    center_island_ellipse: tuple[float, float, float, float, float] | None,
) -> np.ndarray:
    result = mask.copy()
    removed_island = np.zeros_like(mask)

    if image is not None:
        if image.shape[:2] != mask.shape[:2]:
            image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
        removed_island = cv2.bitwise_or(
            removed_island,
            detect_center_island_mask(image, result, center_island_pad, center_island_min_area),
        )

    if center_island_ellipse:
        removed_island = cv2.bitwise_or(
            removed_island,
            cv2.bitwise_and(ellipse_mask(result.shape, center_island_ellipse), result),
        )

    if np.any(removed_island):
        result[removed_island > 0] = 0
    return result


def _sample_indices(frame_count: int, count: int, seed: int | None) -> list[int]:
    if frame_count <= 0:
        raise ValueError("Video frame count is unavailable")
    rng = random.Random(seed)
    count = min(count, frame_count)
    return sorted(rng.sample(range(frame_count), count))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Randomly save video frames after applying a LabelMe road mask."
    )
    parser.add_argument("--video", required=True, help="Input video file path, or a directory with one video.")
    parser.add_argument("--json", required=True, help="LabelMe JSON annotation path.")
    parser.add_argument("--output-dir", required=True, help="Directory for masked frame images.")
    parser.add_argument("--count", type=int, default=50, help="Number of random frames to save. Default: 50.")
    parser.add_argument("--seed", type=int, default=20260703, help="Random seed. Default: 20260703.")
    parser.add_argument("--labels", default="road", help="Comma-separated labels to keep. Default: road.")
    parser.add_argument("--reference-image", help="Image used for auto center-island detection.")
    parser.add_argument(
        "--remove-center-island",
        action="store_true",
        help="Auto-detect the central green roundabout island and remove it from the road mask.",
    )
    parser.add_argument(
        "--center-island-source",
        choices=("frame", "reference"),
        default="frame",
        help="Image source used by --remove-center-island. Default: frame.",
    )
    parser.add_argument(
        "--center-island-pad",
        type=int,
        default=25,
        help="Pixels added around the auto-detected center island ellipse. Default: 25.",
    )
    parser.add_argument(
        "--center-island-min-area",
        type=int,
        default=20000,
        help="Minimum green component area for --remove-center-island. Default: 20000.",
    )
    parser.add_argument(
        "--center-island-ellipse",
        type=_parse_ellipse,
        help="Manually remove an ellipse from the road mask: cx,cy,rx,ry[,angle].",
    )
    parser.add_argument(
        "--outside-color",
        type=_parse_rgb,
        default=(255, 255, 255),
        help="RGB color for non-road regions. Default: 255,255,255.",
    )
    parser.add_argument(
        "--outside-alpha",
        type=float,
        default=1.0,
        help="Opacity of outside color. Default: 1.",
    )
    parser.add_argument("--soft-edge", type=int, default=0, help="Feather the road mask edge. Default: 0.")
    parser.add_argument(
        "--fill-edge-black",
        action="store_true",
        help="Before masking, fill border-connected near-black pixels with --edge-fill-color.",
    )
    parser.add_argument("--black-threshold", type=int, default=10, help="Threshold used by --fill-edge-black.")
    parser.add_argument(
        "--edge-fill-color",
        type=_parse_rgb,
        default=(255, 255, 255),
        help="RGB color for --fill-edge-black. Default: 255,255,255.",
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")

    video_path = _resolve_video(Path(args.video).expanduser())
    json_path = Path(args.json).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    labels = {label.strip() for label in args.labels.split(",") if label.strip()}
    if not labels:
        raise ValueError("--labels must contain at least one label")

    data = _load_labelme_json(json_path)
    reference_path = Path(args.reference_image).expanduser() if args.reference_image else _reference_image_path(json_path, data)
    reference_image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR) if reference_path else None
    if reference_path and reference_image is None:
        raise FileNotFoundError(f"Cannot read reference image: {reference_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _sample_indices(frame_count, args.count, args.seed)
    index_set = set(indices)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_mask: np.ndarray | None = None
    reference_removed_mask: np.ndarray | None = None
    saved: list[dict[str, Any]] = []
    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[WARN] failed to read frame {frame_index}")
            continue

        if base_mask is None or base_mask.shape[:2] != frame.shape[:2]:
            base_mask = _build_mask(
                data,
                frame.shape[:2],
                labels,
            )
            reference_removed_mask = None

        frame_for_mask = frame
        if args.fill_edge_black:
            frame = fill_border_connected_black(frame, args.black_threshold, args.edge_fill_color)

        if args.remove_center_island:
            if args.center_island_source == "frame":
                mask = _remove_center_island(
                    base_mask,
                    frame_for_mask,
                    args.center_island_pad,
                    args.center_island_min_area,
                    args.center_island_ellipse,
                )
            else:
                if reference_removed_mask is None:
                    if reference_image is None:
                        raise ValueError("--center-island-source reference needs --reference-image or JSON imagePath")
                    reference_removed_mask = _remove_center_island(
                        base_mask,
                        reference_image,
                        args.center_island_pad,
                        args.center_island_min_area,
                        args.center_island_ellipse,
                    )
                mask = reference_removed_mask
        elif args.center_island_ellipse:
            mask = _remove_center_island(
                base_mask,
                None,
                args.center_island_pad,
                args.center_island_min_area,
                args.center_island_ellipse,
            )
        else:
            mask = base_mask

        result = apply_mask(frame, mask, args.outside_color, args.outside_alpha, args.soft_edge)

        output_path = output_dir / f"{video_path.stem}_road_frame_{frame_index:06d}.jpg"
        if not cv2.imwrite(str(output_path), result):
            print(f"[WARN] failed to write: {output_path}")
            continue
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        saved.append({"frame_index": frame_index, "timestamp_ms": timestamp_ms, "file": str(output_path)})
        print(f"[OK] saved frame {frame_index}: {output_path}")

    cap.release()

    manifest_path = output_dir / "sampled_frames_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "video": str(video_path),
                "json": str(json_path),
                "count_requested": args.count,
                "seed": args.seed,
                "sampled_indices": indices,
                "saved": saved,
                "missing_indices": sorted(index_set - {item["frame_index"] for item in saved}),
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] saved {len(saved)}/{len(indices)} frames")
    print(f"[OK] wrote manifest: {manifest_path}")
    return 0 if len(saved) == len(indices) else 1


if __name__ == "__main__":
    raise SystemExit(main())
