#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare per-video configs for OpenVTER integrated deployment and export labeling backgrounds.

Features:
1) Generate one JSON config per video under config/demo_config/Integrated deployment.
2) Generate a bootstrap road_config JSON per video (for pre-labeling stage).
3) Export first frame and averaged background image for each video.

Example:
python using/prepare_integrated_deployment.py \
  --video-root E:/drone_data \
  --video E:/drone_data/wu_han/dun_yang_da_sha/dun_yang_da_sha_001.MP4 \
  --template config/demo_config/video_config/20220303_5_E_300_fusion_yolov5.json \
  --integrated-dir "config/demo_config/Integrated deployment" \
  --background-root E:/drone_data/background_figure
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".ts", ".mts", ".m2ts", ".webm", ".mpg", ".mpeg", ".3gp", ".asf", ".rmvb", ".rm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Integrated deployment configs and background figures.")
    parser.add_argument("--video-root", required=True, help="Video root folder, e.g. E:/drone_data")
    parser.add_argument("--template", required=True, help="Template config json path.")
    parser.add_argument("--integrated-dir", required=True, help='Output config folder, e.g. config/demo_config/Integrated deployment')
    parser.add_argument("--background-root", required=True, help="Background output root, e.g. E:/drone_data/background_figure")
    parser.add_argument("--video", default=None, help="Single video path for demo; if omitted, process all videos under --video-root.")
    parser.add_argument("--avg-frames", type=int, default=50, help="Number of frames used to compute average background.")
    parser.add_argument("--skip-existing", action="store_true", default=False, help="Skip videos with existing config and background files.")
    return parser.parse_args()


def to_posix_str(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def sanitize_name(text: str) -> str:
    text = text.strip().replace("\\", "_").replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


def list_videos(video_root: Path, single_video: Optional[Path]) -> List[Path]:
    if single_video is not None:
        return [single_video.resolve()]
    videos = []
    for p in video_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p.resolve())
    return sorted(videos)


def extract_first_and_background(video_path: Path, avg_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    first_frame = None
    frame_count = 0
    acc = None
    try:
        while frame_count < avg_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if first_frame is None:
                first_frame = frame.copy()
                acc = np.zeros_like(frame, dtype=np.float64)
            acc += frame.astype(np.float64)
            frame_count += 1
    finally:
        cap.release()

    if first_frame is None or acc is None or frame_count == 0:
        raise RuntimeError(f"No frames read from video: {video_path}")

    bg = np.clip(acc / float(frame_count), 0, 255).astype(np.uint8)
    return first_frame, bg


def build_bootstrap_road_config(image_path: Path, width: int, height: int) -> Dict:
    # Full-frame road polygon + 4 fixed-point rectangles + one length marker.
    margin_x = max(20, width // 20)
    margin_y = max(20, height // 20)
    fp_w = max(80, width // 8)
    fp_h = max(80, height // 8)

    fp_boxes = [
        (margin_x, margin_y),
        (width - margin_x - fp_w, margin_y),
        (margin_x, height - margin_y - fp_h),
        (width - margin_x - fp_w, height - margin_y - fp_h),
    ]

    shapes = [
        {
            "label": "road",
            "points": [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {},
        }
    ]

    for x, y in fp_boxes:
        shapes.append(
            {
                "label": "fp",
                "points": [[x, y], [x + fp_w, y + fp_h]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            }
        )

    # A rough length reference to avoid None length_per_pixel in downstream code.
    line_y = int(height * 0.9)
    x1 = int(width * 0.1)
    x2 = int(width * 0.2)
    shapes.append(
        {
            "label": "length_10",
            "points": [[x1, line_y], [x2, line_y]],
            "group_id": None,
            "shape_type": "line",
            "flags": {},
        }
    )

    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def map_template_path_value(raw_value, template_dir: Path) -> object:
    # Keep absolute paths unchanged; rewrite relative references to be relative from integrated folder later.
    if isinstance(raw_value, list):
        return [map_template_path_value(v, template_dir) for v in raw_value]
    if not isinstance(raw_value, str):
        return raw_value
    p = Path(raw_value)
    if p.is_absolute():
        return raw_value
    resolved = (template_dir / p).resolve()
    return to_posix_str(resolved)


def main() -> None:
    args = parse_args()

    video_root = Path(args.video_root).resolve()
    template_path = Path(args.template).resolve()
    integrated_dir = Path(args.integrated_dir).resolve()
    background_root = Path(args.background_root).resolve()
    single_video = Path(args.video).resolve() if args.video else None

    integrated_dir.mkdir(parents=True, exist_ok=True)
    road_config_dir = integrated_dir / "road_config"
    road_config_dir.mkdir(parents=True, exist_ok=True)
    background_root.mkdir(parents=True, exist_ok=True)

    with open(template_path, "r", encoding="utf-8") as f:
        template = json.load(f)
    template_dir = template_path.parent

    videos = list_videos(video_root, single_video)
    if not videos:
        raise RuntimeError("No videos found to process.")

    generated = []
    failed = []
    total = len(videos)
    for idx, video_path in enumerate(videos, 1):
        try:
            rel = video_path.relative_to(video_root)
            rel_parent = rel.parent  # e.g. wu_han/dun_yang_da_sha
            scene_out_dir = (background_root / rel_parent).resolve()
            scene_out_dir.mkdir(parents=True, exist_ok=True)

            stem = video_path.stem
            first_frame_name = f"first_frame_{stem}.jpg"
            background_name = f"background_{stem}.jpg"
            first_frame_path = scene_out_dir / first_frame_name
            background_path = scene_out_dir / background_name

            safe_rel = sanitize_name(str(rel.with_suffix("")))
            road_config_path = road_config_dir / f"{safe_rel}.json"
            config_path = integrated_dir / f"{safe_rel}.json"

            if args.skip_existing and first_frame_path.exists() and background_path.exists() and config_path.exists() and road_config_path.exists():
                print(f"[{idx}/{total}] skip existing: {video_path.name}")
                generated.append(
                    {
                        "video": to_posix_str(video_path),
                        "config": to_posix_str(config_path),
                        "road_config": to_posix_str(road_config_path),
                        "first_frame": to_posix_str(first_frame_path),
                        "background": to_posix_str(background_path),
                    }
                )
                continue

            first_frame, background = extract_first_and_background(video_path, args.avg_frames)
            cv2.imwrite(str(first_frame_path), first_frame)
            cv2.imwrite(str(background_path), background)

            h, w = first_frame.shape[:2]

            road_cfg = build_bootstrap_road_config(first_frame_path, w, h)
            with open(road_config_path, "w", encoding="utf-8") as f:
                json.dump(road_cfg, f, ensure_ascii=False, indent=2)

            cfg = dict(template)
            cfg["video_file"] = to_posix_str(video_path)
            cfg.pop("video_folder", None)
            cfg.pop("first_video_name", None)
            cfg.pop("video_num", None)
            cfg.pop("video_name", None)
            cfg["save_folder"] = to_posix_str(scene_out_dir)
            cfg["road_config"] = to_posix_str(road_config_path)
            cfg["stabilize_file"] = f"affine_trans_matrix_{stem}.pkl"

            # Ensure template relative refs still work from new folder by converting to absolute paths.
            if "detection" in cfg:
                cfg["detection"] = map_template_path_value(cfg["detection"], template_dir)
            if "tracking" in cfg:
                cfg["tracking"] = map_template_path_value(cfg["tracking"], template_dir)

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)

            generated.append(
                {
                    "video": to_posix_str(video_path),
                    "config": to_posix_str(config_path),
                    "road_config": to_posix_str(road_config_path),
                    "first_frame": to_posix_str(first_frame_path),
                    "background": to_posix_str(background_path),
                }
            )
            print(f"[{idx}/{total}] done: {video_path.name}")
        except Exception as exc:
            failed.append({"video": to_posix_str(video_path), "error": str(exc)})
            print(f"[{idx}/{total}] failed: {video_path.name} -> {exc}")

    manifest_path = integrated_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(generated), "failed_count": len(failed), "items": generated, "failed": failed}, f, ensure_ascii=False, indent=2)

    print(f"Generated configs: {len(generated)}")
    print(f"Failed: {len(failed)}")
    print(f"Integrated dir: {integrated_dir}")
    print(f"Manifest: {manifest_path}")
    if generated:
        print("Sample:")
        print(json.dumps(generated[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
