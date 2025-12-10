#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick image demo for CenterNet-BBA on sampled frames.
Reuses the video config for video path/road mask, but lets you override the detection config.
Outputs annotated frames under tools/centernet_image_demo/output/ and does not touch the main pipeline.
"""
import argparse
import json
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np

# make repo root importable when running from tools/
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from detection.VehicleDetModule import VehicleDetModule
from utils.config import RoadConfig


def load_det_model(det_config_path: str) -> VehicleDetModule:
    with open(det_config_path, "r", encoding="utf-8") as f:
        det_cfg = json.load(f)
    det_model = VehicleDetModule(**det_cfg)
    det_model.load_model()
    return det_model


def load_road_mask(road_config_path: Optional[str], target_shape: Tuple[int, int], use_mask: bool = True) -> Optional[np.ndarray]:
    if not road_config_path or not use_mask:
        return None
    road_cfg = RoadConfig.fromfile(road_config_path)
    mask = road_cfg["det_mask"]
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    h, w = target_shape
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if mask.dtype != np.uint8:
        mask = (mask != 0).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        return None
    return mask


def sample_frames(video_path: str, stride: int, max_frames: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append((idx, frame))
        idx += 1
    cap.release()
    return frames


def run_demo(args):
    with open(args.video_config, "r", encoding="utf-8") as f:
        vcfg = json.load(f)

    video_folder = vcfg["video_folder"]
    first_video_name = vcfg["first_video_name"]
    video_path = os.path.join(video_folder, first_video_name)

    # allow overriding detection config for model comparison
    if args.det_config:
        det_cfg_path = args.det_config
    else:
        det_cfg_path = os.path.join(os.path.dirname(args.video_config), vcfg["detection"])

    det_model = load_det_model(det_cfg_path)
    bbox_label = vcfg.get("bbox_label", ["score"])

    os.makedirs(args.output_dir, exist_ok=True)

    frames = sample_frames(video_path, args.stride, args.num_frames)
    if not frames:
        raise RuntimeError("No frames sampled; check video path or stride.")

    mask = load_road_mask(
        vcfg.get("road_config"),
        (frames[0][1].shape[0], frames[0][1].shape[1]),
        use_mask=not args.no_mask,
    )

    for idx, frame_bgr in frames:
        frame_for_det = frame_bgr
        if mask is not None:
            frame_for_det = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

        frame_rgb = cv2.cvtColor(frame_for_det, cv2.COLOR_BGR2RGB)
        preds = det_model.inference_img_batch([frame_rgb])[0]
        if preds is None:
            preds_np = np.empty((0, 15), dtype=np.float32)
        elif hasattr(preds, "cpu"):
            preds_np = preds.cpu().numpy()
        else:
            preds_np = preds

        drawn = frame_bgr.copy()
        if preds_np.size > 0:
            drawn = det_model.draw_oriented_bboxs(drawn, preds_np, bbox_label)
        print(f"Frame {idx}: detections={preds_np.shape[0] if preds_np is not None else 0}")

        base = os.path.splitext(first_video_name)[0]
        out_path = os.path.join(args.output_dir, f"{base}_centernet_frame{idx:06d}.jpg")
        cv2.imwrite(out_path, drawn)
        print(f"Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="CenterNet-BBA image demo on sampled frames.")
    parser.add_argument(
        "--video_config",
        default="config/demo_config/video_config/20220303_5_E_300_linux.json",
        help="Path to video config JSON (provides video path and road_config).",
    )
    parser.add_argument(
        "--det_config",
        default="config/demo_config/video_config/centernet_bbavectors.json",
        help="Detection config JSON for CenterNet-BBA (override the detection field in video config).",
    )
    parser.add_argument("--stride", type=int, default=30, help="Sample one frame every N frames.")
    parser.add_argument("--num_frames", type=int, default=5, help="Maximum number of frames to sample.")
    parser.add_argument("--no-mask", action="store_true", help="Disable road mask even if provided in config.")
    parser.add_argument(
        "--output_dir",
        default="tools/centernet_image_demo/output",
        help="Directory to save annotated frames.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
