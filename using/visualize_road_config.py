#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize road_config JSON on top of a background image (or blank canvas).

Usage:
    python using/visualize_road_config.py \
        --road-config config/demo_config/road_config/track_car_empty.json \
        --output out.jpg \
        --background data1/UAV_Videos/xxx/frame0.jpg

The script draws every polygon/line by label:
    - lane_* : blue outline + label text
    - fp     : cyan fill (stabilization mask)
    - roi    : white outline
    - road   : green outline
    - others : yellow outline
"""
import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np


def decode_embedded_image(image_data: str):
    """Return numpy image if `imageData` exists and is base64 string."""
    try:
        img_bytes = base64.b64decode(image_data)
    except Exception:
        return None
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)


def load_background(path: Path, width: int, height: int):
    if path is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    if not path.exists():
        raise FileNotFoundError(f"Background image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read background image: {path}")
    return cv2.resize(img, (width, height))


def draw_label(canvas, text, pts, color):
    center = pts.mean(axis=0).astype(int)
    cv2.putText(canvas, text, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def visualize(road_config_path: Path, output_path: Path, background_path: Path = None):
    with open(road_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    width = int(cfg["imageWidth"])
    height = int(cfg["imageHeight"])

    canvas = None
    if background_path:
        canvas = load_background(background_path, width, height)
    elif cfg.get("imageData"):
        canvas = decode_embedded_image(cfg["imageData"])

    if canvas is None:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

    shape_colors = {
        "lane": (255, 0, 0),      # blue
        "fp": (255, 255, 0),      # cyan filled
        "road": (0, 255, 0),      # green
        "roi": (255, 255, 255),   # white
        "default": (0, 255, 255)  # yellow
    }

    for shape in cfg.get("shapes", []):
        label = shape.get("label", "")
        pts = np.array(shape.get("points", []), dtype=np.int32)
        if pts.size == 0:
            continue
        if label.startswith("lane_"):
            color = shape_colors["lane"]
            cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
            draw_label(canvas, label, pts, color)
        elif label == "fp":
            color = shape_colors["fp"]
            cv2.fillPoly(canvas, [pts], color)
            draw_label(canvas, label, pts, (0, 0, 0))
        elif label == "road":
            color = shape_colors["road"]
            cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
            draw_label(canvas, label, pts, color)
        elif label == "roi":
            color = shape_colors["roi"]
            cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
            draw_label(canvas, label, pts, color)
        else:
            color = shape_colors["default"]
            cv2.polylines(canvas, [pts], True, color, 1, cv2.LINE_AA)
            draw_label(canvas, label, pts, color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"Saved visualization to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize road_config polygons.")
    parser.add_argument("--road-config", required=True, help="config/demo_config/road_config/20220303_5_E_300_1.json")
    parser.add_argument("--output", required=True, help="Path to save visualization image.")
    parser.add_argument("--background", default=None, help="Optional background image.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize(Path(args.road_config), Path(args.output), Path(args.background) if args.background else None)
