#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _parse_rgb(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must be R,G,B, for example 0,0,0")
    try:
        rgb = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("color values must be integers") from exc
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise argparse.ArgumentTypeError("color values must be in [0, 255]")
    return rgb


def _parse_ellipse(value: str) -> tuple[float, float, float, float, float]:
    parts = value.split(",")
    if len(parts) not in {4, 5}:
        raise argparse.ArgumentTypeError("ellipse must be cx,cy,rx,ry or cx,cy,rx,ry,angle")
    try:
        numbers = [float(part.strip()) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ellipse values must be numbers") from exc
    if numbers[2] <= 0 or numbers[3] <= 0:
        raise argparse.ArgumentTypeError("ellipse radii must be positive")
    if len(numbers) == 4:
        numbers.append(0.0)
    return tuple(numbers)  # type: ignore[return-value]


def _rgb_to_bgr(rgb: tuple[int, int, int]) -> np.ndarray:
    return np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)


def _load_labelme_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"JSON top level must be an object: {path}")
    return data


def _shape_points(shape: dict[str, Any]) -> np.ndarray | None:
    points = shape.get("points")
    if not isinstance(points, list) or len(points) < 2:
        return None

    shape_type = shape.get("shape_type") or "polygon"
    if shape_type == "rectangle" and len(points) >= 2:
        (x1, y1), (x2, y2) = points[:2]
        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    elif shape_type != "polygon":
        return None

    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 3:
        return None
    return np.rint(array).astype(np.int32)


def build_label_mask(data: dict[str, Any], size: tuple[int, int], labels: set[str]) -> np.ndarray:
    height, width = size
    mask = np.zeros((height, width), dtype=np.uint8)
    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        raise ValueError("LabelMe JSON field 'shapes' must be a list")

    matched = 0
    for shape in shapes:
        if not isinstance(shape, dict) or str(shape.get("label", "")) not in labels:
            continue
        points = _shape_points(shape)
        if points is None:
            continue
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [points], 255)
        matched += 1

    if matched == 0:
        raise ValueError(f"No polygon/rectangle shapes found for labels: {', '.join(sorted(labels))}")
    return mask


def detect_center_island_mask(
    image: np.ndarray,
    road_mask: np.ndarray,
    pad: int,
    min_area: int,
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_green = cv2.inRange(hsv, np.array([35, 35, 35]), np.array([95, 255, 255]))

    b, g, r = cv2.split(image.astype(np.int16))
    exg_green = ((2 * g - r - b) > 25) & (g > r + 5) & (g > b + 5)

    green = ((hsv_green > 0) | exg_green) & (road_mask > 0)
    green = green.astype(np.uint8) * 255
    green = cv2.morphologyEx(
        green,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    green = cv2.morphologyEx(
        green,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)),
    )

    labels_count, components, stats, centroids = cv2.connectedComponentsWithStats(green, 8)
    if labels_count <= 1:
        raise ValueError("No green island candidate was found inside the road mask")

    image_center = np.array([image.shape[1] / 2.0, image.shape[0] / 2.0])
    best_label = 0
    best_score = -1.0
    for label_id in range(1, labels_count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        centroid = np.array(centroids[label_id])
        distance = float(np.linalg.norm(centroid - image_center))
        score = area / (1.0 + distance * 0.01)
        if score > best_score:
            best_score = score
            best_label = label_id

    if best_label == 0:
        raise ValueError(f"No green island candidate reached --center-island-min-area={min_area}")

    component = (components == best_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Failed to trace center island contour")

    contour = max(contours, key=cv2.contourArea)
    island_mask = np.zeros_like(road_mask)
    if len(contour) >= 5:
        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        axes = (max(1, int(round(axis_a / 2 + pad))), max(1, int(round(axis_b / 2 + pad))))
        cv2.ellipse(
            island_mask,
            (int(round(cx)), int(round(cy))),
            axes,
            float(angle),
            0,
            360,
            255,
            -1,
        )
    else:
        cv2.drawContours(island_mask, [contour], -1, 255, -1)
        if pad > 0:
            kernel_size = pad * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            island_mask = cv2.dilate(island_mask, kernel)

    return cv2.bitwise_and(island_mask, road_mask)


def ellipse_mask(size: tuple[int, int], ellipse: tuple[float, float, float, float, float]) -> np.ndarray:
    height, width = size
    cx, cy, rx, ry, angle = ellipse
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (int(round(cx)), int(round(cy))),
        (int(round(rx)), int(round(ry))),
        angle,
        0,
        360,
        255,
        -1,
    )
    return mask


def fill_border_connected_black(
    image: np.ndarray,
    threshold: int,
    fill_rgb: tuple[int, int, int],
) -> np.ndarray:
    if threshold < 0 or threshold > 255:
        raise ValueError("--black-threshold must be in [0, 255]")

    near_black = np.all(image <= threshold, axis=2).astype(np.uint8)
    labels_count, components = cv2.connectedComponents(near_black, connectivity=8)
    if labels_count <= 1:
        return image

    border_ids = np.unique(
        np.concatenate(
            [
                components[0, :],
                components[-1, :],
                components[:, 0],
                components[:, -1],
            ]
        )
    )
    border_ids = border_ids[border_ids != 0]
    if border_ids.size == 0:
        return image

    result = image.copy()
    border_black = np.isin(components, border_ids)
    result[border_black] = _rgb_to_bgr(fill_rgb).astype(np.uint8)
    return result


def apply_mask(
    image: np.ndarray,
    mask: np.ndarray,
    outside_rgb: tuple[int, int, int],
    outside_alpha: float,
    soft_edge: int,
) -> np.ndarray:
    if outside_alpha < 0.0 or outside_alpha > 1.0:
        raise ValueError("--outside-alpha must be in [0, 1]")

    mask_alpha = mask.astype(np.float32) / 255.0
    if soft_edge > 0:
        kernel = soft_edge * 2 + 1
        mask_alpha = cv2.GaussianBlur(mask_alpha, (kernel, kernel), 0)
        mask_alpha = np.clip(mask_alpha, 0.0, 1.0)

    image_float = image.astype(np.float32)
    outside_color = _rgb_to_bgr(outside_rgb)
    outside = image_float * (1.0 - outside_alpha) + outside_color * outside_alpha
    result = image_float * mask_alpha[..., None] + outside * (1.0 - mask_alpha[..., None])
    return np.clip(result, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Keep LabelMe road polygons and replace irrelevant image regions."
    )
    parser.add_argument("--image", required=True, help="Input background image path.")
    parser.add_argument("--json", required=True, help="LabelMe JSON annotation path.")
    parser.add_argument("--output", required=True, help="Output masked image path.")
    parser.add_argument(
        "--labels",
        default="road",
        help="Comma-separated labels to keep. Default: road.",
    )
    parser.add_argument(
        "--mask-output",
        help="Optional path for the binary road mask image.",
    )
    parser.add_argument(
        "--remove-center-island",
        action="store_true",
        help="Auto-detect the central green roundabout island and remove it from the road mask.",
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
        "--center-island-mask-output",
        help="Optional path for the center island mask removed from the road mask.",
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
        help="Opacity of outside color. 1 keeps outside pure color; 0 keeps original image. Default: 1.",
    )
    parser.add_argument(
        "--soft-edge",
        type=int,
        default=0,
        help="Feather the road mask edge by this many pixels. Default: 0.",
    )
    parser.add_argument(
        "--fill-edge-black",
        action="store_true",
        help="Before masking, fill border-connected near-black pixels with --edge-fill-color.",
    )
    parser.add_argument(
        "--black-threshold",
        type=int,
        default=10,
        help="Threshold used by --fill-edge-black. Default: 10.",
    )
    parser.add_argument(
        "--edge-fill-color",
        type=_parse_rgb,
        default=(255, 255, 255),
        help="RGB color for --fill-edge-black. Default: 255,255,255.",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    json_path = Path(args.json).expanduser()
    output_path = Path(args.output).expanduser()
    labels = {label.strip() for label in args.labels.split(",") if label.strip()}
    if not labels:
        raise ValueError("--labels must contain at least one label")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    data = _load_labelme_json(json_path)
    mask = build_label_mask(data, image.shape[:2], labels)

    if args.fill_edge_black:
        image = fill_border_connected_black(image, args.black_threshold, args.edge_fill_color)

    removed_island = np.zeros_like(mask)
    if args.remove_center_island:
        removed_island = cv2.bitwise_or(
            removed_island,
            detect_center_island_mask(
                image,
                mask,
                args.center_island_pad,
                args.center_island_min_area,
            ),
        )
    if args.center_island_ellipse:
        removed_island = cv2.bitwise_or(
            removed_island,
            cv2.bitwise_and(ellipse_mask(mask.shape, args.center_island_ellipse), mask),
        )
    if np.any(removed_island):
        mask[removed_island > 0] = 0

    result = apply_mask(image, mask, args.outside_color, args.outside_alpha, args.soft_edge)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Failed to write output image: {output_path}")

    if args.mask_output:
        mask_path = Path(args.mask_output).expanduser()
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_path), mask):
            raise RuntimeError(f"Failed to write mask image: {mask_path}")

    if args.center_island_mask_output:
        island_mask_path = Path(args.center_island_mask_output).expanduser()
        island_mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(island_mask_path), removed_island):
            raise RuntimeError(f"Failed to write center island mask image: {island_mask_path}")

    print(f"[OK] wrote masked image: {output_path}")
    if args.mask_output:
        print(f"[OK] wrote road mask: {Path(args.mask_output).expanduser()}")
    if args.center_island_mask_output:
        print(f"[OK] wrote center island mask: {Path(args.center_island_mask_output).expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
