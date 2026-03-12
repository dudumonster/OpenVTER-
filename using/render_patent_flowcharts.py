#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Render patent-style flowcharts as A4 portrait PNG/JPG images.

The charts are designed for direct insertion into Chinese patent DOCX files.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


CANVAS_WIDTH = 2480
CANVAS_HEIGHT = 3508
DPI = (300, 300)
DEFAULT_OUTPUT_DIR = Path(
    r"F:\专利申请\一种基于多模型融合与轨迹物理约束的无人机航拍多类交通参与者轨迹提取方法、系统、设备及存储介质"
)
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\simsun.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
]
LINE_COLOR = (24, 24, 24)
BACKGROUND = (255, 255, 255)
LINE_WIDTH = 5
ARROW_HEAD_LENGTH = 26
ARROW_HEAD_HALF_WIDTH = 12


@dataclass(frozen=True)
class Node:
    text: str
    bbox: tuple[int, int, int, int]
    shape: str = "box"
    max_font_size: int = 68
    min_font_size: int = 26


@dataclass(frozen=True)
class Arrow:
    points: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Diagram:
    file_stem: str
    nodes: tuple[Node, ...]
    arrows: tuple[Arrow, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render patent flowcharts.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for the rendered images.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "jpg"],
        help="Image formats to export, for example: png jpg",
    )
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for font_path in FONT_CANDIDATES:
        if not font_path.exists():
            continue
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except OSError:
            continue
    raise RuntimeError("No usable Chinese font found in the configured font candidates.")


def measure_multiline_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    spacing: int,
) -> tuple[int, int, int, int]:
    if hasattr(draw, "multiline_textbbox"):
        return draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")

    lines = text.splitlines() or [text]
    max_width = 0
    total_height = 0
    for idx, line in enumerate(lines):
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        max_width = max(max_width, right - left)
        total_height += bottom - top
        if idx < len(lines) - 1:
            total_height += spacing
    return (0, 0, max_width, total_height)


def fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    bbox: tuple[int, int, int, int],
    max_font_size: int,
    min_font_size: int,
    padding_x: int = 48,
    padding_y: int = 40,
) -> tuple[ImageFont.FreeTypeFont, int, tuple[int, int, int, int]]:
    box_width = bbox[2] - bbox[0]
    box_height = bbox[3] - bbox[1]

    for font_size in range(max_font_size, min_font_size - 1, -2):
        font = load_font(font_size)
        spacing = max(8, font_size // 4)
        text_bbox = measure_multiline_text(draw, text, font, spacing)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        if text_width <= box_width - 2 * padding_x and text_height <= box_height - 2 * padding_y:
            return font, spacing, text_bbox

    raise ValueError(f"Text does not fit within node bbox: {text!r}")


def draw_text_centered(draw: ImageDraw.ImageDraw, node: Node) -> None:
    font, spacing, text_bbox = fit_font(
        draw,
        node.text,
        node.bbox,
        max_font_size=node.max_font_size,
        min_font_size=node.min_font_size,
    )
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x1, y1, x2, y2 = node.bbox
    draw_x = x1 + (x2 - x1 - text_width) / 2 - text_bbox[0]
    draw_y = y1 + (y2 - y1 - text_height) / 2 - text_bbox[1]
    draw.multiline_text(
        (draw_x, draw_y),
        node.text,
        font=font,
        fill=LINE_COLOR,
        align="center",
        spacing=spacing,
    )


def draw_node(draw: ImageDraw.ImageDraw, node: Node) -> None:
    if node.shape == "ellipse":
        draw.ellipse(node.bbox, outline=LINE_COLOR, width=LINE_WIDTH, fill=BACKGROUND)
    else:
        draw.rounded_rectangle(
            node.bbox,
            radius=32,
            outline=LINE_COLOR,
            width=LINE_WIDTH,
            fill=BACKGROUND,
        )
    draw_text_centered(draw, node)


def draw_arrow(draw: ImageDraw.ImageDraw, arrow: Arrow) -> None:
    if len(arrow.points) < 2:
        return

    for start, end in zip(arrow.points[:-1], arrow.points[1:]):
        draw.line([start, end], fill=LINE_COLOR, width=LINE_WIDTH)

    start = arrow.points[-2]
    end = arrow.points[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return

    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    base_x = end[0] - ARROW_HEAD_LENGTH * ux
    base_y = end[1] - ARROW_HEAD_LENGTH * uy
    left = (base_x + ARROW_HEAD_HALF_WIDTH * px, base_y + ARROW_HEAD_HALF_WIDTH * py)
    right = (base_x - ARROW_HEAD_HALF_WIDTH * px, base_y - ARROW_HEAD_HALF_WIDTH * py)
    draw.polygon([end, left, right], fill=LINE_COLOR)


def validate_diagram(diagram: Diagram) -> None:
    for node in diagram.nodes:
        x1, y1, x2, y2 = node.bbox
        if not (0 <= x1 < x2 <= CANVAS_WIDTH and 0 <= y1 < y2 <= CANVAS_HEIGHT):
            raise ValueError(f"Node bbox is outside canvas: {node}")
    for arrow in diagram.arrows:
        for x, y in arrow.points:
            if not (0 <= x <= CANVAS_WIDTH and 0 <= y <= CANVAS_HEIGHT):
                raise ValueError(f"Arrow point is outside canvas: {arrow}")


def render_diagram(diagram: Diagram, output_dir: Path, formats: Sequence[str]) -> list[Path]:
    validate_diagram(diagram)
    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    for node in diagram.nodes:
        draw_node(draw, node)
    for arrow in diagram.arrows:
        draw_arrow(draw, arrow)

    output_dir.mkdir(parents=True, exist_ok=True)
    exported_paths: list[Path] = []
    for fmt in formats:
        fmt_lower = fmt.lower()
        output_path = output_dir / f"{diagram.file_stem}.{fmt_lower}"
        if fmt_lower in {"jpg", "jpeg"}:
            image.save(output_path, format="JPEG", quality=96, subsampling=0, dpi=DPI)
        elif fmt_lower == "png":
            image.save(output_path, format="PNG", dpi=DPI)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        exported_paths.append(output_path)
    return exported_paths


def mid_x(bbox: tuple[int, int, int, int]) -> int:
    return (bbox[0] + bbox[2]) // 2


def mid_y(bbox: tuple[int, int, int, int]) -> int:
    return (bbox[1] + bbox[3]) // 2


def top_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return (mid_x(bbox), bbox[1])


def bottom_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return (mid_x(bbox), bbox[3])


def top_left_quarter(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return (bbox[0] + (bbox[2] - bbox[0]) // 4, bbox[1])


def top_right_quarter(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return (bbox[2] - (bbox[2] - bbox[0]) // 4, bbox[1])


def bottom_left_quarter(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return (bbox[0] + (bbox[2] - bbox[0]) // 4, bbox[3])


def bottom_right_quarter(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return (bbox[2] - (bbox[2] - bbox[0]) // 4, bbox[3])


def build_compact_vertical_diagram() -> Diagram:
    x1 = 340
    x2 = 2140
    top = 220
    box_height = 220
    gap = 115
    texts = [
        "S1 获取无人机视频\n及道路配置",
        "S2 稳像、掩膜约束\n与分块处理",
        "S3 双模型并行\n目标检测",
        "S4 类别映射与\n旋转框统一表达",
        "S5 全局回投与\n旋转NMS融合",
        "S6 分组多跟踪器\n关联与统一编号",
        "S7 世界坐标映射、车道归属\n与车辆重分类",
        "S8 轨迹补全、运动学平滑\n静止门控与物理约束",
        "S9 输出结构化轨迹\n及可视化结果",
    ]

    nodes: list[Node] = []
    arrows: list[Arrow] = []
    prev_bbox: tuple[int, int, int, int] | None = None
    for idx, text in enumerate(texts):
        y1 = top + idx * (box_height + gap)
        bbox = (x1, y1, x2, y1 + box_height)
        nodes.append(Node(text=text, bbox=bbox))
        if prev_bbox is not None:
            arrows.append(Arrow(points=(bottom_center(prev_bbox), top_center(bbox))))
        prev_bbox = bbox

    return Diagram(
        file_stem="摘要附图_紧凑纵向版",
        nodes=tuple(nodes),
        arrows=tuple(arrows),
    )


def build_branch_fusion_diagram() -> Diagram:
    a1 = (170, 180, 1080, 340)
    a2 = (1400, 180, 2310, 340)
    b = (520, 470, 1960, 630)
    c1 = (170, 790, 1080, 950)
    c2 = (1400, 790, 2310, 950)
    d = (680, 1110, 1800, 1270)
    e = (760, 1430, 1720, 1590)
    f1 = (170, 1750, 1080, 1910)
    f2 = (1400, 1750, 2310, 1910)
    g = (540, 2070, 1940, 2230)
    h = (420, 2390, 2060, 2550)
    i = (620, 2710, 1860, 2870)
    j = (700, 3030, 1780, 3190)

    nodes = (
        Node(text="无人机俯视视频", bbox=a1),
        Node(text="道路配置数据", bbox=a2),
        Node(text="稳像、掩膜裁剪\n与分块处理", bbox=b),
        Node(text="旋转框车辆检测", bbox=c1),
        Node(text="多类目标检测", bbox=c2),
        Node(text="类别映射与\n统一旋转框", bbox=d),
        Node(text="全局旋转NMS", bbox=e),
        Node(text="机动车跟踪器组", bbox=f1),
        Node(text="非机动车/行人\n跟踪器组", bbox=f2),
        Node(text="统一ID管理与\n连续轨迹", bbox=g),
        Node(text="世界坐标映射、车道归属\n与车辆细分类", bbox=h),
        Node(text="轨迹平滑、补全\n与物理约束", bbox=i),
        Node(text="结构化结果与\n可视化", bbox=j),
    )

    arrows = (
        Arrow(points=(bottom_center(a1), (mid_x(a1), 405), (880, 405), top_left_quarter(b))),
        Arrow(points=(bottom_center(a2), (mid_x(a2), 405), (1600, 405), top_right_quarter(b))),
        Arrow(points=(bottom_center(b), (mid_x(b), 710), (mid_x(c1), 710), top_center(c1))),
        Arrow(points=(bottom_center(b), (mid_x(b), 710), (mid_x(c2), 710), top_center(c2))),
        Arrow(points=(bottom_center(c1), (mid_x(c1), 1030), (960, 1030), top_left_quarter(d))),
        Arrow(points=(bottom_center(c2), (mid_x(c2), 1030), (1520, 1030), top_right_quarter(d))),
        Arrow(points=(bottom_center(d), top_center(e))),
        Arrow(points=(bottom_center(e), (mid_x(e), 1670), (mid_x(f1), 1670), top_center(f1))),
        Arrow(points=(bottom_center(e), (mid_x(e), 1670), (mid_x(f2), 1670), top_center(f2))),
        Arrow(points=(bottom_center(f1), (mid_x(f1), 1990), (920, 1990), top_left_quarter(g))),
        Arrow(points=(bottom_center(f2), (mid_x(f2), 1990), (1560, 1990), top_right_quarter(g))),
        Arrow(points=(bottom_center(g), top_center(h))),
        Arrow(points=(bottom_center(h), top_center(i))),
        Arrow(points=(bottom_center(i), top_center(j))),
    )

    return Diagram(
        file_stem="摘要附图_分支融合版",
        nodes=nodes,
        arrows=arrows,
    )


def build_method_overview_diagram() -> Diagram:
    start_bbox = (980, 110, 1500, 240)
    end_bbox = (980, 3270, 1500, 3400)
    x1 = 290
    x2 = 2190
    top = 330
    box_height = 235
    gap = 95
    texts = [
        "步骤1 读取无人机视频、道路配置及\n像素-世界映射参数",
        "步骤2 对视频帧执行稳像处理并进行\n检测区域掩膜约束",
        "步骤3 将当前视频帧切分为\n重叠子图像块",
        "步骤4 对各子图像块分别执行\n旋转框检测和多类目标检测",
        "步骤5 将多模型结果统一为旋转框表达\n并执行全局融合抑制",
        "步骤6 依据类别分组输入多个跟踪器\n得到统一编号轨迹",
        "步骤7 将轨迹映射至世界坐标并进行\n车道判定、车辆重分类与类别平滑",
        "步骤8 对轨迹进行缺帧补全、速度加速度计算\n静止门控和物理约束校验",
        "步骤9 输出结构化轨迹结果、\n统计结果及可视化结果",
    ]

    nodes: list[Node] = [Node(text="开始", bbox=start_bbox, shape="ellipse", max_font_size=54)]
    arrows: list[Arrow] = []
    prev_bbox = start_bbox
    for idx, text in enumerate(texts):
        y1 = top + idx * (box_height + gap)
        bbox = (x1, y1, x2, y1 + box_height)
        nodes.append(Node(text=text, bbox=bbox, max_font_size=62, min_font_size=24))
        arrows.append(Arrow(points=(bottom_center(prev_bbox), top_center(bbox))))
        prev_bbox = bbox
    nodes.append(Node(text="结束", bbox=end_bbox, shape="ellipse", max_font_size=54))
    arrows.append(Arrow(points=(bottom_center(prev_bbox), top_center(end_bbox))))

    return Diagram(
        file_stem="方法总体流程图",
        nodes=tuple(nodes),
        arrows=tuple(arrows),
    )


def build_diagrams() -> Iterable[Diagram]:
    return (
        build_compact_vertical_diagram(),
        build_branch_fusion_diagram(),
        build_method_overview_diagram(),
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    formats = tuple(fmt.lower() for fmt in args.formats)

    for diagram in build_diagrams():
        exported_paths = render_diagram(diagram, output_dir, formats)
        for path in exported_paths:
            print(f"Generated: {path} ({CANVAS_WIDTH}x{CANVAS_HEIGHT}, {DPI[0]}dpi)")


if __name__ == "__main__":
    main()
