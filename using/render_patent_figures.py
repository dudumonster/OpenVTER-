#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Render patent-style figures 1-8 as A4 portrait JPG/PNG images.
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
LANDSCAPE_WIDTH = 3508
LANDSCAPE_HEIGHT = 2480
DPI = (300, 300)
DEFAULT_OUTPUT_DIR = Path(
    r"F:\专利申请\一种基于多模型融合与轨迹物理约束的无人机航拍多类交通参与者轨迹提取方法、系统、设备及存储介质"
)
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\simsun.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
]
LINE_COLOR = (20, 20, 20)
BACKGROUND = (255, 255, 255)
LINE_WIDTH = 5
ARROW_HEAD_LENGTH = 24
ARROW_HEAD_HALF_WIDTH = 12


@dataclass(frozen=True)
class Node:
    text: str
    bbox: tuple[int, int, int, int]
    shape: str = "box"
    max_font_size: int = 64
    min_font_size: int = 24


@dataclass(frozen=True)
class Connector:
    points: tuple[tuple[int, int], ...]
    arrow: bool = True
    line_width: int = LINE_WIDTH
    arrow_head_length: int = ARROW_HEAD_LENGTH
    arrow_head_half_width: int = ARROW_HEAD_HALF_WIDTH
    arrow_gap: int = 0


@dataclass(frozen=True)
class Diagram:
    file_stem: str
    nodes: tuple[Node, ...]
    connectors: tuple[Connector, ...]
    width: int = CANVAS_WIDTH
    height: int = CANVAS_HEIGHT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render patent figures 1-8.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--formats", nargs="+", default=["jpg"], help="Export formats")
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size=size)
            except OSError:
                continue
    raise RuntimeError("No usable Chinese font found.")


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
    padding_x: int = 40,
    padding_y: int = 32,
) -> tuple[ImageFont.FreeTypeFont, int, tuple[int, int, int, int]]:
    box_width = bbox[2] - bbox[0]
    box_height = bbox[3] - bbox[1]
    for font_size in range(max_font_size, min_font_size - 1, -2):
        font = load_font(font_size)
        spacing = max(8, font_size // 4)
        text_bbox = measure_multiline_text(draw, text, font, spacing)
        width = text_bbox[2] - text_bbox[0]
        height = text_bbox[3] - text_bbox[1]
        if width <= box_width - 2 * padding_x and height <= box_height - 2 * padding_y:
            return font, spacing, text_bbox
    raise ValueError(f"Text does not fit: {text}")


def draw_text_centered(draw: ImageDraw.ImageDraw, node: Node) -> None:
    font, spacing, text_bbox = fit_font(
        draw,
        node.text,
        node.bbox,
        node.max_font_size,
        node.min_font_size,
    )
    width = text_bbox[2] - text_bbox[0]
    height = text_bbox[3] - text_bbox[1]
    x1, y1, x2, y2 = node.bbox
    draw_x = x1 + (x2 - x1 - width) / 2 - text_bbox[0]
    draw_y = y1 + (y2 - y1 - height) / 2 - text_bbox[1]
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
            radius=28,
            outline=LINE_COLOR,
            width=LINE_WIDTH,
            fill=BACKGROUND,
        )
    draw_text_centered(draw, node)


def draw_connector(draw: ImageDraw.ImageDraw, connector: Connector) -> None:
    pts = connector.points
    if len(pts) < 2:
        return
    if not connector.arrow:
        for start, end in zip(pts[:-1], pts[1:]):
            draw.line([start, end], fill=LINE_COLOR, width=connector.line_width)
        return
    start = pts[-2]
    raw_end = pts[-1]
    dx = raw_end[0] - start[0]
    dy = raw_end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux = dx / length
    uy = dy / length
    end = (
        int(round(raw_end[0] - ux * connector.arrow_gap)),
        int(round(raw_end[1] - uy * connector.arrow_gap)),
    )
    for seg_start, seg_end in zip(pts[:-2], pts[1:-1]):
        draw.line([seg_start, seg_end], fill=LINE_COLOR, width=connector.line_width)
    draw.line([start, end], fill=LINE_COLOR, width=connector.line_width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    base_x = end[0] - connector.arrow_head_length * ux
    base_y = end[1] - connector.arrow_head_length * uy
    left = (
        base_x + connector.arrow_head_half_width * px,
        base_y + connector.arrow_head_half_width * py,
    )
    right = (
        base_x - connector.arrow_head_half_width * px,
        base_y - connector.arrow_head_half_width * py,
    )
    draw.polygon([end, left, right], fill=LINE_COLOR)


def render_diagram(diagram: Diagram, output_dir: Path, formats: Sequence[str]) -> list[Path]:
    image = Image.new("RGB", (diagram.width, diagram.height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    for node in diagram.nodes:
        draw_node(draw, node)
    for connector in diagram.connectors:
        draw_connector(draw, connector)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for fmt in formats:
        ext = fmt.lower()
        path = output_dir / f"{diagram.file_stem}.{ext}"
        if ext in {"jpg", "jpeg"}:
            image.save(path, format="JPEG", quality=96, subsampling=0, dpi=DPI)
        elif ext == "png":
            image.save(path, format="PNG", dpi=DPI)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        output_paths.append(path)
    return output_paths


def mid_x(bbox: tuple[int, int, int, int]) -> int:
    return (bbox[0] + bbox[2]) // 2


def mid_y(bbox: tuple[int, int, int, int]) -> int:
    return (bbox[1] + bbox[3]) // 2


def top_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return (mid_x(bbox), bbox[1])


def bottom_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return (mid_x(bbox), bbox[3])


def left_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return (bbox[0], mid_y(bbox))


def right_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return (bbox[2], mid_y(bbox))


def build_vertical_flow_diagram(file_stem: str, step_texts: Sequence[str], start_end: bool = False) -> Diagram:
    x1 = 300
    x2 = 2180
    top = 280 if not start_end else 360
    box_height = 250 if len(step_texts) <= 7 else 220
    gap = 90 if len(step_texts) <= 7 else 80
    nodes: list[Node] = []
    connectors: list[Connector] = []

    prev_bbox: tuple[int, int, int, int] | None = None
    if start_end:
        start_bbox = (980, 120, 1500, 250)
        nodes.append(Node(text="开始", bbox=start_bbox, shape="ellipse", max_font_size=54))
        prev_bbox = start_bbox

    for idx, text in enumerate(step_texts):
        y1 = top + idx * (box_height + gap)
        bbox = (x1, y1, x2, y1 + box_height)
        nodes.append(Node(text=text, bbox=bbox))
        if prev_bbox is not None:
            connectors.append(Connector(points=(bottom_center(prev_bbox), top_center(bbox))))
        prev_bbox = bbox

    if start_end and prev_bbox is not None:
        end_bbox = (980, 3275, 1500, 3405)
        nodes.append(Node(text="结束", bbox=end_bbox, shape="ellipse", max_font_size=54))
        connectors.append(Connector(points=(bottom_center(prev_bbox), top_center(end_bbox))))

    return Diagram(file_stem=file_stem, nodes=tuple(nodes), connectors=tuple(connectors))


def build_fig1() -> Diagram:
    start = (90, 120, 390, 235)
    s1 = (470, 105, 910, 250)
    s21 = (990, 105, 1430, 250)
    s22 = (1510, 105, 1950, 250)
    s23 = (2030, 105, 2470, 250)
    s31 = (2550, 105, 2990, 250)

    s32 = (2170, 470, 2610, 615)
    s33 = (2710, 470, 3150, 615)
    s41 = (2250, 760, 3070, 905)
    s42 = (2250, 1035, 3070, 1180)
    s51 = (1820, 1320, 2260, 1465)
    s52 = (2660, 1320, 3100, 1465)
    s53 = (2225, 1595, 2695, 1740)

    s61 = (2235, 1885, 2655, 2010)
    s62 = (1715, 1885, 2135, 2010)
    s63 = (1195, 1885, 1615, 2010)
    s71 = (675, 1885, 1095, 2010)
    s72 = (675, 2160, 1095, 2285)
    s73 = (1195, 2160, 1615, 2285)
    s74 = (1715, 2160, 2135, 2285)
    end = (2235, 2160, 2655, 2285)

    nodes = (
        Node(text="开始", bbox=start, shape="ellipse", max_font_size=44, min_font_size=26),
        Node(text="S1 获取无人机俯视视频及\n道路配置数据", bbox=s1, max_font_size=46, min_font_size=22),
        Node(text="S2-1 在稳像掩膜内\n提取特征点", bbox=s21, max_font_size=46, min_font_size=22),
        Node(text="S2-2 光流跟踪并估计\n部分仿射变换", bbox=s22, max_font_size=46, min_font_size=22),
        Node(text="S2-3 稳像补偿并施加\n检测掩膜", bbox=s23, max_font_size=46, min_font_size=22),
        Node(text="S3-1 重叠滑窗分块", bbox=s31, max_font_size=46, min_font_size=22),
        Node(text="S3-2 旋转框车辆检测", bbox=s32, max_font_size=44, min_font_size=22),
        Node(text="S3-3 多类交通参与者检测", bbox=s33, max_font_size=42, min_font_size=20),
        Node(text="S4-1 类别映射与\n统一旋转框表达", bbox=s41, max_font_size=48, min_font_size=22),
        Node(text="S4-2 回投原图并执行\n全局旋转NMS", bbox=s42, max_font_size=48, min_font_size=22),
        Node(text="S5-1 机动车跟踪器组关联", bbox=s51, max_font_size=42, min_font_size=20),
        Node(text="S5-2 弱势交通参与者\n跟踪器组关联", bbox=s52, max_font_size=42, min_font_size=20),
        Node(text="S5-3 ID偏移统一编号并\n生成全局轨迹", bbox=s53, max_font_size=46, min_font_size=22),
        Node(text="S6-1 像素-世界坐标映射", bbox=s61, max_font_size=42, min_font_size=20),
        Node(text="S6-2 车道归属判定与\n车辆重分类", bbox=s62, max_font_size=44, min_font_size=20),
        Node(text="S6-3 轨迹级类别\n时序平滑", bbox=s63, max_font_size=46, min_font_size=22),
        Node(text="S7-1 缺帧分级补全", bbox=s71, max_font_size=44, min_font_size=22),
        Node(text="S7-2 轨迹平滑与\n运动学计算", bbox=s72, max_font_size=46, min_font_size=22),
        Node(text="S7-3 静止门控与\n物理约束校验", bbox=s73, max_font_size=44, min_font_size=20),
        Node(text="S7-4 输出结构化轨迹及\n可视化结果", bbox=s74, max_font_size=44, min_font_size=20),
        Node(text="结束", bbox=end, shape="ellipse", max_font_size=44, min_font_size=26),
    )

    detect_branch_y = 360
    track_branch_y = 1220
    connectors = (
        Connector(points=(right_center(start), left_center(s1))),
        Connector(points=(right_center(s1), left_center(s21))),
        Connector(points=(right_center(s21), left_center(s22))),
        Connector(points=(right_center(s22), left_center(s23))),
        Connector(points=(right_center(s23), left_center(s31))),
        Connector(points=(bottom_center(s31), (mid_x(s31), detect_branch_y), (mid_x(s32), detect_branch_y), top_center(s32))),
        Connector(points=(bottom_center(s31), (mid_x(s31), detect_branch_y), (mid_x(s33), detect_branch_y), top_center(s33))),
        Connector(points=(bottom_center(s32), (mid_x(s32), 690), top_left_quarter(s41))),
        Connector(points=(bottom_center(s33), (mid_x(s33), 690), top_right_quarter(s41))),
        Connector(points=(bottom_center(s41), top_center(s42))),
        Connector(points=(bottom_center(s42), (mid_x(s42), track_branch_y), (mid_x(s51), track_branch_y), top_center(s51))),
        Connector(points=(bottom_center(s42), (mid_x(s42), track_branch_y), (mid_x(s52), track_branch_y), top_center(s52))),
        Connector(points=(bottom_center(s51), (mid_x(s51), 1510), top_left_quarter(s53))),
        Connector(points=(bottom_center(s52), (mid_x(s52), 1510), top_right_quarter(s53))),
        Connector(points=(bottom_center(s53), top_center(s61))),
        Connector(points=(left_center(s61), right_center(s62))),
        Connector(points=(left_center(s62), right_center(s63))),
        Connector(points=(left_center(s63), right_center(s71))),
        Connector(points=(bottom_center(s71), top_center(s72))),
        Connector(points=(right_center(s72), left_center(s73))),
        Connector(points=(right_center(s73), left_center(s74))),
        Connector(points=(right_center(s74), left_center(end))),
    )
    return Diagram(
        "图1_本发明方法总体流程图",
        nodes,
        connectors,
        width=LANDSCAPE_WIDTH,
        height=LANDSCAPE_HEIGHT,
    )


def build_fig2() -> Diagram:
    system = (560, 180, 1920, 360)
    m1 = (220, 620, 1080, 820)
    m2 = (1400, 620, 2260, 820)
    m3 = (220, 1120, 1080, 1320)
    m4 = (1400, 1120, 2260, 1320)
    m5 = (220, 1620, 1080, 1820)
    m6 = (1400, 1620, 2260, 1820)
    m7 = (810, 2220, 1670, 2420)

    nodes = (
        Node(text="无人机航拍多类交通参与者\n轨迹提取系统", bbox=system, max_font_size=70),
        Node(text="数据获取模块", bbox=m1),
        Node(text="预处理模块", bbox=m2),
        Node(text="多模型融合检测模块", bbox=m3),
        Node(text="分组多目标跟踪模块", bbox=m4),
        Node(text="空间语义增强模块", bbox=m5),
        Node(text="轨迹后处理模块", bbox=m6),
        Node(text="结果输出模块", bbox=m7),
    )

    connectors = (
        Connector(points=(bottom_center(system), (mid_x(system), 470), (mid_x(m1), 470), top_center(m1))),
        Connector(points=(bottom_center(system), (mid_x(system), 470), (mid_x(m2), 470), top_center(m2))),
        Connector(points=(bottom_center(m1), top_center(m3))),
        Connector(points=(bottom_center(m2), top_center(m4))),
        Connector(points=(bottom_center(m3), top_center(m5))),
        Connector(points=(bottom_center(m4), top_center(m6))),
        Connector(points=(bottom_center(m5), (mid_x(m5), 2020), (1020, 2020), top_center(m7))),
        Connector(points=(bottom_center(m6), (mid_x(m6), 2020), (1460, 2020), top_center(m7))),
    )
    return Diagram("图2_本发明系统模块结构框图", nodes, connectors)


def build_fig3() -> Diagram:
    a = (720, 170, 1760, 340)
    b = (680, 520, 1800, 690)
    c1 = (180, 920, 1080, 1090)
    c2 = (1400, 920, 2300, 1090)
    d1 = (180, 1290, 1080, 1460)
    d2 = (1400, 1290, 2300, 1460)
    e = (700, 1710, 1780, 1880)
    f = (700, 2110, 1780, 2280)
    g = (700, 2510, 1780, 2680)
    h = (700, 2910, 1780, 3080)
    nodes = (
        Node(text="预处理后视频帧", bbox=a),
        Node(text="重叠滑窗分块", bbox=b),
        Node(text="旋转框车辆检测模型", bbox=c1),
        Node(text="多类交通参与者检测模型", bbox=c2),
        Node(text="车辆候选检测集", bbox=d1),
        Node(text="多类候选检测集", bbox=d2),
        Node(text="类别映射表对齐", bbox=e),
        Node(text="统一有向包围框表示", bbox=f),
        Node(text="子块结果回投至全图", bbox=g),
        Node(text="全局旋转非极大值抑制\n得到融合检测结果", bbox=h),
    )
    connectors = (
        Connector(points=(bottom_center(a), top_center(b))),
        Connector(points=(bottom_center(b), (mid_x(b), 820), (mid_x(c1), 820), top_center(c1))),
        Connector(points=(bottom_center(b), (mid_x(b), 820), (mid_x(c2), 820), top_center(c2))),
        Connector(points=(bottom_center(c1), top_center(d1))),
        Connector(points=(bottom_center(c2), top_center(d2))),
        Connector(points=(bottom_center(d1), (mid_x(d1), 1590), (980, 1590), (980, 1710), top_center(e))),
        Connector(points=(bottom_center(d2), (mid_x(d2), 1590), (1500, 1590), (1500, 1710), top_center(e))),
        Connector(points=(bottom_center(e), top_center(f))),
        Connector(points=(bottom_center(f), top_center(g))),
        Connector(points=(bottom_center(g), top_center(h))),
    )
    return Diagram("图3_多检测模型融合与类别映射流程图", nodes, connectors)


def build_fig4() -> Diagram:
    a = (720, 170, 1760, 340)
    b = (700, 560, 1780, 730)
    c1 = (160, 980, 1080, 1150)
    c2 = (1400, 980, 2320, 1150)
    d1 = (160, 1380, 1080, 1550)
    d2 = (1400, 1380, 2320, 1550)
    e1 = (160, 1780, 1080, 1950)
    e2 = (1400, 1780, 2320, 1950)
    f = (650, 2300, 1830, 2470)
    g = (700, 2760, 1780, 2930)
    nodes = (
        Node(text="融合检测结果", bbox=a),
        Node(text="按统一类别空间进行\n目标分组", bbox=b),
        Node(text="机动车目标集合", bbox=c1),
        Node(text="弱势交通参与者目标集合", bbox=c2),
        Node(text="机动车跟踪器", bbox=d1),
        Node(text="非机动车/行人跟踪器", bbox=d2),
        Node(text="原始轨迹ID", bbox=e1),
        Node(text="偏移轨迹ID", bbox=e2),
        Node(text="统一编号管理与\n轨迹合并", bbox=f),
        Node(text="全局多类目标连续轨迹", bbox=g),
    )
    connectors = (
        Connector(points=(bottom_center(a), top_center(b))),
        Connector(points=(bottom_center(b), (mid_x(b), 860), (mid_x(c1), 860), top_center(c1))),
        Connector(points=(bottom_center(b), (mid_x(b), 860), (mid_x(c2), 860), top_center(c2))),
        Connector(points=(bottom_center(c1), top_center(d1))),
        Connector(points=(bottom_center(c2), top_center(d2))),
        Connector(points=(bottom_center(d1), top_center(e1))),
        Connector(points=(bottom_center(d2), top_center(e2))),
        Connector(points=(bottom_center(e1), (mid_x(e1), 2130), (980, 2130), top_center(f))),
        Connector(points=(bottom_center(e2), (mid_x(e2), 2130), (1500, 2130), top_center(f))),
        Connector(points=(bottom_center(f), top_center(g))),
    )
    return Diagram("图4_按类别分组的多跟踪器关联流程图", nodes, connectors)


def build_fig5() -> Diagram:
    a = (660, 170, 1820, 340)
    b = (220, 690, 1080, 860)
    c = (1400, 690, 2260, 860)
    d = (700, 1140, 1780, 1310)
    e = (220, 1620, 1080, 1790)
    f = (1400, 1620, 2260, 1790)
    g = (700, 2190, 1780, 2360)
    h = (700, 2760, 1780, 2930)
    nodes = (
        Node(text="含轨迹标识的有向包围框", bbox=a),
        Node(text="四顶点像素坐标", bbox=b),
        Node(text="像素-世界坐标\n仿射映射矩阵 M", bbox=c),
        Node(text="世界坐标顶点集合", bbox=d),
        Node(text="目标中心点", bbox=e),
        Node(text="车道多边形集合", bbox=f),
        Node(text="点内判定与车道归属", bbox=g),
        Node(text="带世界坐标与车道标识的\n增强轨迹结果", bbox=h),
    )
    connectors = (
        Connector(points=(bottom_center(a), (mid_x(a), 500), (mid_x(b), 500), top_center(b))),
        Connector(points=(bottom_center(a), (mid_x(a), 500), (mid_x(c), 500), top_center(c))),
        Connector(points=(bottom_center(b), (mid_x(b), 1010), (960, 1010), top_center(d))),
        Connector(points=(bottom_center(c), (mid_x(c), 1010), (1520, 1010), top_center(d))),
        Connector(points=(bottom_center(d), (mid_x(d), 1450), (mid_x(e), 1450), top_center(e))),
        Connector(points=(bottom_center(d), (mid_x(d), 1450), (mid_x(f), 1450), top_center(f))),
        Connector(points=(bottom_center(e), (mid_x(e), 2000), (980, 2000), top_center(g))),
        Connector(points=(bottom_center(f), (mid_x(f), 2000), (1500, 2000), top_center(g))),
        Connector(points=(bottom_center(g), top_center(h))),
    )
    return Diagram("图5_像素坐标到世界坐标映射及车道归属判定示意图", nodes, connectors)


def build_fig6() -> Diagram:
    steps = [
        "输入原始时序轨迹",
        "构建完整时间轴",
        "按缺口长度执行\n分级补全",
        "轨迹中心点平滑",
        "速度、加速度与\n航向角计算",
        "静止门控",
        "运动学物理约束校验",
        "输出可靠结构化轨迹",
    ]
    return build_vertical_flow_diagram("图6_轨迹后处理（重分类、平滑、补全、门控、约束）流程图", steps, start_end=False)


def build_fig7() -> Diagram:
    center = (760, 1470, 1720, 1680)
    n1 = (120, 530, 960, 700)
    n2 = (1520, 530, 2360, 700)
    n3 = (120, 960, 960, 1130)
    n4 = (1520, 960, 2360, 1130)
    n5 = (120, 1850, 960, 2020)
    n6 = (1520, 1850, 2360, 2020)
    n7 = (120, 2290, 960, 2460)
    n8 = (1520, 2290, 2360, 2460)
    nodes = (
        Node(text="结构化轨迹结果", bbox=center, max_font_size=66),
        Node(text="基础索引字段\nframe_index / output_frame", bbox=n1),
        Node(text="轨迹标识字段\ntrack_id / category", bbox=n2),
        Node(text="像素坐标字段\nx1~x4, y1~y4", bbox=n3),
        Node(text="世界坐标字段\nX1~X4, Y1~Y4", bbox=n4),
        Node(text="空间语义字段\nlane_id / category_name", bbox=n5),
        Node(text="运动学字段\nvelocity / accel / heading", bbox=n6),
        Node(text="轨迹质量字段\nfill_type / gap_size", bbox=n7),
        Node(text="可靠性字段\nis_stationary / phys_violation", bbox=n8),
    )
    connectors = (
        Connector(points=(bottom_center(n1), (mid_x(n1), 1280), (960, 1280), top_center(center)), arrow=False),
        Connector(points=(bottom_center(n2), (mid_x(n2), 1280), (1520, 1280), top_center(center)), arrow=False),
        Connector(points=(right_center(n3), (1100, mid_y(n3)), (1100, 1545), left_center(center)), arrow=False),
        Connector(points=(left_center(n4), (1380, mid_y(n4)), (1380, 1545), right_center(center)), arrow=False),
        Connector(points=(top_center(n5), (mid_x(n5), 1760), (960, 1760), bottom_center(center)), arrow=False),
        Connector(points=(top_center(n6), (mid_x(n6), 1760), (1520, 1760), bottom_center(center)), arrow=False),
        Connector(points=(top_center(n7), (mid_x(n7), 2140), (960, 2140), bottom_center(center)), arrow=False),
        Connector(points=(top_center(n8), (mid_x(n8), 2140), (1520, 2140), bottom_center(center)), arrow=False),
    )
    return Diagram("图7_结构化输出数据字段示意图", nodes, connectors)


def build_fig8() -> Diagram:
    a1 = (170, 180, 1090, 350)
    a2 = (1390, 180, 2310, 350)
    b1 = (170, 700, 1090, 870)
    b2 = (1390, 700, 2310, 870)
    c = (690, 1180, 1790, 1350)
    d1 = (170, 1850, 1090, 2020)
    d2 = (1390, 1850, 2310, 2020)
    e = (690, 2500, 1790, 2670)
    f = (690, 3070, 1790, 3240)
    spine_x = mid_x(c)
    left_bus_x = 620
    right_bus_x = 1860
    upper_join_y1 = 520
    upper_join_y2 = 1020
    upper_merge_y = 1120
    split_y = 1580
    merge_y = 2320
    nodes = (
        Node(text="场景视频与航拍数据", bbox=a1),
        Node(text="道路标注与标尺参数", bbox=a2),
        Node(text="检测/跟踪模型配置", bbox=b1),
        Node(text="输出参数与目录配置", bbox=b2),
        Node(text="批量配置生成器", bbox=c),
        Node(text="道路配置文件", bbox=d1),
        Node(text="视频配置文件", bbox=d2),
        Node(text="集成部署配置清单", bbox=e),
        Node(text="批量场景快速部署", bbox=f),
    )
    connectors = (
        Connector(
            points=(bottom_center(a1), (mid_x(a1), upper_join_y1), (left_bus_x, upper_join_y1)),
            arrow=False,
            line_width=4,
        ),
        Connector(
            points=(bottom_center(a2), (mid_x(a2), upper_join_y1), (right_bus_x, upper_join_y1)),
            arrow=False,
            line_width=4,
        ),
        Connector(
            points=(bottom_center(b1), (mid_x(b1), upper_join_y2), (left_bus_x, upper_join_y2)),
            arrow=False,
            line_width=4,
        ),
        Connector(
            points=(bottom_center(b2), (mid_x(b2), upper_join_y2), (right_bus_x, upper_join_y2)),
            arrow=False,
            line_width=4,
        ),
        Connector(
            points=((left_bus_x, upper_join_y1), (left_bus_x, upper_merge_y), (spine_x, upper_merge_y)),
            arrow=False,
            line_width=4,
        ),
        Connector(
            points=((right_bus_x, upper_join_y1), (right_bus_x, upper_merge_y), (spine_x, upper_merge_y)),
            arrow=False,
            line_width=4,
        ),
        Connector(
            points=((spine_x, upper_merge_y), top_center(c)),
            line_width=4,
            arrow_head_length=30,
            arrow_head_half_width=14,
            arrow_gap=12,
        ),
        Connector(
            points=(bottom_center(c), (spine_x, split_y)),
            arrow=False,
            line_width=4,
        ),
        Connector(
            points=((spine_x, split_y), (mid_x(d1), split_y), top_center(d1)),
            line_width=4,
            arrow_head_length=30,
            arrow_head_half_width=14,
            arrow_gap=12,
        ),
        Connector(
            points=((spine_x, split_y), (mid_x(d2), split_y), top_center(d2)),
            line_width=4,
            arrow_head_length=30,
            arrow_head_half_width=14,
            arrow_gap=12,
        ),
        Connector(
            points=(bottom_center(d1), (mid_x(d1), merge_y), (spine_x, merge_y)),
            arrow=False,
            line_width=4,
        ),
        Connector(
            points=(bottom_center(d2), (mid_x(d2), merge_y), (spine_x, merge_y)),
            arrow=False,
            line_width=4,
        ),
        Connector(
            points=((spine_x, merge_y), top_center(e)),
            line_width=4,
            arrow_head_length=30,
            arrow_head_half_width=14,
            arrow_gap=12,
        ),
        Connector(
            points=(bottom_center(e), top_center(f)),
            line_width=4,
            arrow_head_length=30,
            arrow_head_half_width=14,
            arrow_gap=12,
        ),
    )
    return Diagram("图8_批量集成部署配置自动生成流程图", nodes, connectors)


def top_left_quarter(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return (bbox[0] + (bbox[2] - bbox[0]) // 4, bbox[1])


def top_right_quarter(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return (bbox[2] - (bbox[2] - bbox[0]) // 4, bbox[1])


def top_left_inner(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return (bbox[0] + (bbox[2] - bbox[0]) * 2 // 5, bbox[1])


def top_right_inner(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return (bbox[2] - (bbox[2] - bbox[0]) * 2 // 5, bbox[1])


def build_diagrams() -> Iterable[Diagram]:
    return (
        build_fig1(),
        build_fig2(),
        build_fig3(),
        build_fig4(),
        build_fig5(),
        build_fig6(),
        build_fig7(),
        build_fig8(),
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    formats = tuple(fmt.lower() for fmt in args.formats)
    for diagram in build_diagrams():
        output_paths = render_diagram(diagram, output_dir, formats)
        for path in output_paths:
            print(f"Generated: {path}")


if __name__ == "__main__":
    main()
