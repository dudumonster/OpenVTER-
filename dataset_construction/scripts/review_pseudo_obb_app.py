#!/usr/bin/env python3
"""Streamlit review and A/B comparison app for VisDrone pseudo OBB labels."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st


GLOBAL_NAMES = {
    1: "bicycle",
    2: "motor",
    3: "tricycle",
    4: "awning_tricycle",
}

CLASS_ZH = {
    "bicycle": "自行车",
    "motor": "摩托车/电动车",
    "tricycle": "三轮车",
    "awning_tricycle": "带篷三轮车",
}

STATUS_LABELS = {
    "auto_accept": "自动通过",
    "review": "待审核",
    "reject": "拒绝",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def latest_decisions(path: Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if sample_id:
            decisions[sample_id] = row
    return decisions


def class_name(record: dict[str, Any]) -> str:
    return (
        record.get("target_class_name")
        or record.get("source_class_name")
        or record.get("class_name")
        or "unknown"
    )


def status(record: dict[str, Any]) -> str:
    return record.get("review_status") or record.get("quality", {}).get("quality_status", "")


def image_path(record: dict[str, Any]) -> str:
    return record.get("image_path") or record.get("source_image") or ""


def hbb_xyxy(record: dict[str, Any]) -> list[float]:
    if record.get("source_hbb_xyxy"):
        return record["source_hbb_xyxy"]
    if record.get("hbb_xyxy"):
        return record["hbb_xyxy"]
    x, y, w, h = record.get("source_hbb_xywh") or record.get("hbb_xywh") or [0, 0, 1, 1]
    return [x, y, x + w, y + h]


def crop_xyxy(record: dict[str, Any]) -> list[int]:
    return record.get("crop_box_xyxy") or record.get("expanded_xyxy") or [int(v) for v in hbb_xyxy(record)]


def mask_rel_path(record: dict[str, Any]) -> str:
    return record.get("mask_path") or record.get("mask_crop") or ""


def q(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = record.get("quality", {}).get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def display_class(name: str) -> str:
    return f"{CLASS_ZH.get(name, name)} ({name})"


def load_image(record: dict[str, Any]) -> np.ndarray:
    image = cv2.imread(image_path(record))
    if image is None:
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    return image


def draw_annotations(
    image: np.ndarray,
    record: dict[str, Any],
    pseudo_root: Path,
    line_scale: float = 1.0,
    show_label: bool = True,
    show_corner_numbers: bool = True,
) -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]
    thickness = max(2, int(round(3 * line_scale)))
    thin = max(1, int(round(2 * line_scale)))
    font = max(0.55, 0.7 * line_scale)

    mask_rel = mask_rel_path(record)
    if mask_rel:
        mask_path = pseudo_root / mask_rel
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if mask is not None:
            ex1, ey1, ex2, ey2 = [int(v) for v in crop_xyxy(record)]
            ex1, ey1 = max(ex1, 0), max(ey1, 0)
            ex2, ey2 = min(ex2, w), min(ey2, h)
            mh = min(mask.shape[0], max(ey2 - ey1, 0))
            mw = min(mask.shape[1], max(ex2 - ex1, 0))
            if mh > 0 and mw > 0:
                color = np.zeros_like(canvas)
                color[ey1 : ey1 + mh, ex1 : ex1 + mw][mask[:mh, :mw] > 0] = (0, 150, 255)
                canvas = cv2.addWeighted(canvas, 1.0, color, 0.28, 0)

    x1, y1, x2, y2 = [int(round(v)) for v in hbb_xyxy(record)]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), thickness)

    vlm_box = record.get("vlm_box_xyxy") or []
    if len(vlm_box) == 4:
        vx1, vy1, vx2, vy2 = [int(round(v)) for v in vlm_box]
        cv2.rectangle(canvas, (vx1, vy1), (vx2, vy2), (255, 80, 40), thin)

    points = np.asarray(record.get("obb_points") or [], dtype=np.float32)
    if points.shape == (4, 2):
        cv2.polylines(canvas, [np.round(points).astype(np.int32)], True, (0, 255, 0), thickness)
        for idx, (px, py) in enumerate(points):
            cv2.circle(canvas, (int(round(px)), int(round(py))), max(3, thickness), (0, 0, 255), -1)
            if show_corner_numbers:
                cv2.putText(
                    canvas,
                    str(idx + 1),
                    (int(round(px)) + 5, int(round(py)) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font,
                    (0, 0, 255),
                    thickness,
                    cv2.LINE_AA,
                )

    if show_label:
        label = (
            f"{class_name(record)} {status(record)} "
            f"F={q(record, 'final_score'):.2f} G={q(record, 'geometry_score'):.2f} S={q(record, 'semantic_score'):.2f}"
        )
        cv2.putText(
            canvas,
            label,
            (max(0, x1), max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )
    return canvas


def roi_bounds(record: dict[str, Any], image_shape: tuple[int, int, int], zoom_pad: float) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    boxes: list[list[float]] = [hbb_xyxy(record), crop_xyxy(record)]
    if len(record.get("vlm_box_xyxy") or []) == 4:
        boxes.append(record["vlm_box_xyxy"])
    pts = np.asarray(record.get("obb_points") or [], dtype=np.float32)
    if pts.shape == (4, 2):
        boxes.append([float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())])

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    bw = max(x2 - x1, 20.0)
    bh = max(y2 - y1, 20.0)
    pad = max(bw, bh) * zoom_pad
    return (
        max(0, int(np.floor(x1 - pad))),
        max(0, int(np.floor(y1 - pad))),
        min(w, int(np.ceil(x2 + pad))),
        min(h, int(np.ceil(y2 + pad))),
    )


def crop_view(record: dict[str, Any], pseudo_root: Path, zoom_pad: float, target_width: int) -> np.ndarray:
    image = load_image(record)
    x1, y1, x2, y2 = roi_bounds(record, image.shape, zoom_pad)
    annotated = draw_annotations(image, record, pseudo_root, line_scale=1.0, show_label=False, show_corner_numbers=False)
    crop = annotated[y1:y2, x1:x2]
    if crop.size == 0:
        crop = annotated
    scale = target_width / max(crop.shape[1], 1)
    target_h = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (target_width, target_h), interpolation=cv2.INTER_CUBIC)
    # Draw again after resizing with thicker lines for tiny targets.
    shifted = json.loads(json.dumps(record))
    for key in ("source_hbb_xyxy", "hbb_xyxy", "crop_box_xyxy", "expanded_xyxy", "vlm_box_xyxy"):
        if len(shifted.get(key) or []) == 4:
            shifted[key] = [
                (shifted[key][0] - x1) * scale,
                (shifted[key][1] - y1) * scale,
                (shifted[key][2] - x1) * scale,
                (shifted[key][3] - y1) * scale,
            ]
    if shifted.get("obb_points"):
        shifted["obb_points"] = [[(px - x1) * scale, (py - y1) * scale] for px, py in shifted["obb_points"]]
    shifted["mask_path"] = ""
    shifted["mask_crop"] = ""
    return cv2.cvtColor(
        draw_annotations(
            resized,
            shifted,
            pseudo_root,
            line_scale=1.2,
            show_label=False,
            show_corner_numbers=False,
        ),
        cv2.COLOR_BGR2RGB,
    )


def context_view(record: dict[str, Any], pseudo_root: Path, target_width: int = 900) -> np.ndarray:
    image = draw_annotations(load_image(record), record, pseudo_root, line_scale=1.0)
    h, w = image.shape[:2]
    scale = target_width / max(w, 1)
    resized = cv2.resize(image, (target_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def summary_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "sample_id": r.get("sample_id", ""),
                "class": class_name(r),
                "status": status(r),
                "final": q(r, "final_score"),
                "geometry": q(r, "geometry_score"),
                "semantic": q(r, "semantic_score"),
                "area_ratio": q(r, "area_ratio"),
                "center_shift": q(r, "center_shift"),
                "backend": r.get("vlm_backend", ""),
            }
        )
    return pd.DataFrame(rows)


def comparison_ids(a_records: list[dict[str, Any]], b_records: list[dict[str, Any]]) -> list[str]:
    return sorted(set(r["sample_id"] for r in a_records) & set(r["sample_id"] for r in b_records))


def filter_ids(
    ids: list[str],
    a_by_id: dict[str, dict[str, Any]],
    b_by_id: dict[str, dict[str, Any]],
    cls_filter: str,
    status_filter: str,
    interesting_only: bool,
) -> list[str]:
    filtered = []
    for sid in ids:
        a = a_by_id[sid]
        b = b_by_id[sid]
        if cls_filter != "all" and class_name(a) != cls_filter:
            continue
        if status_filter != "all" and status(a) != status_filter and status(b) != status_filter:
            continue
        if interesting_only and status(a) == status(b) and abs(q(a, "geometry_score") - q(b, "geometry_score")) < 0.08:
            continue
        filtered.append(sid)
    return filtered


def metric_block(record: dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("状态", STATUS_LABELS.get(status(record), status(record)))
    cols[1].metric("Final", f"{q(record, 'final_score'):.3f}")
    cols[2].metric("Geometry", f"{q(record, 'geometry_score'):.3f}")
    cols[3].metric("Semantic", f"{q(record, 'semantic_score'):.3f}")
    st.caption(
        f"area={q(record, 'area_ratio'):.2f}  center={q(record, 'center_shift'):.2f}  "
        f"foreground={q(record, 'foreground_ratio'):.2f}  backend={record.get('vlm_backend', '')}"
    )


def render_compare(
    sid: str,
    a_record: dict[str, Any],
    b_record: dict[str, Any],
    a_root: Path,
    b_root: Path,
    zoom_pad: float,
    crop_width: int,
    show_context: bool,
) -> None:
    st.subheader(sid)
    a_col, b_col = st.columns(2)
    with a_col:
        st.markdown("**A: PureSAM2**")
        metric_block(a_record)
        st.image(crop_view(a_record, a_root, zoom_pad, crop_width), caption="A 放大效果图", use_container_width=True)
        if show_context:
            st.image(context_view(a_record, a_root), caption="A 原图上下文", use_container_width=True)
    with b_col:
        st.markdown("**B: GroundingDINO + SAM2**")
        metric_block(b_record)
        st.image(crop_view(b_record, b_root, zoom_pad, crop_width), caption="B 放大效果图", use_container_width=True)
        if show_context:
            st.image(context_view(b_record, b_root), caption="B 原图上下文", use_container_width=True)


def decision_row(record: dict[str, Any], decision: str, class_id: int, points: list[list[float]], notes: str) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "decision": decision,
        "class_id": class_id,
        "class_name": GLOBAL_NAMES.get(class_id, class_name(record)),
        "obb_points": [[round(float(x), 3), round(float(y), 3)] for x, y in points],
        "notes": notes,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def single_review(root: Path) -> None:
    records = read_jsonl(root / "quality.jsonl")
    if not records:
        records = read_jsonl(root / "review_queue.jsonl")
    if not records:
        st.warning(f"没有找到记录：{root}")
        return

    decisions_path = root / "review_decisions.jsonl"
    decisions = latest_decisions(decisions_path)
    classes = sorted({class_name(r) for r in records})
    cls = st.sidebar.selectbox("类别", ["all", *classes], format_func=lambda x: "全部类别" if x == "all" else display_class(x))
    status_filter = st.sidebar.selectbox("状态", ["all", "auto_accept", "review", "reject"], format_func=lambda x: "全部状态" if x == "all" else STATUS_LABELS.get(x, x))
    show_reviewed = st.sidebar.checkbox("显示已审核", value=True)
    rows = [
        r for r in records
        if (cls == "all" or class_name(r) == cls)
        and (status_filter == "all" or status(r) == status_filter)
        and (show_reviewed or r["sample_id"] not in decisions)
    ]
    if not rows:
        st.info("当前筛选没有样本。")
        return

    idx = st.sidebar.number_input("样本序号", min_value=0, max_value=len(rows) - 1, value=0)
    record = rows[int(idx)]
    st.caption(f"{int(idx) + 1} / {len(rows)}")
    metric_block(record)
    st.image(crop_view(record, root, st.session_state.zoom_pad, st.session_state.crop_width), caption="放大效果图", use_container_width=True)
    if st.session_state.show_context:
        st.image(context_view(record, root), caption="原图上下文", use_container_width=True)

    points = record.get("obb_points") or [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    edited = st.data_editor(pd.DataFrame(points, columns=["x", "y"]), num_rows="fixed", use_container_width=True)
    selected_class = st.selectbox("类别修正", list(GLOBAL_NAMES.values()), index=list(GLOBAL_NAMES.values()).index(class_name(record)) if class_name(record) in GLOBAL_NAMES.values() else 0)
    notes = st.text_area("备注")
    class_id = {name: cid for cid, name in GLOBAL_NAMES.items()}[selected_class]
    c1, c2, c3 = st.columns(3)
    if c1.button("通过", use_container_width=True):
        append_jsonl(decisions_path, decision_row(record, "accept", class_id, edited[["x", "y"]].astype(float).values.tolist(), notes))
        st.rerun()
    if c2.button("保存修改", use_container_width=True):
        append_jsonl(decisions_path, decision_row(record, "edit", class_id, edited[["x", "y"]].astype(float).values.tolist(), notes))
        st.rerun()
    if c3.button("拒绝", use_container_width=True):
        append_jsonl(decisions_path, decision_row(record, "reject", class_id, edited[["x", "y"]].astype(float).values.tolist(), notes))
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="VisDrone pseudo OBB review", layout="wide")
    st.title("VisDrone pseudo OBB 审核与 A/B 对比")

    st.sidebar.header("模式")
    mode = st.sidebar.radio("视图", ["A/B 对比", "单方案审核"], horizontal=True)
    st.sidebar.header("路径")
    a_root = Path(st.sidebar.text_input("A PureSAM2 目录", "dataset_construction/derived/visdrone_pseudo_obb_v2_expA"))
    b_root = Path(st.sidebar.text_input("B Grounded-SAM2 目录", "dataset_construction/derived/visdrone_pseudo_obb_v2"))
    single_root = Path(st.sidebar.text_input("单方案目录", str(b_root)))
    st.sidebar.header("显示")
    st.session_state.zoom_pad = st.sidebar.slider("放大边距", 0.2, 3.0, 1.2, 0.1)
    st.session_state.crop_width = st.sidebar.slider("放大图宽度", 480, 1400, 960, 40)
    st.session_state.show_context = st.sidebar.checkbox("显示原图上下文", value=True)

    if mode == "单方案审核":
        single_review(single_root)
        return

    a_records = read_jsonl(a_root / "quality.jsonl")
    b_records = read_jsonl(b_root / "quality.jsonl")
    if not a_records or not b_records:
        st.warning("A 或 B 的 quality.jsonl 不存在。")
        st.write({"A": str(a_root / "quality.jsonl"), "B": str(b_root / "quality.jsonl")})
        return

    a_by_id = {r["sample_id"]: r for r in a_records}
    b_by_id = {r["sample_id"]: r for r in b_records}
    ids = comparison_ids(a_records, b_records)

    classes = sorted({class_name(a_by_id[sid]) for sid in ids})
    cls = st.sidebar.selectbox("类别", ["all", *classes], format_func=lambda x: "全部类别" if x == "all" else display_class(x))
    status_filter = st.sidebar.selectbox("状态", ["all", "auto_accept", "review", "reject"], format_func=lambda x: "全部状态" if x == "all" else STATUS_LABELS.get(x, x))
    interesting_only = st.sidebar.checkbox("只看差异明显样本", value=True)
    filtered = filter_ids(ids, a_by_id, b_by_id, cls, status_filter, interesting_only)
    if not filtered:
        st.info("当前筛选没有样本。")
        return

    st.sidebar.metric("共同样本", len(ids))
    st.sidebar.metric("当前筛选", len(filtered))
    idx = st.sidebar.number_input("样本序号", min_value=0, max_value=len(filtered) - 1, value=0)
    sid = filtered[int(idx)]
    st.caption(f"{int(idx) + 1} / {len(filtered)}")

    a_df = summary_frame([a_by_id[s] for s in ids])
    b_df = summary_frame([b_by_id[s] for s in ids])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("A 自动通过", f"{(a_df['status'] == 'auto_accept').sum()} / {len(a_df)}")
    c2.metric("B 自动通过", f"{(b_df['status'] == 'auto_accept').sum()} / {len(b_df)}")
    c3.metric("A 平均 Geometry", f"{a_df['geometry'].mean():.3f}")
    c4.metric("B 平均 Geometry", f"{b_df['geometry'].mean():.3f}")

    render_compare(
        sid,
        a_by_id[sid],
        b_by_id[sid],
        a_root,
        b_root,
        st.session_state.zoom_pad,
        st.session_state.crop_width,
        st.session_state.show_context,
    )


if __name__ == "__main__":
    main()
