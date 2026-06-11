#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Interactive trajectory kinematic checker for OpenVTER Final Data CSVs."""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THRESHOLDS = {
    "max_speed": 35.0,                   # m/s
    "max_abs_acc": 10.0,                 # m/s^2
    "max_abs_lon_acc": 10.0,             # m/s^2
    "max_abs_lat_acc": 10.0,             # m/s^2
    "max_delta_speed": 5.0,              # m/s per frame
    "max_delta_velocity_component": 5.0, # m/s per frame
    "max_delta_heading": 30.0,           # deg per frame
    "max_speed_consistency_error": 0.5,  # m/s
    "max_acc_consistency_error": 1.0,    # m/s^2
    "max_heading_motion_error": 45.0,    # deg
}

DERIVATIVE_CONSISTENCY = {
    "velocity_rel_tol": 0.20,
    "velocity_abs_tol": 0.30,      # m/s
    "acceleration_rel_tol": 0.20,
    "acceleration_abs_tol": 0.35,  # m/s^2
}

REQUIRED_TRACK_COLUMNS = [
    "trackId",
    "frame",
    "xCenter",
    "yCenter",
    "heading",
    "xVelocity",
    "yVelocity",
    "xAcceleration",
    "yAcceleration",
    "lonVelocity",
    "latVelocity",
    "lonAcceleration",
    "latAcceleration",
]

NUMERIC_TRACK_COLUMNS = [
    "trackId",
    "frame",
    "xCenter",
    "yCenter",
    "heading",
    "xVelocity",
    "yVelocity",
    "xAcceleration",
    "yAcceleration",
    "lonVelocity",
    "latVelocity",
    "lonAcceleration",
    "latAcceleration",
]

TRACK_META_HINT_COLUMNS = ["trackId", "numFrames", "class"]

ABNORMAL_COLUMNS = [
    "abnormal_speed",
    "abnormal_x_acc",
    "abnormal_y_acc",
    "abnormal_lon_acc",
    "abnormal_lat_acc",
    "abnormal_delta_speed",
    "abnormal_delta_lonVelocity",
    "abnormal_delta_latVelocity",
    "abnormal_delta_heading",
    "abnormal_speed_error",
    "abnormal_acc_error",
    "abnormal_angle_error",
]

TRACK_FIGSIZE = (22, 16)
SUMMARY_FIGSIZE = (22, 16)
GRID_KW = {
    "height_ratios": [1.0, 1.0, 1.0, 1.18],
    "wspace": 0.28,
    "hspace": 0.58,
}
FIGURE_ADJUST = {
    "left": 0.045,
    "right": 0.985,
    "bottom": 0.075,
    "top": 0.93,
}
AXIS_TITLE_PAD = 9
AXIS_LABEL_PAD = 6
FIGURE_TITLE_FONT_SIZE = 14      # 顶部总标题
AXIS_TITLE_FONT_SIZE = 12        # 每个子图标题
AXIS_LABEL_FONT_SIZE = 8      # x/y 轴标签
AXIS_TICK_FONT_SIZE = 8        # 坐标轴刻度
LEGEND_FONT_SIZE = 8           # 图例
ANNOTATION_FONT_SIZE = 8       # 图内标注文字
ABNORMAL_LABEL_FONT_SIZE = 8     # abnormal counts 横轴标签
SUMMARY_FONT_SIZE = 8          # 右下角 summary 文本
'''
SUMMARY_COLUMN_WIDTH     控制每列每行能放多少字
SUMMARY_RIGHT_COLUMN_X   控制右列从多靠右的位置开始
SUMMARY_LINE_SPACING     控制上下行之间的距离
'''
SUMMARY_COLUMN_WIDTH = 32
SUMMARY_RIGHT_COLUMN_X = 0.60
SUMMARY_LINE_SPACING = 1.50


def find_csv_files(folder_path):
    """Find recordingMeta, tracksMeta, and tracks CSV files without hard-coded prefixes."""
    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError("数据子文件夹不存在: %s" % folder_path)

    found = {"recordingMeta": None, "tracksMeta": None, "tracks": None}
    for path in sorted(folder_path.glob("*.csv")):
        name = path.name.lower()
        if "recordingmeta" in name:
            found["recordingMeta"] = path
        elif "tracksmeta" in name:
            found["tracksMeta"] = path
        elif "tracks" in name:
            found["tracks"] = path

    missing = [key for key, value in found.items() if value is None]
    if missing:
        raise FileNotFoundError("在 %s 中没有找到 CSV 文件: %s" % (folder_path, ", ".join(missing)))
    return found


def load_dataset(data_root, folder):
    """Load tracks.csv and tracksMeta.csv from the selected folder."""
    folder_path = Path(data_root) / folder
    csv_files = find_csv_files(folder_path)
    tracks_df = pd.read_csv(str(csv_files["tracks"]))
    tracks_meta_df = pd.read_csv(str(csv_files["tracksMeta"]))
    recording_meta_df = pd.read_csv(str(csv_files["recordingMeta"]))

    missing_tracks = [column for column in REQUIRED_TRACK_COLUMNS if column not in tracks_df.columns]
    if missing_tracks:
        raise ValueError("tracks.csv 缺少必要字段: %s" % ", ".join(missing_tracks))

    missing_meta_hints = [column for column in TRACK_META_HINT_COLUMNS if column not in tracks_meta_df.columns]
    if missing_meta_hints:
        print("提示: tracksMeta.csv 缺少字段: %s；对应信息将显示为 unknown 或从 tracks.csv 推断。" % ", ".join(missing_meta_hints))

    for column in NUMERIC_TRACK_COLUMNS:
        tracks_df[column] = pd.to_numeric(tracks_df[column], errors="coerce")
    if "trackId" in tracks_meta_df.columns:
        tracks_meta_df["trackId"] = pd.to_numeric(tracks_meta_df["trackId"], errors="coerce")
    if "numFrames" in tracks_meta_df.columns:
        tracks_meta_df["numFrames"] = pd.to_numeric(tracks_meta_df["numFrames"], errors="coerce")

    return tracks_df, tracks_meta_df, recording_meta_df


def _frame_rate(recording_meta_df):
    try:
        value = pd.to_numeric(recording_meta_df["frameRate"], errors="coerce").dropna().iloc[0]
        return float(value)
    except Exception:
        return 29.97


def load_quality_report(data_root, folder):
    """Best-effort load of converter quality_report for summary diagnostics."""
    root = Path(data_root).resolve()
    candidates = [
        root.parent / "Adjusted results" / folder / "moving_filtered" / "quality_report.json",
        root.parent / "Adjusted results" / folder / "full" / "quality_report.json",
        root.parent.parent / "Adjusted results" / folder / "moving_filtered" / "quality_report.json",
        root.parent.parent / "Adjusted results" / folder / "full" / "quality_report.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                return {}
    return {}


def normalize_angle_diff(angle_diff):
    """Normalize angle differences to [-180, 180] degrees."""
    return (angle_diff + 180.0) % 360.0 - 180.0


def _safe_max_abs(series):
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return float("nan")
    return float(values.abs().max())


def _safe_max(series):
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return float("nan")
    return float(values.max())


def _safe_p95(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(np.percentile(values.values, 95))


def _safe_ratio(series):
    values = pd.Series(series).dropna()
    if values.empty:
        return 0.0
    return float(values.astype(bool).sum()) / float(len(values))


def _finite_xy_mask(df):
    return np.isfinite(df["xCenter"].values) & np.isfinite(df["yCenter"].values)


def _series_false(index):
    return pd.Series(False, index=index)


def _differentiate(values, frames, frame_rate):
    values = np.asarray(values, dtype=float)
    frames = np.asarray(frames, dtype=float)
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if n == 1:
        out[0] = 0.0
        return out
    for i in range(n):
        if i == 0:
            j0, j1 = 0, 1
        elif i == n - 1:
            j0, j1 = n - 2, n - 1
        else:
            j0, j1 = i - 1, i + 1
        dt = (frames[j1] - frames[j0]) / float(frame_rate)
        out[i] = np.nan if abs(dt) < 1e-12 else (values[j1] - values[j0]) / dt
    return out


def _relative_error(error, reference_norm, abs_tol):
    error = np.asarray(error, dtype=float)
    reference_norm = np.asarray(reference_norm, dtype=float)
    out = np.full(len(error), np.nan, dtype=float)
    mask = np.isfinite(error) & np.isfinite(reference_norm) & (reference_norm >= max(float(abs_tol), 1e-12))
    out[mask] = error[mask] / reference_norm[mask]
    return out


def compute_track_kinematic_checks(track_df, frame_rate=29.97):
    """Compute derived kinematic fields and abnormal-frame flags for one track."""
    df = track_df.copy()
    df = df.sort_values("frame", kind="mergesort").reset_index(drop=True)
    for column in NUMERIC_TRACK_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["speed_xy"] = np.sqrt(df["xVelocity"] ** 2 + df["yVelocity"] ** 2)
    df["speed_lonlat"] = np.sqrt(df["lonVelocity"] ** 2 + df["latVelocity"] ** 2)
    df["speed_error"] = df["speed_xy"] - df["speed_lonlat"]

    df["acc_xy"] = np.sqrt(df["xAcceleration"] ** 2 + df["yAcceleration"] ** 2)
    df["acc_lonlat"] = np.sqrt(df["lonAcceleration"] ** 2 + df["latAcceleration"] ** 2)
    df["acc_error"] = df["acc_xy"] - df["acc_lonlat"]

    df["vx_from_pos"] = _differentiate(df["xCenter"].values, df["frame"].values, frame_rate)
    df["vy_from_pos"] = _differentiate(df["yCenter"].values, df["frame"].values, frame_rate)
    df["velocity_derivative_error"] = np.sqrt((df["xVelocity"] - df["vx_from_pos"]) ** 2 + (df["yVelocity"] - df["vy_from_pos"]) ** 2)
    df["velocity_derivative_reference_norm"] = np.sqrt(df["vx_from_pos"] ** 2 + df["vy_from_pos"] ** 2)
    df["velocity_derivative_relative_error"] = _relative_error(
        df["velocity_derivative_error"].values,
        df["velocity_derivative_reference_norm"].values,
        DERIVATIVE_CONSISTENCY["velocity_abs_tol"],
    )
    velocity_allowed_error = np.maximum(
        DERIVATIVE_CONSISTENCY["velocity_abs_tol"],
        DERIVATIVE_CONSISTENCY["velocity_rel_tol"] * df["velocity_derivative_reference_norm"],
    )
    df["velocity_derivative_over_20pct"] = df["velocity_derivative_error"] > velocity_allowed_error

    df["ax_from_vel"] = _differentiate(df["xVelocity"].values, df["frame"].values, frame_rate)
    df["ay_from_vel"] = _differentiate(df["yVelocity"].values, df["frame"].values, frame_rate)
    df["acceleration_derivative_error"] = np.sqrt((df["xAcceleration"] - df["ax_from_vel"]) ** 2 + (df["yAcceleration"] - df["ay_from_vel"]) ** 2)
    df["acceleration_derivative_reference_norm"] = np.sqrt(df["ax_from_vel"] ** 2 + df["ay_from_vel"] ** 2)
    df["acceleration_derivative_relative_error"] = _relative_error(
        df["acceleration_derivative_error"].values,
        df["acceleration_derivative_reference_norm"].values,
        DERIVATIVE_CONSISTENCY["acceleration_abs_tol"],
    )
    acceleration_allowed_error = np.maximum(
        DERIVATIVE_CONSISTENCY["acceleration_abs_tol"],
        DERIVATIVE_CONSISTENCY["acceleration_rel_tol"] * df["acceleration_derivative_reference_norm"],
    )
    df["acceleration_derivative_over_20pct"] = df["acceleration_derivative_error"] > acceleration_allowed_error

    # Heading-compatible convention for this dataset:
    # 0 deg = +Y, 90 deg = +X. Therefore the motion direction must use
    # atan2(xVelocity, yVelocity), not the standard math angle atan2(y, x).
    df["motion_heading"] = np.degrees(np.arctan2(df["xVelocity"], df["yVelocity"]))
    df["angle_error"] = normalize_angle_diff(df["motion_heading"] - df["heading"])

    if len(df) >= 2:
        df["delta_speed"] = df["speed_xy"].diff()
        df["delta_lonVelocity"] = df["lonVelocity"].diff()
        df["delta_latVelocity"] = df["latVelocity"].diff()
        df["delta_heading"] = normalize_angle_diff(df["heading"].diff())
    else:
        df["delta_speed"] = np.nan
        df["delta_lonVelocity"] = np.nan
        df["delta_latVelocity"] = np.nan
        df["delta_heading"] = np.nan

    df["abnormal_speed"] = df["speed_xy"] > THRESHOLDS["max_speed"]
    df["abnormal_x_acc"] = df["xAcceleration"].abs() > THRESHOLDS["max_abs_acc"]
    df["abnormal_y_acc"] = df["yAcceleration"].abs() > THRESHOLDS["max_abs_acc"]
    df["abnormal_lon_acc"] = df["lonAcceleration"].abs() > THRESHOLDS["max_abs_lon_acc"]
    df["abnormal_lat_acc"] = df["latAcceleration"].abs() > THRESHOLDS["max_abs_lat_acc"]
    df["abnormal_delta_speed"] = df["delta_speed"].abs() > THRESHOLDS["max_delta_speed"]
    df["abnormal_delta_lonVelocity"] = df["delta_lonVelocity"].abs() > THRESHOLDS["max_delta_velocity_component"]
    df["abnormal_delta_latVelocity"] = df["delta_latVelocity"].abs() > THRESHOLDS["max_delta_velocity_component"]
    df["abnormal_delta_heading"] = df["delta_heading"].abs() > THRESHOLDS["max_delta_heading"]
    df["abnormal_speed_error"] = df["speed_error"].abs() > THRESHOLDS["max_speed_consistency_error"]
    df["abnormal_acc_error"] = df["acc_error"].abs() > THRESHOLDS["max_acc_consistency_error"]
    df["abnormal_angle_error"] = df["angle_error"].abs() > THRESHOLDS["max_heading_motion_error"]

    for column in ABNORMAL_COLUMNS:
        df[column] = df[column].fillna(False).astype(bool)
    df["abnormal_frame"] = df[ABNORMAL_COLUMNS].any(axis=1)
    return df


def _meta_value(track_meta_row, column, default="unknown"):
    if track_meta_row is None:
        return default
    try:
        value = track_meta_row[column]
    except Exception:
        return default
    if pd.isnull(value):
        return default
    return value


def _find_meta_row(tracks_meta_df, track_id):
    if tracks_meta_df is None or tracks_meta_df.empty or "trackId" not in tracks_meta_df.columns:
        return None
    ids = pd.to_numeric(tracks_meta_df["trackId"], errors="coerce")
    matches = tracks_meta_df.loc[ids == int(track_id)]
    if matches.empty:
        return None
    return matches.iloc[0]


def summarize_track(track_df, track_meta_row):
    """Return a per-track summary and abnormal ratio."""
    checked = track_df if "speed_xy" in track_df.columns else compute_track_kinematic_checks(track_df)
    track_id = int(checked["trackId"].iloc[0]) if not checked.empty and pd.notnull(checked["trackId"].iloc[0]) else "unknown"
    class_name = _meta_value(track_meta_row, "class", "unknown")
    num_frames = int(_meta_value(track_meta_row, "numFrames", len(checked))) if len(checked) else 0
    num_abnormal = int(checked["abnormal_frame"].sum()) if len(checked) else 0
    abnormal_ratio = float(num_abnormal) / float(len(checked)) if len(checked) else 0.0

    return {
        "trackId": track_id,
        "class": class_name,
        "numFrames": num_frames,
        "max_speed": _safe_max(checked["speed_xy"]),
        "max_acc_xy": _safe_max(checked["acc_xy"]),
        "max_abs_lon_acc": _safe_max_abs(checked["lonAcceleration"]),
        "max_abs_lat_acc": _safe_max_abs(checked["latAcceleration"]),
        "max_abs_delta_speed": _safe_max_abs(checked["delta_speed"]),
        "max_abs_delta_heading": _safe_max_abs(checked["delta_heading"]),
        "max_abs_speed_error": _safe_max_abs(checked["speed_error"]),
        "max_abs_acc_error": _safe_max_abs(checked["acc_error"]),
        "max_abs_heading_motion_error": _safe_max_abs(checked["angle_error"]),
        "max_velocity_derivative_error": _safe_max(checked["velocity_derivative_error"]),
        "p95_velocity_derivative_error": _safe_p95(checked["velocity_derivative_error"]),
        "velocity_derivative_over_20pct_ratio": _safe_ratio(checked["velocity_derivative_over_20pct"]),
        "max_acceleration_derivative_error": _safe_max(checked["acceleration_derivative_error"]),
        "p95_acceleration_derivative_error": _safe_p95(checked["acceleration_derivative_error"]),
        "acceleration_derivative_over_20pct_ratio": _safe_ratio(checked["acceleration_derivative_over_20pct"]),
        "num_abnormal_frames": num_abnormal,
        "abnormal_ratio": abnormal_ratio,
        "nan_lonVelocity": int(checked["lonVelocity"].isnull().sum()),
        "nan_latVelocity": int(checked["latVelocity"].isnull().sum()),
        "nan_lonAcceleration": int(checked["lonAcceleration"].isnull().sum()),
        "nan_latAcceleration": int(checked["latAcceleration"].isnull().sum()),
    }


def _format_number(value, digits=4):
    if value is None:
        return "nan"
    try:
        if math.isnan(float(value)):
            return "nan"
    except Exception:
        return str(value)
    return ("%." + str(digits) + "f") % float(value)


def _print_track_summary(summary):
    keys = [
        "trackId",
        "class",
        "numFrames",
        "max_speed",
        "max_acc_xy",
        "max_abs_lon_acc",
        "max_abs_lat_acc",
        "max_abs_delta_speed",
        "max_abs_delta_heading",
        "max_abs_speed_error",
        "max_abs_acc_error",
        "max_abs_heading_motion_error",
        "max_velocity_derivative_error",
        "p95_velocity_derivative_error",
        "velocity_derivative_over_20pct_ratio",
        "max_acceleration_derivative_error",
        "p95_acceleration_derivative_error",
        "acceleration_derivative_over_20pct_ratio",
        "num_abnormal_frames",
        "abnormal_ratio",
        "nan_lonVelocity",
        "nan_latVelocity",
        "nan_lonAcceleration",
        "nan_latAcceleration",
    ]
    print("track 统计:")
    for key in keys:
        value = summary.get(key)
        if isinstance(value, float):
            value = _format_number(value)
        print("  %s: %s" % (key, value))


def _scatter_abnormal_xy(ax, checked):
    bad = checked["abnormal_frame"] & pd.Series(_finite_xy_mask(checked), index=checked.index)
    if bad.any():
        ax.scatter(checked.loc[bad, "xCenter"], checked.loc[bad, "yCenter"], c="red", marker="x", s=30, label="abnormal")


def _mark_abnormal_points(ax, checked, value_columns, mask_columns):
    mask = _series_false(checked.index)
    for column in mask_columns:
        if column in checked.columns:
            mask = mask | checked[column]
    if not mask.any():
        return
    for column in value_columns:
        if column in checked.columns:
            ax.scatter(checked.loc[mask, "frame"], checked.loc[mask, column], c="red", s=12, zorder=4)


def _plot_y_equals_x(ax, x_values, y_values):
    values = pd.concat([pd.Series(x_values), pd.Series(y_values)], ignore_index=True)
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return
    low = float(values.min())
    high = float(values.max())
    if low == high:
        low -= 1.0
        high += 1.0
    ax.plot([low, high], [low, high], "k--", linewidth=1, label="y=x")


def _apply_axis_spacing(axes):
    for ax in axes:
        ax.title.set_fontsize(AXIS_TITLE_FONT_SIZE)
        ax.xaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
        ax.yaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
        try:
            ax.title.set_pad(AXIS_TITLE_PAD)
        except AttributeError:
            ax.title.set_position((0.5, 1.04))
        ax.xaxis.labelpad = AXIS_LABEL_PAD
        ax.yaxis.labelpad = AXIS_LABEL_PAD
        ax.tick_params(axis="both", labelsize=AXIS_TICK_FONT_SIZE, pad=3)


def _draw_summary_columns(ax, text, fontsize=SUMMARY_FONT_SIZE):
    lines = []
    for line in text.splitlines():
        if not line:
            lines.append(line)
            continue
        wrapped = textwrap.wrap(line, width=SUMMARY_COLUMN_WIDTH, subsequent_indent="  ")
        lines.extend(wrapped or [line])
    split_index = int(math.ceil(len(lines) / 2.0))
    left_text = "\n".join(lines[:split_index])
    right_text = "\n".join(lines[split_index:])
    ax.text(0.00, 0.98, left_text, va="top", ha="left", fontsize=fontsize, family="monospace", linespacing=SUMMARY_LINE_SPACING)
    ax.text(SUMMARY_RIGHT_COLUMN_X, 0.98, right_text, va="top", ha="left", fontsize=fontsize, family="monospace", linespacing=SUMMARY_LINE_SPACING)


def _summary_text(summary, checked):
    abnormal_counts = ["%s: %d" % (column.replace("abnormal_", ""), int(checked[column].sum())) for column in ABNORMAL_COLUMNS]
    lines = [
        "trackId: %s" % summary["trackId"],
        "class: %s" % summary["class"],
        "numFrames: %s" % summary["numFrames"],
        "max_speed: %s" % _format_number(summary["max_speed"]),
        "max_acc_xy: %s" % _format_number(summary["max_acc_xy"]),
        "max_abs_speed_error: %s" % _format_number(summary["max_abs_speed_error"]),
        "max_abs_acc_error: %s" % _format_number(summary["max_abs_acc_error"]),
        "max_abs_heading_motion_error: %s" % _format_number(summary["max_abs_heading_motion_error"]),
        "max_vel_deriv_error: %s" % _format_number(summary["max_velocity_derivative_error"]),
        "p95_vel_deriv_error: %s" % _format_number(summary["p95_velocity_derivative_error"]),
        "vel_deriv_over20_ratio: %s" % _format_number(summary["velocity_derivative_over_20pct_ratio"]),
        "max_acc_deriv_error: %s" % _format_number(summary["max_acceleration_derivative_error"]),
        "p95_acc_deriv_error: %s" % _format_number(summary["p95_acceleration_derivative_error"]),
        "acc_deriv_over20_ratio: %s" % _format_number(summary["acceleration_derivative_over_20pct_ratio"]),
        "num_abnormal_frames: %s" % summary["num_abnormal_frames"],
        "abnormal_ratio: %s" % _format_number(summary["abnormal_ratio"]),
        "NaN lon/lat vel: %s/%s" % (summary["nan_lonVelocity"], summary["nan_latVelocity"]),
        "NaN lon/lat acc: %s/%s" % (summary["nan_lonAcceleration"], summary["nan_latAcceleration"]),
        "",
        "abnormal counts:",
    ]
    return "\n".join(lines + abnormal_counts)


def plot_track_check(track_df, track_meta_row, folder_name, frame_rate=29.97):
    """Plot all checks for one track in a single matplotlib figure."""
    checked = compute_track_kinematic_checks(track_df, frame_rate)
    summary = summarize_track(checked, track_meta_row)
    _print_track_summary(summary)

    fig, axes = plt.subplots(4, 3, figsize=TRACK_FIGSIZE, gridspec_kw=dict(GRID_KW))
    axes = axes.ravel()
    title = "folder=%s, trackId=%s, class=%s, numFrames=%s" % (
        folder_name,
        summary["trackId"],
        summary["class"],
        summary["numFrames"],
    )
    fig.suptitle(title, y=0.975, fontsize=FIGURE_TITLE_FONT_SIZE)
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass

    finite_xy = _finite_xy_mask(checked)

    ax = axes[0]
    ax.plot(checked["xCenter"], checked["yCenter"], "-", linewidth=1, label="trajectory")
    if finite_xy.any():
        first_idx = np.where(finite_xy)[0][0]
        last_idx = np.where(finite_xy)[0][-1]
        ax.scatter(checked.loc[first_idx, "xCenter"], checked.loc[first_idx, "yCenter"], c="green", s=45, label="start")
        ax.scatter(checked.loc[last_idx, "xCenter"], checked.loc[last_idx, "yCenter"], c="red", s=45, label="end")
    _scatter_abnormal_xy(ax, checked)
    ax.set_title("XY trajectory")
    ax.set_xlabel("xCenter")
    ax.set_ylabel("yCenter")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)

    ax = axes[1]
    sc = ax.scatter(checked["xCenter"], checked["yCenter"], c=checked["speed_xy"], s=12)
    ax.set_title("Trajectory colored by speed")
    ax.set_xlabel("xCenter")
    ax.set_ylabel("yCenter")
    ax.axis("equal")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    sc = ax.scatter(checked["xCenter"], checked["yCenter"], c=checked["acc_xy"], s=12)
    ax.set_title("Trajectory colored by acceleration")
    ax.set_xlabel("xCenter")
    ax.set_ylabel("yCenter")
    ax.axis("equal")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[3]
    ax.plot(checked["frame"], checked["heading"], label="heading")
    ax.plot(checked["frame"], np.degrees(np.unwrap(np.radians(checked["heading"]))), label="unwrap heading", linewidth=1)
    _mark_abnormal_points(ax, checked, ["heading"], ["abnormal_delta_heading"])
    ax.set_title("Heading over frames")
    ax.set_xlabel("frame")
    ax.set_ylabel("heading (deg)")
    ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)

    ax = axes[4]
    for column in ["xVelocity", "yVelocity", "lonVelocity", "latVelocity"]:
        ax.plot(checked["frame"], checked[column], label=column, linewidth=1)
    _mark_abnormal_points(
        ax,
        checked,
        ["xVelocity", "yVelocity", "lonVelocity", "latVelocity"],
        ["abnormal_speed", "abnormal_delta_speed", "abnormal_delta_lonVelocity", "abnormal_delta_latVelocity"],
    )
    ax.set_title("Velocity components")
    ax.set_xlabel("frame")
    ax.set_ylabel("velocity (m/s)")
    ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)

    ax = axes[5]
    for column in ["xAcceleration", "yAcceleration", "lonAcceleration", "latAcceleration"]:
        ax.plot(checked["frame"], checked[column], label=column, linewidth=1)
    _mark_abnormal_points(
        ax,
        checked,
        ["xAcceleration", "yAcceleration", "lonAcceleration", "latAcceleration"],
        ["abnormal_x_acc", "abnormal_y_acc", "abnormal_lon_acc", "abnormal_lat_acc"],
    )
    ax.set_title("Acceleration components")
    ax.set_xlabel("frame")
    ax.set_ylabel("acceleration (m/s^2)")
    ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)

    ax = axes[6]
    ax.plot(checked["frame"], checked["speed_xy"], label="speed_xy")
    ax.plot(checked["frame"], checked["speed_lonlat"], label="speed_lonlat")
    _mark_abnormal_points(ax, checked, ["speed_xy", "speed_lonlat"], ["abnormal_speed_error"])
    ax.text(0.02, 0.95, "max_abs_speed_error=%s" % _format_number(summary["max_abs_speed_error"]), transform=ax.transAxes, va="top", fontsize=ANNOTATION_FONT_SIZE)
    ax.set_title("Speed consistency")
    ax.set_xlabel("frame")
    ax.set_ylabel("speed (m/s)")
    ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)

    ax = axes[7]
    ax.plot(checked["frame"], checked["acc_xy"], label="acc_xy")
    ax.plot(checked["frame"], checked["acc_lonlat"], label="acc_lonlat")
    _mark_abnormal_points(ax, checked, ["acc_xy", "acc_lonlat"], ["abnormal_acc_error"])
    ax.text(0.02, 0.95, "max_abs_acc_error=%s" % _format_number(summary["max_abs_acc_error"]), transform=ax.transAxes, va="top", fontsize=ANNOTATION_FONT_SIZE)
    ax.set_title("Acceleration consistency")
    ax.set_xlabel("frame")
    ax.set_ylabel("acceleration (m/s^2)")
    ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)

    ax = axes[8]
    ax.plot(checked["frame"], checked["angle_error"], label="angle_error")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    _mark_abnormal_points(ax, checked, ["angle_error"], ["abnormal_angle_error"])
    ax.set_title("Motion heading vs heading error")
    ax.set_xlabel("frame")
    ax.set_ylabel("angle error (deg)")
    ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)

    ax = axes[9]
    for column in ["delta_speed", "delta_lonVelocity", "delta_latVelocity", "delta_heading"]:
        ax.plot(checked["frame"], checked[column], label=column, linewidth=1)
    _mark_abnormal_points(
        ax,
        checked,
        ["delta_speed", "delta_lonVelocity", "delta_latVelocity", "delta_heading"],
        ["abnormal_delta_speed", "abnormal_delta_lonVelocity", "abnormal_delta_latVelocity", "abnormal_delta_heading"],
    )
    ax.set_title("Frame-to-frame jumps")
    ax.set_xlabel("frame")
    ax.set_ylabel("delta")
    ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)

    ax = axes[10]
    counts = [int(checked[column].sum()) for column in ABNORMAL_COLUMNS]
    labels = [column.replace("abnormal_", "") for column in ABNORMAL_COLUMNS]
    ax.bar(range(len(counts)), counts)
    ax.set_title("Abnormal counts")
    ax.set_ylabel("frames")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=58, ha="right", fontsize=ABNORMAL_LABEL_FONT_SIZE)
    ax.margins(x=0.02)

    ax = axes[11]
    ax.axis("off")
    _draw_summary_columns(ax, _summary_text(summary, checked))
    ax.set_title("Track abnormal summary")

    _apply_axis_spacing(axes)
    fig.subplots_adjust(**FIGURE_ADJUST)
    plt.show()


def _sorted_track_ids(tracks_df):
    ids = pd.to_numeric(tracks_df["trackId"], errors="coerce").dropna().astype(int).unique()
    return sorted(ids.tolist())


def interactive_track_loop(tracks_df, tracks_meta_df, folder_name, track_id=None, frame_rate=29.97):
    """Show one track or iterate over all tracks in trackId order."""
    if track_id is None:
        track_ids = _sorted_track_ids(tracks_df)
    else:
        track_ids = [int(track_id)]

    for current_track_id in track_ids:
        track_rows = tracks_df.loc[pd.to_numeric(tracks_df["trackId"], errors="coerce") == int(current_track_id)]
        if track_rows.empty:
            print("在 folder=%s 中没有找到 trackId=%s。" % (folder_name, current_track_id))
            continue
        meta_row = _find_meta_row(tracks_meta_df, current_track_id)
        class_name = _meta_value(meta_row, "class", "unknown")
        num_frames = int(_meta_value(meta_row, "numFrames", len(track_rows)))
        print("正在显示 folder=%s, trackId=%s, class=%s, numFrames=%s" % (folder_name, current_track_id, class_name, num_frames))
        if track_id is None:
            print("关闭图窗口后将自动显示下一个 track。")
        plot_track_check(track_rows, meta_row, folder_name, frame_rate)


def _compute_all_checks(tracks_df, frame_rate=29.97):
    checked_tracks = []
    for _, group in tracks_df.groupby("trackId", sort=True):
        checked_tracks.append(compute_track_kinematic_checks(group, frame_rate))
    if not checked_tracks:
        return pd.DataFrame()
    return pd.concat(checked_tracks, ignore_index=True)


def _add_class_column(checked_df, tracks_meta_df):
    checked = checked_df.copy()
    if tracks_meta_df is None or tracks_meta_df.empty or "trackId" not in tracks_meta_df.columns or "class" not in tracks_meta_df.columns:
        checked["class"] = "unknown"
        return checked
    class_map = {}
    for _, row in tracks_meta_df.iterrows():
        if pd.notnull(row.get("trackId")):
            class_map[int(row["trackId"])] = row.get("class", "unknown")
    checked["class"] = [class_map.get(int(track_id), "unknown") if pd.notnull(track_id) else "unknown" for track_id in checked["trackId"]]
    return checked


def _hist(ax, series, title, xlabel, bins=80):
    values = pd.to_numeric(series, errors="coerce").dropna()
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")


def _boxplot_by_class(ax, checked, value_column, title, ylabel):
    if "class" not in checked.columns:
        ax.text(0.0, 0.5, "tracksMeta 中没有 class 字段", transform=ax.transAxes, fontsize=ANNOTATION_FONT_SIZE)
        ax.set_title(title)
        return
    classes = []
    values = []
    for class_name in sorted(checked["class"].dropna().unique()):
        class_values = pd.to_numeric(checked.loc[checked["class"] == class_name, value_column], errors="coerce").dropna()
        if not class_values.empty:
            classes.append(str(class_name))
            values.append(class_values.values)
    if values:
        ax.boxplot(values, labels=classes, showfliers=False)
        ax.tick_params(axis="x", labelrotation=45)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def _quality_terminal_stats(quality_report):
    quality = quality_report.get("quality", {}) if isinstance(quality_report, dict) else {}
    terminal = quality.get("terminal_heading_protection", {}) if isinstance(quality, dict) else {}
    return {
        "terminal_heading_protected_frame_count": int(terminal.get("protected_frame_count", 0) or 0),
        "terminal_heading_protected_track_count": int(terminal.get("protected_track_count", 0) or 0),
    }


def _global_summary_text(checked, folder_name, quality_report=None):
    terminal_stats = _quality_terminal_stats(quality_report or {})
    lines = [
        "folder: %s" % folder_name,
        "numTracks: %s" % len(_sorted_track_ids(checked)) if not checked.empty else "numTracks: 0",
        "numRows: %s" % len(checked),
        "max_speed: %s" % _format_number(_safe_max(checked["speed_xy"])),
        "max_acc_xy: %s" % _format_number(_safe_max(checked["acc_xy"])),
        "max_abs_speed_error: %s" % _format_number(_safe_max_abs(checked["speed_error"])),
        "max_abs_acc_error: %s" % _format_number(_safe_max_abs(checked["acc_error"])),
        "max_abs_heading_motion_error: %s" % _format_number(_safe_max_abs(checked["angle_error"])),
        "max_velocity_derivative_error: %s" % _format_number(_safe_max(checked["velocity_derivative_error"])),
        "p95_velocity_derivative_error: %s" % _format_number(_safe_p95(checked["velocity_derivative_error"])),
        "velocity_deriv_over20_ratio: %s" % _format_number(_safe_ratio(checked["velocity_derivative_over_20pct"])),
        "max_acceleration_derivative_error: %s" % _format_number(_safe_max(checked["acceleration_derivative_error"])),
        "p95_acceleration_derivative_error: %s" % _format_number(_safe_p95(checked["acceleration_derivative_error"])),
        "acceleration_deriv_over20_ratio: %s" % _format_number(_safe_ratio(checked["acceleration_derivative_over_20pct"])),
        "terminal_heading_protected_frames: %s" % terminal_stats["terminal_heading_protected_frame_count"],
        "terminal_heading_protected_tracks: %s" % terminal_stats["terminal_heading_protected_track_count"],
        "num_abnormal_rows: %s" % int(checked["abnormal_frame"].sum()),
        "abnormal_ratio: %s" % _format_number(float(checked["abnormal_frame"].sum()) / float(len(checked)) if len(checked) else 0.0),
        "",
        "abnormal counts:",
    ]
    lines.extend(["%s: %d" % (column.replace("abnormal_", ""), int(checked[column].sum())) for column in ABNORMAL_COLUMNS])
    return "\n".join(lines)


def plot_summary(tracks_df, tracks_meta_df, folder_name, frame_rate=29.97, quality_report=None):
    """Plot folder-level summary checks in one matplotlib figure."""
    checked = _add_class_column(_compute_all_checks(tracks_df, frame_rate), tracks_meta_df)
    if checked.empty:
        print("folder=%s 中没有可统计的轨迹数据。" % folder_name)
        return

    fig, axes = plt.subplots(4, 3, figsize=SUMMARY_FIGSIZE, gridspec_kw=dict(GRID_KW))
    axes = axes.ravel()
    title = "summary folder=%s" % folder_name
    fig.suptitle(title, y=0.975, fontsize=FIGURE_TITLE_FONT_SIZE)
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass

    _hist(axes[0], checked["speed_xy"], "speed_xy distribution", "speed_xy (m/s)")
    _hist(axes[1], checked["acc_xy"], "acc_xy distribution", "acc_xy (m/s^2)")
    _hist(axes[2], checked["lonVelocity"], "lonVelocity distribution", "lonVelocity (m/s)")
    _hist(axes[3], checked["latVelocity"], "latVelocity distribution", "latVelocity (m/s)")
    _hist(axes[4], checked["lonAcceleration"], "lonAcceleration distribution", "lonAcceleration (m/s^2)")
    _hist(axes[5], checked["latAcceleration"], "latAcceleration distribution", "latAcceleration (m/s^2)")
    _hist(axes[6], checked["delta_heading"], "delta_heading distribution", "delta_heading (deg)")

    axes[7].scatter(checked["speed_xy"], checked["speed_lonlat"], s=5, alpha=0.4)
    _plot_y_equals_x(axes[7], checked["speed_xy"], checked["speed_lonlat"])
    axes[7].set_title("speed_xy vs speed_lonlat")
    axes[7].set_xlabel("speed_xy")
    axes[7].set_ylabel("speed_lonlat")
    axes[7].legend(loc="best", fontsize=LEGEND_FONT_SIZE)

    axes[8].scatter(checked["acc_xy"], checked["acc_lonlat"], s=5, alpha=0.4)
    _plot_y_equals_x(axes[8], checked["acc_xy"], checked["acc_lonlat"])
    axes[8].set_title("acc_xy vs acc_lonlat")
    axes[8].set_xlabel("acc_xy")
    axes[8].set_ylabel("acc_lonlat")
    axes[8].legend(loc="best", fontsize=LEGEND_FONT_SIZE)

    _boxplot_by_class(axes[9], checked, "speed_xy", "speed_xy by class", "speed_xy (m/s)")
    _boxplot_by_class(axes[10], checked, "acc_xy", "acc_xy by class", "acc_xy (m/s^2)")

    axes[11].axis("off")
    _draw_summary_columns(axes[11], _global_summary_text(checked, folder_name, quality_report))
    axes[11].set_title("Global abnormal summary")

    print(_global_summary_text(checked, folder_name, quality_report))
    _apply_axis_spacing(axes)
    fig.subplots_adjust(**FIGURE_ADJUST)
    plt.show()


def _list_available_folders(data_root):
    root = Path(data_root)
    if not root.exists():
        print("data_root 不存在: %s" % root)
        return
    folders = [child.name for child in sorted(root.iterdir()) if child.is_dir()]
    print("可用子文件夹:")
    for index, folder in enumerate(folders):
        print("[%d] %s" % (index, folder))
    print("")
    print("请使用 --folder 指定要查看的数据集，例如:")
    print('python trajectory_check_visualizer.py --data_root "%s" --folder cao_qiao_001' % root)


def main():
    parser = argparse.ArgumentParser(description="Check and visualize OpenVTER Final Data trajectory kinematic fields.")
    parser.add_argument("--data_root", required=True, help="Final Data root folder.")
    parser.add_argument("--folder", default=None, help="Dataset subfolder under data_root.")
    parser.add_argument("--track_id", type=int, default=None, help="Only show the specified trackId in the selected folder.")
    parser.add_argument("--summary", action="store_true", help="Show folder-level summary plots instead of per-track plots.")
    args = parser.parse_args()

    if not args.folder:
        _list_available_folders(args.data_root)
        return

    tracks_df, tracks_meta_df, recording_meta_df = load_dataset(args.data_root, args.folder)
    frame_rate = _frame_rate(recording_meta_df)
    quality_report = load_quality_report(args.data_root, args.folder)
    if args.summary:
        plot_summary(tracks_df, tracks_meta_df, args.folder, frame_rate, quality_report)
    else:
        interactive_track_loop(tracks_df, tracks_meta_df, args.folder, args.track_id, frame_rate)


if __name__ == "__main__":
    main()
