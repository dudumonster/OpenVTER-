#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert det_bbox_result_*.pkl into a structured CSV/Excel similar to HighD.

Usage:
    python using/output_result.py \
        --pkl data1/UAV_Videos/track_small_raw/output/track_small_raw/det_bbox_result_track_small_raw.pkl
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PIXEL_COORD_NAMES = [
    "x1_px", "y1_px", "x2_px", "y2_px",
    "x3_px", "y3_px", "x4_px", "y4_px"
]
WORLD_COORD_NAMES = [
    "x1_world", "y1_world", "x2_world", "y2_world",
    "x3_world", "y3_world", "x4_world", "y4_world"
]
CATEGORY_NAMES = ["car", "truck", "bus", "freight_car", "van"]


def parse_args():
    parser = argparse.ArgumentParser(description="Export det_bbox_result_*.pkl to csv/xlsx.")
    parser.add_argument("--pkl", required=True, help="Path to det_bbox_result_*.pkl")
    parser.add_argument("--output-dir", default=None, help="Directory to store csv/xlsx (default: <pkl_dir>/saving)")
    return parser.parse_args()


def format_entry(entry):
    if len(entry) == 3:
        frame_idx, output_idx, arr = entry
        frame_time = None
    else:
        frame_idx, output_idx, arr, frame_time = entry
    return frame_idx, output_idx, arr, frame_time


def infer_columns(num_cols):
    base = PIXEL_COORD_NAMES + ["score", "category", "track_id"]
    if num_cols == 11:
        return base
    if num_cols == 19:
        return base + WORLD_COORD_NAMES
    if num_cols == 20:
        return base + WORLD_COORD_NAMES + ["lane_id"]
    raise ValueError(f"Unexpected detection column size: {num_cols}")


def compute_centers(df):
    df["xCenter_px"] = (df["x1_px"] + df["x3_px"]) / 2
    df["yCenter_px"] = (df["y1_px"] + df["y3_px"]) / 2
    if all(col in df.columns for col in WORLD_COORD_NAMES):
        df["xCenter_world"] = (df["x1_world"] + df["x3_world"]) / 2
        df["yCenter_world"] = (df["y1_world"] + df["y3_world"]) / 2
    return df


def add_track_lifetime(df):
    df["track_id"] = df["track_id"].astype(int)
    df.sort_values(["track_id", "frame_index"], inplace=True)
    df["track_lifetime"] = df.groupby("track_id").cumcount()
    return df


def add_heading_velocity(df, fps):
    if not all(col in df.columns for col in WORLD_COORD_NAMES):
        # 没有世界坐标就跳过
        return df
    dt = 1.0 / fps if fps and fps > 0 else 1.0

    # 中心点
    df["xCenter_world"] = (df["x1_world"] + df["x3_world"]) / 2
    df["yCenter_world"] = (df["y1_world"] + df["y3_world"]) / 2

    # 全局速度/加速度
    df["xVelocity"] = df.groupby("track_id")["xCenter_world"].diff() / dt
    df["yVelocity"] = df.groupby("track_id")["yCenter_world"].diff() / dt
    df["xAcceleration"] = df.groupby("track_id")["xVelocity"].diff() / dt
    df["yAcceleration"] = df.groupby("track_id")["yVelocity"].diff() / dt

    # 航向角：仅用速度方向计算
    heading_rad = np.arctan2(df["yVelocity"], df["xVelocity"])
    df["heading"] = np.degrees(heading_rad)

    # 纵/横向速度：将全局速度旋转到车辆朝向系，前向为 +lon，右侧为 +lat
    cos_h = np.cos(heading_rad)
    sin_h = np.sin(heading_rad)
    df["lonVelocity"] = df["xVelocity"] * cos_h + df["yVelocity"] * sin_h
    df["latVelocity"] = -df["xVelocity"] * sin_h + df["yVelocity"] * cos_h
    df["lonAcceleration"] = df["xAcceleration"] * cos_h + df["yAcceleration"] * sin_h
    df["latAcceleration"] = -df["xAcceleration"] * sin_h + df["yAcceleration"] * cos_h
    return df


def add_category_name(df):
    if "category" not in df.columns:
        return df
    cat_map = {i: name for i, name in enumerate(CATEGORY_NAMES)}
    df["category_name"] = df["category"].map(cat_map)
    return df


def main():
    args = parse_args()
    pkl_path = Path(args.pkl)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    traj_info = data["traj_info"]
    fps = data.get("output_info", {}).get("output_fps", None)
    if not traj_info:
        raise RuntimeError("traj_info is empty.")

    sample_entry = traj_info[0]
    sample_cols = sample_entry[2].shape[1] if sample_entry[2].ndim == 2 else sample_entry[2].shape[0]
    col_names = infer_columns(sample_cols)

    base_name = pkl_path.stem
    output_dir = Path(args.output_dir) if args.output_dir else pkl_path.parent / "saving"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for entry in traj_info:
        frame_idx, output_idx, arr, frame_time = format_entry(entry)
        if arr is None or len(arr) == 0:
            continue
        for row in arr:
            record = {
                "recording_id": base_name,
                "frame_index": frame_idx,
                "output_frame": output_idx,
                "frame_time": frame_time,
            }
            for name, value in zip(col_names, row):
                record[name] = value
            records.append(record)

    if not records:
        raise RuntimeError("No detection records found.")

    df = pd.DataFrame(records)
    df = compute_centers(df)
    df = add_track_lifetime(df)
    df = add_category_name(df)
    df = add_heading_velocity(df, fps)

    csv_path = output_dir / f"{base_name}.csv"
    xlsx_path = output_dir / f"{base_name}.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    print(f"Exported {len(df)} rows to:\n - {csv_path}\n - {xlsx_path}")


if __name__ == "__main__":
    main()
