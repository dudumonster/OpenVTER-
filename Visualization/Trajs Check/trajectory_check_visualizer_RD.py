#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RounD sample adapter for the OpenVTER trajectory check visualizer.

This script keeps the same check and plotting logic as trajectory_check_visualizer.py,
but reads the public RounD sample CSVs directly from one folder.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import trajectory_check_visualizer as base


DEFAULT_ROUND_ROOT = Path(r"D:\工作安排\工作汇报整理\2026.5.28 数据集字段探讨\RounD示例")


def find_csv_files(folder_path):
    """Find recordingMeta, tracksMeta, and tracks CSV files in a RounD sample folder."""
    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError("RounD 数据目录不存在: %s" % folder_path)

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
        raise FileNotFoundError("在 %s 中没有找到 RounD CSV 文件: %s" % (folder_path, ", ".join(missing)))
    return found


def load_dataset(data_root, folder=None):
    """Load RounD tracks.csv and tracksMeta.csv from data_root or data_root/folder."""
    folder_path = Path(data_root) / folder if folder else Path(data_root)
    csv_files = find_csv_files(folder_path)
    with csv_files["tracks"].open("r", newline="", encoding="utf-8") as fh:
        tracks_df = pd.read_csv(fh)
    with csv_files["tracksMeta"].open("r", newline="", encoding="utf-8") as fh:
        tracks_meta_df = pd.read_csv(fh)

    missing_tracks = [column for column in base.REQUIRED_TRACK_COLUMNS if column not in tracks_df.columns]
    if missing_tracks:
        raise ValueError("tracks.csv 缺少必要字段: %s" % ", ".join(missing_tracks))

    missing_meta_hints = [column for column in base.TRACK_META_HINT_COLUMNS if column not in tracks_meta_df.columns]
    if missing_meta_hints:
        print("提示: tracksMeta.csv 缺少字段: %s；对应信息将显示为 unknown 或从 tracks.csv 推断。" % ", ".join(missing_meta_hints))

    for column in base.NUMERIC_TRACK_COLUMNS:
        tracks_df[column] = pd.to_numeric(tracks_df[column], errors="coerce")
    if "trackId" in tracks_meta_df.columns:
        tracks_meta_df["trackId"] = pd.to_numeric(tracks_meta_df["trackId"], errors="coerce")
    if "numFrames" in tracks_meta_df.columns:
        tracks_meta_df["numFrames"] = pd.to_numeric(tracks_meta_df["numFrames"], errors="coerce")

    return tracks_df, tracks_meta_df


def _display_name(data_root, folder):
    return folder if folder else Path(data_root).name


def _list_available_folders(data_root):
    root = Path(data_root)
    if not root.exists():
        print("data_root 不存在: %s" % root)
        return
    folders = [child.name for child in sorted(root.iterdir()) if child.is_dir()]
    print("当前目录没有直接找到 RounD 三个 CSV 文件。可用子文件夹:")
    for index, folder in enumerate(folders):
        print("[%d] %s" % (index, folder))
    print("")
    print("请使用 --folder 指定子文件夹，或将 recordingMeta/tracksMeta/tracks CSV 放在 data_root 中。")


def main():
    parser = argparse.ArgumentParser(description="Check and visualize public RounD sample trajectory CSVs.")
    parser.add_argument("--data_root", default=str(DEFAULT_ROUND_ROOT), help="Folder containing RounD recordingMeta/tracksMeta/tracks CSVs.")
    parser.add_argument("--folder", default=None, help="Optional subfolder under data_root.")
    parser.add_argument("--track_id", type=int, default=None, help="Only show the specified trackId.")
    parser.add_argument("--summary", action="store_true", help="Show folder-level summary plots instead of per-track plots.")
    args = parser.parse_args()

    data_path = Path(args.data_root) / args.folder if args.folder else Path(args.data_root)
    try:
        tracks_df, tracks_meta_df = load_dataset(args.data_root, args.folder)
    except FileNotFoundError:
        if args.folder:
            raise
        _list_available_folders(args.data_root)
        return

    folder_name = _display_name(data_path, None)
    if args.summary:
        base.plot_summary(tracks_df, tracks_meta_df, folder_name)
    else:
        base.interactive_track_loop(tracks_df, tracks_meta_df, folder_name, args.track_id)


if __name__ == "__main__":
    main()
