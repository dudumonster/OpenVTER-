#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostics for missing frames and motion spikes in det_bbox_result_*.pkl.

Usage:
    python using/diagnose_track_gaps.py --pkl path/to/det_bbox_result_*.pkl
    python using/diagnose_track_gaps.py --pkl path/to/det_bbox_result_*.pkl --output-dir out
"""
import argparse
import csv
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose track gaps and motion spikes.")
    parser.add_argument("--pkl", required=True, help="Path to det_bbox_result_*.pkl")
    parser.add_argument("--output-dir", default=None, help="Optional output dir for CSV summary")
    parser.add_argument("--top", type=int, default=10, help="Top-N rows to print per section")
    parser.add_argument("--min-frames", type=int, default=10, help="Min frames per track to report")
    parser.add_argument("--gap-thresh", type=int, default=2, help="Gap threshold for missing-frame stats")
    parser.add_argument("--static-speed", type=float, default=0.2,
                        help="Speed threshold to flag near-static tracks (world units)")
    return parser.parse_args()


def format_entry(entry):
    if len(entry) == 3:
        frame_idx, output_idx, arr = entry
        frame_time = None
    else:
        frame_idx, output_idx, arr, frame_time = entry
    return frame_idx, output_idx, arr, frame_time


def extract_centers(row):
    px_pts = row[:8].reshape(4, 2)
    center_px = px_pts.mean(axis=0)
    center_world = None
    if row.shape[0] >= 19:
        world_pts = row[11:19].reshape(4, 2)
        center_world = world_pts.mean(axis=0)
    return center_px, center_world


def summarize_tracks(tracks, fps, args):
    summaries = []
    for track_id, items in tracks.items():
        if len(items) < args.min_frames:
            continue
        items.sort(key=lambda x: x[0])
        frames = np.array([x[0] for x in items], dtype=np.int64)
        centers_px = np.array([x[1] for x in items], dtype=np.float32)
        centers_world = [x[2] for x in items if x[2] is not None]
        use_world = len(centers_world) == len(items)
        centers = np.array(centers_world, dtype=np.float32) if use_world else centers_px

        frame_span = int(frames[-1] - frames[0] + 1)
        missing = max(0, frame_span - len(frames))
        gaps = np.diff(frames) if len(frames) > 1 else np.array([], dtype=np.int64)
        gap_count = int((gaps >= args.gap_thresh).sum()) if gaps.size else 0
        max_gap = int(gaps.max()) if gaps.size else 0

        if len(centers) > 1:
            dxy = np.diff(centers, axis=0)
            dist = np.sqrt((dxy ** 2).sum(axis=1))
            if fps and fps > 0:
                dt = gaps / float(fps)
                dt[dt == 0] = 1.0 / float(fps)
                speed_correct = dist / dt
                speed_naive = dist * float(fps)
            else:
                speed_correct = dist
                speed_naive = dist
            max_speed = float(speed_correct.max()) if speed_correct.size else 0.0
            p95_speed = float(np.percentile(speed_correct, 95)) if speed_correct.size else 0.0
            med_speed = float(np.median(speed_correct)) if speed_correct.size else 0.0
            max_speed_naive = float(speed_naive.max()) if speed_naive.size else 0.0
        else:
            max_speed = p95_speed = med_speed = max_speed_naive = 0.0

        summaries.append({
            "track_id": int(track_id),
            "frames": int(len(frames)),
            "span": frame_span,
            "missing": missing,
            "missing_ratio": missing / float(frame_span) if frame_span else 0.0,
            "gap_count": gap_count,
            "max_gap": max_gap,
            "use_world": use_world,
            "median_speed": med_speed,
            "p95_speed": p95_speed,
            "max_speed": max_speed,
            "max_speed_naive": max_speed_naive,
        })
    return summaries


def print_section(title, rows, top):
    print(f"\n== {title} ==")
    for row in rows[:top]:
        print(row)


def main():
    args = parse_args()
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    traj_info = data.get("traj_info", [])
    fps = data.get("output_info", {}).get("output_fps", None)

    tracks = defaultdict(list)
    total_frames = 0
    for entry in traj_info:
        frame_idx, _, arr, _ = format_entry(entry)
        total_frames += 1
        if arr is None or len(arr) == 0:
            continue
        for row in arr:
            row = np.asarray(row, dtype=np.float32)
            if row.shape[0] <= 10:
                continue
            track_id = int(row[10])
            center_px, center_world = extract_centers(row)
            tracks[track_id].append((int(frame_idx), center_px, center_world))

    summaries = summarize_tracks(tracks, fps, args)
    summaries.sort(key=lambda x: x["track_id"])

    print(f"fps={fps} total_frames={total_frames} tracks={len(tracks)}")
    if not summaries:
        print("No tracks matched the min-frames filter.")
        return

    by_missing = sorted(summaries, key=lambda x: x["missing_ratio"], reverse=True)
    by_gap = sorted(summaries, key=lambda x: x["max_gap"], reverse=True)
    by_speed = sorted(summaries, key=lambda x: x["max_speed"], reverse=True)

    print_section("Top Missing Ratio", by_missing, args.top)
    print_section("Top Max Gap", by_gap, args.top)
    print_section("Top Max Speed", by_speed, args.top)

    static_candidates = [
        s for s in summaries
        if s["use_world"] and s["median_speed"] <= args.static_speed and s["max_speed"] > args.static_speed * 5
    ]
    static_candidates.sort(key=lambda x: x["max_speed"], reverse=True)
    if static_candidates:
        print_section("Static-But-Jittery (median low, max high)", static_candidates, args.top)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (Path(args.pkl).stem + "_diagnostics.csv")
        fieldnames = [
            "track_id", "frames", "span", "missing", "missing_ratio",
            "gap_count", "max_gap", "use_world",
            "median_speed", "p95_speed", "max_speed", "max_speed_naive",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summaries:
                writer.writerow(row)
        print(f"\nWrote CSV summary to: {out_path}")


if __name__ == "__main__":
    main()
