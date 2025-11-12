#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick inspection tool for det_bbox_result_*.pkl files.

Example:
    python using/inspect_det_bbox.py \
        --pkl data1/UAV_Videos/track_small_raw/output/track_small_raw/det_bbox_result_track_small_raw.pkl \
        --tracks 0 10 \
        --frames 1489 1495
"""
import argparse
import pickle
from typing import List, Tuple, Any


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect det_bbox_result_*.pkl structure.")
    parser.add_argument("--pkl", required=True, help="Path to det_bbox_result_*.pkl")
    parser.add_argument("--tracks", nargs="*", type=int, default=None,
                        help="Track IDs to display (default: all).")
    parser.add_argument("--frames", nargs=2, type=int, metavar=("START", "END"), default=None,
                        help="Frame index range to display [inclusive, inclusive].")
    parser.add_argument("--limit", type=int, default=20,
                        help="Max rows to print per frame (default: 20).")
    return parser.parse_args()


def format_entry(entry: Tuple[Any, ...]):
    if len(entry) == 3:
        frame_idx, output_idx, arr = entry
        frame_time = None
    else:
        frame_idx, output_idx, arr, frame_time = entry
    return frame_idx, output_idx, arr, frame_time


def main():
    args = parse_args()
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded keys: {list(data.keys())}")
    traj = data["traj_info"]
    print(f"Total traj entries (frames): {len(traj)}")

    selected_frames = []
    for entry in traj:
        frame_idx, output_idx, arr, frame_time = format_entry(entry)
        if args.frames:
            if not (args.frames[0] <= frame_idx <= args.frames[1]):
                continue
        selected_frames.append((frame_idx, output_idx, arr, frame_time))

    if not selected_frames:
        print("No frames matched the filter.")
        return

    for frame_idx, output_idx, arr, frame_time in selected_frames:
        print(f"\n=== Frame {frame_idx} (output {output_idx}) ===")
        if frame_time is not None:
            print(f" timestamp: {frame_time}")
        rows_printed = 0
        for row in arr:
            track_id = int(row[10]) if row.shape[0] > 10 else -1
            lane_id = int(row[-1]) if row.shape[0] in (20,) else "NA"
            if args.tracks and track_id not in args.tracks:
                continue
            pixel_ct = row[:8].reshape(-1, 2).mean(axis=0)
            print(f" track={track_id:4d} lane={lane_id:4} "
                  f"center_px=({pixel_ct[0]:7.1f},{pixel_ct[1]:7.1f}) "
                  f"score={row[8]:.2f}")
            rows_printed += 1
            if rows_printed >= args.limit:
                print(" ... truncated ...")
                break


if __name__ == "__main__":
    main()
