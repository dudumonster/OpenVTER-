#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze per-track vehicle length statistics from OpenVTER det_bbox_result_*.pkl.

The online pipeline classifies vehicle type by the longer edge of the rotated
box. This script reports that value as length_used_by_code and also keeps the
diagonal length as a diagnostic column.
"""
import argparse
import csv
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path


CATEGORY_NAMES = [
    "car",
    "truck",
    "bus",
    "freight_car",
    "van",
    "pedestrian",
    "people",
    "bicycle",
    "tricycle",
    "awning-tricycle",
    "motor",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze vehicle lengths for tracks, especially van-class tracks."
    )
    parser.add_argument(
        "input",
        help="Path to det_bbox_result_*.pkl or a folder containing one.",
    )
    parser.add_argument(
        "--category",
        default="van",
        help="Category name or id to focus on. Default: van.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: <input_folder>/vehicle_length_analysis_<category>.csv",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.2,
        help="Car/van threshold used for summary. Default: 5.2.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of focused tracks to print. Default: 30.",
    )
    return parser.parse_args()


def find_pkl(input_path):
    path = Path(input_path)
    if path.is_file():
        return path
    matches = sorted(path.glob("det_bbox_result_*.pkl"))
    if not matches:
        raise FileNotFoundError(f"No det_bbox_result_*.pkl found in {path}")
    if len(matches) > 1:
        print("Found multiple pkl files; using:", matches[0])
    return matches[0]


def parse_category(value):
    try:
        return int(value)
    except ValueError:
        pass
    if value not in CATEGORY_NAMES:
        raise ValueError(f"Unknown category {value!r}. Known: {CATEGORY_NAMES}")
    return CATEGORY_NAMES.index(value)


def format_entry(entry):
    if len(entry) == 3:
        frame_idx, output_idx, arr = entry
        frame_time = None
    else:
        frame_idx, output_idx, arr, frame_time = entry
    return frame_idx, output_idx, arr, frame_time


def distance(p1, p2):
    return math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))


def box_dimensions(points):
    """Return (long_edge, short_edge, diagonal)."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    pair_distances = [
        distance(pts[i], pts[j])
        for i in range(4)
        for j in range(i + 1, 4)
    ]
    edge_distances = [distance(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    return max(edge_distances), min(edge_distances), max(pair_distances)


def percentile(values, pct):
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - rank) + ordered[hi] * (rank - lo)


def mean(values):
    return sum(values) / len(values) if values else math.nan


def mode_category(categories):
    counts = Counter(categories)
    max_count = max(counts.values())
    winners = [cat for cat, count in counts.items() if count == max_count]
    return min(winners), counts


def category_name(cat_id):
    if 0 <= cat_id < len(CATEGORY_NAMES):
        return CATEGORY_NAMES[cat_id]
    return f"unknown_{cat_id}"


def summarize_track(track_id, rows):
    cats = [row["category"] for row in rows]
    mode_cat, counts = mode_category(cats)
    code_lengths = [row["length_used_by_code"] for row in rows]
    long_edges = [row["long_edge_length"] for row in rows]
    short_edges = [row["short_edge_length"] for row in rows]
    diagonals = [row["diagonal_length"] for row in rows]
    scores = [row["score"] for row in rows]
    frames = [row["frame_index"] for row in rows]
    van_frames = sum(1 for cat in cats if cat == 4)
    return {
        "track_id": track_id,
        "frame_count": len(rows),
        "start_frame": min(frames),
        "end_frame": max(frames),
        "mode_category_id": mode_cat,
        "mode_category_name": category_name(mode_cat),
        "mode_category_ratio": counts[mode_cat] / len(rows),
        "category_counts": ";".join(
            f"{category_name(cat)}:{count}" for cat, count in sorted(counts.items())
        ),
        "van_frame_ratio": van_frames / len(rows),
        "score_mean": mean(scores),
        "code_len_min": min(code_lengths),
        "code_len_median": percentile(code_lengths, 50),
        "code_len_mean": mean(code_lengths),
        "code_len_p95": percentile(code_lengths, 95),
        "code_len_max": max(code_lengths),
        "long_edge_min": min(long_edges),
        "long_edge_median": percentile(long_edges, 50),
        "long_edge_mean": mean(long_edges),
        "long_edge_p95": percentile(long_edges, 95),
        "long_edge_max": max(long_edges),
        "short_edge_median": percentile(short_edges, 50),
        "diagonal_median": percentile(diagonals, 50),
        "diagonal_p95": percentile(diagonals, 95),
    }


def load_track_rows(pkl_path):
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    traj_info = data.get("traj_info") or []
    tracks = defaultdict(list)
    for entry in traj_info:
        frame_idx, _output_idx, arr, _frame_time = format_entry(entry)
        if arr is None or len(arr) == 0:
            continue
        for row in arr:
            if len(row) <= 10:
                continue
            track_id = int(round(float(row[10])))
            category = int(round(float(row[9])))
            score = float(row[8])
            if len(row) >= 19:
                points = [
                    (row[11], row[12]),
                    (row[13], row[14]),
                    (row[15], row[16]),
                    (row[17], row[18]),
                ]
            else:
                points = [
                    (row[0], row[1]),
                    (row[2], row[3]),
                    (row[4], row[5]),
                    (row[6], row[7]),
                ]
            long_edge, short_edge, diagonal = box_dimensions(points)
            tracks[track_id].append(
                {
                    "frame_index": int(frame_idx),
                    "category": category,
                    "score": score,
                    "length_used_by_code": long_edge,
                    "long_edge_length": long_edge,
                    "short_edge_length": short_edge,
                    "diagonal_length": diagonal,
                }
            )
    return tracks


def write_csv(path, summaries):
    fieldnames = [
        "track_id",
        "frame_count",
        "start_frame",
        "end_frame",
        "mode_category_id",
        "mode_category_name",
        "mode_category_ratio",
        "category_counts",
        "van_frame_ratio",
        "score_mean",
        "code_len_min",
        "code_len_median",
        "code_len_mean",
        "code_len_p95",
        "code_len_max",
        "long_edge_min",
        "long_edge_median",
        "long_edge_mean",
        "long_edge_p95",
        "long_edge_max",
        "short_edge_median",
        "diagonal_median",
        "diagonal_p95",
        "above_threshold_by_code_median",
        "above_threshold_by_long_edge_median",
    ]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main():
    args = parse_args()
    pkl_path = find_pkl(args.input)
    focus_cat = parse_category(args.category)
    focus_name = category_name(focus_cat)
    output_path = (
        Path(args.output)
        if args.output
        else pkl_path.parent / f"vehicle_length_analysis_{focus_name}.csv"
    )

    tracks = load_track_rows(pkl_path)
    summaries = []
    for track_id, rows in tracks.items():
        summary = summarize_track(track_id, rows)
        summary["above_threshold_by_code_median"] = (
            summary["code_len_median"] >= args.threshold
        )
        summary["above_threshold_by_long_edge_median"] = (
            summary["long_edge_median"] >= args.threshold
        )
        summaries.append(summary)

    summaries.sort(
        key=lambda row: (
            row["mode_category_id"] != focus_cat,
            -row["code_len_median"],
            row["track_id"],
        )
    )
    write_csv(output_path, summaries)

    focused = [row for row in summaries if row["mode_category_id"] == focus_cat]
    print(f"Input: {pkl_path}")
    print(f"Tracks total: {len(summaries)}")
    print(f"{focus_name} tracks: {len(focused)}")
    if focused:
        code_above = sum(row["above_threshold_by_code_median"] for row in focused)
        edge_above = sum(row["above_threshold_by_long_edge_median"] for row in focused)
        print(
            f"{focus_name} median length_used_by_code >= {args.threshold}: "
            f"{code_above}/{len(focused)}"
        )
        print(
            f"{focus_name} median long_edge_length >= {args.threshold}: "
            f"{edge_above}/{len(focused)}"
        )
        print(
            f"{focus_name} code_len_median range: "
            f"{min(row['code_len_median'] for row in focused):.3f} - "
            f"{max(row['code_len_median'] for row in focused):.3f}"
        )
        print(
            f"{focus_name} long_edge_median range: "
            f"{min(row['long_edge_median'] for row in focused):.3f} - "
            f"{max(row['long_edge_median'] for row in focused):.3f}"
        )
        print()
        print(
            "Top focused tracks by length_used_by_code median "
            "(track, frames, code_median, diagonal_median, short_edge_median, counts):"
        )
        for row in focused[: args.top]:
            print(
                "  "
                + ", ".join(
                    [
                        f"track={row['track_id']}",
                        f"frames={row['frame_count']}",
                        f"code_med={fmt(row['code_len_median'])}",
                        f"diag_med={fmt(row['diagonal_median'])}",
                        f"short_med={fmt(row['short_edge_median'])}",
                        f"counts={row['category_counts']}",
                    ]
                )
            )
    print(f"CSV written: {output_path}")


if __name__ == "__main__":
    main()
