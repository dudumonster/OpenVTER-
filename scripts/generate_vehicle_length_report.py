#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import math
import pickle
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


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

VEHICLE_CLASSES = {"car", "van", "truck", "bus", "freight_car"}
DEFAULT_LENGTH_BINS = [0.0, 3.5, 4.0, 4.5, 5.0, 5.4, 6.0, 6.8, 8.0, 9.5, 12.0, math.inf]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a reusable HTML vehicle length QA report from OpenVTER "
            "output_server39 results or exported tracksMeta CSV files."
        )
    )
    parser.add_argument(
        "input",
        help=(
            "Scene output directory, such as .../ban_xian_shan/output_server39, "
            "or a directory containing *_tracksMeta.csv files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for report files. Default: <input>/vehicle_length_report.",
    )
    parser.add_argument(
        "--scene-name",
        default=None,
        help="Scene name shown in the report. Default: inferred from the input path.",
    )
    parser.add_argument(
        "--car-threshold",
        type=float,
        default=5.4,
        help="Length threshold between car and van. Default: 5.4.",
    )
    parser.add_argument(
        "--van-threshold",
        type=float,
        default=6.8,
        help="Length threshold between van and truck. Default: 6.8.",
    )
    parser.add_argument(
        "--include-backups",
        action="store_true",
        help="Include directories whose names contain .bak_. They are skipped by default.",
    )
    parser.add_argument(
        "--max-detail-rows",
        type=int,
        default=120,
        help="Maximum suspicious track rows to render in HTML. CSV keeps all rows.",
    )
    return parser.parse_args()


def category_name(cat_id: int) -> str:
    if 0 <= cat_id < len(CATEGORY_NAMES):
        return CATEGORY_NAMES[cat_id]
    return f"unknown_{cat_id}"


def distance(p1: Iterable[float], p2: Iterable[float]) -> float:
    a = list(p1)
    b = list(p2)
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def box_dimensions(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    edge_distances = [distance(points[i], points[(i + 1) % 4]) for i in range(4)]
    pair_distances = [
        distance(points[i], points[j])
        for i in range(4)
        for j in range(i + 1, 4)
    ]
    return max(edge_distances), min(edge_distances), max(pair_distances)


def percentile(values: list[float], pct: float) -> float:
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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def pct(part: int | float, total: int | float) -> float:
    if not total:
        return 0.0
    return float(part) * 100.0 / float(total)


def mode_category(categories: list[int]) -> tuple[int, Counter[int]]:
    counts: Counter[int] = Counter(categories)
    max_count = max(counts.values())
    winners = [cat for cat, count in counts.items() if count == max_count]
    return min(winners), counts


def read_pickle_tracks(pkl_path: Path, video_name: str) -> list[dict[str, Any]]:
    with pkl_path.open("rb") as fh:
        data = pickle.load(fh)

    per_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in data.get("traj_info") or []:
        if len(entry) == 3:
            frame_idx, _output_idx, arr = entry
        else:
            frame_idx, _output_idx, arr, _frame_time = entry
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
                    (float(row[11]), float(row[12])),
                    (float(row[13]), float(row[14])),
                    (float(row[15]), float(row[16])),
                    (float(row[17]), float(row[18])),
                ]
            else:
                points = [
                    (float(row[0]), float(row[1])),
                    (float(row[2]), float(row[3])),
                    (float(row[4]), float(row[5])),
                    (float(row[6]), float(row[7])),
                ]
            long_edge, short_edge, diagonal = box_dimensions(points)
            per_track[track_id].append(
                {
                    "frame": int(frame_idx),
                    "category": category,
                    "score": score,
                    "length": long_edge,
                    "width": short_edge,
                    "diagonal": diagonal,
                }
            )

    tracks: list[dict[str, Any]] = []
    for track_id, rows in per_track.items():
        categories = [row["category"] for row in rows]
        mode_cat, counts = mode_category(categories)
        frames = [row["frame"] for row in rows]
        lengths = [row["length"] for row in rows]
        widths = [row["width"] for row in rows]
        diagonals = [row["diagonal"] for row in rows]
        scores = [row["score"] for row in rows]
        tracks.append(
            {
                "video": video_name,
                "track_id": track_id,
                "class": category_name(mode_cat),
                "frame_count": len(rows),
                "start_frame": min(frames),
                "end_frame": max(frames),
                "length_min": min(lengths),
                "length_median": percentile(lengths, 50),
                "length_mean": mean(lengths),
                "length_p95": percentile(lengths, 95),
                "length_max": max(lengths),
                "width_median": percentile(widths, 50),
                "width_mean": mean(widths),
                "diagonal_median": percentile(diagonals, 50),
                "score_mean": mean(scores),
                "mode_category_ratio": counts[mode_cat] / len(rows),
                "category_counts": ";".join(
                    f"{category_name(cat)}:{count}" for cat, count in sorted(counts.items())
                ),
                "source": str(pkl_path),
            }
        )
    return tracks


def read_tracks_meta(csv_path: Path, video_name: str) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                length = float(row.get("length", "nan"))
                width = float(row.get("width", "nan"))
            except ValueError:
                continue
            cls = row.get("class", "")
            track_id_raw = row.get("trackId") or row.get("track_id") or "0"
            frame_count_raw = row.get("numFrames") or row.get("frame_count") or "1"
            start_frame_raw = row.get("initialFrame") or row.get("start_frame") or "0"
            end_frame_raw = row.get("finalFrame") or row.get("end_frame") or start_frame_raw
            tracks.append(
                {
                    "video": video_name,
                    "track_id": int(float(track_id_raw)),
                    "class": cls,
                    "frame_count": int(float(frame_count_raw)),
                    "start_frame": int(float(start_frame_raw)),
                    "end_frame": int(float(end_frame_raw)),
                    "length_min": length,
                    "length_median": length,
                    "length_mean": length,
                    "length_p95": length,
                    "length_max": length,
                    "width_median": width,
                    "width_mean": width,
                    "diagonal_median": math.nan,
                    "score_mean": math.nan,
                    "mode_category_ratio": 1.0,
                    "category_counts": f"{cls}:1",
                    "source": str(csv_path),
                }
            )
    return tracks


def find_video_inputs(input_dir: Path, include_backups: bool) -> list[dict[str, Any]]:
    candidates: list[Path]
    if any(input_dir.glob("det_bbox_result_*.pkl")) or any(input_dir.glob("*_tracksMeta.csv")):
        candidates = [input_dir]
    else:
        candidates = sorted(path for path in input_dir.iterdir() if path.is_dir())

    video_inputs: list[dict[str, Any]] = []
    for video_dir in candidates:
        if not include_backups and ".bak_" in video_dir.name:
            continue
        pkl_matches = sorted(video_dir.glob("det_bbox_result_*.pkl"))
        csv_matches = sorted(video_dir.glob("*_tracksMeta.csv"))
        tracking_matches = sorted(video_dir.glob(f"tracking_output_*_{video_dir.name}.mp4"))
        video_inputs.append(
            {
                "video": video_dir.name,
                "dir": video_dir,
                "pkl": pkl_matches[0] if pkl_matches else None,
                "tracks_meta": csv_matches[0] if csv_matches else None,
                "tracking_mp4_count": len(tracking_matches),
            }
        )
    return video_inputs


def load_scene_tracks(video_inputs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_tracks: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for item in video_inputs:
        status = {
            "video": item["video"],
            "dir": str(item["dir"]),
            "source_type": "",
            "track_count": 0,
            "vehicle_count": 0,
            "tracking_mp4_count": item["tracking_mp4_count"],
            "ok": False,
            "issue": "",
        }
        try:
            if item["pkl"] is not None:
                tracks = read_pickle_tracks(item["pkl"], item["video"])
                status["source_type"] = "pkl"
            elif item["tracks_meta"] is not None:
                tracks = read_tracks_meta(item["tracks_meta"], item["video"])
                status["source_type"] = "tracksMeta.csv"
            else:
                tracks = []
                status["issue"] = "missing det_bbox_result_*.pkl and *_tracksMeta.csv"
            status["track_count"] = len(tracks)
            status["vehicle_count"] = sum(1 for track in tracks if track["class"] in VEHICLE_CLASSES)
            status["ok"] = bool(tracks)
            all_tracks.extend(tracks)
        except Exception as exc:
            status["issue"] = f"read failed: {exc}"
        statuses.append(status)
    return all_tracks, statuses


def length_bin_label(lo: float, hi: float) -> str:
    if math.isinf(hi):
        return f">={lo:g}"
    return f"{lo:g}-{hi:g}"


def bin_counts(lengths: list[float], bins: list[float]) -> list[tuple[str, int]]:
    counts: list[tuple[str, int]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        count = sum(1 for value in lengths if lo <= value < hi)
        counts.append((length_bin_label(lo, hi), count))
    return counts


def class_stats(tracks: list[dict[str, Any]], cls: str) -> dict[str, Any]:
    selected = [track for track in tracks if track["class"] == cls]
    lengths = [track["length_median"] for track in selected]
    widths = [track["width_median"] for track in selected]
    return {
        "class": cls,
        "count": len(selected),
        "length_min": min(lengths) if lengths else math.nan,
        "length_p25": percentile(lengths, 25),
        "length_median": percentile(lengths, 50),
        "length_mean": mean(lengths),
        "length_p75": percentile(lengths, 75),
        "length_p95": percentile(lengths, 95),
        "length_max": max(lengths) if lengths else math.nan,
        "width_median": percentile(widths, 50),
    }


def build_video_summary(
    video: str,
    tracks: list[dict[str, Any]],
    car_threshold: float,
    van_threshold: float,
) -> dict[str, Any]:
    vehicle_tracks = [track for track in tracks if track["class"] in VEHICLE_CLASSES]
    class_counts = Counter(track["class"] for track in tracks)
    car_tracks = [track for track in tracks if track["class"] == "car"]
    van_tracks = [track for track in tracks if track["class"] == "van"]
    long_cars = [track for track in car_tracks if track["length_median"] >= car_threshold]
    short_vans = [track for track in van_tracks if track["length_median"] < car_threshold]
    long_vans = [track for track in van_tracks if track["length_median"] >= van_threshold]
    return {
        "video": video,
        "tracks_total": len(tracks),
        "vehicle_tracks": len(vehicle_tracks),
        "class_counts": dict(sorted(class_counts.items())),
        "car_count": len(car_tracks),
        "van_count": len(van_tracks),
        "car_ge_threshold": len(long_cars),
        "car_ge_threshold_pct": pct(len(long_cars), len(car_tracks)),
        "van_lt_car_threshold": len(short_vans),
        "van_lt_car_threshold_pct": pct(len(short_vans), len(van_tracks)),
        "van_ge_van_threshold": len(long_vans),
        "van_ge_van_threshold_pct": pct(len(long_vans), len(van_tracks)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def svg_bar_chart(data: list[tuple[str, int]], width: int = 760, height: int = 230) -> str:
    if not data:
        return "<p class='muted'>No data.</p>"
    max_value = max(value for _label, value in data) or 1
    left = 46
    bottom = 34
    top = 16
    usable_height = height - top - bottom
    bar_gap = 5
    bar_width = max(10, (width - left - 18) // len(data) - bar_gap)
    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart' role='img'>"]
    parts.append(f"<line x1='{left}' y1='{height-bottom}' x2='{width-10}' y2='{height-bottom}' class='axis' />")
    for index, (label, value) in enumerate(data):
        x = left + index * (bar_width + bar_gap)
        bar_height = usable_height * value / max_value
        y = height - bottom - bar_height
        parts.append(f"<rect x='{x}' y='{y:.1f}' width='{bar_width}' height='{bar_height:.1f}' class='bar' />")
        parts.append(f"<text x='{x + bar_width / 2:.1f}' y='{y - 4:.1f}' class='value'>{value}</text>")
        parts.append(
            f"<text x='{x + bar_width / 2:.1f}' y='{height - 11}' class='label' "
            f"transform='rotate(-35 {x + bar_width / 2:.1f},{height - 11})'>{html.escape(label)}</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def html_table(headers: list[str], rows: list[list[Any]], cls: str = "") -> str:
    thead = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(fmt(value))}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table class='{cls}'><thead><tr>{thead}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def render_html(
    scene_name: str,
    input_dir: Path,
    output_dir: Path,
    statuses: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    video_summaries: list[dict[str, Any]],
    car_threshold: float,
    van_threshold: float,
    max_detail_rows: int,
) -> str:
    vehicle_tracks = [track for track in tracks if track["class"] in VEHICLE_CLASSES]
    class_counts = Counter(track["class"] for track in tracks)
    vehicle_class_counts = [(cls, class_counts.get(cls, 0)) for cls in ["car", "van", "truck", "bus", "freight_car"]]
    class_stat_rows = []
    for cls in ["car", "van", "truck", "bus", "freight_car", "motor", "pedestrian", "people"]:
        stats = class_stats(tracks, cls)
        if stats["count"]:
            class_stat_rows.append(
                [
                    stats["class"],
                    stats["count"],
                    stats["length_min"],
                    stats["length_p25"],
                    stats["length_median"],
                    stats["length_mean"],
                    stats["length_p75"],
                    stats["length_p95"],
                    stats["length_max"],
                    stats["width_median"],
                ]
            )

    status_rows = [
        [
            row["video"],
            "OK" if row["ok"] else "BAD",
            row["source_type"],
            row["track_count"],
            row["vehicle_count"],
            row["tracking_mp4_count"],
            row["issue"],
        ]
        for row in statuses
    ]

    video_rows = [
        [
            item["video"],
            item["tracks_total"],
            item["vehicle_tracks"],
            item["car_count"],
            item["van_count"],
            f'{item["car_ge_threshold"]} ({item["car_ge_threshold_pct"]:.1f}%)',
            f'{item["van_lt_car_threshold"]} ({item["van_lt_car_threshold_pct"]:.1f}%)',
            f'{item["van_ge_van_threshold"]} ({item["van_ge_van_threshold_pct"]:.1f}%)',
            ", ".join(f"{key}:{value}" for key, value in item["class_counts"].items()),
        ]
        for item in video_summaries
    ]

    suspicious = [
        track
        for track in tracks
        if (track["class"] == "car" and track["length_median"] >= car_threshold)
        or (track["class"] == "van" and track["length_median"] < car_threshold)
        or (track["class"] == "van" and track["length_median"] >= van_threshold)
    ]
    suspicious.sort(key=lambda item: (item["video"], item["class"], -item["length_median"]))
    suspicious_rows = [
        [
            track["video"],
            track["track_id"],
            track["class"],
            track["frame_count"],
            track["start_frame"],
            track["end_frame"],
            track["length_median"],
            track["length_p95"],
            track["width_median"],
            track["category_counts"],
        ]
        for track in suspicious[:max_detail_rows]
    ]

    histogram_sections = []
    for cls in ["car", "van", "truck", "bus", "freight_car"]:
        lengths = [track["length_median"] for track in tracks if track["class"] == cls]
        if lengths:
            histogram_sections.append(
                f"<section><h2>{html.escape(cls)} length histogram</h2>"
                + svg_bar_chart(bin_counts(lengths, DEFAULT_LENGTH_BINS))
                + "</section>"
            )

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ok_count = sum(1 for row in statuses if row["ok"])
    bad_count = len(statuses) - ok_count
    car_tracks = [track for track in tracks if track["class"] == "car"]
    van_tracks = [track for track in tracks if track["class"] == "van"]
    long_cars = sum(1 for track in car_tracks if track["length_median"] >= car_threshold)
    short_vans = sum(1 for track in van_tracks if track["length_median"] < car_threshold)

    css = """
    body { margin: 0; font-family: Arial, sans-serif; color: #20242a; background: #f6f7f9; }
    header { padding: 26px 34px; background: #17212b; color: #fff; }
    main { padding: 24px 34px 42px; }
    h1 { margin: 0 0 8px; font-size: 28px; font-weight: 700; }
    h2 { margin: 0 0 14px; font-size: 20px; }
    section { margin: 0 0 22px; padding: 18px; background: #fff; border: 1px solid #e2e6ea; border-radius: 8px; }
    .muted { color: #68717d; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 18px 0 0; }
    .card { padding: 14px; background: #243241; border-radius: 8px; }
    .card b { display: block; font-size: 24px; margin-bottom: 4px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid #e7eaee; padding: 8px 9px; text-align: left; vertical-align: top; }
    th { background: #f0f3f6; font-weight: 700; position: sticky; top: 0; }
    .scroll { overflow-x: auto; }
    .chart { width: 100%; max-width: 950px; height: auto; }
    .axis { stroke: #9aa4af; stroke-width: 1; }
    .bar { fill: #3772ff; }
    .value { font-size: 11px; text-anchor: middle; fill: #333; }
    .label { font-size: 10px; text-anchor: end; fill: #333; }
    """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(scene_name)} vehicle length report</title>
  <style>{css}</style>
</head>
<body>
<header>
  <h1>{html.escape(scene_name)} vehicle length QA report</h1>
  <div class="muted">Generated: {html.escape(generated)} | Input: {html.escape(str(input_dir))} | Output: {html.escape(str(output_dir))}</div>
  <div class="cards">
    <div class="card"><b>{len(statuses)}</b>video dirs</div>
    <div class="card"><b>{ok_count}</b>readable dirs</div>
    <div class="card"><b>{bad_count}</b>incomplete dirs</div>
    <div class="card"><b>{len(vehicle_tracks)}</b>vehicle tracks</div>
    <div class="card"><b>{long_cars}</b>car >= {car_threshold:g}m ({pct(long_cars, len(car_tracks)):.1f}%)</div>
    <div class="card"><b>{short_vans}</b>van < {car_threshold:g}m ({pct(short_vans, len(van_tracks)):.1f}%)</div>
  </div>
</header>
<main>
  <section>
    <h2>Vehicle class counts</h2>
    {svg_bar_chart(vehicle_class_counts)}
  </section>
  <section>
    <h2>Class length statistics</h2>
    <div class="scroll">
    {html_table(["class", "count", "min", "p25", "median", "mean", "p75", "p95", "max", "width median"], class_stat_rows)}
    </div>
  </section>
  {''.join(histogram_sections)}
  <section>
    <h2>Per-video summary</h2>
    <div class="scroll">
    {html_table(["video", "tracks", "vehicles", "car", "van", f"car >= {car_threshold:g}m", f"van < {car_threshold:g}m", f"van >= {van_threshold:g}m", "class counts"], video_rows)}
    </div>
  </section>
  <section>
    <h2>Suspicious tracks</h2>
    <p class="muted">Rules: car >= {car_threshold:g}m, van < {car_threshold:g}m, or van >= {van_threshold:g}m.</p>
    <div class="scroll">
    {html_table(["video", "track", "class", "frames", "start", "end", "length median", "length p95", "width median", "category counts"], suspicious_rows)}
    </div>
  </section>
  <section>
    <h2>Output completeness</h2>
    <div class="scroll">
    {html_table(["video", "status", "source", "tracks", "vehicles", "tracking mp4 count", "issue"], status_rows)}
    </div>
  </section>
</main>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.exists():
        print(f"Input does not exist: {input_dir}")
        return 2

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_dir / "vehicle_length_report"
    )
    scene_name = args.scene_name or (input_dir.parent.name if input_dir.name.startswith("output") else input_dir.name)

    video_inputs = find_video_inputs(input_dir, args.include_backups)
    tracks, statuses = load_scene_tracks(video_inputs)

    tracks.sort(key=lambda row: (row["video"], row["class"], int(row["track_id"])))
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for track in tracks:
        by_video[track["video"]].append(track)
    video_summaries = [
        build_video_summary(video, by_video.get(video, []), args.car_threshold, args.van_threshold)
        for video in sorted(item["video"] for item in video_inputs)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    track_csv = output_dir / "vehicle_length_tracks.csv"
    summary_csv = output_dir / "vehicle_length_video_summary.csv"
    status_csv = output_dir / "vehicle_length_output_status.csv"
    html_path = output_dir / "vehicle_length_report.html"

    write_csv(
        track_csv,
        tracks,
        [
            "video",
            "track_id",
            "class",
            "frame_count",
            "start_frame",
            "end_frame",
            "length_min",
            "length_median",
            "length_mean",
            "length_p95",
            "length_max",
            "width_median",
            "width_mean",
            "diagonal_median",
            "score_mean",
            "mode_category_ratio",
            "category_counts",
            "source",
        ],
    )
    write_csv(
        summary_csv,
        video_summaries,
        [
            "video",
            "tracks_total",
            "vehicle_tracks",
            "car_count",
            "van_count",
            "car_ge_threshold",
            "car_ge_threshold_pct",
            "van_lt_car_threshold",
            "van_lt_car_threshold_pct",
            "van_ge_van_threshold",
            "van_ge_van_threshold_pct",
            "class_counts",
        ],
    )
    write_csv(
        status_csv,
        statuses,
        ["video", "dir", "source_type", "track_count", "vehicle_count", "tracking_mp4_count", "ok", "issue"],
    )
    html_path.write_text(
        render_html(
            scene_name=scene_name,
            input_dir=input_dir,
            output_dir=output_dir,
            statuses=statuses,
            tracks=tracks,
            video_summaries=video_summaries,
            car_threshold=args.car_threshold,
            van_threshold=args.van_threshold,
            max_detail_rows=args.max_detail_rows,
        ),
        encoding="utf-8",
    )

    ok_count = sum(1 for row in statuses if row["ok"])
    print(f"Read video dirs: {ok_count}/{len(statuses)}")
    print(f"Tracks: {len(tracks)}")
    print(f"HTML report: {html_path}")
    print(f"Track CSV: {track_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Status CSV: {status_csv}")
    return 0 if ok_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
