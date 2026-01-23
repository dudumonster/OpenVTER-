#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export det_bbox_result_*.pkl to Excel with gap filling and physics diagnostics.

Default behavior:
- Build a complete frame timeline per track (f_min..f_max).
- Fill gaps by linear interpolation; large gaps are marked low confidence.
- Stabilize category to the majority class within each track.
- Compute kinematics (velocity/acceleration) using full timeline.
- Apply stationary gating (speed < v_stop for K frames -> speed/accel set to 0).
"""
import argparse
import math
import pickle
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd


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

VEHICLE_CLASS_IDS = {0, 1, 2, 3, 4}


def parse_args():
    parser = argparse.ArgumentParser(description="Fill track gaps and export Excel.")
    parser.add_argument("--pkl", required=True, help="Path to det_bbox_result_*.pkl")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: <pkl_dir>/saving)")
    parser.add_argument("--fps", type=float, default=None, help="Override fps (default: use pkl output_fps)")
    parser.add_argument("--g1", type=int, default=4, help="Small gap threshold (frames)")
    parser.add_argument("--g2", type=int, default=12, help="Medium gap threshold (frames)")
    parser.add_argument("--g3", type=int, default=30, help="Large gap threshold (frames)")
    parser.add_argument("--vmax-vehicle", type=float, default=15.0, help="Vehicle max speed (m/s)")
    parser.add_argument("--amax-vehicle", type=float, default=3.0, help="Vehicle max acceleration (m/s^2)")
    parser.add_argument("--vmax-vru", type=float, default=9.0, help="VRU max speed (m/s)")
    parser.add_argument("--amax-vru", type=float, default=3.0, help="VRU max acceleration (m/s^2)")
    parser.add_argument("--v-stop", type=float, default=0.25, help="Stationary speed threshold (m/s)")
    parser.add_argument("--k-stop", type=int, default=15, help="Stationary window length (frames)")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Median smoothing window for centers before kinematics")
    parser.add_argument("--smooth-method", choices=["median", "savgol"], default="savgol",
                        help="Position smoothing method")
    parser.add_argument("--sg-window", type=int, default=9,
                        help="Savitzky-Golay window length (odd)")
    parser.add_argument("--sg-poly", type=int, default=2,
                        help="Savitzky-Golay polynomial order")
    parser.add_argument("--stat-window", type=int, default=30,
                        help="Stationary dominance window (frames)")
    parser.add_argument("--stat-ratio", type=float, default=0.8,
                        help="Stationary dominance ratio threshold")
    parser.add_argument("--center-x", type=float, default=0.0,
                        help="World center x for tangential/normal components")
    parser.add_argument("--center-y", type=float, default=0.0,
                        help="World center y for tangential/normal components")
    parser.add_argument("--center-eps", type=float, default=1e-3,
                        help="Minimum radius to compute tangential/normal components")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--drop-unreliable-phys", dest="drop_unreliable_phys", action="store_true",
                       help="Set phys values to NaN for unreliable frames (default)")
    group.add_argument("--keep-unreliable-phys", dest="drop_unreliable_phys", action="store_false",
                       help="Keep phys values on unreliable frames")
    parser.set_defaults(drop_unreliable_phys=True)
    return parser.parse_args()


def format_entry(entry):
    if len(entry) == 3:
        frame_idx, output_idx, arr = entry
        frame_time = None
    else:
        frame_idx, output_idx, arr, frame_time = entry
    return frame_idx, output_idx, arr, frame_time


def majority_category(cats):
    counts = Counter(cats)
    max_count = max(counts.values())
    top = [k for k, v in counts.items() if v == max_count]
    return int(min(top))


def safe_mean(points):
    if points is None or len(points) == 0:
        return None
    arr = np.asarray(points, dtype=np.float32)
    if np.isnan(arr).any():
        return None
    return arr.mean(axis=0)


def build_tracks(traj_info):
    tracks = defaultdict(dict)
    frame_meta = {}
    has_world = False
    has_lane = False
    for entry in traj_info:
        frame_idx, output_idx, arr, frame_time = format_entry(entry)
        frame_meta.setdefault(frame_idx, {"output_frame": output_idx, "frame_time": frame_time})
        if arr is None or len(arr) == 0:
            continue
        for row in arr:
            row = np.asarray(row, dtype=np.float32)
            if row.shape[0] <= 10:
                continue
            track_id = int(row[10])
            score = float(row[8])
            category = int(row[9])
            pix = row[0:8].reshape(4, 2)
            world = None
            lane_id = None
            if row.shape[0] >= 19:
                world = row[11:19].reshape(4, 2)
                has_world = True
            if row.shape[0] >= 20:
                lane_id = int(row[19])
                has_lane = True
            # Keep highest score if duplicated in same frame
            existing = tracks[track_id].get(frame_idx, None)
            if existing is None or score > existing["score"]:
                tracks[track_id][frame_idx] = {
                    "pixel": pix,
                    "world": world,
                    "score": score,
                    "category": category,
                    "lane_id": lane_id,
                }
    return tracks, frame_meta, has_world, has_lane


def fill_track(track_frames, frame_range, majority_cat, g1, g2, g3):
    filled = {}
    observed_frames = sorted(track_frames.keys())
    if not observed_frames:
        return filled
    for f in frame_range:
        if f in track_frames:
            item = track_frames[f].copy()
            item["is_observed"] = True
            item["fill_type"] = "observed"
            item["gap_size"] = 0
            filled[f] = item
        else:
            filled[f] = {
                "pixel": None,
                "world": None,
                "score": math.nan,
                "category": majority_cat,
                "lane_id": None,
                "is_observed": False,
                "fill_type": "missing",
                "gap_size": None,
            }
    # Fill gaps by interpolation between observed frames
    for f0, f1 in zip(observed_frames[:-1], observed_frames[1:]):
        gap = f1 - f0 - 1
        if gap <= 0:
            continue
        fill_type = "linear" if gap <= g2 else "predict_low_conf"
        if gap > g2:
            fill_type = "predict_low_conf"
        for f in range(f0 + 1, f1):
            ratio = (f - f0) / float(f1 - f0)
            left = track_frames[f0]
            right = track_frames[f1]
            pix = None
            world = None
            if left["pixel"] is not None and right["pixel"] is not None:
                pix = left["pixel"] + (right["pixel"] - left["pixel"]) * ratio
            if left["world"] is not None and right["world"] is not None:
                world = left["world"] + (right["world"] - left["world"]) * ratio
            filled[f].update({
                "pixel": pix,
                "world": world,
                "score": math.nan,
                "category": majority_cat,
                "lane_id": left.get("lane_id", None),
                "is_observed": False,
                "fill_type": fill_type,
                "gap_size": gap,
            })
    return filled


def _median_smooth_1d(values, window):
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = values.copy()
    for i in range(len(values)):
        s = max(0, i - half)
        e = min(len(values), i + half + 1)
        win = values[s:e]
        win = win[~np.isnan(win)]
        if win.size:
            out[i] = np.median(win)
    return out


def _interp_nan_1d(values):
    idx = np.arange(len(values))
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return values
    return np.interp(idx, idx[valid], values[valid]).astype(np.float32)


def _sanitize_sg_window(window, poly, n):
    if n < 3:
        return None
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    min_window = poly + 2
    if min_window % 2 == 0:
        min_window += 1
    if window < min_window:
        window = min_window
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window < min_window or window < 3:
        return None
    return window


def _savgol_coeffs(window, poly, deriv, delta):
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(x, N=poly + 1, increasing=True)
    pinv = np.linalg.pinv(A)
    coeffs = pinv[deriv] * math.factorial(deriv) / (delta ** deriv)
    return coeffs.astype(np.float32)


def _savgol_filter(values, window, poly, deriv, delta):
    coeffs = _savgol_coeffs(window, poly, deriv, delta)
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    out = np.convolve(padded, coeffs[::-1], mode="valid")
    return out.astype(np.float32)


def _unit_radial(x, y, center_x, center_y, eps):
    dx = x - center_x
    dy = y - center_y
    r = np.sqrt(dx * dx + dy * dy)
    ux = np.full_like(dx, math.nan, dtype=np.float32)
    uy = np.full_like(dy, math.nan, dtype=np.float32)
    valid = r > eps
    ux[valid] = dx[valid] / r[valid]
    uy[valid] = dy[valid] / r[valid]
    return ux, uy


def _central_diff(values, dt):
    n = len(values)
    out = np.full(n, math.nan, dtype=np.float32)
    if n < 2:
        return out
    for i in range(n):
        if i == 0:
            if not np.isnan(values[0]) and not np.isnan(values[1]):
                out[i] = (values[1] - values[0]) / dt
        elif i == n - 1:
            if not np.isnan(values[-1]) and not np.isnan(values[-2]):
                out[i] = (values[-1] - values[-2]) / dt
        else:
            if not np.isnan(values[i - 1]) and not np.isnan(values[i + 1]):
                out[i] = (values[i + 1] - values[i - 1]) / (2.0 * dt)
    return out


def _stationary_dominance_mask(speed, v_stop, window, ratio):
    if window <= 1:
        return np.zeros(len(speed), dtype=bool)
    if window % 2 == 0:
        window += 1
    half = window // 2
    mask = np.zeros(len(speed), dtype=bool)
    for i in range(len(speed)):
        s = max(0, i - half)
        e = min(len(speed), i + half + 1)
        win = speed[s:e]
        valid = ~np.isnan(win)
        if not valid.any():
            continue
        slow_ratio = (win[valid] < v_stop).sum() / float(valid.sum())
        if slow_ratio >= ratio:
            mask[i] = True
    return mask


def _valid_phys_mask(observed_mask, window):
    if window is None or window <= 1:
        return observed_mask.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    n = len(observed_mask)
    if n < window:
        return np.zeros(n, dtype=bool)
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        if not observed_mask[i]:
            continue
        if i < half or i >= (n - half):
            continue
        if observed_mask[i - half:i + half + 1].all():
            out[i] = True
    return out


def compute_kinematics(
    rows,
    fps,
    v_stop,
    k_stop,
    vmax,
    amax,
    use_world,
    smooth_window,
    smooth_method,
    sg_window,
    sg_poly,
    stat_window,
    stat_ratio,
    center_x,
    center_y,
    center_eps,
    drop_unreliable_phys,
):
    centers = []
    for row in rows:
        pts = row["world"] if use_world else row["pixel"]
        center = safe_mean(pts)
        centers.append(center if center is not None else np.array([math.nan, math.nan], dtype=np.float32))
    centers = np.asarray(centers, dtype=np.float32)
    x_raw = centers[:, 0]
    y_raw = centers[:, 1]
    dt = 1.0 / float(fps) if fps and fps > 0 else 1.0

    if smooth_method == "savgol":
        x_pref = _median_smooth_1d(x_raw, smooth_window)
        y_pref = _median_smooth_1d(y_raw, smooth_window)
        x_pref = _interp_nan_1d(x_pref)
        y_pref = _interp_nan_1d(y_pref)
        win = _sanitize_sg_window(sg_window, sg_poly, len(x_pref))
        if win is None:
            smooth_method = "median"
        else:
            x = _savgol_filter(x_pref, win, sg_poly, 0, 1.0)
            y = _savgol_filter(y_pref, win, sg_poly, 0, 1.0)
            x_vel = _savgol_filter(x_pref, win, sg_poly, 1, dt)
            y_vel = _savgol_filter(y_pref, win, sg_poly, 1, dt)
            x_acc = _savgol_filter(x_pref, win, sg_poly, 2, dt)
            y_acc = _savgol_filter(y_pref, win, sg_poly, 2, dt)

    if smooth_method == "median":
        x = _median_smooth_1d(x_raw, smooth_window)
        y = _median_smooth_1d(y_raw, smooth_window)
        x_vel = _central_diff(x, dt)
        y_vel = _central_diff(y, dt)
        if smooth_window > 1:
            x_vel = _median_smooth_1d(x_vel, smooth_window)
            y_vel = _median_smooth_1d(y_vel, smooth_window)
        x_acc = _central_diff(x_vel, dt)
        y_acc = _central_diff(y_vel, dt)

    speed = np.sqrt(x_vel ** 2 + y_vel ** 2)
    speed_raw = speed.copy()

    # Stationary gating
    is_stationary = np.zeros(len(rows), dtype=bool)
    if use_world and len(speed_raw) >= k_stop:
        slow = speed_raw < v_stop
        run_start = None
        for i, val in enumerate(slow):
            if val and run_start is None:
                run_start = i
            if (not val or i == len(slow) - 1) and run_start is not None:
                run_end = i if val else i - 1
                if (run_end - run_start + 1) >= k_stop:
                    is_stationary[run_start:run_end + 1] = True
                run_start = None
        dominant_mask = _stationary_dominance_mask(speed_raw, v_stop, stat_window, stat_ratio)
        is_stationary |= dominant_mask
        x_vel[is_stationary] = 0.0
        y_vel[is_stationary] = 0.0
        speed[is_stationary] = 0.0

    accel = np.sqrt(x_acc ** 2 + y_acc ** 2)

    if use_world:
        x_acc[is_stationary] = 0.0
        y_acc[is_stationary] = 0.0
        accel[is_stationary] = 0.0

    heading = np.degrees(np.arctan2(y_vel, x_vel))
    if use_world:
        heading[is_stationary] = math.nan

    if use_world:
        r_x, r_y = _unit_radial(x, y, center_x, center_y, center_eps)
        t_x = -r_y
        t_y = r_x
        v_t = x_vel * t_x + y_vel * t_y
        v_n = x_vel * r_x + y_vel * r_y
        a_t = x_acc * t_x + y_acc * t_y
        a_n = x_acc * r_x + y_acc * r_y
        valid = ~np.isnan(r_x) & ~np.isnan(r_y)
        v_t[is_stationary & valid] = 0.0
        v_n[is_stationary & valid] = 0.0
        a_t[is_stationary & valid] = 0.0
        a_n[is_stationary & valid] = 0.0
    else:
        r_x = np.full(len(rows), math.nan, dtype=np.float32)
        r_y = np.full(len(rows), math.nan, dtype=np.float32)
        t_x = np.full(len(rows), math.nan, dtype=np.float32)
        t_y = np.full(len(rows), math.nan, dtype=np.float32)
        v_t = np.full(len(rows), math.nan, dtype=np.float32)
        v_n = np.full(len(rows), math.nan, dtype=np.float32)
        a_t = np.full(len(rows), math.nan, dtype=np.float32)
        a_n = np.full(len(rows), math.nan, dtype=np.float32)

    phys_violation = (speed > vmax) | (accel > amax)
    return (
        x_raw,
        y_raw,
        x_vel,
        y_vel,
        speed,
        x_acc,
        y_acc,
        accel,
        heading,
        is_stationary,
        phys_violation,
        r_x,
        r_y,
        t_x,
        t_y,
        v_t,
        v_n,
        a_t,
        a_n,
        x,
        y,
    )


def main():
    args = parse_args()
    pkl_path = Path(args.pkl)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    traj_info = data.get("traj_info", [])
    fps = args.fps if args.fps is not None else data.get("output_info", {}).get("output_fps", None)
    if not traj_info:
        raise RuntimeError("traj_info is empty.")

    tracks, frame_meta, has_world, has_lane = build_tracks(traj_info)
    records = []
    for track_id, frame_map in tracks.items():
        frames = sorted(frame_map.keys())
        if not frames:
            continue
        f_min, f_max = frames[0], frames[-1]
        majority_cat = majority_category([v["category"] for v in frame_map.values()])
        is_vehicle = majority_cat in VEHICLE_CLASS_IDS
        vmax = args.vmax_vehicle if is_vehicle else args.vmax_vru
        amax = args.amax_vehicle if is_vehicle else args.amax_vru

        filled = fill_track(frame_map, range(f_min, f_max + 1), majority_cat, args.g1, args.g2, args.g3)
        rows = [filled[f] for f in range(f_min, f_max + 1)]

        use_world = has_world and all(r["world"] is not None for r in rows)
        (x_c, y_c, x_vel, y_vel, speed, x_acc, y_acc, accel, heading,
         is_stationary, phys_violation, r_x, r_y, t_x, t_y, v_t, v_n, a_t, a_n,
         x_sm, y_sm) = \
            compute_kinematics(
                rows,
                fps,
                args.v_stop,
                args.k_stop,
                vmax,
                amax,
                use_world,
                args.smooth_window,
                args.smooth_method,
                args.sg_window,
                args.sg_poly,
                args.stat_window,
                args.stat_ratio,
                args.center_x,
                args.center_y,
                args.center_eps,
                args.drop_unreliable_phys,
            )

        observed_mask = np.array([bool(r["is_observed"]) for r in rows], dtype=bool)
        if args.smooth_method == "savgol":
            valid_window = _sanitize_sg_window(args.sg_window, args.sg_poly, len(rows))
        else:
            valid_window = args.smooth_window
        valid_phys_mask = _valid_phys_mask(observed_mask, valid_window)
        for idx, f in enumerate(range(f_min, f_max + 1)):
            meta = frame_meta.get(f, {})
            row = rows[idx]
            pix = row["pixel"]
            world = row["world"]
            valid_phys = bool(valid_phys_mask[idx]) if use_world else False
            rec = {
                "recording_id": pkl_path.stem,
                "track_id": track_id,
                "category": majority_cat,
                "category_name": CATEGORY_NAMES[majority_cat] if 0 <= majority_cat < len(CATEGORY_NAMES) else "unknown",
                "is_observed": bool(row["is_observed"]),
                "fill_type": row["fill_type"],
                "gap_size": row["gap_size"],
                "frame_index": f,
                "output_frame": meta.get("output_frame", math.nan),
                "frame_time": meta.get("frame_time", None),
                "score": row["score"],
                "xCenter_world": x_c[idx] if use_world else math.nan,
                "yCenter_world": y_c[idx] if use_world else math.nan,
                "xCenter_px": math.nan,
                "yCenter_px": math.nan,
                "xVelocity": x_vel[idx] if use_world else math.nan,
                "yVelocity": y_vel[idx] if use_world else math.nan,
                "speed": speed[idx] if use_world else math.nan,
                "xAcceleration": x_acc[idx] if use_world else math.nan,
                "yAcceleration": y_acc[idx] if use_world else math.nan,
                "accel": accel[idx] if use_world else math.nan,
                "radial_x": r_x[idx] if use_world else math.nan,
                "radial_y": r_y[idx] if use_world else math.nan,
                "tangent_x": t_x[idx] if use_world else math.nan,
                "tangent_y": t_y[idx] if use_world else math.nan,
                "v_tangent": v_t[idx] if use_world else math.nan,
                "v_normal": v_n[idx] if use_world else math.nan,
                "a_tangent": a_t[idx] if use_world else math.nan,
                "a_normal": a_n[idx] if use_world else math.nan,
                "heading": heading[idx] if use_world else math.nan,
                "is_stationary": bool(is_stationary[idx]) if use_world else False,
                "phys_violation": bool(phys_violation[idx]) if use_world else False,
                "valid_phys": bool(valid_phys) if use_world else False,
            }
            if args.drop_unreliable_phys and use_world and not valid_phys:
                rec["xVelocity"] = math.nan
                rec["yVelocity"] = math.nan
                rec["speed"] = math.nan
                rec["xAcceleration"] = math.nan
                rec["yAcceleration"] = math.nan
                rec["accel"] = math.nan
                rec["v_tangent"] = math.nan
                rec["v_normal"] = math.nan
                rec["a_tangent"] = math.nan
                rec["a_normal"] = math.nan
            if pix is not None:
                rec.update({
                    "x1_px": pix[0, 0], "y1_px": pix[0, 1],
                    "x2_px": pix[1, 0], "y2_px": pix[1, 1],
                    "x3_px": pix[2, 0], "y3_px": pix[2, 1],
                    "x4_px": pix[3, 0], "y4_px": pix[3, 1],
                })
                rec["xCenter_px"] = float(pix[:, 0].mean())
                rec["yCenter_px"] = float(pix[:, 1].mean())
            if world is not None:
                rec.update({
                    "x1_world": world[0, 0], "y1_world": world[0, 1],
                    "x2_world": world[1, 0], "y2_world": world[1, 1],
                    "x3_world": world[2, 0], "y3_world": world[2, 1],
                    "x4_world": world[3, 0], "y4_world": world[3, 1],
                })
            if has_lane:
                rec["lane_id"] = row["lane_id"]
            records.append(rec)

    if not records:
        raise RuntimeError("No records to export.")

    df = pd.DataFrame(records)
    output_dir = Path(args.output_dir) if args.output_dir else pkl_path.parent / "saving"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pkl_path.stem}_filled.xlsx"
    df.to_excel(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
