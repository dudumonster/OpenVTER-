#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit app to explore track trajectories and kinematics from *_filled.xlsx.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def parse_args():
    parser = argparse.ArgumentParser(description="Track visualization app.")
    parser.add_argument("--xlsx", required=True, help="Path to *_filled.xlsx")
    parser.add_argument("--sheet", default=0, help="Excel sheet name or index")
    args, _ = parser.parse_known_args()
    return args


def _safe_unique(series):
    return sorted([v for v in series.dropna().unique()])


def _available_time_cols(df):
    candidates = ["frame_time", "frame_index", "output_frame"]
    cols = []
    for col in candidates:
        if col in df.columns and not df[col].isna().all():
            cols.append(col)
    return cols


def main():
    st.set_page_config(page_title="Track Explorer", layout="wide")
    args = parse_args()
    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        st.error(f"File not found: {xlsx_path}")
        return

    try:
        df = pd.read_excel(xlsx_path, sheet_name=args.sheet)
    except ImportError as exc:
        st.error("Missing dependency for Excel. Please install openpyxl.")
        st.exception(exc)
        return

    if "track_id" not in df.columns:
        st.error("Missing required column: track_id")
        return

    df = df.dropna(subset=["track_id"]).copy()
    df["track_id"] = df["track_id"].astype(int)
    df["track_id_str"] = df["track_id"].astype(str)

    if "frame_time" in df.columns:
        df["frame_time"] = pd.to_numeric(df["frame_time"], errors="coerce")

    st.sidebar.header("Filters")
    time_cols = _available_time_cols(df)
    if not time_cols:
        df["row_index"] = np.arange(len(df))
        time_cols = ["row_index"]
    time_col = st.sidebar.selectbox("Time axis", time_cols, index=0)

    coord_options = []
    if {"xCenter_world", "yCenter_world"}.issubset(df.columns):
        coord_options.append("world")
    if {"xCenter_px", "yCenter_px"}.issubset(df.columns):
        coord_options.append("pixel")
    if not coord_options:
        st.error("Missing center columns (world or pixel).")
        return
    coord = st.sidebar.radio("Coordinate", coord_options, index=0)
    if coord == "world":
        x_col, y_col = "xCenter_world", "yCenter_world"
    else:
        x_col, y_col = "xCenter_px", "yCenter_px"

    categories = _safe_unique(df["category_name"]) if "category_name" in df.columns else []
    if categories:
        selected_categories = st.sidebar.multiselect(
            "Category", categories, default=categories
        )
        df = df[df["category_name"].isin(selected_categories)]

    fill_types = _safe_unique(df["fill_type"]) if "fill_type" in df.columns else []
    if fill_types:
        selected_fill = st.sidebar.multiselect("Fill type", fill_types, default=fill_types)
        df = df[df["fill_type"].isin(selected_fill)]

    if "is_observed" in df.columns:
        observed_only = st.sidebar.checkbox("Observed only", value=False)
        if observed_only:
            df = df[df["is_observed"] == True]

    track_ids = _safe_unique(df["track_id"])
    default_ids = track_ids[: min(10, len(track_ids))]
    selected_ids = st.sidebar.multiselect("Track IDs", track_ids, default=default_ids)
    if selected_ids:
        df = df[df["track_id"].isin(selected_ids)]

    st.sidebar.markdown("---")
    metric_candidates = [
        "speed",
        "accel",
        "xVelocity",
        "yVelocity",
        "xAcceleration",
        "yAcceleration",
        "v_tangent",
        "v_normal",
        "a_tangent",
        "a_normal",
    ]
    available_metrics = [m for m in metric_candidates if m in df.columns]
    default_metrics = [m for m in ["speed", "accel"] if m in available_metrics]
    selected_metrics = st.sidebar.multiselect(
        "Metrics", available_metrics, default=default_metrics
    )

    st.title("Track Explorer")
    st.caption(f"File: {xlsx_path.name}")

    if df.empty:
        st.warning("No data after filtering.")
        return

    df = df.sort_values(["track_id", time_col])

    st.subheader("Trajectory")
    traj_fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color="track_id_str",
        hover_data=[time_col, "track_id_str"],
    )
    traj_fig.update_yaxes(scaleanchor="x", scaleratio=1)
    traj_fig.update_layout(height=600, legend_title_text="track_id")
    st.plotly_chart(traj_fig, use_container_width=True)

    if selected_metrics:
        st.subheader("Metrics Over Time")
        long_df = df.melt(
            id_vars=[time_col, "track_id_str"],
            value_vars=selected_metrics,
            var_name="metric",
            value_name="value",
        )
        metric_fig = px.line(
            long_df,
            x=time_col,
            y="value",
            color="track_id_str",
            facet_row="metric",
        )
        metric_fig.update_yaxes(matches=None)
        metric_fig.update_layout(height=250 * len(selected_metrics))
        st.plotly_chart(metric_fig, use_container_width=True)


if __name__ == "__main__":
    main()
