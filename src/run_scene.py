#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import math
import os
import re
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback for local checks.
    fcntl = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import load_config, run_config
from utils import RoadConfig
from utils.resource_monitor import ResourceMonitor, configure_runtime_threads


VIDEO_SUFFIXES = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV"}
TRAILING_NUMBER_RE = re.compile(r"(\d+)$")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenVTER over every video in one scene.")
    parser.add_argument("--config", required=True, help="Base YAML config to use as a template.")
    parser.add_argument("--scene-dir", required=True, help="Directory that contains scene videos.")
    parser.add_argument(
        "--road-config-dir",
        required=True,
        help="Directory that contains per-video road config JSON files.",
    )
    parser.add_argument(
        "--road-config-pattern",
        default="background_{video_stem}.json",
        help="Pattern used to map a video stem to its road config filename.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional scene output directory. Defaults to <scene-dir>/<base-output-dir-name>.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Only process one specific video filename or stem for quick testing.",
    )
    parser.add_argument(
        "--video-range",
        default=None,
        help="Only process videos whose trailing numeric id falls in a range such as 001-005.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run videos even if their final outputs already exist.",
    )
    args = parser.parse_args()
    if args.video and args.video_range:
        parser.error("--video and --video-range cannot be used together.")
    return args


def _resolve_config_path(config_arg: str) -> Path:
    config_path = Path(config_arg).expanduser()
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    return config_path


def _scene_output_dir(
    base_config: dict[str, Any],
    scene_dir: Path,
    explicit_output_dir: str | None,
) -> Path:
    if explicit_output_dir:
        return Path(explicit_output_dir).expanduser()

    base_output_dir = Path(str(base_config["output_dir"]))
    return scene_dir / base_output_dir.name


def _parse_video_range(video_range: str | None) -> tuple[int, int] | None:
    if video_range is None:
        return None
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", video_range)
    if not match:
        raise ValueError("video range must look like 001-005 or 1-5.")
    start = int(match.group(1))
    end = int(match.group(2))
    if start > end:
        raise ValueError("video range start must be less than or equal to end.")
    return start, end


def _extract_trailing_number(video_path: Path) -> int | None:
    match = TRAILING_NUMBER_RE.search(video_path.stem)
    if not match:
        return None
    return int(match.group(1))


def _list_scene_videos(
    scene_dir: Path,
    selected_video: str | None,
    selected_range: tuple[int, int] | None,
) -> list[Path]:
    video_files = sorted(
        file_path
        for file_path in scene_dir.iterdir()
        if file_path.is_file() and file_path.suffix in VIDEO_SUFFIXES
    )
    if selected_video is not None:
        selected_lower = selected_video.lower()
        filtered = [
            file_path
            for file_path in video_files
            if file_path.name.lower() == selected_lower
            or file_path.stem.lower() == selected_lower
        ]
        if not filtered:
            raise FileNotFoundError(
                f"Video '{selected_video}' was not found under scene directory: {scene_dir}"
            )
        return filtered

    if selected_range is None:
        return video_files

    start, end = selected_range
    filtered = []
    for file_path in video_files:
        video_number = _extract_trailing_number(file_path)
        if video_number is not None and start <= video_number <= end:
            filtered.append(file_path)
    if not filtered:
        raise FileNotFoundError(
            f"No videos matched numeric range {start:03d}-{end:03d} under "
            f"scene directory: {scene_dir}"
        )
    return filtered


def _road_config_for_video(road_config_dir: Path, pattern: str, video_path: Path) -> Path:
    road_config_name = pattern.format(
        video_stem=video_path.stem,
        video_name=video_path.name,
        video_suffix=video_path.suffix,
    )
    road_config_path = road_config_dir / road_config_name
    if not road_config_path.exists():
        raise FileNotFoundError(
            f"Road config not found for video '{video_path.name}': {road_config_path}"
        )
    return road_config_path


def _validate_road_config(road_config_path: Path, video_path: Path) -> None:
    road_config = RoadConfig.fromfile(str(road_config_path))
    length_per_pixel = road_config.get("length_per_pixel")
    if length_per_pixel is None:
        raise ValueError(
            f"Road config for video '{video_path.name}' has no valid length_* "
            f"calibration; length_per_pixel is required: {road_config_path}"
        )
    try:
        length_value = float(length_per_pixel)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Road config for video '{video_path.name}' has invalid "
            f"length_per_pixel={length_per_pixel!r}: {road_config_path}"
        ) from exc
    if not math.isfinite(length_value) or length_value <= 0:
        raise ValueError(
            f"Road config for video '{video_path.name}' has invalid "
            f"length_per_pixel={length_per_pixel!r}: {road_config_path}"
        )


def _video_output_dir(scene_output_dir: Path, video_path: Path) -> Path:
    return scene_output_dir / video_path.stem


def _expected_outputs(
    scene_output_dir: Path,
    video_path: Path,
    config: dict[str, Any],
) -> dict[str, Path]:
    video_output_dir = _video_output_dir(scene_output_dir, video_path)
    pipeline_name = "_".join(config.get("pipeline", ["det"]))
    return {
        "video_output_dir": video_output_dir,
        "stabilize_file": video_output_dir / f"{video_path.stem}_stab.pkl",
        "tracking_video": video_output_dir
        / f"tracking_output_{pipeline_name}_{video_path.stem}.mp4",
        "det_result": video_output_dir / f"det_bbox_result_{video_path.stem}.pkl",
    }


def _is_video_complete(expected_outputs: dict[str, Path]) -> bool:
    return expected_outputs["tracking_video"].exists() and expected_outputs["det_result"].exists()


def _build_video_config(
    base_config: dict[str, Any],
    *,
    scene_dir: Path,
    scene_output_dir: Path,
    road_config_path: Path,
    video_path: Path,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config["data_dir"] = str(scene_dir)
    config["video_file"] = [str(video_path)]
    config["road_config"] = str(road_config_path)
    config["output_dir"] = str(scene_output_dir)
    config["save_folder"] = str(scene_output_dir)
    config["stabilize_file"] = f"{video_path.stem}_stab.pkl"
    return config


def _setup_scene_logger(log_dir: Path, scene_name: str) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"scene_{scene_name}_{timestamp}.log"

    logger = logging.getLogger(f"run_scene.{scene_name}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger, log_path


def _append_status(status_path: Path, payload: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("a", encoding="utf-8") as fh:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            fh.flush()
        finally:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def main() -> None:
    faulthandler.enable()
    args = parse_args()
    config_path = _resolve_config_path(args.config)
    base_config = load_config(config_path)
    thread_status = configure_runtime_threads(base_config.get("cpu_threads"))
    monitor = ResourceMonitor(base_config.get("monitor_interval"))
    selected_range = _parse_video_range(args.video_range)
    force_rerun = args.force or _env_flag("FORCE_RERUN")

    scene_dir = Path(args.scene_dir).expanduser()
    road_config_dir = Path(args.road_config_dir).expanduser()
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")
    if not road_config_dir.exists():
        raise FileNotFoundError(f"Road config directory does not exist: {road_config_dir}")

    scene_output_dir = _scene_output_dir(base_config, scene_dir, args.output_dir)
    scene_name = scene_dir.name
    scene_log_dir = Path(str(base_config["log_dir"])) / scene_name
    logger, log_path = _setup_scene_logger(scene_log_dir, scene_name)

    status_path = scene_output_dir / f"scene_status_{scene_name}.jsonl"
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Project root   : %s", PROJECT_ROOT)
    logger.info("Template config: %s", config_path)
    logger.info("Scene dir      : %s", scene_dir)
    logger.info("Road config dir: %s", road_config_dir)
    logger.info("Scene output   : %s", scene_output_dir)
    logger.info("Scene log file : %s", log_path)
    logger.info("Status file    : %s", status_path)
    logger.info("Runtime threads: %s", thread_status)
    logger.info("%s", monitor.format(stage="scene_start"))
    if args.video:
        logger.info("Single-video mode enabled for: %s", args.video)
    if selected_range:
        logger.info(
            "Video-range mode enabled for ids: %03d-%03d",
            selected_range[0],
            selected_range[1],
        )
    if force_rerun:
        logger.info(
            "Force rerun enabled: completed videos will be re-run and existing outputs "
            "will be overwritten if generated again."
        )
        if _env_flag("FORCE_RERUN"):
            logger.info("FORCE_RERUN=1, existing outputs will be overwritten if generated again.")

    video_files = _list_scene_videos(scene_dir, args.video, selected_range)
    if len(video_files) == 0:
        raise FileNotFoundError(f"No video files were found in scene directory: {scene_dir}")

    logger.info("Discovered %d video(s) to inspect.", len(video_files))

    processed = 0
    skipped = 0
    failed = 0

    for index, video_path in enumerate(video_files, start=1):
        road_config_path = _road_config_for_video(
            road_config_dir,
            args.road_config_pattern,
            video_path,
        )
        video_config = _build_video_config(
            base_config,
            scene_dir=scene_dir,
            scene_output_dir=scene_output_dir,
            road_config_path=road_config_path,
            video_path=video_path,
        )
        expected_outputs = _expected_outputs(scene_output_dir, video_path, video_config)

        if not force_rerun and _is_video_complete(expected_outputs):
            skipped += 1
            logger.info("[%d/%d] Skip completed video: %s", index, len(video_files), video_path.name)
            _append_status(
                status_path,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "scene": scene_name,
                    "video": video_path.name,
                    "status": "skipped",
                    "reason": "outputs already exist",
                    "tracking_video": str(expected_outputs["tracking_video"]),
                    "det_result": str(expected_outputs["det_result"]),
                },
            )
            continue

        logger.info("[%d/%d] Start video: %s", index, len(video_files), video_path.name)
        logger.info("           Road config : %s", road_config_path)
        logger.info("           Output dir  : %s", expected_outputs["video_output_dir"])
        logger.info("%s", monitor.format(stage="video_start", video=video_path.name))
        _append_status(
            status_path,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "scene": scene_name,
                "video": video_path.name,
                "status": "running",
                "road_config": str(road_config_path),
                "output_dir": str(expected_outputs["video_output_dir"]),
            },
        )

        try:
            _validate_road_config(road_config_path, video_path)
            run_config(video_config, config_path=config_path, echo_paths=False)
        except Exception as exc:
            failed += 1
            logger.exception("[%d/%d] Failed video: %s", index, len(video_files), video_path.name)
            logger.info("%s", monitor.format(stage="video_failed", video=video_path.name))
            _append_status(
                status_path,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "scene": scene_name,
                    "video": video_path.name,
                    "status": "failed",
                    "error": repr(exc),
                    "output_dir": str(expected_outputs["video_output_dir"]),
                },
            )
            continue

        if _is_video_complete(expected_outputs):
            processed += 1
            logger.info("[%d/%d] Completed video: %s", index, len(video_files), video_path.name)
            logger.info("%s", monitor.format(stage="video_done", video=video_path.name))
            _append_status(
                status_path,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "scene": scene_name,
                    "video": video_path.name,
                    "status": "done",
                    "tracking_video": str(expected_outputs["tracking_video"]),
                    "det_result": str(expected_outputs["det_result"]),
                    "stabilize_file": str(expected_outputs["stabilize_file"]),
                },
            )
        else:
            failed += 1
            logger.error(
                "[%d/%d] Video finished but outputs are incomplete: %s",
                index,
                len(video_files),
                video_path.name,
            )
            _append_status(
                status_path,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "scene": scene_name,
                    "video": video_path.name,
                    "status": "failed",
                    "error": "expected outputs missing after run",
                    "tracking_video": str(expected_outputs["tracking_video"]),
                    "det_result": str(expected_outputs["det_result"]),
                },
            )

    logger.info(
        "Scene summary: total=%d done=%d skipped=%d failed=%d",
        len(video_files),
        processed,
        skipped,
        failed,
    )
    logger.info("%s", monitor.format(stage="scene_end"))
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
