#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PATH_KEYS = {
    "data_dir",
    "output_dir",
    "log_dir",
    "checkpoint_dir",
    "video_folder",
    "video_file",
    "road_config",
    "save_folder",
    "checkpoint",
    "checkpoint_jit",
    "repo_dir",
    "cfg",
    "weights",
    "stabilize_frame",
}
VARIABLE_RE = re.compile(r"\$\{([^}]+)\}")
FULL_VARIABLE_RE = re.compile(r"^\$\{([^}]+)\}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified OpenVTER launcher")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def _replace_vars(value: str, context: dict[str, Any]) -> Any:
    full_match = FULL_VARIABLE_RE.match(value)
    if full_match:
        key = full_match.group(1)
        if key in context and context[key] is not None:
            return context[key]

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context or context[key] is None:
            return match.group(0)
        return str(context[key])

    previous = None
    current = value
    while previous != current:
        previous = current
        current = VARIABLE_RE.sub(repl, current)
    return current


def _interpolate(data: Any, context: dict[str, Any]) -> Any:
    if isinstance(data, dict):
        result: dict[str, Any] = {}
        local_context = dict(context)
        for key, value in data.items():
            result[key] = _interpolate(value, local_context)
            if not isinstance(result[key], (dict, list)):
                local_context[key] = result[key]
        return result
    if isinstance(data, list):
        return [_interpolate(item, context) for item in data]
    if isinstance(data, str):
        return _replace_vars(data, context)
    return data


def _resolve_path(raw_value: str, config_dir: Path) -> str:
    candidate = Path(os.path.expanduser(raw_value))
    if candidate.is_absolute():
        return str(candidate)
    if raw_value.startswith((".", "..")):
        return str((config_dir / candidate).resolve())
    config_relative = config_dir / candidate
    if config_relative.exists():
        return str(config_relative.resolve())
    return str((PROJECT_ROOT / candidate).resolve())


def _normalize_paths(data: Any, config_dir: Path, key_name: str | None = None) -> Any:
    if isinstance(data, dict):
        return {key: _normalize_paths(value, config_dir, key) for key, value in data.items()}
    if isinstance(data, list):
        if key_name == "video_file":
            return [
                _resolve_path(item, config_dir) if isinstance(item, str) else item
                for item in data
            ]
        return [_normalize_paths(item, config_dir, key_name) for item in data]
    if isinstance(data, str) and key_name in PATH_KEYS:
        return _resolve_path(data, config_dir)
    return data


def _ensure_defaults(config: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    result.setdefault("task", "video_inference")
    result.setdefault("step", 3)
    result.setdefault("config_parameter", 1)
    result.setdefault("multiprocessing", False)
    result.setdefault("workflow_steps", [result["step"]])

    if "output_dir" not in result and "save_folder" in result:
        result["output_dir"] = result["save_folder"]
    if "save_folder" not in result and "output_dir" in result:
        result["save_folder"] = result["output_dir"]
    if "log_dir" not in result:
        result["log_dir"] = str(PROJECT_ROOT / "logs" / Path(result["output_dir"]).name)
    if "checkpoint_dir" not in result:
        result["checkpoint_dir"] = str(PROJECT_ROOT / "checkpoints")
    if "batch_size" in result and "inference_batch_size" not in result:
        result["inference_batch_size"] = result["batch_size"]

    device = result.get("device")
    detection = result.get("detection")
    if device and isinstance(detection, dict) and "device_name" not in detection:
        detection["device_name"] = device
    if device and isinstance(detection, list):
        for item in detection:
            if isinstance(item, dict) and "device_name" not in item:
                item["device_name"] = device
    return result


def _get_video_files(config: dict[str, Any]) -> list[str]:
    if "video_file" in config:
        video_file = config.get("video_file")
        if isinstance(video_file, list):
            return video_file
        if video_file is None:
            return []
        return [video_file]
    if "first_video_name" in config and "video_num" in config:
        video_folder = str(config.get("video_folder"))
        first_video_name = str(config.get("first_video_name"))
        video_num = int(config.get("video_num"))
        return [
            str(Path(video_folder) / first_video_name.format(i + 1))
            for i in range(video_num)
        ]
    video_folder = str(config.get("video_folder"))
    video_name_ls = config.get("video_name") or []
    return [str(Path(video_folder) / str(video_name)) for video_name in video_name_ls]


def _get_stabilize_output_path(config: dict[str, Any]) -> Path | None:
    save_folder = config.get("save_folder")
    stabilize_file = config.get("stabilize_file")
    video_files = _get_video_files(config)
    if not save_folder or not stabilize_file or not video_files:
        return None

    first_video_name = Path(video_files[0]).stem
    if len(video_files) == 1:
        output_folder = Path(str(save_folder)) / first_video_name
    else:
        output_folder = Path(str(save_folder)) / f"{first_video_name}_Num_{len(video_files)}"
    return output_folder / str(stabilize_file)


def _get_runtime_config_dir(config: dict[str, Any]) -> Path:
    save_folder = config.get("save_folder") or config.get("output_dir")
    video_files = _get_video_files(config)
    if not save_folder or not video_files:
        return Path(str(config["output_dir"]))

    first_video_name = Path(video_files[0]).stem
    if len(video_files) == 1:
        return Path(str(save_folder)) / first_video_name
    return Path(str(save_folder)) / f"{first_video_name}_Num_{len(video_files)}"


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        raw_config = yaml.safe_load(fh) or {}
    if not isinstance(raw_config, dict):
        raise ValueError("Top-level YAML config must be a mapping.")

    interpolated = _interpolate(raw_config, {"project_root": PROJECT_ROOT})
    normalized = _normalize_paths(interpolated, config_path.parent)
    return _ensure_defaults(normalized)


def prepare_runtime_config(config: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    output_dir = Path(config["output_dir"])
    log_dir = Path(config["log_dir"])
    checkpoint_dir = Path(config["checkpoint_dir"])
    runtime_config_dir = _get_runtime_config_dir(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    runtime_config_path = runtime_config_dir / "runtime_config.json"
    tmp_path = runtime_config_dir / f".runtime_config.{os.getpid()}.tmp"
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, runtime_config_path)

    return runtime_config_path, output_dir, log_dir, checkpoint_dir


def validate_task(config: dict[str, Any]) -> None:
    task = config.get("task", "video_inference")
    if task != "video_inference":
        raise ValueError(
            f"Unsupported task '{task}'. "
            "This wrapper currently standardizes the OpenVTER video_inference flow."
        )


def run_config(
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
    echo_paths: bool = True,
) -> tuple[Path, Path, Path, Path]:
    runtime_config_path, output_dir, log_dir, checkpoint_dir = prepare_runtime_config(config)

    if echo_paths:
        print(f"Project root   : {PROJECT_ROOT}")
        if config_path is not None:
            print(f"Config file    : {config_path}")
        print(f"Runtime config : {runtime_config_path}")
        print(f"Output dir     : {output_dir}")
        print(f"Log dir        : {log_dir}")
        print(f"Checkpoint dir : {checkpoint_dir}")

    validate_task(config)

    from video_inference_main import run_pipeline

    workflow_steps = config.get("workflow_steps", [config.get("step", 3)])
    if not isinstance(workflow_steps, list) or len(workflow_steps) == 0:
        raise ValueError("workflow_steps must be a non-empty list.")

    for step in workflow_steps:
        step_int = int(step)
        if step_int == 1:
            stabilize_output_path = _get_stabilize_output_path(config)
            if stabilize_output_path is not None and stabilize_output_path.exists():
                print(
                    "Skipping workflow step 1: "
                    f"found existing stabilize file at {stabilize_output_path}"
                )
                continue
        print(f"Running workflow step: {step_int}")
        run_pipeline(
            config_path=str(runtime_config_path),
            step=step_int,
            config_parameter=int(config.get("config_parameter", 1)),
            multiprocessing=bool(config.get("multiprocessing", False)),
        )

    return runtime_config_path, output_dir, log_dir, checkpoint_dir


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    config = load_config(config_path)
    run_config(config, config_path=config_path)


if __name__ == "__main__":
    main()
