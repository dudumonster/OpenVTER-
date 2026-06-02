#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        print(f"[BAD] {path}: cannot read JSON: {exc}")
        return None
    if not isinstance(data, dict):
        print(f"[BAD] {path}: top-level JSON is not an object")
        return None
    return data


def _video_stem_from_config(config: dict[str, Any]) -> str | None:
    video_file = config.get("video_file")
    if isinstance(video_file, list) and video_file:
        return Path(str(video_file[0])).stem
    if isinstance(video_file, str):
        return Path(video_file).stem
    return None


def _video_stems_from_pkl(path: Path) -> list[str]:
    try:
        with path.open("rb") as fh:
            data = pickle.load(fh)
    except Exception as exc:
        return [f"<unreadable pkl: {exc}>"]

    stems: list[str] = []
    if isinstance(data, dict):
        for item in data.get("video_info", []):
            if not isinstance(item, dict):
                continue
            video_name = item.get("video_name")
            if isinstance(video_name, list):
                stems.extend(Path(str(name)).stem for name in video_name)
            elif isinstance(video_name, str):
                stems.append(Path(video_name).stem)
    return stems


def check_video_dir(video_dir: Path) -> list[str]:
    issues: list[str] = []
    expected_stem = video_dir.name

    runtime_config_path = video_dir / "runtime_config.json"
    if not runtime_config_path.exists():
        issues.append("missing runtime_config.json")
    else:
        config = _load_json(runtime_config_path)
        if config is None:
            issues.append("runtime_config.json is unreadable")
        else:
            config_video_stem = _video_stem_from_config(config)
            if config_video_stem != expected_stem:
                issues.append(
                    "runtime_config video_file mismatch: "
                    f"expected {expected_stem}, got {config_video_stem}"
                )
            road_config = str(config.get("road_config", ""))
            if expected_stem not in Path(road_config).stem:
                issues.append(
                    "runtime_config road_config mismatch: "
                    f"expected stem containing {expected_stem}, got {road_config}"
                )

    det_path = video_dir / f"det_bbox_result_{expected_stem}.pkl"
    if not det_path.exists():
        issues.append(f"missing {det_path.name}")
    else:
        pkl_stems = _video_stems_from_pkl(det_path)
        if pkl_stems and expected_stem not in pkl_stems:
            issues.append(
                "det pkl video_info mismatch: "
                f"expected {expected_stem}, got {', '.join(pkl_stems)}"
            )

    tracking_files = list(video_dir.glob(f"tracking_output_*_{expected_stem}.mp4"))
    if not tracking_files:
        issues.append(f"missing tracking_output_*_{expected_stem}.mp4")

    foreign_runtime = [
        path
        for path in video_dir.glob("runtime_config*.json")
        if path.name != "runtime_config.json"
    ]
    if foreign_runtime:
        issues.append(
            "unexpected extra runtime config files: "
            + ", ".join(path.name for path in foreign_runtime)
        )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check OpenVTER per-video outputs for config/result stem mismatches."
    )
    parser.add_argument("scene_output_dir", help="Scene output directory, such as output_server39.")
    args = parser.parse_args()

    output_dir = Path(args.scene_output_dir).expanduser()
    if not output_dir.exists():
        print(f"[BAD] output directory does not exist: {output_dir}")
        return 2

    video_dirs = sorted(path for path in output_dir.iterdir() if path.is_dir())
    if not video_dirs:
        print(f"[BAD] no per-video directories found under: {output_dir}")
        return 2

    issue_count = 0
    for video_dir in video_dirs:
        issues = check_video_dir(video_dir)
        if issues:
            issue_count += len(issues)
            print(f"[BAD] {video_dir.name}")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print(f"[OK]  {video_dir.name}")

    print(f"\nChecked {len(video_dirs)} video directories; issues: {issue_count}")
    return 1 if issue_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
