#!/usr/bin/env python3
"""Validate v2 pipeline readiness and print a dry-run plan.

Checks that paths, config, and data are in place before running the full
VLM-assisted pseudo OBB pipeline.  Does *not* load models or run inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml


# Import from v2 schema module
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from dataset_construction.scripts.vlm_pseudo_obb_v2_schema import (
        ASPECT_PRIORS,
        GLOBAL_NAMES,
        VISDRONE_TO_GLOBAL,
    )
    from dataset_construction.scripts.vlm_prompt_templates import get_all_prompts
except ImportError:
    VISDRONE_TO_GLOBAL = {
        3: (1, "bicycle"), 7: (3, "tricycle"),
        8: (4, "awning_tricycle"), 10: (2, "motor"),
    }
    GLOBAL_NAMES = {0: "motor_vehicle", 1: "bicycle", 2: "motor",
                    3: "tricycle", 4: "awning_tricycle"}
    ASPECT_PRIORS = {"bicycle": (1.2, 8.0), "motor": (1.1, 8.0),
                     "tricycle": (1.0, 6.0), "awning_tricycle": (1.0, 5.0)}


def resolve_split_dir(raw_root: Path, split: str) -> Path:
    direct = raw_root / f"VisDrone2019-DET-{split}"
    nested = direct / f"VisDrone2019-DET-{split}"
    for candidate in (direct, nested):
        if (candidate / "images").is_dir() and (candidate / "annotations").is_dir():
            return candidate
    return direct


def check_path(label: str, path: str | None, required: bool = False) -> dict:
    if not path:
        return {"label": label, "status": "not_configured", "path": ""}
    p = Path(path)
    if p.exists():
        if p.is_file():
            size_mb = round(p.stat().st_size / 1e6, 1)
            return {"label": label, "status": "ok", "path": str(p), "size_mb": size_mb}
        return {"label": label, "status": "ok", "path": str(p)}
    return {"label": label, "status": "missing" if required else "not_found", "path": str(p)}


def check_imports() -> list[dict]:
    results = []
    for mod, desc in [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("yaml", "PyYAML"),
    ]:
        try:
            __import__(mod)
            results.append({"module": mod, "desc": desc, "status": "ok"})
        except ImportError:
            results.append({"module": mod, "desc": desc, "status": "missing"})

    # Optional: SAM2, GroundingDINO
    for mod, desc in [
        ("sam2", "SAM2"),
        ("groundingdino", "GroundingDINO"),
        ("transformers", "HuggingFace Transformers (Florence-2)"),
        ("ultralytics", "Ultralytics (YOLO-World)"),
    ]:
        try:
            __import__(mod)
            results.append({"module": mod, "desc": desc, "status": "ok"})
        except ImportError:
            results.append({"module": mod, "desc": desc, "status": "not_installed"})

    return results


def count_targets(raw_root: Path, splits: list[str]) -> dict:
    counts: dict[str, dict] = {}
    for split in splits:
        split_dir = resolve_split_dir(raw_root, split)
        ann_dir = split_dir / "annotations"
        if not ann_dir.is_dir():
            counts[split] = {"error": f"annotations dir not found: {ann_dir}"}
            continue
        class_counts: Counter = Counter()
        total_images = 0
        for ann_path in sorted(ann_dir.glob("*.txt")):
            total_images += 1
            for line in ann_path.read_text(errors="ignore").splitlines():
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                try:
                    cid = int(parts[5])
                except ValueError:
                    continue
                class_counts[cid] += 1
        targets = {}
        for cid, (gid, gname) in VISDRONE_TO_GLOBAL.items():
            targets[gname] = {
                "visdrone_class_id": cid,
                "target_class_id": gid,
                "count": class_counts.get(cid, 0),
            }
        counts[split] = {
            "total_images": total_images,
            "total_annotations": sum(class_counts.values()),
            "targets": targets,
        }
    return counts


def build_plan(config: dict) -> list[str]:
    """Build a step-by-step dry-run plan."""
    lines = []
    cfg = config
    backend = cfg.get("primary_backend", "grounded_sam2")
    splits = cfg.get("splits", ["train", "val"])
    expand = cfg.get("crop", {}).get("expand_ratio", 0.20)
    quality = cfg.get("quality", {})

    lines.append(f"Backend: {backend}")
    lines.append(f"Splits: {splits}")
    lines.append(f"Expand ratio: {expand}")
    lines.append("")

    lines.append("Pipeline steps:")
    lines.append("  1. Load VisDrone annotations from raw_root")
    lines.append(f"  2. Filter to target classes: {list(VISDRONE_TO_GLOBAL.values())}")
    lines.append(f"  3. For each sample, crop with expand_ratio={expand} (min 40px)")
    if any(k in backend for k in ("grounding", "grounded", "florence", "yoloworld")):
        lines.append("  4a. Run grounding model text-prompt detection on crop")
        lines.append("  4b. Use grounding refined box as SAM2 box prompt")
    else:
        lines.append("  4. Use expanded HBB as SAM2 box prompt")
    lines.append("  5. SAM2 generates mask(s)")
    lines.append("  6. Select best mask (by connectivity + center proximity)")
    lines.append("  7. cv2.minAreaRect → OBB (4-point clockwise)")
    lines.append("  8. Score quality (geometry + semantic)")
    lines.append(f"  9. Classify: auto_accept if final_score >= {quality.get('thresholds', {}).get('auto_accept_final', 0.65)}")
    lines.append("  10. Write auto_accept → YOLO-OBB labels; others → review_queue.jsonl")
    lines.append("")
    lines.append(f"Prompt templates language: {cfg.get('prompts', {}).get('language', 'en')}")
    lines.append("")

    return lines


def plan(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_root = Path(config.get("paths", {}).get("raw_root", ""))
    pseudo_root = Path(config.get("paths", {}).get("pseudo_root", ""))
    yolo_root = Path(config.get("paths", {}).get("yolo_root", ""))
    splits = config.get("splits", ["train", "val"])

    report: dict = {"config": str(config_path), "checks": [], "counts": {}, "plan": []}

    # ---- Path checks ----
    report["checks"].append(check_path("raw_root", str(raw_root), required=True))
    for split in splits:
        sd = resolve_split_dir(raw_root, split)
        report["checks"].append(check_path(f"split_dir [{split}]", str(sd), required=True))
    report["checks"].append(check_path("pseudo_root", str(pseudo_root), required=False))
    report["checks"].append(check_path("yolo_root", str(yolo_root), required=False))

    # ---- Backend checks ----
    backend = config.get("primary_backend", "")
    for key in ("groundingdino", "sam2", "florence2", "yoloworld"):
        be_cfg = config.get(key, {})
        if be_cfg.get("enabled"):
            chk = be_cfg.get("checkpoint_path", "") or be_cfg.get("config_path", "")
            report["checks"].append(check_path(f"{key}_checkpoint", chk, required=True))

    # ---- Import checks ----
    report["checks"].extend(check_imports())

    # ---- Data counts ----
    report["counts"] = count_targets(raw_root, splits)

    # ---- Plan ----
    report["plan"] = build_plan(config)

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    # Also print plan in readable form
    print("\n" + "=" * 60)
    print("DRY-RUN PLAN")
    print("=" * 60)
    for line in report["plan"]:
        print(line)

    print("=" * 60)
    print("CHECKS SUMMARY")
    print("=" * 60)
    errors = [c for c in report["checks"] if c.get("status") in ("missing", "not_configured")]
    if errors:
        print(f"  ISSUES ({len(errors)}):")
        for c in errors:
            label = c.get("label", c.get("module", c.get("desc", "?")))
            print(f"    [{c.get('status', '?')}] {label}: {c.get('path', '')}")
    else:
        print("  All checks passed.")
    not_installed = [c for c in report["checks"] if c.get("status") == "not_installed"]
    if not_installed:
        print(f"  OPTIONAL NOT INSTALLED ({len(not_installed)}):")
        for c in not_installed:
            label = c.get("label", c.get("module", c.get("desc", "?")))
            print(f"    {label} ({c.get('desc', '')})")

    print("\n" + "=" * 60)
    print("TARGET COUNTS")
    print("=" * 60)
    for split, info in report["counts"].items():
        print(f"  {split}: {info.get('total_images', 0)} images, {info.get('total_annotations', 0)} annotations")
        for name, tgt in info.get("targets", {}).items():
            print(f"    {name}: {tgt['count']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate v2 setup and print dry-run plan."
    )
    parser.add_argument(
        "--config",
        default="dataset_construction/configs/vlm_pseudo_obb_v2.yaml",
        help="Path to v2 config YAML",
    )
    return parser


def main() -> None:
    plan(build_parser().parse_args())


if __name__ == "__main__":
    main()
