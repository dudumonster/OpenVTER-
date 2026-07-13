#!/usr/bin/env python3
"""VLM-assisted pseudo OBB v2 pipeline for VisDrone VRU classes.

Architecture:

  VisDrone HBB → crop (+20% margin) → VLM/Grounding text-prompt box refinement
  → SAM2 mask → minAreaRect OBB → v2 quality scoring → auto_accept / review / reject

This script is the *orchestrator* — it does not hard-code model inference.
Each backend lives in ``dataset_construction/scripts/vlm_backends/`` and is
loaded by name from config.

Usage:

  # Dry-run: validate config and count data
  python3 dataset_construction/scripts/vlm_pseudo_obb_v2.py plan

  # Run on val split only (recommended first)
  python3 dataset_construction/scripts/vlm_pseudo_obb_v2.py generate

  # Run with experiment overrides
  python3 dataset_construction/scripts/vlm_pseudo_obb_v2.py generate \\
    --sample-per-class 100 --splits val --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

# -- v2 schema (shared with schema module) --
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset_construction.scripts.vlm_pseudo_obb_v2_schema import (
    ASPECT_PRIORS,
    GLOBAL_NAMES,
    VISDRONE_TO_GLOBAL,
    VLMPseudoObbRecord,
    VLMQualityMetrics,
    append_jsonl,
    classify_review_status,
    expand_box,
    normalize_points,
    obb_from_mask,
    order_points_clockwise,
    postprocess_mask,
    score_v2_quality,
    select_component,
    write_jsonl,
)
from dataset_construction.scripts.vlm_prompt_templates import (
    get_grounding_prompt,
)

# ---------------------------------------------------------------------------
# VLM Backend abstract interface
# ---------------------------------------------------------------------------

class VLMBackend:
    """Abstract interface for a VLM + SAM2 pseudo-OBB backend.

    Subclasses implement ``refine_and_segment()`` which takes an image and an
    expanded crop box and returns (mask, vlm_box, vlm_class, vlm_confidence).
    """

    name: str = "base"

    def refine_and_segment(
        self,
        image_bgr: np.ndarray,
        hbb_xyxy: tuple[int, int, int, int],
        expanded_xyxy: tuple[int, int, int, int],
        class_name: str,
    ) -> dict[str, Any]:
        """Run VLM grounding + SAM2 segmentation on a single crop.

        Returns a dict with keys:
            mask: np.ndarray                        -- binary mask (crop-relative)
            vlm_box_xyxy: list[float] | None        -- VLM refined box (global coords)
            vlm_box_confidence: float
            vlm_class_name: str
            vlm_class_confidence: float
            text_prompt: str
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Pure SAM2 backend (equivalent to v1 SAM2 — no grounding)
# ---------------------------------------------------------------------------

class PureSAM2Backend(VLMBackend):
    """SAM2 box-prompt only, no VLM grounding.  Baseline for experiments."""

    name = "pure_sam2"

    def __init__(self, checkpoint: str, config: str, device: str) -> None:
        self.checkpoint = checkpoint
        self.config = config
        self.device = device
        self._predictor = None
        self._last_image_id: int | None = None

    def _get_predictor(self):
        if self._predictor is None:
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError as e:
                raise ImportError(
                    "SAM2 not installed. Install with: pip install sam2"
                ) from e
            model = build_sam2(self.config, self.checkpoint, device=self.device)
            self._predictor = SAM2ImagePredictor(model)
        return self._predictor

    def refine_and_segment(
        self,
        image_bgr: np.ndarray,
        hbb_xyxy: tuple[int, int, int, int],
        expanded_xyxy: tuple[int, int, int, int],
        class_name: str,
    ) -> dict[str, Any]:
        predictor = self._get_predictor()
        image_id = id(image_bgr)
        if image_id != self._last_image_id:
            predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            self._last_image_id = image_id

        ex1, ey1, ex2, ey2 = expanded_xyxy
        # Use expanded box directly as SAM2 box prompt
        masks, scores, _ = predictor.predict(
            box=np.asarray(expanded_xyxy, dtype=np.float32),
            multimask_output=True,
        )
        best = masks[int(np.argmax(scores))].astype(np.uint8) * 255
        crop_mask = postprocess_mask(best[ey1:ey2, ex1:ex2])

        return {
            "mask": crop_mask,
            "vlm_box_xyxy": None,
            "vlm_box_confidence": 0.0,
            "vlm_class_name": "",
            "vlm_class_confidence": 0.0,
            "text_prompt": f"[box prompt only] {class_name}",
        }


# ---------------------------------------------------------------------------
# GroundingDINO + SAM2 backend
# ---------------------------------------------------------------------------

class GroundingSAM2Backend(VLMBackend):
    """GroundingDINO text-prompt → refined box → SAM2 mask."""

    name = "groundingdino_sam2"

    def __init__(
        self,
        gd_config: str,
        gd_checkpoint: str,
        gd_box_threshold: float,
        gd_text_threshold: float,
        sam2_config: str,
        sam2_checkpoint: str,
        device: str,
    ) -> None:
        self.gd_config = gd_config
        self.gd_checkpoint = gd_checkpoint
        self.gd_box_threshold = gd_box_threshold
        self.gd_text_threshold = gd_text_threshold
        self.sam2_config = sam2_config
        self.sam2_checkpoint = sam2_checkpoint
        self.device = device
        self._gd_model = None
        self._sam2_predictor = None
        self._last_image_id: int | None = None

    @staticmethod
    def is_available() -> bool:
        try:
            import groundingdino  # noqa: F401
            from sam2.build_sam import build_sam2  # noqa: F401
            return True
        except ImportError:
            return False

    def _init_grounding(self):
        if self._gd_model is not None:
            return
        try:
            from groundingdino.util.inference import Model
        except ImportError:
            raise ImportError(
                "GroundingDINO not installed. "
                "Clone https://github.com/IDEA-Research/GroundingDINO and "
                "add to PYTHONPATH."
            )
        self._gd_model = Model(
            model_config_path=self.gd_config,
            model_checkpoint_path=self.gd_checkpoint,
            device=self.device,
        )

    def _init_sam2(self):
        if self._sam2_predictor is not None:
            return
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise ImportError("SAM2 not installed. Install with: pip install sam2")
        model = build_sam2(self.sam2_config, self.sam2_checkpoint, device=self.device)
        self._sam2_predictor = SAM2ImagePredictor(model)

    def _grounding_detect(
        self, crop_bgr: np.ndarray, text_prompt: str
    ) -> tuple[list[float] | None, float]:
        """Run GroundingDINO on a single crop. Returns (best_box_xyxy, confidence)."""
        self._init_grounding()
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        detections = self._gd_model.predict_with_classes(
            image=crop_rgb,
            classes=[text_prompt],
            box_threshold=self.gd_box_threshold,
            text_threshold=self.gd_text_threshold,
        )
        if detections.xyxy is None or len(detections.xyxy) == 0:
            return None, 0.0
        # Take highest-confidence detection
        best_idx = int(np.argmax(detections.confidence))
        box = detections.xyxy[best_idx].tolist()  # [x1, y1, x2, y2] in crop coords
        conf = float(detections.confidence[best_idx])
        return box, conf

    def refine_and_segment(
        self,
        image_bgr: np.ndarray,
        hbb_xyxy: tuple[int, int, int, int],
        expanded_xyxy: tuple[int, int, int, int],
        class_name: str,
    ) -> dict[str, Any]:
        self._init_sam2()
        ex1, ey1, ex2, ey2 = expanded_xyxy
        crop = image_bgr[ey1:ey2, ex1:ex2]
        text_prompt = get_grounding_prompt(class_name, "en")

        # Step 1: GroundingDINO text-prompt detection on crop
        vlm_box_crop, vlm_box_conf = self._grounding_detect(crop, text_prompt)

        # Step 2: Determine box prompt for SAM2
        if vlm_box_crop is not None:
            # Convert crop-relative box to global coords
            sam2_box = (
                vlm_box_crop[0] + ex1,
                vlm_box_crop[1] + ey1,
                vlm_box_crop[2] + ex1,
                vlm_box_crop[3] + ey1,
            )
            vlm_box_global = list(sam2_box)
        else:
            # Fallback: use expanded HBB as box prompt
            sam2_box = tuple(expanded_xyxy)
            vlm_box_global = None

        # Step 3: SAM2 segmentation
        image_id = id(image_bgr)
        if image_id != self._last_image_id:
            self._sam2_predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            self._last_image_id = image_id

        masks, scores, _ = self._sam2_predictor.predict(
            box=np.asarray(sam2_box, dtype=np.float32),
            multimask_output=True,
        )
        best = masks[int(np.argmax(scores))].astype(np.uint8) * 255
        crop_mask = postprocess_mask(best[ey1:ey2, ex1:ex2])

        return {
            "mask": crop_mask,
            "vlm_box_xyxy": vlm_box_global,
            "vlm_box_confidence": round(vlm_box_conf, 4),
            "vlm_class_name": class_name if vlm_box_crop is not None else "",
            "vlm_class_confidence": round(vlm_box_conf, 4),
            "text_prompt": text_prompt,
        }


# ---------------------------------------------------------------------------
# Dummy backend (for dry-run / testing without models)
# ---------------------------------------------------------------------------

class DummyVLMBackend(VLMBackend):
    """Fake backend that returns the expanded HBB as mask with no VLM signal."""

    name = "dummy"

    def refine_and_segment(
        self,
        image_bgr: np.ndarray,
        hbb_xyxy: tuple[int, int, int, int],
        expanded_xyxy: tuple[int, int, int, int],
        class_name: str,
    ) -> dict[str, Any]:
        ex1, ey1, ex2, ey2 = expanded_xyxy
        crop_h = ey2 - ey1
        crop_w = ex2 - ex1
        x1, y1, x2, y2 = hbb_xyxy
        # Create a rectangular mask from HBB within the crop
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        rx1 = max(int(x1 - ex1), 0)
        ry1 = max(int(y1 - ey1), 0)
        rx2 = min(int(x2 - ex1), crop_w)
        ry2 = min(int(y2 - ey1), crop_h)
        mask[ry1:ry2, rx1:rx2] = 255
        return {
            "mask": mask,
            "vlm_box_xyxy": None,
            "vlm_box_confidence": 0.0,
            "vlm_class_name": "",
            "vlm_class_confidence": 0.0,
            "text_prompt": f"[dummy] {class_name}",
        }


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def make_vlm_backend(config: dict) -> VLMBackend:
    """Instantiate a VLM backend from config YAML."""
    backend_name = config.get("primary_backend", "dummy")

    if backend_name == "dummy":
        return DummyVLMBackend()

    if backend_name in ("pure_sam2", "sam2"):
        sam2 = config.get("sam2", {})
        return PureSAM2Backend(
            checkpoint=sam2.get("checkpoint_path", ""),
            config=sam2.get("config_path", ""),
            device=sam2.get("device", "cuda"),
        )

    if backend_name in ("grounded_sam2", "groundingdino_sam2"):
        gd = config.get("groundingdino", {})
        sam2 = config.get("sam2", {})
        backend = GroundingSAM2Backend(
            gd_config=gd.get("config_path", ""),
            gd_checkpoint=gd.get("checkpoint_path", ""),
            gd_box_threshold=gd.get("box_threshold", 0.25),
            gd_text_threshold=gd.get("text_threshold", 0.20),
            sam2_config=sam2.get("config_path", ""),
            sam2_checkpoint=sam2.get("checkpoint_path", ""),
            device=gd.get("device", "cuda"),
        )
        if not backend.is_available():
            print(
                "WARNING: GroundingDINO not importable; falling back to PureSAM2Backend. "
                "Set primary_backend: pure_sam2 in config to suppress this warning.",
                flush=True,
            )
            return PureSAM2Backend(
                checkpoint=sam2.get("checkpoint_path", ""),
                config=sam2.get("config_path", ""),
                device=sam2.get("device", "cuda"),
            )
        return backend

    # Fallback for unknown backends
    print(f"WARNING: unknown backend '{backend_name}', using dummy", flush=True)
    return DummyVLMBackend()


# ---------------------------------------------------------------------------
# Annotation parsing (shared with v1)
# ---------------------------------------------------------------------------

class VisDroneObject:
    __slots__ = ("x", "y", "w", "h", "score", "class_id", "truncation", "occlusion", "line_index")

    def __init__(
        self, x: float, y: float, w: float, h: float,
        score: int, class_id: int, truncation: int, occlusion: int, line_index: int,
    ) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.score = score
        self.class_id = class_id
        self.truncation = truncation
        self.occlusion = occlusion
        self.line_index = line_index

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.x + self.w, self.y + self.h


def parse_annotation(path: Path) -> list[VisDroneObject]:
    objects: list[VisDroneObject] = []
    for idx, line in enumerate(path.read_text(errors="ignore").splitlines()):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 8:
            continue
        try:
            x, y, w, h = map(float, parts[:4])
            score, class_id, truncation, occlusion = map(int, parts[4:8])
        except ValueError:
            continue
        objects.append(VisDroneObject(x, y, w, h, score, class_id, truncation, occlusion, idx))
    return objects


def resolve_split_dir(raw_root: Path, split: str) -> Path:
    direct = raw_root / f"VisDrone2019-DET-{split}"
    nested = direct / f"VisDrone2019-DET-{split}"
    for candidate in (direct, nested):
        if (candidate / "images").is_dir() and (candidate / "annotations").is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find VisDrone split directory for {split} under {raw_root}")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def link_or_copy_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "none":
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
        return
    raise ValueError(f"Unknown copy mode: {mode}")


def draw_preview_v2(
    image_bgr: np.ndarray,
    hbb_xyxy: tuple[int, int, int, int],
    obb_points: np.ndarray | None,
    component_mask: np.ndarray,
    expanded_xyxy: tuple[int, int, int, int],
    vlm_box_xyxy: list[float] | None,
    label: str,
    final_score: float,
) -> np.ndarray:
    """Draw HBB (yellow), VLM box (blue dash), mask overlay (orange), OBB (green)."""
    canvas = image_bgr.copy()
    ex1, ey1, ex2, ey2 = expanded_xyxy
    if component_mask.size:
        overlay = np.zeros_like(canvas)
        crop_overlay = overlay[ey1:ey2, ex1:ex2]
        crop_overlay[component_mask > 0] = (0, 180, 255)
        canvas = cv2.addWeighted(canvas, 1.0, overlay, 0.35, 0)

    # HBB in yellow
    x1, y1, x2, y2 = hbb_xyxy
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # VLM box in blue dashed
    if vlm_box_xyxy and len(vlm_box_xyxy) == 4:
        vx1, vy1, vx2, vy2 = [int(round(v)) for v in vlm_box_xyxy]
        dash_len = 8
        for i in range(0, max(vx2 - vx1, 1), dash_len * 2):
            cv2.line(canvas, (vx1 + i, vy1), (min(vx1 + i + dash_len, vx2), vy1), (255, 100, 50), 1)
            cv2.line(canvas, (vx1 + i, vy2), (min(vx1 + i + dash_len, vx2), vy2), (255, 100, 50), 1)

    # OBB in green
    if obb_points is not None:
        pts = np.round(obb_points).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        for idx, (px, py) in enumerate(obb_points):
            cv2.circle(canvas, (int(round(px)), int(round(py))), 3, (0, 0, 255), -1)

    cv2.putText(
        canvas,
        f"{label} score={final_score:.2f}",
        (max(x1, 0), max(y1 - 6, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def save_data_yaml(yolo_root: Path) -> None:
    data = {
        "path": str(yolo_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": GLOBAL_NAMES,
    }
    with (yolo_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_v2(config: dict, args: argparse.Namespace) -> None:
    """Run the v2 VLM-assisted pseudo OBB pipeline."""
    paths = config.get("paths", {})
    raw_root = Path(paths.get("raw_root", args.raw_root))
    pseudo_root = Path(paths.get("pseudo_root", args.pseudo_root))
    yolo_root = Path(paths.get("yolo_root", args.yolo_root))
    pseudo_root.mkdir(parents=True, exist_ok=True)
    yolo_root.mkdir(parents=True, exist_ok=True)

    cfg_crop = config.get("crop", {})
    expand_ratio = args.expand_ratio if args.expand_ratio is not None else cfg_crop.get("expand_ratio", 0.20)
    min_crop_size = cfg_crop.get("min_crop_size", 40)

    cfg_quality = config.get("quality", {})
    cfg_output = config.get("output", {})

    splits = args.splits or config.get("splits", ["val"])
    sample_per_class = args.sample_per_class
    seed = args.seed

    # Backend
    backend = make_vlm_backend(config)
    print(f"Using backend: {backend.name}", flush=True)

    rng = random.Random(seed)

    all_records: list[dict] = []
    review_candidates: list[dict] = []
    labels_by_image: dict[tuple[str, str], list[str]] = defaultdict(list)
    counters: Counter[str] = Counter()

    # If sampling, collect candidate sample_ids first
    sample_pool: dict[str, list[str]] = defaultdict(list)  # class_name -> [sample_id]

    for split in splits:
        split_dir = resolve_split_dir(raw_root, split)
        image_dir = split_dir / "images"
        ann_dir = split_dir / "annotations"
        images = sorted(image_dir.glob("*.jpg"))
        if args.max_images:
            images = images[: args.max_images]

        for image_index, image_path in enumerate(images, start=1):
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                counters["image_read_failed"] += 1
                continue
            image_h, image_w = image_bgr.shape[:2]
            yolo_image_path = yolo_root / "images" / split / image_path.name
            link_or_copy_image(image_path, yolo_image_path, cfg_output.get("copy_mode", "symlink"))
            objects = parse_annotation(ann_dir / f"{image_path.stem}.txt")

            for obj in objects:
                if obj.class_id not in VISDRONE_TO_GLOBAL:
                    continue
                global_id, class_name = VISDRONE_TO_GLOBAL[obj.class_id]
                sample_id = f"{split}__{image_path.stem}__{obj.line_index:04d}"
                sample_pool[class_name].append(sample_id)

    # Apply sampling if requested
    selected_ids: set[str] = set()
    if sample_per_class and sample_per_class > 0:
        for cls_name, ids in sample_pool.items():
            sampled = rng.sample(ids, min(sample_per_class, len(ids)))
            selected_ids.update(sampled)
        print(
            f"Sampling: {len(selected_ids)} samples from {sum(len(v) for v in sample_pool.values())} total "
            f"({sample_per_class} per class)",
            flush=True,
        )

    # --- Main processing loop ---
    for split in splits:
        split_dir = resolve_split_dir(raw_root, split)
        image_dir = split_dir / "images"
        ann_dir = split_dir / "annotations"
        images = sorted(image_dir.glob("*.jpg"))
        if args.max_images:
            images = images[: args.max_images]

        for image_index, image_path in enumerate(images, start=1):
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                counters["image_read_failed"] += 1
                continue
            image_h, image_w = image_bgr.shape[:2]
            yolo_image_path = yolo_root / "images" / split / image_path.name
            link_or_copy_image(image_path, yolo_image_path, cfg_output.get("copy_mode", "symlink"))
            objects = parse_annotation(ann_dir / f"{image_path.stem}.txt")

            for obj in objects:
                if obj.class_id not in VISDRONE_TO_GLOBAL:
                    continue
                global_id, class_name = VISDRONE_TO_GLOBAL[obj.class_id]

                # Sampling filter
                sample_id = f"{split}__{image_path.stem}__{obj.line_index:04d}"
                if selected_ids and sample_id not in selected_ids:
                    counters["skipped_by_sampling"] += 1
                    continue

                hbb_xyxy_float = obj.xyxy
                hbb_xyxy = tuple(int(round(v)) for v in hbb_xyxy_float)
                expanded = expand_box(hbb_xyxy_float, image_w, image_h, expand_ratio, min_crop_size)
                ex1, ey1, _, _ = expanded

                # VLM + SAM2
                result = backend.refine_and_segment(image_bgr, hbb_xyxy, expanded, class_name)
                mask = result["mask"]
                vlm_box_xyxy = result.get("vlm_box_xyxy")
                vlm_box_conf = result.get("vlm_box_confidence", 0.0)
                vlm_class = result.get("vlm_class_name", "")
                vlm_class_conf = result.get("vlm_class_confidence", 0.0)
                text_prompt = result.get("text_prompt", "")

                # Component selection
                hbb_center_local = (obj.x + obj.w / 2.0 - ex1, obj.y + obj.h / 2.0 - ey1)
                component = select_component(mask, hbb_center_local)

                # Mask → OBB
                obb_points, rect_info = obb_from_mask(component, expanded)
                if obb_points is not None:
                    obb_points[:, 0] = np.clip(obb_points[:, 0], 0, image_w - 1)
                    obb_points[:, 1] = np.clip(obb_points[:, 1], 0, image_h - 1)
                    yolo_values = normalize_points(obb_points, image_w, image_h)
                else:
                    yolo_values = []

                # v2 quality scoring
                metrics = score_v2_quality(
                    class_name=class_name,
                    obb_points=obb_points,
                    rect_info=rect_info,
                    component_mask=component,
                    expanded_xyxy=expanded,
                    image_w=image_w,
                    image_h=image_h,
                    hbb_xywh=(obj.x, obj.y, obj.w, obj.h),
                    occlusion=obj.occlusion,
                    truncation=obj.truncation,
                    vlm_class_name=vlm_class,
                    vlm_class_confidence=vlm_class_conf,
                    vlm_box_confidence=vlm_box_conf,
                    vlm_box_xyxy=vlm_box_xyxy,
                )
                review_decision, queue_reason = classify_review_status(metrics)

                # Save mask crop
                mask_rel = None
                if cfg_output.get("save_mask_crops", True) and component.size:
                    mask_path = pseudo_root / "masks" / split / f"{sample_id}.png"
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(mask_path), component)
                    mask_rel = str(mask_path.relative_to(pseudo_root))

                # Build record
                record = VLMPseudoObbRecord(
                    sample_id=sample_id,
                    split=split,
                    source_dataset="VisDrone2019-DET",
                    image_path=str(image_path),
                    annotation_path=str(ann_dir / f"{image_path.stem}.txt"),
                    image_width=image_w,
                    image_height=image_h,
                    source_hbb_xywh=[round(obj.x, 3), round(obj.y, 3), round(obj.w, 3), round(obj.h, 3)],
                    source_hbb_xyxy=[round(float(v), 3) for v in hbb_xyxy_float],
                    source_class_id=obj.class_id,
                    source_class_name=class_name,
                    source_occlusion=obj.occlusion,
                    source_truncation=obj.truncation,
                    annotation_line_index=obj.line_index,
                    target_class_id=global_id,
                    target_class_name=class_name,
                    crop_box_xyxy=list(expanded),
                    crop_scale=1.0,
                    vlm_backend=backend.name,
                    text_prompt=text_prompt,
                    vlm_box_xyxy=[round(v, 3) for v in vlm_box_xyxy] if vlm_box_xyxy else [],
                    vlm_box_confidence=vlm_box_conf,
                    vlm_class_name=vlm_class,
                    vlm_class_confidence=vlm_class_conf,
                    mask_path=mask_rel or "",
                    obb_points=(
                        [[round(float(x), 3), round(float(y), 3)] for x, y in obb_points]
                        if obb_points is not None else []
                    ),
                    yolo_obb=[global_id, *yolo_values] if yolo_values else [],
                    quality=metrics,
                    review_status=review_decision,
                    queue_reason=queue_reason,
                    failure_reasons=metrics.flags if review_decision == "reject" else [],
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                all_records.append(record.to_dict())
                counters[f"{split}_{class_name}"] += 1

                # Auto-accept: write to YOLO-OBB label pool
                if review_decision == "auto_accept" and yolo_values:
                    labels_by_image[(split, image_path.stem)].append(
                        " ".join([str(global_id), *[f"{v:.6f}" for v in yolo_values]])
                    )

                # Review queue: low quality or random sample
                review_random = (
                    review_decision == "auto_accept"
                    and rng.random() < cfg_output.get("review", {}).get("random_sample_rate", 0.03)
                )
                if review_decision == "review" or review_random:
                    rec_for_queue = dict(record.to_dict())
                    rec_for_queue["queue_reason"] = queue_reason or ("random_sample" if review_random else "low_quality")
                    review_candidates.append(rec_for_queue)

                    # Draw preview
                    if cfg_output.get("save_previews", True):
                        preview_path = pseudo_root / "previews" / split / f"{sample_id}.jpg"
                        preview_path.parent.mkdir(parents=True, exist_ok=True)
                        preview = draw_preview_v2(
                            image_bgr, hbb_xyxy, obb_points, component,
                            expanded, vlm_box_xyxy, class_name,
                            float(metrics.final_score),
                        )
                        cv2.imwrite(str(preview_path), preview)
                        rec_for_queue["preview"] = str(preview_path.relative_to(pseudo_root))

            if cfg_output.get("progress_every") and image_index % cfg_output["progress_every"] == 0:
                print(f"{split}: processed {image_index}/{len(images)} images", flush=True)

    # --- Write YOLO-OBB labels ---
    labels_root = yolo_root / "labels"
    for split in splits:
        split_dir = resolve_split_dir(raw_root, split)
        image_paths = sorted((split_dir / "images").glob("*.jpg"))
        if args.max_images:
            image_paths = image_paths[: args.max_images]
        for image_path in image_paths:
            label_path = labels_root / split / f"{image_path.stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            rows = labels_by_image.get((split, image_path.stem), [])
            label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    # --- Write outputs ---
    write_jsonl(pseudo_root / "quality.jsonl", all_records)
    write_jsonl(pseudo_root / "review_queue.jsonl", review_candidates)
    save_data_yaml(yolo_root)

    # Manifest
    auto_accept_count = sum(1 for r in all_records if r["review_status"] == "auto_accept")
    review_count = sum(1 for r in all_records if r["review_status"] == "review")
    reject_count = sum(1 for r in all_records if r["review_status"] == "reject")

    manifest = {
        "version": "v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend.name,
        "config": {
            "raw_root": str(raw_root),
            "pseudo_root": str(pseudo_root),
            "yolo_root": str(yolo_root),
            "splits": splits,
            "expand_ratio": expand_ratio,
            "sample_per_class": sample_per_class,
            "seed": seed,
        },
        "totals": {
            "records": len(all_records),
            "auto_accept": auto_accept_count,
            "review": review_count,
            "reject": reject_count,
            "review_queue": len(review_candidates),
        },
        "counters": dict(counters),
        "global_names": GLOBAL_NAMES,
    }
    (pseudo_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    default_config = "dataset_construction/configs/vlm_pseudo_obb_v2.yaml"
    default_pseudo = "dataset_construction/derived/visdrone_pseudo_obb_v2"
    default_yolo = "dataset_construction/derived/visdrone_yolo_obb_v2"

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    # --- plan ---
    plan_parser = sub.add_parser("plan", help="Validate config and data, print dry-run plan")
    plan_parser.add_argument("--config", default=default_config)
    plan_parser.set_defaults(func=lambda args: _run_plan(args))

    # --- generate ---
    gen = sub.add_parser("generate", help="Run v2 VLM-assisted pseudo OBB pipeline")
    gen.add_argument("--config", default=default_config,
                     help="Path to v2 config YAML")
    gen.add_argument("--raw-root", default="dataset_construction/data_sources/visdrone/raw")
    gen.add_argument("--pseudo-root", default=default_pseudo)
    gen.add_argument("--yolo-root", default=default_yolo)
    gen.add_argument("--splits", nargs="+", default=None,
                     help="Splits to process (default: from config)")
    gen.add_argument("--expand-ratio", type=float, default=None,
                     help="Override crop expand ratio")
    gen.add_argument("--sample-per-class", type=int, default=0,
                     help="Number of samples per class (0=all)")
    gen.add_argument("--max-images", type=int, default=None,
                     help="Max images per split")
    gen.add_argument("--seed", type=int, default=20260625)
    gen.set_defaults(func=lambda args: _run_generate(args))

    return parser


def _run_plan(args: argparse.Namespace) -> None:
    # Delegate to plan script
    from dataset_construction.scripts import vlm_pseudo_obb_v2_plan
    plan_args = argparse.Namespace(config=args.config)
    vlm_pseudo_obb_v2_plan.plan(plan_args)


def _run_generate(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Run 'python3 dataset_construction/scripts/vlm_pseudo_obb_v2.py plan' first.")
        sys.exit(1)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    generate_v2(config, args)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
