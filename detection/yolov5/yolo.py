#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


class YOLOv5:
    """
    Minimal YOLOv5 wrapper that depends on an existing local YOLOv5 repo.
    It returns detections in OpenVTER's 10-col "4 points + score + cls" format:
      [x1,y1,x2,y1,x2,y2,x1,y2,conf,cls]
    """

    def __init__(
        self,
        weights,
        device_name="cuda:0",
        imgsz=(640, 640),
        confidence=0.25,
        nms_iou=0.45,
        repo_dir=None,
        classes=None,
        agnostic_nms=False,
    ):
        if repo_dir is None or repo_dir == "":
            raise ValueError(
                "YOLOv5 requires `repo_dir` pointing to a local yolov5 repository (with models/, utils/)."
            )
        repo_dir = str(repo_dir)
        if not os.path.isdir(repo_dir):
            raise FileNotFoundError(f"repo_dir not found: {repo_dir}")

        # OpenVTER itself has a top-level package named `utils`, which conflicts with YOLOv5's `utils`.
        # If OpenVTER's `utils` is already imported, YOLOv5 imports like `from utils.datasets import ...`
        # will resolve to OpenVTER's utils and fail. We switch the import context to YOLOv5's repo.
        sys.path.insert(0, repo_dir)
        self._maybe_switch_utils_namespace(repo_dir)

        from utils.general import non_max_suppression

        # letterbox is in different modules across YOLOv5 versions/forks
        try:
            from utils.augmentations import letterbox
        except Exception:
            from utils.datasets import letterbox  # type: ignore

        # scale_boxes is called scale_coords in older YOLOv5 forks
        try:
            from utils.general import scale_boxes as _scale_boxes

            def _scale(img1_shape, boxes, img0_shape):
                return _scale_boxes(img1_shape, boxes, img0_shape)

        except Exception:
            from utils.general import scale_coords as _scale_coords  # type: ignore

            def _scale(img1_shape, boxes, img0_shape):
                _scale_coords(img1_shape, boxes, img0_shape)
                return boxes

        self._letterbox = letterbox
        self._nms = non_max_suppression
        self._scale = _scale

        self.imgsz = tuple(imgsz) if isinstance(imgsz, (list, tuple)) else (int(imgsz), int(imgsz))
        self.conf_thres = float(confidence)
        self.iou_thres = float(nms_iou)
        self.agnostic_nms = bool(agnostic_nms)
        self.classes = classes

        if device_name == "cpu":
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device(device_name)
        else:
            self.device = torch.device("cpu")

        w = str(weights)
        if not os.path.isfile(w):
            raise FileNotFoundError(f"YOLOv5 weights not found: {w}")

        # DetectMultiBackend exists in newer YOLOv5; older forks use attempt_load
        self._backend = None
        try:
            from models.common import DetectMultiBackend  # type: ignore

            self.model = DetectMultiBackend(w, device=self.device, dnn=False, data=None, fp16=False)
            self.model.eval()
            self.stride = int(getattr(self.model, "stride", 32) or 32)
            self._backend = "multibackend"
        except Exception:
            from models.experimental import attempt_load  # type: ignore

            self.model = attempt_load(w, map_location=self.device)
            self.model.eval()
            stride = getattr(self.model, "stride", None)
            self.stride = int(stride.max()) if hasattr(stride, "max") else int(stride or 32)
            self._backend = "attempt_load"
        self._patch_upsample_compat()

    @staticmethod
    def _maybe_switch_utils_namespace(repo_dir: str) -> None:
        """
        Ensure `import utils.*` resolves to YOLOv5's utils package.
        This is required because OpenVTER also defines a top-level `utils` package.
        """
        repo_dir_abs = os.path.abspath(repo_dir)

        def _is_from_repo(module) -> bool:
            mod_file = getattr(module, "__file__", None)
            if not mod_file:
                return False
            try:
                return os.path.abspath(mod_file).startswith(repo_dir_abs)
            except Exception:
                return False

        existing_utils = sys.modules.get("utils")
        if existing_utils is not None and not _is_from_repo(existing_utils):
            # Keep a back-reference for debugging; OpenVTER already imported what it needs.
            sys.modules.setdefault("openvter_utils", existing_utils)

            # Remove conflicting modules so YOLOv5 can import its own `utils.*`.
            for name in list(sys.modules.keys()):
                if name == "utils" or name.startswith("utils."):
                    del sys.modules[name]

    def _patch_upsample_compat(self) -> None:
        """
        Some YOLOv5 .pt weights are pickled model objects created with older torch versions.
        When loaded under newer torch, nn.Upsample.forward expects `recompute_scale_factor`,
        but the unpickled modules may not have that attribute. Add a default to avoid crash.
        """
        try:
            mods = self.model.modules() if hasattr(self.model, "modules") else []
            for m in mods:
                if isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
                    m.recompute_scale_factor = None
        except Exception:
            return

    def _preprocess(self, img_rgb):
        im = self._letterbox(img_rgb, new_shape=self.imgsz, auto=False, stride=int(self.stride or 32))[0]
        im = im.transpose((2, 0, 1))  # HWC -> CHW (RGB)
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device).float() / 255.0
        if im.ndim == 3:
            im = im.unsqueeze(0)
        return im

    def det_images_batch(self, ori_image_ls):
        if len(ori_image_ls) == 0:
            return []
        ims = [self._preprocess(img) for img in ori_image_ls]
        im = torch.cat(ims, dim=0)

        with torch.no_grad():
            pred = self.model(im)
            # attempt_load backend returns (pred, ...) or [pred, ...]
            if isinstance(pred, (list, tuple)):
                pred = pred[0]

        dets = self._nms(
            pred,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            classes=self.classes,
            agnostic=self.agnostic_nms,
            max_det=300,
        )

        out = []
        for i, det in enumerate(dets):
            if det is None or len(det) == 0:
                out.append(np.empty((0, 10), dtype=np.float32))
                continue

            img0 = ori_image_ls[i]
            h0, w0 = img0.shape[:2]
            det[:, :4] = self._scale(im.shape[2:], det[:, :4], (h0, w0)).round()

            det_np = det.detach().cpu().numpy().astype(np.float32)
            boxes = []
            for x1, y1, x2, y2, conf, cls in det_np[:, :6]:
                boxes.append([x1, y1, x2, y1, x2, y2, x1, y2, conf, cls])
            out.append(np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 10), dtype=np.float32))
        return out
