"""
Unified detector adapter for Faster R-CNN (Detectron2) and RF-DETR.
Provides a single pred_detector() that returns Nx4 boxes in xyxy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


# ---------------- RF-DETR helpers ----------------
def _infer_resolution_from_ckpt(ckpt_path: Path) -> Optional[int]:
    try:
        import torch as _torch
        ckpt = _torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sd = ckpt.get("model") if isinstance(ckpt, dict) else None
        if not isinstance(sd, dict):
            return None
        key = "backbone.0.encoder.encoder.embeddings.position_embeddings"
        pe = sd.get(key)
        if pe is None or not hasattr(pe, "shape"):
            return None
        tokens = pe.shape[1]
        n = max(tokens - 1, 1)
        import math
        side = int(round(math.sqrt(n))) * 16
        return side
    except Exception:
        return None


def _build_rfdetr_for_ckpt(weights_path: Path, user_res: Optional[int] = None, num_classes: int = 1):
    try:
        import importlib
        rfdetr = importlib.import_module('rfdetr')
        RFDETRSmall = getattr(rfdetr, 'RFDETRSmall')
        RFDETRNano = getattr(rfdetr, 'RFDETRNano')
        RFDETRMedium = getattr(rfdetr, 'RFDETRMedium')
    except Exception as e:
        raise ImportError(
            "RF-DETR is not installed. Install package 'rfdetr' and its deps."
        ) from e

    ckpt_res = _infer_resolution_from_ckpt(weights_path)
    res = ckpt_res or user_res or 512
    if res <= 448:
        ModelCls = RFDETRNano
        target_res = 384
    elif res <= 544:
        ModelCls = RFDETRSmall
        target_res = 512
    else:
        ModelCls = RFDETRMedium
        target_res = 576

    last_exc: Optional[Exception] = None
    for cls, r in [(ModelCls, target_res), (RFDETRNano, 384), (RFDETRSmall, 512), (RFDETRMedium, 576)]:
        try:
            m = cls(num_classes=num_classes, pretrain_weights=str(weights_path), resolution=r)
            try:
                # optional inference optimization
                m.optimize_for_inference()
            except Exception:
                pass
            return m
        except Exception as e:
            last_exc = e
            continue
    if last_exc:
        raise last_exc
    raise RuntimeError("Failed to build RF-DETR model for checkpoint")


def config_rfdetr(weights_path: str | Path, threshold: float = 0.5, resolution: Optional[int] = None):
    """Load RF-DETR and return a detector dict compatible with pred_detector."""
    wp = Path(weights_path)
    if not wp.exists():
        raise FileNotFoundError(f"RF-DETR weights not found: {weights_path}")
    model = _build_rfdetr_for_ckpt(wp, user_res=resolution, num_classes=1)
    # try to set class name to logo for completeness
    try:
        if not getattr(model.model, "class_names", None):
            model.model.class_names = ["logo"]
    except Exception:
        pass
    return {"type": "rfdetr", "model": model, "threshold": float(threshold)}


def pred_rfdetr(im_path: str, detector: dict) -> torch.Tensor:
    """Run RF-DETR and return Nx4 torch tensor (xyxy)."""
    from PIL import Image
    model = detector["model"]
    thr = float(detector.get("threshold", 0.5))
    img = Image.open(im_path).convert("RGB")
    det = model.predict(img, threshold=thr)

    # det is expected to be supervision.Detections
    try:
        # supervision>=0.20
        if hasattr(det, "xyxy"):
            xyxy = det.xyxy
        else:
            # Some versions return a dict-like
            xyxy = np.asarray(det)
    except Exception:
        # Fallback: try to interpret as numpy-like
        xyxy = getattr(det, "xyxy", None)
        if xyxy is None:
            xyxy = np.asarray(det)

    if xyxy is None or len(xyxy) == 0:
        return torch.empty((0, 4), dtype=torch.float32)
    if not isinstance(xyxy, np.ndarray):
        xyxy = np.asarray(xyxy)
    if xyxy.shape[-1] != 4:
        # attempt to slice first 4 columns
        xyxy = xyxy[:, :4]
    return torch.as_tensor(xyxy, dtype=torch.float32)


# ---------------- Unified API ----------------
def pred_detector(im_path: str, detector: Any) -> torch.Tensor:
    """Return Nx4 xyxy boxes for either RCNN or RF-DETR detector."""
    if isinstance(detector, dict) and detector.get("type") == "rfdetr":
        return pred_rfdetr(im_path, detector)
    # Assume detectron2 DefaultPredictor style
    # Defer import to avoid hard dependency when using RF-DETR
    from logo_recog import pred_rcnn
    return pred_rcnn(im=im_path, predictor=detector)
