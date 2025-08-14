"""
FastAPI server for Phishpedia (RF-DETR + Siamese)
Endpoints:
- GET  /                      -> render index.html (if templates available)
- POST /upload                -> multipart file upload, returns imageUrl
- POST /clear_upload          -> delete uploaded image
- POST /detect                -> UI JSON: {url, imageUrl}
- POST /api/analyze           -> JSON: {url, image_base64}
- POST /api/analyze-bytes     -> raw bytes body; query: ?url=...&preprocess=true/false

Preprocessing (controllable):
- Resize width to PREPROCESS_WIDTH (default 1248)
- Crop top PREPROCESS_CROP_TOP pixels (default 630)
- Toggle via env PHISHPEDIA_PREPROCESS_ENABLE=true/false or per-request query param 'preprocess'.
"""
from __future__ import annotations

import os
import sys
import io
import base64
import time
from typing import Optional

# Add parent directory to Python path for phishpedia imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from fastapi import FastAPI, File, UploadFile, Body, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
from PIL import Image
import numpy as np
import cv2

from phishpedia import PhishpediaWrapper


# ---------- Config ----------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(ROOT_DIR, 'WEBtool')
UPLOAD_DIR = os.path.join(WEB_DIR, 'static', 'uploads')
TEMPLATES_DIR = os.path.join(WEB_DIR, 'templates')

os.makedirs(UPLOAD_DIR, exist_ok=True)

PREPROCESS_ENABLE_DEFAULT = os.environ.get('PHISHPEDIA_PREPROCESS_ENABLE', 'true').lower() in ('1', 'true', 'yes')
PREPROCESS_WIDTH = int(os.environ.get('PHISHPEDIA_PREPROCESS_WIDTH', '1248'))
PREPROCESS_CROP_TOP = int(os.environ.get('PHISHPEDIA_PREPROCESS_CROP_TOP', '630'))

DETECTOR_OVERRIDE = os.environ.get('PHISHPEDIA_DETECTOR_TYPE', None)  # 'rcnn' or 'rfdetr'


def preprocess_image(img: Image.Image, enable: bool = True, target_width: int = PREPROCESS_WIDTH, crop_top: int = PREPROCESS_CROP_TOP) -> Image.Image:
    if not enable:
        return img
    # resize to target width keeping aspect ratio
    w, h = img.size
    if w != target_width:
        ratio = target_width / float(w)
        new_h = int(h * ratio)
        img = img.resize((target_width, new_h), Image.BILINEAR)
    # crop from top
    w, h = img.size
    crop_h = min(crop_top, h)
    img = img.crop((0, 0, w, crop_h))
    return img


app = FastAPI(title="Phishpedia API (FastAPI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static mounts for UI compatibility
app.mount("/static", StaticFiles(directory=os.path.join(WEB_DIR, 'static')), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

templates = Jinja2Templates(directory=TEMPLATES_DIR) if os.path.isdir(TEMPLATES_DIR) else None


# Initialize model
phishpedia = PhishpediaWrapper(detector_override=DETECTOR_OVERRIDE)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if templates is None:
        return HTMLResponse("<h3>Phishpedia API running. Templates directory not found.</h3>")
    # Provide a Flask-like url_for shim for templates expecting filename=... for static files
    def _url_for(name: str, filename: Optional[str] = None):
        try:
            if filename is not None:
                return request.url_for(name, path=filename)
            return request.url_for(name)
        except Exception:
            return "#"
    return templates.TemplateResponse('index.html', {"request": request, "url_for": _url_for})


@app.post("/reload-model")
async def reload_model(payload: dict = Body(...)):
    global phishpedia, DETECTOR_OVERRIDE
    detector = payload.get('detector')
    rebuild_cache = bool(payload.get('rebuild_cache', False))
    if detector is not None:
        d = str(detector).lower()
        if d not in ("rcnn", "rfdetr"):
            raise HTTPException(status_code=400, detail="invalid detector; use rcnn or rfdetr")
        DETECTOR_OVERRIDE = d
    # Reload wrapper and optionally rebuild cache
    phishpedia.reload(detector_override=DETECTOR_OVERRIDE, reload_targetlist=rebuild_cache)
    return {"success": True, "detector": DETECTOR_OVERRIDE or "config_default", "rebuild_cache": rebuild_cache}


@app.post("/upload")
async def upload_file(image: UploadFile = File(...)):
    name = image.filename
    if not name or ('.' not in name) or name.count('.') > 1 or '..' in name or any(sep in name for sep in (os.sep, os.altsep) if sep):
        raise HTTPException(status_code=400, detail="Invalid file name")
    ext = name.rsplit('.', 1)[-1].lower()
    if ext not in {"png", "jpg", "jpeg"}:
        raise HTTPException(status_code=400, detail="Invalid file type")
    save_path = os.path.join(UPLOAD_DIR, name)
    with open(save_path, 'wb') as f:
        f.write(await image.read())
    return {"success": True, "imageUrl": f"/uploads/{name}"}


@app.post("/clear_upload")
async def clear_upload(payload: dict = Body(...)):
    image_url = payload.get('imageUrl')
    if not image_url:
        raise HTTPException(status_code=400, detail="No image URL provided")
    name = image_url.split('/')[-1]
    path = os.path.normpath(os.path.join(UPLOAD_DIR, name))
    if not path.startswith(UPLOAD_DIR):
        raise HTTPException(status_code=400, detail="Invalid file path")
    try:
        os.remove(path)
        return {"success": True}
    except FileNotFoundError:
        return {"success": True}


def _boxes_to_list(pred_boxes) -> list[list[int]]:
    if pred_boxes is None:
        return []
    try:
        import numpy as np
        arr = pred_boxes if isinstance(pred_boxes, np.ndarray) else pred_boxes.detach().cpu().numpy()
        return [[int(x) for x in box[:4]] for box in arr]
    except Exception:
        return []


def _analyze_from_path(url: str, img_path: str, preprocess: bool,
                       siamese_threshold: float | None = None,
                       align: bool | None = None,
                       archeck: bool | None = None) -> dict:
    # preprocessing
    tmp_path = img_path
    if preprocess:
        with Image.open(img_path).convert('RGB') as im:
            pim = preprocess_image(im, enable=True)
            tmp_path = os.path.join(UPLOAD_DIR, f"proc_{os.path.basename(img_path)}")
            pim.save(tmp_path, format='PNG')

    t0 = time.time()
    # Use simplified logo detection only - no phishing analysis
    pred_target, pred_boxes, siamese_conf, logo_recog_time, logo_match_time = phishpedia.detect_logo_brand_only(
        tmp_path, siamese_threshold=siamese_threshold, align=align, archeck=archeck)
    total_time = time.time() - t0

    # cleanup temporary
    if preprocess and tmp_path != img_path:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return {
        "success": True,
        "predicted_brand": pred_target,
        "bboxes": _boxes_to_list(pred_boxes),
        "confidence": float(siamese_conf) if siamese_conf is not None else 0.0,
        "detection_time": round(float(logo_recog_time) + float(logo_match_time), 3),
        "total_time": round(total_time, 3)
    }


@app.post("/api/analyze-bytes")
async def api_analyze_bytes(request: Request):
    # Accept raw bytes body (image), optional query: ?url=...&preprocess=true/false
    url = request.query_params.get('url', '')
    preprocess_q = request.query_params.get('preprocess', None)
    th_q = request.query_params.get('siamese_threshold', None)
    align_q = request.query_params.get('align', None)
    archeck_q = request.query_params.get('archeck', None)
    preprocess = PREPROCESS_ENABLE_DEFAULT if preprocess_q is None else (preprocess_q.lower() in ('1','true','yes'))
    th = float(th_q) if th_q is not None else None
    align = (align_q.lower() in ('1','true','yes')) if align_q is not None else None
    archeck = (archeck_q.lower() in ('1','true','yes')) if archeck_q is not None else None
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty body")
    name = f"bytes_{int(time.time()*1000)}.png"
    path = os.path.join(UPLOAD_DIR, name)
    with open(path, 'wb') as f:
        f.write(data)
    res = _analyze_from_path(url, path, preprocess, siamese_threshold=th, align=align, archeck=archeck)
    try:
        os.remove(path)
    except Exception:
        pass
    return res


# Simple health endpoint
@app.get("/health")
async def health():
    return {"status": "ok", "detector": DETECTOR_OVERRIDE or "config_default", "preprocess": PREPROCESS_ENABLE_DEFAULT}
