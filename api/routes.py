# -*- coding: utf-8 -*-
"""
FastAPI Route Handlers for OAN Pest Inference
=============================================
Provides /predict, /health, /models, /benchmark, and /metrics endpoints.
"""

import os
import sys
import time
import uuid
import io
from typing import Optional, List
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from PIL import Image

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from api.schemas import PredictionResponse, HealthResponse, ModelInfo, BoundingBox
from models.factory import get_model_adapter
from models.registry import load_registry, SHORTLISTED_MODEL_IDS

router = APIRouter()

# Active in-memory adapters
LOADED_ADAPTERS = {}

def get_or_load_adapter(model_id: str):
    if model_id not in LOADED_ADAPTERS:
        adapter = get_model_adapter(model_id)
        adapter.load()
        LOADED_ADAPTERS[model_id] = adapter
    return LOADED_ADAPTERS[model_id]

@router.get("/health", response_model=HealthResponse)
def health():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        device=dev,
        active_models=SHORTLISTED_MODEL_IDS
    )

@router.get("/models", response_model=List[ModelInfo])
def list_models():
    reg = load_registry()
    out = []
    for m in reg:
        out.append(ModelInfo(
            model_id=m["model_id"],
            model_name=m["model_name"],
            category=m.get("category", "CATEGORY C"),
            architecture=m.get("architecture", "unknown"),
            framework=m.get("framework", "PyTorch"),
            size_mb=float(m.get("model_size_mb", 15.0)),
            task=m.get("task", "classification"),
            license=m.get("license", "Apache-2.0"),
            status="available"
        ))
    return out

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    crop: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    language: Optional[str] = Form("en"),
    model_id: Optional[str] = Form("mobilenetv4_conv_large"),
    threshold: Optional[float] = Form(0.40)
):
    # Security checks (Section 32)
    if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail=f"Unsupported MIME type: {image.content_type}")

    contents = await image.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds 15 MB limit")

    req_id = f"oan_{uuid.uuid4().hex[:12]}"
    t0 = time.perf_counter()

    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    # Fallback to yolov8s_pest or mobilenet if specified
    m_id = model_id or "mobilenetv4_conv_large"
    try:
        adapter = get_or_load_adapter(m_id)
        pred = adapter.predict(pil_img, image_id=req_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error on {m_id}: {e}")

    fwd_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # Convert bounding boxes
    boxes = []
    for b in pred.bounding_boxes:
        boxes.append(BoundingBox(
            box_2d=[float(b.get("ymin", 0)), float(b.get("xmin", 0)), float(b.get("ymax", 1)), float(b.get("xmax", 1))],
            label=b.get("label", pred.prediction),
            confidence=float(b.get("confidence", pred.confidence))
        ))

    is_pest = pred.task == "pest_detection"
    return PredictionResponse(
        request_id=req_id,
        crop=crop or "Unspecified",
        diagnosis=pred.prediction,
        pest=pred.prediction if is_pest else None,
        disease=pred.prediction if not is_pest and not pred.unknown else None,
        confidence=pred.confidence,
        uncertain=pred.unknown or pred.confidence < threshold,
        unknown=pred.unknown,
        model_id=pred.model_id,
        model_version="1.0.0",
        recommend_human_review=pred.unknown or pred.confidence < 0.60,
        inference_time_ms=fwd_time_ms,
        bounding_boxes=boxes,
        advisory=pred.explanation
    )

@router.get("/benchmark")
def get_benchmark_summary():
    step10_path = os.path.join(REPO_ROOT, "results", "remediation", "step10_five_class_classifier.json")
    step5_path = os.path.join(REPO_ROOT, "results", "remediation", "step5_summary.json")
    res = {}
    if os.path.exists(step10_path):
        import json
        with open(step10_path, encoding="utf-8") as f:
            res["disease_classifier"] = json.load(f)
    if os.path.exists(step5_path):
        import json
        with open(step5_path, encoding="utf-8") as f:
            res["insect_detector"] = json.load(f)
    return res

@router.get("/metrics")
def get_metrics():
    step7_path = os.path.join(REPO_ROOT, "results", "remediation", "step7_calibration.json")
    step8_path = os.path.join(REPO_ROOT, "results", "remediation", "step8_pilot_sizing.json")
    res = {}
    if os.path.exists(step7_path):
        import json
        with open(step7_path, encoding="utf-8") as f:
            res["calibration"] = json.load(f)
    if os.path.exists(step8_path):
        import json
        with open(step8_path, encoding="utf-8") as f:
            res["pilot_sizing"] = json.load(f)
    return res
