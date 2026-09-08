# -*- coding: utf-8 -*-
"""
OAN Kenya Pest & Disease Inference API Schemas
=============================================
Conforms strictly to Section 27 of the OAN Kenya Specification.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    box_2d: List[float] = Field(..., description="[ymin, xmin, ymax, xmax] normalized [0, 1]")
    label: str
    confidence: float

class PredictionResponse(BaseModel):
    request_id: str
    crop: Optional[str] = "Unknown"
    diagnosis: str
    pest: Optional[str] = None
    disease: Optional[str] = None
    confidence: float
    uncertain: bool
    unknown: bool
    model_id: str
    model_version: str
    recommend_human_review: bool
    inference_time_ms: float
    bounding_boxes: List[BoundingBox] = []
    advisory: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    active_models: List[str]

class ModelInfo(BaseModel):
    model_id: str
    model_name: str
    category: str
    architecture: str
    framework: str
    size_mb: float
    task: str
    license: str
    status: str
