# -*- coding: utf-8 -*-
"""
Base Model Adapter & Normalized Output Schema
==============================================
Defines the standard BasePestModel abstract interface and NormalizedPrediction Pydantic schema
conforming strictly to Sections 7, 8, and 9 of the OAN Kenya Benchmarking Specification.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import os
import time
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from PIL import Image

class BoundingBox(BaseModel):
    """Standardized bounding box representation for object detection models."""
    class_name: str
    confidence: float
    bbox_xyxy: List[float] = Field(description="Bounding box coordinates in [x1, y1, x2, y2] format")

class TopPrediction(BaseModel):
    """Standardized entry in top-N prediction distributions."""
    class_name: str
    confidence: float
    scientific_name: Optional[str] = None

class NormalizedPrediction(BaseModel):
    """
    Normalized prediction schema returned by ALL model adapters.
    Conforms strictly to Section 8 of OAN Kenya Benchmarking Specification.
    Fields unsupported by a particular architecture MUST be set to null.
    """
    model_id: str
    image_id: str = "unknown"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    task: str = "crop_pest_disease_detection"
    prediction: str = "Unknown / Unsupported Class"
    scientific_name: Optional[str] = None
    confidence: float = 0.0
    top_predictions: List[Dict[str, Any]] = Field(default_factory=list)
    bounding_boxes: List[Dict[str, Any]] = Field(default_factory=list)
    severity: Optional[str] = None
    explanation: str = ""
    unknown: bool = False
    inference_time_ms: float = 0.0
    device: str = "cpu"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

class BasePestModel(ABC):
    """
    Abstract base adapter that every candidate AI model must implement.
    The benchmark engine interacts with models SOLELY through this interface.
    """

    def __init__(self, model_id: str, threshold: float = 0.50):
        self.model_id = model_id
        self.threshold = threshold
        self.is_loaded = False
        self.device = "cpu"
        self._load_metadata()

    def _load_metadata(self):
        """Loads model metadata from models/model_registry.yaml."""
        from models.registry import get_model_metadata
        self.metadata = get_model_metadata(self.model_id) or {}

    @abstractmethod
    def load(self) -> bool:
        """
        Loads model weights into memory/device.
        Returns True if loaded successfully, False or raises on failure.
        """
        pass

    @abstractmethod
    def predict(
        self,
        image: Union[str, Image.Image],
        image_id: Optional[str] = None
    ) -> NormalizedPrediction:
        """
        Executes inference on a single image.
        Args:
            image: Path to image file or PIL Image object.
            image_id: Optional identifier for tracking/dataset alignment.
        Returns:
            NormalizedPrediction schema object.
        """
        pass

    def predict_batch(
        self,
        images: List[Union[str, Image.Image]],
        image_ids: Optional[List[str]] = None
    ) -> List[NormalizedPrediction]:
        """
        Batch prediction default implementation.
        Can be overridden by adapters with native vectorized batching.
        """
        results = []
        if image_ids is None:
            image_ids = [f"img_{i:04d}" for i in range(len(images))]
        
        for img, img_id in zip(images, image_ids):
            try:
                pred = self.predict(img, image_id=img_id)
            except Exception as e:
                pred = self._build_error_prediction(img_id, str(e))
            results.append(pred)
        return results

    def get_metadata(self) -> Dict[str, Any]:
        """Returns verified model metadata and technical specifications."""
        return self.metadata

    def health_check(self) -> Dict[str, Any]:
        """Verifies model loading state, device allocation, and memory footprint."""
        return {
            "model_id": self.model_id,
            "is_loaded": self.is_loaded,
            "device": self.device,
            "status": "HEALTHY" if self.is_loaded else "NOT_LOADED",
            "threshold": self.threshold
        }

    def apply_abstention_logic(self, pred: NormalizedPrediction) -> NormalizedPrediction:
        """
        Enforces Section 9: Unknown / Abstention logic.
        If model confidence falls below the configured threshold, flags unknown = True.
        """
        if pred.confidence < self.threshold and not pred.unknown:
            pred.unknown = True
            pred.prediction = "Unknown / Unsupported Class"
            pred.explanation = (
                f"Model confidence ({pred.confidence:.2f}) is below the configured safety threshold "
                f"({self.threshold:.2f}). Expert agronomic validation recommended."
            )
        return pred

    def _build_error_prediction(self, image_id: str, error_msg: str) -> NormalizedPrediction:
        """Constructs a standardized error prediction response."""
        return NormalizedPrediction(
            model_id=self.model_id,
            image_id=image_id or "unknown",
            timestamp=datetime.now(timezone.utc).isoformat(),
            task=self.metadata.get("task", "unknown"),
            prediction="Error",
            scientific_name=None,
            confidence=0.0,
            top_predictions=[],
            bounding_boxes=[],
            severity=None,
            explanation=f"Inference failed: {error_msg}",
            unknown=True,
            inference_time_ms=0.0,
            device=self.device,
            error=error_msg
        )

    def _resolve_image_id(self, image: Union[str, Image.Image], image_id: Optional[str]) -> str:
        """Extracts or generates an image identifier by hashing bytes to eliminate filename leakage."""
        if image_id:
            return image_id
        import hashlib
        if isinstance(image, str) and os.path.exists(image):
            try:
                with open(image, "rb") as f:
                    return hashlib.sha256(f.read()).hexdigest()[:16]
            except Exception:
                pass
        return f"img_{int(time.time() * 1000)}"
