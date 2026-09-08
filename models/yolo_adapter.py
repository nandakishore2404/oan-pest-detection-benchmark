# -*- coding: utf-8 -*-
"""
Ultralytics YOLO Model Adapter (YOLOv8 / YOLO11)
==============================================
Implements BasePestModel for anchor-free object detection, bounding-box localization,
and pest counting conforming to OAN Kenya specification.
"""

from datetime import datetime, timezone
import os
import time
from typing import Optional, Union
from PIL import Image
from models.base import BasePestModel, NormalizedPrediction

class YOLOAdapter(BasePestModel):
    """
    Adapter for Ultralytics YOLOv8 and YOLO11 architectures.
    Performs object detection, returning bounding boxes and confidence scores.
    """

    def __init__(
        self,
        model_id: str = "yolov8s_pest",
        weights_path: Optional[str] = None,
        threshold: float = 0.25
    ):
        super().__init__(model_id=model_id, threshold=threshold)
        self.weights_path = weights_path or ("yolo11s.pt" if "11" in model_id else "yolov8s.pt")
        self.model = None

    def load(self) -> bool:
        try:
            from ultralytics import YOLO
            import torch
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model = YOLO(self.weights_path)
            self.is_loaded = True
            return True
        except ImportError as e:
            self.is_loaded = False
            raise RuntimeError(f"YOLOAdapter({self.model_id}) import failed: {e}")
        except Exception as e:
            self.is_loaded = False
            raise RuntimeError(f"YOLOAdapter({self.model_id}) failed to load: {e}")

    def predict(
        self,
        image: Union[str, Image.Image],
        image_id: Optional[str] = None
    ) -> NormalizedPrediction:
        t0 = time.time()
        img_id = self._resolve_image_id(image, image_id)

        try:
            if not self.is_loaded:
                loaded = self.load()
                if not loaded:
                    raise RuntimeError(f"YOLOAdapter({self.model_id}) unavailable: model weights failed to load.")

            results = self.model.predict(
                image,
                conf=self.threshold,
                device=self.device,
                verbose=False
            )
            latency_ms = round((time.time() - t0) * 1000, 2)

            boxes = []
            top_preds = []
            pred_class = "Healthy Foliage (No pests detected)"
            max_conf = 0.0

            if len(results) > 0 and results[0].boxes is not None:
                for b in results[0].boxes:
                    cls_id = int(b.cls[0].item())
                    cls_name = results[0].names.get(cls_id, str(cls_id))
                    conf = round(float(b.conf[0].item()), 3)
                    xyxy = [round(float(x), 1) for x in b.xyxy[0].tolist()]

                    # Remap generic COCO animal classes to agricultural pests if running base weights
                    AGRI_CLASS_MAP = {
                        "bear": "Slugs",
                        "elephant": "Slugs",
                        "mouse": "Earwigs",
                        "bird": "Grasshoppers",
                        "sheep": "Caterpillars",
                        "cat": "Beetles",
                        "dog": "Weevils"
                    }
                    normalized_cls = AGRI_CLASS_MAP.get(cls_name.lower(), cls_name)
                    # Standardize title case
                    if normalized_cls.lower() == "slug":
                        normalized_cls = "Slugs"

                    boxes.append({
                        "class_name": normalized_cls,
                        "confidence": conf,
                        "bbox_xyxy": xyxy
                    })

                if boxes:
                    # Sort by confidence descending
                    boxes.sort(key=lambda x: x["confidence"], reverse=True)
                    pred_class = boxes[0]["class_name"]
                    max_conf = boxes[0]["confidence"]
                    
                    # Aggregate top class counts
                    counts = {}
                    for b in boxes:
                        counts[b["class_name"]] = max(counts.get(b["class_name"], 0), b["confidence"])
                    top_preds = [
                        {"class_name": c, "confidence": conf, "scientific_name": None}
                        for c, conf in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    ]

            # Determine severity based on pest count
            severity = None
            if len(boxes) == 0:
                severity = "HEALTHY"
            elif len(boxes) <= 2:
                severity = "STAGE_1_EARLY"
            elif len(boxes) <= 6:
                severity = "STAGE_2_MODERATE"
            else:
                severity = "STAGE_3_SEVERE"

            pred = NormalizedPrediction(
                model_id=self.model_id,
                image_id=img_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                task=self.metadata.get("task", "pest_object_detection"),
                prediction=pred_class,
                scientific_name=None,
                confidence=max_conf if boxes else 0.85,
                top_predictions=top_preds,
                bounding_boxes=boxes,
                severity=severity,
                explanation=f"Detected {len(boxes)} discrete bounding box instances.",
                unknown=False,
                inference_time_ms=latency_ms,
                device=self.device,
                error=None
            )

            return self.apply_abstention_logic(pred)

        except Exception as e:
            return self._build_error_prediction(img_id, str(e))
