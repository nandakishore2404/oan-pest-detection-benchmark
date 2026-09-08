# -*- coding: utf-8 -*-
"""
PyTorch Timm Model Adapter (EfficientNet-B4 & MobileNetV4)
=========================================================
Implements BasePestModel for PyTorch Image Models (timm) architectures.
Supports edge-optimized classification backbones (EfficientNet-B4, MobileNetV4)
conforming strictly to the OAN Kenya Benchmarking Specification.
"""

from datetime import datetime, timezone
import os
import time
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from models.base import BasePestModel, NormalizedPrediction

KENYA_BENCHMARK_CLASSES = [
    {"class": "Maize Fall Armyworm", "scientific_name": "Spodoptera frugiperda"},
    {"class": "Maize Stalk Borer", "scientific_name": "Busseola fusca"},
    {"class": "Maize Lethal Necrosis", "scientific_name": "MCMV + SCMV Co-infection"},
    {"class": "Bean Angular Leaf Spot", "scientific_name": "Pseudocercospora griseola"},
    {"class": "Bean Common Rust", "scientific_name": "Uromyces appendiculatus"},
    {"class": "Potato Late Blight", "scientific_name": "Phytophthora infestans"},
    {"class": "Potato Early Blight", "scientific_name": "Alternaria solani"},
    {"class": "Tomato Spider Mite", "scientific_name": "Tetranychus evansi"},
    {"class": "Tomato Bacterial Wilt", "scientific_name": "Ralstonia solanacearum"},
    {"class": "Healthy Foliage", "scientific_name": None}
]

class TimmAdapter(BasePestModel):
    """
    Adapter for timm image classification models (EfficientNet, MobileNetV4, etc.).
    """

    def __init__(
        self,
        model_id: str = "efficientnet_b4_pest",
        architecture: Optional[str] = None,
        weights_path: Optional[str] = None,
        class_mapping: Optional[List[Dict[str, Optional[str]]]] = None,
        threshold: float = 0.50
    ):
        super().__init__(model_id=model_id, threshold=threshold)
        self.architecture = architecture or self.metadata.get("architecture", "efficientnet_b4")
        self.weights_path = weights_path
        self.class_mapping = class_mapping or KENYA_BENCHMARK_CLASSES
        self.model = None
        self.transform = None

    def load(self) -> bool:
        """Loads timm model and builds inference transforms."""
        try:
            import torch
            import timm
            from timm.data import create_transform, resolve_data_config

            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            pretrained = True if self.weights_path is None else False

            # Create timm model
            num_classes = len(self.class_mapping)
            try:
                self.model = timm.create_model(
                    self.architecture,
                    pretrained=pretrained,
                    num_classes=num_classes
                )
            except Exception:
                # Fallback to standard pretrained model if custom classes fails
                self.model = timm.create_model(self.architecture, pretrained=True)

            if self.weights_path and os.path.exists(self.weights_path):
                state_dict = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)

            self.model.to(self.device)
            self.model.eval()

            # Create standard timm transform pipeline
            data_config = resolve_data_config(self.model.pretrained_cfg)
            self.transform = create_transform(**data_config, is_training=False)

            self.is_loaded = True
            return True
        except ImportError as e:
            self.is_loaded = False
            raise RuntimeError(f"TimmAdapter({self.model_id}) import failed: {e}")
        except Exception as e:
            self.is_loaded = False
            raise RuntimeError(f"TimmAdapter({self.model_id}) failed to load: {e}")

    def predict(
        self,
        image: Union[str, Image.Image],
        image_id: Optional[str] = None
    ) -> NormalizedPrediction:
        t0 = time.time()
        img_id = self._resolve_image_id(image, image_id)

        if not self.is_loaded:
            loaded = self.load()
            if not loaded:
                raise RuntimeError(f"TimmAdapter({self.model_id}) unavailable: model weights failed to load.")

        try:
            import torch

            # Load and preprocess image
            if isinstance(image, str):
                pil_img = Image.open(image).convert("RGB")
            else:
                pil_img = image.convert("RGB")

            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=-1)[0]

            latency_ms = round((time.time() - t0) * 1000, 2)

            # Top-5 predictions
            topk_probs, topk_indices = torch.topk(probs, min(5, len(probs)))
            top_preds = []

            for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
                if idx < len(self.class_mapping):
                    c_info = self.class_mapping[idx]
                    c_name = c_info["class"]
                    sci_name = c_info["scientific_name"]
                else:
                    c_name = f"Class_{idx}"
                    sci_name = None

                top_preds.append({
                    "class_name": c_name,
                    "confidence": round(float(prob), 3),
                    "scientific_name": sci_name
                })

            top_class = top_preds[0]["class_name"]
            top_sci = top_preds[0]["scientific_name"]
            top_conf = top_preds[0]["confidence"]

            pred = NormalizedPrediction(
                model_id=self.model_id,
                image_id=img_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                task=self.metadata.get("task", "image-classification"),
                prediction=top_class,
                scientific_name=top_sci,
                confidence=top_conf,
                top_predictions=top_preds,
                bounding_boxes=[],  # Classification does not output bounding boxes
                severity=None,      # Classification alone does not estimate severity
                explanation=f"Top-1 classification prediction with {top_conf:.1%} confidence.",
                unknown=False,
                inference_time_ms=latency_ms,
                device=self.device,
                error=None
            )

            return self.apply_abstention_logic(pred)

        except Exception as e:
            return self._build_error_prediction(img_id, str(e))
