# -*- coding: utf-8 -*-
"""
CerealPestAID Model Adapter
===========================
Implements BasePestModel for CerealPestAID (Sheneman et al., 2024).
Specialized benchmark model covering 26 species of cereal crop pests (Maize, Wheat, Sorghum, Rice)
with PyTorch / ONNX / TFLite runtime support conforming to OAN Kenya specification.
"""

from datetime import datetime, timezone
import os
import time
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from models.base import BasePestModel, NormalizedPrediction

CEREAL_PESTAID_26_SPECIES = [
    {"class": "Fall Armyworm", "scientific_name": "Spodoptera frugiperda"},
    {"class": "African Armyworm", "scientific_name": "Spodoptera exempta"},
    {"class": "African Maize Stalk Borer", "scientific_name": "Busseola fusca"},
    {"class": "Pink Stalk Borer", "scientific_name": "Sesamia calamistis"},
    {"class": "Spotted Stem Borer", "scientific_name": "Chilo partellus"},
    {"class": "Desert Locust", "scientific_name": "Schistocerca gregaria"},
    {"class": "Migratory Locust", "scientific_name": "Locusta migratoria"},
    {"class": "Cereal Aphid", "scientific_name": "Rhopalosiphum padi"},
    {"class": "Corn Leaf Aphid", "scientific_name": "Rhopalosiphum maidis"},
    {"class": "Greenbug", "scientific_name": "Schizaphis graminum"},
    {"class": "Brown Planthopper", "scientific_name": "Nilaparvata lugens"},
    {"class": "Whitebacked Planthopper", "scientific_name": "Sogatella furcifera"},
    {"class": "Rice Leaf Folder", "scientific_name": "Cnaphalocrocis medinalis"},
    {"class": "Rice Stem Borer", "scientific_name": "Scirpophaga incertulas"},
    {"class": "Rice Gall Midge", "scientific_name": "Orseolia oryzae"},
    {"class": "Rice Bug", "scientific_name": "Leptocorisa acuta"},
    {"class": "Sorghum Midge", "scientific_name": "Contarinia sorghicola"},
    {"class": "Sorghum Shoot Fly", "scientific_name": "Atherigona soccata"},
    {"class": "Wheat Stem Sawfly", "scientific_name": "Cephus cinctus"},
    {"class": "Sunn Pest", "scientific_name": "Eurygaster integriceps"},
    {"class": "True Armyworm", "scientific_name": "Mythimna unipuncta"},
    {"class": "Black Cutworm", "scientific_name": "Agrotis ipsilon"},
    {"class": "Lined Click Beetle Wireworm", "scientific_name": "Agriotes lineatus"},
    {"class": "Corn Earworm", "scientific_name": "Helicoverpa zea"},
    {"class": "Southern Green Stink Bug", "scientific_name": "Nezara viridula"},
    {"class": "Healthy Cereal Leaf", "scientific_name": None}
]

class CerealPestAIDAdapter(BasePestModel):
    """
    Adapter for CerealPestAID specialized cereal pest diagnostic models.
    Supports ONNX runtime execution or PyTorch weights.
    """

    def __init__(
        self,
        model_id: str = "cereal_pestaid_b6",
        weights_path: Optional[str] = None,
        threshold: float = 0.50
    ):
        super().__init__(model_id=model_id, threshold=threshold)
        self.weights_path = weights_path
        self.species_list = CEREAL_PESTAID_26_SPECIES
        self.session = None
        self.runtime = "none"

    def load(self) -> bool:
        """Loads ONNX runtime session or PyTorch model weights."""
        # Try ONNX Runtime first (most portable for edge deployment)
        if self.weights_path and self.weights_path.endswith(".onnx"):
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(self.weights_path)
                self.device = "cuda:0" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
                self.runtime = "onnx"
                self.is_loaded = True
                return True
            except Exception as e:
                print(f"Warning: ONNX loading failed: {e}")

        # Try PyTorch
        try:
            import torch
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if self.weights_path and os.path.exists(self.weights_path):
                self.session = torch.load(self.weights_path, map_location=self.device)
                self.runtime = "pytorch"
                self.is_loaded = True
                return True
        except Exception:
            pass

        self.is_loaded = False
        return False

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
                    raise RuntimeError(f"CerealPestAIDAdapter({self.model_id}) unavailable: weights failed to load.")

            import numpy as np

            # Image preprocessing for standard CerealPestAID (224x224 or 528x528 normalized)
            if isinstance(image, str):
                pil_img = Image.open(image).convert("RGB")
            else:
                pil_img = image.convert("RGB")

            img_resized = pil_img.resize((224, 224))
            arr = np.array(img_resized, dtype=np.float32) / 255.0
            # Mean and std normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std
            arr = np.transpose(arr, (2, 0, 1))  # HWC to CHW
            tensor = np.expand_dims(arr, axis=0)  # NCHW

            if self.runtime == "onnx":
                input_name = self.session.get_inputs()[0].name
                raw_out = self.session.run(None, {input_name: tensor})[0][0]
                # Softmax
                exp_vals = np.exp(raw_out - np.max(raw_out))
                probs = exp_vals / np.sum(exp_vals)
            else:
                import torch
                t_tensor = torch.from_numpy(tensor).to(self.device)
                with torch.no_grad():
                    logits = self.session(t_tensor)
                    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            latency_ms = round((time.time() - t0) * 1000, 2)

            top_indices = np.argsort(probs)[::-1][:5]
            top_preds = []

            for idx in top_indices:
                if idx < len(self.species_list):
                    item = self.species_list[idx]
                    top_preds.append({
                        "class_name": item["class"],
                        "confidence": round(float(probs[idx]), 3),
                        "scientific_name": item["scientific_name"]
                    })

            top_pred = top_preds[0]

            pred = NormalizedPrediction(
                model_id=self.model_id,
                image_id=img_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                task=self.metadata.get("task", "cereal_pest_classification"),
                prediction=top_pred["class_name"],
                scientific_name=top_pred["scientific_name"],
                confidence=top_pred["confidence"],
                top_predictions=top_preds,
                bounding_boxes=[],
                severity=None,
                explanation=f"Identified using CerealPestAID 26-species diagnostic model with {top_pred['confidence']:.1%} confidence.",
                unknown=False,
                inference_time_ms=latency_ms,
                device=self.device,
                error=None
            )

            return self.apply_abstention_logic(pred)

        except Exception as e:
            return self._build_error_prediction(img_id, str(e))
