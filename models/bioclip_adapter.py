# -*- coding: utf-8 -*-
"""
BioCLIP (TreeOfLife-10M) Zero-Shot Model Adapter
================================================
Implements BasePestModel for BioCLIP zero-shot taxonomic visual classification.
Evaluates candidate pests and diseases across East African biological taxa
conforming to the OAN Kenya Benchmarking Specification.
"""

from datetime import datetime, timezone
import os
import time
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from models.base import BasePestModel, NormalizedPrediction

# Predefined candidate taxa for East African agricultural pests and diseases
DEFAULT_BIOCLIP_CANDIDATES = [
    {
        "common_name": "Fall Armyworm",
        "scientific_name": "Spodoptera frugiperda",
        "crop": "Maize",
        "prompt": "a photo of Spodoptera frugiperda, fall armyworm on maize"
    },
    {
        "common_name": "African Maize Stalk Borer",
        "scientific_name": "Busseola fusca",
        "crop": "Maize",
        "prompt": "a photo of Busseola fusca, maize stalk borer larva"
    },
    {
        "common_name": "Maize Lethal Necrosis",
        "scientific_name": "Maize chlorotic mottle virus",
        "crop": "Maize",
        "prompt": "a photo of maize lethal necrosis disease with chlorotic mottle"
    },
    {
        "common_name": "Bean Angular Leaf Spot",
        "scientific_name": "Pseudocercospora griseola",
        "crop": "Common Bean",
        "prompt": "a photo of Pseudocercospora griseola, angular leaf spot lesions on bean"
    },
    {
        "common_name": "Bean Common Rust",
        "scientific_name": "Uromyces appendiculatus",
        "crop": "Common Bean",
        "prompt": "a photo of Uromyces appendiculatus, rust pustules on bean leaf"
    },
    {
        "common_name": "Potato Late Blight",
        "scientific_name": "Phytophthora infestans",
        "crop": "Potato",
        "prompt": "a photo of Phytophthora infestans, potato late blight water-soaked lesions"
    },
    {
        "common_name": "Tomato Spider Mite",
        "scientific_name": "Tetranychus evansi",
        "crop": "Tomato",
        "prompt": "a photo of Tetranychus evansi, tomato red spider mite infestation and webbing"
    },
    {
        "common_name": "Desert Locust",
        "scientific_name": "Schistocerca gregaria",
        "crop": "Cereals",
        "prompt": "a photo of Schistocerca gregaria, desert locust"
    },
    {
        "common_name": "Healthy Crop Foliage",
        "scientific_name": None,
        "crop": "All",
        "prompt": "a photo of clean healthy green crop leaves with no pests or disease"
    }
]

class BioCLIPAdapter(BasePestModel):
    """
    BioCLIP zero-shot taxonomic classifier adapter.
    Uses vision-language contrastive embeddings to rank candidate agricultural taxa.
    """

    def __init__(
        self,
        model_id: str = "bioclip_zero_shot",
        candidates: Optional[List[Dict[str, Any]]] = None,
        threshold: float = 0.40
    ):
        super().__init__(model_id=model_id, threshold=threshold)
        self.candidates = candidates or DEFAULT_BIOCLIP_CANDIDATES
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.text_features = None

    def load(self) -> bool:
        """Loads BioCLIP model via open_clip or Hugging Face."""
        try:
            import torch
            import open_clip

            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

            # BioCLIP is typically loaded as ViT-B-16 with hf-hub:imageomics/bioclip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'hf-hub:imageomics/bioclip'
            )
            self.tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
            self.model.to(self.device)
            self.model.eval()

            # Pre-compute text features for the candidate pool
            prompts = [c["prompt"] for c in self.candidates]
            text_tokens = self.tokenizer(prompts).to(self.device)
            with torch.no_grad():
                text_emb = self.model.encode_text(text_tokens)
                self.text_features = text_emb / text_emb.norm(dim=-1, keepdim=True)

            self.is_loaded = True
            return True
        except ImportError as e:
            self.is_loaded = False
            raise RuntimeError(f"BioCLIPAdapter({self.model_id}) import failed: {e}")
        except Exception as e:
            self.is_loaded = False
            raise RuntimeError(f"BioCLIPAdapter({self.model_id}) failed to load: {e}")

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
                    raise RuntimeError(f"BioCLIPAdapter({self.model_id}) unavailable: model weights failed to load.")

            import torch

            if isinstance(image, str):
                pil_img = Image.open(image).convert("RGB")
            else:
                pil_img = image.convert("RGB")

            image_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Cosine similarity * logit scale
                similarity = (image_features @ self.text_features.T) * 100.0
                probs = torch.softmax(similarity, dim=-1)[0]

            latency_ms = round((time.time() - t0) * 1000, 2)

            topk_probs, topk_indices = torch.topk(probs, min(5, len(probs)))
            top_preds = []

            for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
                c = self.candidates[idx]
                top_preds.append({
                    "class_name": f"{c['crop']} - {c['common_name']}",
                    "confidence": round(float(prob), 3),
                    "scientific_name": c["scientific_name"]
                })

            top_prediction = top_preds[0]["class_name"]
            top_sci = top_preds[0]["scientific_name"]
            top_conf = top_preds[0]["confidence"]

            pred = NormalizedPrediction(
                model_id=self.model_id,
                image_id=img_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                task=self.metadata.get("task", "zero-shot-classification"),
                prediction=top_prediction,
                scientific_name=top_sci,
                confidence=top_conf,
                top_predictions=top_preds,
                bounding_boxes=[],
                severity=None,
                explanation=f"BioCLIP zero-shot taxonomic match across {len(self.candidates)} East African candidate taxa.",
                unknown=False,
                inference_time_ms=latency_ms,
                device=self.device,
                error=None
            )

            return self.apply_abstention_logic(pred)

        except Exception as e:
            return self._build_error_prediction(img_id, str(e))
