# -*- coding: utf-8 -*-
"""
Deterministic Mock Pest Detection Adapter
=========================================
Provides a lightweight, deterministic adapter for testing the benchmarking pipeline,
API endpoints, metrics calculation, and UI without requiring heavy model downloads or GPU.
"""

from datetime import datetime, timezone
import os
import time
from typing import Optional, Union
from PIL import Image
from models.base import BasePestModel, NormalizedPrediction

class MockPestModel(BasePestModel):
    """
    Deterministic mock adapter that simulates pest detection and foliar classification.
    Ideal for unit tests, offline continuous integration, and rapid verification.
    """

    def __init__(self, model_id: str = "mock_baseline_model", threshold: float = 0.50):
        super().__init__(model_id=model_id, threshold=threshold)
        self.device = "cpu"

    def load(self) -> bool:
        self.is_loaded = True
        return True

    def predict(
        self,
        image: Union[str, Image.Image],
        image_id: Optional[str] = None
    ) -> NormalizedPrediction:
        t0 = time.time()
        img_id = self._resolve_image_id(image, image_id)

        # Inspect filename or metadata to generate realistic deterministic response
        img_name = img_id.lower()
        is_yolo = "yolo" in self.model_id.lower()
        is_florence = "florence" in self.model_id.lower()
        is_cereal = "cereal" in self.model_id.lower() or "pestaid" in self.model_id.lower()
        is_bioclip = "bioclip" in self.model_id.lower()

        # 12-Class Pest Dataset mapping
        pest_db = {
            "catterpillar": ("Caterpillars", "Lepidoptera (larva)", "STAGE_2_MODERATE", "Chewing defoliation on leaf blade and whorl."),
            "caterpillar": ("Caterpillars", "Lepidoptera (larva)", "STAGE_2_MODERATE", "Chewing defoliation on leaf blade and whorl."),
            "armyworm": ("Caterpillars (Fall Armyworm)", "Spodoptera frugiperda", "STAGE_2_MODERATE", "Distinct whorl feeding windowpaning and frass clusters."),
            "faw": ("Caterpillars (Fall Armyworm)", "Spodoptera frugiperda", "STAGE_2_MODERATE", "Distinct whorl feeding windowpaning and frass clusters."),
            "beetle": ("Beetles", "Coleoptera", "STAGE_2_MODERATE", "Hard-shelled beetle feeding on leaf and stem tissues."),
            "weevil": ("Weevils", "Curculionoidea", "STAGE_2_MODERATE", "Snout beetle causing granular boring damage in grains and storage."),
            "grasshopper": ("Grasshoppers", "Caelifera", "STAGE_2_MODERATE", "Irregular leaf margin stripping and severe defoliation."),
            "earwig": ("Earwigs", "Dermaptera", "STAGE_1_MILD", "Pincer-bearing insect observed in damp plant crevices."),
            "ant": ("Ants", "Formicidae", "STAGE_1_MILD", "Ant activity associated with aphid honeydew guarding."),
            "wasp": ("Wasps", "Hymenoptera", "STAGE_1_MILD", "Solitary/parasitoid wasp active near foliage."),
            "snail": ("Snails", "Gastropoda", "STAGE_2_MODERATE", "Mucus trails and irregular rasping foliage perforations."),
            "slug": ("Slugs", "Gastropoda", "STAGE_2_MODERATE", "Molluscan feeding damage on young seedlings with slime trails."),
            "earthworm": ("Earthworms", "Lumbricina", "HEALTHY", "Beneficial soil organism contributing to soil aeration."),
            "moth": ("Moths", "Lepidoptera (adult)", "STAGE_1_MILD", "Adult reproductive moth on host crop foliage."),
            "bee": ("Bees", "Apis mellifera", "HEALTHY", "Beneficial pollinator visiting crop blossoms."),
            "blight": ("Late Blight", "Phytophthora infestans", "STAGE_3_SEVERE", "Water-soaked expanding brown lesions with chlorotic necrotic border."),
            "als": ("Angular Leaf Spot", "Pseudocercospora griseola", "STAGE_2_MODERATE", "Angular vein-delimited polygonal necrotic lesions on bean foliage."),
            "bean": ("Angular Leaf Spot", "Pseudocercospora griseola", "STAGE_2_MODERATE", "Angular vein-delimited polygonal necrotic lesions on bean foliage.")
        }

        matched_key = None
        for k in pest_db.keys():
            # Avoid partial false matches like 'bee' in 'beetle'
            if k == "bee" and "beetle" in img_name:
                continue
            if k in img_name:
                matched_key = k
                break
        if matched_key is None and isinstance(image, str) and os.path.exists(image):
            # Check if primary YOLO vision model detects an active pest
            try:
                from models.yolo_adapter import YOLOAdapter
                y_adapter = YOLOAdapter()
                y_pred = y_adapter.predict(image)
                if y_pred.prediction and "healthy" not in y_pred.prediction.lower() and y_pred.confidence >= 0.35:
                    pred_lower = y_pred.prediction.lower()
                    for k in pest_db.keys():
                        if k in pred_lower:
                            matched_key = k
                            break
            except Exception:
                pass

        if matched_key:
            p_name, scientific, sev, expl = pest_db[matched_key]
            pred_class = p_name
            conf = 0.92

            # CerealPestAID specialization check: only high confidence on cereal pests
            if is_cereal and matched_key not in {"catterpillar", "caterpillar", "armyworm", "faw", "weevil", "grasshopper", "beetle", "moth"}:
                conf = 0.28  # Abstained on non-cereal soil pests
                expl = "Out-of-distribution for African Cereal Pest model."

            # iBean specialization check: only high confidence on bean diseases
            if "ibean" in self.model_id.lower() and matched_key not in {"bean", "als"}:
                conf = 0.22
                expl = "Out-of-distribution for iBean East Africa model."

            # Bounding box generation: only Object Detection & Grounding models produce boxes
            bboxes = []
            if (is_yolo or is_florence) and conf >= self.threshold:
                bboxes = [
                    {"class_name": pred_class.lower(), "confidence": conf, "bbox_xyxy": [120.0, 160.0, 340.0, 390.0]}
                ]

            top_preds = [
                {"class_name": pred_class, "confidence": conf, "scientific_name": scientific},
                {"class_name": "Alternative Host Pest", "confidence": round((1.0 - conf) * 0.6, 2), "scientific_name": None},
                {"class_name": "Healthy Foliage", "confidence": round((1.0 - conf) * 0.4, 2), "scientific_name": None}
            ]
            severity = sev
            explanation = expl
        else:
            # Fallback Healthy / Unidentified
            pred_class = "Healthy Foliage"
            scientific = None
            conf = 0.72
            bboxes = []
            top_preds = [
                {"class_name": "Healthy Foliage", "confidence": 0.72, "scientific_name": None},
                {"class_name": "Early Blight", "confidence": 0.18, "scientific_name": "Alternaria solani"},
                {"class_name": "Nutrient Deficiency", "confidence": 0.10, "scientific_name": None}
            ]
            severity = "HEALTHY"
            explanation = "No macroscopic pathogenic lesions or discrete insect pests detected."

        latency_ms = round((time.time() - t0 + 0.012) * 1000, 2)

        pred = NormalizedPrediction(
            model_id=self.model_id,
            image_id=img_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            task=self.metadata.get("task", "general_pest_detection"),
            prediction=pred_class,
            scientific_name=scientific,
            confidence=conf,
            top_predictions=top_preds,
            bounding_boxes=bboxes,
            severity=severity,
            explanation=explanation,
            unknown=False,
            inference_time_ms=latency_ms,
            device=self.device,
            error=None
        )

        return self.apply_abstention_logic(pred)
