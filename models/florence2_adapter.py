# -*- coding: utf-8 -*-
"""
Microsoft Florence-2 Vision Foundation Model Adapter
====================================================
Implements BasePestModel for Florence-2-base / Florence-2-large.
Supports multi-task visual pest grounding, object detection, and detailed agricultural captioning
conforming to OAN Kenya specification.
"""

from datetime import datetime, timezone
import os
import time
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from models.base import BasePestModel, NormalizedPrediction

class Florence2Adapter(BasePestModel):
    """
    Adapter for Microsoft Florence-2 foundation vision model.
    Executes object detection (<OD>) and phrase grounding on crop imagery.
    """

    def __init__(
        self,
        model_id: str = "florence_2_base",
        hf_model_name: str = "microsoft/Florence-2-base",
        task_prompt: str = "<OD>",
        threshold: float = 0.30
    ):
        super().__init__(model_id=model_id, threshold=threshold)
        self.hf_model_name = hf_model_name
        self.task_prompt = task_prompt
        self.model = None
        self.processor = None

    def load(self) -> bool:
        """Loads Florence-2 processor and model from Hugging Face."""
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM

            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if "cuda" in self.device else torch.float32

            self.processor = AutoProcessor.from_pretrained(
                self.hf_model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            ).to(self.device)
            self.model.eval()

            self.is_loaded = True
            return True
        except ImportError as e:
            self.is_loaded = False
            raise RuntimeError(f"Florence2Adapter({self.model_id}) import failed: {e}")
        except Exception as e:
            self.is_loaded = False
            raise RuntimeError(f"Florence2Adapter({self.model_id}) failed to load: {e}")

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
                    raise RuntimeError(f"Florence2Adapter({self.model_id}) unavailable: model weights failed to load.")

            import torch

            if isinstance(image, str):
                pil_img = Image.open(image).convert("RGB")
            else:
                pil_img = image.convert("RGB")

            inputs = self.processor(
                text=self.task_prompt,
                images=pil_img,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False
                )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=self.task_prompt,
                image_size=(pil_img.width, pil_img.height)
            )

            latency_ms = round((time.time() - t0) * 1000, 2)

            boxes = []
            top_preds = []
            detected_data = parsed_answer.get(self.task_prompt, {})
            bboxes = detected_data.get("bboxes", [])
            labels = detected_data.get("labels", [])

            for bbox, label in zip(bboxes, labels):
                boxes.append({
                    "class_name": label,
                    "confidence": 0.85,  # Florence-2 output format does not produce explicit confidence per box
                    "bbox_xyxy": [round(float(coord), 1) for coord in bbox]
                })

            pred_class = labels[0] if labels else "Healthy / No specific pests isolated"
            max_conf = 0.85 if labels else 0.50

            # Estimate severity based on detection count
            severity = None
            if len(boxes) == 0:
                severity = "HEALTHY"
            elif len(boxes) <= 2:
                severity = "STAGE_1_EARLY"
            elif len(boxes) <= 5:
                severity = "STAGE_2_MODERATE"
            else:
                severity = "STAGE_3_SEVERE"

            pred = NormalizedPrediction(
                model_id=self.model_id,
                image_id=img_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                task=self.metadata.get("task", "open_vocabulary_detection"),
                prediction=pred_class,
                scientific_name=None,
                confidence=max_conf,
                top_predictions=[{"class_name": l, "confidence": 0.85, "scientific_name": None} for l in set(labels)],
                bounding_boxes=boxes,
                severity=severity,
                explanation=f"Florence-2 prompt '{self.task_prompt}' localized {len(boxes)} visual regions.",
                unknown=False,
                inference_time_ms=latency_ms,
                device=self.device,
                error=None
            )

            return self.apply_abstention_logic(pred)

        except Exception as e:
            return self._build_error_prediction(img_id, str(e))
