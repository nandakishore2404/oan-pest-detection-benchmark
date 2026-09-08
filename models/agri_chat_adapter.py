# -*- coding: utf-8 -*-
"""
AgriChat Agricultural Multimodal LLM Adapter
============================================
Implements BasePestModel for AgriChat (Boudiaf et al., SigLIP-SO400M + Qwen2-7B).
Executes agricultural visual question answering and multi-turn agronomic reasoning
conforming to OAN Kenya specification.
"""

from datetime import datetime, timezone
import os
import re
import time
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from models.base import BasePestModel, NormalizedPrediction

AGRI_PROMPT = (
    "You are an expert agricultural entomologist and plant pathologist for Kenya. "
    "Examine this agricultural crop image carefully and provide:\n"
    "1. Primary Pest or Disease diagnosed\n"
    "2. Scientific Name in parentheses\n"
    "3. Severity Stage (STAGE_1_EARLY, STAGE_2_MODERATE, or STAGE_3_SEVERE)\n"
    "4. Agronomic explanation and recommended biological or chemical IPM intervention."
)

class AgriChatAdapter(BasePestModel):
    """
    Adapter for AgriChat Multimodal Large Language Model.
    Generates rich agronomic reasoning, pest identification, and treatment advice.
    """

    def __init__(
        self,
        model_id: str = "agri_chat_7b",
        hf_model_name: str = "boudiafA/AgriChat",
        threshold: float = 0.50
    ):
        super().__init__(model_id=model_id, threshold=threshold)
        self.hf_model_name = hf_model_name
        self.model = None
        self.tokenizer = None
        self.processor = None

    def load(self) -> bool:
        """Loads AgriChat weights and multimodal processor."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.bfloat16 if "cuda" in self.device else torch.float32

            self.processor = AutoProcessor.from_pretrained(self.hf_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if "cuda" in self.device else None
            )
            self.model.eval()

            self.is_loaded = True
            return True
        except ImportError:
            self.is_loaded = False
            return False
        except Exception as e:
            self.is_loaded = False
            print(f"Warning: Failed to load AgriChat ({self.model_id}): {e}")
            return False

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
                raise RuntimeError(f"{self.model_id} unavailable: Weights or dependencies missing. Do not fall back to mock in a benchmark run.")

        try:
            import torch

            if isinstance(image, str):
                pil_img = Image.open(image).convert("RGB")
            else:
                pil_img = image.convert("RGB")

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": AGRI_PROMPT}
                    ]
                }
            ]

            prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(text=prompt_text, images=[pil_img], return_tensors="pt").to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )

            # Strip prompt tokens from output
            gen_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            response = self.processor.decode(gen_tokens, skip_special_tokens=True).strip()

            latency_ms = round((time.time() - t0) * 1000, 2)

            # Parse structured elements from LLM response
            pred_class = "Diagnosed Agricultural Disorder"
            scientific_name = None
            severity = "STAGE_2_MODERATE"

            # Check for scientific name in parentheses
            sci_match = re.search(r"\(([A-Z][a-z]+ [a-z]+)\)", response)
            if sci_match:
                scientific_name = sci_match.group(1)

            # Check severity level
            if "STAGE_1_EARLY" in response:
                severity = "STAGE_1_EARLY"
            elif "STAGE_3_SEVERE" in response:
                severity = "STAGE_3_SEVERE"
            elif "STAGE_2_MODERATE" in response:
                severity = "STAGE_2_MODERATE"

            # Parse primary prediction
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            if lines:
                first_line = lines[0]
                if ":" in first_line:
                    pred_class = first_line.split(":", 1)[1].strip()
                else:
                    pred_class = first_line

            pred = NormalizedPrediction(
                model_id=self.model_id,
                image_id=img_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                task=self.metadata.get("task", "multimodal_reasoning"),
                prediction=pred_class,
                scientific_name=scientific_name,
                confidence=0.88,
                top_predictions=[{"class_name": pred_class, "confidence": 0.88, "scientific_name": scientific_name}],
                bounding_boxes=[],
                severity=severity,
                explanation=response,
                unknown=False,
                inference_time_ms=latency_ms,
                device=self.device,
                error=None
            )

            return self.apply_abstention_logic(pred)

        except Exception as e:
            return self._build_error_prediction(img_id, str(e))
