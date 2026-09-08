# -*- coding: utf-8 -*-
"""
Frontier Vision-Language Model Adapter (Gemini & OpenAI)
========================================================
Implements BasePestModel for commercial frontier visual reasoning models.
Used as benchmark gold-standard ceilings to evaluate open-weight models against
conforming to OAN Kenya specification.
"""

import base64
from datetime import datetime, timezone
import io
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from models.base import BasePestModel, NormalizedPrediction

FRONTIER_PROMPT = """You are an expert plant pathologist and entomologist evaluating agricultural imagery in Kenya.
Analyze the provided crop image and reply ONLY with a valid JSON object matching this schema:
{
  "prediction": "Name of pest, disease, or 'Healthy Foliage'",
  "scientific_name": "Scientific Latin genus/species or null",
  "confidence": 0.95,
  "severity": "HEALTHY | STAGE_1_EARLY | STAGE_2_MODERATE | STAGE_3_SEVERE",
  "explanation": "Detailed agronomic explanation and biological/chemical IPM recommendation.",
  "unknown": false
}
Do not wrap in markdown quotes. Just raw JSON.
"""

class FrontierVLMAdapter(BasePestModel):
    """
    Adapter for Frontier Vision APIs (Google Gemini, OpenAI GPT-4o).
    Evaluates proprietary models as upper-bound baselines.
    """

    def __init__(
        self,
        model_id: str = "gemini_1_5_pro",
        provider: str = "google",
        threshold: float = 0.50
    ):
        super().__init__(model_id=model_id, threshold=threshold)
        self.provider = provider or ("google" if "gemini" in model_id else "openai")
        self.api_key = None

    def load(self) -> bool:
        """Checks for presence of appropriate provider API keys."""
        if self.provider == "google":
            self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        elif self.provider == "openai":
            self.api_key = os.environ.get("OPENAI_API_KEY")

        if self.api_key:
            self.is_loaded = True
            self.device = "cloud_api"
            return True

        self.is_loaded = False
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
                raise RuntimeError(f"{self.model_id} unavailable: API credentials missing or initialization failed. Do not fall back to mock in a benchmark run.")

        try:
            # Prepare image bytes
            if isinstance(image, str):
                pil_img = Image.open(image).convert("RGB")
            else:
                pil_img = image.convert("RGB")

            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG", quality=85)
            img_bytes = buffered.getvalue()

            if self.provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-1.5-pro")
                response = model.generate_content([FRONTIER_PROMPT, pil_img])
                raw_text = response.text.strip()
            elif self.provider == "openai":
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                b64_img = base64.b64encode(img_bytes).decode("utf-8")
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": FRONTIER_PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=500
                )
                raw_text = response.choices[0].message.content.strip()
            else:
                raise ValueError(f"Unsupported frontier provider: {self.provider}")

            latency_ms = round((time.time() - t0) * 1000, 2)

            # Strip possible markdown code blocks ```json ... ```
            cleaned_json = re.sub(r"^```(json)?", "", raw_text).strip("`").strip()
            parsed = json.loads(cleaned_json)

            pred = NormalizedPrediction(
                model_id=self.model_id,
                image_id=img_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                task=self.metadata.get("task", "frontier_multimodal_baseline"),
                prediction=parsed.get("prediction", "Unknown"),
                scientific_name=parsed.get("scientific_name"),
                confidence=float(parsed.get("confidence", 0.90)),
                top_predictions=[{
                    "class_name": parsed.get("prediction", "Unknown"),
                    "confidence": float(parsed.get("confidence", 0.90)),
                    "scientific_name": parsed.get("scientific_name")
                }],
                bounding_boxes=[],
                severity=parsed.get("severity"),
                explanation=parsed.get("explanation", ""),
                unknown=bool(parsed.get("unknown", False)),
                inference_time_ms=latency_ms,
                device="cloud_api",
                error=None
            )

            return self.apply_abstention_logic(pred)

        except Exception as e:
            return self._build_error_prediction(img_id, str(e))
