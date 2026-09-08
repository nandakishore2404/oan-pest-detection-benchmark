# -*- coding: utf-8 -*-
"""
Ollama Local LLM & Vision Model Adapter
========================================
Lead Architect: Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>
Repository: https://github.com/nandakishore2404/oan-pest-detection-benchmark

Integrates local Ollama instances (e.g. Qwen 2.5 Coder 7B, LLaVA, Llama 3.2 Vision)
for offline agronomic advisory, multimodal visual question answering,
and continuous fine-tuning pipeline orchestration.
"""

import base64
import io
import json
import os
import time
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import requests

OLLAMA_DEFAULT_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

class OllamaAdvisor:
    """
    Client for local Ollama instances running Qwen 2.5 Coder or Vision models.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_DEFAULT_URL,
        model_name: str = "qwen2.5-coder:7b",
        timeout: int = 60
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """Check if Ollama is running and list available models."""
        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if res.status_code == 200:
                models = [m["name"] for m in res.json().get("models", [])]
                return {"status": "online", "available_models": models}
            return {"status": "error", "error": f"HTTP {res.status_code}"}
        except Exception as e:
            return {"status": "offline", "error": str(e)}

    def generate_advisory(
        self,
        diagnosis_results: Dict[str, Any],
        farmer_query: Optional[str] = None,
        language: str = "English"
    ) -> str:
        """
        Synthesizes a tailored agronomic action plan using local Qwen 2.5 Coder.
        """
        system_prompt = (
            "You are an expert agronomist, plant pathologist, and entomologist for the OpenAgriNet (OAN) Kenya program. "
            "Your task is to review the computer vision diagnosis of a farmer's crop and provide an actionable, "
            "practical, and pesticide-safe advisory conforming to Kenya PCPB (Pest Control Products Board) regulations "
            "and KALRO Integrated Pest Management (IPM) guidelines. "
            f"Respond clearly in {language}."
        )

        user_content = f"Computer Vision Diagnosis:\n{json.dumps(diagnosis_results, indent=2)}\n\n"
        if farmer_query:
            user_content += f"Farmer's Specific Question:\n\"{farmer_query}\"\n\n"
        user_content += (
            "Please provide:\n"
            "1. Diagnosis Verification & Severity Assessment\n"
            "2. Immediate Action (PCPB registered active ingredients, pre-harvest interval [PHI], dosage tips)\n"
            "3. Cultural & Biological Controls (Push-pull, sanitation, trap crops)\n"
            "4. Economic Injury Level (EIL) & follow-up scouting advice"
        )

        payload = {
            "model": self.model_name,
            "prompt": f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n",
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9
            }
        }

        try:
            res = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            if res.status_code == 200:
                return res.json().get("response", "").strip()
            return f"Ollama Error (HTTP {res.status_code}): {res.text}"
        except Exception as e:
            return f"Failed to connect to Ollama at {self.base_url}: {e}"

    def analyze_image_vlm(
        self,
        image: Union[str, Image.Image],
        prompt: str = "Diagnose the pest or disease in this crop leaf and suggest treatment."
    ) -> str:
        """
        Sends an image to an Ollama Vision Model (e.g. llava, llama3.2-vision).
        """
        if isinstance(image, str):
            with open(image, "rb") as f:
                img_bytes = f.read()
        else:
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            img_bytes = buf.getvalue()

        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False
        }

        try:
            res = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            if res.status_code == 200:
                return res.json().get("response", "").strip()
            return f"Ollama VLM Error (HTTP {res.status_code}): {res.text}"
        except Exception as e:
            return f"Failed to connect to Ollama VLM at {self.base_url}: {e}"
