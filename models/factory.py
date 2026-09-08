# -*- coding: utf-8 -*-
"""
Model Adapter Factory
=====================
Instantiates standardized BasePestModel adapters for any registered model ID.
Ensures seamless dispatch across YOLO, Timm, BioCLIP, CerealPestAID, Florence-2,
AgriChat, and Frontier architectures.
"""

import os
from typing import Dict, List, Optional
from models.base import BasePestModel
from models.mock_adapter import MockPestModel
from models.yolo_adapter import YOLOAdapter
from models.timm_adapter import TimmAdapter
from models.bioclip_adapter import BioCLIPAdapter
from models.cereal_pestaid_adapter import CerealPestAIDAdapter
from models.florence2_adapter import Florence2Adapter
from models.agri_chat_adapter import AgriChatAdapter
from models.frontier_adapter import FrontierVLMAdapter
from models.registry import load_registry, SHORTLISTED_MODEL_IDS

AFRICA_5CLASS_MAPPING = [
    {"class": "Potato Late Blight", "scientific_name": "Phytophthora infestans"},
    {"class": "Tomato Early Blight", "scientific_name": "Alternaria solani"},
    {"class": "Bean Angular Leaf Spot", "scientific_name": "Pseudocercospora griseola"},
    {"class": "Bean Common Rust", "scientific_name": "Uromyces appendiculatus"},
    {"class": "Healthy Foliage", "scientific_name": None}
]

def get_model_adapter(
    model_id: str,
    threshold: float = 0.40,
    weights_path: Optional[str] = None
) -> BasePestModel:
    """
    Factory constructor returning the appropriate BasePestModel implementation.
    """
    m_id = model_id.lower()

    if "yolo" in m_id:
        return YOLOAdapter(model_id=model_id, weights_path=weights_path, threshold=threshold)

    elif "efficientnet" in m_id:
        return TimmAdapter(
            model_id=model_id,
            architecture="efficientnet_b4",
            weights_path=weights_path,
            threshold=threshold
        )

    elif "mobilenet" in m_id:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_weights = os.path.join(repo_root, "models", "trained", "mobilenetv4_kenya_finetuned.pt")
        actual_weights = weights_path or (default_weights if os.path.exists(default_weights) else None)

        if actual_weights and os.path.exists(actual_weights):
            arch = "mobilenetv4_conv_small"
            classes = AFRICA_5CLASS_MAPPING
        else:
            arch = "mobilenetv4_conv_large"
            classes = None

        return TimmAdapter(
            model_id=model_id,
            architecture=arch,
            weights_path=actual_weights,
            class_mapping=classes,
            threshold=threshold
        )

    elif "ibean" in m_id:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_weights = os.path.join(repo_root, "models", "trained", "mobilenetv4_kenya_finetuned.pt")
        actual_weights = weights_path or (default_weights if os.path.exists(default_weights) else None)
        classes = AFRICA_5CLASS_MAPPING if actual_weights and os.path.exists(actual_weights) else None

        return TimmAdapter(
            model_id=model_id,
            architecture="mobilenetv4_conv_small",
            weights_path=actual_weights,
            class_mapping=classes,
            threshold=threshold
        )

    elif "bioclip" in m_id:
        return BioCLIPAdapter(model_id=model_id, threshold=threshold)

    elif "cereal" in m_id or "pestaid" in m_id:
        return CerealPestAIDAdapter(model_id=model_id, weights_path=weights_path, threshold=threshold)

    elif "florence" in m_id:
        return Florence2Adapter(model_id=model_id, threshold=threshold)

    elif "agrichat" in m_id or "agri_chat" in m_id:
        return AgriChatAdapter(model_id=model_id, threshold=threshold)

    elif "gemini" in m_id:
        return FrontierVLMAdapter(model_id=model_id, provider="google", threshold=threshold)

    elif "gpt" in m_id:
        return FrontierVLMAdapter(model_id=model_id, provider="openai", threshold=threshold)

    elif "mock" in m_id:
        # Only explicitly requested mock testing may return MockPestModel
        return MockPestModel(model_id=model_id, threshold=threshold)

    else:
        raise ValueError(f"Model ID '{model_id}' is not mapped to a real neural adapter. Mock fallback is disabled.")

def get_all_adapters(
    shortlist_only: bool = True,
    threshold: float = 0.40
) -> Dict[str, BasePestModel]:
    """Instantiates adapters for all registered or shortlisted candidate models."""
    models = load_registry()
    adapters = {}
    for m in models:
        mid = m.get("model_id")
        if shortlist_only and mid not in SHORTLISTED_MODEL_IDS:
            continue
        adapters[mid] = get_model_adapter(mid, threshold=threshold)
    return adapters
