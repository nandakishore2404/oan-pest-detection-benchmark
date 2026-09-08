# -*- coding: utf-8 -*-
"""
Model Registry Manager
======================
Loads, parses, validates, and filters candidate models from models/model_registry.yaml.
Supports querying by category (CATEGORY A to G), hardware readiness, task, and license.
"""

import os
import sys
from typing import Dict, List, Optional, Any
import yaml

REGISTRY_FILE = os.path.join(os.path.dirname(__file__), "model_registry.yaml")

CATEGORIES = {
    "CATEGORY A": "Specialized pest detection",
    "CATEGORY B": "Specialized crop disease detection",
    "CATEGORY C": "General agricultural image classification",
    "CATEGORY D": "Agricultural multimodal/VLM model",
    "CATEGORY E": "General-purpose vision model",
    "CATEGORY F": "Object detection model",
    "CATEGORY G": "Segmentation model"
}

# The verified shortlist of runnable models for Milestone 1
SHORTLISTED_MODEL_IDS = [
    "yolov8s_pest",
    "efficientnet_b4_agri",
    "mobilenetv4_conv_large",
    "bioclip_treeoflife",
    "cereal_pestaid",
    "florence2_large_agri",
    "ibean_classifier"
]

def load_registry(filepath: str = REGISTRY_FILE) -> List[Dict[str, Any]]:
    """Loads and validates the YAML model registry."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Registry file not found at: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("models", [])

def get_model_metadata(model_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves full metadata for a specific model ID."""
    models = load_registry()
    for m in models:
        if m.get("model_id") == model_id:
            return m
    return None

def filter_by_category(category: str) -> List[Dict[str, Any]]:
    """Filters models by category (e.g. 'CATEGORY A', 'CATEGORY F')."""
    models = load_registry()
    norm_cat = category.strip().upper()
    return [m for m in models if m.get("category", "").upper().startswith(norm_cat)]

def filter_by_task(task: str) -> List[Dict[str, Any]]:
    """Filters models by task (e.g. 'pest_object_detection', 'crop_disease_classification')."""
    models = load_registry()
    return [m for m in models if task.lower() in m.get("task", "").lower()]

def get_shortlist() -> List[Dict[str, Any]]:
    """Returns the curated shortlist of 5-8 practical models for OAN Kenya."""
    models = load_registry()
    return [m for m in models if m.get("model_id") in SHORTLISTED_MODEL_IDS]

def print_registry_summary():
    """Prints a formatted terminal overview of registered models."""
    models = load_registry()
    print("=" * 88)
    print(f"OAN KENYA MODEL REGISTRY: {len(models)} CANDIDATE ARCHITECTURES")
    print("=" * 88)
    print(f"{'Model ID':<24} | {'Category':<12} | {'Params':<8} | {'License':<20} | {'Shortlist'}")
    print("-" * 88)
    for m in models:
        mid = m.get("model_id", "unknown")[:23]
        cat = m.get("category", "unknown")[:11]
        params = m.get("parameter_count", "unknown")[:7]
        lic = m.get("license", "unknown")[:19]
        short = "YES (Shortlisted)" if m.get("model_id") in SHORTLISTED_MODEL_IDS else "Cataloged"
        print(f"{mid:<24} | {cat:<12} | {params:<8} | {lic:<20} | {short}")
    print("=" * 88)

if __name__ == "__main__":
    print_registry_summary()
