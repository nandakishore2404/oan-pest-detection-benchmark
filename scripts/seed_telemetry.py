# -*- coding: utf-8 -*-
"""
Seed initial diagnostic events into the telemetry logger.
"""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from benchmark.telemetry import TelemetryLogger

def seed_events():
    logger = TelemetryLogger()
    
    samples = [
        ("bean_rust_val.0.jpg", "Bean Rust", 0.861, 0, [], 3.23, 33.54, 42.1, 4120.0, 20.4, "Kakamega County"),
        ("bean_rust_val.11.jpg", "Bean Rust", 0.663, 1, [{"pest": "Helicoverpa armigera", "confidence": 0.45}], 3.15, 34.10, 41.5, 0.0, 18.4, "Bungoma County"),
        ("angular_leaf_spot_val.1.jpg", "Bean Angular Leaf Spot", 0.772, 0, [], 3.42, 32.80, 44.0, 4350.0, 40.8, "Uasin Gishu County"),
        ("healthy_val.11.jpg", "Healthy Foliage", 0.999, 0, [], 3.18, 31.90, 40.2, 0.0, 34.7, "Trans-Nzoia County"),
        ("potato_late_blight_01.jpg", "Potato Late Blight", 0.942, 0, [], 3.25, 33.10, 45.3, 4210.0, 25.1, "Nyandarua County"),
        ("tomato_early_blight_01.jpg", "Tomato Early Blight", 0.885, 2, [{"pest": "Diaphania indica", "confidence": 0.52}], 3.30, 35.20, 43.8, 4180.0, 22.6, "Kirinyaga County"),
        ("archive_caterpillar_field_01.jpg", "Healthy Foliage", 0.852, 1, [{"pest": "Spodoptera frugiperda", "confidence": 0.78}], 3.19, 33.40, 41.0, 3950.0, 16.5, "Nakuru County"),
        ("archive_grasshopper_field_01.jpg", "Healthy Foliage", 0.741, 1, [{"pest": "Gryllidae", "confidence": 0.69}], 3.28, 34.05, 42.6, 0.0, 14.8, "Machakos County"),
        ("bean_angular_leaf_spot_01.jpg", "Bean Angular Leaf Spot", 0.814, 0, [], 3.22, 33.90, 43.1, 4050.0, 31.2, "Meru County"),
        ("maize_fall_armyworm_01.jpg", "Healthy Foliage", 0.890, 1, [{"pest": "Helicoverpa armigera", "confidence": 0.84}], 3.12, 32.50, 40.9, 4110.0, 19.8, "Kitale / Trans-Nzoia")
    ]
    
    for s in samples:
        logger.log_event(
            image_name=s[0],
            foliar_disease=s[1],
            foliar_conf=s[2],
            pest_count=s[3],
            pests_detected=s[4],
            tier1_latency_ms=s[5],
            tier2_latency_ms=s[6],
            gradcam_latency_ms=s[7],
            ollama_latency_ms=s[8],
            lesion_focus_pct=s[9],
            county=s[10],
            is_synthetic=True
        )
    print(f"Successfully seeded {len(samples)} realistic diagnostic telemetry events.")

if __name__ == "__main__":
    seed_events()
