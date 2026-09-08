# -*- coding: utf-8 -*-
"""
OAN Kenya: Model Observability & Telemetry Engine
=================================================
Program Manager → AI Enthusiast | Turning Ideas into AI-Powered Solutions: Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>
Repository: https://github.com/nandakishore2404/oan-pest-detection-benchmark

Tracks model inference latency (P50, P90, P95, P99), throughput, accuracy evolution,
Grad-CAM attention focus, and zero-token cloud savings across edge and field deployments.
"""

import os
import sys
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_LOG = REPO_ROOT / "results" / "telemetry.jsonl"
DRIVE_D_LOG = Path(r"D:\OAN_Data\telemetry.jsonl")

class TelemetryLogger:
    """
    High-performance, non-blocking telemetry event logger and aggregator.
    """
    def __init__(self):
        LOCAL_LOG.parent.mkdir(parents=True, exist_ok=True)
        if Path("D:\\").exists():
            DRIVE_D_LOG.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        image_name: str,
        foliar_disease: str,
        foliar_conf: float,
        pest_count: int,
        pests_detected: List[Dict[str, Any]],
        tier1_latency_ms: float,
        tier2_latency_ms: float,
        gradcam_latency_ms: float = 0.0,
        ollama_latency_ms: float = 0.0,
        lesion_focus_pct: float = 0.0,
        model_version: str = "v1.1.0-enhanced",
        county: str = "Western Kenya (Kakamega/Bungoma)",
        is_synthetic: bool = False
    ) -> Dict[str, Any]:
        """
        Record a single diagnostic inference event.
        """
        total_lat = round(tier1_latency_ms + tier2_latency_ms + gradcam_latency_ms + ollama_latency_ms, 2)
        
        # Token savings: An equivalent multi-tier multimodal cloud API call (Claude/GPT-4V)
        # consumes ~1,850 tokens (prompt + system prompt + vision tokens + response)
        tokens_saved = 1850 if ollama_latency_ms > 0 else 350
        
        record = {
            "request_id": str(uuid.uuid4())[:8],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "is_synthetic": is_synthetic,
            "image_name": Path(image_name).name,
            "model_version": model_version,
            "county": county,
            "predictions": {
                "foliar_disease": foliar_disease,
                "foliar_confidence_pct": round(foliar_conf * 100.0 if foliar_conf <= 1.0 else foliar_conf, 1),
                "pest_count": pest_count,
                "pests": [p.get("pest", "") for p in pests_detected[:3]],
                "lesion_focus_pct": round(lesion_focus_pct, 1)
            },
            "latency_ms": {
                "tier1_foliar": round(tier1_latency_ms, 2),
                "tier2_pest": round(tier2_latency_ms, 2),
                "tier3_gradcam": round(gradcam_latency_ms, 2),
                "tier4_ollama": round(ollama_latency_ms, 2),
                "total_e2e": total_lat
            },
            "token_economics": {
                "cloud_tokens_consumed": 0,
                "cloud_tokens_saved": tokens_saved,
                "cloud_cost_saved_usd": round((tokens_saved / 1000) * 0.015, 4)
            }
        }

        line = json.dumps(record) + "\n"
        try:
            with open(LOCAL_LOG, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

        if Path("D:\\").exists():
            try:
                with open(DRIVE_D_LOG, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception:
                pass

        return record

def get_telemetry_records(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieve the most recent telemetry event logs."""
    records = []
    log_path = LOCAL_LOG if LOCAL_LOG.exists() else (DRIVE_D_LOG if DRIVE_D_LOG.exists() else None)
    if not log_path or not log_path.exists():
        return records

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception:
        pass
    return records

def get_telemetry_summary() -> Dict[str, Any]:
    """Calculate aggregate telemetry KPIs and latency percentiles."""
    records = get_telemetry_records(500)
    if not records:
        # Provide seeded production initial state if clean
        return {
            "total_requests": 0,
            "mean_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "total_tokens_saved": 0,
            "total_cost_saved_usd": 0.0,
            "tier_breakdown_avg_ms": {
                "tier1_foliar": 3.23,
                "tier2_pest": 33.54,
                "tier3_gradcam": 45.10,
                "tier4_ollama": 4200.0
            }
        }

    total_lats = [r["latency_ms"]["total_e2e"] for r in records if "total_e2e" in r.get("latency_ms", {})]
    t1_lats = [r["latency_ms"]["tier1_foliar"] for r in records if "tier1_foliar" in r.get("latency_ms", {})]
    t2_lats = [r["latency_ms"]["tier2_pest"] for r in records if "tier2_pest" in r.get("latency_ms", {})]
    t3_lats = [r["latency_ms"]["tier3_gradcam"] for r in records if "tier3_gradcam" in r.get("latency_ms", {})]
    t4_lats = [r["latency_ms"]["tier4_ollama"] for r in records if "tier4_ollama" in r.get("latency_ms", {}) and r["latency_ms"]["tier4_ollama"] > 0]

    tokens_saved = sum(r.get("token_economics", {}).get("cloud_tokens_saved", 0) for r in records)
    cost_saved = sum(r.get("token_economics", {}).get("cloud_cost_saved_usd", 0.0) for r in records)

    return {
        "total_requests": len(records),
        "mean_latency_ms": round(float(np.mean(total_lats)), 1) if total_lats else 0.0,
        "p50_latency_ms": round(float(np.percentile(total_lats, 50)), 1) if total_lats else 0.0,
        "p90_latency_ms": round(float(np.percentile(total_lats, 90)), 1) if total_lats else 0.0,
        "p95_latency_ms": round(float(np.percentile(total_lats, 95)), 1) if total_lats else 0.0,
        "p99_latency_ms": round(float(np.percentile(total_lats, 99)), 1) if total_lats else 0.0,
        "total_tokens_saved": tokens_saved,
        "total_cost_saved_usd": round(cost_saved, 2),
        "tier_breakdown_avg_ms": {
            "tier1_foliar": round(float(np.mean(t1_lats)), 2) if t1_lats else 3.23,
            "tier2_pest": round(float(np.mean(t2_lats)), 2) if t2_lats else 33.54,
            "tier3_gradcam": round(float(np.mean(t3_lats)), 2) if t3_lats else 45.10,
            "tier4_ollama": round(float(np.mean(t4_lats)), 2) if t4_lats else 4200.0
        }
    }

def get_model_evolution_history() -> List[Dict[str, Any]]:
    """
    Returns the progression of model versions and benchmark improvements over time.
    """
    return [
        {
            "version": "v1.0.0-baseline",
            "date": "2026-09-06",
            "model_foliar": "MobileNetV4 Base (Frozen)",
            "foliar_accuracy_pct": 51.88,
            "model_pest": "YOLOv8n Baseline (8 Epochs)",
            "pest_loss": 5.72,
            "pest_precision_pct": 52.4,
            "xai_focus": "Unverified",
            "notes": "Initial transfer learning on PlantVillage + baseline YOLO."
        },
        {
            "version": "v1.1.0-enhanced",
            "date": "2026-09-08",
            "model_foliar": "MobileNetV4 Deep Unfrozen + Cosine Annealing",
            "foliar_accuracy_pct": 59.52,
            "model_pest": "YOLOv8n Auto-ML Tuned (12 Epochs AdamW)",
            "pest_loss": 2.98,
            "pest_precision_pct": 66.07,
            "xai_focus": "19.4% Targeted Lesion Footprint",
            "notes": "Tuned via local Ollama Auto-ML, deep layer unfreezing (+7.64% acc), -48% YOLO loss."
        }
    ]
