# -*- coding: utf-8 -*-
"""
Benchmarking Evaluation Metrics Engine
=====================================
Calculates quantitative diagnostic metrics, cross-model consensus, latency distributions,
and ground-truth agreement conforming to OAN Kenya specification.
"""

import csv
import os
from typing import Any, Dict, List, Optional
from models.base import NormalizedPrediction

def load_ground_truth_manifest(manifest_path: str) -> Dict[str, Dict[str, Any]]:
    """Loads dataset manifest keyed by both filename and image_id."""
    manifest = {}
    if not os.path.exists(manifest_path):
        return manifest

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "image_id" in row:
                manifest[row["image_id"]] = row
            if "filename" in row:
                manifest[row["filename"]] = row
    return manifest

def calculate_agreement_score(pred: str, ground_truth: str) -> float:
    """
    Computes semantic/token similarity between predicted text and expert label.
    Returns 1.0 for exact/substring match, 0.8 for stem/token overlap, 0.0 otherwise.
    """
    if not pred or not ground_truth:
        return 0.0
    p_norm = pred.lower().strip()
    gt_norm = ground_truth.lower().strip()

    if p_norm == gt_norm or gt_norm in p_norm or p_norm in gt_norm:
        return 1.0

    # Stem-level matching (strip trailing 's' / 'es' and typo normalization)
    p_clean = p_norm.replace("tt", "t").rstrip("s")
    gt_clean = gt_norm.replace("tt", "t").rstrip("s")
    if p_clean == gt_clean or gt_clean in p_clean or p_clean in gt_clean:
        return 1.0

    p_tokens = {w.rstrip("s") for w in p_norm.replace("(", " ").replace(")", " ").replace("/", " ").split()}
    gt_tokens = {w.rstrip("s") for w in gt_norm.replace("(", " ").replace(")", " ").replace("/", " ").split()}
    intersection = p_tokens.intersection(gt_tokens)

    # Filter out common stop-words
    meaningful_overlap = [w for w in intersection if w not in {"the", "a", "of", "and", "in", "on", "leaf", "plant", "stage"}]
    if len(meaningful_overlap) > 0:
        return 0.8
    return 0.0

def compute_benchmark_metrics(
    predictions: List[NormalizedPrediction],
    ground_truth_record: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Computes aggregated performance and consensus metrics for a single benchmark run.
    """
    if not predictions:
        return {}

    latencies = [p.inference_time_ms for p in predictions if p.inference_time_ms > 0]
    confidences = [p.confidence for p in predictions if not p.unknown]
    abstentions = [p for p in predictions if p.unknown]

    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    # Cross-model consensus
    pred_counts: Dict[str, int] = {}
    for p in predictions:
        if not p.unknown and p.prediction != "Error":
            norm_name = p.prediction.split(" [")[0].strip()
            pred_counts[norm_name] = pred_counts.get(norm_name, 0) + 1

    consensus_pred = None
    consensus_votes = 0
    if pred_counts:
        sorted_counts = sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)
        consensus_pred = sorted_counts[0][0]
        consensus_votes = sorted_counts[0][1]

    # Ground truth validation if available
    accuracy_eval = None
    if ground_truth_record:
        gt_target = (
            ground_truth_record.get("pest") or 
            ground_truth_record.get("disease") or 
            ground_truth_record.get("expert_label", "")
        )
        matches = []
        for p in predictions:
            score = calculate_agreement_score(p.prediction, gt_target)
            matches.append({
                "model_id": p.model_id,
                "prediction": p.prediction,
                "target": gt_target,
                "score": score
            })
        accuracy_eval = {
            "ground_truth_target": gt_target,
            "scientific_name": ground_truth_record.get("scientific_name"),
            "severity": ground_truth_record.get("severity"),
            "model_matches": matches,
            "accuracy_ratio": round(sum(m["score"] for m in matches) / len(matches), 2) if matches else 0.0
        }

    return {
        "total_models_evaluated": len(predictions),
        "total_abstentions": len(abstentions),
        "abstention_rate": round(len(abstentions) / len(predictions), 3),
        "avg_latency_ms": avg_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "avg_confidence": avg_confidence,
        "consensus_prediction": consensus_pred,
        "consensus_votes": consensus_votes,
        "ground_truth_evaluation": accuracy_eval
    }
