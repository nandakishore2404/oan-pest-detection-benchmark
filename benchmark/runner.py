# -*- coding: utf-8 -*-
"""
Benchmark Runner
================
Executes candidate pest detection models on the identical agricultural image,
measures runtime latency, validates outputs against unified schemas, and persists results.
Conforms strictly to OAN Kenya Milestone 1 specification.
"""

from datetime import datetime, timezone
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Union
from PIL import Image

from models.base import NormalizedPrediction, BasePestModel
from models.factory import get_model_adapter, get_all_adapters
from models.registry import load_registry, SHORTLISTED_MODEL_IDS
from benchmark.metrics import load_ground_truth_manifest, compute_benchmark_metrics

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "normalized")
MANIFEST_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "golden", "dataset.csv")

def inspect_system_hardware() -> Dict[str, Any]:
    """Inspects and returns local CPU, GPU, OS, and memory hardware status."""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count() or 1,
        "cuda_available": False,
        "gpu_device_name": None,
        "gpu_memory_mb": 0
    }

    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_mb"] = round(props.total_memory / (1024 * 1024), 1)
    except ImportError:
        pass

    return info

class BenchmarkRunner:
    """
    Orchestrates comparative benchmarking across candidate models.
    """

    def __init__(
        self,
        model_ids: Optional[List[str]] = None,
        threshold: float = 0.40,
        output_dir: str = RESULTS_DIR
    ):
        self.threshold = threshold
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.hardware_info = inspect_system_hardware()

        # Resolve models to benchmark
        if not model_ids or model_ids == ["all"]:
            self.model_ids = SHORTLISTED_MODEL_IDS
        elif model_ids == ["full"]:
            all_regs = load_registry()
            self.model_ids = [m["model_id"] for m in all_regs]
        else:
            self.model_ids = model_ids

        # Instantiate adapters
        self.adapters: Dict[str, BasePestModel] = {}
        for mid in self.model_ids:
            try:
                self.adapters[mid] = get_model_adapter(mid, threshold=self.threshold)
            except Exception as e:
                print(f"Warning: Failed to instantiate adapter for {mid}: {e}")

        # Load ground truth manifest
        self.ground_truth = load_ground_truth_manifest(MANIFEST_PATH)

    def run_image(
        self,
        image_path: str,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs all configured models sequentially on the EXACT same image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        filename = os.path.basename(image_path)
        img_id = image_id or os.path.splitext(filename)[0]

        # Resolve ground truth record if available
        gt_record = self.ground_truth.get(img_id) or self.ground_truth.get(filename)

        run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        predictions: List[NormalizedPrediction] = []

        print(f"\nEvaluating Image: {filename} (ID: {img_id})")
        print(f"Candidate Models: {len(self.adapters)}")
        print("-" * 75)

        for mid, adapter in self.adapters.items():
            t_start = time.time()
            try:
                pred = adapter.predict(image_path, image_id=img_id)
            except Exception as e:
                pred = adapter._build_error_prediction(img_id, str(e))

            predictions.append(pred)

            # Persist individual normalized prediction artifact
            pred_filename = f"{run_timestamp}_{img_id}_{mid}.json"
            pred_path = os.path.join(self.output_dir, pred_filename)
            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump(pred.to_dict(), f, indent=2)

        # Compute aggregate metrics
        metrics = compute_benchmark_metrics(predictions, ground_truth_record=gt_record)

        return {
            "image_path": image_path,
            "image_id": img_id,
            "filename": filename,
            "timestamp": run_timestamp,
            "hardware": self.hardware_info,
            "predictions": [p.to_dict() for p in predictions],
            "metrics": metrics,
            "ground_truth": gt_record
        }
