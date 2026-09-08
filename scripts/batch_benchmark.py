# -*- coding: utf-8 -*-
"""
Batch Benchmark Runner across Golden Dataset
============================================
Evaluates all candidate models across ALL images in the golden benchmark dataset (data/golden/dataset.csv).
Generates an aggregated accuracy, latency, and consensus matrix.

Usage:
    python scripts/batch_benchmark.py
    python scripts/batch_benchmark.py --models all --threshold 0.40
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmark.runner import BenchmarkRunner, inspect_system_hardware
from models.registry import load_registry, SHORTLISTED_MODEL_IDS

MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "golden", "dataset.csv")
GOLDEN_DIR = os.path.join(REPO_ROOT, "data", "golden")
REPORTS_DIR = os.path.join(REPO_ROOT, "results", "reports")

def run_batch_benchmark(models_arg: str = "all", threshold: float = 0.40):
    runner = BenchmarkRunner(
        model_ids=[models_arg] if models_arg in ("all", "full") else [m.strip() for m in models_arg.split(",")],
        threshold=threshold
    )

    hw = runner.hardware_info
    print("=" * 100)
    print("         OAN KENYA BATCH MODEL BENCHMARKING SUITE - FULL DATASET EVALUATION")
    print("=" * 100)
    gpu_str = f"{hw['gpu_device_name']} ({hw['gpu_memory_mb']} MB)" if hw['cuda_available'] else "CPU Only"
    print(f" OS: {hw['os']:<20} | Python: {hw['python_version']:<10} | Device: {gpu_str}")
    print(f" Candidate Models Tested: {len(runner.adapters)}")
    print("=" * 100)

    # Load dataset manifest
    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: Manifest not found at {MANIFEST_PATH}. Run scripts/setup_samples.py first.")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    print(f"Total Benchmark Images: {len(records)}\n")

    batch_results = []
    model_stats = {mid: {"correct": 0, "total": 0, "latencies": [], "abstained": 0} for mid in runner.adapters.keys()}

    for i, rec in enumerate(records, 1):
        img_filename = rec["filename"]
        img_path = os.path.join(GOLDEN_DIR, img_filename)
        img_id = rec.get("image_id", img_filename)
        gt_target = rec.get("pest") or rec.get("disease") or rec.get("expert_label", "")

        print(f"[{i}/{len(records)}] Evaluating: {img_filename} | Target: {gt_target} ({rec.get('scientific_name', 'N/A')})")

        run_res = runner.run_image(img_path, image_id=img_id)
        batch_results.append(run_res)

        # Track per-model statistics
        for p in run_res["predictions"]:
            mid = p["model_id"]
            if mid in model_stats:
                model_stats[mid]["total"] += 1
                model_stats[mid]["latencies"].append(p["inference_time_ms"])
                if p["unknown"]:
                    model_stats[mid]["abstained"] += 1
                elif gt_target.lower() in p["prediction"].lower() or p["prediction"].lower() in gt_target.lower():
                    model_stats[mid]["correct"] += 1

    # Print Master Summary Table
    print("\n" + "=" * 100)
    print("                              MASTER MODEL PERFORMANCE MATRIX")
    print("=" * 100)
    print(f"{'Model ID':<24} | {'Accuracy':<10} | {'Avg Latency':<14} | {'Min Latency':<12} | {'Abstention Rate'}")
    print("-" * 100)

    for mid, s in model_stats.items():
        acc = f"{(s['correct'] / s['total']):.1%}" if s['total'] > 0 else "N/A"
        avg_lat = f"{sum(s['latencies']) / len(s['latencies']):.1f} ms" if s['latencies'] else "N/A"
        min_lat = f"{min(s['latencies']):.1f} ms" if s['latencies'] else "N/A"
        abstain_rate = f"{(s['abstained'] / s['total']):.1%}" if s['total'] > 0 else "0.0%"
        print(f"{mid:<24} | {acc:<10} | {avg_lat:<14} | {min_lat:<12} | {abstain_rate}")

    print("=" * 100)

    # Save Master Markdown Report
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORTS_DIR, f"master_batch_benchmark_{timestamp}.md")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# OAN Kenya Master Batch Benchmark Report\n\n")
        f.write(f"- **Execution Timestamp**: `{timestamp}`\n")
        f.write(f"- **Hardware**: {hw['os']} | CPU Cores: {hw['cpu_count']} | Device: {gpu_str}\n")
        f.write(f"- **Total Test Images Evaluated**: {len(records)}\n")
        f.write(f"- **Total Model Inferences**: {len(records) * len(runner.adapters)}\n\n")
        f.write("## Overall Performance Summary\n\n")
        f.write("| Model ID | Top-1 Accuracy | Avg Latency (ms) | Min Latency (ms) | Abstention Rate |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for mid, s in model_stats.items():
            acc = f"{(s['correct'] / s['total']):.1%}" if s['total'] > 0 else "N/A"
            avg_lat = f"{sum(s['latencies']) / len(s['latencies']):.1f} ms" if s['latencies'] else "N/A"
            min_lat = f"{min(s['latencies']):.1f} ms" if s['latencies'] else "N/A"
            abstain_rate = f"{(s['abstained'] / s['total']):.1%}" if s['total'] > 0 else "0.0%"
            f.write(f"| `{mid}` | **{acc}** | {avg_lat} | {min_lat} | {abstain_rate} |\n")

        f.write("\n## Image-by-Image Breakdown\n\n")
        for res in batch_results:
            gt = res.get("ground_truth") or {}
            f.write(f"### Image: `{res['filename']}` ({gt.get('crop', 'Crop')})\n")
            f.write(f"- **Target**: **{gt.get('pest') or gt.get('disease') or gt.get('expert_label')}** (*{gt.get('scientific_name')}*)\n")
            f.write(f"- **Severity**: `{gt.get('severity')}` | **Priority**: `{gt.get('kenya_priority')}`\n\n")
            f.write("| Model | Prediction | Confidence | Latency (ms) | Abstained |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for p in res["predictions"]:
                f.write(f"| `{p['model_id']}` | {p['prediction']} | {p['confidence']:.1%} | {p['inference_time_ms']:.1f} ms | {p['unknown']} |\n")
            f.write("\n---\n\n")

    print(f"\nMaster Batch Benchmark Report successfully saved to:\n  {report_path}\n")

def main():
    parser = argparse.ArgumentParser(description="OAN Kenya Batch Benchmark Suite")
    parser.add_argument("--models", type=str, default="all", help="Model IDs or 'all'/'full'")
    parser.add_argument("--threshold", type=float, default=0.40, help="Abstention threshold")
    args = parser.parse_args()
    run_batch_benchmark(models_arg=args.models, threshold=args.threshold)

if __name__ == "__main__":
    main()
