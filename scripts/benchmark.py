# -*- coding: utf-8 -*-
"""
OAN Kenya Pest Detection Model Benchmarking Lab - CLI Entrypoint
==============================================================
Evaluates candidate computer vision models on the exact same agricultural image,
measures runtime latency, compares predictions, and generates normalized reports.

Usage:
    python scripts/benchmark.py --image data/golden/maize_fall_armyworm_01.jpg --models all
    python scripts/benchmark.py --image data/golden/bean_angular_leaf_spot_01.jpg --models yolov8s_pest,bioclip_treeoflife
"""

import argparse
from datetime import datetime, timezone
import json
import os
import sys

# Ensure repository root is on Python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmark.runner import BenchmarkRunner, inspect_system_hardware
from models.registry import load_registry, SHORTLISTED_MODEL_IDS

def print_hardware_banner(hw: dict):
    """Renders terminal banner with hardware inspection details."""
    print("=" * 96)
    print("       OAN KENYA PEST DETECTION BENCHMARKING LAB - MILESTONE 1 RUNNER")
    print("=" * 96)
    gpu_str = f"{hw['gpu_device_name']} ({hw['gpu_memory_mb']} MB)" if hw['cuda_available'] else "CPU Only (CUDA unavailable)"
    print(f" OS: {hw['os']:<20} | Python: {hw['python_version']:<10} | CPU Cores: {hw['cpu_count']}")
    print(f" Compute Device: {gpu_str}")
    print("=" * 96)

def format_prediction_table(predictions: list, registry_map: dict):
    """Formats and prints comparative model predictions table."""
    print("\n" + "-" * 96)
    print(f"{'Model ID':<22} | {'Category / Task':<20} | {'Prediction':<24} | {'Conf':<6} | {'Time(ms)':<8} | {'Dev':<7} | {'Abstain'}")
    print("-" * 96)

    for p in predictions:
        mid = p["model_id"][:21]
        reg = registry_map.get(p["model_id"], {})
        cat = reg.get("category", p.get("task", "unknown"))[:19]
        pred_label = p["prediction"][:23]
        conf = f"{p['confidence']:.2f}"
        latency = f"{p['inference_time_ms']:.1f}"
        device = p.get("device", "cpu")[:6]
        abstain = "YES (Unknown)" if p.get("unknown") else "No"

        print(f"{mid:<22} | {cat:<20} | {pred_label:<24} | {conf:<6} | {latency:<8} | {device:<7} | {abstain}")

    print("-" * 96)

def generate_markdown_report(result: dict, report_path: str):
    """Generates markdown summary report for the benchmark run."""
    hw = result["hardware"]
    metrics = result["metrics"]
    gt = result.get("ground_truth") or {}
    preds = result["predictions"]

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# OAN Kenya Pest Detection Benchmark Report\n\n")
        f.write(f"- **Image Tested**: `{result['filename']}`\n")
        f.write(f"- **Image ID**: `{result['image_id']}`\n")
        f.write(f"- **Timestamp**: `{result['timestamp']}`\n")
        f.write(f"- **Execution Hardware**: {hw['os']} | CPU: {hw['cpu_count']} cores | GPU: {hw['gpu_device_name'] or 'CPU'}\n\n")

        if gt:
            f.write("## Ground Truth Target\n\n")
            f.write(f"- **Crop**: {gt.get('crop')}\n")
            f.write(f"- **Target Pest/Disease**: {gt.get('pest') or gt.get('disease')}\n")
            f.write(f"- **Scientific Name**: *{gt.get('scientific_name')}*\n")
            f.write(f"- **Severity**: `{gt.get('severity')}`\n")
            f.write(f"- **Kenya Priority**: `{gt.get('kenya_priority')}`\n\n")

        f.write("## Comparative Results\n\n")
        f.write("| Model ID | Task | Prediction | Confidence | Latency (ms) | Device | Abstention |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for p in preds:
            f.write(f"| `{p['model_id']}` | {p['task']} | **{p['prediction']}** | {p['confidence']:.2f} | {p['inference_time_ms']:.1f} ms | `{p['device']}` | {p['unknown']} |\n")

        f.write("\n## Aggregate Metrics\n\n")
        f.write(f"- **Total Models Evaluated**: {metrics.get('total_models_evaluated')}\n")
        f.write(f"- **Consensus Prediction**: **{metrics.get('consensus_prediction')}** ({metrics.get('consensus_votes')} votes)\n")
        f.write(f"- **Average Latency**: {metrics.get('avg_latency_ms')} ms (Min: {metrics.get('min_latency_ms')} ms, Max: {metrics.get('max_latency_ms')} ms)\n")
        f.write(f"- **Abstention Rate**: {metrics.get('abstention_rate'):.1%}\n")

    print(f"\nMarkdown Report Saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="OAN Kenya Pest Detection Model Benchmarking Lab")
    parser.add_argument(
        "--image",
        type=str,
        default=os.path.join(REPO_ROOT, "data", "golden", "maize_fall_armyworm_01.jpg"),
        help="Path to agricultural image for benchmarking"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model IDs, 'all' (shortlisted models), or 'full' (all registered models)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Confidence threshold below which prediction is marked unknown (abstention)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "normalized"),
        help="Directory to persist normalized prediction JSON files"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        default=True,
        help="Generate and save a summary markdown report"
    )

    args = parser.parse_args()

    # Parse models list
    if args.models.strip() in ("all", "full"):
        model_ids = [args.models.strip()]
    else:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]

    # Load registry metadata for tabular display
    registry_list = load_registry()
    registry_map = {m["model_id"]: m for m in registry_list}

    # Initialize runner
    runner = BenchmarkRunner(
        model_ids=model_ids,
        threshold=args.threshold,
        output_dir=args.output_dir
    )

    print_hardware_banner(runner.hardware_info)

    # Execute benchmark run
    result = runner.run_image(args.image)

    # Print comparative table
    format_prediction_table(result["predictions"], registry_map)

    # Print summary metrics
    m = result["metrics"]
    print("\nBENCHMARK RUN SUMMARY:")
    print(f" - Models Evaluated: {m.get('total_models_evaluated')}")
    print(f" - Consensus Prediction: {m.get('consensus_prediction')} ({m.get('consensus_votes')} models agreeing)")
    print(f" - Latency: Mean {m.get('avg_latency_ms')} ms (Range: {m.get('min_latency_ms')} - {m.get('max_latency_ms')} ms)")
    print(f" - Abstention Rate: {m.get('abstention_rate'):.1%} ({m.get('total_abstentions')} models abstained below {args.threshold:.2f} threshold)")

    gt = result.get("ground_truth")
    if gt:
        target = gt.get("pest") or gt.get("disease") or gt.get("expert_label")
        print(f" - Ground Truth Target: {target} ({gt.get('scientific_name')}) [Priority: {gt.get('kenya_priority')}]")

    # Generate report
    if args.save_report:
        report_name = f"benchmark_report_{result['timestamp']}_{result['image_id']}.md"
        report_path = os.path.join(REPO_ROOT, "results", "reports", report_name)
        generate_markdown_report(result, report_path)

    print(f"\nIndividual JSON Predictions saved to: {args.output_dir}")
    print("=" * 96)

if __name__ == "__main__":
    main()
