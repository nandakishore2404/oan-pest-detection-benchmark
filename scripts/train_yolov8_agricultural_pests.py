# -*- coding: utf-8 -*-
"""
Enhanced Agricultural Pest YOLOv8 Fine-Tuning & Benchmarking Pipeline
=====================================================================
Program Manager turned Architect (Powered by AI skills/tools): Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>
Repository: https://github.com/nandakishore2404/oan-pest-detection-benchmark

Integrates Auto-ML hyperparameter profiles from Qwen 2.5 Coder,
stores training runs and checkpoints on Drive D:, and exports
sub-20ms ONNX weights for offline field deployment.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
import json
import shutil
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort

def run_enhanced_yolo_pipeline():
    repo_root = Path(r"c:\Users\nanda\OneDrive\Desktop\DELOITTE\Agri Africa\OAN KENYA\oan-pest-detection-benchmark")
    trained_dir = repo_root / "models" / "trained"
    results_dir = repo_root / "results" / "remediation"
    trained_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Data YAML location on Drive D:
    d_yaml = Path(r"D:\OAN_Data\agricultural_pests_yolo\dataset\data.yaml")
    local_yaml = repo_root / "data" / "agricultural_pests_yolo" / "dataset" / "data.yaml"
    data_yaml = d_yaml if d_yaml.exists() else local_yaml

    # Project output dir on Drive D:
    project_dir = Path(r"D:\OAN_Data\runs\yolov8_agripests_enhanced") if Path("D:\\").exists() else repo_root / "runs" / "agricultural_pests_enhanced"
    project_dir.mkdir(parents=True, exist_ok=True)

    # Ingest Auto-ML config if present
    automl_file = Path(r"D:\OAN_Data\automl_config.json")
    cfg = {
        "epochs": 12,
        "batch": 16,
        "imgsz": 384,
        "optimizer": "AdamW",
        "lr0": 0.002,
        "lrf": 0.01,
        "cos_lr": True,
        "box": 7.5,
        "cls": 0.5,
        "patience": 6,
        "mosaic": 1.0,
        "mixup": 0.1
    }
    if automl_file.exists():
        try:
            with open(automl_file, "r", encoding="utf-8") as f:
                auto_data = json.load(f)
                if "yolov8" in auto_data:
                    cfg.update(auto_data["yolov8"])
                    print("Loaded Auto-ML hyperparameters from Qwen 2.5 Coder:")
                    print(json.dumps(cfg, indent=2))
        except Exception as e:
            print(f"Using calibrated defaults: {e}")

    print("=" * 70)
    print("🌾 OAN KENYA: ENHANCED YOLOV8 AGRICULTURAL PEST FINE-TUNING")
    print("Program Manager turned Architect (Powered by AI skills/tools): Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>")
    print(f"Dataset YAML  : {data_yaml}")
    print(f"Run Directory : {project_dir}")
    print(f"Epochs: {cfg['epochs']}, Optimizer: {cfg['optimizer']}, Cosine LR: {cfg['cos_lr']}")
    print("=" * 70)

    # 1. Load pretrained weights
    print("\n[Step 1/5] Loading Pretrained YOLOv8n Backbone...")
    model = YOLO("yolov8n.pt")

    # 2. Enhanced Fine-Tuning
    print(f"\n[Step 2/5] Training for {cfg['epochs']} Epochs on Real Agricultural Pests...")
    t0_train = time.time()
    train_results = model.train(
        data=str(data_yaml),
        epochs=int(cfg["epochs"]),
        imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]),
        optimizer=str(cfg["optimizer"]),
        lr0=float(cfg["lr0"]),
        lrf=float(cfg["lrf"]),
        cos_lr=bool(cfg["cos_lr"]),
        box=float(cfg["box"]),
        cls=float(cfg["cls"]),
        patience=int(cfg["patience"]),
        mosaic=float(cfg.get("mosaic", 1.0)),
        mixup=float(cfg.get("mixup", 0.0)),
        workers=0,
        device="cpu",
        project=str(project_dir),
        name="train_run",
        exist_ok=True,
        plots=False,
        save=True,
        val=True
    )
    train_duration = time.time() - t0_train
    print(f"Training completed in {train_duration:.2f}s ({train_duration/60:.2f} min)")

    # 3. Model Validation
    print("\n[Step 3/5] Evaluating Fine-Tuned Model on Test/Validation Split...")
    val_metrics = model.val(data=str(data_yaml), imgsz=int(cfg["imgsz"]), batch=int(cfg["batch"]), device="cpu")

    map50 = round(float(val_metrics.box.map50) * 100, 2)
    map50_95 = round(float(val_metrics.box.map) * 100, 2)
    precision = round(float(val_metrics.box.mp) * 100, 2)
    recall = round(float(val_metrics.box.mr) * 100, 2)

    print(f"  Validation Precision : {precision:.2f}%")
    print(f"  Validation Recall    : {recall:.2f}%")
    print(f"  Validation mAP@50    : {map50:.2f}%")
    print(f"  Validation mAP@50-95 : {map50_95:.2f}%")

    # 4. Save PyTorch weights
    best_weights_path = Path(train_results.save_dir) / "weights" / "best.pt"
    dest_pt = trained_dir / "yolov8_agripests_kenya.pt"
    if best_weights_path.exists():
        shutil.copy2(best_weights_path, dest_pt)
        print(f"\n[Step 4/5] Best weights saved to: {dest_pt}")
    else:
        model.save(str(dest_pt))
        print(f"\n[Step 4/5] Model saved to: {dest_pt}")

    # 5. Export Edge-Optimized ONNX
    print("\n[Step 5/5] Exporting Edge-Optimized ONNX Model...")
    onnx_dest = trained_dir / "yolov8_agripests_kenya.onnx"
    export_path = model.export(format="onnx", imgsz=int(cfg["imgsz"]), opset=17, dynamic=False, simplify=True)
    if Path(export_path).exists() and Path(export_path) != onnx_dest:
        shutil.copy2(export_path, onnx_dest)
    print(f"ONNX Model saved to: {onnx_dest}")

    # Benchmark ONNX Latency
    session = ort.InferenceSession(str(onnx_dest), providers=["CPUExecutionProvider"])
    latencies = []
    dummy = np.random.randn(1, 3, int(cfg["imgsz"]), int(cfg["imgsz"])).astype(np.float32)
    input_name = session.get_inputs()[0].name
    for _ in range(30):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_lat = round(float(np.mean(latencies[5:])), 2)
    p95_lat = round(float(np.percentile(latencies[5:], 95)), 2)
    print(f"ONNX Latency (CPU): Avg={avg_lat} ms, P95={p95_lat} ms")

    # Record Results
    results = {
        "model": "YOLOv8n",
        "task": "Agricultural Pest Detection",
        "classes_count": 28,
        "epochs": int(cfg["epochs"]),
        "hyperparameters": cfg,
        "metrics": {
            "precision_pct": precision,
            "recall_pct": recall,
            "map50_pct": map50,
            "map50_95_pct": map50_95
        },
        "latency_ms": {
            "avg": avg_lat,
            "p95": p95_lat
        },
        "weights_pt": str(dest_pt),
        "weights_onnx": str(onnx_dest)
    }

    eval_file = results_dir / "agricultural_pests_finetuned.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation report to: {eval_file}")
    return results

if __name__ == "__main__":
    run_enhanced_yolo_pipeline()
