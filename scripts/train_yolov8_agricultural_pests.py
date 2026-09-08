"""
Agricultural Pest Detection YOLOv8 Fine-Tuning & Benchmarking Pipeline
Lead Architect: Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>
Repository: https://github.com/nandakishore2404/oan-pest-detection-benchmark

This pipeline fine-tunes YOLOv8n on real crop pests damaging East African
agriculture (stem borers, cutworms, bollworms, pod borers, armyworms)
and exports an edge-optimized ONNX model for mobile/offline deployment.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import json
import shutil
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort

def run_pipeline():
    repo_root = Path(r"c:\Users\nanda\OneDrive\Desktop\DELOITTE\Agri Africa\OAN KENYA\oan-pest-detection-benchmark")
    data_yaml = repo_root / "data" / "agricultural_pests_yolo" / "dataset" / "data.yaml"
    trained_dir = repo_root / "models" / "trained"
    results_dir = repo_root / "results" / "remediation"
    
    trained_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("OAN KENYA: YOLOV8 AGRICULTURAL PEST DETECTOR FINE-TUNING")
    print("Lead Architect: Nanda Kishore Kakulla")
    print(f"Dataset YAML: {data_yaml}")
    print("=" * 70)
    
    # 1. Initialize YOLOv8n pretrained weights
    print("\n[Step 1/5] Loading YOLOv8n Pretrained Backbone...")
    model = YOLO("yolov8n.pt")
    
    # 2. Fine-tune on agricultural pests
    print("\n[Step 2/5] Training for 8 Epochs on Real Agricultural Pests...")
    t0_train = time.time()
    train_results = model.train(
        data=str(data_yaml),
        epochs=8,
        imgsz=384,
        batch=16,
        workers=0,
        device="cpu",
        project=str(repo_root / "runs" / "agricultural_pests"),
        name="yolov8_agripests_train",
        exist_ok=True,
        plots=False,
        save=True,
        val=True
    )
    train_duration = time.time() - t0_train
    print(f"Training completed in {train_duration:.2f}s ({train_duration/60:.2f} min)")
    
    # Locate best weights
    best_pt_path = Path(train_results.save_dir) / "weights" / "best.pt"
    if not best_pt_path.exists():
        best_pt_path = Path(train_results.save_dir) / "weights" / "last.pt"
        
    target_pt = trained_dir / "yolov8_agripests_kenya.pt"
    shutil.copy2(best_pt_path, target_pt)
    print(f"Saved best model weights to: {target_pt} ({target_pt.stat().st_size / (1024*1024):.2f} MB)")
    
    # 3. Comprehensive Validation & Benchmarking
    print("\n[Step 3/5] Evaluating Detection Metrics on Held-out Validation Set...")
    eval_model = YOLO(str(target_pt))
    metrics = eval_model.val(data=str(data_yaml), split="val", imgsz=384, batch=16, workers=0, device="cpu")
    
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    
    print(f"  mAP@50     : {map50:.4f} ({map50*100:.2f}%)")
    print(f"  mAP@50-95  : {map50_95:.4f} ({map50_95*100:.2f}%)")
    print(f"  Precision  : {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall     : {recall:.4f} ({recall*100:.2f}%)")
    
    names = eval_model.names
    class_metrics = {}
    key_targets = ["Agrotis", "Chilo suppressalis", "Helicoverpa armigera", "Maruca testulalis Geyer", "Sesamia inferens", "Spodoptera exigua"]
    
    if hasattr(metrics.box, 'maps') and len(metrics.box.maps) == len(names):
        for idx, name in names.items():
            cls_map50 = float(metrics.box.maps[idx])
            class_metrics[name] = {"class_id": idx, "map50": round(cls_map50, 4)}
            if name in key_targets:
                print(f"    - {name:25s} mAP50: {cls_map50:.4f}")
                
    # 4. ONNX Export & Optimization
    print("\n[Step 4/5] Exporting Edge-Optimized ONNX Model...")
    exported_onnx_path = eval_model.export(format="onnx", imgsz=384, simplify=False)
    target_onnx = trained_dir / "yolov8_agripests_kenya.onnx"
    if Path(exported_onnx_path).resolve() != target_onnx.resolve():
        shutil.copy2(exported_onnx_path, target_onnx)
    print(f"Exported ONNX model to: {target_onnx} ({target_onnx.stat().st_size / (1024*1024):.2f} MB)")
    
    # 5. ONNX CPU Latency Benchmarking
    print("\n[Step 5/5] Benchmarking ONNX Runtime Inference Latency on CPU...")
    ort_session = ort.InferenceSession(str(target_onnx), providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 384, 384).astype(np.float32)
    
    # Warmup
    for _ in range(5):
        _ = ort_session.run(None, {input_name: dummy_input})
        
    latencies = []
    for _ in range(30):
        t_start = time.perf_counter()
        _ = ort_session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t_start) * 1000.0)
        
    mean_lat = float(np.mean(latencies))
    min_lat = float(np.min(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = 1000.0 / mean_lat
    
    print(f"  Latency (Mean): {mean_lat:.2f} ms")
    print(f"  Latency (Min) : {min_lat:.2f} ms")
    print(f"  Latency (P95) : {p95_lat:.2f} ms")
    print(f"  Throughput    : {fps:.1f} FPS (CPU)")
    
    benchmark_report = {
        "metadata": {
            "title": "OAN Kenya Agricultural Pest Detection YOLOv8 Benchmark",
            "lead_architect": "Nanda Kishore Kakulla",
            "repository": "https://github.com/nandakishore2404/oan-pest-detection-benchmark",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": "Intel Core Ultra CPU",
            "num_classes": len(names),
            "classes": names
        },
        "training": {
            "epochs": 8,
            "imgsz": 384,
            "batch_size": 16,
            "duration_seconds": round(train_duration, 2),
            "base_checkpoint": "yolov8n.pt"
        },
        "evaluation_metrics": {
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4)
        },
        "key_kenya_pests_evaluated": {
            "Agrotis_Black_Cutworm": class_metrics.get("Agrotis", {}),
            "Chilo_Stem_Borer": class_metrics.get("Chilo suppressalis", {}),
            "Helicoverpa_Bollworm": class_metrics.get("Helicoverpa armigera", {}),
            "Maruca_Legume_Pod_Borer": class_metrics.get("Maruca testulalis Geyer", {}),
            "Sesamia_Pink_Stem_Borer": class_metrics.get("Sesamia inferens", {}),
            "Spodoptera_Armyworm": class_metrics.get("Spodoptera exigua", {})
        },
        "onnx_deployment": {
            "model_path": "models/trained/yolov8_agripests_kenya.onnx",
            "model_size_mb": round(target_onnx.stat().st_size / (1024*1024), 2),
            "mean_latency_ms": round(mean_lat, 2),
            "min_latency_ms": round(min_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "throughput_fps": round(fps, 1),
            "edge_ready": True
        }
    }
    
    out_json = results_dir / "agricultural_pests_finetuned.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)
    print(f"\nBenchmark JSON saved to: {out_json}")
    print("\nPIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_pipeline()
