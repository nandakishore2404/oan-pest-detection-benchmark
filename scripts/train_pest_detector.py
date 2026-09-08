# -*- coding: utf-8 -*-
"""
OAN Kenya Agricultural Pest Detection Training & Transfer Learning Engine
========================================================================
Fine-tunes Ultralytics YOLOv8 on the 12-class agricultural pest dataset:
Ants, Bees, Beetles, Caterpillars, Earthworms, Earwigs, Grasshoppers,
Moths, Slugs, Snails, Wasps, Weevils.

Features:
- Transfer learning from pre-trained COCO representations (yolov8n / yolov8s)
- Optimized for multi-threaded CPU execution (Intel Core Ultra)
- Automatic export to models/trained/best.pt
- Validation on hold-out 1,095 validation images
- Automatic metric logging (mAP@50, precision, recall, loss)

Usage:
    python scripts/train_pest_detector.py --model yolov8n.pt --epochs 5 --imgsz 416
    python scripts/train_pest_detector.py --model yolov8s.pt --epochs 10 --imgsz 640
"""

import argparse
from datetime import datetime, timezone
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import shutil
import sys
import time

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_DATA_YAML = os.path.join(REPO_ROOT, "data", "pest_data_oan.yaml")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "models", "trained")


def train_pest_detector(
    model_name: str = "yolov8n.pt",
    data_yaml: str = DEFAULT_DATA_YAML,
    epochs: int = 5,
    batch_size: int = 16,
    imgsz: int = 416,
    device: str = "cpu",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    export_onnx: bool = True
):
    print("=" * 70)
    print("🌾 OAN Kenya: YOLOv8 Pest Detection Fine-Tuning Engine")
    print("=" * 70)
    print(f"  Base Model Architecture : {model_name}")
    print(f"  Dataset Configuration   : {data_yaml}")
    print(f"  Target Epochs           : {epochs}")
    print(f"  Batch Size              : {batch_size}")
    print(f"  Input Resolution        : {imgsz}x{imgsz}")
    print(f"  Execution Device        : {device.upper()}")
    print(f"  Target Export Directory : {output_dir}")
    print("=" * 70)

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        print(f"❌ Critical Error: Missing required packages: {e}")
        print("Please run: pip install torch torchvision ultralytics")
        sys.exit(1)

    if device == "cpu":
        # Optimize CPU multi-threading
        cpu_count = os.cpu_count() or 4
        torch.set_num_threads(cpu_count)
        print(f"⚡ CPU Multi-Threading Configured: {cpu_count} threads active.")

    os.makedirs(output_dir, exist_ok=True)

    # Initialize model with pre-trained weights
    print(f"\n📦 Initializing base model weights ({model_name})...")
    model = YOLO(model_name)

    start_time = time.time()

    # Launch fine-tuning
    print(f"\n🚀 Launching domain transfer fine-tuning on 12 agricultural pest classes...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=os.path.join(REPO_ROOT, "runs", "train"),
        name="oan_pest_v8",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=1,
        mosaic=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        plots=True,
        save=True,
        verbose=True
    )

    elapsed_mins = (time.time() - start_time) / 60.0
    print(f"\n✅ Training completed in {elapsed_mins:.2f} minutes.")

    # Locate and copy best weights to models/trained/
    train_save_dir = str(model.trainer.save_dir) if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir") else ""
    best_pt_source = os.path.join(train_save_dir, "weights", "best.pt")
    best_pt_dest = os.path.join(output_dir, "best.pt")

    if os.path.exists(best_pt_source):
        shutil.copy2(best_pt_source, best_pt_dest)
        print(f"📦 Successfully exported primary production weights: {best_pt_dest}")
    else:
        # Fallback: save current model state
        model.save(best_pt_dest)
        print(f"📦 Saved model weights directly to: {best_pt_dest}")

    # Optional ONNX export for edge acceleration
    if export_onnx:
        try:
            print("\n🔄 Exporting edge ONNX model for mobile / low-power deployment...")
            trained_model = YOLO(best_pt_dest)
            onnx_path = trained_model.export(format="onnx", imgsz=imgsz)
            print(f"✅ ONNX model successfully exported: {onnx_path}")
        except Exception as e:
            print(f"⚠️ ONNX export warning (optional): {e}")

    # Save training metadata
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": model_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "device": device,
        "training_duration_minutes": round(elapsed_mins, 2),
        "weights_path": best_pt_dest,
        "classes": [
            "Ants", "Bees", "Beetles", "Caterpillars", "Earthworms",
            "Earwigs", "Grasshoppers", "Moths", "Slugs", "Snails",
            "Wasps", "Weevils"
        ]
    }

    meta_path = os.path.join(output_dir, "training_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"📄 Training metadata written to: {meta_path}")

    print("\n🎉 OAN Kenya Pest Detection Model is now trained and production-ready!")
    return best_pt_dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OAN Kenya Pest Detection Model Trainer")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model (yolov8n.pt or yolov8s.pt)")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=416, help="Image resolution")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device: cpu or cuda")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Target weights directory")
    parser.add_argument("--no-onnx", action="store_true", help="Skip ONNX export")

    args = parser.parse_args()
    train_pest_detector(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        output_dir=args.output_dir,
        export_onnx=not args.no_onnx
    )
