# -*- coding: utf-8 -*-
"""
Auto-ML Hyperparameter Optimization via Local Ollama (Qwen 2.5 Coder)
=====================================================================
Program Manager turned Architect (Powered by AI skills/tools): Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>
Repository: https://github.com/nandakishore2404/oan-pest-detection-benchmark

Queries the local Qwen 2.5 Coder LLM running on 127.0.0.1:11434 to analyze
dataset constraints and determine the optimal hyperparameter profile for
both the foliar classifier (MobileNetV4) and pest detector (YOLOv8).
Consumes ZERO cloud API tokens.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import json
import urllib.request
import urllib.error
from pathlib import Path

def run_automl():
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(r"D:\OAN_Data") if Path("D:\\").exists() else repo_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_file = out_dir / "automl_config.json"
    
    print("=" * 70)
    print("🤖 OAN KENYA: AUTO-ML HYPERPARAMETER OPTIMIZATION VIA LOCAL OLLAMA")
    print("Program Manager turned Architect (Powered by AI skills/tools): Nanda Kishore Kakulla")
    print("Storage Target: Drive D: (D:\\OAN_Data)")
    print("=" * 70)
    
    prompt = """
You are an expert Auto-ML and Deep Learning optimization engineer specializing in edge computer vision models (MobileNetV4 and YOLOv8) for agricultural field diagnostics.

Context:
1. Hardware: Multi-core CPU (PyTorch 2.14+cpu).
2. Task 1: MobileNetV4-Conv-Small transfer learning & fine-tuning for East African smallholder crops (5 classes: Potato Late Blight, Tomato Early Blight, Bean Angular Leaf Spot, Bean Rust, Healthy). Target dataset has ~200 Makerere iBean African field images + PlantVillage base. We want to unfreeze deep layers (blocks[-2:]) without catastrophic forgetting.
3. Task 2: YOLOv8n fine-tuning for 28 agricultural pest classes (717 field images). Target: detect small insects (aphids, bollworms, armyworms, beetles) efficiently on CPU.

Output ONLY a single valid JSON object (no markdown quotes, no prose before or after) with this exact schema:
{
  "mobilenetv4": {
    "epochs": 8,
    "batch_size": 16,
    "lr_backbone": 0.0001,
    "lr_head": 0.001,
    "weight_decay": 0.0001,
    "cosine_annealing": true,
    "patience": 4,
    "color_jitter": 0.25,
    "rotation_degrees": 20
  },
  "yolov8": {
    "epochs": 12,
    "batch": 16,
    "imgsz": 384,
    "optimizer": "AdamW",
    "lr0": 0.002,
    "lrf": 0.01,
    "cos_lr": true,
    "box": 7.5,
    "cls": 0.5,
    "patience": 5,
    "mosaic": 1.0,
    "mixup": 0.1
  },
  "rationale": "Brief 1-sentence technical explanation of why these hyperparameters optimize CPU convergence and precision."
}
"""

    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": "qwen2.5-coder:7b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 512
        }
    }
    
    print("\nConnecting to local Ollama daemon (127.0.0.1:11434)...")
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_output = data.get("response", "").strip()
            
            # Extract JSON cleanly
            start_idx = raw_output.find("{")
            end_idx = raw_output.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = raw_output[start_idx:end_idx+1]
                config = json.loads(json_str)
            else:
                raise ValueError("Could not parse JSON from Ollama output")
                
            print("Successfully retrieved Auto-ML profile from Qwen 2.5 Coder!")
            print(json.dumps(config, indent=2))
            
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
                
            print(f"\nSaved Auto-ML config to: {config_file}")
            return config
            
    except Exception as e:
        print(f"Warning: Ollama query failed ({e}). Using expert calibrated CPU defaults.")
        fallback = {
            "mobilenetv4": {
                "epochs": 8,
                "batch_size": 16,
                "lr_backbone": 0.0001,
                "lr_head": 0.001,
                "weight_decay": 0.0001,
                "cosine_annealing": True,
                "patience": 4,
                "color_jitter": 0.25,
                "rotation_degrees": 20
            },
            "yolov8": {
                "epochs": 12,
                "batch": 16,
                "imgsz": 384,
                "optimizer": "AdamW",
                "lr0": 0.002,
                "lrf": 0.01,
                "cos_lr": True,
                "box": 7.5,
                "cls": 0.5,
                "patience": 5,
                "mosaic": 1.0,
                "mixup": 0.1
            },
            "rationale": "Calibrated for fast CPU convergence, high small-pest resolution, and domain adaptation without catastrophic forgetting."
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(fallback, f, indent=2)
        print(f"Saved fallback config to: {config_file}")
        return fallback

if __name__ == "__main__":
    run_automl()
