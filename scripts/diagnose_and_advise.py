# -*- coding: utf-8 -*-
r"""
OAN Kenya: Unified Multi-Tier Diagnostic & Local Ollama Advisory Engine
======================================================================
Program Manager → AI Enthusiast | Turning Ideas into AI-Powered Solutions: Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>
Repository: https://github.com/nandakishore2404/oan-pest-detection-benchmark

Executes the complete sovereign pipeline in a single command:
1. Tier 1 Foliar Pathology (MobileNetV4 ONNX)
2. Tier 2 Agricultural Pest Detection (YOLOv8 ONNX)
3. Grad-CAM Neural Attention Heatmap Card (Saved to D:\OAN_Data\)
4. Tier 3 Local PCPB Agronomic Advisory & Swahili Summary via Ollama (0 API Tokens)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import onnxruntime as ort

def run_unified_diagnosis(image_path: str, model_type: str = "qwen2.5-coder:7b"):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
        
    print("=" * 70)
    print("🌾 OPENAGRINET (OAN) KENYA: SOVEREIGN DIAGNOSTIC & ADVISORY ENGINE")
    print(f"Program Manager → AI Enthusiast | Turning Ideas into AI-Powered Solutions: Nanda Kishore Kakulla")
    print(f"Target Image  : {image_path}")
    print("=" * 70)
    
    # 1. Tier 1 Foliar Disease Diagnosis (MobileNetV4 ONNX)
    print("\n[Step 1/4] Running Tier 1 Foliar Pathology Classifier (ONNX)...")
    foliar_onnx = repo_root / "models" / "trained" / "mobilenetv4_kenya_finetuned.onnx"
    classes = ["Potato_Late_Blight", "Tomato_Early_Blight", "Bean_Angular_Leaf_Spot", "Bean_Rust", "Healthy_Foliage"]
    
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = (np.array(img, dtype=np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = np.expand_dims(np.transpose(arr, (2, 0, 1)), 0).astype(np.float32)
    
    t0 = time.perf_counter()
    session = ort.InferenceSession(str(foliar_onnx), providers=["CPUExecutionProvider"])
    logits = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
    lat_foliar = (time.perf_counter() - t0) * 1000.0
    
    exp_l = np.exp(logits - np.max(logits))
    probs = exp_l / np.sum(exp_l)
    pred_idx = int(np.argmax(probs))
    foliar_diag = classes[pred_idx].replace("_", " ")
    foliar_conf = float(probs[pred_idx])
    
    print(f"  Foliar Disease: {foliar_diag} ({foliar_conf*100:.1f}%) | Latency: {lat_foliar:.2f} ms")
    
    # 2. Tier 2 Agricultural Pest Detection (YOLOv8 ONNX / PyTorch)
    print("\n[Step 2/4] Running Tier 2 Agricultural Pest Detector (YOLOv8)...")
    pest_pt = repo_root / "models" / "trained" / "yolov8_agripests_kenya.pt"
    from ultralytics import YOLO
    yolo_model = YOLO(str(pest_pt))
    
    t1 = time.perf_counter()
    pest_results = yolo_model.predict(source=image_path, conf=0.10, imgsz=384, verbose=False)
    lat_pest = (time.perf_counter() - t1) * 1000.0
    
    pests_detected = []
    if len(pest_results) > 0 and pest_results[0].boxes is not None:
        for b in pest_results[0].boxes:
            cid = int(b.cls[0].item())
            cname = pest_results[0].names[cid]
            pconf = float(b.conf[0].item())
            pests_detected.append({"pest": cname, "confidence": round(pconf, 3)})
            
    print(f"  Pests Detected: {len(pests_detected)} items | Latency: {lat_pest:.2f} ms")
    for p in pests_detected[:3]:
        print(f"    - {p['pest']} ({p['confidence']*100:.1f}%)")
        
    # 3. Grad-CAM Visual Explainability Card
    print("\n[Step 3/4] Generating Grad-CAM Saliency Heatmap...")
    from benchmark.explainability import explain_crop_image
    t2 = time.perf_counter()
    xai_res = explain_crop_image(image_path)
    lat_gradcam = (time.perf_counter() - t2) * 1000.0
    
    out_dir = Path(r"D:\OAN_Data") if Path("D:\\").exists() else repo_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    card_path = out_dir / f"xai_card_{Path(image_path).stem}.jpg"
    xai_res["comparison_card"].save(card_path)
    print(f"  Lesion Focus Score: {xai_res['lesion_focus_pct']:.1f}% | Latency: {lat_gradcam:.2f} ms")
    print(f"  Visual Heatmap Card Saved: {card_path}")
    
    # 4. Tier 3 Local PCPB Advisory via Ollama (0 API Tokens)
    print(f"\n[Step 4/4] Consulting Local Ollama ({model_type}) for PCPB Agronomic Advisory (0 TOKENS)...")
    from benchmark.ollama_adapter import OllamaAdvisor
    advisor = OllamaAdvisor(model_name=model_type)
    
    diagnosis_payload = {
        "foliar_disease": foliar_diag,
        "disease_confidence": round(foliar_conf, 3),
        "lesion_focus_pct": round(xai_res['lesion_focus_pct'], 1),
        "pests_detected": pests_detected,
        "county": "Western Kenya Agricultural Hub (Kakamega / Bungoma)"
    }
    
    t3 = time.perf_counter()
    advisory_text = advisor.generate_advisory(diagnosis_payload, language="English with Swahili Summary")
    lat_ollama = (time.perf_counter() - t3) * 1000.0
    
    print("\n" + "=" * 70)
    print(f"📋 LOCAL OLLAMA PCPB AGRONOMIC ADVISORY (ZERO API TOKENS | Latency: {lat_ollama:.2f} ms)")
    print("=" * 70)
    print(advisory_text)
    print("=" * 70)

    # 5. Log Telemetry Event
    from benchmark.telemetry import TelemetryLogger
    logger = TelemetryLogger()
    event = logger.log_event(
        image_name=image_path,
        foliar_disease=foliar_diag,
        foliar_conf=foliar_conf,
        pest_count=len(pests_detected),
        pests_detected=pests_detected,
        tier1_latency_ms=lat_foliar,
        tier2_latency_ms=lat_pest,
        gradcam_latency_ms=lat_gradcam,
        ollama_latency_ms=lat_ollama,
        lesion_focus_pct=xai_res["lesion_focus_pct"]
    )
    print(f"📊 Telemetry Event Logged [Req ID: {event['request_id']}] | Total E2E: {event['latency_ms']['total_e2e']} ms | Tokens Saved: {event['token_economics']['cloud_tokens_saved']}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OAN Kenya Unified Diagnostic CLI")
    parser.add_argument("--image", required=True, help="Path to crop image file")
    parser.add_argument("--model", default="qwen2.5-coder:7b", help="Ollama model name (e.g. qwen2.5-coder:7b or qwen2.5-coder:1.5b)")
    args = parser.parse_args()
    
    run_unified_diagnosis(args.image, args.model)
