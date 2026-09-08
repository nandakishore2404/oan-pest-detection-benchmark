# -*- coding: utf-8 -*-
"""
Single-Image & Batch Inference CLI for OAN Kenya
================================================
Usage:
    python scripts/predict.py --image path/to/leaf.jpg
    python scripts/predict.py --image path/to/leaf.jpg --model models/trained/mobilenetv4_kenya_finetuned.onnx
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import argparse
import time
from PIL import Image
import numpy as np
import onnxruntime as ort

CLASSES = [
    "Potato_Late_Blight",
    "Tomato_Early_Blight",
    "Bean_Angular_Leaf_Spot",
    "Bean_Rust",
    "Healthy_Foliage"
]

ADVISORIES = {
    "Potato_Late_Blight": "⚠️ HIGH RISK: Potato Late Blight (Phytophthora infestans) detected. Immediately apply PCPB-approved contact fungicide (Mancozeb 80% WP) or systemic fungicide (Metalaxyl-M). Ensure proper drainage and rogue infected plants.",
    "Tomato_Early_Blight": "⚠️ WARNING: Tomato Early Blight (Alternaria solani) detected. Apply PCPB-registered copper-based fungicide or Difenoconazole. Prune lower diseased foliage and avoid overhead irrigation.",
    "Bean_Angular_Leaf_Spot": "⚠️ WARNING: Bean Angular Leaf Spot (Pseudocercospora griseola) detected. Treat with approved Triazole/Strobilurin fungicide spray. Use certified disease-free seeds next season and practice 2-year crop rotation.",
    "Bean_Rust": "⚠️ WARNING: Bean Rust (Uromyces appendiculatus) detected. Apply registered sulfur or azoxystrobin spray at early pustule appearance. Remove infected crop residues post-harvest.",
    "Healthy_Foliage": "✅ NORMAL: Foliage appears healthy with no actionable pathogens detected. Maintain routine scouting and standard cultural nutrition practices."
}

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0).astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="OAN Kenya Foliar Disease Diagnostic CLI")
    parser.add_argument("--image", required=True, help="Path to leaf image file")
    parser.add_argument("--model", default="models/trained/mobilenetv4_kenya_finetuned.onnx", help="Path to ONNX model")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    
    t0 = time.perf_counter()
    tensor = preprocess_image(args.image)
    t_prep = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    logits = session.run(None, {inp_name: tensor})[0][0]
    t_inf = (time.perf_counter() - t1) * 1000

    # Softmax
    exp_l = np.exp(logits - np.max(logits))
    probs = exp_l / np.sum(exp_l)
    
    pred_idx = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    print("\n" + "=" * 60)
    print("🌾 OPENAGRINET (OAN) KENYA — DIAGNOSTIC INFERENCE RESULT")
    print("=" * 60)
    print(f"Image File   : {os.path.basename(args.image)}")
    print(f"Diagnosed ID : {pred_class.replace('_', ' ')}")
    print(f"Confidence   : {confidence * 100:.2f}%")
    print(f"Preprocess   : {t_prep:.2f} ms")
    print(f"Inference    : {t_inf:.2f} ms (ONNX CPU)")
    print("-" * 60)
    print("PROBABILITY DISTRIBUTION:")
    for c, p in zip(CLASSES, probs):
        bar = "#" * int(p * 25)
        print(f"  {c.replace('_', ' '):26} : {p * 100:5.2f}% {bar}")
    print("-" * 60)
    print("AGRONOMIC ADVISORY (PCPB KENYA):")
    print(f"  {ADVISORIES.get(pred_class, 'Consult local extension officer.')}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
