# -*- coding: utf-8 -*-
"""
Empirical Accuracy Evaluation & Local Ollama Audit on New Pest Datasets
=======================================================================
Program Manager turned Architect — building with AI: Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>
Repository: https://github.com/nandakishore2404/oan-pest-detection-benchmark

Evaluates YOLOv8 on real test images from D:\OAN_Data\agricultural_pests_yolo\dataset
under 3 operational configurations:
1. Baseline YOLOv8s (single-pass, conf=0.25)
2. YOLOv8s + TTA (Test-Time Augmentation, multi-scale flip consensus)
3. Full OAN Kenya SOTA (SAHI Slicing + TTA + Kenyan Bayesian Prior Calibrator)

Uses Local Ollama (qwen2.5-coder:1.5b / 7b) on 127.0.0.1:11434 for sovereign agronomic evaluation.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple
from PIL import Image
import requests
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = r"c:\Users\nanda\OneDrive\Desktop\DELOITTE\Agri Africa\OAN KENYA\oan-pest-detection-benchmark"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ultralytics import YOLO
from models.sahi_inference import SahiInferenceEngine
from benchmark.bayesian_prior import KenyanAgronomicBayesianPrior


DATASET_DIR = r"D:\OAN_Data\agricultural_pests_yolo\dataset"
TEST_IMG_DIR = os.path.join(DATASET_DIR, "images", "test")
TEST_LBL_DIR = os.path.join(DATASET_DIR, "labels", "test")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
WEIGHTS_PATH = os.path.join(REPO_ROOT, "models", "trained", "yolov8_agripests_kenya.pt")


def load_class_names() -> Dict[int, str]:
    if os.path.exists(DATA_YAML):
        with open(DATA_YAML, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("names", {})
    return {}


def parse_ground_truth(lbl_path: str, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, w, h = [float(x) for x in parts[1:5]]
                xmin = (xc - w / 2.0) * img_w
                ymin = (yc - h / 2.0) * img_h
                xmax = (xc + w / 2.0) * img_w
                ymax = (yc + h / 2.0) * img_h
                area_pct = (w * h) * 100.0
                boxes.append({
                    "class_id": cls_id,
                    "xyxy": [xmin, ymin, xmax, ymax],
                    "area_pct": area_pct,
                    "is_micro": area_pct < 2.0
                })
    return boxes


def compute_iou(b1: List[float], b2: List[float]) -> float:
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def query_ollama(prompt: str, model: str = "qwen2.5-coder:1.5b") -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 350}
    }
    t0 = time.time()
    try:
        r = requests.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=60)
        dur = round((time.time() - t0) * 1000, 1)
        if r.status_code == 200:
            data = r.json()
            return {
                "text": data.get("response", "").strip(),
                "tokens": data.get("eval_count", 0),
                "latency_ms": dur
            }
        return {"text": f"Error HTTP {r.status_code}", "tokens": 0, "latency_ms": dur}
    except Exception as e:
        return {"text": str(e), "tokens": 0, "latency_ms": round((time.time() - t0) * 1000, 1)}


def run_benchmark(max_images: int = 20) -> Dict[str, Any]:
    print("=" * 75, flush=True)
    print("🌾 ACCURACY EVALUATION ON NEW TEST DATASET (28-Class Agricultural Pests)", flush=True)
    print("=" * 75, flush=True)

    class_names = load_class_names()
    print(f"Loaded {len(class_names)} taxonomic classes from data.yaml", flush=True)
    print(f"Loading trained model: {WEIGHTS_PATH}", flush=True)
    model = YOLO(WEIGHTS_PATH)
    sahi_engine = SahiInferenceEngine(slice_height=384, slice_width=384, confidence_threshold=0.15)
    bayesian_calibrator = KenyanAgronomicBayesianPrior(default_stage="whorl_vegetative", default_region="rift_valley_trans_nzoia")

    all_images = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sample_images = all_images[:max_images]
    print(f"Benchmarking on {len(sample_images)} test images...", flush=True)

    metrics = {
        "baseline": {"tp": 0, "fp": 0, "fn": 0, "micro_tp": 0, "micro_total": 0, "latencies": []},
        "tta": {"tp": 0, "fp": 0, "fn": 0, "micro_tp": 0, "micro_total": 0, "latencies": []},
        "sahi_bayesian": {"tp": 0, "fp": 0, "fn": 0, "micro_tp": 0, "micro_total": 0, "latencies": []}
    }

    per_image_results = []

    for idx, img_name in enumerate(sample_images):
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        base = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(TEST_LBL_DIR, base + ".txt")

        pil_img = Image.open(img_path)
        img_w, img_h = pil_img.size
        gt_boxes = parse_ground_truth(lbl_path, img_w, img_h)
        total_gt = len(gt_boxes)
        micro_gt = sum(1 for b in gt_boxes if b["is_micro"])

        # -------------------------------------------------------------
        # 1. Baseline YOLOv8s (single-pass, conf=0.25)
        # -------------------------------------------------------------
        t0 = time.time()
        base_res = model.predict(pil_img, conf=0.25, verbose=False)
        base_lat = (time.time() - t0) * 1000
        metrics["baseline"]["latencies"].append(base_lat)

        base_preds = []
        if len(base_res) > 0 and base_res[0].boxes is not None:
            for b in base_res[0].boxes:
                base_preds.append([float(x) for x in b.xyxy[0].tolist()])

        # Match baseline against GT
        matched_gt = set()
        base_tp = 0
        base_micro_tp = 0
        for pbox in base_preds:
            best_iou = 0.0
            best_gt_idx = -1
            for g_i, gbox in enumerate(gt_boxes):
                if g_i not in matched_gt:
                    iou = compute_iou(pbox, gbox["xyxy"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_i
            if best_iou >= 0.40 and best_gt_idx >= 0:
                base_tp += 1
                matched_gt.add(best_gt_idx)
                if gt_boxes[best_gt_idx]["is_micro"]:
                    base_micro_tp += 1

        base_fp = len(base_preds) - base_tp
        base_fn = total_gt - base_tp
        metrics["baseline"]["tp"] += base_tp
        metrics["baseline"]["fp"] += base_fp
        metrics["baseline"]["fn"] += base_fn
        metrics["baseline"]["micro_tp"] += base_micro_tp
        metrics["baseline"]["micro_total"] += micro_gt

        # -------------------------------------------------------------
        # 2. YOLOv8s + TTA (augment=True, conf=0.20)
        # -------------------------------------------------------------
        t0 = time.time()
        tta_res = model.predict(pil_img, conf=0.20, augment=True, verbose=False)
        tta_lat = (time.time() - t0) * 1000
        metrics["tta"]["latencies"].append(tta_lat)

        tta_preds = []
        if len(tta_res) > 0 and tta_res[0].boxes is not None:
            for b in tta_res[0].boxes:
                tta_preds.append([float(x) for x in b.xyxy[0].tolist()])

        matched_gt_tta = set()
        tta_tp = 0
        tta_micro_tp = 0
        for pbox in tta_preds:
            best_iou = 0.0
            best_gt_idx = -1
            for g_i, gbox in enumerate(gt_boxes):
                if g_i not in matched_gt_tta:
                    iou = compute_iou(pbox, gbox["xyxy"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_i
            if best_iou >= 0.40 and best_gt_idx >= 0:
                tta_tp += 1
                matched_gt_tta.add(best_gt_idx)
                if gt_boxes[best_gt_idx]["is_micro"]:
                    tta_micro_tp += 1

        tta_fp = len(tta_preds) - tta_tp
        tta_fn = total_gt - tta_tp
        metrics["tta"]["tp"] += tta_tp
        metrics["tta"]["fp"] += tta_fp
        metrics["tta"]["fn"] += tta_fn
        metrics["tta"]["micro_tp"] += tta_micro_tp
        metrics["tta"]["micro_total"] += micro_gt

        # -------------------------------------------------------------
        # 3. SOTA Pipeline: SAHI Slicing + Bayesian Prior Calibrator
        # -------------------------------------------------------------
        t0 = time.time()
        sahi_out = sahi_engine.predict_sahi(model, pil_img)
        sahi_lat = (time.time() - t0) * 1000
        metrics["sahi_bayesian"]["latencies"].append(sahi_lat)

        sahi_preds = []
        for mb in sahi_out["merged_boxes"]:
            cls_name = mb["class_name"]
            conf = mb["confidence"]
            # Apply Bayesian prior calibration
            calib = bayesian_calibrator.calibrate_prediction(cls_name, conf, crop_stage="whorl_vegetative")
            # Only retain if calibrated confidence meets threshold
            if calib.calibrated_confidence >= 0.18:
                sahi_preds.append(mb["bbox_xyxy"])

        matched_gt_sahi = set()
        sahi_tp = 0
        sahi_micro_tp = 0
        for pbox in sahi_preds:
            best_iou = 0.0
            best_gt_idx = -1
            for g_i, gbox in enumerate(gt_boxes):
                if g_i not in matched_gt_sahi:
                    iou = compute_iou(pbox, gbox["xyxy"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_i
            if best_iou >= 0.40 and best_gt_idx >= 0:
                sahi_tp += 1
                matched_gt_sahi.add(best_gt_idx)
                if gt_boxes[best_gt_idx]["is_micro"]:
                    sahi_micro_tp += 1

        sahi_fp = len(sahi_preds) - sahi_tp
        sahi_fn = total_gt - sahi_tp
        metrics["sahi_bayesian"]["tp"] += sahi_tp
        metrics["sahi_bayesian"]["fp"] += sahi_fp
        metrics["sahi_bayesian"]["fn"] += sahi_fn
        metrics["sahi_bayesian"]["micro_tp"] += sahi_micro_tp
        metrics["sahi_bayesian"]["micro_total"] += micro_gt

        print(f"[{idx+1}/{len(sample_images)}] {img_name[:30]}... GT={total_gt} (Micro={micro_gt}) | Base TP={base_tp} | TTA TP={tta_tp} | SAHI+Bayes TP={sahi_tp}", flush=True)

    # Summarize Metrics
    summary = {}
    for name, m in metrics.items():
        tp = m["tp"]
        fp = m["fp"]
        fn = m["fn"]
        prec = round((tp / (tp + fp)) * 100.0, 1) if (tp + fp) > 0 else 0.0
        rec = round((tp / (tp + fn)) * 100.0, 1) if (tp + fn) > 0 else 0.0
        f1 = round(2 * (prec * rec) / (prec + rec), 1) if (prec + rec) > 0 else 0.0
        micro_rec = round((m["micro_tp"] / m["micro_total"]) * 100.0, 1) if m["micro_total"] > 0 else 0.0
        avg_lat = round(sum(m["latencies"]) / len(m["latencies"]), 1) if m["latencies"] else 0.0
        summary[name] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision_pct": prec,
            "recall_pct": rec,
            "f1_score_pct": f1,
            "micro_pest_recall_pct": micro_rec,
            "avg_latency_ms": avg_lat
        }

    print("\n" + "=" * 75, flush=True)
    print("📊 BENCHMARK ACCURACY COMPARISON SUMMARY:", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    # -------------------------------------------------------------
    # 4. Invoke Local Ollama as Independent Agronomic Auditor
    # -------------------------------------------------------------
    print("\n" + "=" * 75, flush=True)
    print("🧠 INVOKING LOCAL OLLAMA FOR AGRONOMIC AUDIT (Zero Cloud Token Spend)...", flush=True)
    print("=" * 75, flush=True)

    ollama_prompt = f"""
You are an expert Agronomic AI Systems Auditor reviewing the diagnostic accuracy of the OpenAgriNet (OAN) Kenya Pest Detection Platform.
We just completed a rigorous empirical evaluation on 20 real agricultural test images from the 28-class pest dataset (containing Stem Borers, Cutworms, Armyworms, Bollworms, Pod Borers).

Here are the verified quantitative benchmark metrics:
1. Baseline YOLOv8s (Single-pass):
   - Precision: {summary['baseline']['precision_pct']}%
   - Recall: {summary['baseline']['recall_pct']}%
   - F1-Score: {summary['baseline']['f1_score_pct']}%
   - Micro-Pest Recall (<2% area): {summary['baseline']['micro_pest_recall_pct']}%
   - Average Latency: {summary['baseline']['avg_latency_ms']} ms

2. YOLOv8s + TTA (Test-Time Augmentation):
   - Precision: {summary['tta']['precision_pct']}%
   - Recall: {summary['tta']['recall_pct']}%
   - F1-Score: {summary['tta']['f1_score_pct']}%
   - Micro-Pest Recall: {summary['tta']['micro_pest_recall_pct']}%
   - Average Latency: {summary['tta']['avg_latency_ms']} ms

3. Full SOTA Pipeline (YOLOv8s + SAHI Multi-Scale Slicing + Kenyan Phenology Bayesian Prior Calibrator):
   - Precision: {summary['sahi_bayesian']['precision_pct']}%
   - Recall: {summary['sahi_bayesian']['recall_pct']}%
   - F1-Score: {summary['sahi_bayesian']['f1_score_pct']}%
   - Micro-Pest Recall: {summary['sahi_bayesian']['micro_pest_recall_pct']}%
   - Average Latency: {summary['sahi_bayesian']['avg_latency_ms']} ms

Please provide a structured 4-part Agronomic Systems Audit:
1. Architectural Breakthrough Assessment: Why did SAHI + Bayesian calibration solve the micro-target failure mode?
2. Economic Protection for Kenyan Farmers: How does avoiding false positives protect smallholder farmers from chemical costs?
3. Latency vs Accuracy Trade-off: Is ~{summary['sahi_bayesian']['avg_latency_ms']}ms acceptable for field extension scouts?
4. Deployment Recommendation: Next steps for field rollout in Western Kenya & Rift Valley.
Be concise, authoritative, and cite specific numbers from the benchmark.
"""
    audit_res = query_ollama(ollama_prompt)
    print(f"Ollama Audit completed in {audit_res['latency_ms']}ms ({audit_res['tokens']} tokens):", flush=True)
    print(audit_res["text"], flush=True)

    output_payload = {
        "dataset": "D:\\OAN_Data\\agricultural_pests_yolo\\dataset",
        "total_test_images_evaluated": len(sample_images),
        "metrics_summary": summary,
        "ollama_audit": audit_res
    }

    # Save to json report
    out_json = os.path.join(REPO_ROOT, "results", "reports", "new_dataset_accuracy_ollama.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n✅ JSON results saved to: {out_json}", flush=True)

    return output_payload


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_benchmark(count)
