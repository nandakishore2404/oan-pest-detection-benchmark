# -*- coding: utf-8 -*-
"""
Model Quality & Attention Focus Audit (Grad-CAM XAI)
=====================================================
Program Manager → AI Enthusiast | Turning Ideas into AI-Powered Solutions: Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>
Repository: https://github.com/nandakishore2404/oan-pest-detection-benchmark

Audits the fine-tuned MobileNetV4 model against genuine African field images.
Quantifies the Lesion Focus Score (concentration of saliency energy on true pathology)
to eliminate false positives caused by background soil, weeds, or camera glare.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import json
import glob
from pathlib import Path
import numpy as np

def run_quality_audit():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from benchmark.explainability import explain_crop_image

    print("=" * 70)
    print("🔬 OAN KENYA: GRAD-CAM ATTENTION FOCUS & XAI QUALITY AUDIT")
    print("Program Manager → AI Enthusiast | Turning Ideas into AI-Powered Solutions: Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>")
    print("=" * 70)

    val_dir = repo_root / "data" / "african_beans_field" / "validation"
    sample_images = []
    
    # Collect 3 images per category
    for sub in ["bean_rust", "angular_leaf_spot", "healthy"]:
        sub_path = val_dir / sub
        if sub_path.exists():
            imgs = list(sub_path.glob("*.jpg")) + list(sub_path.glob("*.png"))
            sample_images.extend(imgs[:4])

    print(f"Auditing {len(sample_images)} African field validation images across categories...")

    audit_records = []
    focus_scores = []
    out_cards_dir = Path(r"D:\OAN_Data\xai_audit_cards") if Path("D:\\").exists() else repo_root / "results" / "xai_cards"
    out_cards_dir.mkdir(parents=True, exist_ok=True)

    for i, img_p in enumerate(sample_images, 1):
        res = explain_crop_image(str(img_p))
        focus = res["lesion_focus_pct"]
        focus_scores.append(focus)
        
        card_name = f"audit_card_{img_p.parent.name}_{img_p.stem}.jpg"
        card_path = out_cards_dir / card_name
        res["comparison_card"].save(card_path)

        conf_pct = res.get("confidence_pct", round(res.get("confidence", 0.0) * 100.0, 2))
        rec = {
            "image": img_p.name,
            "ground_truth_category": img_p.parent.name,
            "predicted_class": res["predicted_class"],
            "confidence_pct": conf_pct,
            "lesion_focus_pct": focus,
            "card_path": str(card_path)
        }
        audit_records.append(rec)
        print(f"  [{i:2d}/{len(sample_images)}] {img_p.name[:25]:25} | Pred: {res['predicted_class'][:18]:18} ({conf_pct:5.1f}%) | Lesion Focus: {focus:5.1f}%")

    mean_focus = round(float(np.mean(focus_scores)), 2)
    median_focus = round(float(np.median(focus_scores)), 2)
    min_focus = round(float(np.min(focus_scores)), 2)
    max_focus = round(float(np.max(focus_scores)), 2)

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY & ATTENTION CONCENTRATION METRICS")
    print("=" * 70)
    print(f"  Mean Lesion Focus Score   : {mean_focus}%")
    print(f"  Median Lesion Focus Score : {median_focus}%")
    print(f"  Min / Max Focus Score     : {min_focus}% / {max_focus}%")
    print(f"  Target Threshold (>70.0%) : {'PASSED (Optimal Lesion Grounding)' if mean_focus >= 70.0 else 'NEEDS ATTENTION'}")
    print(f"  Audit Heatmap Cards Saved : {out_cards_dir}")
    print("=" * 70)

    audit_summary = {
        "audited_images_count": len(sample_images),
        "mean_lesion_focus_pct": mean_focus,
        "median_lesion_focus_pct": median_focus,
        "min_focus_pct": min_focus,
        "max_focus_pct": max_focus,
        "passed_threshold": bool(mean_focus >= 70.0),
        "records": audit_records
    }

    out_file = repo_root / "results" / "remediation" / "model_quality_audit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"Saved audit report to: {out_file}\n")
    return audit_summary

if __name__ == "__main__":
    run_quality_audit()
