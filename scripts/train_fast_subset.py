# -*- coding: utf-8 -*-
"""
Rapid Balanced Fine-Tuning Script for OAN Kenya Pest Detection
=============================================================
Samples a balanced subset across all 12 classes from archive (1),
fine-tunes YOLOv8n on the Intel Core Ultra CPU in ~2-3 minutes,
and saves the production weights to models/trained/best.pt.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import shutil
import sys
import time
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_SRC = r"C:\Users\nanda\OneDrive\Desktop\DELOITTE\Agri Africa\OAN KENYA\SPRINT-3\Pest Control\Pest Data Sets\archive (1)"
FAST_DATA_DIR = os.path.join(REPO_ROOT, "data", "fast_pest_data")

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CLASSES = [
    "Ants", "Bees", "Beetles", "Caterpillars", "Earthworms",
    "Earwigs", "Grasshoppers", "Moths", "Slugs", "Snails",
    "Wasps", "Weevils"
]

def prepare_subset(samples_per_class_train=40, samples_per_class_val=10):
    print("📁 Preparing calibrated balanced subset from archive (1)...")
    train_img_dst = os.path.join(FAST_DATA_DIR, "train", "images")
    train_lbl_dst = os.path.join(FAST_DATA_DIR, "train", "labels")
    val_img_dst = os.path.join(FAST_DATA_DIR, "val", "images")
    val_lbl_dst = os.path.join(FAST_DATA_DIR, "val", "labels")

    for d in [train_img_dst, train_lbl_dst, val_img_dst, val_lbl_dst]:
        os.makedirs(d, exist_ok=True)

    # Helper to sample by class from filename prefix
    src_train_img = os.path.join(DATASET_SRC, "train", "images")
    src_train_lbl = os.path.join(DATASET_SRC, "train", "labels")
    src_val_img = os.path.join(DATASET_SRC, "valid", "images")
    src_val_lbl = os.path.join(DATASET_SRC, "valid", "labels")

    for split, src_img, src_lbl, dst_img, dst_lbl, limit in [
        ("train", src_train_img, src_train_lbl, train_img_dst, train_lbl_dst, samples_per_class_train),
        ("val", src_val_img, src_val_lbl, val_img_dst, val_lbl_dst, samples_per_class_val)
    ]:
        class_counts = {c.lower(): 0 for c in CLASSES}
        all_imgs = os.listdir(src_img)
        for img_name in all_imgs:
            matched_cls = None
            lower_name = img_name.lower()
            for c in CLASSES:
                cl = c.lower()
                # Check match
                if cl in lower_name:
                    matched_cls = cl
                    break
            if matched_cls and class_counts[matched_cls] < limit:
                # Copy image and label
                lbl_name = os.path.splitext(img_name)[0] + ".txt"
                s_img = os.path.join(src_img, img_name)
                s_lbl = os.path.join(src_lbl, lbl_name)
                if os.path.exists(s_img) and os.path.exists(s_lbl):
                    shutil.copy2(s_img, os.path.join(dst_img, img_name))
                    shutil.copy2(s_lbl, os.path.join(dst_lbl, lbl_name))
                    class_counts[matched_cls] += 1

        print(f"  {split.upper()} set sampled: {sum(class_counts.values())} total images across classes: {dict(class_counts)}")

    # Write yaml
    yaml_dict = {
        "path": os.path.abspath(FAST_DATA_DIR).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "nc": 12,
        "names": CLASSES
    }
    yaml_path = os.path.join(FAST_DATA_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f)
    print(f"📄 Fast dataset YAML written to: {yaml_path}")
    return yaml_path

def run_fast_training(yaml_path, epochs=3):
    import torch
    from ultralytics import YOLO

    cpu_count = os.cpu_count() or 4
    torch.set_num_threads(cpu_count)
    print(f"\n🚀 Initializing YOLOv8n transfer learning on {cpu_count} CPU threads for {epochs} epochs...")

    model = YOLO("yolov8n.pt")
    t0 = time.time()
    model.train(
        data=yaml_path,
        epochs=epochs,
        batch=16,
        imgsz=320,
        device="cpu",
        project=os.path.join(REPO_ROOT, "runs", "train"),
        name="fast_oan_pests",
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.002,
        save=True,
        verbose=True
    )
    elapsed = round((time.time() - t0) / 60.0, 2)
    print(f"\n✅ Fine-tuning completed in {elapsed} minutes!")

    # Save to models/trained/best.pt
    save_dir = str(model.trainer.save_dir)
    best_src = os.path.join(save_dir, "weights", "best.pt")
    out_dir = os.path.join(REPO_ROOT, "models", "trained")
    os.makedirs(out_dir, exist_ok=True)
    best_dst = os.path.join(out_dir, "best.pt")

    if os.path.exists(best_src):
        shutil.copy2(best_src, best_dst)
    else:
        model.save(best_dst)
    print(f"📦 Production weights successfully deployed to: {best_dst}")
    return best_dst

if __name__ == "__main__":
    yaml_p = prepare_subset(samples_per_class_train=40, samples_per_class_val=10)
    run_fast_training(yaml_p, epochs=3)
