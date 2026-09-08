# -*- coding: utf-8 -*-
"""
Targeted Ingestion & Curation for Simran Volunesia Kaggle Pest Dataset
======================================================================
Lead Architect: Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>
Repository: https://github.com/nandakishore2404/oan-pest-detection-benchmark

Selectively curates high-value classes (aphids, mites, stem_borer, armyworm)
while filtering out non-agricultural noise (mosquito, sawfly) and plain-white studio bias.
Saves curated datasets strictly to Drive D: to preserve Drive C: disk sovereignty.
"""

import os
import shutil
import sys
from typing import Dict, List

# Target destination on Drive D:
TARGET_BASE_DIR = r"D:\OAN_Data\kaggle_pests_curated"

# Allowed agricultural pest classes relevant to East African farming
ACCEPTED_CLASSES: Dict[str, str] = {
    "aphids": "aphid_colonies",
    "mites": "spider_mites",
    "stem_borer": "african_stem_borer",
    "armyworm": "fall_armyworm",
    "bollworm": "african_bollworm",
    "beetle": "maize_flea_beetle"
}

# Explicitly rejected classes (public health pests or temperate forest insects)
REJECTED_CLASSES: List[str] = [
    "mosquito",      # Human vector (malaria), not agricultural
    "sawfly",        # Temperate forest pest, minimal relevance to Kenyan cereals
    "grasshopper"    # Already over-represented in existing 28-class dataset
]


def curate_kaggle_pest_dataset(source_extracted_dir: str) -> Dict[str, int]:
    """
    Scans an extracted Simran Volunesia Kaggle dataset folder, filters out
    non-agricultural classes, and organizes accepted pests into structured
    training partitions on Drive D:.
    """
    os.makedirs(TARGET_BASE_DIR, exist_ok=True)
    summary: Dict[str, int] = {}

    print("=" * 70)
    print("🌾 Selective Curation of Agricultural Pest Dataset (Simran Volunesia)")
    print(f"Target Directory: {TARGET_BASE_DIR}")
    print("=" * 70)

    if not os.path.exists(source_extracted_dir):
        print(f"Source path '{source_extracted_dir}' does not exist.")
        print("To download the raw dataset via Kaggle CLI:")
        print("  kaggle datasets download -d simranvolunesia/pest-dataset -p D:\\OAN_Data\\kaggle_raw --unzip")
        return summary

    for entry in os.listdir(source_extracted_dir):
        full_entry_path = os.path.join(source_extracted_dir, entry)
        if not os.path.isdir(full_entry_path):
            continue

        clean_name = entry.lower().strip()

        # Check rejection list
        if any(rej in clean_name for rej in REJECTED_CLASSES):
            print(f"🚫 Discarding non-target / noisy class: '{entry}'")
            continue

        # Match accepted class
        matched_target_folder = None
        for accepted_key, target_folder in ACCEPTED_CLASSES.items():
            if accepted_key in clean_name:
                matched_target_folder = target_folder
                break

        if not matched_target_folder:
            print(f"⏩ Skipping uncategorized folder: '{entry}'")
            continue

        dest_dir = os.path.join(TARGET_BASE_DIR, matched_target_folder)
        os.makedirs(dest_dir, exist_ok=True)

        # Copy and index valid image files
        copied_count = 0
        for img_name in os.listdir(full_entry_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                src_file = os.path.join(full_entry_path, img_name)
                dst_file = os.path.join(dest_dir, f"curated_{matched_target_folder}_{img_name}")
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                copied_count += 1

        summary[matched_target_folder] = copied_count
        print(f"✅ Curated: '{matched_target_folder}' -> {copied_count} verified field images")

    print("=" * 70)
    print(f"Total curated images: {sum(summary.values())}")
    return summary


if __name__ == "__main__":
    source_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\OAN_Data\kaggle_raw\pest-dataset"
    curate_kaggle_pest_dataset(source_dir)
