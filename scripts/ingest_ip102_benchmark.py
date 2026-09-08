# -*- coding: utf-8 -*-
"""
IP102 Agricultural Pest Benchmark Ingestion & Kenyan Class Alignment
====================================================================
Program Manager → AI Enthusiast | Turning Ideas into AI-Powered Solutions: Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>
Repository: https://github.com/nandakishore2404/oan-pest-detection-benchmark

Curates and maps the 102-class CVPR benchmark dataset (Wu et al.) to the specific
economic pest priority list of Kenya smallholder farmers (KALRO / OAN Kenya).
Saves curated splits strictly to Drive D: (D:\OAN_Data\ip102_kenya_curated)
"""

import json
import os
import shutil
import sys
from typing import Dict, List, Tuple

TARGET_DIR = r"D:\OAN_Data\ip102_kenya_curated"

# Explicit mapping of IP102 classes to Kenyan Agro-Ecological Priority Names
IP102_TO_KENYA_MAPPING: Dict[str, str] = {
    # Cereals & Grains (Maize, Sorghum, Wheat)
    "fall armyworm": "fall_armyworm",
    "spodoptera": "fall_armyworm",
    "armyworm": "fall_armyworm",
    "stem borer": "african_stem_borer",
    "asiatic rice borer": "african_stem_borer",
    "chilo": "african_stem_borer",
    "corn borer": "african_stem_borer",
    "bollworm": "african_bollworm",
    "helicoverpa": "african_bollworm",
    "corn leaf aphid": "corn_aphids",
    "aphid": "cereal_aphids",
    
    # Legumes & Vegetables (Beans, Cowpea, Cabbage, Tomato)
    "whitefly": "cassava_whitefly",
    "bemisia": "cassava_whitefly",
    "spider mite": "spider_mites",
    "tetranychus": "spider_mites",
    "thrips": "flower_thrips",
    "diamondback moth": "cabbage_diamondback_moth",
    "flea beetle": "flea_beetle",
    
    # Post-Harvest / Storage
    "weevil": "maize_weevil",
    "grain borer": "larger_grain_borer",
    "sitophilus": "maize_weevil"
}


def curate_ip102_subset(source_ip102_dir: str) -> Dict[str, int]:
    """
    Scans the extracted IP102 directory, filters for the high-priority Kenyan pest
    taxa, re-indexes them into YOLO / classification splits, and writes to Drive D:.
    """
    os.makedirs(TARGET_DIR, exist_ok=True)
    stats: Dict[str, int] = {}

    print("=" * 75)
    print("🌾 Ingesting IP102 Benchmark -> Kenyan Agro-Ecological Taxa")
    print(f"Target Destination: {TARGET_DIR}")
    print("=" * 75)

    if not os.path.exists(source_ip102_dir):
        print(f"Source path '{source_ip102_dir}' not found.")
        print("To download the IP102 dataset:")
        print("  1. Official Github: https://github.com/xpwu95/IP102")
        print("  2. Kaggle Mirror: kaggle datasets download -d rishitdagli/ip102 -p D:\\OAN_Data\\ip102_raw --unzip")
        print("  3. Run this script pointing to the unzipped images folder.")
        return stats

    # Iterate over class folders or subdirectories
    for root, dirs, files in os.walk(source_ip102_dir):
        folder_name = os.path.basename(root).lower().strip()
        
        # Match class against Kenyan target taxonomy
        matched_target_class = None
        for key, target_cls in IP102_TO_KENYA_MAPPING.items():
            if key in folder_name:
                matched_target_class = target_cls
                break

        if not matched_target_class:
            continue

        dest_class_dir = os.path.join(TARGET_DIR, matched_target_class)
        os.makedirs(dest_class_dir, exist_ok=True)

        copied = 0
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(root, f)
                dst_path = os.path.join(dest_class_dir, f"ip102_{matched_target_class}_{f}")
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                copied += 1

        stats[matched_target_class] = stats.get(matched_target_class, 0) + copied
        print(f"✅ Ingested {copied} images for '{matched_target_class}' (from '{folder_name}')")

    # Generate metadata catalog manifest
    manifest_path = os.path.join(TARGET_DIR, "dataset_manifest.json")
    manifest_data = {
        "dataset_name": "IP102-Kenya-Curated-Subset",
        "reference": "Wu et al., CVPR 2019: IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition",
        "target_region": "Kenya & East Africa Smallholder Agrosystems",
        "classes_retained": list(stats.keys()),
        "image_counts": stats,
        "total_images": sum(stats.values()),
        "storage_drive": "D:\\"
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print("=" * 75)
    print(f"Curated {sum(stats.values())} field images across {len(stats)} target pest categories.")
    print(f"Manifest written to: {manifest_path}")
    return stats


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else r"D:\OAN_Data\ip102_raw"
    curate_ip102_subset(src)
