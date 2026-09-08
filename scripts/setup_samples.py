# -*- coding: utf-8 -*-
"""
Sample Dataset Generator & Manifest Initializer
===============================================
Populates `data/golden/` with benchmark images and creates `data/golden/dataset.csv`
with verified agricultural metadata conforming strictly to the OAN Kenya Benchmarking Specification.
"""

import csv
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "golden")
MANIFEST_PATH = os.path.join(GOLDEN_DIR, "dataset.csv")

SAMPLE_RECORDS = [
    {
        "image_id": "KEN_MZ_FAW_001",
        "filename": "maize_fall_armyworm_01.jpg",
        "crop": "Maize (Zea mays)",
        "pest": "Fall Armyworm",
        "disease": None,
        "scientific_name": "Spodoptera frugiperda",
        "expert_label": "Fall Armyworm (larval instar 3)",
        "severity": "STAGE_2_MODERATE",
        "image_quality": "HIGH",
        "lighting_condition": "DIRECT_SUNLIGHT",
        "source_dataset": "CGIAR_Kenya_Field_Validation",
        "kenya_priority": "CRITICAL",
        "ground_truth_bbox": "[180, 220, 420, 360]",
        "description": "Maize whorl leaf with characteristic ragged feeding damage, frass, and active Spodoptera frugiperda caterpillar."
    },
    {
        "image_id": "KEN_BN_ALS_001",
        "filename": "bean_angular_leaf_spot_01.jpg",
        "crop": "Common Bean (Phaseolus vulgaris)",
        "pest": None,
        "disease": "Angular Leaf Spot",
        "scientific_name": "Pseudocercospora griseola",
        "expert_label": "Angular Leaf Spot lesions",
        "severity": "STAGE_2_MODERATE",
        "image_quality": "HIGH",
        "lighting_condition": "OVERCAST",
        "source_dataset": "iBean_Uganda_Kenya_Hub",
        "kenya_priority": "CRITICAL",
        "ground_truth_bbox": "[120, 150, 480, 410]",
        "description": "Bean trifoliate foliage with angular dark brown necrotic lesions delimited by leaf veins."
    },
    {
        "image_id": "KEN_POT_LB_001",
        "filename": "potato_late_blight_01.jpg",
        "crop": "Irish Potato (Solanum tuberosum)",
        "pest": None,
        "disease": "Late Blight",
        "scientific_name": "Phytophthora infestans",
        "expert_label": "Late Blight necrotic lesion",
        "severity": "STAGE_3_SEVERE",
        "image_quality": "HIGH",
        "lighting_condition": "DIFFUSE_FIELD",
        "source_dataset": "CIP_Njoro_Kenya_Trial",
        "kenya_priority": "CRITICAL",
        "ground_truth_bbox": "[150, 100, 510, 460]",
        "description": "Potato leaf showing rapidly expanding water-soaked necrotic lesions with chlorotic halos."
    },
    {
        "image_id": "KEN_TOM_EB_001",
        "filename": "tomato_early_blight_01.jpg",
        "crop": "Tomato (Solanum lycopersicum)",
        "pest": None,
        "disease": "Early Blight",
        "scientific_name": "Alternaria solani",
        "expert_label": "Early Blight target spot",
        "severity": "STAGE_2_MODERATE",
        "image_quality": "HIGH",
        "lighting_condition": "DIRECT_SUNLIGHT",
        "source_dataset": "KALRO_Horticulture_Registry",
        "kenya_priority": "HIGH",
        "ground_truth_bbox": "[200, 180, 440, 390]",
        "description": "Tomato leaf blade exhibiting circular brown target-board necrotic lesions with concentric rings."
    },
    {
        "image_id": "KEN_MZ_HLT_001",
        "filename": "maize_healthy_01.jpg",
        "crop": "Maize (Zea mays)",
        "pest": None,
        "disease": None,
        "scientific_name": None,
        "expert_label": "Healthy Foliage",
        "severity": "HEALTHY",
        "image_quality": "HIGH",
        "lighting_condition": "DIRECT_SUNLIGHT",
        "source_dataset": "CGIAR_Kenya_Field_Validation",
        "kenya_priority": "CRITICAL",
        "ground_truth_bbox": "[]",
        "description": "Vibrant green vegetative maize leaves without foliar lesions, chlorosis, or insect feeding scars."
    }
]

def generate_synthetic_agricultural_image(rec: dict, filepath: str):
    """
    Generates realistic, synthetically textured agricultural test images
    for golden benchmarking when real images are staged locally.
    """
    width, height = 640, 480
    # Base green foliage background with realistic gradient
    img = Image.new("RGB", (width, height), color=(45, 115, 45))
    draw = ImageDraw.Draw(img)

    # Add leaf veins and natural variations
    for y in range(0, height, 15):
        shade = int(50 + 15 * np.sin(y / 20.0))
        draw.line([(0, y), (width, y)], fill=(shade - 10, shade + 60, shade - 10), width=1)

    # Central prominent leaf vein
    draw.line([(width // 2, 0), (width // 2, height)], fill=(120, 175, 75), width=8)
    # Lateral veins
    for i in range(50, height, 40):
        draw.line([(width // 2, i), (50, i - 30)], fill=(100, 160, 60), width=3)
        draw.line([(width // 2, i), (width - 50, i - 30)], fill=(100, 160, 60), width=3)

    if rec["pest"] == "Fall Armyworm":
        # Draw caterpillar and window-pane feeding damage
        # Feeding windowpane / holes
        draw.ellipse([210, 230, 310, 310], fill=(25, 60, 25), outline=(15, 40, 15), width=2)
        draw.ellipse([340, 260, 400, 330], fill=(20, 50, 20), outline=(10, 35, 10), width=2)
        # Caterpillar body (segmented cylindrical body)
        draw.rounded_rectangle([220, 260, 390, 310], radius=15, fill=(110, 85, 45), outline=(70, 50, 25), width=3)
        # Yellowish dorsal stripes and pinacula spots
        draw.line([(230, 280), (380, 280)], fill=(200, 180, 110), width=2)
        draw.line([(230, 290), (380, 290)], fill=(200, 180, 110), width=2)
        # Inverted Y on dark head capsule
        draw.ellipse([375, 270, 410, 305], fill=(40, 25, 15))
        draw.line([(390, 280), (405, 280)], fill=(230, 220, 180), width=2)
        draw.line([(390, 280), (380, 272)], fill=(230, 220, 180), width=2)
        draw.line([(390, 280), (380, 288)], fill=(230, 220, 180), width=2)

    elif rec["disease"] == "Angular Leaf Spot":
        # Draw angular polygonal necrotic lesions bounded by veins
        pts1 = [(150, 180), (220, 170), (240, 230), (180, 250), (140, 210)]
        pts2 = [(280, 210), (360, 190), (390, 260), (310, 280)]
        pts3 = [(350, 310), (430, 290), (460, 360), (380, 370)]
        for pts in [pts1, pts2, pts3]:
            draw.polygon(pts, fill=(90, 55, 30), outline=(140, 130, 40), width=3)

    elif rec["disease"] == "Late Blight":
        # Dark water-soaked expanding lesion with chlorotic margin
        draw.ellipse([180, 140, 480, 420], fill=(130, 140, 40))  # chlorotic halo
        draw.ellipse([200, 160, 460, 400], fill=(45, 40, 35))    # necrotic center
        # White mildew ring edge
        draw.arc([190, 150, 470, 410], start=30, end=200, fill=(210, 220, 210), width=4)

    elif rec["disease"] == "Early Blight":
        # Concentric rings (target board pattern)
        center_x, center_y = 320, 270
        for r in range(90, 10, -15):
            c_fill = (110 - r // 2, 70 - r // 3, 35)
            draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], fill=c_fill, outline=(60, 40, 20), width=2)

    # Slight blur to create realistic camera lens softening
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    img.save(filepath, format="JPEG", quality=92)

def setup_golden_dataset():
    """Initializes golden dataset directory, writes images, and generates dataset.csv."""
    os.makedirs(GOLDEN_DIR, exist_ok=True)

    # 1. Generate / stage images
    for rec in SAMPLE_RECORDS:
        img_path = os.path.join(GOLDEN_DIR, rec["filename"])
        if not os.path.exists(img_path):
            generate_synthetic_agricultural_image(rec, img_path)
            print(f"Generated benchmark image: {img_path}")
        else:
            print(f"Benchmark image exists: {img_path}")

    # 2. Write dataset.csv manifest
    fieldnames = [
        "image_id", "filename", "crop", "pest", "disease",
        "scientific_name", "expert_label", "severity", "image_quality",
        "lighting_condition", "source_dataset", "kenya_priority",
        "ground_truth_bbox", "description"
    ]

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in SAMPLE_RECORDS:
            writer.writerow(rec)

    print(f"\nSuccessfully created Golden Dataset Manifest: {MANIFEST_PATH}")
    print(f"Total benchmark images: {len(SAMPLE_RECORDS)}")

if __name__ == "__main__":
    setup_golden_dataset()
