"""
Agricultural Pest Object Detection & Field Advisory CLI Tool
Program Manager turned Architect (Powered by AI skills/tools): Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>
Repository: https://github.com/nandakishore2404/oan-pest-detection-benchmark

Performs bounding-box localization, pest counting, species identification,
and automated PCPB Kenya agronomic advisory on input field images.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
from pathlib import Path
from PIL import Image
import torch
from ultralytics import YOLO

# Pest profile knowledgebase for East African crops
PEST_PROFILES = {
    "Agrotis": {
        "common_name": "Black Cutworm (Agrotis ipsilon)",
        "crops": ["Maize", "Common Beans", "Sorghum", "Vegetables"],
        "damage": "Cuts seedlings at soil level at night, causing severe stand loss.",
        "advisory": {
            "immediate_action": "Inspect soil around cut stems at dusk/dawn. Apply Lambda-cyhalothrin or Chlorpyrifos soil drench (PCPB registered).",
            "cultural_control": "Early land preparation 2-3 weeks before planting to starve larvae; hand-pick cutworms from fresh cuts.",
            "threshold": "> 3 cut seedlings per 100 plants warrants chemical intervention."
        }
    },
    "Chilo suppressalis": {
        "common_name": "Stem Borer (Chilo / Busseola complex)",
        "crops": ["Maize", "Sorghum", "Millet", "Rice"],
        "damage": "Windowing/pin-holes on young leaves, deadheart in seedlings, tunneling inside mature stalks.",
        "advisory": {
            "immediate_action": "Apply Emamectin benzoate (5% WDG) or Chlorantraniliprole into the whorl before larvae bore into stalk.",
            "cultural_control": "Implement push-pull technology using Desmodium repellent and Napier grass trap border; slash and burn stubble after harvest.",
            "threshold": "5% infested plants with fresh leaf windowing at V4-V6 stage."
        }
    },
    "Helicoverpa armigera": {
        "common_name": "African Bollworm / Tomato Fruitworm",
        "crops": ["Tomato", "Cotton", "French Beans", "Maize", "Pigeon Peas"],
        "damage": "Direct boring into flowers, pods, and tomato fruits with characteristic round holes.",
        "advisory": {
            "immediate_action": "Spray Indoxacarb (150 SC) or Flubendiamide; rotate with Bacillus thuringiensis (Bt) kurstaki for organic/IPM compliance.",
            "cultural_control": "Pheromone trapping for male moth monitoring; intercrop with marigold trap crop.",
            "threshold": "1 larva per 5 plants or 2 damaged fruits/pods per 10 inspected."
        }
    },
    "Maruca testulalis Geyer": {
        "common_name": "Legume Pod Borer (Maruca vitrata)",
        "crops": ["Common Beans", "Cowpeas", "Pigeon Peas", "Soybeans"],
        "damage": "Webbing together of flower buds and leaves, feeding on floral organs and pods.",
        "advisory": {
            "immediate_action": "Target floral buds with Deltamethrin or Spinetoram before webbing prevents spray penetration.",
            "cultural_control": "Synchronous planting in farming clusters; spray neem seed kernel extract (NSKE 5%) at flower bud initiation.",
            "threshold": "Presence of webbing on > 10% of sampled flower trusses."
        }
    },
    "Sesamia inferens": {
        "common_name": "Pink Stem Borer",
        "crops": ["Maize", "Sugarcane", "Rice", "Sorghum"],
        "damage": "Bores into lower stem nodes, causing lodging and deadheart.",
        "advisory": {
            "immediate_action": "Granular whorl application of Cartap hydrochloride or systemic diamide insecticides.",
            "cultural_control": "Destroy volunteer maize plants and wild sugarcane alternate hosts along field edges.",
            "threshold": "5% plants showing deadheart."
        }
    },
    "Spodoptera exigua": {
        "common_name": "Beet Armyworm / Spodoptera Complex",
        "crops": ["Vegetables", "Legumes", "Onions", "Maize"],
        "damage": "Skeletonizes leaves, feeds gregariously under silk webbing.",
        "advisory": {
            "immediate_action": "Apply Chlorantraniliprole (Coragen) or Spinetoram (Radiant) under evening conditions.",
            "cultural_control": "Deep ploughing to expose pupae to predatory birds; clean weeding.",
            "threshold": "> 5% defoliation during vegetative phase."
        }
    }
}

def detect_pests(image_path: str, model_path: str = None, conf: float = 0.25):
    repo_root = Path(__file__).resolve().parent.parent
    if model_path is None:
        model_path = repo_root / "models" / "trained" / "yolov8_agripests_kenya.pt"
        
    print("=" * 70)
    print("OAN KENYA: FIELD PEST OBJECT DETECTION & AGRONOMIC ADVISORY")
    print(f"Program Manager turned Architect (Powered by AI skills/tools): Nanda Kishore Kakulla")
    print(f"Input Image   : {image_path}")
    print(f"Model Weights : {model_path}")
    print("=" * 70)
    
    model = YOLO(str(model_path))
    results = model.predict(source=image_path, conf=conf, imgsz=384, verbose=False)
    
    res = results[0]
    boxes = res.boxes
    
    print(f"\nDetection Results:")
    print(f"  Total Pests Detected: {len(boxes)}")
    
    if len(boxes) == 0:
        print("  -> No agricultural pests detected above confidence threshold.")
        print("  -> Status: Foliage appears healthy or below economic injury threshold.")
        return []
        
    detections = []
    for i, b in enumerate(boxes):
        cls_id = int(b.cls[0].item())
        cls_name = res.names[cls_id]
        score = float(b.conf[0].item())
        xyxy = [round(float(x), 1) for x in b.xyxy[0].tolist()]
        
        profile = PEST_PROFILES.get(cls_name, {
            "common_name": cls_name,
            "crops": ["Cereal / Legume / Vegetable Crops"],
            "damage": "Foliar and root pest feeding damage.",
            "advisory": {
                "immediate_action": "Scout field intensity; apply standard broad-spectrum PCPB registered pyrethroid if threshold exceeded.",
                "cultural_control": "Maintain good field sanitation, rotate crops.",
                "threshold": "Consult local ward agricultural extension officer."
            }
        })
        
        print(f"\n[Pest #{i+1}] {profile['common_name']}")
        print(f"  Confidence : {score*100:.1f}%")
        print(f"  BoundingBox: [Xmin={xyxy[0]}, Ymin={xyxy[1]}, Xmax={xyxy[2]}, Ymax={xyxy[3]}]")
        print(f"  Host Crops : {', '.join(profile['crops'])}")
        print(f"  Symptom    : {profile['damage']}")
        print(f"  Action Plan: {profile['advisory']['immediate_action']}")
        print(f"  IPM Control: {profile['advisory']['cultural_control']}")
        print(f"  Threshold  : {profile['advisory']['threshold']}")
        
        detections.append({
            "pest_class": cls_name,
            "confidence": score,
            "bbox": xyxy,
            "profile": profile
        })
        
    return detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OAN Kenya Pest Detection CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input crop image")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLOv8 weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    args = parser.parse_args()
    
    detect_pests(args.image, args.model, args.conf)
