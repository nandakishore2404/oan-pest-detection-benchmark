# -*- coding: utf-8 -*-
"""
Computer Vision Annotation & Bounding Box Visualizer
====================================================
Renders localized bounding boxes, confidence tags, and agronomic diagnostic labels
directly onto agricultural imagery conforming to OAN Kenya specification.
"""

from typing import Any, Dict, List, Optional, Union
from PIL import Image, ImageDraw, ImageFont

# Color palette for bounding box classes
COLOR_PALETTE = {
    "larva": (230, 45, 45),            # Red
    "fall_armyworm": (230, 45, 45),    # Red
    "caterpillar": (230, 45, 45),      # Red
    "borer": (230, 100, 30),           # Dark Orange
    "frass": (215, 170, 35),           # Amber
    "feeding_hole": (190, 140, 40),    # Mustard
    "lesion": (180, 50, 50),           # Crimson
    "spot": (170, 70, 40),             # Rust
    "mildew": (70, 140, 200),          # Cyan/Blue
    "blight": (160, 40, 70),           # Wine
    "healthy": (40, 170, 60),          # Green
    "default": (235, 115, 25)          # Orange
}

def get_class_color(class_name: str) -> tuple:
    """Returns RGB color tuple for a given class name."""
    c_lower = class_name.lower()
    for key, color in COLOR_PALETTE.items():
        if key in c_lower:
            return color
    return COLOR_PALETTE["default"]

def draw_bounding_boxes(
    image: Union[str, Image.Image],
    boxes: List[Dict[str, Any]],
    diagnosis_label: Optional[str] = None,
    confidence: Optional[float] = None
) -> Image.Image:
    """
    Overlays detected bounding boxes and confidence labels onto a copy of the image.
    """
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.copy().convert("RGB")

    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Try loading default font or PIL basic font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for b in boxes:
        bbox = b.get("bbox_xyxy")
        if not bbox or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = bbox
        # Clamp coordinates to image dimensions
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))

        cls_name = b.get("class_name", "Target")
        conf = b.get("confidence", 0.0)
        color = get_class_color(cls_name)

        # Draw outer rectangle outline (width=3)
        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color
            )

        # Label tag text
        conf_str = f"{conf:.0%}" if isinstance(conf, (float, int)) else ""
        label_text = f" {cls_name} {conf_str} "

        # Compute text bounding box
        try:
            bbox_text = draw.textbbox((x1, y1), label_text, font=font)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
        except Exception:
            tw = len(label_text) * 7
            th = 14

        # Draw label background header
        tag_y1 = max(0, y1 - th - 6)
        tag_y2 = max(th + 6, y1)
        tag_x2 = min(width, x1 + tw + 6)
        draw.rectangle([x1, tag_y1, tag_x2, tag_y2], fill=color)
        draw.text((x1 + 3, tag_y1 + 2), label_text, fill=(255, 255, 255), font=font)

    # If diagnosis label provided, draw top diagnostic badge
    if diagnosis_label:
        badge_text = f" [OAN Vision] {diagnosis_label} "
        if confidence:
            badge_text += f"({confidence:.1%}) "
        try:
            b_box = draw.textbbox((10, 10), badge_text, font=font)
            bw = b_box[2] - b_box[0]
            bh = b_box[3] - b_box[1]
        except Exception:
            bw = len(badge_text) * 7
            bh = 14
        draw.rectangle([10, 10, 10 + bw + 8, 10 + bh + 8], fill=(20, 30, 25))
        draw.rectangle([10, 10, 10 + bw + 8, 10 + bh + 8], outline=(60, 180, 80), width=1)
        draw.text((14, 14), badge_text, fill=(100, 240, 120), font=font)

    return img
