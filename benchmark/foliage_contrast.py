# -*- coding: utf-8 -*-
"""
Foliage Contrast & Insect Cuticle Enhancement Module
====================================================
Program Manager → AI Enthusiast | Turning Ideas into AI-Powered Solutions: Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>
Repository: https://github.com/nandakishore2404/oan-pest-detection-benchmark

Extracts the best color-space filtering techniques from agricultural vision literature:
1. HSV Saturation & Value boost to distinguish translucent insect cuticle from chlorophyll.
2. Lab color space 'a*' channel differential (green-magenta axis) to separate insects from leaves.
3. Contrast Limited Adaptive Histogram Equalization (CLAHE) for shadowed canopy inspection.
"""

from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image


def enhance_foliage_contrast(
    image: Union[str, Image.Image, np.ndarray],
    clahe_clip_limit: float = 2.5,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    cuticle_boost_factor: float = 1.25
) -> np.ndarray:
    """
    Applies Lab-space CLAHE and HSV insect cuticle enhancement to foliage images.
    Returns enhanced RGB image as numpy uint8 array.
    """
    if isinstance(image, str):
        bgr = cv2.imread(image)
        if bgr is None:
            raise ValueError(f"Could not load image from {image}")
    elif isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[2] == 3 else image
    else:
        raise TypeError("Unsupported image type")

    # 1. Convert to LAB color space for luminance equalization without color distortion
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge((l_enhanced, a, b))
    bgr_clahe = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 2. Convert to HSV for cuticle saturation boost
    hsv = cv2.cvtColor(bgr_clahe, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # Multiply saturation on non-pure green zones (where insect chitin / body resides)
    # Green hue typically resides between 35 and 85 in OpenCV 0-180 hue range
    is_not_pure_leaf = (h < 35) | (h > 85)
    s[is_not_pure_leaf] = np.clip(s[is_not_pure_leaf] * cuticle_boost_factor, 0, 255)

    hsv_enhanced = cv2.merge((h, s, v)).astype(np.uint8)
    bgr_final = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    rgb_final = cv2.cvtColor(bgr_final, cv2.COLOR_BGR2RGB)

    return rgb_final


def compute_cuticle_saliency_mask(
    image: Union[str, Image.Image, np.ndarray]
) -> np.ndarray:
    """
    Generates a binary/heatmap mask highlighting regions where non-chlorophyll
    insect features diverge from background crop canopy.
    """
    if isinstance(image, str):
        rgb = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = image

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]

    # Mask out green foliage (Hue 35 to 85, Saturation > 40)
    leaf_mask = (h >= 35) & (h <= 85) & (s >= 40)
    pest_saliency = (~leaf_mask).astype(np.uint8) * 255

    # Apply morphological cleaning to remove tiny single-pixel noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pest_saliency_cleaned = cv2.morphologyEx(pest_saliency, cv2.MORPH_OPEN, kernel)

    return pest_saliency_cleaned
