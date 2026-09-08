# -*- coding: utf-8 -*-
"""
Slicing-Aided Hyper Inference (SAHI) Engine for Agricultural Pest Detection
===========================================================================
Program Manager turned Architect (Powered by AI skills/tools): Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>
Repository: https://github.com/nandakishore2404/oan-pest-detection-benchmark

Implements slice-based multi-scale inference inspired by:
1. MDPI Electronics 2024 (13(15), 3008) - "Small target detection in agricultural foliage"
2. Ultralytics - "Object Detection for Pest Control & Slicing Aided Hyper Inference"

Eliminates downsampling resolution loss for microscopic pests (aphids, mites, early-instar larvae).
"""

from datetime import datetime, timezone
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torchvision

from models.base import BasePestModel, NormalizedPrediction


class SahiInferenceEngine:
    """
    Slicing-Aided Hyper Inference (SAHI) engine.
    Slices high-resolution field images into overlapping patches, runs YOLO detection
    across all patches + full resized image, remaps bounding boxes to global coordinates,
    and applies Non-Maximum Suppression (NMS) to merge overlapping detections.
    """

    def __init__(
        self,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.20,
        overlap_width_ratio: float = 0.20,
        iou_threshold: float = 0.45,
        confidence_threshold: float = 0.25,
        include_full_image: bool = True
    ):
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.include_full_image = include_full_image

    def compute_slice_boxes(self, img_width: int, img_height: int) -> List[Tuple[int, int, int, int]]:
        """
        Computes sliding window slice bounding boxes [xmin, ymin, xmax, ymax]
        with specified overlap ratios.
        """
        step_x = int(self.slice_width * (1.0 - self.overlap_width_ratio))
        step_y = int(self.slice_height * (1.0 - self.overlap_height_ratio))

        # If image is smaller than slice dimensions, return single slice for entire image
        if img_width <= self.slice_width and img_height <= self.slice_height:
            return [(0, 0, img_width, img_height)]

        x_starts = list(range(0, max(1, img_width - self.slice_width + 1), max(1, step_x)))
        if not x_starts or x_starts[-1] + self.slice_width < img_width:
            x_starts.append(max(0, img_width - self.slice_width))

        y_starts = list(range(0, max(1, img_height - self.slice_height + 1), max(1, step_y)))
        if not y_starts or y_starts[-1] + self.slice_height < img_height:
            y_starts.append(max(0, img_height - self.slice_height))

        # Deduplicate starts
        x_starts = sorted(list(set(x_starts)))
        y_starts = sorted(list(set(y_starts)))

        slices = []
        for ys in y_starts:
            for xs in x_starts:
                xe = min(xs + self.slice_width, img_width)
                ye = min(ys + self.slice_height, img_height)
                slices.append((xs, ys, xe, ye))

        return slices

    def predict_sahi(
        self,
        yolo_model: Any,
        image: Union[str, Image.Image],
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes full SAHI inference pipeline on a given image using a YOLO model.
        Returns aggregated detections, count of slices, and small-target metrics.
        """
        t0 = time.time()
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
            img_id = image_id or os.path.basename(image)
        else:
            pil_img = image.convert("RGB")
            img_id = image_id or f"img_{int(time.time()*1000)}"

        img_w, img_h = pil_img.size
        slices = self.compute_slice_boxes(img_w, img_h)

        all_boxes: List[List[float]] = []
        all_scores: List[float] = []
        all_classes: List[str] = []

        # 1. Full image inference (macro context)
        if self.include_full_image:
            full_results = yolo_model.predict(
                pil_img,
                conf=self.confidence_threshold,
                verbose=False
            )
            if len(full_results) > 0 and full_results[0].boxes is not None:
                for b in full_results[0].boxes:
                    cls_id = int(b.cls[0].item())
                    cls_name = full_results[0].names.get(cls_id, str(cls_id))
                    conf = float(b.conf[0].item())
                    xyxy = [float(x) for x in b.xyxy[0].tolist()]
                    all_boxes.append(xyxy)
                    all_scores.append(conf)
                    all_classes.append(cls_name)

        # 2. Slice-based inference (micro detail)
        for xs, ys, xe, ye in slices:
            crop = pil_img.crop((xs, ys, xe, ye))
            slice_results = yolo_model.predict(
                crop,
                conf=self.confidence_threshold,
                verbose=False
            )
            if len(slice_results) > 0 and slice_results[0].boxes is not None:
                for b in slice_results[0].boxes:
                    cls_id = int(b.cls[0].item())
                    cls_name = slice_results[0].names.get(cls_id, str(cls_id))
                    conf = float(b.conf[0].item())
                    local_xyxy = [float(x) for x in b.xyxy[0].tolist()]
                    # Remap to global coordinates
                    global_xyxy = [
                        local_xyxy[0] + xs,
                        local_xyxy[1] + ys,
                        local_xyxy[2] + xs,
                        local_xyxy[3] + ys
                    ]
                    all_boxes.append(global_xyxy)
                    all_scores.append(conf)
                    all_classes.append(cls_name)

        latency_ms = round((time.time() - t0) * 1000, 2)

        # 3. Class-aware Non-Maximum Suppression (NMS)
        if not all_boxes:
            return {
                "image_id": img_id,
                "slice_count": len(slices),
                "total_raw_boxes": 0,
                "merged_boxes": [],
                "small_target_count": 0,
                "latency_ms": latency_ms,
                "primary_class": "Healthy Foliage (No pests detected)",
                "max_confidence": 0.0
            }

        # Map class names to numeric class indices for class-aware NMS
        unique_classes = list(set(all_classes))
        merged_boxes = []
        small_target_count = 0
        total_pixels = img_w * img_h

        # Run NMS per class to prevent cross-class suppression
        for cls_name in unique_classes:
            cls_indices = [i for i, c in enumerate(all_classes) if c == cls_name]
            cls_boxes = [all_boxes[i] for i in cls_indices]
            cls_scores = [all_scores[i] for i in cls_indices]

            tensor_boxes = torch.tensor(cls_boxes, dtype=torch.float32)
            tensor_scores = torch.tensor(cls_scores, dtype=torch.float32)

            keep_idx = torchvision.ops.nms(tensor_boxes, tensor_scores, self.iou_threshold)
            
            for k in keep_idx.tolist():
                box = [round(float(coord), 1) for coord in cls_boxes[k]]
                conf = round(float(cls_scores[k]), 3)
                
                # Check small-target threshold (< 2% of total image area, typical for aphids/mites)
                box_w = max(0.0, box[2] - box[0])
                box_h = max(0.0, box[3] - box[1])
                box_area_pct = (box_w * box_h / total_pixels) * 100.0
                is_small_target = box_area_pct < 2.0
                if is_small_target:
                    small_target_count += 1

                merged_boxes.append({
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox_xyxy": box,
                    "box_area_pct": round(box_area_pct, 2),
                    "is_small_target": is_small_target
                })

        # Sort merged boxes by confidence descending
        merged_boxes.sort(key=lambda x: x["confidence"], reverse=True)

        primary_class = merged_boxes[0]["class_name"] if merged_boxes else "Healthy Foliage"
        max_conf = merged_boxes[0]["confidence"] if merged_boxes else 0.0

        return {
            "image_id": img_id,
            "slice_count": len(slices),
            "total_raw_boxes": len(all_boxes),
            "merged_boxes": merged_boxes,
            "small_target_count": small_target_count,
            "latency_ms": latency_ms,
            "primary_class": primary_class,
            "max_confidence": max_conf
        }
