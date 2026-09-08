# -*- coding: utf-8 -*-
"""
OAN Kenya: Explainable AI (XAI) Grad-CAM Module
================================================
Lead Architect: Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>
Repository: https://github.com/nandakishore2404/oan-pest-detection-benchmark

Generates Class Activation Maps (Grad-CAM) showing exact foliar lesion attention,
providing agricultural extension officers and PCPB regulators with visual proof
of neural network diagnostic reasoning.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import timm
from torchvision import transforms

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for PyTorch vision models.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
            
        score = logits[0, target_class]
        score.backward(retain_graph=True)
        
        # Pool gradients across spatial dimensions [B, C, H, W] -> [B, C, 1, 1]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # ReLU to isolate positive contributors to the class
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize between 0 and 1
        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

def overlay_cam_on_image(img_pil: Image.Image, cam: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """
    Overlays normalized CAM heatmap onto original PIL Image using Jet colormap.
    """
    img_rgb = np.array(img_pil.convert("RGB"))
    h, w = img_rgb.shape[:2]
    
    # Resize CAM to image resolution
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_uint8 = np.uint8(255 * np.clip(cam_resized, 0, 1))
    
    # Apply colormap (JET: Red = High Attention, Blue = Low Attention)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    # Blend image and heatmap
    fused = cv2.addWeighted(img_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return Image.fromarray(fused)

def create_explainability_card(
    img_pil: Image.Image,
    cam_overlay: Image.Image,
    pred_class: str,
    confidence: float,
    lesion_focus_pct: float
) -> Image.Image:
    """
    Creates a side-by-side visual comparison card for agronomists and extension officers.
    """
    w, h = 384, 384
    img_resized = img_pil.resize((w, h))
    overlay_resized = cam_overlay.resize((w, h))
    
    banner_h = 70
    canvas = Image.new("RGB", (w * 2 + 30, h + banner_h + 20), color=(255, 255, 255))
    
    canvas.paste(img_resized, (10, banner_h + 10))
    canvas.paste(overlay_resized, (w + 20, banner_h + 10))
    
    canvas_np = np.array(canvas)
    cv2.putText(canvas_np, "OAN Kenya AI Pathology Attention (Grad-CAM XAI)", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 80, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas_np, f"Diagnosis: {pred_class} ({confidence*100:.1f}%) | Lesion Focus: {lesion_focus_pct:.1f}%", (15, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 40, 40), 1, cv2.LINE_AA)
    
    cv2.putText(canvas_np, "Original Field Photo", (15, banner_h + h + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas_np, "Neural Attention Map (Red = Pathology)", (w + 25, banner_h + h + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 20, 20), 1, cv2.LINE_AA)
    
    return Image.fromarray(canvas_np)

def explain_crop_image(
    image_path: str,
    weights_path: str = None,
    architecture: str = "mobilenetv4_conv_small.e2400_r224_in1k",
    class_names: list = None
) -> dict:
    """
    End-to-end function to compute Grad-CAM on a leaf image and return XAI artifacts.
    """
    if class_names is None:
        class_names = [
            "Potato_Late_Blight",
            "Tomato_Early_Blight",
            "Bean_Angular_Leaf_Spot",
            "Bean_Rust",
            "Healthy_Foliage"
        ]
        
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if weights_path is None:
        weights_path = os.path.join(repo_root, "models", "trained", "mobilenetv4_kenya_finetuned.pt")
        
    # Load model
    model = timm.create_model(architecture, pretrained=False, num_classes=len(class_names))
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        elif isinstance(state, dict):
            model.load_state_dict(state)
            
    model.eval()
    
    # Target layer: last residual block with 7x7 spatial resolution
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        target_layer = model.blocks[-1]
    elif hasattr(model, "conv_head"):
        target_layer = model.conv_head
    else:
        target_layer = list(model.children())[-2]
        
    grad_cam = GradCAM(model, target_layer)
    
    # Preprocess image
    img_pil = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_pil).unsqueeze(0)
    
    # Forward pass and prediction
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1).squeeze().tolist()
    pred_id = int(np.argmax(probs))
    pred_class = class_names[pred_id]
    confidence = probs[pred_id]
    
    # Generate CAM
    cam = grad_cam.generate(input_tensor, target_class=pred_id)
    grad_cam.remove_hooks()
    
    # Calculate lesion focus (% of heatmap above 50% threshold)
    high_activation_mask = cam > 0.40
    focus_pct = float(np.sum(high_activation_mask) / cam.size) * 100.0
    
    overlay_img = overlay_cam_on_image(img_pil, cam)
    comparison_card = create_explainability_card(img_pil, overlay_img, pred_class, confidence, focus_pct)
    
    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "confidence_pct": round(confidence * 100.0, 2),
        "lesion_focus_pct": round(focus_pct, 2),
        "overlay_image": overlay_img,
        "comparison_card": comparison_card,
        "cam_array": cam
    }
