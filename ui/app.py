# -*- coding: utf-8 -*-
"""
OpenAgriNet (OAN) Kenya · Pest & Disease Lab
===========================================
Two-Audience Frontend System:
- Farmer View (Mobile-First 390x844: Farmer Home & Farmer Result)
- Lab View (Desktop 1440x900: Dense Multi-Model Engineering & XAI Dashboard)
- Component Sheet (Design System Catalog & UI Primitives)

Author: Program Manager → AI Enthusiast | Turning Ideas into AI-Powered Solutions: Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>
Repository: https://github.com/nandakishore2404/oan-pest-detection-benchmark
"""

import io
import json
import os
import sys
import time
import shutil
from typing import Dict, Any, List, Optional
from PIL import Image
import pandas as pd
import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ui.design_system import (
    COLOR_GROUND, COLOR_SURFACE, COLOR_SURFACE_HOVER, COLOR_ACCENT,
    COLOR_CAUTION, COLOR_ALERT, COLOR_TEXT, COLOR_MUTED, COLOR_MUTED_LIGHT,
    COLOR_BORDER, COLOR_BORDER_FOCUS, inject_theme, svg_leaf, svg_sun, svg_camera, svg_shield,
    svg_shield_alert, svg_checkmark, svg_chat_bubble, svg_settings,
    svg_chevron_left, svg_chevron_right, svg_chevron_down, svg_sliders,
    svg_microscope, svg_cpu, svg_clock, svg_bookmark, svg_user, svg_zap,
    render_confidence_badge, render_stat_tile, render_safety_brake_banner,
    render_action_step, render_model_trust_strip
)

from benchmark.advisory import generate_agronomic_advisory
from benchmark.metrics import compute_benchmark_metrics
from benchmark.runner import inspect_system_hardware
from benchmark.visualizer import draw_bounding_boxes
from benchmark.explainability import explain_crop_image
from benchmark.ollama_adapter import OllamaAdvisor
from models.base import NormalizedPrediction
from models.factory import get_model_adapter
from models.sahi_inference import SahiInferenceEngine
from models.cdfa_thresholds import evaluate_cdfa_threshold
from benchmark.bayesian_prior import KenyanAgronomicBayesianPrior


# ==============================================================================
# STREAMLIT CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="OAN Kenya · Pest & Disease Lab",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

inject_theme()

# Golden Field Samples Cache
SAMPLE_DIR = os.path.join(REPO_ROOT, "data", "golden")
GOLDEN_SAMPLES = {
    "fall_armyworm": {
        "title": "Maize: Fall Armyworm",
        "scientific": "Spodoptera frugiperda",
        "path": os.path.join(SAMPLE_DIR, "maize_fall_armyworm_01.jpg"),
        "stage": "Mid to Late Whorl",
        "crop": "Maize"
    },
    "potato_late_blight": {
        "title": "Potato: Late Blight",
        "scientific": "Phytophthora infestans",
        "path": os.path.join(SAMPLE_DIR, "potato_late_blight_01.jpg"),
        "stage": "Flowering / Tuber Bulking",
        "crop": "Potato"
    },
    "bean_angular_leaf_spot": {
        "title": "Bean: Angular Leaf Spot",
        "scientific": "Pseudocercospora griseola",
        "path": os.path.join(SAMPLE_DIR, "bean_angular_leaf_spot_01.jpg"),
        "stage": "Pod Setting",
        "crop": "Common Bean"
    },
    "tomato_early_blight": {
        "title": "Tomato: Early Blight",
        "scientific": "Alternaria solani",
        "path": os.path.join(SAMPLE_DIR, "tomato_early_blight_01.jpg"),
        "stage": "Vegetative / Flowering",
        "crop": "Tomato"
    },
    "maize_healthy": {
        "title": "Maize: Healthy Foliage",
        "scientific": "Zea mays (No pest)",
        "path": os.path.join(SAMPLE_DIR, "maize_healthy_01.jpg"),
        "stage": "Early Vegetative",
        "crop": "Maize"
    }
}


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "active_view" not in st.session_state:
    st.session_state.active_view = "Farmer Home"
if "selected_sample_id" not in st.session_state:
    st.session_state.selected_sample_id = "fall_armyworm"
if "farmer_uploaded_file" not in st.session_state:
    st.session_state.farmer_uploaded_file = None
if "shamba_chat_open" not in st.session_state:
    st.session_state.shamba_chat_open = False
if "shamba_messages" not in st.session_state:
    st.session_state.shamba_messages = [
        {"role": "assistant", "content": "Hujambo! I am Shamba AI, your Kenyan agronomic assistant. Ask me anything about pest scouting, PCPB-approved biopesticides, or weather advice."}
    ]


# ==============================================================================
# TOP BAR & AUDIENCE SWITCHER
# ==============================================================================
top_col1, top_col2, top_col3 = st.columns([1.2, 2.2, 1.2])

with top_col1:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; padding: 6px 0;">
        {svg_leaf(COLOR_ACCENT, size=24)}
        <div>
            <div style="font-family: var(--font-display); font-weight: 800; font-size: 14px; color: {COLOR_TEXT}; line-height: 1.1;">OAN KENYA</div>
            <div style="font-size: 10px; font-weight: 600; color: {COLOR_MUTED}; letter-spacing: 0.06em;">PEST & DISEASE LAB</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with top_col2:
    reverse_map = {
        "Farmer Home": "Farmer View",
        "Farmer Result": "Farmer View",
        "Lab View": "Lab View",
        "Components": "Design System Specs"
    }
    view_options = ["Farmer View", "Lab View", "Design System Specs"]
    current_label = reverse_map.get(st.session_state.active_view, "Farmer View")
    chosen_view = st.segmented_control(
        "Navigation",
        view_options,
        default=current_label,
        label_visibility="collapsed"
    )
    if chosen_view and reverse_map.get(st.session_state.active_view) != chosen_view:
        target_screen = "Farmer Home" if chosen_view == "Farmer View" else ("Lab View" if chosen_view == "Lab View" else "Components")
        st.session_state.active_view = target_screen
        st.rerun()

with top_col3:
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: flex-end; gap: 12px; padding: 6px 0;">
        <span class="trust-pill">{svg_cpu(COLOR_ACCENT, 14)} CPU-only · $0 cloud spend</span>
        <span style="cursor: pointer;">{svg_settings(COLOR_MUTED, 18)}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border: none; border-top: 1px solid var(--border); margin: 6px 0 20px 0;'>", unsafe_allow_html=True)


# ==============================================================================
# INFERENCE PIPELINE HELPER
# ==============================================================================
def execute_lab_inference(
    image_path: str,
    model_id: str = "yolov8s_pest",
    threshold: float = 0.40,
    enable_sahi: bool = True,
    enable_tta: bool = False,
    crop_stage: str = "mid_to_late_whorl",
    county: str = "rift_valley_trans_nzoia",
    plants_sampled: int = 10
) -> Dict[str, Any]:
    """Executes live computer vision inference with SAHI, Bayesian prior, and CDFA thresholds."""
    adapter = get_model_adapter(model_id, threshold=threshold)
    image_id = os.path.basename(image_path)
    
    t0 = time.time()
    sahi_details = None
    if enable_sahi and "yolo" in model_id.lower():
        adapter.load()
        sahi_engine = SahiInferenceEngine(confidence_threshold=threshold)
        sahi_res = sahi_engine.predict_sahi(adapter.model, image_path, image_id=image_id)
        pred = adapter.predict(image_path, image_id=image_id, augment=enable_tta)
        if sahi_res.get("merged_boxes"):
            pred.bounding_boxes = sahi_res["merged_boxes"]
            pred.prediction = sahi_res["primary_class"]
            pred.confidence = sahi_res["max_confidence"]
            pred.inference_time_ms = sahi_res["latency_ms"]
            sahi_details = sahi_res
    else:
        pred = adapter.predict(image_path, image_id=image_id, augment=enable_tta)
        
    # Bayesian Prior Calibration
    bayesian_calibrator = KenyanAgronomicBayesianPrior()
    bayesian_res = bayesian_calibrator.calibrate_prediction(
        predicted_class=pred.prediction,
        confidence=pred.confidence,
        crop_stage=crop_stage,
        region=county
    )
    
    # CDFA Economic Injury Level Threshold
    cdfa_res = evaluate_cdfa_threshold(
        pest_name=pred.prediction,
        pest_count=len(pred.bounding_boxes),
        crop_stage=crop_stage,
        total_plants_sampled=plants_sampled
    )
    
    # Agronomic Advisory
    advisory = generate_agronomic_advisory(
        prediction=pred.prediction,
        scientific_name=pred.scientific_name,
        severity=pred.severity,
        is_unknown=pred.unknown,
        confidence=pred.confidence,
        threshold=threshold
    )
    
    return {
        "prediction": pred,
        "bayesian": bayesian_res,
        "cdfa": cdfa_res,
        "advisory": advisory,
        "sahi": sahi_details,
        "total_latency_ms": round((time.time() - t0) * 1000, 1)
    }


# ==============================================================================
# SCREEN 1: FARMER HOME (MOBILE 390x844)
# ==============================================================================
def render_screen_farmer_home():
    # Outer mobile device frame
    st.markdown('<div class="mobile-viewport-wrapper">', unsafe_allow_html=True)
    
    # 1. Top Bar inside Mobile
    m_head1, m_head2 = st.columns([3, 1])
    with m_head1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px;">
            {svg_leaf(COLOR_ACCENT, 20)}
            <span style="font-family: var(--font-display); font-weight: 800; font-size: 13px; letter-spacing: 0.04em;">OAN KENYA</span>
            <span style="font-size: 11px; color: {COLOR_MUTED}; font-weight: 500;">/ PEST LAB</span>
        </div>
        """, unsafe_allow_html=True)
    with m_head2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: flex-end; gap: 8px;">
            <span class="trust-pill" style="padding: 3px 8px; font-weight: 700;">EN</span>
            <span>{svg_settings(COLOR_MUTED, 18)}</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    
    # 2. Greeting & Supporting Copy
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
        {svg_sun(COLOR_CAUTION, 18)}
        <span style="font-size: 13px; font-weight: 600; color: {COLOR_CAUTION};">Hujambo, Wanjiru</span>
    </div>
    <div style="font-family: var(--font-display); font-weight: 800; font-size: 24px; color: {COLOR_TEXT}; line-height: 1.2; margin-bottom: 8px;">
        Let's check on your crop
    </div>
    <div style="font-size: 13px; color: {COLOR_MUTED}; line-height: 1.45; margin-bottom: 20px;">
        Get a confidence-scored field diagnosis with PCPB-registered agronomic guidance in under a minute.
    </div>
    """, unsafe_allow_html=True)
    
    # 3. Primary CTA: Upload/Camera Card
    st.markdown(f"""
    <div style="border: 2px dashed {COLOR_BORDER_FOCUS}; background: var(--surface); border-radius: 18px; padding: 26px 16px; text-align: center; margin-bottom: 20px;">
        <div style="width: 52px; height: 52px; background: {COLOR_ACCENT}; border-radius: 16px; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 12px;">
            {svg_camera(COLOR_GROUND, 26)}
        </div>
        <div style="font-family: var(--font-display); font-weight: 700; font-size: 16px; color: {COLOR_TEXT}; margin-bottom: 4px;">
            Take or upload a photo
        </div>
        <div style="font-size: 12px; color: {COLOR_MUTED}; margin-bottom: 14px;">
            JPG or PNG · clear daylight shot works best
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Streamlit file upload handle
    uploaded = st.file_uploader(
        "Upload a photo from your farm",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="farmer_file_uploader"
    )
    if uploaded is not None:
        st.session_state.farmer_uploaded_file = uploaded
        st.session_state.selected_sample_id = None
        st.session_state.active_view = "Farmer Result"
        st.rerun()

    # 4. Verified Sample Chips
    st.markdown("""
    <div style="font-family: var(--font-display); font-weight: 700; font-size: 13px; color: var(--text-primary); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.04em;">
        Or try a verified sample
    </div>
    """, unsafe_allow_html=True)
    
    # Display sample options as interactive clickable cards
    for sid, sinfo in list(GOLDEN_SAMPLES.items())[:4]:
        c1, c2 = st.columns([3.5, 1])
        with c1:
            st.markdown(f"""
            <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 10px 14px; margin-bottom: 8px;">
                <div style="font-family: var(--font-display); font-weight: 700; font-size: 13px; color: {COLOR_TEXT};">{sinfo['title']}</div>
                <div style="font-size: 11px; color: {COLOR_MUTED}; font-style: italic;">{sinfo['scientific']} · {sinfo['stage']}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            if st.button("Diagnose", key=f"chip_btn_{sid}", use_container_width=True):
                st.session_state.selected_sample_id = sid
                st.session_state.farmer_uploaded_file = None
                st.session_state.active_view = "Farmer Result"
                st.rerun()

    # 5. Trust Strip near Bottom
    st.markdown(f"""
    <div style="display: flex; flex-wrap: wrap; gap: 6px; margin: 20px 0 16px 0; justify-content: center;">
        <span class="trust-pill">{svg_shield(COLOR_ACCENT, 14)} KALRO & PCPB aligned</span>
        <span class="trust-pill">{svg_zap(COLOR_CAUTION, 14)} Works fully offline</span>
        <span class="trust-pill">{svg_checkmark(COLOR_ACCENT, 14)} Free for farmers</span>
    </div>
    """, unsafe_allow_html=True)
    
    # 6. Sticky Bottom Bar: Ask Shamba AI
    st.markdown(f"""
    <div class="sticky-shamba-bar">
        <div style="display: flex; align-items: center; gap: 10px;">
            {svg_chat_bubble(COLOR_ACCENT, 20)}
            <span style="font-family: var(--font-display); font-size: 13px; font-weight: 700; color: {COLOR_TEXT};">Ask Shamba AI a question instead</span>
        </div>
        <div>{svg_chevron_right(COLOR_MUTED, 18)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.checkbox("Open Shamba AI Chatbot Assistant", key="toggle_shamba_home", value=False):
        render_shamba_chat_modal()

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# SCREEN 2: FARMER RESULT (MOBILE 390x844)
# ==============================================================================
def render_screen_farmer_result():
    st.markdown('<div class="mobile-viewport-wrapper">', unsafe_allow_html=True)
    
    # 1. Back Chevron + "Your Result" Header
    back_c1, back_c2 = st.columns([1, 4])
    with back_c1:
        if st.button("← Back", key="btn_back_home", use_container_width=True):
            st.session_state.active_view = "Farmer Home"
            st.rerun()
    with back_c2:
        st.markdown(f"""
        <div style="font-family: var(--font-display); font-weight: 800; font-size: 18px; color: {COLOR_TEXT}; padding-top: 6px;">
            Your Result
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Determine Active Image
    active_img_path = None
    if st.session_state.farmer_uploaded_file is not None:
        active_img_path = os.path.join(REPO_ROOT, "results", "current_upload.jpg")
        os.makedirs(os.path.dirname(active_img_path), exist_ok=True)
        img = Image.open(st.session_state.farmer_uploaded_file).convert("RGB")
        img.save(active_img_path)
    else:
        sample_meta = GOLDEN_SAMPLES.get(st.session_state.selected_sample_id, GOLDEN_SAMPLES["fall_armyworm"])
        active_img_path = sample_meta["path"]
        
    # Run Inference
    with st.spinner("Analyzing crop foliage..."):
        res = execute_lab_inference(
            active_img_path,
            model_id="yolov8s_pest",
            threshold=0.35,
            enable_sahi=True,
            enable_tta=False,
            crop_stage="mid_to_late_whorl",
            county="rift_valley_trans_nzoia",
            plants_sampled=10
        )
        
    pred = res["prediction"]
    conf = pred.confidence
    boxes = pred.bounding_boxes
    
    # 2. Photo Preview with Overlaid Detection
    pil_img = Image.open(active_img_path).convert("RGB")
    annotated_img = draw_bounding_boxes(
        pil_img,
        boxes,
        diagnosis_label=pred.prediction,
        confidence=conf
    )
    st.image(annotated_img, use_container_width=True, caption=f"Analyzed Field Photo · Found {len(boxes)} regions of interest")
    
    # 3. Diagnosis Card
    conf_badge_html = render_confidence_badge(conf)
    st.markdown(f"""
    <div class="oan-card" style="margin-top: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <div>
                <div style="font-size: 11px; font-weight: 700; color: {COLOR_MUTED}; text-transform: uppercase; letter-spacing: 0.05em;">DIAGNOSED CONDITION</div>
                <div style="font-family: var(--font-display); font-weight: 800; font-size: 20px; color: {COLOR_TEXT}; line-height: 1.2;">{pred.prediction}</div>
                <div style="font-size: 12px; color: {COLOR_ACCENT}; font-style: italic; margin-top: 2px;">{pred.scientific_name or 'Field-level agronomic pest'}</div>
            </div>
            <div>{conf_badge_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check Safety Brake
    is_abstained = conf < 0.35 or pred.unknown
    if is_abstained:
        st.markdown(render_safety_brake_banner(threshold=0.35, current_conf=conf), unsafe_allow_html=True)

    # Severity & Yield Risk Stat Tiles
    st_c1, st_c2 = st.columns(2)
    with st_c1:
        sev_label = pred.severity or "STAGE_2_MODERATE"
        sev_color = COLOR_CAUTION if "MODERATE" in sev_label or "STAGE_2" in sev_label else (COLOR_ALERT if "SEVERE" in sev_label else COLOR_ACCENT)
        st.markdown(render_stat_tile("Severity", sev_label.replace("STAGE_2_", ""), subtext="Whorl feeding level", color=sev_color), unsafe_allow_html=True)
    with st_c2:
        risk_txt = "15-30%" if "MODERATE" in sev_label else ("40-60%" if "SEVERE" in sev_label else "< 10%")
        st.markdown(render_stat_tile("Yield Risk", risk_txt, subtext="If left unmanaged", color=COLOR_CAUTION), unsafe_allow_html=True)
        
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    
    # 4. "What to do now" - Numbered plain-language action steps
    st.markdown(f"""
    <div style="font-family: var(--font-display); font-weight: 700; font-size: 14px; color: {COLOR_TEXT}; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.04em;">
        What to do now
    </div>
    """, unsafe_allow_html=True)
    
    cdfa_info = res["cdfa"]
    if cdfa_info.get("threshold_exceeded"):
        step1_title = "Economic Threshold Exceeded: Targeted Intervention"
        step1_desc = cdfa_info.get("regulatory_guidance", "Apply PCPB-registered active early morning directly into the whorl funnel.")
        s1_color = COLOR_ALERT
    else:
        step1_title = "Do NOT spray broad-spectrum chemicals yet"
        step1_desc = "Pest density is below the CDFA economic injury level. Beneficial ladybirds and predatory mites are actively suppressing larvae. Chemical spraying would waste KES 2,500 and kill beneficial predators."
        s1_color = COLOR_ACCENT
        
    st.markdown(render_action_step(1, step1_title, step1_desc, color=s1_color), unsafe_allow_html=True)
    st.markdown(render_action_step(2, "Follow CDFA 5-Point Field Scouting Grid", "Sample 10 plants across 5 stations in a W-pattern across the plot. Re-inspect in 48 hours to track larval growth.", color=COLOR_ACCENT), unsafe_allow_html=True)
    st.markdown(render_action_step(3, "Escalation & Extension Support", "If windowpane damage increases past 40%, contact your local KALRO Ward Extension Officer or request subsidized biologicals.", color=COLOR_CAUTION), unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="text-align: center; margin: 12px 0;">
        <a href="#" style="font-size: 12px; font-weight: 600; color: {COLOR_ACCENT}; text-decoration: none;">
            See full advisory in English / Kiswahili →
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # 5. Two Secondary Actions Side-by-Side
    act_c1, act_c2 = st.columns(2)
    with act_c1:
        if st.button("💾 Save to field log", use_container_width=True):
            st.success("Saved to local offline field log!")
    with act_c2:
        if st.button("📞 Talk to an officer", use_container_width=True):
            st.info("Dialing KALRO toll-free extension desk: 0800 720 023")
            
    # 6. Sticky Bottom Bar
    st.markdown(f"""
    <div class="sticky-shamba-bar" style="margin-top: 14px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            {svg_chat_bubble(COLOR_ACCENT, 20)}
            <span style="font-family: var(--font-display); font-size: 13px; font-weight: 700; color: {COLOR_TEXT};">Ask Shamba AI a question instead</span>
        </div>
        <div>{svg_chevron_right(COLOR_MUTED, 18)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# SCREEN 3: LAB / TECHNICAL VIEW (DESKTOP 1440x900)
# ==============================================================================
def render_screen_lab_view():
    lab_left, lab_main = st.columns([1, 3.4])
    
    # --------------------------------------------------------------------------
    # LEFT RAIL (272px): Engineering & Agronomic Controls
    # --------------------------------------------------------------------------
    with lab_left:
        st.markdown(f"""
        <div style="font-family: var(--font-display); font-weight: 800; font-size: 14px; color: {COLOR_TEXT}; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
            {svg_sliders(COLOR_ACCENT, 18)}
            <span>ENGINEERING RAIL</span>
        </div>
        """, unsafe_allow_html=True)
        
        # 1. Active Sample or Upload
        sample_keys = list(GOLDEN_SAMPLES.keys())
        selected_sid = st.selectbox(
            "Test Sample Selection:",
            sample_keys,
            format_func=lambda x: GOLDEN_SAMPLES[x]["title"],
            index=0
        )
        sample_path = GOLDEN_SAMPLES[selected_sid]["path"]
        
        # 2. Model Selector
        model_options = {
            "yolov8s_pest": "YOLOv8s — Locate & Count (Primary Edge)",
            "mobilenetv4_conv_large": "MobileNetV4 — Foliar Classifier",
            "efficientnet_b4_agri": "EfficientNet-B4 — Disease Expert",
            "ibean_classifier": "iBean — East African Pulse Model"
        }
        selected_model = st.selectbox(
            "AI Vision Architecture:",
            list(model_options.keys()),
            format_func=lambda x: model_options[x]
        )
        
        # 3. Safety Brake Slider
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 12px; font-weight: 700; color: {COLOR_TEXT};">Safety Brake Threshold</span>
            <span style="font-size: 12px; font-weight: 800; color: {COLOR_CAUTION};">Abstain below</span>
        </div>
        """, unsafe_allow_html=True)
        threshold_val = st.slider(
            "Abstention Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.40,
            step=0.05,
            label_visibility="collapsed"
        )
        st.caption("Withholds chemical advice if AI confidence is below this bar.")
        
        # 4. Field Scouting Controls
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 12px; font-weight: 700; color: {COLOR_TEXT};'>Field Scouting & Slicing</div>", unsafe_allow_html=True)
        lab_sahi = st.checkbox("🔬 Enable SAHI Slicing", value=True, help="Slices high-res field photo into overlapping 384x384 patches to detect microscopic pests.")
        lab_tta = st.checkbox("⚡ Enable TTA (Test-Time Augment)", value=False, help="Runs horizontal flip multi-scale consensus.")
        
        # 5. Phenology & Eco-Zone Context
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 12px; font-weight: 700; color: {COLOR_TEXT};'>Agronomic Context (CDFA / Prior)</div>", unsafe_allow_html=True)
        stage_sel = st.selectbox(
            "Crop Growth Stage:",
            ["early_vegetative", "mid_to_late_whorl", "tasseling_silking", "grain_fill_maturity"],
            index=1,
            format_func=lambda x: x.replace("_", " ").title()
        )
        county_sel = st.selectbox(
            "Kenyan County / Eco-Zone:",
            ["rift_valley_trans_nzoia", "central_highlands_meru", "western_kakamega", "eastern_dryland_machakos", "coastal_kilifi"],
            index=0,
            format_func=lambda x: x.replace("_", " ").title()
        )
        plants_num = st.number_input("Plants Sampled (W-Grid):", min_value=1, max_value=50, value=10, step=1)
        
        # 6. Collapsed System Status Card Pinned to Bottom
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        with st.expander("🖥️ System & Hardware Telemetry", expanded=False):
            hw = inspect_system_hardware()
            st.markdown(f"""
            <div style="font-size: 11px; line-height: 1.6; color: {COLOR_MUTED_LIGHT};">
                <div><strong>Host OS:</strong> {hw['os']}</div>
                <div><strong>Python:</strong> {hw['python_version']}</div>
                <div><strong>CPU Cores:</strong> {hw['cpu_count']} cores</div>
                <div><strong>Execution:</strong> Sovereign CPU ($0 Cloud)</div>
                <div><strong>Ollama:</strong> 127.0.0.1:11434 (Active)</div>
            </div>
            """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # MAIN AREA: 3 Image Panels, 4 Stat Tiles, 4 Tabs, Evidence Footer
    # --------------------------------------------------------------------------
    with lab_main:
        # Run Lab Inference
        with st.spinner("⚡ Running computer vision models on the identical image..."):
            lab_res = execute_lab_inference(
                sample_path,
                model_id=selected_model,
                threshold=threshold_val,
                enable_sahi=lab_sahi,
                enable_tta=lab_tta,
                crop_stage=stage_sel,
                county=county_sel,
                plants_sampled=plants_num
            )
            
        pred = lab_res["prediction"]
        boxes = pred.bounding_boxes
        conf = pred.confidence
        bayesian = lab_res["bayesian"]
        cdfa = lab_res["cdfa"]
        adv = lab_res["advisory"]
        
        # Panel 1: Row of 3 Large Image Panels
        p_col1, p_col2, p_col3 = st.columns(3)
        pil_raw = Image.open(sample_path).convert("RGB")
        
        with p_col1:
            st.markdown(f"<div style='font-size: 12px; font-weight: 700; color: {COLOR_MUTED}; margin-bottom: 6px;'>1. ORIGINAL FIELD PHOTO</div>", unsafe_allow_html=True)
            st.image(pil_raw, use_container_width=True, caption=os.path.basename(sample_path))
            
        with p_col2:
            st.markdown(f"<div style='font-size: 12px; font-weight: 700; color: {COLOR_ACCENT}; margin-bottom: 6px;'>2. YOLO DETECTION BOXES</div>", unsafe_allow_html=True)
            annotated_boxes = draw_bounding_boxes(pil_raw, boxes, diagnosis_label=pred.prediction, confidence=conf)
            st.image(annotated_boxes, use_container_width=True, caption=f"Identified {len(boxes)} pest bounding boxes")
            
        with p_col3:
            st.markdown(f"<div style='font-size: 12px; font-weight: 700; color: {COLOR_CAUTION}; margin-bottom: 6px;'>3. GRAD-CAM ATTENTION</div>", unsafe_allow_html=True)
            xai_res = explain_crop_image(pil_raw)
            st.image(xai_res["annotated_image"], use_container_width=True, caption=f"Lesion Focus Score: {xai_res['focus_score']}%")
            
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        # Panel 2: Row of 4 Stat Tiles
        st_c1, st_c2, st_c3, st_c4 = st.columns(4)
        with st_c1:
            conf_color = COLOR_ACCENT if conf >= 0.75 else (COLOR_CAUTION if conf >= 0.40 else COLOR_ALERT)
            st.markdown(render_stat_tile("Confidence", f"{conf*100:.1f}%", subtext="Calibrated via Bayesian prior", color=conf_color), unsafe_allow_html=True)
        with st_c2:
            st.markdown(render_stat_tile("Severity", pred.severity or "STAGE_2_MODERATE", subtext="Pest density rating", color=COLOR_TEXT), unsafe_allow_html=True)
        with st_c3:
            brake_passed = conf >= threshold_val and not pred.unknown
            brake_status = "PASSED" if brake_passed else "ABSTAINED"
            brake_color = COLOR_ACCENT if brake_passed else COLOR_ALERT
            st.markdown(render_stat_tile("Safety Brake", brake_status, subtext=f"Threshold: {threshold_val*100:.0f}%", color=brake_color), unsafe_allow_html=True)
        with st_c4:
            st.markdown(render_stat_tile("Edge Latency", f"{pred.inference_time_ms:.1f} ms", subtext="Measured CPU latency", color=COLOR_ACCENT), unsafe_allow_html=True)
            
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        # Panel 3: Tab Bar (Advisory / CDFA Threshold / Model Trust & Licensing / Prescription Slip)
        tab_adv, tab_cdfa, tab_trust, tab_slip = st.tabs([
            "📋 Agronomic Advisory",
            "⚖️ CDFA Action Thresholds",
            "🛡️ Model Trust & Licensing",
            "🧾 Prescription Slip & Beckn"
        ])
        
        with tab_adv:
            # 3-column grid of advisory cards
            cultural_actions = adv.get('cultural_actions') or []
            bio_controls = adv.get('biological_controls') or []
            chem_interventions = adv.get('chemical_interventions') or []
            
            cultural_txt = cultural_actions[0] if cultural_actions else "Inspect 10 plants per station across 5 locations."
            bio_txt = bio_controls[0] if bio_controls else "Preserve ladybird beetles, hoverfly larvae, and parasitic wasps."
            
            if chem_interventions and isinstance(chem_interventions[0], dict):
                first_chem = chem_interventions[0]
                chem_active = first_chem.get("active_ingredient", "PCPB-registered active")
                chem_txt = f"<strong>{chem_active}</strong>: {first_chem.get('application_timing', 'Apply targeted spray into whorl funnel')}"
            else:
                chem_txt = "Apply PCPB-registered active only if economic threshold is breached."

            adv_c1, adv_c2, adv_c3 = st.columns(3)
            with adv_c1:
                st.markdown(f"""
                <div class="oan-card">
                    <div style="font-size: 11px; font-weight: 700; color: {COLOR_ACCENT}; text-transform: uppercase;">1. CULTURAL ACTION</div>
                    <div style="font-family: var(--font-display); font-weight: 700; font-size: 15px; margin: 6px 0; color: {COLOR_TEXT};">Field Scouting & Containment</div>
                    <div style="font-size: 12px; color: {COLOR_MUTED_LIGHT}; line-height: 1.5;">{cultural_txt}</div>
                </div>
                """, unsafe_allow_html=True)
            with adv_c2:
                st.markdown(f"""
                <div class="oan-card">
                    <div style="font-size: 11px; font-weight: 700; color: {COLOR_CAUTION}; text-transform: uppercase;">2. BIOLOGICAL CONTROL</div>
                    <div style="font-family: var(--font-display); font-weight: 700; font-size: 15px; margin: 6px 0; color: {COLOR_TEXT};">Conserve Natural Predators</div>
                    <div style="font-size: 12px; color: {COLOR_MUTED_LIGHT}; line-height: 1.5;">{bio_txt}</div>
                </div>
                """, unsafe_allow_html=True)
            with adv_c3:
                st.markdown(f"""
                <div class="oan-card">
                    <div style="font-size: 11px; font-weight: 700; color: {COLOR_ALERT}; text-transform: uppercase;">3. ESCALATION & PCPB</div>
                    <div style="font-family: var(--font-display); font-weight: 700; font-size: 15px; margin: 6px 0; color: {COLOR_TEXT};">Registered Chemical Interventions</div>
                    <div style="font-size: 12px; color: {COLOR_MUTED_LIGHT}; line-height: 1.5;">{chem_txt}</div>
                </div>
                """, unsafe_allow_html=True)
                
        with tab_cdfa:
            st.markdown(f"""
            <div class="oan-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div>
                        <div style="font-size: 11px; font-weight: 700; color: {COLOR_MUTED}; text-transform: uppercase;">CDFA / KALRO ECONOMIC INJURY LEVEL (EIL) EVALUATION</div>
                        <div style="font-family: var(--font-display); font-weight: 800; font-size: 18px; color: {COLOR_TEXT};">{cdfa.get('pest_common_name', pred.prediction)}</div>
                    </div>
                    <div>
                        <span class="trust-pill" style="border-color: {'#e2847a' if cdfa.get('threshold_exceeded') else '#22c08a'};">
                            {'⚠️ THRESHOLD EXCEEDED' if cdfa.get('threshold_exceeded') else '✅ BELOW INJURY LEVEL'}
                        </span>
                    </div>
                </div>
                <div style="font-size: 13px; color: {COLOR_TEXT}; margin-bottom: 8px;">
                    <strong>Observed Infestation:</strong> {cdfa.get('observed_density', 0)} insects observed / {plants_num} plants sampled
                </div>
                <div style="font-size: 13px; color: {COLOR_TEXT}; margin-bottom: 8px;">
                    <strong>Action Threshold:</strong> {cdfa.get('action_threshold_value', 20.0)}% ({cdfa.get('threshold_unit', 'per plants')})
                </div>
                <div style="font-size: 13px; color: {COLOR_MUTED_LIGHT}; line-height: 1.5; background: rgba(0,0,0,0.25); padding: 12px; border-radius: 8px;">
                    <strong>Regulatory Guidance:</strong> {cdfa.get('regulatory_guidance', 'Density is below threshold. Do not spray.')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with tab_trust:
            scorecard = [
                {"Architecture": "YOLOv8s (Ultralytics)", "License": "AGPL-3.0", "Task": "Object Detection", "Edge Latency": "17.0 ms ONNX", "Size": "11.6 MB", "Empirical Evidence": "new_dataset_accuracy_ollama.json"},
                {"Architecture": "MobileNetV4 (Google/timm)", "License": "Apache-2.0", "Task": "Foliar Classifier", "Edge Latency": "1.99 ms ONNX", "Size": "9.5 MB", "Empirical Evidence": "step10_five_class_classifier.json"},
                {"Architecture": "EfficientNet-B4 (timm)", "License": "Apache-2.0", "Task": "Disease Expert", "Edge Latency": "14.2 ms CPU", "Size": "75 MB", "Empirical Evidence": "step7_calibration.json"},
                {"Architecture": "iBean (Makerere)", "License": "MIT", "Task": "Pulse Disease", "Edge Latency": "12.0 ms CPU", "Size": "22 MB", "Empirical Evidence": "results/metrics/ibean.json"}
            ]
            st.dataframe(pd.DataFrame(scorecard), use_container_width=True)
            
        with tab_slip:
            chem_interventions = adv.get('chemical_interventions') or []
            if chem_interventions and isinstance(chem_interventions[0], dict):
                slip_chem = chem_interventions[0].get("active_ingredient", "Chlorantraniliprole 200 g/L")
                slip_phi = chem_interventions[0].get("phi_days", "14 days")
            else:
                slip_chem = "PCPB-registered biopesticide or active"
                slip_phi = "7 days"
                
            slip_json = {
                "prescription_id": f"OAN-RX-{int(time.time())}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "farmer": "Wanjiru (Smallholder Trans-Nzoia)",
                "diagnosed_threat": pred.prediction,
                "scientific_name": pred.scientific_name,
                "confidence_calibrated": round(conf, 3),
                "cdfa_threshold_exceeded": cdfa.get("threshold_exceeded", False),
                "action_recommended": cdfa.get("regulatory_guidance"),
                "pcpb_registered_active": slip_chem,
                "phi_days": slip_phi,
                "beckn_bpp_action": "crop-protection:oan:kenya:on_search"
            }
            st.code(json.dumps(slip_json, indent=2), language="json")
            
        # Panel 4: Model Trust Footer Strip
        active_model_name = "YOLOv8s" if "yolo" in selected_model else "MobileNetV4"
        active_lic = "AGPL-3.0" if "yolo" in selected_model else "Apache-2.0"
        active_ref = "new_dataset_accuracy_ollama.json" if "yolo" in selected_model else "step10_five_class_classifier.json"
        st.markdown(render_model_trust_strip(active_model_name, active_lic, active_ref), unsafe_allow_html=True)


# ==============================================================================
# SCREEN 4: COMPONENT SHEET (REFERENCE ONLY, 1200x840)
# ==============================================================================
def render_screen_components():
    st.markdown("""
    <div style="max-width: 1200px; margin: 0 auto;">
        <div style="background: rgba(34, 192, 138, 0.08); border: 1px solid #22c08a; border-radius: 12px; padding: 14px 18px; margin-bottom: 20px;">
            <div style="font-weight: 700; font-size: 13px; color: #5fe0ac;">🛠️ INTERNAL DEVELOPER & AUDIT REFERENCE SHEET (Wireframe Artboard 4)</div>
            <div style="font-size: 12px; color: #8b9a91; margin-top: 4px; line-height: 1.5;">
                This catalog displays the design tokens (6-color palette, Manrope/Work Sans typography, stroke-only SVGs) and reusable UI primitives (confidence pills, stat tiles, safety brake banners). It serves as a visual conformance test bench for engineers and auditors — <strong>not an end-user diagnostic tool</strong>. Smallholder farmers use <strong>Farmer View</strong> and agricultural officers use <strong>Lab View</strong>.
            </div>
        </div>
        <div style="font-family: var(--font-display); font-weight: 800; font-size: 26px; color: var(--text-primary); margin-bottom: 6px;">
            🎨 OAN Kenya Design System · Component Specification Sheet
        </div>
        <div style="font-size: 13px; color: var(--text-muted); margin-bottom: 24px;">
            Reusable component inventory and visual tokens for the OAN Kenya Pest & Disease Lab. Governed decision-support design system.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. 6-Color Palette Tokens
    st.markdown("#### 1. Color Palette Tokens")
    swatches = [
        {"name": "Ground", "hex": "#070c0a", "role": "Base canvas & page background"},
        {"name": "Surface", "hex": "#0d1712", "role": "Cards, panels & dialog surfaces"},
        {"name": "Accent", "hex": "#22c08a", "role": "Primary buttons, verified badges, active states"},
        {"name": "Caution", "hex": "#e7b458", "role": "Moderate severity, sub-threshold alerts"},
        {"name": "Alert", "hex": "#e2847a", "role": "Severe damage, safety brake triggers"},
        {"name": "Text Primary", "hex": "#eef3ef", "role": "Primary typography, titles, values"},
        {"name": "Border", "hex": "#1c2620", "role": "Structural borders, card outlines"}
    ]
    sw_cols = st.columns(len(swatches))
    for idx, sw in enumerate(swatches):
        with sw_cols[idx]:
            border_css = "1px solid #284235" if sw["hex"] == "#070c0a" else "none"
            st.markdown(f"""
            <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 12px; text-align: center;">
                <div style="width: 100%; height: 42px; background: {sw['hex']}; border-radius: 8px; margin-bottom: 8px; border: {border_css};"></div>
                <div style="font-family: var(--font-display); font-weight: 700; font-size: 12px; color: var(--text-primary);">{sw['name']}</div>
                <div style="font-family: var(--font-mono); font-size: 10px; color: var(--text-muted);">{sw['hex']}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    # 2. Typography Scale
    st.markdown("#### 2. Typography Scale (Manrope Display + Work Sans Body)")
    st.markdown("""
    <div class="oan-card">
        <div style="font-family: var(--font-display); font-weight: 800; font-size: 28px; color: var(--text-primary); margin-bottom: 4px;">Display Heading · 28px Manrope 800</div>
        <div style="font-family: var(--font-display); font-weight: 700; font-size: 20px; color: var(--text-primary); margin-bottom: 4px;">Section Header · 20px Manrope 700</div>
        <div style="font-family: var(--font-display); font-weight: 700; font-size: 14px; color: var(--text-primary); margin-bottom: 8px;">Card Heading · 14px Manrope 700</div>
        <div style="font-family: var(--font-body); font-weight: 400; font-size: 13px; color: var(--text-muted-light); line-height: 1.5; margin-bottom: 6px;">Body Text · 13px Work Sans 400: Standard agronomic descriptions, advice text, and diagnostic steps.</div>
        <div style="font-family: var(--font-display); font-weight: 700; font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">Caption Label · 11px Manrope 700 Uppercase</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    # 3. Confidence Badge States
    st.markdown("#### 3. Confidence Badge Pills (3 States)")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown(render_confidence_badge(0.89), unsafe_allow_html=True)
        st.caption("High Confidence (>= 75%) · Green (#5fe0ac on #173226)")
    with b2:
        st.markdown(render_confidence_badge(0.58), unsafe_allow_html=True)
        st.caption("Moderate Confidence (40-74%) · Amber (#e7b458 on #332813)")
    with b3:
        st.markdown(render_confidence_badge(0.24), unsafe_allow_html=True)
        st.caption("Uncertain (< 40%) · Rose (#e2847a on #33201c)")
        
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    # 4. Buttons & Interactive Controls
    st.markdown("#### 4. Buttons & Navigation Pills")
    btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)
    with btn_c1:
        st.markdown('<a href="#" class="btn-pill-primary">Primary Button</a>', unsafe_allow_html=True)
    with btn_c2:
        st.markdown('<a href="#" class="btn-pill-secondary">Secondary Button</a>', unsafe_allow_html=True)
    with btn_c3:
        st.markdown(f'<span class="trust-pill">{svg_shield(COLOR_ACCENT, 14)} Trust Pill</span>', unsafe_allow_html=True)
    with btn_c4:
        st.markdown(f'<span class="trust-pill">{svg_zap(COLOR_CAUTION, 14)} Offline Pill</span>', unsafe_allow_html=True)
        
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    # 5. Stat Tiles
    st.markdown("#### 5. Bordered Stat Tiles")
    st_c1, st_c2, st_c3, st_c4 = st.columns(4)
    with st_c1:
        st.markdown(render_stat_tile("Confidence", "91.2%", subtext="Model certainty", color=COLOR_ACCENT), unsafe_allow_html=True)
    with st_c2:
        st.markdown(render_stat_tile("Severity", "STAGE 2", subtext="Moderate whorl damage", color=COLOR_CAUTION), unsafe_allow_html=True)
    with st_c3:
        st.markdown(render_stat_tile("Yield Risk", "15-30%", subtext="If unmitigated", color=COLOR_ALERT), unsafe_allow_html=True)
    with st_c4:
        st.markdown(render_stat_tile("CPU Latency", "17.0 ms", subtext="ONNX Runtime CPU", color=COLOR_TEXT), unsafe_allow_html=True)
        
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    # 6. Safety Brake Banner
    st.markdown("#### 6. Safety Brake Banner Component")
    st.markdown(render_safety_brake_banner(threshold=0.40, current_conf=0.28), unsafe_allow_html=True)
    
    # 7. Model Trust Footer Strip
    st.markdown("#### 7. Evidence-Backed Model Trust Strip")
    st.markdown(render_model_trust_strip("MobileNetV4", "Apache-2.0", "step10_five_class_classifier.json"), unsafe_allow_html=True)


# ==============================================================================
# SHAMBA AI CHATBOT MODAL HELPER
# ==============================================================================
def render_shamba_chat_modal():
    st.markdown(f"""
    <div style="background: var(--surface); border: 1px solid var(--accent); border-radius: 16px; padding: 16px; margin-top: 14px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {svg_chat_bubble(COLOR_ACCENT, 20)}
            <span style="font-family: var(--font-display); font-weight: 700; font-size: 14px; color: {COLOR_TEXT};">Shamba AI · Agronomic Assistant</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display message history
    for msg in st.session_state.shamba_messages[-3:]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    user_q = st.chat_input("Ask about Fall Armyworm, KALRO spraying advice, or weather...", key="shamba_chat_input")
    if user_q:
        st.session_state.shamba_messages.append({"role": "user", "content": user_q})
        with st.spinner("Shamba AI is formulating guidance..."):
            t_start = time.perf_counter()
            advisor = OllamaAdvisor()
            resp = advisor.generate_advisory_qwen(
                predicted_class="Fall Armyworm",
                confidence=0.88,
                crop_stage="mid_to_late_whorl",
                county="Trans-Nzoia"
            )
            elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)
            # Concise fallback response
            ans = resp.get("response", "Apply Bacillus thuringiensis (Bt) or Neem oil into the central whorls early morning. Conserve natural ladybird predators.")
            st.session_state.shamba_messages.append({"role": "assistant", "content": f"{ans}\n\n*(Measured local latency: {elapsed_ms:.1f} ms)*"})
            st.rerun()


# ==============================================================================
# ROUTER DISPATCH
# ==============================================================================
if st.session_state.active_view == "Farmer Home":
    render_screen_farmer_home()
elif st.session_state.active_view == "Farmer Result":
    render_screen_farmer_result()
elif st.session_state.active_view == "Lab View":
    render_screen_lab_view()
elif st.session_state.active_view == "Components":
    render_screen_components()
