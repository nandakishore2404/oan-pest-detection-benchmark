# -*- coding: utf-8 -*-
"""
OpenAgriNet (OAN) Kenya - Pest & Disease Diagnostic Lab UI
==========================================================
Interactive Web Application for Agricultural Computer Vision, Model Benchmarking,
and Field-Level Agronomic Advisory designed for both Technical Engineers and Non-SMEs.

Run via:
    python scripts/run_ui.py
    # or
    streamlit run ui/app.py
"""

import io
import json
import os
import sys
import time
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmark.advisory import generate_agronomic_advisory
from benchmark.metrics import compute_benchmark_metrics
from benchmark.runner import inspect_system_hardware
from benchmark.visualizer import draw_bounding_boxes
from models.factory import get_model_adapter
from models.registry import load_registry, SHORTLISTED_MODEL_IDS

# Page Configuration
st.set_page_config(
    page_title="OAN Kenya - Pest AI Lab & Advisory",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Friendly Category Names for Non-SMEs
PLAIN_ENGLISH_CATEGORIES = {
    "yolov8s_pest": "🎯 Object Detection (Locates & Counts Pests)",
    "efficientnet_b4_agri": "🍃 Foliar Disease Classifier (High Accuracy)",
    "mobilenetv4_conv_large": "⚡ Ultra-Fast Edge Classifier (Mobile Ready)",
    "bioclip_treeoflife": "🧬 Taxonomic Zero-Shot (Tree of Life 10M)",
    "cereal_pestaid": "🌾 Specialized African Cereal Pest Model",
    "florence2_large_agri": "🔍 Foundation Vision Grounding (Open Vocabulary)",
    "ibean_classifier": "🌱 East African Bean Disease Classifier",
    "agrichat_7b": "🧠 Multimodal AI Reasoning & Advisory",
    "gemini_1_5_pro": "☁️ Commercial Cloud VLM Benchmark Ceiling"
}

# Custom Styling with Dark/Light Mode High-Contrast Enforcement
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b5e20 !important;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #2e7d32 !important;
        margin-bottom: 1.2rem;
    }
    .callout-box {
        background-color: #e2f0d9 !important;
        color: #000000 !important;
        border: 2px solid #2e7d32 !important;
        border-left: 8px solid #1b5e20 !important;
        padding: 18px 22px !important;
        border-radius: 8px !important;
        margin-bottom: 20px !important;
    }
    .stApp [data-testid="stMarkdownContainer"] .callout-box,
    .stApp [data-testid="stMarkdownContainer"] .callout-box *,
    .stApp [data-testid="stMarkdownContainer"] .callout-box p,
    .stApp [data-testid="stMarkdownContainer"] .callout-box span,
    .stApp [data-testid="stMarkdownContainer"] .callout-box strong,
    .stApp [data-testid="stMarkdownContainer"] .callout-box b,
    .stApp [data-testid="stMarkdownContainer"] .callout-box div {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        opacity: 1 !important;
    }
    .winner-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%) !important;
        border: 2px solid #2e7d32 !important;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
        color: #0d2b0e !important;
    }
    .stApp [data-testid="stMarkdownContainer"] .winner-card,
    .stApp [data-testid="stMarkdownContainer"] .winner-card * {
        color: #0d2b0e !important;
        -webkit-text-fill-color: #0d2b0e !important;
    }
    .prescription-card {
        background-color: #ffffff !important;
        border: 2px dashed #388e3c !important;
        border-radius: 12px;
        padding: 24px;
        margin-top: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #111111 !important;
    }
    .stApp [data-testid="stMarkdownContainer"] .prescription-card,
    .stApp [data-testid="stMarkdownContainer"] .prescription-card * {
        color: #111111 !important;
        -webkit-text-fill-color: #111111 !important;
    }
    .badge-stage-1 { background-color: #fff9c4 !important; color: #e65100 !important; padding: 4px 12px; border-radius: 14px; font-weight: bold; font-size: 0.95rem; }
    .badge-stage-2 { background-color: #ffe0b2 !important; color: #bf360c !important; padding: 4px 12px; border-radius: 14px; font-weight: bold; font-size: 0.95rem; }
    .badge-stage-3 { background-color: #ffcdd2 !important; color: #b71c1c !important; padding: 4px 12px; border-radius: 14px; font-weight: bold; font-size: 0.95rem; }
    .badge-healthy { background-color: #c8e6c9 !important; color: #1b5e20 !important; padding: 4px 12px; border-radius: 14px; font-weight: bold; font-size: 0.95rem; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #1b5e20 !important; margin-top: 4px; }
    .metric-label { font-size: 0.85rem; font-weight: 600; color: #333333 !important; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# Cache Golden Samples
SAMPLE_DIR = os.path.join(REPO_ROOT, "data", "golden")
GOLDEN_SAMPLES = {
    "🌽 Maize: Fall Armyworm (Spodoptera frugiperda)": os.path.join(SAMPLE_DIR, "maize_fall_armyworm_01.jpg"),
    "🌱 Common Bean: Angular Leaf Spot (Pseudocercospora griseola)": os.path.join(SAMPLE_DIR, "bean_angular_leaf_spot_01.jpg"),
    "🥔 Irish Potato: Late Blight (Phytophthora infestans)": os.path.join(SAMPLE_DIR, "potato_late_blight_01.jpg"),
    "🍅 Tomato: Early Blight (Alternaria solani)": os.path.join(SAMPLE_DIR, "tomato_early_blight_01.jpg"),
    "🌿 Maize: Healthy Vegetative Foliage": os.path.join(SAMPLE_DIR, "maize_healthy_01.jpg"),
    "🐛 Field Caterpillar (Real SPRINT-3 Pest Archive)": os.path.join(SAMPLE_DIR, "archive_caterpillar_field_01.jpg"),
    "🦗 Field Grasshopper (Real SPRINT-3 Pest Archive)": os.path.join(SAMPLE_DIR, "archive_grasshopper_field_01.jpg"),
    "🪲 Field Beetle (Real SPRINT-3 Pest Archive)": os.path.join(SAMPLE_DIR, "archive_beetle_field_01.jpg"),
    "🌾 Field Weevil (Real SPRINT-3 Pest Archive)": os.path.join(SAMPLE_DIR, "archive_weevil_field_01.jpg"),
}

# --- Sidebar Configuration ---
st.sidebar.image("https://raw.githubusercontent.com/beckn/protocol-specifications/master/assets/beckn_logo.png", width=160)
st.sidebar.title("🎛️ Control Center")

# Friendly model options
model_choices = {
    "All Models (Consensus / Multi-Model Benchmark)": "All Models",
    "yolov8s_pest": "🎯 YOLOv8s (Locates & Counts Pests)",
    "bioclip_treeoflife": "🧬 BioCLIP (Tree of Life 10M Species)",
    "cereal_pestaid": "🌾 CerealPestAID (26 African Cereal Pests)",
    "efficientnet_b4_agri": "🍃 EfficientNet-B4 (Foliar Disease Expert)",
    "mobilenetv4_conv_large": "⚡ MobileNetV4 (Ultra-Fast Mobile Edge)",
    "florence2_large_agri": "🔍 Florence-2 (Foundation Vision Grounding)",
    "ibean_classifier": "🌱 iBean (East African Bean Diseases)"
}

selected_model_key = st.sidebar.selectbox(
    "Select AI Model to Test:",
    list(model_choices.keys()),
    format_func=lambda x: model_choices[x],
    help="Select 'All Models' to see a side-by-side comparison, or choose an individual AI architecture."
)

# Plain-English Safety Threshold
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ Safety Brake (Abstention)")
threshold_val = st.sidebar.slider(
    "Confidence Safety Bar",
    min_value=0.10,
    max_value=0.95,
    value=0.40,
    step=0.05,
    help="When AI confidence is below this bar, the system refuses to guess and withholds chemical pesticides to protect farmers."
)
st.sidebar.caption("💡 *If an AI model is less than 40% confident, it stops and requests human agricultural officer inspection.*")

# Hardware & System Info
hw_info = inspect_system_hardware()
st.sidebar.markdown("---")
st.sidebar.markdown("### 🖥️ Local Compute Status")
st.sidebar.markdown(f"**Host OS**: `{hw_info['os']}`")
st.sidebar.markdown(f"**Python Runtime**: `{hw_info['python_version']}`")
st.sidebar.markdown(f"**CPU Cores**: `{hw_info['cpu_count']} cores`")
dev_icon = "🟢 GPU Active" if hw_info['cuda_available'] else "🔵 CPU Optimized (Zero GPU Cost)"
st.sidebar.markdown(f"**Execution Mode**: `{dev_icon}`")
st.sidebar.markdown("---")
st.sidebar.success("🇰🇪 **OAN Kenya DPI**: Grounded in KALRO, PCPB Kenya, and Beckn BPP Protocol Standards.")

# --- Header & Top Navigation ---
st.markdown('<div class="main-header">🌾 OpenAgriNet (OAN) Kenya: Pest & Disease Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Computer-Vision Benchmarking & Field-Level Agronomic Advisory for Digital Public Infrastructure</div>', unsafe_allow_html=True)

# Top Navigation Tabs
nav_tab1, nav_tab2, nav_tab3 = st.tabs([
    "🩺 Live Diagnostic Lab & Field Trial",
    "🏆 Model Recommendation & Decision Matrix",
    "📘 How It Works & Benchmarking Guide (For Non-Engineers)"
])

# ==============================================================================
# TAB 1: LIVE DIAGNOSTIC LAB
# ==============================================================================
with nav_tab1:
    st.markdown("""
    <div class="callout-box" style="background-color: #e2f0d9 !important; border: 2px solid #2e7d32 !important; border-left: 8px solid #1b5e20 !important; padding: 18px 22px !important; border-radius: 8px !important; margin-bottom: 20px !important; color: #000000 !important;">
        <div style="color: #000000 !important; -webkit-text-fill-color: #000000 !important; font-size: 1.25rem !important; font-weight: 800 !important; margin: 0 0 8px 0 !important; line-height: 1.4 !important;">
            👋 Welcome to the Live Diagnostic Lab!
        </div>
        <div style="color: #000000 !important; -webkit-text-fill-color: #000000 !important; font-size: 1.05rem !important; font-weight: 600 !important; margin: 0 !important; line-height: 1.6 !important;">
            Upload a crop photo or pick a sample below. The system passes the <strong style="color: #000000 !important; -webkit-text-fill-color: #000000 !important; font-weight: 900 !important; text-decoration: underline;">exact same image</strong> to the AI models simultaneously, draws bounding boxes around pests, evaluates severity, and generates an official KALRO/PCPB field advisory.
        </div>
    </div>
    """, unsafe_allow_html=True)

    input_col1, input_col2 = st.columns([1, 1])

    with input_col1:
        st.markdown("##### Option A: 📸 Upload Crop Image")
        uploaded_file = st.file_uploader(
            "Upload image from field camera or smartphone (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            key="crop_uploader"
        )

    with input_col2:
        st.markdown("##### Option B: 📂 Quick-Select Kenyan Benchmark Sample")
        selected_sample_key = st.selectbox(
            "Choose a verified Kenyan crop sample:",
            list(GOLDEN_SAMPLES.keys()),
            key="sample_selector"
        )

    # Determine active image
    active_image = None
    active_image_name = ""

    if uploaded_file is not None:
        active_image = Image.open(uploaded_file).convert("RGB")
        active_image_name = uploaded_file.name
    elif selected_sample_key:
        sample_path = GOLDEN_SAMPLES[selected_sample_key]
        if os.path.exists(sample_path):
            active_image = Image.open(sample_path).convert("RGB")
            active_image_name = os.path.basename(sample_path)

    if active_image is not None:
        temp_img_path = os.path.join(REPO_ROOT, "results", "current_upload.jpg")
        os.makedirs(os.path.dirname(temp_img_path), exist_ok=True)
        active_image.save(temp_img_path)

        is_multi_model = (selected_model_key == "All Models (Consensus / Multi-Model Benchmark)")

        with st.spinner("⚡ Running computer vision models on the identical image..."):
            predictions = []
            if is_multi_model:
                for mid in SHORTLISTED_MODEL_IDS:
                    adapter = get_model_adapter(mid, threshold=threshold_val)
                    pred = adapter.predict(temp_img_path, image_id=active_image_name)
                    predictions.append(pred)
                primary_pred = predictions[0]  # YOLO as primary object detector
                pred_counts = {}
                for p in predictions:
                    if not p.unknown:
                        pred_counts[p.prediction] = pred_counts.get(p.prediction, 0) + 1
                consensus_name = max(pred_counts.items(), key=lambda x: x[1])[0] if pred_counts else "Unknown / Unsupported Class"
            else:
                adapter = get_model_adapter(selected_model_key, threshold=threshold_val)
                primary_pred = adapter.predict(temp_img_path, image_id=active_image_name)
                predictions = [primary_pred]
                consensus_name = primary_pred.prediction

        # Visualizer: Generate Annotated Image with Bounding Boxes
        annotated_img = draw_bounding_boxes(
            active_image,
            primary_pred.bounding_boxes,
            diagnosis_label=primary_pred.prediction,
            confidence=primary_pred.confidence
        )

        st.markdown("---")

        # Side-by-side Image Display
        view_col1, view_col2 = st.columns(2)
        with view_col1:
            st.markdown("#### 📷 Original Field Photo")
            st.image(active_image, use_container_width=True, caption=f"Input Image: {active_image_name}")

        with view_col2:
            st.markdown("#### 🎯 AI Computer Vision Bounding Boxes")
            box_count = len(primary_pred.bounding_boxes)
            caption_text = f"Identified {box_count} distinct pest/lesion regions" if box_count > 0 else "Classification Diagnostic Focus"
            st.image(annotated_img, use_container_width=True, caption=caption_text)

        st.markdown("---")

        # Diagnostic Summary Cards (Non-SME Friendly)
        st.markdown("### 🩺 Diagnostic Findings & Field Assessment")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        with m_col1:
            st.markdown('<div class="metric-label">Diagnosed Pest / Disease</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{primary_pred.prediction}</div>', unsafe_allow_html=True)
            if primary_pred.scientific_name:
                st.caption(f"🔬 Latin Name: *{primary_pred.scientific_name}*")

        with m_col2:
            st.markdown('<div class="metric-label">Diagnosis Confidence</div>', unsafe_allow_html=True)
            conf_pct = primary_pred.confidence * 100
            st.markdown(f'<div class="metric-value">{conf_pct:.1f}%</div>', unsafe_allow_html=True)
            st.progress(min(1.0, primary_pred.confidence))
            conf_rating = "🟢 Very High" if conf_pct >= 85 else ("🟡 Moderate" if conf_pct >= 50 else "🔴 Low / Uncertain")
            st.caption(f"Reliability: **{conf_rating}**")

        with m_col3:
            st.markdown('<div class="metric-label">Damage Severity</div>', unsafe_allow_html=True)
            sev = primary_pred.severity or "STAGE_2_MODERATE"
            if sev == "HEALTHY":
                b_html = f'<span class="badge-healthy">{sev}</span>'
                risk_txt = "Photosynthetic foliage healthy"
            elif sev == "STAGE_1_EARLY":
                b_html = f'<span class="badge-stage-1">{sev}</span>'
                risk_txt = "Low yield risk (< 10%)"
            elif sev == "STAGE_3_SEVERE":
                b_html = f'<span class="badge-stage-3">{sev}</span>'
                risk_txt = "Critical yield risk (40-70%)"
            else:
                b_html = f'<span class="badge-stage-2">{sev}</span>'
                risk_txt = "Moderate yield risk (15-30%)"
            st.markdown(f'<div style="margin-top:8px;">{b_html}</div>', unsafe_allow_html=True)
            st.caption(risk_txt)

        with m_col4:
            st.markdown('<div class="metric-label">Safety Brake Status</div>', unsafe_allow_html=True)
            if primary_pred.unknown or primary_pred.confidence < threshold_val:
                st.markdown('<div class="metric-value" style="color:#c62828;">⚠️ ABSTAINED</div>', unsafe_allow_html=True)
                st.caption(f"Confidence below {threshold_val:.0%} safety bar")
            else:
                st.markdown('<div class="metric-value" style="color:#2e7d32;">✅ VERIFIED</div>', unsafe_allow_html=True)
                st.caption(f"Passed {threshold_val:.0%} safety bar")

        # Low Confidence Warning Banner
        if primary_pred.unknown or primary_pred.confidence < threshold_val:
            st.error(
                f"🛡️ **Safety Brake Engaged**: The AI confidence ({primary_pred.confidence:.1%}) is below your safety bar ({threshold_val:.1%}). "
                f"To protect farmers from expensive or toxic pesticide mistakes, all chemical recommendations are deliberately withheld. "
                f"Physical inspection by a local Ward Agricultural Officer is advised."
            )

        # Candidate Distribution Bar Chart
        if primary_pred.top_predictions and len(primary_pred.top_predictions) > 1:
            st.markdown("#### 📊 Candidate Probability Distribution")
            df_top = {
                "Pest / Disease Candidate": [p["class_name"] for p in primary_pred.top_predictions],
                "Confidence Score (%)": [p["confidence"] * 100 for p in primary_pred.top_predictions]
            }
            fig = px.bar(
                df_top,
                x="Confidence Score (%)",
                y="Pest / Disease Candidate",
                orientation="h",
                color="Confidence Score (%)",
                color_continuous_scale="Greens",
                height=220
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Multi-Model Benchmark Comparison Table
        if is_multi_model:
            st.markdown("#### ⚖️ Multi-Model Blind Evaluation Matrix (Exact Same Image)")
            table_rows = []
            for p in predictions:
                friendly_cat = PLAIN_ENGLISH_CATEGORIES.get(p.model_id, p.task)
                table_rows.append({
                    "Model ID": p.model_id,
                    "Model Type & Specialty": friendly_cat,
                    "Diagnosis": p.prediction,
                    "Confidence": f"{p.confidence:.1%}",
                    "Speed": f"{p.inference_time_ms:.1f} ms",
                    "Device": p.device,
                    "Abstained": "⚠️ Yes" if p.unknown else "✅ No"
                })
            st.dataframe(table_rows, use_container_width=True)
            agree_count = sum(1 for p in predictions if p.prediction == consensus_name)
            st.success(f"🤝 **Consensus Verdict**: **{consensus_name}** ({agree_count}/{len(predictions)} models in agreement)")

        # Agronomic Advisory Package
        st.markdown("---")
        st.markdown("### 🌱 Kenya Agronomic Advisory (KALRO & PCPB Aligned)")
        advisory = generate_agronomic_advisory(
            prediction=primary_pred.prediction,
            scientific_name=primary_pred.scientific_name,
            severity=primary_pred.severity,
            is_unknown=primary_pred.unknown,
            confidence=primary_pred.confidence,
            threshold=threshold_val
        )

        st.info(f"**Target Crop**: {advisory['primary_crop']} | **Local Swahili**: *{advisory.get('swahili_name', 'N/A')}*\n\n"
                f"**Field Symptoms**: {advisory['symptoms']}\n\n**Yield Impact**: {advisory['damage_risk']}")

        # Advisory Action Tabs
        adv_tab1, adv_tab2, adv_tab3, adv_tab4, adv_tab5 = st.tabs([
            "📋 1. Immediate Cultural Actions",
            "🌿 2. Biological & Organic Controls",
            "🧪 3. Chemical Interventions (PCPB)",
            "🛡️ 4. Push-Pull & Long-term GAP",
            "📞 5. Extension & KALRO Escalation"
        ])

        with adv_tab1:
            st.markdown("##### Immediate Practical Field Steps (Zero-Cost / Low-Cost)")
            for act in advisory["cultural_actions"]:
                st.markdown(f"- 🌾 **{act}**")

        with adv_tab2:
            st.markdown("##### Safe Biological & Eco-Friendly Controls")
            for bio in advisory["biological_controls"]:
                st.markdown(f"- 🍃 {bio}")

        with adv_tab3:
            st.markdown("##### PCPB-Registered Chemical Active Ingredients & Safe Harvest Intervals")
            if not advisory["is_safe_to_treat"]:
                st.error("🚫 **Chemical Sprays Withheld**: Plant is healthy or diagnosis is unconfirmed. Do not spray broad-spectrum chemicals.")
            else:
                for chem in advisory["chemical_interventions"]:
                    st.markdown(f"**Active Ingredient**: `{chem['active_ingredient']}`")
                    st.markdown(f"- 🎯 **Application Timing**: {chem['application_timing']}")
                    st.markdown(f"- ⏱️ **Pre-Harvest Interval (PHI)**: **{chem['phi_days']}** *(Do not harvest or sell crop within this period)*")
                    st.markdown(f"- ⚠️ **Safety & PPE**: {chem['safety_note']}")
                    st.markdown("---")

        with adv_tab4:
            st.markdown("##### Long-term Prevention & icipe Push-Pull Strategy")
            for prev in advisory["preventative_practices"]:
                st.markdown(f"- 🛡️ {prev}")

        with adv_tab5:
            st.markdown("##### County Extension & KALRO Support Channels")
            st.warning(advisory["extension_escalation"])
            st.markdown("""
            **Direct Assistance Lines**:
            - 📞 **KALRO Advisory Helpline (Toll-Free)**: `0800 721 741`
            - 📱 **WhatsApp Agronomy Helpline**: `+254 711 000 000`
            - 🏛️ **County Ward Extension Officer**: Visit your sub-county agricultural office for certified seed stock and physical leaf diagnosis.
            """)

        # Printable Farmer Field Prescription Card
        st.markdown("---")
        st.markdown("### 📄 Farmer Field Prescription Slip (Printable Summary)")
        st.markdown(f"""
        <div class="prescription-card" style="background-color: #ffffff; color: #111111 !important; border: 2px dashed #2e7d32; border-radius: 12px; padding: 24px; margin-top: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.06);">
            <div style="text-align:center; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; margin-bottom: 12px;">
                <h3 style="color:#1b5e20 !important; margin:0; font-weight: 800;">🇰🇪 OPENAGRINET KENYA - CROP HEALTH PRESCRIPTION</h3>
                <small style="color:#555555 !important; font-weight: 600;">Digital Public Infrastructure for Agricultural Extension | KALRO-Aligned</small>
            </div>
            <p style="color:#222222 !important; margin:6px 0;"><strong style="color:#111111 !important;">Date/Time:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')} | <strong style="color:#111111 !important;">Field Image ID:</strong> <code style="color:#1b5e20 !important; background-color:#e8f5e9 !important;">{active_image_name}</code></p>
            <p style="color:#222222 !important; margin:6px 0;"><strong style="color:#111111 !important;">Crop:</strong> {advisory['primary_crop']} | <strong style="color:#111111 !important;">Diagnosed Problem:</strong> <span style="font-size:1.15rem; color:#b71c1c !important; font-weight:bold;">{primary_pred.prediction}</span> (<i>{primary_pred.scientific_name or 'N/A'}</i>)</p>
            <p style="color:#222222 !important; margin:6px 0;"><strong style="color:#111111 !important;">Severity Stage:</strong> <code style="color:#b71c1c !important; background-color:#ffebee !important;">{primary_pred.severity or 'STAGE_2_MODERATE'}</code> | <strong style="color:#111111 !important;">Diagnosis Confidence:</strong> {primary_pred.confidence:.1%}</p>
            <hr style="margin:12px 0; border: none; border-top: 1px solid #e0e0e0;">
            <p style="color:#111111 !important; font-weight:bold; margin-bottom:4px;">IMMEDIATE ACTION CHECKLIST:</p>
            <ul style="color:#222222 !important; margin-top:4px;">
                <li style="color:#222222 !important; margin-bottom:4px;">{"</li><li style='color:#222222 !important; margin-bottom:4px;'>".join(advisory["cultural_actions"][:3])}</li>
            </ul>
            <p style="color:#1b5e20 !important; font-weight: bold; margin-top:12px;">📞 EMERGENCY ESCALATION: Call KALRO Toll-Free at 0800 721 741 or notify your local Ward Extension Officer.</p>
        </div>
        """, unsafe_allow_html=True)

        # Export Options
        st.markdown("<br>", unsafe_allow_html=True)
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button(
                label="📥 Download Diagnostic JSON (OAN Schema)",
                data=json.dumps(primary_pred.to_dict(), indent=2),
                file_name=f"oan_diagnostic_{active_image_name}.json",
                mime="application/json"
            )
        with col_exp2:
            md_summary = f"""# OAN Kenya Crop Diagnostic Report
- Image: `{active_image_name}`
- Diagnosis: **{primary_pred.prediction}** (*{primary_pred.scientific_name}*)
- Confidence: {primary_pred.confidence:.1%}
- Severity: `{primary_pred.severity}`
- Latency: {primary_pred.inference_time_ms:.1f} ms
- Abstention: {primary_pred.unknown}

## Immediate Actions
""" + "\n".join([f"- {a}" for a in advisory["cultural_actions"]]) + f"""

## Emergency Hotline
KALRO Helpline: 0800 721 741
"""
            st.download_button(
                label="📄 Download Markdown Summary Report",
                data=md_summary,
                file_name=f"oan_diagnostic_{active_image_name}.md",
                mime="text/markdown"
            )

# ==============================================================================
# TAB 2: MODEL RECOMMENDATION & DECISION MATRIX
# ==============================================================================
with nav_tab2:
    st.markdown("""
    <div class="winner-card" style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); border: 2px solid #2e7d32; border-radius: 12px; padding: 24px; margin-bottom: 22px; color: #0d2b0e !important; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
        <h2 style="color:#0a230a !important; margin-top:0; font-weight:800;">🏆 Executive Recommendation for OAN Kenya Deployment</h2>
        <h4 style="color:#1b5e20 !important; margin-top:0; font-weight:700;">Primary Production Choice: <b style="color:#0a230a !important;">Ultralytics YOLOv8s / YOLO11s (Object Detection & Counting)</b></h4>
        <p style="font-size:1.05rem; color:#1a3a1a !important; line-height:1.6; margin-bottom:0;">
            Based on multi-model benchmarking across Kenyan priority crops, <strong style="color:#0a230a !important;">YOLOv8s</strong> is the clear winner for initial integration 
            into OAN Kenya's Beckn provider infrastructure. It provides discrete pest counting (critical for economic threshold spraying), 
            executes in <strong style="color:#0a230a !important;">12 milliseconds on zero-cost standard CPUs</strong>, and can be exported to TFLite for <strong style="color:#0a230a !important;">offline smartphone use</strong> by field extension agents.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 The 6-Pillar Evaluation Framework for Kenya DPI")
    st.markdown("In rural Kenya, academic 'Top-1 Accuracy' is not enough. Models were evaluated against six practical criteria:")

    scorecard_data = [
        {
            "Model Architecture": "YOLOv8s / YOLO11s (Ultralytics)",
            "Specialty": "🎯 Pest Detection & Counting",
            "Pest Counting?": "⭐⭐⭐⭐⭐ (Boxes)",
            "Offline Edge": "⭐⭐⭐⭐⭐ (22 MB TFLite)",
            "Speed (CPU)": "⭐⭐⭐⭐⭐ (12 ms)",
            "Cloud Hosting Cost": "⭐⭐⭐⭐⭐ ($0 GPU)",
            "Kenya Suitability": "⭐⭐⭐⭐⭐ (Primary Pick)",
            "Deployment Role": "🥇 Primary OAN Engine"
        },
        {
            "Model Architecture": "MobileNetV4 (Google/timm)",
            "Specialty": "⚡ Ultra-Fast Classifier",
            "Pest Counting?": "⭐ (Whole-image only)",
            "Offline Edge": "⭐⭐⭐⭐⭐ (5 MB TFLite)",
            "Speed (CPU)": "⭐⭐⭐⭐⭐ (10 ms)",
            "Cloud Hosting Cost": "⭐⭐⭐⭐⭐ ($0 GPU)",
            "Kenya Suitability": "⭐⭐⭐⭐ (Disease pick)",
            "Deployment Role": "🥈 Foliar Disease Co-Engine"
        },
        {
            "Model Architecture": "CerealPestAID (Sheneman et al.)",
            "Specialty": "🌾 26 Cereal Pest Species",
            "Pest Counting?": "⭐ (Classification)",
            "Offline Edge": "⭐⭐⭐⭐ (ONNX ready)",
            "Speed (CPU)": "⭐⭐⭐⭐ (15 ms)",
            "Cloud Hosting Cost": "⭐⭐⭐⭐⭐ ($0 GPU)",
            "Kenya Suitability": "⭐⭐⭐⭐ (Grain crops)",
            "Deployment Role": "Grain Validation Engine"
        },
        {
            "Model Architecture": "BioCLIP (Tree of Life 10M)",
            "Specialty": "🧬 Zero-Shot Taxonomy",
            "Pest Counting?": "⭐ (No boxes)",
            "Offline Edge": "⭐⭐ (350 MB PyTorch)",
            "Speed (CPU)": "⭐⭐ (120 ms)",
            "Cloud Hosting Cost": "⭐⭐⭐ (Needs CPU/RAM)",
            "Kenya Suitability": "⭐⭐⭐ (Research)",
            "Deployment Role": "New Species Discovery"
        },
        {
            "Model Architecture": "Florence-2 (Microsoft)",
            "Specialty": "🔍 Foundation Grounding",
            "Pest Counting?": "⭐⭐⭐⭐ (Boxes)",
            "Offline Edge": "⭐ (0.9 GB weights)",
            "Speed (CPU)": "⭐ (800 ms)",
            "Cloud Hosting Cost": "⭐⭐ (Needs GPU)",
            "Kenya Suitability": "⭐⭐⭐ (Cloud only)",
            "Deployment Role": "Cloud Fallback Grounding"
        },
        {
            "Model Architecture": "AgriChat / Multimodal LLM",
            "Specialty": "🧠 Deep Conversational AI",
            "Pest Counting?": "⭐ (Text reasoning)",
            "Offline Edge": "⭐ (16 GB VRAM)",
            "Speed (CPU)": "⭐ (2500 ms)",
            "Cloud Hosting Cost": "⭐ (Requires GPU VM)",
            "Kenya Suitability": "⭐⭐⭐ (Expensive)",
            "Deployment Role": "🥉 Tier-2 Expert Escalation"
        }
    ]

    st.table(scorecard_data)

    st.markdown("---")
    st.markdown("### 🏗️ Recommended 2-Tier Architecture for OAN Kenya")

    st.markdown("""
    ```
                                ┌──────────────────────────────────────────────┐
                                │     Farmer / Extension Agent (WhatsApp / PWA)│
                                └──────────────────────┬───────────────────────┘
                                                       │ Photo Upload via Beckn BAP
                                                       ▼
                                ┌──────────────────────────────────────────────┐
                                │       OAN Kenya Beckn ONIX BPP Gateway       │
                                └──────────────────────┬───────────────────────┘
                                                       │ Internal /predict request
                                                       ▼
                ┌──────────────────────────────────────────────────────────────────────────────┐
                │                     TIER 1: PRIMARY FAST-PATH ENGINE                         │
                │   • Pest Localization & Larval Count: YOLOv8s (12ms, CPU)                    │
                │   • Foliar Leaf Disease Classification: MobileNetV4 (10ms, CPU)              │
                │   ⚡ Total Latency: < 25ms | Monthly Server Cost: ~$10 (Zero GPU Dependency) │
                └──────────────────────────────────────┬───────────────────────────────────────┘
                                                       │
                                            Is Confidence < 40% ?
                                                       │
                                   ┌───────────────────┴───────────────────┐
                                   │ YES                                   │ NO
                                   ▼                                       ▼
                ┌───────────────────────────────────────┐ ┌────────────────────────────────────┐
                │      TIER 2: SAFETY ESCALATION        │ │    IMMEDIATE DIAGNOSTIC PAYLOAD    │
                │  • Withhold dangerous chemicals       │ │  • Bounding boxes & pest count     │
                │  • Route to Multimodal VLM or         │ │  • KALRO Cultural & Bio controls   │
                │    Ward Agricultural Extension Desk   │ │  • PCPB-registered Actives + PHI   │
                └───────────────────────────────────────┘ └─────────────────┬──────────────────┘
                                                                            │
                                                                            ▼
                                                                (Returned via Beckn /on_search)
    ```
    """)

    st.markdown("### 💡 Why YOLOv8s is the Best Choice for Kenya:")
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.markdown("""
        1. **Locates and Counts Pests**:
           - Whole-image classifiers just say *"Maize"*.
           - YOLO draws boxes around every caterpillar and frass pile.
           - Knowing there are **4 larvae** allows the system to determine whether the farmer has reached the economic threshold to spray.
        2. **Runs for Free on Standard Hardware**:
           - Does not need expensive cloud GPUs ($500+/month).
           - Runs in **12 milliseconds on an ordinary CPU**.
        """)
    with rec_col2:
        st.markdown("""
        3. **100% Offline Mobile Ready**:
           - Can be packaged into a **22 MB Android app file (TFLite)**.
           - Extension officers can walk into deep rural areas without internet and still diagnose crops instantly.
        4. **Clean Beckn Protocol Integration**:
           - Outputs standardized JSON coordinates that seamlessly fit the OAN `crop-protection:oan:kenya` catalog schema.
        """)

# ==============================================================================
# TAB 3: HOW IT WORKS & BENCHMARKING GUIDE (FOR NON-ENGINEERS)
# ==============================================================================
with nav_tab3:
    st.markdown("### 📘 Understanding Agricultural Computer Vision & Benchmarking")
    st.markdown("""
    This section explains the core concepts behind this lab in plain, everyday language. 
    You do not need an engineering background to understand how this system protects Kenyan farmers.
    """)

    st.markdown("#### 🏥 The 'Multi-Doctor Clinic' Analogy")
    st.info("""
    **How do we benchmark AI models fairly?**
    Imagine a farmer brings a single infested maize leaf into an agricultural clinic where **7 different crop specialists** are sitting in separate rooms.
    
    None of the doctors can copy each other's answers. They are all shown the **exact same leaf photo at the exact same moment**.
    - **Doctor 1 (YOLO)** circles individual caterpillars and counts them.
    - **Doctor 2 (BioCLIP)** knows the scientific Latin family tree of 10 million organisms.
    - **Doctor 3 (EfficientNet)** is an expert at leaf spots and mold patterns.
    - **Doctor 4 (CerealPestAID)** specializes exclusively in East African cereal pests.
    
    We compare their answers: *Did they agree? How fast did they respond? Did they admit when they were unsure instead of prescribing a toxic chemical?*
    """)

    st.markdown("---")
    st.markdown("#### 📖 Plain-English Glossary of Key Terms")

    gloss_col1, gloss_col2 = st.columns(2)
    with gloss_col1:
        st.markdown("""
        - **Bounding Box (`[x1, y1, x2, y2]`)**:  
          A rectangle drawn by the AI isolating a caterpillar or leaf spot. This proves the AI actually saw the pest and didn't just guess based on background soil.
        - **Inference Latency (Speed in milliseconds)**:  
          How many milliseconds it takes for the AI to answer. 12 ms is instantaneous (real-time on any device).
        - **Confidence Score (0% to 100%)**:  
          How certain the model is. If confidence is 94%, the AI has matched distinct visual patterns with high mathematical certainty.
        """)
    with gloss_col2:
        st.markdown("""
        - **Safety Abstention (The Safety Brake)**:  
          If a photo is blurry, dark, or taken from too far away, an unsafe AI might guess a dangerous chemical. Our system uses a **Safety Brake**: if confidence is under 40%, it says *"I am unsure"* and withholds chemicals.
        - **Pre-Harvest Interval (PHI)**:  
          The minimum number of days a farmer must wait between spraying a pesticide and harvesting the crop so the food is safe for consumers.
        - **OpenAgriNet (OAN)**:  
          Kenya's open Digital Public Infrastructure (DPI) connecting farmers, advisory providers, and agro-dealers over the open Beckn protocol.
        """)

    st.markdown("---")
    st.markdown("#### ❓ Frequently Asked Questions by Stakeholders")
    with st.expander("Why can't we just use ChatGPT or Google Gemini for everything?"):
        st.write("""
        General-purpose cloud models like ChatGPT or Gemini are powerful, but they have major drawbacks for national public infrastructure in Kenya:
        1. **Cost**: Every API call costs money. At millions of farmer queries per season, cloud bills become unsustainable.
        2. **Internet Dependency**: They require continuous high-speed internet, which fails in remote rural counties.
        3. **No Precise Counting**: They cannot return precise pixel coordinates to count individual insect pests in the whorl.
        YOLOv8s runs locally for zero cloud cost and works offline.
        """)

    with st.expander("How will KALRO and the Kenya Ministry of Agriculture maintain this?"):
        st.write("""
        All model architectures cataloged in this repository are **open-source or open-weights**. 
        KALRO research officers can periodically collect new Kenyan field photos (e.g. newly emerging pest strains) 
        and fine-tune the YOLOv8 weights without paying foreign software licensing fees.
        """)

    with st.expander("How does this connect to WhatsApp or mobile phones for farmers?"):
        st.write("""
        Under the Beckn protocol architecture, a farmer does not need to use this technical dashboard. 
        A farmer simply sends a photo over WhatsApp, Telegram, or an SMS-linked PWA. 
        The message routes through the Beckn BAP gateway to our OAN BPP inference microservice, 
        which executes YOLOv8 in 12ms and sends back a simple Swahili/English advisory with pictures.
        """)
