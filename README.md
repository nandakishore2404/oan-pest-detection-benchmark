# OpenAgriNet (OAN) Kenya: AI Pest & Crop Disease Detection Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20ONNX-orange.svg)](https://pytorch.org/)
[![DPI: Beckn Protocol](https://img.shields.io/badge/Protocol-Beckn%20ONIX-blue.svg)](https://docs.openagrinet.global/)
[![Edge: ONNX Runtime](https://img.shields.io/badge/Edge%20Inference-1.99ms%20CPU-brightgreen.svg)](https://onnxruntime.ai/)

An open-source, edge-optimized computer-vision engine and benchmarking laboratory for automated agricultural pest and foliar disease diagnosis, engineered specifically for the **OpenAgriNet (OAN) Kenya Digital Public Infrastructure (DPI)** under the **Beckn Protocol**.

---

## 🌾 Overview & Two-Tier Architecture

Smallholder farmers in Kenya lose up to 70% of seasonal yields to late-detected foliar pathogens and invasive insect pests. This repository provides a sovereign, field-calibrated **Two-Tier Computer Vision Architecture** designed to run offline on sub-$100 Android smartphones or as a high-throughput Beckn Provider Platform (BPP) microservice:

```
                          ┌───────────────────────────┐
                          │   Farmer Smartphone Photo │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │   Image Quality Gate      │
                          │   (Blur & Plant Filter)   │
                          └─────────────┬─────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
    ┌───────────────────────────────┐       ┌───────────────────────────────┐
    │ Tier 1: Foliar Classifier     │       │ Tier 2: Entomology Detector   │
    │ Model: MobileNetV4 Conv Small │       │ Model: Ultralytics YOLOv8s    │
    │ Size: 9.51 MB ONNX            │       │ Size: 11.6 MB ONNX            │
    │ Latency: 1.99 ms (CPU)        │       │ Latency: 15.5 ms (CPU)        │
    │ Task: Whole-Leaf Pathology    │       │ Task: Bounding Box & Counting │
    │ Output: Pathogen ID + Conf    │       │ Output: Pest Density vs. EIL  │
    └───────────────┬───────────────┘       └───────────────┬───────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │ Advisory Decision Engine  │
                          │ (PCPB Regulatory Guidance)│
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ Localized Swahili/English │
                          │ Actionable Farmer Advice  │
                          └───────────────────────────┘
```

* **Tier 1 (Foliar Disease Classifier)**: Powered by `mobilenetv4_conv_small`. Achieves **87.95% top-1 accuracy** with **100% recall on Potato Late Blight** and **1.99 ms CPU inference**. Fine-tuned with genuine East African smallholder field images from Makerere University.
* **Tier 2 (Insect Pest Detector & Counter)**: Powered by `yolov8s`. Detects and counts individual insect pests to compute **Economic Injury Levels (EIL)** for spray action thresholds.

---

## ⚡ Quickstart

### 1. Environment Setup
Clone the repository and install pinned dependencies:
```bash
git clone https://github.com/<your-org>/oan-pest-detection-benchmark.git
cd oan-pest-detection-benchmark
pip install -r requirements.txt
```

### 2. Run Interactive Streamlit UI
Launch the browser-based diagnostic lab and model evaluation portal:
```bash
streamlit run ui/app.py
```
Open your browser at `http://localhost:8501`. You can upload leaf photos, inspect bounding boxes, view temperature-calibrated probabilities, and test all shortlisted candidate models.

### 3. Run Production Beckn / FastAPI Microservice
Launch the high-performance REST API:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation is available interactively at `http://localhost:8000/docs`.

---

## 🚀 Step-by-Step Guide: How the Kenya Team Can Fine-Tune the Model

As extension officers collect new field photographs across Kenyan counties (e.g., Kakamega, Bungoma, Nakuru, Uasin Gishu, Trans Nzoia), follow this workflow to retrain and adapt the model:

### Step 1: Organize Your New Field Images
Place new field images into the structured data directory:
```
data/
└── kenya_field/
    ├── potato_late_blight/
    ├── tomato_early_blight/
    ├── bean_angular_leaf_spot/
    ├── bean_rust/
    ├── fall_armyworm/
    └── healthy_foliage/
```

### Step 2: Run the Fine-Tuning Script
Execute the domain-adaptation fine-tuning pipeline:
```bash
python scripts/train_kenya_finetune.py --epochs 15 --batch-size 16 --lr 0.0001
```
This script automatically:
* Applies domain-adaptive augmentations for equatorial sunlight, smallholder camera angles, and soil backgrounds.
* Uses loss-weighted cross-entropy to handle class imbalances.
* Logs training and validation metrics per epoch.
* Saves updated PyTorch checkpoints to `models/trained/mobilenetv4_kenya_finetuned.pt`.

### Step 3: Export to ONNX for Android / Edge Deployment
Export the fine-tuned model for mobile offline deployment:
```bash
python scripts/export_onnx.py --model-weights models/trained/mobilenetv4_kenya_finetuned.pt
```
This produces `models/trained/mobilenetv4_kenya_finetuned.onnx` (<10 MB), fully compatible with **ONNX Runtime Mobile** and **Google AI Edge LiteRT** on budget Android smartphones.

### Step 4: Run Single-Image or Batch Inference
Test the updated model on any new farmer image:
```bash
python scripts/predict.py --image path/to/sample_leaf.jpg --model models/trained/mobilenetv4_kenya_finetuned.onnx
```

---

## 📦 Pre-Trained Weights Inventory

All production model weights are stored directly in `models/trained/` and are tracked in this repository:

| Model File | Architecture | Task | Size | Latency (CPU) | Description |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `mobilenetv4_kenya_finetuned.onnx` | MobileNetV4 Conv Small | Foliar Classifier | 9.51 MB | **1.99 ms** | **Recommended Tier 1**: Adapted on East African field data. |
| `mobilenetv4_5class.pt` | MobileNetV4 Conv Small | Foliar Classifier | 10.2 MB | 18.1 ms | Base PyTorch weights for fine-tuning. |
| `best.onnx` | YOLOv8n (Ultralytics) | Insect Detector | 11.6 MB | **15.5 ms** | **Recommended Tier 2**: Edge object detector for pest counting. |
| `best.pt` | YOLOv8n (Ultralytics) | Insect Detector | 6.2 MB | 50.0 ms | Base PyTorch weights for YOLO detector fine-tuning. |

---

## 🏛️ OpenAgriNet (OAN) Beckn ONIX Integration

The engine implements standard Beckn action verbs for decentralized DPI interoperability:

| Beckn Action | Endpoint | Purpose |
| :--- | :--- | :--- |
| `search` | `POST /oan/search` | Discovers available crop diagnostic capabilities. |
| `select` | `POST /oan/select` | Selects target crop diagnostic service and uploads image payload. |
| `init` | `POST /oan/init` | Initializes diagnostic session with GPS county/ward coordinates. |
| `confirm` | `POST /oan/confirm` | Executes Two-Tier neural inference; returns PCPB treatment guidance. |
| `status` | `GET /oan/status` | Polls asynchronous human-in-the-loop second opinion. |

---

## ⚖️ Confidence Calibration & PCPB Kenya Regulatory Safety

To prevent misapplication of costly agrochemicals, the model applies temperature scaling ($T=10.0$) and enforces a **Three-Zone Decision Framework**:
* **Zone 1 ($\ge 85\%$ Confidence)**: Automated PCPB-approved advisory (cultural controls or registered biopesticides).
* **Zone 2 ($50\% - 84\%$ Confidence)**: Flagged for asynchronous Ward Extension Officer second opinion.
* **Zone 3 ($< 50\%$ Confidence)**: Explicit Abstention (*"Diagnosis Uncertain — please retake photo under clearer lighting"*).

## 👨‍💻 Author & Project Architect

* **Nanda Kishore Kakulla** — *Lead AI/ML Solution Architect & Core Contributor*
  * Conceptualized and implemented the **Two-Tier Computer Vision Architecture** (separating foliar pathology from insect counting).
  * Led the P0 technical remediation: eradicated silent mock fallbacks, enforced cryptographic SHA-256 byte hashing, and pinned deep learning frameworks.
  * Executed domain-adaptive fine-tuning on East African smallholder field datasets (+17.17% accuracy gain).
  * Optimized sub-2.5ms edge ONNX runtimes for offline budget Android deployment.
  * Architected Beckn Protocol (ONIX) BPP provider integration for the OpenAgriNet (OAN) Kenya DPI exchange.

---

## 📜 Licensing & Citations

* **Codebase**: Licensed under the [MIT License](LICENSE) (c) 2026 Nanda Kishore Kakulla.
* **Foliar Classification Backbone**: `timm` (Apache 2.0).
* **Detection Engine**: `ultralytics` (AGPL-3.0 / Enterprise).
* **East African Field Dataset**: Makerere University AI Lab & NaCRRI iBean Dataset (MIT License).
