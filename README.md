# OpenAgriNet (OAN) Kenya: AI Pest & Crop Disease Detection Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20ONNX%20%7C%20Ultralytics-orange.svg)](https://pytorch.org/)
[![DPI: Beckn Protocol](https://img.shields.io/badge/Protocol-Beckn%20ONIX-blue.svg)](https://docs.openagrinet.global/)
[![Edge: ONNX Runtime](https://img.shields.io/badge/Edge%20Inference-1.99ms%20CPU-brightgreen.svg)](https://onnxruntime.ai/)
[![Author: Nanda Kishore Kakulla](https://img.shields.io/badge/Lead%20Architect-Nanda%20Kishore%20Kakulla-blue.svg)](https://github.com/nandakishore2404)

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
    │ Model: MobileNetV4 Conv Small │       │ Model: YOLOv8 Agricultural    │
    │ Size: 9.51 MB ONNX            │       │ Size: 11.6 MB ONNX            │
    │ Latency: 1.99 ms (CPU)        │       │ Latency: 17.03 ms (CPU)       │
    │ Task: Whole-Leaf Pathology    │       │ Task: Bounding Box & Counting │
    │ Output: Pathogen ID + Conf    │       │ Output: 28 Pest Classes + EIL │
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

* **Tier 1 (Foliar Disease Classifier)**: Powered by `mobilenetv4_conv_small`. Achieves **87.95% top-1 accuracy** on multi-crop pathogens, **69.05% (+17.17% gain)** on unseen East African smallholder field foliage from Makerere University, with **1.99 ms CPU inference**.
* **Tier 2 (Agricultural Pest Detector & Counter)**: Powered by `yolov8_agripests_kenya`. Fine-tuned on **28 agricultural pest species** (including cutworms, stem borers, bollworms, pod borers, and armyworms) achieving **83.19% precision** and **17.03 ms CPU latency (58.7 FPS)**. Detects individual pests, provides bounding boxes, and calculates Economic Injury Levels (EIL) to trigger spray action thresholds.

---

## ⚡ Quickstart

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/nandakishore2404/oan-pest-detection-benchmark.git
cd oan-pest-detection-benchmark
pip install -r requirements.txt
```

### 2. Run Interactive Streamlit Diagnostic Lab
Launch the browser-based diagnostic lab, pest visualizer, and model benchmarking portal:
```bash
streamlit run ui/app.py
```
Open `http://localhost:8501`. You can upload leaf/pest photos, inspect bounding boxes, view temperature-calibrated probabilities, and test candidate architectures.

### 3. Run Pest Detection CLI
Inspect an image for agricultural crop pests with bounding-box localization and automated PCPB regulatory advisory:
```bash
python scripts/detect_pest.py --image data/agricultural_pests_sample/--2022-04-12-00-40-23_png_jpg.rf.97a93a929db21441f1f4d2c4c45f590c.jpg --conf 0.10
```

### 4. Run Foliar Disease Diagnosis CLI
Diagnose foliar blight, rust, or spot diseases and print Swahili/English advisory:
```bash
python scripts/predict.py --image data/african_field_samples/bean_rust_test_sample.jpg --model models/trained/mobilenetv4_kenya_finetuned.onnx
```

### 5. Run Production Beckn / FastAPI Microservice
Launch the high-performance REST API:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation is available interactively at `http://localhost:8000/docs`.

---

## 🔬 Model Benchmarks & Edge Performance

| Model Architecture | Task | Parameters | Precision / Acc | Latency (CPU) | Throughput | Target Deployment |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`mobilenetv4_kenya_finetuned.onnx`** | Tier 1 Foliar Classifier | 3.8M | 87.95% Top-1 (69.05% Field) | **1.99 ms** | 502 FPS | Android Mobile / Offline Edge |
| **`yolov8_agripests_kenya.onnx`** | Tier 2 Pest Detector (28 classes) | 3.0M | **83.19% Precision** | **17.03 ms** | 58.7 FPS | Android / Raspberry Pi / Drone Edge |
| `mobilenetv4_5class.pt` | Foliar Classifier Base | 3.8M | 87.95% Top-1 | 18.1 ms | 55 FPS | Cloud / PyTorch Training |
| `best.onnx` | General Insect Detector | 3.0M | 74.2% mAP50 | 15.5 ms | 64 FPS | Edge Validation Baseline |

### Key African Crop Pests Detected (28-Class Taxonomy)
* **Black Cutworm (*Agrotis ipsilon*)**: Direct seedling cutter attacking maize and bean stands.
* **Stem Borer (*Chilo suppressalis* / *Busseola fusca*)**: Whorl and stem tunneling cereal destroyer.
* **African Bollworm (*Helicoverpa armigera*)**: Major tomato and French bean export pest.
* **Legume Pod Borer (*Maruca testulalis*)**: Destructive webbing borer of cowpea and bean pods.
* **Pink Stem Borer (*Sesamia inferens*)**: Cereal node tunneling and deadheart inducer.
* **Beet Armyworm (*Spodoptera exigua*)**: Voracious defoliator and cereal seedling pest.
* **Brown & White-Backed Planthoppers (*Nilaparvata*, *Sogatella*)**: Vector and sap feeders.
* **Mole Crickets & Crickets (*Gryllotalpidae*, *Gryllidae*)**: Subterranean root chewers.

---

## 🚀 Step-by-Step Guide: How the Kenya Team Can Fine-Tune the Models

As extension officers collect new field photographs across Kenyan counties (e.g., Kakamega, Bungoma, Nakuru, Uasin Gishu, Trans Nzoia), follow these workflows to retrain and adapt the models:

### Option A: Retraining the Foliar Disease Classifier (Tier 1)
```bash
# 1. Organize new leaf images into data/kenya_field/<class_name>/
# 2. Run domain-adaptation training:
python scripts/train_kenya_finetune.py --epochs 15 --batch-size 16 --lr 0.0001

# 3. Export to ONNX for mobile deployment:
python scripts/export_onnx.py --model-weights models/trained/mobilenetv4_kenya_finetuned.pt
```

### Option B: Retraining the YOLOv8 Pest Detector (Tier 2)
```bash
# Run the end-to-end YOLOv8 fine-tuning, validation, and ONNX export pipeline:
python scripts/train_yolov8_agricultural_pests.py
```
This automatically runs 8 epochs on the agricultural pest dataset, validates detection mAP and precision, saves the updated PyTorch weights (`models/trained/yolov8_agripests_kenya.pt`), exports the ONNX runtime model (`models/trained/yolov8_agripests_kenya.onnx`), and benchmarks CPU latency.

---

## 📦 Pre-Trained Weights Inventory

All production model weights are stored directly in `models/trained/` and are tracked in this repository:

| Model File | Format | Size | Description |
| :--- | :---: | :---: | :--- |
| `mobilenetv4_kenya_finetuned.onnx` | ONNX | 9.51 MB | **Tier 1 Production**: MobileNetV4 foliar disease classifier adapted for African field conditions (1.99 ms CPU). |
| `mobilenetv4_kenya_finetuned.pt` | PyTorch | 9.73 MB | PyTorch checkpoint for foliar fine-tuning. |
| `yolov8_agripests_kenya.onnx` | ONNX | 11.59 MB | **Tier 2 Production**: YOLOv8n detector fine-tuned on 28 agricultural crop pest species (17.03 ms CPU). |
| `yolov8_agripests_kenya.pt` | PyTorch | 5.93 MB | PyTorch checkpoint for pest detector fine-tuning. |
| `best.onnx` | ONNX | 11.60 MB | General garden insect detection ONNX baseline. |
| `best.pt` | PyTorch | 6.22 MB | General insect detection PyTorch baseline. |

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

---

## 👨‍💻 Author & Project Architect

* **Nanda Kishore Kakulla** — *Lead AI/ML Solution Architect & Core Contributor*
  * **GitHub Profile**: [https://github.com/nandakishore2404](https://github.com/nandakishore2404)
  * Conceptualized and implemented the sovereign **Two-Tier Computer Vision Architecture** (separating foliar pathology classification from insect detection and counting).
  * Ingested and benchmarked real East African smallholder field datasets (Makerere iBean, Roboflow Agricultural Pests ODinW-RF100).
  * Executed domain-adaptation fine-tuning on `MobileNetV4 Conv Small` (+17.17% accuracy gain on field imagery) and fine-tuned `YOLOv8n` across 28 real agricultural pest species (83.19% precision).
  * Optimized sub-20ms edge ONNX runtimes (1.99 ms foliar classifier, 17.03 ms pest detector) for offline budget Android smartphone deployment.
  * Architected Beckn Protocol (ONIX) BPP provider integration for the OpenAgriNet (OAN) Kenya DPI exchange.

---

## 📜 Licensing & Citations

* **Codebase**: Licensed under the [MIT License](LICENSE) (c) 2026 Nanda Kishore Kakulla.
* **Foliar Classification Backbone**: `timm` (Apache 2.0).
* **Detection Engine**: `ultralytics` (AGPL-3.0 / Enterprise).
* **East African Field Dataset**: Makerere University AI Lab & NaCRRI iBean Dataset (MIT License).
* **Agricultural Pests Dataset**: Roboflow ODinW-RF100 Challenge (CC BY 4.0).
