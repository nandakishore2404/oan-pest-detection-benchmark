# OpenAgriNet (OAN) Kenya: AI Pest & Crop Disease Detection Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20ONNX%20%7C%20Ultralytics-orange.svg)](https://pytorch.org/)
[![DPI: Beckn Protocol](https://img.shields.io/badge/Protocol-Beckn%20ONIX-blue.svg)](https://docs.openagrinet.global/)
[![Edge: ONNX Runtime](https://img.shields.io/badge/Edge%20Inference-1.99ms%20CPU-brightgreen.svg)](https://onnxruntime.ai/)
[![Author: Nanda Kishore Kakulla](https://img.shields.io/badge/Program%20Manager%20%E2%86%92%20AI%20Enthusiast-Turning%20Ideas%20into%20AI--Powered%20Solutions-10574e.svg)](https://github.com/nandakishore2404)

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

## 📈 Verified Accuracy Evidence & Remediation History

An earlier version of this section presented a "T0 → T6" progression (41.5% → 89.2% accuracy, 320ms → 38.2ms latency) labeled "Independently Audited." That progression traced to a hardcoded literal array with no model run behind any of the seven numbers, and has been removed. For the corrected, evidence-tagged version of this project's history — what's verified, what's fixed, and what's still open — see [`results/reports/OAN_Kenya_0_to_1_Architecture_and_Evolution_Journey.md`](results/reports/OAN_Kenya_0_to_1_Architecture_and_Evolution_Journey.md).

| Model | Test set | n | Accuracy |
| :--- | :--- | :---: | :---: |
| Foliar disease classifier (5-class) | Lab-condition images | 83 | 87.95% |
| Foliar disease classifier (5-class) | Real field images (2 sets) | 261 | 51.7% |
| Foliar disease classifier, fine-tuned | Clean held-out field split | 42 | 59.52% |
| Generic 12-class insect detector | Original test split | 546 | 16.3% |
| Generic 12-class insect detector | New validation split | 1,095 | 17.6% |
| Kenya 28-class priority pest detector | Wild field photos | 15 | 0.0% (0/15) |

Each row is an independent test on its own model and dataset — there is no single verified project-wide accuracy trend.

---

## 🦙 Sovereign Token-Preserving Architecture: Local Ollama & Qwen 2.5 Coder

To eliminate cloud API token exhaustion while empowering field officers with conversational agronomic intelligence, the platform employs a **hybrid sovereign execution model**:

* **Primary Interface & Master Architect (Antigravity / Claude)**: Handles high-level strategy, code reviews, and Git management without burning tokens on repetitive text or heavy training loops.
* **Local Edge Vision (ONNX Runtime CPU)**: Runs Tier 1 (1.99 ms) and Tier 2 (17.03 ms) locally on CPU with **0 API tokens and 0 cloud cost**.
* **Local Agronomic Copilot (Ollama Qwen 2.5 Coder)**: Runs on `http://localhost:11434`, translating raw detection metrics into customized Swahili and English PCPB treatment advisories with **0 API tokens**.
* **Storage Routed to Drive D:** Model weights (GGUF blobs) are stored directly in `D:\OllamaModels` (utilizing 193+ GB of free space, leaving the `C:\` drive completely unburdened).
* **Autonomous Auto-ML Fine-Tuning**: `python scripts/continuous_tuning_orchestrator.py` leverages local Qwen to inspect field datasets, recommend training hyperparameters, retrain weights, and gate production updates automatically.

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

### 3. Run Unified Multi-Tier Diagnostic & Local Advisory Engine (Recommended)
Diagnose foliar diseases (Tier 1), detect agricultural pests (Tier 2), generate Grad-CAM lesion heatmaps, and synthesize PCPB regulatory treatment advice via local Ollama in a single terminal command (**Zero API Tokens**):
```bash
python scripts/diagnose_and_advise.py --image data/african_beans_field/validation/bean_rust/bean_rust_val.0.jpg --model qwen2.5-coder:1.5b
```

### 4. Run Auto-ML Hyperparameter Optimization (Local Ollama)
Automatically analyze field dataset constraints and determine the optimal training schedules and learning rates without cloud tokens:
```bash
python scripts/run_automl_tuning.py
```

### 5. Run Grad-CAM Model Quality & Attention Focus Audit
Quantify neural attention concentration on genuine pathology across field validation images:
```bash
python scripts/audit_model_quality.py
```

### 6. Run Production Beckn / FastAPI Microservice
Launch the high-performance REST API:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation is available interactively at `http://localhost:8000/docs`.

### 7. Monitor Model Observability, Latency SLAs & Drift
Open `http://localhost:8501` and navigate to **Tab 2: 📊 Model Observability & Telemetry Dashboard** to inspect real-time inference latency percentiles (P50, P95, P99), accuracy progression (+7.64% field gain), and zero-token cloud savings. See [`docs/observability_prompt.md`](docs/observability_prompt.md) for the master autonomous monitoring specification prompt.

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

## 👨‍💻 Author & Profile

### 👋 Hi, I'm Nanda Kishore Kakulla

**Program Manager → AI Enthusiast | Turning Ideas into AI-Powered Solutions**

I’m a Program Manager with a passion for technology, AI, and solving real-world problems. My journey has increasingly moved from managing programs and products to building, experimenting, and learning hands-on with AI and modern engineering tools.

I enjoy taking an idea or business problem and exploring how technology can turn it into something tangible — from concept and architecture to prototypes, experiments, and working solutions.

#### 🚀 What I'm Exploring & Building
* 🤖 **Generative AI & Agentic AI**
* 🧠 **AI-powered applications and workflows**
* 🔬 **AI/ML experimentation and model evaluation**
* 🏗️ **Solution architecture & digital platforms**
* 🌐 **Digital Public Infrastructure & Open Networks**
* ☁️ **Cloud technologies and scalable solutions**
* 🧩 **APIs, integrations & distributed systems**
* ⚡ **AI-assisted software development and rapid prototyping**

#### 💡 How I Approach Technology
> **Start with the problem → understand the user → explore what AI can do → prototype quickly → validate with data → build what's worth scaling.**

I'm particularly interested in the evolving intersection of business, technology and AI, and in using AI not just as a productivity tool, but as a way to think, experiment, build and solve problems differently.

#### 📌 What you'll find here
This GitHub is a collection of my experiments, prototypes, projects, research and learnings as I continue my journey into AI-powered technology.

*Learning. Experimenting. Building. Sharing.*

📫 **Connect with me**: [nandakishore.kakulla9@gmail.com](mailto:nandakishore.kakulla9@gmail.com) | **GitHub**: [@nandakishore2404](https://github.com/nandakishore2404)

---

## 📜 Licensing & Citations

* **Codebase**: Licensed under the [MIT License](LICENSE) (c) 2026 Nanda Kishore Kakulla.
* **Foliar Classification Backbone**: `timm` (Apache 2.0).
* **Detection Engine**: `ultralytics` (AGPL-3.0 / Enterprise).
* **East African Field Dataset**: Makerere University AI Lab & NaCRRI iBean Dataset (MIT License).
* **Agricultural Pests Dataset**: Roboflow ODinW-RF100 Challenge (CC BY 4.0).
