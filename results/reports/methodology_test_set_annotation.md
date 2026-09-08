# 🔬 Methodology & Provenance Note: 15-Image / 90-Pest Test Evaluation Benchmark
**Project**: OpenAgriNet (OAN) Kenya — Pest Detection & Surveillance  
**Author**: Program Manager → AI Enthusiast | Turning Ideas into AI-Powered Solutions: Nanda Kishore Kakulla <nandakishore.kakulla9@gmail.com>  
**Repository**: [github.com/nandakishore2404/oan-pest-detection-benchmark](https://github.com/nandakishore2404/oan-pest-detection-benchmark)  
**Date**: September 2026 | **Environment**: Local Sovereign CPU / Drive D:\ & data/benchmark_test_15/

---

## 1. Provenance of the Test Images
The evaluation test set comprises **15 high-resolution agricultural field images** sampled from the held-out test split of the 28-class agricultural pest detection dataset (data/benchmark_test_15/images/ and D:\OAN_Datagricultural_pests_yolo\dataset\images	est).

The dataset captures authentic, unconstrained field conditions:
- Natural smallholder crop foliage (maize, pulses, vegetables) with soil splash, sun glare, and leaf occlusion.
- Multi-pest infestations and variable pest instars (early larvae, adult moths, beetles, borers).

To enable independent verification by any auditor without access to local Drive D:\, the complete set of 15 images, 15 Darknet label files, and data.yaml has been committed directly into the repository at:
[data/benchmark_test_15/](file:///c:/Users/nanda/OneDrive/Desktop/DELOITTE/Agri%20Africa/OAN%20KENYA/oan-pest-detection-benchmark/data/benchmark_test_15)

---

## 2. Ground-Truth Annotation Protocol
- **Format**: Standard Darknet YOLO format: <class_id> <x_center> <y_center> <width> <height> with normalized coordinates in $[0.0, 1.0]$.
- **Total Ground-Truth Pest Instances**: Exactly **90 individual pest bounding boxes** across the 15 images.
- **Taxonomic Coverage**: Key East African crop threats including *Chilo suppressalis* (Asiatic rice/stem borer), *Spodoptera* spp. (Armyworms), *Helicoverpa armigera* (Bollworms), and foliage-chewing chrysomelids.

---

## 3. Micro-Target Classification Criterion
Smallholder field cameras frequently fail to detect tiny nymphs or early instar larvae due to downsampling. In this benchmark:
- An annotation is classified as a **Micro-Target / Micro-Pest** if:
  	ext{Box Area Percentage} = rac{	ext{width} 	imes 	ext{height}}{	ext{Image Width} 	imes 	ext{Image Height}} 	imes 100 < 2.0\%
- **Empirical Breakdown**:
  - Out of 90 ground-truth pest annotations, **73 instances (81.1%)** met the micro-target criterion ($	ext{area} < 2\%$).
  - Only 17 instances (18.9%) were macro-targets ($> 2\%$ frame area).

---

## 4. Empirical Evaluation Results (Honest Ground-Truth Baseline)
Evaluated via [scripts/evaluate_new_datasets_ollama.py](file:///c:/Users/nanda/OneDrive/Desktop/DELOITTE/Agri%20Africa/OAN%20KENYA/oan-pest-detection-benchmark/scripts/evaluate_new_datasets_ollama.py), backed by [
esults/reports/new_dataset_accuracy_ollama.json](file:///c:/Users/nanda/OneDrive/Desktop/DELOITTE/Agri%20Africa/OAN%20KENYA/oan-pest-detection-benchmark/results/reports/new_dataset_accuracy_ollama.json):

| Metric | Baseline YOLOv8s | YOLOv8s + TTA | SAHI Slicing + Bayesian Prior | Empirical Delta / Gain |
| :--- | :---: | :---: | :---: | :---: |
| **True Positives (TP)** | 3 | 13 | 13 | **+10 detections** |
| **False Positives (FP)** | 2 | 10 | 12 | +10 FP (higher sensitivity) |
| **False Negatives (FN)** | 87 | 77 | 77 | -10 missed pests |
| **Precision** | 60.0% | 56.5% | 52.0% | -8.0 pp (recall tradeoff) |
| **Overall Recall** | 3.3% | 14.4% | 14.4% | **+11.1 percentage points** (4.36× relative) |
| **Micro-Pest Recall** | **2.7%** (2/73) | **13.7%** (10/73) | **16.4%** (12/73) | **+13.7 percentage points** (6.07× relative) |
| **Avg Inference Latency** | 46.4 ms | 98.5 ms | 186.5 ms | Edge feasible on CPU |

> [!NOTE]
> **Reporting Standard**: In all subsequent publications and documentation, relative multipliers (e.g. 6.07×) are explicitly accompanied by the absolute percentage points (+13.7 pp, from 2.7% to 16.4%) and the sample size (=15$ images, 90 ground-truth pests, 73 micro-targets).
