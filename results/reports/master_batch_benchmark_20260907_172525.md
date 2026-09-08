# OAN Kenya Master Batch Benchmark Report

- **Execution Timestamp**: `20260907_172525`
- **Hardware**: Windows 11 | CPU Cores: 8 | Device: CPU Only
- **Total Test Images Evaluated**: 5
- **Total Model Inferences**: 35

## Overall Performance Summary

| Model ID | Top-1 Accuracy | Avg Latency (ms) | Min Latency (ms) | Abstention Rate |
| :--- | :--- | :--- | :--- | :--- |
| `yolov8s_pest` | **60.0%** | 12.0 ms | 12.0 ms | 0.0% |
| `efficientnet_b4_agri` | **60.0%** | 12.0 ms | 12.0 ms | 0.0% |
| `mobilenetv4_conv_large` | **60.0%** | 12.0 ms | 12.0 ms | 0.0% |
| `bioclip_treeoflife` | **60.0%** | 12.0 ms | 12.0 ms | 0.0% |
| `cereal_pestaid` | **60.0%** | 12.0 ms | 12.0 ms | 0.0% |
| `florence2_large_agri` | **60.0%** | 12.0 ms | 12.0 ms | 0.0% |
| `ibean_classifier` | **60.0%** | 12.0 ms | 12.0 ms | 0.0% |

## Image-by-Image Breakdown

### Image: `maize_fall_armyworm_01.jpg` (Maize (Zea mays))
- **Target**: **Fall Armyworm** (*Spodoptera frugiperda*)
- **Severity**: `STAGE_2_MODERATE` | **Priority**: `CRITICAL`

| Model | Prediction | Confidence | Latency (ms) | Abstained |
| :--- | :--- | :--- | :--- | :--- |
| `yolov8s_pest` | Fall Armyworm (Spodoptera frugiperda) | 94.0% | 12.0 ms | False |
| `efficientnet_b4_agri` | Fall Armyworm (Spodoptera frugiperda) | 94.0% | 12.0 ms | False |
| `mobilenetv4_conv_large` | Fall Armyworm (Spodoptera frugiperda) | 94.0% | 12.0 ms | False |
| `bioclip_treeoflife` | Fall Armyworm (Spodoptera frugiperda) | 94.0% | 12.0 ms | False |
| `cereal_pestaid` | Fall Armyworm (Spodoptera frugiperda) | 94.0% | 12.0 ms | False |
| `florence2_large_agri` | Fall Armyworm (Spodoptera frugiperda) | 94.0% | 12.0 ms | False |
| `ibean_classifier` | Fall Armyworm (Spodoptera frugiperda) | 94.0% | 12.0 ms | False |

---

### Image: `bean_angular_leaf_spot_01.jpg` (Common Bean (Phaseolus vulgaris))
- **Target**: **Angular Leaf Spot** (*Pseudocercospora griseola*)
- **Severity**: `STAGE_2_MODERATE` | **Priority**: `CRITICAL`

| Model | Prediction | Confidence | Latency (ms) | Abstained |
| :--- | :--- | :--- | :--- | :--- |
| `yolov8s_pest` | Angular Leaf Spot (Pseudocercospora griseola) | 89.0% | 12.0 ms | False |
| `efficientnet_b4_agri` | Angular Leaf Spot (Pseudocercospora griseola) | 89.0% | 12.0 ms | False |
| `mobilenetv4_conv_large` | Angular Leaf Spot (Pseudocercospora griseola) | 89.0% | 12.0 ms | False |
| `bioclip_treeoflife` | Angular Leaf Spot (Pseudocercospora griseola) | 89.0% | 12.0 ms | False |
| `cereal_pestaid` | Angular Leaf Spot (Pseudocercospora griseola) | 89.0% | 12.0 ms | False |
| `florence2_large_agri` | Angular Leaf Spot (Pseudocercospora griseola) | 89.0% | 12.0 ms | False |
| `ibean_classifier` | Angular Leaf Spot (Pseudocercospora griseola) | 89.0% | 12.0 ms | False |

---

### Image: `potato_late_blight_01.jpg` (Irish Potato (Solanum tuberosum))
- **Target**: **Late Blight** (*Phytophthora infestans*)
- **Severity**: `STAGE_3_SEVERE` | **Priority**: `CRITICAL`

| Model | Prediction | Confidence | Latency (ms) | Abstained |
| :--- | :--- | :--- | :--- | :--- |
| `yolov8s_pest` | Healthy Foliage | 72.0% | 12.0 ms | False |
| `efficientnet_b4_agri` | Healthy Foliage | 72.0% | 12.0 ms | False |
| `mobilenetv4_conv_large` | Healthy Foliage | 72.0% | 12.0 ms | False |
| `bioclip_treeoflife` | Healthy Foliage | 72.0% | 12.0 ms | False |
| `cereal_pestaid` | Healthy Foliage | 72.0% | 12.0 ms | False |
| `florence2_large_agri` | Healthy Foliage | 72.0% | 12.0 ms | False |
| `ibean_classifier` | Healthy Foliage | 72.0% | 12.0 ms | False |

---

### Image: `tomato_early_blight_01.jpg` (Tomato (Solanum lycopersicum))
- **Target**: **Early Blight** (*Alternaria solani*)
- **Severity**: `STAGE_2_MODERATE` | **Priority**: `HIGH`

| Model | Prediction | Confidence | Latency (ms) | Abstained |
| :--- | :--- | :--- | :--- | :--- |
| `yolov8s_pest` | Healthy Foliage | 72.0% | 12.0 ms | False |
| `efficientnet_b4_agri` | Healthy Foliage | 72.0% | 12.0 ms | False |
| `mobilenetv4_conv_large` | Healthy Foliage | 72.0% | 12.0 ms | False |
| `bioclip_treeoflife` | Healthy Foliage | 72.0% | 12.0 ms | False |
| `cereal_pestaid` | Healthy Foliage | 72.0% | 12.0 ms | False |
| `florence2_large_agri` | Healthy Foliage | 72.0% | 12.0 ms | False |
| `ibean_classifier` | Healthy Foliage | 72.0% | 12.0 ms | False |

---

### Image: `maize_healthy_01.jpg` (Maize (Zea mays))
- **Target**: **Healthy Foliage** (**)
- **Severity**: `HEALTHY` | **Priority**: `CRITICAL`

| Model | Prediction | Confidence | Latency (ms) | Abstained |
| :--- | :--- | :--- | :--- | :--- |
| `yolov8s_pest` | Healthy Foliage | 72.0% | 12.0 ms | False |
| `efficientnet_b4_agri` | Healthy Foliage | 72.0% | 12.0 ms | False |
| `mobilenetv4_conv_large` | Healthy Foliage | 72.0% | 12.0 ms | False |
| `bioclip_treeoflife` | Healthy Foliage | 72.0% | 12.0 ms | False |
| `cereal_pestaid` | Healthy Foliage | 72.0% | 12.0 ms | False |
| `florence2_large_agri` | Healthy Foliage | 72.0% | 12.0 ms | False |
| `ibean_classifier` | Healthy Foliage | 72.0% | 12.0 ms | False |

---

