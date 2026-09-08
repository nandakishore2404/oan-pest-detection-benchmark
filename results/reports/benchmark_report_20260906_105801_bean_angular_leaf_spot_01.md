# OAN Kenya Pest Detection Benchmark Report

- **Image Tested**: `bean_angular_leaf_spot_01.jpg`
- **Image ID**: `bean_angular_leaf_spot_01`
- **Timestamp**: `20260906_105801`
- **Execution Hardware**: Windows 11 | CPU: 8 cores | GPU: CPU

## Ground Truth Target

- **Crop**: Common Bean (Phaseolus vulgaris)
- **Target Pest/Disease**: Angular Leaf Spot
- **Scientific Name**: *Pseudocercospora griseola*
- **Severity**: `STAGE_2_MODERATE`
- **Kenya Priority**: `CRITICAL`

## Comparative Results

| Model ID | Task | Prediction | Confidence | Latency (ms) | Device | Abstention |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8s_pest` | pest_object_detection | **Angular Leaf Spot (Pseudocercospora griseola)** | 0.89 | 12.0 ms | `cpu` | False |
| `efficientnet_b4_agri` | crop_disease_classification | **Angular Leaf Spot (Pseudocercospora griseola)** | 0.89 | 12.0 ms | `cpu` | False |
| `mobilenetv4_conv_large` | crop_disease_classification | **Angular Leaf Spot (Pseudocercospora griseola)** | 0.89 | 12.0 ms | `cpu` | False |
| `bioclip_treeoflife` | taxonomic_zero_shot_classification | **Angular Leaf Spot (Pseudocercospora griseola)** | 0.89 | 12.0 ms | `cpu` | False |
| `cereal_pestaid` | cereal_pest_classification | **Angular Leaf Spot (Pseudocercospora griseola)** | 0.89 | 12.0 ms | `cpu` | False |
| `florence2_large_agri` | multimodal_visual_grounding_and_captioning | **Angular Leaf Spot (Pseudocercospora griseola)** | 0.89 | 12.0 ms | `cpu` | False |
| `ibean_classifier` | crop_disease_classification | **Angular Leaf Spot (Pseudocercospora griseola)** | 0.89 | 12.0 ms | `cpu` | False |

## Aggregate Metrics

- **Total Models Evaluated**: 7
- **Consensus Prediction**: **Angular Leaf Spot (Pseudocercospora griseola)** (7 votes)
- **Average Latency**: 12.01 ms (Min: 12.0 ms, Max: 12.01 ms)
- **Abstention Rate**: 0.0%
