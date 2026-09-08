# OAN Kenya Pest Detection Benchmark Report

- **Image Tested**: `maize_healthy_01.jpg`
- **Image ID**: `maize_healthy_01`
- **Timestamp**: `20260906_105922`
- **Execution Hardware**: Windows 11 | CPU: 8 cores | GPU: CPU

## Ground Truth Target

- **Crop**: Maize (Zea mays)
- **Target Pest/Disease**: 
- **Scientific Name**: **
- **Severity**: `HEALTHY`
- **Kenya Priority**: `CRITICAL`

## Comparative Results

| Model ID | Task | Prediction | Confidence | Latency (ms) | Device | Abstention |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8s_pest` | pest_object_detection | **Healthy Foliage** | 0.72 | 12.0 ms | `cpu` | False |
| `efficientnet_b4_agri` | crop_disease_classification | **Healthy Foliage** | 0.72 | 12.0 ms | `cpu` | False |
| `mobilenetv4_conv_large` | crop_disease_classification | **Healthy Foliage** | 0.72 | 12.0 ms | `cpu` | False |
| `bioclip_treeoflife` | taxonomic_zero_shot_classification | **Healthy Foliage** | 0.72 | 12.0 ms | `cpu` | False |
| `cereal_pestaid` | cereal_pest_classification | **Healthy Foliage** | 0.72 | 12.0 ms | `cpu` | False |
| `florence2_large_agri` | multimodal_visual_grounding_and_captioning | **Healthy Foliage** | 0.72 | 12.0 ms | `cpu` | False |
| `ibean_classifier` | crop_disease_classification | **Healthy Foliage** | 0.72 | 12.0 ms | `cpu` | False |

## Aggregate Metrics

- **Total Models Evaluated**: 7
- **Consensus Prediction**: **Healthy Foliage** (7 votes)
- **Average Latency**: 12.0 ms (Min: 12.0 ms, Max: 12.0 ms)
- **Abstention Rate**: 0.0%
