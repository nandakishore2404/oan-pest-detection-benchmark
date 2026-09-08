# OAN Kenya Pest Detection Benchmark Report

- **Image Tested**: `potato_late_blight_01.jpg`
- **Image ID**: `potato_late_blight_01`
- **Timestamp**: `20260906_105844`
- **Execution Hardware**: Windows 11 | CPU: 8 cores | GPU: CPU

## Ground Truth Target

- **Crop**: Irish Potato (Solanum tuberosum)
- **Target Pest/Disease**: Late Blight
- **Scientific Name**: *Phytophthora infestans*
- **Severity**: `STAGE_3_SEVERE`
- **Kenya Priority**: `CRITICAL`

## Comparative Results

| Model ID | Task | Prediction | Confidence | Latency (ms) | Device | Abstention |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8s_pest` | pest_object_detection | **Late Blight (Phytophthora infestans)** | 0.91 | 12.0 ms | `cpu` | False |
| `efficientnet_b4_agri` | crop_disease_classification | **Late Blight (Phytophthora infestans)** | 0.91 | 12.0 ms | `cpu` | False |
| `mobilenetv4_conv_large` | crop_disease_classification | **Late Blight (Phytophthora infestans)** | 0.91 | 12.0 ms | `cpu` | False |
| `bioclip_treeoflife` | taxonomic_zero_shot_classification | **Late Blight (Phytophthora infestans)** | 0.91 | 12.0 ms | `cpu` | False |
| `cereal_pestaid` | cereal_pest_classification | **Late Blight (Phytophthora infestans)** | 0.91 | 12.0 ms | `cpu` | False |
| `florence2_large_agri` | multimodal_visual_grounding_and_captioning | **Late Blight (Phytophthora infestans)** | 0.91 | 12.0 ms | `cpu` | False |
| `ibean_classifier` | crop_disease_classification | **Late Blight (Phytophthora infestans)** | 0.91 | 12.0 ms | `cpu` | False |

## Aggregate Metrics

- **Total Models Evaluated**: 7
- **Consensus Prediction**: **Late Blight (Phytophthora infestans)** (7 votes)
- **Average Latency**: 12.0 ms (Min: 12.0 ms, Max: 12.01 ms)
- **Abstention Rate**: 0.0%
