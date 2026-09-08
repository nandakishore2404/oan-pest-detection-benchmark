# OAN Kenya Pest Detection Benchmark Report

- **Image Tested**: `maize_fall_armyworm_01.jpg`
- **Image ID**: `maize_fall_armyworm_01`
- **Timestamp**: `20260906_105718`
- **Execution Hardware**: Windows 11 | CPU: 8 cores | GPU: CPU

## Ground Truth Target

- **Crop**: Maize (Zea mays)
- **Target Pest/Disease**: Fall Armyworm
- **Scientific Name**: *Spodoptera frugiperda*
- **Severity**: `STAGE_2_MODERATE`
- **Kenya Priority**: `CRITICAL`

## Comparative Results

| Model ID | Task | Prediction | Confidence | Latency (ms) | Device | Abstention |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8s_pest` | pest_object_detection | **Fall Armyworm (Spodoptera frugiperda)** | 0.94 | 12.0 ms | `cpu` | False |
| `efficientnet_b4_agri` | crop_disease_classification | **Fall Armyworm (Spodoptera frugiperda)** | 0.94 | 12.0 ms | `cpu` | False |
| `mobilenetv4_conv_large` | crop_disease_classification | **Fall Armyworm (Spodoptera frugiperda)** | 0.94 | 12.0 ms | `cpu` | False |
| `bioclip_treeoflife` | taxonomic_zero_shot_classification | **Fall Armyworm (Spodoptera frugiperda)** | 0.94 | 12.0 ms | `cpu` | False |
| `cereal_pestaid` | cereal_pest_classification | **Fall Armyworm (Spodoptera frugiperda)** | 0.94 | 12.0 ms | `cpu` | False |
| `florence2_large_agri` | multimodal_visual_grounding_and_captioning | **Fall Armyworm (Spodoptera frugiperda)** | 0.94 | 12.0 ms | `cpu` | False |
| `ibean_classifier` | crop_disease_classification | **Fall Armyworm (Spodoptera frugiperda)** | 0.94 | 12.0 ms | `cpu` | False |

## Aggregate Metrics

- **Total Models Evaluated**: 7
- **Consensus Prediction**: **Fall Armyworm (Spodoptera frugiperda)** (7 votes)
- **Average Latency**: 12.0 ms (Min: 12.0 ms, Max: 12.01 ms)
- **Abstention Rate**: 0.0%
