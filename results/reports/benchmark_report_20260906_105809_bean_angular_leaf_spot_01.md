# OAN Kenya Pest Detection Benchmark Report

- **Image Tested**: `bean_angular_leaf_spot_01.jpg`
- **Image ID**: `bean_angular_leaf_spot_01`
- **Timestamp**: `20260906_105809`
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
| `yolov8s_pest` | pest_object_detection | **Unknown / Unsupported Class** | 0.89 | 12.0 ms | `cpu` | True |
| `efficientnet_b4_agri` | crop_disease_classification | **Unknown / Unsupported Class** | 0.89 | 12.0 ms | `cpu` | True |
| `mobilenetv4_conv_large` | crop_disease_classification | **Unknown / Unsupported Class** | 0.89 | 12.0 ms | `cpu` | True |
| `bioclip_treeoflife` | taxonomic_zero_shot_classification | **Unknown / Unsupported Class** | 0.89 | 12.0 ms | `cpu` | True |
| `cereal_pestaid` | cereal_pest_classification | **Unknown / Unsupported Class** | 0.89 | 12.0 ms | `cpu` | True |
| `florence2_large_agri` | multimodal_visual_grounding_and_captioning | **Unknown / Unsupported Class** | 0.89 | 12.0 ms | `cpu` | True |
| `ibean_classifier` | crop_disease_classification | **Unknown / Unsupported Class** | 0.89 | 12.0 ms | `cpu` | True |

## Aggregate Metrics

- **Total Models Evaluated**: 7
- **Consensus Prediction**: **None** (0 votes)
- **Average Latency**: 12.0 ms (Min: 12.0 ms, Max: 12.01 ms)
- **Abstention Rate**: 100.0%
