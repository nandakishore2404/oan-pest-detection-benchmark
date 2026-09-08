# OAN Kenya: Public Agricultural Dataset Benchmark Report

**Dataset Evaluated**: Makerere University AI Lab: East Africa Beans Dataset  
**Source Directory**: `C:\Users\nanda\OneDrive\Desktop\DELOITTE\Agri Africa\OAN KENYA\oan-pest-detection-benchmark\data\external_public\makerere_beans\test`  
**Classes**: angular_leaf_spot, bean_rust, healthy  
**Execution Host**: Windows 11 | CPU Cores: 8 | Python 3.13.5  

## Empirical Performance Matrix

| Model Architecture | Accuracy | Abstention Rate | Avg CPU Latency | Architectural Role in Kenya DPI |
| :--- | :---: | :---: | :---: | :--- |
| `yolov8s_pest` | **16.7%** | 0.0% | 667.8 ms | INSECT DETECTOR (Safely ignores foliar diseases) |
| `efficientnet_b4_agri` | **33.3%** | 0.0% | 43.4 ms | BASELINE/RESEARCH (2/5) |
| `mobilenetv4_conv_large` | **33.3%** | 0.0% | 37.4 ms | FOLIAR TRIAGE CO-ENGINE (4/5) |
| `bioclip_treeoflife` | **33.3%** | 0.0% | 38.8 ms | BASELINE/RESEARCH (2/5) |
| `cereal_pestaid` | **33.3%** | 0.0% | 35.5 ms | CEREAL SPECIFIC (Safely abstains on legumes) |
| `florence2_large_agri` | **33.3%** | 0.0% | 37.8 ms | BASELINE/RESEARCH (2/5) |
| `ibean_classifier` | **33.3%** | 0.0% | 34.1 ms | EAST AFRICA LEGUME WINNER (5/5) |
