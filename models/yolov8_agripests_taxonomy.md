# YOLOv8 Agricultural Pests (28 Classes) — Authoritative Taxonomy & Mapping Specification

**Model Artifacts**:
* ONNX Export: models/trained/yolov8_agripests_kenya.onnx (Input: [1, 3, 384, 384], Output: [1, 32, 3024])
* PyTorch Checkpoint: models/trained/yolov8_agripests_kenya.pt
* Architecture: YOLOv8s (Anchor-Free, Decoupled Head, 4 bounding box coordinates + 28 class probability channels)

---

## 1. Authoritative Class Index Mapping

Extracted directly from the ONNX protobuf metadata property 
ames and verified against model.names in the PyTorch weights binary:

| Class ID | Scientific / Taxon Name | Common Agricultural Description | Primary Host Crops |
| :---: | :--- | :--- | :--- |
| **0** | *Agrotis* spp. | Cutworm larvae | Seedlings, Maize, Vegetables |
| **1** | *Athetis lepigone* | Lepigone armyworm / noctuid moth | Maize stems & ears |
| **2** | *Athetis lineosa* | Grass noctuid moth | Cereals & forage grasses |
| **3** | *Chilo suppressalis* | Asiatic / Striped rice stem borer | Rice, Sorghum, Maize |
| **4** | *Cnaphalocrocis medinalis* | Rice leaf folder | Rice, Cereals |
| **5** | *Creatonotus transiens* | Tiger moth caterpillar | Polyphagous foliage feeder |
| **6** | *Diaphania indica* | Cucumber / Tomato leaf miner moth | Cucurbits, Tomato |
| **7** | *Endotricha consocia* | Pyralid foliage moth | Polyphagous agricultural shrubs |
| **8** | *Euproctis sparsa* | Tussock moth caterpillar | Orchard & field foliage |
| **9** | *Gryllidae* | Field crickets | Seedlings, Ground foliage |
| **10** | *Gryllotalpidae* | Mole crickets | Subterranean roots & tubers |
| **11** | *Helicoverpa armigera* | African bollworm / Corn earworm | Maize cobs, Tomato, Beans, Cotton |
| **12** | *Holotrichia oblita* | White grub / scarab beetle | Roots, Tubers, Maize roots |
| **13** | *Loxostege sticticalis* | Beet webworm | Legumes, Vegetables |
| **14** | *Mamestra brassicae* | Cabbage moth | Cabbage, Brassicas, Legumes |
| **15** | *Maruca testulalis* | Legume pod borer (*Maruca vitrata*) | Common beans, Cowpeas, Pigeon peas |
| **16** | *Mythimna separata* | Oriental armyworm | Maize, Sorghum, Cereals |
| **17** | *Naranga aenescens* | Green rice caterpillar | Cereals, Grasses |
| **18** | *Nilaparvata* spp. | Brown planthopper | Rice, Cereal vascular tissue |
| **19** | *Paracymoriza taiwanalis* | Aquatic crambid moth | Wetland / riparian crops |
| **20** | *Sesamia inferens* | Pink stem borer | Maize, Sorghum, Sugarcane |
| **21** | *Sirthenea flavipes* | Assassin bug (Beneficial predator) | Natural insect predator |
| **22** | *Sogatella furcifera* | White-backed planthopper | Cereals, Sorghum, Grasses |
| **23** | *Spodoptera exigua* | Beet armyworm | Polyphagous foliar feeder |
| **24** | *Spoladea recurvalis* | Hawaiian beet webworm | Amaranth, Spinach, Legumes |
| **25** | *Staurophora celsia* | Malachite moth | Field grasses & forage |
| **26** | *Timandra recompta* | Blood-vein geometer moth | Field weeds, Polygonaceae |
| **27** | *Trichoptera* | Caddisfly (Aquatic bio-indicator) | Irrigation canals & waterways |

---

## 2. Audit Trail & Dataset Reproducibility Note

1. **Class Mapping Provenance**:
   * Previously marked as an unknown mapping by early audit passes because external training configs (data.yaml) on secondary drive D:\ were offline.
   * Fully recovered and validated in Sprint 3 directly from binary metadata inspection (onnx.load() and ultralytics.YOLO()).
2. **Dataset Availability**:
   * The complete 717-image training dataset (D:\OAN_Data\agricultural_pests_yolo\dataset) referenced in 
uns/agricultural_pests/yolov8_agripests_train/args.yaml resides on an external storage mount.
   * To enable offline regression testing in this repository, a curated 15-sample test set is packaged under data/benchmark_test_15/ with matching Darknet format labels.
   * An independent audit on data/benchmark_test_15/ scored **0/15 (0.0%)** top-1 classification accuracy, indicating that the model\'s confidence distribution on unseen wild field photos requires higher sampling scale and calibration before field reliance.
