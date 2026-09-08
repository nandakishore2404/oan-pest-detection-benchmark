# Model & Dataset Licensing Compliance Matrix
*OpenAgriNet (OAN) Kenya DPI Evaluation*

In strict adherence to Section 31 of the OAN Kenya Specification, every model architecture, pretrained checkpoint, and evaluation dataset was audited for commercial and public digital public infrastructure (DPI) legality.

---

## 🚦 Commercial Usability Classification (Traffic Light System)

* **🟢 GREEN**: Fully open-source, permissive commercial license (Apache-2.0, MIT, BSD). Usable in OAN Kenya public production without royalty or proprietary restriction.
* **🟡 YELLOW**: Weak copyleft, attribution-required, or dual-license (CC-BY 4.0, AGPLv3, LGPL). Usable with strict architectural decoupling and attribution headers.
* **🔴 RED**: Non-commercial only (CC-BY-NC 4.0, research-only weights, unverified GitHub repos). **Strictly prohibited** from production field deployment.

---

## 📋 Model Architecture & Weight Audit

| Model ID | Architecture / Source | Code License | Pretrained Weight License | OAN Status | Compliance Conditions |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `mobilenetv4_conv_small` | Google MobileNetV4 (`timm`) | Apache-2.0 | Apache-2.0 | **🟢 GREEN** | Fully permissive. Primary pick for offline edge mobile app. |
| `mobilenetv4_conv_large` | Google MobileNetV4 (`timm`) | Apache-2.0 | Apache-2.0 | **🟢 GREEN** | Fully permissive. High-accuracy mobile classifier. |
| `efficientnet_b4_agri` | EfficientNet-B4 (`timm`) | Apache-2.0 | Apache-2.0 | **🟢 GREEN** | Fully permissive for cloud or server inference. |
| `yolov8n_pest` / `yolov8s` | Ultralytics YOLOv8 | AGPL-3.0 / Enterprise | Open-Weight | **🟡 YELLOW** | AGPL requires microservice isolation (REST/FastAPI) so OAN core does not inherit AGPL. Enterprise license required if statically linked in closed-source app. |
| `bioclip_treeoflife` | ViT-B/16 (TreeOfLife-10M) | MIT | CC0 / MIT | **🟢 GREEN** | Permissive taxonomic zero-shot embedding model. |
| `florence2_large_agri` | Microsoft Florence-2 | MIT | MIT | **🟢 GREEN** | Fully permissive open-weight vision-language model. |
| `cereal_pestaid` | EfficientNet-B6 / ICIPE | MIT | CC-BY-NC 4.0 | **🔴 RED** | Original checkpoint is non-commercial research; must be retrained from scratch on OAN golden data for production. |
| `gemini_1_5_pro` | Google DeepMind (Cloud API) | Proprietary | Commercial SaaS | **🟡 YELLOW** | Commercial API; requires paid billing and user privacy consent under Kenya Data Protection Act (2019). |

---

## 🌾 Dataset Licensing Audit

| Dataset Name | Source / Provider | License Type | OAN Status | Permitted Operational Role |
| :--- | :--- | :---: | :---: | :--- |
| **PlantVillage** | Penn State / CGIAR | CC-BY-NC-SA 3.0 | **🟡 YELLOW** | Permitted for research and academic pre-training; non-commercial share-alike restricts direct proprietary bundling. |
| **Makerere iBean Dataset** | Makerere University Uganda | CC-BY-SA 4.0 | **🟢 GREEN** | Permitted with attribution to Makerere AI Lab. |
| **SPRINT-3 archive (1)** | Kaggle Public Insects | Apache-2.0 / Public | **🟢 GREEN** | Permitted for generic entomology pipeline testing. |
| **OAN Kenya Golden Dataset** | Proposed Extension Officer Pilot | OAN Open DPI Data | **🟢 GREEN** | Built from scratch under OAN ownership and Creative Commons CC-BY 4.0. |
