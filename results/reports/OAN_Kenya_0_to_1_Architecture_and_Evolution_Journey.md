# OpenAgriNet (OAN) Kenya — Corrected Architecture & Evolution Journey
**Initiative-by-initiative accuracy and response-time impact, independently verified**

* **Project**: OpenAgriNet (OAN) Kenya Digital Public Infrastructure (DPI)
* **Client / Stakeholder**: Deloitte Agri Africa / Government of Kenya / KALRO
* **Status**: Production Edge Deployment (Sprint 3)
* **This document replaces**: the prior version of `OAN_Kenya_0_to_1_Architecture_and_Evolution_Journey.md`
* **Why it was replaced**: the previous version's T0→T6 accuracy progression (41.5% → 58.4% → 72.1% → 79.8% → 84.6% → 89.2%) is the exact sequence that was hardcoded as a literal array (`timeline_accuracy = [...]`) in `ui/app.py` with no model run behind any of the seven numbers — confirmed removed from the codebase this session. The same version also states two different, contradictory numbers for the same "small-pest recall" metric (16.4% in the chart header vs. 85.1% in the milestone table) and a "89.2% Foliar Precision" figure that does not match any file in the repository. This version keeps only what can be traced to a real, computed source, tags everything by evidence status, and adds the fixes made in this session.

**Evidence tags used throughout**: **[VERIFIED]** independently computed/reproduced from primary data · **[REPO-SOURCED]** taken from a specific file already in the repo, not independently re-run · **[NOT VERIFIED]** appears in prior narrative reports/UI with no computation behind it found anywhere · **[FIXED THIS SESSION]** a defect found and corrected today, with before/after confirmed live.

---

## 1. What can actually be shown as an accuracy trend

There is **no verified accuracy trend across the full project history** — no set of comparable before/after numbers exists for "Day 0" through "Sprint 2" on a consistent dataset and methodology. What *can* be shown, verified, is the trend on the two models that were independently, repeatedly tested this engagement:

| Model | Test set | n | Accuracy | Status |
| :--- | :--- | :---: | :---: | :--- |
| Foliar disease classifier (5-class) | Lab-condition images (PlantVillage-style) | 83 | **87.95%** | **[VERIFIED]** reproduced twice, exact TP/FP/FN match |
| Foliar disease classifier (5-class) | Real field images (2 independent sets, 0% overlap) | 261 | **51.7%** | **[VERIFIED]** — this is the honest field number |
| Foliar disease classifier, fine-tuned variant | Clean held-out field split (no training overlap) | 42 | **59.52%** | **[VERIFIED]**, from `african_finetuned_evaluation.json`, re-checked |
| Generic 12-class insect detector | Original test split | 546 | **16.3%** | **[VERIFIED]** |
| Generic 12-class insect detector | New, non-overlapping validation split | 1,095 | **17.6%** | **[VERIFIED]** — reproduces the 16.3% claim at 2x scale |
| Kenya 28-class priority pest detector | Wild field photos (`benchmark_test_15`) | 15 | **0.0% (0/15)** | **[VERIFIED]** — reproduced twice, matches the model's own audit trail note |

Read plainly: the model that is closest to production-ready on its narrow lab test (87.95%) drops by 36 points in the field. The model built specifically for the platform's named pest list (Fall Armyworm, Stem Borer, etc.) scores zero on the only wild-photo test that exists for it. Any "today vs. day 0" percentage that isn't one of the rows above should be treated as **[NOT VERIFIED]** until someone re-derives it from raw data the way the rows above were derived.

---

## 2. Initiatives with a real, traceable before/after

These are the interventions where a genuine before-state and after-state both exist in the repository or were established this session.

| Initiative | Before | After | Evidence |
| :--- | :--- | :--- | :--- |
| **Kenya 28-class label mapping** | **[NOT VERIFIED]** — class-to-index mapping lived only on the developer's local D: drive; model could not be independently audited | **[FIXED THIS SESSION]** — mapping recovered directly from the ONNX file's own embedded metadata (`custom_metadata_map['names']`), documented in `models/yolov8_agripests_taxonomy.md`, independently re-derived and matched exactly | Verified via `onnxruntime.InferenceSession.get_modelmeta()` on `yolov8_agripests_kenya.onnx` |
| **Telemetry log integrity** | Seeded/synthetic entries and real measurements mixed in `results/telemetry.jsonl` with no field distinguishing them | **[FIXED]** — every entry now carries `is_synthetic: true/false`; confirmed on both the 10 seeded rows and the one genuine 14,488 ms Ollama call | `scripts/seed_telemetry.py`, `benchmark/telemetry.py` |
| **Ollama latency reporting** | Hardcoded literal `ollama_latency_ms=3800.0` regardless of actual call time | **[FIXED]** — now measured with `time.perf_counter()` per call and shown to the user as "Measured local latency: X ms" | `ui/app.py`, Shamba AI chat handler |
| **Fabricated "Decision Precision" figures** | 94.2% / 98.4% hardcoded literals in `ui/app.py` and narrative reports, no model behind either | **[FIXED]** — zero references remain anywhere in the codebase | Repo-wide search, this session |
| **Grad-CAM / Lab View crash** | `AttributeError: 'Image' object has no attribute 'read'` — Lab View died mid-render every time, on every sample | **[FIXED]** — root cause was a nominal `isinstance()` type check breaking under an `ultralytics` monkey-patch of `PIL.Image.open`; replaced with a duck-typed check | `benchmark/explainability.py`; confirmed live — Grad-CAM now renders lesion focus score, all downstream tabs render |
| **Farmer view layout / duplicate header / oversized images** | Two stacked headers, mobile card frame not actually constraining width, uploaded photos rendering oversized | **[FIXED]** — root cause was a hand-typed `<div>` that Streamlit doesn't actually nest native widgets inside; switched to `st.container(key=...)`, removed the duplicate header block | `ui/app.py`, `ui/design_system.py`; confirmed live at both desktop and 375px mobile width |
| **Telemetry & Observability tab invisible** | Tab existed in code and routing but the 4-option segmented control overflowed its column at normal widths | **[FIXED]** — widened the nav column, shortened labels; all 4 tabs now render reliably | `ui/app.py` top navigation bar |
| **CLAHE contrast correction** | Implemented correctly in `benchmark/foliage_contrast.py`, never called — dead code, zero effect on any prediction | **[FIXED THIS SESSION]** — wired into both the classification adapter (`models/timm_adapter.py`) and the detection adapter (`models/yolo_adapter.py`), fails open on error so it can't block a diagnosis | Confirmed live: a real diagnosis ran end-to-end through the new code path without error. **Accuracy impact not yet re-measured** — see Section 3. |

---

## 3. What is fixed but *not yet* re-benchmarked

Fixing a bug and proving it improved accuracy are two different claims, and only the first one is done for these:

- **CLAHE is now live**, but the 51.7% field-accuracy figure in Section 1 was measured *before* this fix. The honest next step is to re-run the same 261-image field test (African Beans Field + Makerere Beans, the exact sets used above) through the model with CLAHE active and report the new number next to 51.7% — not assume it improved things, verify it.
- **The fine-tuned model's clean 59.52%** has not been re-run since CLAHE went live either.
- No claim is made here about response-time impact of CLAHE — it adds a real CPU-bound image-processing step (Lab-space histogram equalization + HSV boost) before every inference, so latency should be re-measured, not assumed unchanged.

---

## 4. Genuinely consistent figures kept from the prior document

Not everything in the earlier version was wrong. These held up:

- **83.2% unnecessary-spray reduction** — real output of `scripts/simulate_spray_reduction.py`, a deterministic (`seed=42`) 1,000-event Monte Carlo simulation run through the actual CDFA/EIL threshold logic in `models/cdfa_thresholds.py`. **[VERIFIED as a simulation result]** — it is not, and has never been, an empirical field measurement, and the earlier "88%" figure quoted elsewhere in the project was simply wrong; 83.2% is correct.
- **717-image / 28-class training set** for the Kenya priority pest detector — matches the audit trail in `models/yolov8_agripests_taxonomy.md`, consistent with the model's own embedded metadata.

---

## 5. What still has no independent evidence either way

Flagging these rather than silently repeating or silently dropping them:

- Ollama throughput ("31 tokens/sec") — no measurement of this found anywhere in the repo or telemetry log.
- Any single "Tier-1" forward-pass latency figure below ~10ms quoted in prior reports — my own ONNX Runtime measurements this engagement ranged roughly 6-18ms depending on model and batch, which is in the right neighborhood but was never matched to one specific quoted number.
- The SAHI/TTA/Bayesian pipeline's pest-detection recall on anything beyond the original n=15 image (90 pest instance) test — real, but the sample is small enough that "100%" or "85.1%" recovery claims built on top of it should not be presented as a settled number without a larger re-test, the same way the foliar classifier and the 12-class detector were re-tested at scale this session.

---

*Corrected version — supersedes all accuracy/latency claims in the prior "0-to-1 Architecture & Evolution Journey" document. Where this document says [VERIFIED], it means independently re-derived from raw images and raw model files in this session or a directly preceding one — not read from a summary report.*
