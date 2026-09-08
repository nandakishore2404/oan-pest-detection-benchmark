# This document was found to contain fabricated numbers — corrected 2026-09-08

**This file previously presented a "T0 → T6" progression (41.5% → 89.2% accuracy, 320ms → 38.2ms
latency, 34% → 85.1% recall, 72% → 12.6% spray reduction) labeled "Independently Audited."**
That progression traced to a hardcoded literal array with no model run behind any of the seven
milestone numbers — the same fabrication already identified and removed once earlier the same
session, from `ui/app.py`'s `timeline_accuracy` array. It was reintroduced here, in `ui/app.py`'s
Telemetry tab, in `README.md`, and in a second duplicate report file, in a commit titled
`feat(telemetry): publish complete 7-stage initiative-wise accuracy and latency evolution matrix
and docs` (`8fdab15`).

**The single canonical, evidence-tagged version of this history now lives at:**
[`results/reports/OAN_Kenya_0_to_1_Architecture_and_Evolution_Journey.md`](OAN_Kenya_0_to_1_Architecture_and_Evolution_Journey.md)
— read that file for the full breakdown. This file is kept only as a pointer, to avoid two
documents drifting out of sync with each other.

**Summary of what actually survives independent re-derivation** (see the canonical file for
methodology, evidence tags, and per-class detail):

| Model | Test set | n | Accuracy |
| :--- | :--- | :---: | :---: |
| Foliar disease classifier (5-class) | Lab-condition images | 83 | 87.95% |
| Foliar disease classifier (5-class) | Real field images (2 sets) | 261 | 51.7% |
| Foliar disease classifier, fine-tuned | Clean held-out field split | 42 | 59.52% |
| Generic 12-class insect detector | Original test split | 546 | 16.3% |
| Generic 12-class insect detector | New validation split | 1,095 | 17.6% |
| Kenya 28-class priority pest detector | Wild field photos | 15 | 0.0% (0/15) |

No number above should be read as part of a smooth "T0 to T6" trend — each row is its own
independent test, on its own model, on its own dataset. There is no single project-wide accuracy
curve that has been verified.
