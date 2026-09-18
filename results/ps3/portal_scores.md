# Organiser portal scores and model association

The user reported only aggregate portal scores. The earlier uploaded outer ZIP is unavailable for byte-for-byte verification. The saved pre-update `predictions.zip` matches the backed-up W7 Rail and original SHM CSVs exactly. The user identified the 18 September 2026, 22:18 Singapore-time `submission/nebulax/predictions.zip` as the later scored submission. Its present ZIP members match the current CSVs byte for byte; the portal upload itself was not retained independently. `portal_scores.json` pins model, CSV, and current ZIP SHA-256 hashes.

The earlier models, CSVs, reports and local archives are preserved in `submission/history/`.

| Subsystem | Earlier model portal score | Current model portal score | Absolute gain |
|---|---:|---:|---:|
| Rail: W7 → 201-feature coherence LightGBM | 0.7994152046783626 macro F1 | 0.83104 macro F1 | +0.0316247953 |
| SHM: full-feature log-Lasso → positive-skew rainflow blend | 0.971752844618028 (`1 - MAPE`) | 0.972795 (`1 - MAPE`) | +0.0010421554 |

The current Rail coherence model has Train-only selection CV 0.8441 and nested CV 0.8051. Its measured organiser score is 0.83104, to the precision supplied by the user.

For SHM, measured portal MAPE fell from 0.02824716 to 0.027205, a **3.69% relative reduction**. The old and current nested Train MAPE are 0.02031766 and 0.01870004. Under the organiser formula, the corresponding Train estimates are 0.97968234 and 0.98129996: a **0.00161762 absolute validation score gain** and **7.96% relative validation MAPE reduction**.
