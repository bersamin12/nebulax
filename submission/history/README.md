# Previous organiser portal rounds

These local artifacts preserve the model, prediction CSV, validation report and submission
archive for the earlier Rail and SHM portal scores. The exact score values and SHA-256 pins are
in [`results/ps3/portal_scores.json`](../../results/ps3/portal_scores.json), with a readable
summary in [`portal_scores.md`](../../results/ps3/portal_scores.md).

| Archived round | Reported portal score | Preserved under |
|---|---:|---|
| Rail W7 before coherence update | 0.7994152046783626 macro F1 | `20260918_score_round_before_rail_coherence/` |
| SHM before rainflow gate update | 0.971752844618028 (1 - MAPE) | `20260918_organiser_score_before_shm_gate/` |

The user identified the current `submission/nebulax/predictions.zip` as the later scored
submission: Rail 0.83104 macro F1 and SHM 0.972795 (1 - MAPE). The earlier uploaded outer
archive was not retained, so its portal association is based on the user's report and the
matching local archive members. These files are evidence of past rounds; the current four
PS3 predictors and outputs are in `submission/nebulax/`.
