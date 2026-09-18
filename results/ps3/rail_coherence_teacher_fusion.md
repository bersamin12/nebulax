# Rail coherence with frozen encoder probes

Train-only, duplicate-grouped 5-fold × 3 selection CV. The MantisV2 short-window and MOMENT long-window embeddings were extracted from frozen pretrained backbones. Probe scaling and fitting use only each outer training fold. Coherence probabilities are independently fitted on the same folds; all 816 component file IDs and truths match. No Test labels were used.

| coherence weight | macro F1 ± fold SD | Side I F1 | Side II F1 | decisions changed vs coherence |
|---:|---:|---:|---:|---:|
| 0.500 | 0.8263 ± 0.1076 | 0.6079 | 0.8871 | 17 |
| 0.750 | 0.8409 ± 0.1021 | 0.6340 | 0.9058 | 10 |
| 0.875 | 0.8447 ± 0.1112 | 0.6406 | 0.9127 | 3 |
| 1.000 | 0.8441 ± 0.1059 | 0.6454 | 0.9068 | 0 |

The 87.5% coherence blend gains only 0.0007 macro F1 on this post-hoc selection CV, changing three of 816 decisions while lowering Side I F1. Its large encoder weights exceed the 5 MB Rail model limit. This row has no full nested or stress audit and was not distilled or promoted; the current submission remains the 794 KB coherence model.

Validation note: the rerun reproduces the historical W7 + MantisV2 + MOMENT fusion on all 816 held-out predictions. MOMENT alone differs on one near-tie Normal file (`Train256.csv`, 0.491 Normal versus 0.489 Side II in the rerun), recorded in the JSON.
