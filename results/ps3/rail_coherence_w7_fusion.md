# Rail W7 and coherence fusion

Train-only, duplicate-grouped 5-fold × 3 selection CV. Both model probabilities were fitted on exactly the same outer training partitions, mirror-averaged, class-boosted and aligned by held-out file. All 816 coherence predictions reproduce the saved production CV. The low-speed rule is applied after blending. No Test labels were used.

| coherence probability weight | macro F1 ± fold SD | Side I F1 | Side II F1 | changed decisions vs coherence |
|---:|---:|---:|---:|---:|
| 0.00 | 0.8341 ± 0.1008 | 0.5930 | 0.9276 | 10 |
| 0.25 | 0.8341 ± 0.1008 | 0.5930 | 0.9276 | 10 |
| 0.50 | 0.8420 ± 0.1036 | 0.6102 | 0.9337 | 8 |
| 0.75 | 0.8421 ± 0.1101 | 0.6335 | 0.9127 | 2 |
| 1.00 | 0.8441 ± 0.1059 | 0.6454 | 0.9068 | 0 |

No blend exceeds the selected coherence-only row. The model, predictions CSV and four-task ZIP remain unchanged. Earlier W7 blends with the shock-inclusive W6 model and frozen MantisV2/MOMENT features also did not clear the current coherence score.
