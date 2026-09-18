# Rail compact coherence checks

Train-only duplicate-grouped 5-fold × 3 selection CV. The current 201-feature coherence row is the reference. All feature selection and model fitting stay inside each training fold; no Test labels were used.

| model | input features | macro F1 ± fold SD | Side I F1 |
|---|---:|---:|---:|
| promoted coherence LightGBM | 201 | 0.8441 ± 0.1059 | 0.6454 |
| smaller, regularized LightGBM; same 201 features | 201 | 0.8178 ± 0.0841 | 0.6244 |
| remove time and vote blocks; keep coherence | 134 | 0.7599 ± 0.1089 | 0.4807 |

Neither compact row improves macro F1, so the promoted model and submission CSV stay unchanged. Earlier ladder rows also tested a 113-feature model (0.7720) and a 64-feature pruned model on 352 starting features (0.7799); those are distinct feature sets.
