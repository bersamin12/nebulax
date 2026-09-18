# Rail low-speed Normal undersampling

Train-only diagnostic on the promoted coherence row. Each 5-fold × 3 duplicate-grouped training fold randomly retains the stated share of its Normal files below 20 km/h. All fault files and higher-speed Normal files remain. Held-out files are never sampled or augmented. The model still uses its existing balanced class weights and the low-speed prediction rule.

| training-fold retention of low-speed Normal | macro F1 ± fold SD | Side I F1 | Side II F1 |
|---|---:|---:|---:|
| 100% (promoted coherence reference) | 0.8441 ± 0.1059 | 0.6454 | 0.9068 |
| 25% | 0.8410 ± 0.0964 | 0.6343 | 0.9108 |
| 0% | 0.8351 ± 0.0915 | 0.6492 | 0.8782 |

Neither undersampling row improves macro F1, so the promoted model and submission CSV stay unchanged. These are post-hoc Train-only diagnostics; no Test labels were used.
