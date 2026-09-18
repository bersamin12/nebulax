# Rail Moirai-2 last-layer fine-tuning trial

The released Moirai-2 small encoder is adapted inside each training fold. Its last transformer layer and a side-aware head are trained on eight-box car groups. Validation files are excluded from parameter updates and head fitting. No Test files or labels were used. Fault-class loss weight: 1.

Historical W7 selection gate: 0.8365. Current-code W7 reference on these folds: 0.8341 macro F1; Side I F1 0.5930.

| row | macro F1 ± sd | Side I F1 |
|---|---:|---:|
| Moirai-2 last-layer fine-tune | 0.5459 ± 0.0969 | 0.1013 |
| W7 + 25% fine-tuned Moirai-2 | 0.8197 ± 0.0986 | 0.5708 |

Decision: **retain W7: fine-tuned teacher did not beat selection gate**.
