# Rail Ti-MAE architecture trial

An unofficial Ti-MAE implementation has no released general pretrained checkpoint. This trial trains masked reconstruction from scratch on each outer training fold's unlabelled vibration windows, then fits a side-aware logistic probe within that fold. Held-out recordings never enter pretraining. No Test files or labels were used.

| row | macro F1 ± sd | Side I F1 | speed-matched macro F1 |
|---|---:|---:|---:|
| Ti-MAE SSL probe | 0.4027 ± 0.0700 | 0.0807 | 0.3283 |
| W7 + 25% Ti-MAE SSL | 0.8238 ± 0.0980 | 0.5797 | 0.8268 |

Decision: **retain W7: Ti-MAE rows did not beat selection gate**.
