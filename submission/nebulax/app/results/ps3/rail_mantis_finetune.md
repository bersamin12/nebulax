# Rail MantisV2 backbone fine-tuning trial

MantisV2's first three pretrained transformer blocks process 512-sample vibration windows. The input and first block, last block, and all-three-block modes are trained separately inside each of 15 duplicate-grouped folds. The 64 sensors are pooled by rail side. Two windows per sensor are sampled at each training step; all eight cached windows are averaged for held-out prediction. No Test files or labels were used.

Historical W7 selection gate: 0.8365.

| row | macro F1 ± sd | Side I F1 |
|---|---:|---:|
| first_block / fine-tuned only | 0.4810 ± 0.0677 | 0.0000 |
| first_block / W7 + 25% fine-tuned | 0.8237 ± 0.0959 | 0.5771 |
| last_block / fine-tuned only | 0.4643 ± 0.0641 | 0.0000 |
| last_block / W7 + 25% fine-tuned | 0.8237 ± 0.0959 | 0.5771 |
| all_three_blocks / fine-tuned only | 0.4686 ± 0.0836 | 0.0267 |
| all_three_blocks / W7 + 25% fine-tuned | 0.8258 ± 0.0930 | 0.5771 |

Decision: **retain W7: fine-tuning rows did not beat selection gate**.

Nested audit: not run because no fine-tuning row cleared the selection gate.
