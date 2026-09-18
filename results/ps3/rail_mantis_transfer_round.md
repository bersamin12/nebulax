# Rail frozen MantisV2 transfer trial

Frozen external MantisV2 embeddings of eight 512-sample windows from each of 64 vibration sensors. Encoder weights are unchanged. Mean and spread of the 32 sensors on each side form a side-aware linear-probe input. All scaling and classifier fitting is inside each grouped fold. No Test files or labels were used.

Historical W7 selection gate: 0.8365. The W7 reference below is refitted with the current duplicate-grouped inner splits; it can differ slightly from the historical 0.8365 result.

| row | macro F1 ± sd | Side I F1 | speed-matched macro F1 |
|---|---:|---:|---:|
| W7 current-code reference | 0.8341 ± 0.1008 | 0.5930 | 0.8387 |
| MantisV2 frozen linear probe | 0.7056 ± 0.0820 | 0.3921 | 0.7001 |
| W7 + 25% MantisV2 probability blend | 0.8360 ± 0.1000 | 0.6038 | 0.8407 |

Decision: **retain W7: transfer rows did not beat selection gate**.

The blend changed 3 of 816 held-out decisions relative to the current-code W7 reference:

- seed0_fold4 Train169.csv (Normal): Normal → Side II
- seed2_fold0 Train123.csv (Normal): Side I → Normal
- seed2_fold3 Train44.csv (Normal): Side I → Normal

Nested audit: not run because neither transfer row cleared the selection gate.

Retained W7 stress splits:

- contiguous: 0.7488 ± 0.1406; Side I F1 0.4032
- speed_range: 0.7258 ± 0.1006; Side I F1 0.4583

Runtime: 1.2 minutes, excluding the one-time frozen embedding extraction.

## Verification

- 15 grouped validation folds contain 272 unique held-out training files each.
- All 272 cached embeddings have shape (64, 512) and finite values.
- Sensor sides use odd versus even axle-box positions, with an exact block-swap mirror.
- The delivered rail model and predictions were not changed by this trial.
