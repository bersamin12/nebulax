# Door — cross-validation

* scheme: 5 contiguous time blocks of the raw stream, end to end  (5 contiguous time blocks of the raw stream, segmentation and classification re-run end to end per held block)
* seeds: [0]   git rev: `24f6755`   feature version: `door-f1`
* wall seconds: 1154.7

| block | n true | n pred | IoU-F1 | soft recall | soft precision | misses | FPs | wrong label | oracle macro F1 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 22 | 22 | 0.9091 | 0.9091 | 0.9091 | 2 | 2 | 2 | 0.8854 |
| 1 | 22 | 22 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 1.0000 |
| 2 | 22 | 22 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 1.0000 |
| 3 | 22 | 22 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 1.0000 |
| 4 | 22 | 22 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 1.0000 |

**Honest nested/outer IoU-weighted F1 = 0.9818 ± 0.0364** over 5 folds (min 0.9091); oracle-segment macro F1 0.9771 ± 0.0458 (diagnostic only).
The addendum-MUST candidate is selected on inner contiguous folds made only from each outer training partition. The full-ladder winner shipped below was chosen after reading the ladder and is labelled post-hoc; its selection-CV score is not the headline.

## Segmentation check (no labels used)

* gap rule (`dt > 0.5 s`) on `Train.csv`: 110 cycles vs 110 in the answer file, 110 boundary-exact, IoU-F1 (labels ignored) 1.0000.
* fallback state machine on the same stream **with the gaps removed** (timestamps re-stamped at a uniform 20 ms): 110 cycles, IoU-F1 (labels ignored) 1.0000.
* `Test.csv`: 38 cycles.

## Ablations (same frozen scheme)

| variant | IoU-F1 mean | sd | oracle macro F1 |
|---|---|---|---|
| stump on i_mid_rel, baseline features (diagnostic) | 1.0000 | 0.0000 | 1.0000 |
| logreg, physics features (post-hoc ladder candidate) | 1.0000 | 0.0000 | 1.0000 |
| logreg, baseline features, within-block batch norm (TRANSDUCTIVE [R303], never the headline) | 0.9818 | 0.0223 | 0.9782 |
| logreg, baseline features, joint (not per-operation) | 0.9909 | 0.0182 | 0.9891 |
| segmenter = hybrid, baseline features (whole pipeline re-run) | 0.9818 | 0.0364 | 0.9771 |
| segmenter = state, baseline features (whole pipeline re-run) | 0.9818 | 0.0364 | 0.9771 |
