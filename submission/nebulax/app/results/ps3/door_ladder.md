# Door model ladder

Frozen scheme: 5 contiguous time blocks of `Train.csv`; segmentation and classification re-run end to end inside every held block; metric = `nebulax.ps3.scoring.iou_f1`. Oracle-segment macro F1 (classification on the true spans) is a diagnostic only. Every scaler, per-operation baseline, current-vs-position template, back-EMF fit, decision threshold and augmentation is fitted inside the four training blocks - the threshold on **end-to-end IoU-F1 over contiguous inner splits of the training fold**, never on a cycle-level F1 and never on the held block (`ps3_addendum.md` section 6 row 10). PR-AUC (average precision for `Abnormal resistance`) is reported alongside, as the subway-door reference pipeline does under comparable imbalance [R68].

The bold/top ladder row is a **post-hoc selection-CV result**, because the full table was ranked on these same five blocks. It is not the headline. The honest headline re-runs selection among the addendum-MUST candidates inside every outer training partition.

Tie-break, fixed before the table was read: among rows within 1e-9 of the best mean IoU-F1, take the simplest model - fewest features, then the earlier row in the declared ladder order. The stump is a diagnostic and never wins. Two columns are reported but kept out of the tie-break: `margin` (mean held-out |p - 0.5|; MultiRocket's RidgeClassifierCV emits hard 0/1 labels, so its margin is 0.500 for free) and `fit s` (noisy enough to reorder rows when the box is shared).

`abn/38` is a sanity column, not a selection criterion: the row refitted on all of `Train.csv` and run over the distributed `Test.csv`, how many of its 38 cycles it calls abnormal. The training prior is 30/110 = 27 %, and the mid-stroke current on `Test.csv` is cleanly bimodal with 8 cycles above the gap, so a row calling far more than 8 is mis-calibrated off-distribution even though its held-out IoU-F1 is perfect.

| # | row | tier | cite | IoU-F1 mean ± sd | min | wrong | PR-AUC | oracle macro F1 | abn/38 | fit s | n feat |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | stump on i_mid_rel (diagnostic) | diagnostic | [R64] | 1.0000 ± 0.0000 | 1.0000 | 0 | 1.000 | 1.0000 | 8 | 0.49 | 5 |
| 2 | logreg (physics features, separate) | MUST | [R64][R66][R67][R78] | 1.0000 ± 0.0000 | 1.0000 | 0 | 1.000 | 1.0000 | 8 | 4.96 | 24 |
| 3 | lgbm_three_regime (physics, regime-named features) | MUST | [R64] | 1.0000 ± 0.0000 | 1.0000 | 0 | 1.000 | 1.0000 | 14 | 0.96 | 24 |
| 4 | logreg (physics, joint) | ablation | [R64] | 1.0000 ± 0.0000 | 1.0000 | 0 | 1.000 | 1.0000 | 8 | 5.00 | 25 |
| 5 | multirocket_ridge (raw cycle, aug=none) | MUST | [R43][R47][R48] | 1.0000 ± 0.0000 | 1.0000 | 0 | 1.000 | 1.0000 | 8 | 1.44 | 768 |
| 6 | multirocket_ridge (raw cycle, aug=jitter) | ablation | [R296] | 1.0000 ± 0.0000 | 1.0000 | 0 | 1.000 | 1.0000 | 8 | 2.25 | 768 |
| 7 | multirocket_ridge (raw cycle, aug=window warp) | ablation | [R296] | 1.0000 ± 0.0000 | 1.0000 | 0 | 1.000 | 1.0000 | 8 | 2.15 | 768 |
| 8 | quant (raw cycle) | NICE | [R49] | 1.0000 ± 0.0000 | 1.0000 | 0 | 1.000 | 1.0000 | 8 | 3.01 | 768 |
| 9 | litetime (raw cycle, aug=none) | NICE | [R53] | 1.0000 ± 0.0000 | 1.0000 | 0 | 1.000 | 1.0000 | 8 | 11.97 | 768 |
| 10 | litetime (raw cycle, aug=window warp) | ablation | [R296] | 1.0000 ± 0.0000 | 1.0000 | 0 | 1.000 | 1.0000 | 8 | 29.47 | 768 |
| 11 | logreg (physics, no class weighting) | ablation | [R302] | 0.9909 ± 0.0182 | 0.9545 | 1 | 1.000 | 0.9891 | 8 | 4.97 | 24 |
| 12 | random_forest (physics) | NICE | [R68] | 0.9909 ± 0.0182 | 0.9545 | 1 | 1.000 | 0.9891 | 8 | 3.91 | 24 |
| 13 | svm rbf (physics) | NICE | [R64] | 0.9909 ± 0.0182 | 0.9545 | 1 | 1.000 | 0.9891 | 15 | 4.93 | 24 |
| 14 | stacking RF+XGB -> calibrated logreg (physics) | MUST | [R68] | 0.9909 ± 0.0182 | 0.9545 | 1 | 1.000 | 0.9891 | 8 | 7.76 | 24 |
| 15 | logreg (baseline features, separate) | MUST | [R64] | 0.9818 ± 0.0364 | 0.9091 | 2 | 1.000 | 0.9771 | 9 | 5.18 | 5 |
| 16 | logreg (physics, within-block batch norm - TRANSDUCTIVE) | ablation | [R303] | 0.9818 ± 0.0364 | 0.9091 | 2 | 1.000 | 0.9771 | 8 | 8.12 | 24 |

## Addendum MUST rows

| row | cite | ran | IoU-F1 |
|---|---|---|---|
| logreg (baseline features, separate) | [R64] | yes | 0.9818 |
| logreg (physics features, separate) | [R64][R66][R67][R78] | yes | 1.0000 |
| lgbm_three_regime (physics, regime-named features) | [R64] | yes | 1.0000 |
| stacking RF+XGB -> calibrated logreg (physics) | [R68] | yes | 0.9909 |
| multirocket_ridge (raw cycle, aug=none) | [R43][R47][R48] | yes | 1.0000 |

## Selected artefact (post-hoc)

`logreg (physics features, separate)` at selection-CV IoU-F1 1.0000 ± 0.0000. This row was chosen after the ladder was read.

## Honest nested/outer headline

IoU-F1 **0.9818 ± 0.0364** over 5 outer folds and seeds [0]; each outer fold selects only from the addendum-MUST rows on inner contiguous folds.

## Notes

* Every **MUST** row of `docs/research/ps3_addendum.md` section 1.3 ran: the gap segmenter and the command/position state-machine fallback (`door_cv.md`; the fallback is tested on `Train.csv` with the gaps removed), the per-operation fold-fitted baseline with a robust peer z [R78], logistic regression on the physics features, `LGBMThreeRegime` on regime-named features [R64], stacking RF+XGBoost -> calibrated logistic regression with PR-AUC reported [R68], and `multirocket_ridge` on the raw 50 Hz cycle [R43][R47][R48].
* Nothing was skipped for budget: every row finished well inside the 5 min per-row budget.
* **SKIP**ped per the addendum, not attempted: HIVE-COTE 2.0 (~340 h over the benchmark; the ceiling we do not buy [R52]) and audio MNPE + SVM (no microphone channel in this stream [R75]). ClaSP interior segmentation [R176][R177] is a NICE row left out: the gap segmenter already reproduces all 110 answer segments exactly, so a learned interior split has no headroom to win and could only over-segment.
* `litetime x aug=jitter` was not run: the deep tier carries one augmentation contrast and window warping is the higher-ranked transform; the cheap ROCKET row carries the full {none, jitter, warp} sweep instead.
* Augmentation is applied to raw-cycle rows only and inside the training fold only, and is restricted to window warping and magnitude warping / jitter - never rotation, flipping or permutation (a door motor current has a physical sign and a cycle's phases are ordered). Window **slicing** is excluded although it ranks second overall: it assumes every sub-window carries the label, and a resistance signature lives in one phase of the stroke. The addendum's expectation was about +1.5 % through ROCKET [R296]; measured here it is a wash because the un-augmented row is already perfect - reported either way, as required.
* Class weighting is a measured row, not an assumption: `logreg (physics, no class weighting)` is the ablation the addendum asks for [R302].
* The within-block batch-normalisation row is labelled **TRANSDUCTIVE**: it reads the held-out block's own statistics, a named leakage type [R303], so it stays an ablation and can never be the headline however well it scores.
* Segmentation ablations (gap vs hybrid vs the command/position state machine) live in `door_cv.md`: they change the segmenter, not the model; every model row above uses the gap segmenter.
