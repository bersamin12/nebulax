# Rail corrugation - model ladder

The unchanged 22 historical rows are reused. One coherence row was added and the full 23-row nested selection rerun. The prior organiser Rail score was 0.7994152; the new Test score is unknown.

git `83a5696`, feature version `v2`, frozen outer CV = stratified 5-fold by file (grouped on the duplicate fingerprint) x 3 seeds [0, 1, 2], 10586 s wall. Every row is one model x ablation run through the *same* scheme; all numbers are reproducible from `rail_ladder.json`. The table ranks selection-CV rows; the selected row is post-hoc and its value is not the headline.

| # | model | ablation | macro F1 (mean +- sd) | Side I F1 | Side II F1 | speed-matched | n feat | fit s |
|---|---|---|---|---|---|---|---|---|
| 1 | `lgbm` | no shock + same-side axle-box coherence | **0.844** +- 0.106 | 0.645 | 0.907 | 0.851 | 201 | 2.11 |
| 2 | `lgbm` | + no shock channels | **0.836** +- 0.096 | 0.597 | 0.932 | 0.840 | 180 | 1.91 |
| 3 | `lgbm` | Hz bands + speed, wavelength discriminators kept, + mirror TTA | **0.801** +- 0.099 | 0.561 | 0.864 | 0.806 | 352 | 2.65 |
| 4 | `lgbm` | counter-design: Hz bands + speed, nothing distance-derived [R234] | **0.794** +- 0.117 | 0.564 | 0.843 | 0.805 | 245 | 2.13 |
| 5 | `lgbm` | Hz bands + speed, + fold-local speed-baseline residuals, + mirror TTA | **0.793** +- 0.100 | 0.539 | 0.863 | 0.800 | 352 | 4.42 |
| 6 | `lgbm_2view` | two-view ensemble: Hz-band model + wavelength model, no v^2, probability average, + mirror TTA | **0.787** +- 0.102 | 0.505 | 0.875 | 0.791 | 640 | 5.79 |
| 7 | `lgbm` | no v^2, wavelength + Hz bands, + mirror TTA | **0.784** +- 0.104 | 0.503 | 0.870 | 0.787 | 640 | 3.96 |
| 8 | `lgbm` | no v^2, + mirror TTA | **0.782** +- 0.113 | 0.514 | 0.852 | 0.784 | 528 | 3.61 |
| 9 | `lgbm_pruned` | + importance-pruned to 64 columns | **0.780** +- 0.120 | 0.517 | 0.848 | 0.790 | 352 | 8.13 |
| 10 | `lgbm` | + no shock, no time block, no votes | **0.772** +- 0.085 | 0.492 | 0.858 | 0.767 | 113 | 1.52 |
| 11 | `lgbm` | no v^2 normalisation | **0.768** +- 0.099 | 0.474 | 0.852 | 0.769 | 528 | 3.57 |
| 12 | `lgbm_hier` | no v^2, hierarchical + mirror TTA | **0.736** +- 0.113 | 0.411 | 0.821 | 0.737 | 528 | 2.31 |
| 13 | `lgbm_hier` | no v^2, hierarchical present/absent then side | **0.735** +- 0.114 | 0.411 | 0.817 | 0.736 | 528 | 2.28 |
| 14 | `rf` | no v^2, + mirror TTA | **0.733** +- 0.096 | 0.457 | 0.777 | 0.727 | 528 | 4.42 |
| 15 | `lgbm` | baseline features | **0.731** +- 0.090 | 0.395 | 0.831 | 0.728 | 528 | 3.84 |
| 16 | `lgbm` | + no time-domain block | **0.716** +- 0.096 | 0.347 | 0.837 | 0.712 | 224 | 2.21 |
| 17 | `lgbm_windows` | sub-window voting: 3 x 0.5 s windows, Hz bands + speed, + mirror TTA | **0.715** +- 0.144 | 0.423 | 0.759 | 0.713 | 352 | 99.13 |
| 18 | `logreg` | baseline features | **0.713** +- 0.102 | 0.415 | 0.780 | 0.698 | 528 | 0.95 |
| 19 | `logreg` | no v^2, + mirror TTA | **0.701** +- 0.097 | 0.421 | 0.738 | 0.691 | 528 | 0.96 |
| 20 | `svm` | baseline features | **0.680** +- 0.083 | 0.339 | 0.757 | 0.661 | 528 | 1.26 |
| 21 | `rf` | baseline features | **0.624** +- 0.109 | 0.219 | 0.707 | 0.611 | 528 | 4.37 |
| 22 | `multirocket_ridge` | per-side wavelength spectra, no v^2 | **0.496** +- 0.098 | 0.153 | 0.404 | 0.470 | 768 | 12.67 |
| 23 | `multirocket_ridge` | per-side wavelength spectra (4 x 192) | **0.478** +- 0.116 | 0.140 | 0.369 | 0.454 | 768 | 12.96 |

**Post-hoc selection-CV winner: `lgbm` / no shock + same-side axle-box coherence at macro F1 0.844 +- 0.106**, against the baseline's 0.731 +- 0.090. It replaces `models/ps3/rail.pkl`.

## Honest nested/outer headline

Macro F1 **0.805 +- 0.129** over 15 grouped outer folds (5 folds x seeds [0, 1, 2]). All declared ladder rows are selected by grouped inner 3-fold CV inside each outer training partition.

## Stress splits for the winner

| scheme | folds | macro F1 (mean +- sd) |
|---|---|---|
| contiguous | 5 | 0.782 +- 0.135 |
| speed_range | 3 | 0.741 +- 0.117 |

## What the ablations say

* **v^2 normalisation**: costs 0.037 macro F1 (0.731 with, 0.768 without). [R237] predicts it should help by making the roughness->acceleration transfer speed-independent; on this dataset the fault files occupy a narrow speed band (35-67 km/h) and the un-normalised level is itself informative, so removing the speed scaling removes usable signal. Reported as measured, not as predicted.
* **MultiRocket + RidgeClassifierCV on the per-side wavelength spectra**: 0.496 vs LightGBM's 0.731 on the same representation. [R249] finds ROCKET ahead of hand-crafted features on raw accelerometer windows; here the hand-crafted side contrast already encodes what the label describes, so the gap narrows.
* **Hierarchical (present/absent, then which side)**: 0.736 vs 0.768 flat on the same features - the 38 positives support one binary decision better than three-way softmax (`ps3_addendum.md` section 6 row 3).
* **The counter-design arm** [R234] - hand the model fixed 20-5000 Hz bands and the speed instead of normalising speed away - scores 0.794 with nothing distance-derived and 0.801 when the wavelength discriminators are kept alongside, against 0.782 for the distance/wavelength path. Read it honestly: on **this** dataset the faults occupy 35-67 km/h while a third of the Normals sit below 20, so a representation that tracks speed is rewarded by the label distribution, which is exactly the confound the speed-matched column and the leave-speed-range-out split exist to expose. Two things keep it from being pure confound-mining: the winner also leads on the speed-matched subset and on leave-speed-range-out (see the stress table above and `rail_cv.md`), and the gap is well inside one fold-to-fold sd. Note the direction flips with the v^2 flag: in `rail_cv.md`, where v^2 normalisation is on, dropping distance resampling *costs* 0.04 - the Hz bands only win once the v^2 scaling is off, i.e. once the model is allowed to use absolute level, which is itself a speed proxy.
* **Two-view ensemble (Hz-band model + wavelength model, probability average, + mirror TTA)**: scores 0.787 +- 0.102 (Side I 0.505, Side II 0.875), against the single-view Hz model's 0.801 (Side I 0.561) and the single-view wavelength model's 0.782 (Side I 0.514).
* **Feature reduction rows**: base Hz model 0.801 vs no shock 0.836, no time 0.716, minimal 0.772, pruned-64 0.780. Pruning down from 352 columns follows the VSB and LANL winning approaches.
* **Fold-local speed-baseline residuals**: 0.793 +- 0.100 vs base Hz model 0.801.
* **Sub-window voting (3 x 0.5 s windows, + mirror TTA)** [R249]: 0.715 +- 0.144 (Side I 0.423) vs base 0.801.
* **Mirror test-time augmentation** (average p(x) with the side-swapped p(mirror(x))): best TTA row 0.836. The mirror is exact for this sensor layout, so the two views must agree; averaging them costs one extra forward pass.
* The spread across folds (sd 0.090 at n=14 for Side I, ~2.8 per fold) is the honest headline, not a point estimate: [R241] is the realism anchor (81.5 % held-out accuracy over 21 classifiers on railway vibration), and `ps3_addendum.md` section 6 row 5 sets the expectation band at 0.70-0.85 macro F1 with visible spread, against the 0.95+ of rig and simulation studies [R240][R243].
* **Same-side axle-box coherence**: the extra 21 phase-independent Hz-band features raise selection macro F1 from the prior 0.8365 to 0.8441, with Side I F1 0.6454. They average six same-side box pairs per car across eight cars. The full 23-row nested estimate and both stress splits are reported above.

## Rows skipped for budget, with the reason

* **VMD / CEEMDAN / EWT / SPWVD adaptive decompositions** [R247]: per-record, parameter-heavy and slow, with no evidence they beat a wavelength-band PSD on a 3-class macro-F1 task at n=272.
* **Model-based roughness inversion** [R238][R239]: needs track receptance and rail-pad stiffness we do not have; the source's own sensitivity analysis moves the answer 3.5 dB for a 20 % pad-stiffness error.
* **Self-supervised / contrastive pre-training** [R244] and **sim-pretrain -> fine-tune** [R243]: the right shape for 14 labelled positives, but 234 unlabelled 1 s records are far too few for MoCo to pay and no corrugation simulator exists in this repo.
* **Deep 1D-CNN rows** [R240][R242]: a learnable front end on 272 files overfits, and the plan caps the deep tier at one row per family under a strict budget; `multirocket_ridge` is the row that family gets here.
* **GAN / diffusion augmentation**: the mirror swap is exact and free; generative augmentation on 14 spectra is strictly worse and adds a failure mode [R307][R308][R304]; `ps3_addendum.md` section 6 row 5.
* **Comb filter keyed to the wheel circumference** [R238]: implementable but optional at this budget; the side contrast captures most of the wheel-vs-rail separation.
