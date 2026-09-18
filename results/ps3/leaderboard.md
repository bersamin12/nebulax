# PS3 model ladder leaderboard

All metrics below are copied from the four committed ladder JSON files; this report does not refit, predict, or recompute a score. A bold row is the winner of its frozen **selection CV**, not automatically the honest headline. Door and Rail artifacts were chosen after the ladder was inspected, so their selection values are explicitly post-hoc; the final table uses the nested/outer estimates. ACV keeps its pre-registered fixed rule because the six-case pool is exploratory. Rail's `speed < 20 km/h -> Normal` rule is a dataset shortcut, not physics.

## Door — segmentation + classification

Frozen scheme: 5 contiguous time blocks, end-to-end; seeds [0]. Honest headline uses 5 outer folds/cases (nested.iou_f1_mean/.iou_f1_sd).

| model | ablation / feature row | metric mean ± sd | CV folds / seeds | fit s | n feat | augmentation | addendum cite | flags | JSON key path |
|---|---|---:|---|---:|---:|---|---|---|---|
| logreg | logreg (baseline features, separate) | 0.9818 ± 0.0364 | 5 folds; seeds [0] | 5.18 | 5 | off | [R64] | baseline | `rows[0].iou_f1_mean/.iou_f1_sd/.fit_seconds_mean/.n_features` |
| **logreg** | **logreg (physics features, separate)** | **1.0000 ± 0.0000** | 5 folds; seeds [0] | 4.96 | 24 | off | [R64][R66][R67][R78] | winner, physics | `rows[1].iou_f1_mean/.iou_f1_sd/.fit_seconds_mean/.n_features` |
| logreg | logreg (physics, joint) | 1.0000 ± 0.0000 | 5 folds; seeds [0] | 5.00 | 25 | off | [R64] | - | `rows[2].iou_f1_mean/.iou_f1_sd/.fit_seconds_mean/.n_features` |
| logreg_unweighted | logreg (physics, no class weighting) | 0.9909 ± 0.0182 | 5 folds; seeds [0] | 4.97 | 24 | off | [R302] | - | `rows[3].iou_f1_mean/.iou_f1_sd/.fit_seconds_mean/.n_features` |
| stump | stump on i_mid_rel (diagnostic) | 1.0000 ± 0.0000 | 5 folds; seeds [0] | 0.49 | 5 | off | [R64] | - | `rows[4].iou_f1_mean/.iou_f1_sd/.fit_seconds_mean/.n_features` |
| random_forest | random_forest (physics) | 0.9909 ± 0.0182 | 5 folds; seeds [0] | 3.91 | 24 | off | [R68] | - | `rows[5].iou_f1_mean/.iou_f1_sd/.fit_seconds_mean/.n_features` |
| svm | svm rbf (physics) | 0.9909 ± 0.0182 | 5 folds; seeds [0] | 4.93 | 24 | off | [R64] | - | `rows[6].iou_f1_mean/.iou_f1_sd/.fit_seconds_mean/.n_features` |
| lgbm_three_regime | lgbm_three_regime (physics, regime-named features) | 1.0000 ± 0.0000 | 5 folds; seeds [0] | 0.96 | 24 | off | [R64] | - | `rows[7].iou_f1_mean/.iou_f1_sd/.fit_seconds_mean/.n_features` |
| stacking | stacking RF+XGB -> calibrated logreg (physics) | 0.9909 ± 0.0182 | 5 folds; seeds [0] | 7.76 | 24 | off | [R68] | - | `rows[8].iou_f1_mean/.iou_f1_sd/.fit_seconds_mean/.n_features` |
| logreg | logreg (physics, within-block batch norm - TRANSDUCTIVE) | 0.9818 ± 0.0364 | 5 folds; seeds [0] | 8.12 | 24 | off | [R303] | - | `rows[9].iou_f1_mean/.iou_f1_sd/.fit_seconds_mean/.n_features` |
| multirocket_ridge | multirocket_ridge (raw cycle, aug=none) | 1.0000 ± 0.0000 | 5 folds; seeds [0] | 1.44 | 768 | off | [R43][R47][R48] | - | `rows[10].iou_f1_mean/.iou_f1_sd/.fit_seconds_mean/.n_features` |
| multirocket_ridge | multirocket_ridge (raw cycle, aug=jitter) | 1.0000 ± 0.0000 | 5 folds; seeds [0] | 2.25 | 768 | on | [R296] | - | `rows[11].iou_f1_mean/.iou_f1_sd/.fit_seconds_mean/.n_features` |
| multirocket_ridge | multirocket_ridge (raw cycle, aug=window warp) | 1.0000 ± 0.0000 | 5 folds; seeds [0] | 2.15 | 768 | on | [R296] | - | `rows[12].iou_f1_mean/.iou_f1_sd/.fit_seconds_mean/.n_features` |
| quant | quant (raw cycle) | 1.0000 ± 0.0000 | 5 folds; seeds [0] | 3.01 | 768 | off | [R49] | - | `rows[13].iou_f1_mean/.iou_f1_sd/.fit_seconds_mean/.n_features` |
| litetime | litetime (raw cycle, aug=none) | 1.0000 ± 0.0000 | 5 folds; seeds [0] | 11.97 | 768 | off | [R53] | - | `rows[14].iou_f1_mean/.iou_f1_sd/.fit_seconds_mean/.n_features` |
| litetime | litetime (raw cycle, aug=window warp) | 1.0000 ± 0.0000 | 5 folds; seeds [0] | 29.47 | 768 | on | [R296] | - | `rows[15].iou_f1_mean/.iou_f1_sd/.fit_seconds_mean/.n_features` |
| chance | all-Normal on oracle segments | 0.7273 ± 0.0000 | fixed vocabulary / released set | n/a | n/a | off | - | chance floor | `chance_floor.score/.sd` |

Skipped from the addendum:

- HIVE-COTE 2.0: about 340 h over the benchmark; ceiling not bought [R52].
- Audio MNPE + SVM: the controller stream has no microphone channel [R75].
- ClaSP cycle-interior segmentation: NICE only; the gap segmenter is boundary-exact on all 110 training cycles [R176][R177][R64].

## ACV — leaking-car ranking

Frozen scheme: leave-one-case-out (6 cases, exploratory); seeds [0]. Honest headline uses 6 outer folds/cases (selected.score/.sd).

| model | ablation / feature row | metric mean ± sd | CV folds / seeds | fit s | n feat | augmentation | addendum cite | flags | JSON key path |
|---|---|---:|---|---:|---:|---|---|---|---|
| **fixed rule** | **peer_delta_hot_loo** | **1.0000 ± 0.0000** | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | winner | `rows[0].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_hot_loo [all-on-rows/hot] | 1.0000 ± 0.0000 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[1].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_hot_loo [cooling/hottest-quarter] | 1.0000 ± 0.0000 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[2].score/.sd/.fit_seconds/.n_features` |
| fixed rule | robust_peer_z | 1.0000 ± 0.0000 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[3].score/.sd/.fit_seconds/.n_features` |
| fixed rule | robust_peer_z [all-on-rows/hot] | 1.0000 ± 0.0000 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[4].score/.sd/.fit_seconds/.n_features` |
| fixed rule | robust_peer_z [cooling/hottest-quarter] | 1.0000 ± 0.0000 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[5].score/.sd/.fit_seconds/.n_features` |
| fixed rule | above_setpoint_frac | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[6].score/.sd/.fit_seconds/.n_features` |
| fixed rule | above_setpoint_frac [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[7].score/.sd/.fit_seconds/.n_features` |
| fixed rule | above_setpoint_frac [cooling/all] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[8].score/.sd/.fit_seconds/.n_features` |
| fixed rule | above_setpoint_frac [cooling/hottest-quarter] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[9].score/.sd/.fit_seconds/.n_features` |
| fixed rule | cooling_duty | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[10].score/.sd/.fit_seconds/.n_features` |
| fixed rule | cooling_duty [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[11].score/.sd/.fit_seconds/.n_features` |
| fixed rule | cooling_duty [cooling/all] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[12].score/.sd/.fit_seconds/.n_features` |
| fixed rule | ctrl_residual | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[13].score/.sd/.fit_seconds/.n_features` |
| fixed rule | ctrl_residual [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[14].score/.sd/.fit_seconds/.n_features` |
| fixed rule | ctrl_residual [cooling/all] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[15].score/.sd/.fit_seconds/.n_features` |
| fixed rule | ctrl_residual_all | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[16].score/.sd/.fit_seconds/.n_features` |
| fixed rule | ctrl_residual_all [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[17].score/.sd/.fit_seconds/.n_features` |
| fixed rule | ctrl_residual_all [cooling/all] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[18].score/.sd/.fit_seconds/.n_features` |
| fixed rule | ctrl_residual_all [cooling/hottest-quarter] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[19].score/.sd/.fit_seconds/.n_features` |
| fixed rule | guo_residual | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[20].score/.sd/.fit_seconds/.n_features` |
| fixed rule | guo_residual [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[21].score/.sd/.fit_seconds/.n_features` |
| fixed rule | guo_residual [cooling/all] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[22].score/.sd/.fit_seconds/.n_features` |
| fixed rule | guo_residual [cooling/hottest-quarter] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[23].score/.sd/.fit_seconds/.n_features` |
| fixed rule | load_halved_share | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[24].score/.sd/.fit_seconds/.n_features` |
| fixed rule | load_halved_share [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[25].score/.sd/.fit_seconds/.n_features` |
| fixed rule | load_halved_share [cooling/all] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[26].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_all | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[27].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_all [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[28].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_all [cooling/all] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[29].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_all [cooling/hottest-quarter] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[30].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_hot | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.06 | 1 | off | [R251][R252][R253] | baseline, physics | `rows[31].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_hot [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.03 | 1 | off | [R251][R252][R253] | - | `rows[32].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_hot [cooling/all] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 1 | off | [R251][R252][R253] | - | `rows[33].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_hot [cooling/hottest-quarter] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 1 | off | [R251][R252][R253] | - | `rows[34].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_hot_loo [cooling/all] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[35].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_steady | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[36].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_steady [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[37].score/.sd/.fit_seconds/.n_features` |
| fixed rule | persistence_frac | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[38].score/.sd/.fit_seconds/.n_features` |
| fixed rule | persistence_frac [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[39].score/.sd/.fit_seconds/.n_features` |
| fixed rule | persistence_frac [cooling/hottest-quarter] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[40].score/.sd/.fit_seconds/.n_features` |
| fixed rule | robust_peer_z [cooling/all] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[41].score/.sd/.fit_seconds/.n_features` |
| fixed rule | unmet_degree_min | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[42].score/.sd/.fit_seconds/.n_features` |
| fixed rule | unmet_degree_min [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[43].score/.sd/.fit_seconds/.n_features` |
| fixed rule | unmet_degree_min [cooling/all] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[44].score/.sd/.fit_seconds/.n_features` |
| fixed rule | borda_guo_peer | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 2 | off | [R251][R252][R253] | - | `rows[45].score/.sd/.fit_seconds/.n_features` |
| fixed rule | borda_guo_peer [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 2 | off | [R251][R252][R253] | - | `rows[46].score/.sd/.fit_seconds/.n_features` |
| fixed rule | borda_guo_peer [cooling/all] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 2 | off | [R251][R252][R253] | - | `rows[47].score/.sd/.fit_seconds/.n_features` |
| fixed rule | borda_guo_peer [cooling/hottest-quarter] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 2 | off | [R251][R252][R253] | - | `rows[48].score/.sd/.fit_seconds/.n_features` |
| fixed rule | borda_peer_ctrl | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 2 | off | [R251][R252][R253] | - | `rows[49].score/.sd/.fit_seconds/.n_features` |
| fixed rule | borda_peer_ctrl [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 2 | off | [R251][R252][R253] | - | `rows[50].score/.sd/.fit_seconds/.n_features` |
| fixed rule | borda_peer_ctrl [cooling/all] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 2 | off | [R251][R252][R253] | - | `rows[51].score/.sd/.fit_seconds/.n_features` |
| fixed rule | borda_peer_ctrl [cooling/hottest-quarter] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.01 | 2 | off | [R251][R252][R253] | - | `rows[52].score/.sd/.fit_seconds/.n_features` |
| fixed rule | selected (nested, open pool (all 26 rankers)) | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 2.69 | 2 | off | [R251][R252][R253] | - | `rows[53].score/.sd/.fit_seconds/.n_features` |
| fixed rule | selected (nested, selection pool (peer family, 5)) | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.52 | 2 | off | [R251][R252][R253] | - | `rows[54].score/.sd/.fit_seconds/.n_features` |
| fixed rule | zsum_guo_peer | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 2 | off | [R251][R252][R253] | - | `rows[55].score/.sd/.fit_seconds/.n_features` |
| fixed rule | zsum_guo_peer [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 2 | off | [R251][R252][R253] | - | `rows[56].score/.sd/.fit_seconds/.n_features` |
| fixed rule | zsum_guo_peer [cooling/all] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 2 | off | [R251][R252][R253] | - | `rows[57].score/.sd/.fit_seconds/.n_features` |
| fixed rule | zsum_guo_peer [cooling/hottest-quarter] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 2 | off | [R251][R252][R253] | - | `rows[58].score/.sd/.fit_seconds/.n_features` |
| fixed rule | zsum_peer_ctrl | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 2 | off | [R251][R252][R253] | - | `rows[59].score/.sd/.fit_seconds/.n_features` |
| fixed rule | zsum_peer_ctrl [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 2 | off | [R251][R252][R253] | - | `rows[60].score/.sd/.fit_seconds/.n_features` |
| fixed rule | zsum_peer_ctrl [cooling/all] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 2 | off | [R251][R252][R253] | - | `rows[61].score/.sd/.fit_seconds/.n_features` |
| fixed rule | zsum_peer_ctrl [cooling/hottest-quarter] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 2 | off | [R251][R252][R253] | - | `rows[62].score/.sd/.fit_seconds/.n_features` |
| fixed rule | borda_peer_ctrl_unmet | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 3 | off | [R251][R252][R253] | - | `rows[63].score/.sd/.fit_seconds/.n_features` |
| fixed rule | borda_peer_ctrl_unmet [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 3 | off | [R251][R252][R253] | - | `rows[64].score/.sd/.fit_seconds/.n_features` |
| fixed rule | borda_peer_ctrl_unmet [cooling/all] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 3 | off | [R251][R252][R253] | - | `rows[65].score/.sd/.fit_seconds/.n_features` |
| fixed rule | borda_peer_ctrl_unmet [cooling/hottest-quarter] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 3 | off | [R251][R252][R253] | - | `rows[66].score/.sd/.fit_seconds/.n_features` |
| fixed rule | zsum_peer_ctrl_unmet | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 3 | off | [R251][R252][R253] | - | `rows[67].score/.sd/.fit_seconds/.n_features` |
| fixed rule | zsum_peer_ctrl_unmet [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 3 | off | [R251][R252][R253] | - | `rows[68].score/.sd/.fit_seconds/.n_features` |
| fixed rule | zsum_peer_ctrl_unmet [cooling/all] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.03 | 3 | off | [R251][R252][R253] | - | `rows[69].score/.sd/.fit_seconds/.n_features` |
| fixed rule | zsum_peer_ctrl_unmet [cooling/hottest-quarter] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 3 | off | [R251][R252][R253] | - | `rows[70].score/.sd/.fit_seconds/.n_features` |
| fixed rule | borda_all_five | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.03 | 5 | off | [R251][R252][R253] | - | `rows[71].score/.sd/.fit_seconds/.n_features` |
| fixed rule | borda_all_five [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 5 | off | [R251][R252][R253] | - | `rows[72].score/.sd/.fit_seconds/.n_features` |
| fixed rule | borda_all_five [cooling/all] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.03 | 5 | off | [R251][R252][R253] | - | `rows[73].score/.sd/.fit_seconds/.n_features` |
| fixed rule | borda_all_five [cooling/hottest-quarter] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.02 | 5 | off | [R251][R252][R253] | - | `rows[74].score/.sd/.fit_seconds/.n_features` |
| fixed rule | zsum_all_five | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.03 | 5 | off | [R251][R252][R253] | - | `rows[75].score/.sd/.fit_seconds/.n_features` |
| fixed rule | zsum_all_five [all-on-rows/hot] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.03 | 5 | off | [R251][R252][R253] | - | `rows[76].score/.sd/.fit_seconds/.n_features` |
| fixed rule | zsum_all_five [cooling/hottest-quarter] | 0.9792 ± 0.0510 | 6 cases; seeds [0] | 0.03 | 5 | off | [R251][R252][R253] | - | `rows[77].score/.sd/.fit_seconds/.n_features` |
| fixed rule | cooling_duty [cooling/hottest-quarter] | 0.9583 ± 0.1021 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[78].score/.sd/.fit_seconds/.n_features` |
| fixed rule | ctrl_residual [cooling/hottest-quarter] | 0.9583 ± 0.1021 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[79].score/.sd/.fit_seconds/.n_features` |
| fixed rule | load_halved_share [cooling/hottest-quarter] | 0.9583 ± 0.1021 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[80].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_steady [cooling/hottest-quarter] | 0.9583 ± 0.1021 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[81].score/.sd/.fit_seconds/.n_features` |
| fixed rule | unmet_degree_min [cooling/hottest-quarter] | 0.9583 ± 0.1021 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[82].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_steady [cooling/all] | 0.9583 ± 0.0645 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[83].score/.sd/.fit_seconds/.n_features` |
| fixed rule | persistence_frac [cooling/all] | 0.9583 ± 0.0645 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[84].score/.sd/.fit_seconds/.n_features` |
| fixed rule | zsum_all_five [cooling/all] | 0.9375 ± 0.1046 | 6 cases; seeds [0] | 0.03 | 5 | off | [R251][R252][R253] | - | `rows[85].score/.sd/.fit_seconds/.n_features` |
| fixed rule | guo_index | 0.9167 ± 0.1514 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[86].score/.sd/.fit_seconds/.n_features` |
| fixed rule | guo_index [all-on-rows/hot] | 0.9167 ± 0.1514 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[87].score/.sd/.fit_seconds/.n_features` |
| fixed rule | guo_index [cooling/all] | 0.9167 ± 0.1514 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[88].score/.sd/.fit_seconds/.n_features` |
| fixed rule | guo_index [cooling/hottest-quarter] | 0.9167 ± 0.1514 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[89].score/.sd/.fit_seconds/.n_features` |
| fixed rule | mode_switch_rate | 0.8750 ± 0.3062 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[90].score/.sd/.fit_seconds/.n_features` |
| fixed rule | mode_switch_rate [all-on-rows/hot] | 0.8750 ± 0.3062 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[91].score/.sd/.fit_seconds/.n_features` |
| fixed rule | mode_switch_rate [cooling/all] | 0.8750 ± 0.3062 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[92].score/.sd/.fit_seconds/.n_features` |
| fixed rule | mode_switch_rate [cooling/hottest-quarter] | 0.8750 ± 0.3062 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[93].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_trend [cooling/hottest-quarter] | 0.7917 ± 0.3416 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[94].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_trend | 0.7083 ± 0.3764 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[95].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_trend [all-on-rows/hot] | 0.7083 ± 0.3764 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[96].score/.sd/.fit_seconds/.n_features` |
| fixed rule | full_cooling_duty | 0.6875 ± 0.3236 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[97].score/.sd/.fit_seconds/.n_features` |
| fixed rule | full_cooling_duty [all-on-rows/hot] | 0.6875 ± 0.3236 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[98].score/.sd/.fit_seconds/.n_features` |
| fixed rule | full_cooling_duty [cooling/all] | 0.6875 ± 0.3236 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[99].score/.sd/.fit_seconds/.n_features` |
| fixed rule | full_cooling_duty [cooling/hottest-quarter] | 0.6875 ± 0.3236 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[100].score/.sd/.fit_seconds/.n_features` |
| fixed rule | peer_delta_trend [cooling/all] | 0.6667 ± 0.4306 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[101].score/.sd/.fit_seconds/.n_features` |
| fixed rule | pulldown_penalty | 0.3542 ± 0.3572 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[102].score/.sd/.fit_seconds/.n_features` |
| fixed rule | pulldown_penalty [all-on-rows/hot] | 0.3542 ± 0.3572 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[103].score/.sd/.fit_seconds/.n_features` |
| fixed rule | pulldown_penalty [cooling/all] | 0.3542 ± 0.3572 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[104].score/.sd/.fit_seconds/.n_features` |
| fixed rule | pulldown_penalty [cooling/hottest-quarter] | 0.3542 ± 0.3572 | 6 cases; seeds [0] | 0.01 | 1 | off | [R251][R252][R253] | - | `rows[105].score/.sd/.fit_seconds/.n_features` |
| chance | empty-cars-last blind ranking | 0.6042 ± n/a | fixed vocabulary / released set | n/a | n/a | off | - | chance floor | `chance_floor` |
| chance | uniform 8-car ranking | 0.5625 ± n/a | 6 released cases | n/a | n/a | off | - | chance floor | `chance_floor_uniform` |

Skipped from the addendum:

- Virtual refrigerant charge sensor: required liquid-line/suction-line temperatures and subcooling are absent [R262][R257].
- 3R2C grey-box + EKF identification: non-convex per-vehicle identification is unsupported by six cases [R254][R255].
- Supervised classifier / few-shot metric learning: only six labelled episodes exist [R268].
- Physics-simulation hybrid: the cited rail-HVAC work addresses a different fault and a cycle model is outside budget [R267].

## Rail corrugation — three-class classification

Frozen scheme: stratified 5-fold by file (grouped on the duplicate fingerprint) x seeds; seeds [0, 1, 2]. Honest headline uses 15 outer folds/cases (nested.macro_f1_mean/.macro_f1_sd).

| model | ablation / feature row | metric mean ± sd | CV folds / seeds | fit s | n feat | augmentation | addendum cite | flags | JSON key path |
|---|---|---:|---|---:|---:|---|---|---|---|
| logreg | baseline features | 0.7134 ± 0.1018 | 15 folds; seeds [0, 1, 2] | 0.95 | 528 | on | [R231][R232][R233][R237][R250] | - | `rows[0].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| svm | baseline features | 0.6799 ± 0.0830 | 15 folds; seeds [0, 1, 2] | 1.26 | 528 | on | [R231][R232][R233][R237][R250] | - | `rows[1].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| rf | baseline features | 0.6243 ± 0.1089 | 15 folds; seeds [0, 1, 2] | 4.37 | 528 | on | [R231][R232][R233][R237][R250] | - | `rows[2].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| lgbm | baseline features | 0.7311 ± 0.0896 | 15 folds; seeds [0, 1, 2] | 3.84 | 528 | on | [R231][R232][R233][R237][R250] | baseline, physics | `rows[3].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| lgbm | no v^2 normalisation | 0.7679 ± 0.0991 | 15 folds; seeds [0, 1, 2] | 3.57 | 528 | on | [R231][R232][R233][R237][R250] | - | `rows[4].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| lgbm | no v^2, + mirror TTA | 0.7817 ± 0.1125 | 15 folds; seeds [0, 1, 2] | 3.61 | 528 | on | [R231][R232][R233][R237][R250] | - | `rows[5].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| lgbm_hier | no v^2, hierarchical present/absent then side | 0.7348 ± 0.1143 | 15 folds; seeds [0, 1, 2] | 2.28 | 528 | on | [R232] | - | `rows[6].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| lgbm_hier | no v^2, hierarchical + mirror TTA | 0.7362 ± 0.1126 | 15 folds; seeds [0, 1, 2] | 2.31 | 528 | on | [R232] | - | `rows[7].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| lgbm | counter-design: Hz bands + speed, nothing distance-derived [R234] | 0.7940 ± 0.1166 | 15 folds; seeds [0, 1, 2] | 2.13 | 245 | on | [R234] | - | `rows[8].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| lgbm | Hz bands + speed, wavelength discriminators kept, + mirror TTA | 0.8007 ± 0.0987 | 15 folds; seeds [0, 1, 2] | 2.65 | 352 | on | [R234] | - | `rows[9].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| lgbm | no v^2, wavelength + Hz bands, + mirror TTA | 0.7844 ± 0.1036 | 15 folds; seeds [0, 1, 2] | 3.96 | 640 | on | [R234] | - | `rows[10].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| rf | no v^2, + mirror TTA | 0.7331 ± 0.0961 | 15 folds; seeds [0, 1, 2] | 4.42 | 528 | on | [R231][R232][R233][R237][R250] | - | `rows[11].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| logreg | no v^2, + mirror TTA | 0.7007 ± 0.0969 | 15 folds; seeds [0, 1, 2] | 0.96 | 528 | on | [R231][R232][R233][R237][R250] | - | `rows[12].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| multirocket_ridge | per-side wavelength spectra (4 x 192) | 0.4781 ± 0.1159 | 15 folds; seeds [0, 1, 2] | 12.96 | 768 | on | [R249][R43][R47] | - | `rows[13].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| multirocket_ridge | per-side wavelength spectra, no v^2 | 0.4960 ± 0.0982 | 15 folds; seeds [0, 1, 2] | 12.67 | 768 | on | [R249][R43][R47] | - | `rows[14].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| lgbm_2view | two-view ensemble: Hz-band model + wavelength model, no v^2, probability average, + mirror TTA | 0.7870 ± 0.1019 | 15 folds; seeds [0, 1, 2] | 5.79 | 640 | on | [R234][R231][R232][R233] | - | `rows[15].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| lgbm | + no shock channels | 0.8365 ± 0.0960 | 15 folds; seeds [0, 1, 2] | 1.91 | 180 | on | [R234] | - | `rows[16].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| lgbm | + no time-domain block | 0.7162 ± 0.0956 | 15 folds; seeds [0, 1, 2] | 2.21 | 224 | on | [R234] | - | `rows[17].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| lgbm | + no shock, no time block, no votes | 0.7720 ± 0.0855 | 15 folds; seeds [0, 1, 2] | 1.52 | 113 | on | [R234] | - | `rows[18].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| lgbm_pruned | + importance-pruned to 64 columns | 0.7799 ± 0.1203 | 15 folds; seeds [0, 1, 2] | 8.13 | 352 | on | [R234] | - | `rows[19].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| lgbm | Hz bands + speed, + fold-local speed-baseline residuals, + mirror TTA | 0.7928 ± 0.1001 | 15 folds; seeds [0, 1, 2] | 4.42 | 352 | on | [R234] | - | `rows[20].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| lgbm_windows | sub-window voting: 3 x 0.5 s windows, Hz bands + speed, + mirror TTA | 0.7153 ± 0.1438 | 15 folds; seeds [0, 1, 2] | 99.13 | 352 | on | [R249] | - | `rows[21].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| **lgbm** | **no shock + same-side axle-box coherence** | **0.8441 ± 0.1059** | 15 folds; seeds [0, 1, 2] | 2.11 | 201 | on | [R235][R246] | winner | `rows[22].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features` |
| chance | majority class (Normal) | 0.3083 ± 0.0000 | fixed vocabulary / released set | n/a | n/a | off | - | chance floor | `chance_floor.score/.sd` |

Winner stress splits (beside the stratified selection number):

- contiguous: 0.7817 ± 0.1349, 5 folds (`winner_schemes[0].macro_f1_mean/.macro_f1_sd/.n_folds`).
- speed_range: 0.7409 ± 0.1166, 3 folds (`winner_schemes[1].macro_f1_mean/.macro_f1_sd/.n_folds`).

Skipped from the addendum:

- **VMD / CEEMDAN / EWT / SPWVD adaptive decompositions** [R247]: per-record, parameter-heavy and slow, with no evidence they beat a wavelength-band PSD on a 3-class macro-F1 task at n=272.
- **Model-based roughness inversion** [R238][R239]: needs track receptance and rail-pad stiffness we do not have; the source's own sensitivity analysis moves the answer 3.5 dB for a 20 % pad-stiffness error.
- **Self-supervised / contrastive pre-training** [R244] and **sim-pretrain -> fine-tune** [R243]: the right shape for 14 labelled positives, but 234 unlabelled 1 s records are far too few for MoCo to pay and no corrugation simulator exists in this repo.
- **Deep 1D-CNN rows** [R240][R242]: a learnable front end on 272 files overfits, and the plan caps the deep tier at one row per family under a strict budget; `multirocket_ridge` is the row that family gets here.
- **GAN / diffusion augmentation**: the mirror swap is exact and free; generative augmentation on 14 spectra is strictly worse and adds a failure mode [R307][R308][R304]; `ps3_addendum.md` section 6 row 5.
- **Comb filter keyed to the wheel circumference** [R238]: implementable but optional at this budget; the side contrast captures most of the wheel-vs-rail separation.

## SHM — cumulative fatigue damage

Frozen scheme: repeated 5x10-fold outer, nested LOO selection; seeds [0, 1, 2, 3, 4]. Honest headline uses 64 outer folds/cases (nested.score/.score_sd).

| model | ablation / feature row | metric mean ± sd | CV folds / seeds | fit s | n feat | augmentation | addendum cite | flags | JSON key path |
|---|---|---:|---|---:|---:|---|---|---|---|
| rainflow_sn | rainflow_sn/rainflow | 0.9718 ± 0.0004 | 50 folds; seeds [0, 1, 2, 3, 4] | 1.28 | 10 | off | [R269][R270] | - | `rows[0].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| rainflow_sn_fixed | rainflow_sn_fixed/rainflow | 0.9729 ± 0.0002 | 50 folds; seeds [0, 1, 2, 3, 4] | 1.22 | 10 | off | [R269][R270] | physics | `rows[1].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| rainflow_sn | rainflow_sn/rainflow/bias | 0.9729 ± 0.0004 | 50 folds; seeds [0, 1, 2, 3, 4] | 1.99 | 10 | off | [R269][R270] | - | `rows[2].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| lasso_log | lasso_log/stats | 0.8803 ± 0.0048 | 50 folds; seeds [0, 1, 2, 3, 4] | 5.30 | 20 | off | [R272][R273] | baseline | `rows[3].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| lasso_log | lasso_log/rainflow | 0.9771 ± 0.0007 | 50 folds; seeds [0, 1, 2, 3, 4] | 4.58 | 40 | off | [R272][R273] | - | `rows[4].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| lasso_log | lasso_log/spectral | 0.7510 ± 0.0069 | 50 folds; seeds [0, 1, 2, 3, 4] | 6.45 | 27 | off | [R269][R275][R276] | - | `rows[5].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| lasso_log | lasso_log/fds | 0.7473 ± 0.0132 | 50 folds; seeds [0, 1, 2, 3, 4] | 9.27 | 22 | off | [R274] | - | `rows[6].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| lasso_log | lasso_log/rainflow+spectral | 0.9766 ± 0.0008 | 50 folds; seeds [0, 1, 2, 3, 4] | 7.73 | 67 | off | [R269][R275][R276] | - | `rows[7].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| lasso_log | lasso_log/stats+rainflow+spectral+fds | 0.9805 ± 0.0005 | 50 folds; seeds [0, 1, 2, 3, 4] | 8.70 | 109 | off | [R269][R275][R276] | - | `rows[8].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| ridge_log | ridge_log/stats | 0.8809 ± 0.0083 | 50 folds; seeds [0, 1, 2, 3, 4] | 1.48 | 20 | off | [R272][R273] | - | `rows[9].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| ridge_log | ridge_log/rainflow | 0.9781 ± 0.0012 | 50 folds; seeds [0, 1, 2, 3, 4] | 5.98 | 40 | off | [R272][R273] | - | `rows[10].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| ridge_log | ridge_log/spectral | 0.7480 ± 0.0061 | 50 folds; seeds [0, 1, 2, 3, 4] | 1.60 | 27 | off | [R269][R275][R276] | - | `rows[11].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| ridge_log | ridge_log/fds | 0.8017 ± 0.0070 | 50 folds; seeds [0, 1, 2, 3, 4] | 1.63 | 22 | off | [R274] | - | `rows[12].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| ridge_log | ridge_log/rainflow+spectral | 0.9738 ± 0.0007 | 50 folds; seeds [0, 1, 2, 3, 4] | 1.83 | 67 | off | [R269][R275][R276] | - | `rows[13].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| ridge_log | ridge_log/stats+rainflow+spectral+fds | 0.9705 ± 0.0020 | 50 folds; seeds [0, 1, 2, 3, 4] | 1.70 | 109 | off | [R269][R275][R276] | - | `rows[14].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| elastic_log | elastic_log/stats | 0.8795 ± 0.0050 | 50 folds; seeds [0, 1, 2, 3, 4] | 11.62 | 20 | off | [R272][R273] | - | `rows[15].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| elastic_log | elastic_log/rainflow | 0.9770 ± 0.0008 | 50 folds; seeds [0, 1, 2, 3, 4] | 16.35 | 40 | off | [R272][R273] | - | `rows[16].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| elastic_log | elastic_log/spectral | 0.7527 ± 0.0053 | 50 folds; seeds [0, 1, 2, 3, 4] | 19.04 | 27 | off | [R269][R275][R276] | - | `rows[17].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| elastic_log | elastic_log/fds | 0.7445 ± 0.0124 | 50 folds; seeds [0, 1, 2, 3, 4] | 16.73 | 22 | off | [R274] | - | `rows[18].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| elastic_log | elastic_log/rainflow+spectral | 0.9765 ± 0.0010 | 50 folds; seeds [0, 1, 2, 3, 4] | 41.64 | 67 | off | [R269][R275][R276] | - | `rows[19].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| elastic_log | elastic_log/stats+rainflow+spectral+fds | 0.9799 ± 0.0004 | 50 folds; seeds [0, 1, 2, 3, 4] | 66.71 | 109 | off | [R269][R275][R276] | - | `rows[20].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| lgbm | lgbm/stats | 0.8619 ± 0.0125 | 50 folds; seeds [0, 1, 2, 3, 4] | 4.31 | 20 | off | [R272][R273] | - | `rows[21].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| lgbm | lgbm/rainflow | 0.9229 ± 0.0105 | 50 folds; seeds [0, 1, 2, 3, 4] | 5.25 | 40 | off | [R272][R273] | - | `rows[22].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| lgbm | lgbm/spectral | 0.7504 ± 0.0208 | 50 folds; seeds [0, 1, 2, 3, 4] | 4.76 | 27 | off | [R269][R275][R276] | - | `rows[23].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| lgbm | lgbm/fds | 0.7676 ± 0.0087 | 50 folds; seeds [0, 1, 2, 3, 4] | 4.50 | 22 | off | [R274] | - | `rows[24].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| lgbm | lgbm/rainflow+spectral | 0.9259 ± 0.0056 | 50 folds; seeds [0, 1, 2, 3, 4] | 6.68 | 67 | off | [R269][R275][R276] | - | `rows[25].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| lgbm | lgbm/stats+rainflow+spectral+fds | 0.9262 ± 0.0038 | 50 folds; seeds [0, 1, 2, 3, 4] | 8.58 | 109 | off | [R269][R275][R276] | - | `rows[26].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| lasso_log | lasso_log/rainflow/rawtarget | 0.4091 ± 0.0351 | 50 folds; seeds [0, 1, 2, 3, 4] | 29.94 | 40 | off | [R272][R273] | - | `rows[27].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| lasso_log | lasso_log/rainflow/bias | 0.9771 ± 0.0009 | 50 folds; seeds [0, 1, 2, 3, 4] | 5.87 | 40 | off | [R272][R273] | - | `rows[28].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| lasso_log | lasso_log/rainflow/cmixup128 | 0.9767 ± 0.0009 | 50 folds; seeds [0, 1, 2, 3, 4] | 5.53 | 40 | on | [R299] | - | `rows[29].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| lasso_log | lasso_log/stats+rainflow+spectral+fds/rawtarget | 0.3809 ± 0.0175 | 50 folds; seeds [0, 1, 2, 3, 4] | 191.95 | 109 | off | [R269][R275][R276] | - | `rows[30].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| lasso_log | lasso_log/stats+rainflow+spectral+fds/bias | 0.9805 ± 0.0003 | 50 folds; seeds [0, 1, 2, 3, 4] | 15.70 | 109 | off | [R269][R275][R276] | - | `rows[31].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| lasso_log | lasso_log/stats+rainflow+spectral+fds/cmixup128 | 0.9801 ± 0.0010 | 50 folds; seeds [0, 1, 2, 3, 4] | 8.86 | 109 | on | [R269][R275][R276] | - | `rows[32].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| ridge_log | ridge_log/rainflow/rawtarget | 0.4158 ± 0.0228 | 50 folds; seeds [0, 1, 2, 3, 4] | 9.51 | 40 | off | [R272][R273] | - | `rows[33].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| ridge_log | ridge_log/rainflow/bias | 0.9780 ± 0.0011 | 50 folds; seeds [0, 1, 2, 3, 4] | 9.51 | 40 | off | [R272][R273] | - | `rows[34].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| ridge_log | ridge_log/rainflow/cmixup128 | 0.9761 ± 0.0017 | 50 folds; seeds [0, 1, 2, 3, 4] | 8.72 | 40 | on | [R299] | - | `rows[35].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| ridge_log | ridge_log/stats+rainflow+spectral+fds/rawtarget | 0.4007 ± 0.0460 | 50 folds; seeds [0, 1, 2, 3, 4] | 2.30 | 109 | off | [R269][R275][R276] | - | `rows[36].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| ridge_log | ridge_log/stats+rainflow+spectral+fds/bias | 0.9706 ± 0.0020 | 50 folds; seeds [0, 1, 2, 3, 4] | 3.55 | 109 | off | [R269][R275][R276] | - | `rows[37].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| ridge_log | ridge_log/stats+rainflow+spectral+fds/cmixup128 | 0.9607 ± 0.0026 | 50 folds; seeds [0, 1, 2, 3, 4] | 101.44 | 109 | on | [R269][R275][R276] | - | `rows[38].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| **lasso_log** | **lasso_log/stats+rainflow+spectral+fds/bias/physblend50_skewpos** | **0.9813 ± 0.0002** | 50 folds; seeds [0, 1, 2, 3, 4] | 9.42 | 109 | off | [R269][R275][R276] | winner | `rows[39].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features` |
| chance | constant training-label median | 0.0849 ± n/a | fixed vocabulary / released set | n/a | n/a | off | - | chance floor | `chance_floor.score/.sd` |

Skipped from the addendum:

- 1D-CNN / LSTM / TCN on 581k raw samples: 64 files are insufficient and the nearest study gains about 0.003 R2 [R272].
- Multiaxial / critical-plane criteria: only one stress channel is available.
- Rainflow-matrix extrapolation: the record is the label domain, not a shorter history to extrapolate [R279].
- ffpack: GPL-3.0 is incompatible with the shipped deliverable [R269].

## Selected per subsystem

| subsystem | metric | honest nested/outer headline | folds/cases | seeds | artifact | JSON key path |
|---|---|---:|---:|---|---|---|
| door | IoU-weighted F1 | **0.9818 ± 0.0364** | 5 | [0] | `models/ps3/door.pkl` | `nested.iou_f1_mean/.iou_f1_sd` |
| acv | rank decay | **0.9792 ± 0.0510** | 6 | [0] | `models/ps3/acv.pkl` | `selected.score/.sd` |
| rail | macro F1 | **0.8051 ± 0.1291** | 15 | [0, 1, 2] | `models/ps3/rail.pkl` | `nested.macro_f1_mean/.macro_f1_sd` |
| shm | 1 − MAPE | **0.9813** | 64 | [0, 1, 2, 3, 4] | `models/ps3/shm.pkl` | `nested.score/.score_sd` |
