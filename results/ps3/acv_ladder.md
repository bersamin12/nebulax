# ACV ladder - leave-one-case-out rank decay (exploratory, 6 cases)

Feature version `acv-f5`, git `33d00a2`, 4.69 s wall, 106 rows.

Every row is a **rule**, not a fitted model: with 6 cases x 8 cars and one positive per case there is nothing to train a classifier on, and a classifier would be free to learn a car-id prior, which is exactly wrong here. A fixed rule has no parameters to fit inside a fold, so applying it to all six cases *is* its leave-one-case-out score. The two rows re-run rule choice within each held-out case, but neither is a pre-registered headline: the open pool is exploratory and the five-rule restricted pool was defined after inspecting the six-case ladder. The only headline is the pre-registered fixed `baseline_peer_delta_hot`, 0.9792 +/- 0.051 over six exploratory cases (seed-free and deterministic); the 1.0000 row is post-hoc selection evidence.

**No row in this table is the committed headline.** `train()` does not promote a ladder winner: every fixed row here is scored on the same six cases the table is read on, so the top row (`peer_delta_hot_loo`, 1.0000) is **selected on all six cases**, not a held-out number. The committed artefact and the headline in `acv_cv.md` are the pre-registered baseline rule at 0.9792 +/- 0.051. Both produce the identical row for the organisers' Test file.

**Why the fold selects from a five-rule pool, not from all 26.** The `tied at top` column below counts how many candidates were tied at the best training rank decay when that fold made its choice. With the open pool most of it ties on five cases (on one fold 21 of the 26 open-pool candidates are tied), so the pick rests on `acv.separation` (the margin the rule puts between the labelled car and its peers on the training cases), a statistic one wild car can dominate; `acv.selection_pool` therefore restricts the pool to one physically motivated family - peer-normalised indoor-temperature deltas, the primary observable when the telemetry has no pressures - and pre-registers its order (strict leave-one-car-out reference first, as [C1] and [R78] prescribe). Both nested rows land on 0.9792, i.e. on the baseline. That restriction was made after seeing this table, which is the other reason the headline is the baseline and not a row from here.

Chance floor: a signal-free ranker that leaves the empty cars last scores **0.6042** on these six cases (`acv_case_04.xlsx` has four entirely empty cars, so a blind ranking of it already scores 0.8125); a uniformly random ranking of 8 cars scores 0.5625. sd is over the 6 cases, not over seeds (the rules are deterministic).

## Main table (cooling-mode rows, hotter half of the record)

| rank | row | score | sd | top-1 | mean rank of true car | n feat | fit s | note |
|---|---|---|---|---|---|---|---|---|
| 1 | **`peer_delta_hot_loo`** | **1.0000** | 0.0000 | 1.000 | 1.00 | 1 | 0.010 | selected on all six cases (not held out; not the artefact) |
| 2 | **`robust_peer_z`** | **1.0000** | 0.0000 | 1.000 | 1.00 | 1 | 0.010 | tie-break only (<= 3 distinct values per case) |
| 3 | `above_setpoint_frac` | 0.9792 | 0.0510 | 0.833 | 1.17 | 1 | 0.010 |  |
| 4 | `cooling_duty` | 0.9792 | 0.0510 | 0.833 | 1.17 | 1 | 0.010 | tie-break only (constant) |
| 5 | `ctrl_residual` | 0.9792 | 0.0510 | 0.833 | 1.17 | 1 | 0.010 |  |
| 6 | `ctrl_residual_all` | 0.9792 | 0.0510 | 0.833 | 1.17 | 1 | 0.010 |  |
| 7 | `guo_residual` | 0.9792 | 0.0510 | 0.833 | 1.17 | 1 | 0.010 |  |
| 8 | `load_halved_share` | 0.9792 | 0.0510 | 0.833 | 1.17 | 1 | 0.013 | tie-break only (constant) |
| 9 | `peer_delta_all` | 0.9792 | 0.0510 | 0.833 | 1.17 | 1 | 0.011 |  |
| 10 | `peer_delta_hot` | 0.9792 | 0.0510 | 0.833 | 1.17 | 1 | 0.058 |  |
| 11 | `peer_delta_steady` | 0.9792 | 0.0510 | 0.833 | 1.17 | 1 | 0.011 |  |
| 12 | `persistence_frac` | 0.9792 | 0.0510 | 0.833 | 1.17 | 1 | 0.010 |  |
| 13 | `unmet_degree_min` | 0.9792 | 0.0510 | 0.833 | 1.17 | 1 | 0.010 |  |
| 14 | `borda_guo_peer` | 0.9792 | 0.0510 | 0.833 | 1.17 | 2 | 0.016 |  |
| 15 | `borda_peer_ctrl` | 0.9792 | 0.0510 | 0.833 | 1.17 | 2 | 0.016 |  |
| 16 | `selected (nested, open pool (all 26 rankers))` | 0.9792 | 0.0510 | 0.833 | 1.17 | 2 | 2.685 | rule re-chosen per fold |
| 17 | `selected (nested, selection pool (peer family, 5))` | 0.9792 | 0.0510 | 0.833 | 1.17 | 2 | 0.519 | rule re-chosen per fold |
| 18 | `zsum_guo_peer` | 0.9792 | 0.0510 | 0.833 | 1.17 | 2 | 0.017 |  |
| 19 | `zsum_peer_ctrl` | 0.9792 | 0.0510 | 0.833 | 1.17 | 2 | 0.017 |  |
| 20 | `borda_peer_ctrl_unmet` | 0.9792 | 0.0510 | 0.833 | 1.17 | 3 | 0.019 |  |
| 21 | `zsum_peer_ctrl_unmet` | 0.9792 | 0.0510 | 0.833 | 1.17 | 3 | 0.021 |  |
| 22 | `borda_all_five` | 0.9792 | 0.0510 | 0.833 | 1.17 | 5 | 0.025 |  |
| 23 | `zsum_all_five` | 0.9792 | 0.0510 | 0.833 | 1.17 | 5 | 0.028 |  |
| 24 | `guo_index` | 0.9167 | 0.1514 | 0.667 | 1.67 | 1 | 0.010 |  |
| 25 | `mode_switch_rate` | 0.8750 | 0.3062 | 0.833 | 2.00 | 1 | 0.011 |  |
| 26 | `peer_delta_trend` | 0.7083 | 0.3764 | 0.500 | 3.33 | 1 | 0.010 |  |
| 27 | `full_cooling_duty` | 0.6875 | 0.3236 | 0.333 | 3.50 | 1 | 0.010 | tie-break only (<= 3 distinct values per case) |
| 28 | `pulldown_penalty` | 0.3542 | 0.3572 | 0.000 | 6.17 | 1 | 0.012 |  |

## Mask x season-half ablation

`cooling/hot` = cooling-mode rows in the hotter half of the record (default); `cooling/all` = every cooling row; `cooling/hottest-quarter` = the top 25 % of ambient; `all-on-rows/hot` = every row the unit is switched on for, cooling or not.

| row | score | sd | top-1 | note |
|---|---|---|---|---|
| `peer_delta_hot_loo [all-on-rows/hot]` | 1.0000 | 0.0000 | 1.000 |  |
| `peer_delta_hot_loo [cooling/hottest-quarter]` | 1.0000 | 0.0000 | 1.000 |  |
| `robust_peer_z [all-on-rows/hot]` | 1.0000 | 0.0000 | 1.000 | tie-break only (<= 3 distinct values per case) |
| `robust_peer_z [cooling/hottest-quarter]` | 1.0000 | 0.0000 | 1.000 | tie-break only (<= 3 distinct values per case) |
| `above_setpoint_frac [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 |  |
| `above_setpoint_frac [cooling/all]` | 0.9792 | 0.0510 | 0.833 |  |
| `above_setpoint_frac [cooling/hottest-quarter]` | 0.9792 | 0.0510 | 0.833 |  |
| `cooling_duty [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 | tie-break only (constant) |
| `cooling_duty [cooling/all]` | 0.9792 | 0.0510 | 0.833 | tie-break only (constant) |
| `ctrl_residual [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 |  |
| `ctrl_residual [cooling/all]` | 0.9792 | 0.0510 | 0.833 |  |
| `ctrl_residual_all [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 |  |
| `ctrl_residual_all [cooling/all]` | 0.9792 | 0.0510 | 0.833 |  |
| `ctrl_residual_all [cooling/hottest-quarter]` | 0.9792 | 0.0510 | 0.833 |  |
| `guo_residual [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 |  |
| `guo_residual [cooling/all]` | 0.9792 | 0.0510 | 0.833 |  |
| `guo_residual [cooling/hottest-quarter]` | 0.9792 | 0.0510 | 0.833 |  |
| `load_halved_share [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 | tie-break only (constant) |
| `load_halved_share [cooling/all]` | 0.9792 | 0.0510 | 0.833 | tie-break only (constant) |
| `peer_delta_all [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 |  |
| `peer_delta_all [cooling/all]` | 0.9792 | 0.0510 | 0.833 |  |
| `peer_delta_all [cooling/hottest-quarter]` | 0.9792 | 0.0510 | 0.833 |  |
| `peer_delta_hot [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 |  |
| `peer_delta_hot [cooling/all]` | 0.9792 | 0.0510 | 0.833 |  |
| `peer_delta_hot [cooling/hottest-quarter]` | 0.9792 | 0.0510 | 0.833 |  |
| `peer_delta_hot_loo [cooling/all]` | 0.9792 | 0.0510 | 0.833 |  |
| `peer_delta_steady [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 |  |
| `persistence_frac [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 |  |
| `persistence_frac [cooling/hottest-quarter]` | 0.9792 | 0.0510 | 0.833 |  |
| `robust_peer_z [cooling/all]` | 0.9792 | 0.0510 | 0.833 | tie-break only (<= 3 distinct values per case) |
| `unmet_degree_min [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 |  |
| `unmet_degree_min [cooling/all]` | 0.9792 | 0.0510 | 0.833 |  |
| `borda_guo_peer [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 |  |
| `borda_guo_peer [cooling/all]` | 0.9792 | 0.0510 | 0.833 |  |
| `borda_guo_peer [cooling/hottest-quarter]` | 0.9792 | 0.0510 | 0.833 |  |
| `borda_peer_ctrl [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 |  |
| `borda_peer_ctrl [cooling/all]` | 0.9792 | 0.0510 | 0.833 |  |
| `borda_peer_ctrl [cooling/hottest-quarter]` | 0.9792 | 0.0510 | 0.833 |  |
| `zsum_guo_peer [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 |  |
| `zsum_guo_peer [cooling/all]` | 0.9792 | 0.0510 | 0.833 |  |
| `zsum_guo_peer [cooling/hottest-quarter]` | 0.9792 | 0.0510 | 0.833 |  |
| `zsum_peer_ctrl [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 |  |
| `zsum_peer_ctrl [cooling/all]` | 0.9792 | 0.0510 | 0.833 |  |
| `zsum_peer_ctrl [cooling/hottest-quarter]` | 0.9792 | 0.0510 | 0.833 |  |
| `borda_peer_ctrl_unmet [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 |  |
| `borda_peer_ctrl_unmet [cooling/all]` | 0.9792 | 0.0510 | 0.833 |  |
| `borda_peer_ctrl_unmet [cooling/hottest-quarter]` | 0.9792 | 0.0510 | 0.833 |  |
| `zsum_peer_ctrl_unmet [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 |  |
| `zsum_peer_ctrl_unmet [cooling/all]` | 0.9792 | 0.0510 | 0.833 |  |
| `zsum_peer_ctrl_unmet [cooling/hottest-quarter]` | 0.9792 | 0.0510 | 0.833 |  |
| `borda_all_five [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 |  |
| `borda_all_five [cooling/all]` | 0.9792 | 0.0510 | 0.833 |  |
| `borda_all_five [cooling/hottest-quarter]` | 0.9792 | 0.0510 | 0.833 |  |
| `zsum_all_five [all-on-rows/hot]` | 0.9792 | 0.0510 | 0.833 |  |
| `zsum_all_five [cooling/hottest-quarter]` | 0.9792 | 0.0510 | 0.833 |  |
| `cooling_duty [cooling/hottest-quarter]` | 0.9583 | 0.1021 | 0.833 | tie-break only (constant) |
| `ctrl_residual [cooling/hottest-quarter]` | 0.9583 | 0.1021 | 0.833 |  |
| `load_halved_share [cooling/hottest-quarter]` | 0.9583 | 0.1021 | 0.833 | tie-break only (constant) |
| `peer_delta_steady [cooling/hottest-quarter]` | 0.9583 | 0.1021 | 0.833 |  |
| `unmet_degree_min [cooling/hottest-quarter]` | 0.9583 | 0.1021 | 0.833 |  |
| `peer_delta_steady [cooling/all]` | 0.9583 | 0.0645 | 0.667 |  |
| `persistence_frac [cooling/all]` | 0.9583 | 0.0645 | 0.667 |  |
| `zsum_all_five [cooling/all]` | 0.9375 | 0.1046 | 0.667 |  |
| `guo_index [all-on-rows/hot]` | 0.9167 | 0.1514 | 0.667 |  |
| `guo_index [cooling/all]` | 0.9167 | 0.1514 | 0.667 |  |
| `guo_index [cooling/hottest-quarter]` | 0.9167 | 0.1514 | 0.667 |  |
| `mode_switch_rate [all-on-rows/hot]` | 0.8750 | 0.3062 | 0.833 |  |
| `mode_switch_rate [cooling/all]` | 0.8750 | 0.3062 | 0.833 |  |
| `mode_switch_rate [cooling/hottest-quarter]` | 0.8750 | 0.3062 | 0.833 |  |
| `peer_delta_trend [cooling/hottest-quarter]` | 0.7917 | 0.3416 | 0.500 |  |
| `peer_delta_trend [all-on-rows/hot]` | 0.7083 | 0.3764 | 0.500 |  |
| `full_cooling_duty [all-on-rows/hot]` | 0.6875 | 0.3236 | 0.333 | tie-break only (<= 3 distinct values per case) |
| `full_cooling_duty [cooling/all]` | 0.6875 | 0.3236 | 0.333 | tie-break only (<= 3 distinct values per case) |
| `full_cooling_duty [cooling/hottest-quarter]` | 0.6875 | 0.3236 | 0.333 | tie-break only (<= 3 distinct values per case) |
| `peer_delta_trend [cooling/all]` | 0.6667 | 0.4306 | 0.500 |  |
| `pulldown_penalty [all-on-rows/hot]` | 0.3542 | 0.3572 | 0.000 |  |
| `pulldown_penalty [cooling/all]` | 0.3542 | 0.3572 | 0.000 |  |
| `pulldown_penalty [cooling/hottest-quarter]` | 0.3542 | 0.3572 | 0.000 |  |

## Per-fold selection

A fold's rule is chosen on its five training cases by rank decay, then top-1, then `acv.separation` (the training margin between the labelled car and its peers, in MAD units), then the fewest features, then the pool order. `tied at top` counts the candidates the first two criteria could not separate - the reason the third exists.

| pool | held-out case | rule chosen on the other five | tied at top | train separation |
|---|---|---|---|---|
| selected (nested, open pool (all 26 rankers)) | `acv_case_01.xlsx` | `peer_delta_hot_loo` | 2 of 26 | 4.49 |
| selected (nested, open pool (all 26 rankers)) | `acv_case_02.xlsx` | `peer_delta_hot_loo` | 2 of 26 | 5.59 |
| selected (nested, open pool (all 26 rankers)) | `acv_case_03.xlsx` | `peer_delta_hot_loo` | 2 of 26 | 4.98 |
| selected (nested, open pool (all 26 rankers)) | `acv_case_04.xlsx` | `peer_delta_steady` | 21 of 26 | 9.04 |
| selected (nested, open pool (all 26 rankers)) | `acv_case_05.xlsx` | `peer_delta_hot_loo` | 2 of 26 | 5.62 |
| selected (nested, open pool (all 26 rankers)) | `acv_case_06.xlsx` | `peer_delta_hot_loo` | 3 of 26 | 2.98 |
| selected (nested, selection pool (peer family, 5)) | `acv_case_01.xlsx` | `peer_delta_hot_loo` | 3 of 5 | 4.49 |
| selected (nested, selection pool (peer family, 5)) | `acv_case_02.xlsx` | `peer_delta_hot_loo` | 3 of 5 | 5.59 |
| selected (nested, selection pool (peer family, 5)) | `acv_case_03.xlsx` | `peer_delta_hot_loo` | 3 of 5 | 4.98 |
| selected (nested, selection pool (peer family, 5)) | `acv_case_04.xlsx` | `peer_delta_steady` | 5 of 5 | 9.04 |
| selected (nested, selection pool (peer family, 5)) | `acv_case_05.xlsx` | `peer_delta_hot_loo` | 3 of 5 | 5.62 |
| selected (nested, selection pool (peer family, 5)) | `acv_case_06.xlsx` | `peer_delta_hot_loo` | 3 of 5 | 2.98 |

## Skipped for budget, and why

* **Supervised classifiers / deep models** - 6 cases x 8 cars = 48 rows with 6 positives. The finder's explicit "do not" ([C1] practitioner notes); nothing to gain and a car-id prior to lose.
* **Virtual refrigerant-charge sensors and RP-1043-style decoupling features** ([C7], [C8]) - they need subcooling, superheat, pressures or power. Five of the seven files carry eight temperature/mode parameters per car and none of those signals; only `acv_case_04.xlsx` has pressures, and the **test file does not**, so a feature built on them could never run on the held-out case.
* **3R2C grey-box + extended Kalman filter** ([C4]) - 4-6 h and non-convex identification for a pull-down feature the seven peer cars already supply for free; the cheap version (`pulldown_penalty`) is in the table.
* **Per-position (car-id) offset correction** - the finder suggests subtracting a per-position feature offset when the faulty car is not uniform across cases. It is not uniform here (cars 01, 02, 03, 01, 04, 06), but the offset would have to be estimated from training cases whose own faulty car sits in the same skewed positions, so it would systematically discount exactly the cars that are usually faulty. Deliberately **not implemented**: the ranker must never carry a car-id prior, in either direction.
