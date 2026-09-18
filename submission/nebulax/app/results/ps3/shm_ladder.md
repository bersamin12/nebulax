# SHM ladder - cumulative fatigue damage

This round reuses the unchanged 39 historical shm-f2 rows, adds one positive-skew physics blend, and reruns six-row nested LOO selection. The organiser's prior SHM score was 0.971752844618028; the blend subsequently scored 0.972795 on the reported 22:18 submission ZIP.

Selection metric: repeated 5x10-fold outer MAPE (mean +- sd over the 5 repeats, seeds [0, 1, 2, 3, 4]).
LOO is reported alongside. git rev `9cfb2f5`, features `shm-f2`,
1212.62 s wall for 40 rows.

| # | model / features / ablation | n feat | 5x10-fold MAPE | score | LOO MAPE | LOO low mode | LOO high mode | fit s |
|---|---|---|---|---|---|---|---|---|
| 1 | `lasso_log/stats+rainflow+spectral+fds/bias/physblend50_skewpos` | 109 | 0.0187 +- 0.0002 | 0.9813 | 0.0187 | 0.0206 | 0.0144 | 9.4 |
| 2 | `lasso_log/stats+rainflow+spectral+fds/bias` | 109 | 0.0195 +- 0.0003 | 0.9805 | 0.0198 | 0.0220 | 0.0147 | 15.7 |
| 3 | `lasso_log/stats+rainflow+spectral+fds` | 109 | 0.0195 +- 0.0005 | 0.9805 | 0.0198 | 0.0217 | 0.0152 | 8.7 |
| 4 | `lasso_log/stats+rainflow+spectral+fds/cmixup128` | 109 | 0.0199 +- 0.0010 | 0.9801 | 0.0197 | 0.0217 | 0.0149 | 8.9 |
| 5 | `elastic_log/stats+rainflow+spectral+fds` | 109 | 0.0201 +- 0.0004 | 0.9799 | 0.0197 | 0.0221 | 0.0140 | 66.7 |
| 6 | `ridge_log/rainflow` | 40 | 0.0219 +- 0.0012 | 0.9781 | 0.0218 | 0.0254 | 0.0133 | 6.0 |
| 7 | `ridge_log/rainflow/bias` | 40 | 0.0220 +- 0.0011 | 0.9780 | 0.0216 | 0.0253 | 0.0130 | 9.5 |
| 8 | `lasso_log/rainflow` | 40 | 0.0229 +- 0.0007 | 0.9771 | 0.0227 | 0.0230 | 0.0221 | 4.6 |
| 9 | `lasso_log/rainflow/bias` | 40 | 0.0229 +- 0.0009 | 0.9771 | 0.0226 | 0.0232 | 0.0212 | 5.9 |
| 10 | `elastic_log/rainflow` | 40 | 0.0230 +- 0.0008 | 0.9770 | 0.0229 | 0.0234 | 0.0218 | 16.4 |
| 11 | `lasso_log/rainflow/cmixup128` | 40 | 0.0233 +- 0.0009 | 0.9767 | 0.0226 | 0.0228 | 0.0219 | 5.5 |
| 12 | `lasso_log/rainflow+spectral` | 67 | 0.0234 +- 0.0008 | 0.9766 | 0.0231 | 0.0235 | 0.0224 | 7.7 |
| 13 | `elastic_log/rainflow+spectral` | 67 | 0.0235 +- 0.0010 | 0.9765 | 0.0228 | 0.0231 | 0.0219 | 41.6 |
| 14 | `ridge_log/rainflow/cmixup128` | 40 | 0.0239 +- 0.0017 | 0.9761 | 0.0231 | 0.0267 | 0.0147 | 8.7 |
| 15 | `ridge_log/rainflow+spectral` | 67 | 0.0262 +- 0.0007 | 0.9738 | 0.0244 | 0.0277 | 0.0165 | 1.8 |
| 16 | `rainflow_sn/rainflow/bias` | 10 | 0.0271 +- 0.0004 | 0.9729 | 0.0268 | 0.0303 | 0.0187 | 2.0 |
| 17 | `rainflow_sn_fixed/rainflow` | 10 | 0.0271 +- 0.0002 | 0.9729 | 0.0271 | 0.0318 | 0.0162 | 1.2 |
| 18 | `rainflow_sn/rainflow` | 10 | 0.0282 +- 0.0004 | 0.9718 | 0.0283 | 0.0330 | 0.0169 | 1.3 |
| 19 | `ridge_log/stats+rainflow+spectral+fds/bias` | 109 | 0.0294 +- 0.0020 | 0.9706 | 0.0267 | 0.0318 | 0.0147 | 3.5 |
| 20 | `ridge_log/stats+rainflow+spectral+fds` | 109 | 0.0295 +- 0.0020 | 0.9705 | 0.0267 | 0.0318 | 0.0146 | 1.7 |
| 21 | `ridge_log/stats+rainflow+spectral+fds/cmixup128` | 109 | 0.0393 +- 0.0026 | 0.9607 | 0.0346 | 0.0406 | 0.0206 | 101.4 |
| 22 | `lgbm/stats+rainflow+spectral+fds` | 109 | 0.0738 +- 0.0038 | 0.9262 | 0.0786 | 0.0561 | 0.1320 | 8.6 |
| 23 | `lgbm/rainflow+spectral` | 67 | 0.0741 +- 0.0056 | 0.9259 | 0.0680 | 0.0552 | 0.0981 | 6.7 |
| 24 | `lgbm/rainflow` | 40 | 0.0771 +- 0.0105 | 0.9229 | 0.0736 | 0.0547 | 0.1184 | 5.2 |
| 25 | `ridge_log/stats` | 20 | 0.1191 +- 0.0083 | 0.8809 | 0.1172 | 0.1396 | 0.0641 | 1.5 |
| 26 | `lasso_log/stats` | 20 | 0.1197 +- 0.0048 | 0.8803 | 0.1172 | 0.1434 | 0.0553 | 5.3 |
| 27 | `elastic_log/stats` | 20 | 0.1205 +- 0.0050 | 0.8795 | 0.1179 | 0.1435 | 0.0571 | 11.6 |
| 28 | `lgbm/stats` | 20 | 0.1381 +- 0.0125 | 0.8619 | 0.1256 | 0.1212 | 0.1358 | 4.3 |
| 29 | `ridge_log/fds` | 22 | 0.1983 +- 0.0070 | 0.8017 | 0.1957 | 0.2047 | 0.1745 | 1.6 |
| 30 | `lgbm/fds` | 22 | 0.2324 +- 0.0087 | 0.7676 | 0.2421 | 0.2418 | 0.2429 | 4.5 |
| 31 | `elastic_log/spectral` | 27 | 0.2473 +- 0.0053 | 0.7527 | 0.2470 | 0.2597 | 0.2169 | 19.0 |
| 32 | `lasso_log/spectral` | 27 | 0.2490 +- 0.0069 | 0.7510 | 0.2469 | 0.2598 | 0.2165 | 6.5 |
| 33 | `lgbm/spectral` | 27 | 0.2496 +- 0.0208 | 0.7504 | 0.2406 | 0.2659 | 0.1808 | 4.8 |
| 34 | `ridge_log/spectral` | 27 | 0.2520 +- 0.0061 | 0.7480 | 0.2521 | 0.2655 | 0.2203 | 1.6 |
| 35 | `lasso_log/fds` | 22 | 0.2527 +- 0.0132 | 0.7473 | 0.2411 | 0.2483 | 0.2240 | 9.3 |
| 36 | `elastic_log/fds` | 22 | 0.2555 +- 0.0124 | 0.7445 | 0.2469 | 0.2545 | 0.2288 | 16.7 |
| 37 | `ridge_log/rainflow/rawtarget` | 40 | 0.5842 +- 0.0228 | 0.4158 | 0.5732 | 0.7352 | 0.1896 | 9.5 |
| 38 | `lasso_log/rainflow/rawtarget` | 40 | 0.5909 +- 0.0351 | 0.4091 | 0.5800 | 0.7439 | 0.1916 | 29.9 |
| 39 | `ridge_log/stats+rainflow+spectral+fds/rawtarget` | 109 | 0.5993 +- 0.0460 | 0.4007 | 0.5689 | 0.7377 | 0.1692 | 2.3 |
| 40 | `lasso_log/stats+rainflow+spectral+fds/rawtarget` | 109 | 0.6191 +- 0.0175 | 0.3809 | 0.6474 | 0.8431 | 0.1840 | 191.9 |

**Winner: `lasso_log/stats+rainflow+spectral+fds/bias/physblend50_skewpos`** (109 features). Beats the plan's baseline (`lasso_log/stats`, 0.1197): **True**.

Honest (nested) number, selection re-run inside every outer fold over the top 6 rows: **MAPE 0.0187, score 0.9813** (low mode 0.0205, high mode 0.0143). Specs chosen per fold: {'elastic_log/stats+rainflow+spectral+fds': 1, 'lasso_log/stats+rainflow+spectral+fds/bias/physblend50_skewpos': 62, 'lasso_log/stats+rainflow+spectral+fds/cmixup128': 1}.

## The diagnostic gate and the noise floor

Per `docs/research/ps3_addendum.md` section 6 row 11, the rainflow diagnostic ran **before**
any ladder row (`results/ps3/shm_diagnostic.md`). It found that the organisers' label *is*
plain rainflow + Palmgren-Miner on the published channel at `m = 5` with the residue counted
as half cycles [R270], `C ~ 7.4e8`, log-log slope 0.997: MAPE 0.0254 from one fitted
constant. That is the physics row every ML row here has to beat, and the `rainflow_sn*` rows
above are exactly it, fitted fold-locally.

Two consequences the addendum names:

* the leftover scatter tracks the irregularity factor (`sp_alpha1` r = +0.40) as well as the
  skewness (r = -0.54), so the **FDS bands became a MUST** [R274] and are in the feature
  table - the winning row uses them;
* the **noise floor** row (addendum section 6 row 12) is reported in `shm_diagnostic.md`
  rather than as a CV row, because it is not a model: windows are resampled with replacement
  to rebuild full-length records [R271][R269]. Read it as an upper bound - these files are
  visibly non-stationary inside a file, so the window spread is mostly reproducible structure.
  The honest empirical ceiling is the physics row's residual sd in log space, ~0.039, and the
  winner's nested MAPE of 0.0187 sits just below half of it.

## MAPE split by damage mode

The labels are bimodal (median 0.10, 45 files below 0.25 and 19 above), and MAPE is relative,
so the **low-damage mode decides the score** (addendum section 4.4). Every row above reports
both columns; the winner's split is in `shm_ladder.json`, and the nested number splits
0.0205 (low) / 0.0143 (high).

## Augmentation

One row, as the addendum prescribes (section 6 row 14): **C-Mixup** [R299] in log-feature
space, whole-file mixing with pairs drawn by a Gaussian kernel on label distance, generated
inside the training fold only and never on a held-out file. Cropping and vanilla mixup do not
preserve a cumulative-damage label, so they are not attempted. The `cmixup128` rows above
carry the measured effect.

## Rows skipped for budget

* Deep sequence models (1D-CNN / LSTM / TCN) on the 581k raw samples: n = 64 files, and the
  wind-DEL literature measures a temporal-convolution network on the full series at +0.003 R2
  over three scalars [R272]. Not worth a GPU slot here (addendum section 4.3, SKIP).
* Rainflow-matrix extrapolation [R279]: we are not extrapolating to a longer life - the
  record *is* the label's domain.
* Multiaxial / critical-plane criteria: one stress channel, so there is no second channel.
* Per-band rainflow FDS [R274]: the SDOF bank is evaluated on the PSD instead (same construction,
  one Welch per file rather than 11 rainflow passes).
* `ffpack` / `FLife` / `fatpack` as dependencies: no installs are permitted, and `ffpack` is
  GPL-3.0 [R269]. Every rainflow and spectral formula here is hand-implemented.
* The Haibach two-slope knee is swept in the diagnostic but never shipped: [R291] is flagged
  SECONDARY-SOURCE and must not reach a deliverable unverified.
