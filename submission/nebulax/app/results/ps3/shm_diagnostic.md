# SHM diagnostic - does rainflow + Miner reproduce the organisers' label?

In an all-64-file diagnostic sweep, 4-point rainflow with half-cycle residue and Miner
exponent m=5 reproduces the labels to MAPE 0.0254 after fitting one global constant C.
Because counting method, residue convention, exponent, and C were selected/fitted using all
64 labels, this is an in-sample label-reproduction diagnostic, not an outer-CV estimate; the
fold-local physics rows provide the corresponding predictive evidence. The team's earlier
reading ("LOO MAPE >= 0.78 for any exponent") missed the residue convention, which is the
largest lever here (Marsh et al. 2016 on residue processing).

| best variant | 4point rainflow, residue `half`, `miner`, m = 5 |
|---|---|
| MAPE with one fitted `C` | **0.0254** (score 0.9746) |
| log-log slope vs label | 0.9965 (1.0 = literal Miner) |
| fitted S-N constant `C` | 7.35e+08 (sigma_a^m N = C, arbitrary stress units) |
| R2 of log damage on log label | 0.99879 |
| residual sd in log space | 0.0391 |

## Top 15 variants

| # | method | residue | correction | m | ratio cv % | MAPE (fitted C) | log-log slope | R2 |
|---|---|---|---|---|---|---|---|---|
| 1 | 4point | half | miner | 5 | 4.06 | 0.0254 | 0.9965 | 0.99879 |
| 2 | 3point | half | miner | 5 | 4.06 | 0.0254 | 0.9965 | 0.99879 |
| 3 | 4point | half | knee | 5 | 4.06 | 0.0254 | 0.9965 | 0.99879 |
| 4 | 3point | half | knee | 5 | 4.06 | 0.0254 | 0.9965 | 0.99879 |
| 5 | 4point | half | haibach | 5 | 4.06 | 0.0254 | 0.9965 | 0.99879 |
| 6 | 3point | half | haibach | 5 | 4.06 | 0.0254 | 0.9965 | 0.99879 |
| 7 | 3point | close | miner | 5 | 6.47 | 0.0520 | 1.0082 | 0.99684 |
| 8 | 3point | close | knee | 5 | 6.47 | 0.0520 | 1.0082 | 0.99684 |
| 9 | 3point | close | haibach | 5 | 6.47 | 0.0520 | 1.0082 | 0.99684 |
| 10 | 3point | full | miner | 5 | 6.67 | 0.0560 | 0.9731 | 0.99711 |
| 11 | 3point | full | knee | 5 | 6.67 | 0.0560 | 0.9731 | 0.99711 |
| 12 | 3point | full | haibach | 5 | 6.67 | 0.0560 | 0.9731 | 0.99711 |
| 13 | 4point | close | miner | 5 | 6.59 | 0.0574 | 1.0071 | 0.99658 |
| 14 | 4point | close | knee | 5 | 6.59 | 0.0574 | 1.0071 | 0.99658 |
| 15 | 4point | close | haibach | 5 | 6.59 | 0.0574 | 1.0071 | 0.99658 |

## What the leftover scatter tracks

Correlation of the log residual (`log(label / Miner sum)`) with the log features:

| feature | r |
|---|---|
| `st_skew` | -0.537 |
| `sp_alpha1` | +0.404 |
| `sp_vanmarcke` | -0.337 |
| `sp_centroid` | +0.336 |
| `sp_alpha2` | +0.290 |
| `sp_nu0` | +0.250 |
| `st_kurtosis` | +0.171 |

Bandwidth-ish terms (`sp_alpha1`, `sp_alpha2`, `sp_vanmarcke`) and the distribution shape
(`st_skew`, `st_kurtosis`) both appear, i.e. part of the residual is a broad-band / non-Gaussian
effect rather than pure realisation noise - which is exactly what the log-space regression on
the rainflow + spectral features picks up in `shm_ladder.md`.

## Realisation noise floor

Each of 16 files was cut into 16 contiguous windows and the
Miner damage (m = 5) counted per window; 400 pseudo-records of the same
total length were then built by resampling windows with replacement.

* per-window damage cv inside a file: **1.607**
* bootstrapped full-record damage cv: **0.391** (median 0.371)

Read this as an **upper** bound, not a floor: these records are plainly non-stationary inside a
file (a train changes speed and line), so the window-to-window spread is mostly real,
reproducible structure rather than realisation noise - which is why the fitted physics row
already reaches MAPE 0.0254, far below the bootstrap number. The honest
empirical ceiling is the physics row's residual sd in log space, **0.0391**, and
the ladder's job is to explain the part of that residual which tracks bandwidth and skewness.

Plots: `shm_diagnostic/damage_vs_label.png` (log damage vs log label, one panel per residue
convention) and `shm_diagnostic/residual_structure.png` (the exponent sweep, and the residual
against the irregularity factor and the kurtosis).

## References

Ids are `docs/research/references.md` (`docs/research/ps3_addendum.md` section 4).

* **[R270]** Marsh, Wignall, Thies, Barltrop et al., *Review and application of rainflow residue
  processing techniques for accurate fatigue damage estimation*, Int. J. Fatigue 82:757-765,
  2016 - the residue convention, which is the whole story here (discard -> half moves the
  MAPE from 0.114 to 0.025).
* **[R269]** Zorman, Slavic, Boltezar, *Vibration fatigue by spectral methods: a review with
  open-source support*, MSSP 190:110149, 2023 - the closed-form narrow-band damage that makes
  `log D` affine in `log`(amplitude scale), and the source (with [R275][R276]) the Dirlik and
  Tovo-Benasciutti formulae in `shm_features.spectral_features` were hand-implemented from.
  `FLife` itself is **not** a dependency: no installs.
* **[R271]** Marques, Benasciutti, Tovo, *Variability of the fatigue damage due to the
  randomness of a stationary vibration load*, Int. J. Fatigue 141:105891, 2020 - the
  noise-floor bootstrap above.
* **[R274]** Proner & Mucchi, *A multi-axial Fatigue Damage Spectrum...*, MSSP 226:112362,
  2025 - the FDS construction. The addendum makes the FDS bands a MUST when the diagnostic
  scatter tracks the irregularity factor, which it does (`sp_alpha1` r = +0.40 above), so the
  `fds` family is in the feature table and in every `all`-family ladder row.
* **[R277]** Wang & Serra, *Vibration fatigue damage estimation by new stress correction based
  on kurtosis control of random excitation loadings*, Sensors 21(13):4518, 2021 - the
  non-Gaussian reading of the kurtosis and skewness correlations above.
* **[R290]** ASTM E1049-85(2017) defines the 3-point counting reimplemented here. Paywalled
  standard, not opened - cited by number, no clause is paraphrased.
* **[R291]** The Haibach two-slope rule (`k* = 2k - 1`) is **SECONDARY-SOURCE** in the
  reference list. It is swept above as one diagnostic variant and it does not win; **no
  shipped model uses it**, and it must not appear in a deliverable until a primary source is
  read.
