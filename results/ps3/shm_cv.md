# SHM cumulative fatigue damage - baseline CV

* spec `lasso_log/stats` (20 features, `shm-f2`)
* git rev `36d7e03`, 4.86 s wall
* scheme: loo + repeated 5x10-fold; metric `max(0, 1 - MAPE)` (`nebulax.ps3.scoring.mape_score`)

| scheme | MAPE | score | MAPE low mode (< 0.25) | MAPE high mode | worst APE |
|---|---|---|---|---|---|
| LOO (n=64) | 0.1172 | **0.8828** | 0.1434 (n=45) | 0.0553 (n=19) | 0.743 |
| repeated 5x10-fold | 0.1197 +- 0.0048 | 0.8803 | 0.1437 | 0.0590 | 0.752 |

Worst LOO files:

| file | true | predicted | APE |
|---|---|---|---|
| `train56.csv` | 0.0401 | 0.0698 | 0.743 |
| `train09.csv` | 0.0336 | 0.0477 | 0.417 |
| `train13.csv` | 0.0286 | 0.0182 | 0.364 |
| `train59.csv` | 0.0915 | 0.1215 | 0.327 |
| `train50.csv` | 0.0402 | 0.0510 | 0.269 |

> This file is the **baseline** row the plan freezes first (the plan's stats-20 + Lasso in
> log space). It is not the shipped model: `shm_ladder.md` has the full table, the winner and
> the honest nested number, and `models/ps3/shm.json` records which spec the artefact holds.

Every number here is reproducible from `shm_cv.json` (per-file predictions are in it).
Fold-local: the scaler, the Lasso alpha path, the S-N exponent and any bias correction are
fitted inside each training fold; the held-out file is never seen, never augmented.
