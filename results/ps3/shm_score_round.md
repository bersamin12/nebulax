# SHM round after organiser feedback

The organiser reported **0.971752844618028** for the previous SHM CSV
(MAPE 0.028247155381972). The later positive-skew blend scored **0.972795**
(MAPE 0.027205) from the 22:18 submission ZIP: a measured score gain of
**0.00104216** and 3.69% relative MAPE reduction. No Test labels or per-file
Test errors were available.

The previous full-feature log-Lasso had Train-only selection MAPE 0.019459
and nested LOO MAPE 0.020318. Five compact corrections to the m=5 rainflow
damage formula and a full-feature Lasso with the m=5 term fixed as an offset
scored worse than that model in LOO and repeated CV. Those rows are in
`shm_physics_residual_round.json`.

The one promoted change is a fixed rule: when a recording's stress skew is
positive, average the Lasso prediction with a plain m=5 rainflow prediction
at equal weight. The rainflow scale is the median log ratio fitted on the
training fold; the Lasso, scaler, alpha and bias are also fitted inside that
fold. Negative-skew recordings continue through the same Lasso. The sign
threshold and 50% weight were chosen after exploratory LOO diagnostics, so
the repeated and nested scores are conditional evidence rather than a fresh
independent test.

| Train-only check | Previous MAPE | Gate MAPE |
|---|---:|---:|
| Leave one file out | 0.019805 | **0.018734** |
| Repeated 5 × 10-fold | 0.019459 | **0.018695** |
| Nested LOO, six-row shortlist | 0.020318 | **0.018700** |
| Five nested 4-fold repeats, fixed gate | 0.019917 | **0.018727** |
| Hold out skew quartiles | 0.022933 | **0.020920** |
| Hold out highest-skew quartile | 0.028204 | **0.022477** |
| Hold out spectral-centroid quartiles | 0.020651 | **0.019401** |

The full ladder audit reused the unchanged 39 historical `shm-f2` rows,
evaluated this one new row on the same LOO and repeated folds, then reran
six-row nested selection. Inner selection chose the gate in **62 of 64** outer
folds. The refitted model is **6,212 bytes**. Ten of 16 Test predictions
change by 0.09%–1.76%. The later reported organiser score confirms an aggregate improvement.

The previous model, reports, CSV and submission ZIP are backed up under
`data/ps3_backups/20260918_organiser_score_before_shm_gate/`. The new model
and CSV were promoted only after the production predictor reproduced all
16 independently calculated candidate predictions to 1e-12 relative and
absolute tolerance. Focused SHM tests, including exact blend arithmetic
and fold isolation, passed.
The root CLI, packaged app CLI, submission CSV and ZIP member agree byte for byte
on all 16 distributed Test inputs.
