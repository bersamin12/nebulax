# Rail feature Mixup round

Fold-local Train-only checks on the promoted coherence row. Each fault training row is mixed with a distinct same-class training row at a nearby speed. These are convex interpolations of the 201 aggregate features; held-out rows are never altered. The inner class-boost calibration synthesises its own rows from each inner training partition and evaluates only untouched validation rows.

| variant | grouped 5-fold × 3 macro F1 ± SD | Side I F1 | Side II F1 |
|---|---:|---:|---:|
| no Mixup, selected coherence model | 0.8441 ± 0.1059 | 0.6454 | 0.9068 |
| same-class feature Mixup, Beta(0.4, 0.4) | 0.8409 ± 0.1075 | 0.6498 | 0.8943 |
| same-class feature Mixup, Beta(2, 2) | 0.8187 ± 0.1021 | 0.6016 | 0.8788 |
| fault/Normal feature Mixup, weighted soft target, Beta(0.4, 0.4) | 0.8157 ± 0.1086 | 0.5660 | 0.9053 |

All Mixup rows score below the selected model. The cross-class row interpolates each fault
training row with a distinct speed-near Normal training row, then represents the interpolated
target using two identical feature rows weighted by its fault and Normal mixture fractions.
Its 15-fold detail is in `rail_mixup_soft_round.json`. No Test labels were used; the model and
submission CSV are unchanged.
