# ACV - leave-one-case-out (exploratory)

Rule `baseline_peer_delta_hot`: features ['peer_delta_hot'], aggregation `single`, tie-break ['ctrl_residual'], hot quantile 0.5.
Feature version `acv-f5`, git `33d00a2`, 5.18 s wall warm / 317.28 s cold (every feature frame rebuilt from the xlsx).

**Mean rank-decay 0.9792 +/- 0.051** over 6 cases (top-1 0.833), **exploratory**. A signal-free ranker that leaves the empty cars last scores 0.6042 on these six cases - that, not the 0.5625 a uniformly random ranking of 8 cars scores, is the floor this number has to beat (`acv_case_04.xlsx` has four entirely empty cars, so a blind ranking of it already scores 0.8125).

This headline is the **pre-registered baseline** rule (`baseline_peer_delta_hot`), evaluated on all six cases. The ladder's best row, `peer_delta_hot_loo`, reaches 1.0000 - but it was **selected on all six cases**, so that number is a selection result, not a held-out one, and `train()` deliberately does not promote it: a rule picked on the same six cases it is then scored on cannot be the headline under the plan's fold-local rule. Both rules produce the identical row for the organisers' Test file, so nothing about the deliverable turns on the choice - only the claim does. `acv_ladder.md` keeps the full table.

Re-choosing the rule inside every fold (the pre-registered pool of `acv.selection_pool`, five training cases per fold, rank decay then top-1 then `acv.separation`) scores 0.9792 - the same as this fixed baseline. The folds do not all pick the same rule: `peer_delta_hot_loo` on 5 of 6 folds, `peer_delta_steady` on 1 of 6 folds.

Six cases is too few for a confidence interval: one case moving one rank changes the mean by 0.021, and the 0.9792 -> 1.0000 gap in the ladder is exactly that - one rank step on `acv_case_04.xlsx`, whose top two cars are 4.2 mK apart. The ranker is peer-normalised inside each case, so no statistic can cross cases.

| held-out case | true car | rank | score | ranked_cars | rule chosen on the other 5 |
|---|---|---|---|---|---|
| `acv_case_01.xlsx` | 01 | 1 | 1.000 | `01|03|04|02|07|06|08|05` | `baseline_peer_delta_hot` |
| `acv_case_02.xlsx` | 02 | 1 | 1.000 | `02|03|08|01|07|04|06|05` | `baseline_peer_delta_hot` |
| `acv_case_03.xlsx` | 03 | 1 | 1.000 | `03|02|04|01|08|07|05|06` | `baseline_peer_delta_hot` |
| `acv_case_04.xlsx` | 01 | 2 | 0.875 | `04|01|02|03|05|06|07|08` | `baseline_peer_delta_hot` |
| `acv_case_05.xlsx` | 04 | 1 | 1.000 | `04|02|07|06|03|01|08|05` | `baseline_peer_delta_hot` |
| `acv_case_06.xlsx` | 06 | 1 | 1.000 | `06|08|04|02|03|05|01|07` | `baseline_peer_delta_hot` |
