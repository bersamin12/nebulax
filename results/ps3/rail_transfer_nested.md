# Rail transfer nested audit

All 13 declared W7 and transfer rows are ranked inside each outer training partition using grouped 3-fold CV. Outer held predictions come from the same fixed recipes in the completed selection run. All external encoders are frozen.

Nested macro F1: **0.8363 ± 0.0995**; Side I F1 0.6038; speed-matched macro F1 0.8412.

Nested gate: 0.7471. Status: **passes nested gate**.

Selection counts:

- W7 / W7 current-code reference: 1
- frozen transfer / W7 + 12.5% Mantis short + 12.5% MOMENT long: 1
- frozen transfer / W7 + 12.5% Mantis short + 12.5% UniTS long: 3
- frozen transfer / W7 + 25% MOMENT long: 1
- frozen transfer / W7 + 25% MOMENT short: 4
- frozen transfer / W7 + 25% Mantis long: 2
- frozen transfer / W7 + 25% UniTS long: 3

W7 versus the short-Mantis/long-MOMENT fusion only:

- Nested macro F1 0.8381 ± 0.0988; Side I F1 0.6038.
- W7 / W7 current-code reference: 6 outer folds
- frozen transfer / W7 + 12.5% Mantis short + 12.5% MOMENT long: 9 outer folds
