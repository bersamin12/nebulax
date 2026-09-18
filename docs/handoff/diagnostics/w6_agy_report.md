# W6 — Rail corrugation: two-view ensemble ladder row report

## files_written
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w6/prev/rail.pkl`: backup of committed rail model artefact before ladder run
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w6/prev/rail.json`: backup of committed rail model metadata JSON before ladder run
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w6/prev/rail_cv.json`: backup of committed rail CV results JSON before ladder run
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w6/prev/rail_cv.md`: backup of committed rail CV report Markdown before ladder run
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w6/prev/rail_ladder.json`: backup of committed rail ladder JSON before ladder run
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w6/prev/rail_ladder.md`: backup of committed rail ladder report Markdown before ladder run
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w6/prev/rail_predictions.csv`: backup of committed rail test predictions CSV before ladder run
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w6/prev/rail_predictions_baseline.csv`: backup of committed rail baseline predictions CSV before ladder run
- `/mnt/berstorage/nebulax/nebulax/ps3/rail.py`: added `_TwoViewLGBM` wrapper class, threaded `columns` parameter through `_make_estimator`, `_fit_estimators`, and `fit_rail`, appended row 15 (`lgbm_2view`) to `LADDER_ROWS`, added citation `[R234][R231][R232][R233]` in `_rail_cite`, added two-view ensemble comparison to `_ladder_findings`, and updated `RailTask.predict` to build mirror features and call `mdl.proba_tta` when `mdl.tta` is True (Deliverable B)
- `/mnt/berstorage/nebulax/tests/test_ps3_rail.py`: added `test_two_view_ensemble_fits_predicts_and_averages_submodels` verifying fitting, vocabulary labels, and probability averaging of submodels
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w6/rail_cli.csv`: CLI predictions output verifying md5 equality with results CSV
- `/mnt/berstorage/nebulax/models/ps3/rail.json`: restored byte-for-byte from `prev/` under decision rule
- `/mnt/berstorage/nebulax/models/ps3/rail.pkl`: restored byte-for-byte from `prev/` under decision rule
- `/mnt/berstorage/nebulax/results/ps3/rail_cv.json`: restored byte-for-byte from `prev/` under decision rule
- `/mnt/berstorage/nebulax/results/ps3/rail_cv.md`: restored byte-for-byte from `prev/` under decision rule
- `/mnt/berstorage/nebulax/results/ps3/rail_ladder.json`: restored byte-for-byte from `prev/` under decision rule
- `/mnt/berstorage/nebulax/results/ps3/rail_ladder.md`: restored byte-for-byte from `prev/` under decision rule
- `/mnt/berstorage/nebulax/results/ps3/rail_predictions.csv`: restored byte-for-byte from `prev/` under decision rule
- `/mnt/berstorage/nebulax/results/ps3/rail_predictions_baseline.csv`: restored byte-for-byte from `prev/` under decision rule
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w6/report.md`: this handover and execution report

## commands_run
- Ladder run:
  `cd /mnt/berstorage/nebulax && setsid nohup /home/administrator/miniconda3/envs/nebulax/bin/python scripts/ps3_train.py --task rail --seeds 0 1 2 --n-jobs 4 --ladder --tag baseline > /tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w6/train.log 2>&1 &`
  Start: `2026-09-18T11:13:16+08:00`
  End: `2026-09-18T12:09:10+08:00` (PID 1162023, wall time 3354.81 s / 55.9 min)
  Exit code: 0
- Deliverable B CLI verification command:
  `/home/administrator/miniconda3/envs/nebulax/bin/python scripts/ps3_predict.py --task rail --input /mnt/berstorage/nebulax/readingmaterials/problem_statement/PS3/02_Datasets/Rail_Corrugation/Test --output /tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w6/rail_cli.csv -q && md5sum /tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w6/rail_cli.csv results/ps3/rail_predictions.csv`
  Exit code: 0, both md5 `59ebeb1506a63e4c81a869943b72db82`
- Final pytest suite:
  `/home/administrator/miniconda3/envs/nebulax/bin/python -m pytest tests/test_ps3_rail.py tests/test_ps3_rail_verify.py tests/test_ps3_stream.py tests/test_api_ps3.py tests/test_ps3_cli.py tests/test_ps3_report.py -q`
  Summary: `97 passed, 189 warnings in 137.49s (0:02:17)`

## ladder result
- New row (`lgbm_2view`, row 15: two-view probability ensemble: Hz-band model + wavelength model, no v^2, + mirror TTA):
  - Selection-CV Macro F1: **0.7870 ± 0.1019**
  - Side I F1: **0.5054** (Normal F1: 0.9809, Side II F1: 0.8748, speed-matched F1: 0.791)
- Comparison against the two single-view rows:
  - Hz-band single-view row (`lgbm`, row 9: Hz bands + speed, wavelength discriminators kept, + mirror TTA):
    Selection-CV Macro F1: **0.8007 ± 0.0987** (Side I F1: 0.5613, Side II F1: 0.8637, Normal F1: 0.9772)
  - Wavelength single-view row (`lgbm`, row 5: no v^2, + mirror TTA):
    Selection-CV Macro F1: **0.7817 ± 0.1125** (Side I F1: 0.5143, Side II F1: 0.8521, Normal F1: 0.9787)
  - Single-model union row (`lgbm`, row 10: no v^2, wavelength + Hz bands, + mirror TTA):
    Selection-CV Macro F1: **0.7844 ± 0.1036** (Side I F1: 0.5035, Side II F1: 0.8701, Normal F1: 0.9795)
- Nested headline:
  - New nested outer estimate across 15 grouped outer folds: **0.7662 ± 0.1083** (speed-matched: 0.7690 ± 0.1181, Normal: 0.9759, Side I: 0.5010, Side II: 0.8219)
  - Previous nested headline: **0.7639 ± 0.1090**
  - Outer fold selection counts across the 15 outer folds:
    - `lgbm / no v^2 normalisation`: 5 folds
    - `lgbm / no v^2, wavelength + Hz bands, + mirror TTA`: 3 folds
    - `lgbm / Hz bands + speed, wavelength discriminators kept, + mirror TTA`: 2 folds
    - `lgbm / no v^2, + mirror TTA`: 2 folds
    - `lgbm_2view / two-view ensemble`: 2 folds
    - `lgbm / counter-design`: 1 fold
- Procedure winner:
  - Selected winner: `lgbm` / row 9 (`Hz bands + speed, wavelength discriminators kept, + mirror TTA`) at macro F1 **0.8007 ± 0.0987**
- Stress splits for the winner:
  - `contiguous` (5 folds): macro F1 **0.7456 ± 0.1423** (Normal: 0.9756, Side I: 0.4222, Side II: 0.8390)
  - `speed_range` (3 folds): macro F1 **0.6423 ± 0.0927** (Normal: 0.9186, Side I: 0.3000, Side II: 0.7083)

## decision
- **do not ship**
- Deciding clause: Clause (a) of the decision rule failed.
  The rule specifies: *"Ship the ensemble only if both hold in the new rail_ladder.json: (a) the ensemble row is the selection-CV winner (winner.model == 'lgbm_2view'), and (b) the nested headline (nested.macro_f1_mean) is not lower than the previous 0.7639 by more than 0.01."*
  While clause (b) held (nested macro F1 was 0.7662 vs 0.7639), clause (a) failed because the selection-CV winner chosen by the procedure was row 9 (`lgbm` at 0.8007 ± 0.0987), whereas `lgbm_2view` scored 0.7870 ± 0.1019.
  Following the brief's instruction, all files in `scratchpad/w6/prev/` (`models/ps3/rail.{pkl,json}` and `results/ps3/rail_*.{json,md,csv}`) have been restored byte-for-byte, while preserving the declared code changes (`nebulax/ps3/rail.py` and `tests/test_ps3_rail.py`).

## if shipped
- Not shipped (decision was do not ship).
- Artefacts and hashes restored to baseline state:
  - `models/ps3/rail.json`: `544f86069c4f5093e5ba37a32d08638f`
  - `models/ps3/rail.pkl`: `70320d088018a192a5afdab3a840b222`
  - `submission/nebulax/rail_predictions.csv`: `59ebeb1506a63e4c81a869943b72db82`
- Equality proof: CLI output `rail_cli.csv` == `results/ps3/rail_predictions.csv` == `submission/nebulax/rail_predictions.csv` (all md5 `59ebeb1506a63e4c81a869943b72db82`).

## open findings
- **Harness vs Frozen Procedure discrepancy**: In the disposable exploratory harness (`diag_rail5.py`), plain ungrouped 5-fold CV without boost tuning suggested an ensemble gain (0.816 -> 0.830). Under the frozen, honest, grouped-split procedure with inner stratified boost tuning, the two-view ensemble achieved 0.7870 ± 0.1019, falling behind the single-view Hz-band model (0.8007 ± 0.0987). This demonstrates the necessity of testing modelling hypotheses through the frozen protocol rather than informal scripts.
- **Outer Fold Selection**: In the nested cross-validation, `lgbm_2view` was selected in 2 of the 15 outer folds by inner 3-fold CV. The overall nested outer estimate slightly improved from 0.7639 to 0.7662.
- **TTA Equivariance in API/CLI Path**: Deliverable B resolved the discrepancy in `RailTask.predict`, ensuring that whenever `mdl.tta` is True, mirror test-time augmentation is applied consistently across both batch and streaming/CLI execution paths.

## status
- **done**. Both deliverables completed, ladder measured, decision rule strictly applied, artefacts restored, and all required tests passing.
