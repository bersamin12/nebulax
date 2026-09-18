# W7 — Rail corrugation: feature-reduction rows, fold-local speed baseline, sub-window voting report

## files_written
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w7/prev/rail.pkl`: backup of committed rail model artefact before ladder run
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w7/prev/rail.json`: backup of committed rail model metadata JSON before ladder run
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w7/prev/rail_cv.json`: backup of committed rail CV results JSON before ladder run
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w7/prev/rail_cv.md`: backup of committed rail CV report Markdown before ladder run
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w7/prev/rail_ladder.json`: backup of committed rail ladder JSON before ladder run
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w7/prev/rail_ladder.md`: backup of committed rail ladder report Markdown before ladder run
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w7/prev/rail_predictions.csv`: backup of committed rail test predictions CSV before ladder run
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w7/prev/rail_predictions_baseline.csv`: backup of committed rail baseline predictions CSV before ladder run
- `/mnt/berstorage/nebulax/data/ps3_cache/rail_win/w1/`: sub-window feature cache (1020 files: 3 x 0.5 s slices for all 340 recordings)
- `/mnt/berstorage/nebulax/nebulax/ps3/rail_windows.py`: sub-window extraction, cache directory, and loader functions
- `/mnt/berstorage/nebulax/nebulax/ps3/rail_features.py`: added `shock` boolean toggle to `AGG_DEFAULT` and `aggregate`; added `raw_path` attribute to `RailFeatures`
- `/mnt/berstorage/nebulax/nebulax/ps3/rail.py`: implemented `_PrunedLGBM`, `_fit_speed_baseline`, `_add_speed_baseline_residuals`, `_get_window_features`, `lgbm_windows` logic in `fit_rail`/`proba`/`proba_tta`, added ladder rows 16-21, citations, findings, and mirror-TTA handling
- `/mnt/berstorage/nebulax/tests/test_ps3_rail.py`: added unit tests for `lgbm_2view`, feature reduction options (R1, R2, R3), `lgbm_pruned` (R4), `speed_baseline` (R5), and `lgbm_windows` (R6)
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w7/rail_cli.csv`: CLI predictions output verifying md5 equality with `results/ps3/rail_predictions.csv` and submission CSV (`dae7c1c0e922e554533aae053f7b262d`)
- `/mnt/berstorage/nebulax/models/ps3/rail.json`: committed re-shipped model metadata JSON (MD5 `0db85f40c483828d96167a23bdd7eca7`)
- `/mnt/berstorage/nebulax/models/ps3/rail.pkl`: committed re-shipped model weights (MD5 `a73f4fd86f22b4e3ede2f72d58bd14a8`)
- `/mnt/berstorage/nebulax/results/ps3/rail_cv.json`: updated rail CV results JSON
- `/mnt/berstorage/nebulax/results/ps3/rail_cv.md`: updated rail CV report Markdown
- `/mnt/berstorage/nebulax/results/ps3/rail_ladder.json`: updated rail ladder JSON recording Row 16 winner
- `/mnt/berstorage/nebulax/results/ps3/rail_ladder.md`: updated rail ladder report Markdown
- `/mnt/berstorage/nebulax/results/ps3/rail_predictions.csv`: updated rail predictions CSV
- `/mnt/berstorage/nebulax/results/ps3/leaderboard.json`: updated PS3 multi-task leaderboard JSON
- `/mnt/berstorage/nebulax/results/ps3/leaderboard.md`: updated PS3 multi-task leaderboard Markdown
- `/mnt/berstorage/nebulax/results/ps3/ablation_heatmap.html`: regenerated Plotly ablation heatmap
- `/mnt/berstorage/nebulax/docs/ps3_writeup.md`: updated headline table and text with new Rail numbers
- `/mnt/berstorage/nebulax/submission/nebulax/Optional_Items/write_up.md`: updated submission copy of write-up
- `/mnt/berstorage/nebulax/submission/nebulax/rail_predictions.csv`: updated submission rail predictions CSV (MD5 `dae7c1c0e922e554533aae053f7b262d`)
- `/mnt/berstorage/nebulax/submission/nebulax/predictions.zip`: regenerated 4-task submission archive
- `/mnt/berstorage/nebulax/submission/nebulax/app/`: refreshed self-contained app copies of `nebulax/ps3/rail.py`, `nebulax/ps3/rail_features.py`, `nebulax/ps3/rail_windows.py`, `models/ps3/rail.{pkl,json}`, `results/ps3/rail_*.{json,md}`, `results/ps3/leaderboard.{json,md}`, `results/ps3/ablation_heatmap.html`
- `/mnt/berstorage/nebulax/tests/test_ps3_stream.py`: updated the 3 frozen MD5 pins for rail
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w7/report.md`: this execution and handover report

## commands_run
- Sub-window cache extraction:
  `/home/administrator/miniconda3/envs/nebulax/bin/python /tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w7/run_extract.py`
  Start: `13:58:20` | End: `13:59:46` | Wall time: 86.2s | Extracted 340 x 3 = 1020 slices into `data/ps3_cache/rail_win/w1/`
- Full ladder run:
  `cd /mnt/berstorage/nebulax && setsid nohup /home/administrator/miniconda3/envs/nebulax/bin/python scripts/ps3_train.py --task rail --seeds 0 1 2 --n-jobs 4 --ladder --tag baseline > /tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w7/train.log 2>&1 &`
  Start: `2026-09-18T14:16:23+08:00` | End: `2026-09-18T17:03:29+08:00` (PID 1187255, wall time 9890.99 s / 164.8 min)
  Exit code: 0
- CLI prediction verification:
  `/home/administrator/miniconda3/envs/nebulax/bin/python scripts/ps3_predict.py --task rail --input /mnt/berstorage/nebulax/readingmaterials/problem_statement/PS3/02_Datasets/Rail_Corrugation/Test --output /tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w7/rail_cli.csv -q`
  Exit code: 0 | Verified md5 `dae7c1c0e922e554533aae053f7b262d` matches `results/ps3/rail_predictions.csv`
- Submission regeneration:
  `/home/administrator/miniconda3/envs/nebulax/bin/python scripts/ps3_submission.py`
  Exit code: 0 | Packed 4 subsystems into `submission/nebulax/predictions.zip`
- Report regeneration:
  `/home/administrator/miniconda3/envs/nebulax/bin/python -m nebulax.ps3.report`
  Exit code: 0 | Rebuilt `results/ps3/leaderboard.{md,json}` and `ablation_heatmap.html`
- Final pytest suite:
  `/home/administrator/miniconda3/envs/nebulax/bin/python -m pytest tests/test_ps3_rail.py tests/test_ps3_rail_verify.py tests/test_ps3_stream.py tests/test_api_ps3.py tests/test_ps3_cli.py tests/test_ps3_report.py -q`
  Result: 101 passed in 2m42s, exit code 0

## ladder result
- **Shipped Baseline Winner (Row 9, W4/W6 benchmark)**:
  `lgbm` / `Hz bands + speed, wavelength discriminators kept, + mirror TTA`:
  - Selection-CV Macro F1: **0.8007 ± 0.0987**
  - Side I F1: **0.5613** | Side II F1: **0.8637** | Normal F1: **0.9772**
  - Speed-matched Macro F1: **0.8062**
  - Features: **352**
- **New W7 Rows (Selection-CV comparison vs shipped winner's 0.8007 ± 0.0987)**:
  - **Row 16 (R1, WINNER)**: `lgbm` / `+ no shock channels`:
    - Selection-CV Macro F1: **0.8365 ± 0.0960** (+0.0358 gain over shipped winner)
    - Side I F1: **0.5975** (vs 0.5613 baseline)
    - Side II F1: **0.9319** (vs 0.8637 baseline)
    - Normal F1: **0.9800** (vs 0.9772 baseline)
    - Speed-matched Macro F1: **0.8403** (vs 0.8062 baseline)
    - Features: **180** (pruned 172 shock channels)
  - **Row 17 (R2)**: `lgbm` / `+ no time-domain block`:
    - Selection-CV Macro F1: **0.7162 ± 0.0956** (-0.0845 vs baseline)
    - Side I F1: **0.3475** | Side II F1: **0.8374** | Normal F1: **0.9637** | Speed-matched: **0.7123** | Features: **224**
  - **Row 18 (R3)**: `lgbm` / `+ no shock, no time block, no votes`:
    - Selection-CV Macro F1: **0.7720 ± 0.0855** (-0.0287 vs baseline)
    - Side I F1: **0.4923** | Side II F1: **0.8576** | Normal F1: **0.9659** | Speed-matched: **0.7672** | Features: **113**
  - **Row 19 (R4)**: `lgbm_pruned` / `+ importance-pruned to 64 columns`:
    - Selection-CV Macro F1: **0.7799 ± 0.1203** (-0.0208 vs baseline)
    - Side I F1: **0.5171** | Side II F1: **0.8476** | Normal F1: **0.9750** | Speed-matched: **0.7901** | Features: **352 -> 64**
  - **Row 20 (R5)**: `lgbm` / `Hz bands + speed, + fold-local speed-baseline residuals, + mirror TTA`:
    - Selection-CV Macro F1: **0.7928 ± 0.1001** (-0.0079 vs baseline)
    - Side I F1: **0.5387** | Side II F1: **0.8631** | Normal F1: **0.9765** | Speed-matched: **0.8001** | Features: **352**
  - **Row 21 (R6)**: `lgbm_windows` / `sub-window voting: 3 x 0.5 s windows, Hz bands + speed, + mirror TTA`:
    - Selection-CV Macro F1: **0.7153 ± 0.1438** (-0.0854 vs baseline)
    - Side I F1: **0.4234** | Side II F1: **0.7588** | Normal F1: **0.9637** | Speed-matched: **0.7127** | Features: **352**
- **Other Existing Rows (for context)**:
  - Row 15 (`lgbm_2view`): **0.7870 ± 0.1019** | Side I: **0.5054** | Side II: **0.8748** | Speed-matched: **0.7907**
  - Row 10 (`lgbm` union): **0.7844 ± 0.1036** | Side I: **0.5035** | Side II: **0.8701** | Speed-matched: **0.7868**
  - Row 8 (`lgbm` counter-design): **0.7940 ± 0.1166** | Side I: **0.5644** | Side II: **0.8434** | Speed-matched: **0.8050**
  - Row 5 (`lgbm` no v^2 + TTA): **0.7817 ± 0.1125** | Side I: **0.5143** | Side II: **0.8521** | Speed-matched: **0.7845**
  - Row 4 (`lgbm` no v^2): **0.7679 ± 0.0991** | Side I: **0.4743** | Side II: **0.8521** | Speed-matched: **0.7693**
  - Row 3 (`lgbm` baseline): **0.7311 ± 0.0896** | Side I: **0.3949** | Side II: **0.8312** | Speed-matched: **0.7283**
- **Nested Headline**:
  - Honest outer estimate across 15 grouped outer folds: **0.7571 ± 0.1232**
  - Speed-matched outer estimate: **0.7605 ± 0.1360**
  - Class F1: Normal: **0.9752**, Side I: **0.4571**, Side II: **0.8388**
  - Previous nested headline: **0.7639 ± 0.1090** (delta: **-0.0068**, within 0.01 tolerance)
  - Outer fold selection counts across the 15 outer folds:
    - `lgbm / no v^2 normalisation`: 5 folds
    - `lgbm_pruned / + importance-pruned to 64 columns`: 4 folds
    - `lgbm / + no shock channels`: 2 folds
    - `lgbm / no v^2, + mirror TTA`: 2 folds
    - `lgbm / no v^2, wavelength + Hz bands, + mirror TTA`: 1 fold
    - `lgbm_2view / two-view ensemble`: 1 fold
- **Stress Splits for Row 16 Winner**:
  - `contiguous` (5 folds): **0.7488 ± 0.1406** (improved from 0.7456 ± 0.1423; Normal: 0.9756, Side I: 0.4032, Side II: 0.8675)
  - `speed_range` (3 folds): **0.7258 ± 0.1006** (substantially improved from 0.6423 ± 0.0927: +0.0835 gain; Normal: 0.9365, Side I: 0.4583, Side II: 0.7825)

## decision
- **SHIP**
- Deciding clauses:
  1. Clause (a): **PASSED**. Row 16 (`lgbm` / `+ no shock channels`) is the clear selection-CV winner at **0.8365 ± 0.0960**, beating the baseline row 9 (0.8007 ± 0.0987) by +0.0358. Row index 16 >= 16 (new W7 row).
  2. Clause (b): **PASSED**. The nested outer estimate across the 15 outer folds is **0.7571 ± 0.1232**, which is 0.7639 - 0.0068. The difference (0.0068) is strictly less than the 0.01 margin required by the decision rule.
- Consequently, the rail model artefact has been re-shipped: model files, submission CSV, full 4-subsystem predictions zip, MD5 pins in `test_ps3_stream.py`, report leaderboard, documentation write-up, and app copies have all been regenerated and updated.

## if shipped
- **Shipped**: Yes.
- New committed hashes:
  - `models/ps3/rail.json`: `0db85f40c483828d96167a23bdd7eca7`
  - `models/ps3/rail.pkl`: `a73f4fd86f22b4e3ede2f72d58bd14a8`
  - `submission/nebulax/rail_predictions.csv`: `dae7c1c0e922e554533aae053f7b262d`
- Equivariance verification:
  - CLI output (`rail_cli.csv`): `dae7c1c0e922e554533aae053f7b262d`
  - Results output (`results/ps3/rail_predictions.csv`): `dae7c1c0e922e554533aae053f7b262d`
  - Submission output (`submission/nebulax/rail_predictions.csv`): `dae7c1c0e922e554533aae053f7b262d`
  - All three files are byte-identical.

## open findings
1. **Shock Channel Pruning is Highly Effective**: Dropping the 172 shock accelerometer columns (`shock: False`) improved Selection-CV Macro F1 from 0.8007 to 0.8365. Shock sensors pick up track/bogie impacts and transient jolts that dilute the steady-state corrugation acoustic emission captured by the axlebox vibration channels. Pruning them directly improved Side I F1 from 0.5613 to 0.5975 and Side II F1 from 0.8637 to 0.9319.
2. **Speed Extrapolation Greatly Improved**: In the leave-speed-range-out stress split (3 folds), the shock-free model achieved **0.7258 ± 0.1006** vs **0.6423 ± 0.0927** for the previous baseline. Shock signals vary dramatically with speed, so removing them eliminated a major source of speed confounding.
3. **Time-Domain Block is Essential**: Row 17 (`+ no time-domain block`) collapsed to **0.7162 ± 0.0956** (Side I F1 0.3475), demonstrating that broadband time statistics (RMS, peak, kurtosis, crest factor) provide an indispensable level anchor that spectral bands alone cannot substitute.
4. **Sub-Window Voting Degrades Performance**: 0.5 s sub-windows (Row 21: 0.7153) suffered from inadequate frequency resolution in the low-frequency acoustic bands and high sample variance per frame. The full 1.0 s integration window remains superior for axlebox vibration.
5. **Inner Fold Selection Diversity**: In nested CV, 6 distinct candidate configurations were selected across the 15 outer folds (`no v^2`: 5, `pruned`: 4, `+ no shock`: 2, `no v^2 + TTA`: 2, `2view`: 1, `wavelength + Hz`: 1), reflecting the tight contest among the leading LightGBM configurations on 2/3 partitions of the training data.

## status
- **done**. W7 brief executed in full. Declared ladder rows R1-R6 implemented and tested, sub-window cache created, frozen validation procedure run, decision rule satisfied, artefacts re-shipped with updated pins and documentation, and all 101 unit/regression tests passing cleanly.
