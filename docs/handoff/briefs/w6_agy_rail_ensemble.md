# W6 — Rail corrugation: two-view ensemble ladder row (handover to agy / Gemini, 18 Sep 2026)

You are taking over one bounded modelling change on /mnt/berstorage/nebulax: add a **two-view
probability-average ensemble** as a new declared row of the rail model ladder, run the frozen
validation procedure, and re-ship the rail artefact **only if the frozen procedure selects it**.
Read `scratchpad/w6/findings.md` (path below) first: it is the evidence this brief is based on.

Paths: repo `/mnt/berstorage/nebulax`; findings and diagnostic scripts in
`/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w6/`
(`diag_rail5.py` is the disposable harness that measured the ensemble; do not copy it into the
package, it is not fold-grouped and does not tune boosts).

## Rules (same as every agent on this repo)

- Python: `/home/administrator/miniconda3/envs/nebulax/bin/python`, run from the repo root
  (`python -m pytest ...`, `python scripts/...`). No editable install.
- Do NOT `git commit/add/stash/checkout` (the orchestrator commits). Do NOT install packages.
  Do NOT write into `readingmaterials/problem_statement/` (organisers' data, read-only).
- Do NOT edit `nebulax/ps3/common.py`, `nebulax/ps3/scoring.py`, `nebulax/ps3/submission.py`,
  `nebulax/ps3/rail_features.py` (the cache format is frozen; everything you need is in the cache),
  or any door / ACV / SHM module or artefact. Do NOT edit any test except the hash table in
  `tests/test_ps3_stream.py` (see "Re-ship" below) and `tests/test_ps3_rail*.py` if a test
  genuinely encodes the old ladder length.
- Do NOT run the full pytest suite (~10 min). Run only the files named in "Tests".
- There is a uvicorn already running on port 8830 for the user; do not stop it and do not start
  another server on 8000, 5173 or 8830.
- Heavy loops: 4 workers max (`--n-jobs 4`). One ladder + nested run takes roughly 1.5-2 h wall;
  start it early, detached (`setsid nohup ... > scratchpad/w6/train.log 2>&1 &`), and keep working
  on the code-path items while it runs. Check it with `tail`, not with a polling loop.
- Nothing you do may touch the other three tasks' artefacts, CSVs or the door/ACV/SHM hashes in
  `tests/test_ps3_stream.py`.
- No verification run is scheduled after you; write the report (format below) and stop.

## Why

Rail macro F1 is limited by Side I (F1 0.49 on the honest nested CV; Normal 0.975, Side II 0.82).
The orchestrator's diagnostic (findings.md) shows the misses are confident Normals that no
threshold recovers, that a per-box model and per-car features do not help out of fold, and that
the one cheap thing that does help is averaging two models trained on the two representations
already in the feature cache: the shipped winner's Hz-band view and the wavelength view, both
without v^2 normalisation. In a disposable harness this moved 0.816 -> 0.830 macro F1 (Side I
0.559 -> 0.594). Whether it survives the **frozen, grouped, boost-tuned, nested** procedure of
`nebulax/ps3/rail.py` is what you are going to find out; the procedure decides, not the harness.

## Deliverable A — the ensemble as a ladder row

Read `nebulax/ps3/rail.py` top to bottom before touching it. Relevant pieces:
`RailModel` (dataclass; `proba` reindexes the design matrix to `self.columns` and averages the
seed estimators; `proba_tta` averages with the side-swapped mirror view; `predict_labels`
applies the boosts and the low-speed rule), `_make_estimator(kind, seed, n_jobs)`,
`_fit_estimators`, `fit_rail` (builds ONE design matrix from `opts` via
`rail_features.aggregate`, mirror-swap augmentation, inner boost tuning), `cross_validate`,
`nested_cross_validate` (iterates `rows=LADDER_ROWS`, so a new row is automatically a nested
candidate), `LADDER_ROWS`, `run_ladder` (ranks rows on the selection CV, refits the winner,
writes `rail_ladder.{json,md}`, replaces `models/ps3/rail.*` and writes
`results/ps3/rail_predictions.csv` when the winner beats the baseline), `_ladder_markdown`,
`_ladder_findings`, `_rail_cite`, `train`, `predict_test_set`, `RailTask.predict`.

Recommended shape (smallest change that keeps every existing path working):

1. New estimator kind `"lgbm_2view"` in `_make_estimator`: a small sklearn-style wrapper class
   (`fit(X, y)`, `predict_proba(X)`, `classes_`) that holds two LightGBM classifiers with the
   same hyper-parameters as the `"lgbm"` kind, one fitted on the Hz-view columns and one on the
   wavelength-view columns, and returns the arithmetic mean of their probabilities. The
   wrapper needs the column names to pick the views: pass `columns` into `_make_estimator`
   (add a keyword argument with default `None`; every existing call keeps working) and thread
   it through `_fit_estimators` / `fit_rail` (the column list already exists there as `columns`).
   View selection by column-name prefix, from `rail_features.aggregate`:
   Hz view = every column except `*_wl_*` (i.e. the `vib_wl_`/`shock_wl_` blocks);
   wavelength view = every column except `*_hz_*`. Shared columns (time-domain block `_t_`,
   discriminators `_d_`, `*_agree_*`, `*_lamspread_*`, `car_vote_*`, speed columns) go to both.
   Assert both views are non-empty at fit time.
2. New ladder row appended at the END of `LADDER_ROWS` (do not reorder; `run_ladder` and the
   markdown reference the baseline by row index 3):
   `("two-view ensemble: Hz-band model + wavelength model, no v^2, probability average, + mirror TTA",
     "lgbm_2view", {"opts": {"hz": True, "v2_normalise": False}, "tta": True})`
   (`wavelength` stays True by default so the union design matrix carries both blocks).
   Give it a citation in `_rail_cite` (it is [R234] for the Hz view plus [R231][R232][R233] for
   the wavelength view). Add one line to `_ladder_findings` / `_ladder_markdown` that reports the
   ensemble against its two single-view rows so the write-up can quote it.
3. Everything else (mirror-swap augmentation, inner boost tuning, TTA, low-speed rule, seeds,
   grouped splits, stress splits) is inherited unchanged. Do not add any other row and do not
   change any existing row, hyper-parameter or split.
4. Unit test in `tests/test_ps3_rail.py`: `fit_rail(..., kind="lgbm_2view", opts={"hz": True,
   "v2_normalise": False})` on the fixture/synthetic data fits, predicts labels from the fixed
   vocabulary, and its `proba` equals the mean of the two view models' probabilities (expose the
   two sub-models on the wrapper so the test can check this). Keep it seconds-fast (`seeds=(0,)`,
   `n_jobs=1`), like the neighbouring tests.

Then run the frozen procedure exactly as W4 did:

```
cd /mnt/berstorage/nebulax && setsid nohup /home/administrator/miniconda3/envs/nebulax/bin/python \
  scripts/ps3_train.py --task rail --seeds 0 1 2 --n-jobs 4 --ladder --tag baseline \
  > /tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w6/train.log 2>&1 &
```

It rewrites `results/ps3/rail_cv.{json,md}`, `results/ps3/rail_ladder.{json,md}`,
`results/ps3/rail_predictions_baseline.csv`, and, when the ladder winner beats the baseline,
`models/ps3/rail.{pkl,json}` and `results/ps3/rail_predictions.csv`. Before launching, copy the
current `models/ps3/rail.pkl`, `models/ps3/rail.json`, `results/ps3/rail_*.{json,md,csv}` to
`scratchpad/w6/prev/` so the previous state can be restored byte-for-byte if the decision
rule below says "do not ship".

## Decision rule (honesty first)

Ship the ensemble **only if both** hold in the new `rail_ladder.json`:
(a) the ensemble row is the selection-CV winner (`winner.model == "lgbm_2view"`), and
(b) the nested headline (`nested.macro_f1_mean`) is not lower than the previous 0.7639 by more
than 0.01 (it is an honest outer estimate of the whole selection procedure; adding a candidate
may move it either way, and a drop means the procedure over-selects).
If (a) fails, or (b) fails: restore every file from `scratchpad/w6/prev/`, keep the code (the
row stays in the ladder as a declared, measured, non-selected row), keep the NEW
`rail_ladder.{json,md}` and `rail_cv.{json,md}` only if the restored artefact is the one they
describe (they will not be: the ladder json records the winner it refitted, so restore those
too), and say so in the report. Never hand-pick the ensemble because the harness liked it.

## Deliverable B — the app path applies TTA when the model asks for it

`RailTask.predict` calls `mdl.proba(X)` and ignores `mdl.tta`, while `predict_test_set` uses the
mirror view. Make `RailTask.predict` build the mirror design matrix from `feats.attrs["rail_features"]`
(`rail_features.mirror` then `aggregate` with `mdl.opts`) and call `mdl.proba_tta(X, X_mirror)`
when `mdl.tta` is set, so the CLI/API path and the results CSV agree by construction. Then prove
it: `python scripts/ps3_predict.py --task rail --input <PS3>/02_Datasets/Rail_Corrugation/Test
--output scratchpad/w6/rail_cli.csv` must be byte-identical (`md5sum`) to
`results/ps3/rail_predictions.csv` for whichever artefact ends up in `models/ps3/`. This holds
today for the shipped model (both md5 59ebeb1506a63e4c81a869943b72db82); it must still hold after.
`nebulax/ps3/stream.py::stream_rail` goes through `RailTask.run`, so it inherits the fix; its
tests pin final-frame equality with the batch rows, run them.

## Re-ship (only when the decision rule says ship)

1. `python scripts/ps3_submission.py --tasks rail` regenerates
   `submission/nebulax/rail_predictions.csv` and `submission/nebulax/predictions.zip`
   (it validates the CSV; read its docstring for the exact flags; do NOT touch
   `predictions_baseline.zip`). Confirm the new submission CSV md5 equals
   `results/ps3/rail_predictions.csv` and the CLI output from Deliverable B.
2. `tests/test_ps3_stream.py` pins md5s in `FROZEN_MD5` for `models/ps3/rail.json`,
   `models/ps3/rail.pkl` and `submission/nebulax/rail_predictions.csv`. Replace exactly those
   three values with the new md5s (compute them with `md5sum`; never hand-type). Leave the
   other nine entries untouched.
3. `python -m nebulax.ps3.report` (it has a `main`; check `--help`) rebuilds
   `results/ps3/leaderboard.{md,json}` and `ablation_heatmap.html` from the ladder jsons.
4. Update the rail numbers quoted in prose: `docs/ps3_writeup.md` (the "Rail" row of the
   headline table, currently 0.7639 +- 0.1090, and any sentence naming the selected rail row or
   "15 candidates"), `results/ps3/leaderboard.md` is regenerated, `README.md` if it quotes the
   rail number. Grep for `0.7639`, `0.801`, `Hz bands + speed` to find them all. Describe the
   ensemble honestly: "post-hoc selected; the nested headline is X", exactly as the current text
   does for the Hz-band row.
5. `submission/nebulax/app/` is a self-contained copy of the package and models (see
   `scripts/ps3_submission.py` and `docs/ps3_writeup.md` for how it was built). If it embeds
   `models/ps3/rail.*` or `nebulax/ps3/rail.py`, refresh those copies the same way W4 did;
   report what you found and what you refreshed.

## Order of work

1. Copy the current rail artefacts and results to `scratchpad/w6/prev/`.
2. Deliverable A code + unit test; run `tests/test_ps3_rail.py` (fast).
3. Launch the ladder run detached (command above). Note the start time.
4. Deliverable B while it runs; run `tests/test_ps3_rail.py tests/test_ps3_rail_verify.py
   tests/test_ps3_stream.py tests/test_api_ps3.py tests/test_ps3_cli.py -q`.
5. When the run exits: read `rail_ladder.md` and `rail_cv.md`, apply the decision rule, then
   either re-ship (steps 1-5 above) or restore, then re-run the tests in step 4 plus
   `tests/test_ps3_report.py`.
6. Write the report.

## Tests

`python -m pytest tests/test_ps3_rail.py tests/test_ps3_rail_verify.py tests/test_ps3_stream.py tests/test_api_ps3.py tests/test_ps3_cli.py tests/test_ps3_report.py -q`
All must pass at the end, whichever branch of the decision rule applied.

## Report

Write `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w6/report.md`:
- files_written (every path, one line each, what changed)
- commands_run (the ladder launch line, its start/end times and exit code, the test line and its
  summary line)
- ladder result: the new row's selection-CV macro F1 +- sd and Side I F1 next to the two
  single-view rows; the new nested headline vs 0.7639 +- 0.1090; the winner the procedure chose;
  the stress-split numbers for the winner
- decision: ship / do not ship, with the rule clause that decided it
- if shipped: old vs new test prediction counts (Normal / Side I / Side II), the list of Test
  files whose label changed, the three new md5s, and the md5 equality proof (results csv ==
  submission csv == CLI csv)
- open findings (anything you saw and did not fix; do not fix outside this brief's scope)
- status: done / partial, and what is left
