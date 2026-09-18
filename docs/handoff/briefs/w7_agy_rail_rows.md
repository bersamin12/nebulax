# W7 — Rail corrugation: feature-reduction rows, fold-local speed baseline, sub-window voting (handover to agy / Gemini, 18 Sep 2026, 13:45)

You are taking over one bounded modelling round on /mnt/berstorage/nebulax. Same shape as W6
(brief `w6_agy_rail_ensemble.md`, which you executed; its rules, decision rule and re-ship steps
apply here unchanged, re-read it now). This round adds several new **declared ladder rows** to
`nebulax/ps3/rail.py`, runs the frozen procedure ONCE, and re-ships the rail artefact only if
the procedure selects a new row. Evidence behind the rows: `scratchpad/w6/findings.md` and the
competition survey in the orchestrator's notes below.

**Hard clock.** The event starts at 17:00. The ladder run must be launched by **14:50** at the
latest with whatever rows are ready; anything not ready by then is dropped and reported as not
attempted. Do not let the sub-window work delay the launch.

Paths: repo `/mnt/berstorage/nebulax`; scratch `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w7/`
(write `train.log`, `extract.log`, `report.md` there). The byte-for-byte backup of the shipped
artefacts already exists in `scratchpad/w6/prev/` (unchanged since W6; verify the md5s match
the three rail pins in `tests/test_ps3_stream.py` before you start, and restore from there if
the decision rule says do not ship).

## Rules (delta from W6)

- Everything in the W6 rules holds: nebulax python, no commit, no install, no full suite, no
  edits to `common.py` / `scoring.py` / `submission.py` / other tasks, no server on 8000/5173/8830.
- `nebulax/ps3/rail_features.py` MAY be edited this round, under two constraints: (1) the on-disk
  cache format and `FEATURE_VERSION` "v2" stay untouched and every existing cache file stays
  valid; (2) `aggregate(feats, AGG_DEFAULT)` and `aggregate(feats, <every existing ladder opts>)`
  must return exactly the same columns and values as before (new behaviour only behind NEW opts
  whose default reproduces today's output). `tests/test_ps3_rail.py`, `tests/test_ps3_rail_verify.py`
  and `tests/test_ps3_stream.py` guard this; run them after every edit to that file.
- CPU: the ladder run uses `--n-jobs 4`; the sub-window extraction may run at 4 workers in
  parallel with it (24 cores on the box). Nothing else above that.
- Append new rows at the END of `LADDER_ROWS`; never reorder or edit existing rows.

## Rows to add (in this priority order; each is one tuple in `LADDER_ROWS`)

### R1-R4: feature reduction (cache-only, cheapest, highest prior)
The shipped winner has 352 columns for 272 files. The VSB and LANL winners pruned hard. Add
opts to `aggregate` (all default True so nothing changes without them):
`"shock": bool` (drop the shock-channel blocks), `"time": bool` exists already,
`"discriminators"`/`"votes"` exist already. Rows, all `lgbm`, all with the winner's base opts
`{"wavelength": False, "hz": True, "v2_normalise": False}` and `tta=True`:
- R1 `+ no shock channels`: base + `{"shock": False}`
- R2 `+ no time-domain block`: base + `{"time": False}`
- R3 `+ no shock, no time block, no votes` (Hz bands + contrast + discriminators + speed only)
- R4 `+ importance-pruned to 64 columns`: kind `"lgbm_pruned"`: fit the winner once on the
  training fold, keep the 64 columns with the highest gain importance (summed over the seed
  models), refit on those. Fold-local by construction because it happens inside `fit_rail`.
  Implement as a wrapper estimator like `_TwoViewLGBM` (it receives `columns` already); the
  selected column names go into the model meta for the report.

### R5: fold-local speed baseline residuals (cache-only)
The three Side I misses at 66-67 km/h sit among nine Normals at 65-71 km/h; the ladder measured
the theoretical v^2 normalisation as a cost. Replace it with a data-fitted one: inside
`fit_rail`, for every dB-level column of the design matrix (the `*_hz_*`, `*_wl_*` band levels
and the `logrms`/`logpeak` time columns), fit `level = a + b * log(speed)` by least squares on
the **Normal files of the training fold only**, and add residual columns
`<name>_resid = level - (a + b log speed)`. Store `(a, b)` per column on the `RailModel` and
apply them in `proba` (or a small transform step before the estimator) so prediction is
identical for the CLI, the API and `predict_test_set`. The mirror augmentation must use the
same coefficients (fit once on the un-mirrored Normal files). Row:
`("Hz bands + speed, + fold-local speed-baseline residuals, + mirror TTA", "lgbm",
  {"opts": {"wavelength": False, "hz": True, "v2_normalise": False, "speed_baseline": True}, "tta": True})`
where `speed_baseline` is a new opt read by `fit_rail`, not by `aggregate`.
If this needs more than ~45 minutes of work, skip it and say so; R1-R4 and R6 come first.

### R6: sub-window voting (needs a new cache; start the extraction FIRST, then code R1-R5)
Every 1 s file becomes K = 3 windows of 0.5 s at 0.25 s hops (samples [0,5000), [2500,7500),
[5000,10000)). New module `nebulax/ps3/rail_windows.py`:
- `window_cache_dir(version="w1")` -> `data/ps3_cache/rail_win/w1/`;
- `build_window_cache(paths, n_jobs=4)`: for every file, `rail_features.read_rail_csv`, slice the
  three windows, call `rail_features.extract_array(window, name=f"{stem}#w{k}.csv")`, and write
  `<stem>#w{k}.parquet` + `<stem>#w{k}.spec.npy` exactly like `_extract_one` does (same schema, so
  `load_feature_cache`-style loading and `aggregate` work unchanged). Skip files already cached.
  ~40 min for the 340 files at 4 workers; launch it detached to `scratchpad/w7/extract.log`
  as the very first thing you do after reading this brief:
  `setsid nohup python -c "..." > .../extract.log 2>&1 &` (or a tiny script under scratchpad).
- `load_window_features(file_ids) -> (RailFeatures of windows, parent index array)`.
Model kind `"lgbm_windows"`: in `fit_rail`, when kind is `lgbm_windows`, expand the training
files into their windows (labels inherited from the parent), apply mirror augmentation per
window, fit the usual seed ensemble on the window design matrix; in `proba`, expand the input
files into windows, average the window probabilities per parent file (arithmetic mean), then
the usual boosts / low-speed rule on the file-level speed. Boost tuning inside `fit_rail` must
group windows by parent file (extend the existing `inner_groups` logic) so a window is never
scored against a sibling of its own file. The outer CV and the stress splits stay file-level and
need no change. `RailTask.predict`, `predict_test_set` and `stream_rail` must all work with this
kind: they receive one file's `RailFeatures` built by `RailTask.load` from the raw array, so the
model needs a code path that windows a raw array on the fly (`extract_array` on the three
slices) when the cache has no entry; keep `RailTask.load` returning the full-file features and
let the window model do its own windowing from `feats.attrs["rail_features"]` plus the raw
path if needed. If on-the-fly windowing needs the raw csv again, store the path on the
RailFeatures in `RailTask.load` (a new optional attribute, default None).
Row: `("sub-window voting: 3 x 0.5 s windows, Hz bands + speed, + mirror TTA", "lgbm_windows",
       {"opts": {"wavelength": False, "hz": True, "v2_normalise": False}, "tta": True})`.
Cite [R249] (short accelerometer windows) and the survey note below.
If the window cache is not complete by 14:45, launch the ladder without R6 and report it as not
attempted.

## Orchestrator's survey note (for the write-up, if a row ships)
No Kaggle competition exists on axle-box vibration. The nearest signal competitions with scarce
positives were won by hand-built window statistics into gradient boosting with aggressive
feature pruning (VSB Power Line Fault Detection 1st place: pulse statistics, ~300 -> 68 columns;
LANL Earthquake 1st place: a LightGBM on four features) and by sampling many overlapping
training windows per recording with a per-recording vote (LANL, CareerCon 2019). Deep 1-D CNNs
with online augmentation only won where thousands of frames per class existed (ICPHM 2023).

## Procedure, decision, re-ship

1. 13:50 start the window extraction (detached, 4 workers). Verify W6 `prev/` md5s.
2. Code R1-R4 (+ unit tests: one fast test per new kind/opt in `tests/test_ps3_rail.py`), then
   R5 if time allows, then R6 once the cache exists. Run
   `python -m pytest tests/test_ps3_rail.py tests/test_ps3_rail_verify.py tests/test_ps3_stream.py -q`
   after each piece.
3. By 14:50 at the latest, launch ONE ladder run (same line as W6, log to `scratchpad/w7/train.log`).
   Rows not ready are left out of `LADDER_ROWS` (do not leave half-implemented rows in).
4. While it runs: make sure `RailTask.predict` / CLI / stream work for every new kind on one Test
   file (`scripts/ps3_predict.py --task rail --input <one Test csv> --output scratchpad/w7/one.csv`
   with a model you fit in a scratch script on a 40-file subset; do not overwrite `models/ps3/`).
5. When the run exits: apply the W6 decision rule with the SAME two clauses ((a) a NEW row is the
   selection-CV winner; (b) nested headline not below 0.7639 by more than 0.01). If it says ship,
   follow the W6 re-ship steps (submission zip, the three md5 pins, `python -m nebulax.ps3.report`,
   write-up numbers, `submission/nebulax/app/` refresh). If not, restore from `scratchpad/w6/prev/`
   and keep the code.
6. Final tests: the W6 test line plus `tests/test_ps3_cli.py tests/test_ps3_report.py`.
7. Report to `scratchpad/w7/report.md` in the W6 report format, with one line per row:
   selection-CV macro F1 +- sd, Side I F1, speed-matched, vs the shipped winner's 0.8007 +- 0.0987.
