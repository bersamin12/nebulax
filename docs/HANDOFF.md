# NEBULA X — PS3 handoff (updated 18 Sep 2026)

Written by the orchestrator (Claude) for whoever picks the project up from here. Everything below
is verifiable from the repo; where a number comes from a JSON file, the file is named.

## 1. State at a glance

| item | state |
|---|---|
| Repo | https://github.com/bersamin12/nebulatemp.git, branch `master`, HEAD `83a5696` (history was rewritten on 18 Sep to purge the LTA PDFs; anyone with an older clone must re-clone) |
| Uncommitted work | the W7 rail re-ship (new rail artefact, results, write-up numbers, hash pins, submission zip, app copy). See §4 before committing |
| Python env | `/home/administrator/miniconda3/envs/nebulax/bin/python` (the base miniconda python lacks deps; never use it) |
| App server | uvicorn restarted after this implementation on port **8830** (`http://127.0.0.1:8830`, twin page `?page=twin`). Restart after later API edits or web rebuilds with the `nebulax` Python environment. |
| Ports reserved | 8000 and 5173 (dev), 8765 is the README default for judges |
| Organiser data | `readingmaterials/problem_statement/PS3/` (gitignored, read-only, 7.6 GB clone) |
| Feature caches | Current rail feature code uses `data/ps3_cache/rail/v3` (per-file); W7's frozen model metadata is v2 and remains loadable. The measured window experiment uses `data/ps3_cache/rail_win/w1`. All caches are gitignored. |

## 2. Deliverables checklist (organiser rules, PS3 spec §"Submission")

| deliverable | status | where |
|---|---|---|
| `predictions.zip` with the four `*_predictions.csv` at archive root | regenerated 18 Sep 17:15 by the W7 re-ship, validated by `scripts/ps3_submission.py` | `submission/nebulax/predictions.zip` |
| Demo video ≤ 3 min showing all four tabs | **NOT DONE** — placeholder only | `submission/nebulax/demo_video.PLACEHOLDER.md`; script in `docs/demo.md` (timed run sheet + six-system twin walkthrough) |
| Runnable app for a non-technical user | done; self-contained copy | `submission/nebulax/app/` (`./run.sh 8765`), README inside |
| Optional write-up, code, models | done | `submission/nebulax/Optional_Items/`, `docs/ps3_writeup.md` |
| Team folder name | **still `submission/nebulax`** — rename to the registered team name before packaging (`scripts/ps3_submission.py --team <name>` regenerates into `submission/<name>/`) |
| Rules check on pre-event code | **not confirmed** with the organisers |
| Stale file | `submission/nebulax/predictions_final.zip` was deleted; use only the validated `predictions.zip`. |

## 3. Scores (honest headlines; selection values are post-hoc and must not be quoted as the headline)

| task | metric | honest headline | selection CV | source |
|---|---|---|---|---|
| Door | IoU-weighted F1 | 0.982 ± 0.036 (5 contiguous blocks, nested) | 1.000 on 5 blocks (post-hoc) | `results/ps3/door_ladder.json` |
| ACV | rank decay | 0.979 ± 0.051 (leave-one-case-out, 6 cases, exploratory) | pre-registered rule, no selection | `results/ps3/acv_ladder.json` |
| Rail | macro F1 | **0.757 ± 0.123** (nested, 15 outer folds, 22 candidates) | 0.836 ± 0.096 ("no shock channels" row) | `results/ps3/rail_ladder.json` |
| SHM | 1 − MAPE | 0.980 (nested) | — | `results/ps3/shm_ladder.json` |

`results/ps3/leaderboard.md` is generated from these JSONs by `python -m nebulax.ps3.report`; do not
hand-edit it.

Expected organiser scores (rough): door 0.95–1.0, ACV 0.875–1.0 (one test case), rail 0.65–0.85,
SHM 0.82–0.88. Rail's test set has ~3–4 Side I files, so its score moves in steps of ~0.1 and is
mostly luck at this point.

## 4. The rail work on 18 Sep (W6, W7) and what to do with it

Full evidence: `docs/handoff/diagnostics/w6_rail_findings.md` (read this first),
`docs/handoff/diagnostics/w6_agy_report.md`, `docs/handoff/diagnostics/w7_agy_report.md`, and
`results/ps3/rail_ladder.md`.

What was learned:
- Side I (14 train files) is the whole problem: nested F1 0.46–0.50 vs Normal 0.98, Side II 0.82.
- The side label is **not** a left/right level contrast under the documented odd/even sensor
  layout (sign agrees with the label on 20/38 fault files = chance). Both rails show the same
  fault excess on every axle box. So side-sign rules, per-box models (0.44) and per-car profile
  features (no OOF gain) are dead ends.
- The five always-missed Side I files are confident Normals (p ≥ 0.93): the slowest fault, the
  quietest fault, and three at 66–67 km/h among nine Normals. No threshold or prior boost helps.
- Rows measured on the frozen selection CV (`rail_ladder.json`): no shock channels **0.836**;
  previous winner 0.801; fold-local speed baseline 0.793; two-view ensemble 0.787; pruned-64
  0.780; no shock/time/votes 0.772; no time block 0.716; sub-window voting 0.715.
- The nested headline moved 0.764 → 0.757 with 22 candidates (more candidates = noisier
  selection). The pre-declared decision rule tolerated a 0.01 drop, so the "no shock channels"
  row was shipped. It changes five Test predictions (Test5, Test32, Test56 → Side I; Test9 →
  Side II; Test33 Side II → Side I); counts 60/2/6 → 56/6/6. Risk: over-calling Side I.

**Verification of the W7 re-ship (orchestrator, 18 Sep 17:40):** the model, model json, results
CSV, submission CSV, the CSV inside `predictions.zip`, the CLI output and the app copy all carry
the same md5s as the pins in `tests/test_ps3_stream.py` (pkl `a73f4fd8…`, json `0db85f40…`, csv
`dae7c1c0…`); the zip holds exactly the four CSVs. The focused test line below was re-run by the
orchestrator on 18 Sep 17:50: 101 passed, exit 0. To re-check:
```bash
md5sum models/ps3/rail.pkl models/ps3/rail.json submission/nebulax/rail_predictions.csv results/ps3/rail_predictions.csv
grep -n 'rail' tests/test_ps3_stream.py | grep -i 'md5\|"models/ps3/rail\|rail_predictions'   # pins must equal the md5s above
python scripts/ps3_predict.py --task rail --input readingmaterials/problem_statement/PS3/02_Datasets/Rail_Corrugation/Test --output /tmp/rail_cli.csv && md5sum /tmp/rail_cli.csv   # must equal the CSVs above
unzip -l submission/nebulax/predictions.zip     # exactly four CSVs at the root
python -m pytest tests/test_ps3_rail.py tests/test_ps3_rail_verify.py tests/test_ps3_stream.py tests/test_api_ps3.py tests/test_ps3_cli.py tests/test_ps3_report.py -q
```
If you prefer the previous, more conservative artefact, the byte-exact backup is in the W6
scratchpad `prev/` folder (may be gone with `/tmp`); otherwise `git show 83a5696:models/ps3/rail.pkl`
etc. restores it, and the three pins in `tests/test_ps3_stream.py` go back to
`70320d088018a192a5afdab3a840b222` (pkl), `544f86069c4f5093e5ba37a32d08638f` (json),
`59ebeb1506a63e4c81a869943b72db82` (csv).

At the W7 handoff, ideas not yet tried included shorter/more windows with the full-resolution
model as a second view; a "no shock" + "pruned" combination; 10 seeds instead of 3. Deep and
ROCKET rows, generative augmentation, transductive test use: measured or argued out, see
`results/ps3/rail_ladder.md` "skipped" section.

Subsequent diagnostics are preserved in `results/ps3/rail_revised_round.md`,
`rail_sensor_augmentation_round.md`, `rail_transfer_compare.md`, and the related transfer and
fine-tuning reports. Robust RMS, mirrored calibration, bounded gain jitter and sensor masking
did not clear the W7 selection gate. External encoder probes are exploratory; one blend cleared
the selection gate but needs its nested audit before any promotion. The ongoing Chronos extraction
is independent. The shipped W7 artefact and 68-row Test CSV remain unchanged.

## 5. How to run things

```bash
cd /mnt/berstorage/nebulax && export PATH=/home/administrator/miniconda3/envs/nebulax/bin:$PATH

# app (judges' default port)
uvicorn nebulax.api.main:app --host 127.0.0.1 --port 8765          # open http://127.0.0.1:8765/  (?page=twin for the fleet twin)
# web rebuild after editing web/src (then restart uvicorn)
(cd web && npm run build)
# CLI, Info-Kit style (task inferred from the input)
python predict.py --input readingmaterials/problem_statement/PS3/02_Datasets/Door/Test.csv --output door_predictions.csv
# retrain one task with its ladder and nested CV (rail ≈ 2.5 h with 22 rows; door/acv/shm minutes)
python scripts/ps3_train.py --task rail --seeds 0 1 2 --n-jobs 4 --ladder --tag baseline
# rebuild the submission zip and the four CSVs, validated
python scripts/ps3_submission.py --team nebulax
# leaderboard + heatmap from the ladder JSONs
python -m nebulax.ps3.report
# tests: focused (≈4 min; test_ps3_stream reproduces the CSVs from real data) and full (≈10 min, run once before a release commit)
python -m pytest tests/test_ps3_*.py tests/test_api_ps3.py -q
python -m pytest -q --ignore=tests/test_models_deep.py
```

## 6. Conventions that keep the submission honest

- Every scaler, threshold, boost, template and augmentation is fitted inside the training fold;
  model selection is nested or on a frozen split; the headline quoted anywhere is the nested
  number, and post-hoc selections are labelled as such (`docs/ps3_contract.md`).
- `tests/test_ps3_stream.py::FROZEN_MD5` pins the four models and four submission CSVs. Changing
  an artefact means changing exactly those pins, computed with `md5sum`, never typed by hand.
- The animated twin (W5) is a demo device: the last frame of every stream equals the batch
  predictor's rows; intermediate frames are causal previews. Rail/SHM cross-file order is a
  captioned demo device, not a claim.
- Rail: `speed < 20 km/h → Normal` is a dataset shortcut, documented as such, not physics.
- Nothing under `nebulax/sim`, `bench`, `demo`, the replay API or the old twin changes.

## 7. Implementation and remaining delivery items

- The twin now shows six systems: simulated brake air supply and axle bearing, plus LTA PS3 Door,
  ACV, Rail and SHM. The simulated door stays in the legacy simulator API but is not displayed.
  The LTA drawer action replays a model-predicted Test example; it is labelled prediction-only,
  with a browser-upload fallback when local Test data is unavailable. The layer plays into the
  chosen prediction and holds that frame; STOP restores the previous stream and cursor.
- `_DoorSeries` was removed and the millisecond prefix test fixed. The rail caption waits for
  pulse transitions before showing speed. SHM time uses its `sample` unit.
- The old verifier outputs were removed from `results/verify/`; the two cited audit documents
  now live in `docs/audits/`. The stale W4 `predictions_final.zip` was deleted.
- `/stream` rolls back upload-session rows and files if preview generation fails. Rail's v3
  features, duplicate grouping, fold-local boosts and augmentation were reviewed. The measured
  `lgbm_windows` loser and W7 artefact are retained for reproducibility.
- Delivery work still outside this implementation: record the ≤3-minute demo video and choose
  the registered team folder name. Organiser confirmation on pre-event code remains open.

Verification on 18 Sep: the focused PS3 Stream/Door/Common/API/Rail/CLI/Report suite passed;
`npm run build` passed; the packaged app received the new API, stream module and built assets;
`predictions.zip` validates (123 rows), and all four local predicted-example endpoints returned
HTTP 200 (Door `Test.csv`, ACV `acv_test_case.xlsx`, Rail `Test5.csv`, SHM `test02.csv`).
The W7 model and saved rail CSV hashes still match the pins in §4. The server was restarted and
`/` plus `/api/ps3/tasks` returned HTTP 200. The self-contained app ran on a temporary port;
without local Test data, its example route returned 404 as expected for the browser-upload
fallback. The release copy is 16 MB after removing ignored derived rail cache, and its CLI
reproduced the shipped `Test5.csv` prediction. The root rail CLI reproduced the saved Test CSV byte
for byte (md5 `dae7c1c0e922e554533aae053f7b262d`).

## 8. Working with delegate agents (how W5–W7 were run)

- Orchestrator writes a brief (`docs/handoff/briefs/`), the agent implements, never commits or
  installs, never runs the full suite, and writes a report in the format the brief specifies.
- agy (Gemini 3.8 Flash High) is launched by a human from the repo root:
  `setsid nohup agy --model gemini-3.8-flash-high --dangerously-skip-permissions --print-timeout 10h -p "<prompt>" > <log> 2>&1 &`
- Modelling changes go in as declared ladder rows through `scripts/ps3_train.py --ladder`; the
  ship rule is "the new row is the selection-CV winner AND the nested headline does not fall by
  more than 0.01", with a byte-exact backup of the artefacts taken first.
- Contract modules (`common.py`, `scoring.py`, `submission.py`) are frozen for agents.

## 9. File map (PS3 only)

```
nebulax/ps3/            common.py (task protocol, timestamp codec, model IO)  scoring.py (organiser metrics)
                        door.py door_features.py  acv.py acv_features.py  shm.py shm_features.py
                        rail.py rail_features.py rail_windows.py  stream.py (animated previews)  report.py  submission.py
nebulax/api/ps3.py      /api/ps3/tasks, /{task}/predict (batched, session token), /{task}/stream, /{task}/example/stream, /results/{token}.csv
scripts/                ps3_train.py  ps3_predict.py  ps3_submission.py ; predict.py at the root (Info-Kit CLI)
models/ps3/             {door,acv,rail,shm}.{pkl,json}   (hash-pinned)
results/ps3/            *_cv.{json,md} *_ladder.{json,md} *_predictions.csv leaderboard.{md,json} ablation_heatmap.html
submission/nebulax/     predictions.zip predictions_baseline.zip *_predictions.csv app/ Optional_Items/ demo_video.PLACEHOLDER.md
web/src/                predict/ (landing page)  components/ (twin, LayersPanel, DetailPanel)  state/ps3StreamStore.js ; web/dist built
docs/                   ps3_writeup.md demo.md ps3_contract.md provenance.md research/ps3_addendum.md handoff/
tests/                  test_ps3_*.py test_api_ps3.py test_ps3_stream.py (hash pins)
```
