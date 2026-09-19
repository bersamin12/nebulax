# NEBULA X - PS3 handoff (updated 19 Sep 2026)

Written by the orchestrator (Claude) for whoever picks the project up from here. Everything below
is verifiable from the repo; where a number comes from a JSON file, the file is named. The
previous handoff (18 Sep) described the W7 rail re-ship; the rail and SHM artefacts have since
been replaced once more (section 4) and the web app was rebuilt around an overview page and a
guided Digital Twin page (section 5).

## 1. State at a glance

| item | state |
|---|---|
| Repo | https://github.com/bersamin12/nebulax.git, branch `main`. History was restarted on 19 Sep: `d2f6424` "Initialize NEBULA X PS3 and bearing/brake research" holds everything up to the 18 Sep artefacts, `454bab1` adds the web app work of 19 Sep. Anyone holding a clone of the old `nebulatemp` repository must re-clone. |
| Uncommitted work | none, apart from a stray `Pasted image.png` at the repo root (not part of the project; delete it) |
| Python envs | research: `/home/administrator/miniconda3/envs/nebulax/bin/python` (full `environment.yml`: xgboost, torch, aeon, ...). App: `.conda-app/` created by `scripts/start_app.py` from `submission/nebulax/app/requirements.lock.txt` (no xgboost, stumpy, torch, aeon, river). The base miniconda python lacks deps; never use it. |
| App server | `python scripts/start_app.py` (rebuilds `web/dist`, then uvicorn on **8765**, `http://127.0.0.1:8765/`). A second server from the research env has been running on 8830 (`pgrep -af uvicorn`); kill it when no longer needed. |
| Ports reserved | 8000 and 5173 (dev), 8765 is the README default for judges |
| Organiser data | `readingmaterials/problem_statement/PS3/02_Datasets/` (gitignored, read-only, 7.6 GB); `nebulax.ps3.common.test_dir(task)` resolves each Test folder |
| Feature caches | `data/ps3_cache/` (gitignored); rail features at `rail/v3` per file, the shipped rail model metadata says `v2` and loads fine |

## 2. Deliverables checklist (organiser rules, PS3 spec "Submission")

| deliverable | status | where |
|---|---|---|
| `predictions.zip` with the four `*_predictions.csv` at archive root | regenerated 18 Sep 22:18 (rail coherence artefact + SHM skew gate); exactly four CSVs, md5s equal the pins in `tests/test_ps3_stream.py::FROZEN_MD5` (checked 19 Sep) | `submission/nebulax/predictions.zip` |
| Demo video <= 3 min showing all four tabs | **NOT DONE**, placeholder only | `submission/nebulax/demo_video.PLACEHOLDER.md`; script in `docs/demo.md` (its rail/SHM numbers are the 18 Sep afternoon ones, refresh them from section 3 before recording) |
| Runnable app for a non-technical user | done; self-contained copy, code and built site synced on 19 Sep | `submission/nebulax/app/` (`./run.sh 8765`), README inside |
| Optional write-up, code, models | done | `submission/nebulax/Optional_Items/`, `docs/ps3_writeup.md` |
| Team folder name | **still `submission/nebulax`**; the app brands itself "Team Bus MRT Walk". Rename before packaging (`scripts/ps3_submission.py --team <name>` regenerates into `submission/<name>/`) |
| Rules check on pre-event code | **not confirmed** with the organisers |

## 3. Scores (honest headlines; selection values are post-hoc and must not be quoted as the headline)

| task | metric | honest headline | selection CV | organiser portal | source |
|---|---|---|---|---|---|
| Door | IoU-weighted F1 | 0.9818 +- 0.0364 (5 contiguous blocks, nested) | 1.000 (post-hoc) | not reported | `results/ps3/door_ladder.json` `nested.iou_f1_mean` |
| ACV | rank decay | 0.9792 +- 0.0510 (leave-one-case-out, 6 cases, exploratory) | pre-registered rule, no selection | not reported | `results/ps3/acv_ladder.json` `selected.score` |
| Rail | macro F1 | **0.8051 +- 0.1291** (nested, 15 outer folds, 23 candidates) | 0.8441 +- 0.1059 ("no shock + same-side axle-box coherence") | **0.83104** for the shipped CSV (previous W7 CSV: 0.7994) | `results/ps3/rail_ladder.json` `nested.macro_f1_mean` |
| SHM | 1 - MAPE | **0.9813** (nested LOO, 64 folds) | gate chosen in 62 of 64 outer folds | **0.972795** for the shipped CSV (previous: 0.971753) | `results/ps3/shm_ladder.json` `nested.score` |

`results/ps3/leaderboard.md` is generated from these JSONs by `python -m nebulax.ps3.report`; do
not hand-edit it. The overview page of the app quotes the same four numbers from the leaderboard.

## 4. What changed in the models since the 18 Sep handoff

Rail (evening of 18 Sep): one row, "no shock + same-side axle-box coherence" (lgbm, 201
features), was added to the 22 historical rows and the full 23-row nested CV rerun. It won the
frozen selection CV (0.8441 vs W7's 0.8365) and cleared the declared nested gate (0.8051 vs the
0.7471 floor), so it replaced the W7 artefact: `models/ps3/rail.pkl` (794 KB, md5
`4e725692227a2a060838e529fe4bcf9a`), `rail.json` (`13cbd218cc2bc8f462a28ef588760d56`),
`rail_predictions.csv` (`65feab749662d2aac226a192e470867c`; 57 Normal, 4 Side I, 7 Side II). It
changed two of the 68 Test predictions against W7 and the portal scored it 0.83104. Full story:
`results/ps3/rail_ladder.md`, `docs/ps3_writeup.md` (rail section), and the exploratory rounds in
`results/ps3/rail_revised_round.md`, `rail_sensor_augmentation_round.md`, `rail_transfer_*.md`,
`rail_moirai2_*.md`, `rail_timae_ssl.md`, `rail_foundation_round.md`. None of the encoder or
fine-tuning probes beat the shipped row; the encoder weights also exceed the 5 MB artefact limit.

SHM (evening of 18 Sep): a fixed physics gate was promoted. When a recording's stress skew is
positive, the log-Lasso prediction is averaged at equal weight with a plain m=5 rainflow
prediction (rainflow scale = median log ratio fitted on the training fold). Nested LOO MAPE
0.0203 -> 0.0187; the portal confirmed 0.971753 -> 0.972795. Ten of 16 Test predictions moved by
0.09 to 1.76 percent. `results/ps3/shm_score_round.md` has the audit table; the model is 6 KB.

The earlier W6/W7 diagnostics still hold and are the reason the rail work stopped where it did:
Side I is the whole problem (the five always-missed Side I files are confident Normals), the side
label is not a left/right level contrast, and side-sign rules, per-box models and per-car profile
features are dead ends. Read `docs/handoff/diagnostics/w6_rail_findings.md` before trying more.

To re-check the shipped artefacts:
```bash
md5sum models/ps3/*.pkl models/ps3/*.json submission/nebulax/*_predictions.csv   # must equal tests/test_ps3_stream.py::FROZEN_MD5
unzip -l submission/nebulax/predictions.zip                                        # exactly four CSVs at the root
python -m pytest tests/test_ps3_rail.py tests/test_ps3_rail_verify.py tests/test_ps3_stream.py tests/test_api_ps3.py tests/test_ps3_cli.py tests/test_ps3_report.py -q
```

## 5. The web app after 19 Sep (commit `454bab1`)

Design source: Artifact https://claude.ai/artifact/Br7QdkbkvyhZbHxqFeQPPm (Design canvas; a copy
of its main page is `docs/design/Main.dc.html`). Everything below is implemented in `web/src/`.

- **Overview page** (`web/src/overview/`, default route): hero, the four model cards with their
  architecture diagrams and literature, the EDA and ablation carousels, the two exploratory
  systems (UORED-VAFCLS bearings, MetroPT-3 brake air supply) as a two-page carousel, the team.
  Images live in `web/public/overview/`. Header nav: OVERVIEW / DIGITAL TWIN, START TUTORIAL,
  OPEN DIGITAL TWIN. Brand: "Team Bus MRT Walk · Train Digital Twin"; no em dashes in page text.
- **Digital Twin page** (`?page=predict`, `web/src/predict/`):
  - all six systems as compact tiles (`TaskColumn.jsx`): the four models with their CV score,
    Brake air supply and Axle bearing marked `(EXPLORATORY)`; clicking a tile selects it and
    shows its description, clicking it again hides it. The old PREDICTIONS (4) / ALL SIX toggle
    and the `?systems=all` parameter are gone.
  - **file check dialog** (`FileConfirm.jsx`, rules in `fileCheck.js`): a pick never queues
    directly. Door: header chips with the required Datetime / Motor current / Door leaf position
    highlighted (aliases transcribed from `door_features.CANONICAL_COLUMNS`); Rail: 129 columns
    with a speed column first; SHM: headerless single numeric column; ACV: the server's
    `POST /api/ps3/acv/validate` report. Only matching files are queued; the server re-checks.
  - **tutorial** (`Tutorial.jsx`): eight-step spotlight tour, auto-starts on a first visit
    (localStorage `nx.tour.seen`), restarts from TUTORIAL in the header or `?tour=1`
    (`?tour=N` jumps to step N). The dim is four panels around the spotlight, so the highlighted
    control stays usable while the tour is open.
  - **(i) definitions** for every explanation variable (`numberDefs.js`; keys match the
    `explanation.numbers` the API sends per task).
  - **STOP** beside RUN aborts the request in flight; rows already predicted stay, the cut-off
    file and the rest of the queue go back to `queued`. A file the server had finished anyway
    comes back as "already predicted" on the next RUN and is marked done, not failed.
  - a chevron on every predictions row, a "CLICK A ROW TO INSPECT IT" hint, and a grey guidance
    line under the table that says what to do next per system and state.
- **Scaling** (`App.jsx`): the 1440 x 900 console scales to fill the window, up to 3x on
  2.5K/4K screens (the 3D viewport draws at device pixel ratio x console scale,
  `lib/consoleScale.js`), down to 0.6 on a small desktop window (then the stage scrolls).
  Phones and tablets get a "limited support" bar and the console fitted to the screen.
- **Backend changes** (also synced to `submission/nebulax/app/nebulax/`): `nebulax/models/
  __init__.py` imports each model family tolerantly and lists the skipped ones in
  `nebulax.models.UNAVAILABLE` with one `RuntimeWarning`; `xgboost` is imported lazily inside
  `StackingRFXGBLogReg`. Without this the door predictor failed in the app env
  ("No module named 'xgboost'") because `door_features` reaches `nebulax.models.physics` through
  the package. `POST /api/ps3/{task}/stream` now returns `errors` instead of a 500 when the one
  upload was refused (a duplicate name after STOP).
- Verified on 19 Sep with headless Chrome driven over CDP (`google-chrome --headless=new`): tour
  click-through, STOP at 2/6 rail files then RUN to 6 rows, the four dialog variants, 2560 x 1440
  and iPhone renders; Door `Test.csv` through the API in the app env: 38 rows, 0 errors;
  `pytest tests/test_api_ps3.py tests/test_ps3_stream.py` in the research env: exit 0 (the hash
  pins in section 4 held).
- Not done: the console does not reflow (by design), the fleet twin page (`?page=twin`) still
  uses "—" as its empty-cell placeholder in `lib/format.js`, and the demo video.

## 6. How to run things

```bash
cd /mnt/berstorage/nebulaxmain

# app, one command (creates .conda-app on first run, builds web/dist, serves on 8765)
python scripts/start_app.py                      # add --no-browser on a server; --check only verifies
# or by hand, research env
export PATH=/home/administrator/miniconda3/envs/nebulax/bin:$PATH
(cd web && npm run build) && uvicorn nebulax.api.main:app --host 127.0.0.1 --port 8765
# pages: /  (overview)   /?page=predict  (digital twin)   /?page=predict&tour=1   /?page=twin (fleet twin)

# CLI, Info-Kit style (task inferred from the input)
python predict.py --input readingmaterials/problem_statement/PS3/02_Datasets/Door/Test.csv --output door_predictions.csv
# retrain one task with its ladder and nested CV (rail ~ 3 h with 23 rows; door/acv/shm minutes)
python scripts/ps3_train.py --task rail --seeds 0 1 2 --n-jobs 4 --ladder --tag baseline
# rebuild the submission zip and the four CSVs, validated
python scripts/ps3_submission.py --team nebulax
# leaderboard + heatmap from the ladder JSONs
python -m nebulax.ps3.report
# tests: focused (~4 min; test_ps3_stream reproduces the CSVs from real data) and full (~10 min, once before a release commit)
python -m pytest tests/test_ps3_*.py tests/test_api_ps3.py -q
python -m pytest -q --ignore=tests/test_models_deep.py
```

After editing `nebulax/` or `web/src/`, copy the change into `submission/nebulax/app/` too
(`diff -rq nebulax submission/nebulax/app/nebulax -x __pycache__` must be empty; replace
`submission/nebulax/app/web/dist` with a fresh `web/dist`).

## 7. Conventions that keep the submission honest

- Every scaler, threshold, boost, template, gate and augmentation is fitted inside the training
  fold; model selection is nested or on a frozen split; the headline quoted anywhere is the
  nested number, and post-hoc selections are labelled as such (`docs/ps3_contract.md`).
- `tests/test_ps3_stream.py::FROZEN_MD5` pins the four models and four submission CSVs. Changing
  an artefact means changing exactly those pins, computed with `md5sum`, never typed by hand.
- The animated twin is a demo device: the last frame of every stream equals the batch
  predictor's rows; intermediate frames are causal previews.
- Rail: `speed < 20 km/h -> Normal` is a dataset shortcut, documented as such, not physics.
- The app shows only what the API sends: the explanation panel renders `explanation.numbers`,
  `trace` and `viewport` as they arrive, and the file check dialog is a preview of the server's
  own reader rules, never a substitute for them.
- Nothing under `nebulax/sim`, `bench`, `demo`, the replay API or the old twin changes.

## 8. Remaining delivery items

1. Record the <= 3-minute demo video (`docs/demo.md` run sheet; the app's tutorial follows the
   same left-to-right order, so `?page=predict&tour=1` is a good opening shot).
2. Choose the registered team folder name and regenerate `submission/<name>/`.
3. Get organiser confirmation on pre-event code.
4. Optional: sweep "—" out of the fleet twin page if the no-em-dash rule is meant to cover it.

## 9. Working with delegate agents (how W5-W7 were run)

- Orchestrator writes a brief (`docs/handoff/briefs/`), the agent implements, never commits or
  installs, never runs the full suite, and writes a report in the format the brief specifies
  (`docs/handoff/diagnostics/w*_agy_report.md`).
- agy (Gemini 3.8 Flash High) is launched by a human from the repo root:
  `setsid nohup agy --model gemini-3.8-flash-high --dangerously-skip-permissions --print-timeout 10h -p "<prompt>" > <log> 2>&1 &`
- Modelling changes go in as declared ladder rows through `scripts/ps3_train.py --ladder`; the
  ship rule is "the new row is the selection-CV winner AND the nested headline does not fall by
  more than 0.01", with a byte-exact backup of the artefacts taken first.
- Contract modules (`common.py`, `scoring.py`, `submission.py`) are frozen for agents.

## 10. File map (PS3 and app)

```
nebulax/ps3/            common.py (task protocol, timestamp codec, model IO)  scoring.py (organiser metrics)
                        door.py door_features.py  acv.py acv_features.py  shm.py shm_features.py
                        rail.py rail_features.py rail_windows.py  stream.py (animated previews)  report.py  submission.py
nebulax/models/         __init__.py (tolerant family imports, UNAVAILABLE)  boosting.py (lazy xgboost)  physics.py (door DWT features) ...
nebulax/api/ps3.py      /api/ps3/tasks, /{task}/predict (batched, session token), /{task}/stream, /acv/validate, /{task}/example/stream, /results/{token}.csv
scripts/                start_app.py (one-command launcher)  ps3_train.py  ps3_predict.py  ps3_submission.py ; predict.py at the root (Info-Kit CLI)
models/ps3/             {door,acv,rail,shm}.{pkl,json}   (hash-pinned)
results/ps3/            *_cv.{json,md} *_ladder.{json,md} *_predictions.csv leaderboard.{md,json} ablation_heatmap.html  + per-round rail/shm notes
submission/nebulax/     predictions.zip *_predictions.csv app/ (self-contained copy) Optional_Items/ demo_video.PLACEHOLDER.md
web/src/                App.jsx (shell, scaling, routes)  overview/ (landing page)  predict/ (digital twin: TaskColumn, FileConfirm, fileCheck,
                        ResultsTable, ExplanationPanel, numberDefs, Tutorial, usePs3Predict)  components/ (Header, viewport, twin panels)
                        lib/consoleScale.js  styles.css ; web/public/overview/ images ; web/dist built (gitignored)
docs/                   HANDOFF.md (this file) run_app.md app_contract.md ps3_contract.md ps3_writeup.md demo.md design/Main.dc.html handoff/{briefs,diagnostics}
tests/                  test_ps3_*.py test_api_ps3.py test_ps3_stream.py (hash pins)
```
