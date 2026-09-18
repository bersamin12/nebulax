# W5 — Animated six-system fleet twin (handover to agy / Gemini, 18 Sep 2026)

You are taking over one well-bounded feature on /mnt/berstorage/nebulax. The acceptance tests
already exist and are frozen: **`tests/test_ps3_stream.py`**. Read its module docstring first; it
is the contract. Your job is to make every test in that file pass without editing it, then wire
the result into the Fleet twin page so all six systems animate.

## Rules (same as every agent on this repo)

- Python: `/home/administrator/miniconda3/envs/nebulax/bin/python`, run from the repo root
  (`python -m pytest ...`). No editable install. Node/npm are on PATH, `web/node_modules` is
  installed.
- Do NOT `git commit/add/stash` (the orchestrator commits). Do NOT install packages. Do NOT write
  into `readingmaterials/problem_statement/` (organisers' data, read-only).
- Do NOT edit `tests/test_ps3_stream.py`. Do NOT edit or re-train anything under `models/ps3/`,
  `submission/`, `results/ps3/`. Do NOT change the outputs of `nebulax/ps3/{door,acv,rail,shm}.py`
  (`test_frozen_artefacts_and_submission_unchanged` and the four `_batch_rows` comparisons pin
  this). You may add small read-only helpers to those modules if you need them, but the safer
  pattern is to import their existing functions from the new `nebulax/ps3/stream.py`.
- Do NOT run the full pytest suite (≈10 min). Run only:
  `python -m pytest tests/test_ps3_stream.py tests/test_api_ps3.py tests/test_ps3_common.py -q`
  plus `tests/test_ps3_<task>.py` for any task module you touched.
- Ports 8000 and 5173 are reserved; pick your own for any server you start and stop it after.
- Heavy loops: 4 workers max.
- No verification run is scheduled after you; write the report at the end (format below) and stop.

## Why this exists

The submission is done (commit 47b6438): four PS3 subsystems, `submission/nebulax/predictions.zip`,
the upload-and-predict app (landing page) and the frozen replay twin (`?page=twin`, six-car sim
with brake air supply, axle bearings and doors). The demo video will be recorded from the twin
page, and the user wants **all six systems animated on the one 8-car model**: the two sim systems
from the replay as today, plus the four PS3 systems from the uploaded Test files.

Only two PS3 datasets carry a real clock, so "animated" is defined per task, and honesty rules
are part of the contract:

| System | Source | What animates | Honesty rule |
|---|---|---|---|
| Door | PS3 `Door/Test.csv` (50 Hz stream, ms timestamps) | one frame per completed cycle; the leaf tints as each cycle closes | frame k = the rows of cycles 0..k, causal |
| ACV | PS3 `acv_test_case.xlsx` (30 s telemetry, 8 cars) | the ranking recomputed on the prefix seen so far; the leaking car drifts to rank 1 | prefix ranking = the batch pipeline on the truncated workbook |
| SHM | PS3 `SHM/Test/test*.csv` (one stress channel) | running Miner damage climbing inside a file | final × D(prefix)/D(full), monotone, ends on the submitted value |
| Rail corrugation | PS3 `Rail_Corrugation/Test/Test*.csv` (1 s at 10 kHz) | speed from the tacho as the second plays; class appears at the end of the second | no class before the whole second; cross-file order is a demo device and is captioned as such |
| Brake air supply | replay sim | already animated | unchanged |
| Axle bearing | replay sim | already animated | unchanged |

**The last frame of every stream carries exactly the rows the batch predictor writes.** Intermediate
frames are causal previews. Every caption on the twin must say so ("causal preview · final frame =
submitted row").

## Deliverable A — backend (make the Python tests pass first)

1. **`nebulax/ps3/stream.py`** (new). `Frame` dataclass (field order in the test docstring; add
   `as_dict()`), `STREAMERS` dict, `register_streamer(name, fn)`, `stream_file(task, path,
   model=None, *, max_frames=200)`; unknown task -> `KeyError`. Register the four streamers at
   import. When `model is None` load the committed artefact the same way the API does
   (`nebulax.api.ps3.task_model(name)` honours `NEBULAX_PS3_MODELS`; `common.load_model` does not)
   — use `task_model` so tests that point the model dir at a temp folder behave.
   - **door**: `door_features.load_stream`, `segment`, `cycle_features`, then
     `DoorTask.predict(feats, model)` once on the full stream; `result.extras["proba"]` gives
     `p_abnormal` per cycle; `feats.table["end_ms"]` / `start_ms` give the clock (t = end_ms of
     cycle k minus the stream's first row, in seconds). The classifier is per-cycle given the
     artefact, so slicing the full result is causal; `test_door_frames_are_one_per_cycle_and_causal`
     checks that by truncating the raw stream. If it fails, do a true prefix loop instead.
     `file_id` is `""` (as the batch `PredictionResult`). Viewport per cycle: car 1, `door_L1`,
     crit for "Abnormal resistance", else ok.
   - **acv**: `acv_features.load_case(path)` -> `ACVCase`; a prefix is
     `ACVCase(file_id, cars, time[:k], {name: df.iloc[:k] for ...}, unmapped, meta)`; then
     `ACVTask().featurise(case_k, ranker)` (pass the ranker so the mask matches), set
     `feats.attrs["file_id"]`, `ACVTask().predict(feats, ranker)`. Cutoffs: `n_steps = min(max_frames,
     n_rows)` evenly spaced row counts, last = all rows. `t` = hours from `time[0]` to `time[k-1]`.
     The final frame's rows must be `ACVTask().run(path, model).to_rows(...)` verbatim (simplest:
     compute the batch result once and use its rows for the last frame; the prefix result at
     k = n_rows must equal it anyway). Every frame ranks every header car exactly once (cars with
     no data sort last, as the batch code already does).
   - **shm**: `x = shm_features.load_signal(path)`; batch result via `SHMTask().run(path, model)`
     gives `final` (float of `rows[0]["prediction"]`). Cutoffs: `n_steps = min(max_frames, n)`
     evenly spaced sample counts, last = n. For each cutoff:
     `r, _, c = rainflow_cycles(x[:cut], method="4point", residue="half")`;
     `D = miner_damage(r, c, exponent=5.0)`; value = `final * D / D_full`; row
     `{"file_id": name, "prediction": f"{value:.9g}"}`; `numbers = {"damage_running": value,
     "damage_final": final, "cycles_so_far": float(r.size)}`; `t = float(cut)`. This literal
     form is what the test's reference computes (rel 1e-6). Cost measured on this box: one full
     pass 0.17 s, so 60 frames ≈ 6 s per 581k-sample file. Do not "optimise" into a different
     residue treatment; if you want it faster, cache the turning points and slice them, but the
     boundary point of each prefix must still be treated as `turning_points(x[:cut])` would.
     Viewport: `bogie_frame`, health as the batch (`crit` ≥ 0.6, `warn` ≥ MODE_SPLIT, else ok)
     using the running value.
   - **rail**: batch result via `RailTask().run(path, model)` (rows + numbers). Read the raw array
     once (`rail_features.read_rail_csv`); `n_steps = min(max_frames, 10)`; frame k at
     `t = (k+1)/n_steps` seconds; `numbers["speed_kmh_so_far"] = speed_from_pulse(arr[:round(t*FS_HZ), 0])`;
     intermediate `rows = []`; the final frame carries the batch rows and the batch numbers
     (plus `speed_kmh_so_far`). Viewport: the batch viewport on the final frame; on intermediate
     frames the same car/side with health `ok`.
   - All `numbers` values must be finite floats (NaN -> drop the key or 0.0, as the batch code does).

2. **`POST /api/ps3/{task}/stream`** in `nebulax/api/ps3.py`. Same multipart contract as
   `/predict` (`files`, optional `session` form field or query param), **exactly one file**
   (400 with a detail containing "one file" for 0 or >1). Reuse the `/predict` code path for the
   batch prediction and the session append — factor the per-file loop into a helper both routes
   call, so the CSV bytes cannot diverge — then call `stream_file(task, saved_path, model,
   max_frames=60)` and add `"frames": [f.as_dict() ...]` to the same response body. Errors mirror
   `/predict` (same `detail` for a wrong suffix; 404 unknown task; 413 over the cap; 503 when the
   task is unavailable). `GET /api/ps3/tasks` output must not change.

3. **`web/src/api.js`**: `postPs3Stream(task, file, session=null, opts)` next to `postPs3Predict`.
   **`web/dev/mockServer.mjs`**: a `/api/ps3/:task/stream` route with made-up frames of the same
   shape (door: one frame per mock row; others: 10 frames), so the page can be built offline.

## Deliverable B — web (the twin page animates six systems)

4. **`mergeHealthMaps(layers, prefs)`** in `web/src/components/viewport/componentMap.js` — pure,
   no imports (the file has none; keep it importable from node — the test runs it under node
   22). `layers = {twin: Map|null, door, acv, rail, shm}` (PS3 entries are `ps3HealthMap()`
   results or null), `prefs = {axleboxes: "rail"|"bearing", doors: "twin"|"ps3"}`. Returns one
   `Map` by `componentKey()`. Rail boxes and twin bearing boxes share keys
   (`${car}|bearing|axlebox_${axle}${L|R}`, see `railBoxComponent`); doors share
   `${car}|door|door_L${k}`. For each shared group, only the preferred source's entries survive
   (the other source's entries in that group are dropped even where the preferred one paints
   nothing); every other key from every source is kept. Use `indexHealthMap()` to flatten each
   PS3 layer first.

5. **PS3 stream store** `web/src/state/ps3StreamStore.js` (module-level, `useSyncExternalStore`
   like `replayStore.js`): per task, the list of files streamed in this browser session, each
   with its frames, plus a per-layer playhead. The predict page (`web/src/predict/`) calls
   `postPs3Stream` instead of `postPs3Predict` when the "animate on the twin" toggle is on (default
   on) and pushes the frames into the store; the batch semantics of the page (rows table, CSV
   download, explanation panel) are unchanged because the response body is a superset. Keep
   `postPs3Predict` for batches larger than one file if you prefer; then stream each file
   afterwards. The store survives the page switch (App.jsx switches pages by query param without
   a reload).

6. **Twin page layers** (`web/src/App.jsx`, `?page=twin`): a "LAYERS" panel listing the six
   systems: brake air supply, axle bearing, door (twin), door (PS3 upload), ACV, rail
   corrugation, SHM — each with on/off, a source badge (`sim · MetroPT-3 calibrated`, `sim ·
   Ottawa calibrated`, `LTA PS3 upload`), and the two conflict prefs (axle boxes: rail | bearing;
   doors: twin | PS3). PS3 layers with no streamed files show "upload on the Predict page".
   - **Clocks.** One transport bar (existing play/pause/speed) drives everything. Each PS3 layer
     has its own clock: the layer's files play back to back, each file's frames spread over
     `FILE_SECONDS` of wall time at speed 1 (default 6 s per file; door: the whole stream over
     30 s), scaled by the transport speed relative to the default, looping when the last file
     ends. Pause pauses all six. A per-layer progress line under the panel shows file id, frame
     step/n, and the layer's `t` with its unit.
   - **Tint.** Compute `mergeHealthMaps({twin: indexFrame(train), door, acv, rail, shm}, prefs)`
     every animation tick, where each PS3 layer's map is `ps3HealthMap({task, rows: frame.rows,
     explanations: [ {file_id, viewport: frame.viewport, numbers: frame.numbers} ], fileId})`
     for the current frame. Feed it to `TrainViewport` through a new optional `overlay` prop
     (a Map by componentKey) that overrides only the tint lookup while the frame keeps driving
     door leaf animation, picking, ticker and lanes. Do not break the two existing modes
     (`frame` alone, `healthMap` alone).
   - **Captions.** Under the viewport in twin mode, one line per active PS3 layer:
     `DOOR · cycle 12/38 · causal preview · final frame = submitted row`;
     `RAIL · Test17.csv · 0.6 s · 54 km/h · demo ordering (files are unordered)`;
     `SHM · test03.csv · D = 0.412 → 0.807 · Miner accumulation, ends on submitted value`;
     `ACV · 14.2 h · car 01 rank 1 · prefix ranking`. The rail and SHM ones must carry the
     "demo ordering" words.
   - **Detail panel**: clicking a PS3-painted mesh shows the current frame's numbers and the
     layer's file id; the existing sim components keep their panel.
   - `TrainElevationFallback` (WebGL off) gets the same overlay (it already takes a healthMap; the
     cheapest route is to pass `{components: Object.fromEntries(overlay)}`).

7. **`docs/demo.md`**: add a "Six-system twin" section to the 3-minute script: predict page →
   upload the four Test inputs with animate on (door Test.csv, acv_test_case.xlsx, 3 rail files,
   3 SHM files — the upload of 3+3 takes ~40 s on this box; say so) → switch to Fleet twin →
   turn on all layers → let it play 30 s → click one mesh per PS3 layer. Note which captions the
   presenter reads out (the honesty lines).

## Order of work and time budget (≈6–10 h)

1. `stream.py` + door/shm/acv/rail streamers until
   `python -m pytest tests/test_ps3_stream.py -q -k "not route and not merge"` is green (the
   real-data tests run here because the datasets are on this box; each must stream one file in
   under 60 s).
2. The route; `-k route` green; `tests/test_api_ps3.py` still green.
3. `mergeHealthMaps`; `-k merge` green.
4. Store, predict-page toggle, twin layers, captions, fallback, mock server, `npm run build` in
   `web/` clean (`npx vite build`, no warnings about missing exports).
5. Screenshot both pages against the real API (start `uvicorn nebulax.api.main:app --port <free>`
   with `NEBULAX_WEB_DIST` or the vite preview on another free port; headless chromium
   `--screenshot`; save under
   `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w5/`).
   Earlier screenshots of the app are in that scratchpad root (`ps3-*-realapi.png`) if you want
   the reference look.
6. `docs/demo.md` section. Stop every process you started.

If you run short of time, deliver in this order: door + SHM + ACV streamers and the route (tests
1–5, 7–8), then the twin door/SHM/ACV layers, then rail, then the fallback and mock server.

## Report (final message AND written to
`/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w5/report.md`)

`files_written`, `commands_run`, `tests` (each command + its verbatim summary line),
`status` (done / partial: what is missing), `findings_open` (anything in the contract you could
not meet and why — do not silently deviate), `demo_notes` (upload timings you measured, the
captions as rendered).
