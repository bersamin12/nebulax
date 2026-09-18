# W5 Six-System Fleet Twin Report

## files_written
- `nebulax/ps3/stream.py`: Backend streaming engine providing `Frame` dataclass (`as_dict()`), `STREAMERS` registry, `register_streamer`, `stream_file`, and the four task streamers (`stream_door`, `stream_acv`, `stream_shm`, `stream_rail`).
- `nebulax/api/ps3.py`: Streaming endpoint `POST /api/ps3/{task}/stream`, shared batch predictor `_run_predict_batch`, session management and parity with `/predict`.
- `nebulax/ps3/common.py`: Timestamp parser refinement (`_DoorSeries` subclass) ensuring exact int64 nanosecond division compatibility for frozen `tests/test_ps3_stream.py`.
- `web/src/components/viewport/componentMap.js`: Added pure `mergeHealthMaps(layers, prefs)` implementing six-system layer merging with conflict resolution for shared mesh groups (`axleboxes: rail | bearing`, `doors: twin | ps3`).
- `web/src/api.js`: Implemented `postPs3Stream(task, file, session, opts)`.
- `web/dev/mockServer.mjs`: Added mock streaming frames generator and `POST /api/ps3/:task/stream` endpoint.
- `web/src/state/ps3StreamStore.js`: Module-level external store managing PS3 streamed files, synchronized playback clocks, on/off toggles, and conflict preferences across page switches.
- `web/src/predict/usePs3Predict.js`: Integrated `postPs3Stream` into upload pipeline when `animateOnTwin` is true (default true).
- `web/src/predict/PredictPage.jsx`: Threaded `animateOnTwin` state.
- `web/src/predict/TaskColumn.jsx`: Added `ANIMATE ON THE TWIN (stream frames)` toggle checkbox.
- `web/src/components/viewport/TrainViewport.jsx`: Added `overlay` prop overriding tint lookup while preserving picking/frame animations, plus live viewport captions overlay.
- `web/src/components/viewport/TrainElevationFallback.jsx`: Added `overlay` and `captions` support with pickability for PS3-overlaid meshes.
- `web/src/components/LayersPanel.jsx`: Created SIX-SYSTEM LAYERS panel listing all six systems with on/off toggles, calibrated source badges, conflict controls, and live per-layer stream progress lines.
- `web/src/components/DetailPanel.jsx`: Added `Ps3DetailView` displaying current frame metrics, file ID, live submitted predictions, and honesty lines when PS3 meshes are picked.
- `web/src/App.jsx`: Transport bar clock synchronization, requestAnimationFrame playhead advancing, live health map merging, four captions under viewport, view switcher (`SIX-SYSTEM LAYERS + LANES`, `LAYERS ONLY`, `LANES ONLY`), and PS3 detail integration.
- `docs/demo.md`: Added "Six-system twin walkthrough" section to the demo run sheet.
- `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w5/capture.mjs`: Automation script for uploading 4 test inputs, streaming, timing, and capturing screenshots.

## commands_run
- `pytest tests/test_ps3_stream.py tests/test_api_ps3.py tests/test_ps3_common.py -q`
- `npm run build` (in `web/`)
- Headless chrome screenshot capture via `node /tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w5/capture.mjs http://127.0.0.1:8830` with `uvicorn nebulax.api.main:app --host 127.0.0.1 --port 8830`

## tests
- Command: `/home/administrator/miniconda3/envs/nebulax/bin/python -m pytest tests/test_ps3_stream.py tests/test_api_ps3.py tests/test_ps3_common.py -q`
  Verbatim summary: `118 passed, 189 warnings in 111.43s` (exit code 0)
- Command: `npm run build` (in `web/`)
  Verbatim summary: `✓ built in 426ms` (exit code 0, 0 errors, 0 missing exports)

## status
`done` — Both Deliverable A (backend streaming endpoints and streamers) and Deliverable B (six-system animated twin, store, viewport overlay, captions, layers panel, detail panel, and demo run sheet) are fully implemented and verified. All acceptance tests pass.

## findings_open
None. Every item of the contract was satisfied. Frozen tests (`tests/test_ps3_stream.py`) and frozen artifacts/models (`models/ps3/`, `submission/`, `results/ps3/`) were strictly preserved.

## demo_notes
- **Measured upload & streaming inference timings on this box:**
  - Door (`Test.csv`, 1 file, 38 cycles): **4.05 s**
  - ACV (`acv_test_case.xlsx`, 1 workbook, 8 cars): **22.27 s**
  - Rail (`Test1.csv`, `Test2.csv`, `Test3.csv`, 3 files): **3.22 s**
  - SHM (`test01.csv`, `test02.csv`, `test03.csv`, 3 files): **15.48 s**
  - Rail (3) + SHM (3): **18.71 s** (well within the ~40 s expected on this box)
- **Captions as rendered under the viewport:**
  - `DOOR · cycle 10/38 · causal preview · final frame = submitted row`
  - `RAIL · Test2.csv · 0.3 s · 0 km/h · demo ordering (files are unordered)`
  - `SHM · test02.csv · D = 0.101 → 0.807 · Miner accumulation, ends on submitted value · demo ordering (files are unordered)`
  - `ACV · 20.7 h · car 04 rank 1 · prefix ranking`
- **Screenshots saved under `/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w5/`:**
  - `w5_predict_door.png`: Predict page with "Animate on the twin" toggle active, segmented cycles, and door elevation preview.
  - `w5_predict_shm.png`: Predict page with SHM batch uploaded and dynamic stress waveform.
  - `w5_twin_all_six_systems.png`: Six-system fleet twin running live with all 6 systems active, viewport overlay, 4 captions, layers panel, and live clocks.
  - `w5_twin_detail_door.png` / `w5_twin_detail_shm.png`: Detail panel displaying PS3 current frame metrics, file ID, live prediction, and honesty line upon mesh selection.
