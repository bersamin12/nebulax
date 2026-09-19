# Run the NEBULA X app locally

Run commands from this repository checkout. The app has one FastAPI server for the REST API, WebSocket replay and built React site. The PS3 page predicts faults from uploaded files using the four saved models in `models/ps3/`. The Fleet twin page uses the included compact bearing and brake replay, or the larger generated simulation and score data when available.

## Fastest start

For the container build and deployed Google Cloud app, see [the Google Cloud guide](google_cloud.md). With Docker Desktop's Linux engine running, `docker compose up --build` starts the same four-model web app at `http://127.0.0.1:8765/`.

Install Node.js/npm and either Python 3.11 or Conda. Then run this **one command** from the repository root on Windows, macOS or Linux:

```sh
python scripts/start_app.py
```

The launcher creates a local Python 3.11 environment (`.venv`, or `.conda-app` when Conda must provide Python 3.11), installs the pinned app dependencies, runs `npm ci` if needed, builds `web/dist`, checks that all four saved PS3 models exist, starts Uvicorn and opens `http://127.0.0.1:8765/`. If Conda reports that its Anaconda channels require Terms of Service acceptance, the launcher accepts the channels named in that error (`pkgs/main`, `pkgs/r`, `pkgs/msys2`) and retries once. The first start needs internet access to install dependencies. Later starts reuse the Python and Node installations, but rebuild the frontend to include current source changes. Press **Ctrl-C** in the terminal to stop the server. The environments and build files are local generated files; do not commit them.

Options:

```sh
python scripts/start_app.py --check                 # inspect prerequisites, do not change files
python scripts/start_app.py --port 8000 --no-browser
python scripts/start_app.py --prepare-fleet         # also generate and score the optional Fleet twin
```

The default command starts the full web/API process and **all four PS3 predictors**. It does not retrain models or regenerate the fleet on every start. The saved models are already in this repository. `--prepare-fleet` installs the additional packages in `requirements-fleet.txt`; it is a longer, CPU and disk intensive setup and only generates the fleet data when missing. It uses `results/runs.parquet` for the selected benchmark models. If the raw MetroPT-3 dataset is absent, it scores the synthetic fleet only.

## What is in this checkout

| Component | Current repository state | Needed for |
|---|---|---|
| `models/ps3/{door,acv,rail,shm}.pkl` | Present | PS3 prediction |
| `web/dist` | Built automatically by the launcher | Browser page |
| `demo_data/sim/index.json`, `demo_data/scores/scores.parquet` | Included compact T04/T09 replay | Fleet twin replay and fault injection in a fresh checkout |
| `data/sim/index.json`, `data/scores/scores.parquet` | Generated with `--prepare-fleet`; takes precedence over `demo_data` | Larger Fleet twin |
| `readingmaterials/problem_statement/PS3/02_Datasets/` | Supplied separately by organisers; not in this checkout | Released Test uploads and any PS3 retraining |
| `data/raw/metropt3/` | Downloaded separately | The optional real MetroPT-3 replay unit |

The app opens on the **Overview** page: a short landing page (the four cross-validation results, the four-step workflow, where each subsystem sits on the train) followed by collapsed sections holding the detail (each PS3 model with its architecture diagram and literature, the EDA, the ablation ladders, the two exploratory systems, the glossary), then the team. **Open the prediction workspace** in the hero or the PREDICT nav leads to the predict page (`?page=predict`); **Take the tour** opens it with a guided tour that spotlights one control at a time (the header's TUTORIAL button and `?page=predict&tour=1` do the same; it never starts by itself). The control inside the spotlight stays live, so a reader can pick a system, drop files or press RUN while the tour explains it. On the predict page every explanation variable has an (i) icon with its definition.

The app can start without the released datasets. Upload the organiser Test files through the **Digital Twin** (predict) page, or use compatible files with the required schemas. Use **Door** for one door stream CSV, **ACV** for an Excel case workbook, **Rail** for rail CSV recordings, and **SHM** for headerless stress CSVs. Choose a task, select or drop its files, press **RUN**, inspect the explanations and download the generated CSV. The Rail folder can contain many files; the page sends batches and the server scores them sequentially. **STOP** next to RUN ends a run after the file in flight: the rows already predicted stay, the rest of the queue waits for the next RUN. Click any row of the predictions table to inspect it (the train model and the explanation follow).

The overview is an ordinary responsive page and reads on a phone. The prediction workspace is built for desktop browsers: a fixed 1440 by 900 console scaled to fill the window, up on a 2.5K or 4K screen (the 3D view is drawn at the matching pixel ratio, so it stays sharp) and down to 60 percent on a small one, below which the page scrolls. Phones, tablets and windows narrower than 700 px get a notice asking for a larger screen, with an "open anyway" option that shows the console fitted to the screen under a limited-support bar.

On the predict page the six systems are a list with a thumbnail each (the four models with their CV score, the two exploratory systems marked as such); choosing one opens its description under it and closes the previous one, choosing it again hides the description. The 3D train orbits freely (drag, scroll to zoom, double-click to reset) and drifts back to the task's home framing a few seconds after the pointer is released. Choosing files opens a check dialog that lists the columns each file carries against what the model expects (Door: the required Datetime, Motor current and Door leaf position columns; Rail: 129 columns; SHM: a headerless single column; ACV: the server's workbook report) and queues only the files that match. For Door and ACV the dialog, and a strip under the file picker, also offer **Simulate missing columns**: switching an optional field off removes that column from every file of the next RUN before it is loaded (`drop_columns` on `POST /api/ps3/{task}/predict`), so the model's own fallback for an absent field is what runs; the explanation panel says which fields were removed. Rail needs all 129 columns and SHM is one column, so they have nothing to switch off.

Check the running backend at `http://127.0.0.1:8765/api/health` and the PS3 task list at `http://127.0.0.1:8765/api/ps3/tasks`. The task list reports whether each saved model can be loaded. The bundled fleet replay covers 12–14 September 2026 UTC and two trains.

## Manual modular startup

If you prefer to manage each step yourself, use Python **3.11** in a fresh environment and run:

```sh
python -m pip install -r submission/nebulax/app/requirements.lock.txt
cd web
npm ci
npm run build
cd ..
python -m uvicorn nebulax.api.main:app --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/`. Run Uvicorn from the repository root because the code and default data paths resolve there. `web/dist/index.html` must exist before Uvicorn starts or the browser route will not be mounted. The app dependency lock is sufficient for normal inference; `environment.yml` is the broader research environment for training and benchmarking. The advisory integration is optional and uses `ANTHROPIC_API_KEY` if configured; the app has a template fallback without the key.

To build the Fleet twin manually, first install the broader dependencies needed by the benchmark winners:

```sh
python -m pip install -r requirements-fleet.txt
python scripts/generate.py --subsystem all --n-trains 10 --days 30 --seed 0 --out data/sim --p-healthy 0.4 --max-faults 1 --store-every 10 --workers 4
python scripts/score_for_demo.py --skip-metropt3
```

The scoring step needs `results/runs.parquet`, which is present here. Remove `--skip-metropt3` only after supplying its raw data under `data/raw/metropt3/`. `--prepare-fleet` in the launcher performs these steps when required. Fleet setup may take many minutes and uses substantial disk space.

## CLI and development modes

The CLI uses the same PS3 prediction path and CSV writer as the app:

```sh
python predict.py --input path/to/Door/Test.csv --output door_predictions.csv
python predict.py --input path/to/ACV/Test --output acv_predictions.csv
python predict.py --input path/to/Rail_Corrugation/Test --output rail_predictions.csv
python predict.py --input path/to/SHM/Test --output shm_predictions.csv
```

For frontend hot reload, start the API on port **8000** and Vite in a second terminal:

```sh
python scripts/start_app.py --port 8000 --no-browser
```

```sh
cd web
npm run dev
```

Open `http://127.0.0.1:5173/`. Vite proxies `/api` and the replay WebSocket to port 8000. If the frontend is the only code being edited, leave the API running and let Vite rebuild in the browser. For backend changes, restart the API process; the one-command launcher does not enable Uvicorn reload.

## Common problems

| Symptom | Action |
|---|---|
| `Python 3.11 is required` | Install Python 3.11 or Conda. The launcher can create `.conda-app` when Conda is available. Python 3.14 is not the supported app environment. |
| `npm` not found | Install Node.js, then reopen the terminal. The checked-in `web/package-lock.json` is used by `npm ci`. |
| Root URL returns 404 | Build `web/dist` with `npm ci` and `npm run build`, then restart the API. |
| Predict page reports a model unavailable | Confirm all four `models/ps3/*.pkl` files exist and the pinned dependencies installed successfully. Check `/api/ps3/tasks` for detail. |
| Fleet twin has no trains | Check that `demo_data/sim/index.json` and `demo_data/scores/scores.parquet` exist, or run `python scripts/start_app.py --prepare-fleet`. |
| Port 8765 is occupied | Use `python scripts/start_app.py --port 8766`. |
| Organiser Test paths are missing | Obtain the released files separately; the repository intentionally does not redistribute raw datasets. |

See `docs/ps3_model_pipeline.md` for each predictor's inputs and processing, `docs/ps3_contract.md` for CSV schemas, and `docs/architecture.md` for the Fleet twin data flow.
