# NEBULA X — Track 3

## Problem Statement 3 submission

NEBULA X attempts all four released PS3 tasks: Door segmentation/classification, ACV leaking-car
ranking, Rail corrugation classification and SHM fatigue-damage regression. The landing page is a
single upload → explain → download workflow.

From the repository root, with Node.js and Python 3.11 or Conda available:

```bash
python scripts/start_app.py
# open http://127.0.0.1:8765/
```

The launcher installs app dependencies, builds the frontend and starts the server. A compact
bearing and brake air supply replay is included in `demo_data/`. Use
`python scripts/start_app.py --prepare-fleet` to generate and score a larger optional Fleet twin.
See `docs/run_app.md` for manual steps, development mode and troubleshooting.

### Run with Docker

With Docker Desktop running, use these commands from the repository root:

```powershell
docker compose up -d                 # start the existing image
docker compose up --build -d         # rebuild after code, dependency, frontend, or model changes
docker compose logs -f app           # follow app logs (Ctrl-C stops following logs)
docker compose down                  # stop the app
```

Open `http://127.0.0.1:8765/`. For the deployed app, Cloud Run proxy, batch predictions,
monitoring, costs, and deployment plans, see the [Google Cloud guide](docs/google_cloud.md).

After the first dependency install, the app runs offline. Select a subsystem, choose the
corresponding released Test file/folder,
press **RUN**, inspect a result and download the organiser-format CSV. Rail's 68 files are uploaded
in bounded batches and processed sequentially.

The Info Kit-compatible CLI infers the task from the input:

```bash
python predict.py --input readingmaterials/problem_statement/PS3/02_Datasets/Door/Test.csv --output door_predictions.csv
python predict.py --input readingmaterials/problem_statement/PS3/02_Datasets/ACV/Test --output acv_predictions.csv
python predict.py --input readingmaterials/problem_statement/PS3/02_Datasets/Rail_Corrugation/Test --output rail_predictions.csv
python predict.py --input readingmaterials/problem_statement/PS3/02_Datasets/SHM/Test --output shm_predictions.csv
```

Useful submission files:

| path | purpose |
|---|---|
| `submission/nebulax/predictions.zip` | four final organiser-schema CSVs at archive root |
| `submission/nebulax/predictions_baseline.zip` | frozen baseline comparison |
| `docs/ps3_writeup.md` | approach, validation, ladder and limitations |
| `docs/ps3_model_pipeline.md` | inputs, models and processing for all four tasks |
| `docs/run_app.md` | one-command and manual local startup guide |
| `docs/demo.md` | rehearsed ≤3 minute recording script |
| `results/ps3/leaderboard.md` | machine-sourced model ladder with JSON key provenance |
| `docs/ps3_contract.md` | task, explanation, schema and fold-local contracts |
| `readingmaterials/problem_statement/PS3/` | official PS3 statement, four Info Kits and example output schemas |
| `results/ps3/portal_scores.md` and `submission/history/` | prior portal score record and archived artifacts |

Run focused PS3 tests with:

```bash
python -m pytest tests/test_ps3_*.py tests/test_api_ps3.py
```

## Axle bearing and brake air supply research

The research portion retains the axle bearing and brake air supply work: Ottawa bearing and
MetroPT pneumatic adapters, simulators, benchmark results, and the optional advisory. The old
Cranfield and synthetic Door research datasets and their benchmark results were removed. The
scored PS3 Door task, model, result files and app workflow remain. The **Fleet twin** page runs
from the bundled two-train replay in a fresh checkout and uses the larger local data when present.

| area | paths |
|---|---|
| behavioural twin | `nebulax/sim/`, `scripts/generate.py`, `data/sim/` |
| health twin and API | `nebulax/demo/`, `nebulax/api/`, `nebulax/advisory/` |
| adapters/features/models | `nebulax/adapters/`, `nebulax/features/`, `nebulax/models/` |
| benchmark | `nebulax/bench/`, `configs/`, `results/leaderboard.md` |
| 3D web app | `web/`, `assets3d/` |

Python is not installed editable: run commands from the repository root. The research environment
is described by `environment.yml`; app dependencies are pinned in
`submission/nebulax/app/requirements.lock.txt`. Node dependencies are locked by `web/package-lock.json`. Raw
dataset provenance, licences and redistribution cautions are recorded in `docs/provenance.md`.

To reproduce the earlier twin rather than PS3:

```bash
python scripts/download_data.py --dataset all --out data/raw
python scripts/generate.py --subsystem all --n-trains 10 --days 30 --seed 0 --out data/sim --p-healthy 0.4 --max-faults 1 --store-every 10
python scripts/run_bench.py --config configs/core.yaml --max-workers 3
python scripts/score_for_demo.py
```

See `docs/architecture.md`, `docs/results.md`, `docs/parameters.md` and `docs/provenance.md`
for the retained research record.
