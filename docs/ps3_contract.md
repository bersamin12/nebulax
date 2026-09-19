# PS3 contract

The interface every Problem Statement 3 agent codes against. Frozen on 18 Sep 2026; changing
anything here means changing `tests/test_ps3_common.py` and telling the other PS3 agents.

Code: `nebulax/ps3/common.py` (types, paths, timestamps, registry), `nebulax/ps3/scoring.py`
(the four organiser metrics), `nebulax/ps3/submission.py` (validator + packer).
Tests: `python -m pytest tests/test_ps3_scoring.py tests/test_ps3_common.py`.

---

## 1. The `Task` protocol

A subsystem module is `nebulax/ps3/<name>.py` for `name` in `door | acv | rail | shm`. It defines
one task object and registers it at import time:

```python
from nebulax.ps3.common import BaseTask, PredictionResult, Trace, Viewport, register_task

class RailTask(BaseTask):
    name = "rail"                      # label / accepts / output_filename come from BaseTask

    def load(self, path):  ...         # -> Raw    one input file
    def featurise(self, raw): ...      # -> Feats  usually a DataFrame, one row per predicted item
    def predict(self, feats, model=None): ...   # -> PredictionResult

register_task(RailTask())
```

| Attribute | door | acv | rail | shm |
|---|---|---|---|---|
| `label` | `Door` | `ACV` | `Rail Corrugation` | `SHM` |
| `accepts` | `.csv` | `.xlsx`, `.xls` | `.csv` | `.csv` |
| `output_filename` | `door_predictions.csv` | `acv_predictions.csv` | `rail_predictions.csv` | `shm_predictions.csv` |

Consumers call `get_task(name)` (never `import nebulax.ps3.rail` directly - `get_task` imports the
module lazily and raises a clear `KeyError` when it does not exist yet), then either the three
steps or the `BaseTask.run(path, model=None)` shortcut. `model=None` means "load the committed
artefact". `to_rows(result)` and `explain(result)` have working defaults on `BaseTask`; override
them when a task needs more than a pass-through.

`PredictionResult` (one per input file; for door, one for the whole `Test.csv` stream):

| Field | Meaning |
|---|---|
| `task` | `door \| acv \| rail \| shm` |
| `file_id` | source basename **with** extension (`""` for the door stream) |
| `rows` | the organiser-schema rows that land in the CSV (section 3) |
| `numbers` | headline explanatory numbers, `dict[str, float]` |
| `trace` | a `Trace` for the explanation chart, or `None` |
| `viewport` | a `Viewport` for the 3D twin, or `None` |
| `extras` | the task's own scratch space (per-item frames, importances, warnings) - never written to a CSV |

## 2. The `Explanation` payload

`explain(result)` returns an `Explanation`; `explanation.as_dict()` is exactly the JSON the API
sends and the predict page renders, so keep it small (a few hundred trace points, not 10,000):

```json
{
  "file_id": "Test7.csv",
  "numbers": {"side_i_score": 0.81, "speed_kmh": 52.0},
  "trace": {"x": [2.0, 2.5, 3.0], "y": [0.1, 0.9, 0.2],
            "marks": [{"x": 2.5, "label": "6.3 cm peak", "kind": "peak"}],
            "label": "wavelength PSD, Side I axleboxes"},
  "viewport": {"car": 3, "side": "I", "health": "crit", "component": "axlebox_c3_p1"}
}
```

* `numbers` - every value is coerced to `float`; use readable snake_case keys, they are shown as-is.
* `trace.x` - the task's own x axis (door: native timestamp strings; rail: wavelength in cm; SHM:
  stress range; ACV: ISO timestamps). `x` and `y` must be the same length.
* `trace.marks` - free-form annotation dicts; `{"x": ..., "label": str, "kind": str}` for a point,
  add `"x1"` for a span (door uses spans for the predicted cycle boundaries).
* `viewport.car` - `1..8` or `None`; `side` - `"I" | "II" | "L" | "R"` or `None`;
  `health` - `"ok" | "warn" | "crit"`; `component` - a mesh/agent-readable name
  (`car3_ac1`, `door_L1`, `axlebox_c3_p1`, `car5_bogie1`).

Constructors validate: an out-of-range car, an unknown health state or an x/y length mismatch
raises immediately rather than shipping a broken payload to the browser.

## 3. CSV schemas (`nebulax.ps3.submission.CSV_HEADERS`)

Copied from `04_Example_Submission/`; header order is part of the contract.

| Task | Header | Row |
|---|---|---|
| door | `start_time,end_time,prediction` | one row per **predicted segment** in the single `Test.csv` stream; no `file_id`. `prediction` in `Normal \| Abnormal resistance` |
| acv | `file_id,ranked_cars` | `ranked_cars` = every car in that file, most- to least-likely faulty, two-digit ids exactly as the file's own headers spell them, joined by `\|` |
| rail | `file_id,prediction` | `prediction` in `Normal \| Side I \| Side II` |
| shm | `file_id,prediction` | a single positive finite cumulative-damage number |

`file_id` is the source file name **including its extension**, case-sensitive, with no directory
part. Door timestamps may be written in the native `Y-M-D-H-M-S-ms` form (`2023-7-5-0-11-17-664`,
nothing zero padded, last field whole milliseconds) or any ISO form; use
`common.format_door_timestamp` and the round trip is exact.

## 4. Validator and packer

```python
from nebulax.ps3.submission import expected_ids_for, pack, validate_csv, validate_zip

validate_csv("rail", path, expected_ids_for("rail")).raise_for_errors()
pack([door_csv, acv_csv, rail_csv, shm_csv], "submission/<team>/predictions.zip")
validate_zip(zip_path, {"rail": expected_ids_for("rail")}).raise_for_errors()
```

`validate_csv(task, path, expected_ids=None, *, expected_cars=None)` returns a `ValidationReport`
(`ok`, `errors`, `warnings`, `n_rows`, `ids`, `raise_for_errors()`) and checks:

* header exactly as the example, in order (door may carry one extra trailing `confidence`, which
  the Door Info Kit permits - reported as a warning);
* `file_id` is a bare, case-sensitive basename with the task's extension;
* no duplicate keys; set equality with `expected_ids` when given (door: `expected_ids` must be
  `None`, it has no id column);
* no empty cells and no NaN/Inf tokens anywhere;
* label vocabularies (`DOOR_LABELS`, `RAIL_LABELS`); SHM predictions positive and finite;
* ACV `ranked_cars`: two-digit ids joined by `|`, no repeats, and - when `expected_cars` maps the
  file id to `common.acv_car_ids(<that xlsx>)` - every car in the file exactly once;
* door: timestamps parse, `start < end`, no two predicted segments overlap (touching is fine).

`pack` writes the CSVs at the archive root only, refuses anything not named `*_predictions.csv`,
and is byte-deterministic (fixed member order and timestamps) so a re-run is diffable.
`validate_zip` re-extracts every member and re-runs `validate_csv`. Attempting a subset of the
four subsystems is fine - that is what the specification's Overall/Average split is for
(`scoring.combined_scores`).

## 5. Metrics (`nebulax/ps3/scoring.py`)

Exact transcriptions of the Info Kits; the tests reproduce every worked example in them.

| Function | Subsystem | Returns |
|---|---|---|
| `iou_f1(true_segments, pred_segments)` | door | `score`, `soft_recall`, `soft_precision`, `matches`, `misses`, `false_positives`, counts |
| `rank_decay(ranked, true_car, n=None)` / `rank_decay_mean(...)` | acv | `(n - (r - 1)) / n`, 0 when the true car is not ranked |
| `macro_f1(y_true, y_pred)` / `class_f1_report(...)` | rail | macro F1 over the **fixed** three-label vocabulary, absent classes included as F1 = 0 |
| `mape_score(y_true, y_pred)` | shm | `mape`, `score = max(0, 1 - mape)`, per-file `ape` |
| `combined_scores(per_task)` | all | `overall` (sum / 4) and `average` (sum / attempted) |

Segments are `(start, end, label)` with the boundary as a datetime, a native/ISO string or epoch
milliseconds; a DataFrame with `start_time`/`end_time`/`prediction` (or `t_start`/`t_end`/`status`)
is accepted directly, so `common.read_door_segments(...)` output drops straight in.

**IoU matching**, per the kit: same-label only, `IoU > 0`, one-to-one, highest IoU first, credit =
the IoU itself. **Tie-break (ours, the kit does not specify one):** equal-IoU candidates are
ordered by `(-iou, true_index, pred_index)`, so the earliest true segment wins, then the earliest
prediction. Pass segments in stream order for a reproducible result.

## 6. Paths and artefacts

| Constant | Path | Notes |
|---|---|---|
| `data_root()` | `readingmaterials/problem_statement/PS3/02_Datasets` | override with `NEBULAX_PS3_DATA`; **read-only**, never write into the organisers' clone |
| `MODEL_DIR` | `models/ps3` | committed artefacts, `<task>.pkl` + `<task>.json` sidecar, < 5 MB each (`save_model` refuses more) |
| `CACHE_DIR` | `data/ps3_cache` | gitignored; per-file features, resampled spectra |
| `RESULTS_DIR` | `results/ps3` | `<task>_cv.json` / `.md`, `leaderboard.md` |

`dataset_dir/train_dir/test_dir/labels_path(task)` resolve the organisers' folder names
(`Door`, `ACV`, `Rail_Corrugation`, `SHM`); door has no `Train/`/`Test/` subfolders, so both point
at the dataset dir. `save_model(task, obj, meta)` writes the pickle plus a json sidecar carrying
the git rev, the save time and whatever CV summary the trainer passes; `load_model(task)` and
`model_meta(task)` read them back.

## 7. The fold-local rule (non-negotiable)

Every scaler, threshold, template, rule weight, feature selection, bias correction and
augmentation is fitted **inside the training fold**. Model selection is nested (inner split) or on
a frozen outer split - never on the CV that produces the headline number. Augmented copies stay in
their source file's fold and held-out files are never augmented. The frozen validation schemes are
in the plan's W4 section: door = 5 contiguous time blocks of the raw stream, segmentation and
classification re-run end to end per block; rail = stratified 5-fold by file x 3 seeds plus
contiguous-filename-block and leave-speed-range-out stress splits; acv = leave-one-case-out with
the ranker frozen before each held-out case (six cases: exploratory); shm = LOO plus repeated
5x10-fold outer with nested inner selection. Every `results/ps3/<task>_cv.json` records the
scheme, the seeds and the git rev.

## 8. Fixtures

`tests/fixtures/ps3/` holds tiny verbatim slices of all four datasets (see its `README.md` for
exactly how each was cut). They are schema fixtures - 100 rows of a 10 kHz rail file is 10 ms of
signal - so a test that needs a full second of data synthesises it.
