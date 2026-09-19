# App contract (W3): scores, API, replay frames, advisory

Everything in `nebulax/demo/`, `nebulax/api/`, `nebulax/advisory/` and `web/` codes to this file.
Change it only by editing this file first. Frozen names are in backticks.

## 1. The fleet the demo shows

* Source `sim`, trains `T01`..`T10`, sim clock **2026-09-01T00:00Z .. 2026-09-30T23:59Z** (30 days).
  `data/sim/index.json` lists every run: `run_id`, `subsystem`, `train_id`, `car`, `component_id`,
  `healthy`, `fault_types`. Telemetry lives in `data/sim/source=sim/run_id=<run_id>/` as
  `telemetry.parquet` (long schema), `features.parquet`, `events.parquet`, `fault_log.parquet`,
  `meta.json`. Ground truth for the whole fleet: `data/sim/fault_log.parquet`.
* Instrumentation is sparse on purpose: per train there are 2 door runs (one door leaf each),
  1 pneumatic run (`apu_1`, `car=0` = unit level) and 2 bearing runs (one car each, all 8 boxes
  `axlebox_1L..4R`). A door run's telemetry has `component_id in {door_XY, train}` and its
  `car`. Everything else on the 6-car schematic has **no telemetry** and renders as `nodata`.
* MetroPT-3 is shown as one extra unit `MP3` (real Porto metro APU, Feb–Jul 2020) with the
  pneumatic pick only. Its clock is its own; the replay never mixes it with the sim fleet.
* A 6-car set: cars 1..6, doors `door_L1..L4` / `door_R1..R4` per car, `apu_1` per car
  (unit-level in the sim, `car=0` maps to the schematic's car 3 APU), 8 axle boxes per car.

## 2. Winners (from `results/leaderboard.md`, "Selected per subsystem"; frozen)

| subsystem | dataset | model (registry name) | input_kind | window | peer_norm | note |
|---|---|---|---|---|---|---|
| door | sim | `cusum_cycle_scalar` | `cycle_features` | - | True | val lift 4.24x, 4/12 events at budget |
| pneumatic | sim | `sparse_autoencoder` | `window_stats` | 360 | False | 3/3 events, precision 3/16 |
| bearing | sim | `cusum_cycle_scalar` | `cycle_features` | - | True | val lift 50.8x, 10/12 |
| pneumatic | metropt3 | `lgbm_residual` params `{"channels":[0,2,4,6,8,10,12],"n_jobs":4}` | `raw_window` | 360 | False | hand-picked, selection deferred on the page (no validation failure); 4/4 events, FA 0.105/scored-day, the only 4/4 row inside the FA budget on test |

The exact benchmark rows (config, params, split, thresholds) are in `results/runs.parquet`;
`nebulax/demo/scoring.py` reads its `WINNERS` from that frame by `(dataset, dataset_subsystem,
model, input_kind, window, peer_norm, contamination)` and fails loudly if a row is missing, is
not `status=ok`, or if more than one row matches. `contamination == 0.0` is part of the join
key: on door and bearing those six other keys match **two** ok rows that differ only in
contamination (0.0 and 0.05), and the leaderboard row this table quotes (door 4/12, bearing
10/12) is the clean fit.

Cranfield (door) and Ottawa (bearing) picks are **classification** stories on the slides, not
replay content: nothing in the app scores them.

## 3. Scores (`data/scores/`, produced by `scripts/score_for_demo.py`)

Schema = `nebulax.schema.SCORES_COLUMNS` exactly:
`timestamp, train_id, car, subsystem, component_id, model, score, threshold, alert, top_signals_json`.

* One row per **feature row** the winner scores (door cycle, pneumatic 360-step window,
  bearing 5-min window). `timestamp` = the row's `t_end`.
* `threshold` = the false-alarm-budget threshold, calibrated **exactly as the benchmark does**
  (`nebulax.bench.thresholds.calibrate` on the validation slice of the training trains only).
* `alert` = row belongs to an alarm **episode** (`nebulax.bench.metrics.episodes`: score >
  threshold for >= 3 rows **consecutive in time** on one series - a run is cut wherever the
  step exceeds the fold's `max_step_s` - and qualifying runs merged under a 1 h gap). A single
  row above threshold is therefore `alert=False`; the UI shows it as `warn`.
* Episodes are detected on the **scoreable** rows only, exactly the population the benchmark
  measures on: the sim blanks every row of a component after its `t_failure` (no repair is
  modelled), MetroPT-3 blanks the 24 h after a failure. A blanked row is still written with
  its score and its threshold, and always carries `alert=False`. `scored_train_days` in the
  manifest is counted on the same rows (per train, `nebulax.bench.metrics.train_days`), so
  `false_episodes_per_scored_train_day` is comparable with `results/runs.parquet`'s
  `budget_false_alarms_per_train_day`.
* `top_signals_json` = JSON list, at most 5 items, `[{"signal": <feature column>, "z": <float>,
  "value": <float>}, ...]`, ordered by |z| descending. `z` is the robust z-score of that feature
  against the fitted training distribution (median/MAD); if the model exposes a per-feature
  attribution use it instead and say so in the manifest.
* Protocol for the fleet: **leave-one-train-out**. For train `Tk` the winner is fitted on the
  healthy rows of the other nine trains (same `train_regime=normal_only` the benchmark used,
  its own validation carve for the threshold), then scores every row of `Tk`. Ten fits per
  subsystem; all are cheap except the sparse autoencoder (~17 s each, GPU if present).
* Files:
  * `data/scores/scores.parquet` - all three subsystems, all sim runs.
  * `data/scores/metropt3.parquet` - `train_id="MP3"`, `car=0`, `component_id="apu_1"`, the
    `lgbm_residual` w=360 row's own threshold, scored on the benchmark's test slice (11 Apr ->).
  * `data/scores/episodes.parquet` - one row per alarm episode, and the episode is the one
    `nebulax.bench.metrics.episodes` returned when `alert` was written (never a maximal run of
    array-adjacent `alert` rows: a door raises no cycle overnight, so two alarm blocks either
    side of the ~5 h gap are two episodes). Columns: `train_id, car, subsystem,
    component_id, model, t_start, t_end, peak_score, n_rows, episode_id, fault_type,
    t_onset, t_failure, lead_to_failure_h, matched, H_s` (joined to the fault log: matched
    when the episode overlaps `[t_onset - H, t_failure]`). `H_s` is the horizon that decided
    the row - **72 h** (259200) for the sim, **48 h** (172800) for MetroPT-3, each the H of
    its own benchmark row - so one file cannot silently mix two horizons.
  * `data/scores/manifest.json` - per subsystem: the winner row's `config_hash`, model, params,
    input_kind, window, peer_norm, thresholds per held-out train, fit seconds, n rows scored,
    n rows scoreable, `scored_train_days`, episode counts (matched / false), git rev,
    `generated_at`. The API serves it verbatim.
  * `data/scores/models/<subsystem>/<held_out_train>.pkl` - fitted winner + threshold +
    training feature medians/MADs, so `POST /sim/inject` re-scores without refitting.

Python API (`nebulax/demo/scoring.py`):

```python
WINNERS: dict[str, WinnerSpec]                      # keys door, pneumatic, bearing, metropt3
def fit_winner(subsystem: str, *, exclude_train: str | None, index=None) -> FittedWinner
def load_fitted(subsystem: str, held_out_train: str) -> FittedWinner   # from the pkl
def score_run(fw: FittedWinner, run_dir: Path) -> pd.DataFrame          # SCORES_COLUMNS
def score_frames(fw: FittedWinner, long: pd.DataFrame, features: pd.DataFrame,
                 *, train_id: str, car: int, component_id: str) -> pd.DataFrame
def episodes_from_scores(scores: pd.DataFrame, fault_log: pd.DataFrame | None, *,
                         H: float = 3 * 86400.0, k: int = 3, merge_gap_s: float = 3600.0,
                         max_step_s: float | None = None, scoreable=None) -> pd.DataFrame
def scoreable_from_fault_log(scores: pd.DataFrame, fault_log: pd.DataFrame | None, *,
                             rule: str = "post_failure") -> np.ndarray
```

`k`, `merge_gap_s` and `max_step_s` are the fold's own episode definition (`FittedWinner`, or
`k_consecutive` / `merge_gap_s` / `episode_max_step_s` in `manifest.json`); `max_step_s=None`
measures the budget off the frame. `scoreable` is the mask above; `None` re-derives it from
the frame. `scripts/score_for_demo.py --episodes-only` rebuilds `episodes.parquet` and the
manifest's episode / event / train-day numbers from a published `scores.parquet` and
`manifest.json` alone - no fit, no loader, under a minute.

`FittedWinner` is picklable and carries `subsystem, model_name, params, input_kind, window,
peer_norm, threshold, feature_names, train_median, train_mad, held_out_train, fitted_at`.

## 4. API (`nebulax/api/`, FastAPI, `uvicorn nebulax.api.main:app --port 8000`)

All routes under `/api`. JSON, timestamps ISO-8601 UTC strings. Reads `data/scores/*` and the sim
partitions lazily with pyarrow filters; never loads a whole telemetry parquet into memory.

| route | returns |
|---|---|
| `GET /api/health` | `{status:"ok", scores_loaded:bool, n_trains, clock:{start,end}}` |
| `GET /api/trains` | `[{train_id, line:"NSL", cars:6, instrumented:[{car, subsystem, component_id, run_id}], faults:[fault-log rows]}]` (MP3 included with its own clock) |
| `GET /api/train/{id}/state?ts=` | one **frame** (section 5) for that train at `ts` (default: end of clock) |
| `GET /api/train/{id}/component/{cid}/series?signal=&from=&to=&car=&max_points=2000` | `{signal, unit, points:[[ts, value], ...]}`; downsampled by min/max bucketing to `max_points`; door cycles come from the stored waveforms only |
| `GET /api/train/{id}/component/{cid}/scores?from=&to=&car=` | `{model, threshold, points:[[ts, score, alert]], top_signals: <latest top_signals_json>}` |
| `GET /api/train/{id}/component/{cid}/cycle?ts=&car=` | the door cycle waveform nearest `ts`: `{t:[...], pos:[...], pos_ref:[...], current:[...], pwm:[...]}`, 404 for non-door |
| `GET /api/alerts?ts=&train=` | episodes open or closed at `ts`, newest first, each with `advisory: null | Advisory` |
| `GET /api/kpis?ts=` | `{open_alerts, median_lead_h, fa_per_train_day, events_detected, events_total}` computed from episodes up to `ts` |
| `GET /api/bench/selection` | manifest + the six selection rows of `results/leaderboard.md` as JSON (dataset, subsystem, model, val lift, recall, precision, FA) |
| `WS /api/replay` | client sends `{cmd:"play"|"pause"|"seek"|"speed", ts?, speed?}`; server pushes frames (section 5) at 10 Hz wall-clock while playing; `speed` = sim seconds per wall second (default 8640 = 1 day / 10 s) |
| `POST /api/sim/inject` | body `{train_id, car, component_id, fault_type, severity_ramp_days, seed?}`; regenerates that component for the replay window with the fault onset at the current replay `ts` (or `t_onset` in body), re-scores with the fitted winner, writes to an in-memory overlay that the replay and all GET routes read first; returns `{run_id, n_rows, episodes:[...], elapsed_s}`; 400 for unknown fault types (`nebulax.schema.FAULT_TYPES`). Must finish in < 60 s for a door (30 days of one leaf, store_every=10) - shorten `days` to the remaining replay window if needed |
| `POST /api/advisory/{episode_id}` | runs `nebulax.advisory.advise(...)` for that episode, caches the result, returns the `Advisory` |
| `DELETE /api/sim/inject` | clears the overlay |

Static: `app.mount("/", StaticFiles(directory="web/dist", html=True))` when `web/dist` exists, so
one process serves the demo.

## 5. Replay frame (WS and `/state`)

```json
{
  "ts": "2026-09-14T08:30:00Z",
  "speed": 8640,
  "playing": true,
  "trains": [
    {"train_id": "T01",
     "components": [
       {"car": 1, "subsystem": "door", "component_id": "door_L1",
        "score": 3.2, "threshold": 2.4, "alert": true, "health": "crit",
        "model": "cusum_cycle_scalar", "top_signals": [{"signal": "closing_time", "z": 5.1, "value": 3.9}]},
       {"car": 3, "subsystem": "pneumatic", "component_id": "apu_1", "score": 0.1, "threshold": 0.26,
        "alert": false, "health": "ok", "model": "sparse_autoencoder", "top_signals": []},
       {"car": 2, "subsystem": "bearing", "component_id": "axlebox_3R", "health": "warn", "...": "..."}
     ]}
  ],
  "alerts": [{"episode_id": "T01-door_L1-3", "train_id": "T01", "car": 1, "subsystem": "door",
              "component_id": "door_L1", "t_start": "...", "t_end": null, "peak_score": 5.9,
              "fault_type": "friction", "lead_to_failure_h": 91.5, "advisory": null}],
  "kpis": {"open_alerts": 2, "median_lead_h": 61.0, "fa_per_train_day": 0.02,
           "events_detected": 9, "events_total": 14},
  "ticker": ["08:29 T01 car 1 door L1 · CUSUM 3.2 > 2.4 · episode open 6 h", "..."]
}
```

`health`: `crit` = inside an alarm episode at `ts`; `warn` = latest score > threshold but no
episode yet, or > 0.8 x threshold; `ok` = scored and below; `nodata` = component not
instrumented. Components not instrumented are **omitted** from `components`; the UI fills the
schematic with `nodata` for every other id. The latest score is the last scored row with
`timestamp <= ts` and not older than 6 h for every subsystem (doors are out of service up to 5.1 h every night, so the earlier 2 h door window went stale mid-episode each night; changed 18 Sep); older than that -> `stale: true` and
`health` falls back to `ok`.

## 6. Advisory (`nebulax/advisory/`)

```python
class Advisory(BaseModel):
    summary: str                 # <= 2 sentences, plain language for a depot engineer
    evidence: list[str]          # 2-5 bullets quoting the numbers it was given
    likely_component: str
    likely_fault: str            # one of schema.FAULT_TYPES[subsystem] or "unknown"
    recommended_action: str
    urgency: Literal["low", "medium", "high"]
    confidence: float            # 0..1
    cbm_steps: dict[str, str]    # keys: state_detection, health_assessment, prognostic_assessment, advisory
    source: Literal["claude", "template"]
    model: str                   # "claude-opus-5" or "template-v1"

def advise(alert: AlertContext, *, client=None, timeout_s: float = 20.0) -> Advisory
```

`AlertContext` (pydantic): `episode_id, train_id, car, subsystem, component_id, model_name, score,
threshold, t_start, t_end, duration_h, top_signals, recent_events: list[str], fleet_peer_summary:
str, glossary: str, fault_candidates: list[str]`. `advise` uses `claude-opus-5` through
`client.messages.parse(..., output_format=AdvisoryDraft)` with `output_config={"effort": "low"}`,
`max_tokens=1024`, the system prompt cached with `cache_control`; on missing
`ANTHROPIC_API_KEY`, any exception or timeout it returns the **template** advisory
(`source="template"`). Never blocks the demo on the network: `advise` itself runs the call on a
daemon thread with a hard join timeout, and the API calls `advise` in a thread.

`AdvisoryDraft` (`nebulax.advisory.schema`) is `Advisory` minus `source`/`model` and with
`cbm_steps` as a closed 4-field object. It exists because the SDK builds the structured-output
grammar from the pydantic model, and `Advisory`'s `cbm_steps: dict[str, str]` transforms to
`{"type": "object", "properties": {}, "additionalProperties": false}` - the model could only ever
return an empty dict. `AdvisoryDraft.to_advisory()` returns the `Advisory` with `source="claude"`,
`model="claude-opus-5"` and `likely_fault` narrowed to the alert's subsystem. `thinking` is set to
`{"type": "disabled"}`: Claude Opus 5 runs adaptive thinking by default and would spend the frozen
1024-token budget before writing the advisory.

Fix round 1 (18 Sep, after verification): (a) `Advisory.narrow_to` / `AdvisoryDraft.to_advisory`
clamp `confidence` to 0.3 whenever `likely_fault` ends up `"unknown"`; (b) the Claude call gets
`timeout = max(1.0, 0.8 * timeout_s)` and `max_retries = 0` so the SDK gives up before the daemon
join fires; (c) the template classifier resolves every top signal through an alias chain
(`raw -> _peer_delta/_peer_z stripped -> _<stat> stripped`) and scores faults by combined
evidence, with door `obstruction` / `limit_switch` reached through `recent_events` tokens
(`obstruction`, `ls_timeout`) because those are event types, not feature columns.
