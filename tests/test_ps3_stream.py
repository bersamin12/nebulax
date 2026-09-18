"""Acceptance tests for the animated PS3 twin (W5): ``nebulax/ps3/stream.py`` and the stream route.

These tests were written **before** the implementation and define its contract. The implementer
(see the W5 brief) codes against the names imported here; nothing in this file may be edited to
make an implementation pass. The one rule that matters more than any other:

    THE LAST FRAME OF EVERY STREAM CARRIES EXACTLY THE ROWS THE BATCH PREDICTOR WRITES.

Intermediate frames are causal previews (computed from a prefix of the file); the final frame is
the submission. The scored pipelines (``nebulax/ps3/{door,acv,rail,shm}.py``), the committed
artefacts under ``models/ps3/`` and ``submission/nebulax/*.csv`` are frozen - the last test here
pins their hashes.

Contract (``nebulax.ps3.stream``)
---------------------------------
``Frame`` dataclass with fields, in this order::

    task: str                 one of TASK_NAMES
    file_id: str              basename of the input ("" for door, as in PredictionResult)
    step: int                 0-based index of this frame within the file
    n_steps: int              total frames for this file
    t: float                  position on the file's own clock (see t_unit)
    t_unit: str               door "s" (seconds since the stream's first row); acv "h" (hours since
                              the first timestamp); shm "sample" (sample index); rail "s" (0..1)
    progress: float           0 < progress <= 1, non-decreasing, 1.0 on the final frame
    rows: list[dict]          organiser-schema rows AS THEY STAND at this step
    numbers: dict[str, float] finite floats only
    viewport: dict            Viewport.as_dict() shape
    final: bool               True on the last frame only

``Frame.as_dict()`` -> JSON-serialisable dict with exactly those keys.

``stream_file(task, path, model=None, *, max_frames=200) -> list[Frame]`` - dispatches on task
name through ``STREAMERS`` (``dict[str, Callable[[path, model, max_frames], list[Frame]]]``);
``register_streamer(name, fn)`` adds or replaces an entry (tests register dummies).

Per-task semantics (frozen):

door  one frame per completed cycle, in stream order; ``n_steps`` = number of cycles (max_frames
      is ignored); frame k's ``rows`` are the organiser rows of cycles 0..k; ``t`` = end of cycle
      k in seconds since the first row; ``numbers`` has ``p_abnormal`` for cycle k. CAUSAL: the
      stream truncated at the end of cycle k yields frames[:k+1] with identical rows.
acv   ``n_steps = min(max_frames, n_rows)`` frames at evenly spaced row-prefix cutoffs, the last
      being the whole file; frame rows = the ranking of the prefix computed exactly as
      ``ACVTask.run`` computes it for a workbook truncated at that row (hot quantile and masks
      over the prefix, same ranker artefact); ``t`` in hours since the first timestamp.
shm   ``n_steps = min(max_frames, n_samples)`` frames at evenly spaced sample cutoffs, the last
      being the whole file; ``rows[0]["prediction"]`` = final_prediction * D(prefix) / D(full)
      where D = Miner damage at exponent 5 of the repo's 4-point rainflow with the half-cycle
      residue (``shm_features.rainflow_cycles(x, method="4point", residue="half")`` +
      ``miner_damage(ranges, counts, exponent=5.0)``); formatted ``f"{value:.9g}"``; ``t`` = cutoff
      sample index; ``numbers`` has ``damage_running`` (the same float) and ``damage_final``.
rail  ``n_steps = min(max_frames, 10)`` frames at t = k/n_steps seconds through the 1 s file;
      intermediate frames have ``rows == []`` (a class needs the whole second) and
      ``numbers["speed_kmh_so_far"]`` = ``rail_features.speed_from_pulse`` over the samples seen so
      far (``pulse[:round(t * FS_HZ)]``); the final frame carries the batch rows and the batch ``numbers``.

API (``nebulax/api/ps3.py``): ``POST /api/ps3/{task}/stream`` - same multipart contract as
``/predict`` (``files``, optional ``session``) but exactly one file; the file is predicted with
the batch path and appended to the session exactly as ``/predict`` would, then streamed; the
response is the ``/predict`` body plus ``"frames": [Frame.as_dict(), ...]``. Errors mirror
``/predict`` (400 wrong suffix with the same ``detail``, 404 unknown task, 413 over the cap).

Web (``web/src/components/viewport/componentMap.js``): ``mergeHealthMaps(layers, prefs)`` - pure,
importable from node. ``layers`` is ``{twin: Map|null, door, acv, rail, shm}`` where each PS3
entry is a ``ps3HealthMap()`` result or null and ``twin`` is an ``indexFrame()`` Map or null;
``prefs`` is ``{axleboxes: "rail"|"bearing", doors: "twin"|"ps3"}``. Returns one Map keyed by
componentKey(); for every axle-box key only the preferred source's entry survives, for every
door key only the preferred source's entry survives, and no key from any other mesh group is
dropped.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nebulax.api import ps3 as api_ps3
from nebulax.ps3 import common
from nebulax.ps3.common import (
    TASK_NAMES,
    BaseTask,
    PredictionResult,
    Trace,
    Viewport,
    get_task,
    parse_door_timestamps,
    to_ms,
)
from nebulax.ps3.submission import CSV_HEADERS

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "ps3"
DOOR_CSV = FIXTURES / "door" / "train_slice.csv"
ACV_XLSX = FIXTURES / "acv" / "case01_slice.xlsx"
ACV_WIDE_XLSX = FIXTURES / "acv" / "case04_wide_slice.xlsx"
SHM_CSV = FIXTURES / "shm" / "train01_slice.csv"
SUBMISSION = REPO / "submission" / "nebulax"
MODEL_DIR = REPO / "models" / "ps3"

# Pin the submitted artefacts. Rail coherence and SHM skew-gated physics were promoted after W5;
# any later model or CSV change must update this contract and recheck final-frame equality.
FROZEN_MD5 = {
    "models/ps3/acv.json": "af07c4fddfe6df0367f834f92b312fa6",
    "models/ps3/acv.pkl": "016d5aa53ec7c3521b12dded5dc065d0",
    "models/ps3/door.json": "2da3931e74333674ca7014b94f5fbc4b",
    "models/ps3/door.pkl": "da03fe23d75f61973bab33d79fd92972",
    "models/ps3/rail.json": "13cbd218cc2bc8f462a28ef588760d56",
    "models/ps3/rail.pkl": "4e725692227a2a060838e529fe4bcf9a",
    "models/ps3/shm.json": "ecf2616cd43ed889b59737ae10789382",
    "models/ps3/shm.pkl": "9492f37b6a8e8024823ac58a618afbfc",
    "submission/nebulax/acv_predictions.csv": "ee5f0bfe75235d5ff3013b24be47ca2c",
    "submission/nebulax/door_predictions.csv": "b7cd00d119683182eab68c5392274eee",
    "submission/nebulax/rail_predictions.csv": "65feab749662d2aac226a192e470867c",
    "submission/nebulax/shm_predictions.csv": "b822c92edfbac958273527e46e2dbeba",
}

FRAME_KEYS = ["task", "file_id", "step", "n_steps", "t", "t_unit", "progress", "rows", "numbers", "viewport", "final"]
T_UNIT = {"door": "s", "acv": "h", "shm": "sample", "rail": "s"}


def _stream_mod():
    """Import lazily so the whole module reports one clear failure while stream.py is missing."""
    return importlib.import_module("nebulax.ps3.stream")


def _real_data_root() -> Path | None:
    root = common.data_root()
    return root if (root / "Door" / "Test.csv").exists() else None


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    return pd.read_csv(csv_path, dtype=str).fillna("").to_dict("records")


def _batch_rows(task_name: str, path: Path) -> list[dict[str, Any]]:
    """What the submission path writes for this file: the reference every final frame must equal."""
    task = get_task(task_name)
    result = api_ps3.run_file(task, path, api_ps3.task_model(task_name))
    return [{k: str(v) for k, v in r.items()} for r in task.to_rows(result)]


def _rows_str(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in r.items()} for r in rows]


def _check_frame_shape(frames: list[Any], task_name: str) -> None:
    """The invariants every stream obeys, whatever the task."""
    assert frames, "a stream must have at least one frame"
    n = len(frames)
    last_progress = 0.0
    for i, f in enumerate(frames):
        d = f.as_dict()
        assert list(d.keys()) == FRAME_KEYS, f"frame {i}: keys {list(d.keys())}"
        json.dumps(d)  # JSON-serialisable, no numpy scalars
        assert d["task"] == task_name
        assert d["step"] == i
        assert d["n_steps"] == n
        assert d["t_unit"] == T_UNIT[task_name]
        assert isinstance(d["t"], float) and math.isfinite(d["t"])
        assert 0.0 < d["progress"] <= 1.0
        assert d["progress"] >= last_progress, f"frame {i}: progress went backwards"
        last_progress = d["progress"]
        assert d["final"] is (i == n - 1)
        assert isinstance(d["rows"], list)
        for row in d["rows"]:
            assert list(row.keys()) == list(CSV_HEADERS[task_name]), f"frame {i}: row keys {list(row.keys())}"
        for k, v in d["numbers"].items():
            assert isinstance(k, str)
            assert isinstance(v, float) and math.isfinite(v), f"frame {i}: numbers[{k!r}] = {v!r}"
        Viewport(**d["viewport"])  # the viewport shape the 3D twin already understands
    assert frames[-1].as_dict()["progress"] == 1.0
    if len(frames) > 1:
        ts = [f.t for f in frames]
        assert ts == sorted(ts), "t must be non-decreasing"


def _synth_rail_second(*, speed_kmh: float, side: str | None = None, lam_mm: float = 50.0, amp: float = 8.0, seed: int = 0, n: int = 10000) -> np.ndarray:
    """A ``(n, 129)`` recording: 90-tooth pulse at ``speed_kmh`` plus optional corrugation on one side
    (the same construction as ``tests/test_ps3_rail.py::synth_rail``)."""
    from nebulax.ps3 import rail_features as rf

    rng = np.random.default_rng(seed)
    v = speed_kmh / 3.6
    dist = np.arange(n) / rf.FS_HZ * v
    arr = np.zeros((n, rf.N_CHANNELS + 1), dtype=np.float32)
    arr[:, 0] = (np.floor(dist / (rf.PULSE_DISTANCE_M / 2)).astype(np.int64) % 2).astype(np.float32)
    arr[:, 1:] = rng.normal(0.0, 1.0, size=(n, rf.N_CHANNELS)).astype(np.float32)
    if side is not None:
        boxes = np.flatnonzero(rf.side_mask(side))
        cols = np.sort(np.concatenate([2 * boxes, 2 * boxes + 1]))
        arr[:, 1:][:, cols] += (amp * np.sin(2 * np.pi * dist / (lam_mm / 1000.0)))[:, None].astype(np.float32)
    return arr


def _write_rail_csv(arr: np.ndarray, path: Path) -> Path:
    pd.DataFrame(arr).to_csv(path, index=False, header=[f"c{i}" for i in range(arr.shape[1])], float_format="%.4f")
    return path


# ======================================================================================
# 1. Final-frame equality on the committed fixtures (the test that matters)
# ======================================================================================


@pytest.mark.parametrize(
    "task_name,path",
    [("door", DOOR_CSV), ("acv", ACV_XLSX), ("acv", ACV_WIDE_XLSX), ("shm", SHM_CSV)],
    ids=["door", "acv", "acv-wide", "shm"],
)
def test_final_frame_equals_batch_rows_on_fixtures(task_name: str, path: Path) -> None:
    stream = _stream_mod()
    frames = stream.stream_file(task_name, path)
    _check_frame_shape(frames, task_name)
    assert _rows_str(frames[-1].rows) == _batch_rows(task_name, path)


def test_final_frame_equals_batch_rows_on_a_synthetic_rail_second(tmp_path: Path) -> None:
    """The rail fixture is 10 ms; a class needs the whole second, so synthesise one."""
    stream = _stream_mod()
    path = _write_rail_csv(_synth_rail_second(speed_kmh=54.0, side="II", lam_mm=60.0, amp=12.0, seed=3), tmp_path / "Synth7.csv")
    frames = stream.stream_file("rail", path)
    _check_frame_shape(frames, "rail")
    assert 1 <= len(frames) <= 10
    for f in frames[:-1]:
        assert f.rows == [], "rail has no class before the whole second is seen"
        assert "speed_kmh_so_far" in f.numbers
    assert _rows_str(frames[-1].rows) == _batch_rows("rail", path)
    assert frames[-1].rows[0]["file_id"] == "Synth7.csv"
    assert "speed_kmh" in frames[-1].numbers


# ======================================================================================
# 2. The same equality against the real Test files and the submitted CSVs (skips off-box)
# ======================================================================================


@pytest.mark.skipif(_real_data_root() is None, reason="organisers' PS3 datasets not present")
@pytest.mark.parametrize("task_name", ["door", "acv", "shm", "rail"])
def test_final_frames_reproduce_the_submitted_csv(task_name: str) -> None:
    """Every final frame is a row of ``submission/nebulax/<task>_predictions.csv`` and, taken
    together, the frames reproduce that CSV exactly (rail and SHM run every distributed file)."""
    stream = _stream_mod()
    root = _real_data_root()
    assert root is not None
    submitted = _read_rows(SUBMISSION / common.OUTPUT_FILENAMES[task_name])
    if task_name == "door":
        frames = stream.stream_file("door", root / "Door" / "Test.csv")
        _check_frame_shape(frames, "door")
        assert len(frames) == len(submitted) == 38
        assert _rows_str(frames[-1].rows) == submitted
        return
    inputs = sorted(common.test_dir(task_name, root=root).iterdir(), key=lambda p: common.natural_key(p.name))
    inputs = [p for p in inputs if p.suffix.lower() in common.ACCEPTED_SUFFIXES[task_name]]
    got: list[dict[str, str]] = []
    budget_s = 0.0
    for p in inputs:
        t0 = time.perf_counter()
        frames = stream.stream_file(task_name, p, max_frames=40)
        budget_s = max(budget_s, time.perf_counter() - t0)
        _check_frame_shape(frames, task_name)
        got.extend(_rows_str(frames[-1].rows))
    assert got == submitted
    # a demo cannot wait: one file must stream in well under a minute on this box
    assert budget_s < 60.0, f"{task_name}: slowest file took {budget_s:.1f} s to stream"


# ======================================================================================
# 3. SHM: running damage is the Miner accumulation profile, monotone, ends on the prediction
# ======================================================================================


def test_shm_running_damage_is_monotone_and_follows_miner() -> None:
    from nebulax.ps3 import shm_features as sfeat

    stream = _stream_mod()
    frames = stream.stream_file("shm", SHM_CSV, max_frames=25)
    _check_frame_shape(frames, "shm")
    assert len(frames) == 25
    values = [float(f.rows[0]["prediction"]) for f in frames]
    assert all(v > 0 and math.isfinite(v) for v in values)
    assert values == sorted(values), "cumulative damage cannot decrease"
    final = float(_batch_rows("shm", SHM_CSV)[0]["prediction"])
    assert frames[-1].rows[0]["prediction"] == f"{final:.9g}"
    assert values[0] < values[-1], "the curve must actually climb"
    for f in frames:
        assert f.numbers["damage_final"] == pytest.approx(final)
        assert f.numbers["damage_running"] == pytest.approx(float(f.rows[0]["prediction"]), rel=1e-9)
        assert f.rows[0]["file_id"] == "train01_slice.csv"

    # independent reference: the repo's own rainflow + Miner on the same prefixes
    x = sfeat.load_signal(SHM_CSV)

    def miner(prefix: np.ndarray) -> float:
        r, _m, c = sfeat.rainflow_cycles(prefix, method="4point", residue="half")
        return sfeat.miner_damage(r, c, exponent=5.0)

    d_full = miner(x)
    assert d_full > 0
    for f in frames:
        cut = int(round(f.t))
        assert 1 <= cut <= x.size
        expected = final * miner(x[:cut]) / d_full
        assert float(f.rows[0]["prediction"]) == pytest.approx(expected, rel=1e-6, abs=1e-12), f"step {f.step} (cut {cut})"
    assert int(round(frames[-1].t)) == x.size


# ======================================================================================
# 4. ACV: a prefix frame equals the batch pipeline run on the truncated workbook
# ======================================================================================


@pytest.mark.parametrize("path", [ACV_XLSX, ACV_WIDE_XLSX], ids=["8-param", "60-param"])
def test_acv_prefix_frame_equals_truncated_workbook(path: Path, tmp_path: Path) -> None:
    stream = _stream_mod()
    n_rows = len(pd.read_excel(path))
    frames = stream.stream_file("acv", path, max_frames=5)
    _check_frame_shape(frames, "acv")
    assert len(frames) == min(5, n_rows)
    assert frames[0].t == 0.0 or frames[0].t > 0.0  # hours since the first timestamp
    full = pd.read_excel(path)
    for f in frames:
        # the cutoff is recoverable from the frame's clock: rows whose time <= first + t hours
        times = pd.to_datetime(full.iloc[:, [i for i, c in enumerate(full.columns) if "time" in str(c).lower()][0]])
        cutoff = int((times <= times.iloc[0] + pd.Timedelta(hours=f.t) + pd.Timedelta(seconds=1)).sum())
        assert 1 <= cutoff <= n_rows
        trunc = tmp_path / f"{path.stem}_{f.step}.xlsx"
        full.iloc[:cutoff].to_excel(trunc, index=False)
        expected = _batch_rows("acv", trunc)
        assert len(expected) == 1
        assert f.rows[0]["ranked_cars"] == expected[0]["ranked_cars"], f"step {f.step}, cutoff {cutoff}"
        assert f.rows[0]["file_id"] == path.name
    assert frames[-1].rows == _batch_rows("acv", path)


def test_acv_every_frame_ranks_every_header_car_exactly_once() -> None:
    stream = _stream_mod()
    cars = common.acv_car_ids(ACV_XLSX)
    for f in stream.stream_file("acv", ACV_XLSX, max_frames=6):
        ranked = f.rows[0]["ranked_cars"].split("|")
        assert sorted(ranked) == cars and len(set(ranked)) == len(cars)


# ======================================================================================
# 5. Door: frames appear only when a cycle closes, and streaming a prefix is a prefix of the stream
# ======================================================================================


def _door_prefix(path: Path, end_ms: int, out: Path) -> Path:
    """The raw stream cut after the last row whose timestamp is <= ``end_ms`` (header kept verbatim)."""
    raw = pd.read_csv(path)
    ts = parse_door_timestamps(raw["Datetime"])
    keep = ts.astype("int64") <= end_ms
    raw[keep.to_numpy()].to_csv(out, index=False)
    return out


def test_door_frames_are_one_per_cycle_and_causal(tmp_path: Path) -> None:
    stream = _stream_mod()
    frames = stream.stream_file("door", DOOR_CSV)
    _check_frame_shape(frames, "door")
    batch = _batch_rows("door", DOOR_CSV)
    assert len(batch) == 3, "the fixture holds exactly three labelled cycles"
    assert len(frames) == 3
    for k, f in enumerate(frames):
        assert _rows_str(f.rows) == batch[: k + 1], f"frame {k} must hold cycles 0..{k}"
        assert "p_abnormal" in f.numbers and 0.0 <= f.numbers["p_abnormal"] <= 1.0
        assert f.file_id == ""
        # the frame's clock is the end of its cycle, in seconds since the first row
        end_s = (to_ms(batch[k]["end_time"]) - to_ms(batch[0]["start_time"])) / 1000.0
        assert f.t == pytest.approx(end_s, abs=0.021)
    # causality: the stream cut at the end of cycle 1 gives exactly frames[:2]
    cut = _door_prefix(DOOR_CSV, to_ms(batch[1]["end_time"]), tmp_path / "prefix.csv")
    prefix_frames = stream.stream_file("door", cut)
    assert len(prefix_frames) == 2
    assert [_rows_str(f.rows) for f in prefix_frames] == [_rows_str(f.rows) for f in frames[:2]]
    assert [f.numbers["p_abnormal"] for f in prefix_frames] == pytest.approx([f.numbers["p_abnormal"] for f in frames[:2]])


# ======================================================================================
# 6. Rail: one frame per tenth of a second, speed from the samples seen so far
# ======================================================================================


def test_rail_intermediate_frames_track_the_pulse_speed(tmp_path: Path) -> None:
    from nebulax.ps3 import rail_features as rf

    stream = _stream_mod()
    arr = _synth_rail_second(speed_kmh=40.0, seed=5)
    path = _write_rail_csv(arr, tmp_path / "Synth1.csv")
    frames = stream.stream_file("rail", path, max_frames=4)
    _check_frame_shape(frames, "rail")
    assert len(frames) == 4
    for f in frames:
        assert 0.0 < f.t <= 1.0
        n_seen = int(round(f.t * rf.FS_HZ))
        expected = rf.speed_from_pulse(arr[:n_seen, 0])
        assert f.numbers["speed_kmh_so_far"] == pytest.approx(expected, rel=1e-6)
    assert frames[-1].t == pytest.approx(1.0)
    assert frames[-1].rows[0]["prediction"] in common.RAIL_LABELS


# ======================================================================================
# 7 + 8. The stream route: same session semantics as /predict, frames appended, same errors
# ======================================================================================


class _DummyTask(BaseTask):
    """Predicts from the file name; the route, not the model, is under test."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__()

    def load(self, path):  # type: ignore[override]
        return Path(path)

    def featurise(self, raw):  # type: ignore[override]
        return raw

    def predict(self, feats, model=None):  # type: ignore[override]
        p = Path(feats)
        if self.name == "shm":
            rows = [{"file_id": p.name, "prediction": 0.25}]
        elif self.name == "rail":
            rows = [{"file_id": p.name, "prediction": "Side I"}]
        elif self.name == "acv":
            rows = [{"file_id": p.name, "ranked_cars": "|".join(common.acv_car_ids(p))}]
        else:
            rows = [
                {"start_time": "2023-7-5-0-0-0-0", "end_time": "2023-7-5-0-0-4-500", "prediction": "Normal"},
                {"start_time": "2023-7-5-0-0-30-0", "end_time": "2023-7-5-0-0-33-800", "prediction": "Abnormal resistance"},
            ]
        return PredictionResult(
            task=self.name,
            file_id="" if self.name == "door" else p.name,
            rows=rows,
            numbers={"x": 1.0},
            trace=Trace(x=[0, 1], y=[0.0, 1.0]),
            viewport=Viewport(car=1, health="ok", component="door_L1"),
        )


def _dummy_streamer(task_name: str):
    """A streamer that ends on the dummy task's own rows - the contract the route must keep."""

    def _stream(path: Any, model: Any = None, max_frames: int = 200) -> list[Any]:
        stream = _stream_mod()
        task = get_task(task_name)
        result = task.predict(task.featurise(task.load(path)), model)
        rows = task.to_rows(result)
        if task_name == "door":
            steps = [rows[:1], rows]
        else:
            steps = [[], rows]
        n = len(steps)
        return [
            stream.Frame(
                task=task_name,
                file_id=result.file_id,
                step=i,
                n_steps=n,
                t=float(i + 1),
                t_unit=T_UNIT[task_name],
                progress=(i + 1) / n,
                rows=[dict(r) for r in step_rows],
                numbers={"x": 1.0},
                viewport=Viewport(car=1, health="ok", component="door_L1").as_dict(),
                final=(i == n - 1),
            )
            for i, step_rows in enumerate(steps)
        ]

    return _stream


@pytest.fixture
def isolated_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NEBULAX_PS3_UPLOADS", str(tmp_path / "uploads"))
    monkeypatch.setenv("NEBULAX_PS3_RESULTS", str(tmp_path / "results_ps3"))
    monkeypatch.setenv("NEBULAX_PS3_MODELS", str(tmp_path / "models_ps3"))
    api_ps3.reset_sessions()
    yield
    api_ps3.reset_sessions()


@pytest.fixture
def client(isolated_api) -> TestClient:
    app = FastAPI()
    app.include_router(api_ps3.router)
    return TestClient(app)


@pytest.fixture
def dummies(monkeypatch: pytest.MonkeyPatch):
    stream = _stream_mod()
    for name in TASK_NAMES:
        monkeypatch.setitem(common.TASKS, name, _DummyTask(name))
        monkeypatch.setitem(stream.STREAMERS, name, _dummy_streamer(name))


def _upload(path: Path) -> tuple[str, tuple[str, bytes, str]]:
    return ("files", (path.name, path.read_bytes(), "application/octet-stream"))


@pytest.mark.parametrize("task_name,path", [("shm", SHM_CSV), ("door", DOOR_CSV), ("acv", ACV_XLSX)])
def test_stream_route_matches_predict_and_appends_frames(client: TestClient, dummies, task_name: str, path: Path) -> None:
    body = client.post(f"/api/ps3/{task_name}/stream", files=[_upload(path)]).json()
    assert set(body) >= {"session", "task", "rows", "explanations", "csv_url", "n_done", "files", "errors", "frames"}
    assert body["task"] == task_name and body["n_done"] == 1 and body["errors"] == []
    frames = body["frames"]
    assert isinstance(frames, list) and frames
    assert [list(f.keys()) for f in frames] == [FRAME_KEYS] * len(frames)
    assert frames[-1]["final"] is True and all(f["final"] is False for f in frames[:-1])
    # the session rows come from the batch path, and the last frame is exactly those rows
    assert frames[-1]["rows"] == body["rows"]
    csv_text = client.get(body["csv_url"]).text
    got = pd.read_csv(pd.io.common.StringIO(csv_text), dtype=str).fillna("").to_dict("records")
    assert got == [{k: str(v) for k, v in r.items()} for r in body["rows"]]
    # the same file through /predict in a fresh session gives byte-identical CSV
    plain = client.post(f"/api/ps3/{task_name}/predict", files=[_upload(path)]).json()
    assert client.get(plain["csv_url"]).content == csv_text.encode()


def test_stream_route_continues_a_predict_session(client: TestClient, dummies, tmp_path: Path) -> None:
    a = tmp_path / "test01.csv"
    b = tmp_path / "test02.csv"
    shutil.copy(SHM_CSV, a)
    shutil.copy(SHM_CSV, b)
    first = client.post("/api/ps3/shm/predict", files=[_upload(a)]).json()
    second = client.post("/api/ps3/shm/stream", files=[_upload(b)], data={"session": first["session"]}).json()
    assert second["session"] == first["session"]
    assert second["files"] == ["test01.csv", "test02.csv"]
    assert [r["file_id"] for r in second["rows"]] == ["test01.csv", "test02.csv"]
    assert second["frames"][-1]["rows"] == [second["rows"][1]]


def test_stream_route_errors_mirror_predict(client: TestClient, dummies, tmp_path: Path) -> None:
    wrong = tmp_path / "notes.txt"
    wrong.write_text("hello", encoding="utf-8")
    a = client.post("/api/ps3/shm/predict", files=[_upload(wrong)])
    b = client.post("/api/ps3/shm/stream", files=[_upload(wrong)])
    assert a.status_code == b.status_code == 400
    assert a.json()["detail"] == b.json()["detail"]
    assert client.post("/api/ps3/nope/stream", files=[_upload(SHM_CSV)]).status_code == 404
    two = client.post(
        "/api/ps3/shm/stream",
        files=[_upload(SHM_CSV), ("files", ("test09.csv", SHM_CSV.read_bytes(), "application/octet-stream"))],
    )
    assert two.status_code == 400
    assert "one file" in two.json()["detail"].lower()
    none = client.post("/api/ps3/shm/stream")
    assert none.status_code == 400


def test_tasks_route_unchanged_by_stream_route(client: TestClient, dummies) -> None:
    """Adding the route must not change /tasks: it still lists the four tasks with the same keys."""
    body = client.get("/api/ps3/tasks").json()
    assert [t["name"] for t in body] == list(TASK_NAMES)
    assert {"name", "label", "accepts", "output_filename", "available", "cv"} <= set(body[0])


def test_local_door_example_is_prediction_only_and_has_no_session(
    client: TestClient, dummies, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "door"
    root.mkdir()
    shutil.copy(DOOR_CSV, root / "Test.csv")
    monkeypatch.setattr(common, "test_dir", lambda name: root)
    body = client.get("/api/ps3/door/example/stream").json()
    assert body["task"] == "door" and body["source"] == "local organiser Test data"
    assert body["files"] == ["Test.csv"] and "session" not in body
    assert body["example"]["prediction_only"] is True
    assert body["example"]["frame_index"] == 1
    assert body["frames"][1]["rows"][-1]["prediction"] == "Abnormal resistance"
    assert not api_ps3._SESSIONS


def test_local_shm_example_selects_highest_prediction(
    client: TestClient, dummies, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "shm"
    root.mkdir()
    for name in ("test01.csv", "test02.csv"):
        (root / name).write_text("dummy\n", encoding="utf-8")
    monkeypatch.setattr(common, "test_dir", lambda name: root)
    task = common.TASKS["shm"]
    original = task.predict
    def predict(feats, model=None):
        result = original(feats, model)
        result.rows[0]["prediction"] = 0.8 if Path(feats).name == "test02.csv" else 0.2
        return result
    monkeypatch.setattr(task, "predict", predict)
    body = client.get("/api/ps3/shm/example/stream").json()
    assert body["example"]["file"] == "test02.csv"
    assert body["rows"][0]["prediction"] == 0.8


def test_stream_failure_rolls_back_existing_session(
    client: TestClient, dummies, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = _stream_mod()
    first = client.post("/api/ps3/shm/predict", files=[_upload(SHM_CSV)]).json()
    token = first["session"]
    second = tmp_path / "test02.csv"
    shutil.copy(SHM_CSV, second)
    def fail(*args, **kwargs):
        raise RuntimeError("stream failed")
    monkeypatch.setitem(stream.STREAMERS, "shm", fail)
    with pytest.raises(RuntimeError, match="stream failed"):
        client.post("/api/ps3/shm/stream", files=[_upload(second)], data={"session": token})
    session = api_ps3._SESSIONS[token]
    assert session.files == first["files"] and session.rows == first["rows"]
    assert session.n_batches == 1
    assert not (session.dir / "batch002").exists()


# ======================================================================================
# 9. Web: one source per shared mesh group (pure helper, run under node)
# ======================================================================================

_NODE_SCRIPT = r"""
import { componentKey, indexHealthMap, ps3HealthMap, mergeHealthMaps, railBoxComponent } from "%(module)s";

const twin = new Map();
// a replay frame index: car 2 bearing axle boxes, car 1 doors, car 3 apu, all "warn"
for (let a = 1; a <= 4; a++) for (const s of ["L", "R"]) twin.set(componentKey(2, "bearing", `axlebox_${a}${s}`), { health: "warn", label: "twin bearing" });
for (let k = 1; k <= 4; k++) twin.set(componentKey(1, "door", `door_L${k}`), { health: "warn", label: "twin door" });
twin.set(componentKey(0, "pneumatic", "apu_1"), { health: "warn", label: "twin apu" });

const door = ps3HealthMap({ task: "door", rows: [{ start_time: "a", end_time: "b", prediction: "Abnormal resistance" }] });
const rail = ps3HealthMap({
  task: "rail",
  rows: [{ file_id: "Test1.csv", prediction: "Side I" }],
  explanations: [{ file_id: "Test1.csv", viewport: { car: 2, side: "I", health: "crit", component: "axlebox_c2_p1" }, top_boxes: [{ car: 2, position: 1 }, { car: 2, position: 3 }, { car: 5, position: 7 }] }],
});
const acv = ps3HealthMap({ task: "acv", rows: [{ file_id: "x.xlsx", ranked_cars: "03|01|02|04|05|06|07|08" }] });
const shm = ps3HealthMap({ task: "shm", rows: [{ file_id: "test01.csv", prediction: "0.62" }] });

const boxKeyTwin = componentKey(2, "bearing", "axlebox_1L");
const b = railBoxComponent(2, 1);
const boxKeyRail = componentKey(b.car, b.subsystem, b.component_id);
const doorKey = componentKey(1, "door", "door_L1");
const apuKey = componentKey(0, "pneumatic", "apu_1");
const acKey = componentKey(3, "acv", "ac_1");
const railKey = componentKey(0, "rail", "rail_I");
const bogieKey = componentKey(1, "shm", "bogie_1");

function check() {
  const m = mergeHealthMaps({ twin, door, acv, rail, shm });
  if (!(m instanceof Map)) throw new Error("mergeHealthMaps must return a Map");
  const boxes = [...m.entries()].filter(([k]) => /axlebox/.test(k));
  const labels = new Set(boxes.map(([, v]) => v.label));
  if (!boxes.length) throw new Error("no axle boxes painted");
  for (const [, v] of boxes) if (v.label !== "twin bearing") throw new Error(`axlebox painted by the wrong source: ${v.label}`);
  const doors = [...m.entries()].filter(([k]) => /\|door\|/.test(k));
  if (!doors.length) throw new Error("no doors painted");
  for (const [, v] of doors) if (!String(v.label).startsWith("1 abnormal")) throw new Error(`door painted by the wrong source: ${v.label}`);
  for (const k of [apuKey, acKey, railKey, bogieKey]) if (!m.has(k)) throw new Error(`merge dropped ${k}`);
  if (m.get(acKey).health !== "crit") throw new Error("acv rank 1 must stay crit");
  if (m.get(apuKey).label !== "twin apu") throw new Error("twin apu must survive");
}
check();
const m2 = mergeHealthMaps({ twin: null, door, acv, rail, shm });
if ([...m2.keys()].some((k) => /\|bearing\|/.test(k))) throw new Error("rail must not tint axle boxes");
const m3 = mergeHealthMaps({ twin, door: null, acv: null, rail: null, shm: null });
if (m3.get(boxKeyTwin)?.label !== "twin bearing") throw new Error("twin bearing must tint axle boxes");
if ([...m3.keys()].some((k) => /\|door\|/.test(k))) throw new Error("simulated doors must not enter the twin");
console.log("ok");
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_merge_health_maps_paints_each_shared_mesh_group_from_one_source(tmp_path: Path) -> None:
    module = (REPO / "web" / "src" / "components" / "viewport" / "componentMap.js").resolve()
    script = tmp_path / "check.mjs"
    script.write_text(_NODE_SCRIPT % {"module": module.as_posix()}, encoding="utf-8")
    proc = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert proc.stdout.strip().endswith("ok")


# ======================================================================================
# 10. The scored pipeline is frozen: artefacts and submitted CSVs keep their hashes
# ======================================================================================


def test_frozen_artefacts_and_submission_unchanged() -> None:
    _stream_mod()  # importing the streamers must not touch anything below
    for rel, want in FROZEN_MD5.items():
        got = hashlib.md5((REPO / rel).read_bytes()).hexdigest()
        assert got == want, f"{rel} changed ({got}); W5 must not re-train or re-pack"


def test_streamers_are_registered_for_all_four_tasks() -> None:
    stream = _stream_mod()
    assert set(stream.STREAMERS) == set(TASK_NAMES)
    assert callable(stream.register_streamer)
    with pytest.raises(KeyError):
        stream.stream_file("brakes", SHM_CSV)
