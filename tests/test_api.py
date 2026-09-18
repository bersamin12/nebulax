"""Tests for the demo API (``docs/app_contract.md`` sections 4 and 5).

Everything runs against a **fake** ``data/`` built in ``tmp_path``: one door partition written
through ``nebulax.schema.write_dataset`` with three stored cycles, a scores frame carrying one
three-row alarm episode plus a warn/stale/ok component each, the episode and manifest files,
and a two-row ``results/leaderboard.md``. No test touches the real ``data/``.

The two modules the API only imports lazily - ``nebulax.demo.scoring`` and ``nebulax.advisory``,
both written in parallel to the same contract - are faked in ``sys.modules`` where they are
needed, so this file passes whether or not they exist yet.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nebulax import schema as S
from nebulax.api.main import create_app, selection_rows
from nebulax.api.series import downsample_minmax
from nebulax.api.state import FleetState, Settings

T0 = pd.Timestamp("2026-09-01T00:00:00Z")
THRESHOLD = 2.0

#: Cycle geometry of the fake door run: 50 samples at 20 ms opening, a 30 s dwell, 50 closing.
CYCLE_PERIOD_S = 200.0
DWELL_S = 30.0
N_SAMPLES = 50


# --------------------------------------------------------------------------------------
# The fake dataset
# --------------------------------------------------------------------------------------


def _door_long(n_cycles: int = 3) -> pd.DataFrame:
    frames = []
    for k in range(n_cycles):
        base = T0 + pd.Timedelta(seconds=k * CYCLE_PERIOD_S)
        for leg, offset in (("open", 0.0), ("close", DWELL_S + N_SAMPLES * 0.02)):
            ts = pd.date_range(
                base + pd.Timedelta(seconds=offset), periods=N_SAMPLES, freq="20ms", tz="UTC"
            ).as_unit("ms")
            ramp = np.linspace(0.0, 0.725, N_SAMPLES)
            pos = ramp if leg == "open" else ramp[::-1]
            for sig, val in (
                ("pos", pos),
                ("pos_ref", pos * 1.01),
                ("current", 2.0 + 0.5 * np.sin(np.linspace(0, np.pi, N_SAMPLES)) + 0.3 * k),
                ("pwm", np.linspace(-0.5, 0.5, N_SAMPLES)),
                ("vel", np.gradient(pos, 0.02)),
            ):
                frames.append(
                    pd.DataFrame(
                        {
                            "timestamp": ts,
                            "source": "sim",
                            "run_id": "door_0000",
                            "train_id": "T01",
                            "car": np.int8(3),
                            "subsystem": "door",
                            "component_id": "door_L1",
                            "signal": sig,
                            "value": val.astype(np.float32),
                        }
                    )
                )
    ctx = pd.date_range(T0, periods=n_cycles * 4, freq="60s", tz="UTC").as_unit("ms")
    for sig, val in (("speed", 11.0), ("T_amb", 29.5), ("in_service", 1.0), ("load_frac", 0.4)):
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": ctx,
                    "source": "sim",
                    "run_id": "door_0000",
                    "train_id": "T01",
                    "car": np.int8(0),
                    "subsystem": "train",
                    "component_id": "train",
                    "signal": sig,
                    "value": np.full(len(ctx), val, dtype=np.float32),
                }
            )
        )
    return S.coerce_long(pd.concat(frames, ignore_index=True))


def _door_features(n_cycles: int = 3) -> pd.DataFrame:
    starts = [T0 + pd.Timedelta(seconds=k * CYCLE_PERIOD_S) for k in range(n_cycles)]
    return S.coerce_features(
        pd.DataFrame(
            {
                "run_id": "door_0000",
                "source": "sim",
                "train_id": "T01",
                "car": np.int8(3),
                "subsystem": "door",
                "component_id": "door_L1",
                "cycle_id": np.arange(n_cycles, dtype=np.int64),
                "t_start": pd.DatetimeIndex(starts).as_unit("ms"),
                "t_end": pd.DatetimeIndex(
                    [s + pd.Timedelta(seconds=DWELL_S + 2 * N_SAMPLES * 0.02) for s in starts]
                ).as_unit("ms"),
                "closing_time": np.linspace(2.6, 3.4, n_cycles),
                "i_peak": np.linspace(5.0, 7.5, n_cycles),
                "is_faulty": [False] * n_cycles,
            }
        )
    )


def _run_fault_log() -> pd.DataFrame:
    return S.coerce_fault_log(
        pd.DataFrame(
            {
                "run_id": ["door_0000"],
                "train_id": ["T01"],
                "car": np.array([3], dtype=np.int8),
                "subsystem": ["door"],
                "component_id": ["door_L1"],
                "fault_type": ["friction"],
                "t_onset": [T0 + pd.Timedelta(hours=4)],
                "t_failure": [T0 + pd.Timedelta(hours=8)],
                "t_functional_failure": [pd.NaT],
                "gamma": np.array([2.0], dtype=np.float32),
                "shape": ["power"],
                "params_json": ["{}"],
            }
        )
    )


def _run_events() -> pd.DataFrame:
    return S.coerce_events(
        pd.DataFrame(
            {
                "run_id": ["door_0000", "door_0000"],
                "timestamp": [T0 + pd.Timedelta(hours=5, minutes=10), T0 + pd.Timedelta(hours=6)],
                "train_id": ["T01", "T01"],
                "car": np.array([3, 3], dtype=np.int8),
                "subsystem": ["door", "door"],
                "component_id": ["door_L1", "door_L1"],
                "event": ["obstruction", "reversal"],
                "detail_json": ['{"force_N": 150}', "{}"],
            }
        )
    )


def _scores() -> pd.DataFrame:
    """Four components, one per health state the contract defines.

    ``T01/door_L1`` runs a three-row alarm episode at 05:00-07:00; ``T01/apu_1`` sits between
    ``0.8 x threshold`` and the threshold (warn); ``T01/axlebox_1L`` stays low (ok); and
    ``T02/door_R1`` stops being scored after 00:00 (stale from 06:00 on).
    """
    rows = []

    def add(train_id, car, subsystem, component_id, model, times, scores, alerts, top=None):
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": pd.DatetimeIndex(times).as_unit("ms"),
                    "train_id": train_id,
                    "car": np.int8(car),
                    "subsystem": subsystem,
                    "component_id": component_id,
                    "model": model,
                    "score": np.asarray(scores, dtype=np.float32),
                    "threshold": np.float32(THRESHOLD),
                    "alert": np.asarray(alerts, dtype=bool),
                    "top_signals_json": top or ["[]"] * len(scores),
                }
            )
        )

    hours = [T0 + pd.Timedelta(hours=h) for h in range(12)]
    door_scores = [0.5] * 5 + [3.0, 3.5, 3.2] + [0.4] * 4
    door_alerts = [False] * 5 + [True] * 3 + [False] * 4
    top = [json.dumps([{"signal": "closing_time", "z": 5.1, "value": 3.9}])] * 12
    add("T01", 3, "door", "door_L1", "cusum_cycle_scalar", hours, door_scores, door_alerts, top)
    add("T01", 0, "pneumatic", "apu_1", "sparse_autoencoder", hours, [1.8] * 12, [False] * 12)
    add("T01", 3, "bearing", "axlebox_1L", "cusum_cycle_scalar", hours, [0.1] * 12, [False] * 12)
    add("T02", 3, "door", "door_R1", "cusum_cycle_scalar", hours[:1], [0.2], [False])
    return S.coerce_scores(pd.concat(rows, ignore_index=True))


def _episodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "train_id": ["T01", "T01"],
            "car": np.array([3, 0], dtype=np.int8),
            "subsystem": ["door", "pneumatic"],
            "component_id": ["door_L1", "apu_1"],
            "model": ["cusum_cycle_scalar", "sparse_autoencoder"],
            "t_start": pd.DatetimeIndex(
                [T0 + pd.Timedelta(hours=5), T0 + pd.Timedelta(hours=9)]
            ).as_unit("ms"),
            "t_end": pd.DatetimeIndex(
                [T0 + pd.Timedelta(hours=7), T0 + pd.Timedelta(hours=10)]
            ).as_unit("ms"),
            "peak_score": np.array([3.5, 2.1], dtype=np.float32),
            "n_rows": np.array([3, 3], dtype=np.int64),
            "episode_id": ["T01-door_L1-1", "T01-apu_1-1"],
            "fault_type": ["friction", None],
            "t_onset": pd.DatetimeIndex([T0 + pd.Timedelta(hours=4), pd.NaT]).as_unit("ms"),
            "t_failure": pd.DatetimeIndex([T0 + pd.Timedelta(hours=8), pd.NaT]).as_unit("ms"),
            "lead_to_failure_h": np.array([3.0, np.nan], dtype=np.float32),
            "matched": np.array([True, False]),
        }
    )


LEADERBOARD = """# Benchmark

## Task: `ad`

| dataset | model |
|---|---|
| sim | cusum_cycle_scalar |

## Selected per subsystem

| subsystem | dataset | model | split | input_kind | window | selected on | val lift | VUS-PR (test) | test lift | episodes (n) | recall@budget | precision | FA/scored-day | fit (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| door | sim | **cusum_cycle_scalar** | sim_loo_unit | cycle_features | - | val_lift | 4.24x | 0.722 | 2.68x | 31 | 0.333 (4/12) | 0.571 (4/7) | 0.016 | 0.1 |
| | | _chance floor (`random_score`): VUS-PR 0.710_ | | | | | | | | | | | | |
| bearing | sim | **cusum_cycle_scalar** | sim_loo_unit | cycle_features | - | val_lift | 50.82x | 0.658 | 39.94x | 12 | 0.833 (10/12) | 0.833 (10/12) | 0.008 | 0.8 |

## Notes
"""


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A miniature ``data/`` + ``results/`` tree, complete enough for every route."""
    data = tmp_path / "data"
    sim = data / "sim"
    S.write_dataset(
        _door_long(),
        _door_features(),
        _run_fault_log(),
        _run_events(),
        sim,
        source="sim",
        run_id="door_0000",
        meta={"store_every": 10},
    )
    runs = [
        {
            "run_id": "door_0000",
            "subsystem": "door",
            "train_id": "T01",
            "car": 3,
            "component_id": "door_L1",
            "healthy": False,
            "fault_types": ["friction"],
            "path": "source=sim/run_id=door_0000",
        },
        {
            "run_id": "pneumatic_0000",
            "subsystem": "pneumatic",
            "train_id": "T01",
            "car": 0,
            "component_id": "apu_1",
            "healthy": True,
            "fault_types": [],
            "path": "source=sim/run_id=pneumatic_0000",
        },
        {
            "run_id": "bearing_0000",
            "subsystem": "bearing",
            "train_id": "T01",
            "car": 3,
            "component_id": "axlebox_1L",
            "healthy": True,
            "fault_types": [],
            "path": "source=sim/run_id=bearing_0000",
        },
        {
            "run_id": "door_0001",
            "subsystem": "door",
            "train_id": "T02",
            "car": 3,
            "component_id": "door_R1",
            "healthy": True,
            "fault_types": [],
            "path": "source=sim/run_id=door_0001",
        },
    ]
    (sim / "index.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    _run_fault_log().to_parquet(sim / "fault_log.parquet", engine="pyarrow", index=False)

    scores_dir = data / "scores"
    scores_dir.mkdir(parents=True)
    _scores().to_parquet(scores_dir / "scores.parquet", engine="pyarrow", index=False)
    _episodes().to_parquet(scores_dir / "episodes.parquet", engine="pyarrow", index=False)
    (scores_dir / "manifest.json").write_text(
        json.dumps({"door": {"model": "cusum_cycle_scalar", "config_hash": "deadbeef"}}),
        encoding="utf-8",
    )
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "leaderboard.md").write_text(LEADERBOARD, encoding="utf-8")
    return tmp_path


@pytest.fixture
def settings(data_root: Path) -> Settings:
    return Settings(
        data_dir=data_root / "data",
        results_dir=data_root / "results",
        web_dist=data_root / "no-such-dist",
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


def _ts(hours: float) -> str:
    return (T0 + pd.Timedelta(hours=hours)).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------------------
# health / trains
# --------------------------------------------------------------------------------------


def test_health_reports_the_loaded_clock(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["scores_loaded"] is True
    assert body["n_trains"] == 2
    assert body["clock"]["start"] == "2026-09-01T00:00:00Z"
    assert body["clock"]["end"] == _ts(11)


def test_app_boots_without_scores(tmp_path: Path) -> None:
    """The contract requires the app to boot on a laptop with no ``data/scores`` yet."""
    client = TestClient(create_app(Settings(data_dir=tmp_path / "data", results_dir=tmp_path / "r")))
    body = client.get("/api/health").json()
    assert body["scores_loaded"] is False
    frame = client.get("/api/kpis").json()
    assert frame["open_alerts"] == 0
    assert client.get("/api/alerts").json() == []


def test_trains_lists_instrumentation_and_faults(client: TestClient) -> None:
    trains = {t["train_id"]: t for t in client.get("/api/trains").json()}
    assert set(trains) == {"T01", "T02"}
    t01 = trains["T01"]
    assert t01["line"] == "NSL" and t01["cars"] == S.MAX_CAR
    ids = {(c["car"], c["component_id"]) for c in t01["instrumented"]}
    assert (3, "door_L1") in ids and (0, "apu_1") in ids
    # a bearing run instruments all eight boxes of its car
    assert len([c for c in t01["instrumented"] if c["subsystem"] == "bearing"]) == 8
    assert t01["faults"][0]["fault_type"] == "friction"
    assert t01["faults"][0]["t_onset"] == _ts(4)


# --------------------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------------------


def _component(frame: dict[str, Any], component_id: str) -> dict[str, Any]:
    comps = [c for t in frame["trains"] for c in t["components"] if c["component_id"] == component_id]
    assert comps, f"{component_id} missing from the frame"
    return comps[0]


def test_frame_marks_an_open_episode_crit(client: TestClient) -> None:
    frame = client.get("/api/train/T01/state", params={"ts": _ts(6.5)}).json()
    door = _component(frame, "door_L1")
    assert door["health"] == "crit"
    assert door["alert"] is True
    assert door["score"] == pytest.approx(3.5)
    assert door["top_signals"][0]["signal"] == "closing_time"
    assert frame["ts"] == _ts(6.5)


def test_frame_health_warn_and_ok(client: TestClient) -> None:
    frame = client.get("/api/train/T01/state", params={"ts": _ts(4.5)}).json()
    assert _component(frame, "door_L1")["health"] == "ok"  # 0.5 of a threshold of 2.0
    assert _component(frame, "apu_1")["health"] == "warn"  # 1.8 > 0.8 x 2.0
    assert _component(frame, "axlebox_1L")["health"] == "ok"


def test_frame_marks_an_old_row_stale(client: TestClient) -> None:
    """A door is stale six hours after its last scored cycle (not two: doors sit idle up to 5.1 h
    every night), and falls back to ``ok``."""
    frame = client.get("/api/train/T02/state", params={"ts": _ts(6.5)}).json()
    door = _component(frame, "door_R1")
    assert door["stale"] is True
    assert door["health"] == "ok"
    for h in (1, 5):
        fresh = client.get("/api/train/T02/state", params={"ts": _ts(h)}).json()
        assert _component(fresh, "door_R1")["stale"] is False, h


def test_frame_omits_components_that_are_not_scored(client: TestClient) -> None:
    frame = client.get("/api/train/T01/state", params={"ts": _ts(6.5)}).json()
    ids = {c["component_id"] for t in frame["trains"] for c in t["components"]}
    assert ids == {"door_L1", "apu_1", "axlebox_1L"}
    assert "door_L2" not in ids and "axlebox_4R" not in ids


def test_frame_before_the_first_scored_row_is_empty(client: TestClient) -> None:
    frame = client.get("/api/train/T01/state", params={"ts": "2026-08-31T00:00:00Z"}).json()
    assert frame["trains"][0]["components"] == []
    assert frame["alerts"] == []


def test_state_404s_for_an_unknown_train(client: TestClient) -> None:
    assert client.get("/api/train/T99/state").status_code == 404


def test_state_400s_on_an_unparseable_ts(client: TestClient) -> None:
    assert client.get("/api/train/T01/state", params={"ts": "yesterday"}).status_code == 400


def test_frame_truncates_an_open_episode_at_ts(client: TestClient) -> None:
    """Mid-episode the frame must not leak the episode's future peak or its end."""
    frame = client.get("/api/train/T01/state", params={"ts": _ts(5.5)}).json()
    alert = frame["alerts"][0]
    assert alert["episode_id"] == "T01-door_L1-1"
    assert alert["t_end"] is None and alert["open"] is True
    assert alert["peak_score"] == pytest.approx(3.0)  # 3.5 only arrives at 06:00
    later = client.get("/api/train/T01/state", params={"ts": _ts(8)}).json()
    assert later["alerts"][0]["t_end"] == _ts(7)
    assert later["alerts"][0]["peak_score"] == pytest.approx(3.5)


def test_frame_ticker_holds_the_latest_transitions(client: TestClient) -> None:
    frame = client.get("/api/train/T01/state", params={"ts": _ts(8)}).json()
    assert len(frame["ticker"]) == 2
    assert "episode closed" in frame["ticker"][0]
    assert "episode open" in frame["ticker"][1]
    assert client.get("/api/train/T01/state", params={"ts": _ts(1)}).json()["ticker"] == []


# --------------------------------------------------------------------------------------
# alerts / kpis
# --------------------------------------------------------------------------------------


def test_alerts_are_newest_first_and_filterable(client: TestClient) -> None:
    alerts = client.get("/api/alerts", params={"ts": _ts(11)}).json()
    assert [a["episode_id"] for a in alerts] == ["T01-apu_1-1", "T01-door_L1-1"]
    assert all(a["advisory"] is None for a in alerts)
    assert client.get("/api/alerts", params={"ts": _ts(6)}).json()[0]["episode_id"] == "T01-door_L1-1"
    assert client.get("/api/alerts", params={"ts": _ts(11), "train": "T02"}).json() == []


def test_kpis_count_open_episodes_and_events(client: TestClient) -> None:
    mid = client.get("/api/kpis", params={"ts": _ts(6)}).json()
    assert mid["open_alerts"] == 1
    assert mid["events_total"] == 1
    assert mid["events_detected"] == 1
    # the door episode is still running at 06:00, so its lead is not known yet
    assert mid["median_lead_h"] is None
    end = client.get("/api/kpis", params={"ts": _ts(11)}).json()
    assert end["open_alerts"] == 0
    assert end["median_lead_h"] == pytest.approx(3.0)  # closed and matched by then
    assert end["fa_per_train_day"] > 0  # the unmatched pneumatic episode
    assert end["n_trains"] == 2


# --------------------------------------------------------------------------------------
# series / scores / cycle
# --------------------------------------------------------------------------------------


def test_series_downsamples_within_its_budget(client: TestClient) -> None:
    body = client.get(
        "/api/train/T01/component/door_L1/series",
        params={"signal": "current", "max_points": 20},
    ).json()
    assert body["unit"] == "A"
    assert body["n_raw"] == 3 * 2 * N_SAMPLES
    assert 0 < len(body["points"]) <= 20
    stamps = [pd.Timestamp(p[0]) for p in body["points"]]
    assert stamps == sorted(stamps)


def test_series_respects_the_time_filter(client: TestClient) -> None:
    body = client.get(
        "/api/train/T01/component/door_L1/series",
        params={"signal": "pos", "from": _ts(0), "to": T0 + pd.Timedelta(seconds=5)},
    ).json()
    assert body["n_raw"] == N_SAMPLES  # the first opening leg only
    assert all(pd.Timestamp(p[0]) <= T0 + pd.Timedelta(seconds=5) for p in body["points"])


def test_series_returns_every_point_when_it_fits(client: TestClient) -> None:
    body = client.get(
        "/api/train/T01/component/door_L1/series",
        params={"signal": "pos", "max_points": 1000},
    ).json()
    assert len(body["points"]) == body["n_raw"] == 300


def test_series_404s_without_a_partition_and_400s_on_a_bad_signal(client: TestClient) -> None:
    # door_R1 is in index.json but its partition was never written
    assert (
        client.get("/api/train/T02/component/door_R1/series", params={"signal": "pos"}).status_code == 404
    )
    assert (
        client.get("/api/train/T01/component/door_L1/series", params={"signal": "TP2"}).status_code == 400
    )


def test_series_pins_to_a_car_and_serves_train_context(client: TestClient) -> None:
    pinned = client.get(
        "/api/train/T01/component/door_L1/series", params={"signal": "pos", "car": 3}
    ).json()
    assert pinned["n_raw"] == 300
    assert (
        client.get(
            "/api/train/T01/component/door_L1/series", params={"signal": "pos", "car": 5}
        ).status_code
        == 404
    )
    context = client.get(
        "/api/train/T01/component/train/series", params={"signal": "speed"}
    ).json()
    assert context["unit"] == "m/s"
    assert context["n_raw"] == 12


def test_downsample_minmax_keeps_the_extremes() -> None:
    ts = np.arange(1000, dtype=np.int64) * 1_000_000_000
    vals = np.zeros(1000)
    vals[137] = 9.0
    vals[600] = -9.0
    points = downsample_minmax(ts, vals, 40)
    assert len(points) <= 40
    values = [p[1] for p in points]
    assert max(values) == 9.0 and min(values) == -9.0


def test_scores_route_serves_points_and_top_signals(client: TestClient) -> None:
    body = client.get("/api/train/T01/component/door_L1/scores").json()
    assert body["model"] == "cusum_cycle_scalar"
    assert body["threshold"] == pytest.approx(THRESHOLD)
    assert len(body["points"]) == 12
    assert body["points"][5][2] is True
    assert body["top_signals"][0]["signal"] == "closing_time"
    window = client.get(
        "/api/train/T01/component/door_L1/scores", params={"from": _ts(5), "to": _ts(7)}
    ).json()
    assert len(window["points"]) == 3
    assert client.get("/api/train/T01/component/door_L9/scores").status_code == 404


def test_cycle_returns_the_nearest_stored_waveform(client: TestClient) -> None:
    body = client.get(
        "/api/train/T01/component/door_L1/cycle",
        params={"ts": (T0 + pd.Timedelta(seconds=CYCLE_PERIOD_S)).isoformat()},
    ).json()
    assert body["n"] == 2 * N_SAMPLES  # opening + closing of one cycle, never two cycles
    assert body["t"][0] == 0.0
    assert body["t"][-1] == pytest.approx(DWELL_S + 2 * N_SAMPLES * 0.02 - 0.02, abs=0.05)
    assert len(body["pos"]) == len(body["current"]) == len(body["pwm"]) == body["n"]
    assert body["t_start"].startswith("2026-09-01T00:03:20")


def test_cycle_widens_its_search_and_keeps_whole_cycles(client: TestClient) -> None:
    """A ts before the first cycle (or after the last) still returns a complete waveform."""
    first = client.get(
        "/api/train/T01/component/door_L1/cycle", params={"ts": "2026-08-30T00:00:00Z"}
    ).json()
    assert first["n"] == 2 * N_SAMPLES
    assert first["t_start"] == "2026-09-01T00:00:00Z"
    last = client.get(
        "/api/train/T01/component/door_L1/cycle", params={"ts": "2026-09-02T00:00:00Z"}
    ).json()
    assert last["n"] == 2 * N_SAMPLES
    assert last["t_start"].startswith("2026-09-01T00:06:40")


def test_cycle_404s_for_a_non_door(client: TestClient) -> None:
    assert client.get("/api/train/T01/component/apu_1/cycle").status_code == 404


# --------------------------------------------------------------------------------------
# bench selection
# --------------------------------------------------------------------------------------


def test_bench_selection_parses_the_leaderboard(client: TestClient) -> None:
    body = client.get("/api/bench/selection").json()
    assert body["manifest"]["door"]["config_hash"] == "deadbeef"
    rows = body["selection"]
    assert [r["subsystem"] for r in rows] == ["door", "bearing"]  # the note row is dropped
    assert rows[0]["model"] == "cusum_cycle_scalar"  # bold stripped
    assert rows[0]["val_lift"] == "4.24x"
    assert rows[0]["recall"] == "0.333 (4/12)"
    assert rows[0]["precision"] == "0.571 (4/7)"
    assert rows[0]["fa_per_scored_day"] == "0.016"


def test_selection_rows_tolerates_a_missing_leaderboard(tmp_path: Path) -> None:
    assert selection_rows(tmp_path) == []


def test_selection_rows_match_the_real_leaderboard() -> None:
    """The retained bearing and pneumatic picks in ``results/leaderboard.md``."""
    rows = selection_rows(Path(__file__).resolve().parents[1] / "results")
    if not rows:
        pytest.skip("results/leaderboard.md has not been generated in this checkout")
    assert len(rows) == 4
    assert {(r["subsystem"], r["dataset"]) for r in rows} == {
        ("pneumatic", "sim"),
        ("pneumatic", "metropt3"),
        ("bearing", "sim"),
        ("bearing", "ottawa"),
    }


# --------------------------------------------------------------------------------------
# replay websocket
# --------------------------------------------------------------------------------------


def test_replay_play_pause_seek(client: TestClient) -> None:
    with client.websocket_connect("/api/replay") as ws:
        first = ws.receive_json()
        assert first["playing"] is False
        assert first["ts"] == "2026-09-01T00:00:00Z"

        ws.send_json({"cmd": "play", "speed": 60.0})
        frames = [ws.receive_json() for _ in range(3)]
        stamps = [f["ts"] for f in frames]
        assert stamps == sorted(stamps)
        assert stamps[-1] > stamps[0]
        assert all(f["playing"] for f in frames)
        assert all(f["speed"] == 60.0 for f in frames)

        ws.send_json({"cmd": "pause"})
        paused = _drain_until(ws, lambda f: not f["playing"])
        assert paused["playing"] is False

        ws.send_json({"cmd": "seek", "ts": _ts(6.5)})
        sought = _drain_until(ws, lambda f: f.get("ts") == _ts(6.5))
        assert sought["ts"] == _ts(6.5)
        door = _component(sought, "door_L1")
        assert door["health"] == "crit"


def test_replay_rejects_a_bad_command(client: TestClient) -> None:
    with client.websocket_connect("/api/replay") as ws:
        ws.receive_json()
        ws.send_json({"cmd": "teleport"})
        assert "error" in ws.receive_json()
        ws.send_json({"cmd": "seek"})
        assert "error" in ws.receive_json()


def _drain_until(ws, predicate, limit: int = 40) -> dict[str, Any]:
    """Read frames until one satisfies ``predicate`` - frames already in flight when a command
    was sent are not an error, they are the 10 Hz push."""
    for _ in range(limit):
        frame = ws.receive_json()
        if predicate(frame):
            return frame
    raise AssertionError("no frame satisfied the predicate within the limit")


# --------------------------------------------------------------------------------------
# inject
# --------------------------------------------------------------------------------------


def _fake_scoring(monkeypatch: pytest.MonkeyPatch, *, model: str = "sparse_autoencoder") -> types.ModuleType:
    """Stand in for ``nebulax.demo.scoring`` (contract section 3, written in parallel)."""

    class FittedWinner:
        subsystem = "pneumatic"
        model_name = model
        threshold = 1.0
        params: dict[str, Any] = {}

    def load_fitted(subsystem: str, held_out_train: str | None = None) -> FittedWinner:
        fw = FittedWinner()
        fw.subsystem = subsystem
        fw.held_out_train = held_out_train
        return fw

    def score_frames(fw, long, features, *, train_id, car, component_id):
        n = len(features)
        score = np.linspace(0.2, 3.0, n)
        return S.coerce_scores(
            pd.DataFrame(
                {
                    "timestamp": features["t_end"].to_numpy(),
                    "train_id": train_id,
                    "car": np.int8(car),
                    "subsystem": fw.subsystem,
                    "component_id": component_id,
                    "model": fw.model_name,
                    "score": score,
                    "threshold": 1.0,
                    "alert": score > 1.0,
                    "top_signals_json": [json.dumps([{"signal": "TP2", "z": 4.0, "value": 7.1}])] * n,
                }
            )
        )

    def episodes_from_scores(scores, fault_log):
        above = scores[scores["score"] > scores["threshold"]]
        if not len(above):
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "episode_id": ["injected-1"],
                "train_id": [str(above["train_id"].iloc[0])],
                "car": np.array([int(above["car"].iloc[0])], dtype=np.int8),
                "subsystem": [str(above["subsystem"].iloc[0])],
                "component_id": [str(above["component_id"].iloc[0])],
                "model": [str(above["model"].iloc[0])],
                "t_start": [above["timestamp"].iloc[0]],
                "t_end": [pd.NaT],
                "peak_score": [float(above["score"].max())],
                "n_rows": [int(len(above))],
                "fault_type": ["air_leak"],
                "t_onset": [above["timestamp"].iloc[0]],
                "t_failure": [pd.NaT],
                "lead_to_failure_h": [12.0],
                "matched": [True],
            }
        )

    scoring = types.ModuleType("nebulax.demo.scoring")
    scoring.load_fitted = load_fitted
    scoring.score_frames = score_frames
    scoring.episodes_from_scores = episodes_from_scores
    demo = types.ModuleType("nebulax.demo")
    demo.scoring = scoring
    monkeypatch.setitem(sys.modules, "nebulax.demo", demo)
    monkeypatch.setitem(sys.modules, "nebulax.demo.scoring", scoring)
    return scoring


def test_inject_rejects_an_unknown_fault_type(client: TestClient) -> None:
    body = {
        "train_id": "T01",
        "car": 0,
        "component_id": "apu_1",
        "fault_type": "gremlins",
        "severity_ramp_days": 2.0,
    }
    response = client.post("/api/sim/inject", json=body)
    assert response.status_code == 400
    assert "gremlins" in response.json()["detail"]


def test_inject_rejects_a_fault_from_another_subsystem(client: TestClient) -> None:
    body = {
        "train_id": "T01",
        "car": 0,
        "component_id": "apu_1",
        "fault_type": "friction",  # a door fault
        "severity_ramp_days": 2.0,
    }
    assert client.post("/api/sim/inject", json=body).status_code == 400


def test_inject_rejects_an_unknown_component(client: TestClient) -> None:
    body = {
        "train_id": "T01",
        "car": 2,
        "component_id": "pantograph_1",
        "fault_type": "friction",
        "severity_ramp_days": 2.0,
    }
    assert client.post("/api/sim/inject", json=body).status_code == 400


def test_inject_rejects_a_non_positive_ramp(client: TestClient) -> None:
    body = {
        "train_id": "T01",
        "car": 0,
        "component_id": "apu_1",
        "fault_type": "air_leak",
        "severity_ramp_days": 0.0,
    }
    assert client.post("/api/sim/inject", json=body).status_code == 422


def test_inject_overlay_takes_precedence(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """One real pneumatic day is simulated, re-scored by the fake winner, and every GET route
    must then read the overlay rather than ``data/scores``."""
    _fake_scoring(monkeypatch)
    before = client.get("/api/train/T01/component/apu_1/scores").json()
    assert len(before["points"]) == 12

    response = client.post(
        "/api/sim/inject",
        json={
            "train_id": "T01",
            "car": 0,
            "component_id": "apu_1",
            "fault_type": "air_leak",
            "severity_ramp_days": 2.0,
            "seed": 7,
            "ts": _ts(11),
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["window"]["days"] == 1
    assert body["n_rows"] > 0
    assert body["elapsed_s"] < 60.0
    assert body["episodes"][0]["episode_id"] == "inj-T01-apu_1-0"  # the injector mints the id
    assert body["window"]["max_days"] == 30  # pneumatic fits the whole replay clock

    after = client.get("/api/train/T01/component/apu_1/scores").json()
    assert len(after["points"]) == body["n_rows"] != len(before["points"])
    assert after["top_signals"][0]["signal"] == "TP2"

    # the overlay's telemetry answers /series too, without a partition on disk
    series = client.get(
        "/api/train/T01/component/apu_1/series", params={"signal": "TP2", "max_points": 50}
    ).json()
    assert series["n_raw"] > 0 and len(series["points"]) <= 50

    alerts = client.get("/api/alerts", params={"ts": _ts(11)}).json()
    assert "inj-T01-apu_1-0" in {a["episode_id"] for a in alerts}
    assert "T01-apu_1-1" not in {a["episode_id"] for a in alerts}  # shadowed, same component

    assert client.delete("/api/sim/inject").json() == {"cleared": True}
    restored = client.get("/api/train/T01/component/apu_1/scores").json()
    assert len(restored["points"]) == 12
    assert client.get("/api/health").json()["overlay"] is False


def test_inject_503s_without_the_scoring_module(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Until ``nebulax.demo.scoring`` lands the route says so instead of exploding."""
    import nebulax.api.inject as inject_mod

    def boom(*args: Any, **kwargs: Any):
        raise ModuleNotFoundError("No module named 'nebulax.demo'")

    monkeypatch.setattr(inject_mod, "_simulate", boom)
    response = client.post(
        "/api/sim/inject",
        json={
            "train_id": "T01",
            "car": 0,
            "component_id": "apu_1",
            "fault_type": "air_leak",
            "severity_ramp_days": 2.0,
            "ts": _ts(11),
        },
    )
    assert response.status_code == 503


# --------------------------------------------------------------------------------------
# advisory
# --------------------------------------------------------------------------------------


def _fake_advisory(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stand in for ``nebulax.advisory`` (contract section 6) and count the calls."""
    calls: dict[str, Any] = {"n": 0, "contexts": []}

    class AlertContext:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class Advisory:
        def __init__(self, **kwargs: Any) -> None:
            self._data = kwargs

        def model_dump(self, mode: str = "python") -> dict[str, Any]:
            return dict(self._data)

    def advise(ctx, *, client=None, timeout_s: float = 20.0) -> Advisory:
        calls["n"] += 1
        calls["contexts"].append(ctx)
        return Advisory(
            summary="Door L1 closing slower every cycle.",
            evidence=[f"score {ctx.score} vs threshold {ctx.threshold}"],
            likely_component=ctx.component_id,
            likely_fault="friction",
            recommended_action="Inspect the belt tension at the next depot visit.",
            urgency="medium",
            confidence=0.7,
            cbm_steps={"state_detection": "alarm", "advisory": "inspect"},
            source="template",
            model="template-v1",
        )

    module = types.ModuleType("nebulax.advisory")
    module.AlertContext = AlertContext
    module.Advisory = Advisory
    module.advise = advise
    monkeypatch.setitem(sys.modules, "nebulax.advisory", module)
    return calls


def test_advisory_is_built_and_cached(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_advisory(monkeypatch)
    first = client.post("/api/advisory/T01-door_L1-1")
    assert first.status_code == 200
    body = first.json()
    assert body["likely_component"] == "door_L1"
    assert body["source"] == "template"
    assert calls["n"] == 1

    ctx = calls["contexts"][0]
    assert ctx.episode_id == "T01-door_L1-1"
    assert ctx.subsystem == "door" and ctx.car == 3
    assert ctx.duration_h == pytest.approx(2.0)
    assert ctx.top_signals[0]["signal"] == "closing_time"
    assert any("obstruction" in e for e in ctx.recent_events)
    assert "friction" in ctx.fault_candidates
    assert "median" in ctx.fleet_peer_summary

    again = client.post("/api/advisory/T01-door_L1-1")
    assert again.json() == body
    assert calls["n"] == 1  # served from the cache

    forced = client.post("/api/advisory/T01-door_L1-1", params={"refresh": True})
    assert forced.status_code == 200
    assert calls["n"] == 2


def test_advisory_shows_up_on_the_alert(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_advisory(monkeypatch)
    client.post("/api/advisory/T01-door_L1-1")
    alerts = client.get("/api/alerts", params={"ts": _ts(11)}).json()
    door = [a for a in alerts if a["episode_id"] == "T01-door_L1-1"][0]
    assert door["advisory"]["urgency"] == "medium"


def test_advisory_404s_for_an_unknown_episode(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_advisory(monkeypatch)
    assert client.post("/api/advisory/nope").status_code == 404
    assert client.get("/api/advisory/T01-door_L1-1").status_code == 404  # nothing cached yet


# --------------------------------------------------------------------------------------
# frame_at performance (the WS pushes ten a second)
# --------------------------------------------------------------------------------------


def test_frame_at_is_fast(settings: Settings) -> None:
    import time

    state = FleetState(settings)
    state.frame_at(T0 + pd.Timedelta(hours=6))  # warm the caches
    t0 = time.perf_counter()
    for _ in range(20):
        state.frame_at(T0 + pd.Timedelta(hours=6))
    per_call_ms = (time.perf_counter() - t0) / 20 * 1000
    assert per_call_ms < 25.0, f"frame_at took {per_call_ms:.1f} ms per call"


# --------------------------------------------------------------------------------------
# Integration with the parallel deliverables, when they exist
# --------------------------------------------------------------------------------------


def test_real_advisory_module_accepts_our_context(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``build_alert_context`` must satisfy the real ``AlertContext`` (``extra="forbid"``).

    Skipped while ``nebulax.advisory`` is still being written; with no API key set the module
    takes its template path, so this never touches the network.
    """
    pytest.importorskip("nebulax.advisory")
    from nebulax.advisory import AlertContext, advise

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from nebulax.api.main import build_alert_context

    state = FleetState(settings)
    rec = state.episode("T01-door_L1-1")
    assert rec is not None
    advisory = advise(AlertContext(**build_alert_context(state, rec)))
    assert advisory.source == "template"
    assert advisory.likely_fault in (*S.FAULT_TYPES["door"], "unknown")
    assert set(advisory.cbm_steps) == {
        "state_detection",
        "health_assessment",
        "prognostic_assessment",
        "advisory",
    }


def test_real_scoring_module_exposes_the_contract(settings: Settings) -> None:
    """``nebulax.demo.scoring`` must offer what ``inject`` calls. Skipped until it lands."""
    scoring = pytest.importorskip("nebulax.demo.scoring")
    for name in ("load_fitted", "score_frames", "WINNERS"):
        assert hasattr(scoring, name), f"nebulax.demo.scoring is missing {name}"


# --------------------------------------------------------------------------------------
# Fix round 1 regressions
# --------------------------------------------------------------------------------------
#
# One test per ``must_fix`` / ``should_fix`` of the API fix brief (17 Sep):
#   1. MetroPT-3 never leaks into the sim fleet's frame / kpis / alerts / ticker;
#   2. a bearing injection installs eight overlay series, one per axle box;
#   3. a non-JSON text frame on the replay socket is answered, not fatal;
#   + staleness beating an open episode, the patched inject window, train-context signals on a
#     component, the 404 for an unknown train, ``inj-`` episode ids, open-alert fields, the
#     ticker format and the clock cap.

MP3_T0 = pd.Timestamp("2020-04-17T00:00:00Z")
MP3_END = pd.Timestamp("2020-07-20T00:00:00Z")
#: The first of the four MetroPT-3 failure-report rows (``nebulax.adapters.metropt3``).
MP3_FIRST_ONSET = pd.Timestamp("2020-04-18T00:00:00Z")


def _mp3_scores() -> pd.DataFrame:
    times = [MP3_T0 + pd.Timedelta(hours=h) for h in range(12)] + [MP3_END]
    return S.coerce_scores(
        pd.DataFrame(
            {
                "timestamp": pd.DatetimeIndex(times).as_unit("ms"),
                "train_id": "MP3",
                "car": np.int8(0),
                "subsystem": "pneumatic",
                "component_id": "apu_1",
                "model": "lgbm_residual",
                "score": np.array([0.5] * 5 + [3.0, 3.5, 3.2] + [2.5] * 4 + [0.4], dtype=np.float32),
                "threshold": np.float32(THRESHOLD),
                "alert": np.array([False] * 5 + [True] * 3 + [True] * 4 + [False]),
                "top_signals_json": ["[]"] * len(times),
            }
        )
    )


def _mp3_episodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "train_id": ["MP3", "MP3"],
            "car": np.array([0, 0], dtype=np.int8),
            "subsystem": ["pneumatic", "pneumatic"],
            "component_id": ["apu_1", "apu_1"],
            "model": ["lgbm_residual", "lgbm_residual"],
            "t_start": pd.DatetimeIndex(
                [MP3_T0 + pd.Timedelta(hours=5), MP3_T0 + pd.Timedelta(hours=9)]
            ).as_unit("ms"),
            "t_end": pd.DatetimeIndex(
                [MP3_T0 + pd.Timedelta(hours=7), MP3_T0 + pd.Timedelta(hours=10)]
            ).as_unit("ms"),
            "peak_score": np.array([3.5, 2.5], dtype=np.float32),
            "n_rows": np.array([3, 3], dtype=np.int64),
            "episode_id": ["MP3-apu_1-0", "MP3-apu_1-1"],
            "fault_type": ["air_leak", None],
            "t_onset": pd.DatetimeIndex([MP3_FIRST_ONSET, pd.NaT]).as_unit("ms"),
            "t_failure": pd.DatetimeIndex([MP3_FIRST_ONSET + pd.Timedelta(hours=24), pd.NaT]).as_unit("ms"),
            "lead_to_failure_h": np.array([43.0, np.nan], dtype=np.float32),
            "matched": np.array([True, False]),
        }
    )


@pytest.fixture
def mp3_client(data_root: Path, settings: Settings) -> TestClient:
    """The same fake fleet with the MetroPT-3 unit bolted on: its own 2020 clock, two of its
    own episodes and the adapter's four failure-report rows behind it."""
    scores_dir = data_root / "data" / "scores"
    _mp3_scores().to_parquet(scores_dir / "metropt3.parquet", engine="pyarrow", index=False)
    pd.concat([_episodes(), _mp3_episodes()], ignore_index=True).to_parquet(
        scores_dir / "episodes.parquet", engine="pyarrow", index=False
    )
    return TestClient(create_app(settings))


# -- must_fix 1: MetroPT-3 must not leak into the sim fleet ------------------------------


def test_sim_kpis_ignore_the_metropt3_unit(mp3_client: TestClient) -> None:
    """MP3's 2020 episodes started long before any sim ts, so without the split they would
    count as false alarms on the 2026 fleet from the very first frame."""
    early = mp3_client.get("/api/kpis", params={"ts": "2026-09-01T00:02:09Z"}).json()
    assert early["fa_per_train_day"] == 0.0
    assert early["open_alerts"] == 0
    assert early["events_total"] == 1  # the one sim fault whose detection window has opened
    assert early["n_trains"] == 2  # sim trains only

    end = mp3_client.get("/api/kpis", params={"ts": _ts(11)}).json()
    assert end["events_total"] == 1
    assert end["median_lead_h"] == pytest.approx(3.0)  # the sim door episode, not MP3's 43 h


def test_sim_frame_and_alerts_ignore_the_metropt3_unit(mp3_client: TestClient) -> None:
    frame = mp3_client.get("/api/train/T01/state", params={"ts": _ts(11)}).json()
    assert {a["train_id"] for a in frame["alerts"]} == {"T01"}
    assert all("MP3" not in line for line in frame["ticker"])
    assert frame["kpis"]["n_trains"] == 2
    fleet_alerts = mp3_client.get("/api/alerts", params={"ts": _ts(11)}).json()
    assert "MP3" not in {a["train_id"] for a in fleet_alerts}


def test_metropt3_has_its_own_clock_kpis_and_alerts(mp3_client: TestClient) -> None:
    trains = {t["train_id"]: t for t in mp3_client.get("/api/trains").json()}
    assert trains["MP3"]["clock"]["start"] == "2020-04-17T00:00:00Z"
    assert len(trains["MP3"]["faults"]) == 4  # the failure-report rows

    kpis = mp3_client.get("/api/kpis", params={"train": "MP3"}).json()
    assert kpis["n_trains"] == 1
    assert kpis["events_total"] == 4  # all four MetroPT windows have opened by its clock end
    assert kpis["events_detected"] == 1
    assert kpis["median_lead_h"] == pytest.approx(43.0)
    # one false alarm over its own 94-day clock, not over the sim fleet's 30 days
    assert kpis["elapsed_days"] == pytest.approx((MP3_END - MP3_T0) / pd.Timedelta(days=1))
    assert kpis["fa_per_train_day"] == pytest.approx(1.0 / kpis["elapsed_days"])

    state = mp3_client.get("/api/train/MP3/state").json()
    assert state["ts"] == "2020-07-20T00:00:00Z"  # MP3's clock end, not the sim fleet's
    assert {a["train_id"] for a in state["alerts"]} == {"MP3"}
    assert state["kpis"] == kpis
    assert all("MP3" in line for line in state["ticker"])


def test_kpis_404s_for_an_unknown_train(client: TestClient) -> None:
    assert client.get("/api/kpis", params={"train": "T99"}).status_code == 404


# -- must_fix 2: a bearing injection is eight series -------------------------------------


def _fake_bearing_inject(monkeypatch: pytest.MonkeyPatch, t0: pd.Timestamp, n: int = 4) -> None:
    """Replace the bearing simulator and the winner with cheap stand-ins that keep the shape
    the real ones have: one run, eight axle boxes, ``n`` windows each."""
    import nebulax.api.inject as inject_mod

    times = pd.DatetimeIndex([t0 + pd.Timedelta(minutes=5 * k) for k in range(n)]).as_unit("ms")

    def fake_simulate(subsystem, scenario, service, rng, store_every):
        long = pd.DataFrame(
            {
                "timestamp": list(times) * len(S.AXLEBOX_COMPONENT_IDS),
                "component_id": np.repeat(list(S.AXLEBOX_COMPONENT_IDS), n),
                "car": np.int8(scenario.car),
                "signal": "acc_rms",
                "value": np.linspace(0.1, 1.0, n * len(S.AXLEBOX_COMPONENT_IDS)),
            }
        )
        features = pd.DataFrame({"t_end": list(times) * len(S.AXLEBOX_COMPONENT_IDS)})
        return long, features, pd.DataFrame()

    monkeypatch.setattr(inject_mod, "_simulate", fake_simulate)

    class FittedWinner:
        subsystem = "bearing"
        model_name = "cusum_cycle_scalar"
        threshold = 1.0

    def load_fitted(subsystem: str, held_out_train: str | None = None) -> FittedWinner:
        return FittedWinner()

    def score_frames(fw, long, features, *, train_id, car, component_id):
        """The real one returns **eight** series for a bogie: the per-row keys win."""
        rows = []
        for cid in S.AXLEBOX_COMPONENT_IDS:
            hot = cid == "axlebox_3R"
            rows.append(
                pd.DataFrame(
                    {
                        "timestamp": times,
                        "train_id": train_id,
                        "car": np.int8(car),
                        "subsystem": "bearing",
                        "component_id": cid,
                        "model": fw.model_name,
                        "score": np.linspace(2.0, 9.0, n) if hot else np.linspace(0.1, 0.3, n),
                        "threshold": 1.0,
                        "alert": np.array([hot] * n),
                        "top_signals_json": [
                            json.dumps([{"signal": "T_box" if hot else "acc_rms", "z": 6.0, "value": 91.0}])
                        ]
                        * n,
                    }
                )
            )
        return S.coerce_scores(pd.concat(rows, ignore_index=True))

    def episodes_from_scores(scores, fault_log):
        hot = scores[scores["component_id"].astype(str) == "axlebox_3R"]
        return pd.DataFrame(
            {
                "episode_id": ["whatever-the-scorer-called-it"],
                "train_id": [str(hot["train_id"].iloc[0])],
                "car": np.array([int(hot["car"].iloc[0])], dtype=np.int8),
                "subsystem": ["bearing"],
                "component_id": ["axlebox_3R"],
                "model": ["cusum_cycle_scalar"],
                "t_start": [hot["timestamp"].iloc[0]],
                "t_end": [pd.NaT],
                "peak_score": [float(hot["score"].max())],
                "n_rows": [int(len(hot))],
                "fault_type": ["hot_axle_box"],
                "t_onset": [hot["timestamp"].iloc[0]],
                "t_failure": [pd.NaT],
                "lead_to_failure_h": [8.0],
                "matched": [True],
            }
        )

    scoring = types.ModuleType("nebulax.demo.scoring")
    scoring.load_fitted = load_fitted
    scoring.score_frames = score_frames
    scoring.episodes_from_scores = episodes_from_scores
    demo = types.ModuleType("nebulax.demo")
    demo.scoring = scoring
    monkeypatch.setitem(sys.modules, "nebulax.demo", demo)
    monkeypatch.setitem(sys.modules, "nebulax.demo.scoring", scoring)


def test_bearing_inject_keys_every_axle_box_separately(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bogie's eight boxes must land in eight overlay series, not one pile: before the fix
    ``/scores`` for one box returned all eight scores per timestamp."""
    onset = T0 + pd.Timedelta(hours=10)
    _fake_bearing_inject(monkeypatch, onset)
    response = client.post(
        "/api/sim/inject",
        json={
            "train_id": "T01",
            "car": 3,
            "component_id": "axlebox_3R",
            "fault_type": "hot_axle_box",
            "severity_ramp_days": 1.0,
            "ts": _ts(10),
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["window"]["max_days"] == 17  # the measured cap for a bogie, under the 60 s budget
    assert body["episodes"][0]["episode_id"] == "inj-T01-axlebox_3R-0"

    hot = client.get("/api/train/T01/component/axlebox_3R/scores", params={"car": 3}).json()
    stamps = [p[0] for p in hot["points"]]
    assert len(stamps) == len(set(stamps)) == 4  # one point per timestamp, not eight
    assert hot["top_signals"][0]["signal"] == "T_box"

    cool = client.get("/api/train/T01/component/axlebox_1R/scores", params={"car": 3}).json()
    assert len(cool["points"]) == 4
    assert cool["points"][-1][1] < hot["points"][-1][1]

    frame = client.get("/api/train/T01/state", params={"ts": _ts(10.25)}).json()
    box = _component(frame, "axlebox_3R")
    assert box["health"] == "crit"
    assert box["score"] > 0
    assert box["top_signals"][0]["signal"] == "T_box"
    assert _component(frame, "axlebox_1R")["health"] == "ok"
    assert _component(frame, "axlebox_1R")["top_signals"][0]["signal"] == "acc_rms"
    # all eight boxes of the injected bogie are now scored components of the frame
    boxes = {c["component_id"] for t in frame["trains"] for c in t["components"] if c["subsystem"] == "bearing"}
    assert boxes == set(S.AXLEBOX_COMPONENT_IDS)


# -- must_fix 3: a non-JSON frame is answered, the socket stays open ----------------------


def test_replay_answers_a_non_json_frame_and_stays_open(client: TestClient) -> None:
    with client.websocket_connect("/api/replay") as ws:
        ws.receive_json()
        ws.send_text("this is not json")
        answer = ws.receive_json()
        assert "error" in answer
        # the session is still alive: the next command is served as usual
        ws.send_json({"cmd": "seek", "ts": _ts(6.5)})
        sought = _drain_until(ws, lambda f: f.get("ts") == _ts(6.5))
        assert _component(sought, "door_L1")["health"] == "crit"
        ws.send_text("{still not json")
        assert "error" in ws.receive_json()
        ws.send_json({"cmd": "pause"})
        assert _drain_until(ws, lambda f: f.get("playing") is False)["ts"] == _ts(6.5)


# -- should_fix ---------------------------------------------------------------------------


def test_staleness_beats_an_open_episode(settings: Settings) -> None:
    """Contract section 5: a stale component reports ``ok`` even inside an open episode - but
    keeps ``alert`` and says ``stale``."""
    from nebulax.api.state import EpisodeRec

    state = FleetState(settings)
    key = ("T02", 3, "door", "door_R1")
    state.episodes.append(
        EpisodeRec(
            episode_id="T02-door_R1-0",
            train_id="T02",
            car=3,
            subsystem="door",
            component_id="door_R1",
            model="cusum_cycle_scalar",
            t_start_ns=int(T0.value),
            t_end_ns=None,
            peak_score=9.0,
            n_rows=3,
            fault_type="friction",
            t_onset_ns=None,
            t_failure_ns=None,
            lead_to_failure_h=None,
            matched=False,
        )
    )
    state._index_episodes()
    fresh = state.component_state(key, int((T0 + pd.Timedelta(hours=1)).value))
    assert fresh["stale"] is False and fresh["health"] == "crit"
    still = state.component_state(key, int((T0 + pd.Timedelta(hours=5)).value))
    assert still["stale"] is False and still["health"] == "crit"  # inside a nightly door gap
    stale = state.component_state(key, int((T0 + pd.Timedelta(hours=6.5)).value))
    assert stale["stale"] is True
    assert stale["health"] == "ok"
    assert stale["alert"] is False  # the row's own flag, untouched


def test_inject_patches_the_series_instead_of_replacing_it(settings: Settings) -> None:
    """The injection covers a window; everything scored before it stays on the chart."""
    state = FleetState(settings)
    key = ("T01", 0, "pneumatic", "apu_1")
    base = state.series_for(key)
    assert len(base.ts) == 12
    times = pd.DatetimeIndex([T0 + pd.Timedelta(hours=h) for h in (8, 9, 10, 11)]).as_unit("ms")
    scores = S.coerce_scores(
        pd.DataFrame(
            {
                "timestamp": times,
                "train_id": "T01",
                "car": np.int8(0),
                "subsystem": "pneumatic",
                "component_id": "apu_1",
                "model": "sparse_autoencoder",
                "score": np.array([5.0, 6.0, 7.0, 8.0]),
                "threshold": 1.0,
                "alert": np.array([True] * 4),
                "top_signals_json": ["[]"] * 4,
            }
        )
    )
    state.apply_overlay(key=key, scores=scores, episodes=None, long=None, features=None)
    merged = state.series_for(key)
    assert len(merged.ts) == 12  # 8 kept + 4 injected, not 4
    assert merged.ts[0] == base.ts[0]
    assert merged.score[7] == pytest.approx(1.8)  # the on-disk row at 07:00
    assert merged.score[8] == pytest.approx(5.0)  # the first injected row


def test_series_reads_train_context_rows_for_a_component(client: TestClient) -> None:
    """``speed`` lives on the run's ``component_id="train"``, ``car=0`` rows, so asking a door
    leaf for it reads those and says where the answer came from."""
    body = client.get(
        "/api/train/T01/component/door_L1/series", params={"signal": "speed", "car": 3}
    ).json()
    assert body["source_component"] == "train"
    assert body["component_id"] == "door_L1"
    assert body["car"] == 0
    assert body["n_raw"] == 12
    assert body["unit"] == "m/s"
    own = client.get(
        "/api/train/T01/component/door_L1/series", params={"signal": "pos", "car": 3}
    ).json()
    assert own["source_component"] == "door_L1"


def test_inject_404s_for_an_unknown_train(client: TestClient) -> None:
    """Checked before the fitted winner is looked for, so it is a 404 and never a 503."""
    response = client.post(
        "/api/sim/inject",
        json={
            "train_id": "T99",
            "car": 0,
            "component_id": "apu_1",
            "fault_type": "air_leak",
            "severity_ramp_days": 2.0,
            "ts": _ts(11),
        },
    )
    assert response.status_code == 404
    assert "T99" in response.json()["detail"]


def test_clearing_the_overlay_drops_the_injected_advisories(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_scoring(monkeypatch)
    _fake_advisory(monkeypatch)
    body = client.post(
        "/api/sim/inject",
        json={
            "train_id": "T01",
            "car": 0,
            "component_id": "apu_1",
            "fault_type": "air_leak",
            "severity_ramp_days": 2.0,
            "seed": 7,
            "ts": _ts(11),
        },
    ).json()
    episode_id = body["episodes"][0]["episode_id"]
    assert episode_id.startswith("inj-")
    assert client.post(f"/api/advisory/{episode_id}").status_code == 200
    assert client.get(f"/api/advisory/{episode_id}").status_code == 200
    # a scored episode's advisory survives the clear; the injected one does not
    assert client.post("/api/advisory/T01-door_L1-1").status_code == 200
    client.delete("/api/sim/inject")
    assert client.get(f"/api/advisory/{episode_id}").status_code == 404
    assert client.get("/api/advisory/T01-door_L1-1").status_code == 200


def test_an_open_alert_carries_no_future_knowledge(client: TestClient) -> None:
    """``matched`` and ``lead_to_failure_h`` are whole-episode values, so they are published
    only once the episode has closed at or before ``ts``."""
    mid = client.get("/api/alerts", params={"ts": _ts(5.5)}).json()[0]
    assert mid["open"] is True and mid["final"] is False
    assert mid["matched"] is None and mid["lead_to_failure_h"] is None
    assert mid["n_rows"] == 3  # kept: the UI labels it as the episode's own size
    done = client.get("/api/alerts", params={"ts": _ts(8)}).json()[0]
    assert done["final"] is True
    assert done["matched"] is True
    assert done["lead_to_failure_h"] == pytest.approx(3.0)


def test_ticker_follows_the_contract_format(client: TestClient) -> None:
    from nebulax.api.state import _g, model_label

    frame = client.get("/api/train/T01/state", params={"ts": _ts(6)}).json()
    assert frame["ticker"] == ["05:00 T01 car 3 door L1 · CUSUM 3 > 2 · episode open 1 h"]
    later = client.get("/api/train/T01/state", params={"ts": _ts(8)}).json()["ticker"]
    assert later[1] == "05:00 T01 car 3 door L1 · CUSUM 3 > 2 · episode open 3 h"
    assert later[0].startswith("07:00 T01 car 3 door L1 · CUSUM")
    assert "episode closed after 2 h" in later[0]
    assert model_label("sparse_autoencoder") == "SAE"
    assert model_label("lgbm_residual") == "LGBM"
    assert _g(3.9123e10) == "3.91e+10"  # three significant figures, contract example


def test_health_does_not_leak_the_data_directory(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert "data_dir" not in body
    assert set(body) == {"status", "scores_loaded", "n_trains", "clock", "overlay"}


def test_the_sim_clock_never_runs_past_the_last_day(tmp_path: Path) -> None:
    """The bearing windows close just after midnight on 1 Oct; the replay clock stops at
    2026-09-30T23:59:59Z all the same (contract section 1)."""
    scores_dir = tmp_path / "data" / "scores"
    scores_dir.mkdir(parents=True)
    times = pd.DatetimeIndex(
        [pd.Timestamp("2026-09-30T23:55:00Z"), pd.Timestamp("2026-10-01T00:00:00Z")]
    ).as_unit("ms")
    S.coerce_scores(
        pd.DataFrame(
            {
                "timestamp": times,
                "train_id": "T01",
                "car": np.int8(3),
                "subsystem": "bearing",
                "component_id": "axlebox_1L",
                "model": "cusum_cycle_scalar",
                "score": np.array([0.1, 0.2]),
                "threshold": 1.0,
                "alert": np.array([False, False]),
                "top_signals_json": ["[]", "[]"],
            }
        )
    ).to_parquet(scores_dir / "scores.parquet", engine="pyarrow", index=False)
    state = FleetState(Settings(data_dir=tmp_path / "data", results_dir=tmp_path / "r"))
    assert state.clock["end"] == "2026-09-30T23:59:59Z"


def test_the_alert_context_leaves_the_glossary_to_the_advisory_module(settings: Settings) -> None:
    from nebulax.api.main import build_alert_context

    state = FleetState(settings)
    ctx = build_alert_context(state, state.episode("T01-door_L1-1"))
    assert ctx["glossary"] == ""
    assert "median" in ctx["fleet_peer_summary"]
    assert "alarm episode" in ctx["fleet_peer_summary"]  # the methodology rides here instead
