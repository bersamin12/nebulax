"""Tests for ``nebulax.demo.scoring`` and ``scripts/score_for_demo.py``.

Everything runs against a tiny fleet written into ``tmp_path`` with
:func:`nebulax.schema.write_dataset` and a hand-made ``runs.parquet`` holding one row per
winner: no test here reads ``data/``, and the only model used is ``cusum_cycle_scalar``
(mean/std fit, no GPU, milliseconds).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nebulax import schema as S
from nebulax.bench import metrics as M
from nebulax.demo import scoring as sc

REPO_ROOT = Path(__file__).resolve().parents[1]

T0 = pd.Timestamp("2026-09-01T00:00:00Z")
TRAINS = ("T01", "T02", "T03")
N_DOOR_CYCLES = 200
N_BEARING_WINDOWS = 120
BOXES = ("axlebox_1L", "axlebox_1R", "axlebox_2L", "axlebox_2R",
         "axlebox_3L", "axlebox_3R", "axlebox_4L", "axlebox_4R")

#: The one seeded fault: T03's right-hand door, onset at cycle 120, failure at cycle 180.
FAULT_TRAIN = "T03"
FAULT_COMPONENT = "door_R1"
FAULT_ONSET_CYCLE = 120
FAULT_FAILURE_CYCLE = 180
DOOR_PITCH_S = 300.0
BEARING_PITCH_S = 300.0
H_S = 3 * 86400.0


# --------------------------------------------------------------------------------------
# A tiny fleet
# --------------------------------------------------------------------------------------


def _long_stub(run_id: str, train_id: str, car: int, subsystem: str, component_id: str) -> pd.DataFrame:
    """Four legal telemetry rows - ``write_dataset`` wants a long table, the cycle_features
    winners never read one."""
    signal = S.SIGNALS[subsystem][0]
    ts = pd.date_range(T0, periods=4, freq="1s", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "source": "sim",
            "run_id": run_id,
            "train_id": train_id,
            "car": car,
            "subsystem": subsystem,
            "component_id": component_id,
            "signal": signal,
            "value": np.arange(4, dtype=np.float32),
        }
    )


def _door_features(rng: np.random.Generator, run_id: str, train_id: str, car: int, component_id: str,
                   *, faulty: bool) -> pd.DataFrame:
    k = np.arange(N_DOOR_CYCLES)
    t_start = T0 + pd.to_timedelta(k * DOOR_PITCH_S, unit="s")
    drift = np.zeros(N_DOOR_CYCLES)
    if faulty:
        ramp = np.clip((k - FAULT_ONSET_CYCLE) / (FAULT_FAILURE_CYCLE - FAULT_ONSET_CYCLE), 0.0, None)
        drift = 6.0 * ramp
    return pd.DataFrame(
        {
            "run_id": run_id,
            "source": "sim",
            "train_id": train_id,
            "car": np.int8(car),
            "subsystem": "door",
            "component_id": component_id,
            "cycle_id": k.astype(np.int64),
            "t_start": t_start,
            "t_end": t_start + pd.Timedelta(seconds=30),
            "closing_time": (3.0 + 0.05 * rng.standard_normal(N_DOOR_CYCLES) + drift).astype(np.float32),
            "i_peak": (4.0 + 0.1 * rng.standard_normal(N_DOOR_CYCLES) + 0.5 * drift).astype(np.float32),
            "T_motor": (30.0 + 0.5 * rng.standard_normal(N_DOOR_CYCLES)).astype(np.float32),
        }
    )


def _bearing_features(rng: np.random.Generator, run_id: str, train_id: str, car: int) -> pd.DataFrame:
    k = np.arange(N_BEARING_WINDOWS)
    rows = []
    for box in BOXES:
        t_start = T0 + pd.to_timedelta(k * BEARING_PITCH_S, unit="s")
        rows.append(
            pd.DataFrame(
                {
                    "run_id": run_id,
                    "source": "sim",
                    "train_id": train_id,
                    "car": np.int8(car),
                    "subsystem": "bearing",
                    "component_id": box,
                    "cycle_id": k.astype(np.int64),
                    "t_start": t_start,
                    "t_end": t_start + pd.Timedelta(seconds=BEARING_PITCH_S),
                    "T_box_mean": (40.0 + 0.4 * rng.standard_normal(N_BEARING_WINDOWS)).astype(np.float32),
                    "vib_rms_mean": (1.0 + 0.05 * rng.standard_normal(N_BEARING_WINDOWS)).astype(np.float32),
                    "T_amb": (28.0 + 0.2 * rng.standard_normal(N_BEARING_WINDOWS)).astype(np.float32),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _fault_log_row(run_id: str, train_id: str, car: int, component_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": [run_id],
            "train_id": [train_id],
            "car": [np.int8(car)],
            "subsystem": ["door"],
            "component_id": [component_id],
            "fault_type": ["friction"],
            "t_onset": [T0 + pd.to_timedelta(FAULT_ONSET_CYCLE * DOOR_PITCH_S, unit="s")],
            "t_failure": [T0 + pd.to_timedelta(FAULT_FAILURE_CYCLE * DOOR_PITCH_S, unit="s")],
            "t_functional_failure": [pd.NaT],
            "gamma": [np.float32(1.0)],
            "shape": ["power"],
            "params_json": ["{}"],
        }
    )


@pytest.fixture(scope="module")
def fleet(tmp_path_factory) -> Path:
    """A three-train fleet: two door runs per train (so a door has a concurrent peer) and one
    bearing run per train (eight axle boxes, whose peer group is inside the one run)."""
    from nebulax.features.cycles import label_from_fault_log

    root = tmp_path_factory.mktemp("fleet")
    rng = np.random.default_rng(0)
    logs: list[pd.DataFrame] = []
    index_runs: list[dict] = []

    for i, train in enumerate(TRAINS):
        for j, (component, car) in enumerate((("door_L1", 1), ("door_R1", 2))):
            run_id = f"door_{i:02d}{j}"
            faulty = train == FAULT_TRAIN and component == FAULT_COMPONENT
            feats = _door_features(rng, run_id, train, car, component, faulty=faulty)
            fl = _fault_log_row(run_id, train, car, component) if faulty else S.empty_fault_log()
            feats = label_from_fault_log(feats, fl)
            S.write_dataset(
                _long_stub(run_id, train, car, "door", component), feats, fl, None,
                root, "sim", run_id, meta={"subsystem": "door", "train_id": train, "car": car},
            )
            logs.append(fl)
            index_runs.append({"run_id": run_id, "subsystem": "door", "train_id": train, "car": car,
                               "component_id": component})
        run_id = f"bearing_{i:02d}"
        feats = _bearing_features(rng, run_id, train, 1)
        feats = label_from_fault_log(feats, S.empty_fault_log())
        S.write_dataset(
            _long_stub(run_id, train, 1, "bearing", BOXES[0]), feats, None, None,
            root, "sim", run_id, meta={"subsystem": "bearing", "train_id": train, "car": 1},
        )
        index_runs.append({"run_id": run_id, "subsystem": "bearing", "train_id": train, "car": 1,
                           "component_id": BOXES[0]})

    S.coerce_fault_log(pd.concat([f for f in logs if len(f)], ignore_index=True)).to_parquet(
        root / "fault_log.parquet", engine="pyarrow", index=False
    )
    (root / "index.json").write_text(json.dumps({"runs": index_runs}), encoding="utf-8")
    _write_runs_parquet(root / "runs.parquet")
    return root


def _runs_row(key: str, **over) -> dict:
    want = sc.WINNER_KEYS[key]
    row = {
        "dataset": want["dataset"],
        "dataset_subsystem": want["dataset_subsystem"],
        "model": want["model"],
        "input_kind": want["input_kind"],
        "window": float("nan") if want["window"] is None else float(want["window"]),
        "peer_norm": bool(want["peer_norm"]),
        "contamination": 0.0,
        "params": "{}",
        "data_kwargs": '{"max_runs": 8}',
        "split": want.get("split", "sim_loo_unit"),
        "feature_set": "default",
        "train_regime": "normal_only",
        "seed": 0,
        "H": H_S,
        "max_minutes": 20.0,
        "config_hash": f"hash_{key}",
        "k_consecutive": 3.0,
        "merge_gap_s": 3600.0,
        "max_step_s": 450.0,
        "window_seconds": 30.0,
        "threshold_budget": 1.0,
        "status": "ok",
    }
    row.update(over)
    return row


def _write_runs_parquet(path: Path) -> None:
    pd.DataFrame([_runs_row("door"), _runs_row("bearing")]).to_parquet(path, engine="pyarrow", index=False)


def _winner(fleet: Path, key: str) -> sc.WinnerSpec:
    return sc.load_winners(fleet / "runs.parquet", keys=[key])[key]


@pytest.fixture(scope="module")
def door_fleet(fleet: Path) -> sc.FleetIndex:
    return sc.load_fleet("door", root=fleet, runs_path=fleet / "runs.parquet", cache_dir=None)


@pytest.fixture(scope="module")
def bearing_fleet(fleet: Path) -> sc.FleetIndex:
    return sc.load_fleet("bearing", root=fleet, runs_path=fleet / "runs.parquet", cache_dir=None)


# --------------------------------------------------------------------------------------
# WINNERS
# --------------------------------------------------------------------------------------


def test_winners_come_from_the_results_frame(fleet: Path):
    spec = _winner(fleet, "door")
    assert spec.model == "cusum_cycle_scalar"
    assert spec.input_kind == "cycle_features"
    assert spec.peer_norm is True
    assert spec.window is None
    assert spec.config_hash == "hash_door"
    # max_runs capped the sweep, never the demo.
    assert "max_runs" not in spec.data_kwargs


def test_missing_winner_row_fails_loudly(tmp_path: Path):
    path = tmp_path / "runs.parquet"
    pd.DataFrame([_runs_row("bearing")]).to_parquet(path, engine="pyarrow", index=False)
    with pytest.raises(ValueError, match="no benchmark row for winner 'door'"):
        sc.load_winners(path, keys=["door"])


def test_failed_winner_row_fails_loudly(tmp_path: Path):
    path = tmp_path / "runs.parquet"
    pd.DataFrame([_runs_row("door", status="failed")]).to_parquet(path, engine="pyarrow", index=False)
    with pytest.raises(ValueError, match="status="):
        sc.load_winners(path, keys=["door"])


def test_ambiguous_winner_row_fails_loudly(tmp_path: Path):
    path = tmp_path / "runs.parquet"
    pd.DataFrame([_runs_row("door"), _runs_row("door", config_hash="hash_door_2")]).to_parquet(
        path, engine="pyarrow", index=False
    )
    with pytest.raises(ValueError, match="ambiguous"):
        sc.load_winners(path, keys=["door"])


# --------------------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------------------


def test_leave_one_out_never_fits_on_the_held_out_train(door_fleet: sc.FleetIndex, monkeypatch):
    data = door_fleet.data
    trains = np.asarray(data.labels["train_id"].astype(str))
    seen: dict[str, np.ndarray] = {}

    from nebulax.bench import runner as R

    real_fit = R._fit_model

    def spy(model, d, spec, train_idx, **kw):
        seen["train_idx"] = np.asarray(train_idx)
        return real_fit(model, d, spec, train_idx, **kw)

    monkeypatch.setattr(sc.R, "_fit_model", spy)
    fw = sc.fit_winner("door", exclude_train="T03", index=door_fleet)

    fitted_trains = set(trains[seen["train_idx"]])
    assert "T03" not in fitted_trains
    assert fitted_trains <= {"T01", "T02"}
    # normal_only, contamination 0: every fitted row is healthy...
    assert not data.y_binary[seen["train_idx"]].any()
    # ...and the count is the healthy rows of the training trains minus the validation carve.
    split = door_fleet.split_for("T03")
    healthy_train_rows = int((~data.y_binary[split.train].astype(bool)).sum())
    assert fw.n_train_rows == healthy_train_rows == seen["train_idx"].size
    assert set(np.asarray(trains)[split.test]) == {"T03"}
    assert fw.held_out_train == "T03"
    assert np.isfinite(fw.threshold)


def test_load_fleet_refuses_a_partial_fleet(fleet: Path):
    """The loaders sample by default (``max_runs``, the streamed path's four-run fallback);
    scoring a demo fleet that is missing runs must be an error, not a quiet ``nodata``."""
    import dataclasses

    spec = _winner(fleet, "bearing")
    partial = dataclasses.replace(spec, data_kwargs={"run_ids": ["bearing_00", "bearing_01"]})
    with pytest.raises(ValueError, match="missing 1 of 3 run"):
        sc.load_fleet("bearing", root=fleet, cache_dir=None, spec=partial)


def test_fitted_winner_carries_the_contract_attributes(door_fleet: sc.FleetIndex):
    fw = sc.fit_winner("door", exclude_train="T01", index=door_fleet)
    for attr in (
        "subsystem", "model_name", "params", "input_kind", "window", "peer_norm", "threshold",
        "feature_names", "train_median", "train_mad", "held_out_train", "fitted_at",
    ):
        assert hasattr(fw, attr), attr
    assert len(fw.feature_names) == len(fw.train_median) == len(fw.train_mad)
    assert fw.attribution == "robust_z"
    # peer columns really are in the feature space (the peer_norm axis is not decoration)
    assert any(n.endswith("_peer_delta") for n in fw.feature_names)


def test_fitted_winner_round_trips_through_a_pickle(door_fleet: sc.FleetIndex, tmp_path: Path):
    fw = sc.fit_winner("door", exclude_train="T01", index=door_fleet)
    sc.save_fitted(fw, scores_dir=tmp_path)
    back = sc.load_fitted("door", "T01", scores_dir=tmp_path)
    assert back.threshold == fw.threshold
    assert back.feature_names == fw.feature_names
    np.testing.assert_allclose(back.train_median, fw.train_median)


# --------------------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------------------


def test_scores_have_the_frozen_schema(door_fleet: sc.FleetIndex):
    fw = sc.fit_winner("door", exclude_train="T02", index=door_fleet)
    scores = sc.score_held_out(fw, door_fleet)
    assert list(scores.columns) == list(S.SCORES_COLUMNS)
    for col, dtype in S.SCORES_COLUMNS.items():
        if dtype == "object":
            continue
        assert str(scores[col].dtype) == dtype, col
    assert scores["timestamp"].dt.tz is not None
    assert set(scores["train_id"].astype(str)) == {"T02"}
    assert (scores["threshold"] == np.float32(fw.threshold)).all()

    payload = json.loads(scores["top_signals_json"].iloc[0])
    assert isinstance(payload, list) and len(payload) <= sc.MAX_TOP_SIGNALS
    assert set(payload[0]) == {"signal", "z", "value"}
    assert payload[0]["signal"] in fw.feature_names
    z = [abs(p["z"]) for p in payload]
    assert z == sorted(z, reverse=True)


def test_scores_cover_every_row_of_the_held_out_train(door_fleet: sc.FleetIndex):
    fw = sc.fit_winner("door", exclude_train="T02", index=door_fleet)
    scores = sc.score_held_out(fw, door_fleet)
    split = door_fleet.split_for("T02")
    assert len(scores) == split.test.size == 2 * N_DOOR_CYCLES


def _bare_winner(threshold: float = 1.0, **kw) -> sc.FittedWinner:
    return sc.FittedWinner(
        subsystem="door", model_name="cusum_cycle_scalar", params={}, input_kind="cycle_features",
        window=None, peer_norm=True, threshold=threshold, feature_names=["a"],
        train_median=np.zeros(1), train_mad=np.ones(1), held_out_train="T01", fitted_at="",
        k_consecutive=3, merge_gap_s=3600.0, max_step_s=450.0, window_seconds=300.0, **kw,
    )


def test_alert_needs_k_consecutive_rows():
    fw = _bare_winner(threshold=1.0)
    t = T0 + pd.to_timedelta(np.arange(12) * 300.0, unit="s")
    series = np.full(12, "s1", dtype=object)
    # two above-threshold rows, then a three-row run: only the second becomes an episode.
    scores = np.zeros(12)
    scores[1:3] = 5.0
    scores[6:9] = 5.0
    alert = sc._alert_flags(fw, scores, t.to_numpy(), series)
    assert not alert[1:3].any()
    assert alert[6:9].all()
    assert alert.sum() == 3


def test_alert_never_pools_two_components():
    fw = _bare_winner(threshold=1.0)
    t = T0 + pd.to_timedelta(np.tile(np.arange(4) * 300.0, 2), unit="s")
    series = np.asarray(["s1"] * 4 + ["s2"] * 4, dtype=object)
    scores = np.zeros(8)
    scores[2:6] = 5.0  # two rows on s1 and two on s2 - no episode on either
    alert = sc._alert_flags(fw, scores, t.to_numpy(), series)
    assert not alert.any()


def test_alert_respects_the_time_contiguity_budget():
    fw = _bare_winner(threshold=1.0)
    # three above-threshold rows, but a week apart: array-adjacent, not time-adjacent.
    t = T0 + pd.to_timedelta([0.0, 7 * 86400.0, 14 * 86400.0], unit="s")
    alert = sc._alert_flags(fw, np.full(3, 5.0), t.to_numpy(), np.full(3, "s1", dtype=object))
    assert not alert.any()


# --------------------------------------------------------------------------------------
# score_frames / score_run
# --------------------------------------------------------------------------------------


def test_score_frames_equals_score_run(fleet: Path, door_fleet: sc.FleetIndex):
    fw = sc.fit_winner("door", exclude_train="T01", index=door_fleet)
    run_dir = fleet / "source=sim" / "run_id=door_000"
    feats = pd.read_parquet(run_dir / "features.parquet")
    long = pd.read_parquet(run_dir / "telemetry.parquet")

    from_run = sc.score_run(fw, run_dir)
    from_frames = sc.score_frames(fw, long, feats, train_id="T01", car=1, component_id="door_L1")
    pd.testing.assert_frame_equal(from_run, from_frames)
    assert len(from_run) == N_DOOR_CYCLES
    assert list(from_run.columns) == list(S.SCORES_COLUMNS)


def test_score_run_reproduces_the_fleet_scores_when_the_peers_are_in_the_run(
    fleet: Path, bearing_fleet: sc.FleetIndex
):
    """A bearing run holds all eight of its axle boxes, so its peer group is complete inside
    the one run and the isolated path must give the fleet's own numbers back."""
    fw = sc.fit_winner("bearing", exclude_train="T03", index=bearing_fleet)
    fleet_scores = sc.score_held_out(fw, bearing_fleet)
    run_scores = sc.score_run(fw, fleet / "source=sim" / "run_id=bearing_02")
    assert len(fleet_scores) == len(run_scores) == len(BOXES) * N_BEARING_WINDOWS
    key = ["component_id", "timestamp"]
    a = fleet_scores.sort_values(key).reset_index(drop=True)
    b = run_scores.sort_values(key).reset_index(drop=True)
    np.testing.assert_allclose(a["score"].to_numpy(), b["score"].to_numpy(), rtol=1e-5, atol=1e-6)
    assert (a["alert"].to_numpy() == b["alert"].to_numpy()).all()


def test_a_lone_door_scores_with_zeroed_peer_columns(fleet: Path, door_fleet: sc.FleetIndex):
    """The documented fallback: peer_normalise over one door yields NaN peer columns, which
    ``_finite`` turns into 0.0 - "exactly average against its peers"."""
    fw = sc.fit_winner("door", exclude_train="T01", index=door_fleet)
    feats = pd.read_parquet(fleet / "source=sim" / "run_id=door_000" / "features.parquet")
    X, _ = sc._cycle_matrix(fw, feats)
    peer_cols = [i for i, n in enumerate(fw.feature_names) if n.endswith("_peer_delta")]
    assert peer_cols
    assert np.allclose(X[:, peer_cols], 0.0)


# --------------------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------------------


def _scores_stub(alert: list[bool], t0: pd.Timestamp = T0) -> pd.DataFrame:
    n = len(alert)
    return S.coerce_scores(
        pd.DataFrame(
            {
                "timestamp": t0 + pd.to_timedelta(np.arange(n) * 300.0, unit="s"),
                "train_id": "T03",
                "car": np.int8(2),
                "subsystem": "door",
                "component_id": FAULT_COMPONENT,
                "model": "cusum_cycle_scalar",
                "score": np.where(alert, 9.0, 0.1).astype(np.float32),
                "threshold": np.float32(1.0),
                "alert": np.asarray(alert, dtype=bool),
                "top_signals_json": ["[]"] * n,
            }
        )
    )


def test_episodes_from_scores_matches_and_computes_the_lead(fleet: Path):
    fault_log = pd.read_parquet(fleet / "fault_log.parquet")
    n = 200
    alert = [False] * n
    alert[100:106] = [True] * 6  # inside [onset - H, failure]
    eps = sc.episodes_from_scores(_scores_stub(alert), fault_log, H=H_S)
    assert len(eps) == 1
    row = eps.iloc[0]
    assert list(eps.columns) == list(sc.EPISODE_COLUMNS)
    assert bool(row["matched"]) is True
    assert row["fault_type"] == "friction"
    assert row["n_rows"] == 6
    assert row["episode_id"] == f"T03-{FAULT_COMPONENT}-0"
    t_failure = T0 + pd.to_timedelta(FAULT_FAILURE_CYCLE * DOOR_PITCH_S, unit="s")
    expected = (t_failure - row["t_start"]).total_seconds() / 3600.0
    assert row["lead_to_failure_h"] == pytest.approx(expected, rel=1e-5)
    assert row["peak_score"] == pytest.approx(9.0)


def test_episodes_far_from_the_fault_are_false_alarms_with_nan_lead(fleet: Path):
    fault_log = pd.read_parquet(fleet / "fault_log.parquet")
    # The fault window opens at onset - 72 h, i.e. well before t=0 here, so put the episode
    # after the failure instead.
    n = 200
    alert = [False] * n
    alert[190:196] = [True] * 6
    eps = sc.episodes_from_scores(_scores_stub(alert), fault_log, H=H_S)
    assert len(eps) == 1
    assert not bool(eps.iloc[0]["matched"])
    assert np.isnan(eps.iloc[0]["lead_to_failure_h"])
    assert eps.iloc[0]["fault_type"] == ""


def test_episodes_are_numbered_per_component(fleet: Path):
    fault_log = pd.read_parquet(fleet / "fault_log.parquet")
    n = 60
    alert = [False] * n
    alert[5:9] = [True] * 4
    alert[40:44] = [True] * 4
    eps = sc.episodes_from_scores(_scores_stub(alert), fault_log, H=H_S)
    assert list(eps["episode_id"]) == [f"T03-{FAULT_COMPONENT}-0", f"T03-{FAULT_COMPONENT}-1"]


def test_two_faults_on_one_component_are_counted_separately(fleet: Path):
    """The MetroPT-3 shape: several events on the same component. ``matched`` assigns each
    episode to one fault (so its ``fault_type`` is unambiguous) while ``faults_detected``
    answers per fault, so the recall numerator cannot disagree with its denominator."""
    n = 400
    alert = [False] * n
    alert[100:106] = [True] * 6
    alert[300:306] = [True] * 6
    scores = _scores_stub(alert)
    two = pd.DataFrame(
        {
            "train_id": ["T03", "T03"],
            "car": [np.int8(2)] * 2,
            "subsystem": ["door", "door"],
            "component_id": [FAULT_COMPONENT] * 2,
            "fault_type": ["friction", "backlash"],
            "t_onset": [T0 + pd.Timedelta(seconds=100 * 300.0), T0 + pd.Timedelta(seconds=300 * 300.0)],
            "t_failure": [T0 + pd.Timedelta(seconds=150 * 300.0), T0 + pd.Timedelta(seconds=350 * 300.0)],
        }
    )
    eps = sc.episodes_from_scores(scores, two, H=0.0)
    assert len(eps) == 2 and eps["matched"].all()
    assert list(eps["fault_type"]) == ["friction", "backlash"]
    assert sc.faults_detected(eps, two, H=0.0).tolist() == [True, True]
    summary = sc.event_summary(eps, two, scores, H=0.0)
    assert summary["events_total"] == 2 and summary["events_detected"] == 2
    assert summary["episodes_false"] == 0


def test_episodes_split_on_a_gap_wider_than_max_step_s():
    """The door's ~5 h nightly gap: two array-adjacent alarm blocks, two episodes."""
    n = 20
    scores = _scores_stub([True] * n)
    # push the second half a full night later, leaving the two blocks array-adjacent
    ts = pd.to_datetime(scores["timestamp"], utc=True)
    ts.iloc[10:] = ts.iloc[10:] + pd.Timedelta(hours=5)
    scores["timestamp"] = ts.astype(S.TIMESTAMP_DTYPE)
    eps = sc.episodes_from_scores(scores, None, max_step_s=450.0, merge_gap_s=3600.0, k=3)
    assert len(eps) == 2
    assert list(eps["n_rows"]) == [10, 10]
    assert list(eps["episode_id"]) == [f"T03-{FAULT_COMPONENT}-0", f"T03-{FAULT_COMPONENT}-1"]


def test_two_runs_closer_than_merge_gap_s_are_one_episode():
    """A single below-threshold row does not end an episode: the two runs are merged."""
    alert = [False] * 30
    alert[10:14] = [True] * 4
    alert[15:19] = [True] * 4
    scores = _scores_stub(alert)
    scores.loc[14, "alert"] = True  # the merged gap row keeps its (low) score
    eps = sc.episodes_from_scores(scores, None, max_step_s=450.0, merge_gap_s=3600.0, k=3)
    assert len(eps) == 1
    assert int(eps.iloc[0]["n_rows"]) == 9


def test_episodes_record_the_horizon_they_were_matched_with(fleet: Path):
    fault_log = pd.read_parquet(fleet / "fault_log.parquet")
    alert = [False] * 200
    alert[100:106] = [True] * 6
    eps = sc.episodes_from_scores(_scores_stub(alert), fault_log, H=2 * 86400.0)
    assert list(eps.columns) == list(sc.EPISODE_COLUMNS)
    assert eps["H_s"].tolist() == [2 * 86400.0]


def test_scoreable_from_fault_log_blanks_the_rows_the_benchmark_blanks():
    n = 40
    scores = _scores_stub([False] * n)
    t_failure = T0 + pd.Timedelta(seconds=20 * DOOR_PITCH_S)
    fl = pd.DataFrame(
        {
            "train_id": ["T03"],
            "car": [np.int8(2)],
            "subsystem": ["door"],
            "component_id": [FAULT_COMPONENT],
            "fault_type": ["friction"],
            "t_onset": [T0],
            "t_failure": [t_failure],
        }
    )
    # sim: everything after t_failure, to the end of the run
    keep = sc.scoreable_from_fault_log(scores, fl, rule="post_failure")
    assert keep.sum() == 21 and not keep[21:].any()
    # MetroPT-3: only the 24 h after the failure (300 s rows -> 288 of them, but the frame ends first)
    keep = sc.scoreable_from_fault_log(scores, fl, rule="post_repair")
    assert keep.sum() == 21
    # a fault log with no failure time blanks nothing
    fl2 = fl.assign(t_failure=[pd.NaT])
    assert sc.scoreable_from_fault_log(scores, fl2, rule="post_failure").all()
    assert sc.scoreable_from_fault_log(scores, None).all()


def test_blanked_rows_are_not_episodes_and_are_not_scored_train_days():
    """post-``t_failure`` rows keep their score, but they are not a population."""
    n = 40
    alert = [False] * n
    scores = _scores_stub(alert)
    scores.loc[30:, "score"] = np.float32(9.0)  # trivially anomalous, post-failure
    t_failure = T0 + pd.Timedelta(seconds=25 * DOOR_PITCH_S)
    fl = pd.DataFrame(
        {
            "train_id": ["T03"], "car": [np.int8(2)], "subsystem": ["door"],
            "component_id": [FAULT_COMPONENT], "fault_type": ["friction"],
            "t_onset": [T0], "t_failure": [t_failure],
        }
    )
    keep = sc.scoreable_from_fault_log(scores, fl, rule="post_failure")
    eps = sc.episodes_from_scores(scores, fl, H=H_S, max_step_s=450.0, scoreable=keep)
    assert eps.empty
    summary = sc.event_summary(eps, fl, scores, H=H_S, scoreable=keep)
    assert summary["n_rows_scoreable"] == 26
    assert summary["scored_train_days"] == pytest.approx(25 * DOOR_PITCH_S / 86400.0)
    # ... and counting every row instead would inflate the false-alarm denominator
    assert sc.event_summary(eps, fl, scores, H=H_S)["scored_train_days"] == pytest.approx(
        (n - 1) * DOOR_PITCH_S / 86400.0
    )


def test_episodes_from_empty_scores_is_an_empty_typed_frame():
    eps = sc.episodes_from_scores(S.empty_scores(), None)
    assert list(eps.columns) == list(sc.EPISODE_COLUMNS)
    assert eps.empty


def test_episodes_agree_with_the_benchmark_episode_definition(door_fleet: sc.FleetIndex, fleet: Path):
    fw = sc.fit_winner("door", exclude_train=FAULT_TRAIN, index=door_fleet)
    scores = sc.score_held_out(fw, door_fleet)
    eps = sc.episodes_from_scores(scores, pd.read_parquet(fleet / "fault_log.parquet"), H=H_S)
    assert len(eps)
    for _, row in eps.iterrows():
        sel = scores[
            (scores["train_id"].astype(str) == row["train_id"])
            & (scores["component_id"].astype(str) == row["component_id"])
            & (scores["timestamp"] >= row["t_start"])
            & (scores["timestamp"] <= row["t_end"])
        ].sort_values("timestamp")
        # >= k rows, every one strictly above the threshold, no gap wider than max_step_s
        assert len(sel) == row["n_rows"] >= fw.k_consecutive
        assert (sel["score"].to_numpy() > np.float32(fw.threshold)).all()
        gaps = np.diff(M.to_epoch_seconds(sel["timestamp"].to_numpy()))
        assert gaps.size == 0 or gaps.max() <= fw.max_step_s
        assert row["peak_score"] == pytest.approx(float(sel["score"].max()), rel=1e-5)


# --------------------------------------------------------------------------------------
# The batch script
# --------------------------------------------------------------------------------------


def _load_script():
    path = REPO_ROOT / "scripts" / "score_for_demo.py"
    spec = importlib.util.spec_from_file_location("score_for_demo", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["score_for_demo"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def scored(fleet: Path, tmp_path_factory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("scores")
    script = _load_script()
    rc = script.main(
        [
            "--out", str(out),
            "--sim-root", str(fleet),
            "--runs", str(fleet / "runs.parquet"),
            "--cache-dir", "",
            "--subsystems", "door,bearing",
        ]
    )
    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    return out, manifest


def test_script_writes_the_contract_files(scored):
    out, _ = scored
    assert (out / "scores.parquet").exists()
    assert (out / "episodes.parquet").exists()
    assert (out / "manifest.json").exists()
    for train in TRAINS:
        assert (out / "models" / "door" / f"{train}.pkl").exists()
        assert (out / "models" / "bearing" / f"{train}.pkl").exists()


def test_script_scores_every_row_of_every_run(scored):
    out, _ = scored
    scores = pd.read_parquet(out / "scores.parquet")
    assert list(scores.columns) == list(S.SCORES_COLUMNS)
    per_sub = scores.groupby("subsystem", observed=True).size().to_dict()
    assert per_sub["door"] == len(TRAINS) * 2 * N_DOOR_CYCLES
    assert per_sub["bearing"] == len(TRAINS) * len(BOXES) * N_BEARING_WINDOWS
    assert set(scores["train_id"].astype(str)) == set(TRAINS)


def test_manifest_has_the_contract_keys(scored):
    _, manifest = scored
    assert {"generated_at", "git_rev", "subsystems", "protocol", "files"} <= set(manifest)
    for key in ("door", "bearing"):
        entry = manifest["subsystems"][key]
        required = {
            "config_hash", "model", "params", "input_kind", "window", "peer_norm",
            "thresholds", "fit_seconds", "n_rows_scored", "episodes_matched", "episodes_false",
            "events_total", "events_detected", "scored_train_days", "attribution",
        }
        assert required <= set(entry), sorted(required - set(entry))
        assert set(entry["thresholds"]) == set(TRAINS)
        assert all(np.isfinite(v) for v in entry["thresholds"].values())
        assert entry["config_hash"] == f"hash_{key}"


def test_manifest_counts_the_seeded_fault(scored):
    _, manifest = scored
    assert manifest["subsystems"]["door"]["events_total"] == 1
    assert manifest["subsystems"]["bearing"]["events_total"] == 0


def test_rerunning_is_idempotent_and_skips_up_to_date_subsystems(scored, fleet: Path, caplog):
    out, _ = scored
    before = pd.read_parquet(out / "scores.parquet")
    mtime = (out / "scores.parquet").stat().st_mtime_ns
    script = _load_script()
    rc = script.main(
        ["--out", str(out), "--sim-root", str(fleet), "--runs", str(fleet / "runs.parquet"),
         "--cache-dir", "", "--subsystems", "door,bearing"]
    )
    assert rc == 0
    assert (out / "scores.parquet").stat().st_mtime_ns == mtime
    pd.testing.assert_frame_equal(before, pd.read_parquet(out / "scores.parquet"))


def test_forced_rerun_reproduces_the_same_scores(scored, fleet: Path):
    out, _ = scored
    before = pd.read_parquet(out / "scores.parquet")
    script = _load_script()
    rc = script.main(
        ["--out", str(out), "--sim-root", str(fleet), "--runs", str(fleet / "runs.parquet"),
         "--cache-dir", "", "--subsystems", "door", "--force"]
    )
    assert rc == 0
    after = pd.read_parquet(out / "scores.parquet")
    key = ["subsystem", "train_id", "component_id", "timestamp"]
    pd.testing.assert_frame_equal(
        before.sort_values(key).reset_index(drop=True),
        after.sort_values(key).reset_index(drop=True),
    )


def test_episodes_only_reproduces_the_scoring_run_without_fitting(scored, fleet: Path, tmp_path):
    """``--episodes-only`` rebuilds episodes.parquet from what is on disk, byte for byte.

    It reads no feature table and fits nothing: the per-row thresholds are in
    ``scores.parquet`` and the episode definition is in the manifest entry.
    """
    out, _ = scored
    work = tmp_path / "episodes_only"
    shutil.copytree(out, work)
    shutil.rmtree(work / "models")  # nothing here may need a fitted winner
    before = pd.read_parquet(work / "episodes.parquet")
    before_manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))

    script = _load_script()
    rc = script.main(
        ["--out", str(work), "--sim-root", str(fleet), "--runs", str(fleet / "runs.parquet"),
         "--subsystems", "door,bearing", "--episodes-only"]
    )
    assert rc == 0
    after = pd.read_parquet(work / "episodes.parquet")
    key = ["subsystem", "train_id", "component_id", "t_start"]
    pd.testing.assert_frame_equal(
        before.sort_values(key).reset_index(drop=True),
        after.sort_values(key).reset_index(drop=True),
    )
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    for key_ in ("door", "bearing"):
        entry, was = manifest["subsystems"][key_], before_manifest["subsystems"][key_]
        for col in ("episodes_total", "episodes_matched", "episodes_false", "events_detected",
                    "events_total", "n_rows_scored", "scored_train_days"):
            assert entry[col] == was[col], (key_, col)
        # every alert row is covered by an episode row: the two views cannot disagree
        assert entry["n_alert_rows_outside_episodes"] == 0
        assert entry["scoreable_rule"] == "post_failure"
        assert entry["n_rows_blanked_by_rule"] == was["n_rows_blanked_by_loader"]
    assert not (work / "models").exists()


def test_fitted_winners_on_disk_can_rescore_a_run(scored, fleet: Path):
    out, _ = scored
    fw = sc.load_fitted("bearing", "T03", scores_dir=out)
    scores = sc.score_run(fw, fleet / "source=sim" / "run_id=bearing_02")
    assert len(scores) == len(BOXES) * N_BEARING_WINDOWS
    assert list(scores.columns) == list(S.SCORES_COLUMNS)


# --------------------------------------------------------------------------------------
# MetroPT-3: needs the real recording
# --------------------------------------------------------------------------------------

METROPT_RAW = REPO_ROOT / "data" / "raw" / "metropt3"


@pytest.mark.skipif(not METROPT_RAW.exists(), reason="needs data/raw/metropt3")
def test_metropt3_winner_is_a_model_attribution():
    spec = sc.load_winners()["metropt3"]
    assert spec.model == "lgbm_residual"
    assert spec.input_kind == "raw_window"
    assert sc._attribution_mode(spec) == "model"
