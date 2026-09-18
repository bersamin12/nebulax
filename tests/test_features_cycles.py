"""Tests for nebulax.features.cycles: fault-log join and peer normalisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nebulax import schema as S
from nebulax.features import cycles as CY

DAY = 86_400.0


def _bare_features(t0: pd.Timestamp, n=20, component_id="door_L1", run_id="run0", period_s=60.0) -> pd.DataFrame:
    starts = t0 + pd.to_timedelta(np.arange(n) * period_s, unit="s")
    return pd.DataFrame(
        {
            "run_id": run_id,
            "source": "sim",
            "train_id": "T01",
            "car": np.int8(3),
            "subsystem": "door",
            "component_id": component_id,
            "cycle_id": np.arange(n, dtype=np.int64),
            "t_start": starts,
            "t_end": starts + pd.Timedelta(seconds=3),
            "i_peak": np.linspace(5.0, 6.0, n),
        }
    )


def _fault_row(run_id, component_id, fault_type, t_onset, t_failure, gamma=1.0, shape="power"):
    return {
        "run_id": run_id,
        "train_id": "T01",
        "car": np.int8(3),
        "subsystem": "door",
        "component_id": component_id,
        "fault_type": fault_type,
        "t_onset": t_onset,
        "t_failure": t_failure,
        "t_functional_failure": pd.NaT,
        "gamma": np.float32(gamma),
        "shape": shape,
        "params_json": "{}",
    }


def test_label_from_fault_log_empty_fault_log_is_all_healthy(t0):
    feats = S.coerce_features(_bare_features(t0))
    out = CY.label_from_fault_log(feats, S.empty_fault_log())
    assert (out["fault_type"].astype(str) == "healthy").all()
    assert not out["is_faulty"].any()
    assert out["severity"].eq(0.0).all()
    assert out["rul_s"].isna().all()
    assert not out["alarm_window_3d"].any()
    S.validate_features(out, require_labels=True)


def test_label_from_fault_log_power_law_matches_degradation_trajectory(t0):
    n = 20
    period_s = 60.0
    feats = S.coerce_features(_bare_features(t0, n=n, period_s=period_s))
    onset = t0 + pd.Timedelta(seconds=5 * period_s)
    failure = t0 + pd.Timedelta(seconds=15 * period_s)
    fl = S.coerce_fault_log(
        pd.DataFrame([_fault_row("run0", "door_L1", "friction", onset, failure, gamma=2.0)])
    )
    out = CY.label_from_fault_log(feats, fl)

    te = feats["t_end"]
    u = ((te - onset) / (failure - onset)).clip(lower=0.0, upper=1.0).to_numpy(dtype=np.float64)
    expected_sev = np.clip(u**2.0, 0.0, 1.0)

    np.testing.assert_allclose(out["severity"].to_numpy(dtype=np.float64), expected_sev, atol=1e-5)
    assert bool((out["is_faulty"].to_numpy() == (expected_sev > 0)).all())
    before = (te < onset).to_numpy()
    assert (out.loc[before, "fault_type"].astype(str) == "healthy").all()
    after = ~before
    assert (out.loc[after, "fault_type"].astype(str) == "friction").all()


def test_label_from_fault_log_step_shape(t0):
    n = 10
    period_s = 60.0
    feats = S.coerce_features(_bare_features(t0, n=n, period_s=period_s))
    onset = t0 + pd.Timedelta(seconds=4 * period_s)
    fl = S.coerce_fault_log(
        pd.DataFrame([_fault_row("run0", "door_L1", "limit_switch", onset, onset, shape="step")])
    )
    out = CY.label_from_fault_log(feats, fl)
    expected_faulty = (feats["t_end"] >= onset).to_numpy()
    assert bool((out["is_faulty"].to_numpy() == expected_faulty).all())
    assert bool((out.loc[out["is_faulty"], "severity"] == 1.0).all())


def test_label_from_fault_log_rul_and_alarm_window(t0):
    n = 30
    period_s = 3600.0  # hourly cycles
    feats = S.coerce_features(_bare_features(t0, n=n, period_s=period_s))
    onset = t0 + pd.Timedelta(hours=2)
    failure = t0 + pd.Timedelta(hours=20)
    fl = S.coerce_fault_log(
        pd.DataFrame([_fault_row("run0", "door_L1", "friction", onset, failure, gamma=1.0)])
    )
    alarm_window_s = 5 * 3600.0
    out = CY.label_from_fault_log(feats, fl, alarm_window_s=alarm_window_s)
    faulty = out[out["is_faulty"]]
    rul = faulty["rul_s"].to_numpy(dtype=np.float64)
    assert (rul >= 0.0).all()
    expected_alarm = rul <= alarm_window_s
    np.testing.assert_array_equal(faulty["alarm_window_3d"].to_numpy(), expected_alarm)
    # right at failure time, RUL must be ~0
    assert rul.min() == pytest.approx(0.0, abs=1.0)


def test_label_from_fault_log_step_label_with_unknown_failure_time_has_no_rul(t0):
    """A Cranfield-like static fault log: seeded condition from the first sample, no failure.

    ``nebulax.adapters.cranfield`` writes exactly this row per recording - ``shape="step"``,
    ``t_onset`` = first sample, ``t_failure=NaT`` - because the bench fixture never fails.
    The int64 NaT sentinel must never reach the RUL arithmetic: an unknown failure time is
    ``rul_s=NaN`` and no alarm, not ``rul_s=0`` with a 3-day alarm on every row.
    """
    n = 12
    feats = S.coerce_features(_bare_features(t0, n=n, period_s=30.0))
    fl = S.coerce_fault_log(
        pd.DataFrame(
            [_fault_row("run0", "door_L1", "friction", feats["t_start"].iloc[0], pd.NaT, shape="step")]
        )
    )
    out = CY.label_from_fault_log(feats, fl)

    assert out["is_faulty"].all()
    assert (out["fault_type"].astype(str) == "friction").all()
    assert out["severity"].eq(1.0).all()  # step: severity is closed-form without t_failure
    assert out["rul_s"].isna().all()
    assert not out["alarm_window_3d"].any()
    S.validate_features(out, require_labels=True)


def test_label_from_fault_log_unknown_failure_time_leaves_power_severity_unknown(t0):
    # the power law's u = (t - onset)/(t_failure - onset) has no denominator without a
    # failure time, so severity is NaN - faulty, magnitude unquantified - never a guess.
    feats = S.coerce_features(_bare_features(t0, n=6, period_s=30.0))
    onset = t0 - pd.Timedelta(seconds=1)
    fl = S.coerce_fault_log(
        pd.DataFrame([_fault_row("run0", "door_L1", "friction", onset, pd.NaT, gamma=2.0, shape="power")])
    )
    out = CY.label_from_fault_log(feats, fl)

    assert out["is_faulty"].all()
    assert (out["fault_type"].astype(str) == "friction").all()
    assert out["severity"].isna().all()
    assert out["rul_s"].isna().all()
    assert not out["alarm_window_3d"].any()
    S.validate_features(out, require_labels=True)


def test_label_from_fault_log_ignores_fault_rows_with_unknown_onset(t0):
    feats = S.coerce_features(_bare_features(t0, n=8, period_s=30.0))
    fl = S.coerce_fault_log(
        pd.DataFrame([_fault_row("run0", "door_L1", "friction", pd.NaT, pd.NaT, shape="step")])
    )
    out = CY.label_from_fault_log(feats, fl)
    # an onset-less row has no timeline: it must not "cover" everything from -2**63 ns on
    assert not out["is_faulty"].any()
    assert (out["fault_type"].astype(str) == "healthy").all()
    assert out["rul_s"].isna().all()


def test_label_from_fault_log_known_failure_time_still_counts_down(t0):
    # the NaT guard must not disturb the ordinary run-to-failure case
    feats = S.coerce_features(_bare_features(t0, n=5, period_s=3600.0))
    onset = t0 - pd.Timedelta(hours=1)
    failure = t0 + pd.Timedelta(hours=10)
    fl = S.coerce_fault_log(pd.DataFrame([_fault_row("run0", "door_L1", "friction", onset, failure)]))
    out = CY.label_from_fault_log(feats, fl)
    rul = out["rul_s"].to_numpy(dtype=np.float64)
    assert np.isfinite(rul).all()
    expected = (failure - out["t_end"]).dt.total_seconds().to_numpy()
    np.testing.assert_allclose(rul, expected, rtol=1e-5)


def test_label_from_fault_log_only_labels_matching_component(t0):
    feats_a = _bare_features(t0, n=10, component_id="door_L1")
    feats_b = _bare_features(t0, n=10, component_id="door_L2")
    feats = S.coerce_features(pd.concat([feats_a, feats_b], ignore_index=True))
    onset = t0 + pd.Timedelta(minutes=1)
    failure = t0 + pd.Timedelta(minutes=30)
    fl = S.coerce_fault_log(pd.DataFrame([_fault_row("run0", "door_L1", "friction", onset, failure)]))
    out = CY.label_from_fault_log(feats, fl)
    assert out.loc[out["component_id"].astype(str) == "door_L2", "is_faulty"].eq(False).all()
    assert out.loc[out["component_id"].astype(str) == "door_L1", "is_faulty"].any()


# ---------------------------------------------------------------- peer_normalise


def test_peer_normalise_leave_one_out_matches_hand_computation():
    df = pd.DataFrame(
        {
            "run_id": ["r0"] * 4,
            "car": [1, 1, 1, 1],
            "component_id": ["axlebox_1L", "axlebox_2L", "axlebox_1R", "axlebox_2R"],
            "T_box_mean": [50.0, 60.0, 40.0, 44.0],
        }
    )
    out = CY.peer_normalise(df, group_cols=["run_id", "car"], same_side=True, value_cols=["T_box_mean"])
    # side L peers: 50, 60 -> for row 0, "others" = [60.0] -> delta = 50-60 = -10
    row0 = out.iloc[0]
    assert row0["T_box_mean_peer_delta"] == pytest.approx(-10.0)
    row1 = out.iloc[1]
    assert row1["T_box_mean_peer_delta"] == pytest.approx(10.0)
    # side R peers: 40, 44 -> row 2 other = 44 -> delta = 40-44 = -4
    row2 = out.iloc[2]
    assert row2["T_box_mean_peer_delta"] == pytest.approx(-4.0)


def test_peer_normalise_without_same_side_pools_all_peers():
    df = pd.DataFrame(
        {
            "run_id": ["r0"] * 3,
            "cycle_id": [5, 5, 5],
            "component_id": ["door_L1", "door_L2", "door_R1"],
            "i_peak": [6.0, 6.0, 12.0],
        }
    )
    out = CY.peer_normalise(df, group_cols=["run_id", "cycle_id"], same_side=False, value_cols=["i_peak"])
    # row 2 (i_peak=12): others = [6, 6] -> mean 6 -> delta = 6
    assert out.iloc[2]["i_peak_peer_delta"] == pytest.approx(6.0)
    # rows 0/1 (i_peak=6): others = [6, 12] -> mean 9 -> delta = -3
    assert out.iloc[0]["i_peak_peer_delta"] == pytest.approx(-3.0)


def test_peer_normalise_insufficient_peers_is_nan():
    df = pd.DataFrame(
        {
            "run_id": ["r0", "r1"],
            "car": [1, 1],
            "component_id": ["axlebox_1L", "axlebox_1L"],
            "T_box_mean": [50.0, 60.0],
        }
    )
    out = CY.peer_normalise(df, group_cols=["run_id", "car"], value_cols=["T_box_mean"], min_peers=1)
    # each row is alone in its (run_id, car) group -> zero "other" peers -> NaN
    assert out["T_box_mean_peer_delta"].isna().all()
    assert out["T_box_mean_peer_z"].isna().all()


def test_peer_normalise_default_value_cols_excludes_keys_and_labels():
    df = pd.DataFrame(
        {
            "run_id": ["r0"] * 2,
            "source": ["sim"] * 2,
            "train_id": ["T01"] * 2,
            "car": np.array([1, 1], dtype=np.int8),
            "subsystem": ["bearing"] * 2,
            "component_id": ["axlebox_1L", "axlebox_2L"],
            "cycle_id": np.array([0, 0], dtype=np.int64),
            "t_start": pd.to_datetime(["2026-09-18T09:00:00Z"] * 2),
            "t_end": pd.to_datetime(["2026-09-18T09:05:00Z"] * 2),
            "T_box_mean": [50.0, 60.0],
            "is_faulty": [False, False],
        }
    )
    out = CY.peer_normalise(df, group_cols=["run_id", "car"])
    assert "T_box_mean_peer_delta" in out.columns
    assert "is_faulty_peer_delta" not in out.columns
    assert "cycle_id_peer_delta" not in out.columns
