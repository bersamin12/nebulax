"""Tests for :mod:`nebulax.sim.door` - the bi-parting electric door simulator.

Covers the frozen contract (schema conformance of all three tables, ``store_every``,
bit-reproducibility, the 1-day budget), the physics anchors from the plan (healthy closing
2-3 s, functional failure reachable), and the rail_phm.md corrections (>= 2 s closing
warning, obstruction detection within 0.3 s in the 100-200 N band, ``nff`` labelling).

Everything here runs on one simulated day or less, so the whole file is well inside 60 s.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pandas as pd
import pytest

from nebulax import schema as S
from nebulax.sim import door as D
from nebulax.sim.common import (
    SEC_PER_DAY,
    DegradationTrajectory,
    ServiceParams,
    ShockProcess,
    generate_service,
    sample_scenarios,
)

SHORT = ServiceParams(service_start_h=5.5, service_end_h=8.5)  # ~60 cycles, fast
SHORT_T0 = SHORT.service_start_h * 3600.0
SHORT_SPAN = (SHORT.service_end_h - SHORT.service_start_h) * 3600.0


def _service(days: int = 1, seed: int = 0, params: ServiceParams | None = None):
    return generate_service(days, np.random.default_rng(seed), params=params, train_id="T01")


def _run(faults=None, *, seed: int = 0, days: int = 1, params=None, store_every: int = 10, **kw):
    service = _service(days, seed, params)
    rng = np.random.default_rng(seed + 991)
    return service, D.simulate(
        kw.pop("door_params", D.DoorParams()),
        faults,
        service,
        rng,
        store_every,
        run_id=kw.pop("run_id", "t_run"),
        car=kw.pop("car", 3),
        component_id=kw.pop("component_id", "door_L1"),
        **kw,
    )


def _traj(fault_type: str, *, onset_d: float = 0.20, fail_d: float = 0.95, gamma: float = 1.0, shape="power"):
    """Trajectory in fractions of a simulated day (for the full-day fixtures)."""
    return DegradationTrajectory(
        fault_type=fault_type,
        subsystem="door",
        component_id="door_L1",
        t_onset=onset_d * SEC_PER_DAY,
        t_failure=(onset_d if shape == "step" else fail_d) * SEC_PER_DAY,
        gamma=gamma,
        shape=shape,
    )


def _straj(fault_type: str, *, onset: float = 0.15, fail: float = 0.85, gamma: float = 1.0, shape="power"):
    """Trajectory in fractions of the SHORT service window, so severity really sweeps 0 -> 1."""
    t_on = SHORT_T0 + onset * SHORT_SPAN
    return DegradationTrajectory(
        fault_type=fault_type,
        subsystem="door",
        component_id="door_L1",
        t_onset=t_on,
        t_failure=t_on if shape == "step" else SHORT_T0 + fail * SHORT_SPAN,
        gamma=gamma,
        shape=shape,
    )


_CONST_SPAN = 1.0e7  # s: a ramp so slow that one service window sees a constant severity


def _const(fault_type: str, s: float, *, mid: float = SHORT_T0 + 0.5 * SHORT_SPAN):
    """A trajectory pinned at a constant severity ``s`` over the whole SHORT window.

    Severity regressors are trained over the *whole* [0, 1] axis, so every severity has to
    carry distinguishable telemetry; injecting a ramp only ever samples the axis, and a
    plateau at the top hides a saturation cliff.  This is the injection the acceptance band
    in :func:`test_friction_severity_band_is_usable_end_to_end` needs.
    """
    if s <= 0.0:
        return None
    return [
        DegradationTrajectory(
            fault_type=fault_type,
            subsystem="door",
            component_id="door_L1",
            t_onset=mid - s * _CONST_SPAN,
            t_failure=mid + (1.0 - s) * _CONST_SPAN,
            gamma=1.0,
            jitter_sigma=0.0,
        )
    ]


# --------------------------------------------------------------------------------------
# Module-scoped runs (a simulated day is not cheap; every test reuses these)
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def healthy_day():
    service = _service(1, 0)
    rng = np.random.default_rng(7)
    t0 = time.perf_counter()
    long, feats, events = D.simulate(
        D.DoorParams(), None, service, rng, 10, run_id="h", car=3, component_id="door_L1"
    )
    return service, long, feats, events, time.perf_counter() - t0


@pytest.fixture(scope="module")
def friction_day():
    service = _service(1, 0)
    rng = np.random.default_rng(3)
    traj = _traj("friction", onset_d=0.25, fail_d=0.75, gamma=1.5)
    long, feats, events = D.simulate(
        D.DoorParams(), [traj], service, rng, 10, run_id="f", car=3, component_id="door_L1"
    )
    return service, long, feats, events, traj


# --------------------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------------------


def test_healthy_day_is_schema_conformant(healthy_day):
    service, long, feats, events, _ = healthy_day
    S.validate_long(long)
    S.validate_features(feats, require_labels=True)
    S.validate_events(events)
    # one cycle per dwell, all of them, regardless of store_every
    assert len(feats) == len(service.dwells())
    assert feats["cycle_id"].is_monotonic_increasing
    assert (feats["t_end"] >= feats["t_start"]).all()
    assert set(feats["subsystem"].astype("string")) == {"door"}
    assert set(feats["component_id"].astype("string")) == {"door_L1"}


def test_long_carries_door_signals_and_train_context(healthy_day):
    _, long, _, _, _ = healthy_day
    sub = long["subsystem"].astype("string")
    assert set(sub.unique()) == {"door", "train"}
    door_sigs = set(long.loc[sub == "door", "signal"].astype("string").unique())
    assert door_sigs == set(S.SIGNALS["door"]), door_sigs
    ctx = set(long.loc[sub == "train", "signal"].astype("string").unique())
    assert ctx == set(S.CONTEXT_SIGNALS)
    assert (long.loc[sub == "train", "car"] == 0).all()
    assert np.isfinite(long["value"].to_numpy(dtype=np.float32)[~long["value"].isna()]).all()


def test_one_simulated_day_is_under_ten_seconds(healthy_day):
    *_, secs = healthy_day
    assert secs < 10.0, f"one healthy door-day took {secs:.2f} s"


def test_features_carry_the_planned_columns(healthy_day):
    _, _, feats, _, _ = healthy_day
    required = {
        "closing_time", "opening_time", "i_peak", "i_mean_cruise", "i_rms_cruise", "i_end",
        "energy_J", "pwm_mean", "pos_err_max", "pos_err_rms", "reversal_count", "obstruction",
        "ls_timeout", "dropout_frac", "T_motor", "current_profile_50",
    }
    assert required <= set(feats.columns), sorted(required - set(feats.columns))
    prof = feats["current_profile_50"].iloc[0]
    assert len(prof) == D.DoorParams().profile_n == 50
    assert all(np.isfinite(np.asarray(feats["current_profile_50"].tolist(), dtype=float)).ravel())


def test_simulation_is_bit_reproducible():
    _, (l1, f1, e1) = _run(params=SHORT, seed=11)
    _, (l2, f2, e2) = _run(params=SHORT, seed=11)
    pd.testing.assert_frame_equal(f1.drop(columns=["current_profile_50"]), f2.drop(columns=["current_profile_50"]))
    np.testing.assert_array_equal(
        np.asarray(f1["current_profile_50"].tolist()), np.asarray(f2["current_profile_50"].tolist())
    )
    pd.testing.assert_frame_equal(l1, l2)
    pd.testing.assert_frame_equal(e1, e2)


def test_store_every_thins_waveforms_but_never_features():
    _, (long_a, feat_a, _) = _run(params=SHORT, store_every=1)
    _, (long_b, feat_b, _) = _run(params=SHORT, store_every=20)
    assert len(feat_a) == len(feat_b)
    door_a = long_a[long_a["subsystem"].astype("string") == "door"]
    door_b = long_b[long_b["subsystem"].astype("string") == "door"]
    assert len(door_b) < len(door_a) / 5
    assert len(door_b) > 0


def test_invalid_arguments_are_rejected():
    service = _service(1, 0, SHORT)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="component_id"):
        D.simulate(D.DoorParams(), None, service, rng, component_id="apu_1")
    with pytest.raises(ValueError, match="car"):
        D.simulate(D.DoorParams(), None, service, rng, car=99, component_id="door_L1")
    with pytest.raises(ValueError, match="store_every"):
        D.simulate(D.DoorParams(), None, service, rng, 0, component_id="door_L1")


# --------------------------------------------------------------------------------------
# Healthy physics anchors
# --------------------------------------------------------------------------------------


def test_healthy_closing_time_is_two_to_three_seconds(healthy_day):
    _, _, feats, _, _ = healthy_day
    clean = feats[feats["reversal_count"] == 0]
    assert len(clean) > 0.9 * len(feats)
    assert 2.0 <= float(clean["closing_time"].median()) <= 3.0
    assert 2.0 <= float(clean["opening_time"].median()) <= 3.0
    # the plant is not deterministic: per-cycle scatter must show up
    assert float(clean["closing_time"].std()) > 0.0
    assert float(clean["i_rms_cruise"].std()) > 0.0


def test_healthy_currents_stay_below_the_obstruction_trip(healthy_day):
    p = D.DoorParams()
    _, _, feats, _, _ = healthy_day
    clean = feats[feats["reversal_count"] == 0]
    assert float(clean["i_rms_cruise"].max()) < p.i_obs_a
    assert float(clean["i_peak"].max()) <= p.i_limit_peak_a + 1e-6
    # the force band the detector lives in is the 100-200 N one from rail_phm 1.4
    assert 100.0 <= p.f_obs_trip_n <= p.f_limit_n <= 200.0


def test_healthy_door_raises_no_limit_switch_timeouts(healthy_day):
    _, _, feats, events, _ = healthy_day
    assert float(feats["ls_timeout"].sum()) == 0.0
    assert not (events["event"].astype("string") == "functional_failure").any()
    assert float(feats["obstruction"].mean()) < 0.10


def test_healthy_labels_are_healthy_or_nff(healthy_day):
    _, _, feats, _, _ = healthy_day
    types = set(feats["fault_type"].astype("string").unique())
    assert types <= {"healthy", "nff"}
    assert float(feats["severity"].max()) == 0.0
    assert not feats["is_faulty"].any()
    # rail_phm 1.3: an alarm with nothing injected is the no-fault-found class
    alarmed = (feats["obstruction"] > 0) | (feats["ls_timeout"] > 0)
    assert (feats.loc[alarmed, "fault_type"].astype("string") == "nff").all()


def test_closing_warning_precedes_movement_by_two_seconds(healthy_day):
    """PRM TSI 4.2.2.3.2 [R165]: the closing signal starts >= 2 s before the door moves."""
    _, long, feats, _, _ = healthy_day
    assert bool((feats["warning_time"] >= 2.0).all())
    wide = S.to_wide(long[long["subsystem"].astype("string") == "door"], "door")
    stored = feats[(feats["cycle_id"] % 10 == 0) & (feats["reversal_count"] == 0)]
    row = stored.iloc[len(stored) // 2]
    seg = wide[(wide["timestamp"] >= row["t_start"]) & (wide["timestamp"] <= row["t_end"])]
    seg = seg[seg["door_key"].notna() & seg["vel"].notna()].reset_index(drop=True)
    key = seg["door_key"].to_numpy(dtype=float)
    vel = seg["vel"].to_numpy(dtype=float)
    ts = seg["timestamp"].to_numpy()
    release_drop = np.flatnonzero(np.diff(key) < -0.5)
    closing_start = np.flatnonzero(vel < -0.02)
    assert release_drop.size and closing_start.size
    gap = (ts[closing_start[0]] - ts[release_drop[0]]) / np.timedelta64(1, "s")
    assert gap >= 2.0 - 1e-6, f"only {gap:.2f} s of closing warning"


# --------------------------------------------------------------------------------------
# Faults
# --------------------------------------------------------------------------------------


def test_friction_ramp_is_monotone_and_reaches_functional_failure(friction_day):
    _, _, feats, events, traj = friction_day
    live = feats[(feats["severity"] > 0) & (feats["rul_s"] > 0)]
    assert len(live) > 20
    rho = live["severity"].corr(live["i_rms_cruise"], method="spearman")
    assert rho > 0.8, f"cruise current is not monotone in severity (rho={rho:.3f})"
    healthy_med = float(feats.loc[feats["severity"] == 0, "i_rms_cruise"].median())
    assert float(live["i_rms_cruise"].max()) > 1.8 * healthy_med

    assert traj.t_functional_failure is not None
    assert 0.0 < traj.t_functional_failure <= 1.0 * SEC_PER_DAY
    ff = events[events["event"].astype("string") == "functional_failure"]
    assert len(ff) == 1
    assert (feats["fault_type"].astype("string") == "friction").any()
    assert feats.loc[feats["severity"] > 0, "is_faulty"].all()
    assert float(feats["rul_s"].min()) >= 0.0
    assert feats["alarm_window_3d"].any()


@pytest.fixture(scope="module")
def friction_band():
    """Friction held at a constant s = 0 / 0.3 / 1 for a whole SHORT window each."""
    return {
        sev: _run(_const("friction", sev), params=SHORT, seed=0)[1][1] for sev in (0.0, 0.3, 1.0)
    }


def test_friction_severity_band_is_usable_end_to_end(friction_band):
    """The whole severity axis has to carry telemetry, not just its bottom.

    A ramp fixture only samples the axis and hides both failure modes a friction map can
    have: a saturation cliff (every cycle above some s looks identical) and a seizure (the
    door stops moving, so the features collapse to zeros).  This asserts the acceptance band
    directly, at constant severity: plan "Door" fault map ``F_c = F_c0 (1 + 2 s)``,
    ``b = b0 (1 + 1.5 s)`` against the force ceiling of rail_phm 1.4 [R67][R85].
    """
    p = D.DoorParams()
    healthy, mid, full = friction_band[0.0], friction_band[0.3], friction_band[1.0]
    h_close = float(healthy["closing_time"].median())
    h_cur = float(healthy["i_mean_cruise"].median())
    h_rev = float((healthy["reversal_count"] > 0).mean())

    # s = 1: a badly worn door, but still a door - finite features, the leaf still opens,
    # and both headline features are well clear of healthy.
    cols = ["closing_time", "opening_time", "i_mean_cruise", "i_rms_cruise", "energy_J"]
    assert np.isfinite(full[cols].to_numpy(dtype=float)).all()
    assert float(full["closing_time"].median()) >= 1.6 * h_close
    assert float(full["i_mean_cruise"].median()) >= 1.4 * h_cur
    assert float(full["pos_open_max"].median()) > 0.95 * p.stroke_m
    assert float(full["pos_close_max"].median()) > 0.95 * p.stroke_m

    # s = 0.3: still in service - under the functional-failure limit, current already
    # visibly up, and no phantom obstructions.  The obstruction detector answers an abrupt
    # rise over the cycle's own baseline [R65], so gradual wear must not read as an
    # obstacle at any severity - otherwise "friction raises the current" and "no phantom
    # reversal" are mutually exclusive.
    assert float(mid["closing_time"].median()) < p.closing_time_limit_s
    assert float(mid["i_mean_cruise"].median()) > 1.2 * h_cur
    assert float((mid["reversal_count"] > 0).mean()) <= h_rev + 0.05
    assert float((full["reversal_count"] > 0).mean()) <= h_rev + 0.05

    # no dead zone in between: the current axis is strictly ordered
    assert h_cur < float(mid["i_mean_cruise"].median()) < float(full["i_mean_cruise"].median())


def test_fault_log_row_carries_the_functional_failure(friction_day):
    service, _, _, _, traj = friction_day
    row = traj.to_fault_log_row(run_id="f", train_id="T01", car=3, t0=service.t0)
    log = S.coerce_fault_log(pd.DataFrame([row]))
    S.validate_fault_log(log)
    assert pd.notna(log["t_functional_failure"].iloc[0])
    assert log["t_functional_failure"].iloc[0] <= log["t_failure"].iloc[0]


def test_obstruction_is_detected_within_300_ms():
    """rail_phm 1.4 [R67][R85]: 100-200 N operating point, detection inside 0.3 s.

    A design assumption reported as OUR acceptance metric, never as an EN 14752 requirement.
    """
    _, (_, feats, events) = _run([_straj("obstruction", onset=0.0, fail=0.4)], params=SHORT, seed=5)
    obs = feats[feats["obstruction"] > 0]
    assert len(obs) > 0
    lat = obs["obs_detect_s"].to_numpy(dtype=float)
    lat = lat[np.isfinite(lat)]
    assert lat.size > 0
    assert lat.max() <= D.DoorParams().obs_detect_deadline_s
    kinds = events["event"].astype("string")
    assert (kinds == "obstruction").any() and (kinds == "reversal").any()
    assert float(obs["reversal_count"].max()) >= 1.0
    assert float(obs["closing_time"].max()) > float(feats["closing_time"].median())


def test_limit_switch_step_fault_times_the_door_out():
    _, (_, feats, events) = _run([_straj("limit_switch", onset=0.15, shape="step")], params=SHORT, seed=2)
    after = feats[feats["severity"] > 0]
    assert len(after) > 5
    assert float(after["ls_timeout"].mean()) > 0.9
    kinds = events["event"].astype("string")
    assert (kinds == "ls_timeout").any()
    assert (kinds == "door_fault").any()
    assert (kinds == "functional_failure").any()


def test_dcu_dropout_puts_nan_into_every_channel():
    _, (long, feats, events) = _run(
        [_straj("dcu_dropout", onset=0.0, fail=0.2)], params=SHORT, seed=4, store_every=1
    )
    assert float(feats["dropout_frac"].max()) > 0.05
    door = long[long["subsystem"].astype("string") == "door"]
    nan_by_signal = door.groupby("signal", observed=True)["value"].apply(lambda s: s.isna().mean())
    assert (nan_by_signal > 0).all(), nan_by_signal.to_dict()
    assert (events["event"].astype("string") == "dropout").any()


def test_brush_wear_raises_current_and_drops_the_current_channel():
    """R = R0(1 + 0.8 s) and k_t, k_e x (1 - 0.2 s): more amps for the same force, a lower
    back-EMF, and commutation dropouts on the current channel only."""
    _, (long, feats, _) = _run(
        [_straj("brush_wear", onset=0.25, fail=0.65)], params=SHORT, seed=6, store_every=1
    )
    early = feats[feats["severity"] < 0.05]
    late = feats[feats["severity"] > 0.8]
    assert len(early) > 3 and len(late) > 3
    assert float(late["i_rms_cruise"].median()) > 1.1 * float(early["i_rms_cruise"].median())
    # back-EMF falls with k_e, so the duty needed at cruise falls even though amps rise
    assert abs(float(late["pwm_mean"].median())) < abs(float(early["pwm_mean"].median()))
    door = long[long["subsystem"].astype("string") == "door"]
    nan_frac = door.groupby("signal", observed=True)["value"].apply(lambda s: s.isna().mean())
    assert nan_frac["current"] > nan_frac["pos"]


def test_backlash_widens_the_tracking_error():
    _, (_, feats, _) = _run([_straj("backlash", onset=0.25, fail=0.65)], params=SHORT, seed=8)
    early = feats[feats["severity"] < 0.05]
    late = feats[feats["severity"] > 0.8]
    assert len(early) > 3 and len(late) > 3
    assert float(late["pos_err_rms"].median()) > float(early["pos_err_rms"].median())


def test_shock_wear_eats_the_closing_position_ceiling():
    """rail_phm 4.1 [R69][R71]: shocks walk the closing-position ceiling down."""
    shocks = ShockProcess(inter_arrival_s=(600.0, 1200.0)).sample(
        SEC_PER_DAY, np.random.default_rng(5), t_start=SHORT_T0
    )
    traj = DegradationTrajectory(
        fault_type="shock_wear",
        subsystem="door",
        component_id="door_L1",
        t_onset=SHORT_T0,
        t_failure=SEC_PER_DAY,
        shape="shock",
        shocks=shocks,
    )
    _, (_, feats, events) = _run([traj], params=SHORT, seed=9)
    assert float(feats["meta_shock_count"].sum()) > 0
    assert (events["event"].astype("string") == "shock").any()
    late = feats[feats["severity"] > 0.3]
    if len(late):
        assert float(late["pos_close_max"].min()) < D.DoorParams().stroke_m


def test_shock_count_is_metadata_and_the_estimator_is_observable():
    """The leakage fix: the latent shock process rides along as ``meta_shock_count`` (useful for
    analysis, invisible to the model) and the per-cycle statistics carry ``shock_jump_count_est``,
    which a real DCU could compute - correlated with the truth, never equal to it."""
    shocks = ShockProcess().sample(SEC_PER_DAY, np.random.default_rng(7))
    traj = DegradationTrajectory(
        fault_type="shock_wear",
        subsystem="door",
        component_id="door_L1",
        t_onset=0.0,
        t_failure=SEC_PER_DAY,
        shape="shock",
        shocks=shocks,
    )
    _, (_, feats, _) = _run([traj], seed=7)

    x_cols = S.feature_columns(feats)
    assert "meta_shock_count" in feats.columns  # kept, for analysis
    assert "meta_shock_count" not in x_cols  # but never a model input
    assert "shock_jump_count_est" in x_cols
    assert not any(c.startswith(S.METADATA_PREFIX) for c in x_cols)
    assert not ({*S.FEATURE_KEY_COLUMNS, *S.LABEL_COLUMNS} & set(x_cols))

    truth = feats["meta_shock_count"].to_numpy(dtype=float).cumsum()
    est = feats["shock_jump_count_est"].to_numpy(dtype=float)
    assert truth[-1] >= 8  # the fixture really does shock the door
    assert float(np.corrcoef(truth, est)[0, 1]) > 0.8  # it tracks the damage
    assert not np.array_equal(truth, est)  # but it is an estimate, not the process
    assert 0.0 < est[-1] < truth[-1]  # and it undercounts: jams blur later steps
    assert (np.diff(est) >= 0.0).all()  # a count, so monotone


def test_shock_estimator_is_quiet_without_shocks(healthy_day):
    """A step detector that fires on a healthy door would be noise, not a feature."""
    _, _, feats, _, _ = healthy_day
    assert float(feats["meta_shock_count"].sum()) == 0.0
    assert float(feats["shock_jump_count_est"].to_numpy()[-1]) <= 2.0  # out of ~430 cycles


def test_estimate_shock_jumps_finds_a_step_and_ignores_a_spike():
    """The estimator's own contract: a permanent step counts once, a one-cycle passenger
    obstruction does not count at all."""
    base = np.full(300, 0.72) + np.random.default_rng(3).normal(0.0, 1e-4, 300)
    stepped = base.copy()
    stepped[150:] -= 0.05
    assert float(D.estimate_shock_jumps((stepped,)).sum()) == 1.0
    spiky = base.copy()
    spiky[150] -= 0.05
    assert float(D.estimate_shock_jumps((spiky,)).sum()) == 0.0
    assert D.estimate_shock_jumps((np.zeros(0),)).size == 0
    with pytest.raises(ValueError, match="share one length"):
        D.estimate_shock_jumps((base, base[:10]))


# --------------------------------------------------------------------------------------
# Faults container and params
# --------------------------------------------------------------------------------------


def test_doorfaults_maps_trajectories_onto_slots():
    f = D.DoorFaults.from_trajectories([_traj("friction"), _traj("limit_switch", shape="step")])
    assert f.friction is not None and f.limit_switch is not None
    assert f.brush_wear is None
    assert len(f.active()) == 2
    assert not f.healthy
    assert f.component_id() == "door_L1"
    assert D.DoorFaults.from_trajectories(None).healthy
    assert D.DoorFaults.from_trajectories(f) is f


def test_doorfaults_rejects_foreign_and_duplicate_trajectories():
    bad = DegradationTrajectory("air_leak", "pneumatic", "apu_1", 0.0, 10.0)
    with pytest.raises(ValueError, match="subsystem"):
        D.DoorFaults.from_trajectories([bad])
    with pytest.raises(ValueError, match="two trajectories"):
        D.DoorFaults.from_trajectories([_traj("friction"), _traj("friction")])
    mixed = DegradationTrajectory("friction", "door", "door_R3", 0.0, 10.0)
    with pytest.raises(ValueError, match="one door"):
        D.DoorFaults.from_trajectories([_traj("backlash"), mixed]).component_id()


def test_doorparams_derived_quantities_and_cranfield_variant():
    p = D.DoorParams()
    assert p.gear_gain == pytest.approx(20.0 / 0.03)
    assert p.force_per_amp == pytest.approx(0.75 * 0.35 * p.gear_gain)
    assert p.m_eff == pytest.approx(2 * 40.0 + 3e-5 * p.gear_gain**2)
    assert p.i_limit_a == pytest.approx(p.f_limit_n / p.force_per_amp)
    assert p.i_limit_peak_a > p.i_limit_a  # inertia + seal feed-forward
    assert p.v_bus == 110.0 and p.stroke_m == 0.725

    c = D.DoorParams.cranfield()
    assert c.v_bus == 24.0
    assert c.stroke_m != p.stroke_m and c.n_leaves == 1
    assert p.with_(v_bus=48.0).v_bus == 48.0

    _, (long, feats, events) = _run(params=SHORT, seed=1, door_params=c, component_id="door_R2")
    S.validate_long(long)
    S.validate_features(feats, require_labels=True)
    S.validate_events(events)
    assert float(feats["closing_time"].median()) > 0.5
    assert np.isfinite(feats["i_rms_cruise"].to_numpy(dtype=float)).all()


def test_simulate_scenario_uses_the_scenario_identity():
    rng = np.random.default_rng(12)
    scen = [s for s in sample_scenarios(6, "door", rng, p_healthy=0.0, days=1, components=["door_L2"])][0]
    service = _service(1, 0, SHORT)
    long, feats, events = D.simulate_scenario(scen, service, np.random.default_rng(scen.seed))
    S.validate_features(feats, require_labels=True)
    assert set(feats["run_id"].astype("string")) == {scen.run_id}
    assert set(feats["component_id"].astype("string")) == {scen.component_id}
    assert int(feats["car"].iloc[0]) == scen.car
    rows = scen.fault_log_rows(service.t0)
    S.validate_fault_log(S.coerce_fault_log(pd.DataFrame(rows)))

    with pytest.raises(ValueError, match="subsystem"):
        D.simulate_scenario(dataclasses.replace(scen, subsystem="bearing"), service, rng)


# --------------------------------------------------------------------------------------
# Calibration against the Cranfield rig (scripts/calibrate_door.py)
# --------------------------------------------------------------------------------------


def test_obs_base_seed_kills_the_phantom_reversal_on_the_cranfield_variant():
    """A zero-seeded detector baseline makes a door's own cruise current look like an obstacle.

    The asymmetric EWMA climbs towards an *alarming* change with ``obs_base_tau_s`` (0.40 s),
    so from zero it needs ``tau . ln(i_cruise / i_obs_delta_a)`` to reach the operating point -
    0.42 s on ``DoorParams.cranfield()`` - while the current path arms once the acceleration
    ramp is over, 0.25 s in.  Every cycle whose cruise current cleared ``f_obs_trip_n``
    therefore phantom-reversed: **100 %** of them at friction ``s >= 0.5`` against a 1.4 %
    healthy baseline, which also made ``closing_time`` non-monotone in severity (6.32 s at
    s = 0.5 against 6.30 s at s = 1.0) and so useless as a degradation feature.
    """
    c = D.DoorParams.cranfield()
    rates, closing = {}, {}
    for seed_on in (False, True):
        p = c.with_(obs_base_seed=seed_on)
        for s in (0.0, 0.5, 1.0):
            feats = _run(_const("friction", s), params=SHORT, seed=0, door_params=p)[1][1]
            rates[(seed_on, s)] = float((feats["reversal_count"] > 0).mean())
            closing[(seed_on, s)] = float(np.median(feats["closing_time"]))

    # the defect, pinned so it cannot come back unnoticed
    assert rates[(False, 0.5)] > 0.9 and rates[(False, 1.0)] > 0.9
    assert closing[(False, 0.5)] >= closing[(False, 1.0)]  # non-monotone in severity
    # the fix: at s = 0.5 the phantom rate is back at the healthy baseline ...
    assert rates[(True, 0.5)] <= rates[(True, 0.0)] + 0.05
    # ... and at s = 1.0 what survives is a real minority, not every cycle
    assert rates[(True, 1.0)] < 0.25
    assert rates[(True, 1.0)] < 0.5 * rates[(False, 1.0)]
    # closing time is monotone in severity again, which it was not before
    assert closing[(True, 0.0)] < closing[(True, 0.5)] < closing[(True, 1.0)]
    # and the current channel is untouched by the detector change
    assert closing[(True, 1.0)] < closing[(False, 1.0)]


def test_cranfield_variant_scales_its_thresholds_to_its_own_stroke():
    """Inherited absolute thresholds are meaningless on a stroke 7x shorter than the door's."""
    c = D.DoorParams.cranfield()
    d = D.DoorParams()
    # spalling ripple must fit inside the stroke many times over, at the screw lead
    assert c.mis_lambda_m == D.CRANFIELD_SCREW_LEAD_M
    assert c.stroke_m / c.mis_lambda_m >= 10.0
    # ... and the lead is the same number that sets the rotational-to-linear gain
    assert c.pulley_r_m * 2.0 * np.pi == pytest.approx(D.CRANFIELD_SCREW_LEAD_M)
    # the tracking-error trip keeps the door's stroke fraction, not its absolute 20 mm
    assert c.pos_err_obs_m == pytest.approx(D.OBS_POS_ERR_STROKE_FRAC * c.stroke_m)
    assert c.pos_err_obs_m / c.stroke_m == pytest.approx(d.pos_err_obs_m / d.stroke_m, rel=1e-6)
    assert c.pos_err_obs_m < 0.25 * d.pos_err_obs_m


def test_the_shipped_door_is_barely_touched_by_the_cranfield_calibration():
    """The constants were fitted on a rig; the 110 V door they were not fitted on must not move.

    Seeding the detector baseline is a no-op on the shipped door because its 0.50 s
    acceleration ramp already outlasts the 0.34 s the unseeded baseline needs to climb - a
    0.16 s accident this removes.  Measured over 69 cycles x 3 severities: **0 cycles differ**
    at s = 0 and s = 0.5, **1 cycle** differs at s = 1.0, and every aggregate is identical.
    """
    p = D.DoorParams()
    assert p.obs_base_seed is True
    assert p.mis_lambda_m == 0.30 and p.pos_err_obs_m == 0.020
    assert (p.k_friction_c, p.k_friction_b, p.backlash_m, p.k_mis) == (2.0, 1.5, 0.008, 3.0)
    for s in (0.0, 1.0):
        a = _run(_const("friction", s), params=SHORT, seed=0, door_params=p)[1][1]
        b = _run(_const("friction", s), params=SHORT, seed=0, door_params=p.with_(obs_base_seed=False))[1][1]
        for col in ("closing_time", "i_mean_cruise", "reversal_count"):
            av, bv = a[col].to_numpy(float), b[col].to_numpy(float)
            differing = int(np.sum(~np.isclose(av, bv, equal_nan=True)))
            assert differing <= 1, f"{col}: {differing} of {len(av)} cycles changed"
            assert np.nanmedian(av) == pytest.approx(np.nanmedian(bv), rel=1e-6), col
