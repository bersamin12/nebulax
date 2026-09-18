"""Contract tests for nebulax.sim.common: severity laws, the service generator, the sensor model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nebulax import schema as S
from nebulax.sim import common as C

DAY = C.SEC_PER_DAY


# ---------------------------------------------------------------- degradation


@pytest.mark.parametrize("gamma", [1.0, 2.0, 3.0])
def test_power_severity_is_monotone_and_bounded(gamma):
    traj = C.DegradationTrajectory(
        fault_type="friction", subsystem="door", component_id="door_L1",
        t_onset=5 * DAY, t_failure=12 * DAY, gamma=gamma,
    )
    t = np.linspace(0, 30 * DAY, 4001)
    s = traj.severity(t)
    assert s.shape == t.shape
    assert np.all(np.diff(s) >= -1e-12), "severity must be non-decreasing"
    assert s.min() == 0.0 and s.max() == pytest.approx(1.0)
    assert s[t < 5 * DAY].max() == 0.0
    assert s[t >= 12 * DAY].min() == pytest.approx(1.0)
    assert traj.severity(8.5 * DAY) == pytest.approx(0.5**gamma, abs=1e-9)


def test_step_severity_jumps_at_onset():
    traj = C.DegradationTrajectory(
        fault_type="limit_switch", subsystem="door", component_id="door_L3",
        t_onset=7 * DAY, t_failure=7 * DAY, shape="step",
    )
    assert traj.severity(7 * DAY - 1.0) == 0.0
    assert traj.severity(7 * DAY) == 1.0
    assert np.all(np.diff(traj.severity(np.linspace(0, 30 * DAY, 1000))) >= 0.0)


def test_shock_severity_is_a_monotone_staircase(rng):
    shocks = C.ShockProcess().sample(20 * DAY, rng, t_start=2 * DAY)
    traj = C.DegradationTrajectory(
        fault_type="shock_wear", subsystem="door", component_id="door_L1",
        t_onset=2 * DAY, t_failure=20 * DAY, shape="shock", shocks=shocks,
    )
    t = np.linspace(0, 20 * DAY, 3000)
    s = traj.severity(t)
    assert np.all(np.diff(s) >= -1e-12)
    assert s[0] == 0.0 and s[-1] == pytest.approx(1.0)  # many shocks over 18 days saturate
    gaps = np.diff(shocks.times)
    assert gaps.min() >= 3600.0 and gaps.max() <= 7200.0
    assert 0.02 <= shocks.magnitudes.min() <= shocks.magnitudes.max() <= 0.14
    assert shocks.count(2 * DAY, 3 * DAY) >= 1


def test_jitter_perturbs_but_stays_in_range(rng):
    traj = C.DegradationTrajectory(
        fault_type="air_leak", subsystem="pneumatic", component_id="apu_1",
        t_onset=2 * DAY, t_failure=10 * DAY, gamma=2.0,
    )
    t = np.linspace(2 * DAY, 10 * DAY, 500)
    clean = traj.severity(t)
    jittered = traj.severity(t, rng=rng)
    assert np.all((jittered >= 0.0) & (jittered <= 1.0))
    assert not np.allclose(clean, jittered)
    assert np.abs(jittered - clean).mean() < 0.05  # jitter_sigma = 0.05, multiplicative


def test_trajectory_rejects_illegal_combinations():
    with pytest.raises(ValueError, match="fault_type 'air_leak' is not legal"):
        C.DegradationTrajectory(fault_type="air_leak", subsystem="door", component_id="door_L1",
                                t_onset=0.0, t_failure=DAY)
    with pytest.raises(ValueError, match="component_id 'apu_1' is not legal"):
        C.DegradationTrajectory(fault_type="friction", subsystem="door", component_id="apu_1",
                                t_onset=0.0, t_failure=DAY)
    with pytest.raises(ValueError, match="t_failure .* < t_onset"):
        C.DegradationTrajectory(fault_type="friction", subsystem="door", component_id="door_L1",
                                t_onset=DAY, t_failure=0.0)
    with pytest.raises(ValueError, match="requires a sampled ShockSeries"):
        C.DegradationTrajectory(fault_type="shock_wear", subsystem="door", component_id="door_L1",
                                t_onset=0.0, t_failure=DAY, shape="shock")


def test_fault_log_row_matches_schema(t0):
    traj = C.DegradationTrajectory(
        fault_type="bearing_degradation", subsystem="bearing", component_id="axlebox_2R",
        t_onset=3 * DAY, t_failure=9 * DAY, gamma=2.5, params={"mu_gain": 4.0},
    )
    traj.t_functional_failure = 8.5 * DAY
    row = traj.to_fault_log_row(run_id="r0", train_id="T01", car=2, t0=t0)
    assert set(row) == set(S.FAULT_LOG_COLUMNS)
    log = S.coerce_fault_log(pd.DataFrame([row]))
    S.validate_fault_log(log)
    assert log["t_onset"].iloc[0] == t0 + pd.Timedelta(days=3)
    assert log["t_functional_failure"].iloc[0] == t0 + pd.Timedelta(days=8, hours=12)


# ---------------------------------------------------------------- speed profile


def test_speed_profile_is_trapezoidal_to_22_ms():
    v = C.speed_profile(C.RunCycle(duration_s=120.0, v_max=22.0, accel=1.0, decel=1.0), dt=0.5)
    assert v.size == 240
    assert v[0] == 0.0
    assert v.max() == pytest.approx(22.0)
    assert v[-1] < 1.0
    peak = int(np.argmax(v >= 22.0))
    assert np.all(np.diff(v[:peak]) >= 0) and np.all(np.diff(v[peak + 1:]) <= 0)
    assert np.all(v >= 0.0)


def test_short_run_is_triangular_and_never_reaches_v_max():
    cycle = C.RunCycle(duration_s=20.0, v_max=22.0, accel=1.0, decel=1.0)
    assert cycle.v_peak == pytest.approx(10.0)
    v = C.speed_profile(cycle, dt=0.1)
    assert v.max() == pytest.approx(10.0, abs=0.1)


def test_speed_profile_rejects_bad_dt():
    with pytest.raises(ValueError, match="dt must be > 0"):
        C.speed_profile(C.RunCycle(120.0), dt=0.0)


# ---------------------------------------------------------------- ambient / load


def test_ambient_stays_inside_the_singapore_envelope(rng):
    prof = C.AmbientProfile.singapore_routine()
    t = np.linspace(0, 30 * DAY, 60_000)
    temp = prof.temperature(t, rng=rng)
    assert temp.min() >= 24.0 and temp.max() <= 33.0, "routine sweep is 24-33 C [rail_phm 0]"
    assert temp.max() - temp.min() > 5.0, "a constant ambient is forbidden: diurnal shape required"
    stress = C.AmbientProfile.singapore_stress().temperature(t, rng=rng)
    assert stress.min() >= 19.0 and stress.max() <= 37.0


def test_ambient_has_a_diurnal_shape_peaking_mid_afternoon():
    prof = C.AmbientProfile.singapore_routine()
    hours = np.arange(0, 24, 0.25)
    temp = prof.temperature(hours * 3600.0)
    assert 13.0 <= hours[int(np.argmax(temp))] <= 17.0
    assert 1.0 <= hours[int(np.argmin(temp))] <= 5.0


def test_ambient_day_offsets_are_stable_across_chunked_calls(rng):
    prof = C.AmbientProfile.singapore_routine()
    offsets = prof.day_offsets(5, rng)
    t = np.linspace(0, 5 * DAY, 500)
    a = prof.temperature(t, offsets=offsets)
    b = np.concatenate([prof.temperature(chunk, offsets=offsets) for chunk in np.array_split(t, 7)])
    np.testing.assert_allclose(a, b)


def test_passenger_load_has_two_peaks():
    hours = np.arange(0, 24, 0.1)
    load = C.passenger_load(hours)
    assert load.min() >= 0.02 and load.max() <= 1.0
    am = load[(hours > 6) & (hours < 11)].max()
    pm = load[(hours > 16) & (hours < 21)].max()
    midday = load[(hours > 12) & (hours < 14)].max()
    assert am > midday and pm > midday


# ---------------------------------------------------------------- service generator


def test_generate_service_shapes_and_columns(rng):
    svc = C.generate_service(3, rng)
    assert list(svc.segments.columns) == list(C.SEGMENT_COLUMNS)
    assert svc.days == 3 and svc.duration_s == 3 * DAY
    assert set(svc.segments["kind"].astype(str).unique()) == {"run", "dwell", "depot"}
    assert svc.segments["seg_id"].is_monotonic_increasing
    assert svc.segments["t_start"].is_monotonic_increasing
    # segments tile the whole span with no gaps and no overlap
    np.testing.assert_allclose(
        svc.segments["t_start"].to_numpy()[1:], svc.segments["t_end"].to_numpy()[:-1], atol=1e-6
    )
    assert svc.segments["t_start"].iloc[0] == 0.0
    assert svc.segments["t_end"].iloc[-1] == pytest.approx(3 * DAY)
    runs = svc.runs()
    dwells = svc.dwells()
    assert runs["duration_s"].between(90.0, 150.0).all()
    assert dwells["duration_s"].between(25.0, 45.0).all()
    assert len(runs) > 400  # ~19 h of service per day
    assert runs["v_peak"].max() == pytest.approx(22.0)
    assert (svc.segments.loc[~svc.segments["in_service"], "kind"].astype(str) == "depot").all()


def test_service_timeline_is_consistent_with_the_segments(rng):
    svc = C.generate_service(2, rng)
    tl = svc.timeline(dt=1.0)
    assert len(tl) == int(2 * DAY)
    assert tl.speed.shape == tl.load_frac.shape == tl.T_amb.shape == tl.in_service.shape
    assert tl.speed.min() >= 0.0 and tl.speed.max() <= 22.0 + 1e-9
    assert tl.T_amb.min() >= 24.0 and tl.T_amb.max() <= 33.0
    assert np.all(tl.speed[~tl.in_service] == 0.0), "no motion while stabled"
    assert np.all(tl.load_frac[~tl.in_service] == 0.0)
    # 05:30-00:30 service window -> 19 h of 24 in service
    assert 0.74 < tl.in_service.mean() < 0.80
    assert tl.timestamp[0] == svc.t0
    sl = svc.timeline(dt=1.0, t_start=3600.0, t_end=7200.0)
    assert len(sl) == 3600 and sl.t[0] == 3600.0


def test_service_context_long_is_schema_conformant(rng):
    svc = C.generate_service(1, rng, train_id="T07")
    long = svc.context_long(dt=10.0, source="sim", run_id="run0")
    S.validate_long(long)
    assert set(long["signal"].astype("string").unique()) == set(S.CONTEXT_SIGNALS)
    assert (long["component_id"].astype("string") == "train").all()
    assert (long["car"] == 0).all()
    wide = S.to_wide(long, "train")
    assert len(wide) == int(DAY / 10)


def test_generate_service_is_reproducible():
    a = C.generate_service(2, np.random.default_rng(7)).segments
    b = C.generate_service(2, np.random.default_rng(7)).segments
    pd.testing.assert_frame_equal(a, b)


def test_generate_service_rejects_zero_days(rng):
    with pytest.raises(ValueError, match="days must be >= 1"):
        C.generate_service(0, rng)


# ---------------------------------------------------------------- sensor model


def test_apply_sensor_noise_quantisation_and_drift(rng):
    n = 20_000
    x = np.zeros(n)
    spec = C.SensorSpec(noise_sigma=0.5, quantum=0.25, bias=1.0, drift_per_day=2.0)
    y = C.apply_sensor(x, spec, rng, dt=1.0)
    assert y.dtype == np.float32
    assert not np.isnan(y).any()
    # quantised to the 0.25 grid
    np.testing.assert_allclose(np.remainder(y.astype(np.float64), 0.25), 0.0, atol=1e-5)
    assert y[:100].mean() == pytest.approx(1.0, abs=0.2)
    drift = y[-100:].mean() - y[:100].mean()
    assert drift == pytest.approx(2.0 * n / C.SEC_PER_DAY, abs=0.3)
    assert np.std(y) > 0.4


def test_apply_sensor_burst_dropouts_are_bursty(rng):
    n = 100_000
    spec = C.SensorSpec(dropout_burst_prob=1e-3, dropout_len_mean=20.0)
    y = C.apply_sensor(np.ones(n), spec, rng)
    nan = np.isnan(y)
    frac = nan.mean()
    assert 0.005 < frac < 0.20, f"dropout fraction {frac:.4f} out of the expected band"
    # bursts, not isolated samples: mean run length should be near dropout_len_mean
    edges = np.diff(np.concatenate([[0], nan.view(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    assert starts.size == ends.size and starts.size > 20
    assert (ends - starts).mean() > 5.0
    assert np.nanmin(y) == 1.0  # untouched samples keep their value


def test_apply_sensor_is_pure_and_handles_empty(rng):
    x = np.arange(10, dtype=np.float64)
    before = x.copy()
    C.apply_sensor(x, C.SensorSpec(noise_sigma=1.0), rng)
    np.testing.assert_array_equal(x, before)
    assert C.apply_sensor(np.empty(0), C.SensorSpec(), rng).size == 0
    clipped = C.apply_sensor(np.array([-5.0, 5.0]), C.SensorSpec(clip=(0.0, 1.0)), rng)
    np.testing.assert_allclose(clipped, [0.0, 1.0])


# ---------------------------------------------------------------- scenarios


def test_sample_scenarios_respects_p_healthy_and_priors(rng):
    scenarios = C.sample_scenarios(400, "door", rng, p_healthy=0.4, days=30)
    assert len(scenarios) == 400
    healthy = [s for s in scenarios if s.healthy]
    assert 0.33 < len(healthy) / len(scenarios) < 0.47
    assert all(s.faults == () for s in healthy)
    faulty = [s for s in scenarios if not s.healthy]
    types = {f.fault_type for s in faulty for f in s.faults}
    assert types <= set(C.FAULT_PRIORS["door"])
    # electrical/switch class dominates the observed ScotRail distribution [rail_phm 4.1]
    counts = pd.Series([f.fault_type for s in faulty for f in s.faults]).value_counts(normalize=True)
    assert counts.get("limit_switch", 0) + counts.get("dcu_dropout", 0) > 0.3
    assert {s.component_id for s in scenarios} == set(S.DOOR_COMPONENT_IDS)
    for s in faulty:
        for f in s.faults:
            assert 5 * DAY <= f.t_onset <= 18 * DAY
            assert f.subsystem == "door" and f.component_id == s.component_id
            if f.shape == "power":
                assert 3 * DAY <= f.duration_s <= 12 * DAY
                assert 1.0 <= f.gamma <= 3.0


def test_sample_scenarios_is_reproducible_and_validates_args():
    a = C.sample_scenarios(10, "bearing", np.random.default_rng(3))
    b = C.sample_scenarios(10, "bearing", np.random.default_rng(3))
    assert [x.run_id for x in a] == [x.run_id for x in b]
    assert [f.fault_type for x in a for f in x.faults] == [f.fault_type for x in b for f in x.faults]
    with pytest.raises(ValueError, match="unknown subsystem"):
        C.sample_scenarios(5, "brakes", np.random.default_rng(0))
    with pytest.raises(ValueError, match="p_healthy"):
        C.sample_scenarios(5, "door", np.random.default_rng(0), p_healthy=1.5)


def test_scenario_fault_log_rows_validate(t0, rng):
    scenarios = C.sample_scenarios(6, "pneumatic", rng, p_healthy=0.0, days=30)
    rows = [r for s in scenarios for r in s.fault_log_rows(t0)]
    log = S.coerce_fault_log(pd.DataFrame(rows))
    S.validate_fault_log(log)
    assert len(log) == 6


def test_fault_priors_sum_to_one_and_use_legal_types():
    for sub, priors in C.FAULT_PRIORS.items():
        assert sum(priors.values()) == pytest.approx(1.0)
        assert set(priors) <= set(S.FAULT_TYPES[sub])


def test_to_timestamp_round_trip(t0):
    ts = C.to_timestamp(np.array([0.0, 1.5, 86_400.0]), t0)
    assert str(ts.dtype) == "datetime64[ms, UTC]"
    assert ts[1] == t0 + pd.Timedelta(milliseconds=1500)
    assert C.to_timestamp(60.0, t0) == t0 + pd.Timedelta(minutes=1)


def test_thirty_day_service_and_timeline_are_fast(rng):
    import time

    t = time.perf_counter()
    svc = C.generate_service(30, rng)
    tl = svc.timeline(dt=1.0)
    elapsed = time.perf_counter() - t
    assert len(tl) == int(30 * DAY)
    assert elapsed < 10.0, f"30-day service+timeline took {elapsed:.1f}s"
