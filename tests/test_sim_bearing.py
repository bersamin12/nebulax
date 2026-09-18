"""Contract + physics tests for nebulax.sim.bearing.

The physics assertions are deliberately tied to the numbers in ``docs/research/rail_phm.md``
section 4.3, because that document (not the plan) is the authority on the constants:

* a healthy box rises **20-26 K** above ambient at 80 km/h, scaling as ``v**0.43``;
* a defect moves ``T_box`` by **~1 K per severity step**, buried in an **8.4 K** healthy
  inter-box spread - the axle box is ~15x less sensitive than the rings;
* kurtosis **rises then collapses** (5.56 -> 21.69 -> 8.06 -> 3.29) instead of rising
  monotonically, and the healthy crest intercept is **4.5**, not 3;
* functional failure follows the published Netherlands Railways rules (same-side differential
  levels, 80 C absolute level 4), not the withdrawn 90 C / 30 K pair.

Every simulated run here is one day, which keeps the module well under the 60 s budget.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from nebulax import schema as S
from nebulax.sim import bearing as B
from nebulax.sim.common import (
    SEC_PER_DAY,
    SEGMENT_COLUMNS,
    AmbientProfile,
    DegradationTrajectory,
    SensorSpec,
    Service,
    ServiceParams,
    generate_service,
)

DAY = SEC_PER_DAY
FAST = 10**9  # store_every large enough that only one telemetry window is materialised


# ---------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def params() -> B.BearingParams:
    return B.BearingParams()


@pytest.fixture(scope="module")
def service():
    return generate_service(1, np.random.default_rng(11), AmbientProfile.singapore_routine())


def _run(service, faults=None, *, seed: int = 5, params=None, store_every: int = FAST, car: int = 3):
    p = params or B.BearingParams()
    return B.simulate(
        p,
        B.BearingFaults.from_trajectories(faults or ()),
        service,
        np.random.default_rng(seed),
        store_every,
        run_id="t0",
        car=car,
    )


@pytest.fixture(scope="module")
def healthy(service):
    return _run(service)


def _box(feat: pd.DataFrame, box: str) -> pd.DataFrame:
    return feat[feat["component_id"] == box]


# ---------------------------------------------------------------------------- geometry


def test_box_ids_match_the_registry():
    assert B.BOX_IDS == S.AXLEBOX_COMPONENT_IDS
    assert len(B.BOX_IDS) == 8


@pytest.mark.parametrize("box", B.BOX_IDS)
def test_peer_group_is_the_same_side_of_the_train(box):
    """[R151]: dT is measured against the median of the peers on the SAME SIDE, never pooled."""
    peers = B.same_side_boxes(box)
    assert len(peers) == 3
    assert all(B.side_of(p) == B.side_of(box) for p in peers)
    assert box not in peers
    opp = B.opposite_box(box)
    assert B.axle_of(opp) == B.axle_of(box) and B.side_of(opp) != B.side_of(box)


def test_motor_car_leading_bearing_is_loaded_harder(params):
    """[R150]: +13.5 % on the leading bearing of a motor-car wheelset; trailer cars uniform."""
    motor = params.position_gain(3)
    trailer = params.position_gain(1)
    assert np.allclose(trailer, 1.0)
    lead = np.isin([B.axle_of(b) for b in B.BOX_IDS], (1, 3))
    assert np.allclose(motor[lead], 1.0 + params.motor_lead_load_gain)
    assert np.allclose(motor[~lead], 1.0)


# ---------------------------------------------------------------------------- thermal law


def test_steady_rise_lands_in_the_derived_80kmh_band(params):
    """rail_phm 4.3.1 derives 20-26 K at 80 km/h from [R149] Table 1 + Eq. 25."""
    rise = params.steady_rise(22.2, 0.5)
    assert 20.0 <= rise <= 26.0, f"steady rise {rise:.1f} K outside the derived 20-26 K band"


def test_convection_follows_the_published_v057_correlation(params):
    """[R149] Eq. 25: ``h_a ~ v^0.57`` above the natural-convection floor."""
    v = np.array([5.0, 10.0, 20.0])
    forced = params.hA(v) - params.hA_0_w_per_k
    assert np.allclose(forced / forced[0], (v / v[0]) ** 0.57)
    assert params.hA(0.0) == pytest.approx(params.hA_0_w_per_k)


def test_implemented_speed_exponent_is_near_the_derived_043(params):
    """``dT ~ v^0.43`` is the ideal; the natural-convection floor lifts it to ~0.5. We assert
    the implemented value rather than pretending the floor is not there."""
    e = params.speed_exponent(8.0, 22.2)
    assert 0.40 <= e <= 0.65, f"implemented speed exponent {e:.3f} far from the derived 0.43"


def test_friction_power_grows_with_speed_and_load(params):
    p_slow = params.friction_power_w(8.0, 0.5, params.mu_0)
    p_fast = params.friction_power_w(22.0, 0.5, params.mu_0)
    p_heavy = params.friction_power_w(22.0, 1.0, params.mu_0)
    assert 0 < p_slow < p_fast < p_heavy
    assert params.friction_power_w(0.0, 0.5, params.mu_0) == 0.0


def test_severity_gain_is_15x_smaller_than_the_hot_box_gain(params):
    """The W1 correction: ordinary defect growth barely warms the box; only hot_axle_box does."""
    assert params.mu_gain_hot_box > 15.0 * params.mu_gain_degradation
    assert params.dT_per_severity_k <= 3.0


def test_first_order_scan_matches_a_naive_loop(rng):
    n, k = 2000, 3
    alpha = rng.uniform(0.95, 0.999, (n, k))
    beta = rng.normal(0.0, 0.05, (n, k))
    T0 = rng.normal(30.0, 1.0, k)
    got = B._first_order_scan(alpha, beta, T0)
    want = np.empty_like(beta)
    T = T0.copy()
    for i in range(n):
        T = alpha[i] * T + beta[i]
        want[i] = T
    assert np.allclose(got, want, atol=1e-9)


# ---------------------------------------------------------------------------- vibration law


def test_kurtosis_rises_then_collapses(params):
    """W1 CORRECTION. ``3 + 6 s^2`` is falsified; the measured inner-race sequence is
    5.56 -> 21.69 -> 8.06 -> 3.29 as the defect widens [R157]."""
    s = np.array([0.15, 0.35, 0.6, 1.0])
    kurt = params.vib_kurt_healthy + params.vib_kurt_gain * B.severity_bump(
        s, s_peak=params.kurt_s_peak, w_lo=params.kurt_w_lo, w_hi=params.kurt_w_hi
    )
    assert kurt[1] > kurt[0] and kurt[1] > kurt[2] > kurt[3], "kurtosis must rise then collapse"
    assert np.allclose(kurt, [5.56, 21.69, 8.06, 3.29], atol=2.5)
    assert kurt[3] == pytest.approx(params.vib_kurt_healthy, abs=1.0), "s=1 returns to healthy"


def test_severity_bump_is_bounded_and_zero_at_zero():
    s = np.linspace(0.0, 1.0, 501)
    b = B.severity_bump(s)
    assert b[0] == 0.0
    assert 0.0 <= b.min() <= b.max() <= 1.0
    assert s[int(np.argmax(b))] == pytest.approx(0.35, abs=0.02)


def test_healthy_vibration_intercepts_are_the_measured_ones(params):
    """[R157]: healthy kurtosis 2.76-2.96 and healthy crest 4.22-5.59 (the plan's 3 is falsified)."""
    assert 2.7 <= params.vib_kurt_healthy <= 3.1
    assert 4.2 <= params.vib_crest_healthy <= 5.6


def test_healthy_rms_floor_is_the_right_order_of_magnitude(params):
    """[R150] anchor: 8.8 m/s^2 RMS at 300 km/h; scaled by v^2 that is ~0.6 m/s^2 at 22 m/s."""
    assert 0.3 <= params.vib_rms_healthy <= 1.2
    assert params.vib_speed_exp_healthy > params.vib_speed_exp_defect


# ---------------------------------------------------------------------------- schema contract


def test_healthy_run_is_schema_conformant(healthy):
    long, feat, ev = healthy
    S.validate_long(long, strict=True)
    S.validate_features(feat, require_labels=True)
    S.validate_events(ev, strict=True)
    assert list(long.columns) == list(S.LONG_COLUMNS)
    assert list(feat.columns[: len(S.FEATURE_KEY_COLUMNS)]) == list(S.FEATURE_KEY_COLUMNS)


def test_long_carries_the_five_box_signals_plus_context(healthy):
    long, _, _ = healthy
    bearing = long[long["subsystem"] == "bearing"]
    sigs = set(bearing["signal"].astype(str))
    assert set(B.BEARING_SIGNALS) <= sigs <= set(S.SIGNALS["bearing"])
    assert set(bearing["component_id"].astype(str)) == set(B.BOX_IDS)
    ctx = long[long["subsystem"] == "train"]
    assert set(ctx["signal"].astype(str)) == set(S.CONTEXT_SIGNALS)
    assert (ctx["car"] == 0).all()


def test_wide_view_joins_context_without_resampling(healthy):
    long, _, _ = healthy
    wide = S.to_wide(long, "bearing", signals=["T_box"], include_context=True)
    assert {"T_box", "speed", "T_amb", "load_frac"} <= set(wide.columns)
    assert wide["T_amb"].notna().any()


def test_feature_table_has_every_documented_window_feature(healthy):
    _, feat, _ = healthy
    required = {
        "T_box_mean", "T_box_max", "dT_peer_same_side", "dT_opposite", "T_amb",
        "v_mean", "v_max", "load_frac", "dwell_fraction", "dwell_fraction_last_hour",
        "thermal_residual", "vib_rms_mean", "vib_kurt_mean", "vib_crest_mean", "vib_bpfo_mean",
    }
    assert required <= set(feat.columns)
    assert len(feat) == feat["cycle_id"].nunique() * 8
    assert (feat["t_end"] - feat["t_start"]).dt.total_seconds().unique().tolist() == [300.0]


def test_labels_are_healthy_and_finite_on_a_healthy_run(healthy):
    _, feat, _ = healthy
    assert (feat["fault_type"].astype(str) == "healthy").all()
    assert not feat["is_faulty"].any()
    assert not feat["alarm_window_3d"].any()
    assert (feat["severity"] == 0).all()
    assert feat["rul_s"].isna().all()


def test_run_is_bit_reproducible(service):
    a = _run(service, seed=17)[1]
    b = _run(service, seed=17)[1]
    pd.testing.assert_frame_equal(a, b)
    c = _run(service, seed=18)[1]
    assert not np.allclose(a["T_box_mean"], c["T_box_mean"])


# ---------------------------------------------------------------------------- healthy physics


def test_healthy_rise_and_inter_box_spread_match_the_literature(healthy):
    """In-service rise in the derived band, and a healthy spread of the measured order (8.4 K)."""
    _, feat, _ = healthy
    ins = feat[feat["in_service_frac"] > 0.99]
    rise = (ins["T_box_mean"] - ins["T_amb"]).mean()
    assert 15.0 <= rise <= 28.0, f"in-service mean rise {rise:.1f} K"
    per_box = feat.groupby("component_id", observed=True)["T_box_mean"].mean()
    spread = float(per_box.max() - per_box.min())
    assert 3.0 <= spread <= 14.0, f"healthy inter-box spread {spread:.1f} K (R149 measured 8.4 K)"


def test_healthy_run_never_reaches_an_ns_alarm_rule_code(healthy, params):
    """Singapore ambient puts a healthy box at 50-57 C, so the 80 C absolute line must not trip -
    but the margin is small, which is exactly rail_phm section 0's point."""
    _, feat, ev = healthy
    assert feat["T_box_max"].max() < params.level4_abs_c
    assert feat["dT_peer_same_side"].abs().max() < params.level1_dT_k
    assert "hot_box_alarm" not in set(ev["event"].astype(str))
    assert (feat["alarm_rule_code"] == 0).all()


def test_ambient_has_a_diurnal_shape_in_the_singapore_envelope(healthy):
    _, feat, _ = healthy
    amb = feat.groupby("cycle_id", observed=True)["T_amb"].first()
    assert 19.0 <= amb.min() and amb.max() <= 37.0
    assert amb.max() - amb.min() > 4.0, "ambient must swing, not sit at a constant"


def test_dwell_fraction_is_a_real_input(healthy):
    """Duty cycle is a model input, not decoration: it varies, and the hourly form is the
    causal 12-window rolling mean of it."""
    _, feat, _ = healthy
    g = _box(feat, "axlebox_1L").sort_values("cycle_id")
    dw = g["dwell_fraction"].to_numpy()
    assert 0.0 <= dw.min() < dw.max() <= 1.0
    assert dw.std() > 0.01
    last_h = g["dwell_fraction_last_hour"].to_numpy()
    assert last_h[0] == pytest.approx(dw[0], abs=1e-5), "first window has no history to average"
    assert last_h[11] == pytest.approx(dw[:12].mean(), rel=1e-4)


# --- the duty-cycle causality experiment -------------------------------------------------
#
# [R151]: "the bearing cools significantly by the wind while driving and warms up while
# standing still". This is the one claim in the module docstring that a lumped node does NOT
# give you for free, so it gets a controlled experiment rather than a correlation: two
# services that are bit-identical for two hours of 120 s runs / 30 s dwells and then either
# keep running or stand still for 30 minutes, with every source of noise switched off.


def _fixed_ambient(c: float = 30.0) -> AmbientProfile:
    return dataclasses.replace(
        AmbientProfile(), daily_min_c=c, daily_max_c=c, day_offset_sigma_c=0.0, noise_sigma_c=0.0
    )


def _noiseless(p: B.BearingParams) -> B.BearingParams:
    """Kill per-box scatter, sensor noise/quantisation/dropout and the station-fault mode, so
    the only thing left in T_box is the thermal model."""
    return dataclasses.replace(
        p,
        mu_scatter_sigma=0.0,
        hA_scatter_sigma=0.0,
        sensor_bias_sigma_k=0.0,
        sensors={k: SensorSpec() for k in p.sensors},
        wayside_station_fault_prob=0.0,
    )


def _metronome_service(*, standing: bool, prefix_s: float, stand_s: float, tail_s: float):
    """A hand-built Service: ``prefix_s`` of 120 s runs / 30 s dwells, then either one
    ``stand_s`` dwell (``standing``) or more of the same tiling, then ``tail_s`` of service."""
    rows: list[dict] = []
    t = 0.0

    def add(kind: str, dur: float) -> None:
        nonlocal t
        rows.append(
            dict(seg_id=len(rows), day=0, kind=kind, t_start=t, t_end=t + dur, duration_s=dur,
                 v_peak=22.0 if kind == "run" else 0.0, distance_m=0.0, load_frac=0.3,
                 T_amb=30.0, hour_of_day=0.0, in_service=True)
        )
        t += dur

    def tile(span: float) -> None:
        end = t + span
        while t < end:
            add("run", 120.0)
            add("dwell", 30.0)

    tile(prefix_s)
    t_stand = t
    if standing:
        add("dwell", stand_s)
    else:
        tile(stand_s)
    tile(tail_s)
    seg = pd.DataFrame(rows)[list(SEGMENT_COLUMNS)]
    days = int(np.ceil(t / DAY))
    return (
        Service(
            segments=seg, days=days, t0=pd.Timestamp("2026-09-01T00:00:00Z"),
            params=ServiceParams(), ambient_profile=_fixed_ambient(),
            day_offsets=np.zeros(days), train_id="T01",
        ),
        t_stand,
    )


@pytest.fixture(scope="module")
def stand_vs_service():
    """``(t_stand, {standing: (t, T_box)}, {standing: features})`` for the paired experiment."""
    p = _noiseless(B.BearingParams())
    traces, feats, t_stand = {}, {}, None
    for standing in (True, False):
        svc, t_stand = _metronome_service(
            standing=standing, prefix_s=2 * 3600.0, stand_s=1800.0, tail_s=1800.0
        )
        lg, ft, _ = B.simulate(
            p, (), svc, np.random.default_rng(0), 1, run_id="duty", car=1
        )
        one = lg[(lg["signal"].astype(str) == "T_box")
                 & (lg["component_id"].astype(str) == "axlebox_1L")].sort_values("timestamp")
        traces[standing] = (
            (one["timestamp"] - svc.t0).dt.total_seconds().to_numpy(),
            one["value"].to_numpy(dtype=float),
        )
        feats[standing] = _box(ft, "axlebox_1L").sort_values("cycle_id")
    return t_stand, traces, feats


def test_a_terminus_stand_warms_the_box_relative_to_service(stand_vs_service):
    """[R151] with the sign checked: standing must make the box HOTTER than staying in
    service, and it must keep rising for the whole stand - not merely cool more slowly."""
    t_stand, traces, _ = stand_vs_service
    (ts, Ts), (tm, Tm) = traces[True], traces[False]

    def at(t_arr, y, when):
        return float(y[int(np.searchsorted(t_arr, t_stand + when))])

    deltas = {d: at(ts, Ts, d) - at(tm, Tm, d) for d in (60.0, 300.0, 900.0, 1800.0)}
    assert all(v > 0.0 for v in deltas.values()), f"standing must be warmer, got {deltas}"
    assert deltas[1800.0] > 0.5, f"30 min of standing only bought {deltas[1800.0]:.2f} K"
    assert deltas[60.0] < deltas[300.0] < deltas[900.0] < deltas[1800.0], deltas

    i0 = int(np.searchsorted(ts, t_stand))
    i1 = int(np.searchsorted(ts, t_stand + 1800.0))
    stand_trace = Ts[i0:i1]
    assert stand_trace[-1] > stand_trace[0] + 0.5, "the standing box must rise, not decay"
    assert stand_trace.argmax() > 0.8 * len(stand_trace), "the peak must be at the END of the stand"


def test_the_stand_is_visible_in_the_dwell_features(stand_vs_service):
    """The same experiment seen through the 5-minute feature table: the stand windows are the
    dwell_fraction ~ 1 windows, and they are the warm ones."""
    t_stand, _, feats = stand_vs_service
    standing, moving = feats[True], feats[False]
    t0 = standing["t_start"].min()
    sec = (standing["t_start"] - t0).dt.total_seconds().to_numpy()
    inside = (sec >= t_stand) & (sec < t_stand + 1800.0)
    assert inside.sum() >= 5

    before = sec < t_stand  # the two hours the two services share exactly
    dw = standing["dwell_fraction"].to_numpy()
    assert dw[inside].min() > 0.95, "a terminus stand is a dwell_fraction ~ 1 window"
    assert dw[before].max() < 0.5, "ordinary service must stay well below it"
    assert moving["dwell_fraction"].to_numpy()[inside].max() < 0.5

    warm = standing["T_box_mean"].to_numpy()[inside] - moving["T_box_mean"].to_numpy()[inside]
    assert (warm > 0.0).all(), f"stand windows must run warmer, got {warm}"
    assert warm[-1] > warm[0], "and the gap must widen across the stand"


def test_thermal_residual_is_small_and_centred_on_a_healthy_run(healthy):
    _, feat, _ = healthy
    r = feat["thermal_residual"].to_numpy()
    assert np.isfinite(r).all()
    assert abs(float(np.mean(r))) < 1.0
    assert float(np.std(r)) < 3.0


# ---------------------------------------------------------------------------- degradation


@pytest.fixture(scope="module")
def degraded(service):
    """One full 0 -> 1 severity sweep on axlebox_3R over the simulated day."""
    traj = DegradationTrajectory(
        fault_type="bearing_degradation", subsystem="bearing", component_id="axlebox_3R",
        t_onset=0.0, t_failure=DAY, gamma=1.0, jitter_sigma=0.0,
    )
    return _run(service, (traj,)), traj


def test_degradation_moves_vibration_hard(degraded):
    (_, feat, _), _ = degraded
    bad = _box(feat, "axlebox_3R")
    good = _box(feat, "axlebox_3L")
    assert bad["vib_kurt_mean"].max() > 12.0, "kurtosis must peak well above healthy"
    assert good["vib_kurt_mean"].max() < 5.0
    assert bad["vib_bpfo_mean"].max() > 0.2
    # A healthy box is NOT BPFO-silent. Ottawa measures a real healthy envelope floor
    # (scripts/calibrate_bearing.py: outer-race records carry only 10.05x more BPFO band
    # amplitude than healthy ones), so vib_bpfo_healthy > 0 and the defect has to clear a
    # floor rather than emerge from exactly zero - the same trivial-separability error
    # rail_phm 4.3.2 caught on temperature.
    assert 0.02 < good["vib_bpfo_mean"].max() < 0.25
    assert bad["vib_bpfo_mean"].max() > 4.0 * good["vib_bpfo_mean"].max()
    assert bad["vib_rms_mean"].max() > 1.5 * good["vib_rms_mean"].max()
    assert bad["vib_crest_mean"].max() > 6.0


def test_kurtosis_collapses_again_at_full_severity(degraded):
    """The headline W1 correction, end to end: the severe class must NOT be trivially separable
    on kurtosis - a widened spall stops being impulsive [R157]."""
    (_, feat, _), _ = degraded
    g = _box(feat, "axlebox_3R").sort_values("cycle_id")
    g = g[g["v_mean"] > 5.0]  # kurtosis is only meaningful while the train is moving
    k = g["vib_kurt_mean"].to_numpy()
    i_peak = int(np.argmax(k))
    assert 0 < i_peak < len(k) - 1
    assert k[-1] < 0.4 * k[i_peak], "kurtosis must collapse back toward healthy at s -> 1"


def test_degradation_barely_moves_temperature(service, degraded):
    """rail_phm 4.3.2: the axle box gains +1.02 K where the rollers gain +15.6 K - ~15x less
    sensitive. The thermal move must stay buried in the healthy inter-box spread."""
    (_, feat, ev), _ = degraded
    base = _run(service, (), seed=5)[1]  # identical rng stream, no fault
    d = float(_box(feat, "axlebox_3R")["T_box_mean"].mean() - _box(base, "axlebox_3R")["T_box_mean"].mean())
    per_box = base.groupby("component_id", observed=True)["T_box_mean"].mean()
    spread = float(per_box.max() - per_box.min())
    assert 0.05 < d < 3.0, f"defect moved T_box by {d:.2f} K; expected ~1 K"
    assert d < spread, f"thermal move {d:.2f} K must stay below the healthy spread {spread:.2f} K"
    assert "hot_box_alarm" not in set(ev["event"].astype(str))


def test_degradation_labels_track_the_trajectory(degraded):
    (_, feat, _), traj = degraded
    bad = _box(feat, "axlebox_3R").sort_values("cycle_id")
    assert (bad["fault_type"].astype(str) == "bearing_degradation").all()
    assert bad["is_faulty"].all()
    assert bad["severity"].is_monotonic_increasing
    assert bad["severity"].iloc[-1] == pytest.approx(1.0, abs=0.02)
    assert bad["rul_s"].is_monotonic_decreasing
    assert bad["alarm_window_3d"].all()  # the whole 1-day run is inside 3 days of t_failure
    assert not _box(feat, "axlebox_3L")["is_faulty"].any()


# ---------------------------------------------------------------------------- hot axle box


@pytest.fixture(scope="module")
def hot(service):
    traj = DegradationTrajectory(
        fault_type="hot_axle_box", subsystem="bearing", component_id="axlebox_2L",
        t_onset=0.30 * DAY, t_failure=0.30 * DAY + 10 * 3600.0, gamma=3.0, jitter_sigma=0.0,
    )
    return _run(service, (traj,)), traj


def test_hot_axle_box_is_the_only_fault_that_drives_temperature(hot):
    (_, feat, _), _ = hot
    bad = _box(feat, "axlebox_2L")
    assert bad["T_box_max"].max() > 80.0
    assert bad["dT_peer_same_side"].max() > 30.0
    assert _box(feat, "axlebox_2R")["T_box_max"].max() < 80.0


def test_ns_alarm_rule_codes_escalate_in_order(hot, params):
    """[R151]: level 1 at dT > 30 K, level 3 at dT > 50 K, level 4 at 80 C absolute."""
    (_, feat, ev), _ = hot
    bad = _box(feat, "axlebox_2L").sort_values("cycle_id")
    lv = bad["alarm_rule_code"].to_numpy()
    assert lv.max() >= 3
    first = {int(v): int(np.argmax(lv >= v)) for v in (1, 3) if (lv >= v).any()}
    assert first[1] <= first[3]
    alarms = ev[ev["event"].astype(str) == "hot_box_alarm"]
    assert len(alarms) >= 1
    assert set(alarms["component_id"].astype(str)) == {"axlebox_2L"}
    detail = json.loads(alarms.iloc[0]["detail_json"])
    assert detail["level"] >= 3 and "dT_same_side_K" in detail


def test_functional_failure_is_written_back_to_the_trajectory(hot):
    (_, feat, ev), traj = hot
    assert traj.t_functional_failure is not None
    assert traj.t_onset < traj.t_functional_failure <= DAY
    ff = ev[ev["event"].astype(str) == "functional_failure"]
    assert len(ff) == 1 and ff.iloc[0]["component_id"] == "axlebox_2L"
    # RUL now counts down to the functional failure, not to the nominal t_failure.
    bad = _box(feat, "axlebox_2L").sort_values("cycle_id")
    tail = bad[bad["t_end"] >= bad["t_start"].iloc[0] + pd.Timedelta(seconds=traj.t_functional_failure)]
    assert (tail["rul_s"] == 0).all()


def test_fault_log_row_carries_the_functional_failure(hot, t0):
    _, traj = hot
    row = traj.to_fault_log_row(run_id="t0", train_id="T01", car=3, t0=t0)
    df = S.coerce_fault_log(pd.DataFrame([row]))
    S.validate_fault_log(df, strict=True)
    assert df["t_functional_failure"].notna().all()
    assert df["fault_type"].iloc[0] == "hot_axle_box"


# ---------------------------------------------------------------------------- sensor faults


def test_sensor_stuck_freezes_the_channel_without_touching_physics(service):
    traj = DegradationTrajectory(
        fault_type="sensor_stuck", subsystem="bearing", component_id="axlebox_4R",
        t_onset=0.5 * DAY, t_failure=0.5 * DAY, shape="step",
    )
    _, feat, _ = _run(service, (traj,), seed=5)
    g = _box(feat, "axlebox_4R").sort_values("cycle_id")
    after = g[g["t_end"] > g["t_start"].iloc[0] + pd.Timedelta(seconds=0.55 * DAY)]
    assert after["T_box_std"].max() == pytest.approx(0.0, abs=1e-5)
    assert after["T_box_mean"].std() == pytest.approx(0.0, abs=1e-4)
    assert _box(feat, "axlebox_4L")["T_box_mean"].std() > 1.0


def test_sensor_offset_shifts_only_the_affected_box(service, params):
    traj = DegradationTrajectory(
        fault_type="sensor_offset", subsystem="bearing", component_id="axlebox_1R",
        t_onset=0.4 * DAY, t_failure=0.4 * DAY, shape="step",
    )
    _, feat, _ = _run(service, (traj,), seed=5)
    base = _run(service, (), seed=5)[1]
    g = _box(feat, "axlebox_1R").sort_values("cycle_id").reset_index(drop=True)
    b = _box(base, "axlebox_1R").sort_values("cycle_id").reset_index(drop=True)
    d = (g["T_box_mean"] - b["T_box_mean"]).to_numpy()
    assert d[:100] == pytest.approx(np.zeros(100), abs=1e-4)
    assert d[-1] == pytest.approx(params.sensor_offset_k, abs=0.2)
    other = _box(feat, "axlebox_1L")["T_box_mean"].to_numpy()
    assert other == pytest.approx(_box(base, "axlebox_1L")["T_box_mean"].to_numpy(), abs=1e-4)


# ---------------------------------------------------------------------------- wayside


def test_wayside_snapshots_are_sparse_and_biased(service, params):
    long, _, ev = _run(service, (), seed=5)
    ws = long[long["signal"].astype(str) == "T_box_wayside"]
    passes = ev[ev["event"].astype(str) == "wayside_pass"]
    assert len(passes) == params.wayside_passes_per_day
    assert len(ws) == len(passes) * 8
    # a snapshot, not a continuous channel: 3 instants out of 86 400 onboard samples
    assert ws["timestamp"].nunique() == len(passes) <= 10
    assert set(ws["component_id"].astype(str)) == set(B.BOX_IDS)
    assert ws["value"].notna().all()


def test_station_fault_mode_is_caught_by_the_ns_data_quality_guard(service):
    """[R151]: '>= 4 boxes hot in one pass => blame the measurement station'. Ours re-expresses
    the published absolute 50 C as a rise above ambient, because a HEALTHY Singapore box already
    sits at 50-57 C and the absolute form would fire on every pass."""
    p = dataclasses.replace(B.BearingParams(), wayside_station_fault_prob=1.0)
    _, _, ev = _run(service, (), seed=5, params=p)
    kinds = ev["event"].astype(str)
    assert (kinds == "station_fault").sum() >= 1
    # A station bias on boxes that have not warmed up yet is genuinely invisible to an
    # absolute guard; what must hold is that "suppressed" and the box count agree exactly.
    for d in ev[kinds == "wayside_pass"]["detail_json"]:
        det = json.loads(d)
        assert det["suppressed"] == (det["n_boxes_hot"] >= p.wayside_guard_boxes)
    assert any(json.loads(d)["suppressed"] for d in ev[kinds == "wayside_pass"]["detail_json"])


def test_guard_does_not_fire_on_a_healthy_singapore_run(service):
    _, _, ev = _run(service, (), seed=5)
    passes = ev[ev["event"].astype(str) == "wayside_pass"]
    assert len(passes) and not any(json.loads(d)["suppressed"] for d in passes["detail_json"])


# ---------------------------------------------------------------------------- store_every


def test_store_every_thins_telemetry_but_never_features(service):
    long_a, feat_a, _ = _run(service, (), seed=5, store_every=4)
    long_b, feat_b, _ = _run(service, (), seed=5, store_every=16)
    n = lambda d: len(d[(d["subsystem"] == "bearing") & (d["signal"].astype(str) == "T_box")])  # noqa: E731
    assert n(long_a) > n(long_b) > 0
    assert len(feat_a) == len(feat_b)
    assert feat_a["cycle_id"].nunique() == feat_b["cycle_id"].nunique()


def test_tail_before_failure_is_always_stored(service):
    """The contract: every cycle in the last 2 days before failure is kept regardless of thinning."""
    traj = DegradationTrajectory(
        fault_type="hot_axle_box", subsystem="bearing", component_id="axlebox_2L",
        t_onset=0.6 * DAY, t_failure=0.9 * DAY, gamma=3.0,
    )
    long, _, _ = _run(service, (traj,), seed=5, store_every=10**6)
    tb = long[(long["subsystem"] == "bearing") & (long["signal"].astype(str) == "T_box")]
    assert tb["timestamp"].nunique() > 0.9 * 86400, "the whole 1-day run is inside the failure tail"


# ---------------------------------------------------------------------------- input validation


def test_faults_router_rejects_a_non_bearing_trajectory():
    door = DegradationTrajectory(
        fault_type="friction", subsystem="door", component_id="door_L1", t_onset=0.0, t_failure=DAY
    )
    with pytest.raises(ValueError, match="subsystem"):
        B.BearingFaults.from_trajectories([door])


def test_faults_router_sorts_by_fault_type():
    made = [
        DegradationTrajectory(fault_type=ft, subsystem="bearing", component_id=cid,
                              t_onset=0.0, t_failure=DAY,
                              shape="step" if ft.startswith("sensor") else "power")
        for ft, cid in (("outer_race", "axlebox_1L"), ("hot_axle_box", "axlebox_2L"),
                        ("sensor_stuck", "axlebox_3L"), ("sensor_offset", "axlebox_4L"))
    ]
    fl = B.BearingFaults.from_trajectories(made)
    assert len(fl.degradation) == len(fl.hot_box) == len(fl.sensor_stuck) == len(fl.sensor_offset) == 1
    assert len(fl.all()) == 4 and bool(fl)
    assert not B.BearingFaults()


def test_simulate_rejects_bad_arguments(service, params):
    with pytest.raises(ValueError, match="car must lie"):
        B.simulate(params, None, service, np.random.default_rng(0), car=0)
    with pytest.raises(ValueError, match="store_every"):
        B.simulate(params, None, service, np.random.default_rng(0), store_every=0)
    with pytest.raises(TypeError, match="BearingParams"):
        B.simulate(object(), None, service, np.random.default_rng(0))  # type: ignore[arg-type]


def test_short_service_is_rejected_with_a_precise_message():
    svc = generate_service(1, np.random.default_rng(1))
    p = dataclasses.replace(B.BearingParams(), window_s=2 * SEC_PER_DAY)
    with pytest.raises(ValueError, match="shorter than one"):
        B.simulate(p, None, svc, np.random.default_rng(0))


def test_selftest_entrypoint_runs():
    assert B.main(["--days", "1", "--seed", "3", "--store-every", "180"]) == 0
