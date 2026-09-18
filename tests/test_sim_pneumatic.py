"""Contract + physics tests for nebulax.sim.pneumatic (brake air supply / APU).

Anchors asserted here, with their sources:

* control band 8.06 / 10.2 bar, LPS 7.0 bar                             MEASURED on MetroPT-3
* Motor_current 0.04 / 3.79 / 5.92 A, NOT [R87]'s nominal 0 / 4 / 7 A   MEASURED on MetroPT-3
* load/unload transitions 40 s / 3 s, not instantaneous                [R155, rail_phm 2.3b]
* leak is a sharp-edged orifice, Q ~ d^2 and rising with P_abs         [R154, rail_phm 4.2]
* air leak shrinks ``idle_run_ratio`` / ``t_off``                      [R101, rail_phm 4.2]
* a valve leak shortens the charge slope WITHOUT raising the OFF decay [rail_phm 4.2]
* compressor duty ~ 0.09-0.12                                          MEASURED on MetroPT-3
* TP2 ~0 bar off/unloaded and 9.29 bar loaded; H1 the exact opposite   MEASURED on MetroPT-3
* TP2 - TP3 while loaded is POSITIVE (+0.304 bar measured)             MEASURED on MetroPT-3
* functional failure: LPS > 60 s in service, or oil > 95 C             (plan)

Every run here is one simulated day or two, so the whole module stays well inside 60 s.

Every constant marked MEASURED is produced by ``scripts/calibrate_pneumatic.py`` on the
failure-free MetroPT-3 window (1 Feb - 31 Mar 2020); re-running that script is the audit and
should report ``CONFIRMED`` on every row of its table. Where a measurement and the dataset
descriptor [R87] disagree - the loaded current is 5.92 A, not "about 7 A" - the measurement
wins and the test says so.
"""

from __future__ import annotations

import time

import json

import numpy as np
import pandas as pd
import pytest

from nebulax import schema as S
from nebulax.sim import pneumatic as PN
from nebulax.sim.common import SEC_PER_DAY, DegradationTrajectory, generate_service

DAY = SEC_PER_DAY
SEED = 20260918


def _service(days: int = 1, seed: int = SEED):
    return generate_service(days, np.random.default_rng(seed), train_id="T01")


def _trajectory(fault_type: str, *, onset_d: float, fail_d: float, gamma: float = 2.0):
    shape = "step" if fault_type == "dryer_valve_stuck" else "power"
    return DegradationTrajectory(
        fault_type=fault_type,
        subsystem="pneumatic",
        component_id="apu_1",
        t_onset=onset_d * DAY,
        t_failure=fail_d * DAY,
        gamma=gamma,
        shape=shape,
    )


def _run(fault_type: str | None = None, *, days: int = 1, store_every: int = 60, **kw):
    service = _service(days)
    faults = PN.PneumaticFaults()
    if fault_type is not None:
        faults = PN.PneumaticFaults.from_trajectories([_trajectory(fault_type, **kw)])
    rng = np.random.default_rng(SEED)
    long, feats, events = PN.simulate(
        PN.PneumaticParams(),
        faults,
        service,
        rng,
        store_every=store_every,
        run_id=f"test_{fault_type or 'healthy'}",
    )
    return long, feats, events, faults, service


@pytest.fixture(scope="module")
def healthy():
    return _run(None, days=1, store_every=1)


@pytest.fixture(scope="module")
def leaky():
    return _run("air_leak", days=1, store_every=60, onset_d=0.15, fail_d=1.0)


# ------------------------------------------------------------------ schema conformance


def test_healthy_run_is_schema_conformant(healthy):
    long, feats, events, _, _ = healthy
    S.validate_long(long, strict=True)
    S.validate_features(feats, require_labels=True)
    S.validate_events(events, strict=True)
    assert len(long) and len(feats) and len(events)


def test_every_registered_pneumatic_signal_is_emitted(healthy):
    long, _, _, _, _ = healthy
    emitted = set(long.loc[long["subsystem"] == "pneumatic", "signal"].astype(str))
    assert emitted == set(S.SIGNALS["pneumatic"])
    # MetroPT-3 names verbatim, typo included
    assert set(S.METROPT3_SIGNALS) <= emitted
    assert "DV_eletric" in emitted and "Flowmeter" in emitted


def test_context_rows_ride_along_on_the_train_pseudo_subsystem(healthy):
    long, _, _, _, _ = healthy
    ctx = long[long["subsystem"] == "train"]
    assert set(ctx["signal"].astype(str)) == set(S.CONTEXT_SIGNALS)
    assert (ctx["component_id"].astype(str) == "train").all()
    assert (ctx["car"] == 0).all()
    wide = S.to_wide(long, "pneumatic", include_context=True)
    assert wide[["speed", "load_frac", "T_amb", "in_service"]].notna().all().all()
    # the rows are built from the masked timeline for memory reasons; they must still be
    # exactly what Service.context_long() would have emitted
    _, _, _, _, service = healthy
    ref = service.context_long(1.0, source="sim", run_id="test_healthy")
    key = ["timestamp", "signal"]
    a = ctx.sort_values(key).reset_index(drop=True)
    b = ref.sort_values(key).reset_index(drop=True)
    assert len(a) == len(b)
    assert a["signal"].astype(str).equals(b["signal"].astype(str))
    assert a["timestamp"].equals(b["timestamp"])
    assert np.allclose(a["value"], b["value"])


def test_apu_is_a_unit_level_component(healthy):
    long, feats, events, _, _ = healthy
    for df in (long[long["subsystem"] == "pneumatic"], feats, events):
        assert (df["car"] == 0).all()
        assert set(df["component_id"].astype(str)) == {"apu_1"}


def test_cycle_feature_columns_are_all_present(healthy):
    _, feats, _, _, _ = healthy
    for col in PN.CYCLE_FEATURE_COLUMNS:
        assert col in feats.columns, col
        assert feats[col].dtype == np.float32
    assert feats["cycle_id"].is_monotonic_increasing


# ------------------------------------------------------------------ control constants


def test_control_band_start_806_stop_102_measured_on_metropt3(healthy):
    """MEASURED: the unit loads at 8.06 bar and unloads at 10.21, both from the cycle
    extremes corrected for MetroPT-3's 10 s sampling lag. [R87] only bounds the band
    ("starts below 8.2, stops above 10.2"); the file pins it."""
    long, _, _, _, _ = healthy
    wide = S.to_wide(long, "pneumatic")
    p = PN.PneumaticParams()
    assert (p.P_start_bar, p.P_stop_bar, p.P_lps_bar) == (8.06, 10.2, 7.0)
    assert p.P_start_bar < 8.2 and p.P_stop_bar >= 10.2  # ... and still inside [R87]'s bounds
    res = wide["Reservoirs"].to_numpy()
    # sensor noise is 11.3 mbar + 2 mbar quantisation (measured); allow 5 sigma
    assert res.min() > p.P_start_bar - 0.25
    assert res.max() < p.P_stop_bar + 0.10
    # the compressor really does cycle across the band
    assert res.max() - res.min() > 1.5
    assert (wide["LPS"].to_numpy() == 0).all()


def test_motor_current_has_the_three_measured_levels(healthy):
    """MEASURED on MetroPT-3, and the measurement corrects the descriptor.

    [R87] gives the three states as "about 0 / 4 / 7 A". The file's own histogram is three
    clean clusters at **0.04 / 3.79 / 5.92 A** with empty gaps at 0.25-3.5 A and 4.25-4.75 A -
    so the loaded plateau is 5.9 A, not 7. The simulator now reproduces the measurement.
    """
    long, feats, _, _, _ = healthy
    wide = S.to_wide(long, "pneumatic")
    i = wide["Motor_current"].to_numpy()
    comp = wide["COMP"].to_numpy()
    off = (comp == 1) & (wide["DV_eletric"].to_numpy() == 0)
    # loaded 5.92 A median on MetroPT-3 (p5-p95 5.36-6.22)
    assert 5.6 <= float(np.nanmean(i[comp == 0])) <= 6.3
    # a distinct offloaded plateau at 3.79 A and a zero plateau
    assert np.isclose(np.nanmedian(i[off & (i > 1.0)]), 3.785, atol=0.2)
    assert float(np.nanmin(i)) < 0.2
    assert 5.6 <= float(feats["I_loaded_mean"].mean()) <= 6.3
    # current falls with oil temperature (thinner oil): the calibrated coefficient is negative
    assert PN.PneumaticParams().I_loaded_kT < 0.0
    assert float(feats["I_start_peak"].mean()) > 8.5  # direct-on-line start peak


def test_load_and_unload_transitions_are_40s_and_3s(healthy):
    """[R155]: sump relief on unload ~40 s, repressurise on reload ~3 s - not instantaneous.

    The offloaded phase is a **timer** (``t_unload_s + t_hold_s`` = 416 s, which is what
    MetroPT-3's p25-p75 of 416-426 s says), but the timer is not the only way out: a large air
    demand can pull the reservoir below ``P_start`` during the hold and reload early. The real
    file shows exactly that lower tail (``t_unloaded`` p1-p10 = 9-10 s), so the timer has to be
    the MODE here, not an invariant - it was asserted as an invariant until 15 Sep 2026, when
    the burst schedule was still one fixed draw per stop.
    """
    _, feats, _, _, _ = healthy
    p = PN.PneumaticParams()
    t_unl = feats["t_unloaded"].to_numpy()
    timer = p.t_unload_s + p.t_hold_s
    assert np.isclose(np.median(t_unl), timer, atol=1.5)
    assert float(np.mean(np.isclose(t_unl, timer, atol=1.5))) > 0.6
    assert (t_unl <= timer + 1.5).all()  # the timer is a hard ceiling
    assert t_unl.min() >= p.t_unload_s  # ... and the 40 s blowdown a hard floor
    assert p.t_unload_s == 40.0 and p.t_reload_s == 3.0


def test_twin_tower_dryer_switches_on_its_parameterised_period(healthy):
    long, feats, events, _, _ = healthy
    p = PN.PneumaticParams()
    wide = S.to_wide(long, "pneumatic")
    towers = wide["Towers"].to_numpy()
    flips = int(np.sum(np.abs(np.diff(towers)) > 0.5))
    loaded_s = float((wide["COMP"].to_numpy() == 0).sum())
    assert flips > 0
    # switching runs on loaded time only, so period ~= loaded_s / flips
    assert 0.6 * p.tower_period_s < loaded_s / flips < 1.6 * p.tower_period_s
    assert float(feats["tower_switches"].mean()) > 1.0
    assert (feats["purge_count"] == feats["tower_switches"]).all()
    assert {"tower_switch", "purge"} <= set(events["event"].astype(str))


def test_duty_cycle_matches_the_metropt3_air_balance(healthy):
    """MEASURED: MetroPT-3's loaded fraction is 0.089 over whole compressor cycles (0.114 over
    the raw stream, which also counts logger outages). The consumption schedule is scaled to
    that air balance by ``scripts/calibrate_pneumatic.py``; the first cut's 0.27 came from a
    consumption tuning that was ~7x too large.

    The anchor is the **median** cycle and the **time-weighted** duty, not the mean over
    cycles: MetroPT-3's per-cycle duty is strongly right-skewed (median 0.081, mean 0.196,
    13 % of healthy cycles above 0.5), because a big consumer can hold the compressor on for a
    whole cycle. Asserting the mean inside a narrow band would be asserting that the demand is
    homogeneous, which is exactly what the file falsifies.
    """
    _, feats, _, _, _ = healthy
    svc = feats["in_service_frac"].to_numpy() > 0.5
    duty = feats["duty_ratio"].to_numpy()
    assert 0.07 <= float(np.nanmedian(duty[svc])) <= 0.13, float(np.nanmedian(duty[svc]))
    # time-weighted over the whole run, the one number the air balance actually fixes
    t_loaded = feats["t_loaded"].to_numpy()
    span = t_loaded + feats["t_unloaded"].to_numpy() + feats["t_off"].to_numpy()
    assert 0.06 <= float(t_loaded.sum() / span.sum()) <= 0.12
    # right-skewed like the real file, but nowhere near it: ours has no duty ~ 1 population
    assert float(np.nanmean(duty[svc])) > float(np.nanmedian(duty[svc]))
    assert np.all(np.isclose(feats["idle_run_ratio"], feats["t_off"] / feats["t_loaded"], rtol=1e-4))
    # DOE band: a stabled train (leak only) must look "well maintained", >= 9.0 idle/run
    assert float(np.nanmax(feats["idle_run_ratio"].to_numpy()[~svc])) > 9.0


# ------------------------------------------------------------------ the leak law


def test_leak_is_a_sharp_edged_orifice_scaling_as_d_squared():
    p = PN.PneumaticParams()
    p_ref = p.leak_ref_pressure_barg + p.P_atm_bar
    q = PN.leak_nls(p_ref, np.array([0.40, 0.79, 1.59, 3.18]), p)
    doe = np.array([PN.DOE_LEAK_TABLE_NLS[d] for d in (0.40, 0.79, 1.59, 3.18)])
    # our reference point is the 0.55 mm / 0.2 NL/s pair; the DOE table is reproduced to ~20 %
    assert np.allclose(q, doe, rtol=0.25), (q, doe)
    # exact d^2 scaling
    assert PN.leak_nls(p_ref, 1.10, p) == pytest.approx(4.0 * PN.leak_nls(p_ref, 0.55, p))


def test_leak_rises_with_supply_pressure_not_constant():
    p = PN.PneumaticParams()
    lo = PN.leak_nls(p.P_start_bar + p.P_atm_bar, p.leak_d0_mm, p)
    hi = PN.leak_nls(p.P_stop_bar + p.P_atm_bar, p.leak_d0_mm, p)
    assert hi > lo * 1.15  # a constant-NL/s draw would give hi == lo
    assert 0.2 < lo < 0.4 and 0.25 < hi < 0.45  # ~0.3 NL/s in band, the demo_runs tuning


def test_orifice_diameter_runs_055_to_288_mm():
    """``s = 0`` is 0.55 mm (confirmed by MetroPT-3's OFF decay); ``s = 1`` is 2.88 mm.

    CALIBRATED: MetroPT-3's own air-leak episodes bound the endpoint from below - during
    29 May - 7 Jun 2020 the unit ran continuously and settled at 8.38 bar, an equivalent
    2.66 mm orifice, and on 18 Apr it settled at 8.86 bar (2.57 mm) - but both were repaired
    *before* functional failure. ``s = 1`` is therefore defined as the leak that just beats
    the compressor at the 7.0 bar LPS trip, 2.88 mm. rail_phm 4.2's "roughly 2.2 mm" was a
    back-calculation from the plan's own discarded constant-draw law, not a measurement.
    """
    p = PN.PneumaticParams()
    assert PN.orifice_diameter_mm(0.0, p) == pytest.approx(0.55)
    assert PN.orifice_diameter_mm(1.0, p) == pytest.approx(2.88)
    # above the worst real episode (2.66 mm) and below the DOE 1/8 in row (3.18 mm)
    assert 2.66 <= p.leak_d1_mm <= 3.18
    assert p.leak_area_ratio_max == pytest.approx((2.88 / 0.55) ** 2)
    s = np.linspace(0, 1, 50)
    assert np.all(np.diff(PN.orifice_diameter_mm(s, p)) > 0)


# ------------------------------------------------------------------ fault signatures


def test_air_leak_shrinks_idle_run_ratio_and_raises_flow(healthy, leaky):
    """[R101]: air leakage makes compressor idle time shorter than run time."""
    _, f_ok, _, _, _ = healthy
    _, f_bad, _, _, _ = leaky
    late = f_bad["severity"].to_numpy() > 0.5
    assert late.any()
    ok = f_ok["in_service_frac"].to_numpy() > 0.5
    bad = late & (f_bad["in_service_frac"].to_numpy() > 0.5)
    assert np.nanmedian(f_bad["idle_run_ratio"].to_numpy()[bad]) < 0.75 * np.nanmedian(
        f_ok["idle_run_ratio"].to_numpy()[ok]
    )
    assert np.nanmedian(f_bad["t_off"].to_numpy()[bad]) < np.nanmedian(f_ok["t_off"].to_numpy()[ok])
    assert np.nanmax(f_bad["Flowmeter_max"]) > np.nanmax(f_ok["Flowmeter_max"])
    assert np.nanmean(f_bad["duty_ratio"].to_numpy()[bad]) > np.nanmean(f_ok["duty_ratio"].to_numpy()[ok])


def test_leak_stays_well_below_the_regulatory_catastrophic_bound(healthy, leaky):
    """[R156]: 60 cfm = 28.3 NL/s and 5 psi/min = 0.34 bar/min are a whole freight consist's
    brake pipe, measured with the brake valve in neutral - i.e. with NO genuine consumption.
    So the flow form is compared against our worst leak, and the pressure form only on the
    stabled cycles; an in-service window legitimately decays faster because the train is
    braking into it."""
    p = PN.PneumaticParams()
    worst_leak = float(PN.leak_nls(p.P_stop_bar + p.P_atm_bar, p.leak_d1_mm, p))
    # The MetroPT-3-calibrated endpoint is larger than the plan's guess, so the margin to the
    # freight bound narrows from ~10x to ~3.3x. Still an order-of-magnitude statement: 28.3 NL/s
    # is a whole consist's brake pipe, ours is one unit's main reservoir.
    assert worst_leak < 28.3 / 3.0, worst_leak
    for fixture in (healthy, leaky):
        _, feats, _, _, _ = fixture
        stabled = feats["in_service_frac"].to_numpy() < 0.5
        assert stabled.any()
        worst = float(np.nanmin(feats["dP_dt_off"].to_numpy()[stabled])) * 60.0  # bar/min
        assert worst > -0.34, worst


def test_valve_leak_shortens_the_charge_slope_without_raising_the_off_decay():
    """[rail_phm 4.2]: this is exactly what separates a valve leak from a reservoir leak.

    Two days per arm, not one: since the burst schedule became heavy-tailed the per-cycle OFF
    decay has a real spread (MetroPT-3's own p25-p75 is -0.00166..-0.00095 bar/s), so a median
    taken over the three cycles a one-day run leaves inside the severity window is noise. The
    physics asserted is unchanged.
    """
    _, f_ok, _, _, _ = _run(None, days=2, store_every=600)
    _, f_in, _, _, _ = _run("valve_leak_inlet", days=2, onset_d=0.2, fail_d=1.0)
    _, f_air, _, _, _ = _run("air_leak", days=2, onset_d=0.2, fail_d=1.0)
    # Both arms are compared on the cycles that still HAVE an OFF phase, because dP_dt_off is
    # NaN by construction once the fault drives the duty to 1. That break-even arrives at very
    # different severities for the two faults, and the asymmetry is itself the point: a valve
    # leak takes delivery *away* (it never reaches break-even inside the scale) while a
    # reservoir leak *adds demand* and, on the MetroPT-3-calibrated APU whose healthy draw is
    # already 82 % leak, crosses it at s ~ 0.13. So the severity bands cannot be equal.
    live = lambda f: (f["in_service_frac"].to_numpy() > 0.5) & (f["t_off"].to_numpy() > 0.0)
    sel_valve = live(f_in) & (f_in["severity"].to_numpy() > 0.30) & (f_in["severity"].to_numpy() < 0.85)
    sel_air = live(f_air) & (f_air["severity"].to_numpy() > 0.02)
    ok = live(f_ok)
    assert sel_valve.sum() > 8 and sel_air.sum() > 8

    slope_ok = np.nanmedian(f_ok["dP_dt_loaded"].to_numpy()[ok])
    slope_valve = np.nanmedian(f_in["dP_dt_loaded"].to_numpy()[sel_valve])
    assert slope_valve < 0.9 * slope_ok  # charge slope falls
    decay_ok = np.nanmedian(f_ok["dP_dt_off"].to_numpy()[ok])
    decay_valve = np.nanmedian(f_in["dP_dt_off"].to_numpy()[sel_valve])
    assert decay_valve > 1.3 * decay_ok  # ... but the OFF decay does NOT (less negative == equal)
    # A reservoir leak DOES steepen the OFF decay, on every cycle that still has one.
    decay_air = np.nanmedian(f_air["dP_dt_off"].to_numpy()[sel_air])
    assert decay_air < 1.5 * decay_ok


# ------------------------------------------------------ TP2 / H1 placement (MEASURED)

#: MetroPT-3 medians by compressor state, measured on the failure-free window
#: 1 Feb - 31 Mar 2020 (445,298 rows) with the state cut ``scripts/calibrate_pneumatic.py``
#: uses (``Motor_current`` < 2 A off, > 5 A loaded; 279,701 / 114,672 / 50,925 samples).
#: ``TP2`` is the compressor discharge tap - live only while the unit delivers - and ``H1``
#: is the cyclonic-separator discharge tap, which does the opposite. Before 15 Sep 2026 the
#: simulator emitted ``H1`` inverted (high while loaded); these anchors are the regression.
METROPT3_TP2_BY_STATE = {"off": -0.012, "unloaded": -0.012, "loaded": 9.294}
METROPT3_H1_BY_STATE = {"off": 8.706, "unloaded": 9.662, "loaded": -0.012}
METROPT3_TP3_BY_STATE = {"off": 8.718, "unloaded": 9.678, "loaded": 9.162}
#: ... and, while loaded, ``TP2 - TP3`` and ``H1 - TP3``.
METROPT3_TP2_MINUS_TP3_LOADED = 0.304  # p25-p75 0.156 .. 0.396
METROPT3_H1_MINUS_TP3_LOADED = -9.168

#: Simulated levels are compared to the measured ones within this band. It is loose enough to
#: absorb the one known offset (our discharge drop is the sum of three handbook [R155] drops,
#: 0.55 bar, against 0.304 bar measured - see ``_cycle_features``) and the fact that our
#: reservoir-pressure distribution inside a state is not MetroPT-3's.
LEVEL_TOL_BAR = 0.5
#: A channel that the real file pins at its zero offset (-0.012 bar) must read ~0 here; the
#: module's clean floor is 0.0 because the SensorSpec clips at 0.
ZERO_TOL_BAR = 0.05


def _states_from_current(current: np.ndarray) -> np.ndarray:
    """OFF / UNLOADED / LOADED (0/1/2), the cut scripts/calibrate_pneumatic.py uses."""
    i = np.asarray(current, dtype=np.float64)
    return np.where(i < 2.0, 0, np.where(i < 5.0, 1, 2))


def _wide_by_state(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    wide = S.to_wide(long, "pneumatic")
    state = _states_from_current(wide["Motor_current"].to_numpy())
    return {name: wide[state == code] for code, name in ((0, "off"), (1, "unloaded"), (2, "loaded"))}


def test_tp2_and_h1_have_the_measured_state_shape(healthy):
    """TP2 live only while loaded, H1 only while NOT loaded - MEASURED on MetroPT-3."""
    long, _, _, _, _ = healthy
    by_state = _wide_by_state(long)
    assert all(len(f) > 100 for f in by_state.values())

    for name, frame in by_state.items():
        tp2 = float(np.nanmedian(frame["TP2"]))
        h1 = float(np.nanmedian(frame["H1"]))
        # off-state floors: whichever channel the real file pins at zero must be ~0 here
        if name == "loaded":
            assert abs(h1 - METROPT3_H1_BY_STATE[name]) < ZERO_TOL_BAR, f"H1 while {name}"
            assert abs(tp2 - METROPT3_TP2_BY_STATE[name]) < LEVEL_TOL_BAR, f"TP2 while {name}"
            assert tp2 > 5.0  # ... and it is the discharge pressure, not a floor
        else:
            assert abs(tp2 - METROPT3_TP2_BY_STATE[name]) < ZERO_TOL_BAR, f"TP2 while {name}"
            assert abs(h1 - METROPT3_H1_BY_STATE[name]) < LEVEL_TOL_BAR, f"H1 while {name}"
            assert h1 > 5.0  # ... H1 holds panel pressure, it does not collapse
        assert abs(float(np.nanmedian(frame["TP3"])) - METROPT3_TP3_BY_STATE[name]) < LEVEL_TOL_BAR


def test_loaded_channel_differences_match_metropt3_in_sign_and_scale(healthy):
    long, _, _, _, _ = healthy
    loaded = _wide_by_state(long)["loaded"]
    dp = float(np.nanmedian(loaded["TP2"] - loaded["TP3"]))
    # TP2 is upstream of the separator + filter + dryer, so the difference is POSITIVE while
    # loaded: MetroPT-3 measures +0.304 bar (it was the sign of this that the old placement
    # made meaningless once H1 was corrected).
    assert dp > 0.0
    assert abs(dp - METROPT3_TP2_MINUS_TP3_LOADED) < LEVEL_TOL_BAR
    # H1 is dumped while loaded: ~ -9.2 bar below the panel, not above it.
    d_h1 = float(np.nanmedian(loaded["H1"] - loaded["TP3"]))
    assert d_h1 < 0.0
    assert abs(d_h1 - METROPT3_H1_MINUS_TP3_LOADED) < LEVEL_TOL_BAR
    # H1 tracks the panel in both non-delivering states (measured H1 - TP3 = -0.012 bar there)
    for name in ("off", "unloaded"):
        frame = _wide_by_state(long)[name]
        assert abs(float(np.nanmedian(frame["H1"] - frame["TP3"]))) < ZERO_TOL_BAR


def test_h1_loaded_mean_feature_is_near_zero_like_the_real_file(healthy):
    """``H1_loaded_mean`` is ~0 in MetroPT-3 too; the fix is what makes the columns agree."""
    _, feats, _, _, _ = healthy
    assert abs(float(np.nanmedian(feats["H1_loaded_mean"]))) < ZERO_TOL_BAR


def test_clogged_filter_opens_tp2_minus_tp3_and_leaves_h1_alone(healthy):
    """The corrected placement re-routes the fault onto TP2 - TP3, with H1 as the control.

    TP2 (compressor discharge) is upstream of the air/oil filter and TP3 (panel) downstream of
    it, so clogging opens that difference. H1 sits on the reservoir side and is dumped while
    loaded, so it must not move at all - the old ``TP2 - H1`` channel is meaningless now.
    """
    long_ok, f_ok, _, _, _ = healthy
    long_bad, f_bad, _, _, _ = _run("clogged_filter", days=1, store_every=1, onset_d=0.1, fail_d=0.5)
    late = f_bad["severity"].to_numpy() > 0.8
    dp_ok = float(np.nanmedian(f_ok["TP2_minus_TP3_mean"]))
    dp_bad = float(np.nanmedian(f_bad["TP2_minus_TP3_mean"].to_numpy()[late]))
    assert dp_bad > dp_ok + 0.5  # dp_filter = 0.15*(1 + 6 s)

    # H1 while loaded is ~0 with and without the fault: the negative control
    h1_ok = float(np.nanmedian(f_ok["H1_loaded_mean"]))
    h1_bad = float(np.nanmedian(f_bad["H1_loaded_mean"].to_numpy()[late]))
    assert abs(h1_ok) < ZERO_TOL_BAR and abs(h1_bad) < ZERO_TOL_BAR
    assert dp_bad - dp_ok > 3.0 * abs(h1_bad - h1_ok)

    # ... and H1 in the states where it IS alive keeps reading the panel, fault or not
    for name in ("off", "unloaded"):
        for source in (long_ok, long_bad):
            frame = _wide_by_state(source)[name]
            assert abs(float(np.nanmedian(frame["H1"] - frame["TP3"]))) < ZERO_TOL_BAR


def test_dryer_valve_stuck_is_a_step_that_stops_tower_switching():
    _, feats, events, faults, _ = _run("dryer_valve_stuck", days=1, onset_d=0.4, fail_d=0.4)
    traj = faults.dryer_valve_stuck
    assert traj is not None and traj.shape == "step"
    sev = feats["severity"].to_numpy()
    before = feats["tower_switches"].to_numpy()[sev < 0.5]
    # skip the one cycle that straddles the step: it switched before the valve stuck
    after = feats["tower_switches"].to_numpy()[sev >= 0.5][1:]
    assert before.mean() > 1.0 and after.max() == 0.0
    assert feats["purge_count"].to_numpy()[sev >= 0.5][1:].max() == 0.0
    # a stuck-open purge valve keeps venting, so the duty rises
    assert np.nanmean(feats["duty_ratio"].to_numpy()[sev >= 0.5]) > np.nanmean(
        feats["duty_ratio"].to_numpy()[sev < 0.5]
    )


def test_oil_leak_heats_the_sump_and_trips_the_oil_level_switch(healthy):
    long, f_bad, events, _, _ = _run("oil_leak", days=1, onset_d=0.1, fail_d=0.6)
    _, f_ok, _, _, _ = healthy
    late = f_bad["severity"].to_numpy() > 0.8
    assert np.nanmax(f_bad["T_oil_max"].to_numpy()[late]) > np.nanmax(f_ok["T_oil_max"]) + 3.0
    wide = S.to_wide(long, "pneumatic")
    assert wide["Oil_level"].to_numpy().max() == 1.0  # flips at s ~ 0.6
    assert "oil_low" in set(events["event"].astype(str))
    # hysteresis: the level switch must not chatter
    assert int((events["event"].astype(str) == "oil_low").sum()) == 1


def test_compressor_wear_cuts_delivery_and_raises_current(healthy):
    _, f_ok, _, _, _ = healthy
    _, f_bad, _, _, _ = _run("compressor_wear", days=1, onset_d=0.1, fail_d=0.5)
    late = f_bad["severity"].to_numpy() > 0.8
    ok = f_ok["in_service_frac"].to_numpy() > 0.5
    bad = late & (f_bad["in_service_frac"].to_numpy() > 0.5)
    assert np.nanmedian(f_bad["dP_dt_loaded"].to_numpy()[bad]) < np.nanmedian(
        f_ok["dP_dt_loaded"].to_numpy()[ok]
    )
    assert np.nanmedian(f_bad["I_loaded_mean"].to_numpy()[bad]) > np.nanmedian(
        f_ok["I_loaded_mean"].to_numpy()[ok]
    )


# ------------------------------------------------------------------ failure + labels


def test_functional_failure_is_written_back_and_logged():
    service = _service(2)
    traj = _trajectory("air_leak", onset_d=0.2, fail_d=1.0, gamma=1.0)
    faults = PN.PneumaticFaults.from_trajectories([traj])
    assert traj.t_functional_failure is None
    long, feats, events = PN.simulate(
        PN.PneumaticParams(), faults, service, np.random.default_rng(SEED), store_every=120
    )
    assert traj.t_functional_failure is not None
    assert traj.t_onset < traj.t_functional_failure <= 2 * DAY
    ff = events[events["event"].astype(str) == "functional_failure"]
    assert len(ff) == 1
    # the fault log the caller builds must validate with the written-back timestamp
    row = traj.to_fault_log_row(run_id="r", train_id=service.train_id, car=0, t0=service.t0)
    log = S.coerce_fault_log(pd.DataFrame([row]))
    S.validate_fault_log(log, strict=True)
    assert pd.notna(log["t_functional_failure"].iloc[0])


def test_labels_are_monotone_and_bounded(leaky):
    _, feats, _, _, _ = leaky
    sev = feats["severity"].to_numpy()
    assert np.all(np.diff(sev) >= -1e-6), "clean severity must stay monotone"
    assert sev.min() == 0.0 and sev.max() <= 1.0
    rul = feats["rul_s"].to_numpy()
    assert np.all(np.diff(rul) <= 1e-3) and rul.min() >= 0.0
    assert (feats["is_faulty"].to_numpy() == (sev > 0.0)).all()
    ftypes = set(feats["fault_type"].astype(str))
    assert ftypes <= set(S.FAULT_TYPES["pneumatic"])
    assert ftypes == {"healthy", "air_leak"}


def test_healthy_run_has_no_labels_beyond_healthy(healthy):
    _, feats, _, _, _ = healthy
    assert set(feats["fault_type"].astype(str)) == {"healthy"}
    assert not feats["is_faulty"].any()
    assert not feats["alarm_window_3d"].any()
    assert feats["severity"].to_numpy().max() == 0.0
    assert feats["rul_s"].isna().all()


# ------------------------------------------------------------------ contract mechanics


def test_store_every_thins_telemetry_but_never_features():
    service = _service(1)
    out = {}
    for every in (1, 6):
        long, feats, _ = PN.simulate(
            PN.PneumaticParams(), None, service, np.random.default_rng(SEED), store_every=every
        )
        out[every] = (long, feats)
    assert len(out[1][1]) == len(out[6][1]), "features are emitted for every compressor cycle"
    n1 = int((out[1][0]["subsystem"] == "pneumatic").sum())
    n6 = int((out[6][0]["subsystem"] == "pneumatic").sum())
    assert 0 < n6 < n1 / 3
    # the context rows are thinned with the same mask
    ctx1 = int((out[1][0]["subsystem"] == "train").sum())
    ctx6 = int((out[6][0]["subsystem"] == "train").sum())
    assert ctx6 / ctx1 == pytest.approx(n6 / n1, rel=0.05)


def test_bit_reproducible_for_a_fixed_seed():
    service = _service(1)
    runs = [
        PN.simulate(PN.PneumaticParams(), None, service, np.random.default_rng(11), store_every=30)
        for _ in range(2)
    ]
    pd.testing.assert_frame_equal(runs[0][0], runs[1][0])
    pd.testing.assert_frame_equal(runs[0][1], runs[1][1])


def test_one_simulated_day_is_well_inside_the_budget():
    service = _service(1)
    t = time.perf_counter()
    PN.simulate(PN.PneumaticParams(), None, service, np.random.default_rng(3), store_every=30)
    assert time.perf_counter() - t < 10.0


def test_params_reject_nonsense():
    with pytest.raises(ValueError, match="P_lps_bar < P_start_bar < P_stop_bar"):
        PN.PneumaticParams(P_stop_bar=7.5)
    with pytest.raises(ValueError, match="V_res_l"):
        PN.PneumaticParams(V_res_l=0.0)
    with pytest.raises(ValueError, match="leak_d1_mm"):
        PN.PneumaticParams(leak_d1_mm=0.1)
    with pytest.raises(ValueError, match="burst_draw_prob"):
        PN.PneumaticParams(burst_draw_prob=0.0)
    with pytest.raises(ValueError, match="big_event_prob"):
        PN.PneumaticParams(big_event_prob=1.0)
    with pytest.raises(ValueError, match="burst_draw_prob \\+ big_event_prob"):
        PN.PneumaticParams(burst_draw_prob=0.9, big_event_prob=0.5, big_event_mult=1.0)
    with pytest.raises(ValueError, match="burst_shape_k/burst_max_nls"):
        PN.PneumaticParams(burst_max_nls=0.0)
    with pytest.raises(ValueError, match="unit-mean burst mixture"):
        PN.PneumaticParams(big_event_prob=0.01, big_event_mult=200.0)
    with pytest.raises(ValueError, match="aux_active_frac"):
        PN.PneumaticParams(aux_active_frac=1.5)
    with pytest.raises(ValueError, match="aux_block_s"):
        PN.PneumaticParams(aux_block_s=0.0)
    with pytest.raises(ValueError, match="no SensorSpec"):
        PN.PneumaticParams(sensors={"TP2": PN.PneumaticParams().sensors["TP2"]})
    assert PN.PneumaticParams().with_(V_res_l=800.0).V_res_l == 800.0


def test_faults_container_rejects_mismatched_trajectories():
    door = DegradationTrajectory(
        fault_type="friction", subsystem="door", component_id="door_L1", t_onset=0.0, t_failure=DAY
    )
    with pytest.raises(ValueError, match="not simulated"):
        PN.PneumaticFaults.from_trajectories([door])
    leak = _trajectory("air_leak", onset_d=0.1, fail_d=1.0)
    with pytest.raises(ValueError, match="two 'air_leak' trajectories"):
        PN.PneumaticFaults.from_trajectories([leak, leak])
    with pytest.raises(ValueError, match="step fault"):
        PN.PneumaticFaults(
            dryer_valve_stuck=DegradationTrajectory(
                fault_type="dryer_valve_stuck", subsystem="pneumatic", component_id="apu_1",
                t_onset=0.1 * DAY, t_failure=DAY, shape="power",
            )
        )
    with pytest.raises(ValueError, match="expected 'air_leak'"):
        PN.PneumaticFaults(air_leak=_trajectory("oil_leak", onset_d=0.1, fail_d=1.0))
    assert not PN.PneumaticFaults()
    assert PN.PneumaticFaults.from_trajectories([leak]).active() == (leak,)
    assert PN.PneumaticFaults.from_trajectories(None).active() == ()


def test_simulate_rejects_a_foreign_component_and_bad_store_every():
    service = _service(1)
    with pytest.raises(ValueError, match="store_every"):
        PN.simulate(PN.PneumaticParams(), None, service, np.random.default_rng(0), store_every=0)
    with pytest.raises(ValueError, match="not legal for 'pneumatic'"):
        PN.simulate(PN.PneumaticParams(), None, service, np.random.default_rng(0), component_id="door_L1")
    with pytest.raises(TypeError, match="PneumaticParams"):
        PN.simulate(object(), None, service, np.random.default_rng(0))


def test_selftest_entry_point_reports_the_anchors():
    report = PN.selftest(days=1, seed=1, store_every=60)
    assert 0.07 <= report["duty_in_service"] <= 0.16
    assert 5.6 <= report["I_loaded_mean_A"] <= 6.3
    assert 55.0 <= report["T_oil_mean_C"] <= 85.0
    assert report["n_features"] > 50 and report["n_events"] > 50
    assert PN.main(["--days", "1", "--store-every", "60"]) == 0


# ------------------------------------------------------------------ LPS reachability and the
# ------------------------------------------------------------------ post-failure regime


@pytest.fixture(scope="module")
def lps_ramp():
    """A single air-leak trajectory ramped to s = 1 over three days.

    This is the acceptance case for rail_phm 4.2's lead-time table, whose air-leak row is
    "detected >= 150 min **before LPS fires**" [R93]: LPS has to be reachable from ONE air
    leak, with no help from a second fault. It is only reachable because ``Q_comp_nls`` and
    ``leak_d1_mm`` were sized against each other - see the module docstring.
    """
    return _run("air_leak", days=3, store_every=60, onset_d=0.3, fail_d=2.0, gamma=1.0)


def test_a_single_air_leak_eventually_asserts_lps(lps_ramp):
    long, feats, events, faults, service = lps_ramp
    lps = events[events["event"] == "lps"]
    assert len(lps) > 0, "a lone air leak must be able to trip LPS"
    # the event log is the evidence: telemetry is thinned by store_every, and the terminal
    # excursion falls in the trailing continuous run, which is not a complete cycle
    p_at_trip = [json.loads(d)["P_res"] for d in lps["detail_json"]]
    # the event detail rounds P_res to 3 dp, so a trip exactly at the threshold reads 7.000
    assert min(p_at_trip) < PN.PneumaticParams().P_lps_bar
    assert max(p_at_trip) <= PN.PneumaticParams().P_lps_bar + 1e-3
    # ... and LPS is a *late* symptom: nothing trips while the leak is still small
    first_lps = lps["timestamp"].min()
    assert (first_lps - pd.Timestamp(service.t0)).total_seconds() > 0.5 * DAY


def test_the_compressor_loses_the_band_before_lps_not_at_it(lps_ramp):
    """The failure is progressive: duty -> 1 first, LPS only once the leak beats delivery.

    With the MetroPT-3-calibrated leak endpoint the compressor stops cycling altogether at
    ``s ~ 0.73`` - past that it is one continuous run, the trailing partial cycle is dropped,
    and **no feature rows exist above that severity**. That is the physics (a leak that beats
    the compressor has no duty cycle left to report) and it is a real constraint on the RUL
    training set, so the test asserts it explicitly rather than hiding it.
    """
    _, feats, _, _, _ = lps_ramp
    svc = feats["in_service_frac"].to_numpy() > 0.5
    sev = feats["severity"].to_numpy()
    mid = svc & (sev > 0.3) & (sev < 0.5)
    top = svc & (sev > 0.65)
    assert mid.any() and top.any()
    assert sev.max() < 0.95, "cycles must die out before s = 1 with the calibrated endpoint"
    assert np.nanmean(feats["duty_ratio"].to_numpy()[mid]) < np.nanmean(
        feats["duty_ratio"].to_numpy()[top]
    )
    assert np.nanmean(feats["duty_ratio"].to_numpy()[top]) > 0.8
    assert np.nanmedian(feats["idle_run_ratio"].to_numpy()[top]) < 0.1


def test_post_failure_rows_are_labelled_rather_than_truncated(lps_ramp):
    """``simulate`` never truncates; the caller decides. Contract for scripts/generate.py."""
    _, feats, _, faults, service = lps_ramp
    t_func = faults.active()[0].t_functional_failure
    assert t_func is not None
    t_end_s = (pd.DatetimeIndex(feats["t_end"]) - pd.Timestamp(service.t0)).total_seconds()
    post = t_end_s > t_func
    assert post.any(), "the run must continue past functional failure"
    assert (feats["rul_s"].to_numpy()[post] == 0.0).all()
    assert feats["is_faulty"].to_numpy()[post].all()
    assert (feats["rul_s"].to_numpy()[~post] > 0.0).all()


def test_oil_node_saturates_at_the_thermal_protection_ceiling():
    """Past functional failure the sump must stop at a physical number, not integrate away."""
    service = _service(2)
    tl = service.timeline(dt=1.0)
    n = len(tl)
    p = PN.PneumaticParams()
    cons = PN._consumption(service, tl.t, tl.seg_id, p)
    sev = {name: np.zeros(n) for name in PN.PNEUMATIC_FAULTS}
    for name in ("air_leak", "compressor_wear", "oil_leak"):
        sev[name][:] = 1.0

    capped, _, _ = PN._run_core(n, cons["demand"], tl.T_amb, tl.in_service, sev, p)
    uncapped, _, _ = PN._run_core(
        n, cons["demand"], tl.T_amb, tl.in_service, sev, p.with_(T_oil_max_c=1e4)
    )
    assert uncapped["Oil_temperature"].max() > p.T_oil_max_c, "the cap has to actually bind"
    assert capped["Oil_temperature"].max() == pytest.approx(p.T_oil_max_c)
    assert capped["Oil_temperature"].max() > p.T_oil_fail_c  # failure still detectable
    # the ceiling must not disturb the pressure side: identical reservoir trace
    assert np.allclose(capped["Reservoirs"], uncapped["Reservoirs"])
    assert np.array_equal(capped["state"], uncapped["state"])
    with pytest.raises(ValueError, match="T_oil_max_c"):
        PN.PneumaticParams(T_oil_max_c=50.0)


def test_healthy_oil_temperature_sits_in_the_metropt_band(healthy):
    """hA_oil_w_per_k is calibrated so the duty-weighted sump lands in MetroPT-3's 55-75 C."""
    _, feats, _, _, _ = healthy
    t_max = feats["T_oil_max"].to_numpy()
    assert 55.0 <= np.nanmedian(t_max) <= 75.0, np.nanmedian(t_max)
    assert np.nanmax(t_max) < PN.PneumaticParams().T_oil_fail_c


# ------------------------------------------------- heterogeneous demand (MEASURED, 15 Sep 2026)
#
# MetroPT-3's healthy cycles are NOT interchangeable: ``t_off`` spreads 0-1,814 s (sd 600 s,
# median 903, 17 % of cycles at zero), ``duty_ratio`` spreads 0.015-0.99, the per-cycle
# quiescent OFF draw spreads 0.205-0.351-0.97 NL/s (p95-median-p5) and the draw per phase is
# 0.733 (loaded) / 0.473 (unloaded) / 0.342 (off) NL/s. Until 15 Sep 2026 every stop drew the
# same ``brake_nl_per_stop`` and every dwell the same air-spring share, so every simulated
# cycle landed on the mean and the two-sample KS distances stayed at 0.46-0.75 even where the
# medians agreed to 3 %. These tests pin the replacement, whose defining property is that it
# changes the DISTRIBUTION and not one calibrated MEAN.


@pytest.fixture(scope="module")
def healthy_21d():
    """One 21-day healthy run - the same days/seed ``scripts/calibrate_pneumatic.py`` uses."""
    rng = np.random.default_rng(3)
    service = generate_service(21, rng, train_id="T01")
    _, feats, _ = PN.simulate(
        PN.PneumaticParams(), None, service, rng, store_every=10**6, run_id="test_21d"
    )
    return feats


def test_burst_multiplier_mixture_is_unit_mean_and_heavy_tailed():
    """``E[M] = 1`` exactly, so ``brake_nl_per_stop`` keeps meaning what it was calibrated as."""
    p = PN.PneumaticParams()
    m = PN._burst_multipliers(np.random.default_rng(0), 400_000, p)
    assert m.mean() == pytest.approx(1.0, rel=0.05)
    # most events draw nothing at all (a blended ED brake takes the stop)
    assert (m == 0.0).mean() == pytest.approx(1.0 - p.burst_draw_prob - p.big_event_prob, abs=0.01)
    assert np.median(m) == 0.0
    # ... and the mass is in a tail, not in a spike at the mean
    assert m.max() > 100.0
    assert (m > 10.0).mean() < 0.05
    assert m.min() >= 0.0
    assert PN._burst_multipliers(np.random.default_rng(0), 0, p).size == 0


def test_consumption_keeps_every_calibrated_mean_and_only_changes_the_spread():
    """The whole point: same air, different distribution.

    ``rng=None`` is the *expected* schedule (what ``fit_consumption`` fits its hour bands
    against) and must still be exactly one ``brake_nl_per_stop`` per run and one air-spring
    share per dwell; ``rng`` given must integrate to the same air.
    """
    p = PN.PneumaticParams()
    service = _service(7)
    tl = service.timeline(dt=1.0)
    seg = service.segments
    kind = seg["kind"].to_numpy().astype(str)
    load = seg["load_frac"].to_numpy(dtype=np.float64)
    n_run = int((kind == "run").sum())
    spring_nl = p.spring_nl_per_dwell + p.spring_nl_per_unit_load * np.abs(
        np.diff(load, prepend=load[0])
    )
    expected = n_run * p.brake_nl_per_stop + float(spring_nl[kind == "dwell"].sum())

    flat = PN._consumption(service, tl.t, tl.seg_id, p)
    # to the litre: one mean brake application per run, one mean levelling per dwell. (The
    # split between the two channels can move a few seconds' worth when a brake application
    # stretched by the supply-line choke runs into the following dwell, so the invariant is
    # the total; the calibration's hour-band basis only ever sees the total.)
    # (rel 1e-4, not exact: an event still in flight when the timeline ends is truncated)
    assert float((flat["brake"] + flat["spring"]).sum()) == pytest.approx(expected, rel=1e-4)
    assert float((flat["brake"] + flat["spring"]).max()) <= p.burst_max_nls + 1e-9
    assert np.allclose(flat["aux"][tl.in_service.astype(bool)], p.aux_nls)

    drawn = PN._consumption(service, tl.t, tl.seg_id, p, np.random.default_rng(5))
    burst_flat = float((flat["brake"] + flat["spring"]).sum())
    burst_drawn = float((drawn["brake"] + drawn["spring"]).sum())
    assert burst_drawn == pytest.approx(burst_flat, rel=0.30)  # same budget, Monte-Carlo n
    assert float(drawn["aux"].mean()) == pytest.approx(float(flat["aux"].mean()), rel=0.20)
    # ... but nothing like the same schedule. Both are choked at the same ``burst_max_nls``,
    # so the same air occupies the same total time either way; what changes is how it is
    # PARCELLED - few long episodes instead of one short one per stop.
    def episodes(x: np.ndarray) -> np.ndarray:
        live = np.concatenate([[0], (x > 0.0).astype(np.int8), [0]])
        edges = np.flatnonzero(np.diff(live) != 0).reshape(-1, 2)
        return edges[:, 1] - edges[:, 0]

    e_flat, e_drawn = episodes(flat["brake"] + flat["spring"]), episodes(drawn["brake"] + drawn["spring"])
    assert e_drawn.size < 0.4 * e_flat.size  # far fewer events...
    assert e_flat.max() < 60  # ... one stop's worth each, 11-19 s
    assert e_drawn.max() > 600  # ... against multi-minute episodes


def test_supply_line_choke_bounds_the_burst_and_loses_no_air():
    """One pipe feeds brake and springs, so demands queue instead of adding without limit.

    The bound is not cosmetic: without it two overlapping large events hold the draw above the
    compressor's net delivery long enough to walk the sump through the 95 C functional-failure
    line on a *healthy* run.
    """
    p = PN.PneumaticParams()
    service = _service(7)
    tl = service.timeline(dt=1.0)
    cons = PN._consumption(service, tl.t, tl.seg_id, p, np.random.default_rng(5))
    burst = cons["brake"] + cons["spring"]
    assert float(burst.max()) <= p.burst_max_nls + 1e-9
    assert p.burst_max_nls < p.Q_comp_nls * (1.0 - p.purge_frac)  # the reservoir can keep up
    # queued, not discarded: at most one event's worth is still in the pipe at the end
    raw = PN._spread_events(
        len(tl), 1.0, float(tl.t[0]),
        service.segments["t_end"].to_numpy(dtype=np.float64) - p.brake_s,
        np.where(service.segments["kind"].to_numpy().astype(str) == "run", p.brake_nl_per_stop, 0.0),
        np.full(len(service.segments), p.brake_s),
        p,
    )
    assert float(raw.sum()) == pytest.approx(
        int((service.segments["kind"].to_numpy().astype(str) == "run").sum()) * p.brake_nl_per_stop,
        rel=1e-6,
    )
    below = 0.5 * p.burst_max_nls * np.ones(1000, dtype=np.float64)
    assert PN._choke(below, 1.0, p) is below  # nothing to queue when every draw is under the cap
    over = np.zeros(1000, dtype=np.float64)
    over[10:20] = 4.0 * p.burst_max_nls
    served = PN._choke(over, 1.0, p)
    assert float(served.max()) <= p.burst_max_nls + 1e-9
    assert float(served.sum()) == pytest.approx(float(over.sum()), rel=1e-9)  # queued, not lost


def test_t_off_spreads_like_metropt3_over_a_21_day_run(healthy_21d):
    """MEASURED: real ``t_off`` sd is 600 s with 17 % of cycles at zero; ours was 299 s and 0 %.

    The KS distance against MetroPT-3's 3,141 healthy cycles falls 0.480 -> 0.091 on ``t_off``
    and 0.620 -> 0.226 on ``dP_dt_off`` because of this spread; see
    ``scripts/calibrate_pneumatic.py`` and docs/parameters.md pneumatic section 4.
    """
    t_off = healthy_21d["t_off"].to_numpy(dtype=np.float64)
    assert t_off.size > 800
    assert float(np.std(t_off)) > 400.0, float(np.std(t_off))
    assert float(np.mean(t_off == 0.0)) > 0.05  # cycles a big consumer robs of their OFF phase
    assert 700.0 <= float(np.median(t_off)) <= 1100.0  # MetroPT-3: 903 s
    assert float(t_off.max()) > 1400.0
    # the OFF decay inherits the spread, which is what the KS on dP_dt_off measures
    decay = healthy_21d["dP_dt_off"].to_numpy(dtype=np.float64)
    decay = decay[np.isfinite(decay)]
    assert -0.00145 <= float(np.median(decay)) <= -0.00105  # MetroPT-3: -0.001228 bar/s
    assert float(np.percentile(decay, 5)) < -0.0025  # MetroPT-3: -0.003385 bar/s


def test_the_three_compressor_phases_draw_different_amounts(healthy_21d):
    """MEASURED per phase: 0.733 (loaded) / 0.473 (unloaded) / 0.342 (off) NL/s.

    The phases cannot be equal, because the compressor loads BECAUSE the demand is high: an
    event large enough to matter pulls the reservoir below ``P_start`` and therefore spends its
    air inside a loaded window by construction. Nothing in ``_run_core`` is told this - it
    falls out of the pressure feedback once the schedule is heavy-tailed. The loaded and OFF
    legs are readable straight off the feature table; the full three-way row (0.599 / 0.516 /
    0.348 NL/s simulated) is printed by ``scripts/calibrate_pneumatic.py::sim_phase_demand_nls``.
    """
    p = PN.PneumaticParams()
    v_over_p = p.V_res_l / p.P_atm_bar
    net = p.Q_comp_nls * (1.0 - p.purge_frac)
    loaded = net - float(np.nanmedian(healthy_21d["dP_dt_loaded"].to_numpy())) * v_over_p
    off = -float(np.nanmedian(healthy_21d["dP_dt_off"].to_numpy())) * v_over_p
    assert loaded > 1.4 * off, (loaded, off)
    assert 0.25 <= off <= 0.45  # MetroPT-3: 0.342 NL/s, and the leak alone is 0.27
    assert 0.45 <= loaded <= 0.95  # MetroPT-3: 0.733 NL/s
