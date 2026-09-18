"""Tests for scripts/calibrate_pneumatic.py - the MetroPT-3 -> APU-simulator calibration.

Everything except one small integration test runs on a **synthetic MetroPT-3-shaped frame**
built by :func:`_fake_frame`, whose true constants are known exactly, so each estimator can be
checked for *recovery* rather than merely for not crashing. The integration test reads the first
few days of the real CSV and is skipped when ``data/raw/metropt3`` is absent. The whole module
stays far inside the 60 s budget because nothing here reads the full 218 MB file.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nebulax.sim import pneumatic as PN

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "calibrate_pneumatic.py"
RAW_DIR = REPO_ROOT / "data" / "raw" / "metropt3"
CSV_NAME = "MetroPT3(AirCompressor).csv"


def _load_module():
    spec = importlib.util.spec_from_file_location("calibrate_pneumatic", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["calibrate_pneumatic"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cp():
    return _load_module()


# ------------------------------------------------------------------ the synthetic record

#: Ground truth of :func:`_fake_frame`. Every estimator in the script has to recover these.
TRUE = {
    "p_start": 8.0,
    "p_stop": 10.0,
    "t_loaded": 100.0,
    "t_unloaded": 400.0,
    "t_off": 900.0,
    "i_off": 0.05,
    "i_unloaded": 3.8,
    "i_loaded": 6.0,
    "tower_period": 60.0,
    "oil_tau": 800.0,
    "oil_sink": 50.0,
    "oil_rise_unloaded": 10.0,
    "oil_rise_loaded": 40.0,
    "dt": 10.0,
}


def _fake_frame(
    t0: str = "2020-02-01", days: float = 20.0, leak: float = 1.0, seed: int = 0
) -> pd.DataFrame:
    """A MetroPT-3-shaped 10 s record with known constants.

    One saw-tooth cycle is ``t_loaded`` of charging from ``p_start`` to ``p_stop`` followed by
    ``t_unloaded`` and ``t_off`` of linear decay back down, with the current and the oil node
    driven off the same state sequence. ``leak`` scales the decay rate (and shortens the fall),
    which is how the failure window is made.
    """
    rng = np.random.default_rng(seed)
    dt = TRUE["dt"]
    n = int(days * 86400 / dt)
    t = np.arange(n, dtype=np.float64) * dt
    cycle = TRUE["t_loaded"] + (TRUE["t_unloaded"] + TRUE["t_off"]) / leak
    phase = np.mod(t, cycle)
    loaded = phase < TRUE["t_loaded"]
    unloaded = ~loaded & (phase < TRUE["t_loaded"] + TRUE["t_unloaded"] / leak)
    off = ~loaded & ~unloaded
    band = TRUE["p_stop"] - TRUE["p_start"]
    press = np.where(
        loaded,
        TRUE["p_start"] + band * phase / TRUE["t_loaded"],
        TRUE["p_stop"] - band * (phase - TRUE["t_loaded"]) / ((TRUE["t_unloaded"] + TRUE["t_off"]) / leak),
    )
    current = np.where(loaded, TRUE["i_loaded"], np.where(unloaded, TRUE["i_unloaded"], TRUE["i_off"]))
    # oil node, integrated with the same first-order law the script fits
    rise = np.where(loaded, TRUE["oil_rise_loaded"], np.where(unloaded, TRUE["oil_rise_unloaded"], 0.0))
    b = dt / TRUE["oil_tau"]
    oil = np.empty(n)
    T = TRUE["oil_sink"]
    for k in range(n):
        T += b * (TRUE["oil_sink"] + rise[k] - T)
        oil[k] = T
    # towers flip every tower_period of LOADED time
    cum_loaded = np.cumsum(loaded.astype(np.float64)) * dt
    towers = (np.floor(cum_loaded / TRUE["tower_period"]) % 2.0)
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp(t0) + pd.to_timedelta(t, unit="s"),
            "TP2": np.where(loaded, press + 0.3, 0.0),
            "TP3": press,
            "H1": press - 0.15,
            "DV_pressure": np.where(unloaded, press, 0.0),
            "Reservoirs": press,
            "Oil_temperature": oil,
            "Motor_current": current,
            "COMP": (~loaded).astype(float),
            "DV_eletric": loaded.astype(float),
            "Towers": towers,
            "MPG": (press < TRUE["p_start"] + 0.05).astype(float),
            "LPS": (press < 7.0).astype(float),
            "Pressure_switch": unloaded.astype(float),
            "Oil_level": np.ones(n),
            "Caudal_impulses": np.ones(n),
        }
    )
    assert rng is not None
    return frame


@pytest.fixture(scope="module")
def fake(cp):
    """A normal window plus a June 'failure' window with a 4x leak, as one file."""
    normal = _fake_frame("2020-02-01", days=20.0, leak=1.0)
    failing = _fake_frame("2020-05-29", days=9.0, leak=4.0, seed=1)
    return pd.concat([normal, failing], ignore_index=True)


# ------------------------------------------------------------------ windowing


def test_the_normal_window_is_the_failure_free_stretch(cp, fake):
    mask, n_excluded = cp.normal_window_mask(fake["timestamp"])
    kept = fake.loc[mask, "timestamp"]
    assert kept.min() >= pd.Timestamp(cp.NORMAL_WINDOW[0])
    assert kept.max() < pd.Timestamp(cp.NORMAL_WINDOW[1])
    # every UCI episode is later than 31 March, so the +/-3 day guard removes nothing - that
    # is the whole reason Feb-Mar is the calibration window, and it is asserted, not assumed
    assert n_excluded == 0
    assert len(cp.FAILURE_EPISODES) == 4
    assert cp.EXCLUSION_DAYS == 3.0


def test_the_guard_band_really_does_cut_when_a_failure_is_inside_it(cp):
    """Same guard, moved onto a window that does overlap an episode."""
    ts = pd.Series(pd.date_range("2020-04-16", "2020-04-21", freq="10s"))
    guard = pd.Timedelta(days=cp.EXCLUSION_DAYS)
    a, b = (pd.Timestamp(x) for x in cp.FAILURE_EPISODES[0])
    inside = ((ts >= a - guard) & (ts <= b + guard)).to_numpy()
    assert inside.all(), "this whole span is inside the 18 Apr guard band"


def test_segment_states_uses_the_adapter_thresholds(cp):
    from nebulax.adapters import metropt3 as M

    assert (cp.I_OFF_MAX, cp.I_LOADED_MIN) == (M._I_OFF_MAX, M._I_LOADED_MIN)
    st = cp.segment_states(np.array([0.0, 1.99, 2.0, 4.9, 5.0, 9.3]))
    assert st.tolist() == [0, 0, 1, 1, 2, 2]


# ------------------------------------------------------------------ cycle extraction


def test_extract_cycles_recovers_the_synthetic_durations(cp, fake):
    mask, _ = cp.normal_window_mask(fake["timestamp"])
    cycles = cp.extract_cycles(fake[mask].reset_index(drop=True))
    assert len(cycles) > 1000
    for key in ("t_loaded", "t_unloaded", "t_off"):
        assert cycles[key].median() == pytest.approx(TRUE[key], abs=TRUE["dt"])
    assert cycles["P_res_min"].median() == pytest.approx(TRUE["p_start"], abs=0.05)
    assert cycles["P_res_max"].median() == pytest.approx(TRUE["p_stop"], abs=0.05)
    # the endpoint rates must add up to the band across the cycle
    band = TRUE["p_stop"] - TRUE["p_start"]
    assert cycles["charge_rate"].median() == pytest.approx(band / TRUE["t_loaded"], rel=0.15)
    assert cycles["fall_rate"].median() == pytest.approx(
        band / (TRUE["t_unloaded"] + TRUE["t_off"]), rel=0.05
    )
    assert cycles["off_rate"].median() < 0.0 and cycles["unloaded_rate"].median() < 0.0


def test_extract_cycles_refuses_a_window_with_no_complete_cycle(cp):
    flat = _fake_frame(days=0.002)
    flat["Motor_current"] = 0.0
    with pytest.raises(ValueError, match="no complete compressor cycle"):
        cp.extract_cycles(flat)


def test_a_logger_gap_is_not_counted_as_compressor_off_time(cp):
    """The file has 190 gaps over 30 min; they must not inflate ``t_off``."""
    frame = _fake_frame(days=0.5)
    gapped = pd.concat([frame.iloc[:200], frame.iloc[200:]], ignore_index=True)
    gapped.loc[200:, "timestamp"] = gapped.loc[200:, "timestamp"] + pd.Timedelta(hours=2)
    cycles = cp.extract_cycles(gapped)
    assert cycles["t_off"].max() <= TRUE["t_off"] + cp.MAX_SAMPLE_DT_S + TRUE["dt"]


# ------------------------------------------------------------------ the estimators


def test_effective_volume_inverts_the_receiver_relation(cp, fake):
    """``V`` recovered from the rates must reproduce the rates it was derived from."""
    mask, _ = cp.normal_window_mask(fake["timestamp"])
    m = cp.measure(fake[mask].reset_index(drop=True), fake)
    p = PN.PneumaticParams()
    v = cp.effective_volume_l(m, p)
    assert v > 0.0
    net = p.Q_comp_nls * (1.0 - p.purge_frac)
    assert m.compressor_rate == pytest.approx(net * p.P_atm_bar / v, rel=1e-9)
    # ... and it is a pure ratio: double the pinned delivery, double the volume
    assert cp.effective_volume_l(m, p.with_(Q_comp_nls=2.0 * p.Q_comp_nls)) == pytest.approx(2.0 * v)


def test_oil_thermal_fit_recovers_the_synthetic_node(cp, fake):
    mask, _ = cp.normal_window_mask(fake["timestamp"])
    frame = fake[mask].reset_index(drop=True)
    t = frame["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64).astype(float)
    raw_dt = np.diff(t, append=t[-1] + TRUE["dt"])
    state = cp.segment_states(frame["Motor_current"].to_numpy())
    beta, r2, b = cp.fit_oil_thermal(t, frame["Oil_temperature"].to_numpy(), state, raw_dt)
    assert 1.0 / b == pytest.approx(TRUE["oil_tau"], rel=0.05)
    sink = beta[cp.STATE_OFF] / b
    assert sink == pytest.approx(TRUE["oil_sink"], abs=0.5)
    assert beta[cp.STATE_LOADED] / b - sink == pytest.approx(TRUE["oil_rise_loaded"], rel=0.1)
    assert beta[cp.STATE_UNLOADED] / b - sink == pytest.approx(TRUE["oil_rise_unloaded"], rel=0.15)
    assert r2 > 0.5


def test_measure_reads_the_band_currents_and_tower_period(cp, fake):
    mask, _ = cp.normal_window_mask(fake["timestamp"])
    m = cp.measure(fake[mask].reset_index(drop=True), fake)
    assert m.p_load_sample == pytest.approx(TRUE["p_start"], abs=0.05)
    assert m.p_unload_sample == pytest.approx(TRUE["p_stop"], abs=0.05)
    # the lag correction pushes the switch points outward, never inward
    assert m.p_start <= m.p_load_sample and m.p_stop >= m.p_unload_sample
    assert (m.i_off, m.i_unloaded, m.i_loaded) == pytest.approx(
        (TRUE["i_off"], TRUE["i_unloaded"], TRUE["i_loaded"])
    )
    assert m.tower_period_s == pytest.approx(TRUE["tower_period"], abs=TRUE["dt"])
    assert m.duty_cycles == pytest.approx(
        TRUE["t_loaded"] / (TRUE["t_loaded"] + TRUE["t_unloaded"] + TRUE["t_off"]), rel=0.1
    )


def test_the_failure_window_shows_the_t_off_shrinkage_that_sets_s_equals_one(cp, fake):
    mask, _ = cp.normal_window_mask(fake["timestamp"])
    m = cp.measure(fake[mask].reset_index(drop=True), fake)
    # the synthetic June window leaks 4x, so its OFF phase must be ~4x shorter
    assert m.t_off_leak_window_s < 0.5 * m.t_off_s
    assert m.leak_window_duty > m.duty_cycles


def test_leak_endpoint_puts_s_equals_one_exactly_at_the_lps_trip(cp, fake):
    mask, _ = cp.normal_window_mask(fake["timestamp"])
    m = cp.measure(fake[mask].reset_index(drop=True), fake)
    p = PN.PneumaticParams()
    aux = 0.076
    ends = cp.leak_endpoint_mm(m, p, aux)
    # by construction, at the chosen endpoint the leak at 7 bar equals the net delivery
    leak_at_trip = float(PN.leak_nls(p.P_lps_bar + p.P_atm_bar, ends["chosen_d1_mm"], p))
    assert leak_at_trip == pytest.approx(p.Q_comp_nls * (1.0 - p.purge_frac) - aux, rel=1e-6)
    # a lower equilibrium pressure always implies a bigger hole
    assert ends["lps_d1_mm"] > ends["start_d1_mm"]
    assert ends["worst_case_nls"] < 28.3 / 3.0  # [R156] freight bound


def test_measure_sensor_noise_recovers_a_known_sigma(cp):
    frame = _fake_frame(days=2.0)
    rng = np.random.default_rng(3)
    sigma = 0.02
    frame["Reservoirs"] = frame["Reservoirs"] + rng.normal(0.0, sigma, len(frame))
    state = cp.segment_states(frame["Motor_current"].to_numpy())
    t = frame["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64).astype(float)
    raw_dt = np.diff(t, append=t[-1] + TRUE["dt"])
    got, quantum = cp.measure_sensor_noise(frame, state, raw_dt, "Reservoirs")
    assert got == pytest.approx(sigma, rel=0.15)
    assert quantum > 0.0


def test_ks_table_is_zero_for_identical_samples_and_one_for_disjoint(cp):
    a = pd.DataFrame({c: np.linspace(1.0, 2.0, 200) for c in cp.KS_COLUMNS})
    b = pd.DataFrame({c: np.linspace(1.0, 2.0, 200) for c in cp.KS_COLUMNS})
    table = cp.ks_table(a, b)
    assert set(table["feature"]) == set(cp.KS_COLUMNS)
    assert (table["ks"] == 0.0).all()
    far = pd.DataFrame({c: np.linspace(90.0, 99.0, 50) for c in cp.KS_COLUMNS})
    assert (cp.ks_table(a, far)["ks"] == 1.0).all()


# ------------------------------------------------------------------ the leak-ramp detector


def _ramp_table(dip: tuple[float, float] | None, level: float = 6.0, low: float = 0.5) -> pd.DataFrame:
    """A 30-day per-cycle table at a constant healthy level, optionally dipping in ``dip`` days."""
    day = np.arange(0.0, 30.0, 1.0 / 96.0)  # a cycle every 15 min, as the simulator gives
    idle = np.full(day.size, level, dtype=np.float64)
    if dip is not None:
        idle[(day >= dip[0]) & (day < dip[1])] = low
    return pd.DataFrame(
        {
            "day": day,
            "idle_run_ratio": idle,
            "duty_ratio": 1.0 / (1.0 + idle),
            "t_off": idle * 100.0,
            "in_service_frac": np.ones(day.size),
        }
    )


def test_rolling_median_is_trailing_and_respects_min_periods(cp):
    day = np.arange(0.0, 1.0, 1.0 / 24.0)  # hourly
    x = np.arange(day.size, dtype=np.float64)
    out = cp._rolling_median(day, x, window_h=5.0, min_periods=3)
    assert np.isnan(out[:2]).all() and np.isfinite(out[2:]).all()
    # trailing: the last point medians the last five samples, i.e. hours 19..23
    assert out[-1] == pytest.approx(np.median(x[-5:]))


def test_the_alarm_needs_a_sustained_crossing_not_a_dip(cp):
    """The 24 h hold is the whole reason the rule does not fire on a healthy unit."""
    day = np.arange(0.0, 10.0, 1.0 / 96.0)
    stat = np.ones(day.size)
    stat[(day >= 2.0) & (day < 2.2)] = 0.0  # a ~5 h excursion
    assert np.isnan(cp._first_sustained(day, stat, 0.5, True, persist_h=24.0))
    stat[(day >= 5.0) & (day < 7.0)] = 0.0  # a 2-day excursion
    # dated at the END of the hold: the rule cannot declare before it has 24 h of evidence
    assert cp._first_sustained(day, stat, 0.5, True, persist_h=24.0) == pytest.approx(6.0, abs=0.05)
    # ... and a crossing too close to the end of the record cannot be confirmed
    late = np.ones(day.size)
    late[day >= 9.5] = 0.0
    assert np.isnan(cp._first_sustained(day, late, 0.5, True, persist_h=24.0))


def test_leak_ramp_alarm_baselines_on_the_units_own_first_days(cp):
    """A permanent step after the baseline window fires; the same level inside it does not."""
    _, alarm = cp._leak_ramp_alarm(
        _ramp_table((12.0, 30.0)), "idle_run_ratio", True, 5.0, 12.0, 24.0, cp.LEAK_RAMP_BASELINE_DAY
    )
    assert alarm == pytest.approx(13.0, abs=0.6)  # step at day 12 + the 24 h hold
    # a flat record never fires, whatever the percentile: that is the false-alarm audit
    thr, none_alarm = cp._leak_ramp_alarm(
        _ramp_table(None), "idle_run_ratio", True, 5.0, 12.0, 24.0, cp.LEAK_RAMP_BASELINE_DAY
    )
    assert thr == pytest.approx(6.0) and np.isnan(none_alarm)
    # a dip that happens *inside* the baseline window widens the baseline, so it is not an alarm
    _, inside = cp._leak_ramp_alarm(
        _ramp_table((2.0, 4.0)), "idle_run_ratio", True, 5.0, 12.0, 24.0, cp.LEAK_RAMP_BASELINE_DAY
    )
    assert np.isnan(inside)


def test_leak_ramp_run_emits_the_columns_the_lead_time_table_reads(cp):
    tab, ff = cp.leak_ramp_run(2.0, seed=0, days=3, onset_day=0.5, failure_day=2.0)
    assert list(tab.columns) == ["day", "idle_run_ratio", "duty_ratio", "t_off", "in_service_frac"]
    assert len(tab) > 50 and np.all(np.diff(tab["day"].to_numpy()) >= 0.0)
    assert 0.0 < tab["day"].max() <= 3.0
    assert np.isfinite(ff) and 0.5 < ff <= 3.0  # a ramp this steep must reach functional failure
    healthy, ff_ok = cp.leak_ramp_run(None, seed=0, days=3)
    assert np.isnan(ff_ok)
    # the healthy control shares its demand realisation with the ramp up to the onset
    early = tab["day"].to_numpy() < 0.5
    assert np.allclose(tab["t_off"].to_numpy()[early], healthy["t_off"].to_numpy()[: int(early.sum())])


def test_the_documented_detector_settings_are_the_ones_section_3_quotes(cp):
    """docs/parameters.md pneumatic section 3 quotes these five numbers in its prose."""
    assert cp.LEAK_RAMP_WINDOW_H == 12.0
    assert cp.LEAK_RAMP_PERSIST_H == 24.0
    assert cp.LEAK_RAMP_BASELINE_DAY == 8.0
    assert cp.LEAK_RAMP_ONSET_DAY == 8.0 and cp.LEAK_RAMP_FAILURE_DAY == 26.0
    assert [(n, q) for n, _c, _b, q in cp.LEAK_RAMP_DETECTORS] == [("idle_run_ratio", 5.0), ("duty_ratio", 95.0)]


def test_leak_ramp_report_renders_the_false_alarm_row(cp):
    table = pd.DataFrame(
        {
            "detector": ["idle_run_ratio", "idle_run_ratio"],
            "gamma": [2.0, np.nan],
            "seed": [3.0, 3.0],
            "threshold": [5.0, 5.0],
            "alarm_day": [12.0, np.nan],
            "ff_day": [16.0, np.nan],
            "lead_d": [4.0, np.nan],
            "n_cycles": [2000.0, 2000.0],
        }
    )
    table.attrs["settings"] = {
        "days": 30.0,
        "onset_day": 8.0,
        "failure_day": 26.0,
        "window_h": 12.0,
        "persist_h": 24.0,
        "baseline_day": 8.0,
    }
    text = cp.leak_ramp_report(table)
    assert "healthy" in text and "never" in text and "4.00" in text


# ------------------------------------------------------------------ the audit contract


def test_recommend_constants_reports_every_simulator_knob_it_touched(cp, fake):
    mask, _ = cp.normal_window_mask(fake["timestamp"])
    m = cp.measure(fake[mask].reset_index(drop=True), fake)
    updates = cp.recommend_constants(m)
    names = {u.name for u in updates}
    for expected in (
        "V_res_l", "P_start_bar", "P_stop_bar", "t_hold_s", "tower_period_s", "I_loaded_a",
        "I_loaded_kp", "I_loaded_kT", "hA_oil_w_per_k", "P_heat_loaded_w", "T_sump_offset_c",
        "aux_nls", "brake_nl_per_stop", "leak_d0_mm", "leak_d1_mm", "unloaded_vent_nls",
        "Q_comp_nls", "C_oil_j_per_k",
    ):
        assert expected in names, expected
    assert all(u.tag in {"measured", "derived", "calibrated", "ours"} for u in updates)
    assert all(u.evidence for u in updates)
    assert all(u.status in {"CHANGE", "CONFIRMED"} for u in updates)
    # the pinned rows must never be reported as a change
    assert {u.name for u in updates if not u.changed} >= {"Q_comp_nls", "C_oil_j_per_k", "leak_d0_mm"}
    text = cp.report(m, updates, None)
    assert "CONTROL BAND" in text and "OIL NODE" in text and "LEAK ACCEPTANCE WINDOW" in text


def test_the_simulator_defaults_are_the_calibrated_values(cp):
    """Regression guard: these are what ``scripts/calibrate_pneumatic.py`` measured on the real
    file (see ``docs/parameters.md``). If a default drifts, re-run the script, do not edit this
    list - the numbers in the docs, the module docstring and here must agree."""
    p = PN.PneumaticParams()
    assert (p.V_res_l, p.P_start_bar, p.P_stop_bar, p.P_lps_bar) == (290.0, 8.06, 10.2, 7.0)
    assert (p.t_unload_s, p.t_hold_s, p.t_reload_s) == (40.0, 376.0, 3.0)
    assert p.tower_period_s == 59.0
    assert (p.I_unloaded_a, p.I_loaded_a, p.I_loaded_kp) == (3.785, 5.618, 0.2231)
    assert p.I_loaded_kT == -0.0114 and p.T_oil_ref_c == 58.0
    assert (p.hA_oil_w_per_k, p.P_heat_loaded_w, p.P_heat_unloaded_w) == (32.0, 1323.0, 320.0)
    assert p.T_sump_offset_c == 33.5 and p.C_oil_j_per_k == 25_000.0
    assert (p.aux_nls, p.brake_nl_per_stop, p.unloaded_vent_nls) == (0.076, 22.0, 0.098)
    assert (p.spring_nl_per_dwell, p.spring_nl_per_unit_load) == (5.5, 66.0)
    assert (p.leak_d0_mm, p.leak_d1_mm) == (0.55, 2.88)
    assert p.sensors["Reservoirs"].quantum == 0.002
    assert p.sensors["Oil_temperature"].noise_sigma == 0.065
    assert p.sensors["Motor_current"].quantum == 0.0025


def test_params_still_validate_the_two_new_fields():
    with pytest.raises(ValueError, match="unloaded_vent_nls"):
        PN.PneumaticParams(unloaded_vent_nls=-0.1)
    with pytest.raises(ValueError, match="T_sump_offset_c"):
        PN.PneumaticParams(T_sump_offset_c=-1.0)
    assert PN.PneumaticParams().with_(T_sump_offset_c=0.0).T_sump_offset_c == 0.0


# ------------------------------------------------------------------ real-data integration


@pytest.mark.skipif(not (RAW_DIR / CSV_NAME).exists(), reason="data/raw/metropt3 not present")
def test_real_slice_reproduces_the_documented_anchors(cp, tmp_path):
    """The first ~4 days of the real file, not the whole 218 MB - this must stay fast."""
    slice_rows = 36_000
    raw = pd.read_csv(RAW_DIR / CSV_NAME, nrows=slice_rows)
    raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")])
    raw.to_csv(tmp_path / CSV_NAME, index=False)
    frame = cp.load_metropt3(tmp_path)
    assert frame["timestamp"].iloc[0] == pd.Timestamp("2020-02-01")
    cycles = cp.extract_cycles(frame)
    assert len(cycles) > 50
    # the anchors documented in docs/parameters.md, on real data
    assert 8.0 <= cycles["P_res_min"].median() <= 8.15
    assert 10.05 <= cycles["P_res_max"].median() <= 10.2
    assert 90.0 <= cycles["t_loaded"].median() <= 130.0
    assert 400.0 <= cycles["t_unloaded"].median() <= 430.0
    assert 5.6 <= cycles["I_loaded_mean"].median() <= 6.3
    state = cp.segment_states(frame["Motor_current"].to_numpy())
    assert np.median(frame["Motor_current"].to_numpy()[state == cp.STATE_UNLOADED]) == pytest.approx(
        3.785, abs=0.15
    )
    assert cp.load_metropt3(tmp_path).shape[1] == 16


def test_load_metropt3_names_the_expected_file_when_the_dir_is_wrong(cp, tmp_path):
    with pytest.raises(FileNotFoundError, match=re.escape(CSV_NAME)):
        cp.load_metropt3(tmp_path / "nope")
