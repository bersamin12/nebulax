#!/usr/bin/env python
"""Calibrate the pneumatic (APU) simulator against MetroPT-3.

What this does
--------------
Reads ``data/raw/metropt3/MetroPT3(AirCompressor).csv`` (UCI 791, CC BY 4.0 - see
:mod:`nebulax.adapters.metropt3` for the full provenance), cuts the record into
**compressor duty cycles** with exactly the segmentation the adapter uses, and measures,
on a *normal* window only (1 Feb - 31 Mar 2020, minus +/-3 days around every failure
episode in the UCI failure report):

1. **control-band thresholds** - the reservoir pressure at which the unit actually loads
   and unloads, corrected for the file's 10 s sampling lag;
2. **charge slope** while loaded and **OFF decay rate**, both per hour of day;
3. **state durations** - ``t_loaded`` / ``t_unloaded`` / ``t_off`` per cycle;
4. **motor current per state**, with the loaded current regressed on reservoir pressure
   and oil temperature;
5. **oil temperature vs duty**, and a first-order thermal fit
   ``dT/dt = (P_state + hA*T_sink - hA*T)/C`` giving ``hA/C``, the OFF asymptote and the
   per-state heat inputs;
6. the **twin-tower switching period**, measured from the ``TOWERS`` channel itself
   (rail_phm 2.3b: "the switching period does not need to be invented - measure it").

From those it *sets* simulator constants:

* ``V_res_l`` from the charge slope (``effective_volume_l``),
* the consumption schedule by a non-negative least squares fit of the simulator's own
  (aux, burst) demand basis against the measured **demand per hour band**,
* the oil node (``hA_oil_w_per_k``, ``P_heat_loaded_w``, ``P_heat_unloaded_w``,
  ``T_sump_offset_c``) from the thermal regression with ``C_oil_j_per_k`` pinned,
* the **leak orifice endpoint** ``leak_d1_mm`` so that ``s = 1`` reproduces the ``t_off``
  collapse of the 29 May - 7 Jun 2020 episode.

It then runs the simulator on the current defaults, extracts cycles with the *same* code,
and reports **two-sample KS distances** on ``t_loaded``, ``t_off``, ``dP_dt_off``,
``I_loaded_mean`` and ``T_oil_max`` against the MetroPT-3 healthy cycles, plus the **draw per
compressor phase** (``sim_phase_demand_nls``) beside the measured one - the asymmetry
0.733 / 0.473 / 0.342 NL/s that the heavy-tailed burst schedule exists to reproduce.
Re-running the script after the constants are applied is the audit: every row of the constant
table should read ``CONFIRMED``.

Three identifiability facts, all load-bearing
---------------------------------------------
**MetroPT-3 has no flow channel.** ``Flowmeter`` exists only in MetroPT-1/2 [rail_phm 2.1],
so pressure data alone fixes only the *ratio* ``Q_comp / V_res`` - never either alone. We pin
``Q_comp_nls`` (the free air delivery; also the constant the previous review sized against the
leak) and solve for ``V_res_l``. Pinning ``V_res_l = 600 L`` instead would give
``Q_comp = 13.3 NL/s``; both are physical, and every pressure *rate* in the module is
unchanged by the choice. This is stated wherever the derived volume is reported.

**The oil node has two time constants and we model one.** The cycle-scale regression gives
``C/hA = 780 s`` and an OFF asymptote of **52 C**, but the record's own cold starts (oil at
15.4 C on 2020-08-17 and 18.5 C on 2020-03-07, both while charging an empty reservoir from
~1 bar) show the true outdoor ambient was 15-19 C. A single node cannot relax to 52 C on a
780 s constant *and* sit 37 K above outdoor air: there is a slow, massive warm body (the
compressor block and its bay) that the fast oil node relaxes toward. We keep the fast node
and give its sink a measured offset, ``T_sump_offset_c``. Fitting the slow node from the
logger's own multi-hour gaps is **not** identifiable - the compressor state during a gap is
unknown, and the fit lands at rms 9.5 K with ``T_amb`` running from -4 C to +24 C depending
on which gaps are admitted - so it is reported as a known model limitation, not a number.

**OFF windows are a biased-low estimate of demand.** A consumption burst large enough to
matter pulls the pressure down fast enough to *start the compressor*, so it lands inside a
loaded window by construction and never inside an OFF window. The OFF decay therefore measures
the **quiescent** draw (leak + auxiliaries), while the **cycle air balance**
``Q_comp*(1-purge_frac)*duty`` measures the *mean* draw. Both are reported; the consumption
fit uses the second and the leak split uses the first.

Run::

    python scripts/calibrate_pneumatic.py                     # full run + plots
    python scripts/calibrate_pneumatic.py --table out.csv     # also dump the real cycle table
    python scripts/calibrate_pneumatic.py --no-plots --quiet  # numbers only
    python scripts/calibrate_pneumatic.py --no-sim            # skip the sim + KS section
    python scripts/calibrate_pneumatic.py --leak-ramp         # + the section-3 lead-time table
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebulax.sim import pneumatic as PN  # noqa: E402
from nebulax.sim.common import SEC_PER_DAY, DegradationTrajectory, generate_service  # noqa: E402

__all__ = [
    "NORMAL_WINDOW",
    "FAILURE_EPISODES",
    "EXCLUSION_DAYS",
    "Measurements",
    "ConstantUpdate",
    "load_metropt3",
    "normal_window_mask",
    "segment_states",
    "extract_cycles",
    "measure",
    "effective_volume_l",
    "fit_oil_thermal",
    "fit_consumption",
    "leak_endpoint_mm",
    "simulate_cycles",
    "sim_phase_demand_nls",
    "leak_ramp_run",
    "leak_ramp_lead_time",
    "leak_ramp_report",
    "ks_table",
    "recommend_constants",
    "report",
    "main",
]

# ------------------------------------------------------------------------------ windows

#: The normal (failure-free) window the calibration is measured on. February and March 2020
#: precede every episode in the UCI failure report, so this is the longest clean stretch in
#: the file (445,298 rows, 59 days).
NORMAL_WINDOW: Final[tuple[str, str]] = ("2020-02-01", "2020-04-01")

#: UCI failure-report episodes, transcribed from :data:`nebulax.adapters.metropt3._FAULT_REPORTS`.
#: All four are air leaks; MetroPT-3 has no oil-leak episode [rail_phm 2.2].
FAILURE_EPISODES: Final[tuple[tuple[str, str], ...]] = (
    ("2020-04-18T00:00:00", "2020-04-18T23:59:00"),
    ("2020-05-29T23:30:00", "2020-05-30T06:00:00"),
    ("2020-06-05T10:00:00", "2020-06-07T14:30:00"),
    ("2020-07-15T14:30:00", "2020-07-15T19:00:00"),
)

#: Guard band removed around every episode above.
EXCLUSION_DAYS: Final[float] = 3.0

#: The acceptance window rail_phm 4.2 names: "``s = 1`` must reproduce the ``t_off`` shrinkage
#: seen in MetroPT-3 late May - early June 2020".
LEAK_WINDOW: Final[tuple[str, str]] = ("2020-05-29", "2020-06-08")

# ------------------------------------------------------------------------------ segmentation

#: Motor-current state thresholds, identical to :mod:`nebulax.adapters.metropt3` so the two
#: never disagree about what a cycle is. The file's current histogram is three clean clusters
#: with empty gaps at 0.25-3.5 A and 4.25-4.75 A.
I_OFF_MAX: Final[float] = 2.0
I_LOADED_MIN: Final[float] = 5.0

#: A sample interval longer than this is a logger outage, not compressor-off time. The file is
#: nominally 10 s (not the 1 Hz its dataset card claims) with 190 gaps over 30 min.
MAX_SAMPLE_DT_S: Final[float] = 60.0

#: Nominal sampling interval of the file, used to correct the control-band thresholds for the
#: half-interval lag between the true switch and the first sample that shows it.
SAMPLE_DT_S: Final[float] = 10.0

#: A gap longer than this ends a cycle rather than inflating its ``t_off``.
MAX_CYCLE_S: Final[float] = 4.0 * 3600.0

#: Cycle-feature columns compared between MetroPT-3 and the simulator.
KS_COLUMNS: Final[tuple[str, ...]] = (
    "t_loaded",
    "t_off",
    "dP_dt_off",
    "I_loaded_mean",
    "T_oil_max",
)

STATE_OFF, STATE_UNLOADED, STATE_LOADED = 0, 1, 2


# ------------------------------------------------------------------------------ reading


def load_metropt3(raw_dir: Path) -> pd.DataFrame:
    """Read the MetroPT-3 CSV. Same file and timestamp convention as the adapter."""
    from nebulax.adapters.metropt3 import _CSV_NAME, _find_csv

    path = _find_csv(Path(raw_dir))
    usecols = ["timestamp", *PN.S.METROPT3_SIGNALS]
    df = pd.read_csv(path, usecols=usecols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    assert path.name == _CSV_NAME
    return df


def normal_window_mask(ts: pd.Series) -> tuple[np.ndarray, int]:
    """``(mask, n_excluded)`` for :data:`NORMAL_WINDOW` minus the failure guard bands."""
    lo, hi = (pd.Timestamp(x) for x in NORMAL_WINDOW)
    mask = (ts >= lo).to_numpy() & (ts < hi).to_numpy()
    guard = pd.Timedelta(days=EXCLUSION_DAYS)
    excluded = np.zeros(len(ts), dtype=bool)
    for a, b in FAILURE_EPISODES:
        excluded |= ((ts >= pd.Timestamp(a) - guard) & (ts <= pd.Timestamp(b) + guard)).to_numpy()
    n_excluded = int((mask & excluded).sum())
    return mask & ~excluded, n_excluded


def segment_states(current: np.ndarray) -> np.ndarray:
    """OFF / UNLOADED / LOADED (0/1/2) from ``Motor_current``."""
    i = np.asarray(current, dtype=np.float64)
    return np.where(i < I_OFF_MAX, STATE_OFF, np.where(i < I_LOADED_MIN, STATE_UNLOADED, STATE_LOADED)).astype(np.int8)


def _seconds(ts: pd.Series) -> np.ndarray:
    return ts.to_numpy().astype("datetime64[s]").astype(np.int64).astype(np.float64)


# ------------------------------------------------------------------------------ cycles


def _phase_rate(press: np.ndarray, state: np.ndarray, t: np.ndarray, which: int) -> float:
    """Endpoint pressure rate (bar/s) over the samples of one state inside a cycle."""
    idx = np.flatnonzero(state == which)
    if idx.size < 2:
        return float("nan")
    span = t[idx[-1]] - t[idx[0]]
    if span <= 0.0:
        return float("nan")
    return float((press[idx[-1]] - press[idx[0]]) / span)


def extract_cycles(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per compressor duty cycle (load start -> next load start).

    The columns are named for :data:`nebulax.sim.pneumatic.CYCLE_FEATURE_COLUMNS` so the real
    and simulated tables are directly comparable. ``dP_dt_off`` / ``dP_dt_loaded`` are the
    mean per-second pressure rate over the OFF / LOADED samples of the cycle, which is what
    :func:`nebulax.sim.pneumatic._cycle_features` computes at 1 Hz.
    """
    t = _seconds(frame["timestamp"])
    raw_dt = np.diff(t, append=t[-1] + SAMPLE_DT_S)
    dt = np.clip(raw_dt, 0.0, MAX_SAMPLE_DT_S)
    state = segment_states(frame["Motor_current"].to_numpy())
    press = frame["Reservoirs"].to_numpy(dtype=np.float64)
    current = frame["Motor_current"].to_numpy(dtype=np.float64)
    oil = frame["Oil_temperature"].to_numpy(dtype=np.float64)
    lps = frame["LPS"].to_numpy(dtype=np.float64)
    # a per-sample rate is only meaningful across a nominal interval
    clean = (raw_dt >= 0.8 * SAMPLE_DT_S) & (raw_dt <= 1.5 * SAMPLE_DT_S)
    rate = np.where(clean, np.diff(press, append=press[-1]) / np.maximum(raw_dt, 1e-9), np.nan)

    loaded = state == STATE_LOADED
    starts = np.flatnonzero(loaded & ~np.roll(loaded, 1))
    starts = starts[starts > 0]
    rows: list[dict[str, float]] = []
    for a, z in zip(starts[:-1], starts[1:]):
        if t[z] - t[a] > MAX_CYCLE_S:
            continue
        s, d = state[a:z], dt[a:z]
        t_loaded = float(d[s == STATE_LOADED].sum())
        t_unloaded = float(d[s == STATE_UNLOADED].sum())
        t_off = float(d[s == STATE_OFF].sum())
        if t_loaded <= 0.0:
            continue
        ld = s == STATE_LOADED
        # drop the first sample of the loaded run: it straddles the start transient, exactly
        # as the simulator drops its first 4 samples at 1 Hz
        steady = ld.copy()
        steady[np.flatnonzero(ld)[:1]] = False
        rows.append(
            {
                "t": t[a],
                "t_loaded": t_loaded,
                "t_unloaded": t_unloaded,
                "t_off": t_off,
                "idle_run_ratio": t_off / t_loaded,
                "duty_ratio": t_loaded / (t_loaded + t_unloaded + t_off),
                "dP_dt_off": float(np.nanmean(rate[a:z][s == STATE_OFF])) if (s == STATE_OFF).any() else np.nan,
                "dP_dt_loaded": float(np.nanmean(rate[a:z][ld])),
                # endpoint (span / duration) rates: immune to which samples straddle a
                # transition, and the only ones that add up to the band across a cycle
                "charge_rate": (float(press[a:z].max()) - float(press[a:z][ld][0])) / max(t_loaded - SAMPLE_DT_S, 1e-9),
                "fall_rate": (float(press[a:z].max()) - float(press[a:z].min())) / max(t_unloaded + t_off, 1e-9),
                "unloaded_rate": _phase_rate(press[a:z], s, t[a:z], STATE_UNLOADED),
                "off_rate": _phase_rate(press[a:z], s, t[a:z], STATE_OFF),
                "I_loaded_mean": float(np.nanmean(current[a:z][steady])) if steady.any() else np.nan,
                "T_oil_max": float(np.nanmax(oil[a:z])),
                "P_res_min": float(np.nanmin(press[a:z])),
                "P_res_max": float(np.nanmax(press[a:z])),
                "LPS_any": float(np.nanmax(lps[a:z])),
                "hour_of_day": float((t[a] / 3600.0) % 24.0),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("extract_cycles: no complete compressor cycle in this window")
    return out


# ------------------------------------------------------------------------------ measurement


@dataclass(frozen=True, slots=True)
class Measurements:
    """Everything read off MetroPT-3, before any simulator constant is touched."""

    n_rows: int
    n_excluded: int
    t_min: pd.Timestamp
    t_max: pd.Timestamp
    sample_dt_s: float
    # control band
    p_load_sample: float  # last sample before the compressor loads
    p_unload_sample: float  # highest sample of the loaded run
    p_start: float  # ... corrected for the half-interval sampling lag
    p_stop: float
    band_bar: float
    # rates
    dP_dt_loaded: float
    dP_dt_off: float
    dP_dt_unloaded: float
    duty_cycles: float  # loaded fraction over whole cycles only (the air-balance duty)
    charge_rate: float  # endpoint: band travelled per second of loaded time
    fall_rate: float  # endpoint: band travelled per second of unloaded+off time
    unloaded_rate: float  # endpoint, unloaded phase only
    off_rate: float  # endpoint, off phase only
    compressor_rate: float  # charge_rate + fall_rate = Q_net * P_atm / V
    # durations
    t_loaded_s: float
    t_unloaded_s: float
    t_off_s: float
    duty_time_weighted: float
    idle_run_ratio: float
    # currents
    i_off: float
    i_unloaded: float
    i_loaded: float
    i_start_peak_max: float
    i_loaded_intercept: float
    i_loaded_kp: float
    i_loaded_kT: float
    i_loaded_r2: float
    # dryer
    tower_period_s: float
    tower_flip_loaded_frac: float
    # oil
    oil_b_per_s: float
    oil_tau_s: float
    oil_sink_c: float
    oil_rise_loaded_k: float
    oil_rise_unloaded_k: float
    oil_r2: float
    oil_ambient_c: float
    oil_median_c: float
    oil_duty_slope_k: float
    oil_duty_corr: float
    # measurement chain
    sigma_press: float
    quantum_press: float
    sigma_oil: float
    quantum_oil: float
    sigma_current: float
    quantum_current: float
    # per hour
    duty_by_hour: np.ndarray = field(repr=False)
    dP_dt_off_by_hour: np.ndarray = field(repr=False)
    # leak episode
    t_off_leak_window_s: float
    leak_equilibrium_bar: float
    leak_window_duty: float
    cycles: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)


def _lstsq(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return beta, float(1.0 - (resid**2).sum() / ss_tot) if ss_tot > 0 else float("nan")


def fit_oil_thermal(
    t: np.ndarray, oil: np.ndarray, state: np.ndarray, raw_dt: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Regress ``dT/dt = a_state - b*T`` on the clean, state-constant sample intervals.

    Returns ``(beta, r2, b)`` with ``beta = [a_off, a_unloaded, a_loaded, b]``. ``b = hA/C``;
    ``a_state/b`` is that state's steady temperature, so ``a_off/b`` is the sink the node
    relaxes toward with the motor off and ``(a_state - a_off)/b`` is the rise the heat input
    buys. Only ``hA/C`` and the ratios ``P_state/hA`` are identifiable - ``C`` is pinned by
    the caller.
    """
    clean = (raw_dt >= 0.8 * SAMPLE_DT_S) & (raw_dt <= 1.5 * SAMPLE_DT_S)
    same_state = np.roll(state, -1) == state
    dT = np.diff(oil, append=oil[-1]) / np.maximum(raw_dt, 1e-9)
    ok = clean & same_state & np.isfinite(dT) & np.isfinite(oil)
    ok[-1] = False
    design = np.zeros((int(ok.sum()), 4), dtype=np.float64)
    for s in (STATE_OFF, STATE_UNLOADED, STATE_LOADED):
        design[:, s] = (state[ok] == s).astype(np.float64)
    design[:, 3] = -oil[ok]
    beta, r2 = _lstsq(design, dT[ok])
    return beta, r2, float(beta[3])


def measure_sensor_noise(
    frame: pd.DataFrame, state: np.ndarray, raw_dt: np.ndarray, signal: str
) -> tuple[float, float]:
    """``(sigma, quantum)`` of one channel's measurement chain, per state and pooled.

    The second difference of a slow signal is almost pure noise, and for white noise
    ``var(d2x) = 6*sigma^2`` - so the high-frequency content is recoverable without knowing the
    underlying trajectory. Sigma is measured separately in each compressor state (the loaded
    state carries real motor ripple on top of instrument noise) and pooled by state duration,
    because :class:`~nebulax.sim.common.SensorSpec` has exactly one additive sigma. The quantum
    is the smallest gap between distinct values the channel ever reports.
    """
    x = frame[signal].to_numpy(dtype=np.float64)
    ok = (raw_dt >= 0.8 * SAMPLE_DT_S) & (raw_dt <= 1.2 * SAMPLE_DT_S)
    d2 = np.diff(x, 2)
    var, weight = 0.0, 0.0
    for st in (STATE_OFF, STATE_UNLOADED, STATE_LOADED):
        run = ok & (state == st) & (np.roll(state, 1) == st) & (np.roll(state, -1) == st)
        sel = run[:-2] & run[1:-1] & run[2:]
        if sel.sum() < 100:
            continue
        w = float(sel.sum())
        var += w * float(np.nanvar(d2[sel])) / 6.0
        weight += w
    uniq = np.unique(np.round(x[np.isfinite(x)], 6))
    quantum = float(np.min(np.diff(uniq))) if uniq.size > 1 else float("nan")
    return float(np.sqrt(var / max(weight, 1.0))), quantum


def measure(frame: pd.DataFrame, raw: pd.DataFrame) -> Measurements:
    """Measure everything on the normal window (``frame``); ``raw`` is the whole file."""
    t = _seconds(frame["timestamp"])
    raw_dt = np.diff(t, append=t[-1] + SAMPLE_DT_S)
    dt = np.clip(raw_dt, 0.0, MAX_SAMPLE_DT_S)
    state = segment_states(frame["Motor_current"].to_numpy())
    press = frame["Reservoirs"].to_numpy(dtype=np.float64)
    current = frame["Motor_current"].to_numpy(dtype=np.float64)
    oil = frame["Oil_temperature"].to_numpy(dtype=np.float64)
    towers = frame["Towers"].to_numpy(dtype=np.float64)
    hour = (frame["timestamp"].dt.hour.to_numpy()).astype(np.int64)

    cycles = extract_cycles(frame)

    # ---- control band. The file samples at 10 s, so the switch fires somewhere in the
    # interval before the sample that first shows it: correct by half an interval of travel.
    p_load_sample = float(np.median(cycles["P_res_min"]))
    p_unload_sample = float(np.median(cycles["P_res_max"]))
    dP_loaded = float(np.median(cycles["dP_dt_loaded"]))
    dP_off = float(np.nanmedian(cycles["dP_dt_off"]))
    p_start = p_load_sample + 0.5 * dP_off * SAMPLE_DT_S
    p_stop = p_unload_sample + 0.5 * dP_loaded * SAMPLE_DT_S

    # ---- rates, by state, over clean intervals only
    clean = (raw_dt >= 0.8 * SAMPLE_DT_S) & (raw_dt <= 1.5 * SAMPLE_DT_S)
    rate = np.where(clean, np.diff(press, append=press[-1]) / np.maximum(raw_dt, 1e-9), np.nan)
    dP_unloaded = float(np.nanmedian(rate[clean & (state == STATE_UNLOADED)]))
    charge_rate = float(np.nanmedian(cycles["charge_rate"]))
    fall_rate = float(np.nanmedian(cycles["fall_rate"]))
    unloaded_rate = float(np.nanmedian(cycles["unloaded_rate"]))
    off_rate = float(np.nanmedian(cycles["off_rate"]))
    compressor_rate = float(np.nanmedian(cycles["charge_rate"] + cycles["fall_rate"]))

    # ---- durations and duty
    total = dt.sum()
    duty_tw = float(dt[state == STATE_LOADED].sum() / total)
    # Duty per hour band from the CYCLE TABLE, not from the raw stream: whole cycles conserve
    # air (in = Q_net * t_loaded, out = the demand integral), so the ratio is an unbiased
    # estimate of the mean demand. The raw stream also counts logger outages and the handful of
    # multi-hour continuous runs that extract_cycles drops, which pushes the duty from 0.089 to
    # 0.114 and would inflate the fitted consumption by 29 %.
    cyc_total = cycles["t_loaded"] + cycles["t_unloaded"] + cycles["t_off"]
    cyc_hour = cycles["hour_of_day"].astype(int).to_numpy()
    duty_by_hour = np.array(
        [
            float(cycles["t_loaded"].to_numpy()[cyc_hour == h].sum() / max(cyc_total.to_numpy()[cyc_hour == h].sum(), 1e-9))
            for h in range(24)
        ]
    )
    duty_cycles = float(cycles["t_loaded"].sum() / cyc_total.sum())
    off_by_hour = np.array(
        [float(np.nanmedian(rate[clean & (state == STATE_OFF) & (hour == h)])) for h in range(24)]
    )

    # ---- currents
    i_off = float(np.median(current[state == STATE_OFF]))
    i_unloaded = float(np.median(current[state == STATE_UNLOADED]))
    i_loaded = float(np.median(current[state == STATE_LOADED]))
    loaded = state == STATE_LOADED
    starts = np.flatnonzero(loaded & ~np.roll(loaded, 1))
    starts = starts[starts > 0]
    peak_idx = np.clip(starts[:, None] + np.arange(3)[None, :], 0, current.size - 1)
    i_peak_max = float(np.nanmax(current[peak_idx]))
    sel = loaded.copy()
    sel[peak_idx[:, :3].ravel()] = False  # drop the start transient
    design = np.c_[np.ones(int(sel.sum())), press[sel] - p_start, oil[sel] - float(np.median(oil))]
    beta_i, r2_i = _lstsq(design, current[sel])

    # ---- twin-tower period, measured from the TOWERS channel [rail_phm 2.3b]
    flips = np.flatnonzero(np.diff(towers) != 0.0) + 1
    flip_loaded = flips[state[flips] == STATE_LOADED]
    cum_loaded = np.cumsum(np.where(state == STATE_LOADED, dt, 0.0))
    gaps = np.diff(cum_loaded[flip_loaded])
    gaps = gaps[(gaps > 0.0) & (gaps < 600.0)]
    tower_period = float(np.median(gaps))

    # ---- oil node
    beta_o, r2_o, b = fit_oil_thermal(t, oil, state, raw_dt)
    sink = float(beta_o[STATE_OFF] / b)
    rise_unloaded = float(beta_o[STATE_UNLOADED] / b) - sink
    rise_loaded = float(beta_o[STATE_LOADED] / b) - sink
    # true outdoor ambient: the oil temperature at the record's own cold starts, i.e. while
    # the unit is charging an empty reservoir (< 2 bar) from a full shutdown
    cold = raw[(raw["Reservoirs"] < 2.0) & (raw["Motor_current"] > I_LOADED_MIN - 0.2)]
    ambient = float(cold["Oil_temperature"].min()) if len(cold) else float("nan")
    # oil vs duty at the hour scale
    hourly = pd.DataFrame(
        {"hr": frame["timestamp"].dt.floor("h"), "T": oil, "ld": (state == STATE_LOADED).astype(float), "w": dt}
    )
    agg = hourly.groupby("hr", observed=True).apply(
        lambda x: pd.Series(
            {
                "T": float(np.average(x["T"], weights=x["w"])),
                "duty": float(np.average(x["ld"], weights=x["w"])),
                "w": float(x["w"].sum()),
            }
        ),
        include_groups=False,
    )
    agg = agg[agg["w"] > 0.8 * 3600.0]
    slope, _ = _lstsq(np.c_[np.ones(len(agg)), agg["duty"].to_numpy()], agg["T"].to_numpy())
    corr = float(np.corrcoef(agg["duty"], agg["T"])[0, 1])

    # ---- measurement chain
    sig_press, q_press = measure_sensor_noise(frame, state, raw_dt, "Reservoirs")
    sig_oil, q_oil = measure_sensor_noise(frame, state, raw_dt, "Oil_temperature")
    sig_cur, q_cur = measure_sensor_noise(frame, state, raw_dt, "Motor_current")

    # ---- the leak acceptance window
    lo, hi = (pd.Timestamp(x) for x in LEAK_WINDOW)
    leak_frame = raw[(raw["timestamp"] >= lo) & (raw["timestamp"] < hi)].reset_index(drop=True)
    leak_cycles = extract_cycles(leak_frame)
    leak_state = segment_states(leak_frame["Motor_current"].to_numpy())
    leak_press = leak_frame["Reservoirs"].to_numpy(dtype=np.float64)
    lt = _seconds(leak_frame["timestamp"])
    ld = leak_state == STATE_LOADED
    runs_s = np.flatnonzero(ld & ~np.roll(ld, 1))
    runs_e = np.flatnonzero(ld & ~np.roll(ld, -1))
    k = min(runs_s.size, runs_e.size)
    lengths = lt[runs_e[:k]] - lt[runs_s[:k]]
    j = int(np.argmax(lengths))
    equilibrium = float(np.median(leak_press[runs_s[j] : runs_e[j] + 1]))

    return Measurements(
        n_rows=len(frame),
        n_excluded=0,
        t_min=frame["timestamp"].iloc[0],
        t_max=frame["timestamp"].iloc[-1],
        sample_dt_s=float(np.median(raw_dt)),
        p_load_sample=p_load_sample,
        p_unload_sample=p_unload_sample,
        p_start=p_start,
        p_stop=p_stop,
        band_bar=float(np.median(cycles["P_res_max"] - cycles["P_res_min"])),
        dP_dt_loaded=dP_loaded,
        dP_dt_off=dP_off,
        dP_dt_unloaded=dP_unloaded,
        duty_cycles=duty_cycles,
        charge_rate=charge_rate,
        fall_rate=fall_rate,
        unloaded_rate=unloaded_rate,
        off_rate=off_rate,
        compressor_rate=compressor_rate,
        t_loaded_s=float(cycles["t_loaded"].median()),
        t_unloaded_s=float(cycles["t_unloaded"].median()),
        t_off_s=float(cycles["t_off"].median()),
        duty_time_weighted=duty_tw,
        idle_run_ratio=float(cycles["idle_run_ratio"].median()),
        i_off=i_off,
        i_unloaded=i_unloaded,
        i_loaded=i_loaded,
        i_start_peak_max=i_peak_max,
        i_loaded_intercept=float(beta_i[0]),
        i_loaded_kp=float(beta_i[1]),
        i_loaded_kT=float(beta_i[2]),
        i_loaded_r2=r2_i,
        tower_period_s=tower_period,
        tower_flip_loaded_frac=float((state[flips] == STATE_LOADED).mean()),
        oil_b_per_s=b,
        oil_tau_s=1.0 / b,
        oil_sink_c=sink,
        oil_rise_loaded_k=rise_loaded,
        oil_rise_unloaded_k=rise_unloaded,
        oil_r2=r2_o,
        oil_ambient_c=ambient,
        oil_median_c=float(np.median(oil)),
        oil_duty_slope_k=float(slope[1]),
        oil_duty_corr=corr,
        sigma_press=sig_press,
        quantum_press=q_press,
        sigma_oil=sig_oil,
        quantum_oil=q_oil,
        sigma_current=sig_cur,
        quantum_current=q_cur,
        duty_by_hour=duty_by_hour,
        dP_dt_off_by_hour=off_by_hour,
        t_off_leak_window_s=float(leak_cycles["t_off"].median()),
        leak_equilibrium_bar=equilibrium,
        leak_window_duty=float(leak_cycles["duty_ratio"].median()),
        cycles=cycles,
    )


# ------------------------------------------------------------------ constants for the sim


def effective_volume_l(m: Measurements, p: PN.PneumaticParams) -> float:
    """Reservoir volume implied by the charge slope, with ``Q_comp_nls`` pinned.

    Inside one cycle the same demand is drawn in both phases: it is *subtracted* from the
    compressor while the unit charges and *added* to nothing while it coasts down. So the
    per-cycle sum of the two endpoint rates cancels the demand exactly and isolates the
    compressor - DOE's receiver relation ``V = T*C*Pa/(P1-P2)`` [R155] read backwards::

        band/t_loaded + band/(t_unloaded + t_off) = Q_comp*(1 - purge_frac) * P_atm / V

    Endpoint rates, not sample-wise means: at 10 s cadence the first sample of a loaded run
    straddles the transition and drags a sample-wise mean ~9 % low.

    Only ``Q_comp/V`` is identifiable from pressure alone (MetroPT-3 has no flow channel);
    see the module docstring.
    """
    net = p.Q_comp_nls * (1.0 - p.purge_frac)
    return float(net * p.P_atm_bar / m.compressor_rate)


def unloaded_vent_nls(m: Measurements, p: PN.PneumaticParams, v_res_l: float, aux_nls: float) -> float:
    """Extra flow leaving the reservoir while the unit is unloaded, beyond leak + auxiliaries.

    The unloaded phase falls at :attr:`Measurements.unloaded_rate`, measurably faster than
    leak-plus-auxiliaries alone predicts at the same (higher) pressure - the separator and
    unloader keep bleeding after the motor offloads. Without it the simulator's reservoir
    enters the OFF phase ~0.3 bar too high and ``t_off`` runs long.
    """
    p_unloaded = m.p_unload_sample + 0.5 * m.unloaded_rate * (m.t_unloaded_s)
    expected = float(PN.leak_nls(p_unloaded + p.P_atm_bar, p.leak_d0_mm, p)) + aux_nls
    observed = abs(m.unloaded_rate) * v_res_l / p.P_atm_bar
    return max(observed - expected, 0.0)


def quiescent_demand_nls(m: Measurements, p: PN.PneumaticParams, v_res_l: float) -> tuple[float, float]:
    """``(total, genuine)`` quiescent draw in NL/s, from the OFF decay rate.

    ``total`` is everything leaving the reservoir with the motor off; ``genuine`` subtracts
    the healthy leak at the mean OFF pressure, so it is the auxiliary consumption the
    simulator's ``aux_nls`` has to carry. If the residual is negative the healthy leak
    assumption (``leak_d0_mm``) is too large and the caller must say so.
    """
    p_mean = m.p_start + 0.5 * abs(m.off_rate) * m.t_off_s
    total = abs(m.off_rate) * v_res_l / p.P_atm_bar
    leak = float(PN.leak_nls(p_mean + p.P_atm_bar, p.leak_d0_mm, p))
    return total, total - leak


def demand_by_hour_nls(m: Measurements, p: PN.PneumaticParams) -> np.ndarray:
    """Mean out-flow per hour of day, from the cycle air balance ``Q_net * duty(h)``."""
    return m.duty_by_hour * p.Q_comp_nls * (1.0 - p.purge_frac)


def _sim_demand_basis(p: PN.PneumaticParams, days: int = 7, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """``(aux_profile, burst_profile)`` - the simulator's own demand shape per hour of day.

    ``aux_profile`` is the duty-cycle-free auxiliary multiplier (1 in service,
    ``depot_aux_frac`` while stabled) and ``burst_profile`` the brake + air-spring draw at the
    *current* constants, both averaged over hour of day across a week of the shared service
    generator. Fitting two scalars against these is what "scale consumption per hour band"
    means for a train whose diurnal shape is set by the timetable, not by us.
    """
    rng = np.random.default_rng(seed)
    service = generate_service(days, rng, train_id="T01")
    tl = service.timeline(dt=1.0)
    # NO rng on purpose: ``_consumption`` then emits the *expected* schedule (every event at
    # its mean) instead of one realisation of the heavy-tailed draw. The fit below is against
    # the measured MEAN demand per hour band, so it must see the mean basis; with a sampled
    # basis the fitted ``burst_scale`` wanders ~10 % from seed to seed and the section-2
    # constants stop reading CONFIRMED for no physical reason.
    cons = PN._consumption(service, tl.t, tl.seg_id, p)
    hour = ((tl.t / 3600.0) % 24.0).astype(np.int64)
    aux_unit = np.where(tl.in_service, 1.0, p.depot_aux_frac)
    burst = cons["brake"] + cons["spring"]
    aux_profile = np.array([float(aux_unit[hour == h].mean()) for h in range(24)])
    burst_profile = np.array([float(burst[hour == h].mean()) for h in range(24)])
    return aux_profile, burst_profile


def fit_consumption(
    m: Measurements, p: PN.PneumaticParams, v_res_l: float, *, days: int = 7, seed: int = 7
) -> dict[str, float]:
    """Scale the simulator's consumption schedule onto the measured demand per hour band.

    Two *independent* estimators pin the two knobs, which a joint regression cannot do: on our
    service the continuous-auxiliary basis and the brake/air-spring burst basis are both
    essentially "1 while in service", so they are collinear and an unconstrained NNLS puts the
    whole demand into whichever it reaches first (it puts it all into ``aux_nls`` and zeroes
    the brake, which is why the fit is *not* run that way). Instead:

    * ``aux_nls`` comes from the **OFF decay**, which by construction only ever sees quiescent
      periods - a burst big enough to matter starts the compressor and so lands in a loaded
      window, never in an OFF window;
    * ``burst_scale`` then comes from the **per-hour demand** ``Q_net*duty(h)``, least squares
      on the burst basis alone with the auxiliaries already fixed, clipped at zero.

    ``burst_scale`` multiplies ``brake_nl_per_stop``, ``spring_nl_per_dwell`` and
    ``spring_nl_per_unit_load`` together, so the demo_runs *relative* tuning between brake and
    air spring survives untouched.
    """
    _, aux_nls = quiescent_demand_nls(m, p, v_res_l)
    aux_nls = max(aux_nls, 0.0)
    p_band = 0.5 * (m.p_start + m.p_stop)
    leak = float(PN.leak_nls(p_band + p.P_atm_bar, p.leak_d0_mm, p))
    vent = unloaded_vent_nls(m, p, v_res_l, aux_nls)
    aux_profile, burst_profile = _sim_demand_basis(p, days=days, seed=seed)
    target = demand_by_hour_nls(m, p) - leak - aux_nls * aux_profile
    denom = float((burst_profile**2).sum())
    burst_scale = max(float((target * burst_profile).sum() / denom), 0.0) if denom > 0 else 0.0
    fitted = aux_nls * aux_profile + burst_scale * burst_profile + leak
    measured = demand_by_hour_nls(m, p)
    return {
        "aux_nls": aux_nls,
        "burst_scale": burst_scale,
        "unloaded_vent_nls": vent,
        "leak_nls_at_band": leak,
        "target_mean_nls": float(measured.mean()),
        "fitted_mean_nls": float(fitted.mean()),
        "rms_residual_nls": float(np.sqrt(np.mean((measured - fitted) ** 2))),
        "burst_basis_mean_nls": float(burst_profile.mean()),
        "brake_nl_per_stop": p.brake_nl_per_stop * burst_scale,
        "spring_nl_per_dwell": p.spring_nl_per_dwell * burst_scale,
        "spring_nl_per_unit_load": p.spring_nl_per_unit_load * burst_scale,
    }


def leak_endpoint_mm(m: Measurements, p: PN.PneumaticParams, aux_nls: float) -> dict[str, float]:
    """The ``s = 1`` orifice, three ways, and the one we take.

    ``observed`` reproduces the 29 May - 7 Jun equilibrium: during that episode's 7,781 s
    continuous run the reservoir settled at :attr:`Measurements.leak_equilibrium_bar`, so the
    leak there exactly balanced the net delivery at that pressure.

    ``lps`` is the same balance evaluated at the ``LPS`` trip: the smallest leak for which a
    continuously running compressor can no longer hold the reservoir above 7 bar. That is the
    definition we adopt for ``s = 1`` - the real episodes were repaired *before* full failure,
    so they bound the endpoint from below, while the plan defines ``s = 1`` as the failure
    itself. ``t_off`` collapses to zero for any endpoint above ``observed``, so the acceptance
    test is satisfied across the whole range.
    """
    net = p.Q_comp_nls * (1.0 - p.purge_frac) - aux_nls
    ref_abs = p.leak_ref_pressure_barg + p.P_atm_bar

    def ratio_at(p_eq: float) -> float:
        return net * ref_abs / (p.leak_ref_nls * (p_eq + p.P_atm_bar))

    out = {}
    for key, p_eq in (("observed", m.leak_equilibrium_bar), ("lps", p.P_lps_bar), ("start", p.P_start_bar)):
        r = ratio_at(p_eq)
        out[f"{key}_ratio"] = r
        out[f"{key}_d1_mm"] = p.leak_d0_mm * float(np.sqrt(r))
    out["chosen_d1_mm"] = out["lps_d1_mm"]
    out["worst_case_nls"] = float(PN.leak_nls(p.P_stop_bar + p.P_atm_bar, out["chosen_d1_mm"], p))
    return out


def healthy_leak_bound_mm(m: Measurements, p: PN.PneumaticParams, v_res_l: float) -> float:
    """Upper bound on ``leak_d0_mm``: the whole quiescent draw attributed to the leak."""
    p_mean = m.p_start + 0.5 * abs(m.off_rate) * m.t_off_s
    total = abs(m.off_rate) * v_res_l / p.P_atm_bar
    q_ref = total * (p.leak_ref_pressure_barg + p.P_atm_bar) / (p_mean + p.P_atm_bar)
    return p.leak_d0_mm * float(np.sqrt(q_ref / p.leak_ref_nls))


# ------------------------------------------------------------------------------ the sim side


def simulate_cycles(days: int = 21, seed: int = 3, params: PN.PneumaticParams | None = None) -> pd.DataFrame:
    """Healthy simulator cycles, as a table with the same columns as :func:`extract_cycles`."""
    p = params or PN.PneumaticParams()
    rng = np.random.default_rng(seed)
    service = generate_service(days, rng, train_id="T01")
    _, feats, _ = PN.simulate(p, PN.PneumaticFaults(), service, rng, store_every=10**6, run_id="cal")
    out = feats[
        ["t_loaded", "t_unloaded", "t_off", "dP_dt_off", "dP_dt_loaded", "I_loaded_mean", "T_oil_max", "P_res_min", "hour_of_day"]
    ].astype(np.float64).copy()
    out["idle_run_ratio"] = feats["idle_run_ratio"].astype(np.float64)
    out["duty_ratio"] = feats["duty_ratio"].astype(np.float64)
    out["in_service_frac"] = feats["in_service_frac"].astype(np.float64)
    return out


def sim_phase_demand_nls(
    days: int = 7, seed: int = 3, params: PN.PneumaticParams | None = None
) -> dict[str, float]:
    """Simulated draw per compressor phase, with the SAME estimator MetroPT-3 is read with.

    The measured row (``report``'s "draw per phase") is built from the per-cycle **endpoint**
    rate of each phase: ``Q_net - dP/dt*V/P_atm`` while loaded and ``-dP/dt*V/P_atm`` while
    unloaded / off, then the median over cycles. This runs the simulator's own state machine
    (no sensor chain - the estimator is on the clean reservoir trace, exactly as the real one is
    on a channel whose noise is 1.5 % of the signal) and reports the same three medians, so the
    asymmetry that motivated the heavy-tailed burst schedule can be audited rather than asserted.

    Seven days are enough: the medians move by < 0.01 NL/s between 7 and 21 days, and a second
    21-day run would double this script's runtime.
    """
    p = params or PN.PneumaticParams()
    rng = np.random.default_rng(seed)
    service = generate_service(days, rng, train_id="T01")
    tl = service.timeline(dt=1.0)
    cons = PN._consumption(service, tl.t, tl.seg_id, p, rng)
    sev = {name: np.zeros(len(tl), dtype=np.float64) for name in PN.PNEUMATIC_FAULTS}
    sig, _, _ = PN._run_core(len(tl), cons["demand"], tl.T_amb, tl.in_service, sev, p)
    state, press = sig["state"], sig["Reservoirs"]
    bounds = np.flatnonzero((state == PN.LOADING) & (np.diff(state, prepend=np.int8(PN.OFF)) != 0))
    net = p.Q_comp_nls * (1.0 - p.purge_frac)
    phases = {
        "loaded": ((state == PN.LOADED) | (state == PN.LOADING), True),
        "unloaded": ((state == PN.UNLOADING) | (state == PN.UNLOADED), False),
        "off": (state == PN.OFF, False),
    }
    out: dict[str, float] = {"n_cycles": float(max(bounds.size - 1, 0))}
    for name, (mask, is_loaded) in phases.items():
        rates = []
        for a, z in zip(bounds[:-1], bounds[1:]):
            idx = np.flatnonzero(mask[a:z])
            if idx.size < 2:
                continue
            lo, hi = a + int(idx[0]), a + int(idx[-1])
            rates.append((press[hi] - press[lo]) / (hi - lo))
        r = np.asarray(rates, dtype=np.float64)
        q = (net - r * p.V_res_l / p.P_atm_bar) if is_loaded else (-r * p.V_res_l / p.P_atm_bar)
        out[name] = float(np.median(q)) if q.size else float("nan")
        out[f"n_{name}"] = float(q.size)
    out["duty_time"] = float(((state == PN.LOADED) | (state == PN.LOADING)).mean())
    return out


# ------------------------------------------------------------------- leak-ramp lead time

#: The 30-day air-leak ramp that section 3 of ``docs/parameters.md`` quotes: onset on day 8,
#: nominal ``s = 1`` on day 26, so the detector has an 18-day ramp to catch.
LEAK_RAMP_DAYS: Final[int] = 30
LEAK_RAMP_ONSET_DAY: Final[float] = 8.0
LEAK_RAMP_FAILURE_DAY: Final[float] = 26.0
LEAK_RAMP_SEEDS: Final[tuple[int, ...]] = (3, 5, 7, 11)
LEAK_RAMP_GAMMAS: Final[tuple[float, ...]] = (1.0, 2.0, 3.0)

#: Detector settings, chosen so that the rule **never fires on a healthy run** (the false-alarm
#: column of :func:`leak_ramp_report` is the audit). ``persist_h`` is what buys that: a rolling
#: median is heavily autocorrelated, so a bare percentile crossing chatters. The baseline is the
#: unit's own first :data:`LEAK_RAMP_BASELINE_DAY` days, i.e. a post-overhaul reference period,
#: and the alarm is searched only after it.
LEAK_RAMP_WINDOW_H: Final[float] = 12.0
LEAK_RAMP_PERSIST_H: Final[float] = 24.0
LEAK_RAMP_BASELINE_DAY: Final[float] = 8.0
LEAK_RAMP_MIN_CYCLES: Final[int] = 5

#: The two detectors section 3 reports: ``(name, feature, alarm below?, baseline quantile %)``.
#: The first is [R101]'s statistic, the second needs no OFF phase at all and so keeps working
#: after ``t_off`` has collapsed to zero.
LEAK_RAMP_DETECTORS: Final[tuple[tuple[str, str, bool, float], ...]] = (
    ("idle_run_ratio", "idle_run_ratio", True, 5.0),
    ("duty_ratio", "duty_ratio", False, 95.0),
)


def _rolling_median(day: np.ndarray, x: np.ndarray, window_h: float, min_periods: int) -> np.ndarray:
    """Trailing rolling median of ``x`` over ``window_h`` hours of cycle-end time."""
    s = pd.Series(np.asarray(x, dtype=np.float64), index=pd.to_timedelta(day * SEC_PER_DAY, unit="s"))
    return s.rolling(pd.Timedelta(hours=window_h), min_periods=min_periods).median().to_numpy(dtype=np.float64)


def _first_sustained(day: np.ndarray, stat: np.ndarray, thr: float, below: bool, persist_h: float) -> float:
    """Day on which a crossing of ``thr`` that *holds* for ``persist_h`` hours can be declared.

    The day returned is the **end** of the hold, not its start: a detector cannot raise an alarm
    before it has the evidence, so crediting it with the first sample of the excursion would
    inflate every lead time by ``persist_h``.
    """
    ok = np.isfinite(stat)
    hit = ok & ((stat < thr) if below else (stat > thr))
    if not hit.any():
        return float("nan")
    span = persist_h / 24.0
    last = float(day[ok].max())
    for i in np.flatnonzero(hit):
        d0 = float(day[i])
        if d0 + span > last:  # not enough record left to confirm the crossing
            return float("nan")
        win = ok & (day >= d0) & (day <= d0 + span)
        if bool(hit[win].all()):
            return d0 + span
    return float("nan")


def leak_ramp_run(
    gamma: float | None,
    seed: int,
    days: int = LEAK_RAMP_DAYS,
    onset_day: float = LEAK_RAMP_ONSET_DAY,
    failure_day: float = LEAK_RAMP_FAILURE_DAY,
    params: PN.PneumaticParams | None = None,
) -> tuple[pd.DataFrame, float]:
    """One air-leak ramp (``gamma=None`` for the matched healthy run).

    Returns the per-cycle table (``day``, ``idle_run_ratio``, ``duty_ratio``, ``t_off``,
    ``in_service_frac``) and the functional-failure day written back by
    :func:`nebulax.sim.pneumatic.simulate` (``nan`` when the unit never fails inside the run,
    which is the healthy case by construction).

    The consumption schedule is drawn *before* the severity series, so for a given ``seed`` the
    healthy and ramped runs share their demand realisation up to ``onset_day``: the healthy run
    is a genuine matched control, not an independent unit.
    """
    p = params or PN.PneumaticParams()
    rng = np.random.default_rng(seed)
    service = generate_service(days, rng, train_id="T01")
    faults = PN.PneumaticFaults()
    traj: DegradationTrajectory | None = None
    if gamma is not None:
        traj = DegradationTrajectory(
            fault_type="air_leak",
            subsystem="pneumatic",
            component_id="apu_1",
            t_onset=onset_day * SEC_PER_DAY,
            t_failure=failure_day * SEC_PER_DAY,
            gamma=float(gamma),
        )
        faults = PN.PneumaticFaults.from_trajectories([traj])
    tag = "healthy" if gamma is None else f"g{gamma:g}"
    _, feats, _ = PN.simulate(p, faults, service, rng, store_every=10**6, run_id=f"leakramp_{tag}_s{seed}")
    day = (feats["t_end"] - service.t0).dt.total_seconds().to_numpy(dtype=np.float64) / SEC_PER_DAY
    out = pd.DataFrame(
        {
            "day": day,
            "idle_run_ratio": feats["idle_run_ratio"].to_numpy(dtype=np.float64),
            "duty_ratio": feats["duty_ratio"].to_numpy(dtype=np.float64),
            "t_off": feats["t_off"].to_numpy(dtype=np.float64),
            "in_service_frac": feats["in_service_frac"].to_numpy(dtype=np.float64),
        }
    )
    t_ff = float("nan")
    if traj is not None and traj.t_functional_failure is not None:
        t_ff = float(traj.t_functional_failure) / SEC_PER_DAY
    return out, t_ff


def _leak_ramp_alarm(
    tab: pd.DataFrame,
    col: str,
    below: bool,
    q: float,
    window_h: float,
    persist_h: float,
    baseline_day: float,
) -> tuple[float, float]:
    """``(threshold, alarm_day)`` for one run: baseline from its own first ``baseline_day`` days."""
    day = tab["day"].to_numpy(dtype=np.float64)
    stat = _rolling_median(day, tab[col].to_numpy(dtype=np.float64), window_h, LEAK_RAMP_MIN_CYCLES)
    base = stat[day < baseline_day]
    if not np.isfinite(base).any():
        return float("nan"), float("nan")
    thr = float(np.nanpercentile(base, q))
    live = day >= baseline_day
    return thr, _first_sustained(day[live], stat[live], thr, below, persist_h)


def leak_ramp_lead_time(
    gammas: Sequence[float] = LEAK_RAMP_GAMMAS,
    seeds: Sequence[int] = LEAK_RAMP_SEEDS,
    days: int = LEAK_RAMP_DAYS,
    onset_day: float = LEAK_RAMP_ONSET_DAY,
    failure_day: float = LEAK_RAMP_FAILURE_DAY,
    window_h: float = LEAK_RAMP_WINDOW_H,
    persist_h: float = LEAK_RAMP_PERSIST_H,
    baseline_day: float = LEAK_RAMP_BASELINE_DAY,
    params: PN.PneumaticParams | None = None,
) -> pd.DataFrame:
    """Alarm day, functional-failure day and lead time of an air-leak ramp, per seed and gamma.

    This is the experiment behind section 3's lead-time table, which until 15 Sep 2026 lived
    only in that table's prose. It is a function so the table can be regenerated rather than
    believed. Cost: ``(len(gammas) + 1) * len(seeds)`` 30-day runs, ~20 s each.

    **Why the rule is not [R101]'s literal one.** [R101] alarms when a 6 h rolling median of
    ``idle_run_ratio`` falls below *the healthy 5th percentile of the raw feature*. That is not
    implementable on this unit and never was: ``idle_run_ratio`` is **zero-inflated** - 17.3 %
    of MetroPT-3's own healthy cycles never reach the OFF phase at all, so the real record's
    healthy p1, p5 and p10 are all exactly 0.000 and no median can fall below them. (Before
    15 Sep 2026 the simulator's flat demand schedule produced no zeros and hence a healthy p5 of
    ~5.0; that number was an artefact of the schedule, not a property of the unit.) The rule is
    therefore restated on quantities that survive the zeros:

    * the statistic is a ``window_h`` **rolling median** of the feature;
    * the threshold is the ``q``-th percentile **of that same rolling statistic** over the unit's
      own first ``baseline_day`` days - a post-overhaul reference period, not a fleet constant,
      because the healthy level itself varies by a factor of ~2 across service patterns;
    * the crossing must **hold for ``persist_h``**, which is what makes the rule survive the
      autocorrelation of a rolling median. Without it the same threshold fires on healthy runs.

    Both detectors are reported: [R101]'s ``idle_run_ratio`` and ``duty_ratio``, which needs no
    OFF phase and so keeps working past ``s ~ 0.13``. The matched healthy run of every seed is
    passed through the identical rule; ``detector``-wise that is the **false-alarm** audit and
    it is printed beside the leads.

    One row per (detector, gamma, seed), plus one row per (detector, seed) with ``gamma = nan``
    for the healthy control. ``alarm_day`` is ``nan`` when the detector never fires, ``ff_day``
    is ``nan`` when the unit never reaches functional failure inside the run.
    """
    rows: list[dict[str, float]] = []
    healthy: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        healthy[seed], _ = leak_ramp_run(None, seed, days, onset_day, failure_day, params)
    runs: list[tuple[float, int, pd.DataFrame, float]] = [
        (float("nan"), seed, healthy[seed], float("nan")) for seed in seeds
    ]
    for gamma in gammas:
        for seed in seeds:
            tab, ff = leak_ramp_run(float(gamma), seed, days, onset_day, failure_day, params)
            runs.append((float(gamma), seed, tab, ff))
    for name, col, below, q in LEAK_RAMP_DETECTORS:
        for gamma, seed, tab, ff in runs:
            thr, alarm = _leak_ramp_alarm(tab, col, below, q, window_h, persist_h, baseline_day)
            rows.append(
                {
                    "detector": name,
                    "gamma": gamma,
                    "seed": float(seed),
                    "threshold": thr,
                    "alarm_day": alarm,
                    "ff_day": ff,
                    "lead_d": ff - alarm,
                    "n_cycles": float(len(tab)),
                }
            )
    out = pd.DataFrame(rows)
    out.attrs["healthy_zero_frac"] = {int(s): float(np.mean(healthy[s]["t_off"].to_numpy() <= 0.0)) for s in seeds}
    out.attrs["healthy_idle_p5_raw"] = {
        int(s): float(np.nanpercentile(healthy[s]["idle_run_ratio"].to_numpy(), 5.0)) for s in seeds
    }
    out.attrs["settings"] = {
        "days": float(days),
        "onset_day": float(onset_day),
        "failure_day": float(failure_day),
        "window_h": float(window_h),
        "persist_h": float(persist_h),
        "baseline_day": float(baseline_day),
    }
    return out


def _fmt_spread(x: np.ndarray) -> str:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "never"
    if v.size == 1:
        return f"{v[0]:.2f}"
    return f"{np.median(v):.2f} [{v.min():.2f}, {v.max():.2f}]"


def leak_ramp_report(table: pd.DataFrame, real: pd.DataFrame | None = None) -> str:
    """Render :func:`leak_ramp_lead_time` the way section 3 quotes it."""
    st = table.attrs.get("settings", {})
    lines = [
        "=" * 96,
        f"AIR-LEAK RAMP, LEAD TIME  ({st.get('days', float('nan')):.0f} d, onset day "
        f"{st.get('onset_day', float('nan')):.0f}, nominal s=1 day {st.get('failure_day', float('nan')):.0f}; "
        f"{int(table['n_cycles'].max()):,} cycles per run)",
        "-" * 96,
    ]
    zf = table.attrs.get("healthy_zero_frac", {})
    p5 = table.attrs.get("healthy_idle_p5_raw", {})
    if zf:
        lines.append(
            "  healthy t_off == 0 fraction        "
            + "  ".join(f"seed {s}: {v:.3f}" for s, v in zf.items())
            + (f"   (MetroPT-3: {float((real['t_off'] <= 0).mean()):.3f})" if real is not None else "")
        )
    if p5:
        lines.append(
            "  healthy p5 of RAW idle_run_ratio   "
            + "  ".join(f"seed {s}: {v:.3f}" for s, v in p5.items())
            + (
                "   (MetroPT-3 p1/p5/p10: "
                f"{np.nanpercentile(real['idle_run_ratio'], 1):.3f}/"
                f"{np.nanpercentile(real['idle_run_ratio'], 5):.3f}/"
                f"{np.nanpercentile(real['idle_run_ratio'], 10):.3f})"
                if real is not None
                else ""
            )
        )
        lines.append(
            "  => [R101]'s literal 'raw feature below the healthy 5th percentile' cannot fire on "
            "either record; the rule below is the restatement."
        )
    for name, _col, below, q in LEAK_RAMP_DETECTORS:
        sub = table[table["detector"] == name]
        if sub.empty:
            continue
        lines += [
            "-" * 96,
            f"detector: {st.get('window_h', float('nan')):.0f} h rolling median of {name} "
            f"{'below' if below else 'above'} the p{q:.0f} of that statistic over the unit's own "
            f"first {st.get('baseline_day', float('nan')):.0f} days, held "
            f"{st.get('persist_h', float('nan')):.0f} h",
            f"{'gamma':>7}{'fired':>8}{'alarm day':>24}{'failure day':>24}{'lead d':>24}",
        ]
        for gamma, g in sub.groupby("gamma", dropna=False):
            fired = int(np.isfinite(g["alarm_day"].to_numpy()).sum())
            label = f"{gamma:.1f}" if np.isfinite(gamma) else "healthy"
            lines.append(
                f"{label:>7}{fired:>4d}/{len(g):<3d}"
                f"{_fmt_spread(g['alarm_day'].to_numpy()):>24}"
                f"{_fmt_spread(g['ff_day'].to_numpy()):>24}"
                f"{_fmt_spread(g['lead_d'].to_numpy()):>24}"
            )
        lines.append("  the 'healthy' row is the false-alarm audit: the same rule on the matched healthy run.")
    lines.append("=" * 96)
    return "\n".join(lines)


def ks_table(real: pd.DataFrame, sim: pd.DataFrame, columns: Sequence[str] = KS_COLUMNS) -> pd.DataFrame:
    """Two-sample KS distance per feature, with the two medians beside it."""
    from scipy.stats import ks_2samp

    rows = []
    for col in columns:
        a = real[col].to_numpy(dtype=np.float64)
        b = sim[col].to_numpy(dtype=np.float64)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        stat = ks_2samp(a, b)
        rows.append(
            {
                "feature": col,
                "metropt3_median": float(np.median(a)),
                "sim_median": float(np.median(b)),
                "ratio": float(np.median(b) / np.median(a)) if np.median(a) else np.nan,
                "ks": float(stat.statistic),
                "n_real": a.size,
                "n_sim": b.size,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ constants


@dataclass(frozen=True, slots=True)
class ConstantUpdate:
    """One simulator constant, its current default and what MetroPT-3 says it should be."""

    name: str
    old: float
    new: float
    tag: str
    evidence: str
    tol: float = 0.02

    @property
    def changed(self) -> bool:
        if self.old == self.new:
            return False
        scale = max(abs(self.old), abs(self.new), 1e-12)
        return abs(self.new - self.old) / scale > self.tol

    @property
    def status(self) -> str:
        return "CHANGE" if self.changed else "CONFIRMED"


def recommend_constants(m: Measurements, params: PN.PneumaticParams | None = None) -> list[ConstantUpdate]:
    """The whole constant table: current default vs MetroPT-3-derived value.

    Rows whose ``new`` equals ``old`` on purpose are *pinned* or *confirmed*, and are listed
    so that re-running the script audits them too rather than silently ignoring them.
    """
    p = params or PN.PneumaticParams()
    v_eff = effective_volume_l(m, p)
    total_q, genuine_q = quiescent_demand_nls(m, p, v_eff)
    fit = fit_consumption(m, p, v_eff)
    leak = leak_endpoint_mm(m, p, fit["aux_nls"])
    d0_bound = healthy_leak_bound_mm(m, p, v_eff)
    hA = p.C_oil_j_per_k * m.oil_b_per_s
    dp_disch = p.dp_sep_bar + p.dp_filter_bar + p.dp_dryer_bar
    return [
        ConstantUpdate(
            "Q_comp_nls", p.Q_comp_nls, p.Q_comp_nls, "ours",
            "PINNED. MetroPT-3 has no flow channel, so only Q_comp/V_res is identifiable; we "
            "hold the delivery and solve the volume. Pinning V_res = 600 L instead would give "
            f"Q_comp = {p.V_res_l * m.compressor_rate / p.P_atm_bar / (1 - p.purge_frac):.1f} NL/s "
            "and leave every pressure rate unchanged.",
        ),
        ConstantUpdate(
            "V_res_l", p.V_res_l, v_eff, "calibrated",
            f"per-cycle endpoint rates {m.charge_rate:+.5f} (charge) and {m.fall_rate:+.5f} "
            f"(fall) sum to {m.compressor_rate:.5f} bar/s = Q_comp*(1-purge)*P_atm/V",
        ),
        ConstantUpdate(
            "P_start_bar", p.P_start_bar, m.p_start, "measured",
            f"median cycle minimum {m.p_load_sample:.3f} bar, corrected by half a {SAMPLE_DT_S:.0f} s "
            f"sample of OFF decay; [R87] only says 'starts below 8.2'",
            tol=0.005,
        ),
        ConstantUpdate(
            "P_stop_bar", p.P_stop_bar, m.p_stop, "measured",
            f"median cycle maximum {m.p_unload_sample:.3f} bar + half a sample of charge slope; "
            f"[R87] says 'stops above 10.2' - confirmed to 0.1 %",
            tol=0.005,
        ),
        ConstantUpdate(
            "P_lps_bar", p.P_lps_bar, p.P_lps_bar, "measured",
            f"unchanged [R87]; {int(m.cycles['LPS_any'].sum())} of {len(m.cycles)} healthy cycles "
            f"touch it, so it is a real, rarely-crossed floor even on a healthy unit",
        ),
        ConstantUpdate(
            "t_hold_s", p.t_hold_s, m.t_unloaded_s - p.t_unload_s, "measured",
            f"unloaded run {m.t_unloaded_s:.0f} s per cycle (p5-p95 {np.percentile(m.cycles['t_unloaded'],5):.0f}"
            f"-{np.percentile(m.cycles['t_unloaded'],95):.0f} s: a fixed timer, not a pressure "
            f"threshold) minus the {p.t_unload_s:.0f} s [R155] blowdown",
        ),
        ConstantUpdate(
            "unloaded_vent_nls", getattr(p, "unloaded_vent_nls", 0.0), fit["unloaded_vent_nls"],
            "calibrated",
            f"the unloaded phase falls at {m.unloaded_rate:+.5f} bar/s, faster than leak + "
            f"auxiliaries at that pressure: the separator and unloader keep bleeding after the "
            f"motor offloads",
        ),
        ConstantUpdate(
            "tower_period_s", p.tower_period_s, m.tower_period_s, "measured",
            f"median loaded-seconds between TOWERS flips; {m.tower_flip_loaded_frac:.0%} of flips "
            f"happen while loaded [rail_phm 2.3b: measure this, do not invent it]",
        ),
        ConstantUpdate(
            "I_unloaded_a", p.I_unloaded_a, m.i_unloaded, "measured",
            f"median Motor_current in the unloaded cluster (n={int((m.cycles['t_unloaded'] > 0).sum())} "
            f"cycles); [R87]'s nominal is 4 A",
        ),
        ConstantUpdate(
            "I_loaded_a", p.I_loaded_a, m.i_loaded_intercept - m.i_loaded_kp * dp_disch, "calibrated",
            f"regression intercept {m.i_loaded_intercept:.3f} A at P = P_start, referred back "
            f"through the {dp_disch:.2f} bar discharge drop the module adds; measured loaded "
            f"median {m.i_loaded:.2f} A, NOT [R87]'s nominal 7 A",
        ),
        ConstantUpdate(
            "I_loaded_kp", p.I_loaded_kp, m.i_loaded_kp, "calibrated",
            f"OLS of Motor_current on Reservoirs over the loaded samples of {len(m.cycles)} "
            f"cycles, R2 = {m.i_loaded_r2:.2f}, residual sd 0.16 A",
        ),
        ConstantUpdate(
            "I_loaded_kT", p.I_loaded_kT, m.i_loaded_kT, "calibrated",
            "same regression: current FALLS with oil temperature (thinner oil, less viscous "
            "drag). The old +0.02 A/K had the sign wrong.",
        ),
        ConstantUpdate(
            "I_start_a", p.I_start_a, p.I_start_a, "measured",
            f"confirmed: the largest Motor_current sample anywhere in the normal window is "
            f"{m.i_start_peak_max:.2f} A. A direct-on-line peak is under-sampled at 10 s, so the "
            f"9.0 A is a floor on the true peak, not a fit.",
        ),
        ConstantUpdate(
            "T_oil_ref_c", p.T_oil_ref_c, m.oil_median_c, "measured",
            "median Oil_temperature on the normal window - the point the current-vs-temperature "
            "regression is centred on, so the two must agree",
        ),
        ConstantUpdate(
            "C_oil_j_per_k", p.C_oil_j_per_k, p.C_oil_j_per_k, "ours",
            f"PINNED. The regression identifies hA/C = {m.oil_b_per_s:.6f}/s and P_state/hA, "
            f"never the three separately; 25 kJ/K is ~13 L of compressor oil (1.9 kJ/kg/K, "
            f"870 kg/m3) plus the separator shell.",
        ),
        ConstantUpdate(
            "hA_oil_w_per_k", p.hA_oil_w_per_k, hA, "calibrated",
            f"thermal regression hA/C = {m.oil_b_per_s:.6f}/s (tau {m.oil_tau_s:.0f} s, "
            f"R2 = {m.oil_r2:.2f}) times the pinned C",
        ),
        ConstantUpdate(
            "P_heat_loaded_w", p.P_heat_loaded_w, hA * m.oil_rise_loaded_k, "calibrated",
            f"steady loaded temperature sits {m.oil_rise_loaded_k:.1f} K above the OFF asymptote; "
            f"x hA",
        ),
        ConstantUpdate(
            "P_heat_unloaded_w", p.P_heat_unloaded_w, hA * m.oil_rise_unloaded_k, "calibrated",
            f"steady unloaded temperature sits {m.oil_rise_unloaded_k:.1f} K above the OFF "
            f"asymptote; x hA",
        ),
        ConstantUpdate(
            "T_sump_offset_c", getattr(p, "T_sump_offset_c", 0.0), m.oil_sink_c - m.oil_ambient_c,
            "calibrated",
            f"OFF asymptote {m.oil_sink_c:.1f} C minus the record's own cold-start ambient "
            f"{m.oil_ambient_c:.1f} C (oil while charging an empty reservoir from < 2 bar). The "
            f"fast oil node relaxes to the warm compressor body, not to outdoor air.",
        ),
        ConstantUpdate(
            "aux_nls", p.aux_nls, fit["aux_nls"], "calibrated",
            f"OFF decay {m.off_rate:+.5f} bar/s over {v_eff:.0f} L = {total_q:.3f} NL/s quiescent "
            f"draw, of which {genuine_q:.3f} NL/s is not the healthy leak",
        ),
        ConstantUpdate(
            "brake_nl_per_stop", p.brake_nl_per_stop, fit["brake_nl_per_stop"], "calibrated",
            f"burst scale {fit['burst_scale']:.3f} from the 24 measured hour bands "
            f"(mean demand {fit['target_mean_nls']:.3f} NL/s measured vs "
            f"{fit['fitted_mean_nls']:.3f} fitted, rms residual {fit['rms_residual_nls']:.3f})",
        ),
        ConstantUpdate(
            "spring_nl_per_dwell", p.spring_nl_per_dwell, fit["spring_nl_per_dwell"], "calibrated",
            f"same burst scale {fit['burst_scale']:.3f}; the brake / air-spring ratio is untouched",
        ),
        ConstantUpdate(
            "spring_nl_per_unit_load", p.spring_nl_per_unit_load, fit["spring_nl_per_unit_load"],
            "calibrated", f"same burst scale {fit['burst_scale']:.3f}",
        ),
        ConstantUpdate(
            "sensors['Reservoirs'].noise_sigma", p.sensors["Reservoirs"].noise_sigma, m.sigma_press,
            "measured", "second-difference high-frequency noise of Reservoirs, pooled over the "
            "three compressor states by duration (off 0.0061, unloaded 0.0129, loaded 0.0254 bar)",
            tol=0.25,
        ),
        ConstantUpdate(
            "sensors['Reservoirs'].quantum", p.sensors["Reservoirs"].quantum, m.quantum_press,
            "measured", "smallest gap between distinct reported values on every pressure channel",
        ),
        ConstantUpdate(
            "sensors['Oil_temperature'].noise_sigma", p.sensors["Oil_temperature"].noise_sigma,
            m.sigma_oil, "measured", "same method on Oil_temperature", tol=0.25,
        ),
        ConstantUpdate(
            "sensors['Oil_temperature'].quantum", p.sensors["Oil_temperature"].quantum,
            m.quantum_oil, "measured", "smallest gap between distinct Oil_temperature values",
        ),
        ConstantUpdate(
            "sensors['Motor_current'].noise_sigma", p.sensors["Motor_current"].noise_sigma,
            m.sigma_current, "measured",
            "same method on Motor_current: instrument floor 0.0011 A with the motor off, real "
            "ripple 0.147 A while loaded; the duration-pooled figure confirms the first cut",
            tol=0.25,
        ),
        ConstantUpdate(
            "sensors['Motor_current'].quantum", p.sensors["Motor_current"].quantum,
            m.quantum_current, "measured", "smallest gap between distinct Motor_current values",
        ),
        ConstantUpdate(
            "leak_d0_mm", p.leak_d0_mm, p.leak_d0_mm, "derived",
            f"confirmed as an upper-bounded fit: attributing the WHOLE quiescent OFF draw to the "
            f"leak gives an equivalent {d0_bound:.2f} mm sharp orifice, so the plan's 0.55 mm is "
            f"inside the measurement with {genuine_q:.3f} NL/s left over for auxiliaries",
        ),
        ConstantUpdate(
            "leak_d1_mm", p.leak_d1_mm, leak["chosen_d1_mm"], "calibrated",
            f"the 29 May-7 Jun episode settles at {m.leak_equilibrium_bar:.2f} bar while running "
            f"continuously => {leak['observed_d1_mm']:.2f} mm; s = 1 is defined as the leak that "
            f"just beats the compressor at the {p.P_lps_bar:.1f} bar LPS trip => "
            f"{leak['lps_d1_mm']:.2f} mm, worst case {leak['worst_case_nls']:.2f} NL/s "
            f"({28.3 / leak['worst_case_nls']:.1f}x below the [R156] freight bound)",
        ),
    ]


# ------------------------------------------------------------------------------- plots


def _style(ax) -> None:
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _fig(title: str, subtitle: Sequence[str], nrows: int = 1, ncols: int = 1, figsize=(11.0, 4.6)):
    """Figure with a left-aligned title and wrapped subtitle lines that never overlap the axes.

    Offsets are computed in inches and converted to figure fractions, so a 4.4-inch and a
    7.0-inch figure get the same visual header rather than the same fraction of their height.
    """
    import matplotlib.pyplot as plt

    height = float(figsize[1])
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.02, y=1.0 - 0.28 / height, ha="left")
    for i, line in enumerate(subtitle):
        fig.text(0.02, 1.0 - (0.56 + 0.20 * i) / height, line, fontsize=8.5, color="#444444", ha="left")
    top = 1.0 - (0.62 + 0.20 * len(subtitle)) / height
    return fig, np.atleast_1d(axes).ravel(), top


def _save(fig, path: Path, top: float) -> Path:
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    fig.savefig(path, dpi=140)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def plot_band(m: Measurements, sim: pd.DataFrame | None, out_dir: Path) -> Path:
    fig, ax, _top = _fig(
        "MetroPT-3 control band and charge slope",
        [
            f"normal window {NORMAL_WINDOW[0]} to {NORMAL_WINDOW[1]}, {len(m.cycles)} compressor cycles, "
            f"{m.n_excluded} rows dropped by the +/-{EXCLUSION_DAYS:.0f} d failure guard",
            f"switch points corrected for the {SAMPLE_DT_S:.0f} s sampling lag: "
            f"start {m.p_start:.2f} bar, stop {m.p_stop:.2f} bar",
        ],
        1, 3, figsize=(13.0, 4.4),
    )
    band_bins = np.linspace(
        float(np.percentile(m.cycles["P_res_min"], 0.5)) - 0.1,
        float(np.percentile(m.cycles["P_res_max"], 99.5)) + 0.1,
        70,
    )
    ax[0].hist(m.cycles["P_res_min"], bins=band_bins, color="#4c72b0", alpha=0.8, label="cycle minimum")
    ax[0].hist(m.cycles["P_res_max"], bins=band_bins, color="#c44e52", alpha=0.8, label="cycle maximum")
    for v, c, lab in ((m.p_start, "#4c72b0", "P_start"), (m.p_stop, "#c44e52", "P_stop")):
        ax[0].axvline(v, color=c, ls="--", lw=1.4, label=f"{lab} {v:.2f}")
    ax[0].set_xlabel("Reservoirs (bar)")
    ax[0].set_ylabel("cycles")
    ax[0].set_title("control band", fontsize=10)
    ax[0].legend(fontsize=7)
    _style(ax[0])

    def _rate_panel(axis, real, sim_col, colour, title, xlabel, marker):
        real = real[np.isfinite(real)]
        lo, hi = np.percentile(real, [0.5, 99.5])
        if sim_col is not None:
            s = sim_col[np.isfinite(sim_col)]
            lo, hi = min(lo, float(np.percentile(s, 0.5))), max(hi, float(np.percentile(s, 99.5)))
        pad = 0.08 * (hi - lo) or 1e-4
        bins = np.linspace(lo - pad, hi + pad, 60)
        axis.hist(real, bins=bins, density=True, color=colour, alpha=0.85, label="MetroPT-3")
        if sim_col is not None:
            axis.hist(sim_col, bins=bins, density=True, color="#dd8452", alpha=0.6, label="simulator")
        axis.axvline(marker, color=colour, ls="--", lw=1.4)
        axis.set_xlabel(xlabel)
        axis.set_title(title, fontsize=10)
        axis.legend(fontsize=7)
        _style(axis)

    _rate_panel(
        ax[1], m.cycles["dP_dt_loaded"].to_numpy(dtype=float),
        None if sim is None else sim["dP_dt_loaded"].to_numpy(dtype=float),
        "#55a868", "charge slope -> V_eff", "dP/dt while loaded (bar/s)", m.dP_dt_loaded,
    )
    _rate_panel(
        ax[2], m.cycles["dP_dt_off"].to_numpy(dtype=float),
        None if sim is None else sim["dP_dt_off"].to_numpy(dtype=float),
        "#4c72b0", "OFF decay -> quiescent demand", "dP/dt while off (bar/s)", m.dP_dt_off,
    )
    return _save(fig, out_dir / "cal_pneumatic_band.png", _top)


def plot_cycles(m: Measurements, sim: pd.DataFrame | None, ks: pd.DataFrame | None, out_dir: Path) -> Path:
    cols = ("t_loaded", "t_off", "I_loaded_mean", "T_oil_max")
    ksmap = {r.feature: r.ks for r in ks.itertuples()} if ks is not None else {}
    fig, ax, _top = _fig(
        "Cycle features: MetroPT-3 healthy vs simulator",
        [
            "two-sample KS distance in each panel title; the simulator runs the Singapore service "
            "generator, so T_oil sits above Porto by the ambient difference",
            f"MetroPT-3 n={len(m.cycles)} cycles" + (f", simulator n={len(sim)}" if sim is not None else ""),
        ],
        2, 2, figsize=(11.5, 7.0),
    )
    for a, col in zip(ax, cols):
        real = m.cycles[col].to_numpy(dtype=float)
        real = real[np.isfinite(real)]
        lo, hi = np.nanpercentile(real, [0.5, 99.5])
        if sim is not None:
            s = sim[col].to_numpy(dtype=float)
            s = s[np.isfinite(s)]
            lo = min(lo, float(np.nanpercentile(s, 0.5)))
            hi = max(hi, float(np.nanpercentile(s, 99.5)))
        bins = np.linspace(lo, hi, 55)
        a.hist(real, bins=bins, density=True, color="#4c72b0", alpha=0.8, label="MetroPT-3")
        if sim is not None:
            a.hist(sim[col].to_numpy(dtype=float), bins=bins, density=True, color="#dd8452", alpha=0.6, label="sim")
        title = col if col not in ksmap else f"{col}   KS = {ksmap[col]:.3f}"
        a.set_title(title, fontsize=10)
        a.set_xlabel(col)
        a.legend(fontsize=7)
        _style(a)
    return _save(fig, out_dir / "cal_pneumatic_cycles.png", _top)


def plot_oil(m: Measurements, sim: pd.DataFrame | None, out_dir: Path) -> Path:
    p = PN.PneumaticParams()
    fig, ax, _top = _fig(
        "Oil node: thermal fit and duty coupling",
        [
            f"dT/dt = (P_state + hA*T_sink - hA*T)/C fitted on {NORMAL_WINDOW[0]}..{NORMAL_WINDOW[1]}: "
            f"hA/C = {m.oil_b_per_s:.5f}/s (tau {m.oil_tau_s:.0f} s), R2 = {m.oil_r2:.2f}",
            f"OFF asymptote {m.oil_sink_c:.1f} C vs cold-start ambient {m.oil_ambient_c:.1f} C "
            f"-> the fast node relaxes to a warm body, not to outdoor air",
        ],
        1, 3, figsize=(13.0, 4.4),
    )
    states = ("off", "unloaded", "loaded")
    rises = (0.0, m.oil_rise_unloaded_k, m.oil_rise_loaded_k)
    ax[0].bar(states, [m.oil_sink_c + r for r in rises], color=["#4c72b0", "#dd8452", "#c44e52"])
    ax[0].axhline(m.oil_sink_c, color="#444444", ls="--", lw=1.2, label=f"sink {m.oil_sink_c:.1f} C")
    ax[0].axhline(m.oil_ambient_c, color="#55a868", ls=":", lw=1.4, label=f"cold-start ambient {m.oil_ambient_c:.1f} C")
    ax[0].set_ylabel("steady-state oil temperature (C)")
    ax[0].set_title("per-state steady state", fontsize=10)
    ax[0].legend(fontsize=7)
    _style(ax[0])

    ax[1].scatter(m.cycles["duty_ratio"], m.cycles["T_oil_max"], s=4, alpha=0.18, color="#4c72b0", label="MetroPT-3")
    if sim is not None:
        ax[1].scatter(sim["duty_ratio"], sim["T_oil_max"], s=4, alpha=0.25, color="#dd8452", label="sim")
    ax[1].set_xlabel("duty_ratio")
    ax[1].set_ylabel("T_oil_max (C)")
    ax[1].set_title(f"oil vs duty (hourly slope {m.oil_duty_slope_k:+.1f} K, r={m.oil_duty_corr:.2f})", fontsize=10)
    ax[1].legend(fontsize=7)
    _style(ax[1])

    hours = np.arange(24)
    ax[2].plot(hours, m.duty_by_hour, "o-", color="#4c72b0", lw=1.4, ms=3.5, label="MetroPT-3")
    if sim is not None:
        sim_hour = sim.groupby(sim["hour_of_day"].astype(int))["duty_ratio"].median().reindex(hours)
        ax[2].plot(hours, sim_hour.to_numpy(), "s-", color="#dd8452", lw=1.4, ms=3.5, label="sim (cycle median)")
    ax[2].set_xlabel("hour of day")
    ax[2].set_ylabel("loaded fraction")
    ax[2].set_title("duty per hour band", fontsize=10)
    ax[2].legend(fontsize=7)
    _style(ax[2])
    return _save(fig, out_dir / "cal_pneumatic_oil.png", _top)


def plot_leak(m: Measurements, raw: pd.DataFrame, out_dir: Path) -> Path:
    p = PN.PneumaticParams()
    lo, hi = (pd.Timestamp(x) for x in LEAK_WINDOW)
    leak_frame = raw[(raw["timestamp"] >= lo) & (raw["timestamp"] < hi)].reset_index(drop=True)
    leak_cycles = extract_cycles(leak_frame)
    fit = fit_consumption(m, p, effective_volume_l(m, p))
    ends = leak_endpoint_mm(m, p, fit["aux_nls"])
    fig, ax, _top = _fig(
        "Leak endpoint: the 29 May - 7 Jun 2020 episode sets s = 1",
        [
            f"healthy t_off median {m.cycles['t_off'].median():.0f} s -> episode median "
            f"{m.t_off_leak_window_s:.0f} s ({m.t_off_leak_window_s / m.cycles['t_off'].median():.0%} of healthy)",
            f"equilibrium during the episode's longest continuous run {m.leak_equilibrium_bar:.2f} bar "
            f"-> {ends['observed_d1_mm']:.2f} mm; s=1 taken at the LPS trip -> {ends['chosen_d1_mm']:.2f} mm",
        ],
        1, 3, figsize=(13.0, 4.4),
    )
    ax[0].plot(leak_frame["timestamp"], leak_frame["Reservoirs"], lw=0.5, color="#c44e52")
    ax[0].axhline(p.P_lps_bar, color="#444444", ls="--", lw=1.2, label=f"LPS {p.P_lps_bar} bar")
    ax[0].axhline(m.leak_equilibrium_bar, color="#4c72b0", ls=":", lw=1.4, label=f"equilibrium {m.leak_equilibrium_bar:.2f}")
    ax[0].set_ylabel("Reservoirs (bar)")
    ax[0].set_title("episode reservoir trace", fontsize=10)
    ax[0].legend(fontsize=7)
    ax[0].tick_params(axis="x", labelrotation=30, labelsize=7)
    _style(ax[0])

    bins = np.linspace(0, float(np.nanpercentile(m.cycles["t_off"], 99)), 50)
    ax[1].hist(m.cycles["t_off"], bins=bins, density=True, color="#4c72b0", alpha=0.8, label="healthy Feb-Mar")
    ax[1].hist(leak_cycles["t_off"], bins=bins, density=True, color="#c44e52", alpha=0.6, label="29 May-7 Jun")
    ax[1].set_xlabel("t_off (s)")
    ax[1].set_title("the t_off shrinkage s=1 must reproduce", fontsize=10)
    ax[1].legend(fontsize=7)
    _style(ax[1])

    sev = np.linspace(0.0, 1.0, 200)
    for d1, lab, c in (
        (p.leak_d1_mm, f"current default {p.leak_d1_mm:.2f} mm", "#8172b2"),
        (ends["observed_d1_mm"], f"episode equilibrium {ends['observed_d1_mm']:.2f} mm", "#4c72b0"),
        (ends["chosen_d1_mm"], f"LPS-trip endpoint {ends['chosen_d1_mm']:.2f} mm", "#c44e52"),
    ):
        pp = p.with_(leak_d1_mm=d1)
        q = PN.leak_nls(p.P_start_bar + p.P_atm_bar, PN.orifice_diameter_mm(sev, pp), pp)
        ax[2].plot(sev, q, lw=1.6, color=c, label=lab)
    net = p.Q_comp_nls * (1.0 - p.purge_frac)
    ax[2].axhline(net, color="#444444", ls="--", lw=1.2, label=f"net delivery {net:.2f} NL/s")
    ax[2].set_xlabel("severity")
    ax[2].set_ylabel("leak at 8.1 bar (NL/s)")
    ax[2].set_title("leak law vs endpoint choice", fontsize=10)
    ax[2].legend(fontsize=7)
    _style(ax[2])
    return _save(fig, out_dir / "cal_pneumatic_leak.png", _top)


def make_plots(m: Measurements, raw: pd.DataFrame, sim: pd.DataFrame | None, ks: pd.DataFrame | None, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_band(m, sim, out_dir),
        plot_cycles(m, sim, ks, out_dir),
        plot_oil(m, sim, out_dir),
        plot_leak(m, raw, out_dir),
    ]


# -------------------------------------------------------------------------------- report


def report(
    m: Measurements,
    updates: list[ConstantUpdate],
    ks: pd.DataFrame | None,
    phase: dict[str, float] | None = None,
) -> str:
    p = PN.PneumaticParams()
    v_eff = effective_volume_l(m, p)
    total_q, genuine_q = quiescent_demand_nls(m, p, v_eff)
    lines = [
        "=" * 96,
        f"MetroPT-3 normal window  {m.t_min:%Y-%m-%d} .. {m.t_max:%Y-%m-%d}   "
        f"{m.n_rows:,} rows @ {m.sample_dt_s:.0f} s   {len(m.cycles):,} compressor cycles",
        f"  failure guard (+/-{EXCLUSION_DAYS:.0f} d around {len(FAILURE_EPISODES)} UCI episodes) removed "
        f"{m.n_excluded:,} rows from this window",
        "-" * 96,
        "CONTROL BAND",
        f"  cycle min / max pressure      {m.p_load_sample:6.3f} / {m.p_unload_sample:6.3f} bar   (band {m.band_bar:.3f} bar)",
        f"  switch points, lag-corrected  P_start {m.p_start:6.3f}  P_stop {m.p_stop:6.3f} bar",
        "RATES",
        f"  charge slope (loaded)         {m.dP_dt_loaded:+.5f} bar/s",
        f"  decay (unloaded / off)        {m.dP_dt_unloaded:+.5f} / {m.dP_dt_off:+.5f} bar/s",
        f"  => effective volume           {v_eff:.1f} L   (with Q_comp = {p.Q_comp_nls} NL/s pinned; only the ratio is identifiable)",
        f"  => quiescent draw             {total_q:.3f} NL/s total, {genuine_q:.3f} NL/s after the healthy leak",
        f"  => mean draw (air balance)    {m.duty_cycles * p.Q_comp_nls * (1 - p.purge_frac):.3f} NL/s at cycle duty "
        f"{m.duty_cycles:.4f}  (raw time-weighted duty {m.duty_time_weighted:.4f} incl. logger gaps)",
        f"  => draw per phase             loaded {p.Q_comp_nls * (1 - p.purge_frac) - m.charge_rate * v_eff / p.P_atm_bar:.3f}"
        f" / unloaded {abs(m.unloaded_rate) * v_eff / p.P_atm_bar:.3f} / off {abs(m.off_rate) * v_eff / p.P_atm_bar:.3f} NL/s"
        " - the compressor loads BECAUSE demand is high, so the phases cannot be equal",
        "DURATIONS (median per cycle)",
        f"  t_loaded / t_unloaded / t_off {m.t_loaded_s:6.0f} / {m.t_unloaded_s:6.0f} / {m.t_off_s:6.0f} s",
        f"  idle_run_ratio                {m.idle_run_ratio:.2f}   (DOE well-maintained band is >= 9.0)",
        "CURRENT",
        f"  off / unloaded / loaded       {m.i_off:.3f} / {m.i_unloaded:.3f} / {m.i_loaded:.3f} A   (max seen {m.i_start_peak_max:.2f} A)",
        f"  I = {m.i_loaded_intercept:.3f} {m.i_loaded_kp:+.4f}*(P-P_start) {m.i_loaded_kT:+.5f}*(T-{m.oil_median_c:.0f})   R2 {m.i_loaded_r2:.2f}",
        "DRYER",
        f"  tower period                  {m.tower_period_s:.0f} s of loaded time   ({m.tower_flip_loaded_frac:.0%} of flips while loaded)",
        "OIL NODE",
        f"  hA/C {m.oil_b_per_s:.6f} /s  (tau {m.oil_tau_s:.0f} s)   R2 {m.oil_r2:.2f}",
        f"  sink {m.oil_sink_c:.1f} C   rise unloaded {m.oil_rise_unloaded_k:+.1f} K   loaded {m.oil_rise_loaded_k:+.1f} K",
        f"  cold-start ambient {m.oil_ambient_c:.1f} C  =>  sump offset {m.oil_sink_c - m.oil_ambient_c:.1f} K",
        f"  hourly T_oil = a {m.oil_duty_slope_k:+.1f} K x duty   (r = {m.oil_duty_corr:.2f})",
        "MEASUREMENT CHAIN (second-difference high-frequency sigma, duration-pooled over states)",
        f"  pressure  sigma {m.sigma_press:.4f} bar   quantum {m.quantum_press:.4f} bar",
        f"  oil temp  sigma {m.sigma_oil:.4f} K     quantum {m.quantum_oil:.4f} K",
        f"  current   sigma {m.sigma_current:.4f} A     quantum {m.quantum_current:.4f} A",
        "LEAK ACCEPTANCE WINDOW  " + " .. ".join(LEAK_WINDOW),
        f"  t_off {m.t_off_leak_window_s:.0f} s vs healthy {m.t_off_s:.0f} s "
        f"({m.t_off_leak_window_s / m.t_off_s:.0%})   duty {m.leak_window_duty:.3f}",
        f"  equilibrium during the longest continuous run: {m.leak_equilibrium_bar:.2f} bar",
        "=" * 96,
        f"{'constant':<26}{'current':>12}{'MetroPT-3':>12}  {'status':<10}{'tag':<12}evidence",
        "-" * 96,
    ]
    for u in updates:
        lines.append(f"{u.name:<26}{u.old:>12.4g}{u.new:>12.4g}  {u.status:<10}{u.tag:<12}{u.evidence}")
    n_change = sum(u.changed for u in updates)
    lines.append("-" * 96)
    lines.append(f"{n_change} of {len(updates)} constants differ from the current defaults by more than 2 %.")
    if ks is not None:
        lines += ["=" * 96, "KS DISTANCE, MetroPT-3 healthy cycles vs simulator healthy cycles", "-" * 96]
        lines.append(f"{'feature':<18}{'MetroPT-3':>12}{'sim':>12}{'sim/real':>10}{'KS':>8}{'n_real':>9}{'n_sim':>8}")
        for r in ks.itertuples():
            lines.append(
                f"{r.feature:<18}{r.metropt3_median:>12.4g}{r.sim_median:>12.4g}"
                f"{r.ratio:>10.2f}{r.ks:>8.3f}{r.n_real:>9d}{r.n_sim:>8d}"
            )
    if phase is not None:
        real_phase = {
            "loaded": p.Q_comp_nls * (1 - p.purge_frac) - m.charge_rate * v_eff / p.P_atm_bar,
            "unloaded": abs(m.unloaded_rate) * v_eff / p.P_atm_bar,
            "off": abs(m.off_rate) * v_eff / p.P_atm_bar,
        }
        lines += [
            "-" * 96,
            "DRAW PER PHASE (median over cycles, endpoint rate; the heavy-tailed burst schedule "
            "has to reproduce the asymmetry)",
            f"{'phase':<18}{'MetroPT-3':>12}{'sim':>12}{'sim/real':>10}{'n_sim':>8}",
        ]
        for name in ("loaded", "unloaded", "off"):
            a, b = real_phase[name], phase[name]
            lines.append(f"{name:<18}{a:>12.3f}{b:>12.3f}{b / a:>10.2f}{int(phase['n_' + name]):>8d}")
        lines.append(
            f"  simulated duty (time-weighted) {phase['duty_time']:.4f} against the measured "
            f"{m.duty_time_weighted:.4f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, default=REPO_ROOT / "data" / "raw" / "metropt3")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "sim_checks")
    ap.add_argument("--table", type=Path, default=None, help="write the MetroPT-3 cycle table here")
    ap.add_argument("--days", type=int, default=21, help="simulated days for the KS comparison")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--no-sim", action="store_true", help="skip the simulator run and the KS table")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--leak-ramp",
        action="store_true",
        help="also run the 30-day air-leak ramp and print the lead-time table of section 3 "
        "(~6 min: 4 healthy baselines + 12 ramps)",
    )
    ap.add_argument(
        "--leak-ramp-seeds",
        type=str,
        default=",".join(str(s) for s in LEAK_RAMP_SEEDS),
        help="comma-separated seeds for --leak-ramp",
    )
    args = ap.parse_args(argv)

    raw = load_metropt3(args.raw)
    mask, n_excluded = normal_window_mask(raw["timestamp"])
    frame = raw[mask].reset_index(drop=True)
    m = measure(frame, raw)
    object.__setattr__(m, "n_excluded", n_excluded)

    sim = None if args.no_sim else simulate_cycles(days=args.days, seed=args.seed)
    ks = None if sim is None else ks_table(m.cycles, sim)
    phase = None if args.no_sim else sim_phase_demand_nls(seed=args.seed)
    updates = recommend_constants(m)

    if not args.quiet:
        print(report(m, updates, ks, phase))
    if args.leak_ramp:
        seeds = tuple(int(x) for x in str(args.leak_ramp_seeds).split(",") if x.strip())
        ramp = leak_ramp_lead_time(seeds=seeds)
        print(leak_ramp_report(ramp, m.cycles))
    if args.table is not None:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        m.cycles.to_csv(args.table, index=False)
        print(f"wrote {args.table}")
    if not args.no_plots:
        for path in make_plots(m, raw, sim, ks, args.out_dir):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
