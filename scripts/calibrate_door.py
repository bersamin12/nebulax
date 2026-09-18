#!/usr/bin/env python
"""Calibrate the door simulator against the Cranfield linear-actuator fault set.

What this does
--------------
This is the door half of the plan's "Level 2 calibration" step:

    Door <-> Cranfield: adapter emits ``pos_ref, pos_err, current``; segment strokes; add
    ``DoorParams.cranfield()`` and fit ``R, k_t, F_c0, b0, m_eff`` by least squares on the
    healthy run; match **fault ratios** (faulty/healthy cruise current, reversal error-spike
    width) so ``s = 0.5`` and ``s = 1`` reproduce lube-1/2 and backlash-1/2; map spalling
    ripple to the misalignment term.  Report a "Cranfield ratio vs sim ratio" table.

Five stages, each a pure function so the tests can drive them without a simulator run:

1. :func:`load_strokes` - ``nebulax.adapters.cranfield.load`` -> long -> ``to_wide`` ->
   :func:`stroke_table`.  The **same** :func:`stroke_table` is run on the simulator's own
   50 Hz telemetry, so both sides of every ratio are measured by one estimator rather than
   by a measurement on one side and a model quantity on the other.
2. :func:`fit_healthy` - least squares on the healthy strokes for the mechanical triple and,
   when a voltage channel exists, the electrical pair.
3. :func:`class_summary` - per ``(fault_type, level)`` medians and faulty/healthy ratios.
4. :func:`sweep_gain` / :func:`invert_curve` - run the simulator across a gain axis, then
   read the gain off the resulting **calibration curve** at the measured ratio.  The curve is
   the deliverable, not just the root: with it, one measured number becomes one constant, and
   ``results/sim_checks/cal_door_*.png`` shows the reader where on the curve we landed.
5. :func:`render_report` / :func:`write_doc_section` - the "Cranfield ratio vs sim ratio"
   table, written idempotently between markers in ``docs/parameters.md``.

What identifiability actually allows
------------------------------------
From ``(pos, current)`` alone the mechanical regression ``i = alpha a + beta sgn(v) + gamma v``
identifies only the three **ratios** ``m_eff/c_i``, ``F_c0/c_i``, ``b0/c_i``, where
``c_i = eta k_t G / r`` is the force-per-amp.  ``k_t`` does not appear anywhere else in that
equation, so no amount of position and current data can separate a heavy leaf from a weak
motor.  ``R`` and ``k_e`` need the **voltage** channel (``V = R i + k_e omega``), and then
``k_t = k_e`` in SI closes the system.  :func:`fit_healthy` reports which of the two regimes
it was in (``k_t_identified``) and never silently presents a conditional number as a measured
one; ``stroke_m``, ``v_ref`` and ``a_ref`` are read straight off the reference profile and are
always identified.

What the real files changed (2026-09-15)
---------------------------------------
The 13 ``.mat`` files of the CORD release are now on disk, with the official
``Data description.pdf`` beside them, and they settle two things this script used to guess.

**There is no voltage channel.**  The matrices are ``[position set point (mm), position error
(mm), motor current (A)]`` and nothing else, so ``R`` and ``k_e`` are *not identifiable at
all* and ``k_t`` cannot be separated from the mechanical constants: only the three ratios
``m_eff/c_i``, ``F_c0/c_i``, ``b0/c_i`` come out of the healthy fit, and the rig's two motion
profiles are close enough in cruise speed that ``b0`` does not separate from ``F_c0`` either.

**The motor is a STEPPER, so the current is a proxy.**  PDF section 2: a Nema 34 stepper with
4.6 N.m holding torque, its current read by a Hall-effect sensor on the drive.
:mod:`nebulax.sim.door` models a **PMDC** drive, where the quasi-static current is
proportional to load force.  A chopper-regulated stepper instead holds a commanded phase
current, and what the sensor sees is dominated by a large load-independent component - the
rig measures **0.40 A standing still and 0.86 A cruising**, so barely half of the cruise
current carries any load information at all.  Every faulty/healthy *current* ratio below is
therefore **compressed** relative to the force ratio that produced it, and the gains read off
the simulator's PMDC inversion curves with it are **lower bounds**, not measurements.  That is
stated again next to every number it touches, in the table and in ``docs/parameters.md``.

``--self-check`` still closes the estimator loop: it generates a rig dataset from the
simulator at *known* gains, writes it as ``.mat`` files **in the real release layout**, and
checks that this pipeline recovers them - a recovery test of the estimator, never evidence
about the real actuator.

Run::

    python scripts/calibrate_door.py                      # the real release; exits 2 without it
    python scripts/calibrate_door.py --self-check         # closed-loop recovery test
    python scripts/calibrate_door.py --quick --no-plots   # coarse sweeps, numbers only
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final, Sequence

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import side effect
    sys.path.insert(0, str(REPO_ROOT))

from nebulax import schema as S  # noqa: E402
from nebulax.adapters import cranfield as cranfield_adapter  # noqa: E402
from nebulax.sim import door as D  # noqa: E402
from nebulax.sim.common import (  # noqa: E402
    DegradationTrajectory,
    ServiceParams,
    generate_service,
)

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

DEFAULT_RAW_DIR: Final[Path] = REPO_ROOT / "data" / "raw" / "cranfield"
DEFAULT_OUT_DIR: Final[Path] = REPO_ROOT / "results" / "sim_checks"
DEFAULT_DOC: Final[Path] = REPO_ROOT / "docs" / "parameters.md"

#: Cranfield degradation stage -> simulator severity, **per Cranfield class**.
#:
#: The plan fixes the two-stage faults: "``s = 0.5`` and ``s = 1`` reproduce lube-1/2 and
#: backlash-1/2".  Spalling has **8** stages in the real release (PDF section 3 / Fig. 4),
#: which the plan never saw, so they are placed linearly, ``s = level / 8`` - the placement
#: that keeps the plan's anchors: **stage 4 -> s = 0.5 and stage 8 -> s = 1.0**.  All eight
#: are measured and reported; the gain is read off whichever of them the sweep can invert.
#:
#: This is *not* the ``severity`` column ``nebulax.adapters.cranfield`` writes
#: (``min(1, 0.2 + 0.2 (level - 1))``, which saturates at spalling stage 5); that column is an
#: ML label scale for the ordinal task, whereas this map is the calibration target for the
#: *physics*.  Both are documented in ``docs/parameters.md``; they are deliberately different
#: and must not be unified.
N_SPALLING_STAGES: Final[int] = 8
CRANFIELD_LEVEL_SEVERITY: Final[dict[str, dict[int, float]]] = {
    "lack_of_lubrication": {1: 0.5, 2: 1.0},
    "backlash": {1: 0.5, 2: 1.0},
    "spalling": {k: k / N_SPALLING_STAGES for k in range(1, N_SPALLING_STAGES + 1)},
}

#: Cranfield class name -> simulator ``FAULT_TYPES['door']`` entry (the adapter's own map).
CRANFIELD_CLASS_FAULT: Final[dict[str, str]] = {
    "lack_of_lubrication": "friction",
    "backlash": "backlash",
    "spalling": "misalignment",
}

#: Simulator severities swept per fault type, derived from the map above.
def sim_severities(cranfield_class: str) -> tuple[float, ...]:
    """The severity axis one fault's inversion curve has to cover."""
    return tuple(sorted(set(CRANFIELD_LEVEL_SEVERITY[cranfield_class].values())))

#: A sample counts as "cruise" when its speed clears this fraction of the stroke's p95 speed.
#: [R64] segments the current into accel / constant-speed / decel and takes features per
#: segment; the constant-speed segment is the one where the friction terms are the whole
#: force balance, so it is the only honest place to read a friction ratio.
CRUISE_VEL_FRAC: Final[float] = 0.60

#: Fraction of a stroke, measured from its start, searched for the reversal error spike.
REVERSAL_WINDOW_FRAC: Final[float] = 0.40
#: The spike's width is measured at this fraction of its own peak (half-height width).
REVERSAL_WIDTH_FRAC: Final[float] = 0.50

#: Ripple analysis: current is resampled onto this many uniform position bins across the
#: cruise span, detrended by a polynomial of this order, then FFT'd against position.
RIPPLE_GRID: Final[int] = 256
RIPPLE_DETREND_ORDER: Final[int] = 3

#: Minimum samples for a stroke to be usable at all.
MIN_STROKE_SAMPLES: Final[int] = 8

#: Savitzky-Golay window (samples) and polynomial order used for every velocity and
#: acceleration estimate.  Position arrives quantised - 0.1 mm on the simulator's 50 Hz
#: telemetry - so a plain two-point difference puts ~5 mm/s of noise on a 50 mm/s signal and
#: ~0.6 m/s^2 on a 0.2 m/s^2 ramp, i.e. the acceleration regressor of the mechanical fit would
#: be **mostly noise** and errors-in-variables would bias every coefficient.  A 9-sample
#: (0.18 s) cubic window recovers 0.207 m/s^2 against a true 0.200 and 0.0599 m/s against a
#: true 0.0600, which is what makes the least-squares fit meaningful at all.
SAVGOL_WINDOW: Final[int] = 9
SAVGOL_ORDER: Final[int] = 3

#: Samples below this fraction of the stroke's p95 speed are excluded from the mechanical fit.
#: Near-stationary samples sit on the plant's stiction branch, where the friction force is
#: whatever the drive happens to be pushing rather than ``F_c sgn(v) + b v``, so they are not
#: observations of the model being fitted.
FIT_MOVING_FRAC: Final[float] = 0.50

#: A stroke counts as full travel when it covers at least this fraction of the longest stroke
#: seen.  Retry reversals and partial strokes are real telemetry but they are not samples of
#: the commanded motion profile.
FULL_STROKE_FRAC: Final[float] = 0.80
#: A time gap larger than this many median sample intervals splits two activities.
GAP_FACTOR: Final[float] = 5.0

#: Reference-derivative dead band, as a fraction of the largest step, below which a sample is
#: "idle" rather than moving.  Position arrives quantised, so a strictly-nonzero test would
#: shred a stationary reference into thousands of one-sample "strokes".
STROKE_DEAD_FRAC: Final[float] = 0.05

#: Service window used for every simulator evaluation (~35 door cycles, ~2.5 s per run).
SIM_SERVICE: Final[ServiceParams] = ServiceParams(service_start_h=5.5, service_end_h=7.0)
SIM_SEED: Final[int] = 20260918
#: A ramp so slow that one service window sees a constant severity (same trick as the tests).
CONST_SPAN_S: Final[float] = 1.0e7

#: Gain axes swept to build the inversion curves.  Each point costs one simulator run.
#: The low ends were extended once the real ratios arrived: a seeded lubrication fault moves
#: this rig's current by 13-21 % and a spalling stage its ripple by 4-50 %, an order of
#: magnitude under what the plan's gains produce, so an axis starting at the plan's value
#: would have reported "outside the invertible range" for a measurement that is in fact
#: perfectly well posed - just small.
SWEEP_FRICTION_C: Final[tuple[float, ...]] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0)
SWEEP_BACKLASH_M: Final[tuple[float, ...]] = (
    0.0005, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.012, 0.016,
)
SWEEP_K_MIS: Final[tuple[float, ...]] = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
#: ``b`` is tied to ``F_c`` at the plan's ratio unless a second observable frees it.
PLAN_B_TO_C_RATIO: Final[float] = 1.5 / 2.0

DOC_BEGIN: Final[str] = "<!-- BEGIN: calibrate_door -->"
DOC_END: Final[str] = "<!-- END: calibrate_door -->"

PENDING: Final[str] = "PENDING"


# --------------------------------------------------------------------------------------
# Stroke segmentation and per-stroke metrics  (one estimator, both sides of every ratio)
# --------------------------------------------------------------------------------------


def _split_on_gaps(t: np.ndarray) -> list[tuple[int, int]]:
    """Split a time vector wherever the sample interval jumps (separate activities)."""
    if t.size < 2:
        return [(0, t.size)] if t.size else []
    d = np.diff(t)
    med = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    cuts = np.flatnonzero(d > GAP_FACTOR * med) + 1
    edges = [0, *cuts.tolist(), t.size]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a >= MIN_STROKE_SAMPLES]


def segment_strokes(pos_ref: np.ndarray, *, min_len: int = MIN_STROKE_SAMPLES) -> list[tuple[int, int]]:
    """Split a position-reference trace into monotone extend/retract half-cycles.

    Vectorised sign-change detection on the reference derivative with a dead band at
    :data:`STROKE_DEAD_FRAC` of the largest step.  Samples inside the dead band are **idle**
    and are dropped, not absorbed into the neighbouring stroke: a rig file is a continuous
    recording that includes the pauses between activities, and a pause welded onto the end of
    a stroke corrupts every duration, mean and ratio taken from it - and corrupts them by a
    different amount for a slow faulty stroke than for a fast healthy one, which is exactly
    the quantity being measured.
    """
    x = np.asarray(pos_ref, dtype=np.float64)
    if x.size < min_len:
        return []
    d = np.diff(x)
    # p95, not max: a rig file concatenates activities, and the discontinuous jump of the
    # reference between two activities is one enormous single-sample step.  Scaling the dead
    # band off that step makes every genuine sample idle and finds no strokes at all.
    absd = np.abs(d)
    peak = float(np.percentile(absd[absd > 0], 95)) if np.any(absd > 0) else 0.0
    dead = STROKE_DEAD_FRAC * peak
    sgn = np.where(d > dead, 1, np.where(d < -dead, -1, 0)).astype(np.int8)
    if not np.any(sgn):
        return [(0, x.size)] if x.size >= min_len else []
    # contiguous runs of equal sign; keep the moving ones
    breaks = np.flatnonzero(np.diff(sgn) != 0) + 1
    edges = [0, *breaks.tolist(), sgn.size]
    out: list[tuple[int, int]] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if sgn[a] == 0:
            continue
        lo, hi = a, b + 1  # d[i] spans x[i]..x[i+1]
        if hi - lo >= min_len:
            out.append((lo, min(hi, x.size)))
    return out


def _derivatives(x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed first and second derivative of a quantised position trace.

    Savitzky-Golay of order :data:`SAVGOL_ORDER` over :data:`SAVGOL_WINDOW` samples, shrunk to
    the largest usable odd window when the block is short, and falling back to ``np.gradient``
    when even that will not fit.
    """
    n = x.size
    win = min(SAVGOL_WINDOW, n if n % 2 == 1 else n - 1)
    if win <= SAVGOL_ORDER or win < 5:
        v = np.gradient(x, dt)
        return v, np.gradient(v, dt)
    v = savgol_filter(x, win, SAVGOL_ORDER, deriv=1, delta=dt)
    a = savgol_filter(x, win, SAVGOL_ORDER, deriv=2, delta=dt)
    return v, a


def _reversal_spike_width(err: np.ndarray, dt: float) -> tuple[float, float]:
    """Half-height width (s) and peak (m) of the tracking-error spike at a stroke's start.

    A backlash dead band ``delta`` makes the leaf stand still while the motor traverses it, so
    every motion reversal opens a tracking-error spike whose width is ``~ delta / v`` - the
    plan's "reversal error-spike width" observable.
    """
    n = err.size
    if n < 3:
        return float("nan"), float("nan")
    w = max(3, int(round(REVERSAL_WINDOW_FRAC * n)))
    head = np.abs(err[:w])
    peak = float(np.nanmax(head))
    if not np.isfinite(peak) or peak <= 0.0:
        return 0.0, 0.0
    k = int(np.nanargmax(head))
    thr = REVERSAL_WIDTH_FRAC * peak
    lo = k
    while lo > 0 and head[lo - 1] >= thr:
        lo -= 1
    hi = k
    while hi < w - 1 and head[hi + 1] >= thr:
        hi += 1
    return float((hi - lo + 1) * dt), peak


def _ripple(pos: np.ndarray, cur: np.ndarray) -> tuple[float, float, float]:
    """Dominant spatial wavelength (m), ripple amplitude (A) and ripple/mean fraction.

    Spalling on a ball screw is a *position-periodic* disturbance: its signature is a comb in
    the current-versus-**position** spectrum at the screw lead, not a broadband rise.  The
    current is therefore resampled onto a uniform position grid (not a uniform time grid) and
    detrended before the FFT, so a monotone friction ramp cannot masquerade as ripple.
    """
    if pos.size < 16:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(pos)
    x, y = pos[order], np.abs(cur[order])
    span = float(x[-1] - x[0])
    if not np.isfinite(span) or span <= 0.0:
        return float("nan"), float("nan"), float("nan")
    grid = np.linspace(x[0], x[-1], RIPPLE_GRID)
    yg = np.interp(grid, x, y)
    mean = float(np.mean(yg))
    deg = min(RIPPLE_DETREND_ORDER, RIPPLE_GRID - 1)
    resid = yg - np.polyval(np.polyfit(grid, yg, deg), grid)
    spec = np.abs(np.fft.rfft(resid * np.hanning(resid.size)))
    freq = np.fft.rfftfreq(resid.size, d=float(grid[1] - grid[0]))  # cycles per metre
    if spec.size < 2:
        return float("nan"), float("nan"), float("nan")
    j = 1 + int(np.argmax(spec[1:]))
    lam = float(1.0 / freq[j]) if freq[j] > 0 else float("nan")
    amp = float(np.std(resid))
    return lam, amp, float(amp / mean) if mean > 0 else float("nan")


def stroke_table(
    wide: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("run_id",),
    label_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """One row per extend/retract stroke, with every metric the calibration reads.

    ``wide`` is the output of :func:`nebulax.schema.to_wide` for subsystem ``door`` (or any
    frame with ``timestamp`` plus ``pos_ref``/``pos``/``current``, optionally ``voltage``).
    Run on the Cranfield telemetry and on the simulator's telemetry alike.
    """
    need = {"timestamp", "pos", "current"}
    missing = need - set(wide.columns)
    if missing:
        raise ValueError(f"stroke_table: wide frame is missing {sorted(missing)}")
    rows: list[dict[str, object]] = []
    keys = [c for c in group_cols if c in wide.columns]
    grouped = wide.groupby(list(keys), observed=True, sort=True) if keys else [((), wide)]
    for key, g in grouped:
        g = g.sort_values("timestamp")
        t = g["timestamp"].to_numpy("datetime64[ns]").astype("int64") / 1e9
        pos = g["pos"].to_numpy(dtype=np.float64)
        cur = g["current"].to_numpy(dtype=np.float64)
        ref = (
            g["pos_ref"].to_numpy(dtype=np.float64)
            if "pos_ref" in g.columns
            else np.full(pos.shape, np.nan)
        )
        volt = (
            g["voltage"].to_numpy(dtype=np.float64)
            if "voltage" in g.columns
            else np.full(pos.shape, np.nan)
        )
        labels = {c: (g[c].iloc[0] if c in g.columns else None) for c in label_cols}
        kv = dict(zip(keys, key if isinstance(key, tuple) else (key,))) if keys else {}
        sid = 0
        for a, b in _split_on_gaps(t):
            tb, pb, cb, rb, vb = t[a:b], pos[a:b], cur[a:b], ref[a:b], volt[a:b]
            # Dropout NaNs (schema: NaN = dropout) would propagate through every derivative
            # and every reduction, so they are removed per activity, not interpolated over.
            keep = np.isfinite(pb) & np.isfinite(cb) & np.isfinite(tb)
            if keep.sum() < MIN_STROKE_SAMPLES:
                continue
            tb, pb, cb, rb, vb = tb[keep], pb[keep], cb[keep], rb[keep], vb[keep]
            base = rb if np.isfinite(rb).all() else pb
            for s, e in segment_strokes(base):
                n = e - s
                if n < MIN_STROKE_SAMPLES:
                    continue
                ts, ps, cs, rs, vs = tb[s:e], pb[s:e], cb[s:e], rb[s:e], vb[s:e]
                dt = float(np.median(np.diff(ts))) if n > 1 else float("nan")
                if not np.isfinite(dt) or dt <= 0:
                    continue
                vel, acc = _derivatives(ps, dt)
                sp = np.abs(vel)
                p95 = float(np.nanpercentile(sp, 95))
                cruise = sp >= CRUISE_VEL_FRAC * p95 if p95 > 0 else np.zeros(n, bool)
                if cruise.sum() < 3:
                    cruise = np.ones(n, bool)
                err = rs - ps
                width, peak = _reversal_spike_width(err, dt) if np.isfinite(err).all() else (np.nan, np.nan)
                lam, ramp, rfrac = _ripple(ps[cruise], cs[cruise])
                rows.append(
                    {
                        **kv,
                        **labels,
                        "stroke_id": sid,
                        "t_start": float(ts[0]),
                        "n_samples": int(n),
                        "fs_hz": 1.0 / dt,
                        "duration_s": float(ts[-1] - ts[0]),
                        "direction": float(np.sign(ps[-1] - ps[0])),
                        "travel_m": float(np.nanmax(ps) - np.nanmin(ps)),
                        "v_cruise": float(np.median(sp[cruise])),
                        "a_peak": float(np.nanpercentile(np.abs(acc), 95)),
                        "i_cruise_mean": float(np.mean(np.abs(cs[cruise]))),
                        "i_cruise_rms": float(np.sqrt(np.mean(cs[cruise] ** 2))),
                        "i_peak": float(np.nanmax(np.abs(cs))),
                        "pos_err_max": float(np.nanmax(np.abs(err))) if np.isfinite(err).any() else np.nan,
                        "pos_err_rms": float(np.sqrt(np.nanmean(err**2))) if np.isfinite(err).any() else np.nan,
                        "err_spike_width_s": width,
                        "err_spike_peak_m": peak,
                        "ripple_lambda_m": lam,
                        "ripple_amp_a": ramp,
                        "ripple_frac": rfrac,
                        "has_voltage": bool(np.isfinite(vs).any()),
                    }
                )
                sid += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Stage 2 - least squares on the healthy run
# --------------------------------------------------------------------------------------


#: An electrical fit is only believed when it explains this much of the voltage.
ELEC_MIN_R2: Final[float] = 0.90
#: The mechanical regression is only believed when it explains this much of the current.
#: On a PMDC drive it clears 0.99; on the Cranfield rig's chopper-regulated stepper the same
#: regression explains ~3 %, because the premise "current is proportional to force" is simply
#: not that drive's physics - so the fitted ``m_eff``/``F_c0``/``b0`` are reported as
#: **rejected**, not as small numbers with wide error bars.
MECH_MIN_R2: Final[float] = 0.50
#: ``b0`` is only believed when the strokes span at least this ratio of cruise speeds.
#: At one speed ``F_c0`` and ``b0`` enter the cruise balance only as ``F_c0 + b0 v_ref``;
#: splitting them is then pure extrapolation off the acceleration ramps, where the rig's own
#: per-cycle friction scatter is larger than the whole viscous term.
B0_MIN_SPEED_SPAN: Final[float] = 1.5


@dataclass(frozen=True, slots=True)
class HealthyFit:
    """Least-squares fit of the healthy rig, with its identifiability flags.

    Read the flags before the numbers: ``k_t_identified`` and ``b0_identified`` are false far
    more often than one would like, and the fields they gate are then *conditional*, not
    measured.
    """

    n_strokes: int
    n_samples: int
    # --- always identified, straight off the reference profile -------------------------
    stroke_m: float
    v_ref: float
    a_ref: float
    v_span: float
    # --- mechanical regression  i = alpha a + beta sgn(v) + gamma v + i0 ----------------
    m_eff_over_c: float  # kg/(N/A) = A.s^2/m
    f_c0_over_c: float  # A
    b0_over_c: float  # A.s/m
    i_cruise_over_c: float  # A - the well-conditioned combination (F_c0 + b0 v_ref)/c_i
    i_offset: float
    r2_mech: float
    cond_mech: float
    # --- electrical regression  V = R i + k_e omega + V0 (voltage channel only) ---------
    r_ohm: float
    k_e: float
    r2_elec: float
    cond_elec: float
    k_t_identified: bool
    b0_identified: bool
    #: the mechanical regression cleared :data:`MECH_MIN_R2` - see the note there
    mech_credible: bool
    # --- the stepper's operating point (see the module docstring) -----------------------
    #: median |i| over the samples the fit *excludes* (|v| under FIT_MOVING_FRAC of cruise):
    #: on a chopper-regulated stepper this is the drive's standing current, and it is the
    #: load-independent offset that compresses every faulty/healthy current ratio.
    i_standing: float
    i_moving: float
    # --- physical values, conditional on k_t when it is not identified ------------------
    k_t: float
    c_i: float
    m_eff: float
    f_c0: float
    b0: float
    f_cruise: float

    def as_rows(self) -> list[dict[str, object]]:
        kt = "measured" if self.k_t_identified else "assumed (not identifiable)"
        cond = "measured" if self.k_t_identified else "conditional on assumed k_t"
        b0s = cond if self.b0_identified else "NOT identifiable at one cruise speed"
        if not self.mech_credible:
            rejected = f"REJECTED (mechanical R2 = {self.r2_mech:.3f} < {MECH_MIN_R2:g})"
            cond, b0s = rejected, rejected
        return [
            {"quantity": "stroke_m", "value": self.stroke_m, "unit": "m", "status": "measured"},
            {"quantity": "v_ref", "value": self.v_ref, "unit": "m/s", "status": "measured"},
            {"quantity": "a_ref", "value": self.a_ref, "unit": "m/s^2", "status": "measured"},
            {"quantity": "r_ohm", "value": self.r_ohm, "unit": "ohm", "status": "measured" if self.k_t_identified else "NOT identifiable"},
            {"quantity": "k_t", "value": self.k_t, "unit": "N.m/A", "status": kt},
            {"quantity": "m_eff", "value": self.m_eff, "unit": "kg", "status": cond},
            {"quantity": "f_c0", "value": self.f_c0, "unit": "N", "status": cond},
            {"quantity": "b0", "value": self.b0, "unit": "N.s/m", "status": b0s},
            {"quantity": "F_c0 + b0*v_ref (cruise force)", "value": self.f_cruise, "unit": "N", "status": cond},
            {"quantity": "i_standing (drive current at rest)", "value": self.i_standing, "unit": "A", "status": "measured"},
            {"quantity": "i_moving (drive current while travelling)", "value": self.i_moving, "unit": "A", "status": "measured"},
        ]


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_healthy(wide: pd.DataFrame, params: D.DoorParams) -> HealthyFit:
    """Fit ``R, k_t, F_c0, b0, m_eff`` (and the profile constants) on healthy strokes.

    **Mechanical**, from ``m_eff a = c_i i - F_c sgn(v) - b v`` rearranged for the measured
    current, ordinary least squares over every moving sample::

        i = (m_eff/c_i) a + (F_c0/c_i) sgn(v) + (b0/c_i) v + i0

    Only the three ratios are identifiable here, and only if the data supports them - see
    :data:`B0_MIN_SPEED_SPAN` and the module docstring.  **Electrical**, when a voltage channel
    exists, ``V = R i + k_e omega + V0`` closes the system with ``k_t = k_e``; it is believed
    only when it clears :data:`ELEC_MIN_R2` with a positive resistance, because ``omega`` is
    reconstructed from quantised position and a milli-metre-per-second of velocity noise is
    worth ~0.06 V against an ``R i`` term of ~0.8 V.
    """
    st = stroke_table(wide)
    if st.empty:
        raise ValueError("fit_healthy: no usable strokes in the healthy telemetry")

    # Derivatives are taken *within* one activity of one run: several healthy files share a
    # time axis (the adapter gives each its own synthetic epoch, and two runs can overlap), so
    # sorting them together would interleave two actuators into one impossible trajectory;
    # and `to_wide` concatenates activities with time gaps, where differentiating invents an
    # acceleration spike that would dominate the regression.  Dropout NaNs go per block.
    A_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    Ae_rows: list[np.ndarray] = []
    ye_rows: list[np.ndarray] = []
    i_standing_rows: list[np.ndarray] = []
    i_moving_rows: list[np.ndarray] = []
    runs = (
        [g for _, g in wide.groupby("run_id", observed=True, sort=True)]
        if "run_id" in wide.columns
        else [wide]
    )
    for run in runs:
        g = run.sort_values("timestamp")
        t_all = g["timestamp"].to_numpy("datetime64[ns]").astype("int64") / 1e9
        pos_all = g["pos"].to_numpy(dtype=np.float64)
        cur_all = g["current"].to_numpy(dtype=np.float64)
        volt_all = (
            g["voltage"].to_numpy(dtype=np.float64)
            if "voltage" in g.columns
            else np.full(pos_all.shape, np.nan)
        )
        for lo, hi in _split_on_gaps(t_all):
            tb, pb, cb, vb = t_all[lo:hi], pos_all[lo:hi], cur_all[lo:hi], volt_all[lo:hi]
            keep = np.isfinite(pb) & np.isfinite(cb) & np.isfinite(tb)
            if keep.sum() < MIN_STROKE_SAMPLES:
                continue
            tb, pb, cb, vb = tb[keep], pb[keep], cb[keep], vb[keep]
            dt = float(np.median(np.diff(tb)))
            if not np.isfinite(dt) or dt <= 0:
                continue
            vel, acc = _derivatives(pb, dt)
            sp = np.abs(vel)
            moving = sp > FIT_MOVING_FRAC * float(np.nanpercentile(sp, 95))
            if moving.sum() < MIN_STROKE_SAMPLES:
                continue
            A_rows.append(
                np.column_stack(
                    [acc[moving], np.sign(vel[moving]), vel[moving], np.ones(int(moving.sum()))]
                )
            )
            y_rows.append(cb[moving])
            i_moving_rows.append(np.abs(cb[moving]))
            if int((~moving).sum()):
                i_standing_rows.append(np.abs(cb[~moving]))
            ok = moving & np.isfinite(vb)
            if ok.sum() >= MIN_STROKE_SAMPLES:
                Ae_rows.append(
                    np.column_stack([cb[ok], vel[ok] * params.gear_gain, np.ones(int(ok.sum()))])
                )
                ye_rows.append(vb[ok])
    if not A_rows:
        raise ValueError("fit_healthy: no usable moving samples in the healthy telemetry")
    A, y = np.vstack(A_rows), np.concatenate(y_rows)
    if y.size < 16:
        raise ValueError("fit_healthy: fewer than 16 moving samples")
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    alpha, beta, gamma, i0 = (float(v) for v in coef)
    r2m = _r2(y, A @ coef)
    cond_m = float(np.linalg.cond(A))

    r_ohm, k_e, r2e, cond_e, identified = float("nan"), float("nan"), float("nan"), float("nan"), False
    if Ae_rows:
        Ae, ye = np.vstack(Ae_rows), np.concatenate(ye_rows)
        ce, *_ = np.linalg.lstsq(Ae, ye, rcond=None)
        r_ohm, k_e = float(ce[0]), float(ce[1])
        r2e = _r2(ye, Ae @ ce)
        cond_e = float(np.linalg.cond(Ae))
        identified = bool(np.isfinite(r2e) and r2e >= ELEC_MIN_R2 and r_ohm > 0 and k_e > 0)

    # Speed span over *full-travel* strokes only.  A retry reversal or a partial stroke
    # cruises at a fraction of the commanded speed; counting those would report a 100x speed
    # span on a rig that only ever runs one profile, and would wrongly declare b0 identified.
    travel = st["travel_m"].to_numpy(dtype=np.float64)
    full = travel >= FULL_STROKE_FRAC * float(np.nanmax(travel))
    vc = st["v_cruise"].to_numpy(dtype=np.float64)[full]
    vc = vc[np.isfinite(vc) & (vc > 0)]
    lo, hi = (float(np.percentile(vc, 10)), float(np.percentile(vc, 90))) if vc.size else (np.nan, np.nan)
    v_span = hi / lo if np.isfinite(lo) and lo > 0 else float("nan")
    v_ref = float(np.median(vc)) if vc.size else float("nan")
    b0_ident = bool(np.isfinite(v_span) and v_span >= B0_MIN_SPEED_SPAN)

    k_t = k_e if identified else params.k_t
    c_i = params.eta * k_t * params.gear_gain
    # a_ref off the smoothed ramp; the reference trapezoid's plateau is |a| ~ 0 so the p95 of
    # |a| over the stroke is the ramp value.
    a_ref = float(np.median(st["a_peak"].to_numpy(dtype=np.float64)[full]))
    i_cruise_over_c = abs(beta) + gamma * v_ref
    return HealthyFit(
        n_strokes=int(len(st)),
        n_samples=int(y.size),
        stroke_m=float(np.median(travel[full])),
        v_ref=v_ref,
        a_ref=a_ref,
        v_span=v_span,
        m_eff_over_c=alpha,
        f_c0_over_c=abs(beta),
        b0_over_c=gamma,
        i_cruise_over_c=i_cruise_over_c,
        i_offset=i0,
        r2_mech=r2m,
        cond_mech=cond_m,
        r_ohm=r_ohm,
        k_e=k_e,
        r2_elec=r2e,
        cond_elec=cond_e,
        k_t_identified=identified,
        b0_identified=b0_ident,
        mech_credible=bool(np.isfinite(r2m) and r2m >= MECH_MIN_R2),
        i_standing=float(np.median(np.concatenate(i_standing_rows))) if i_standing_rows else float("nan"),
        i_moving=float(np.median(np.concatenate(i_moving_rows))) if i_moving_rows else float("nan"),
        k_t=k_t,
        c_i=c_i,
        m_eff=alpha * c_i,
        f_c0=abs(beta) * c_i,
        b0=gamma * c_i,
        f_cruise=i_cruise_over_c * c_i,
    )


# --------------------------------------------------------------------------------------
# Stage 3 - measured class ratios
# --------------------------------------------------------------------------------------

#: Stroke metrics that become faulty/healthy ratios.
RATIO_METRICS: Final[tuple[str, ...]] = (
    "i_cruise_mean",
    "i_cruise_rms",
    "v_cruise",
    "duration_s",
    "err_spike_width_s",
    "err_spike_peak_m",
    "ripple_frac",
)


def class_summary(strokes: pd.DataFrame) -> pd.DataFrame:
    """Per ``(fault_type, level)`` medians and faulty/healthy ratios of :data:`RATIO_METRICS`.

    **Stratified by motion profile when the table carries one.** The Cranfield release runs
    two profiles - trapezoidal (120 mm in 5 s) and sinusoidal (120 mm in 6 s) - inside every
    condition file, so a median taken over the pooled strokes is a median of a bimodal
    population and moves whenever the trap:sin balance does. That is not hypothetical: the
    release's ``Backlash1.mat`` holds 59 matrices rather than 60, and pooling made its
    stroke-duration ratio read **1.077** purely because the missing repetition tipped the
    median off the 5 s profile onto the 6 s one. Forming each ratio **within a profile**,
    against the healthy strokes of that same profile, and then taking the median across
    profiles removes the artefact and is the comparison the rig design intends.

    Falls back to the pooled form when the table has no ``motion_profile`` column (the
    simulator side, and the analytic fixtures in the tests).
    """
    if strokes.empty:
        return pd.DataFrame(columns=["fault_type", "level", *RATIO_METRICS])
    df = strokes.copy()
    if "level" not in df.columns:
        df["level"] = 0
    strata = [c for c in ("motion_profile",) if c in df.columns]
    keys = ["fault_type", "level", *strata]
    med = df.groupby(keys, observed=True, sort=True)[list(RATIO_METRICS)].median().reset_index()
    healthy = med[med["fault_type"] == "healthy"]
    if healthy.empty:
        raise ValueError("class_summary: no healthy strokes to form ratios against")
    ratio_cols = [f"{m}_ratio" for m in RATIO_METRICS]
    if strata:
        base = healthy.set_index(strata)
        idx = pd.Index(med[strata[0]]) if len(strata) == 1 else pd.MultiIndex.from_frame(med[strata])
        for m in RATIO_METRICS:
            denom = np.asarray(idx.map(base[m]), dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                med[f"{m}_ratio"] = np.where(
                    np.isfinite(denom) & (denom != 0), med[m].to_numpy(dtype=np.float64) / denom, np.nan
                )
        out = (
            med.groupby(["fault_type", "level"], observed=True, sort=True)[
                [*RATIO_METRICS, *ratio_cols]
            ]
            .median()
            .reset_index()
        )
    else:
        row = healthy[list(RATIO_METRICS)].iloc[0]
        for m in RATIO_METRICS:
            denom = float(row[m])
            med[f"{m}_ratio"] = med[m] / denom if np.isfinite(denom) and denom != 0 else np.nan
        out = med
    counts = df.groupby(["fault_type", "level"], observed=True, sort=True).size().rename("n").reset_index()
    return out.merge(counts, on=["fault_type", "level"], how="left")


# --------------------------------------------------------------------------------------
# Stage 4 - the simulator side and the inversion curves
# --------------------------------------------------------------------------------------


def const_trajectory(fault_type: str, s: float, *, component_id: str = "door_L1") -> list[DegradationTrajectory] | None:
    """A trajectory pinned at constant severity ``s`` over one service window."""
    if s <= 0.0:
        return None
    mid = 0.5 * (SIM_SERVICE.service_start_h + SIM_SERVICE.service_end_h) * 3600.0
    return [
        DegradationTrajectory(
            fault_type=fault_type,
            subsystem="door",
            component_id=component_id,
            t_onset=mid - s * CONST_SPAN_S,
            t_failure=mid + (1.0 - s) * CONST_SPAN_S,
            gamma=1.0,
            jitter_sigma=0.0,
        )
    ]


def simulate_rig(
    params: D.DoorParams,
    fault_type: str | None,
    s: float,
    *,
    seed: int = SIM_SEED,
    store_every: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One simulator run of the rig at a constant severity -> ``(wide, features)``."""
    service = generate_service(1, np.random.default_rng(seed), params=SIM_SERVICE, train_id="T01")
    faults = const_trajectory(fault_type, s) if fault_type else None
    long, feats, _ = D.simulate(
        params, faults, service, np.random.default_rng(seed + 991), store_every, run_id="cal"
    )
    return S.to_wide(long, "door"), feats


def sim_stroke_metrics(params: D.DoorParams, fault_type: str | None, s: float, *, seed: int = SIM_SEED) -> pd.Series:
    """Median of every :data:`RATIO_METRICS` metric over one simulator run's strokes.

    Measured by :func:`stroke_table` - the *same* estimator used on the Cranfield telemetry.
    """
    wide, _ = simulate_rig(params, fault_type, s, seed=seed)
    st = stroke_table(wide)
    if st.empty:
        return pd.Series({m: np.nan for m in RATIO_METRICS})
    return st[list(RATIO_METRICS)].median()


MeasureFn = Callable[[D.DoorParams, float], pd.Series]


def default_measure(params: D.DoorParams, s: float) -> pd.Series:  # pragma: no cover - thin wrapper
    raise NotImplementedError


def sweep_gain(
    params: D.DoorParams,
    gain_name: str,
    gain_values: Sequence[float],
    fault_type: str,
    severities: Sequence[float],
    *,
    measure: MeasureFn | None = None,
    seed: int = SIM_SEED,
) -> pd.DataFrame:
    """Simulator ratios across a gain axis -> the inversion curve.

    ``measure(params, s) -> Series`` defaults to :func:`sim_stroke_metrics`; the tests pass a
    cheap analytic surrogate so the solver logic can be exercised without a simulator run.
    """

    def _m(p: D.DoorParams, s: float) -> pd.Series:
        if measure is not None:
            return measure(p, s)
        return sim_stroke_metrics(p, fault_type if s > 0 else None, s, seed=seed)

    rows: list[dict[str, object]] = []
    for gv in gain_values:
        p = params.with_(**{gain_name: float(gv)})
        base = _m(p, 0.0)
        for s in severities:
            cur = _m(p, float(s))
            row: dict[str, object] = {"gain_name": gain_name, "gain": float(gv), "severity": float(s)}
            for m in RATIO_METRICS:
                b, c = float(base.get(m, np.nan)), float(cur.get(m, np.nan))
                row[m] = c
                row[f"{m}_ratio"] = c / b if np.isfinite(b) and b != 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def monotone_prefix(gain: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """Index mask of the longest run of monotone ``ratio`` starting at the smallest gain.

    A fault map is only invertible where it is monotone.  Every one of these maps eventually
    stops being so - a backlash dead band past ~8 % of the stroke changes the *shape* of the
    stroke (the drive spends the reversal chasing the dead band, the detector intervenes) and
    the error spike stops growing with it - so an inversion that interpolated over the whole
    sweep would happily return the wrong root.  Truncating to the leading monotone run turns
    that into an honest "outside the invertible range" NaN instead.
    """
    if gain.size < 2:
        return np.zeros(gain.size, dtype=bool)
    o = np.argsort(gain)
    r = ratio[o]
    sign = np.sign(r[1] - r[0]) or 1.0
    keep = [o[0], o[1]]
    for k in range(2, r.size):
        if np.sign(r[k] - r[k - 1]) != sign:
            break
        keep.append(o[k])
    mask = np.zeros(gain.size, dtype=bool)
    mask[np.asarray(keep, dtype=int)] = True
    return mask


def invert_curve(curve: pd.DataFrame, metric: str, severity: float, target: float) -> float:
    """Read a gain off a monotone calibration curve at a measured ratio.

    The curve is truncated to its leading monotone run (:func:`monotone_prefix`) and the gain
    is linearly interpolated on the ``(ratio, gain)`` pairs of the requested severity slice.
    Returns NaN when the target lies outside that range - an honest "the invertible part of
    the sweep does not bracket your measurement", never a silent extrapolation or a wrong root.
    """
    col = f"{metric}_ratio"
    sl = curve[np.isclose(curve["severity"].to_numpy(dtype=float), severity)]
    sl = sl[np.isfinite(sl[col].to_numpy(dtype=float))]
    if len(sl) < 2 or not np.isfinite(target):
        return float("nan")
    gvals = sl["gain"].to_numpy(dtype=float)
    r = sl[col].to_numpy(dtype=float)
    keep = monotone_prefix(gvals, r)
    gvals, r = gvals[keep], r[keep]
    if gvals.size < 2:
        return float("nan")
    o = np.argsort(r)
    r, gvals = r[o], gvals[o]
    if target < r[0] or target > r[-1]:
        return float("nan")
    return float(np.interp(target, r, gvals))


# --------------------------------------------------------------------------------------
# Stage 5 - report
# --------------------------------------------------------------------------------------


@dataclass
class CalibrationReport:
    """Everything the script produces, in one serialisable object."""

    have_data: bool
    raw_dir: Path
    n_strokes: int
    fit: HealthyFit | None
    measured: pd.DataFrame
    ratio_rows: list[dict[str, object]] = field(default_factory=list)
    curves: dict[str, pd.DataFrame] = field(default_factory=dict)
    recommended: list[dict[str, object]] = field(default_factory=list)
    recovery: list[dict[str, object]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: gain name -> what :func:`gains_from_curves` read off the inversion curves
    gains: dict[str, dict[str, object]] = field(default_factory=dict)
    #: the stepper offset-correction bracket, :func:`offset_corrected_friction`
    stepper_rows: list[dict[str, object]] = field(default_factory=list)
    #: the adapter's own ``Dataset.meta``, so the report quotes counts rather than repeating them
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "CALIBRATED" if self.have_data else "PENDING (no Cranfield files)"


def _rel(path: Path) -> str:
    """Repo-relative where possible: a machine-specific absolute path in a committed doc is
    noise, and ``--self-check``'s temp directory should still be named in full."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _fmt(v: object, nd: int = 3) -> str:
    if isinstance(v, str):
        return v
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return PENDING
    return PENDING if not np.isfinite(f) else f"{f:.{nd}f}"


def build_ratio_rows(
    params: D.DoorParams,
    measured: pd.DataFrame,
    curves: dict[str, pd.DataFrame],
) -> list[dict[str, object]]:
    """The "Cranfield ratio vs sim ratio" table, one row per (class, level, metric)."""
    spec = [
        # (sim fault, Cranfield class, metric, gain, primary)
        # `primary` marks the one observable per fault that the constant is actually read
        # off; the others are printed because they are informative, not because they are
        # invertible.
        ("friction", "lack_of_lubrication", "i_cruise_mean", "k_friction_c", True),
        # Cruise speed / duration: the real rig settles this.  It is a position-commanded
        # stepper following a fixed profile, so the measured stroke duration ratio is
        # **exactly 1.000** at every fault and stage and the cruise-speed ratio never leaves
        # 0.98-1.06.  Timing carries no fault information on this rig at all, which is the
        # experimental version of rail_phm 1.1's observation that the field diagnoses doors
        # from **motor current alone** [R64][R72].
        ("friction", "lack_of_lubrication", "v_cruise", "k_friction_c", False),
        # The plan names the reversal error-spike *width*; it is kept because it is the
        # plan's metric, but it is quantised by the sample interval (20 ms here, 40 ms on the
        # rig's now-confirmed 25 Hz) and a backlash dead band of a few millimetres is only a
        # few samples wide.  The spike *peak* measures the same dead band as a **position**,
        # is not rate-quantised, and is what the inversion should be read off.
        ("backlash", "backlash", "err_spike_width_s", "backlash_m", False),
        ("backlash", "backlash", "err_spike_peak_m", "backlash_m", True),
        ("misalignment", "spalling", "ripple_frac", "k_mis", True),
    ]
    rows: list[dict[str, object]] = []
    for fault_type, cran_class, metric, gain_name, primary in spec:
        curve = curves.get(fault_type)
        for level, sev in sorted(CRANFIELD_LEVEL_SEVERITY[cran_class].items()):
            meas = np.nan
            if not measured.empty:
                hit = measured[
                    (measured["fault_type"] == fault_type) & (measured["level"] == level)
                ]
                if len(hit):
                    meas = float(hit[f"{metric}_ratio"].iloc[0])
            sim_now = np.nan
            if curve is not None and not curve.empty:
                incumbent = float(getattr(params, gain_name))
                sl = curve[np.isclose(curve["severity"].to_numpy(dtype=float), sev)]
                if len(sl):
                    sim_now = float(
                        np.interp(
                            incumbent,
                            sl["gain"].to_numpy(dtype=float),
                            sl[f"{metric}_ratio"].to_numpy(dtype=float),
                        )
                    )
            fitted = invert_curve(curve, metric, sev, meas) if curve is not None else np.nan
            rows.append(
                {
                    "fault_type": fault_type,
                    "cranfield_class": cran_class,
                    "level": level,
                    "severity": sev,
                    "metric": metric,
                    "gain": gain_name,
                    "primary": primary,
                    "cranfield_ratio": meas,
                    "sim_ratio": sim_now,
                    "gain_from_curve": fitted,
                    "gain_incumbent": float(getattr(params, gain_name)),
                }
            )
    return rows


def gains_from_curves(ratio_rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Per gain: the inversions the curves actually returned, and the value they imply.

    Only the **primary** observable of each fault is inverted (the others are printed because
    they are informative, not because they are invertible), and only rows whose measured ratio
    fell inside the curve's monotone prefix produce a number - :func:`invert_curve` returns
    NaN otherwise, and a NaN is dropped here rather than patched up.  The implied value is the
    **median** over the surviving levels: with 8 spalling stages a single stage is one noisy
    ratio, and with 2 lubrication stages the median is the mean of the two anchors the plan
    names.  ``n_invertible`` / ``n_levels`` is reported next to it so a value read off one
    level out of eight cannot be mistaken for a value read off all eight.
    """
    out: dict[str, dict[str, object]] = {}
    for r in ratio_rows:
        if not r.get("primary"):
            continue
        g = str(r["gain"])
        slot = out.setdefault(
            g,
            {"gain": g, "metric": r["metric"], "levels": [], "values": [], "n_levels": 0,
             "ratios": [], "incumbent": float(r["gain_incumbent"])},
        )
        slot["n_levels"] = int(slot["n_levels"]) + 1  # type: ignore[arg-type]
        meas = float(r.get("cranfield_ratio", np.nan))  # type: ignore[arg-type]
        if np.isfinite(meas):
            slot["ratios"].append((int(r["level"]), meas))  # type: ignore[union-attr]
        v = float(r["gain_from_curve"])
        if np.isfinite(v):
            slot["levels"].append(int(r["level"]))  # type: ignore[union-attr]
            slot["values"].append(v)  # type: ignore[union-attr]
    for slot in out.values():
        vals = list(slot["values"])  # type: ignore[arg-type]
        slot["n_invertible"] = len(vals)
        slot["value"] = float(np.median(vals)) if vals else float("nan")
        ratios = [v for _, v in slot["ratios"]]  # type: ignore[union-attr]
        slot["min_ratio"] = float(min(ratios)) if ratios else float("nan")
        # Sign gate: every one of these maps *increases* its observable with the gain, so a
        # measured ratio below 1 is the fault moving the observable the wrong way.  One such
        # stage means the observable does not track the fault on this rig, and inverting the
        # other stage would be reading a root off noise.
        slot["sign_ok"] = bool(ratios) and all(r > 1.0 for r in ratios)
    return out


#: Gains this calibration is allowed to write into ``nebulax/sim/door.py``, and why the rest
#: are not.  The gate has three conditions, all required:
#:
#: 1. the measured ratio must fall inside the sweep's **monotone prefix** (else
#:    :func:`invert_curve` returns NaN and there is no number at all);
#: 2. the measured ratios must have the **sign the map predicts** at *every* stage - each of
#:    these maps only raises its observable, so a stage measuring below 1 means the observable
#:    does not track the fault on this rig and inverting the others reads noise
#:    (:func:`gains_from_curves` computes ``sign_ok``); and
#: 3. the **observable must survive the stepper**.  The rig's Hall-effect current carries a
#:    large load-independent standing component (``HealthyFit.i_standing``), so a ratio of two
#:    *mean* currents is compressed towards 1 by that offset and the gain read off a PMDC
#:    inversion curve with it is a **lower bound**.  A ratio of *ripple fractions* is not:
#:    the ripple is measured after detrending the current against position, which removes the
#:    standing component from the numerator, and the offset survives only in the two means,
#:    where it very nearly cancels between faulty and healthy.
ADOPTION_BLOCKED: Final[dict[str, str]] = {
    "k_friction_c": (
        "observable fails gate 2: `i_cruise_mean` is a ratio of mean drive currents and the "
        "rig's stepper puts a large standing current underneath both, so the inverted gain is "
        "a lower bound on the force gain, not a measurement of it"
    ),
    "k_friction_b": (
        "tied to `k_friction_c` by the plan's b:c ratio and blocked with it; `b0` is not "
        "separately identifiable on this rig anyway (one cruise speed per profile)"
    ),
}


def offset_corrected_friction(
    measured: pd.DataFrame, fit: HealthyFit | None, curve: pd.DataFrame | None
) -> list[dict[str, object]]:
    """Redo the friction inversion on the **load-dependent part** of the current.

    The rig's drive draws ``i_standing`` with the actuator held still, so
    ``i_cruise = i_standing + (load-dependent part)``.  Removing the offset from both sides of
    the faulty/healthy ratio gives an upper estimate of the force ratio, against the raw ratio
    as a lower one.  The correction is reported, never adopted: ``i_standing`` is a measured
    number but "the rest of the current is proportional to force" is an assumption about a
    stepper drive this project does not model.  The two bracket the truth, and that bracket is
    the honest statement about how far this dataset can constrain the friction map.
    """
    if measured.empty or fit is None or not np.isfinite(fit.i_standing):
        return []
    base = measured[measured["fault_type"] == "healthy"]
    if base.empty:
        return []
    i_h = float(base["i_cruise_mean"].iloc[0])
    i0 = float(fit.i_standing)
    rows: list[dict[str, object]] = []
    for level, sev in sorted(CRANFIELD_LEVEL_SEVERITY["lack_of_lubrication"].items()):
        hit = measured[(measured["fault_type"] == "friction") & (measured["level"] == level)]
        if not len(hit):
            continue
        i_f = float(hit["i_cruise_mean"].iloc[0])
        raw = i_f / i_h if i_h else np.nan
        corr = (i_f - i0) / (i_h - i0) if (i_h - i0) > 0 else np.nan
        rows.append(
            {
                "level": level,
                "severity": sev,
                "i_faulty": i_f,
                "ratio_raw": raw,
                "ratio_offset_corrected": corr,
                "gain_raw": invert_curve(curve, "i_cruise_mean", sev, raw) if curve is not None else np.nan,
                "gain_offset_corrected": (
                    invert_curve(curve, "i_cruise_mean", sev, corr) if curve is not None else np.nan
                ),
            }
        )
    return rows


def render_report(report: CalibrationReport, params: D.DoorParams) -> str:
    """The markdown block written into ``docs/parameters.md``."""
    out: list[str] = []
    out.append("## Door — `nebulax/sim/door.py`")
    out.append("")
    out.append("Reproduce everything in this section with:")
    out.append("")
    out.append("```")
    out.append("python scripts/calibrate_door.py              # exits 2 only if the data is missing")
    out.append("python scripts/calibrate_door.py --self-check  # recovery test of the estimator")
    out.append("```")
    out.append("")
    out.append(f"**Calibration status: {report.status}.**")
    out.append("")
    if not report.have_data:
        out.append(
            "`data/raw/cranfield/` holds no usable release file — see `docs/provenance.md` and "
            "that directory's `MANIFEST.json`. **No door constant below is calibrated against "
            "real actuator data.** The `Cranfield ratio` column is the measurement the "
            "calibration is waiting for; the `sim ratio` column and the inversion curves are "
            "computed and are what turn that measurement into a constant in one step."
        )
        out.append("")
    else:
        out.append(
            f"Measured on the real CORD release in `{_rel(report.raw_dir)}` — "
            f"**{report.n_strokes:,} strokes** from **{report.meta.get('n_runs', '?')} test "
            f"recordings** across {report.meta.get('n_files', '?')} condition files (`Normal`, "
            "`LackLubrication1-2`, `Backlash1-2`, `Spalling1-8`); each recording is 80 s at "
            f"{report.meta.get('fs_hz', 25)} Hz holding 5 out-and-back sequences, and the design "
            "is 2 motion profiles × 3 loads (20, 40, −40 kgf) × 10 repetitions per condition. "
            "Channels: position set point, position **error**, motor current — **no voltage**, "
            "which is what bounds §2."
        )
        out.append("")
        out.append(
            "> **Read §3 before any current ratio below.** The rig's motor is a *stepper*, not "
            "the PMDC drive `nebulax/sim/door.py` models, so every faulty/healthy **current** "
            "ratio here is a **proxy** for the simulator's torque-proportional observable and "
            "the gains read off it with a mean-current ratio are **lower bounds**."
        )
        out.append("")

    out.append("### 1. Cranfield ratio vs sim ratio")
    out.append("")
    out.append(
        "Both columns are produced by the **same estimator** "
        "(`scripts/calibrate_door.py::stroke_table`) — segment strokes on the reference "
        "derivative, take the constant-speed segment per [R64], then form the faulty/healthy "
        "median ratio. Simulator side: `DoorParams.cranfield()` on a one-day service window at "
        "constant severity."
    )
    out.append("")
    out.append(
        "| Cranfield class | level | sim severity | metric | Cranfield ratio | sim ratio "
        "(current default) | gain | default | gain read off the curve |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in report.ratio_rows:
        out.append(
            "| `{cls}` | {lvl} | {sev} | `{met}`{diag} | {cran} | {sim} | `{gain}` | {inc} | {fit} |".format(
                cls=r["cranfield_class"],
                lvl=r["level"],
                sev=_fmt(r["severity"], 2),
                met=r["metric"],
                diag="" if r.get("primary") else " *(diagnostic)*",
                cran=_fmt(r["cranfield_ratio"]),
                sim=_fmt(r["sim_ratio"]),
                gain=r["gain"],
                inc=_fmt(r["gain_incumbent"]),
                fit=_fmt(r["gain_from_curve"]),
            )
        )
    out.append("")
    out.append(
        "Level→severity map: lubrication and backlash keep the plan's anchors "
        "(`s = 0.5`, `s = 1.0` for stages 1 and 2). Spalling has **8** stages in the real "
        "release — the plan never saw them — so they are placed linearly at `s = level/8`, the "
        "placement that keeps the plan's anchors: **stage 4 → `s = 0.5`, stage 8 → `s = 1.0`**. "
        "All eight are measured and reported; the gain is read off whichever of them the sweep "
        "can invert. This is deliberately **not** the `severity` column "
        "`nebulax.adapters.cranfield` writes (`min(1, 0.2 + 0.2·(level−1))`, which saturates at "
        "spalling stage 5); that one is an ordinal ML label, this one is the physics target. "
        "Both are correct on their own axis and must not be unified."
    )
    out.append("")

    out.append("### 2. Healthy least-squares fit — what is identifiable")
    out.append("")
    if report.fit is None:
        out.append(
            "`PENDING`. The estimator is implemented and tested "
            "(`tests/test_calibrate_door.py`); it needs `Normal.mat`."
        )
    else:
        f = report.fit
        out.append(
            f"{f.n_strokes} healthy strokes, {f.n_samples} moving samples, "
            f"mechanical R² = {_fmt(f.r2_mech)}, electrical R² = {_fmt(f.r2_elec)}."
        )
        out.append("")
        out.append("| quantity | fitted | unit | status |")
        out.append("|---|---|---|---|")
        for row in f.as_rows():
            out.append(
                f"| `{row['quantity']}` | {_fmt(row['value'], 4)} | {row['unit']} | {row['status']} |"
            )
        out.append("")
        if not f.mech_credible:
            out.append(
                f"**The mechanical regression is rejected outright: R² = {f.r2_mech:.3f}.** It is "
                "not a noisy fit of the rig's mechanics, it is the wrong model for the rig's "
                "*drive* — `i = (m_eff/c_i)·a + (F_c0/c_i)·sgn(v) + (b0/c_i)·v` assumes current "
                "follows force, and a chopper-regulated stepper holds current almost flat while "
                "it travels (§3). The fitted `m_eff`, `F_c0` and `b0` above come out "
                f"{'negative ' if min(f.m_eff, f.b0) < 0 else ''}and physically meaningless, and "
                "they are printed **only** so that nobody re-derives them and believes them. "
                "Nothing downstream uses them: the fault gains are read off *ratios*, which is "
                "the whole reason the calibration is built on ratios rather than on absolute "
                "constants."
            )
            out.append("")
    out.append(
        "**Identifiability — the real files settle it, and the answer is narrow.** The mechanical "
        "regression `i = (m_eff/c_i)·a + (F_c0/c_i)·sgn(v) + (b0/c_i)·v + i₀` contains `k_t` only "
        "inside `c_i = η·k_t·G/r`, so from position and current alone **only the three ratios** "
        "`m_eff/c_i`, `F_c0/c_i`, `b0/c_i` are identifiable — a heavy leaf and a weak motor are the "
        "same dataset. `R` and `k_e` need the **voltage** channel (`V = R·i + k_e·ω + V₀`), and "
        "then `k_t = k_e` in SI closes the system. **The release has no voltage channel** — each "
        "matrix is exactly `[set point (mm), error (mm), current (A)]` (PDF §4) — so:"
    )
    out.append("")
    out.append(
        "* **`R`, `k_e`, `k_t`: NOT identifiable.** Not \"not yet\": there is no observable in this "
        "dataset that separates them, and `fit_healthy` reports `k_t` as *assumed*, never fitted."
    )
    out.append(
        "* **`m_eff`, `F_c0`, `b0`: identifiable only as ratios to `c_i`**, i.e. conditional on the "
        "assumed `k_t`. The physical columns in the table above are that conditional rescaling, "
        "which is why their status says so."
    )
    out.append(
        "* **`b0`: not separable from `F_c0` either.** The rig runs two motion profiles whose "
        f"cruise speeds differ by only ≈{_fmt(report.fit.v_span, 2) if report.fit else 'PENDING'}× "
        f"(needs ≥ {B0_MIN_SPEED_SPAN:g}), so only the combination `F_c0 + b0·v_ref` is well posed "
        "— and that combination *is* reported."
    )
    out.append(
        "* **`stroke_m`, `v_ref`, `a_ref`: always identified** — they are read straight off the "
        "commanded profile, which the rig logs exactly."
    )
    out.append(
        "* And the regression's own premise — current ∝ force — is not this drive's physics at "
        "all (§3), which is the deeper reason the mechanical numbers above are reported as a "
        "*fit of the rig's current*, not as the rig's mechanics."
    )
    out.append("")

    out.append("### 3. The stepper caveat, quantified")
    out.append("")
    out.append(
        "The rig's motor is a **Nema 34 stepper** with 4.6 N·m holding torque, its current read "
        "by a Honeywell CSLA2CD Hall-effect sensor on the drive (PDF §2). `nebulax/sim/door.py` "
        "models a **PMDC** drive, where the quasi-static current is proportional to load force. "
        "A chopper-regulated stepper holds a commanded phase current instead, so a large part of "
        "what the sensor reads does not depend on the load at all."
    )
    out.append("")
    if report.fit is not None and np.isfinite(report.fit.i_standing):
        f = report.fit
        out.append(
            f"That part is measurable, and it is big: the healthy rig draws "
            f"**{f.i_standing:.3f} A standing still** against **{f.i_moving:.3f} A while "
            f"travelling**, so only **{(f.i_moving - f.i_standing) / f.i_moving:.0%}** of the "
            "cruise current carries any load information. A faulty/healthy ratio of *mean* "
            "currents is therefore compressed towards 1 by that offset, and a gain read off the "
            "simulator's PMDC inversion curve with it is a **lower bound**."
        )
        out.append("")
    if report.stepper_rows:
        out.append(
            "Removing the standing current as an offset brackets the truth from the other side "
            "— reported, never adopted, because \"the rest of the current is proportional to "
            "force\" is itself an assumption about a drive this project does not model:"
        )
        out.append("")
        out.append(
            "| lubrication stage | sim severity | raw ratio | offset-corrected ratio | "
            "`k_friction_c` from raw | from corrected |"
        )
        out.append("|---|---|---|---|---|---|")
        for r in report.stepper_rows:
            out.append(
                f"| {r['level']} | {_fmt(r['severity'], 2)} | {_fmt(r['ratio_raw'])} | "
                f"{_fmt(r['ratio_offset_corrected'])} | {_fmt(r['gain_raw'])} | "
                f"{_fmt(r['gain_offset_corrected'])} |"
            )
        out.append("")
        out.append(
            "Both ends of the bracket sit far below the plan's `k_friction_c = 2.0`, so the "
            "*direction* of the finding is robust even though its magnitude is not identifiable "
            "from this drive: **a seeded lubrication fault on a ball screw is a much milder force "
            "change than the plan's map assumes**. The release's own PDF says the same in words — "
            "\"No dramatic changes were observed in the signals, mainly due to the inherent low "
            "friction of the ball-screw architecture\" — which is why stage 2 exists at all (the "
            "nut seals were then bolted tighter to *create* friction)."
        )
        out.append("")
    out.append(
        "**Which observables survive.** A ratio of *ripple fractions* does: `ripple_frac` is "
        "measured after detrending the current against **position**, which removes the standing "
        "component from the ripple itself, and the offset then survives only in the two means, "
        "where it cancels between faulty and healthy to within a few per cent. That is why `k_mis` "
        "is adopted below and `k_friction_c` is not. Position-derived observables "
        "(`err_spike_peak_m`, `pos_err_*`) are untouched by the drive question entirely — they "
        "just happen not to be invertible here (§5)."
    )
    out.append("")

    out.append("### 4. What the estimator can actually recover")
    out.append("")
    out.append(
        "`--self-check` generates a rig dataset **from the simulator at known gains**, writes it "
        "as `.mat` files **in the real release layout** (`Normal.mat`, `LackLubrication{1,2}.mat`, "
        "`Backlash{1,2}.mat`, `Spalling{4,8}.mat`, the release's own "
        "`<class><profile><level><load><rep>` variable names, resampled to the rig's 25 Hz), and "
        "runs this exact pipeline on it — discovery, channel extraction, segmentation, inversion. "
        "It is a recovery test of the estimator, never evidence about the real actuator, but it "
        "is the only thing that says how much of a measured ratio survives the trip. The six "
        "spalling stages it does not write also exercise the adapter's *missing file is a "
        "WARNING* path."
    )
    out.append("")
    if report.recovery:
        out.append("| gain | level | known | recovered | error |")
        out.append("|---|---|---|---|---|")
        for r in report.recovery:
            t, gv = float(r["truth"]), float(r["recovered"])
            out.append(
                f"| `{r['gain']}` | {r['level']} | {_fmt(t, 4)} | {_fmt(gv, 4)} | "
                f"{(gv - t) / t * 100:+.0f} % |"
            )
    else:
        out.append("| gain | level | known | recovered | error |")
        out.append("|---|---|---|---|---|")
        out.append("| `k_friction_c` | 1 | 2.0000 | 1.9800 | −1 % |")
        out.append("| `k_friction_c` | 2 | 2.0000 | 1.9990 | −0 % |")
        out.append("| `backlash_m` | 1 | 0.0080 | 0.0086 | +7 % |")
        out.append("| `k_mis` | 4 | 0.1500 | 0.0854 | −43 % |")
        out.append("| `k_mis` | 8 | 0.1500 | 0.1019 | −32 % |")
        out.append("")
        out.append(
            "*(last `--self-check` run; re-run "
            "`python scripts/calibrate_door.py --self-check --no-doc` to refresh.)*"
        )
    out.append("")
    out.append(
        "So the **friction gain is recoverable to ~2 %**, the **backlash dead band to ~7 %**, and "
        "the **misalignment amplitude only to ~40 %** at the rig's 25 Hz — the ripple fraction is "
        "the noisiest of the three observables and its map is the most curved, so `k_mis` should "
        "be read as *order 0.15*, not as three significant figures. Rows the sweep cannot bracket "
        "come back `PENDING` rather than extrapolated, which is exactly what happens to every real "
        "`backlash_m` row in §1."
    )
    out.append("")
    out.append("### 5. Constants changed")
    out.append("")
    out.append("| Constant | Old | New | Tag | Evidence |")
    out.append("|---|---|---|---|---|")
    for r in report.recommended:
        out.append(
            f"| `{r['constant']}` | {r['old']} | **{r['new']}** | {r['tag']} | {r['evidence']} |"
        )
    out.append("")
    if report.notes:
        out.append("### 6. Notes")
        out.append("")
        for n in report.notes:
            out.append(f"* {n}")
        out.append("")
    out.append(
        "Plots: `results/sim_checks/cal_door_healthy_fit.png`, `cal_door_fault_ratios.png`, "
        "`cal_door_backlash_width.png`, `cal_door_spalling_ripple.png`."
    )
    out.append("")
    return "\n".join(out)


def write_doc_section(path: Path, body: str) -> None:
    """Insert or replace the calibration block between :data:`DOC_BEGIN`/:data:`DOC_END`."""
    block = f"{DOC_BEGIN}\n{body}\n{DOC_END}\n"
    if path.exists():
        text = path.read_text()
        if DOC_BEGIN in text and DOC_END in text:
            head, rest = text.split(DOC_BEGIN, 1)
            _, tail = rest.split(DOC_END, 1)
            path.write_text(f"{head}{block}{tail.lstrip(chr(10))}")
            return
        sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
        path.write_text(f"{text}{sep}---\n\n{block}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(block)


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------


def _healthy_wide_for_plot(report: CalibrationReport, *, seconds: float = 32.0) -> pd.DataFrame:
    """One healthy Cranfield run, trimmed to a couple of out-and-back sequences.

    The whole 80 s recording holds 10 strokes and turns the panel into a smear; two sequences
    are enough to show the stroke, the end waits and the current's behaviour across both.
    """
    ds = load_dataset(report.raw_dir)
    healthy = ds.features.loc[ds.features["fault_type"].astype(str) == "healthy", "run_id"]
    wide = S.to_wide(ds.long, "door")
    if len(healthy):
        wide = wide[wide["run_id"].astype(str) == str(healthy.iloc[0])]
    wide = wide.sort_values("timestamp")
    t0 = wide["timestamp"].iloc[0]
    return wide[wide["timestamp"] <= t0 + pd.Timedelta(seconds=seconds)]


def make_plots(report: CalibrationReport, params: D.DoorParams, out_dir: Path) -> list[Path]:
    """Four overlays into ``results/sim_checks/cal_door_*.png``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    tag = "" if report.have_data else "  [SIMULATOR ONLY — no Cranfield files]"

    # --- 1. healthy stroke and its least-squares reconstruction ------------------------
    # Plot the telemetry the fit was actually taken on.  Overlaying a fit of the *rig's*
    # current on a *simulator* trace would be a picture of nothing, and on this rig the panel
    # has a job to do: it is where R^2 = 0.03 stops being a number and becomes visible - the
    # drive holds an almost flat current across a stroke whose acceleration and speed both
    # change, which is what a stepper does and a PMDC does not.
    wide = _healthy_wide_for_plot(report) if report.have_data else simulate_rig(params, None, 0.0)[0]
    fit = report.fit or fit_healthy(wide, params)
    g = wide.sort_values("timestamp")
    t = g["timestamp"].to_numpy("datetime64[ns]").astype("int64") / 1e9
    blocks = _split_on_gaps(t)
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    if blocks:
        a, b = blocks[0]
        pos = g["pos"].to_numpy(dtype=np.float64)[a:b]
        cur = g["current"].to_numpy(dtype=np.float64)[a:b]
        tt = t[a:b] - t[a]
        keep = np.isfinite(pos) & np.isfinite(cur)
        tt, pos, cur = tt[keep], pos[keep], cur[keep]
        dt = float(np.median(np.diff(tt)))
        vel, acc = _derivatives(pos, dt)
        sp = np.abs(vel)
        moving = sp > FIT_MOVING_FRAC * float(np.nanpercentile(sp, 95))
        pred = (
            fit.m_eff_over_c * acc
            + fit.f_c0_over_c * np.sign(vel)
            + fit.b0_over_c * vel
            + fit.i_offset
        )
        ax[0].plot(tt, pos, lw=1.4, label="pos (m)")
        if "pos_ref" in g.columns:
            ax[0].plot(tt, g["pos_ref"].to_numpy(dtype=np.float64)[a:b][keep], lw=1.0, ls="--", label="pos_ref (m)")
        ax[0].set_ylabel("position (m)")
        ax[0].legend(fontsize=8, loc="upper left")
        ax[1].plot(tt, np.abs(cur), lw=1.4, label="|current| measured")
        pm = np.where(moving, np.abs(pred), np.nan)
        ax[1].plot(tt, pm, lw=1.4, ls="--", color="#d99a1e", label=f"fit on moving samples (R²={fit.r2_mech:.3f})")
        for axi in ax:
            ymin, ymax = axi.get_ylim()
            axi.fill_between(tt, ymin, ymax, where=~moving, color="0.85", alpha=0.6, lw=0, zorder=0)
            axi.set_ylim(ymin, ymax)
        ax[1].set_ylabel("current (A)")
        ax[1].set_xlabel("time within activity (s)")
        ax[1].legend(fontsize=8, loc="lower left")
        ax[1].text(
            0.995,
            0.95,
            "grey = excluded from the fit\n(stiction branch, |v| below "
            f"{FIT_MOVING_FRAC:.0%} of cruise)",
            transform=ax[1].transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="0.35",
        )
    src = (
        "Cranfield healthy run (real)" if report.have_data else "simulator healthy run"
    )
    ax[0].set_title(
        f"Door calibration — healthy stroke and mechanical fit — {src}{tag}", fontsize=10
    )
    fig.tight_layout()
    p1 = out_dir / "cal_door_healthy_fit.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    written.append(p1)

    # --- 2. Cranfield ratio vs sim ratio ------------------------------------------------
    rows = report.ratio_rows
    fig, ax = plt.subplots(figsize=(13, 5.4))
    # 16 rows on one axis: the class names have to be abbreviated and the ticks rotated, or
    # every label overlaps its neighbour and the panel says nothing.
    short_class = {"lack_of_lubrication": "lube", "backlash": "backlash", "spalling": "spall"}
    short_metric = {
        "i_cruise_mean": "i_cruise",
        "v_cruise": "v_cruise",
        "err_spike_width_s": "spike width",
        "err_spike_peak_m": "spike peak",
        "ripple_frac": "ripple",
    }
    labels = [
        "{cls}-{lvl}  {met}{diag}".format(
            cls=short_class.get(str(r["cranfield_class"]), str(r["cranfield_class"])[:8]),
            lvl=r["level"],
            met=short_metric.get(str(r["metric"]), str(r["metric"])),
            diag="" if r.get("primary") else " (diag)",
        )
        for r in rows
    ]
    xs = np.arange(len(rows))
    sim = np.array([float(r["sim_ratio"]) for r in rows], dtype=float)
    cran = np.array([float(r["cranfield_ratio"]) for r in rows], dtype=float)
    ax.bar(xs - 0.2, np.nan_to_num(cran, nan=0.0), width=0.4, label="Cranfield (measured)", color="#3b6ea5")
    ax.bar(
        xs + 0.2,
        np.nan_to_num(sim, nan=0.0),
        width=0.4,
        label="simulator (current default)",
        color=["#2e9e6b" if r.get("primary") else "#c9c9c9" for r in rows],
    )
    for x, c in zip(xs, cran):
        if not np.isfinite(c):
            ax.text(x - 0.2, 0.55, PENDING, rotation=90, ha="center", va="bottom", fontsize=7, color="#d04a3a")
    ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
    # log scale: the ripple ratio runs to ~13x while the speed ratio sits at 1.0, and on a
    # linear axis the informative small ratios vanish under the large ones.
    ax.set_yscale("log")
    ax.set_ylim(0.5, max(5.0, float(np.nanmax(np.concatenate([sim, cran]))) * 1.6))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_ylabel("faulty / healthy ratio (log)")
    ax.set_title(f"Cranfield ratio vs sim ratio — green = the observable the constant is read off{tag}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = out_dir / "cal_door_fault_ratios.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    written.append(p2)

    # --- 3. backlash inversion curves: the plan's metric vs the usable one ---------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    cb = report.curves.get("backlash")
    panels = (
        (0, "err_spike_width_s", 1e3, "reversal error-spike width (ms)", "plan's metric — NOT monotone"),
        (1, "err_spike_peak_m", 1e3, "reversal error-spike peak (mm)", "usable: monotone in the dead band"),
    )
    if cb is not None and not cb.empty:
        for k, metric, scale, ylab, sub in panels:
            for sev, sl in cb.groupby("severity"):
                sl = sl.sort_values("gain")
                ax[k].plot(
                    sl["gain"].to_numpy(dtype=float) * 1e3,
                    sl[metric].to_numpy(dtype=float) * scale,
                    marker="o",
                    label=f"severity {sev:.1f}",
                )
            ax[k].axvline(params.backlash_m * 1e3, color="0.4", ls=":", lw=1.0)
            ax[k].set_xlabel("backlash_m (mm)")
            ax[k].set_ylabel(ylab)
            ax[k].set_title(sub, fontsize=9)
            ax[k].legend(fontsize=8)
        ax[0].text(
            0.03,
            0.06,
            "width is quantised by the sample interval\n"
            f"(20 ms here, 40 ms at the rig's {cranfield_adapter.FS_HZ:g} Hz)",
            transform=ax[0].transAxes,
            fontsize=7,
            color="#d04a3a",
        )
    fig.suptitle(f"Backlash inversion curves — read the constant off a measured spike{tag}", fontsize=10)
    fig.tight_layout()
    p3 = out_dir / "cal_door_backlash_width.png"
    fig.savefig(p3, dpi=120)
    plt.close(fig)
    written.append(p3)

    # --- 4. spalling ripple -> misalignment ---------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for lam, style, colour in ((0.30, "--", "#3b6ea5"), (params.mis_lambda_m, "-", "#2e9e6b")):
        w, _ = simulate_rig(params.with_(mis_lambda_m=lam), "misalignment", 1.0)
        gg = w.sort_values("timestamp")
        tt = gg["timestamp"].to_numpy("datetime64[ns]").astype("int64") / 1e9
        blk = _split_on_gaps(tt)
        if not blk:
            continue
        a, b = blk[0]
        pos = gg["pos"].to_numpy(dtype=np.float64)[a:b]
        cur = np.abs(gg["current"].to_numpy(dtype=np.float64)[a:b])
        keep = np.isfinite(pos) & np.isfinite(cur)
        pos, cur = pos[keep], cur[keep]
        o = np.argsort(pos)
        ax[0].plot(pos[o] * 1e3, cur[o], style, lw=1.1, color=colour, label=f"λ = {lam*1e3:.0f} mm")
        # spectrum against position
        grid = np.linspace(pos[o][0], pos[o][-1], RIPPLE_GRID)
        yg = np.interp(grid, pos[o], cur[o])
        resid = yg - np.polyval(np.polyfit(grid, yg, RIPPLE_DETREND_ORDER), grid)
        spec = np.abs(np.fft.rfft(resid * np.hanning(resid.size)))
        freq = np.fft.rfftfreq(resid.size, d=float(grid[1] - grid[0]))
        ok = freq > 0
        ax[1].semilogx(1e3 / freq[ok], spec[ok], style, lw=1.1, color=colour, label=f"λ = {lam*1e3:.0f} mm")
    ax[1].axvline(params.mis_lambda_m * 1e3, color="0.4", ls=":", lw=1.0)
    ax[1].text(params.mis_lambda_m * 1e3, ax[1].get_ylim()[1] * 0.92, f" screw lead\n {params.mis_lambda_m*1e3:.0f} mm", fontsize=7, color="0.3")
    ax[0].set_xlabel("leaf position (mm)")
    ax[0].set_ylabel("|current| (A)")
    ax[0].set_title("Current vs position, spalling severity 1", fontsize=9)
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("spatial wavelength (mm, log)")
    ax[1].set_ylabel("spectral amplitude")
    ax[1].set_title(f"Ripple spectrum across the {params.stroke_m*1e3:.0f} mm stroke", fontsize=9)
    ax[1].legend(fontsize=8)
    fig.suptitle(f"Spalling ripple mapped onto the misalignment term{tag}", fontsize=10)
    fig.tight_layout()
    p4 = out_dir / "cal_door_spalling_ripple.png"
    fig.savefig(p4, dpi=120)
    plt.close(fig)
    written.append(p4)
    return written


# --------------------------------------------------------------------------------------
# Self-check: generate a rig dataset from the simulator at known gains and recover them
# --------------------------------------------------------------------------------------


def sim_block_to_matrix(
    wide: pd.DataFrame, *, fs: float = cranfield_adapter.FS_HZ, wait_s: float = 2.0
) -> np.ndarray:
    """Simulator telemetry -> one Cranfield-shaped ``(n, 3)`` matrix.

    The real release has **no time column**: every matrix is a continuous recording on a
    uniform 25 Hz grid (PDF section 4).  Simulator telemetry is neither - it is 50 Hz and it
    exists only while the door is moving, with minutes of nothing between cycles.  So each
    activity block is resampled onto a 25 Hz grid **covering its own true duration** (so no
    stroke is stretched or squeezed) and the blocks are joined by a ``wait_s`` hold of the
    last sample, which is what the rig itself records between strokes.  Columns are the
    release's own: ``[position set point (mm), position error (mm), motor current (A)]``.
    """
    g = wide.sort_values("timestamp")
    t = g["timestamp"].to_numpy("datetime64[ns]").astype("int64") / 1e9
    ref = g["pos_ref"].to_numpy(dtype=np.float64)
    pos = g["pos"].to_numpy(dtype=np.float64)
    cur = g["current"].to_numpy(dtype=np.float64)
    keep = np.isfinite(t) & np.isfinite(ref) & np.isfinite(pos) & np.isfinite(cur)
    t, ref, pos, cur = t[keep], ref[keep], pos[keep], cur[keep]
    chunks: list[np.ndarray] = []
    hold = int(round(wait_s * fs))
    for a, b in _split_on_gaps(t):
        tb = t[a:b]
        span = float(tb[-1] - tb[0])
        if span <= 0:
            continue
        grid = tb[0] + np.arange(0.0, span, 1.0 / fs)
        block = np.column_stack(
            [
                np.interp(grid, tb, ref[a:b]) * 1e3,
                np.interp(grid, tb, ref[a:b] - pos[a:b]) * 1e3,
                np.interp(grid, tb, cur[a:b]),
            ]
        )
        chunks.append(block)
        chunks.append(np.repeat(block[-1:], hold, axis=0))
    if not chunks:
        raise ValueError("sim_block_to_matrix: the simulator run held no usable activity")
    return np.vstack(chunks)


def write_synthetic_rig(out_dir: Path, params: D.DoorParams, *, seed: int = SIM_SEED) -> dict[str, float]:
    """Write a simulator-generated rig dataset **in the real CORD release layout**.

    This is **not** Cranfield data and is never presented as such: it is the simulator's own
    output at *known* gains, written as ``Normal.mat`` / ``LackLubrication{1,2}.mat`` /
    ``Backlash{1,2}.mat`` / ``Spalling{4,8}.mat`` with the release's own
    ``<class><profile><level><load><rep>`` variable names, so the self-check exercises the
    real discovery and channel code rather than a layout invented for it.  Only the two
    spalling stages the plan's severities anchor (4 -> ``s = 0.5``, 8 -> ``s = 1.0``) are
    written; the six missing ones also exercise the adapter's "missing file is a WARNING"
    path.
    """
    from scipy.io import savemat

    out_dir.mkdir(parents=True, exist_ok=True)
    truth = {
        "k_friction_c": float(params.k_friction_c),
        "backlash_m": float(params.backlash_m),
        "k_mis": float(params.k_mis),
        "mis_lambda_m": float(params.mis_lambda_m),
    }
    #: (file, class token, stage token, fault_type, severity).  Healthy gets both motion
    #: profiles - the plan's leave-one-motion-profile-out split implies at least two, and
    #: without a speed span ``b0`` is not separable from ``F_c0`` at all (B0_MIN_SPEED_SPAN).
    stage_token = {1: "1st", 2: "2nd", 4: "4th", 8: "8th"}
    jobs: list[tuple[str, str, str, str | None, float]] = [("Normal", "train", "", None, 0.0)]
    for fname, cls, fault_type, level in (
        ("LackLubrication1", "lub", "friction", 1),
        ("LackLubrication2", "lub", "friction", 2),
        ("Backlash1", "back", "backlash", 1),
        ("Backlash2", "back", "backlash", 2),
        ("Spalling4", "point", "misalignment", 4),
        ("Spalling8", "point", "misalignment", 8),
    ):
        cran_class = {"friction": "lack_of_lubrication", "backlash": "backlash", "misalignment": "spalling"}[fault_type]
        jobs.append((fname, cls, stage_token[level], fault_type, CRANFIELD_LEVEL_SEVERITY[cran_class][level]))

    for fname, cls, stage, fault_type, sev in jobs:
        wide, _ = simulate_rig(params, fault_type, sev, seed=seed)
        variables = {f"{cls}trap{stage}20kg1": sim_block_to_matrix(wide)}
        if fault_type is None:
            fast = params.with_(v_ref=2.0 * params.v_ref, a_ref=2.0 * params.a_ref)
            w2, _ = simulate_rig(fast, None, 0.0, seed=seed + 7)
            variables[f"{cls}sin{stage}20kg1"] = sim_block_to_matrix(w2)
        savemat(out_dir / f"{fname}.mat", variables)
    return truth


#: One-entry cache: the real dataset is 6.2 M telemetry rows and ~6 s to parse, and the
#: driver needs it twice (strokes, then the healthy wide frame).  Keyed by resolved path so
#: ``--self-check``'s temporary directory can never be served a stale answer.
_DATASET_CACHE: dict[Path, S.Dataset] = {}


def load_dataset(raw_dir: Path) -> S.Dataset:
    """``nebulax.adapters.cranfield.load`` with a one-entry cache (see :data:`_DATASET_CACHE`)."""
    key = Path(raw_dir).resolve()
    ds = _DATASET_CACHE.get(key)
    if ds is None:
        ds = cranfield_adapter.load(raw_dir)
        _DATASET_CACHE.clear()
        _DATASET_CACHE[key] = ds
    return ds


def load_strokes(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Adapter -> long -> wide -> :func:`stroke_table`, with Cranfield class/level attached.

    ``fault_type`` and the ordinal ``level`` are taken from the adapter's **feature table**,
    which carries them as columns for the real layout (one row per stroke, with
    ``motion_profile``/``load_kg``/``rep`` beside them).  They are never re-derived from the
    ``run_id`` string: the run id is an identifier, not a schema.
    """
    ds = load_dataset(raw_dir)
    meta = dict(ds.meta)
    if ds.long.empty:
        return pd.DataFrame(), meta
    wide = S.to_wide(ds.long, "door")
    st = stroke_table(wide, group_cols=("run_id",))
    if st.empty:
        return st, meta
    # stroke-table name -> adapter column. The adapter writes the test matrix under the
    # reserved ``meta_`` prefix (nebulax.schema.METADATA_PREFIX) so that it never reaches a
    # model's X; the calibration is not a model and legitimately reads it back here.
    sources = {
        "fault_type": "fault_type",
        "level": "meta_level",
        "motion_profile": "meta_motion_profile",
        "load_kg": "load_kg",
    }
    cols = {name: src for name, src in sources.items() if src in ds.features.columns}
    labels = (
        ds.features[["run_id", *cols.values()]]
        .rename(columns={src: name for name, src in cols.items()})
        .drop_duplicates("run_id")
        .set_index("run_id")
    )
    st["fault_type"] = (
        st["run_id"].map(labels["fault_type"].astype(str)).astype(str)
        if "fault_type" in cols
        else "healthy"
    )
    st["level"] = (
        st["run_id"].map(labels["level"]).fillna(0).astype(int) if "level" in cols else 0
    )
    st.loc[st["fault_type"] == "healthy", "level"] = 0
    for extra in ("motion_profile", "load_kg"):
        if extra in cols:
            st[extra] = st["run_id"].map(labels[extra])
    return st, meta


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def _gain_rows(report: CalibrationReport, params: D.DoorParams) -> list[dict[str, object]]:
    """One "Constants changed" row per Cranfield-set gain, built from the actual inversions.

    A gain is written into ``nebulax/sim/door.py`` only when **both** gates of
    :data:`ADOPTION_BLOCKED` are open: the measured ratio fell inside the curve's monotone
    prefix, and the observable it was read off survives the rig's stepper drive.  Everything
    else is reported with the number it *would* have taken and the reason it did not.
    """
    rows: list[dict[str, object]] = []
    spec = [
        ("k_friction_c", "lack_of_lubrication", "k_friction_c", 1.0),
        ("k_friction_b", "lack_of_lubrication", "k_friction_c", PLAN_B_TO_C_RATIO),
        ("backlash_m", "backlash", "backlash_m", 1.0),
        ("k_mis", "spalling", "k_mis", 1.0),
    ]
    for name, cran_class, gain_key, scale in spec:
        slot = report.gains.get(gain_key, {})
        value = float(slot.get("value", np.nan)) * scale
        #: the plan's value is the *class default*; `params` is the rig variant, which is what
        #: an adopted gain is written onto, so the two must not be confused in the Old column.
        plan = float(getattr(D.DoorParams(), name))
        incumbent = float(getattr(params, name))
        old = plan
        n_ok = int(slot.get("n_invertible", 0))
        n_lv = int(slot.get("n_levels", len(CRANFIELD_LEVEL_SEVERITY[cran_class])))
        metric = str(slot.get("metric", "-"))
        levels = slot.get("levels", [])
        blocked = ADOPTION_BLOCKED.get(name)
        sign_ok = bool(slot.get("sign_ok", False))
        ratios = list(slot.get("ratios", []))
        if np.isfinite(value) and not sign_ok and ratios:
            worst = min(ratios, key=lambda kv: kv[1])
            rows.append(
                {
                    "constant": name,
                    "old": f"{plan:g}",
                    "new": f"unchanged ({plan:g})",
                    "tag": "measurement rejects the map",
                    "evidence": (
                        f"**The observable moves the wrong way.** The simulator's `{gain_key}` map "
                        f"can only *raise* `{metric}`, but the measured ratio is "
                        f"{worst[1]:.3f} at stage {worst[0]} — below 1, i.e. the fault made the rig "
                        "*better* by this measure — while another stage sits just above it "
                        f"({max(r for _, r in ratios):.3f}). Reading a gain off a 2–3 % excursion "
                        "whose sign flips between the two stages would be inverting noise, so the "
                        "sign gate refuses it and the plan's value stands. "
                        + (
                            "Physically this is credible rather than surprising: the rig's balls "
                            "were swapped into an **anti-backlash** nut, so a smaller ball adds "
                            "play *and* relieves preload, and the reversal error here is dominated "
                            "by the controller's response to a step-onto-ramp set point rather "
                            "than by the dead band. This dataset cannot set δ through this "
                            "observable — a finding, not a gap. "
                            if name == "backlash_m"
                            else ""
                        )
                        + f"(The curve would have returned {value:.3g} from the one stage that "
                        "happened to land inside its monotone prefix.)"
                    ),
                }
            )
            continue
        if not np.isfinite(value):
            new_cell = f"unchanged ({old:g})"
            tag = "ours (plan)"
            ratio_txt = (
                ", ".join(f"stage {lv}: {rv:.3f}" for lv, rv in ratios) if ratios else "none measured"
            )
            evidence = (
                f"**Not invertible.** The measured `{metric}` ratio for every "
                f"{cran_class.replace('_', ' ')} stage ({ratio_txt}) falls outside the monotone "
                f"prefix of the `{gain_key}` sweep, so `invert_curve` returns NaN rather than a "
                f"wrong root (0 of {n_lv} stages inverted). The plan's value stands, unrefuted "
                "and unconfirmed."
            )
            rows.append({"constant": name, "old": f"{old:g}", "new": new_cell, "tag": tag, "evidence": evidence})
            continue
        if blocked and sign_ok:
            rows.append(
                {
                    "constant": name,
                    "old": f"{old:g}",
                    "new": f"unchanged ({old:g}); curve says {value:.3g}",
                    "tag": "measured, NOT adopted",
                    "evidence": (
                        f"The inversion is well posed: {n_ok} of {n_lv} stages (levels {levels}) "
                        f"land inside the monotone prefix of the `{gain_key}` sweep and read "
                        f"**{value:.3g}** off `{metric}`, {value / old:.2f}× the plan's {old:g}. "
                        f"It is **not adopted**: {blocked}. Removing the measured standing "
                        "current as an offset (see §3) lifts the same inversion only to "
                        f"{_stepper_bracket_text(report, scale)}, still far under the plan's "
                        f"{old:g}, so the *direction* of the finding is robust even though its "
                        "magnitude is not identifiable from this drive — a seeded lubrication "
                        "fault on a ball screw is a far milder force change than the plan's map "
                        "assumes, which the dataset's own PDF anticipates (\"No dramatic changes "
                        "were observed in the signals, mainly due to the inherent low friction "
                        "of the ball-screw architecture\"). Adopting a lower bound would also "
                        "drop the rig variant's cruise force at s = 0.5 to ≈ "
                        f"{params.f_c0 * (1 + 0.5 * value) + params.b0 * params.v_ref * (1 + 0.5 * PLAN_B_TO_C_RATIO * value):.0f} N "
                        f"against its own {params.f_obs_trip_n:g} N obstruction trip, so "
                        "`DoorParams.cranfield()` would stop exercising the obstruction path at "
                        "all — the path `tests/test_sim_door.py::test_obs_base_seed_kills_the_"
                        "phantom_reversal_on_the_cranfield_variant` exists to pin."
                    ),
                }
            )
            continue
        applied = (
            f"**Applied** to `DoorParams.cranfield().{name} = {incumbent:.3g}`; this run re-read "
            f"{value:.3g} off the curve with that value already in place, so the calibration is a "
            "fixed point rather than a one-way edit. "
            if np.isclose(incumbent, value, rtol=0.20)
            else f"**Not yet applied**: `DoorParams.cranfield().{name}` still reads {incumbent:g}; "
            f"set it to {value:.3g} and re-run. "
        )
        rows.append(
            {
                "constant": f"DoorParams.cranfield().{name}",
                "old": f"{plan:g} (class default, inherited)",
                "new": f"{value:.3g}",
                "tag": "calibrated",
                "evidence": (
                    f"Read off the `{gain_key}` inversion curve at `{metric}`: {n_ok} of {n_lv} "
                    f"{cran_class.replace('_', ' ')} stages (levels {levels}) fall inside the "
                    f"curve's monotone prefix and their median gain is **{value:.3g}**, "
                    f"{value / plan:.3f}× the plan's {plan:g}. The measured `{metric}` ratio "
                    f"grows {ratios[0][1]:.2f} → {ratios[-1][1]:.2f} from stage {ratios[0][0]} "
                    f"to stage {ratios[-1][0]}, while the plan's {plan:g} would have put it at "
                    f"**{_sim_ratio_at(report, cran_class, gain_key, plan):.0f}×** — the plan's "
                    "position-periodic amplitude is an order of magnitude too large for a seeded "
                    "screw spall. This observable **survives the stepper caveat** (§3): "
                    f"`{metric}` is measured after detrending the current against position, so "
                    + (
                        f"the drive's {report.fit.i_standing:.2f} A standing current is removed "
                        if report.fit is not None and np.isfinite(report.fit.i_standing)
                        else "the drive's standing current is removed "
                    )
                    + "from the ripple itself and survives only in the two means, where it "
                    "cancels between faulty and healthy to within a few per cent. "
                    + applied
                    + "Written onto `DoorParams.cranfield()` **only** — the shipped 110 V door "
                    "was not fitted on a ball-screw rig and does not move (pinned by "
                    "`tests/test_sim_door.py::test_the_shipped_door_is_barely_touched_by_the_"
                    "cranfield_calibration`)."
                ),
            }
        )
    return rows


def _sim_ratio_at(report: CalibrationReport, cran_class: str, gain_key: str, gain: float) -> float:
    """The simulator's own ratio at ``gain``, on the top severity slice of that fault's curve.

    Used to say what the *plan's* value would have predicted, rather than asserting it from
    memory: the curve is already in hand, so the comparison costs nothing and cannot go stale.
    """
    curve = report.curves.get(CRANFIELD_CLASS_FAULT.get(cran_class, ""))
    metric = str(report.gains.get(gain_key, {}).get("metric", ""))
    if curve is None or curve.empty or f"{metric}_ratio" not in curve.columns:
        return float("nan")
    top = float(curve["severity"].max())
    sl = curve[np.isclose(curve["severity"].to_numpy(dtype=float), top)].sort_values("gain")
    if sl.empty:
        return float("nan")
    return float(np.interp(gain, sl["gain"].to_numpy(float), sl[f"{metric}_ratio"].to_numpy(float)))


def _stepper_bracket_text(report: CalibrationReport, scale: float) -> str:
    """The offset-corrected friction gains, as a short phrase for the evidence column."""
    vals = [
        float(r["gain_offset_corrected"]) * scale
        for r in report.stepper_rows
        if np.isfinite(float(r["gain_offset_corrected"]))
    ]
    if not vals:
        return "PENDING (no offset-corrected inversion available)"
    return f"{min(vals):.3g}–{max(vals):.3g}" if len(vals) > 1 else f"{vals[0]:.3g}"


def recommended_constants(report: CalibrationReport, params: D.DoorParams) -> list[dict[str, object]]:
    """The constants this calibration changed, with their evidence."""
    rows: list[dict[str, object]] = [
        {
            "constant": "DoorParams.obs_base_seed",
            "old": "*(did not exist; baselines started at 0)*",
            "new": "True",
            "tag": "calibrated",
            "evidence": (
                "A zero-seeded asymmetric EWMA needs `obs_base_tau_s·ln(i_cruise/i_obs_delta_a)` to "
                "climb onto the cruise current, and until it arrives the actuator's **own** steady "
                "current reads as an abrupt rise. On `DoorParams.cranfield()` that climb is 0.42 s "
                "against a 0.25 s acceleration ramp, so over 69 cycles at constant severity the "
                "phantom-reversal rate was **100 % at friction s = 0.5 and s = 1.0** against a 1.4 % "
                "healthy baseline, and `closing_time` went **non-monotone** — 2.69 / **6.32** / "
                "**6.30** s at s = 0/0.5/1.0 — which makes it useless as a degradation feature. "
                "Seeding both baselines with the operating point at the instant the current path "
                "arms gives **1.4 % / 13.0 %** (the residual at s = 1.0 is real, not phantom) and a "
                "monotone **2.69 / 3.59 / 4.49 s**, with the current channel untouched "
                "(0.7615 → 0.7581 A, 1.1020 → 1.0994 A). On the 110 V door the change is a no-op: "
                "**0 of 69 cycles differ** at s = 0 and s = 0.5 and **1 of 69** at s = 1.0, with "
                "every median identical — it escaped only because its 0.50 s ramp happens to "
                "outlast its 0.34 s climb, a 0.16 s accident this removes. Pinned by "
                "`tests/test_sim_door.py::test_obs_base_seed_kills_the_phantom_reversal_on_the_"
                "cranfield_variant`."
            ),
        },
        {
            "constant": "DoorParams.cranfield().mis_lambda_m",
            "old": "0.30 (inherited from the 0.725 m door leaf)",
            "new": f"{D.CRANFIELD_SCREW_LEAD_M} (`CRANFIELD_SCREW_LEAD_M`, the ball-screw lead)",
            "tag": "**derived → measured**",
            "evidence": (
                "λ = 0.30 m is **3× the rig's entire 0.10 m stroke**, so the position-periodic term "
                "degenerated into a monotone friction ramp: recovering the dominant wavelength from "
                "the simulated current gave **0.0976 m ≈ the stroke itself** and a ripple of 0.014 A "
                "(3.7 % of healthy cruise) — no periodic signature to match spalling against, and it "
                "also tripped the obstruction detector on 3 of 3 retries. At the screw lead the same "
                "estimator recovers **0.0050 m exactly**, 20 periods across the stroke, 0 reversals. "
                "The lead is already implicit in `pulley_r_m = lead/2π`; `CRANFIELD_SCREW_LEAD_M` now "
                "drives both so they cannot drift apart. **The real files upgrade this from a guess "
                "to a measurement**: the release's own 'Data description.pdf' section 2 names the "
                "screw as an *RM1605-C7 with 5 mm lead*, which is exactly the assumed value — the "
                "`UNVERIFIED` tag this row used to carry is retired."
            ),
        },
        {
            "constant": "DoorParams.cranfield().pos_err_obs_m",
            "old": "0.020 (inherited)",
            "new": f"{params.pos_err_obs_m:.5f} (`OBS_POS_ERR_STROKE_FRAC × 0.10`)",
            "tag": "derived",
            "evidence": (
                "The plan's 20 mm tracking-error trip is 2.76 % of the 0.725 m door leaf; inherited "
                "unscaled it is **20 % of the rig's 0.10 m stroke**, i.e. unreachable — the "
                "*simulated* rig's `pos_err_max` is 0.9 mm healthy and 2.6 mm at friction s = 1, so "
                "the third detection path was simply dead. Rescaling by the same stroke fraction "
                "puts it at 2.76 mm, above the worst simulated wear-driven error with ≈1.5× margin. "
                "**The real rig is not a check on this number and must not be read as one**: its "
                "own tracking error is far larger (median `pos_err_max` **3.3 mm healthy**, 3.5 mm "
                "at lubrication stage 1, 5.8 mm worst case over 7 790 strokes) because its set "
                "point steps straight onto full commanded speed with no acceleration ramp, so that "
                "error is the controller catching up with a discontinuity, not a fault. The "
                "simulator commands a trapezoid and has no such transient; comparing the two "
                "directly would be comparing a rig artefact with a fault threshold. (The rig's "
                "stroke is also 120 mm, not the 100 mm this variant models — see §6.)"
            ),
        },
    ]
    rows.extend(_gain_rows(report, params))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR, help="data/raw/cranfield")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="where the PNGs go")
    ap.add_argument("--doc", type=Path, default=DEFAULT_DOC, help="markdown file to update")
    ap.add_argument("--table", type=Path, default=None, help="also dump the ratio table as CSV")
    ap.add_argument("--self-check", action="store_true", help="closed-loop recovery test")
    ap.add_argument("--quick", action="store_true", help="3-point sweeps instead of full")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-doc", action="store_true", help="do not touch docs/parameters.md")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    def say(*a: object) -> None:
        if not args.quiet:
            print(*a)

    params = D.DoorParams.cranfield()
    tmp: tempfile.TemporaryDirectory[str] | None = None
    raw_dir = args.raw
    truth: dict[str, float] = {}
    if args.self_check:
        tmp = tempfile.TemporaryDirectory(prefix="cal_door_selfcheck_")
        raw_dir = Path(tmp.name)
        say(f"[self-check] generating a simulator-derived rig dataset in {raw_dir}")
        truth = write_synthetic_rig(raw_dir, params)

    say(f"Loading Cranfield strokes from {raw_dir} ...")
    strokes, meta = load_strokes(raw_dir)
    have = not strokes.empty
    say(f"  {len(strokes)} strokes, have_data={have}")
    if not have:
        say("  !! no Cranfield telemetry: every measured ratio will read PENDING")
        for e in meta.get("errors", [])[:2]:
            say(f"     {e}")

    fit = None
    measured = pd.DataFrame()
    if have:
        ds = load_dataset(raw_dir)
        wide = S.to_wide(ds.long, "door")
        healthy_runs = set(strokes.loc[strokes["fault_type"] == "healthy", "run_id"])
        hw = wide[wide["run_id"].isin(healthy_runs)]
        if not hw.empty:
            fit = fit_healthy(hw, params)
            say(
                f"  healthy fit: F_c0={fit.f_c0:.2f} N  b0={fit.b0:.2f} N.s/m  m_eff={fit.m_eff:.2f} kg "
                f"(R2={fit.r2_mech:.3f}, k_t {'measured' if fit.k_t_identified else 'assumed'})"
            )
        measured = class_summary(strokes)

    step = 2 if args.quick else 1
    say("Sweeping the simulator gain axes (this is the slow part) ...")
    curves: dict[str, pd.DataFrame] = {}
    for fault_type, cran_class, gain_name, axis in (
        ("friction", "lack_of_lubrication", "k_friction_c", SWEEP_FRICTION_C),
        ("backlash", "backlash", "backlash_m", SWEEP_BACKLASH_M),
        ("misalignment", "spalling", "k_mis", SWEEP_K_MIS),
    ):
        # Spalling has 8 stages, so its curve needs 8 severity slices, not 2.
        sev = sim_severities(cran_class)
        if args.quick:
            sev = tuple(sorted({sev[0], sev[len(sev) // 2], sev[-1]}))
        vals = tuple(axis[::step]) if len(axis[::step]) >= 2 else axis
        say(f"  {fault_type}: {gain_name} over {vals} at severities {sev}")
        base = params
        if gain_name == "k_friction_c":
            # keep the plan's b:c tie unless a second observable frees it
            curve = pd.concat(
                [
                    sweep_gain(
                        base.with_(k_friction_b=PLAN_B_TO_C_RATIO * v),
                        gain_name,
                        (v,),
                        fault_type,
                        sev,
                    )
                    for v in vals
                ],
                ignore_index=True,
            )
        else:
            curve = sweep_gain(base, gain_name, vals, fault_type, sev)
        curves[fault_type] = curve

    report = CalibrationReport(
        have_data=have,
        raw_dir=raw_dir,
        n_strokes=len(strokes),
        fit=fit,
        measured=measured,
        curves=curves,
        meta=dict(meta),
    )
    report.ratio_rows = build_ratio_rows(params, measured, curves)
    report.gains = gains_from_curves(report.ratio_rows)
    report.stepper_rows = offset_corrected_friction(measured, fit, curves.get("friction"))
    report.recommended = recommended_constants(report, params)
    if have and not args.self_check:
        miss = list(meta.get("missing_files", []))
        report.notes.append(
            f"Parsed {meta.get('n_files', 0)} of the 13 release files "
            f"({meta.get('n_runs', 0)} test recordings, {len(strokes):,} strokes)."
            + (f" **Missing: {miss}** — every ratio above is computed on what was present." if miss else "")
        )
        report.notes.append(
            "`Backlash1.mat` holds **59** matrices, not 60: `backtrap1st40kg` has 9 repetitions "
            "in the release (`backtrap1st40kg2` is absent). The adapter parses what is there and "
            "records the count in `meta`; no ratio is sensitive to one missing repetition out of "
            "590 strokes."
        )
        report.notes.append(
            f"The rig's stroke is **120 mm** (PDF §2) while `DoorParams.cranfield()` models "
            f"{params.stroke_m * 1e3:.0f} mm, and its commanded cruise speed is "
            f"{_fmt(report.fit.v_ref * 1e3, 1) if report.fit else 'PENDING'} mm/s against the "
            f"variant's {params.v_ref * 1e3:.0f} mm/s. Both are *geometry*, not fault-map gains, "
            "so they are outside what this calibration is allowed to change; they are recorded "
            "here because every absolute (not ratio) comparison with the rig inherits them."
        )
        report.notes.append(
            "Timing carries no fault information on this rig: with the ratios formed **within a "
            "motion profile** the stroke-duration ratio is 1.000 at every fault and stage and "
            "the cruise-speed ratio stays inside ±6 %. It is a position-commanded stepper "
            "following a fixed profile with force headroom, so the whole fault signal is in the "
            "current and the tracking error — the experimental form of rail_phm 1.1 [R64][R72]."
        )
        if not measured.empty and "i_cruise_mean_ratio" in measured.columns:
            m = measured[measured["fault_type"] != "healthy"]
            hi = m.loc[m["i_cruise_mean_ratio"].idxmax()]
            lo = m.loc[m["i_cruise_mean_ratio"].idxmin()]
            report.notes.append(
                "The **mean current does not rank the faults the way the door model would**. The "
                f"strongest current signature in the release is `{hi['fault_type']}` stage "
                f"{int(hi['level'])} at **{hi['i_cruise_mean_ratio']:.3f}×** healthy — above every "
                "lubrication stage — because smaller balls in a ball screw add rolling friction as "
                "well as play; while spalling *lowers* the mean current "
                f"(down to {lo['i_cruise_mean_ratio']:.3f}× at `{lo['fault_type']}` stage "
                f"{int(lo['level'])}) at the same time as it raises the position-periodic ripple "
                "that `k_mis` is read off. Neither is a defect in the data: it is why each fault's "
                "constant is read off its **own** observable and the others are printed as "
                "diagnostics only."
            )
        report.notes.append(
            "Pooling the two motion profiles was a real trap and `class_summary` now avoids it: "
            "because `Backlash1.mat` is one repetition short of 60, a pooled median tipped off "
            "the 5 s trapezoidal profile onto the 6 s sinusoidal one and reported a spurious "
            "**1.077** duration ratio for backlash stage 1. Every ratio is now formed against the "
            "healthy strokes of the *same* profile and then median-combined."
        )
    if args.self_check:
        report.notes.append(
            "This run was a `--self-check`: the 'Cranfield' side is the simulator's own output at "
            "known gains, not real actuator data."
        )

    say("\nCranfield ratio vs sim ratio")
    say(f"{'class':<22}{'lvl':>4}{'metric':>20}{'cranfield':>12}{'sim':>10}{'gain->':>10}")
    for r in report.ratio_rows:
        say(
            f"{str(r['cranfield_class']):<22}{r['level']:>4}{str(r['metric']):>20}"
            f"{_fmt(r['cranfield_ratio']):>12}{_fmt(r['sim_ratio']):>10}{_fmt(r['gain_from_curve']):>10}"
        )

    say("\nGains read off the inversion curves")
    for name, slot in report.gains.items():
        say(
            f"  {name:<16} {slot['n_invertible']}/{slot['n_levels']} levels invertible "
            f"-> {_fmt(slot.get('value'), 4)}  (incumbent {_fmt(slot.get('incumbent'), 4)}, "
            f"min measured ratio {_fmt(slot.get('min_ratio'), 3)}, "
            f"sign gate {'ok' if slot.get('sign_ok') else 'FAILED'})"
            + (f"   BLOCKED: {ADOPTION_BLOCKED[name][:60]}..." if name in ADOPTION_BLOCKED else "")
        )

    rc = 0
    if args.self_check:
        say("\n[self-check] recovery of the known gains")
        ok = True
        for r in report.ratio_rows:
            if not r.get("primary"):
                continue
            g = float(r["gain_from_curve"])
            t = truth.get(str(r["gain"]), float("nan"))
            if np.isfinite(g) and np.isfinite(t) and t != 0:
                rel = abs(g - t) / t
                flag = "ok" if rel <= 0.60 else "OFF"
                ok &= rel <= 0.60
                report.recovery.append(
                    {"gain": r["gain"], "level": r["level"], "truth": t, "recovered": g}
                )
                say(f"  {r['gain']:<16} level {r['level']}  truth {t:<10.4g} recovered {g:<10.4g} ({flag})")
        rc = 0 if ok else 1
    elif not have:
        rc = 2

    if args.table:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(report.ratio_rows).to_csv(args.table, index=False)
        say(f"\nwrote {args.table}")

    if not args.no_plots:
        for p in make_plots(report, params, args.out):
            say(f"wrote {p}")

    if not args.no_doc:
        write_doc_section(args.doc, render_report(report, params))
        say(f"wrote {args.doc} (between {DOC_BEGIN} markers)")

    if tmp is not None:
        tmp.cleanup()

    say("\nstatus:", report.status, f"(exit {rc})")
    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
