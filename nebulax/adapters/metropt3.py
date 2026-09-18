"""MetroPT-3 (UCI 791) adapter -> ``pneumatic`` subsystem (Porto metro APU proxy).

Source: "MetroPT-3 (Air Compressor)" dataset, UCI ML Repository id 791, DOI 10.24432/C5VW3R,
licence CC BY 4.0. One CSV, ``MetroPT3(AirCompressor).csv``, 1,516,948 rows, nominally 10-second
cadence (not the 1 Hz the dataset card advertises -- see ``_read_raw``), 2020-02-01 -> 2020-09-01,
15 channels (7 analogue + 8 digital) covering one air-production-unit (APU): main reservoir
pressure, panel pressure, dryer differential pressure, oil temperature, motor current, and the
compressor/valve/dryer digitals. See ``docs/research/rail_phm.md`` sections 2 and 4.2 for the
physics this adapter's feature engineering follows.

What ``load()`` emits
----------------------
``long``     : all 15 MetroPT-3 signals verbatim (:data:`nebulax.schema.METROPT3_SIGNALS`), one
               unit-level component (``apu_1``, ``car=0``), no unit conversion (the file already
               uses bar / degC / A, matching the schema's pneumatic units).
``features`` : one row per **compressor cycle** (a full load/unload/off duty cycle, segmented from
               ``Motor_current`` -- see ``_classify_state``), per the plan's "compressor-cycle table
               keyed on COMP transitions" and rail_phm 4.2's duty-cycle features -- see
               ``_build_cycle_features``'s docstring for the full column table, including
               ``transition_frac`` (rail_phm's transition mask, section "Detector design
               additions": fraction of a cycle's samples within :data:`_TRANSITION_WINDOW_S` of a
               state change, a ``Towers`` flip, or a ``COMP`` edge -- see :func:`transition_mask`).
               NOT the separate
               "10-second aggregate" window-stats table the plan also mentions -- that is generic
               fixed-window aggregation with no MetroPT-specific logic, so it is left to
               ``nebulax.features`` (the ``window_stats`` input_kind models build it themselves from
               ``long`` via ``to_wide`` + rolling aggregation, using :func:`transition_mask` for its
               own ``is_transition`` column); duplicating it here would just be
               ``features/windows.py`` copy-pasted into an adapter. Recorded in ``meta``.
``fault_log`` : the 4 air-leak episodes from the UCI page's free-text failure-report table,
               hard-coded. **Dates are UCI's own report, not independently verified against the
               paper** -- ``meta['fault_log_note']`` says so and the CLI prints them for a human to
               check. MetroPT-3 has air-leak failures only (no oil leak; that is MetroPT-1/2)
               [rail_phm 2.2, 4.2].
``events``   : compressor start/stop, dryer tower switches, purge pulses, LPS trips and oil-low
               transitions -- all in :data:`nebulax.schema.EVENT_TYPES`.

Timestamp convention
---------------------
The CSV timestamp has no timezone. Per the adapter contract ("sources with a local-time clock
state the assumed offset"), we treat it as already UTC (no shift applied) so that the telemetry and
the hard-coded fault-report timestamps -- which come from the same clock -- stay aligned relative to
each other, whatever the true offset turns out to be. This mirrors the UCI dataset's own reporting
(Portugal, so the true offset is WET/WEST, UTC+0/+1) but we do not attempt a DST-aware shift because
there is no cited source for the recorder's actual clock discipline.

Accepted deferral (15 Sep 2026, orchestrator decision)
--------------------------------------------------------
The plan's separate "10-second aggregate" table (analogue mean/max per channel, digital duty
fraction, and an ``is_transition`` column from :func:`transition_mask`) is **not** built by this
adapter -- see "What ``load()`` emits" above. Ownership is assigned to the benchmark data loader
``nebulax/bench/data.py`` in the W2 stage, which should call :func:`transition_mask` on
``to_wide(long, "pneumatic")`` for its ``is_transition`` column when it is written; this module
stays scoped to the compressor-cycle table. See ``docs/provenance.md``'s MetroPT-3 entry for the
full accepted-deferral note.

This adapter's ``features`` table is also not a supervised dataset: only 6 of the 10,395
compressor cycles :func:`load` emits carry ``is_faulty=True``, because a leak episode collapses
into one multi-day cycle rather than many short faulty ones. MetroPT-3 is an anomaly-detection
dataset here, scored on events/alarm windows (``fault_log``, ``alarm_window_3d``), never treated
as a supervised positive class on the cycle table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from nebulax.schema import (
    APU_COMPONENT_IDS,
    Dataset,
    FAULT_LOG_COLUMNS,
    METROPT3_SIGNALS,
    coerce_events,
    coerce_fault_log,
    coerce_features,
    coerce_long,
    to_long,
)

__all__ = ["load", "transition_mask"]

SOURCE: Final[str] = "metropt3"
SUBSYSTEM: Final[str] = "pneumatic"
COMPONENT_ID: Final[str] = APU_COMPONENT_IDS[0]  # "apu_1"
RUN_ID: Final[str] = "metropt3"
TRAIN_ID: Final[str] = "porto_apu"

_CSV_NAME: Final[str] = "MetroPT3(AirCompressor).csv"

# Compressor state thresholds on Motor_current (A), fit by inspecting the file's own histogram
# (three clusters: ~0 A off, ~3.4-4.0 A unloaded/idling, ~5.3-6.5 A loaded -- see docs/provenance.md
# for the exact counts). Not from the paper; ours, and documented as such.
_I_OFF_MAX: Final[float] = 2.0
_I_LOADED_MIN: Final[float] = 5.0

# A logger gap this long or longer is treated as missing acquisition time, not real compressor-off
# duration, so per-sample duration is capped here before being summed into cycle durations. 331 of
# 1.5M inter-sample gaps in the file exceed 60 s; the largest is ~2 days.
_MAX_SAMPLE_DT_S: Final[float] = 60.0

# rail_phm.md, "Detector design additions" (pneumatic section): "Emit an `is_transition` mask and
# exclude +/-5 s around `Towers` flips and `COMP` load/offload edges from point scoring; score
# those regions with a change-point detector instead." CONFIRMED (rail_phm states the number
# itself; this is not our invention, so no UNVERIFIED tag). "COMP load/offload edges" is widened
# here to every state change of :func:`_classify_state` (off/unloaded/loaded) PLUS `COMP`'s own
# edges read directly (see :func:`transition_mask`): the `Motor_current`-derived cut does not
# exactly subsume `COMP` -- on the real file 703 of 33,846 (2.08 %) `COMP` digital edges lag the
# nearest `Motor_current`/`Towers` event by one 10 s sample, so `COMP` is ORed in on its own to
# get exact coverage, consistent with the plan's "compressor state changes (load, unload, off)".
#
# CAVEAT (does not change the number, which is rail_phm's own): this file's native cadence is
# ~10 s, so a +/-5 s window is below one sample period -- in practice it flags essentially only
# the edge sample itself rather than a temporal neighbourhood either side of it (empirically, on
# the real file the immediately preceding sample is flagged only ~21 % of the time and the next
# one ~40 %). A caller that actually needs a multi-sample neighbourhood excluded at this file's
# cadence should pass a larger ``window_s`` to :func:`transition_mask` (``window_s=15`` masks
# ~15.5 % of samples on the real file, vs. 6.16 % at the default); the default is kept at
# rail_phm's literal 5 s for fidelity to the spec, not because it excludes a neighbourhood here.
_TRANSITION_WINDOW_S: Final[float] = 5.0

# UCI additional_info.summary free-text failure-report table (MetroPT-3 dataset card, id 791),
# transcribed verbatim from data/raw/metropt3/MANIFEST.json. All four are reported as "Air Leak" /
# "High stress" -- MetroPT-3 has no oil-leak episode (that is MetroPT-1/2's F3). The report's own
# row numbering repeats "#1" for two different rows; we keep the report's text as-is in
# ``params_json`` and number our own rows sequentially.
_FAULT_REPORTS: Final[tuple[dict[str, Any], ...]] = (
    {
        "t_onset": "2020-04-18T00:00:00Z",
        "t_failure": "2020-04-18T23:59:00Z",
        "report": "Air Leak, High stress",
    },
    {
        "t_onset": "2020-05-29T23:30:00Z",
        "t_failure": "2020-05-30T06:00:00Z",
        "report": "Air Leak, High stress, Maintenance on 30Apr at 12:00 (sic, as printed on the UCI page)",
    },
    {
        "t_onset": "2020-06-05T10:00:00Z",
        "t_failure": "2020-06-07T14:30:00Z",
        "report": "Air Leak, High stress, Maintenance on 8Jun at 16:00",
    },
    {
        "t_onset": "2020-07-15T14:30:00Z",
        "t_failure": "2020-07-15T19:00:00Z",
        "report": "Air Leak, High stress, Maintenance on 16Jul at 00:00",
    },
)

_ALARM_LEAD: Final[pd.Timedelta] = pd.Timedelta(days=3)


def _find_csv(raw_dir: Path) -> Path:
    path = raw_dir / _CSV_NAME
    if path.exists():
        return path
    raise FileNotFoundError(
        f"metropt3 adapter: expected {path} (run scripts/download_data.py --dataset metropt3, "
        f"or pass the directory containing '{_CSV_NAME}' as raw_dir); raw_dir={raw_dir} "
        f"contains: {sorted(p.name for p in raw_dir.glob('*')) if raw_dir.exists() else '<missing>'}"
    )


def _read_raw(raw_dir: Path) -> pd.DataFrame:
    """Read the CSV, drop the row-index column, parse the timestamp (assumed UTC)."""
    path = _find_csv(raw_dir)
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    missing = [c for c in METROPT3_SIGNALS if c not in df.columns]
    if missing:
        raise ValueError(f"metropt3 adapter: {path} is missing expected column(s) {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return df


def _classify_state(motor_current: np.ndarray) -> np.ndarray:
    """0 = off, 1 = unloaded (idling), 2 = loaded (compressing), from Motor_current alone."""
    return np.where(motor_current < _I_OFF_MAX, 0, np.where(motor_current >= _I_LOADED_MIN, 2, 1)).astype(np.int8)


def _sample_durations_s(ts: pd.Series) -> np.ndarray:
    """Forward-looking per-sample duration (time to the next sample), capped at ``_MAX_SAMPLE_DT_S``
    and with the final sample given the file's own median cadence."""
    t = ts.to_numpy(dtype="datetime64[ms]").astype(np.int64) / 1000.0
    dt = np.empty(len(t), dtype=np.float64)
    if len(t) > 1:
        dt[:-1] = np.diff(t)
        dt[-1] = np.median(dt[:-1])
    elif len(t) == 1:
        dt[0] = 0.0
    return np.minimum(dt, _MAX_SAMPLE_DT_S)


def _timestamp_seconds(ts: pd.Series) -> np.ndarray:
    """``timestamp`` as float epoch seconds (ms resolution is ample for a 10 s-cadence file)."""
    return ts.to_numpy(dtype="datetime64[ms]").astype(np.int64) / 1000.0


def _backward_diff_rate(ts: pd.Series, values: np.ndarray) -> np.ndarray:
    """Per-sample ``d(values)/dt`` using the *preceding* sample, at whatever cadence ``ts`` has.

    This is the irregular-cadence generalisation of
    :func:`nebulax.sim.pneumatic._cycle_features`'s fixed-grid ``np.diff(P, prepend=P[0]) / dt``:
    the sample where a phase (loaded/off/...) begins carries the transition-in delta and is
    attributed to the phase it lands in, exactly like the simulator's convention, instead of the
    phase's own internal ``(last - first) / duration`` (which drops the entry edge). For a
    uniform-cadence trace restricted to one contiguous phase run the two are algebraically
    identical (a telescoping sum divided by the sample count vs. divided by the same count via
    ``reduceat``); they differ only when the cadence is irregular, which is why this uses the
    actual per-sample ``dt`` rather than a fixed one.

    ``values[0]`` has no preceding sample, so ``rate[0]`` is ``0.0`` -- matching the simulator's
    ``prepend=P[0]`` convention (``dP[0] = P[0] - P[0] = 0``), not ``NaN``, so the two sides treat
    a record's very first sample identically instead of the adapter excluding it from a mean the
    simulator includes it in. A gap of
    :data:`_MAX_SAMPLE_DT_S` or more between two samples is capped the same way
    :func:`_sample_durations_s` caps it, so a multi-day logger outage cannot produce a
    near-zero-but-technically-huge-denominator rate that silently mutes real dynamics either side
    of it.
    """
    t = _timestamp_seconds(ts)
    v = np.asarray(values, dtype=np.float64)
    rate = np.full(len(t), np.nan, dtype=np.float64)
    if len(t) > 0:
        # No preceding sample for index 0 -- matches the simulator's own convention
        # (``np.diff(P, prepend=P[0])`` gives ``dP[0] = P[0] - P[0] = 0``, not NaN), so a record
        # that starts already mid-cycle (never true for the real MetroPT-3 file, which starts
        # OFF) gets the same cycle-0 rate on both sides instead of the adapter silently excluding
        # that one sample from the mean while the simulator includes it as a zero.
        rate[0] = 0.0
    if len(t) > 1:
        dt_back = np.minimum(np.diff(t), _MAX_SAMPLE_DT_S)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate[1:] = np.where(dt_back > 0, np.diff(v) / dt_back, np.nan)
    return rate


def transition_mask(df: pd.DataFrame, *, window_s: float = _TRANSITION_WINDOW_S) -> np.ndarray:
    """Per-sample ``is_transition`` flag, at this file's native ~10 s cadence: ``True`` within
    ``window_s`` seconds of a compressor state change (:func:`_classify_state`'s off/unloaded/
    loaded cut on ``Motor_current``), a ``Towers`` flip, or -- when the column is present -- a
    ``COMP`` digital edge, per rail_phm.md's transition mask (see :data:`_TRANSITION_WINDOW_S`).
    ``COMP`` is read directly (not just inferred from ``Motor_current``) because the two lag each
    other by up to one sample on the real file: 703 of 33,846 (2.08 %) ``COMP`` edges are not
    coincident with the nearest ``Motor_current``/``Towers`` event, so OR-ing ``COMP``'s own edges
    in is needed for exact "COMP load/offload edges" coverage as rail_phm names it, at a cost of
    +0.05 pp masked samples (6.164 % -> 6.210 % on the real file).

    CAVEAT: at this file's ~10 s cadence, ``window_s`` below one sample period (the 5 s default)
    flags essentially only the edge sample itself, not a temporal neighbourhood either side of it
    -- see :data:`_TRANSITION_WINDOW_S` for measured numbers. Pass a larger ``window_s`` if a
    caller needs an actual multi-sample exclusion zone at this cadence.

    This is the ingredient the generic 10-second-aggregate / ``window_stats`` table needs to
    exclude transition samples from point scoring. It is **not** materialised as a ``long``
    column by this adapter: ``nebulax.schema.SIGNALS["pneumatic"]`` is a closed vocabulary of
    physical channels, and adding a synthetic signal there is a schema change outside this
    fix's scope. Callers building that table downstream from ``long`` (see the module docstring
    -- ``nebulax.features``, via ``to_wide`` + rolling aggregation) can compute this same flag by
    calling this function on the wide frame. Used internally by :func:`load` to compute each
    cycle's ``transition_frac``.

    ``df`` must have ``timestamp``, ``Motor_current`` and ``Towers`` columns; ``COMP`` is used
    when present (it always is for :func:`load`'s own call, since it is one of
    :data:`nebulax.schema.METROPT3_SIGNALS`) but is optional here so this function also works on
    a caller-built frame that dropped it. The returned array is aligned with ``df``'s rows (no
    reindexing/sorting is done here).
    """
    state = _classify_state(df["Motor_current"].to_numpy(dtype=np.float64))
    towers = df["Towers"].to_numpy()
    changed = np.r_[False, state[1:] != state[:-1]] | np.r_[False, towers[1:] != towers[:-1]]
    if "COMP" in df.columns:
        comp = df["COMP"].to_numpy()
        changed = changed | np.r_[False, comp[1:] != comp[:-1]]
    t = _timestamp_seconds(df["timestamp"])
    event_t = t[changed]
    if event_t.size == 0:
        return np.zeros(len(df), dtype=bool)
    idx = np.searchsorted(event_t, t)
    lo = event_t[np.clip(idx - 1, 0, event_t.size - 1)]
    hi = event_t[np.clip(idx, 0, event_t.size - 1)]
    return np.minimum(np.abs(t - lo), np.abs(t - hi)) <= window_s


def _build_long(df: pd.DataFrame) -> pd.DataFrame:
    wide = df[["timestamp", *METROPT3_SIGNALS]]
    return to_long(
        wide,
        SUBSYSTEM,
        signals=METROPT3_SIGNALS,
        source=SOURCE,
        run_id=RUN_ID,
        train_id=TRAIN_ID,
        car=0,
        component_id=COMPONENT_ID,
        dropna=False,
    )


def _build_events(df: pd.DataFrame, state: np.ndarray) -> pd.DataFrame:
    """Discrete events: compressor start/stop, tower switches, purge pulses, LPS, oil-low."""
    n = len(df)
    active = state != 0
    start = active & ~np.r_[False, active[:-1]]
    stop = (~active) & np.r_[False, active[:-1]]

    def _rising(col: str) -> np.ndarray:
        v = df[col].to_numpy()
        return (v[1:] > 0.5) & (v[:-1] <= 0.5)

    def _changed(col: str) -> np.ndarray:
        v = df[col].to_numpy()
        return v[1:] != v[:-1]

    rows: list[dict[str, Any]] = []
    ts = df["timestamp"]

    def _add(mask: np.ndarray, offset: int, event: str) -> None:
        idx = np.flatnonzero(mask) + offset
        for i in idx:
            rows.append(
                {
                    "run_id": RUN_ID,
                    "timestamp": ts.iloc[int(i)],
                    "train_id": TRAIN_ID,
                    "car": 0,
                    "subsystem": SUBSYSTEM,
                    "component_id": COMPONENT_ID,
                    "event": event,
                    "detail_json": "{}",
                }
            )

    _add(start, 0, "comp_load")
    _add(stop, 0, "comp_off")
    if n > 1:
        _add(_changed("Towers"), 1, "tower_switch")
        _add(_rising("Pressure_switch"), 1, "purge")
        _add(_rising("LPS"), 1, "lps")
        _add(_rising("Oil_level"), 1, "oil_low")

    if not rows:
        from nebulax.schema import empty_events

        return empty_events()
    return coerce_events(pd.DataFrame(rows))


def _build_cycle_features(
    df: pd.DataFrame, state: np.ndarray, dt: np.ndarray, is_transition: np.ndarray
) -> pd.DataFrame:
    """One row per compressor duty cycle: [start of an off->active transition, next such start).

    Per-cycle ``*_loaded_*`` / ``*_off_*`` feature definitions (kept identical, by construction
    and by ``tests/test_adapter_metropt3.py::test_loaded_off_features_match_simulator``, to
    :func:`nebulax.sim.pneumatic._cycle_features`. **Not** echoed in that function's own
    docstring -- ``nebulax/sim/pneumatic.py`` is out of scope for this fix (only
    ``TP2_minus_TP3_mean`` / ``H1_loaded_mean`` semantics are documented there) and is left
    untouched; this table is the single source of truth, cross-checked against the simulator's
    actual output (not a hand-picked helper) by
    ``tests/test_adapter_metropt3.py::test_loaded_off_features_match_simulator``):

    ====================  =============================================  =======================
    column                definition                                     restricted to
    ====================  =============================================  =======================
    ``t_loaded``          summed sample duration                         ``state == 2`` (loaded)
    ``t_unloaded``        summed sample duration                         ``state == 1`` (unloaded)
    ``t_off``             summed sample duration                         ``state == 0`` (off)
    ``I_loaded_mean``     mean ``Motor_current``                         ``state == 2`` AND at
                                                                          least 4 samples into the
                                                                          cycle (drops the
                                                                          starting-current
                                                                          transient, matching the
                                                                          simulator's own
                                                                          "first 4 samples of the
                                                                          cycle" exclusion)
    ``H1_loaded_mean``    mean ``H1``                                    ``state == 2``
    ``TP2_minus_TP3_mean`` mean ``TP2 - TP3``                            ``state == 2`` (fixed here
                                                                          -- was the whole cycle,
                                                                          which includes the ~0 bar
                                                                          ``TP2`` off/unloaded
                                                                          samples and so does not
                                                                          share a meaning with the
                                                                          simulator's identically-
                                                                          named column)
    ``dP_dt_loaded``      mean of :func:`_backward_diff_rate`            ``state == 2``
                          (``Reservoirs``)
    ``dP_dt_off``         mean of :func:`_backward_diff_rate`            ``state == 0``
                          (``Reservoirs``)
    ``I_start_peak``      max ``Motor_current``                          whole cycle (fixed here --
                                                                          was the first 3 samples
                                                                          only, matching neither the
                                                                          simulator's whole-cycle
                                                                          ``fmax.reduceat`` nor the
                                                                          real file, where the
                                                                          current maximum is not
                                                                          always at cycle start)
    ``tower_switches``    count of ``Towers`` flips                      whole cycle, including a
                                                                          flip on the cycle's own
                                                                          first sample (fixed here --
                                                                          was excluded, unlike the
                                                                          simulator's plain
                                                                          ``add.reduceat`` sum)
    ``purge_count``       count of ``Pressure_switch`` rising edges      whole cycle, including one
                                                                          on the cycle's own first
                                                                          sample (same fix as
                                                                          ``tower_switches``, for the
                                                                          same reason)
    ``transition_frac``   mean of :func:`transition_mask`                whole cycle
    ====================  =============================================  =======================

    ``dP_dt_loaded`` / ``dP_dt_off`` used to be ``(last - first) Reservoirs`` over the phase
    divided by the phase's summed duration, which drops the transition-in sample's own pressure
    delta (a fencepost: a contiguous run of *n* samples has only *n - 1* internal diffs, so
    dividing by the phase's *n*-sample duration understates the rate). ``_backward_diff_rate``
    instead gives every sample -- including the one where the phase begins -- its own
    ``d(Reservoirs)/dt`` against the *preceding* sample (whichever phase that was in), then means
    that per-sample rate over the phase's samples: for a uniform-cadence run this reduces to
    exactly the simulator's ``mean(diff(Reservoirs) over the phase) / dt``, and it generalises
    correctly to this file's irregular ~10 s cadence.

    ``state`` here is the adapter's 3-way :func:`_classify_state` cut on ``Motor_current``
    (0 off / 1 unloaded / 2 loaded), which is the ``Motor_current >= 5 A`` / COMP-derived cut
    ``scripts/calibrate_pneumatic.py`` also uses; the simulator's 5-state machine collapses onto
    it as ``loaded = state in {LOADING, LOADED}``, ``off = state == OFF``.

    ``I_start_peak`` / ``tower_switches`` / ``purge_count`` used to disagree with the simulator's
    ``nebulax.sim.pneumatic._cycle_features`` in ways an earlier synthetic-trace test could not
    see, because that trace happened to put the current's global maximum in the cycle's first two
    samples and its one ``Towers`` flip away from every cycle boundary: ``I_start_peak`` restricted
    the max to the first 3 samples (the simulator takes ``fmax`` over the whole cycle -- on the
    real file the two differ on 97.9 % of cycles), and ``tower_switches`` / ``purge_count`` dropped
    a flip/pulse landing exactly on the cycle's own first sample (the simulator's plain
    ``add.reduceat`` sum counts it -- 7.8 % of the real file's ``Towers`` flips land there). All
    three are now plain whole-cycle reductions, matching the simulator exactly.
    """
    active = state != 0
    start = active & ~np.r_[False, active[:-1]]
    cycle_id_raw = np.cumsum(start) - 1  # samples before the first start get -1 (dropped below)
    keep = cycle_id_raw >= 0
    if not keep.any():
        return pd.DataFrame()

    seg = pd.DataFrame(
        {
            "cycle_id": cycle_id_raw[keep],
            "timestamp": df["timestamp"].to_numpy()[keep],
            "state": state[keep],
            "dt": dt[keep],
            "motor_current": df["Motor_current"].to_numpy()[keep],
            "reservoirs": df["Reservoirs"].to_numpy()[keep],
            "oil_temperature": df["Oil_temperature"].to_numpy()[keep],
            "tp2": df["TP2"].to_numpy()[keep],
            "tp3": df["TP3"].to_numpy()[keep],
            "h1": df["H1"].to_numpy()[keep],
            "lps": df["LPS"].to_numpy()[keep],
            "is_transition": is_transition[keep],
            "dp_rate": _backward_diff_rate(df["timestamp"], df["Reservoirs"].to_numpy())[keep],
        }
    )
    seg["pos_in_cycle"] = seg.groupby("cycle_id").cumcount()
    seg["dt_off"] = np.where(seg["state"] == 0, seg["dt"], 0.0)
    seg["dt_unloaded"] = np.where(seg["state"] == 1, seg["dt"], 0.0)
    seg["dt_loaded"] = np.where(seg["state"] == 2, seg["dt"], 0.0)
    # drop the starting-current transient from I_loaded_mean: the first 4 samples of the cycle,
    # matching nebulax.sim.pneumatic._cycle_features's own "steady" exclusion.
    steady = (seg["state"] == 2) & (seg["pos_in_cycle"] >= 4)
    seg["i_loaded"] = np.where(steady, seg["motor_current"], np.nan)
    seg["h1_loaded"] = np.where(seg["state"] == 2, seg["h1"], np.nan)
    seg["dp_rate_loaded"] = np.where(seg["state"] == 2, seg["dp_rate"], np.nan)
    seg["dp_rate_off"] = np.where(seg["state"] == 0, seg["dp_rate"], np.nan)
    # loaded-only, matching nebulax.sim.pneumatic._cycle_features's TP2_minus_TP3_mean: this used
    # to average over the WHOLE cycle (dominated by the ~0 bar TP2 off/unloaded samples, ~-7.9 bar
    # on real data), which is a different quantity from the simulator's identically-named column.
    seg["tp2_minus_tp3_loaded"] = np.where(seg["state"] == 2, seg["tp2"] - seg["tp3"], np.nan)

    g = seg.groupby("cycle_id", sort=True)
    out = pd.DataFrame(
        {
            "t_start": g["timestamp"].min(),
            "t_end": g["timestamp"].max(),
            "t_loaded": g["dt_loaded"].sum(),
            "t_unloaded": g["dt_unloaded"].sum(),
            "t_off": g["dt_off"].sum(),
            "I_loaded_mean": g["i_loaded"].mean(),
            # whole cycle, matching nebulax.sim.pneumatic._cycle_features's
            # fmax.reduceat(I, bounds) -- was max over the first 3 samples only (fixed here).
            "I_start_peak": g["motor_current"].max(),
            "T_oil_max": g["oil_temperature"].max(),
            "TP2_minus_TP3_mean": g["tp2_minus_tp3_loaded"].mean(),
            "H1_loaded_mean": g["h1_loaded"].mean(),
            "LPS_any": g["lps"].max() > 0.5,
            "transition_frac": g["is_transition"].mean(),
            "dP_dt_loaded": g["dp_rate_loaded"].mean(),
            "dP_dt_off": g["dp_rate_off"].mean(),
        }
    )

    denom = out["t_loaded"] + out["t_unloaded"] + out["t_off"]
    out["duty_ratio"] = np.where(denom > 0, out["t_loaded"] / denom, np.nan)
    out["idle_run_ratio"] = np.where(out["t_loaded"] > 0, out["t_off"] / out["t_loaded"], np.nan)
    out["hour_of_day"] = out["t_start"].dt.hour + out["t_start"].dt.minute / 60.0

    # tower_switches / purge_count per cycle, from the same rising/changed logic as _build_events
    # but grouped by cycle instead of emitted as events. Counted over the whole cycle, including a
    # flip/pulse landing on the cycle's own first sample -- matching
    # nebulax.sim.pneumatic._cycle_features's plain add.reduceat(tower_flip / purge_start, bounds)
    # sum, which has no such exclusion (was excluded here, fixed above in this run).
    towers = df["Towers"].to_numpy()
    pressure_switch = df["Pressure_switch"].to_numpy()
    tower_changed = np.r_[False, towers[1:] != towers[:-1]]
    purge_rising = np.r_[False, (pressure_switch[1:] > 0.5) & (pressure_switch[:-1] <= 0.5)]
    seg2 = pd.DataFrame(
        {
            "cycle_id": cycle_id_raw[keep],
            "tower_changed": tower_changed[keep],
            "purge_rising": purge_rising[keep],
        }
    )
    out["tower_switches"] = seg2.groupby("cycle_id")["tower_changed"].sum().reindex(out.index).fillna(0).astype(int)
    out["purge_count"] = seg2.groupby("cycle_id")["purge_rising"].sum().reindex(out.index).fillna(0).astype(int)

    out = out.reset_index()
    out["cycle_id"] = out["cycle_id"].astype(np.int64)
    out["run_id"] = RUN_ID
    out["source"] = SOURCE
    out["train_id"] = TRAIN_ID
    out["car"] = 0
    out["subsystem"] = SUBSYSTEM
    out["component_id"] = COMPONENT_ID
    return out


def _label_cycles(features: pd.DataFrame, fault_log: pd.DataFrame) -> pd.DataFrame:
    """Join is_faulty / fault_type / alarm_window_3d from the hard-coded fault windows."""
    if features.empty or fault_log.empty:
        features = features.copy()
        features["is_faulty"] = False
        features["fault_type"] = "healthy"
        features["alarm_window_3d"] = False
        return features

    features = features.copy()
    is_faulty = np.zeros(len(features), dtype=bool)
    alarm = np.zeros(len(features), dtype=bool)
    t_start = features["t_start"]
    t_end = features["t_end"]
    for row in fault_log.itertuples():
        onset, failure = row.t_onset, row.t_failure
        is_faulty |= (t_start <= failure).to_numpy() & (t_end >= onset).to_numpy()
        lead = onset - _ALARM_LEAD
        alarm |= (t_start <= failure).to_numpy() & (t_end >= lead).to_numpy()
    features["is_faulty"] = is_faulty
    features["fault_type"] = np.where(is_faulty, "air_leak", "healthy")
    features["alarm_window_3d"] = alarm
    return features


def _build_fault_log() -> pd.DataFrame:
    rows = []
    for report in _FAULT_REPORTS:
        rows.append(
            {
                "run_id": RUN_ID,
                "train_id": TRAIN_ID,
                "car": 0,
                "subsystem": SUBSYSTEM,
                "component_id": COMPONENT_ID,
                "fault_type": "air_leak",
                "t_onset": report["t_onset"],
                "t_failure": report["t_failure"],
                "t_functional_failure": pd.NaT,
                "gamma": np.nan,
                "shape": None,
                "params_json": json.dumps(
                    {
                        "source": "UCI additional_info.summary failure-report table (dataset id 791)",
                        "report_text": report["report"],
                        "verify_dates": True,
                    }
                ),
            }
        )
    df = pd.DataFrame(rows, columns=list(FAULT_LOG_COLUMNS))
    return coerce_fault_log(df)


def load(raw_dir: str | Path) -> Dataset:
    """Load the MetroPT-3 CSV under ``raw_dir`` into a schema-conformant :class:`Dataset`.

    Raises ``FileNotFoundError`` if the CSV is not where expected.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"metropt3 adapter: raw_dir {raw_dir} does not exist; expected it to contain "
            f"'{_CSV_NAME}' (run scripts/download_data.py --dataset metropt3)"
        )
    df = _read_raw(raw_dir)
    motor_current = df["Motor_current"].to_numpy(dtype=np.float64)
    state = _classify_state(motor_current)
    dt = _sample_durations_s(df["timestamp"])
    is_trans = transition_mask(df)

    long = coerce_long(_build_long(df))
    fault_log = _build_fault_log()
    events = _build_events(df, state)
    cycle_features = _build_cycle_features(df, state, dt, is_trans)
    if len(cycle_features):
        cycle_features = _label_cycles(cycle_features, fault_log)
        features = coerce_features(cycle_features)
    else:
        from nebulax.schema import empty_features

        features = empty_features()

    n_gap_capped = int((dt >= _MAX_SAMPLE_DT_S).sum())
    meta: dict[str, Any] = {
        "doi": "10.24432/C5VW3R",
        "uci_id": 791,
        "licence": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/legalcode)",
        "citation": (
            "Veloso, B., Ribeiro, R.P., Gama, J., Pereira, P.M. (2022). MetroPT-3 Dataset. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5VW3R"
        ),
        "source_file": _CSV_NAME,
        "n_rows_raw": int(len(df)),
        "nominal_cadence_s": float(np.median(dt[dt > 0])) if (dt > 0).any() else None,
        "n_gaps_over_60s": n_gap_capped,
        "timestamp_assumption": (
            "no timezone stated in the source; treated as UTC without shift so telemetry and the "
            "hard-coded fault-report timestamps (same clock) stay mutually aligned"
        ),
        "unit_conversion": "none; TP2/TP3/H1/Reservoirs/DV_pressure already bar, Oil_temperature "
        "already degC, Motor_current already A -- matches nebulax.schema pneumatic units",
        "component_mapping": f"single APU -> component_id={COMPONENT_ID!r}, car=0 (unit-level)",
        "compressor_state_thresholds": {
            "off_max_A": _I_OFF_MAX,
            "loaded_min_A": _I_LOADED_MIN,
            "note": "ours, fit from this file's own trimodal Motor_current histogram, not from the paper",
        },
        "fault_log_note": (
            "the 4 episodes are transcribed verbatim from the UCI page's free-text failure-report "
            "table (see data/raw/metropt3/MANIFEST.json); dates are NOT independently re-verified "
            "against the underlying paper -- verify before quoting as ground truth"
        ),
        "features_note": (
            "features = compressor-cycle table only (one row per load/unload/off duty cycle, "
            "segmented from Motor_current). The plan's separate '10-second aggregate' window-stats "
            "table with its own is_transition column is a deliberate, documented scope boundary of "
            "THIS fix task (declared files: nebulax/adapters/metropt3.py + its tests, "
            "docs/provenance.md, one docs/parameters.md bullet) -- it is generic fixed-window "
            "aggregation with no MetroPT-specific logic and belongs in nebulax.features / the "
            "window_stats input_kind, built from `long` at bench time, a module this task does not "
            "touch; transition_mask() is the is_transition ingredient that table needs, per "
            "rail_phm.md's transition mask; see the module docstring and "
            "_build_cycle_features's docstring table, and docs/provenance.md's metropt3 entry for "
            "the full acceptance note."
        ),
        "transition_window_s": _TRANSITION_WINDOW_S,
        "transition_window_note": (
            "5.0 s is rail_phm.md's own number ('+/-5 s around Towers flips and COMP load/offload "
            "edges'); at this file's ~10 s cadence that is below one sample period, so it flags "
            "essentially only the edge sample, not a neighbourhood either side of it (measured: the "
            "preceding sample is flagged ~21% of the time, the next one ~40%). transition_mask() "
            "now also reads COMP directly (not just Motor_current/Towers) for exact edge coverage."
        ),
        "n_cycles": int(len(features)),
        "n_events": int(len(events)),
    }
    return Dataset(long=long, features=features, fault_log=fault_log, events=events, meta=meta)
