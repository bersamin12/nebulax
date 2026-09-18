"""Door subsystem: stream loading, cycle segmentation, per-cycle physics features.

Problem Statement 3 hands the Door subsystem **one continuous 50 Hz stream** (`Train.csv`,
`Test.csv`) rather than one file per cycle, so everything downstream starts here:

``load_stream`` -> ``segment`` -> ``cycle_features`` -> :class:`FoldBaseline` (fold-local) -> model.

Three facts about the released data drive the design (measured 17 Sep 2026, see
``results/ps3/door_cv.md``):

* the stream is sampled at 50 Hz (20 ms) inside a cycle and the recorder simply stops between
  cycles, so 109 of the 110 `Train.csv` cycle boundaries coincide with a timestamp gap > 0.5 s
  and the 110th is the first row of the file: the gap rule reproduces all 110 answer segments
  **exactly** (start, end and row count) and cuts `Test.csv` into 38 cycles;
* every row belongs to a labelled cycle - there are no idle rows between cycles - which is why
  the fallback segmenter (:func:`segment_by_state`) has to work off the position sweep and the
  command edges rather than an "is the door moving" flag;
* door leaf position runs 0 <-> ~700 counts but the extremes differ between the streams
  (`Test.csv` reaches 807), so every travel-phase window is computed **within the cycle** from
  its own endpoints, never from a global constant.

Feature design follows `docs/research/ps3_addendum.md` section 1.2: abnormal resistance is
mechanical work the motor must supply, so it shows as **excess current at a given position**
rather than a global mean shift. Hence per-regime current statistics on the accelerate / cruise /
decelerate split [R64], ``i_mid`` over 15-85 % travel, a back-EMF residual, the mechanical energy
proxy, the position of the current peak, the distance from a fold-fitted current-vs-position
template, cycle duration against the controller's commanded open/close time registers, and the
two coarsest DWT detail bands of the current plus the breakaway peak [R66][R67]. Per-operation
fold-fitted baselines with a robust peer z are the published answer to the Info Kit's core
problem 2, "distributions differ among doors" [R78].

Nothing in this module is fitted on labels except :class:`FoldBaseline`, which is the single
fold-local object in the door pipeline: it is fitted on the training fold's rows only (plan
section W4, ``docs/ps3_contract.md`` section 7). The ``mode="batch"`` variant - the declared
ablation - re-estimates the same statistics from the block being predicted, unlabelled, and is
mirrored independently inside every held block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from nebulax.ps3.common import DOOR_LABELS, parse_door_timestamps

__all__ = [
    "FEATURE_VERSION",
    "GAP_SECONDS",
    "PHASE_EDGES",
    "N_PROFILE_BINS",
    "RATIO_CLIP",
    "Z_CLIP",
    "CANONICAL_COLUMNS",
    "BASELINE_FEATURES",
    "PHYSICS_FEATURES",
    "DoorStream",
    "DoorFeats",
    "load_stream",
    "segment_by_gaps",
    "segment_by_state",
    "segment",
    "cycle_features",
    "FoldBaseline",
    "label_segments",
]

#: Bump when a feature definition changes; it keys the parquet cache and lands in the CV json.
FEATURE_VERSION = "door-f1"

#: A timestamp step longer than this starts a new cycle (Door Info Kit section 2.2: the recorder
#: stops between cycles). 0.5 s = 25 sample periods at 50 Hz, i.e. well clear of jitter.
GAP_SECONDS = 0.5

#: Travel-fraction window edges splitting a stroke into opening / cruise / closing phases.
#: The mid window (15-85 % of travel) is the brief's ``i_mid`` window.
PHASE_EDGES: tuple[float, float] = (0.15, 0.85)

#: Bins of the current-versus-position profile used for the template-distance feature.
N_PROFILE_BINS = 64

#: Guard rails on the fold-relative features (see :meth:`FoldBaseline.transform`): a ratio is
#: clipped to ``[0, RATIO_CLIP]`` and a robust z to ``+-Z_CLIP``. Ten times the fold baseline is
#: already far beyond any door fault in the training stream (the worst abnormal cycle sits at
#: 3.2x), so the clip only ever bites on a broken or out-of-distribution channel.
RATIO_CLIP = 10.0
Z_CLIP = 25.0

#: ``canonical name -> accepted header spellings`` (matched case-insensitively, punctuation and
#: whitespace stripped). The organisers' header uses "Motor electrodynamic force" while the
#: parameter list calls it "Motor back electromotive force"; both are accepted.
CANONICAL_COLUMNS: dict[str, tuple[str, ...]] = {
    "datetime": ("datetime", "time", "timestamp"),
    "current": ("motorcurrentma", "motorcurrent", "current"),
    "voltage": ("motorvoltage10mv", "motorvoltage", "voltage"),
    "emf": (
        "motorelectrodynamicforce",
        "motorbackelectromotiveforce",
        "backemf",
        "emf",
    ),
    "open_time": ("dooropeningtime1s", "dooropeningtime01s", "dooropeningtime"),
    "close_time": ("doorclosingtime1s", "doorclosingtime01s", "doorclosingtime"),
    "close_cmd": ("closecommand", "closecmd"),
    "open_cmd": ("opencommand", "opencmd"),
    "dcsr": ("dcsr",),
    "dcsl": ("dcsl",),
    "dlsr": ("dlsr",),
    "dlsl": ("dlsl",),
    "door_opened": ("dooropened",),
    "door_locked": ("doorlocked",),
    "is_opening": ("doorisopening", "isopening", "opening"),
    "is_closing": ("doorisclosing", "isclosing", "closing"),
    "position": ("doorleafposition", "doorposition", "position"),
}

#: Columns a stream cannot do without (the rest degrade gracefully to NaN features).
_REQUIRED = ("datetime", "current", "position")

#: The brief's baseline feature set: operation, mid-stroke current, mean/peak current, mean
#: voltage, duration - each one relative to the **training fold's** per-operation baseline.
BASELINE_FEATURES: tuple[str, ...] = (
    "i_mean_cruise_rel",
    "i_mean_rel",
    "i_peak_rel",
    "v_mean_rel",
    "duration_rel",
)

#: The ladder's physics feature set (plan W4 door row): back-EMF/resistance residuals, energy,
#: peak position, phase currents and the current-vs-position template distance.
PHYSICS_FEATURES: tuple[str, ...] = (
    "i_mean_cruise_rel",
    "i_mean_cruise_z",
    "i_mean_opening_rel",
    "i_mean_closing_rel",
    "i_max_cruise_rel",
    "i_mean_rel",
    "i_peak_rel",
    "i_peak_frac",
    "i_std_cruise_rel",
    "charge_rel",
    "energy_rel",
    "v_mean_rel",
    "v_mean_cruise_rel",
    "emf_mean_cruise_rel",
    "r_proxy_rel",
    "i_per_speed_rel",
    "emf_per_speed_rel",
    "emf_residual_z",
    "speed_cruise_rel",
    "duration_rel",
    "duration_vs_cmd_rel",
    "dwt_coarse1_rel",
    "dwt_coarse2_rel",
    "prof_dist",
)


def _norm_key(name: Any) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


@dataclass
class DoorStream:
    """One continuous door stream: canonical columns plus the parsed timestamps.

    ``table`` carries the canonical float columns (:data:`CANONICAL_COLUMNS` keys minus
    ``datetime``); ``t`` is a ``datetime64[ms]`` Series aligned to it; ``source`` is the file the
    stream came from (``""`` for an in-memory frame) and ``extra_columns`` lists header columns
    that are not part of the documented 17 (kept as a warning, never used).
    """

    table: pd.DataFrame
    t: pd.Series
    source: str = ""
    extra_columns: tuple[str, ...] = ()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.table)


def load_stream(path_or_frame: Path | str | pd.DataFrame) -> DoorStream:
    """Read a door CSV (or accept a DataFrame) into a :class:`DoorStream`.

    Raises ``ValueError`` with a message naming the offending column for an empty file, a
    missing required column or an unparseable timestamp; unknown extra columns are ignored and
    recorded on the stream.
    """
    source = ""
    if isinstance(path_or_frame, pd.DataFrame):
        raw = path_or_frame.copy()
    else:
        source = str(path_or_frame)
        p = Path(path_or_frame)
        if not p.exists():
            raise FileNotFoundError(f"door stream not found: {p}")
        if p.stat().st_size == 0:
            raise ValueError(f"{p.name}: file is empty (expected a header row plus 50 Hz samples)")
        try:
            raw = pd.read_csv(p)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"{p.name}: file is empty (expected a header row plus 50 Hz samples)") from exc
    if raw.empty:
        raise ValueError(f"{Path(source).name or 'door stream'}: no data rows after the header")

    by_key = {_norm_key(c): c for c in raw.columns}
    found: dict[str, str] = {}
    for canon, aliases in CANONICAL_COLUMNS.items():
        for alias in aliases:
            if alias in by_key:
                found[canon] = by_key[alias]
                break
    missing = [c for c in _REQUIRED if c not in found]
    if missing:
        raise ValueError(
            f"{Path(source).name or 'door stream'}: missing required column(s) {missing}; "
            f"got {list(raw.columns)}"
        )
    used = set(found.values())
    extra = tuple(str(c) for c in raw.columns if c not in used)

    t = parse_door_timestamps(raw[found["datetime"]])
    if t.isna().any():
        bad = raw[found["datetime"]][t.isna()].iloc[0]
        raise ValueError(f"{Path(source).name or 'door stream'}: unparseable timestamp {bad!r}")
    if not t.is_monotonic_increasing:
        order = np.argsort(t.values, kind="stable")
        raw = raw.iloc[order]
        t = t.iloc[order]

    cols: dict[str, np.ndarray] = {}
    for canon in CANONICAL_COLUMNS:
        if canon == "datetime":
            continue
        if canon in found:
            cols[canon] = pd.to_numeric(raw[found[canon]], errors="coerce").to_numpy(dtype=float)
        else:
            cols[canon] = np.full(len(raw), np.nan, dtype=float)
    table = pd.DataFrame(cols).reset_index(drop=True)
    return DoorStream(table=table, t=t.reset_index(drop=True), source=source, extra_columns=extra)


# --------------------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------------------


def segment_by_gaps(stream: DoorStream, gap_seconds: float = GAP_SECONDS) -> list[tuple[int, int]]:
    """Half-open row spans ``[i0, i1)``, a new cycle wherever ``dt > gap_seconds``.

    The segment's end is the **last row of the span**, never trimmed and never extrapolated, so
    the emitted boundaries are timestamps that exist in the stream (millisecond exact).
    """
    t = stream.t
    if len(t) == 0:
        return []
    dt = t.diff().dt.total_seconds().to_numpy()
    cuts = [0, *(int(i) for i in np.flatnonzero(dt > float(gap_seconds))), len(t)]
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def segment_by_state(
    stream: DoorStream,
    *,
    min_rows: int = 25,
    hysteresis: float = 0.08,
    travel_fraction: float = 0.5,
) -> list[tuple[int, int]]:
    """Fallback segmenter: command edges plus a hysteretic position-sweep state machine.

    Used when a stream has no usable timestamp gaps (the Door Info Kit warns not to rely on any
    single column). A boundary is placed where

    * the open/close command word changes (a new command always starts a new cycle), or
    * the leaf position, having swept toward one extreme, reverses by more than ``hysteresis``
      of the observed travel - the state machine only accepts a reversal after the sweep has
      covered ``travel_fraction`` of the range, so encoder jitter mid-stroke cannot split a cycle.

    Spans shorter than ``min_rows`` (0.5 s at 50 Hz) are merged into the preceding cycle.
    """
    n = len(stream.table)
    if n == 0:
        return []
    pos = stream.table["position"].to_numpy(dtype=float)
    if np.all(~np.isfinite(pos)):
        return [(0, n)]
    pos = pd.Series(pos).ffill().bfill().to_numpy(dtype=float)
    span = float(np.nanmax(pos) - np.nanmin(pos))
    band = max(hysteresis * span, 1.0)
    need = max(travel_fraction * span, 1.0)

    cmd = np.zeros(n, dtype=float)
    for col, weight in (("open_cmd", 1.0), ("close_cmd", 2.0)):
        values = stream.table[col].to_numpy(dtype=float)
        cmd += np.nan_to_num(values, nan=0.0) * weight
    cmd_edge = np.zeros(n, dtype=bool)
    cmd_edge[1:] = cmd[1:] != cmd[:-1]

    cuts = [0]
    anchor = pos[0]
    extreme = pos[0]
    direction = 0
    for i in range(1, n):
        if cmd_edge[i]:
            cuts.append(i)
            anchor = extreme = pos[i]
            direction = 0
            continue
        p = pos[i]
        if direction == 0:
            if abs(p - anchor) > band:
                direction = 1 if p > anchor else -1
                extreme = p
            continue
        if (direction > 0 and p > extreme) or (direction < 0 and p < extreme):
            extreme = p
            continue
        reversed_by = (extreme - p) if direction > 0 else (p - extreme)
        swept = abs(extreme - anchor)
        if reversed_by > band and swept >= need:
            cuts.append(i)
            anchor = extreme = p
            direction = 0
    cuts.append(n)

    spans: list[tuple[int, int]] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b <= a:
            continue
        if spans and (b - a) < min_rows:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return spans


def segment(
    stream: DoorStream,
    *,
    method: str = "auto",
    gap_seconds: float = GAP_SECONDS,
    min_rows: int = 25,
) -> list[tuple[int, int]]:
    """Cut a stream into cycles.

    ``method``:

    * ``"gap"`` - timestamp gaps only (the baseline, exact on `Train.csv`);
    * ``"state"`` - the command/position state machine only;
    * ``"hybrid"`` - gaps, then the state machine **inside** each gap span (ablation: catches a
      boundary the recorder did not mark);
    * ``"auto"`` (default) - gaps when the stream has at least one, else the state machine.
    """
    key = str(method).lower()
    if key == "gap":
        return segment_by_gaps(stream, gap_seconds)
    if key == "state":
        return segment_by_state(stream, min_rows=min_rows)
    if key == "auto":
        spans = segment_by_gaps(stream, gap_seconds)
        return spans if len(spans) > 1 else segment_by_state(stream, min_rows=min_rows)
    if key == "hybrid":
        out: list[tuple[int, int]] = []
        for a, b in segment_by_gaps(stream, gap_seconds):
            sub = DoorStream(
                table=stream.table.iloc[a:b].reset_index(drop=True),
                t=stream.t.iloc[a:b].reset_index(drop=True),
                source=stream.source,
            )
            out.extend((a + i0, a + i1) for i0, i1 in segment_by_state(sub, min_rows=min_rows))
        return out
    raise ValueError(f"unknown door segmentation method {method!r} (gap|state|hybrid|auto)")


# --------------------------------------------------------------------------------------
# Per-cycle features
# --------------------------------------------------------------------------------------


@dataclass
class DoorFeats:
    """What :meth:`nebulax.ps3.door.DoorTask.featurise` returns.

    ``table`` is one row per predicted cycle (raw, *not* fold-normalised); ``profiles`` is the
    ``(n_cycles, N_PROFILE_BINS)`` current-versus-travel profile used for the template distance;
    ``cycles`` keeps the per-cycle arrays the explanation trace needs.
    """

    table: pd.DataFrame
    profiles: np.ndarray
    cycles: list[dict[str, Any]] = field(default_factory=list)
    spans: list[tuple[int, int]] = field(default_factory=list)
    source: str = ""
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.table)


def _safe(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _dwt_coarse_bands(current: np.ndarray) -> tuple[float, float]:
    """Log energy of the two coarsest db4 detail bands of the cycle's current [R66][R67].

    The wavelet door-monitoring papers read the obstruction signature off the coarse detail
    levels of the motor current; a 140-190 sample cycle supports about five db4 levels, so
    "coarsest two" is resolved per cycle rather than pinned to levels 8/9. PyWavelets is not
    installed, so this reuses the periodised Mallat cascade already in
    ``nebulax.models.physics.dwt_detail_energies``.
    """
    from nebulax.models.physics import dwt_detail_energies

    values = np.nan_to_num(np.asarray(current, dtype=float), nan=0.0)
    if values.size < 8:
        return float("nan"), float("nan")
    energies = np.asarray(dwt_detail_energies(values[None, :], wavelet="db4", n_levels=9))
    if energies.size == 0:
        return float("nan"), float("nan")
    band = energies[0]
    coarse1 = float(np.log1p(band[-1]))
    coarse2 = float(np.log1p(band[-2])) if band.size > 1 else coarse1
    return coarse1, coarse2


def _phase_masks(frac: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = PHASE_EDGES
    opening = frac < lo
    cruise = (frac >= lo) & (frac <= hi)
    closing = frac > hi
    if not cruise.any():  # degenerate (a 1-2 row span): treat everything as cruise
        cruise = np.ones_like(frac, dtype=bool)
    return opening, cruise, closing


def cycle_features(
    stream: DoorStream,
    spans: Sequence[tuple[int, int]] | None = None,
    *,
    keep_cycles: bool = True,
) -> DoorFeats:
    """Physics features for every cycle span, one row each.

    Feature names carry their stroke phase (``_opening`` / ``_cruise`` / ``_closing``) so that
    ``nebulax.models.boosting.LGBMThreeRegime`` can build its regime groups from the names, and
    cycle-level names carry none so they are shared by every regime.
    """
    spans = list(spans) if spans is not None else segment(stream)
    tbl = stream.table
    t_ms = stream.t.to_numpy(dtype="datetime64[ms]").astype("int64")
    cur = tbl["current"].to_numpy(dtype=float)
    vol = tbl["voltage"].to_numpy(dtype=float)
    emf = tbl["emf"].to_numpy(dtype=float)
    pos = tbl["position"].to_numpy(dtype=float)
    open_time = tbl["open_time"].to_numpy(dtype=float)
    close_time = tbl["close_time"].to_numpy(dtype=float)
    opening_flag = tbl["is_opening"].to_numpy(dtype=float)
    closing_flag = tbl["is_closing"].to_numpy(dtype=float)
    open_cmd = tbl["open_cmd"].to_numpy(dtype=float)
    dcs = np.nanmean(np.vstack([tbl["dcsr"].to_numpy(float), tbl["dcsl"].to_numpy(float)]), axis=0)
    dls = np.nanmean(np.vstack([tbl["dlsr"].to_numpy(float), tbl["dlsl"].to_numpy(float)]), axis=0)

    rows: list[dict[str, Any]] = []
    profiles = np.zeros((len(spans), N_PROFILE_BINS), dtype=float)
    cycles: list[dict[str, Any]] = []
    grid = np.linspace(0.0, 1.0, N_PROFILE_BINS)

    for k, (a, b) in enumerate(spans):
        sl = slice(a, b)
        n = b - a
        secs = (t_ms[sl] - t_ms[a]) / 1000.0
        duration = float(secs[-1]) if n else 0.0
        i = np.abs(cur[sl])
        p = pos[sl]
        v = vol[sl]
        e = emf[sl]

        p0, p1 = float(p[0]), float(p[-1])
        travel = p1 - p0
        if abs(travel) < 1e-9:  # no net movement: fall back to the extremes seen in the span
            travel = float(np.nanmax(p) - np.nanmin(p)) or 1.0
            frac = (p - np.nanmin(p)) / travel
        else:
            frac = (p - p0) / travel
        frac = np.clip(np.nan_to_num(frac, nan=0.0), 0.0, 1.0)

        if travel > 0:
            operation = "Open"
        elif travel < 0:
            operation = "Close"
        else:  # pragma: no cover - degenerate, resolved by the controller flags
            operation = "Open" if _safe(opening_flag[sl]) >= _safe(closing_flag[sl]) else "Close"
        if not np.isfinite(travel) or travel == 0:
            flag_open, flag_close = _safe(opening_flag[sl]), _safe(closing_flag[sl])
            if np.isfinite(flag_open) and np.isfinite(flag_close) and flag_open != flag_close:
                operation = "Open" if flag_open > flag_close else "Close"
            elif np.isfinite(_safe(open_cmd[sl])):
                operation = "Open" if _safe(open_cmd[sl]) >= 0.5 else "Close"

        op_mask, cr_mask, cl_mask = _phase_masks(frac)
        dt = np.diff(secs, prepend=secs[0]) if n > 1 else np.zeros(1)
        speed = np.abs(np.gradient(frac, secs)) if n > 2 and duration > 0 else np.zeros(n)
        speed_cruise = _safe(speed[cr_mask])
        i_cruise = _safe(i[cr_mask])
        v_cruise = _safe(v[cr_mask])
        e_cruise = _safe(e[cr_mask])
        eps = 1e-6

        row: dict[str, Any] = {
            "cycle": k,
            "start_ms": int(t_ms[a]),
            "end_ms": int(t_ms[b - 1]),
            "operation": operation,
            "op_code": 1.0 if operation == "Open" else 0.0,
            "n_rows": float(n),
            "duration": duration,
            "travel": abs(float(travel)),
            # cycle-level current
            "i_mean": _safe(i),
            "i_rms": float(np.sqrt(np.nanmean(np.square(i)))) if n else float("nan"),
            "i_std": float(np.nanstd(i)) if n else float("nan"),
            "i_med": float(np.nanmedian(i)) if n else float("nan"),
            "i_peak": float(np.nanmax(i)) if n else float("nan"),
            "i_peak_frac": float(frac[int(np.nanargmax(i))]) if n else float("nan"),
            "charge": float(np.nansum(i * dt)),
            "energy": float(np.nansum(i * v * 10.0 * dt)),
            "v_mean": _safe(v),
            "emf_mean": _safe(e),
            # phase currents (names feed LGBMThreeRegime's regime grouping)
            "i_mean_opening": _safe(i[op_mask]),
            "i_max_opening": float(np.nanmax(i[op_mask])) if op_mask.any() else float("nan"),
            "i_mean_cruise": i_cruise,
            "i_max_cruise": float(np.nanmax(i[cr_mask])) if cr_mask.any() else float("nan"),
            "i_std_cruise": float(np.nanstd(i[cr_mask])) if cr_mask.any() else float("nan"),
            "i_mean_closing": _safe(i[cl_mask]),
            "i_max_closing": float(np.nanmax(i[cl_mask])) if cl_mask.any() else float("nan"),
            "v_mean_cruise": v_cruise,
            "emf_mean_cruise": e_cruise,
            "dur_opening": float(dt[op_mask].sum()) if op_mask.any() else 0.0,
            "dur_cruise": float(dt[cr_mask].sum()) if cr_mask.any() else 0.0,
            "dur_closing": float(dt[cl_mask].sum()) if cl_mask.any() else 0.0,
            # physics residuals
            "speed_cruise": speed_cruise,
            "speed_mean": (1.0 / duration) if duration > 0 else float("nan"),
            "r_proxy": (10.0 * v_cruise - e_cruise) / max(i_cruise, eps),
            "i_per_speed": i_cruise / max(speed_cruise, eps),
            "emf_per_speed": e_cruise / max(speed_cruise, eps),
            "sw_dcs_frac": _safe(dcs[sl]),
            "sw_dls_frac": _safe(dls[sl]),
        }
        # duration against the controller's own commanded time register (0.1 s units), the
        # addendum's "cycle duration against the commanded open/close time" feature [R64].
        # The register reads 0 on some `Test.csv` rows (not latched yet), and a mean over the
        # cycle would then divide by almost nothing - so take the median of the **valid**
        # (non-zero, finite) readings and leave the feature missing when there are none.
        register = (open_time if operation == "Open" else close_time)[sl]
        valid = register[np.isfinite(register) & (register > 0)]
        commanded = float(np.median(valid)) * 0.1 if valid.size else 0.0
        row["duration_vs_cmd"] = duration / commanded if commanded > 0 else float("nan")
        coarse = _dwt_coarse_bands(i)
        row["dwt_coarse1"], row["dwt_coarse2"] = coarse[0], coarse[1]
        rows.append(row)

        order = np.argsort(frac, kind="stable")
        profiles[k] = np.interp(grid, frac[order], i[order])
        if keep_cycles:
            cycles.append(
                {
                    "cycle": k,
                    "t_ms": t_ms[sl].copy(),
                    "secs": secs,
                    "current": i,
                    "position": p,
                    "frac": frac,
                    "voltage": v,
                    "emf": e,
                    "operation": operation,
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(columns=["cycle", "start_ms", "end_ms", "operation", "op_code"])
    warnings = [f"ignored unknown column {c!r}" for c in stream.extra_columns]
    return DoorFeats(
        table=table,
        profiles=profiles,
        cycles=cycles,
        spans=list(spans),
        source=stream.source,
        warnings=warnings,
    )


# --------------------------------------------------------------------------------------
# The fold-local baseline
# --------------------------------------------------------------------------------------

#: Raw columns that get a ``_rel`` (ratio to the fold baseline) and ``_z`` (robust z) companion.
_BASELINE_COLS: tuple[str, ...] = (
    "i_mean",
    "i_rms",
    "i_std",
    "i_med",
    "i_peak",
    "charge",
    "energy",
    "v_mean",
    "emf_mean",
    "i_mean_opening",
    "i_max_opening",
    "i_mean_cruise",
    "i_max_cruise",
    "i_std_cruise",
    "i_mean_closing",
    "i_max_closing",
    "v_mean_cruise",
    "emf_mean_cruise",
    "speed_cruise",
    "r_proxy",
    "i_per_speed",
    "emf_per_speed",
    "duration",
    "travel",
    "duration_vs_cmd",
    "dwt_coarse1",
    "dwt_coarse2",
)

_PASS_THROUGH: tuple[str, ...] = (
    "op_code",
    "i_peak_frac",
    "dur_opening",
    "dur_cruise",
    "dur_closing",
    "sw_dcs_frac",
    "sw_dls_frac",
)


class FoldBaseline:
    """Per-operation normal baseline, **fitted inside one training fold**.

    For each operation (``Open`` / ``Close``) it stores a robust centre and scale for every
    column in :data:`_BASELINE_COLS`, a mean current-versus-travel template, and the median
    template distance of the fitting rows. :meth:`transform` turns a raw feature table into the
    fold-relative one the classifiers see (``<col>_rel`` = value / centre, ``<col>_z`` =
    robust z, ``prof_dist`` = template distance / scale, ``emf_residual_z`` = the back-EMF
    residual ``emf - k*speed`` with ``k`` least-squares fitted on the same rows).

    ``mode``:

    * ``"fold"`` (default) - centres come from the training fold; when labels are supplied only
      its ``Normal`` rows are used, otherwise the median over the fold (robust while fewer than
      half the cycles are abnormal - 27 % in `Train.csv`);
    * ``"batch"`` - the declared ablation: the same statistics are re-estimated from whatever
      batch is being predicted, unlabelled, so nothing crosses from the training fold. Mirrored
      independently inside each held block.
    """

    def __init__(self, mode: str = "fold") -> None:
        if mode not in ("fold", "batch"):
            raise ValueError(f"FoldBaseline mode must be 'fold' or 'batch', got {mode!r}")
        self.mode = mode
        self.stats_: dict[str, dict[str, tuple[float, float]]] = {}
        self.templates_: dict[str, np.ndarray] = {}
        self.template_scale_: dict[str, float] = {}
        self.emf_k_: dict[str, tuple[float, float]] = {}
        self.global_: dict[str, tuple[float, float]] = {}
        self.feature_names_: list[str] = []

    # -- fitting ------------------------------------------------------------------------
    def fit(self, feats: DoorFeats, y: Sequence[str] | None = None) -> "FoldBaseline":
        """Fit the baseline on a training fold's cycles (`y` = their labels, when known)."""
        table = feats.table.reset_index(drop=True)
        prof = np.asarray(feats.profiles, dtype=float)
        if y is not None:
            labels = np.asarray([str(v) for v in y])
            if labels.size != len(table):
                raise ValueError(f"FoldBaseline.fit: {len(table)} rows but {labels.size} labels")
            keep = labels == DOOR_LABELS[0]
            if keep.sum() < 3:  # too few normals to build a template - use every row
                keep = np.ones(len(table), dtype=bool)
        else:
            keep = np.ones(len(table), dtype=bool)

        self.stats_, self.templates_, self.template_scale_, self.emf_k_ = {}, {}, {}, {}
        self.resid_scale_: dict[str, float] = {}
        for op in ("Open", "Close"):
            mask = keep & (table["operation"].to_numpy() == op)
            if mask.sum() == 0:
                continue
            sub = table.loc[mask]
            self._fit_one(op, sub, prof[np.flatnonzero(mask)])
        # a catch-all group so an unseen operation still normalises against something
        if keep.any():
            self._fit_one("*", table.loc[keep], prof[np.flatnonzero(keep)])
        self.feature_names_ = list(self.transform(feats).columns)
        return self

    def _fit_one(self, op: str, sub: pd.DataFrame, prof: np.ndarray) -> None:
        stats: dict[str, tuple[float, float]] = {}
        for col in _BASELINE_COLS:
            if col not in sub.columns:
                continue
            values = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                stats[col] = (1.0, 1.0)
                continue
            centre = float(np.median(values))
            mad = float(np.median(np.abs(values - centre)))
            scale = 1.4826 * mad if mad > 0 else (abs(centre) * 0.05 if centre else 1.0)
            stats[col] = (centre if abs(centre) > 1e-12 else 1e-9, scale if scale > 0 else 1.0)
        self.stats_[op] = stats
        template = prof.mean(axis=0) if prof.size else np.zeros(N_PROFILE_BINS)
        self.templates_[op] = template
        if prof.size:
            dists = np.linalg.norm(prof - template, axis=1)
            scale = float(np.median(dists))
            if scale <= 0:
                scale = float(np.mean(dists)) or 1.0
        else:  # pragma: no cover
            scale = 1.0
        self.template_scale_[op] = scale
        speed = pd.to_numeric(sub.get("speed_cruise", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        emf = pd.to_numeric(sub.get("emf_mean_cruise", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        ok = np.isfinite(speed) & np.isfinite(emf)
        if ok.sum() >= 3 and float(np.ptp(speed[ok])) > 1e-9:
            k, c = (float(v) for v in np.polyfit(speed[ok], emf[ok], 1))
        else:
            k, c = 0.0, (float(np.mean(emf[ok])) if ok.any() else 0.0)
        self.emf_k_[op] = (k, c)
        resid = emf[ok] - (k * speed[ok] + c) if ok.any() else np.zeros(1)
        rscale = 1.4826 * float(np.median(np.abs(resid)))
        self.resid_scale_[op] = rscale if rscale > 0 else (float(np.std(resid)) or 1.0)

    # -- transform ----------------------------------------------------------------------
    def transform(self, feats: DoorFeats) -> pd.DataFrame:
        """Fold-relative features, one row per cycle, aligned to ``feats.table``."""
        if self.mode == "batch":
            batch = FoldBaseline(mode="fold")
            batch.fit(feats, y=None)
            return batch._transform_with(feats)
        if not self.stats_:
            raise RuntimeError("FoldBaseline.transform called before fit()")
        return self._transform_with(feats)

    def _transform_with(self, feats: DoorFeats) -> pd.DataFrame:
        table = feats.table.reset_index(drop=True)
        prof = np.asarray(feats.profiles, dtype=float)
        out: dict[str, np.ndarray] = {}
        ops = table["operation"].to_numpy() if "operation" in table.columns else np.array(["*"] * len(table))
        groups = [op if op in self.stats_ else "*" for op in ops]
        for col in _BASELINE_COLS:
            if col not in table.columns:
                continue
            values = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
            rel = np.empty(len(table), dtype=float)
            zed = np.empty(len(table), dtype=float)
            for idx, group in enumerate(groups):
                centre, scale = self.stats_.get(group, {}).get(col, (1.0, 1.0))
                rel[idx] = values[idx] / centre if centre else np.nan
                zed[idx] = (values[idx] - centre) / scale if scale else 0.0
            out[f"{col}_rel"] = rel
            out[f"{col}_z"] = zed
        for col in _PASS_THROUGH:
            if col in table.columns:
                out[col] = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
        dist = np.zeros(len(table), dtype=float)
        resid = np.zeros(len(table), dtype=float)
        speed = pd.to_numeric(table.get("speed_cruise", pd.Series(np.nan, index=table.index)), errors="coerce").to_numpy(float)
        emf = pd.to_numeric(table.get("emf_mean_cruise", pd.Series(np.nan, index=table.index)), errors="coerce").to_numpy(float)
        for idx, group in enumerate(groups):
            template = self.templates_.get(group)
            scale = self.template_scale_.get(group, 1.0) or 1.0
            if template is None or prof.size == 0:
                dist[idx] = 0.0
            else:
                dist[idx] = float(np.linalg.norm(prof[idx] - template)) / scale
            k, c = self.emf_k_.get(group, (0.0, 0.0))
            rscale = self.resid_scale_.get(group, 1.0) or 1.0
            value = emf[idx] - (k * speed[idx] + c)
            resid[idx] = value / rscale if np.isfinite(value) else 0.0
        out["prof_dist"] = dist
        out["emf_residual_z"] = resid
        frame = pd.DataFrame(out, index=table.index).replace([np.inf, -np.inf], np.nan)
        # A missing ratio means "no evidence", which is the baseline value 1.0 - filling it with
        # 0.0 would read as a 100 % collapse and dominate a linear model. Ratios are then clipped
        # to [0, RATIO_CLIP] and robust z-scores to +-Z_CLIP so that one out-of-distribution
        # channel (an unlatched controller register, a different door's travel range) cannot
        # swing a fitted direction by hundreds of standard deviations.
        for col in frame.columns:
            if col.endswith("_rel"):
                frame[col] = frame[col].fillna(1.0).clip(0.0, RATIO_CLIP)
            elif col.endswith("_z"):
                frame[col] = frame[col].fillna(0.0).clip(-Z_CLIP, Z_CLIP)
            else:
                frame[col] = frame[col].fillna(0.0)
        return frame

    def fit_transform(self, feats: DoorFeats, y: Sequence[str] | None = None) -> pd.DataFrame:
        return self.fit(feats, y).transform(feats)


# --------------------------------------------------------------------------------------
# Labelling predicted segments from the answer file (training only)
# --------------------------------------------------------------------------------------


def label_segments(
    feats: DoorFeats,
    answer: pd.DataFrame,
    *,
    min_iou: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Attach ``Train_Segments_Answer.csv`` labels to segmented cycles by best IoU.

    Returns ``(labels, matched)``: ``labels[i]`` is the status of the answer segment with the
    highest IoU against cycle ``i`` (``""`` when nothing overlaps by at least ``min_iou``) and
    ``matched`` is the boolean mask of cycles that found one. Unmatched cycles are dropped from
    training rather than guessed at.
    """
    from nebulax.ps3.scoring import iou as _iou

    starts = feats.table["start_ms"].to_numpy(dtype="int64")
    ends = feats.table["end_ms"].to_numpy(dtype="int64")
    a_start = answer["t_start"].to_numpy(dtype="datetime64[ms]").astype("int64")
    a_end = answer["t_end"].to_numpy(dtype="datetime64[ms]").astype("int64")
    a_label = answer["label"].astype(str).to_numpy()

    labels = np.full(len(starts), "", dtype=object)
    matched = np.zeros(len(starts), dtype=bool)
    for i, (s, e) in enumerate(zip(starts, ends)):
        best, best_j = 0.0, -1
        for j, (ts, te) in enumerate(zip(a_start, a_end)):
            value = _iou((s, e), (ts, te))
            if value > best:
                best, best_j = value, j
        if best_j >= 0 and best >= min_iou:
            labels[i] = a_label[best_j]
            matched[i] = True
    return labels, matched
