"""Physics-grounded detectors and the three wrapper rows of ``configs/model_ladder.yaml``.

Six ladder rows live here (family / input_kind / task exactly as the yaml has them):

===============================  ================  ================  =====
name                             family            input_kind        task
===============================  ================  ================  =====
``duty_cycle_ratio_cusum``       physics_feature   cycle_features    ad
``kurtogram_envelope_bpfo``      physics_feature   raw_window        ad
``peer_delta_temperature``       physics_feature   window_stats      ad
``thermal_residual_model``       physics_residual  window_stats      ad
``dwt_w8w9_startpeak``           time_frequency    raw_window        ad
``transition_mask``              exclusion_rule    raw_window        cpd
``peer_normalisation``           wrapper           cycle_features    ad
``peer_normalisation_shared``    wrapper           window_stats      ad
``k_of_n_corroboration``         wrapper           window_stats      ad
===============================  ================  ================  =====

Column addressing
-----------------
``nebulax.bench.runner`` hands every model that asks for it the **row context** of the slice
it is fitting or scoring (``nebulax.bench.base.CONTEXT_KEYS``): ``feature_names`` (the
columns of a 2-D ``X``, the channels of a 3-D one), ``series`` (the component id per row) and
``unit`` (the train / bearing / rig). Labels are never passed. Physics rows need a *named*
quantity (``idle_run_ratio``, ``T_box``, the door current channel), so every such model
declares ``feature_names`` on ``_fit`` and takes the same trio of constructor keywords:

``feature_names``  optional ``list[str]`` fallback for **direct** use (tests, notebooks);
                   the runner's names win when it supplies them,
``<x>_index``      an explicit integer column index, which always wins,
``<x>``            a column *name*, resolved against the names by :func:`match_column`
                   (exact hit, then ``<signal>_mean...``, then any other suffix chain, then
                   a substring hit - so "the TP2 column" is ``TP2_mean_mean``, the level,
                   rather than ``TP2_mean_std``, its spread).

A named column that is genuinely **absent** from a known feature table raises ``ValueError``
naming the column and the table rather than silently landing on column 0. With no names and
no index at all the documented fallback (column 0, or the highest-variance column) applies,
and the resolved index is recorded in ``self.feature_index_`` for audit.

Peer grouping
-------------
Peers are *rows that share a unit and a timestamp (within ``time_tol_s``) but a different
series*: the sibling doors of one dwell, the eight axle boxes of one window
(:func:`peer_group_ids`). Grouping by timestamp alone pooled ten trains and two runs into
one "peer" group on the shipped bearing table and chained neighbouring cycles together, so
both ``unit`` and ``series`` reach the model through the row-context channel and a group now
holds at most one row per series.

The R151 same-side rule reads the side straight off the component id
(:func:`series_sides`): the L/R letter of ``door_L1``..``door_R4`` / ``axlebox_1L``. That is
a property of the component, so - unlike the row-position parity it replaces - it does not
change when equal-timestamp rows are reordered.

Sampling rate
-------------
``kurtogram_envelope_bpfo`` also declares ``fs_hz``: the rate the runner's loader actually
*delivered* the raw samples at (Ottawa's ``fs_hz_out = 10,500 Hz`` after ``decimate=4``, not
the sensor's 42 kHz), so the envelope-spectrum frequency axis is right instead of 4x off.

Peer statistic and box geometry
-------------------------------
``peer_normalisation_shared`` pools its same-side peers by **median**
(:data:`PEER_STATISTICS`), as its ladder row requires, and ``thermal_residual_model`` takes
the **specific** adjacent box (:func:`bogie_partner`, ``axlebox_1L`` <-> ``axlebox_2L``) and
the corresponding opposite box (``axlebox_3R`` -> ``axlebox_3L``), never a whole-side average.

No model in this module ever reads a label.
"""

from __future__ import annotations

import math
import re
from typing import Any, ClassVar, Final, Sequence

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from nebulax.bench import metrics as M
from nebulax.bench.base import AnomalyDetector, BaseModel
from nebulax.bench.registry import build, register
from nebulax.features.vibration import BearingGeometry, envelope_spectrum_feats

__all__ = [
    "DutyCycleRatioCusum",
    "KurtogramEnvelopeBpfo",
    "PeerDeltaTemperature",
    "ThermalResidualModel",
    "DwtW8W9StartPeak",
    "TransitionMask",
    "PeerNormalisation",
    "PeerNormalisationShared",
    "KOfNCorroboration",
    "times_seconds",
    "resolve_feature_index",
    "resolve_feature_indices",
    "match_column",
    "base_signal",
    "component_id",
    "series_sides",
    "component_positions",
    "bogie_partner",
    "peer_group_ids",
    "peer_delta",
    "PEER_STATISTICS",
    "robust_baseline",
    "sanitise",
]

#: Floor for every robust scale, so a constant training channel can never divide by zero.
_SCALE_FLOOR: Final[float] = 1e-9
#: ``1.4826 * MAD`` is the normal-consistent robust sigma.
_MAD_K: Final[float] = 1.4826


# ======================================================================================
# shared helpers (also imported by nebulax.models.drift)
# ======================================================================================


def times_seconds(t: Any, n: int) -> np.ndarray:
    """``t`` as float epoch seconds of length ``n``; ``arange(n)`` when unusable.

    A model must never crash because the runner passed ``None`` (unit tests, ``fit_score``)
    or a column of ``NaT``: the row index is then a perfectly good monotone clock.
    """
    if t is None:
        return np.arange(n, dtype=np.float64)
    s = np.asarray(M.to_epoch_seconds(np.asarray(t).reshape(-1)), dtype=np.float64)
    if s.size != n or not np.isfinite(s).any():
        return np.arange(n, dtype=np.float64)
    if not np.isfinite(s).all():  # patch NaT rows onto the surrounding clock
        idx = np.arange(n, dtype=np.float64)
        good = np.isfinite(s)
        s = np.interp(idx, idx[good], s[good])
    return s


def sanitise(x: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Finite float64 copy of ``x``: NaN/inf replaced by ``fill`` (0 by default)."""
    out = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(out), out, float(fill))


def robust_baseline(x: np.ndarray) -> tuple[float, float]:
    """``(median, 1.4826 * MAD)`` of the finite entries, scale floored away from zero.

    Falls back to the standard deviation when the MAD is degenerate (more than half the
    training values identical - common for duty fractions and digital channels).
    """
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    med = float(np.median(v))
    scale = float(_MAD_K * np.median(np.abs(v - med)))
    if not np.isfinite(scale) or scale <= _SCALE_FLOOR:
        scale = float(np.std(v))
    if not np.isfinite(scale) or scale <= _SCALE_FLOOR:
        scale = 1.0
    return med, max(scale, _SCALE_FLOOR)


def _columns_of(X: np.ndarray) -> int:
    return int(X.shape[-1])


#: Statistic suffixes :mod:`nebulax.features.stats` appends to a base signal name. Stripping
#: them turns ``"T_box_wayside_mean"`` into the *sensor* ``"T_box_wayside"``, which is what the
#: same-sensor exclusion in :class:`ThermalResidualModel` reasons about.
_STAT_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        "mean",
        "std",
        "min",
        "max",
        "slope",
        "rms",
        "kurtosis",
        "crest",
        "skew",
        "p2p",
        "median",
        "ac",
        "energy",
        "sum",
        "count",
        "frac",
        "shape",
        "impulse",
        "factor",
    }
)


def base_signal(name: str) -> str:
    """``"T_box_wayside_mean" -> "T_box_wayside"``: the column name minus its stat suffixes."""
    parts = str(name).split("_")
    while len(parts) > 1 and parts[-1].lower() in _STAT_SUFFIXES:
        parts.pop()
    return "_".join(parts)


def _column_rank(column: str, want: str) -> float | None:
    """How good a match ``column`` is for the signal ``want`` (lower = better, None = no match).

    The tables the runner builds name a column ``<signal>_<stat>[_<stat>]`` (MetroPT-3 window
    stats aggregate an aggregate: ``TP2_mean_mean``), so "the TP2 column" must prefer the
    *level* of TP2 over its spread: an exact hit beats a ``<signal>_mean...`` column, which
    beats any other suffix chain, which beats a bare substring hit.
    """
    a, b = str(column).lower(), str(want).lower()
    if a == b:
        return 0.0
    if a.startswith(b + "_"):
        suffix = a[len(b) + 1 :].split("_")
        return 1.0 + sum(0.0 if tok in ("mean", "") else 1.0 for tok in suffix) + 0.001 * len(suffix)
    if b in a:
        return 10.0 + 0.001 * len(a)
    return None


def match_column(want: str, names: Sequence[str]) -> int | None:
    """Index of the best column for the signal ``want``, or ``None`` when nothing matches."""
    ranked = [(r, i) for i, nm in enumerate(names) if (r := _column_rank(nm, want)) is not None]
    return min(ranked)[1] if ranked else None


def _preview(names: Sequence[str], k: int = 12) -> str:
    head = ", ".join(str(n) for n in list(names)[:k])
    return head + (", ..." if len(names) > k else "")


def resolve_feature_index(
    feature: str | int | None,
    *,
    index: int | None = None,
    feature_names: Sequence[str] | None = None,
    n_features: int,
    patterns: Sequence[str] = (),
    aliases: Sequence[str] = (),
    required: bool = True,
    what: str = "feature",
) -> int:
    """Turn a name / index / pattern list into one validated column index.

    Resolution order: explicit ``index`` -> integer ``feature`` -> ``feature`` (and any
    ``aliases``) matched against ``feature_names`` by :func:`match_column` -> the first
    ``patterns`` regex that matches a name.

    ``feature_names`` is the **table's own column list**, handed to ``_fit`` by the runner
    (``nebulax.bench.base.CONTEXT_KEYS``); a named column that is genuinely absent from it
    raises ``ValueError`` naming the column and the table, instead of silently resolving to
    column 0 (which is what made ``idle_run_ratio`` read ``t_loaded`` on the shipped
    pneumatic cycle table). With **no** names at all the documented fallback is column 0.
    """
    if index is not None:
        i = int(index)
        if not -n_features <= i < n_features:
            raise ValueError(f"{what}: column index {i} out of range for {n_features} columns")
        return i % n_features
    if isinstance(feature, (int, np.integer)) and not isinstance(feature, bool):
        return resolve_feature_index(None, index=int(feature), n_features=n_features, what=what)

    names = [str(s) for s in feature_names] if feature_names is not None else []
    if names and len(names) == n_features:
        wanted = [feature, *aliases] if isinstance(feature, str) else list(aliases)
        for w in wanted:
            j = match_column(str(w), names)
            if j is not None:
                return j
        for pat in patterns:
            rx = re.compile(pat, re.IGNORECASE)
            hit = [i for i, nm in enumerate(names) if rx.search(nm)]
            if hit:
                return hit[0]
        if required and (isinstance(feature, str) or aliases):
            tried = " / ".join(str(w) for w in wanted)
            raise ValueError(
                f"{what}: no column for {tried!r} in this feature table "
                f"({n_features} columns: {_preview(names)}). Pass an explicit column index, "
                f"or run this row on a table that carries it."
            )
    return 0


def resolve_feature_indices(
    features: Sequence[str | int] | None,
    *,
    indices: Sequence[int] | None = None,
    feature_names: Sequence[str] | None = None,
    n_features: int,
    patterns: Sequence[str] = (),
    aliases: dict[str, Sequence[str]] | None = None,
    default_k: int = 5,
    variance: np.ndarray | None = None,
    on_missing: str = "error",
    what: str = "features",
) -> list[int]:
    """Several columns, same rules as :func:`resolve_feature_index`.

    ``on_missing="error"`` (the default) raises when a *named* member is absent from a known
    ``feature_names``; ``"drop"`` votes on the members that did resolve. With nothing
    specified: every column matching ``patterns`` (when names are known), else the
    ``default_k`` columns with the largest training variance.
    """
    if on_missing not in ("error", "drop"):
        raise ValueError(f"{what}: on_missing must be error|drop, got {on_missing!r}")
    if indices is not None:
        return [int(i) % n_features for i in indices]
    names = [str(s) for s in feature_names] if feature_names is not None else []
    known = bool(names) and len(names) == n_features
    alias_map = dict(aliases or {})
    if features is not None:
        out: list[int] = []
        missing: list[str] = []
        for f in features:
            if isinstance(f, (int, np.integer)) and not isinstance(f, bool):
                j: int | None = int(f) % n_features
            elif known:
                j = None
                for w in (str(f), *alias_map.get(str(f), ())):
                    j = match_column(w, names)
                    if j is not None:
                        break
                if j is None:
                    missing.append(str(f))
            else:  # no names: the historical positional fallback, index 0 for a bare name
                j = resolve_feature_index(f, n_features=n_features, required=False, what=what)
            if j is not None and j not in out:
                out.append(j)
        if missing and on_missing == "error":
            raise ValueError(
                f"{what}: {', '.join(missing)} not present in this feature table "
                f"({n_features} columns: {_preview(names)}). Pass feature_indices=[...], "
                f"name columns this table has, or set on_missing='drop'."
            )
        return out or [0]

    if known and patterns:
        hit = []
        for pat in patterns:
            rx = re.compile(pat, re.IGNORECASE)
            hit.extend(i for i, nm in enumerate(names) if rx.search(nm) and i not in hit)
        if hit:
            return hit
    if n_features <= default_k or variance is None:
        return list(range(n_features))
    v = sanitise(variance, 0.0)
    return sorted(np.argsort(v)[::-1][:default_k].tolist())


def _order_by_time(t_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(order, inverse)`` - the runner does not promise time-sorted rows."""
    order = np.argsort(t_s, kind="stable")
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return order, inv


def rolling_time_stat(
    x: np.ndarray, t_s: np.ndarray, window_s: float, *, stat: str = "median", min_periods: int = 1
) -> np.ndarray:
    """Backward-looking rolling statistic over a ``window_s`` **time** window.

    Rows are sorted by time internally and the result is returned in the caller's row order.
    NaN values are skipped (pandas semantics); an all-NaN window yields the series median.
    """
    n = x.size
    order, inv = _order_by_time(t_s)
    ts = t_s[order]
    # a strictly increasing integer-ns index keeps pandas happy with duplicate timestamps
    idx = pd.to_datetime(np.round(ts * 1e9).astype("int64"), unit="ns")
    s = pd.Series(np.asarray(x, dtype=np.float64)[order], index=idx)
    roll = s.rolling(f"{max(float(window_s), 1e-3)}s", min_periods=max(int(min_periods), 1))
    out = getattr(roll, stat)().to_numpy(dtype=np.float64)
    fill = float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0
    out = np.where(np.isfinite(out), out, fill)
    return out[inv] if out.size == n else np.full(n, fill)


def cusum(d: np.ndarray, k: float = 0.5) -> np.ndarray:
    """One-sided CUSUM ``S_i = max(0, S_{i-1} + d_i - k)`` over a standardised deviation."""
    out = np.empty(d.size, dtype=np.float64)
    s = 0.0
    kk = float(k)
    for i, v in enumerate(d):
        s = max(0.0, s + (float(v) if math.isfinite(float(v)) else 0.0) - kk)
        out[i] = s
    return out


def component_id(series: Any) -> np.ndarray:
    """The component part of a ``BenchData.series`` id: ``"door_0000/door_L1" -> "door_L1"``."""
    s = np.asarray(series, dtype=object).reshape(-1)
    return np.array([str(v).rsplit("/", 1)[-1] for v in s], dtype=object)


#: A side marker is a bare ``L``/``R`` inside the *component* part of a series id - the ``L``
#: of ``door_L1``..``door_L4`` or the trailing ``L`` of ``axlebox_1L`` - never a letter that is
#: part of a word (the ``R`` of ``door``, the ``L`` of ``axlebox``).
_SIDE_RX: Final[re.Pattern[str]] = re.compile(r"(?<![A-Za-z])([LR])(?![A-Za-z])")


def series_sides(series: Any) -> np.ndarray:
    """``'L'`` / ``'R'`` / ``''`` per row, from the L/R letter of the component id (R151).

    This is the *defensible* same-side rule: it is a property of the component's identity, so
    it cannot change when equal-timestamp rows are reordered (the row-position parity it
    replaces did). Tables whose series ids carry no side letter (MetroPT's single APU, an
    Ottawa recording, a Cranfield rig run) get ``''`` and therefore all sit on one side.
    """
    comp = component_id(series)
    out = np.empty(comp.size, dtype=object)
    for i, name in enumerate(comp):
        hit = _SIDE_RX.findall(str(name).upper())
        out[i] = hit[-1] if hit else ""
    return out


#: The last run of digits in a component id is its position: ``axlebox_3R`` -> 3,
#: ``door_L1`` -> 1. A component with no digits at all gets 0 (one nominal position).
_INDEX_RX: Final[re.Pattern[str]] = re.compile(r"(\d+)(?!.*\d)")


def component_positions(series: Any | None, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``(side, index)`` per row: the L/R letter and the position number of the component.

    ``"bearing_0000/axlebox_3R" -> ("R", 3)``, ``"door_0000/door_L1" -> ("L", 1)``. A table
    whose series ids carry neither (MetroPT's single APU, an Ottawa recording) gets
    ``("", 0)`` for every row, which makes every position-aware rule a documented no-op
    rather than a wrong answer.
    """
    if series is None:
        return np.full(n, "", dtype=object), np.zeros(n, dtype=np.int64)
    side = np.array([str(v) for v in series_sides(series)], dtype=object)
    comp = component_id(series)
    idx = np.array(
        [int(m.group(1)) if (m := _INDEX_RX.search(str(c))) else 0 for c in comp], dtype=np.int64
    )
    return side, idx


def bogie_partner(idx: np.ndarray) -> np.ndarray:
    """The other axle of the same bogie: ``1<->2``, ``3<->4``, ... (R110's "adjacent" box).

    A four-wheelset car is two bogies of two wheelsets, so the *adjacent* box on one side is
    the other box of that bogie - the one sharing its suspension, its load path and its
    thermal environment - not "some other box on this side". Position ``0`` (a component with
    no position number) is its own partner, which the caller then discards as a self-match.
    """
    i = np.asarray(idx, dtype=np.int64)
    return np.where(i <= 0, i, np.where(i % 2 == 1, i + 1, i - 1))


def peer_group_ids(
    t_s: np.ndarray,
    tol_s: float,
    *,
    unit: Any | None = None,
    series: Any | None = None,
) -> np.ndarray:
    """Group id per row: **one unit's** rows at one timestamp, at most one row per series.

    Peers are the *other components of the same train at the same window*: rows that share a
    ``unit`` and a timestamp (within ``tol_s``) but carry a different ``series``. The runner
    hands both arrays to any model that declares them (``nebulax.bench.base.CONTEXT_KEYS``),
    so the grouping no longer has to guess from the timeline alone.

    Two properties matter and are both enforced here:

    * groups never span units - the shipped bearing table interleaves 10 trains x 2 runs on
      one clock, and a timestamp-only rule pooled all 160 boxes of a window into one group;
    * a group holds **at most one row per series**, so a chain of rows ``tol_s`` apart cannot
      walk across cycles: a new group starts as soon as a series repeats or the row leaves
      the anchor's ``tol_s`` window.

    With ``unit``/``series`` omitted the old timestamp-chaining behaviour is kept (direct use
    in tests, tables with a single component).
    """
    n = t_s.size
    u = np.asarray(unit, dtype=object).reshape(-1) if unit is not None else np.zeros(n, dtype=np.int8)
    s = np.asarray(series, dtype=object).reshape(-1) if series is not None else None
    if u.size != n:
        u = np.zeros(n, dtype=np.int8)
    if s is not None and s.size != n:
        s = None
    tol = float(tol_s)
    order = np.lexsort((t_s, np.asarray([str(v) for v in u], dtype=object)))
    gid = np.empty(n, dtype=np.int64)
    g = -1
    anchor = -np.inf
    cur_unit: Any = object()
    seen: set[Any] = set()
    for i in order:
        key = str(u[i])
        new = key != cur_unit or t_s[i] - anchor > tol or (s is not None and s[i] in seen)
        if new:
            g += 1
            anchor = t_s[i]
            cur_unit = key
            seen = set()
        if s is not None:
            seen.add(s[i])
        gid[i] = g
    return gid


def check_peers_exist(gid: np.ndarray, series: Any, *, unit: Any, what: str, require: bool) -> int:
    """Rows that actually have a peer; raise when the row context says there are none.

    A peer-differential row whose every group holds one member emits a **constant** score,
    which no metric in the protocol can rank - and the reason is always a property of the
    slice, not of the model. Better to say that loudly at fit time than to ship a flat column
    into the leaderboard. Only checked when the runner actually supplied ``series``.

    **What the shipped tables really carry** (measured on the uncached fleet, 16 Sep 2026, not
    assumed): ``sim``/``bearing`` has eight axle boxes per train (``axlebox_1L..4L``,
    ``axlebox_1R..4R``) and ``sim``/``door`` has **two** door series per train over 10 trains
    and 20 runs - so both have peers, and the door row fits on the real temporal normal-only
    slice with ~95 k peer-supported rows. The case this guard is really for is a *slice* with
    one series: a leave-one-unit-out fold that holds out a single component, a ``SEQUENTIAL``
    row's per-series scoring pass, or a single-recording table (Ottawa, Cranfield, MetroPT's
    one APU).
    """
    sizes = np.bincount(np.asarray(gid, dtype=np.int64))
    n_with = int((sizes[np.asarray(gid, dtype=np.int64)] > 1).sum())
    if n_with == 0 and series is not None and require:
        n_series = len(set(str(v) for v in np.asarray(series, dtype=object).reshape(-1)))
        n_unit = len(set(str(v) for v in np.asarray(unit, dtype=object).reshape(-1))) if unit is not None else 1
        raise ValueError(
            f"{what}: no peer anywhere in this slice - {n_series} series over {n_unit} unit(s), and no "
            "two of them share a unit and a timestamp. A peer-differential row would emit a constant "
            "score. Run it on a slice whose units really have sibling components (the synthetic fleet "
            "has eight axle boxes and two doors per train), widen time_tol_s if the siblings are "
            "timestamped apart, or pass require_peers=False to accept the degenerate zero output."
        )
    return n_with


#: How the peer reference is pooled inside a group. ``"mean"`` is the leave-one-out
#: arithmetic mean (:func:`nebulax.features.cycles.peer_normalise`); ``"median"`` is the
#: leave-one-out **median**, which ``peer_normalisation_shared`` requires - with four boxes
#: a side, one hot box drags the mean of its own reference up by a quarter of its excursion,
#: while the median of the remaining three is untouched.
PEER_STATISTICS: Final[tuple[str, ...]] = ("mean", "median")


def _loo_median_mad(V: np.ndarray, inv: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out median and MAD of each column within each group, vectorised by size.

    Groups are tiny by construction (two boxes of a wheelset, four doors of a side), so the
    groups are bucketed by member count and every bucket is done in one numpy pass: for a
    bucket of ``m`` members, the ``m`` "drop member j" medians are ``m`` medians of an
    ``(n_groups_in_bucket, m - 1, n_features)`` block. NaNs are already imputed by the caller.
    """
    n, f = V.shape
    med = np.full((n, f), np.nan)
    mad = np.full((n, f), np.nan)
    order = np.argsort(inv, kind="stable")
    sizes = np.bincount(inv, minlength=n_groups)
    starts = np.r_[0, np.cumsum(sizes)]
    for m in np.unique(sizes[sizes >= 2]):
        gsel = np.flatnonzero(sizes == m)
        rows = np.stack([order[starts[g] : starts[g] + m] for g in gsel])  # (G, m)
        block = V[rows]  # (G, m, f)
        for j in range(int(m)):
            others = np.delete(block, j, axis=1)  # (G, m-1, f)
            cen = np.median(others, axis=1)  # (G, f)
            med[rows[:, j]] = cen
            mad[rows[:, j]] = _MAD_K * np.median(np.abs(others - cen[:, None, :]), axis=1)
    return med, mad


def peer_delta(
    values: np.ndarray,
    gid: np.ndarray,
    *,
    side: np.ndarray | None = None,
    standardise: bool = False,
    statistic: str = "mean",
) -> np.ndarray:
    """Leave-one-out peer delta of ``values`` (n, f) within each ``gid`` (and ``side``) group.

    ``statistic="mean"`` mirrors :func:`nebulax.features.cycles.peer_normalise`:
    ``x - mean(other members)``, ``/ std(others)`` when ``standardise``.
    ``statistic="median"`` is the robust form ``configs/model_ladder.yaml`` requires for
    ``peer_normalisation_shared`` - "axle boxes must be compared against the **median** of
    peers on the SAME SIDE of the train (R151)" - ``x - median(others)``, standardised by the
    others' MAD. With four same-side boxes the distinction is material: a leave-one-out mean
    lets the *other* degrading box of the same side contaminate the reference, the median of
    three does not. Groups with a single member get 0.0 (no peer, no evidence) rather than
    NaN, because the score contract forbids NaN.

    **The standardising denominator is floored.** A same-side peer group is small by
    construction - two boxes of a wheelset, four doors of a side - and the leave-one-out
    spread of *one* peer is undefined (and of two peers, often near zero), which made the
    standardised delta collapse to a constant zero or explode by eight orders of magnitude.
    So the denominator is ``max(loo_spread, 0.1 * column robust scale)``, and the column's
    own robust scale when there is a single peer: with no within-group spread to measure,
    "in units of this feature's healthy scatter" is the honest normalisation.
    """
    if statistic not in PEER_STATISTICS:
        raise ValueError(f"peer_delta: statistic must be one of {PEER_STATISTICS}, got {statistic!r}")
    V = np.asarray(values, dtype=np.float64)
    if V.ndim == 1:
        V = V[:, None]
    if side is None:
        key: np.ndarray = np.asarray(gid, dtype=np.int64)
    else:  # any hashable side label ('L'/'R', a position parity, ...) splits each group
        codes = np.unique(np.asarray(side, dtype=object), return_inverse=True)[1]
        key = np.asarray(gid, dtype=np.int64) * (int(codes.max()) + 1) + codes.astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    finite = np.isfinite(V)
    Vz = np.where(finite, V, 0.0)
    cnt = np.zeros((uniq.size, V.shape[1]))
    s1 = np.zeros_like(cnt)
    s2 = np.zeros_like(cnt)
    np.add.at(cnt, inv, finite.astype(np.float64))
    np.add.at(s1, inv, Vz)
    np.add.at(s2, inv, Vz * Vz)
    loo_n = np.maximum(cnt[inv] - finite.astype(np.float64), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(loo_n > 0, (s1[inv] - Vz) / np.where(loo_n > 0, loo_n, 1.0), 0.0)
        var = np.where(loo_n > 0, (s2[inv] - Vz * Vz) / np.where(loo_n > 0, loo_n, 1.0) - mean**2, 0.0)
    spread = np.sqrt(np.clip(var, 0.0, None))
    center = mean
    if statistic == "median":
        med, mad = _loo_median_mad(Vz, inv, uniq.size)
        center = np.where(np.isfinite(med), med, mean)
        spread = np.where(np.isfinite(mad), mad, spread)
    delta = np.where(finite & (loo_n > 0), V - center, 0.0)
    if not standardise:
        return sanitise(delta)
    col_scale = np.array([robust_baseline(V[:, j])[1] for j in range(V.shape[1])], dtype=np.float64)
    floor = np.maximum(0.1 * col_scale, _SCALE_FLOOR)
    denom = np.where(loo_n >= 2, np.maximum(spread, floor[None, :]), np.broadcast_to(col_scale, V.shape))
    return sanitise(delta / np.maximum(denom, _SCALE_FLOOR))


def _build_inner(name: str, params: dict[str, Any] | None, extra: dict[str, Any] | None = None) -> BaseModel:
    merged = dict(params or {})
    merged.update(extra or {})
    return build(name, **merged)


# ======================================================================================
# pneumatic - duty cycle
# ======================================================================================


def held_crossing_cusum(
    dev: np.ndarray, t_s: np.ndarray, *, hold_s: float, k: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """The R101 **sustained-crossing** CUSUM. ``dev`` and ``t_s`` must be in time order.

    ``dev > 0`` marks a row on the wrong side of the baseline percentile. Evidence is
    accumulated as a one-sided CUSUM from the **first** row of a crossing, but nothing is
    emitted until that crossing has held for ``hold_s`` (``docs/parameters.md``, pneumatic
    §3: "the crossing must hold for 24 h ... the alarm is dated at the end of the 24 h hold").
    The moment the statistic comes back over the baseline the crossing is broken and both the
    accumulator and the emitted score return to zero - a unit that recovers is not left
    elevated. Returns ``(score, held)``.

    So a 12 h excursion emits exactly zero, a 24 h+ excursion emits the CUSUM of the whole
    excursion (the hold costs 24 h of delay, not 24 h of evidence), and a recovery resets.
    """
    n = dev.size
    out = np.zeros(n, dtype=np.float64)
    held = np.zeros(n, dtype=bool)
    s = 0.0
    start = 0.0
    inside = False
    hold = max(float(hold_s), 0.0)
    kk = float(k)
    for i in range(n):
        d = float(dev[i]) if math.isfinite(float(dev[i])) else 0.0
        if d <= 0.0:  # back over the baseline: the crossing (and the evidence) is gone
            inside, s = False, 0.0
            continue
        if not inside:
            inside, s, start = True, 0.0, float(t_s[i])
        s = max(0.0, s + d - kk)
        if float(t_s[i]) - start >= hold:
            held[i] = True
            out[i] = s
    return out, held


@register("duty_cycle_ratio_cusum", input_kind="cycle_features", family="physics_feature", task="ad")
class DutyCycleRatioCusum(AnomalyDetector):
    """[R101] idle/run duty rule as a one-sided CUSUM on a rolling median, with the 24 h hold.

    ``idle_run_ratio = t_off / t_loaded`` collapses as an air leak grows, so the alarm is a
    *downward* excursion. ``docs/parameters.md`` (pneumatic, 15 Sep 2026 restatement) shows
    the literal "raw p5" rule is unimplementable - the raw feature is zero-inflated (17.3 %
    of healthy MetroPT-3 cycles have ``t_off = 0``) - and restates it as three explicit
    parts, all three of which this row implements:

    1. the threshold is the ``quantile`` **of the 6 h rolling median**, not of the raw feature;
    2. the baseline is the unit's own first ``baseline_days`` (a post-overhaul reference);
    3. the crossing must **hold for ``hold_h`` = 24 h** - a rolling median is heavily
       autocorrelated, and with no hold both detectors fire on 2 of 4 healthy runs.

    :func:`held_crossing_cusum` carries part 3: the score is zero until the crossing has held
    for 24 h (the alarm is dated at the *end* of the hold, as the document requires), then it
    is the CUSUM of the standardised shortfall accumulated over the whole crossing, and it
    returns to zero the moment the rolling median comes back over the baseline. A 12 h
    excursion therefore scores exactly 0, and a recovered unit is not left elevated.

    ``SEQUENTIAL = True``: the hold and the CUSUM are running state, so the runner scores one
    ``series`` (one APU / one component) at a time in ascending ``t``. ``series`` also reaches
    ``_fit``/``_score`` directly, so the rolling median is never taken across two components.

    **The zero-inflation escape hatch is OFF by default** (``fallback_feature=None``).
    ``docs/parameters.md`` retracts the raw-feature percentile because ``idle_run_ratio`` is
    zero-inflated; on the shipped synthetic fleet the *rolling* statistic is zero-inflated too
    (14-16 % of cycles have ``t_off = 0`` and whole 6 h windows are all-zero), so the healthy
    p5 of the rolling median can be **0.000** and a strictly-below crossing can then never
    happen. ``fallback_feature="duty_ratio"`` makes the row switch, on such a degenerate
    baseline, to the companion statistic the same document reports beside it - ``duty_ratio``,
    which "needs no OFF phase and so keeps working past ``s ~ 0.13``" - as an **upward**
    crossing.

    That substitution is a *different detector* from the one this ladder row names, and the
    benchmark results row carries only the ``RunSpec`` fields, so a silent switch would be
    invisible in the leaderboard: a row labelled ``duty_cycle_ratio_cusum`` would in fact be
    ``duty_ratio``/upward with no way to tell. It is therefore opt-in - the default is the
    literal R101 rule, which is allowed to be unfireable on a degenerate baseline and says so
    through ``baseline_ == ref_min_`` - and whenever it *is* asked for, the fitted model
    persists exactly what it resolved to: ``feature_name_``, ``feature_index_``,
    ``direction_`` and ``fallback_used_``.

    Defaults: ``feature="idle_run_ratio"``, ``feature_index=None``, ``feature_names=None``,
    ``window_h=6.0``, ``baseline_days=8.0``, ``hold_h=24.0``, ``quantile=0.05``, ``k=0.5``,
    ``direction="down"``, ``fallback_feature=None`` (the literal rule; pass ``"duty_ratio"``
    to allow the documented substitution), ``fallback_direction="up"``, ``min_periods=3``.
    """

    SEQUENTIAL: ClassVar[bool] = True
    _PATTERNS: Final[tuple[str, ...]] = (r"idle_run_ratio", r"duty_ratio", r"duty")
    #: ``duty_ratio`` needs no OFF phase and keeps working past ``s ~ 0.13`` where
    #: ``idle_run_ratio`` saturates at zero (``docs/parameters.md``), so it is the documented
    #: stand-in on a table that carries one but not the other.
    _ALIASES: Final[tuple[str, ...]] = ("duty_ratio", "COMP_duty")

    def __init__(
        self,
        feature: str | int | None = "idle_run_ratio",
        *,
        feature_index: int | None = None,
        feature_names: Sequence[str] | None = None,
        window_h: float = 6.0,
        baseline_days: float = 8.0,
        hold_h: float = 24.0,
        quantile: float = 0.05,
        k: float = 0.5,
        direction: str = "down",
        fallback_feature: str | None = None,
        fallback_direction: str = "up",
        min_periods: int = 3,
    ) -> None:
        super().__init__(
            feature=feature,
            feature_index=feature_index,
            feature_names=list(feature_names) if feature_names is not None else None,
            window_h=window_h,
            baseline_days=baseline_days,
            hold_h=hold_h,
            quantile=quantile,
            k=k,
            direction=direction,
            fallback_feature=fallback_feature,
            fallback_direction=fallback_direction,
            min_periods=min_periods,
        )
        for d, what in ((direction, "direction"), (fallback_direction, "fallback_direction")):
            if d not in ("down", "up", "both"):
                raise ValueError(f"duty_cycle_ratio_cusum: {what} must be down|up|both, got {d!r}")
        self.feature = feature
        self.feature_index = feature_index
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.window_h = float(window_h)
        self.baseline_days = float(baseline_days)
        self.hold_h = float(hold_h)
        self.quantile = float(quantile)
        self.k = float(k)
        self.direction = direction
        self.fallback_feature = fallback_feature
        self.fallback_direction = fallback_direction
        self.min_periods = int(min_periods)
        self.feature_index_: int = 0
        self.feature_name_: str = ""
        self.direction_: str = direction
        self.fallback_used_: bool = False
        self.baseline_: float = 0.0
        self.ref_min_: float = 0.0
        self.ref_max_: float = 0.0
        self.scale_: float = 1.0

    def _statistic(
        self, X: np.ndarray, t: np.ndarray | None, series: Any | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The 6 h rolling median of the duty feature, taken **per series**, and the clock."""
        t_s = times_seconds(t, X.shape[0])
        x = sanitise(X[:, self.feature_index_], fill=np.nan)
        fill = float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0
        x = np.where(np.isfinite(x), x, fill)
        win = self.window_h * 3600.0
        if series is None:
            return rolling_time_stat(x, t_s, win, stat="median", min_periods=self.min_periods), t_s
        s = np.asarray(series, dtype=object).reshape(-1)
        roll = np.empty(x.size, dtype=np.float64)
        for key in pd.unique(s):
            sel = np.flatnonzero(s == key)
            roll[sel] = rolling_time_stat(
                x[sel], t_s[sel], win, stat="median", min_periods=self.min_periods
            )
        return roll, t_s

    def _fit(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        *,
        feature_names: Sequence[str] | None = None,
        series: Any | None = None,
        **kwargs: Any,
    ) -> None:
        names = feature_names if feature_names is not None else self.feature_names
        self.feature_index_ = resolve_feature_index(
            self.feature,
            index=self.feature_index,
            feature_names=names,
            n_features=_columns_of(X),
            patterns=self._PATTERNS,
            aliases=self._ALIASES,
            what="duty_cycle_ratio_cusum.feature",
        )
        self.feature_name_ = str(names[self.feature_index_]) if names is not None else ""
        self.direction_ = self.direction
        self.fallback_used_ = False
        self._calibrate(X, t, series)
        if self._degenerate(X, t, series) and self.fallback_feature is not None and names is not None:
            j = resolve_feature_index(
                self.fallback_feature,
                feature_names=names,
                n_features=_columns_of(X),
                required=False,
                what="duty_cycle_ratio_cusum.fallback_feature",
            )
            if j != self.feature_index_ and str(names[j]).lower().startswith(
                str(self.fallback_feature).lower()
            ):
                self.feature_index_, self.feature_name_ = j, str(names[j])
                self.direction_, self.fallback_used_ = self.fallback_direction, True
                self._calibrate(X, t, series)

    def _calibrate(self, X: np.ndarray, t: np.ndarray | None, series: Any | None) -> None:
        roll, t_s = self._statistic(X, t, series)
        span = self.baseline_days * 86400.0
        base_rows = t_s <= (np.nanmin(t_s) + span) if np.isfinite(t_s).any() else np.ones(roll.size, bool)
        if base_rows.sum() < max(5, self.min_periods):
            base_rows = np.ones(roll.size, dtype=bool)
        ref = roll[base_rows]
        q = self.quantile if self.direction_ == "down" else 1.0 - self.quantile
        self.baseline_ = float(np.nanquantile(ref, q))
        self.ref_min_ = float(np.nanmin(ref))
        self.ref_max_ = float(np.nanmax(ref))
        _, self.scale_ = robust_baseline(ref)

    def _degenerate(self, X: np.ndarray, t: np.ndarray | None, series: Any | None) -> bool:
        """True when the fitted baseline sits on the floor/ceiling of its own reference.

        A zero-inflated statistic puts its own 5th percentile *at* the minimum, and then
        ``roll < baseline`` is unsatisfiable: the rule cannot fire at all, on any unit.
        """
        eps = 1e-12
        if self.direction_ == "down":
            return self.baseline_ <= self.ref_min_ + eps
        if self.direction_ == "up":
            return self.baseline_ >= self.ref_max_ - eps
        return False

    def _score(
        self, X: np.ndarray, t: np.ndarray | None = None, *, series: Any | None = None
    ) -> np.ndarray:
        roll, t_s = self._statistic(X, t, series)
        if self.direction_ == "down":
            dev = (self.baseline_ - roll) / self.scale_
        elif self.direction_ == "up":
            dev = (roll - self.baseline_) / self.scale_
        else:
            dev = np.abs(roll - self.baseline_) / self.scale_
        dev = sanitise(dev)
        out = np.zeros(X.shape[0], dtype=np.float64)
        hold_s = self.hold_h * 3600.0
        groups = (
            [np.arange(X.shape[0])]
            if series is None
            else [
                np.flatnonzero(np.asarray(series, dtype=object).reshape(-1) == key)
                for key in pd.unique(np.asarray(series, dtype=object).reshape(-1))
            ]
        )
        for sel in groups:  # running state never crosses a component
            order = sel[np.argsort(t_s[sel], kind="stable")]
            out[order] = held_crossing_cusum(dev[order], t_s[order], hold_s=hold_s, k=self.k)[0]
        return out


# ======================================================================================
# bearing - kurtogram / envelope / thermal
# ======================================================================================


@register("kurtogram_envelope_bpfo", input_kind="raw_window", family="physics_feature", task="ad")
class KurtogramEnvelopeBpfo(AnomalyDetector):
    """[R117] spectral-kurtosis band -> Hilbert envelope -> BPFO/BPFI/BSF/FTF amplitudes.

    Wraps :func:`nebulax.features.vibration.envelope_spectrum_feats` and turns the selected
    amplitudes into a normal-only score: robust z against the training median/MAD of each
    feature, aggregated by ``aggregate``.

    **The default feature set is the ladder row verbatim** - "spectral kurtosis band ->
    Hilbert envelope -> BPFO/BPFI/BSF/FTF harmonics; emit ``sk_band_fc``, ``sk_band_bw``,
    ``sk_max``":

    ``("env_bpfo_energy", "env_bpfi_energy", "env_bsf_energy", "env_ftf_energy",
    "sk_band_fc", "sk_band_bw", "sk_max")``

    Each ``env_<name>_energy`` is the **sum of that family's harmonic amplitudes** in m/s2
    (BPFO/BPFI 3 harmonics, BSF 2, FTF 1 - see ``vibration._HARMONICS``); pass
    ``harmonics=True`` to score the individual ``env_<name>_h<k>`` harmonics instead of the
    per-family sums. ``env_bpfo_energy`` is the quantity ``docs/parameters.md`` calls
    "BPFO amp" (see that function's scaling note). ``sk_band_fc``/``sk_band_bw`` are the
    kurtogram-selected band: a resonance that moves or widens is itself bearing evidence.

    **Sampling rate.** The frequency axis of the envelope spectrum is only right if the
    model knows the rate the samples were *delivered* at. Ottawa's raw windows are decimated
    (``decimate=4`` -> ``fs_hz_out=10500.0``, not the sensor's 42 kHz), so a hard-coded rate
    put every BPFO/BPFI/BSF/FTF line 4x off. ``fs=None`` (the default) therefore takes the
    rate from the runner's ``fs_hz`` row context (``nebulax.bench.base.CONTEXT_KEYS``: the
    loader's ``fs_hz_out``/``fs_hz`` after decimation); ``fs=<float>`` is an explicit
    override that always wins, and with neither the documented fallback is
    :data:`DEFAULT_FS_HZ` (42 kHz, Ottawa's native rate). The rate actually used is recorded
    on the fitted model as ``fs_hz_``, and scoring a slice delivered at a *different* rate
    raises ``ValueError`` rather than silently re-scaling the spectrum.

    Defaults: ``fs=None`` (auto, from the row context; fallback ``DEFAULT_FS_HZ`` = 42 kHz,
    Ottawa's native rate), ``channel=None`` (auto: a ``vib``/``acc``
    channel when the runner supplies channel names, else channel 0), ``shaft_rpm=1800.0``,
    ``speed_ms=None``, ``wheel_radius_m=0.425``, ``n_elements=9``,
    ``ball_diameter_m=7.94e-3``, ``pitch_diameter_m=39.0e-3``, ``contact_angle_deg=0.0``,
    ``max_level=3``, ``features=DEFAULT_FEATURES``, ``harmonics=False``, ``aggregate="max"``,
    ``max_fit_windows=512``, ``seed=0``.
    """

    _AGGS: Final[tuple[str, ...]] = ("max", "mean", "sum")
    _CHANNEL_PATTERNS: Final[tuple[str, ...]] = (r"vib", r"acc", r"^x$", r"^y$")
    #: Last-resort sampling rate when neither ``fs`` nor the runner's ``fs_hz`` is given:
    #: Ottawa's native rate. Any real benchmark run gets the delivered rate instead.
    DEFAULT_FS_HZ: Final[float] = 42_000.0
    #: Relative tolerance for "the score slice was delivered at the fitted rate".
    _FS_RTOL: Final[float] = 1e-3
    #: The ladder row's feature set: all four fault families plus the three kurtogram columns.
    DEFAULT_FEATURES: Final[tuple[str, ...]] = (
        "env_bpfo_energy",
        "env_bpfi_energy",
        "env_bsf_energy",
        "env_ftf_energy",
        "sk_band_fc",
        "sk_band_bw",
        "sk_max",
    )
    #: ``harmonics=True``: the individual harmonic amplitudes behind those four sums.
    HARMONIC_FEATURES: Final[tuple[str, ...]] = (
        "env_bpfo_h1",
        "env_bpfo_h2",
        "env_bpfo_h3",
        "env_bpfi_h1",
        "env_bpfi_h2",
        "env_bpfi_h3",
        "env_bsf_h1",
        "env_bsf_h2",
        "env_ftf_h1",
        "sk_band_fc",
        "sk_band_bw",
        "sk_max",
    )

    def __init__(
        self,
        *,
        fs: float | None = None,
        channel: int | str | None = None,
        shaft_rpm: float = 1800.0,
        speed_ms: float | None = None,
        wheel_radius_m: float = 0.425,
        n_elements: int = 9,
        ball_diameter_m: float = 7.94e-3,
        pitch_diameter_m: float = 39.0e-3,
        contact_angle_deg: float = 0.0,
        max_level: int = 3,
        features: Sequence[str] | None = None,
        harmonics: bool = False,
        aggregate: str = "max",
        max_fit_windows: int = 512,
        seed: int = 0,
    ) -> None:
        feats = list(
            features
            if features is not None
            else (self.HARMONIC_FEATURES if harmonics else self.DEFAULT_FEATURES)
        )
        super().__init__(
            fs=fs,
            channel=channel,
            shaft_rpm=shaft_rpm,
            speed_ms=speed_ms,
            wheel_radius_m=wheel_radius_m,
            n_elements=n_elements,
            ball_diameter_m=ball_diameter_m,
            pitch_diameter_m=pitch_diameter_m,
            contact_angle_deg=contact_angle_deg,
            max_level=max_level,
            features=feats,
            harmonics=harmonics,
            aggregate=aggregate,
            max_fit_windows=max_fit_windows,
            seed=seed,
        )
        if aggregate not in self._AGGS:
            raise ValueError(f"kurtogram_envelope_bpfo: aggregate must be one of {self._AGGS}")
        self.fs = None if fs is None else float(fs)
        self.channel = channel
        self.harmonics = bool(harmonics)
        self.shaft_rpm = float(shaft_rpm)
        self.speed_ms = None if speed_ms is None else float(speed_ms)
        self.wheel_radius_m = float(wheel_radius_m)
        self.geometry = BearingGeometry(
            n_elements=int(n_elements),
            ball_diameter_m=float(ball_diameter_m),
            pitch_diameter_m=float(pitch_diameter_m),
            contact_angle_deg=float(contact_angle_deg),
        )
        self.max_level = int(max_level)
        self.features = feats
        self.aggregate = aggregate
        self.max_fit_windows = int(max_fit_windows)
        self.seed = int(seed)
        self.feature_names_: list[str] = list(feats)
        #: The sampling rate the fitted spectra were computed at (Hz). See the class docstring.
        self.fs_hz_: float = self.DEFAULT_FS_HZ if self.fs is None else self.fs
        self.channel_: int = 0 if not isinstance(channel, (int, np.integer)) else int(channel)
        self.center_: np.ndarray = np.zeros(len(feats))
        self.scale_: np.ndarray = np.ones(len(feats))

    def _resolve_fs(self, fs_hz: float | None) -> float:
        """The rate to analyse at: an explicit ``fs`` wins, then the runner's, then the default."""
        if self.fs is not None:
            return self.fs
        if fs_hz is not None and np.isfinite(float(fs_hz)) and float(fs_hz) > 0.0:
            return float(fs_hz)
        return float(self.DEFAULT_FS_HZ)

    def _feature_table(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"kurtogram_envelope_bpfo: expects raw_window (n, L, c), got {X.shape}")
        ch = self.channel_ % X.shape[2]
        rpm_or_speed = self.speed_ms if self.speed_ms is not None else self.shaft_rpm
        is_speed = self.speed_ms is not None
        rows = np.empty((X.shape[0], len(self.feature_names_)), dtype=np.float64)
        for i in range(X.shape[0]):
            feats = envelope_spectrum_feats(
                X[i, :, ch].astype(np.float64),
                self.fs_hz_,
                rpm_or_speed,
                self.geometry,
                is_speed_ms=is_speed,
                wheel_radius_m=self.wheel_radius_m,
                max_level=self.max_level,
            )
            rows[i] = [float(feats.get(k, np.nan)) for k in self.feature_names_]
        return rows

    def _fit(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        *,
        feature_names: Sequence[str] | None = None,
        fs_hz: float | None = None,
        **kwargs: Any,
    ) -> None:
        if X.ndim != 3:
            raise ValueError(f"kurtogram_envelope_bpfo: expects raw_window (n, L, c), got {X.shape}")
        self.fs_hz_ = self._resolve_fs(fs_hz)
        self.channel_ = resolve_feature_index(
            self.channel,
            feature_names=feature_names,
            n_features=X.shape[2],
            patterns=self._CHANNEL_PATTERNS,
            required=False,
            what="kurtogram_envelope_bpfo.channel",
        )
        idx = np.arange(X.shape[0])
        if idx.size > self.max_fit_windows:
            idx = np.sort(np.random.default_rng(self.seed).choice(idx, self.max_fit_windows, replace=False))
        tab = self._feature_table(X[idx])
        cen, sc = [], []
        for j in range(tab.shape[1]):
            m, s = robust_baseline(tab[:, j])
            cen.append(m)
            sc.append(s)
        self.center_ = np.asarray(cen, dtype=np.float64)
        self.scale_ = np.asarray(sc, dtype=np.float64)

    def _score(
        self, X: np.ndarray, t: np.ndarray | None = None, *, fs_hz: float | None = None
    ) -> np.ndarray:
        got = self._resolve_fs(fs_hz)
        if not math.isclose(got, self.fs_hz_, rel_tol=self._FS_RTOL):
            raise ValueError(
                f"kurtogram_envelope_bpfo: fitted at {self.fs_hz_:g} Hz but this slice is delivered "
                f"at {got:g} Hz. The envelope-spectrum frequency axis (BPFO/BPFI/BSF/FTF) is not "
                "comparable across sampling rates, so the healthy baseline would be meaningless. "
                "Fit and score on the same loader settings, or pin fs=<rate> explicitly."
            )
        z = (self._feature_table(X) - self.center_) / self.scale_
        z = sanitise(z, 0.0)
        if self.aggregate == "max":
            return z.max(axis=1)
        if self.aggregate == "sum":
            return z.sum(axis=1)
        return z.mean(axis=1)


@register("peer_delta_temperature", input_kind="window_stats", family="physics_feature", task="ad")
class PeerDeltaTemperature(AnomalyDetector):
    """[R112-R114] axle-box temperature against its peers at the same instant.

    The absolute box temperature tracks speed, load and ambient; the *difference* to the
    peer boxes does not, which is why every operator rule is a differential one. Peers are
    the other rows of the **same unit** at the same timestamp (``time_tol_s``) carrying a
    **different series** (:func:`peer_group_ids`, fed by the runner's ``series``/``unit``
    context); the score is the robust z of the leave-one-out peer delta against its healthy
    training distribution, so "hotter than my peers, by more than healthy scatter" scores high.

    The peer reference is the **median** of the other boxes in the group (``statistic``,
    :data:`PEER_STATISTICS`): with four boxes a side, a leave-one-out arithmetic mean lets a
    second degrading box drag its own reference up, the median of the rest does not.

    Defaults: ``feature="T_box"``, ``feature_index=None``, ``feature_names=None``,
    ``time_tol_s=1.0``, ``same_side=False`` (R151 like-for-like, from the component's L/R
    letter - see :func:`series_sides`), ``side_from="series"``, ``side_stride=2``,
    ``standardise=False``, ``statistic="median"``, ``signed=True``, ``require_peers=True``.

    An individual group with no peer contributes a delta of 0 - "no differential evidence"
    rather than NaN. A *table* with no peers anywhere is refused at fit time
    (:func:`check_peers_exist`), because the row would then emit a constant score.
    """

    _PATTERNS: Final[tuple[str, ...]] = (r"t_?box.*mean", r"t_?box", r"temp.*mean", r"temp")

    def __init__(
        self,
        feature: str | int | None = "T_box",
        *,
        feature_index: int | None = None,
        feature_names: Sequence[str] | None = None,
        time_tol_s: float = 1.0,
        same_side: bool = False,
        side_from: str = "series",
        side_stride: int = 2,
        standardise: bool = False,
        statistic: str = "median",
        signed: bool = True,
        require_peers: bool = True,
    ) -> None:
        super().__init__(
            feature=feature,
            feature_index=feature_index,
            feature_names=list(feature_names) if feature_names is not None else None,
            time_tol_s=time_tol_s,
            same_side=same_side,
            side_from=side_from,
            side_stride=side_stride,
            standardise=standardise,
            statistic=statistic,
            signed=signed,
            require_peers=require_peers,
        )
        if statistic not in PEER_STATISTICS:
            raise ValueError(
                f"peer_delta_temperature: statistic must be one of {PEER_STATISTICS}, got {statistic!r}"
            )
        self.feature = feature
        self.feature_index = feature_index
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.time_tol_s = float(time_tol_s)
        self.same_side = bool(same_side)
        self.side_from = _check_side_from(side_from, self.name)
        self.side_stride = max(int(side_stride), 1)
        self.standardise = bool(standardise)
        self.statistic = str(statistic)
        self.signed = bool(signed)
        self.require_peers = bool(require_peers)
        self.feature_index_: int = 0
        self.feature_name_: str = ""
        self.n_rows_with_peers_: int = 0
        self.center_: float = 0.0
        self.scale_: float = 1.0

    def _delta(
        self, X: np.ndarray, t: np.ndarray | None, series: Any | None, unit: Any | None
    ) -> np.ndarray:
        t_s = times_seconds(t, X.shape[0])
        gid = peer_group_ids(t_s, self.time_tol_s, unit=unit, series=series)
        side = sides_for(
            t_s, gid, series, same_side=self.same_side, side_from=self.side_from, stride=self.side_stride
        )
        x = sanitise(X[:, self.feature_index_], fill=np.nan)
        fill = float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0
        x = np.where(np.isfinite(x), x, fill)
        return peer_delta(x, gid, side=side, standardise=self.standardise, statistic=self.statistic)[:, 0]

    def _fit(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        *,
        feature_names: Sequence[str] | None = None,
        series: Any | None = None,
        unit: Any | None = None,
        **kwargs: Any,
    ) -> None:
        names = feature_names if feature_names is not None else self.feature_names
        self.feature_index_ = resolve_feature_index(
            self.feature,
            index=self.feature_index,
            feature_names=names,
            n_features=_columns_of(X),
            patterns=self._PATTERNS,
            what="peer_delta_temperature.feature",
        )
        self.feature_name_ = str(names[self.feature_index_]) if names is not None else ""
        gid = peer_group_ids(times_seconds(t, X.shape[0]), self.time_tol_s, unit=unit, series=series)
        self.n_rows_with_peers_ = check_peers_exist(
            gid, series, unit=unit, what=self.name, require=self.require_peers
        )
        self.center_, self.scale_ = robust_baseline(self._delta(X, t, series, unit))

    def _score(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        *,
        series: Any | None = None,
        unit: Any | None = None,
    ) -> np.ndarray:
        z = (self._delta(X, t, series, unit) - self.center_) / self.scale_
        return sanitise(z if self.signed else np.abs(z))


def _group_bounds(g_sorted: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` of each run of equal values in a sorted group-id array."""
    if g_sorted.size == 0:
        return []
    change = np.flatnonzero(np.diff(g_sorted) != 0)
    starts = np.r_[0, change + 1]
    ends = np.r_[change, g_sorted.size - 1]
    return list(zip(starts.tolist(), ends.tolist()))


#: ``side_from="series"`` is the defensible R151 rule (the component's own L/R letter);
#: ``"position"`` is the legacy row-position stride, kept only as an explicit opt-in because
#: it is row-order dependent and therefore not a component-identity rule.
SIDE_SOURCES: Final[tuple[str, ...]] = ("series", "position")


def _check_side_from(value: str, what: str) -> str:
    if value not in SIDE_SOURCES:
        raise ValueError(f"{what}: side_from must be one of {SIDE_SOURCES}, got {value!r}")
    return value


def sides_for(
    t_s: np.ndarray,
    gid: np.ndarray,
    series: Any | None,
    *,
    same_side: bool,
    side_from: str = "series",
    stride: int = 2,
) -> np.ndarray | None:
    """The side label per row, or ``None`` when peers must not be split by side.

    ``side_from="series"`` (the default) reads the L/R letter off the component id, so the
    label belongs to the component and is invariant to the order equal-timestamp rows arrive
    in. ``"position"`` reproduces the old row-position stride and is only reachable by asking
    for it explicitly. Series ids with no side letter all share one side, which makes the
    rule a no-op on tables that have no L/R geometry (MetroPT, Ottawa, Cranfield).
    """
    if not same_side:
        return None
    if side_from == "series" and series is not None:
        return series_sides(series)
    if side_from == "position" and stride > 1:
        order, _ = _order_by_time(t_s)
        pos = np.zeros(gid.size, dtype=np.int64)
        for start, end in _group_bounds(gid[order]):
            pos[order[start : end + 1]] = np.arange(end + 1 - start)
        return pos % int(stride)
    return None


@register("thermal_residual_model", input_kind="window_stats", family="physics_residual", task="ad")
class ThermalResidualModel(AnomalyDetector):
    """[R110] ``T_box`` regressed on its **operating context**; the residual is the detector.

    A ridge regression predicts the box temperature from the ladder row's stated inputs -
    "speed, ambient, load, adjacent and opposite axlebox temperatures" - on *healthy*
    training rows only. ``thermal_residual = T_box - T_box_pred`` is then robust-standardised;
    a positive residual is a box running hotter than its own operating point explains.

    Two covariate sources, both physical:

    ``named columns``   ``speed`` / ``ambient`` / ``T_amb`` / ``load`` / ``dwell`` / ``duty``
                        columns of the feature table, found through ``feature_names``;
    ``peer temperature``  the **adjacent** (same-side) and **opposite** box temperature at the
                        same ``unit`` and timestamp, built from the runner's ``series`` /
                        ``unit`` context by :func:`peer_group_ids` and :func:`series_sides`.

    Columns derived from the **target's own sensor** are excluded outright
    (:func:`base_signal`): regressing ``T_box_mean`` on ``T_box_max``/``T_box_slope`` - or on
    the wayside twin of the same box - fits the fault as well as the context and turns the
    residual into noise. That silent same-sensor fit is what this row used to do on the
    shipped bearing table, and it is now impossible: with ``require_physical=True`` (the
    default) a table that supplies *neither* source raises ``ValueError`` naming exactly what
    is missing, instead of quietly becoming a different method.

    Defaults: ``target="T_box"``, ``target_index=None``, ``covariates=None`` (auto, as
    above), ``covariate_indices=None``, ``feature_names=None``, ``peer_covariates=True``,
    ``time_tol_s=1.0``, ``require_physical=True``, ``alpha=1.0``, ``max_covariates=24``,
    ``signed=True``.

    :meth:`residual` is public so ``page_hinkley_thermal_residual`` can stream exactly this
    quantity without duplicating the fit. A scored slice that holds a **single** series (the
    runner scores ``SEQUENTIAL`` rows one component at a time) has no peers: those rows fall
    back to the fitted training median of the peer columns and are counted in
    ``n_peerless_rows_``, so the peer term is then a constant offset rather than a silent
    same-sensor regression.
    """

    _TARGET_PATTERNS: Final[tuple[str, ...]] = (r"t_?box.*mean", r"t_?box", r"temp.*mean", r"temp")
    #: The operating context R110 names. ``t_?box`` is deliberately **not** here - the peer
    #: box temperatures come from the peer channel, not from this row's own sensor columns.
    _COV_PATTERNS: Final[tuple[str, ...]] = (
        r"speed",
        r"ambient",
        r"t_?amb",
        r"load",
        r"dwell",
        r"duty",
    )
    _COV_WANTED: Final[str] = "speed / ambient / load / dwell"

    def __init__(
        self,
        *,
        target: str | int | None = "T_box",
        target_index: int | None = None,
        covariates: Sequence[str | int] | None = None,
        covariate_indices: Sequence[int] | None = None,
        feature_names: Sequence[str] | None = None,
        peer_covariates: bool = True,
        time_tol_s: float = 1.0,
        require_physical: bool = True,
        alpha: float = 1.0,
        max_covariates: int = 24,
        signed: bool = True,
    ) -> None:
        super().__init__(
            target=target,
            target_index=target_index,
            covariates=list(covariates) if covariates is not None else None,
            covariate_indices=list(covariate_indices) if covariate_indices is not None else None,
            feature_names=list(feature_names) if feature_names is not None else None,
            peer_covariates=peer_covariates,
            time_tol_s=time_tol_s,
            require_physical=require_physical,
            alpha=alpha,
            max_covariates=max_covariates,
            signed=signed,
        )
        self.target = target
        self.target_index = target_index
        self.covariates = list(covariates) if covariates is not None else None
        self.covariate_indices = list(covariate_indices) if covariate_indices is not None else None
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.peer_covariates = bool(peer_covariates)
        self.time_tol_s = float(time_tol_s)
        self.require_physical = bool(require_physical)
        self.alpha = float(alpha)
        self.max_covariates = int(max_covariates)
        self.signed = bool(signed)
        self.target_index_: int = 0
        self.target_name_: str = ""
        self.covariate_index_: list[int] = []
        self.covariate_names_: list[str] = []
        self.peer_columns_: list[str] = []
        self.model_: Any = None
        self.fill_: np.ndarray = np.zeros(0)
        self.peer_fill_: np.ndarray = np.zeros(0)
        self.n_peerless_rows_: int = 0
        self.center_: float = 0.0
        self.scale_: float = 1.0

    # -- the peer covariate ---------------------------------------------------------------
    def _peer_temperatures(
        self, y: np.ndarray, t: np.ndarray | None, series: Any | None, unit: Any | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(adjacent, opposite)`` box temperature per row, plus a "row had a peer" mask.

        The ladder row names two **specific** boxes, not two side averages:

        ``adjacent``  the other box of the same bogie on the *same side*
                      (:func:`bogie_partner`: ``axlebox_1L`` <-> ``axlebox_2L``,
                      ``axlebox_3L`` <-> ``axlebox_4L``), falling back to the nearest other
                      same-side index in the group when that box is not in this slice;
        ``opposite``  the corresponding box on the *other side* of the same wheelset
                      (``axlebox_3R`` -> ``axlebox_3L``), falling back to the nearest
                      other-side index.

        Averaging every same-side box and every opposite box - which is what this used to do -
        throws away exactly the like-for-like comparison R151 asks for: bearing 1 of a motor
        car carries ~13.5 % more roller-raceway contact force than bearing 2 and runs
        measurably hotter, so "the mean of the other three boxes on my side" is a different,
        systematically biased reference. Peers are looked up within a peer group (one unit,
        one timestamp - :func:`peer_group_ids`); a box with neither peer present gets NaN,
        which :meth:`_design` fills with the fitted training median and counts in
        ``n_peerless_rows_``. A series id that carries **no** position number has no adjacent
        box and one that carries no side letter has no opposite box: those resolve to NaN
        rather than to the row's own temperature, so a table with no axle-box geometry at all
        (a single APU, one Ottawa recording) offers no peer covariate and ``require_physical``
        refuses it instead of quietly regressing ``T_box`` on ``T_box``.
        """
        n = y.size
        t_s = times_seconds(t, n)
        gid = peer_group_ids(t_s, self.time_tol_s, unit=unit, series=series)
        side, idx = component_positions(series, n)
        other = np.where(side == "L", "R", np.where(side == "R", "L", side))
        finite = np.isfinite(y)
        yv = np.where(finite, y, np.nan)

        frame = pd.DataFrame({"g": gid, "s": side, "i": idx, "y": yv})
        # one value per (group, side, index); duplicates within a group cannot happen because
        # peer_group_ids already allows at most one row per series
        table = frame.groupby(["g", "s", "i"], sort=False)["y"].mean()

        def lookup(want_side: np.ndarray, want_idx: np.ndarray) -> np.ndarray:
            hit = table.reindex(pd.MultiIndex.from_arrays([gid, want_side, want_idx]))
            return np.asarray(hit.to_numpy(np.float64))

        # a component with no position number is its own "partner" and a component with no
        # side letter is its own "opposite": both would read the target's own sensor straight
        # back into the design matrix, which is the exact same-sensor regression this row
        # refuses, so they resolve to NaN (no peer) instead
        partner = bogie_partner(idx)
        same_ok, opp_ok = partner != idx, other != side
        same = np.where(same_ok, lookup(side, partner), np.nan)
        opp = np.where(opp_ok, lookup(other, idx), np.nan)
        same, opp = self._nearest_fallback(table, gid, side, other, idx, same, opp, same_ok, opp_ok)
        had = np.isfinite(same) | np.isfinite(opp)
        return np.column_stack([same, opp]), had

    @staticmethod
    def _nearest_fallback(
        table: pd.Series,
        gid: np.ndarray,
        side: np.ndarray,
        other: np.ndarray,
        idx: np.ndarray,
        same: np.ndarray,
        opp: np.ndarray,
        same_ok: np.ndarray,
        opp_ok: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fill the rows whose named partner is absent with the nearest index on that side.

        ``same_ok`` / ``opp_ok`` carry the geometry guards from the caller, so a component with
        no position number never picks up "the nearest box on my side" (that would be itself)
        and one with no side letter never picks up an "opposite" box (likewise itself).
        """
        need = np.flatnonzero((same_ok & ~np.isfinite(same)) | (opp_ok & ~np.isfinite(opp)))
        if need.size == 0:
            return same, opp
        members: dict[tuple[Any, Any], list[tuple[int, float]]] = {}
        for (g, sd, i), v in table.items():
            if np.isfinite(v):
                members.setdefault((g, sd), []).append((int(i), float(v)))
        for r in need:
            for want_side, out, ok in ((side[r], same, same_ok[r]), (other[r], opp, opp_ok[r])):
                if not ok or np.isfinite(out[r]):
                    continue
                pool = members.get((gid[r], want_side))
                if not pool:
                    continue
                # never the row's own box: on the same side that is its own index, on the
                # opposite side the guard above has already ruled out "the same side again"
                cand = [(abs(i - idx[r]), i, v) for i, v in pool if not (want_side == side[r] and i == idx[r])]
                if cand:
                    out[r] = min(cand)[2]
        return same, opp

    def _design(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        series: Any | None = None,
        unit: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        y = sanitise(X[:, self.target_index_], fill=np.nan)
        Z = np.asarray(X[:, self.covariate_index_], dtype=np.float64)
        Z = np.where(np.isfinite(Z), Z, self.fill_[None, :]) if Z.size else Z
        if not self.peer_columns_:
            return y, Z
        peers, had = self._peer_temperatures(y, t, series, unit)
        self.n_peerless_rows_ = int((~had).sum())
        peers = np.where(np.isfinite(peers), peers, self.peer_fill_[None, :])
        return y, (np.concatenate([Z, peers], axis=1) if Z.size else peers)

    def _auto_covariates(self, n_f: int, names: Sequence[str] | None) -> list[int]:
        """Named operating-context columns, minus everything derived from the target sensor."""
        known = names is not None and len(names) == n_f
        if not known:
            return [j for j in range(n_f) if j != self.target_index_]
        assert names is not None
        own = base_signal(str(names[self.target_index_])).lower()
        hit: list[int] = []
        for pat in self._COV_PATTERNS:
            rx = re.compile(pat, re.IGNORECASE)
            hit.extend(j for j, nm in enumerate(names) if rx.search(str(nm)) and j not in hit)
        return [j for j in hit if not str(names[j]).lower().startswith(own)]

    def _fit(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        *,
        feature_names: Sequence[str] | None = None,
        series: Any | None = None,
        unit: Any | None = None,
        **kwargs: Any,
    ) -> None:
        from sklearn.linear_model import Ridge

        names = feature_names if feature_names is not None else self.feature_names
        n_f = _columns_of(X)
        self.target_index_ = resolve_feature_index(
            self.target,
            index=self.target_index,
            feature_names=names,
            n_features=n_f,
            patterns=self._TARGET_PATTERNS,
            what="thermal_residual_model.target",
        )
        self.target_name_ = str(names[self.target_index_]) if names is not None else ""
        Xf = sanitise(X, fill=np.nan)
        y_all = Xf[:, self.target_index_]

        if self.covariate_indices is not None or self.covariates is not None:
            cov = resolve_feature_indices(
                self.covariates,
                indices=self.covariate_indices,
                feature_names=names,
                n_features=n_f,
                what="thermal_residual_model.covariates",
            )
        else:
            cov = self._auto_covariates(n_f, names)
        cov = [j for j in cov if j != self.target_index_]
        if self.require_physical and names is not None:
            # The same-sensor exclusion applies to EXPLICIT covariates too: T_box_max is not a
            # physical covariate of T_box_mean, however it was asked for.
            own = base_signal(str(names[self.target_index_])).lower()
            same = [str(names[j]) for j in cov if base_signal(str(names[j])).lower() == own]
            if same:
                raise ValueError(
                    f"thermal_residual_model: covariates {same} are statistics of the target's own "
                    f"sensor ({own}); a same-sensor regression is not the named method. Drop them or "
                    f"pass require_physical=False."
                )
        if len(cov) > self.max_covariates:  # keep the most informative columns
            with np.errstate(invalid="ignore"):
                yv = np.where(np.isfinite(y_all), y_all, np.nanmedian(y_all))
                cols = np.where(np.isfinite(Xf[:, cov]), Xf[:, cov], 0.0)
                num = np.abs(((cols - cols.mean(0)) * (yv - yv.mean())[:, None]).mean(0))
                den = cols.std(0) * (yv.std() + _SCALE_FLOOR)
                corr = sanitise(num / np.where(den > 0, den, np.nan), 0.0)
            cov = [cov[i] for i in np.argsort(corr)[::-1][: self.max_covariates]]
        self.covariate_index_ = sorted(cov)
        self.covariate_names_ = [str(names[j]) for j in self.covariate_index_] if names is not None else []

        # peer (adjacent / opposite box) covariates, when the row context actually has peers
        self.peer_columns_ = []
        self.peer_fill_ = np.zeros(0)
        if self.peer_covariates:
            peers, had = self._peer_temperatures(y_all, t, series, unit)
            if had.any():
                own = float(np.nanmedian(y_all)) if np.isfinite(y_all).any() else 0.0
                fill = [
                    float(np.median(c[np.isfinite(c)])) if np.isfinite(c).any() else own
                    for c in peers.T
                ]
                self.peer_fill_ = np.asarray(fill, dtype=np.float64)
                self.peer_columns_ = ["peer_same_side", "peer_opposite_side"]

        if not self.covariate_index_ and not self.peer_columns_:
            where = f" ({_preview(names)})" if names is not None else ""
            if self.require_physical:
                raise ValueError(
                    "thermal_residual_model: no physical operating-context covariate for target "
                    f"{self.target_name_ or self.target_index_!r}. This feature table carries none of "
                    f"{self._COV_WANTED}{where}, and the row context supplies no peer series "
                    "(adjacent / opposite axle-box temperature at the same unit and timestamp). "
                    "Regressing the target on statistics of its own sensor would make the residual a "
                    "different method, so it is refused: pass covariates=[...]/covariate_indices=[...] "
                    "explicitly, run this row on a table that carries the operating context, or set "
                    "require_physical=False to accept the non-physical fallback."
                )
            self.covariate_index_ = [j for j in range(n_f) if j != self.target_index_] or [
                self.target_index_
            ]
            self.covariate_names_ = (
                [str(names[j]) for j in self.covariate_index_] if names is not None else []
            )

        col = Xf[:, self.covariate_index_]
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(col, axis=0) if col.size else np.zeros(0)
        self.fill_ = np.where(np.isfinite(med), med, 0.0) if col.size else np.zeros(0)
        y, Z = self._design(Xf, t, series, unit)
        ok = np.isfinite(y)
        if ok.sum() < 3:
            y = np.where(ok, y, 0.0)
            ok = np.ones(y.size, dtype=bool)
        self.model_ = Ridge(alpha=self.alpha).fit(Z[ok], np.where(np.isfinite(y[ok]), y[ok], 0.0))
        self.center_, self.scale_ = robust_baseline(self.residual(Xf, t, series, unit))

    def residual(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        series: Any | None = None,
        unit: Any | None = None,
    ) -> np.ndarray:
        """``T_box - T_box_pred`` per row (raw kelvin-ish units, not standardised)."""
        y, Z = self._design(np.asarray(X, dtype=np.float64), t, series, unit)
        pred = np.asarray(self.model_.predict(Z), dtype=np.float64)
        return sanitise(np.where(np.isfinite(y), y - pred, 0.0))

    def standardised_residual(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        series: Any | None = None,
        unit: Any | None = None,
    ) -> np.ndarray:
        """The residual as a robust z against the healthy training residual."""
        return sanitise((self.residual(X, t, series, unit) - self.center_) / self.scale_)

    def _score(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        *,
        series: Any | None = None,
        unit: Any | None = None,
    ) -> np.ndarray:
        z = self.standardised_residual(X, t, series, unit)
        return z if self.signed else np.abs(z)


# ======================================================================================
# door - wavelet start peak
# ======================================================================================

#: Daubechies-4 (``db4``) decomposition low-pass filter, PyWavelets' ``dec_lo`` ordering.
_DB4_LO: Final[np.ndarray] = np.array(
    [
        -0.010597401784997278,
        0.032883011666982945,
        0.030841381835986965,
        -0.18703481171888114,
        -0.02798376941698385,
        0.6308807679295904,
        0.7148465705525415,
        0.23037781330885523,
    ],
    dtype=np.float64,
)
#: ``db2`` (Daubechies-4-tap) low-pass, offered as ``wavelet="db2"``.
_DB2_LO: Final[np.ndarray] = np.array(
    [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    dtype=np.float64,
)
_HAAR_LO: Final[np.ndarray] = np.array([0.7071067811865476, 0.7071067811865476], dtype=np.float64)
_WAVELETS: Final[dict[str, np.ndarray]] = {"db4": _DB4_LO, "db2": _DB2_LO, "haar": _HAAR_LO}


def _qmf(lo: np.ndarray) -> np.ndarray:
    """Quadrature-mirror high-pass from a decomposition low-pass filter."""
    hi = lo[::-1].copy()
    hi[1::2] *= -1.0
    return hi


def dwt_detail_energies(sig: np.ndarray, wavelet: str = "db4", n_levels: int = 9) -> np.ndarray:
    """``(n, L)`` -> ``(n, levels_used)`` of per-level detail energies (Mallat, periodised).

    A vectorised stand-in for ``pywt.wavedec`` (**PyWavelets is not installed in this
    environment**, see the model docstring): periodic extension, FFT convolution with the
    db4/db2/haar QMF pair, decimation by two, one energy per detail band. Decomposition
    stops when the approximation is shorter than the filter, so ``levels_used`` can be
    smaller than ``n_levels`` for short windows.
    """
    try:
        lo = _WAVELETS[wavelet]
    except KeyError:
        raise ValueError(f"dwt_detail_energies: wavelet must be one of {sorted(_WAVELETS)}, got {wavelet!r}") from None
    hi = _qmf(lo)
    a = np.asarray(sig, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    out: list[np.ndarray] = []
    f = lo.size
    for _ in range(int(n_levels)):
        if a.shape[1] < f:
            break
        ext = np.concatenate([a, a[:, : f - 1]], axis=1)  # periodisation: wrap the tail round
        approx = fftconvolve(ext, lo[None, ::-1], mode="valid", axes=1)[:, ::2]
        detail = fftconvolve(ext, hi[None, ::-1], mode="valid", axes=1)[:, ::2]
        out.append(np.sum(detail * detail, axis=1))
        a = approx
        if a.shape[1] < 2:
            break
    if not out:
        return np.zeros((sig.shape[0] if sig.ndim > 1 else 1, 0), dtype=np.float64)
    return np.stack(out, axis=1)


@register("dwt_w8w9_startpeak", input_kind="raw_window", family="time_frequency", task="ad")
class DwtW8W9StartPeak(AnomalyDetector):
    """[R66, R67] door-motor current: wavelet detail levels 8/9 plus the start-peak amplitude.

    The obstruction/wear signature of a door drive sits in the start transient: the two
    coarse detail bands (``w8``, ``w9``) carry the slow mechanical envelope and the first
    quarter of the stroke carries the breakaway current peak. Each feature is robust-z'd
    against healthy training and aggregated.

    The analysed channel is the **door motor current**, resolved by name from the channel
    names the runner passes (``current`` is channel 3 of both shipped door raw-window
    layouts: ``pos_ref, pos, vel, current, ...`` on the synthetic fleet and on Cranfield).
    Channel 0 there is ``pos_ref``, the commanded position, which carries no motor signature
    at all - so a bare index default is not safe and ``channel`` is a *name*.
    ``channel_index=<int>`` overrides it; with no channel names at all (direct use on a
    single-channel array) the documented fallback is channel 0.

    Defaults: ``channel="current"``, ``channel_index=None``, ``channel_names=None``,
    ``wavelet="db4"``, ``levels=(8, 9)``, ``start_frac=0.25``, ``use_start_peak=True``,
    ``aggregate="max"``, ``log_energy=True``.

    **Deviation from the ladder row** (``impl: pywt``): PyWavelets is *not installed* in the
    frozen environment and the brief forbids installing anything, so the db4 filter bank is
    applied directly with :func:`dwt_detail_energies` (numpy/scipy, periodised Mallat
    cascade - the same decomposition ``pywt.wavedec(..., mode="periodization")`` performs).
    Levels deeper than the window supports are clamped to the deepest available level and
    the effective list is exposed as ``self.levels_used_``.
    """

    _CHANNEL_PATTERNS: Final[tuple[str, ...]] = (r"^i_", r"current", r"motor")

    def __init__(
        self,
        *,
        channel: int | str = "current",
        channel_index: int | None = None,
        channel_names: Sequence[str] | None = None,
        wavelet: str = "db4",
        levels: Sequence[int] = (8, 9),
        start_frac: float = 0.25,
        use_start_peak: bool = True,
        aggregate: str = "max",
        log_energy: bool = True,
    ) -> None:
        super().__init__(
            channel=channel,
            channel_index=channel_index,
            channel_names=list(channel_names) if channel_names is not None else None,
            wavelet=wavelet,
            levels=list(levels),
            start_frac=start_frac,
            use_start_peak=use_start_peak,
            aggregate=aggregate,
            log_energy=log_energy,
        )
        if aggregate not in ("max", "mean", "sum"):
            raise ValueError(f"dwt_w8w9_startpeak: aggregate must be max|mean|sum, got {aggregate!r}")
        if not 0.0 < float(start_frac) <= 1.0:
            raise ValueError(f"dwt_w8w9_startpeak: start_frac must be in (0, 1], got {start_frac}")
        self.channel = channel
        self.channel_index = channel_index
        self.channel_names = list(channel_names) if channel_names is not None else None
        self.channel_: int = 0
        self.wavelet = str(wavelet)
        self.levels = [int(v) for v in levels]
        self.start_frac = float(start_frac)
        self.use_start_peak = bool(use_start_peak)
        self.aggregate = aggregate
        self.log_energy = bool(log_energy)
        self.levels_used_: list[int] = []
        self.center_: np.ndarray = np.zeros(0)
        self.scale_: np.ndarray = np.ones(0)

    def _features(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"dwt_w8w9_startpeak: expects raw_window (n, L, c), got {X.shape}")
        ch = self.channel_ % X.shape[2]
        sig = sanitise(X[:, :, ch], fill=0.0)
        want = max(self.levels)
        energies = dwt_detail_energies(sig, self.wavelet, want)
        n_avail = energies.shape[1]
        if not self.levels_used_:
            self.levels_used_ = [min(int(v), max(n_avail, 1)) for v in self.levels]
        cols = [energies[:, min(lv, n_avail) - 1] if n_avail else np.zeros(sig.shape[0]) for lv in self.levels_used_]
        feats = np.stack(cols, axis=1)
        if self.log_energy:
            feats = np.log1p(np.clip(feats, 0.0, None))
        if self.use_start_peak:
            head = max(int(round(self.start_frac * sig.shape[1])), 1)
            peak = np.max(np.abs(sig[:, :head]), axis=1)
            body = np.median(np.abs(sig), axis=1)
            ratio = peak / np.where(body > _SCALE_FLOOR, body, 1.0)
            feats = np.concatenate([feats, peak[:, None], ratio[:, None]], axis=1)
        return sanitise(feats)

    def _fit(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        *,
        feature_names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if X.ndim != 3:
            raise ValueError(f"dwt_w8w9_startpeak: expects raw_window (n, L, c), got {X.shape}")
        names = feature_names if feature_names is not None else self.channel_names
        self.channel_ = resolve_feature_index(
            self.channel,
            index=self.channel_index,
            feature_names=names,
            n_features=X.shape[2],
            patterns=self._CHANNEL_PATTERNS,
            required=X.shape[2] > 1,
            what="dwt_w8w9_startpeak.channel",
        )
        self.levels_used_ = []
        tab = self._features(X)
        stats = [robust_baseline(tab[:, j]) for j in range(tab.shape[1])]
        self.center_ = np.asarray([m for m, _ in stats])
        self.scale_ = np.asarray([s for _, s in stats])

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        z = np.abs(sanitise((self._features(X) - self.center_) / self.scale_))
        if self.aggregate == "max":
            return z.max(axis=1)
        if self.aggregate == "sum":
            return z.sum(axis=1)
        return z.mean(axis=1)


# ======================================================================================
# pneumatic - the exclusion rule
# ======================================================================================


@register("transition_mask", input_kind="raw_window", family="exclusion_rule", task="cpd")
class TransitionMask(AnomalyDetector):
    """[R105, R155, R87] **not a detector**: the COMP/Towers transition exclusion rule.

    The ladder row says exactly what this is: "emit ``is_transition`` and exclude +/-5 s
    around **Towers flips and COMP load/offload edges** from POINT scoring, scoring those
    regions with a change-point detector instead. Mask width is physically grounded: sump
    relief on unload ~40 s, reload repressurise ~3 s (R155)." All three clauses are here.

    **Which channels.** ``COMP`` (the compressor load/offload signal) and ``Towers`` (the
    drying-tower flip) are resolved **by name** from the channel names the runner passes as
    ``feature_names`` (``nebulax.bench.base.CONTEXT_KEYS``), so on the shipped pneumatic raw
    layout they land on channels 7 and 9 and on MetroPT-3 on its own ``COMP``/``Towers``
    channels. Picking whichever single channel happened to have the largest training step -
    TP2, a pressure trace, on real data - is not the rule the ladder states, so it is only
    the fallback for a table that carries **no channel names at all** (direct use on a bare
    array). A named channel missing from a named table raises ``ValueError``
    (``on_missing="drop"`` masks on the ones that did resolve, ``channels=[...]``/``channel=``
    pins indices outright).

    **How an edge is detected.** ``COMP``/``Towers`` are *digital*: a channel whose training
    values sit on at most ``max_levels`` distinct levels is treated as such and **any**
    within-window change larger than ``digital_tol`` (scaled by the level span) is a flip.
    An analog channel falls back to the robust step rule (``step_sigma`` MADs above the
    training median step). A row is a transition when *any* watched channel has an edge; the
    flag is then dilated by ``mask_width_s`` on both sides, converted to a row count with the
    measured row pitch of ``t`` (``metrics.row_pitch_seconds``) so it means seconds whatever
    the stride.

    **What the masked rows score.** Not zero. Zeroing was the bug: it threw away every
    genuine event that happened to coincide with a compressor edge and it made the row's
    output discontinuous. Instead there are two statistics, each robust-standardised against
    its own healthy training distribution so both are in "healthy sigmas":

    ``point``        the inner detector's score (``inner=<registered name>``), or the built-in
                     per-window level deviation ``|z|`` - used on **un**-masked rows;
    ``change-point`` a CUSUM of that level deviation, accumulated over the whole time-ordered
                     stream (a change-point statistic needs the continuity, so it is *run*
                     across the slice and *read off* inside the masked regions) - used on the
                     masked rows, which is the ladder's "scoring those regions with a
                     change-point detector instead".

    The two branches are combined and then min-max mapped through
    :func:`nebulax.bench.metrics.cpd_score`, like every other ``task: cpd`` row, so the
    emitted score is continuous in ``[0, 1]``. ``masked_score="constant"`` restores the legacy
    behaviour (every masked row gets ``masked_value``) as an explicit ablation.

    Defaults: ``inner=None``, ``inner_params=None``, ``channels=("COMP", "Towers")``,
    ``channel=None`` (pin a single index), ``channel_indices=None``, ``channel_names=None``,
    ``on_missing="error"``, ``mask_width_s=5.0``, ``step_sigma=4.0``, ``digital_tol=1e-6``,
    ``max_levels=4``, ``masked_score="changepoint"``, ``masked_value=0.0``, ``k=0.5``.
    """

    #: Running CUSUM and time dilation are sequential state: the runner scores one series at a
    #: time on a fresh copy, so interleaved components never share a mask or evidence.
    SEQUENTIAL: ClassVar[bool] = True

    #: The two ladder-named channels, and the patterns that find them in a named table.
    DEFAULT_CHANNELS: Final[tuple[str, ...]] = ("COMP", "Towers")
    _PATTERNS: Final[tuple[str, ...]] = (r"^COMP", r"^Towers")
    _MASKED_SCORES: Final[tuple[str, ...]] = ("changepoint", "constant")

    def __init__(
        self,
        *,
        inner: str | None = None,
        inner_params: dict[str, Any] | None = None,
        channels: Sequence[str | int] | None = DEFAULT_CHANNELS,
        channel: int | None = None,
        channel_indices: Sequence[int] | None = None,
        channel_names: Sequence[str] | None = None,
        on_missing: str = "error",
        mask_width_s: float = 5.0,
        step_sigma: float = 4.0,
        digital_tol: float = 1e-6,
        max_levels: int = 4,
        masked_score: str = "changepoint",
        masked_value: float = 0.0,
        k: float = 0.5,
        **inner_kwargs: Any,
    ) -> None:
        super().__init__(
            inner=inner,
            inner_params=dict(inner_params or {}),
            channels=list(channels) if channels is not None else None,
            channel=channel,
            channel_indices=list(channel_indices) if channel_indices is not None else None,
            channel_names=list(channel_names) if channel_names is not None else None,
            on_missing=on_missing,
            mask_width_s=mask_width_s,
            step_sigma=step_sigma,
            digital_tol=digital_tol,
            max_levels=max_levels,
            masked_score=masked_score,
            masked_value=masked_value,
            k=k,
            **inner_kwargs,
        )
        if masked_score not in self._MASKED_SCORES:
            raise ValueError(
                f"transition_mask: masked_score must be one of {self._MASKED_SCORES}, got {masked_score!r}"
            )
        self.inner = inner
        self.inner_params = dict(inner_params or {})
        self.inner_kwargs = dict(inner_kwargs)
        self.channels = list(channels) if channels is not None else None
        self.channel = channel
        self.channel_indices = list(channel_indices) if channel_indices is not None else None
        self.channel_names = list(channel_names) if channel_names is not None else None
        self.on_missing = on_missing
        self.mask_width_s = float(mask_width_s)
        self.step_sigma = float(step_sigma)
        self.digital_tol = float(digital_tol)
        self.max_levels = int(max_levels)
        self.masked_score = masked_score
        self.masked_value = float(masked_value)
        self.k = float(k)
        self.channels_: list[int] = []
        self.channel_names_: list[str] = []
        self.digital_: np.ndarray = np.zeros(0, dtype=bool)
        self.edge_threshold_: np.ndarray = np.zeros(0)
        self.level_center_: np.ndarray = np.zeros(0)
        self.level_scale_: np.ndarray = np.ones(0)
        self.point_center_: float = 0.0
        self.point_scale_: float = 1.0
        self.cp_center_: float = 0.0
        self.cp_scale_: float = 1.0
        self.inner_: BaseModel | None = None

    # -- transition detection -----------------------------------------------------------
    @property
    def channel_(self) -> int:
        """The first watched channel - kept so ``channel=<i>`` reads back as it used to."""
        return self.channels_[0] if self.channels_ else 0

    def _step_magnitude(self, X: np.ndarray, ch: int) -> np.ndarray:
        sig = sanitise(X[:, :, ch], fill=0.0)
        if sig.shape[1] < 2:
            return np.zeros(sig.shape[0])
        return np.max(np.abs(np.diff(sig, axis=1)), axis=1)

    def _levels(self, X: np.ndarray) -> np.ndarray:
        """Per-window mean of each watched channel, ``(n, len(channels_))``."""
        return sanitise(
            np.stack(
                [np.nanmean(sanitise(X[:, :, c % X.shape[2]], fill=np.nan), axis=1) for c in self.channels_],
                axis=1,
            )
        )

    def edge_flags(self, X: Any) -> np.ndarray:
        """``(n, len(channels_))`` boolean: did this channel have an edge inside this window?"""
        arr = np.asarray(X, dtype=np.float64)
        mags = np.stack([self._step_magnitude(arr, c % arr.shape[2]) for c in self.channels_], axis=1)
        return mags > np.maximum(self.edge_threshold_[None, :], _SCALE_FLOOR)

    def is_transition(self, X: Any, t: np.ndarray | None = None) -> np.ndarray:
        """Boolean per-row ``is_transition`` flag (the column the ladder row asks us to emit).

        True when **any** watched channel (COMP / Towers by default) has an edge inside the
        window, dilated by ``mask_width_s`` on both sides.
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(f"transition_mask: expects raw_window (n, L, c), got {arr.shape}")
        flag = self.edge_flags(arr).any(axis=1)
        t_s = times_seconds(t, arr.shape[0])
        pitch = float(M.row_pitch_seconds(t_s, fallback=1.0)) or 1.0
        half = int(math.ceil(self.mask_width_s / pitch)) if self.mask_width_s > 0 else 0
        if half <= 0 or not flag.any():
            return flag
        order, inv = _order_by_time(t_s)
        f = flag[order].astype(np.float64)
        kern = np.ones(2 * half + 1)
        dil = np.convolve(f, kern, mode="same") > 0.0
        return dil[inv]

    # -- model ---------------------------------------------------------------------------
    def _resolve_channels(self, X: np.ndarray, names: Sequence[str] | None) -> None:
        n_c = X.shape[2]
        if self.channel is not None:  # legacy single-channel pin
            self.channels_ = [int(self.channel) % n_c]
        elif self.channel_indices is not None:
            self.channels_ = [int(i) % n_c for i in self.channel_indices]
        elif names is not None and len(names) == n_c:
            self.channels_ = resolve_feature_indices(
                self.channels,
                feature_names=names,
                n_features=n_c,
                patterns=self._PATTERNS,
                default_k=n_c,
                on_missing=self.on_missing,
                what="transition_mask.channels",
            )
        else:  # no names anywhere: the documented fallback, the most step-like channel
            mags = [float(np.nanmax(self._step_magnitude(X, c), initial=0.0)) for c in range(n_c)]
            self.channels_ = [int(np.argmax(mags)) if mags else 0]
        self.channel_names_ = (
            [str(names[c]) for c in self.channels_] if names is not None and len(names) == n_c else []
        )

    def _calibrate_edges(self, X: np.ndarray) -> None:
        """Per-channel edge threshold: a level flip for a digital channel, robust for analog."""
        digital, thr = [], []
        for c in self.channels_:
            sig = sanitise(X[:, :, c % X.shape[2]], fill=0.0)
            levels = np.unique(np.round(sig, 9))
            is_digital = levels.size <= self.max_levels
            digital.append(is_digital)
            if is_digital:
                span = float(levels.max() - levels.min()) if levels.size > 1 else 0.0
                thr.append(max(0.5 * span, self.digital_tol))
            else:
                cen, sc = robust_baseline(self._step_magnitude(X, c % X.shape[2]))
                thr.append(cen + self.step_sigma * sc)
        self.digital_ = np.asarray(digital, dtype=bool)
        self.edge_threshold_ = np.asarray(thr, dtype=np.float64)

    def _fit(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        *,
        feature_names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if X.ndim != 3:
            raise ValueError(f"transition_mask: expects raw_window (n, L, c), got {X.shape}")
        names = feature_names if feature_names is not None else self.channel_names
        self._resolve_channels(X, names)
        self._calibrate_edges(X)
        lvl = self._levels(X)
        stats = [robust_baseline(lvl[:, j]) for j in range(lvl.shape[1])]
        self.level_center_ = np.asarray([m for m, _ in stats])
        self.level_scale_ = np.asarray([s for _, s in stats])

        mask = self.is_transition(X, t)
        if self.inner is not None:
            self.inner_ = _build_inner(self.inner, self.inner_params, self.inner_kwargs)
            keep = ~mask
            if keep.sum() < max(3, int(0.05 * X.shape[0])):
                keep = np.ones(X.shape[0], dtype=bool)
            t_keep = None if t is None else np.asarray(t).reshape(-1)[keep]
            self.inner_.fit(X[keep], t_keep)
        # both statistics are standardised against healthy training so the two branches of
        # the emitted score are in the same units ("healthy sigmas"), not two arbitrary scales
        self.point_center_, self.point_scale_ = robust_baseline(self._point_statistic(X, t))
        self.cp_center_, self.cp_scale_ = robust_baseline(self._changepoint_statistic(X, t))

    def _deviation(self, X: np.ndarray) -> np.ndarray:
        """Per-row ``max`` over watched channels of the standardised level deviation."""
        z = np.abs(sanitise((self._levels(X) - self.level_center_) / self.level_scale_))
        return z.max(axis=1)

    def _point_statistic(self, X: np.ndarray, t: np.ndarray | None) -> np.ndarray:
        """The POINT score: the inner detector's, or the built-in level deviation."""
        if self.inner_ is not None:
            return sanitise(np.asarray(self.inner_.score(X, t), dtype=np.float64))
        return self._deviation(X)

    def _changepoint_statistic(self, X: np.ndarray, t: np.ndarray | None) -> np.ndarray:
        """The CHANGE-POINT score for the masked regions: CUSUM of the level deviation.

        Run over the whole time-ordered slice - a cumulative statistic has no meaning
        restarted at every mask boundary - and read off inside the masked regions.
        """
        t_s = times_seconds(t, X.shape[0])
        order, inv = _order_by_time(t_s)
        return cusum(self._deviation(X)[order], self.k)[inv]

    def _score(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        mask = self.is_transition(X, t)
        point = sanitise((self._point_statistic(X, t) - self.point_center_) / self.point_scale_)
        if self.masked_score == "constant":  # legacy ablation: masked rows get a constant
            return np.where(mask, self.masked_value, M.cpd_score(X.shape[0], statistic=point))
        cp = sanitise((self._changepoint_statistic(X, t) - self.cp_center_) / self.cp_scale_)
        # both branches are already in healthy sigmas, so ONE min-max over the combined vector
        # keeps them comparable and still honours the cpd rows' [0, 1] score-adapter contract
        return M.cpd_score(X.shape[0], statistic=np.where(mask, cp, point))


# ======================================================================================
# wrappers
# ======================================================================================


class _PeerNormalisationBase(AnomalyDetector):
    """Shared machinery for the two peer-normalisation wrapper rows."""

    _DEFAULT_SAME_SIDE: ClassVar[bool] = False
    #: How the peer reference is pooled (:data:`PEER_STATISTICS`). The shared axle-box row
    #: overrides this to ``"median"``, which is what its ladder row requires.
    _DEFAULT_STATISTIC: ClassVar[str] = "mean"

    def __init__(
        self,
        *,
        inner: str | None = None,
        inner_params: dict[str, Any] | None = None,
        time_tol_s: float = 1.0,
        standardise: bool = True,
        statistic: str | None = None,
        keep_raw: bool = False,
        same_side: bool | None = None,
        side_from: str = "series",
        side_stride: int = 2,
        feature_indices: Sequence[int] | None = None,
        aggregate: str = "max",
        require_peers: bool = True,
        **inner_kwargs: Any,
    ) -> None:
        same = self._DEFAULT_SAME_SIDE if same_side is None else bool(same_side)
        stat = self._DEFAULT_STATISTIC if statistic is None else str(statistic)
        super().__init__(
            inner=inner,
            inner_params=dict(inner_params or {}),
            time_tol_s=time_tol_s,
            standardise=standardise,
            statistic=stat,
            keep_raw=keep_raw,
            same_side=same,
            side_from=side_from,
            side_stride=side_stride,
            feature_indices=list(feature_indices) if feature_indices is not None else None,
            aggregate=aggregate,
            require_peers=require_peers,
            **inner_kwargs,
        )
        if aggregate not in ("max", "mean", "sum"):
            raise ValueError(f"{self.name}: aggregate must be max|mean|sum, got {aggregate!r}")
        if stat not in PEER_STATISTICS:
            raise ValueError(f"{self.name}: statistic must be one of {PEER_STATISTICS}, got {stat!r}")
        self.inner = inner
        self.inner_params = dict(inner_params or {})
        self.inner_kwargs = dict(inner_kwargs)
        self.time_tol_s = float(time_tol_s)
        self.standardise = bool(standardise)
        self.statistic = stat
        self.keep_raw = bool(keep_raw)
        self.same_side = same
        self.side_from = _check_side_from(side_from, self.name)
        self.side_stride = max(int(side_stride), 1)
        self.feature_indices = list(feature_indices) if feature_indices is not None else None
        self.aggregate = aggregate
        self.require_peers = bool(require_peers)
        self.n_rows_with_peers_: int = 0
        self.cols_: list[int] = []
        self.inner_: BaseModel | None = None
        self.center_: np.ndarray = np.zeros(0)
        self.scale_: np.ndarray = np.ones(0)

    def _sides(self, t_s: np.ndarray, gid: np.ndarray, series: Any | None) -> np.ndarray | None:
        return sides_for(
            t_s, gid, series, same_side=self.same_side, side_from=self.side_from, stride=self.side_stride
        )

    def transform(
        self,
        X: np.ndarray,
        t: np.ndarray | None,
        series: Any | None = None,
        unit: Any | None = None,
    ) -> np.ndarray:
        """Peer-normalised feature matrix handed to the inner model."""
        Xf = sanitise(X, fill=np.nan)
        col = Xf[:, self.cols_]
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(col, axis=0)
        col = np.where(np.isfinite(col), col, np.where(np.isfinite(med), med, 0.0)[None, :])
        t_s = times_seconds(t, X.shape[0])
        gid = peer_group_ids(t_s, self.time_tol_s, unit=unit, series=series)
        delta = peer_delta(
            col,
            gid,
            side=self._sides(t_s, gid, series),
            standardise=self.standardise,
            statistic=self.statistic,
        )
        return np.concatenate([delta, col], axis=1).astype(np.float32) if self.keep_raw else delta.astype(np.float32)

    def _fit(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        *,
        series: Any | None = None,
        unit: Any | None = None,
        **kwargs: Any,
    ) -> None:
        n_f = _columns_of(X)
        self.cols_ = (
            [int(i) % n_f for i in self.feature_indices] if self.feature_indices is not None else list(range(n_f))
        )
        gid = peer_group_ids(times_seconds(t, X.shape[0]), self.time_tol_s, unit=unit, series=series)
        self.n_rows_with_peers_ = check_peers_exist(
            gid, series, unit=unit, what=self.name, require=self.require_peers
        )
        Z = self.transform(X, t, series, unit)
        if self.inner is not None:
            self.inner_ = _build_inner(self.inner, self.inner_params, self.inner_kwargs)
            self.inner_.fit(Z, t)
        else:
            stats = [robust_baseline(Z[:, j]) for j in range(Z.shape[1])]
            self.center_ = np.asarray([m for m, _ in stats])
            self.scale_ = np.asarray([s for _, s in stats])

    def _score(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        *,
        series: Any | None = None,
        unit: Any | None = None,
    ) -> np.ndarray:
        Z = self.transform(X, t, series, unit)
        if self.inner_ is not None:
            return sanitise(np.asarray(self.inner_.score(Z, t), dtype=np.float64))
        z = np.abs(sanitise((Z - self.center_) / self.scale_))
        if self.aggregate == "max":
            return z.max(axis=1)
        if self.aggregate == "sum":
            return z.sum(axis=1)
        return z.mean(axis=1)


@register("peer_normalisation", input_kind="cycle_features", family="wrapper", task="ad")
class PeerNormalisation(_PeerNormalisationBase):
    """[R78] fleet peer normalisation as a wrapper around any registered detector.

    Each row is replaced by its leave-one-out difference (``standardise=True``: z) against
    the *other* rows sharing its timestamp - the sibling doors of the same dwell - so a
    fleet-wide drift (temperature, duty, a software change) cancels and only a component's
    own deviation survives. ``inner=<registered name>`` is then fitted and scored on the
    normalised matrix; ``inner=None`` scores the peer z directly.

    Defaults: ``inner=None``, ``inner_params=None`` (extra ``**kwargs`` are forwarded to the
    inner too), ``time_tol_s=1.0``, ``standardise=True``, ``keep_raw=False``,
    ``same_side=False``, ``side_from="series"``, ``side_stride=2``,
    ``statistic="mean"`` (:data:`PEER_STATISTICS`), ``feature_indices=None``
    (all columns), ``aggregate="max"``, ``require_peers=True``.

    **Peer identity:** a peer is a row of the *same unit* at the *same timestamp* on a
    *different series* (:func:`peer_group_ids`, from the runner's ``series``/``unit``
    context), never merely "a row within ``time_tol_s``" - that pooled ten trains and two
    runs into one group on the shipped tables. The loader-side ``peer_norm=True`` axis
    remains the alternative when the normalisation should happen before caching.

    **Applicability, measured rather than assumed.** On the shipped synthetic fleet
    ``sim``/``door``/``cycle_features`` carries 20 door series over 10 trains - **two doors
    per train** (two runs of the same unit, e.g. ``T01`` = ``door_0000/door_L1`` +
    ``door_0010/door_L3``) - so the sibling set is *not* empty and this row fits normally
    there: on the temporal normal-only training slice it finds ~95 k peer-supported rows out
    of ~114 k. ``sim``/``bearing`` has eight boxes per train and works as intended too. An
    earlier version of this docstring claimed one door per train and that the row must raise
    on ``sim``/``door``; that was wrong and is retracted here.

    Note that the two doors of a train are not always on the same side (``T03`` is
    ``door_L3`` + ``door_R1``), which is why this row pools sides (``same_side=False``) and
    only the axle-box row :class:`PeerNormalisationShared` splits by side.

    What *does* leave a slice peerless is a slice holding a single series - a leave-one-unit-out
    fold that holds out one component, or a single-recording table; ``require_peers=True`` (the
    default) then raises at fit time instead of emitting a constant zero.
    """

    _DEFAULT_SAME_SIDE = False


@register("peer_normalisation_shared", input_kind="window_stats", family="wrapper", task="ad")
class PeerNormalisationShared(_PeerNormalisationBase):
    """[R78, R110, R112, R150, R151] the W1 same-side correction of :class:`PeerNormalisation`.

    Axle boxes must be compared against the median of the peers on the **same side** of the
    train, not pooled across the car: on a motor car bearing 1 carries ~13.5 % higher
    roller-raceway contact force than bearing 2 and runs measurably hotter, while the
    trailer-car boxes of one wheelset are thermally identical (R150/R151). Like-for-like
    positions only.

    The statistic is the **median** of those same-side peers, as the ladder row states, not a
    leave-one-out arithmetic mean: with four boxes a side, one box running away drags a mean
    reference up by a quarter of its own excursion (and a second degrading box corrupts the
    reference outright), while the median of the remaining three is untouched. See
    :func:`peer_delta` and :data:`PEER_STATISTICS`; ``statistic="mean"`` restores the pooled
    arithmetic form for an ablation.

    Defaults: as :class:`PeerNormalisation` but ``same_side=True`` and ``statistic="median"``.

    **Same-side identity:** the side is the **L/R letter of the component id** that the
    runner passes as ``series`` (``axlebox_1L``/``axlebox_3R``, ``door_L1``/``door_R4`` -
    see :func:`series_sides`). It is a property of the component, so shuffling equal-timestamp
    rows cannot change a single transformed value; the row-position parity this replaces
    could, and did. ``side_from="position"`` restores that legacy stride explicitly (with
    ``side_stride``) and ``same_side=False`` restores the pooled behaviour. A table whose
    series ids carry no side letter has one side, which makes the rule a documented no-op
    rather than a wrong answer.
    """

    _DEFAULT_SAME_SIDE = True
    _DEFAULT_STATISTIC = "median"


@register("k_of_n_corroboration", input_kind="window_stats", family="wrapper", task="ad")
class KOfNCorroboration(AnomalyDetector):
    """[R95] raise only when at least ``k`` of ``n`` independent channels agree.

    The ladder's example is "``>= 3`` of {TP2, TP3, Motor_current, Oil_temperature,
    idle_run_ratio} anomalous". The score is the **k-th largest member score**, so
    ``score > threshold`` holds exactly when at least ``k`` members are above that
    threshold - a continuous statistic that keeps the voting semantics intact for episode
    building and for AUPRC/VUS-PR.

    Members are either single feature columns (robust z against healthy training, the
    default) or one inner detector per member when ``inner=<registered name>`` is given -
    each inner instance is fitted on its own single-column view and its scores are
    robust-standardised so the vote compares like with like.

    **The default vote is the ladder row verbatim**:
    ``features=("TP2", "TP3", "Motor_current", "Oil_temperature", "idle_run_ratio")``,
    resolved against the runner's ``feature_names`` - so on MetroPT-3 window stats the
    members are the *levels* ``TP2_mean_mean``/``TP3_mean_mean``/``Motor_current_mean_mean``/
    ``Oil_temperature_mean_mean``, not eight arbitrary high-variance columns.

    ``idle_run_ratio`` is a **per-cycle** quantity and is genuinely absent from every
    ``window_stats`` table, so the fifth member falls back through the documented aliases
    ``duty_ratio`` -> ``COMP_duty`` (on MetroPT-3 window stats: ``COMP_duty_mean``). That is
    the same physical idle/run duty measured over the window instead of over the cycle, and
    the vote is on ``|z|``, so an inverse-monotone stand-in votes identically. A member that
    resolves to nothing at all raises ``ValueError`` naming it (``on_missing="drop"`` votes
    on the rest instead). ``member_names_`` records what each member resolved to.

    Defaults: ``k=3``, ``features=DEFAULT_MEMBERS``, ``feature_indices=None``,
    ``feature_names=None``, ``inner=None``, ``inner_params=None``, ``on_missing="error"``,
    ``signed=False`` (a channel is "anomalous" when it deviates in either direction),
    ``max_members=8``. With **no** feature names at all (direct use on a bare array) the
    members are the ``max_members`` highest-variance training columns, as before.
    """

    _PATTERNS: Final[tuple[str, ...]] = (
        r"^TP2",
        r"^TP3",
        r"motor_current",
        r"oil_temperature",
        r"idle_run_ratio",
    )
    #: R95's example vote, which is also this row's default.
    DEFAULT_MEMBERS: Final[tuple[str, ...]] = (
        "TP2",
        "TP3",
        "Motor_current",
        "Oil_temperature",
        "idle_run_ratio",
    )
    #: Documented stand-ins for a member a given feature table does not carry under its own
    #: name (the duty family: ``idle_run_ratio`` is per-cycle, ``COMP_duty`` per-window).
    MEMBER_ALIASES: Final[dict[str, tuple[str, ...]]] = {
        "idle_run_ratio": ("duty_ratio", "COMP_duty"),
        "Motor_current": ("I_loaded_mean", "current"),
        "Oil_temperature": ("T_oil_max", "T_oil"),
    }

    def __init__(
        self,
        *,
        k: int = 3,
        features: Sequence[str | int] | None = None,
        feature_indices: Sequence[int] | None = None,
        feature_names: Sequence[str] | None = None,
        inner: str | None = None,
        inner_params: dict[str, Any] | None = None,
        on_missing: str = "error",
        signed: bool = False,
        max_members: int = 8,
        **inner_kwargs: Any,
    ) -> None:
        members = list(features) if features is not None else list(self.DEFAULT_MEMBERS)
        super().__init__(
            k=k,
            features=members,
            feature_indices=list(feature_indices) if feature_indices is not None else None,
            feature_names=list(feature_names) if feature_names is not None else None,
            inner=inner,
            inner_params=dict(inner_params or {}),
            on_missing=on_missing,
            signed=signed,
            max_members=max_members,
            **inner_kwargs,
        )
        if int(k) < 1:
            raise ValueError(f"k_of_n_corroboration: k must be >= 1, got {k}")
        self.k = int(k)
        self.features = members
        self.feature_indices = list(feature_indices) if feature_indices is not None else None
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.inner = inner
        self.inner_params = dict(inner_params or {})
        self.inner_kwargs = dict(inner_kwargs)
        self.on_missing = on_missing
        self.signed = bool(signed)
        self.max_members = int(max_members)
        self.members_: list[int] = []
        self.member_names_: list[str] = []
        self.inners_: list[BaseModel] = []
        self.center_: np.ndarray = np.zeros(0)
        self.scale_: np.ndarray = np.ones(0)

    def _member_matrix(self, X: np.ndarray, t: np.ndarray | None) -> np.ndarray:
        col = sanitise(X[:, self.members_], fill=0.0)
        if not self.inners_:
            return col
        return np.stack(
            [sanitise(np.asarray(m.score(col[:, [j]].astype(np.float32), t))) for j, m in enumerate(self.inners_)],
            axis=1,
        )

    def _fit(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        *,
        feature_names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        n_f = _columns_of(X)
        names = feature_names if feature_names is not None else self.feature_names
        known = names is not None and len(names) == n_f
        with np.errstate(invalid="ignore"):
            var = np.nanvar(sanitise(X, fill=np.nan), axis=0)
        self.members_ = resolve_feature_indices(
            self.features if (known or self.feature_indices is not None) else None,
            indices=self.feature_indices,
            feature_names=names,
            n_features=n_f,
            patterns=self._PATTERNS,
            aliases=self.MEMBER_ALIASES,
            default_k=self.max_members,
            variance=var,
            on_missing=self.on_missing,
            what="k_of_n_corroboration.features",
        )
        self.member_names_ = [str(names[j]) for j in self.members_] if known else []
        self.inners_ = []
        if self.inner is not None:
            col = sanitise(X[:, self.members_], fill=0.0).astype(np.float32)
            for j in range(len(self.members_)):
                m = _build_inner(self.inner, self.inner_params, self.inner_kwargs)
                m.fit(col[:, [j]], t)
                self.inners_.append(m)
        Z = self._member_matrix(X, t)
        stats = [robust_baseline(Z[:, j]) for j in range(Z.shape[1])]
        self.center_ = np.asarray([m for m, _ in stats])
        self.scale_ = np.asarray([s for _, s in stats])

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        z = sanitise((self._member_matrix(X, t) - self.center_) / self.scale_)
        if not self.signed:
            z = np.abs(z)
        k = min(self.k, z.shape[1])
        # k-th largest per row: score > thr  <=>  at least k members > thr
        part = np.sort(z, axis=1)[:, ::-1]
        return part[:, k - 1]
