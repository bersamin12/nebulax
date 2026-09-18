"""Benchmark data loaders: one function per dataset, one :class:`BenchData` out.

The registry contract (``nebulax/bench/base.py``) says a model is handed ``X`` and nothing
else: ``(n, f)`` float32 for ``window_stats`` / ``cycle_features``, ``(n, L, c)`` float32 for
``raw_window``. Everything a model must *not* see - unit ids, ground-truth labels, ordinal
stages, the split groups - travels beside ``X`` on the :class:`BenchData`, never inside it.
``X`` is built from :func:`nebulax.schema.feature_columns` alone, so ``meta_*`` columns are
structurally excluded.

The four datasets
-----------------
``metropt3``  Porto APU. This module builds the **10-second aggregate table** the plan asks
              for (analog mean/max, digital duty fraction, ``is_transition`` from
              ``nebulax.adapters.metropt3.transition_mask``) on top of the adapter's ``long``,
              then windows it at 60 and 360 steps (10 min / 1 h). ``cycle_features`` is the
              adapter's own compressor-cycle table. Ground truth = the four air-leak episodes;
              a 24 h **post-repair mask** follows each one.
``cranfield`` Linear-actuator door proxy. ``cycle_features`` = one row per *test* (the 13
              conditions x profile x load x rep matrix), aggregated from the adapter's
              per-stroke rows; ``raw_window`` = 4 s windows at 25 Hz carrying the test id so a
              window-level model can be majority-voted to a test-level prediction.
``ottawa``    UORED-VAFCLS bearings. 0.25 s and 1 s windows of each 10 s record; ``raw_window``
              decimates 4x (42 kHz -> 10.5 kHz) so a deep model sees a tractable L. Groups are
              bearing ids, never records.
``sim``       The synthetic fleet, per subsystem. Peer-normalisation on/off and a train-time
              contamination knob (0 % / 5 %) are exposed here because both change ``X``, not
              the model.

Caching
-------
Every build is keyed by a hash of its arguments and cached under ``data/features/<hash>.npz``
(+ ``<hash>.labels.parquet`` + ``<hash>.json``). MetroPT-3 alone costs ~30 s and ~6 GB of peak
RSS to parse from CSV, so a sweep over 40 models must not pay it 40 times. ``X`` is float32
throughout and no loader ever holds two datasets at once - peak RSS stays under 12 GB.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from nebulax import schema as S
from nebulax.bench.metrics import Event, to_epoch_seconds

__all__ = [
    "BenchData",
    "DATASETS",
    "DEFAULT_CACHE_DIR",
    "load_bench",
    "load_metropt3",
    "load_cranfield",
    "load_ottawa",
    "load_sim",
    "SIM_SPLIT_UNIT_COLS",
    "SIM_FAULT_LOG",
    "PEER_TIME_TOLERANCE_S",
    "PEER_GROUP_COL",
    "PeerGrouping",
    "metropt3_aggregate",
    "train_rows",
    "majority_vote",
    "cache_key",
    "CACHE_VERSION",
    "LOADER_SEMANTICS",
    "CRANFIELD_CLS_TARGETS",
    "loader_accepted_kwargs",
    "opaque_series_id",
    "REQUIRED_MASKS",
    "sim_series_id",
]

LOGGER = logging.getLogger(__name__)

#: Datasets this module can build, in ladder order (A, B, C, D).
DATASETS: tuple[str, ...] = ("metropt3", "cranfield", "ottawa", "sim")
DEFAULT_CACHE_DIR: Path = Path("data/features")
#: Bumped whenever a loader's output changes shape or meaning, so an old cache entry can never
#: be mistaken for a current one. It is part of every cache key.
#: Bump this whenever the SHAPE of a cached BenchData changes - new label columns, new masks,
#: a different feature construction - so an old cache can never be read back as a current one.
#: It is part of :func:`cache_key`, so a bump simply misses every stale file and rebuilds.
#:
#: 2: labels carry :data:`PEER_GROUP_COL` when ``peer_norm=True``. A version-1 peer_norm cache
#:    has no such column, which would make ``splits._atomic_peer_supports`` silently do nothing
#:    and let a temporal cut fall inside a peer group - the exact leak that column exists to
#:    prevent. :func:`_cache_load` also refuses such a file outright, because a cache that is
#:    merely *missed* is a rebuild while a cache that is *misread* is a wrong number.
#: 3: ``meta["loader_kwargs"]`` is always written, so ``runner._check_axes_against_loader``
#:    can compare a RunSpec's axes with what the cached table was actually built from. A
#:    version-2 file may lack it, which made that check inert; such files are missed by the
#:    key and, belt and braces, refused by :func:`_cache_load`.
#: 4: the ``series`` array (component id per row) is stored; a version-3 file has none and
#:    would silently fall back to ``unit``, pooling sibling components again. Refused.
#: 5: the loaders' OUTPUT SEMANTICS are part of the key (:data:`LOADER_SEMANTICS`) and every
#:    dataset's required masks (:data:`REQUIRED_MASKS`) are checked on load. Version 4 covered
#:    the array *shape* only, so an entry written between two loader changes of the same shape
#:    (Cranfield series=rig vs series=recording; sim without post_failure) replayed silently.
CACHE_VERSION: int = 5

#: Per-dataset semantic version of what the loader emits. Bump the dataset's number whenever
#: its labels, events, series, masks or feature construction change meaning, even when no
#: array changes shape: it is folded into :func:`cache_key`, so old entries are simply missed.
#: sim 4: ``max_runs`` completes whole trains (a three-run request loads the sibling runs too).
#: cranfield/ottawa 3: opaque series ids (the recording id no longer spells the class).
LOADER_SEMANTICS: dict[str, int] = {"metropt3": 1, "cranfield": 3, "ottawa": 3, "sim": 4}

#: Masks a cached entry of each dataset must carry; a file without them predates the loader
#: that introduced them and is refused by :func:`_cache_load` (second line of defence).
REQUIRED_MASKS: dict[str, tuple[str, ...]] = {
    "metropt3": ("is_transition", "post_repair", "scoreable"),
    "sim": ("post_failure", "scoreable"),
    "cranfield": ("scoreable",),
    "ottawa": ("scoreable",),
}

#: MetroPT-3 aggregate bin, seconds. The file's own native cadence is ~10 s, so this is a
#: regularising re-bin (it removes jitter and gives a uniform time axis) rather than a
#: down-sampling: expect ~1 raw sample per bin.
METROPT_BIN_S: float = 10.0
#: A window spanning a logger gap longer than this is never emitted (MetroPT-3 has 331 gaps
#: over 60 s and 190 over 30 min).
METROPT_MAX_GAP_S: float = 60.0
#: Post-repair blanking after each MetroPT failure, per the plan.
POST_REPAIR_S: float = 24 * 3600.0
#: Default alarm-window horizon H (seconds) - 3 days, matching ``alarm_window_3d``.
DEFAULT_H_S: float = 3 * 86400.0


# --------------------------------------------------------------------------------------
# The container
# --------------------------------------------------------------------------------------


@dataclass
class BenchData:
    """One dataset, one ``input_kind``, everything the runner needs and nothing a model may see.

    Attributes
    ----------
    X : ``(n, f)`` float32 for ``window_stats`` / ``cycle_features``, ``(n, L, c)`` float32 for
        ``raw_window``. Built only from :func:`nebulax.schema.feature_columns`.
    feature_names : ``f`` names for a 2-D ``X``, ``c`` channel names for a 3-D one.
    t_start, t_end : ``(n,)`` ``datetime64[ms]``; ``t_end`` is what every metric and split uses.
    unit : ``(n,)`` object - ``train_id`` (MetroPT / sim), ``bearing_XX`` (Ottawa), the rig id
        (Cranfield). The false-alarm denominator (train-days) is counted per unit.
    group : ``(n,)`` object - the default grouping id for splits (run / bearing / test id).
    series : ``(n,)`` object - the finest time series the rows form, i.e. the **component**
        an alarm episode is raised on (``run_id/component_id`` on the synthetic fleet, where
        one train carries two doors or two bogies with eight axle boxes each; the unit
        elsewhere). Episode contiguity, the row pitch and event attribution are all per
        series, so three cycles pooled from two sibling doors are never "3 consecutive
        windows" and an alarm on a healthy sibling never detects its neighbour's fault.
        Defaults to ``unit``.
    labels : ``(n,)``-row frame with ``is_faulty, fault_type, severity, rul_s`` plus the
        dataset's ordinal ``stage``, its ``meta_*`` columns and the key columns splits group on.
        **Never** an input to a model.
    events : ground truth as ``(series, t_onset, t_failure)`` for the AD event metrics;
        ``Event.unit`` is matched against ``series``.
    window_seconds : nominal duration covered by one row, for the false-alarm budget.
    masks : named boolean masks, always including ``scoreable`` (rows that count towards test
        metrics), and per dataset ``is_transition`` / ``post_repair`` (MetroPT) and
        ``post_failure`` (synthetic fleet).
    meta : free loader metadata. ``meta["timeline"] == "fabricated"`` marks a dataset whose
        timestamps are synthetic anchors placed by the adapter (Cranfield's hourly tests,
        Ottawa's 20 s record spacing): train-days and the per-train-day false-alarm budget
        are meaningless there and the runner publishes them as not applicable.
    """

    dataset: str
    subsystem: str
    input_kind: str
    X: np.ndarray
    feature_names: list[str]
    t_start: np.ndarray
    t_end: np.ndarray
    unit: np.ndarray
    group: np.ndarray
    labels: pd.DataFrame
    events: list[Event]
    window_seconds: float
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    series: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = int(self.X.shape[0])
        if self.series is None:
            self.series = np.asarray(self.unit, dtype=object).reshape(-1).copy()
        for name in ("t_start", "t_end", "unit", "group", "series"):
            arr = np.asarray(getattr(self, name)).reshape(-1)
            if arr.size != n:
                raise ValueError(f"BenchData({self.dataset}): X has {n} rows but {name} has {arr.size}")
            setattr(self, name, arr)
        if len(self.labels) != n:
            raise ValueError(f"BenchData({self.dataset}): X has {n} rows but labels has {len(self.labels)}")
        if self.input_kind in ("window_stats", "cycle_features") and self.X.ndim != 2:
            raise ValueError(f"BenchData({self.dataset}): {self.input_kind} needs a 2-D X, got {self.X.shape}")
        if self.input_kind == "raw_window" and self.X.ndim != 3:
            raise ValueError(f"BenchData({self.dataset}): raw_window needs a 3-D X, got {self.X.shape}")
        self.masks.setdefault("scoreable", np.ones(n, dtype=bool))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(np.prod(self.X.shape[1:]))

    @property
    def y_binary(self) -> np.ndarray:
        """``is_faulty`` as an int8 array - the pointwise AD target."""
        return self.labels["is_faulty"].to_numpy(dtype=np.int8)

    @property
    def y_class(self) -> np.ndarray:
        """The CLS target as strings: ``labels["class_target"]`` when the loader set one (the
        Cranfield 13-class condition via ``cls_target="condition"``), else ``fault_type``."""
        col = "class_target" if "class_target" in self.labels.columns else "fault_type"
        return self.labels[col].astype(str).to_numpy()

    @property
    def stage(self) -> np.ndarray:
        """Ordinal severity stage (dataset-specific; NaN where the dataset has none)."""
        return self.labels["stage"].to_numpy(dtype=np.float64)

    @property
    def rul_s(self) -> np.ndarray:
        return self.labels["rul_s"].to_numpy(dtype=np.float64)

    def target(self, task: str) -> np.ndarray:
        """The supervised target for ``task`` (``cls`` -> fault type, ``rul`` -> ``rul_s``)."""
        if task == "cls":
            return self.y_class
        if task == "rul":
            return self.rul_s
        return self.y_binary

    def subset(self, idx: np.ndarray) -> "BenchData":
        """A row subset - used by the runner only for reporting; models get plain arrays."""
        i = np.asarray(idx, dtype=np.int64).reshape(-1)
        return BenchData(
            dataset=self.dataset,
            subsystem=self.subsystem,
            input_kind=self.input_kind,
            X=self.X[i],
            feature_names=list(self.feature_names),
            t_start=self.t_start[i],
            t_end=self.t_end[i],
            unit=self.unit[i],
            group=self.group[i],
            labels=self.labels.iloc[i].reset_index(drop=True),
            events=list(self.events),
            window_seconds=self.window_seconds,
            masks={k: v[i] for k, v in self.masks.items()},
            meta=dict(self.meta),
            series=self.series[i],
        )

    def events_in(self, idx: np.ndarray, H: float = 0.0) -> list[Event]:
        """The ground-truth events whose **detection window** ``[t_onset - H, t_failure]``
        overlaps the time span of ``idx``.

        This is what makes "recall over N events" honest per slice: the report prints the number
        of events actually present in the slice being scored, not the dataset's total. On
        MetroPT-3 the ``metropt_temporal`` test slice (11 April onward) happens to contain all
        four air-leak episodes, because all four begin after the cut; the ``metropt_contaminated``
        test slice (4 June onward) contains two of them. Neither number is assumed anywhere - it
        is counted per slice, which is the whole point of this method.

        ``H`` must be the same horizon the metrics are scored with, because that is what
        defines the window. With ``H = 0`` (the default, for callers that only want events
        that have already begun) a slice that ends inside an event's run-up - the last three
        days before a failure that falls just after the cut - is credited with **no** event,
        while :func:`nebulax.bench.metrics.event_metrics` would happily detect one there. The
        denominator and the numerator of the recall figure then disagree. The runner passes
        ``spec.H``.
        """
        i = np.asarray(idx, dtype=np.int64).reshape(-1)
        if i.size == 0:
            return []
        t = to_epoch_seconds(self.t_end)[i]
        u = np.asarray(self.series)[i]
        h = float(H) if np.isfinite(H) else 0.0
        # Bounds PER SERIES (component). A single global [min, max] over the slice would credit a fleet with
        # every unit's events whenever any unit was observed at that time - on a
        # leave-one-unit-out fold the recall denominator would then count failures on trains
        # that are not in the fold at all.
        bounds: dict[Any, tuple[float, float]] = {}
        for unit in pd.unique(u):
            tu = t[u == unit]
            tu = tu[np.isfinite(tu)]
            if tu.size:
                bounds[unit] = (float(tu.min()), float(tu.max()))
        out = []
        for ev in self.events:
            span = bounds.get(ev.unit)
            if span is None:
                continue
            lo, hi = span
            end = ev.t_failure if np.isfinite(ev.t_failure) else np.inf
            if end >= lo and (ev.t_onset - h) <= hi:
                out.append(ev)
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "subsystem": self.subsystem,
            "input_kind": self.input_kind,
            "n_rows": len(self),
            "X_shape": list(self.X.shape),
            "n_features": self.n_features,
            "n_units": int(pd.unique(self.unit).size),
            "n_groups": int(pd.unique(self.group).size),
            "n_events": len(self.events),
            "n_faulty": int(self.y_binary.sum()),
            "window_seconds": self.window_seconds,
            "t_min": str(pd.Timestamp(self.t_end.min())) if len(self) else None,
            "t_max": str(pd.Timestamp(self.t_end.max())) if len(self) else None,
        }


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _as_ms(t: Any) -> np.ndarray:
    """Any timestamp-ish array -> ``datetime64[ms]`` (tz dropped; everything here is UTC)."""
    if isinstance(t, (pd.Series, pd.Index)):
        t = pd.DatetimeIndex(t)
        return (t.tz_localize(None) if t.tz is not None else t).to_numpy().astype("datetime64[ms]")
    arr = np.asarray(t)
    if arr.dtype.kind == "M":
        return arr.astype("datetime64[ms]")
    return pd.to_datetime(pd.Series(arr), utc=True).dt.tz_localize(None).to_numpy().astype("datetime64[ms]")


def sim_series_id(run_id: Any, component_id: Any) -> np.ndarray:
    """``run_id/component_id`` - the series key of the synthetic fleet (one door, one axle box)."""
    r = np.asarray(run_id, dtype=object).reshape(-1).astype(str)
    c = np.asarray(component_id, dtype=object).reshape(-1).astype(str)
    return np.char.add(np.char.add(r, "/"), c).astype(object)


def opaque_series_id(ids: Any) -> np.ndarray:
    """Recording ids that do not spell the class.

    Cranfield's and Ottawa's ``run_id`` literally contain the condition (``cranfield_Normal_...``,
    ``ottawa_B_11_1``), and the runner hands ``series`` to any model that declares it - a type
    code that transfers across the bearing / rep / profile fence. The series id is therefore a
    short hash of the run id: still one id per recording, matched by events through the same
    function, but carrying no label.
    """
    arr = np.asarray(ids, dtype=object).reshape(-1).astype(str)
    return np.array(["rec_" + hashlib.sha256(x.encode()).hexdigest()[:12] for x in arr], dtype=object)


def _events_from_fault_log(fault_log: pd.DataFrame, unit_col: str | None = "train_id") -> list[Event]:
    """Fault-log rows -> :class:`~nebulax.bench.metrics.Event` list.

    ``unit_col`` is the column the event is keyed by and must be the loader's ``series``
    (``train_id`` for MetroPT, ``run_id`` = the recording for Cranfield and Ottawa);
    ``unit_col=None`` keys by :func:`sim_series_id` (``run_id/component_id``) so a synthetic
    event is attributed to the component that actually failed, not to every component of
    the train. Rows whose ``fault_type`` is ``healthy`` (Cranfield writes one fault-log row
    per test, healthy tests included) are not events: they would otherwise enter the recall
    denominator as failures that no detector can find.
    """
    if fault_log is None or fault_log.empty:
        return []
    out: list[Event] = []
    for row in fault_log.itertuples():
        if str(getattr(row, "fault_type", "")).lower() == "healthy":
            continue
        if unit_col is None:
            key = str(sim_series_id(row.run_id, row.component_id)[0])
        elif unit_col == "run_id":
            key = str(opaque_series_id([row.run_id])[0])  # matches the recording's opaque series id
        else:
            key = getattr(row, unit_col)
        out.append(
            Event.from_any(
                key,
                getattr(row, "t_onset"),
                getattr(row, "t_failure"),
                getattr(row, "fault_type", "unknown"),
            )
        )
    return out


def _label_rows(
    t_end: np.ndarray,
    unit: np.ndarray,
    events: Sequence[Event],
    *,
    H: float = DEFAULT_H_S,
    default_fault_type: str = "healthy",
) -> pd.DataFrame:
    """Per-row ``is_faulty / fault_type / rul_s / alarm_window`` from a list of events.

    A row is faulty when its ``t_end`` lies inside ``[t_onset, t_failure]`` of an event on its
    own unit (open-ended when ``t_failure`` is NaN); it is in the alarm window from
    ``t_onset - H``. ``rul_s`` counts down to ``t_failure`` and is **NaN** when the event has
    no failure time - unknown is unknown, never zero (schema convention).
    """
    t = to_epoch_seconds(t_end).reshape(-1)
    u = np.asarray(unit, dtype=object).reshape(-1)
    n = t.size
    is_faulty = np.zeros(n, dtype=bool)
    alarm = np.zeros(n, dtype=bool)
    rul = np.full(n, np.nan, dtype=np.float64)
    ftype = np.full(n, default_fault_type, dtype=object)
    for ev in events:
        same = u == ev.unit
        end = ev.t_failure if np.isfinite(ev.t_failure) else np.inf
        inside = same & (t >= ev.t_onset) & (t <= end)
        is_faulty |= inside
        alarm |= same & (t >= ev.t_onset - H) & (t <= end)
        ftype[inside] = ev.fault_type
        if np.isfinite(ev.t_failure):
            rul[inside] = np.minimum(np.where(np.isnan(rul[inside]), np.inf, rul[inside]), ev.t_failure - t[inside])
    return pd.DataFrame(
        {
            "is_faulty": is_faulty,
            "fault_type": pd.Series(ftype, dtype="object").astype(str),
            "severity": np.where(is_faulty, np.nan, 0.0).astype(np.float32),
            "rul_s": rul.astype(np.float32),
            "alarm_window": alarm,
        }
    )


def _numeric_matrix(df: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """``(n, f)`` float32 from ``cols``, expanding equal-length list columns into ``col_0..``.

    Vector features (the simulator's ``current_profile_50``) are legitimate model inputs but
    cannot live in a float matrix as lists; unequal-length or non-numeric columns are dropped
    and named in the log rather than silently coerced.
    """
    blocks: list[np.ndarray] = []
    names: list[str] = []
    dropped: list[str] = []
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            blocks.append(s.to_numpy(dtype=np.float32).reshape(-1, 1))
            names.append(c)
        elif pd.api.types.is_bool_dtype(s):
            blocks.append(s.to_numpy(dtype=np.float32).reshape(-1, 1))
            names.append(c)
        elif s.dtype == object and len(s) and isinstance(s.iloc[0], (list, tuple, np.ndarray)):
            lens = {len(v) for v in s if v is not None}
            if len(lens) == 1:
                k = lens.pop()
                blocks.append(np.stack([np.asarray(v, dtype=np.float32) for v in s]).reshape(-1, k))
                names.extend(f"{c}_{i}" for i in range(k))
            else:
                dropped.append(f"{c} (ragged list lengths {sorted(lens)[:4]})")
        else:
            dropped.append(f"{c} ({s.dtype})")
    if dropped:
        LOGGER.info("bench.data: dropped non-numeric feature column(s): %s", ", ".join(dropped))
    if not blocks:
        raise ValueError("bench.data: no numeric feature columns survived; nothing to build X from")
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False), names


def _finite(X: np.ndarray) -> np.ndarray:
    """Replace non-finite values with 0.0 in place-ish. Models must never see NaN in ``X``."""
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def majority_vote(pred: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Window-level predictions -> one prediction per group (Cranfield's test-level vote).

    Returns ``(group_ids, voted_predictions)`` with group ids in first-appearance order. Ties
    go to the label that appears first in sorted order, which is deterministic and stated
    rather than left to dict insertion luck.
    """
    p = np.asarray(pred).reshape(-1)
    g = np.asarray(groups, dtype=object).reshape(-1)
    if p.size != g.size:
        raise ValueError(f"majority_vote: pred has {p.size} rows but groups has {g.size}")
    uniq = pd.unique(g)
    out = []
    for key in uniq:
        vals, counts = np.unique(p[g == key].astype(str), return_counts=True)
        out.append(vals[np.lexsort((vals, -counts))[0]])
    return np.asarray(uniq, dtype=object), np.asarray(out)


def train_rows(
    data: BenchData,
    train_idx: np.ndarray,
    *,
    train_regime: str = "normal_only",
    contamination: float = 0.0,
    seed: int = 0,
    task: str = "ad",
) -> np.ndarray:
    """Filter a split's training indices according to the training regime.

    ``normal_only``   drop every faulty row, then add back a ``contamination`` fraction of
                      them (0.0 = a clean semi-supervised fit, 0.05 = the ladder's 5 %
                      contamination ablation). The fraction is of the *kept* training set, so
                      ``contamination=0.05`` means 5 % of the rows a model fits on are faulty.
    ``all``           every training row, labels ignored (genuinely unsupervised / supervised).

    ``task`` guards the regime: ``normal_only`` is a **semi-supervised anomaly-detection**
    idea - fit the model on healthy behaviour and call the rest anomalous - and it is
    meaningless, in fact destructive, for a supervised task. Applying it to ``cls`` deletes
    every faulty row and leaves a classifier with one class to learn; applying it to ``rul``
    deletes exactly the run-to-failure rows the regressor exists to fit. For ``cls`` and
    ``rul`` the regime is therefore ignored (with a log line) and every training row is kept;
    ``rul`` additionally drops rows whose target is not finite, which no regressor can use.

    Returns a sorted int64 index array; never touches validation or test rows.
    """
    idx = np.asarray(train_idx, dtype=np.int64).reshape(-1)
    if task in ("cls", "rul"):
        if train_regime == "normal_only":
            LOGGER.info(
                "train_rows: train_regime='normal_only' ignored for task=%s - dropping the faulty "
                "rows would remove the very rows a supervised model must learn from",
                task,
            )
        if task == "rul":
            keep = np.isfinite(data.rul_s[idx])
            if not keep.all():
                LOGGER.info("train_rows(rul): dropping %d training row(s) with a non-finite target", int((~keep).sum()))
            idx = idx[keep]
        return np.sort(idx)
    if train_regime == "all":
        return np.sort(idx)
    if train_regime != "normal_only":
        raise ValueError(f"train_rows: unknown train_regime {train_regime!r}; expected 'normal_only' or 'all'")
    faulty = data.y_binary[idx].astype(bool)
    clean, dirty = idx[~faulty], idx[faulty]
    if contamination <= 0 or dirty.size == 0:
        return np.sort(clean)
    n_add = int(round(contamination * clean.size / max(1e-9, 1.0 - contamination)))
    n_add = min(n_add, dirty.size)
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([clean, rng.choice(dirty, size=n_add, replace=False)]))


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------


def cache_key(dataset: str, **kwargs: Any) -> str:
    """Stable 16-hex hash of ``(dataset, sorted kwargs)`` - the cache file stem."""
    payload = json.dumps(
        {"v": CACHE_VERSION, "sem": LOADER_SEMANTICS.get(dataset, 0), "dataset": dataset, **{k: kwargs[k] for k in sorted(kwargs)}},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path, Path]:
    return cache_dir / f"{key}.npz", cache_dir / f"{key}.labels.parquet", cache_dir / f"{key}.json"


def _cache_load(cache_dir: Path, key: str) -> BenchData | None:
    npz_p, lab_p, meta_p = _cache_paths(cache_dir, key)
    if not (npz_p.exists() and lab_p.exists() and meta_p.exists()):
        return None
    try:
        with np.load(npz_p, allow_pickle=True) as z:
            arrays = {k: z[k] for k in z.files}
        labels = pd.read_parquet(lab_p, engine="pyarrow")
        meta = json.loads(meta_p.read_text())
    except Exception as exc:  # pragma: no cover - a corrupt cache must never fail a sweep
        LOGGER.warning("bench.data: ignoring unreadable cache %s (%s)", npz_p, exc)
        return None
    if meta.get("meta", {}).get("peer_norm") and PEER_GROUP_COL not in labels.columns:
        # Belt and braces behind CACHE_VERSION: a peer-normalised table without the group id
        # cannot be split safely, so refuse it rather than silently skipping the atomic
        # placement that keeps a temporal cut out of a peer group.
        LOGGER.warning(
            "bench.data: ignoring stale peer_norm cache %s - it predates %s and cannot be split "
            "atomically; rebuilding",
            npz_p,
            PEER_GROUP_COL,
        )
        return None
    if "series" not in arrays:
        LOGGER.warning("bench.data: ignoring cache %s without the series array; rebuilding", npz_p)
        return None
    have_masks = {k[len("mask_") :] for k in arrays if k.startswith("mask_")}
    missing = [m for m in REQUIRED_MASKS.get(str(meta.get("dataset")), ()) if m not in have_masks]
    if missing:
        LOGGER.warning("bench.data: ignoring cache %s without required mask(s) %s; rebuilding", npz_p, missing)
        return None
    if not isinstance(meta.get("meta", {}).get("loader_kwargs"), dict):
        LOGGER.warning(
            "bench.data: ignoring cache %s without meta.loader_kwargs - the runner could not "
            "check its axes against the spec; rebuilding",
            npz_p,
        )
        return None
    masks = {k[len("mask_") :]: arrays[k] for k in arrays if k.startswith("mask_")}
    return BenchData(
        dataset=meta["dataset"],
        subsystem=meta["subsystem"],
        input_kind=meta["input_kind"],
        X=arrays["X"],
        feature_names=list(meta["feature_names"]),
        t_start=arrays["t_start"].astype("datetime64[ms]"),
        t_end=arrays["t_end"].astype("datetime64[ms]"),
        unit=arrays["unit"].astype(object),
        group=arrays["group"].astype(object),
        labels=labels,
        events=[Event(**e) for e in meta["events"]],
        window_seconds=float(meta["window_seconds"]),
        masks=masks,
        meta={**meta.get("meta", {}), "cache_key": key, "cache_hit": True},
        series=arrays["series"].astype(object),
    )


def _cache_store(cache_dir: Path, key: str, data: BenchData) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_p, lab_p, meta_p = _cache_paths(cache_dir, key)
    arrays: dict[str, Any] = {
        "X": data.X,
        "t_start": data.t_start.astype("datetime64[ms]"),
        "t_end": data.t_end.astype("datetime64[ms]"),
        "unit": np.asarray(data.unit, dtype=str),
        "group": np.asarray(data.group, dtype=str),
        "series": np.asarray(data.series, dtype=str),
    }
    arrays.update({f"mask_{k}": v for k, v in data.masks.items()})
    np.savez(npz_p, **arrays)
    data.labels.to_parquet(lab_p, engine="pyarrow", compression="zstd", index=False)
    meta_p.write_text(
        json.dumps(
            {
                "dataset": data.dataset,
                "subsystem": data.subsystem,
                "input_kind": data.input_kind,
                "feature_names": list(data.feature_names),
                "window_seconds": float(data.window_seconds),
                "events": [e.as_dict() for e in data.events],
                "meta": {k: v for k, v in data.meta.items() if _jsonable(v)},
            },
            indent=2,
            default=str,
        )
    )


def _jsonable(v: Any) -> bool:
    try:
        json.dumps(v, default=str)
        return True
    except Exception:  # pragma: no cover
        return False


# --------------------------------------------------------------------------------------
# A: MetroPT-3
# --------------------------------------------------------------------------------------

_METRO_ANALOG: tuple[str, ...] = (
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
)
_METRO_DIGITAL: tuple[str, ...] = (
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
)


def metropt3_aggregate(long: pd.DataFrame, *, bin_s: float = METROPT_BIN_S) -> pd.DataFrame:
    """The plan's **10-second aggregate table**, built from the adapter's ``long`` telemetry.

    One row per ``bin_s``-second bin with

    * ``<sig>_mean`` and ``<sig>_max`` for each analog channel,
    * ``<sig>_duty`` (the fraction of the bin the channel is 1) for each digital channel,
    * ``is_transition`` - ``True`` when any sample in the bin is flagged by
      :func:`nebulax.adapters.metropt3.transition_mask` (compressor state change, ``Towers``
      flip or ``COMP`` edge),
    * ``n_samples`` - how many raw samples the bin actually had, so a bin that is really a
      logger gap is visible rather than indistinguishable from a quiet one.

    The reduction is a ``reduceat`` over the sorted time axis, not a ``groupby``: at 1.5 M
    rows and ~1.5 M bins (the file's native cadence is already ~10 s, so this bin is a
    regulariser) ``groupby`` would dominate the load time.
    """
    from nebulax.adapters.metropt3 import transition_mask

    wide = S.to_wide(long, "pneumatic").sort_values("timestamp", kind="stable").reset_index(drop=True)
    if wide.empty:
        raise ValueError("metropt3_aggregate: no pneumatic rows in the long table")
    trans = transition_mask(wide)

    t_s = to_epoch_seconds(wide["timestamp"])
    bins = np.floor(t_s / float(bin_s)).astype(np.int64)
    edges = np.flatnonzero(np.r_[True, np.diff(bins) != 0])
    counts = np.diff(np.r_[edges, bins.size]).astype(np.float64)

    out: dict[str, np.ndarray] = {}
    for sig in _METRO_ANALOG:
        v = wide[sig].to_numpy(dtype=np.float64) if sig in wide.columns else np.full(bins.size, np.nan)
        ok = np.isfinite(v)
        c = np.add.reduceat(ok.astype(np.float64), edges)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{sig}_mean"] = np.where(c > 0, np.add.reduceat(np.where(ok, v, 0.0), edges) / np.where(c == 0, 1, c), np.nan)
        mx = np.maximum.reduceat(np.where(ok, v, -np.inf), edges)
        out[f"{sig}_max"] = np.where(c > 0, mx, np.nan)
    for sig in _METRO_DIGITAL:
        v = wide[sig].to_numpy(dtype=np.float64) if sig in wide.columns else np.full(bins.size, np.nan)
        ok = np.isfinite(v)
        c = np.add.reduceat(ok.astype(np.float64), edges)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{sig}_duty"] = np.where(c > 0, np.add.reduceat(np.where(ok, v, 0.0), edges) / np.where(c == 0, 1, c), np.nan)

    t_bin = (bins[edges] * float(bin_s) * 1000.0).astype("int64").astype("datetime64[ms]")
    agg = pd.DataFrame(out)
    agg.insert(0, "timestamp", t_bin)
    agg["is_transition"] = np.maximum.reduceat(trans.astype(np.float64), edges) > 0
    agg["n_samples"] = counts
    agg["train_id"] = str(wide["train_id"].iloc[0])
    agg["component_id"] = str(wide["component_id"].iloc[0])
    agg["run_id"] = str(wide["run_id"].iloc[0])
    return agg


def load_metropt3(
    *,
    input_kind: str = "window_stats",
    window: int = 60,
    stride: int | None = None,
    feature_set: str = "all",
    H: float = DEFAULT_H_S,
    raw_dir: str | Path | None = None,
    transition_frac_max: float = 0.5,
    **_: Any,
) -> BenchData:
    """MetroPT-3 (dataset **A**): the Porto APU air-leak benchmark.

    ``input_kind="window_stats"``   :func:`nebulax.features.stats.window_stats` over the 10-s
        aggregate at ``window`` = 60 (10 min) or 360 (1 h) steps. ``window=1`` is also legal
        and yields the aggregate rows themselves, which is the only form where the
        ``is_transition`` exclusion bites at sample resolution.
    ``input_kind="raw_window"``     the same windows, un-summarised: ``(n, window, c)``.
    ``input_kind="cycle_features"`` the adapter's compressor-cycle table (10,395 rows).

    ``feature_set`` is ``"all"`` (analog mean+max and digital duty), ``"analog"`` (mean+max
    only) or ``"mean"`` (one column per channel). Ground truth is the four air-leak episodes;
    ``masks["post_repair"]`` blanks 24 h after each ``t_failure`` and ``masks["is_transition"]``
    flags windows whose transition fraction exceeds ``transition_frac_max``.

    **Never treat the cycle table as a supervised positive class**: only 6 of its 10,395 rows
    carry ``is_faulty`` because one leak episode collapses into a single multi-day cycle. This
    is an anomaly-detection dataset scored on the four events.
    """
    from nebulax.adapters import metropt3 as adapter

    ds = adapter.load(Path(raw_dir) if raw_dir else Path("data/raw/metropt3"))
    events = _events_from_fault_log(ds.fault_log)

    if input_kind == "cycle_features":
        feats = ds.features
        cols = S.feature_columns(feats)
        X, names = _numeric_matrix(feats, cols)
        t_start, t_end = _as_ms(feats["t_start"]), _as_ms(feats["t_end"])
        unit = feats["train_id"].astype(str).to_numpy(dtype=object)
        labels = _label_rows(t_end, unit, events, H=H)
        labels["stage"] = np.nan
        labels["run_id"] = feats["run_id"].astype(str).to_numpy()
        labels["train_id"] = unit
        dur = (to_epoch_seconds(t_end) - to_epoch_seconds(t_start))
        return BenchData(
            dataset="metropt3",
            subsystem="pneumatic",
            input_kind="cycle_features",
            X=_finite(X),
            feature_names=names,
            t_start=t_start,
            t_end=t_end,
            unit=unit,
            group=unit.copy(),
            labels=labels,
            events=events,
            window_seconds=float(np.nanmedian(dur)) if dur.size else 0.0,
            masks=_metro_masks(
                t_end,
                unit,
                events,
                feats["transition_frac"].to_numpy(dtype=np.float64)
                if "transition_frac" in feats
                else np.zeros(len(feats)),
                transition_frac_max,
            ),
            meta={"source": "compressor-cycle table", "adapter_meta_keys": sorted(ds.meta)},
        )

    agg = metropt3_aggregate(ds.long)
    del ds
    channels = _metro_channels(feature_set)
    from nebulax.features.stats import feature_names as stat_feature_names
    from nebulax.features.stats import window_stats
    from nebulax.features.windows import make_windows

    L = int(window)
    if L < 1:
        raise ValueError(f"load_metropt3: window must be >= 1, got {window}")
    step = int(stride) if stride is not None else max(1, L // 2)

    if L == 1:
        keep = agg[[*channels, "is_transition"]].to_numpy(dtype=np.float32)
        Xw = keep[:, None, :]
        t_end = _as_ms(agg["timestamp"]) + np.timedelta64(int(METROPT_BIN_S * 1000), "ms")
        t_start = _as_ms(agg["timestamp"])
        unit = agg["train_id"].astype(str).to_numpy(dtype=object)
    else:
        W = make_windows(
            agg[["timestamp", "train_id", "component_id", "run_id", *channels, "is_transition"]],
            [*channels, "is_transition"],
            L=L,
            stride=step,
            fs=1.0 / METROPT_BIN_S,
            max_gap_s=METROPT_MAX_GAP_S,
        )
        if len(W) == 0:
            raise ValueError("load_metropt3: no windows survived gap splitting; check window/stride")
        Xw = W.X
        t_end = _as_ms(W.t_end)
        t_start = t_end - np.timedelta64(int(L * METROPT_BIN_S * 1000), "ms")
        unit = np.asarray(W.train_id, dtype=object)

    trans_frac = np.nanmean(Xw[:, :, -1].astype(np.float64), axis=1)
    X_raw = np.ascontiguousarray(Xw[:, :, :-1], dtype=np.float32)

    if input_kind == "raw_window":
        X, names = X_raw, list(channels)
    elif input_kind == "window_stats":
        if L == 1:
            X, names = X_raw[:, 0, :], list(channels)
        else:
            X = window_stats(X_raw)
            names = stat_feature_names(channels)
    else:
        raise ValueError(f"load_metropt3: unknown input_kind {input_kind!r}")

    labels = _label_rows(t_end, unit, events, H=H)
    labels["stage"] = np.nan
    labels["transition_frac"] = trans_frac.astype(np.float32)
    labels["train_id"] = unit
    return BenchData(
        dataset="metropt3",
        subsystem="pneumatic",
        input_kind=input_kind,
        X=_finite(np.asarray(X, dtype=np.float32)),
        feature_names=list(names),
        t_start=t_start,
        t_end=t_end,
        unit=unit,
        group=unit.copy(),
        labels=labels,
        events=events,
        window_seconds=float(L * METROPT_BIN_S),
        masks=_metro_masks(t_end, unit, events, trans_frac, transition_frac_max),
        meta={"aggregate_bin_s": METROPT_BIN_S, "window": L, "stride": step, "feature_set": feature_set},
    )


def _metro_channels(feature_set: str) -> list[str]:
    if feature_set in ("all", "default"):
        return [f"{s}_{k}" for s in _METRO_ANALOG for k in ("mean", "max")] + [f"{s}_duty" for s in _METRO_DIGITAL]
    if feature_set == "analog":
        return [f"{s}_{k}" for s in _METRO_ANALOG for k in ("mean", "max")]
    if feature_set == "mean":
        return [f"{s}_mean" for s in _METRO_ANALOG]
    raise ValueError(
        f"load_metropt3: unknown feature_set {feature_set!r}; expected all|analog|mean "
        f"('default' = 'all')"
    )


def _metro_masks(
    t_end: np.ndarray, unit: np.ndarray, events: Sequence[Event], trans_frac: np.ndarray, frac_max: float
) -> dict[str, np.ndarray]:
    """``is_transition`` (windows too full of state changes) and the 24 h post-repair blanking."""
    t = to_epoch_seconds(t_end)
    u = np.asarray(unit, dtype=object)
    is_trans = np.asarray(trans_frac, dtype=np.float64) > float(frac_max)
    post = np.zeros(t.size, dtype=bool)
    for ev in events:
        if np.isfinite(ev.t_failure):
            post |= (u == ev.unit) & (t > ev.t_failure) & (t <= ev.t_failure + POST_REPAIR_S)
    return {"is_transition": is_trans, "post_repair": post, "scoreable": ~(is_trans | post)}


# --------------------------------------------------------------------------------------
# B: Cranfield
# --------------------------------------------------------------------------------------

_CRANFIELD_FS: float = 25.0
_CRANFIELD_CHANNELS: tuple[str, ...] = ("pos_ref", "pos", "vel", "current")


def load_cranfield(
    *,
    input_kind: str = "cycle_features",
    window: int | None = None,
    stride: int | None = None,
    feature_set: str = "test",
    raw_dir: str | Path | None = None,
    cls_target: str = "fault_type",
    **_: Any,
) -> BenchData:
    """Cranfield linear actuator (dataset **B**): the door proxy with a real fault matrix.

    ``input_kind="cycle_features"``
        ``feature_set="test"`` (default, what the plan asks for) gives **one sample per test**:
        the per-stroke features of the test's 10 strokes reduced to mean / std / min / max, so
        a test is one row and leave-one-load-out cannot leak strokes of the same recording
        across the fence. ``feature_set="stroke"`` keeps the adapter's per-stroke rows.
    ``input_kind="raw_window"``
        4 s windows (``window`` defaults to ``4 s * 25 Hz = 100`` samples) over
        ``pos_ref, pos, vel, current``. Every window carries ``meta_test_id``, so a window
        model's predictions can be reduced with :func:`majority_vote` to the test-level
        prediction the test matrix is actually labelled at.

    Labels: ``fault_type`` is the 4-class target (healthy / backlash / lack_lubrication /
    spalling), ``meta_condition`` the 13-class one (class x level, the release's own
    conditions) and ``stage`` the ordinal severity level (0 for normal, 1-8 for spalling).
    """
    from nebulax.adapters import cranfield as adapter

    ds = adapter.load(Path(raw_dir) if raw_dir else Path("data/raw/cranfield"))
    feats = ds.features
    if feats.empty:
        raise ValueError("load_cranfield: the adapter returned an empty feature table (no .mat files?)")

    if input_kind in ("cycle_features", "window_stats"):
        if feature_set in ("test", "default"):
            df = _cranfield_per_test(feats)
        elif feature_set == "stroke":
            df = feats.reset_index(drop=True).copy()
        else:
            raise ValueError(
                f"load_cranfield: unknown feature_set {feature_set!r}; expected test|stroke "
                f"('default' = 'test')"
            )
        X, names = _numeric_matrix(df, S.feature_columns(df))
        t_start, t_end = _as_ms(df["t_start"]), _as_ms(df["t_end"])
        unit = df["train_id"].astype(str).to_numpy(dtype=object)
        group = df["meta_test_id"].astype(str).to_numpy(dtype=object)
        labels = _cranfield_labels(df)
        labels = _cranfield_class_target(labels, cls_target)
        dur = to_epoch_seconds(t_end) - to_epoch_seconds(t_start)
        return BenchData(
            dataset="cranfield",
            subsystem="door",
            input_kind=input_kind,
            X=_finite(X),
            feature_names=names,
            t_start=t_start,
            t_end=t_end,
            unit=unit,
            group=group,
            labels=labels,
            # Each test is an independent 80 s recording on fabricated hourly timestamps: the
            # recording (run_id) is the series, never the rig, so "3 consecutive windows"
            # cannot be three unrelated tests and an event is the faulty recording itself.
            events=_events_from_fault_log(ds.fault_log, unit_col="run_id"),
            window_seconds=float(np.nanmedian(dur)) if dur.size else 0.0,
            masks={},
            series=opaque_series_id(df["run_id"]),
            meta={"feature_set": feature_set, "n_tests": int(pd.unique(group).size), "timeline": "fabricated"},
        )

    if input_kind != "raw_window":
        raise ValueError(f"load_cranfield: unknown input_kind {input_kind!r}")

    from nebulax.features.windows import make_windows

    L = int(window) if window else int(round(4.0 * _CRANFIELD_FS))
    step = int(stride) if stride is not None else L
    wide = S.to_wide(ds.long, "door", signals=list(_CRANFIELD_CHANNELS))
    W = make_windows(wide, list(_CRANFIELD_CHANNELS), L=L, stride=step, fs=_CRANFIELD_FS)
    if len(W) == 0:
        raise ValueError(f"load_cranfield: no raw windows at L={L}; each recording is only 80 s long")
    run_ids = np.asarray(W.run_id if W.run_id is not None else W.train_id, dtype=object)

    # Map each window's run back to its test row (one run per (file, variable) recording).
    per_run = feats.drop_duplicates("run_id").set_index(feats.drop_duplicates("run_id")["run_id"].astype(str))
    wanted = [str(r) for r in pd.unique(run_ids)]
    missing = [r for r in wanted if r not in per_run.index]
    if missing:
        raise ValueError(
            f"load_cranfield(raw_window): {len(missing)} windowed recording(s) have no row in the "
            f"feature table (first: {missing[0]}); the adapter found no stroke in them"
        )
    lut = per_run.loc[wanted]
    row_of = {str(k): i for i, k in enumerate(lut.index)}
    pos = np.array([row_of[str(r)] for r in run_ids], dtype=np.int64)
    df = lut.iloc[pos].reset_index(drop=True)

    t_end = _as_ms(W.t_end)
    labels = _cranfield_labels(df)
    labels = _cranfield_class_target(labels, cls_target)
    return BenchData(
        dataset="cranfield",
        subsystem="door",
        input_kind="raw_window",
        X=_finite(np.ascontiguousarray(W.X, dtype=np.float32)),
        feature_names=list(_CRANFIELD_CHANNELS),
        t_start=t_end - np.timedelta64(int(L / _CRANFIELD_FS * 1000), "ms"),
        t_end=t_end,
        unit=np.asarray(W.train_id, dtype=object),
        group=df["meta_test_id"].astype(str).to_numpy(dtype=object),
        labels=labels,
        events=_events_from_fault_log(ds.fault_log, unit_col="run_id"),
        window_seconds=float(L / _CRANFIELD_FS),
        masks={},
        series=opaque_series_id(run_ids),
        meta={"window": L, "stride": step, "fs_hz": _CRANFIELD_FS, "vote_group": "meta_test_id", "timeline": "fabricated"},
    )


def _cranfield_per_test(feats: pd.DataFrame) -> pd.DataFrame:
    """Per-stroke rows -> one row per test, strokes reduced to mean/std/min/max."""
    keys = [
        "run_id",
        "source",
        "train_id",
        "car",
        "subsystem",
        "component_id",
        "fault_type",
        "severity",
        "is_faulty",
        "rul_s",
        "alarm_window_3d",
        "meta_class",
        "meta_level",
        "meta_test_id",
        "meta_motion_profile",
        "meta_rep",
        "load_kg",
    ]
    keys = [k for k in keys if k in feats.columns]
    value_cols = [c for c in S.feature_columns(feats) if c not in ("load_kg",)]
    g = feats.groupby("meta_test_id", observed=True, sort=False)
    agg = g[value_cols].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    head = g[keys].first()
    out = pd.concat([head, agg], axis=1).reset_index(drop=True)
    out["t_start"] = g["t_start"].min().to_numpy()
    out["t_end"] = g["t_end"].max().to_numpy()
    out["cycle_id"] = np.arange(len(out), dtype=np.int64)
    out["n_strokes"] = g.size().to_numpy().astype(np.float32)
    return out


CRANFIELD_CLS_TARGETS: tuple[str, ...] = ("fault_type", "condition")


def _cranfield_class_target(labels: pd.DataFrame, cls_target: str) -> pd.DataFrame:
    """Select the Cranfield classification target: the 4-class ``fault_type`` (default) or the
    13-class ``condition`` (``meta_condition``: fault type x level, healthy included), exposed
    to the runner as ``labels["class_target"]`` so ``BenchData.y_class`` and the split
    stratification see one column. ``cls_target`` is a loader keyword, hence part of the cache
    key and of ``RunSpec.data_kwargs`` (config identity)."""
    if cls_target not in CRANFIELD_CLS_TARGETS:
        raise ValueError(f"load_cranfield: cls_target must be one of {CRANFIELD_CLS_TARGETS}, got {cls_target!r}")
    labels = labels.copy()
    labels["class_target"] = labels["meta_condition" if cls_target == "condition" else "fault_type"].astype(str)
    return labels


def _cranfield_labels(df: pd.DataFrame) -> pd.DataFrame:
    """4-class fault type, 13-class condition and the ordinal stage, plus the split groups."""
    level = df["meta_level"].to_numpy(dtype=np.float64)
    cls = df["meta_class"].astype(str).to_numpy()
    condition = np.array([c if l == 0 else f"{c}{int(l)}" for c, l in zip(cls, level)], dtype=object)
    return pd.DataFrame(
        {
            "is_faulty": df["is_faulty"].to_numpy(dtype=bool),
            "fault_type": df["fault_type"].astype(str).to_numpy(),
            "severity": df["severity"].to_numpy(dtype=np.float32),
            "rul_s": df["rul_s"].to_numpy(dtype=np.float32),
            "alarm_window": df.get("alarm_window_3d", pd.Series(False, index=df.index)).to_numpy(dtype=bool),
            "stage": level.astype(np.float32),
            "meta_class": cls,
            "meta_condition": condition.astype(str),
            "meta_level": level.astype(np.int32),
            "meta_test_id": df["meta_test_id"].astype(str).to_numpy(),
            "meta_motion_profile": df["meta_motion_profile"].astype(str).to_numpy(),
            "meta_rep": df["meta_rep"].to_numpy(dtype=np.int32),
            "load_kg": df["load_kg"].to_numpy(dtype=np.float32),
            "run_id": df["run_id"].astype(str).to_numpy(),
            "train_id": df["train_id"].astype(str).to_numpy(),
        }
    )


# --------------------------------------------------------------------------------------
# C: Ottawa
# --------------------------------------------------------------------------------------

_OTTAWA_DECIMATE: int = 4


def load_ottawa(
    *,
    input_kind: str = "window_stats",
    window: float = 1.0,
    feature_set: str = "time",
    decimate: int = _OTTAWA_DECIMATE,
    raw_dir: str | Path | None = None,
    **_: Any,
) -> BenchData:
    """Ottawa UORED-VAFCLS bearings (dataset **C**).

    ``window`` is the window length in **seconds**: 1.0 or 0.25, the two the adapter emits.
    ``feature_set`` is ``"time"`` (time-domain statistics + temperature + operating point) or
    ``"time+env"`` (also the envelope-spectrum / fast-kurtogram band features).

    ``input_kind="raw_window"`` re-reads the raw 42 kHz records and **decimates 4x** to
    10.5 kHz, giving ``L = 10500`` (1 s) or ``2625`` (0.25 s) samples on one accelerometer
    channel. Splits group on ``meta_bearing_id``: two records of the same physical bearing
    (healthy-ish "developing" and "faulty") must never straddle the fence.
    """
    from nebulax.adapters import ottawa as adapter

    raw = Path(raw_dir) if raw_dir else Path("data/raw/ottawa")

    if input_kind == "raw_window":
        return _ottawa_raw(raw, window=float(window), decimate=int(decimate))

    ds = adapter.load(raw)
    feats = ds.features
    sel = np.isclose(feats["window_s"].to_numpy(dtype=np.float64), float(window))
    if not sel.any():
        have = sorted(pd.unique(feats["window_s"]).tolist())
        raise ValueError(f"load_ottawa: window={window} s not in the adapter's table; available = {have}")
    df = feats.loc[sel].reset_index(drop=True)
    cols = _ottawa_columns(df, feature_set)
    X, names = _numeric_matrix(df, cols)
    t_start, t_end = _as_ms(df["t_start"]), _as_ms(df["t_end"])
    unit = df["train_id"].astype(str).to_numpy(dtype=object)
    labels = _ottawa_labels(df)
    return BenchData(
        dataset="ottawa",
        subsystem="bearing",
        input_kind=input_kind,
        X=_finite(X),
        feature_names=names,
        t_start=t_start,
        t_end=t_end,
        unit=unit,
        group=labels["meta_bearing_id"].astype(str).to_numpy(dtype=object),
        labels=labels,
        # One 10 s record per (bearing, state) is the series; its fault-log row is the event.
        events=_events_from_fault_log(ds.fault_log, unit_col="run_id"),
        window_seconds=float(window),
        masks={},
        series=opaque_series_id(df["run_id"]),
        meta={
            "feature_set": feature_set,
            "window_s": float(window),
            "n_bearings": int(pd.unique(labels["meta_bearing_id"]).size),
            "timeline": "fabricated",
        },
    )


def _ottawa_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    cols = S.feature_columns(df)
    cols = [c for c in cols if c != "window_s"]
    if feature_set in ("time", "default"):
        return [c for c in cols if not (c.startswith("env_") or c.startswith("sk_") or c == "shaft_hz")]
    if feature_set in ("time+env", "envelope", "all"):
        return cols
    raise ValueError(
        f"load_ottawa: unknown feature_set {feature_set!r}; expected time|time+env "
        f"('default' = 'time')"
    )


def _ottawa_labels(df: pd.DataFrame) -> pd.DataFrame:
    state = df["meta_state"].astype(str).to_numpy()
    return pd.DataFrame(
        {
            "is_faulty": df["is_faulty"].to_numpy(dtype=bool),
            "fault_type": df["fault_type"].astype(str).to_numpy(),
            "severity": df["severity"].to_numpy(dtype=np.float32),
            "rul_s": df["rul_s"].to_numpy(dtype=np.float32),
            "alarm_window": df.get("alarm_window_3d", pd.Series(False, index=df.index)).to_numpy(dtype=bool),
            "stage": state.astype(np.float32),
            "meta_bearing_id": df["meta_bearing_id"].to_numpy(dtype=np.int32),
            "meta_class": df["meta_class"].astype(str).to_numpy(),
            "meta_state": state,
            "meta_state_label": df["meta_state_label"].astype(str).to_numpy(),
            "run_id": df["run_id"].astype(str).to_numpy(),
            "train_id": df["train_id"].astype(str).to_numpy(),
        }
    )


def _ottawa_raw(raw_dir: Path, *, window: float, decimate: int) -> BenchData:
    """Raw accelerometer windows, decimated. Uses the adapter's own file discovery/parsing so
    the naming convention (``<class>_<bearing>_<state>``) has exactly one implementation."""
    from scipy.signal import decimate as sp_decimate

    from nebulax.adapters import ottawa as adapter

    files = adapter._record_files(raw_dir)  # noqa: SLF001 - single source of truth for the layout
    if not files:
        raise FileNotFoundError(f"load_ottawa(raw_window): no Ottawa records under {raw_dir}")
    fs_out = adapter.FS_HZ / decimate
    L = int(round(window * fs_out))
    rows: list[dict[str, Any]] = []
    blocks: list[np.ndarray] = []
    for i, path in enumerate(files):
        cls, bearing_id, state = adapter._parse_name(path)  # noqa: SLF001
        rec = adapter._read_record(path)  # noqa: SLF001
        x = rec["Accelerometer"].to_numpy(dtype=np.float64) * 9.80665
        xd = sp_decimate(x, decimate, ftype="fir", zero_phase=True).astype(np.float32)
        n_win = xd.size // L
        if n_win == 0:
            continue
        blocks.append(xd[: n_win * L].reshape(n_win, L, 1))
        anchor = pd.Timestamp("2020-01-01T00:00:00Z") + pd.Timedelta(seconds=i * 20.0)
        for k in range(n_win):
            rows.append(
                {
                    "run_id": f"ottawa_{path.stem}",
                    "train_id": f"bearing_{bearing_id:02d}",
                    "meta_bearing_id": np.int32(bearing_id),
                    "meta_class": cls,
                    "meta_state": state,
                    "meta_state_label": adapter._STATE_LABEL[state],  # noqa: SLF001
                    "fault_type": adapter._CLASS_FAULT[cls],  # noqa: SLF001
                    "severity": np.float32(adapter._STATE_SEVERITY[state]),  # noqa: SLF001
                    "is_faulty": state != "0",
                    "rul_s": np.float32(np.nan),
                    "alarm_window": False,
                    "stage": np.float32(float(state)),
                    "t_start": anchor + pd.Timedelta(seconds=k * window),
                    "t_end": anchor + pd.Timedelta(seconds=(k + 1) * window),
                }
            )
    if not blocks:
        raise ValueError(f"load_ottawa(raw_window): window={window}s is longer than a decimated record")
    df = pd.DataFrame(rows)
    X = np.concatenate(blocks, axis=0)
    t_end = _as_ms(df["t_end"])
    labels = df[
        [
            "is_faulty",
            "fault_type",
            "severity",
            "rul_s",
            "alarm_window",
            "stage",
            "meta_bearing_id",
            "meta_class",
            "meta_state",
            "meta_state_label",
            "run_id",
            "train_id",
        ]
    ].reset_index(drop=True)
    events = [
        Event.from_any(str(opaque_series_id([u])[0]), t, np.nan, f)
        for u, t, f in df.loc[df["is_faulty"]].groupby("run_id", observed=True).first().reset_index()[
            ["run_id", "t_start", "fault_type"]
        ].itertuples(index=False)
    ]
    return BenchData(
        dataset="ottawa",
        subsystem="bearing",
        input_kind="raw_window",
        X=_finite(X),
        feature_names=["vib_acc"],
        t_start=_as_ms(df["t_start"]),
        t_end=t_end,
        unit=df["train_id"].astype(str).to_numpy(dtype=object),
        group=df["meta_bearing_id"].astype(str).to_numpy(dtype=object),
        labels=labels,
        events=events,
        window_seconds=float(window),
        masks={},
        series=opaque_series_id(df["run_id"]),
        meta={"decimate": decimate, "fs_hz_out": fs_out, "L": L, "n_records": len(files), "timeline": "fabricated"},
    )


# --------------------------------------------------------------------------------------
# D: synthetic fleet
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PeerGrouping:
    """How one sim subsystem's **concurrent siblings** are identified.

    ``group_cols``  the keys that make a peer group. A group must contain more than one row
                    or peer-normalisation produces nothing at all (a leave-one-out mean over a
                    single row is NaN), which is a silent failure: the NaNs become constant
                    zero columns and ``peer_norm=True`` quietly means "add 48 zeros".
    ``same_side``   compare only against peers on the same side of the vehicle (axle boxes:
                    the sunny side runs hotter, so a cross-side comparison is noise).
    ``members_per_group``  how many rows a full peer group holds, **declared** rather than
                    measured. It fixes the feature schema: a peer z-score needs at least three
                    members, so with two it is dropped. Measuring this on the table would let
                    the held-out rows decide which columns the model is fitted on, since the
                    feature table is built before the split. The declaration is checked against
                    the data (:func:`_sim_peer_normalise` raises on a mismatch), and a check
                    that aborts is not a check that selects.
    ``safe_units``  the split-unit columns a group is guaranteed to sit inside. This is what
                    makes pre-split peer normalisation legitimate: peers are concurrent rows
                    on the same unit, so no held-out unit can enter a training row's peer
                    statistic *as long as the split holds out one of these columns*. The
                    runner refuses ``peer_norm=True`` with any other split.
    """

    group_cols: list[str]
    same_side: bool
    safe_units: tuple[str, ...]
    members_per_group: int = 0


#: Per-subsystem peer groupings on the synthetic fleet, checked against the real tables by
#: ``tests/test_bench_data.py``. ``pneumatic`` is deliberately absent: the fleet has exactly
#: one APU per train and one row per compressor cycle, so a concurrent sibling does not exist
#: and there is nothing to normalise against - ``_sim_peer_normalise`` refuses rather than
#: emitting all-NaN columns.
_SIM_PEER_GROUPS: dict[str, PeerGrouping] = {
    # Two doors per train share a dwell: identical t_start and cycle_id across the train's two
    # runs (verified on the real fleet - 132,055 groups of exactly 2, no singletons).
    "door": PeerGrouping(["train_id", "cycle_id"], False, ("train_id",), members_per_group=2),
    # Eight axle boxes per run share a window; same_side halves that to four.
    "bearing": PeerGrouping(["run_id", "t_end"], True, ("train_id", "run_id"), members_per_group=8),
}
_SIM_FS: dict[str, float] = {"door": 100.0, "pneumatic": 1.0, "bearing": 1.0}


def load_sim(
    *,
    subsystem: str = "door",
    input_kind: str = "cycle_features",
    window: int | None = None,
    stride: int | None = None,
    peer_norm: bool = False,
    run_ids: Sequence[str] | None = None,
    max_runs: int | None = None,
    H: float = DEFAULT_H_S,
    raw_dir: str | Path | None = None,
    **_: Any,
) -> BenchData:
    """The synthetic fleet (dataset **D**), one subsystem at a time.

    ``subsystem`` is ``door`` / ``pneumatic`` / ``bearing`` - the per-subsystem breakdown the
    plan asks for; never all three in one ``BenchData``, because they share no feature space.

    ``input_kind="cycle_features"`` (default) uses the simulator's own per-cycle / per-window
    feature table - 264 k door cycles, 18 k compressor cycles, 1.38 M bearing windows.
    ``window_stats`` / ``raw_window`` stream the raw telemetry **run by run** (808 M rows in
    total across the fleet, so it is never loaded at once) and window it at the subsystem's
    native rate; ``max_runs`` caps how many runs are read, completed to whole trains
    (:func:`_sim_complete_units`) so peer groups and leave-one-unit-out folds stay intact.

    ``peer_norm=True`` adds :func:`nebulax.features.cycles.peer_normalise` columns - the
    leave-one-out comparison against sibling doors at the same dwell, or the same-side axle
    boxes at the same window. On the bearing subsystem it is the primary detector, not a
    nicety: at Singapore ambient an absolute temperature line is very weak (ladder
    ``operating_environment.consequences.bearing``).

    Training-set **contamination** is not applied here - it is a property of the training
    slice, so the runner applies it with :func:`train_rows` after the split.
    """
    from nebulax.adapters import synthetic as adapter

    root = Path(raw_dir) if raw_dir else Path("data/sim")
    if subsystem not in _SIM_FS:
        raise ValueError(f"load_sim: unknown subsystem {subsystem!r}; expected one of {sorted(_SIM_FS)}")
    ids = list(run_ids) if run_ids else None
    if ids is None and max_runs is not None:
        ids = _sim_complete_units(root, subsystem, _sim_run_ids(root, subsystem)[: int(max_runs)])

    if input_kind == "window_stats" and window is None:
        raise ValueError(
            "load_sim(input_kind='window_stats'): a window length is required. Without one this "
            "used to hand back the simulator's per-cycle feature table labelled as window "
            "statistics, which is a different feature space with a different row meaning - two "
            "runs that differ only in input_kind would have been the same numbers. Pass "
            "window=<samples> (with data.max_runs to cap how much telemetry is streamed), or "
            "ask for input_kind='cycle_features', which is what that table actually is."
        )
    if input_kind == "cycle_features" and window is None:
        ds = adapter.load(root, subsystem=subsystem, run_ids=ids, include_long=False)
        feats = ds.features.reset_index(drop=True)
        if peer_norm:
            feats = _sim_peer_normalise(feats, subsystem)
        X, names = _numeric_matrix(feats, S.feature_columns(feats))
        t_start, t_end = _as_ms(feats["t_start"]), _as_ms(feats["t_end"])
        unit = feats["train_id"].astype(str).to_numpy(dtype=object)
        labels = _sim_labels(feats)
        dur = to_epoch_seconds(t_end) - to_epoch_seconds(t_start)
        loaded_runs = [str(r) for r in pd.unique(feats["run_id"])]
        events, provenance = _sim_events(root, loaded_runs, ds.fault_log)
        series = sim_series_id(feats["run_id"], feats["component_id"])
        # One overlap rule for both input kinds: a row whose END is at or after its component's
        # onset is faulty. The simulator stamps a cycle by its start, so the cycle that straddles
        # the onset (3 of 264k door cycles) would otherwise stay healthy here and faulty in the
        # window table.
        labels = _sim_extend_labels_to_run_end(labels, t_end, series, events)
        post = _sim_post_failure_mask(t_end, series, events)
        return BenchData(
            dataset="sim",
            subsystem=subsystem,
            input_kind=input_kind,
            X=_finite(X),
            feature_names=names,
            t_start=t_start,
            t_end=t_end,
            unit=unit,
            group=feats["run_id"].astype(str).to_numpy(dtype=object),
            labels=labels,
            events=events,
            window_seconds=float(np.nanmedian(dur)) if dur.size else 0.0,
            masks={"post_failure": post, "scoreable": ~post},
            series=series,
            meta={
                "peer_norm": peer_norm,
                "n_runs": len(loaded_runs),
                "fault_log_source": provenance,
            },
        )

    if peer_norm:
        raise ValueError(
            f"load_sim(input_kind={input_kind!r}, peer_norm=True): peer normalisation is defined on "
            f"the per-cycle feature table (a cycle has concurrent sibling cycles), not on raw "
            f"telemetry windows, and was previously accepted and then silently ignored - the "
            f"peer_norm=True and peer_norm=False rows were byte-identical apart from their config "
            f"hash. Use input_kind='cycle_features' for the peer_norm ablation."
        )
    return _sim_windows(
        root,
        subsystem=subsystem,
        input_kind=input_kind,
        window=window or 128,
        stride=stride,
        run_ids=ids or _sim_complete_units(root, subsystem, _sim_run_ids(root, subsystem)[: int(max_runs) if max_runs else 4]),
        H=H,
    )


#: The fleet's canonical ground-truth file, as the plan names it. The per-run partitions carry
#: the same rows, but "the events came from data/sim/fault_log.parquet" is a contract a reader
#: can check in one command, and a partition-assembled list is not.
SIM_FAULT_LOG = "fault_log.parquet"


def _sim_events(root: Path, run_ids: Sequence[str] | None, fallback: pd.DataFrame | None) -> tuple[list[Event], str]:
    """Ground-truth events for the synthetic fleet, from ``<root>/fault_log.parquet``.

    Returns ``(events, provenance)``. The canonical file is read and filtered to ``run_ids``
    (``None`` = every run). Only if it is absent - a trimmed test fixture - do we fall back to
    the per-run partition logs the adapter assembles, and the provenance string says so, so a
    results row is never silently scored against a different ground truth than the one the
    plan names.
    """
    path = Path(root) / SIM_FAULT_LOG
    if path.exists():
        fl = pd.read_parquet(path)
        if run_ids is not None and "run_id" in fl.columns:
            fl = fl[fl["run_id"].astype(str).isin({str(r) for r in run_ids})]
        return _events_from_fault_log(fl, unit_col=None), str(path)
    LOGGER.warning("load_sim: %s is missing; falling back to the per-run partition fault logs", path)
    return _events_from_fault_log(fallback, unit_col=None), "partition fault logs (canonical file missing)"


def _sim_complete_units(root: Path, subsystem: str, run_ids: Sequence[str]) -> list[str]:
    """Extend a truncated run list with every other run of the same subsystem on the SAME
    trains, so ``max_runs`` never leaves a train with half its components: a peer group
    (the two doors of a train, the 16 axle boxes) must be whole or ``peer_norm=True`` fails
    on it, and a leave-one-unit-out fold must hold out a whole train."""
    index = json.loads((root / "index.json").read_text())
    runs = [r for r in index["runs"] if r["subsystem"] == subsystem]
    train_of = {r["run_id"]: r.get("train_id") for r in runs}
    trains = {train_of.get(r) for r in run_ids}
    return sorted({r["run_id"] for r in runs if r.get("train_id") in trains} | set(run_ids))


def _sim_run_ids(root: Path, subsystem: str) -> list[str]:
    index = json.loads((root / "index.json").read_text())
    return sorted(r["run_id"] for r in index["runs"] if r["subsystem"] == subsystem)


#: Columns that a synthetic-fleet split can ever hold out (``sim_loo_unit`` and
#: ``sim_random_unit`` hold out ``train_id``, ``sim_run_kfold`` holds out ``run_id``).
SIM_SPLIT_UNIT_COLS: tuple[str, ...] = ("train_id", "run_id")


def _assert_peer_groups_within_units(
    feats: pd.DataFrame,
    group_cols: Sequence[str],
    subsystem: str,
    safe_units: Sequence[str] = SIM_SPLIT_UNIT_COLS,
) -> None:
    """Refuse a peer grouping that could carry a held-out unit into a training row's features.

    Peer-normalisation is computed on the whole feature table, before the runner splits it.
    That is only safe because a peer group is a set of *concurrent siblings* - the same cycle
    on the same unit - so every peer of a training row is a row the model would also have at
    inference time, and never a row from a unit the split is holding out. It is not safe in
    general: a grouping keyed on ``train_id`` alone would let a ``sim_run_kfold`` training row
    borrow the mean of a run in the test fold.

    Rather than trust that the current keys happen to be fine, check it: every peer group must
    sit inside a single value of every column a sim split can hold out. Raises ``ValueError``
    naming the offending column if not, so the failure is a loud one at load time instead of
    an optimistic number in the results table.
    """
    if not group_cols:
        return
    gcode = feats.groupby(list(group_cols), sort=False, observed=True, dropna=False).ngroup().to_numpy()
    n_groups = int(gcode.max()) + 1 if gcode.size else 0
    for col in safe_units:
        if col not in feats.columns or col in group_cols:
            continue
        ucode = pd.factorize(feats[col], use_na_sentinel=False)[0].astype(np.int64)
        n_units = int(ucode.max()) + 1 if ucode.size else 1
        n_pairs = int(np.unique(gcode.astype(np.int64) * n_units + ucode).size)
        if n_pairs > n_groups:
            raise ValueError(
                f"load_sim(peer_norm, subsystem={subsystem!r}): peer groups {list(group_cols)} span "
                f"more than one {col!r} ({n_pairs - n_groups} group(s) do), so a peer statistic "
                f"could cross a split that holds {col!r} out, which PeerGrouping.safe_units claims "
                f"is safe. Fix the declaration in _SIM_PEER_GROUPS, add {col!r} to the group keys, "
                f"or peer-normalise after splitting."
            )


#: Column carrying each row's peer-group id when ``peer_norm=True``. It is ``meta_``-prefixed,
#: so :func:`nebulax.schema.feature_columns` keeps it out of ``X``, and
#: :func:`nebulax.bench.splits.make_split` uses it to place a whole peer group on one side of a
#: temporal cut or purge it entirely. Peer features are computed before the split, so a group
#: that lands on both sides is a genuine leak; this column is what lets the split *prove* that
#: never happens instead of arguing that it is unlikely.
PEER_GROUP_COL: str = "meta_peer_group"

#: How far apart two peers may **begin** and still count as simultaneous. Doors on one train
#: start their dwell at the same instant; a second is generous for that and five orders of
#: magnitude below the day-level cuts of every temporal preset.
PEER_TIME_TOLERANCE_S: float = 1.0


def _group_spread_s(feats: pd.DataFrame, group_cols: Sequence[str], col: str) -> np.ndarray:
    """Per-row: how many seconds ``col`` spans within that row's group."""
    g = feats.groupby(list(group_cols), sort=False, observed=True, dropna=False)[col]
    hi = to_epoch_seconds(g.transform("max").to_numpy())
    lo = to_epoch_seconds(g.transform("min").to_numpy())
    return hi - lo


def _assert_peer_groups_time_coincident(
    feats: pd.DataFrame,
    group_cols: Sequence[str],
    subsystem: str,
    tol_s: float = PEER_TIME_TOLERANCE_S,
) -> None:
    """Refuse a peer grouping whose members do not **begin** at the same moment.

    This says the grouping *means* what "peer" is supposed to mean - the same event seen on
    sibling components, one dwell and two doors, one window and eight axle boxes - rather than
    the same component at another moment, which would be a look-ahead dressed up as a
    comparison.

    It is **not** the safety argument for a temporal split. A start tolerance shows only that a
    cut would have to fall inside ``tol_s`` to separate a group; it does not show that no cut
    does. Safety comes from :data:`PEER_GROUP_COL`: the group id travels with the rows and
    :func:`nebulax.bench.splits.make_split` places every peer group atomically, purging any
    group a cut would split. That is a proof rather than an argument about likelihood.

    ``t_end`` is deliberately **not** required to coincide. Two doors that begin closing
    together can finish up to ~9 s apart on the real fleet, and that difference is precisely
    the quantity the peer comparison measures - requiring it to be zero would forbid the
    feature that makes the grouping worth having. The spread is logged, not enforced.
    """
    if feats.empty or "t_start" not in feats.columns:
        return
    spread = _group_spread_s(feats, group_cols, "t_start")
    worst = float(np.nanmax(spread)) if spread.size else 0.0
    if worst > float(tol_s):
        raise ValueError(
            f"load_sim(peer_norm=True, subsystem={subsystem!r}): peer groups {list(group_cols)} are not "
            f"time-coincident - {int(np.count_nonzero(spread > float(tol_s)))} row(s) sit in a group whose "
            f"'t_start' spans up to {worst:.3g} s (tolerance {tol_s} s). Peers must begin together, or a "
            f"temporal split can put a training row's peer in the test period. Add the time columns to the "
            f"group keys, or peer-normalise after splitting."
        )
    if "t_end" in feats.columns:
        ends = _group_spread_s(feats, group_cols, "t_end")
        LOGGER.info(
            "load_sim(peer_norm, %s): peer groups start within %.3g s of each other and end within %.3g s "
            "(the end spread is the signal, not a violation)",
            subsystem,
            worst,
            float(np.nanmax(ends)) if ends.size else 0.0,
        )


def _sim_peer_normalise(feats: pd.DataFrame, subsystem: str) -> pd.DataFrame:
    """Add leave-one-out peer columns for ``subsystem``, or raise saying why it cannot.

    Three failure modes are refused loudly instead of producing plausible-looking zeros:
    a subsystem with no sibling relation at all (pneumatic), a grouping that turns out to be
    singleton on this table, and a grouping whose peer columns come back with no usable
    values. All three previously yielded ``peer_norm=True`` runs whose peer features were
    constant, i.e. an ablation axis that measured nothing.
    """
    from nebulax.features.cycles import peer_normalise

    grouping = _SIM_PEER_GROUPS.get(subsystem)
    if grouping is None:
        raise ValueError(
            f"load_sim(peer_norm=True, subsystem={subsystem!r}): this subsystem has no concurrent "
            f"siblings on the synthetic fleet (one APU per train, one row per compressor cycle), so "
            f"there is nothing to peer-normalise against. Run it with peer_norm=False; peer "
            f"normalisation is defined for {sorted(_SIM_PEER_GROUPS)}."
        )
    group_cols = [c for c in grouping.group_cols if c in feats.columns]
    value_cols = [c for c in S.feature_columns(feats) if pd.api.types.is_numeric_dtype(feats[c])]
    if len(group_cols) != len(grouping.group_cols) or not value_cols:
        raise ValueError(
            f"load_sim(peer_norm=True, subsystem={subsystem!r}): peer grouping {grouping.group_cols} "
            f"needs columns this feature table does not have (present: {sorted(feats.columns)[:20]}...)"
        )

    sizes = feats.groupby(group_cols, sort=False, observed=True, dropna=False).size()
    if grouping.members_per_group and int(sizes.max()) != grouping.members_per_group:
        raise ValueError(
            f"load_sim(peer_norm=True, subsystem={subsystem!r}): PeerGrouping declares "
            f"{grouping.members_per_group} members per peer group but the largest group on this table "
            f"holds {int(sizes.max())}. The declaration fixes the FEATURE SCHEMA, so it must not be "
            f"guessed from the data - fix _SIM_PEER_GROUPS to match the fleet."
        )
    if int(sizes.max()) < 2:
        raise ValueError(
            f"load_sim(peer_norm=True, subsystem={subsystem!r}): every peer group keyed on "
            f"{group_cols} is a singleton ({len(sizes)} groups over {len(feats)} rows), so a "
            f"leave-one-out peer statistic is NaN for every row and the peer columns would be "
            f"constant. Fix the grouping in _SIM_PEER_GROUPS rather than shipping dead features."
        )
    _assert_peer_groups_within_units(feats, group_cols, subsystem, grouping.safe_units)
    _assert_peer_groups_time_coincident(feats, group_cols, subsystem)

    out = peer_normalise(feats, group_cols, value_cols=value_cols, same_side=grouping.same_side).drop(
        columns=["_peer_side"], errors="ignore"
    )
    out = _drop_undefined_peer_columns(feats, out, subsystem, grouping.members_per_group)
    # The group id travels with the rows as a meta_ column (so it is excluded from X by
    # nebulax.schema.feature_columns) and lets splits.make_split place a whole peer group in
    # one partition or purge it - see PEER_GROUP_COL.
    out[PEER_GROUP_COL] = (
        feats.groupby(group_cols, sort=False, observed=True, dropna=False).ngroup().to_numpy()
    )
    _assert_peer_features_informative(feats, out, subsystem)
    return out


def _drop_undefined_peer_columns(
    before: pd.DataFrame, after: pd.DataFrame, subsystem: str, members_per_group: int
) -> pd.DataFrame:
    """Drop the peer columns this **grouping** cannot define, decided from the grouping alone.

    A leave-one-out peer *standard deviation* needs at least two peers, i.e. a group of three.
    A door dwell holds exactly two doors, so every ``*_peer_z`` column is NaN for every row and
    ``_finite`` would hand the model 24 constant zeros - harmless to a tree, misleading to a
    reader, and exactly the shape of the bug where ``peer_norm=True`` looked like it was doing
    something. ``*_peer_delta`` needs one peer and survives.

    The rule is deliberately a function of ``members_per_group`` - a number **declared** on the
    :class:`PeerGrouping`, not measured on the table - and the column suffix. Neither the
    feature values nor the observed group sizes may decide the schema: the feature table is
    built before the split, so anything read off it lets the held-out rows determine which
    columns the model is fitted on. That is a small leak, but it is a leak, and it is the kind
    that is invisible in the results row.
    """
    if members_per_group >= 3 or members_per_group <= 0:
        return after
    new_cols = [c for c in after.columns if c not in before.columns]
    dead = [c for c in new_cols if c.endswith("_peer_z")]
    if dead:
        LOGGER.info(
            "load_sim(peer_norm, %s): dropping %d of %d peer column(s) that this grouping cannot "
            "define - a peer z-score needs >= 3 group members and PeerGrouping declares %d",
            subsystem,
            len(dead),
            len(new_cols),
            members_per_group,
        )
    return after.drop(columns=dead)


def _assert_peer_features_informative(before: pd.DataFrame, after: pd.DataFrame, subsystem: str) -> None:
    """Refuse a peer-normalisation that produced no information.

    ``_finite`` later turns NaN into 0.0, so an all-NaN peer column reaches the model as a
    constant zero and the ``peer_norm`` ablation silently compares a model against itself.

    This is an **assertion**: it either lets the whole feature table through unchanged or
    aborts the load. It never selects columns, so unlike a value-based drop it cannot let the
    held-out rows influence what the model is fitted on.
    """
    # Only real peer features count. meta_peer_group is added for the splitter and varies by
    # construction, so counting it would let a table whose every actual peer column is constant
    # pass this check - which is precisely the failure the check exists to catch.
    new_cols = [
        c
        for c in after.columns
        if c not in before.columns and (c.endswith("_peer_delta") or c.endswith("_peer_z"))
    ]
    if not new_cols:
        raise ValueError(f"load_sim(peer_norm=True, subsystem={subsystem!r}): produced no peer feature columns")
    informative = 0
    for c in new_cols:
        v = pd.to_numeric(after[c], errors="coerce").to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size and float(np.nanstd(v)) > 0.0:
            informative += 1
    if informative == 0:
        raise ValueError(
            f"load_sim(peer_norm=True, subsystem={subsystem!r}): all {len(new_cols)} peer columns are "
            f"empty or constant, so peer_norm=True would be indistinguishable from peer_norm=False "
            f"with {len(new_cols)} zero columns bolted on."
        )


def _sim_post_failure_mask(t_end: np.ndarray, series: np.ndarray, events: Sequence[Event]) -> np.ndarray:
    """Rows of a component after its ``t_failure``.

    The simulator models no repair: after functional failure the component keeps running,
    degraded, to the end of the run, and its feature table keeps ``is_faulty=True``. Those
    rows are trivially anomalous and would be counted twice - as easy positives for the
    threshold-free metrics and, because no event window extends past ``t_failure``, as false
    alarms for the operational ones. They are blanked from ``scoreable`` exactly as MetroPT
    blanks its 24 h post-repair window.
    """
    t = to_epoch_seconds(t_end)
    s = np.asarray(series, dtype=object)
    post = np.zeros(t.size, dtype=bool)
    for ev in events:
        if np.isfinite(ev.t_failure):
            post |= (s == ev.unit) & (t > ev.t_failure)
    return post


def _sim_extend_labels_to_run_end(labels: pd.DataFrame, t_end: np.ndarray, series: np.ndarray, events: Sequence[Event]) -> pd.DataFrame:
    """Make raw-window labels agree with the simulator's own per-cycle ``is_faulty``.

    The feature table marks a component faulty from ``t_onset`` to the END of the run (there
    is no repair); ``_label_rows`` stops at ``t_failure``. Both input kinds of one dataset
    must score against one ground truth, so the window labels are extended the same way
    (the rows past ``t_failure`` are then blanked by :func:`_sim_post_failure_mask`).
    """
    t = to_epoch_seconds(t_end)
    s = np.asarray(series, dtype=object)
    is_faulty = labels["is_faulty"].to_numpy(dtype=bool).copy()
    ftype = labels["fault_type"].to_numpy(dtype=object).copy()
    for ev in events:
        after = (s == ev.unit) & (t >= ev.t_onset)
        is_faulty |= after
        ftype[after & (ftype == "healthy")] = ev.fault_type
    labels = labels.copy()
    labels["is_faulty"] = is_faulty
    labels["fault_type"] = ftype
    return labels


def _sim_labels(feats: pd.DataFrame) -> pd.DataFrame:
    """Per-row labels and split keys for the synthetic fleet. Never an input to a model."""
    sev = feats["severity"].to_numpy(dtype=np.float32) if "severity" in feats else np.full(len(feats), np.nan, np.float32)
    out = pd.DataFrame(
        {
            "is_faulty": feats["is_faulty"].to_numpy(dtype=bool),
            "fault_type": feats["fault_type"].astype(str).to_numpy(),
            "severity": sev,
            "rul_s": feats["rul_s"].to_numpy(dtype=np.float32),
            "alarm_window": feats.get("alarm_window_3d", pd.Series(False, index=feats.index)).to_numpy(dtype=bool),
            "stage": np.floor(np.nan_to_num(sev, nan=-1.0) * 4.0).astype(np.float32),
            "run_id": feats["run_id"].astype(str).to_numpy(),
            "train_id": feats["train_id"].astype(str).to_numpy(),
            "component_id": feats["component_id"].astype(str).to_numpy(),
            "car": feats["car"].to_numpy(dtype=np.int16),
        }
    )
    # The peer-group id has to reach the split (splits._atomic_peer_supports) or a temporal cut
    # could fall inside a group whose features were computed before the split.
    if PEER_GROUP_COL in feats.columns:
        out[PEER_GROUP_COL] = feats[PEER_GROUP_COL].to_numpy()
    return out


def _sim_windows(
    root: Path,
    *,
    subsystem: str,
    input_kind: str,
    window: int,
    stride: int | None,
    run_ids: Sequence[str],
    H: float,
) -> BenchData:
    """Stream the fleet's raw telemetry run by run and window it. One run in memory at a time."""
    from nebulax.adapters import synthetic as adapter
    from nebulax.features.stats import feature_names as stat_feature_names
    from nebulax.features.stats import window_stats
    from nebulax.features.windows import make_windows

    fs = _SIM_FS[subsystem]
    sigs = [s for s in S.SIGNALS[subsystem]]
    L, step = int(window), int(stride) if stride is not None else max(1, int(window) // 2)
    Xs: list[np.ndarray] = []
    t_ends: list[np.ndarray] = []
    units: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    comps: list[np.ndarray] = []
    canonical = (Path(root) / SIM_FAULT_LOG).exists()
    events, fault_log_source = _sim_events(root, [str(r) for r in run_ids], None)
    for rid in run_ids:
        ds = adapter.load(root, run_ids=[rid], include_long=True)
        wide = S.to_wide(ds.long, subsystem, signals=sigs)
        if not canonical and ds.fault_log is not None:
            # Canonical file missing (trimmed fixture): collect EVERY requested run's partition
            # log, keyed like the labels and episodes (run_id/component_id), not by train.
            events.extend(_events_from_fault_log(ds.fault_log, unit_col=None))
        del ds
        if wide.empty:
            continue
        W = make_windows(wide, sigs, L=L, stride=step, fs=fs, max_gap_s=max(3.0 / fs, 5.0))
        del wide
        if len(W) == 0:
            continue
        Xs.append(np.ascontiguousarray(W.X, dtype=np.float32))
        t_ends.append(_as_ms(W.t_end))
        units.append(np.asarray(W.train_id, dtype=object))
        groups.append(np.full(len(W), rid, dtype=object))
        comps.append(np.asarray(W.component_id, dtype=object))
    if not Xs:
        raise ValueError(f"load_sim: no windows built for subsystem={subsystem} runs={list(run_ids)}")
    Xw = np.concatenate(Xs, axis=0)
    t_end = np.concatenate(t_ends)
    unit = np.concatenate(units)
    group = np.concatenate(groups)
    comp = np.concatenate(comps)
    if input_kind == "raw_window":
        X, names = Xw, sigs
    elif input_kind == "window_stats":
        X, names = window_stats(Xw), stat_feature_names(sigs)
    else:
        raise ValueError(f"load_sim: unknown input_kind {input_kind!r}")
    series = sim_series_id(group, comp)
    labels = _label_rows(t_end, series, events, H=H)  # per component: a healthy sibling stays healthy
    labels = _sim_extend_labels_to_run_end(labels, t_end, series, events)  # same truth as cycle_features
    post = _sim_post_failure_mask(t_end, series, events)
    labels["stage"] = np.nan
    labels["run_id"] = group
    labels["train_id"] = unit
    labels["component_id"] = comp
    return BenchData(
        dataset="sim",
        subsystem=subsystem,
        input_kind=input_kind,
        X=_finite(np.asarray(X, dtype=np.float32)),
        feature_names=list(names),
        t_start=t_end - np.timedelta64(int(L / fs * 1000), "ms"),
        t_end=t_end,
        unit=unit,
        group=group,
        labels=labels,
        events=events,
        window_seconds=float(L / fs),
        masks={"post_failure": post, "scoreable": ~post},
        series=series,
        meta={
            "window": L,
            "stride": step,
            "fs_hz": fs,
            "run_ids": list(run_ids),
            "fault_log_source": fault_log_source,
        },
    )


# --------------------------------------------------------------------------------------
# Dispatcher + cache
# --------------------------------------------------------------------------------------

_LOADERS = {
    "metropt3": load_metropt3,
    "cranfield": load_cranfield,
    "ottawa": load_ottawa,
    "sim": load_sim,
}


#: Axis keywords the runner passes to EVERY loader (RunSpec.loader_kwargs merges them in), so
#: a loader that has no use for one of them still accepts it.
UNIVERSAL_LOADER_KWARGS: frozenset[str] = frozenset({"input_kind", "feature_set", "H", "peer_norm", "subsystem", "window"})


def loader_accepted_kwargs(dataset: str) -> frozenset[str]:
    """Keywords ``load_bench(dataset, ...)`` accepts: the loader's named parameters plus the
    universal axis keywords. Anything else is a typo or a keyword meant for another loader."""
    import inspect

    fn = _LOADERS[dataset]
    named = {
        n for n, p in inspect.signature(fn).parameters.items()
        if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }
    return frozenset(named | UNIVERSAL_LOADER_KWARGS)


def load_bench(
    dataset: str,
    *,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    **kwargs: Any,
) -> BenchData:
    """Build (or read from cache) one :class:`BenchData`.

    All loader keywords - ``subsystem``, ``input_kind``, ``window``, ``stride``,
    ``feature_set``, ``peer_norm``, ``H``, ``raw_dir``, ... - are part of the cache key, so
    two runs that differ in any of them never share a cache entry. ``use_cache=False`` forces
    a rebuild and still writes the result.
    """
    if dataset not in _LOADERS:
        raise ValueError(f"load_bench: unknown dataset {dataset!r}; expected one of {DATASETS}")
    unknown = sorted(set(kwargs) - loader_accepted_kwargs(dataset))
    if unknown:
        # A keyword the loader ignores would still fork the cache, be recorded in
        # meta["loader_kwargs"] and be published in the results row as if it had had an effect.
        raise ValueError(
            f"load_bench({dataset!r}): unknown loader keyword(s) {unknown}; this loader accepts "
            f"{sorted(loader_accepted_kwargs(dataset))}"
        )
    key = cache_key(dataset, **kwargs)
    cdir = Path(cache_dir) if cache_dir else None
    if cdir is not None and use_cache:
        hit = _cache_load(cdir, key)
        if hit is not None:
            LOGGER.info("bench.data: cache hit %s (%s)", key, dataset)
            return hit
    data = _LOADERS[dataset](**kwargs)
    # Record what the loader was actually called with, so the runner can cross-check the
    # results row against it (runner.run_one). Survives the cache: the kwargs are part of the
    # cache key, so a hit necessarily replays the same call.
    data.meta["loader_kwargs"] = {k: kwargs[k] for k in sorted(kwargs) if _jsonable(kwargs[k])}
    data.meta["cache_key"] = key
    data.meta["cache_hit"] = False
    if cdir is not None:
        try:
            _cache_store(cdir, key, data)
        except Exception as exc:  # pragma: no cover - a read-only data dir must not fail a run
            LOGGER.warning("bench.data: could not write cache %s (%s)", key, exc)
    return data
