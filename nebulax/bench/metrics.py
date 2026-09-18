"""Evaluation metrics for the NEBULA X benchmark.

Binding protocol (plan "Evaluation protocol", ``configs/model_ladder.yaml:evaluation``):

* **No point adjustment. Ever.** Nothing in this module rewrites a score or a prediction
  using the ground truth before measuring it. ``point_adjusted_f1`` and ``vus_roc`` are on
  the ladder's ``excluded`` list and are deliberately absent here.
* **Alarm episodes** are ``score > threshold`` for ``>= k`` consecutive windows (k = 3),
  merged when the gap between them is ``< merge_gap_s`` (1 h). See :func:`episodes`.
* **Primary AD metrics** are operational: event recall inside ``[t_onset - H, t_failure]``,
  false alarms per train-day, and the lead-time distribution (:func:`event_metrics`).
* **Threshold-free** metrics are AUROC, AUPRC and VUS-PR (:func:`auroc`, :func:`auprc`,
  :func:`vus_pr`).
* **CLS** metrics are macro-F1, balanced accuracy and the confusion matrix
  (:func:`cls_metrics`); ordinal severity is checked with :func:`monotonicity`.
* **CPD** metrics are ``covering`` and ``f1_at_annotation_margin`` (:func:`covering`,
  :func:`f1_at_margin`, integrated over margins by :func:`f1_over_margins`), fed by the
  change-point score adapter (:func:`cpd_score`) that turns discrete breakpoints / boolean
  drift flags into the continuous per-window score VUS-PR needs.

Every metric here is pure: arrays in, floats/dicts out, no I/O and no global state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from nebulax.bench import vus_tsb_ad as _V

__all__ = [
    "Episode",
    "Event",
    "episodes",
    "max_step_seconds",
    "row_pitch_seconds",
    "event_metrics",
    "auroc",
    "auprc",
    "r_auc_pr",
    "vus_pr",
    "cls_metrics",
    "monotonicity",
    "segments_from_cps",
    "f1_at_margin",
    "f1_over_margins",
    "covering",
    "cpd_metrics",
    "cpd_score",
    "DEFAULT_K_CONSECUTIVE",
    "DEFAULT_MERGE_GAP_S",
    "DEFAULT_CPD_MARGIN",
]

#: Protocol defaults (``configs/model_ladder.yaml:evaluation.episode_definition``).
DEFAULT_K_CONSECUTIVE: int = 3
DEFAULT_MERGE_GAP_S: float = 3600.0
#: ``evaluation.cpd_metrics.f1_at_annotation_margin.params.margin_timesteps``.
DEFAULT_CPD_MARGIN: int = 5
#: Slack on the nominal row step before two rows count as *not* time-adjacent. A row whose
#: predecessor is more than ``step * DEFAULT_MAX_STEP_FACTOR`` (floored by ``step +
#: DEFAULT_MAX_STEP_FLOOR_S``) behind it starts a new run, so "k consecutive windows" means
#: consecutive **in time**, not merely adjacent in the score array. ``step`` is the larger of
#: the window duration and the measured row pitch (:func:`row_pitch_seconds`): a per-cycle
#: table's rows are spaced by the cycle period, not by the 36 s a door cycle lasts.
DEFAULT_MAX_STEP_FACTOR: float = 1.5
DEFAULT_MAX_STEP_FLOOR_S: float = 60.0

_SECONDS_PER_DAY: float = 86400.0


# --------------------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------------------


def to_epoch_seconds(t: Any) -> np.ndarray:
    """Any timestamp-ish array -> float64 epoch **seconds**.

    Accepts ``datetime64`` of any unit, tz-aware pandas series/index, or a plain float
    array that is already in seconds (returned unchanged, as float64). ``NaT`` becomes NaN.
    """
    arr = np.asarray(t)
    if arr.dtype.kind == "M":
        ns = arr.astype("datetime64[ns]")
        out = ns.astype("float64") / 1e9
        # NaT casts to the int64 minimum, a FINITE -9.22e9 s, which would silently pass every
        # np.isfinite guard downstream (open-ended events, train-days, split membership).
        out[np.isnat(ns)] = np.nan
        return out
    if isinstance(t, (pd.Series, pd.Index, pd.DatetimeIndex)):
        return to_epoch_seconds(pd.DatetimeIndex(t).tz_localize(None).to_numpy())
    if arr.dtype.kind == "O":
        return to_epoch_seconds(pd.to_datetime(pd.Series(arr), utc=True).dt.tz_localize(None).to_numpy())
    return arr.astype(np.float64, copy=False)


def _scalar_seconds(t: Any) -> float:
    """One timestamp-ish scalar -> float epoch seconds (NaN for NaT/None)."""
    if t is None:
        return float("nan")
    if isinstance(t, (int, float, np.integer, np.floating)):
        return float(t)
    return float(to_epoch_seconds(np.asarray([t]))[0])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of ``True`` in a 1-D boolean mask, as inclusive ``(start, end)`` pairs."""
    if mask.size == 0:
        return []
    padded = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def row_pitch_seconds(times: Any, units: Any | None = None, *, fallback: float = 0.0) -> float:
    """Seconds between two consecutive rows of the same unit - what one row is worth in time.

    Rows are spaced by the *stride* (window stats), by the cycle period (per-cycle tables) or
    by the record cadence, never by the window duration, so the contiguity rule behind
    "k consecutive windows" and the VUS buffer must be counted in this unit. Gaps are taken
    **within** each unit when ``units`` is given: ten trains interleaved on one timeline would
    otherwise report a tenth of the true pitch. The result is the median positive gap; with
    fewer than two rows or no finite positive gap it is ``fallback``.
    """
    t = to_epoch_seconds(np.asarray(times).reshape(-1))
    if t.size < 2:
        return float(fallback)
    if units is None:
        groups = [np.arange(t.size)]
    else:
        u = np.asarray(units, dtype=object).reshape(-1)
        if u.size != t.size:
            raise ValueError(f"row_pitch_seconds: units has {u.size} entries for {t.size} times")
        groups = [np.flatnonzero(u == k) for k in pd.unique(u)]
    gaps: list[np.ndarray] = []
    for g in groups:
        if g.size >= 2:
            d = np.diff(np.sort(t[g]))
            gaps.append(d[np.isfinite(d) & (d > 0)])
    pooled = np.concatenate(gaps) if gaps else np.empty(0)
    return float(np.median(pooled)) if pooled.size else float(fallback)


def max_step_seconds(
    window_seconds: float | None,
    row_pitch_s: float | None = None,
    *,
    factor: float = DEFAULT_MAX_STEP_FACTOR,
    floor_s: float = DEFAULT_MAX_STEP_FLOOR_S,
) -> float:
    """Largest gap between two consecutive rows that still counts as *time-adjacent*.

    The scored slice is not dense: MetroPT windows are split at data gaps, the runner drops
    non-scoreable rows (post-repair blanking, compressor transitions) before measuring, and
    a cross-validation fold is a subset of the timeline. Array adjacency therefore does not
    imply time adjacency, so ``episodes`` needs an explicit step budget.

    The step is the larger of ``window_seconds`` and ``row_pitch_s`` (the measured gap
    between consecutive rows of one unit, :func:`row_pitch_seconds`). The window alone is
    wrong for any table whose rows are sparser than the window: the synthetic door table has
    36 s cycles every ~155 s, so a window-derived 96 s budget would never let three cycles
    count as consecutive and the >= 3-window rule could never fire. The pitch alone is wrong
    for overlapping windows (600 s windows every 300 s), where one dropped row is still
    inside the previous window.

    ``inf`` when neither value is a finite positive number means "no contiguity check" - the
    caller has not told us what one row is worth.
    """
    cands = [float(v) for v in (window_seconds, row_pitch_s) if v is not None]
    cands = [v for v in cands if np.isfinite(v) and v > 0.0]
    if not cands:
        return float("inf")
    w = max(cands)
    return max(w * float(factor), w + float(floor_s))


def _split_on_gaps(a: int, b: int, t: np.ndarray, max_step_s: float) -> list[tuple[int, int]]:
    """Cut an inclusive index run ``[a, b]`` wherever the time step exceeds ``max_step_s``."""
    if b <= a or not np.isfinite(max_step_s):
        return [(a, b)]
    gaps = np.flatnonzero(np.diff(t[a : b + 1]) > max_step_s)
    if gaps.size == 0:
        return [(a, b)]
    out: list[tuple[int, int]] = []
    start = a
    for g in gaps.tolist():
        out.append((start, a + g))
        start = a + g + 1
    out.append((start, b))
    return out


# --------------------------------------------------------------------------------------
# Alarm episodes
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Episode:
    """One alarm episode: a run (possibly several merged runs) of above-threshold windows.

    ``i_start`` / ``i_end`` index into the *caller's* score array (inclusive). ``t_start`` /
    ``t_end`` are epoch seconds so that arithmetic with ``H`` and ``merge_gap_s`` is trivial
    and unit-free.
    """

    unit: str
    i_start: int
    i_end: int
    t_start: float
    t_end: float
    peak_score: float
    n_windows: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Event:
    """One ground-truth fault episode: ``(unit, t_onset, t_failure)`` in epoch seconds.

    ``t_failure`` may be NaN for a statically-labelled condition that never fails (Cranfield,
    Ottawa); :func:`event_metrics` then treats the detection window as
    ``[t_onset - H, +inf)`` bounded by the unit's last observed window.
    """

    unit: str
    t_onset: float
    t_failure: float
    fault_type: str = "unknown"

    @classmethod
    def from_any(cls, unit: Any, t_onset: Any, t_failure: Any, fault_type: Any = "unknown") -> "Event":
        return cls(str(unit), _scalar_seconds(t_onset), _scalar_seconds(t_failure), str(fault_type))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def episodes(
    scores: np.ndarray,
    times: Any,
    threshold: float,
    *,
    k: int = DEFAULT_K_CONSECUTIVE,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    units: np.ndarray | None = None,
    max_step_s: float | None = None,
) -> list[Episode]:
    """Alarm episodes: ``score > threshold`` for ``>= k`` consecutive windows, merged when the
    gap between two runs is ``< merge_gap_s`` seconds.

    Parameters
    ----------
    scores : ``(n,)`` per-window anomaly score, higher = more anomalous.
    times : ``(n,)`` window end timestamps (``datetime64``, tz-aware pandas, or epoch seconds).
    threshold : scalar; the comparison is **strictly greater** (protocol wording).
    k : minimum consecutive above-threshold windows for an episode to exist at all. The
        ``k``-of-``k`` rule is applied *before* merging, so two 2-window blips 5 min apart do
        **not** become one episode when ``k=3``.
    merge_gap_s : two qualifying runs closer than this in time are one episode.
    units : ``(n,)`` series id per window - the component an alarm is raised on
        (``BenchData.series``: one door, one axle box; the train only where the train is the
        component). Runs never cross a series boundary and episodes are never merged across
        series, so cycles of two sibling doors can never pool into "3 consecutive windows".
        ``None`` = a single series.
    max_step_s : maximum gap between two rows that still counts as *consecutive*. Runs are
        cut wherever the time step exceeds it, **before** the ``k``-of-``k`` rule and before
        merging, so "3 consecutive windows" cannot be satisfied by three isolated points
        weeks apart. Build it with :func:`max_step_seconds` from the scored table's
        ``window_seconds``; ``None``/``inf`` disables the check (array adjacency only) and
        should be used only when one row's nominal duration is genuinely unknown.

    Returns episodes sorted by ``(unit, t_start)``. Rows are grouped by unit and sorted by
    time internally, so the caller does not have to pre-sort; ``i_start``/``i_end`` still
    index the caller's original arrays.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    t = to_epoch_seconds(times).reshape(-1)
    if s.size != t.size:
        raise ValueError(f"episodes: scores has {s.size} rows but times has {t.size}")
    if k < 1:
        raise ValueError(f"episodes: k must be >= 1, got {k}")
    if merge_gap_s < 0:
        raise ValueError(f"episodes: merge_gap_s must be >= 0, got {merge_gap_s}")
    step = float("inf") if max_step_s is None else float(max_step_s)
    if step <= 0:
        raise ValueError(f"episodes: max_step_s must be > 0 (or None), got {max_step_s}")
    u = np.full(s.size, "all", dtype=object) if units is None else np.asarray(units, dtype=object).reshape(-1)
    if u.size != s.size:
        raise ValueError(f"episodes: units has {u.size} rows but scores has {s.size}")

    out: list[Episode] = []
    for unit in pd.unique(u):
        sel = np.flatnonzero(u == unit)
        order = sel[np.argsort(t[sel], kind="stable")]
        su, tu = s[order], t[order]
        above = su > threshold
        contiguous: list[tuple[int, int]] = []
        for a, b in _runs(above):
            contiguous.extend(_split_on_gaps(a, b, tu, step))
        blocks = [(a, b) for a, b in contiguous if (b - a + 1) >= k]
        if not blocks:
            continue
        merged: list[list[int]] = [list(blocks[0])]
        for a, b in blocks[1:]:
            if (tu[a] - tu[merged[-1][1]]) < merge_gap_s:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        for a, b in merged:
            idx = order[a : b + 1]
            out.append(
                Episode(
                    unit=str(unit),
                    i_start=int(idx[0]),
                    i_end=int(idx[-1]),
                    t_start=float(tu[a]),
                    t_end=float(tu[b]),
                    peak_score=float(np.max(su[a : b + 1])),
                    n_windows=int(b - a + 1),
                )
            )
    out.sort(key=lambda e: (e.unit, e.t_start))
    return out


def train_days(times: Any, units: np.ndarray | None = None) -> float:
    """Observed **train-days**: summed per-unit observed span (last window - first window),
    in days. One unit observed for 12 h contributes 0.5; a single window contributes 0.

    This is the denominator of ``false_alarms_per_train_day`` and is computed from the scored
    slice itself, so a short validation slice cannot fake a generous false-alarm budget.
    """
    t = to_epoch_seconds(times).reshape(-1)
    if t.size == 0:
        return 0.0
    u = np.full(t.size, "all", dtype=object) if units is None else np.asarray(units, dtype=object).reshape(-1)
    total = 0.0
    for unit in pd.unique(u):
        tu = t[u == unit]
        tu = tu[np.isfinite(tu)]
        if tu.size >= 2:
            total += float(tu.max() - tu.min()) / _SECONDS_PER_DAY
    return total


def event_metrics(
    eps: Sequence[Episode],
    events: Sequence[Event],
    H: float,
    *,
    times: Any = None,
    units: np.ndarray | None = None,
    n_train_days: float | None = None,
) -> dict[str, Any]:
    """Operational AD metrics for one scored slice.

    An event is **detected** when at least one episode of the same series (``Episode.unit``
    == ``Event.unit``, both the component id) overlaps its
    detection window ``[t_onset - H, t_failure]`` (``H`` in seconds; ``t_failure`` NaN means
    the window stays open). Episodes overlapping no event window on their unit are **false
    alarms**. Nothing is point-adjusted: the episode is either inside the window or it is not.

    ``n_train_days`` is the false-alarm denominator; pass it, or pass ``times``/``units`` and
    it is computed with :func:`train_days`.

    Returns
    -------
    dict with ``n_events``, ``n_events_detected``, ``event_recall`` (NaN when there are no
    events - never 0.0, so a dataset with nothing to find cannot look like a failure),
    ``n_episodes``, ``n_false_alarms``, ``train_days``, ``false_alarms_per_train_day``,
    ``lead_times_s`` (one per detected event, from the first detecting episode's start to
    ``t_failure``), ``median_lead_time_s``, ``mean_lead_time_s``, ``lead_times_to_onset_s``
    and ``detected`` (bool per event, in the order given).
    """
    if n_train_days is None:
        if times is None:
            raise ValueError("event_metrics: pass n_train_days, or times (+ units) to derive it")
        n_train_days = train_days(times, units)
    H = float(H)

    hit_episode = np.zeros(len(eps), dtype=bool)
    detected: list[bool] = []
    leads: list[float] = []
    leads_onset: list[float] = []

    for ev in events:
        lo = ev.t_onset - H
        hi = ev.t_failure if np.isfinite(ev.t_failure) else np.inf
        first: Episode | None = None
        for j, e in enumerate(eps):
            if e.unit != ev.unit:
                continue
            if e.t_end >= lo and e.t_start <= hi:
                hit_episode[j] = True
                if first is None or e.t_start < first.t_start:
                    first = e
        detected.append(first is not None)
        if first is not None:
            leads.append(float(ev.t_failure - first.t_start) if np.isfinite(ev.t_failure) else float("nan"))
            leads_onset.append(float(ev.t_onset - first.t_start))

    n_fa = int((~hit_episode).sum())
    n_ev = len(events)
    n_det = int(sum(detected))
    finite_leads = [x for x in leads if np.isfinite(x)]
    return {
        "n_events": n_ev,
        "n_events_detected": n_det,
        "event_recall": (n_det / n_ev) if n_ev else float("nan"),
        "n_episodes": len(eps),
        "n_false_alarms": n_fa,
        "train_days": float(n_train_days),
        "false_alarms_per_train_day": (n_fa / n_train_days) if n_train_days > 0 else float("nan"),
        "lead_times_s": leads,
        "lead_times_to_onset_s": leads_onset,
        "median_lead_time_s": float(np.median(finite_leads)) if finite_leads else float("nan"),
        "mean_lead_time_s": float(np.mean(finite_leads)) if finite_leads else float("nan"),
        "detected": detected,
    }


# --------------------------------------------------------------------------------------
# Threshold-free pointwise metrics
# --------------------------------------------------------------------------------------


def _clean_pair(y_true: Any, score: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).reshape(-1).astype(np.int8)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    if y.size != s.size:
        raise ValueError(f"metric: y_true has {y.size} rows but score has {s.size}")
    ok = np.isfinite(s)
    return y[ok], s[ok]


def auroc(y_true: Any, score: Any) -> float:
    """Pointwise ROC-AUC. NaN when only one class is present (undefined, never 0.5)."""
    from sklearn.metrics import roc_auc_score

    y, s = _clean_pair(y_true, score)
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def auprc(y_true: Any, score: Any) -> float:
    """Pointwise average precision (PR-AUC). NaN when only one class is present."""
    from sklearn.metrics import average_precision_score

    y, s = _clean_pair(y_true, score)
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


# --------------------------------------------------------------------------------------
# VUS-PR  (vendored - see nebulax/bench/vus_tsb_ad.py)
# --------------------------------------------------------------------------------------
#
# Source   : TSB-AD, https://github.com/TheDatumOrg/TSB-AD, file
#            ``TSB_AD/evaluation/basic_metrics.py`` at commit
#            6beac72e11d1155ade40870492c00d0d1cfdcaaf.
# Licence  : Apache-2.0 (the plan says "MIT"; the upstream repository is Apache-2.0 at this
#            commit, and nebulax/bench/vus_tsb_ad.py records the licence that applies).
# Paper    : J. Paparrizos, P. Boniol, T. Palpanas, R. S. Tsay, A. Elmore, M. J. Franklin,
#            "Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series
#            Anomaly Detection", PVLDB 15(11), 2022.
#
# The implementation is a faithful port of upstream's ``RangeAUC_volume_opt`` /``RangeAUC``
# living in its own module, so there is exactly one body of code to diff against upstream.
# A previous revision of this file carried a *re-implementation* written from the published
# definition, which differed from upstream (e.g. 0.41667 vs 0.58333 on the 6-point case in
# ``test_r_auc_pr_hand_computed_case``) because it used a different buffer geometry, recall
# normalisation and PR integration rule. Do not re-derive it here: import it.
# --------------------------------------------------------------------------------------


def r_auc_pr(y_true: Any, score: Any, window: int = 0) -> float:
    """TSB-AD's **R-AUC-PR** at a single buffer size ``window``.

    Thin wrapper over :func:`nebulax.bench.vus_tsb_ad.r_auc_pr` that first drops non-finite
    pairs the way every other metric in this module does. NaN when the slice has only one
    class, or when upstream's own segmenter degenerates (see the vendored module).
    """
    y, s = _clean_pair(y_true, score)
    y = (y > 0).astype(np.int8)
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return _V.r_auc_pr(y, s, window=int(window))


def vus_pr(
    y_true: Any,
    score: Any,
    *,
    max_window: int = 10,
    window_sizes: Sequence[int] | None = None,
) -> float:
    """TSB-AD's **VUS-PR** - the ladder's ``primary_metric``: the volume under the
    precision/recall/buffer surface, i.e. the mean R-AUC-PR over buffer sizes
    ``0..max_window``.

    Identical to ``TSB_AD.evaluation.metrics.get_metrics(score, labels,
    slidingWindow=max_window)["VUS-PR"]`` - pinned by
    ``tests/test_bench_metrics.py::test_vus_pr_matches_upstream_tsb_ad_fixtures``.

    ``max_window`` is roughly how many windows a real anomaly may be detected early or late
    by; the runner derives it from ``window_seconds`` (~1 h) and records it on the results row
    as ``vus_buffer_windows`` / ``vus_buffer_seconds``, because the number is meaningless
    without it. ``window_sizes`` overrides the buffer set with an explicit list (each entry
    is evaluated with the single-window :func:`r_auc_pr`, so it is *not* upstream's volume
    and is only for diagnostics). Returns NaN when the slice has only one class.
    """
    if window_sizes is not None:
        vals = [r_auc_pr(y_true, score, int(w)) for w in window_sizes]
        finite = [v for v in vals if np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")
    y, s = _clean_pair(y_true, score)
    y = (y > 0).astype(np.int8)
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return _V.vus_pr(y, s, max_window=int(max_window))


def cls_metrics(y_true: Any, y_pred: Any, *, labels: Sequence[Any] | None = None) -> dict[str, Any]:
    """Macro-F1, balanced accuracy, accuracy, per-class F1 and the confusion matrix.

    ``labels`` fixes the class order (and therefore the confusion-matrix axes); it defaults to
    the sorted union of ``y_true`` and ``y_pred`` so a model that never predicts a class still
    gets a square matrix with that class in it.
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    if yt.size != yp.size:
        raise ValueError(f"cls_metrics: y_true has {yt.size} rows but y_pred has {yp.size}")
    lab = list(labels) if labels is not None else sorted(set(yt.tolist()) | set(yp.tolist()), key=str)
    per_class = f1_score(yt, yp, labels=lab, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(yt, yp, labels=lab, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "accuracy": float(accuracy_score(yt, yp)),
        "labels": [str(x) for x in lab],
        "per_class_f1": [float(x) for x in per_class],
        "confusion_matrix": confusion_matrix(yt, yp, labels=lab).tolist(),
        "n": int(yt.size),
    }


def monotonicity(score: Any, stage: Any) -> float:
    """Spearman rho of anomaly score against an **ordinal severity stage**.

    +1 means the score ranks the stages exactly in order (what a usable health index looks
    like); ~0 means the score carries no severity information. NaN if either side is constant.
    """
    from scipy.stats import spearmanr

    s = np.asarray(score, dtype=np.float64).reshape(-1)
    g = np.asarray(stage, dtype=np.float64).reshape(-1)
    if s.size != g.size:
        raise ValueError(f"monotonicity: score has {s.size} rows but stage has {g.size}")
    ok = np.isfinite(s) & np.isfinite(g)
    if ok.sum() < 3 or np.unique(s[ok]).size < 2 or np.unique(g[ok]).size < 2:
        return float("nan")
    rho = spearmanr(s[ok], g[ok]).statistic
    return float(rho)


# --------------------------------------------------------------------------------------
# Change-point detection: metrics + the score adapter
# --------------------------------------------------------------------------------------
#
# ``covering`` and ``f1_at_annotation_margin`` are ports of TCPDBench
# (https://github.com/alan-turing-institute/TCPDBench, MIT licence, (c) The Alan Turing
# Institute), per ``configs/model_ladder.yaml:evaluation.cpd_metrics``. Same honest
# provenance note as VUS-PR: written from the published definitions, not fetched.


def segments_from_cps(cps: Iterable[int], n: int) -> list[set[int]]:
    """Change points -> the partition of ``range(n)`` they induce.

    ``cps`` are the **start indices of new segments** (0 and ``n`` are implicit and ignored).
    """
    bounds = sorted({0, int(n)} | {int(c) for c in cps if 0 < int(c) < int(n)})
    return [set(range(bounds[i], bounds[i + 1])) for i in range(len(bounds) - 1)]


def _true_positives(T: set[int], X: set[int], margin: int) -> set[int]:
    """TCPDBench's greedy no-double-counting matching of predictions to annotations."""
    tp: set[int] = set()
    remaining = set(X)
    for tau in sorted(T):
        close = sorted((abs(tau - x), x) for x in remaining if abs(tau - x) <= margin)
        if not close:
            continue
        tp.add(tau)
        remaining.discard(close[0][1])
    return tp


def f1_at_margin(
    true_cps: Iterable[int],
    pred_cps: Iterable[int],
    *,
    margin: int = DEFAULT_CPD_MARGIN,
    alpha: float = 0.5,
    annotations: Mapping[Any, Iterable[int]] | None = None,
) -> dict[str, float]:
    """F1 at an annotation margin (TCPDBench ``f_measure``), with precision and recall.

    A prediction counts once, for the nearest unmatched true change point within ``margin``
    timesteps. With several annotators, pass ``annotations`` (``{annotator: cps}``); recall is
    then averaged over annotators against the union for precision, exactly as TCPDBench does.
    Returns ``{"f1", "precision", "recall", "margin"}``; NaN when there is nothing to score.
    """
    X = {int(c) for c in pred_cps}
    ann: dict[Any, set[int]] = (
        {k: {int(c) for c in v} for k, v in annotations.items()}
        if annotations is not None
        else {0: {int(c) for c in true_cps}}
    )
    ann = {k: v for k, v in ann.items() if v}
    if not ann:
        return {"f1": float("nan"), "precision": float("nan"), "recall": float("nan"), "margin": float(margin)}
    union: set[int] = set().union(*ann.values())
    if not X:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "margin": float(margin)}
    precision = len(_true_positives(union, X, margin)) / len(X)
    recall = float(np.mean([len(_true_positives(v, X, margin)) / len(v) for v in ann.values()]))
    denom = alpha * recall + (1.0 - alpha) * precision
    f1 = (precision * recall / denom) if denom > 0 else 0.0
    return {"f1": float(f1), "precision": float(precision), "recall": float(recall), "margin": float(margin)}


def f1_over_margins(
    true_cps: Iterable[int],
    pred_cps: Iterable[int],
    *,
    margins: Sequence[int] = tuple(range(1, DEFAULT_CPD_MARGIN * 2 + 1)),
    alpha: float = 0.5,
) -> dict[str, Any]:
    """F1 **integrated over the margin** rather than fixed at one value - the ladder's own
    caveat on ``f1_at_annotation_margin`` (same argument VUS makes for its buffer).

    Returns the mean F1 over ``margins``, the per-margin curve, and the F1 at
    :data:`DEFAULT_CPD_MARGIN` so the single-margin number stays reportable.
    """
    curve = {int(m): f1_at_margin(true_cps, pred_cps, margin=int(m), alpha=alpha)["f1"] for m in margins}
    vals = [v for v in curve.values() if np.isfinite(v)]
    return {
        "f1_mean_over_margins": float(np.mean(vals)) if vals else float("nan"),
        "f1_at_margin_5": f1_at_margin(true_cps, pred_cps, margin=DEFAULT_CPD_MARGIN, alpha=alpha)["f1"],
        "margins": [int(m) for m in margins],
        "f1_curve": [float(curve[int(m)]) for m in margins],
    }


def covering(true_cps: Iterable[int], pred_cps: Iterable[int], n: int) -> float:
    """TCPDBench's **covering metric**
    ``C(G, G') = (1/T) * sum_{A in G} |A| * max_{A' in G'} J(A, A')`` where ``G`` is the
    ground-truth partition, ``G'`` the predicted one and ``J`` the Jaccard index.

    1.0 is a perfect segmentation; a single-segment prediction scores the largest true
    segment's share of the series.
    """
    n = int(n)
    if n <= 0:
        return float("nan")
    G = segments_from_cps(true_cps, n)
    Gp = segments_from_cps(pred_cps, n)
    total = 0.0
    for A in G:
        best = max((len(A & B) / len(A | B)) for B in Gp)
        total += len(A) * best
    return float(total / n)


def cpd_metrics(
    true_cps: Iterable[int],
    pred_cps: Iterable[int],
    n: int,
    *,
    margin: int = DEFAULT_CPD_MARGIN,
) -> dict[str, Any]:
    """The ladder's two primary segmentation metrics in one call: ``covering`` and
    ``f1_at_annotation_margin`` (plus the margin-integrated variant)."""
    out: dict[str, Any] = {"covering": covering(true_cps, pred_cps, n)}
    out.update({f"cpd_{k}": v for k, v in f1_at_margin(true_cps, pred_cps, margin=margin).items()})
    out.update(f1_over_margins(true_cps, pred_cps))
    out["n_pred_cps"] = int(len({int(c) for c in pred_cps}))
    out["n_true_cps"] = int(len({int(c) for c in true_cps}))
    return out


def cpd_score(
    n: int,
    *,
    breakpoints: Iterable[int] | None = None,
    profile: Any = None,
    statistic: Any = None,
    tau_windows: float = 10.0,
) -> np.ndarray:
    """**The change-point score adapter.** Discrete/boolean CPD output -> a continuous
    per-window score that VUS-PR, AUPRC and :func:`episodes` can all consume.

    Exactly one of the three inputs is used, matching
    ``configs/model_ladder.yaml:evaluation.cpd_score_adapter.mapping``:

    ``breakpoints``
        Offline segmenters (ruptures / changeforest / aeon): an **exponential kernel on the
        distance to the nearest breakpoint**, ``exp(-d / tau_windows)``, so a window sitting
        on a detected change scores 1.0 and the score decays with distance. No breakpoints
        -> all-zero score (an honest "nothing detected", not NaN).
    ``profile``
        ClaSP and Bayesian online CPD: the per-timestamp curve is already continuous, so it
        is only min-max normalised to ``[0, 1]`` (constant curve -> all zeros).
    ``statistic``
        River drift detectors: the running cumulative statistic (Page-Hinkley) or the ADWIN
        drift magnitude - a non-negative running quantity, forward-filled over NaN and
        min-max normalised. A **boolean** flag array is accepted too and is turned into its
        running count before normalising, which is what makes an otherwise-uninformative
        boolean stream rankable.

    Returns ``(n,)`` float64 in ``[0, 1]``, higher = more anomalous.
    """
    n = int(n)
    given = [x is not None for x in (breakpoints, profile, statistic)]
    if sum(given) != 1:
        raise ValueError("cpd_score: pass exactly one of breakpoints=, profile=, statistic=")

    if breakpoints is not None:
        bkps = np.array(sorted({int(b) for b in breakpoints if 0 <= int(b) <= n}), dtype=np.float64)
        if bkps.size == 0:
            return np.zeros(n, dtype=np.float64)
        idx = np.arange(n, dtype=np.float64)
        d = np.abs(idx[:, None] - bkps[None, :]).min(axis=1) if bkps.size < 4096 else _min_dist(idx, bkps)
        tau = max(float(tau_windows), 1e-9)
        return np.exp(-d / tau)

    raw = profile if profile is not None else statistic
    x = np.asarray(raw, dtype=np.float64).reshape(-1)
    if x.size != n:
        raise ValueError(f"cpd_score: curve has {x.size} points but n={n}")
    if statistic is not None and np.asarray(raw).dtype == bool:
        x = np.cumsum(np.asarray(raw, dtype=np.float64).reshape(-1))
    if not np.isfinite(x).any():
        return np.zeros(n, dtype=np.float64)
    x = pd.Series(x).ffill().bfill().to_numpy(dtype=np.float64)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    return np.zeros(n, dtype=np.float64) if hi <= lo else (x - lo) / (hi - lo)


def _min_dist(idx: np.ndarray, bkps: np.ndarray) -> np.ndarray:
    """Distance to the nearest breakpoint without the (n, k) broadcast, for long series."""
    pos = np.searchsorted(bkps, idx)
    lo = bkps[np.clip(pos - 1, 0, bkps.size - 1)]
    hi = bkps[np.clip(pos, 0, bkps.size - 1)]
    return np.minimum(np.abs(idx - lo), np.abs(idx - hi))
