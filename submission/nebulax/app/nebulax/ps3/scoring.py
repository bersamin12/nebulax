"""The four organiser metrics for Problem Statement 3, re-implemented exactly.

Each function is a line-by-line transcription of the formula disclosed in the corresponding Info
Kit, so a number produced here is the number `judge_leaderboard.py` will produce:

* :func:`iou_f1` - Door, *IoU-weighted F1* (`Door_Subsystem_Info_Kit.md` section 4).
* :func:`rank_decay` - ACV, *linear rank decay* ``(n - (r - 1)) / n`` (`ACV_..._Info_Kit.md` section 4).
* :func:`macro_f1` - Rail, *macro F1* over the fixed three-class vocabulary (`Rail_..._Info_Kit.md` section 4).
* :func:`mape_score` - SHM, ``max(0, 1 - MAPE)`` (`SHM_Info_Kit.md` section 4).

Plus :func:`combined_scores` for the two cross-subsystem figures in the specification section 5.3.

Every function is pure: values in, floats/dicts out. No I/O, no global state, no fitting - and in
particular nothing here ever looks at a prediction before deciding how to score it (no point
adjustment, no threshold search), matching `nebulax/bench/metrics.py`'s protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from nebulax.ps3.common import RAIL_LABELS, to_ms

__all__ = [
    "Segment",
    "Match",
    "iou",
    "iou_f1",
    "rank_decay",
    "rank_decay_mean",
    "macro_f1",
    "macro_f1_from_class_f1",
    "class_f1_report",
    "mape_score",
    "combined_scores",
]


# --------------------------------------------------------------------------------------
# Door: IoU-weighted F1
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """One labelled interval, in integer epoch-milliseconds."""

    start: int
    end: int
    label: str


@dataclass(frozen=True)
class Match:
    """One greedy true<->pred assignment and the IoU credit it earns."""

    true_index: int
    pred_index: int
    iou: float
    label: str


def _coerce_segments(obj: Any, *, what: str) -> list[Segment]:
    """Accept tuples, mappings, or a DataFrame of segments; normalise to :class:`Segment`."""
    if isinstance(obj, pd.DataFrame):
        df = obj
        start_col = next((c for c in ("t_start", "start_time", "start") if c in df.columns), None)
        end_col = next((c for c in ("t_end", "end_time", "end") if c in df.columns), None)
        label_col = next((c for c in ("label", "status", "prediction") if c in df.columns), None)
        if start_col is None or end_col is None or label_col is None:
            raise ValueError(f"{what}: DataFrame needs start/end/label columns, got {list(df.columns)}")
        rows: Iterable[Any] = (
            (a, b, c) for a, b, c in zip(df[start_col], df[end_col], df[label_col], strict=True)
        )
    else:
        rows = obj

    out: list[Segment] = []
    for i, row in enumerate(rows):
        if isinstance(row, Segment):
            start, end, label = row.start, row.end, row.label
        elif isinstance(row, Mapping):
            try:
                start = row[next(k for k in ("start", "t_start", "start_time") if k in row)]
                end = row[next(k for k in ("end", "t_end", "end_time") if k in row)]
                label = row[next(k for k in ("label", "status", "prediction") if k in row)]
            except StopIteration as exc:
                raise ValueError(f"{what}[{i}]: mapping needs start/end/label keys, got {sorted(row)}") from exc
        else:
            seq = list(row)
            if len(seq) != 3:
                raise ValueError(f"{what}[{i}]: expected (start, end, label), got {row!r}")
            start, end, label = seq
        s, e = to_ms(start), to_ms(end)
        if e < s:
            raise ValueError(f"{what}[{i}]: end {end!r} is before start {start!r}")
        out.append(Segment(s, e, str(label).strip()))
    return out


def iou(a: Any, b: Any) -> float:
    """Intersection over union of two segments, exactly as the Door Info Kit defines it.

    ``intersection = max(0, min(ends) - max(starts))``,
    ``union = len(a) + len(b) - intersection``, ``IoU = 0`` when ``union <= 0``.
    """
    (sa, ea), (sb, eb) = (to_ms(a[0]), to_ms(a[1])), (to_ms(b[0]), to_ms(b[1]))
    inter = max(0, min(ea, eb) - max(sa, sb))
    union = (ea - sa) + (eb - sb) - inter
    return 0.0 if union <= 0 else float(inter) / float(union)


def iou_f1(true_segments: Any, pred_segments: Any) -> dict[str, Any]:
    """Door metric: greedy one-to-one, same-label, IoU-weighted F1.

    Matching (Info Kit section 4.1): candidate pairs are same-label with ``IoU > 0``; pairs are
    assigned **highest IoU first**, one-to-one, each segment consumed when matched. Credit for a
    match is its IoU itself (section 4.2)::

        soft_recall    = sum(IoU over matches) / n_true
        soft_precision = sum(IoU over matches) / n_pred
        score          = harmonic mean (0 when both are 0)

    **Tie-break (ours, the kit does not specify one):** equal-IoU candidates are ordered by
    ``(-iou, true_index, pred_index)``, i.e. the earliest true segment wins, then the earliest
    prediction. Segments are compared in the order given, so pass them in stream order for a
    reproducible result. Ties only change *which* equally-good pairing is chosen, never the score,
    unless the tied pairs overlap - which is why the rule is fixed rather than left to sort order.

    Returns ``score``, ``soft_recall``, ``soft_precision``, ``matches`` (list of :class:`Match`
    dicts), ``misses`` (unmatched true indices), ``false_positives`` (unmatched pred indices) and
    the counts behind them.
    """
    truth = _coerce_segments(true_segments, what="true_segments")
    preds = _coerce_segments(pred_segments, what="pred_segments")

    candidates: list[tuple[float, int, int]] = []
    for ti, t in enumerate(truth):
        for pi, p in enumerate(preds):
            if t.label != p.label:
                continue
            value = iou((t.start, t.end), (p.start, p.end))
            if value > 0.0:
                candidates.append((value, ti, pi))
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

    used_true: set[int] = set()
    used_pred: set[int] = set()
    matches: list[Match] = []
    for value, ti, pi in candidates:
        if ti in used_true or pi in used_pred:
            continue
        used_true.add(ti)
        used_pred.add(pi)
        matches.append(Match(true_index=ti, pred_index=pi, iou=float(value), label=truth[ti].label))

    iou_sum = float(sum(m.iou for m in matches))
    soft_recall = iou_sum / len(truth) if truth else 0.0
    soft_precision = iou_sum / len(preds) if preds else 0.0
    denom = soft_recall + soft_precision
    score = 0.0 if denom <= 0 else 2.0 * soft_recall * soft_precision / denom

    matches.sort(key=lambda m: (m.true_index, m.pred_index))
    return {
        "score": float(score),
        "soft_recall": float(soft_recall),
        "soft_precision": float(soft_precision),
        "matches": [asdict(m) for m in matches],
        "misses": [i for i in range(len(truth)) if i not in used_true],
        "false_positives": [i for i in range(len(preds)) if i not in used_pred],
        "n_true": len(truth),
        "n_pred": len(preds),
        "n_matches": len(matches),
        "n_misses": len(truth) - len(matches),
        "n_false_positives": len(preds) - len(matches),
        "iou_sum": iou_sum,
    }


# --------------------------------------------------------------------------------------
# ACV: linear rank decay
# --------------------------------------------------------------------------------------


def rank_decay(ranked: Sequence[Any], true_car: Any, n: int | None = None) -> float:
    """ACV metric for one case file: ``(n - (r - 1)) / n``, or 0 when the true car is absent.

    ``ranked`` is most- to least-likely faulty, using the file's own two-digit car ids; ``n``
    defaults to the number of cars ranked. Ids are compared as stripped strings, so pass them
    exactly as the headers spell them (``"03"``, never ``3`` or ``"Car 3"``). A duplicate id
    scores at its first position - the submission validator is what rejects duplicates.
    """
    items = [str(x).strip() for x in list(ranked)]
    total = int(n) if n is not None else len(items)
    if total <= 0:
        return 0.0
    target = str(true_car).strip()
    if target not in items:
        return 0.0
    r = items.index(target) + 1
    if r > total:
        return 0.0
    return float(total - (r - 1)) / float(total)


def rank_decay_mean(rankings: Mapping[str, Sequence[Any]], true_cars: Mapping[str, Any]) -> dict[str, Any]:
    """Mean :func:`rank_decay` over cases, keyed by file id (the ACV ``primary_metric``).

    Every key of ``true_cars`` is scored; a case with no ranking submitted scores 0.
    """
    per_file = {
        fid: rank_decay(list(rankings.get(fid, [])), car)
        for fid, car in sorted(true_cars.items())
    }
    score = float(np.mean(list(per_file.values()))) if per_file else 0.0
    return {"score": score, "per_file": per_file, "n": len(per_file)}


# --------------------------------------------------------------------------------------
# Rail: macro F1 over the fixed vocabulary
# --------------------------------------------------------------------------------------


def class_f1_report(
    y_true: Any, y_pred: Any, labels: Sequence[str] = RAIL_LABELS
) -> dict[str, Any]:
    """Per-class precision / recall / F1 / support plus macro F1, over a **fixed** vocabulary.

    Classes absent from both arrays still appear, with F1 = 0 - that is the whole point of macro
    F1 on this dataset (Rail Info Kit section 4: "a class the model never gets right contributes
    an F1 of 0 to the average, regardless of how rare that class is"). Zero-denominator precision
    or recall is 0, matching ``sklearn``'s ``zero_division=0``.
    """
    yt = np.asarray(list(y_true), dtype=object).reshape(-1)
    yp = np.asarray(list(y_pred), dtype=object).reshape(-1)
    if yt.size != yp.size:
        raise ValueError(f"macro_f1: y_true has {yt.size} rows but y_pred has {yp.size}")
    lab = [str(x) for x in labels]
    yt = np.array([str(v).strip() for v in yt], dtype=object)
    yp = np.array([str(v).strip() for v in yp], dtype=object)
    unknown = sorted({str(v) for v in np.concatenate([yt, yp])} - set(lab)) if yt.size else []

    per_class: dict[str, dict[str, float]] = {}
    for c in lab:
        tp = int(np.sum((yt == c) & (yp == c)))
        fp = int(np.sum((yt != c) & (yp == c)))
        fn = int(np.sum((yt == c) & (yp != c)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(tp + fn),
        }
    matrix = [[int(np.sum((yt == a) & (yp == b))) for b in lab] for a in lab]
    return {
        "macro_f1": float(np.mean([per_class[c]["f1"] for c in lab])) if lab else 0.0,
        "labels": lab,
        "per_class": per_class,
        "per_class_f1": [per_class[c]["f1"] for c in lab],
        "confusion_matrix": matrix,
        "accuracy": float(np.mean(yt == yp)) if yt.size else 0.0,
        "n": int(yt.size),
        "unknown_labels": unknown,
    }


def macro_f1(y_true: Any, y_pred: Any, labels: Sequence[str] = RAIL_LABELS) -> float:
    """Rail metric: unweighted mean of the per-class F1 over ``labels`` (default the three rail classes)."""
    return class_f1_report(y_true, y_pred, labels)["macro_f1"]


def macro_f1_from_class_f1(values: Iterable[float]) -> float:
    """Macro F1 straight from per-class F1 values - the form the Info Kit's worked example uses."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


# --------------------------------------------------------------------------------------
# SHM: max(0, 1 - MAPE)
# --------------------------------------------------------------------------------------


def mape_score(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """SHM metric: ``MAPE = mean(|true - pred| / |true|)``, ``score = max(0, 1 - MAPE)``.

    Returns ``mape``, ``score``, ``n`` and the per-file absolute percentage errors (``ape``), which
    the CV report splits by damage mode. True values must be finite and non-zero - the organisers'
    damage labels are strictly positive, and a zero would make the percentage undefined.
    """
    yt = np.asarray(list(y_true), dtype=float).reshape(-1)
    yp = np.asarray(list(y_pred), dtype=float).reshape(-1)
    if yt.size != yp.size:
        raise ValueError(f"mape_score: y_true has {yt.size} rows but y_pred has {yp.size}")
    if yt.size == 0:
        raise ValueError("mape_score: no rows")
    if not np.all(np.isfinite(yt)) or np.any(yt == 0.0):
        raise ValueError("mape_score: y_true must be finite and non-zero (percentage error is relative to it)")
    if not np.all(np.isfinite(yp)):
        raise ValueError("mape_score: y_pred contains NaN/Inf")
    ape = np.abs(yt - yp) / np.abs(yt)
    mape = float(ape.mean())
    return {"mape": mape, "score": float(max(0.0, 1.0 - mape)), "n": int(yt.size), "ape": ape.tolist()}


# --------------------------------------------------------------------------------------
# Specification section 5.3: the two cross-subsystem figures
# --------------------------------------------------------------------------------------


def combined_scores(per_task: Mapping[str, float]) -> dict[str, Any]:
    """``Overall`` = sum over all four subsystems / 4; ``Average`` = mean over those attempted.

    Unattempted subsystems are simply absent from ``per_task`` (or ``None``); they contribute 0 to
    Overall and are excluded from Average.
    """
    attempted = {k: float(v) for k, v in per_task.items() if v is not None}
    total = float(sum(attempted.values()))
    return {
        "overall": total / 4.0,
        "average": (total / len(attempted)) if attempted else 0.0,
        "attempted": sorted(attempted),
        "per_task": attempted,
    }
