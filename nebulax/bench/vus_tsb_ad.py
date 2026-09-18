"""VUS-PR and R-AUC-PR, **vendored from TSB-AD**.

Source
------
Repository  https://github.com/TheDatumOrg/TSB-AD
File        ``TSB_AD/evaluation/basic_metrics.py``
Commit      ``6beac72e11d1155ade40870492c00d0d1cfdcaaf`` (pinned; fetched 2026-09-15)
Licence     **Apache License 2.0** - see https://github.com/TheDatumOrg/TSB-AD/blob/main/LICENSE
            (the NEBULA X plan says "MIT licence"; the upstream repository is in fact
            Apache-2.0 at this commit, and this header records the licence that actually
            applies. Apache-2.0 permits redistribution with attribution and a statement of
            changes, both given below.)

Papers
------
* J. Paparrizos et al., "Volume Under the Surface: A New Accuracy Evaluation Measure for
  Time-Series Anomaly Detection", PVLDB 15(11), 2022.
* Q. Liu and J. Paparrizos, "The Elephant in the Room: Towards A Reliable Time-Series
  Anomaly Detection Benchmark", NeurIPS 2024 (TSB-AD).

What this file is
-----------------
A faithful port of the upstream ``basic_metricor`` methods that VUS-PR is built from:
``range_convers_new``, ``new_sequence``, ``sequencing``, ``RangeAUC_volume_opt`` and
``RangeAUC``. TSB-AD's ``get_metrics`` computes ``VUS-PR`` as
``generate_curve(labels, score, slidingWindow, version="opt", thre=250)[-1]``, i.e. the
``avg_ap_3d`` returned by :func:`range_auc_volume_opt`, and that is exactly what
:func:`vus_pr` returns here.

**Changes from upstream** (required by Apache-2.0 s.4(b), and deliberately kept to the
minimum that keeps the numbers identical):

1. The methods are module-level functions instead of ``basic_metricor`` methods; the
   arithmetic, loop structure, index ranges and integration rules are unchanged.
2. Type hints, docstrings and ``float(...)`` on the returned scalars were added.
3. Guards were added for inputs upstream does not defend against and which a benchmark
   sweep hits routinely: an empty array, a constant score, and a label array with **no**
   positives or **all** positives (upstream divides by ``len(L)`` or ``np.sum(pred)`` and
   raises/emits NaN). These return ``nan`` rather than raising. When positives exist, no
   guard is on the computation path, so vendored parity is exact - see
   ``tests/test_bench_metrics.py::test_vus_pr_matches_upstream_tsb_ad_fixtures``.
4. ``RangeAUC`` (the single-window helper, **not** VUS-PR) divides by ``len(L)`` where ``L``
   comes from running the integer-transition segmenter over the *buffered float* label; for
   some windows that finds no segment and upstream raises ``ZeroDivisionError``. That case
   returns ``nan`` here.
5. ``thre`` is clamped to at most ``len(score)`` distinct sample points, because upstream's
   ``np.linspace(0, len(score)-1, thre)`` silently repeats thresholds on short arrays.
   Repeated thresholds contribute zero-width PR rectangles, so this changes no value; it
   only avoids the wasted work.

Nothing else in ``nebulax`` may re-implement these; :mod:`nebulax.bench.metrics` imports
from here so there is exactly one definition to diff against upstream.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "TSB_AD_SOURCE",
    "TSB_AD_COMMIT",
    "TSB_AD_LICENCE",
    "range_convers_new",
    "new_sequence",
    "sequencing",
    "range_auc_volume_opt",
    "range_auc",
    "vus_pr",
    "r_auc_pr",
]

TSB_AD_SOURCE: str = (
    "https://github.com/TheDatumOrg/TSB-AD/blob/"
    "6beac72e11d1155ade40870492c00d0d1cfdcaaf/TSB_AD/evaluation/basic_metrics.py"
)
TSB_AD_COMMIT: str = "6beac72e11d1155ade40870492c00d0d1cfdcaaf"
TSB_AD_LICENCE: str = "Apache-2.0"

#: Upstream's default number of threshold samples (``get_metrics(..., thre=250)``).
DEFAULT_THRE: int = 250


def range_convers_new(label: np.ndarray) -> list[tuple[int, int]]:
    """Upstream ``basic_metricor.range_convers_new``: maximal non-zero segments as inclusive
    ``(start, end)`` pairs."""
    anomaly_starts = np.where(np.diff(label) == 1)[0] + 1
    (anomaly_ends,) = np.where(np.diff(label) == -1)
    if len(anomaly_ends):
        if not len(anomaly_starts) or anomaly_ends[0] < anomaly_starts[0]:
            anomaly_starts = np.concatenate([[0], anomaly_starts])
    if len(anomaly_starts):
        if not len(anomaly_ends) or anomaly_ends[-1] < anomaly_starts[-1]:
            anomaly_ends = np.concatenate([anomaly_ends, [len(label) - 1]])
    return list(zip(anomaly_starts.tolist(), anomaly_ends.tolist()))


def new_sequence(label: np.ndarray, sequence_original: list[tuple[int, int]], window: int) -> list[tuple[int, int]]:
    """Upstream ``basic_metricor.new_sequence``: the anomaly segments widened by
    ``window // 2`` on each side and merged where they now touch."""
    a = max(sequence_original[0][0] - window // 2, 0)
    sequence_new: list[tuple[int, int]] = []
    for i in range(len(sequence_original) - 1):
        if sequence_original[i][1] + window // 2 < sequence_original[i + 1][0] - window // 2:
            sequence_new.append((a, sequence_original[i][1] + window // 2))
            a = sequence_original[i + 1][0] - window // 2
    sequence_new.append((a, min(sequence_original[len(sequence_original) - 1][1] + window // 2, len(label) - 1)))
    return sequence_new


def sequencing(x: np.ndarray, L: list[tuple[int, int]], window: int = 5) -> np.ndarray:
    """Upstream ``basic_metricor.sequencing``: the buffered label, weight
    ``sqrt(1 - d / window)`` over ``window // 2`` points either side of every segment."""
    label = x.copy().astype(float)
    length = len(label)
    for k in range(len(L)):
        s = L[k][0]
        e = L[k][1]
        x1 = np.arange(e + 1, min(e + window // 2 + 1, length))
        label[x1] += np.sqrt(1 - (x1 - e) / (window))
        x2 = np.arange(max(s - window // 2, 0), s)
        label[x2] += np.sqrt(1 - (s - x2) / (window))
    return np.minimum(np.ones(length), label)


def _degenerate(labels: np.ndarray, score: np.ndarray) -> bool:
    """True when upstream's arithmetic is undefined (see change 3 in the module docstring)."""
    if labels.size == 0 or score.size != labels.size:
        return True
    p = float(np.sum(labels))
    return p <= 0.0 or p >= labels.size


def range_auc_volume_opt(
    labels_original: np.ndarray, score: np.ndarray, windowSize: int, thre: int = DEFAULT_THRE
) -> tuple[float, float]:
    """Upstream ``basic_metricor.RangeAUC_volume_opt``, returning ``(VUS-ROC, VUS-PR)``.

    Upstream returns six values and ``generate_curve`` keeps the last two (``avg_auc_3d``,
    ``avg_ap_3d``); the four curve arrays are only used for plotting, so they are not built.
    """
    labels_original = np.asarray(labels_original).astype(int).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if _degenerate(labels_original, score):
        return float("nan"), float("nan")
    windowSize = int(max(0, windowSize))
    thre = int(np.clip(thre, 2, max(2, score.size)))

    window_3d = np.arange(0, windowSize + 1, 1)
    P = np.sum(labels_original)
    seq = range_convers_new(labels_original)
    l = new_sequence(labels_original, seq, windowSize)

    score_sorted = -np.sort(-score)
    auc_3d = np.zeros(windowSize + 1)
    ap_3d = np.zeros(windowSize + 1)

    grid = np.linspace(0, len(score) - 1, thre).astype(int)
    N_pred = np.zeros(thre)
    for k, i in enumerate(grid):
        N_pred[k] = np.sum(score >= score_sorted[i])
    tp = np.zeros(thre)  # upstream keeps this all-zero in the 'opt' path

    for window in window_3d:
        labels_extended = sequencing(labels_original, seq, window)
        L = new_sequence(labels_extended, seq, window)

        TF_list = np.zeros((thre + 2, 2))
        Precision_list = np.ones(thre + 1)
        j = 0

        for i in grid:
            threshold = score_sorted[i]
            pred = score >= threshold
            labels = labels_extended.copy()
            existence = 0

            for seg in L:
                labels[seg[0] : seg[1] + 1] = labels_extended[seg[0] : seg[1] + 1] * pred[seg[0] : seg[1] + 1]
                if (pred[seg[0] : (seg[1] + 1)] > 0).any():
                    existence += 1
            for seg in seq:
                labels[seg[0] : seg[1] + 1] = 1

            TP = 0.0
            N_labels = 0.0
            for seg in l:
                TP += np.dot(labels[seg[0] : seg[1] + 1], pred[seg[0] : seg[1] + 1])
                N_labels += np.sum(labels[seg[0] : seg[1] + 1])

            TP += tp[j]
            FP = N_pred[j] - TP

            existence_ratio = existence / len(L)
            P_new = (P + N_labels) / 2
            recall = min(TP / P_new, 1)

            TPR = recall * existence_ratio
            N_new = len(labels) - P_new
            FPR = FP / N_new
            Precision = TP / N_pred[j] if N_pred[j] else 1.0

            j += 1
            TF_list[j] = [TPR, FPR]
            Precision_list[j] = Precision

        TF_list[j + 1] = [1, 1]  # otherwise range-AUC stops earlier than (1, 1)

        width = TF_list[1:, 1] - TF_list[:-1, 1]
        height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
        auc_3d[window] = np.dot(width, height)

        width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
        height_PR = Precision_list[1:]
        ap_3d[window] = np.dot(width_PR, height_PR)

    return float(np.sum(auc_3d) / len(window_3d)), float(np.sum(ap_3d) / len(window_3d))


def range_auc(labels: np.ndarray, score: np.ndarray, window: int = 0, thre: int = DEFAULT_THRE) -> tuple[float, float]:
    """Upstream ``basic_metricor.RangeAUC(..., AUC_type="window", plot_ROC=True)``, returning
    ``(R-AUC-ROC, R-AUC-PR)`` at the **single** buffer size ``window``.

    Note that upstream's ``RangeAUC`` buffers the label with ``extend_postive_range`` while
    ``RangeAUC_volume_opt`` uses ``sequencing``; the two differ by one index at the trailing
    edge of a segment. Both are reproduced as they are, because VUS-PR is the volume version
    and R-AUC-PR is the single-window one - swapping either would silently change published
    numbers.
    """
    labels = np.asarray(labels).astype(int).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if _degenerate(labels, score):
        return float("nan"), float("nan")
    thre = int(np.clip(thre, 2, max(2, score.size)))
    window = int(max(0, window))

    score_sorted = -np.sort(-score)
    P = np.sum(labels)
    labels = _extend_postive_range(labels, window=window)
    L = range_convers_new(labels)
    if not L:
        # Upstream defect, reproduced as a guard rather than as a crash: after buffering, the
        # label array is float, so ``range_convers_new``'s ``np.diff(label) == 1`` test can
        # match nothing, and upstream then divides by ``len(L) == 0``. VUS-PR does not go
        # through this path (it segments the ORIGINAL integer labels), so this only affects
        # the single-window R-AUC-PR helper.
        return float("nan"), float("nan")

    TPR_list = [0.0]
    FPR_list = [0.0]
    Precision_list = [1.0]
    for i in np.linspace(0, len(score) - 1, thre).astype(int):
        pred = score >= score_sorted[i]
        TPR, FPR, Precision = _tpr_fpr_range_auc(labels, pred, P, L)
        TPR_list.append(TPR)
        FPR_list.append(FPR)
        Precision_list.append(Precision)
    TPR_list.append(1.0)
    FPR_list.append(1.0)

    tpr = np.array(TPR_list)
    fpr = np.array(FPR_list)
    prec = np.array(Precision_list)
    auc_range = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    ap_range = float(np.sum((tpr[1:-1] - tpr[:-2]) * prec[1:]))
    return auc_range, ap_range


def _extend_postive_range(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Upstream ``basic_metricor.extend_postive_range`` (used by ``RangeAUC`` only)."""
    label = x.copy().astype(float)
    L = range_convers_new(label)
    length = len(label)
    for k in range(len(L)):
        s = L[k][0]
        e = L[k][1]
        x1 = np.arange(e, min(e + window // 2, length))
        label[x1] += np.sqrt(1 - (x1 - e) / (window)) if window else 0.0
        x2 = np.arange(max(s - window // 2, 0), s)
        label[x2] += np.sqrt(1 - (s - x2) / (window)) if window else 0.0
    return np.minimum(np.ones(length), label)


def _tpr_fpr_range_auc(labels: np.ndarray, pred: np.ndarray, P: float, L: list[tuple[int, int]]):
    """Upstream ``basic_metricor.TPR_FPR_RangeAUC``."""
    indices = np.where(labels == 1)[0]
    product = labels * pred
    TP = np.sum(product)
    newlabels = product.copy()
    newlabels[indices] = 1

    P_new = (P + np.sum(newlabels)) / 2
    recall = min(TP / P_new, 1)

    existence = 0
    for seg in L:
        if np.sum(product[seg[0] : (seg[1] + 1)]) > 0:
            existence += 1
    existence_ratio = existence / len(L)

    TPR_RangeAUC = recall * existence_ratio
    FP = np.sum(pred) - TP
    N_new = len(labels) - P_new
    FPR_RangeAUC = FP / N_new
    Precision_RangeAUC = TP / np.sum(pred) if np.sum(pred) else 1.0
    return TPR_RangeAUC, FPR_RangeAUC, Precision_RangeAUC


def vus_pr(y_true: np.ndarray, score: np.ndarray, *, max_window: int = 10, thre: int = DEFAULT_THRE) -> float:
    """TSB-AD's ``VUS-PR``: the mean R-AUC-PR over buffer sizes ``0..max_window``.

    Identical to ``get_metrics(score, labels, slidingWindow=max_window)["VUS-PR"]``.
    """
    return range_auc_volume_opt(y_true, score, max_window, thre)[1]


def r_auc_pr(y_true: np.ndarray, score: np.ndarray, *, window: int = 0, thre: int = DEFAULT_THRE) -> float:
    """TSB-AD's ``R-AUC-PR`` at one buffer size (``RangeAUC(..., plot_ROC=True)[1]``)."""
    return range_auc(y_true, score, window, thre)[1]
