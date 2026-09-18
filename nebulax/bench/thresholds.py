"""Threshold calibration - **validation scores only**.

``configs/model_ladder.yaml:evaluation.threshold_calibration`` says ``fit_on:
validation_only`` and the ladder lists ``test_data_in_threshold_calibration`` as *forbidden*.
That rule is enforced three ways here, not just documented:

1. **By API shape.** :func:`calibrate` takes a :class:`ValidationScores` bundle, and that
   bundle refuses to be constructed with ``part != "val"``. There is no argument on any
   public function of this module that a test-slice score array could legally be passed as.
2. **By call order in the runner.** ``nebulax.bench.runner`` fits, scores the validation
   slice, calibrates, and only then computes test scores - the test score array does not
   exist as a Python object at the moment :func:`calibrate` is called. One test
   (``tests/test_bench_runner.py::test_no_test_index_reaches_calibrate_or_fit``) monkeypatches
   both entry points and asserts it.
3. **By audit trail.** The returned :class:`ThresholdSet` carries the row indices it was
   calibrated on. ``ThresholdSet.as_dict`` puts their count, range and a 16-hex digest into
   the results row, and the runner writes the indices themselves to
   ``<splits_dir>/<config_hash>.calibration.parquet``, so any published threshold can be
   joined back to the exact validation rows it came from and checked against the split
   parquet after the fact.

Three thresholds are produced for every run, all from the same validation scores:

``budget``  the most sensitive threshold a **descending first-violation scan** reaches while
            still meeting the operational false-alarm budget of ``<= 1 alarm episode per 7
            train-days``, using the protocol's own episode definition (>= 3 consecutive
            windows, merged under 1 h), so the budget is counted in *episodes an operator
            would see*, not in points. It is the lowest candidate **visited** before the first
            budget violation, which is not necessarily the lowest budget-compliant candidate
            on the grid - the episode count is not monotone in the threshold. The error
            direction is the safe one; see :func:`calibrate`.
``q995`` / ``q999``  the 99.5 % and 99.9 % score quantiles the protocol also asks to report.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from nebulax.bench.metrics import (
    DEFAULT_K_CONSECUTIVE,
    DEFAULT_MERGE_GAP_S,
    episodes,
    max_step_seconds,
    row_pitch_seconds,
    train_days,
)

__all__ = [
    "ValidationScores",
    "Threshold",
    "ThresholdSet",
    "calibrate",
    "quantile_thresholds",
    "DEFAULT_FAR_BUDGET",
    "DEFAULT_QUANTILES",
]

#: ``<= 1 episode per 7 train-days`` -> episodes per train-day.
DEFAULT_FAR_BUDGET: float = 1.0 / 7.0
#: Also reported, per ``evaluation.threshold_calibration.also_report_quantiles``.
DEFAULT_QUANTILES: tuple[float, ...] = (0.995, 0.999)


class TestScoresLeakedError(RuntimeError):
    """Raised when something tries to calibrate on anything but the validation slice."""


@dataclass(frozen=True, slots=True)
class ValidationScores:
    """The **only** thing :func:`calibrate` accepts: scores from the validation slice.

    ``part`` must be ``"val"``. It exists so that handing calibration a test slice is a
    ``TestScoresLeakedError`` at construction time rather than a silently optimistic number,
    and so the object itself records which row indices were used.
    """

    scores: np.ndarray
    times: np.ndarray
    units: np.ndarray
    index: np.ndarray
    part: str = "val"
    #: component id per row (``BenchData.series``); episodes are counted per series while
    #: train-days are counted per unit. ``None`` = the units.
    series: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.part != "val":
            raise TestScoresLeakedError(
                f"ValidationScores(part={self.part!r}): thresholds may only be calibrated on the "
                f"validation slice (evaluation.threshold_calibration.fit_on = validation_only)"
            )
        s = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        object.__setattr__(self, "scores", s)
        object.__setattr__(self, "times", np.asarray(self.times).reshape(-1))
        object.__setattr__(self, "units", np.asarray(self.units, dtype=object).reshape(-1))
        object.__setattr__(self, "index", np.asarray(self.index, dtype=np.int64).reshape(-1))
        ser = self.units if self.series is None else np.asarray(self.series, dtype=object).reshape(-1)
        object.__setattr__(self, "series", ser)
        n = s.size
        for name in ("times", "units", "index", "series"):
            got = getattr(self, name).size
            if got != n:
                raise ValueError(f"ValidationScores: scores has {n} rows but {name} has {got}")

    def __len__(self) -> int:
        return int(self.scores.size)


@dataclass(frozen=True, slots=True)
class Threshold:
    """One calibrated threshold plus what it actually achieved **on validation**."""

    name: str
    value: float
    method: str
    val_episodes: int
    val_train_days: float
    val_far_per_train_day: float
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": float(self.value),
            "method": self.method,
            "val_episodes": int(self.val_episodes),
            "val_train_days": float(self.val_train_days),
            "val_far_per_train_day": float(self.val_far_per_train_day),
        }


@dataclass(frozen=True, slots=True)
class ThresholdSet:
    """All thresholds for one run, with the audit trail of what they were fitted on."""

    thresholds: dict[str, Threshold]
    calibrated_on_index: np.ndarray
    n_val_rows: int
    window_seconds: float
    k_consecutive: int
    merge_gap_s: float
    max_step_s: float = float("inf")

    @property
    def primary(self) -> Threshold:
        """The operational threshold: the false-alarm-budget one."""
        return self.thresholds["budget"]

    def __getitem__(self, key: str) -> Threshold:
        return self.thresholds[key]

    def items(self):
        return self.thresholds.items()

    def index_digest(self) -> str:
        """16-hex digest of the exact row indices this was calibrated on.

        The full index array is far too large for a results cell, so the row carries this
        digest and the runner writes the indices themselves to
        ``<splits_dir>/<config_hash>.calibration.parquet``. Recomputing the digest over that
        file's ``row_index`` column proves the published threshold came from those rows and no
        others - which is the audit trail this module's docstring promises.
        """
        idx = np.asarray(self.calibrated_on_index, dtype=np.int64)
        return hashlib.sha256(idx.tobytes() + str(idx.size).encode()).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_val_rows": int(self.n_val_rows),
            "window_seconds": float(self.window_seconds),
            "k_consecutive": int(self.k_consecutive),
            "merge_gap_s": float(self.merge_gap_s),
            "max_step_s": float(self.max_step_s),
            "calibrated_on_n": int(np.asarray(self.calibrated_on_index).size),
            "calibrated_on_digest": self.index_digest(),
            "calibrated_on_min": int(np.min(self.calibrated_on_index)) if self.n_val_rows else -1,
            "calibrated_on_max": int(np.max(self.calibrated_on_index)) if self.n_val_rows else -1,
        }
        for name, thr in self.thresholds.items():
            out[f"threshold_{name}"] = float(thr.value)
            out[f"threshold_{name}_val_far_per_train_day"] = float(thr.val_far_per_train_day)
            out[f"threshold_{name}_val_episodes"] = int(thr.val_episodes)
            # The denominator travels with the numerator so the runner can micro-average the
            # rate over folds instead of averaging per-fold rates - see runner._aggregate_folds.
            out[f"threshold_{name}_val_train_days"] = float(thr.val_train_days)
        return out


def quantile_thresholds(scores_val: Any, quantiles: Sequence[float] = DEFAULT_QUANTILES) -> dict[str, float]:
    """``{"q995": ..., "q999": ...}`` - plain score quantiles of the validation slice.

    Non-finite scores are dropped first (a model that emits NaN has already failed
    ``AnomalyDetector.score``'s own check, so this is belt-and-braces).
    """
    s = np.asarray(scores_val, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {f"q{str(q).replace('0.', '').ljust(3, '0')}": float("nan") for q in quantiles}
    out: dict[str, float] = {}
    for q in quantiles:
        key = "q" + f"{q:.4f}".split(".")[1].rstrip("0")
        out[key] = float(np.quantile(s, q))
    return out


def _far_at(
    thr: float, vs: ValidationScores, k: int, merge_gap_s: float, n_days: float, max_step_s: float
) -> tuple[int, float]:
    eps = episodes(
        vs.scores, vs.times, thr, k=k, merge_gap_s=merge_gap_s, units=vs.series, max_step_s=max_step_s
    )
    if not np.isfinite(n_days):
        return len(eps), float("nan")  # fabricated timeline: no train-days to divide by
    far = (len(eps) / n_days) if n_days > 0 else float("inf" if eps else 0.0)
    return len(eps), far


def calibrate(
    validation: ValidationScores,
    *,
    far_budget_episodes_per_train_day: float = DEFAULT_FAR_BUDGET,
    window_seconds: float = 600.0,
    k_consecutive: int = DEFAULT_K_CONSECUTIVE,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    n_candidates: int = 128,
    max_step_s: float | None = None,
    budget_applicable: bool = True,
) -> ThresholdSet:
    """Calibrate the operational threshold on **validation scores only**.

    The budget threshold is the most sensitive candidate reachable by scanning the search
    grid downwards from the top and **stopping at the first candidate that breaks the budget**
    (``far_budget_episodes_per_train_day * train_days``). Candidates are ``n_candidates``
    values spanning the median to the maximum validation score. This is a first-violation
    scan, not an exhaustive grid search and not a bisection: the episode count is *not*
    monotone in the threshold (lowering it can merge two episodes into one, or split one into
    two when the surviving windows fall more than ``merge_gap_s`` apart), so a lower candidate
    might also have satisfied the budget. The error direction is deliberately the safe one -
    stopping early can only return a threshold at or above the true boundary, i.e. a *less*
    sensitive detector that raises no more alarms than the budget allows.

    A solution always exists: the comparison is strictly greater, so the grid's top point (the
    maximum validation score) raises no alarms at all. Two flags in ``meta`` say what the
    search actually found - ``budget_binding`` is ``False`` when even the grid's *lowest*
    candidate (the validation median) stays inside the budget, which means the budget is not
    what is setting this threshold and the model could be run more sensitively; ``silent`` is
    ``True`` when the chosen threshold raises **no** validation episode at all, i.e. the score
    has no usable operating point on this slice - a calibrated but mute detector, which the
    report must show as such rather than as a clean zero-false-alarm result.

    Parameters
    ----------
    validation : :class:`ValidationScores`. Test scores cannot be passed - see the module
        docstring.
    far_budget_episodes_per_train_day : default ``1/7`` = the protocol's "<= 1 episode per 7
        train-days".
    window_seconds : the scored window length; carried into the result so the runner can
        record what the budget was counted against and one input to the episode contiguity
        budget (with the row pitch measured on the validation slice). Train-days come from the
        observed timestamps, which already account for gaps.
    k_consecutive, merge_gap_s, max_step_s : the episode definition, identical to the one used
        at test time; calibrating with a different one would make the budget meaningless. The
        runner passes ``max_step_s`` computed ONCE per run from the whole table's row pitch,
        so calibration and test never count episodes under different contiguity budgets.
        ``None`` derives it from the validation slice itself (stand-alone use).
    budget_applicable : ``False`` on a dataset with a fabricated timeline (Cranfield, Ottawa:
        the adapter placed the recordings on synthetic anchors), where "train-days" is not a
        quantity. The ``budget`` threshold is then the 99.5 % validation quantile, its
        train-days and false-alarm rate are NaN and its method says why, so the leaderboard
        can print "n/a" instead of a rate nobody measured.
    """
    if not isinstance(validation, ValidationScores):
        raise TypeError(
            "thresholds.calibrate: pass a ValidationScores bundle (built from the VALIDATION "
            "slice only); see nebulax/bench/thresholds.py's module docstring"
        )
    s = validation.scores
    finite = s[np.isfinite(s)]
    if finite.size == 0:
        raise ValueError("thresholds.calibrate: validation scores are empty or all non-finite")

    n_days = train_days(validation.times, validation.units) if budget_applicable else float("nan")
    budget_eps = far_budget_episodes_per_train_day * n_days
    # Same episode definition as test time, contiguity rule included (see metrics.episodes).
    if max_step_s is None:
        max_step_s = max_step_seconds(window_seconds, row_pitch_seconds(validation.times, validation.series))
    max_step_s = float(max_step_s)

    lo_q, hi = float(np.quantile(finite, 0.5)), float(finite.max())
    grid = np.unique(np.concatenate([np.linspace(lo_q, hi, int(n_candidates)), [hi]]))
    # Descending first-violation scan: walk the grid downwards and stop at the first candidate
    # that breaks the budget; the lowest candidate VISITED before that wins. This is neither a
    # bisection nor an exhaustive search - see the docstring for why, and for why stopping
    # early can only err towards a less sensitive detector.
    best: tuple[float, int, float] | None = None
    reached_bottom = False
    if not budget_applicable:
        q995 = float(np.quantile(finite, 0.995))
        n_eps, _ = _far_at(q995, validation, k_consecutive, merge_gap_s, n_days, max_step_s)
        thresholds: dict[str, Threshold] = {
            "budget": Threshold(
                name="budget",
                value=q995,
                method="score_quantile_0.995 (false-alarm budget not applicable: fabricated timeline)",
                val_episodes=n_eps,
                val_train_days=float("nan"),
                val_far_per_train_day=float("nan"),
                meta={
                    "far_budget_episodes_per_train_day": float(far_budget_episodes_per_train_day),
                    "budget_episodes_allowed": float("nan"),
                    "budget_binding": False,
                    "budget_applicable": False,
                    "silent": bool(n_eps == 0),
                    "n_candidates": 0,
                    "max_step_s": float(max_step_s),
                },
            )
        }
        return _finish(thresholds, s, validation, k_consecutive, merge_gap_s, n_days, max_step_s, quantiles, window_seconds)
    for j, value in enumerate(grid[::-1]):
        n_eps, far = _far_at(float(value), validation, k_consecutive, merge_gap_s, n_days, max_step_s)
        if n_eps <= budget_eps:
            best = (float(value), n_eps, far)
            reached_bottom = j == grid.size - 1
        else:
            break
    if best is None:  # pragma: no cover - grid[-1] == max never fires (strict >)
        value = float(np.nextafter(hi, np.inf))
        n_eps, far = _far_at(value, validation, k_consecutive, merge_gap_s, n_days, max_step_s)
        best = (value, n_eps, far)
    silent = bool(best[1] == 0)

    thresholds: dict[str, Threshold] = {
        "budget": Threshold(
            name="budget",
            value=best[0],
            method="far_budget_episodes",
            val_episodes=best[1],
            val_train_days=n_days,
            val_far_per_train_day=best[2],
            meta={
                "far_budget_episodes_per_train_day": float(far_budget_episodes_per_train_day),
                "budget_episodes_allowed": float(budget_eps),
                "budget_binding": bool(not reached_bottom),
                "budget_applicable": True,
                "silent": silent,
                "n_candidates": int(grid.size),
                "max_step_s": float(max_step_s),
            },
        )
    }
    return _finish(thresholds, s, validation, k_consecutive, merge_gap_s, n_days, max_step_s, quantiles, window_seconds)


def _finish(
    thresholds: dict[str, Threshold],
    s: np.ndarray,
    validation: ValidationScores,
    k_consecutive: int,
    merge_gap_s: float,
    n_days: float,
    max_step_s: float,
    quantiles: Sequence[float],
    window_seconds: float,
) -> ThresholdSet:
    """Add the quantile thresholds beside the budget one and assemble the set."""
    for name, value in quantile_thresholds(s, quantiles).items():
        n_eps, far = _far_at(float(value), validation, k_consecutive, merge_gap_s, n_days, max_step_s)
        thresholds[name] = Threshold(
            name=name,
            value=float(value),
            method="score_quantile",
            val_episodes=n_eps,
            val_train_days=n_days,
            val_far_per_train_day=far,
            meta={"quantile": name},
        )
    return ThresholdSet(
        thresholds=thresholds,
        calibrated_on_index=validation.index.copy(),
        n_val_rows=len(validation),
        window_seconds=float(window_seconds),
        k_consecutive=int(k_consecutive),
        merge_gap_s=float(merge_gap_s),
        max_step_s=float(max_step_s),
    )
