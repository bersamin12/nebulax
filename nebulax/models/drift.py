"""River drift detectors as change-point (``task="cpd"``) rows of the model ladder.

=====================================  ===============  ===============  =====
name                                   family           input_kind       task
=====================================  ===============  ===============  =====
``page_hinkley_cycle_scalar``          drift_detector   cycle_features   cpd
``adwin_kswin_residual``               drift_detector   cycle_features   cpd
``page_hinkley_residual``              drift_detector   raw_window       cpd
``adwin_residual``                     drift_detector   raw_window       cpd
``page_hinkley_thermal_residual``      drift_detector   window_stats     cpd
=====================================  ===============  ===============  =====

Every row is the same three-stage pipeline:

1. **a scalar stream** - one number per row, in time order: a named cycle feature, the
   maximum standardised channel residual of a raw window, or ``T_box - T_box_pred`` from
   :class:`nebulax.models.physics.ThermalResidualModel`;
2. **a river detector** (``river.drift.PageHinkley`` / ``ADWIN`` / ``KSWIN``, BSD-3-Clause)
   updated one sample at a time, with its *continuous* internal quantity recorded at every
   step - the Page-Hinkley cumulative statistic ``max(S_inc - min S_inc, max S_dec - S_dec)``
   for the ``cumulative_statistic`` adapter, ``|ADWIN estimation - healthy mean| / sigma``
   (plus ``-log10 p`` from KSWIN) for the ``drift_magnitude`` adapter;
3. **the ladder's score adapter** - :func:`nebulax.bench.metrics.cpd_score`
   (``statistic=``), which forward-fills and min-max normalises the running quantity into
   the ``[0, 1]`` per-window score every AD metric, the episode builder and the threshold
   calibrator consume. The boolean ``drift_detected`` flag alone is not rankable, which is
   exactly why the adapter exists.

Monotone evidence
-----------------
River's detectors **reset** their statistic after each detection, and both the Page-Hinkley
cumulative sum and the ADWIN/KSWIN magnitudes fluctuate *between* detections, so a raw
emission saw-tooths back down after a permanent step: all five rows showed 100-125 negative
post-step increments, the largest a 0.18 drop, which is nonsense for a degradation indicator
and costs VUS-PR directly.

Two things fix it and both are on by default:

``accumulate=True``  adds the detector's own threshold once per detection, so the curve does
                     not fall back to zero when river resets its internal state;
``monotone=True``    emits the **running maximum** of that accumulated evidence
                     (``np.maximum.accumulate`` over the time-ordered stream). Evidence is
                     cumulative by definition - a test that has already seen a 6-sigma
                     excursion has not un-seen it - so the emitted statistic is monotone
                     non-decreasing by construction, and ``metrics.cpd_score`` (a min-max
                     map) preserves that. The detectors themselves are untouched:
                     ``drift_detected`` and ``n_drifts_`` still report the raw behaviour.

The cost is stated plainly: a row that rises on a transient never comes back down within one
scored slice, so a false alarm stays raised until the next slice. That is the right trade for
a *change-point* score (the ladder's ``cumulative_statistic`` / ``drift_magnitude``
adapters), and ``monotone=False`` restores the saw-tooth for an ablation.

Row context
-----------
Every row declares ``feature_names`` / ``series`` / ``unit`` on ``_fit``
(``nebulax.bench.base.CONTEXT_KEYS``), so the named cycle feature and the named raw channels
(``TP2``, ``TP3``, ``Motor_current``) resolve against the table the runner actually built
instead of collapsing to column 0. All five rows set ``SEQUENTIAL = True``: the runner scores
them one ``series`` at a time in ascending ``t``, so two interleaved doors or sixteen axle
boxes never share a running statistic.

No model here reads a label; the healthy baseline comes only from the rows the runner passes
to ``fit``.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Final, Sequence

import numpy as np
from river import drift

from nebulax.bench import metrics as M
from nebulax.bench.base import AnomalyDetector
from nebulax.bench.registry import register
from nebulax.models.physics import (
    ThermalResidualModel,
    _order_by_time,
    resolve_feature_index,
    resolve_feature_indices,
    robust_baseline,
    sanitise,
    times_seconds,
)

__all__ = [
    "PageHinkleyCycleScalar",
    "AdwinKswinResidual",
    "PageHinkleyResidual",
    "AdwinResidual",
    "PageHinkleyThermalResidual",
    "page_hinkley_statistic",
    "adwin_statistic",
    "kswin_statistic",
    "running_max",
]

_EPS: Final[float] = 1e-12


def running_max(x: np.ndarray) -> np.ndarray:
    """The running maximum of accumulated drift evidence - see "Monotone evidence" above.

    ``x`` must already be in **time order**. Evidence a sequential test has accumulated is
    never withdrawn, so the emitted curve is the largest evidence seen so far and is monotone
    non-decreasing by construction.
    """
    return np.maximum.accumulate(np.asarray(x, dtype=np.float64))


# ======================================================================================
# the three river statistics (continuous, non-negative, one value per stream sample)
# ======================================================================================


def page_hinkley_statistic(
    x: np.ndarray, *, accumulate: bool = True, monotone: bool = True, **params: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Stream ``x`` through ``river.drift.PageHinkley``; return ``(statistic, drift_flags)``.

    ``river.drift.PageHinkley`` exposes only the boolean ``drift_detected`` publicly - its
    cumulative test quantity lives on private attributes - so the statistic is computed here
    from **our own running sums**, mirroring the published Page-Hinkley recursion river
    implements (a fading mean ``m_i``, ``S_inc <- alpha*S_inc + (x_i - m_i) - delta``,
    ``S_dec <- alpha*S_dec + (x_i - m_i) + delta``, reset after a detection):

    ``statistic = max(S_inc - min S_inc, max S_dec - S_dec)`` for ``mode="both"``, and the
    matching one-sided quantity otherwise. The detector itself is still river's, and still
    the authority on *when* a drift is declared: the flags come from ``drift_detected`` and
    the sums reset exactly when river resets. There is deliberately **no silent fallback** -
    an earlier version caught ``AttributeError`` on river's private attributes and emitted a
    trivial all-zero statistic, which would have shipped a flat column to the leaderboard
    had river's internals moved.

    ``accumulate`` adds ``threshold`` per detection so the curve does not fall back to zero
    when river resets, and ``monotone`` emits :func:`running_max` of that accumulated
    evidence so a permanent step can never be followed by a score drop. ``x`` must be in
    time order; the flags are always raw.
    """
    det = drift.PageHinkley(**params)
    mode = str(params.get("mode", "both"))
    if mode not in ("up", "down", "both"):
        raise ValueError(f"page_hinkley_statistic: mode must be up|down|both, got {mode!r}")
    thr = float(params.get("threshold", 50.0))
    alpha = float(params.get("alpha", 1.0 - 1e-4))
    delta = float(params.get("delta", 0.005))
    min_instances = int(params.get("min_instances", 30))  # river withholds detection before this
    n = x.size
    out = np.zeros(n, dtype=np.float64)
    flags = np.zeros(n, dtype=bool)
    bonus = 0.0
    # our own copy of river's state, reset in lockstep with it
    count, mean, s_inc, s_dec = 0, 0.0, 0.0, 0.0
    min_inc, max_dec = math.inf, -1.0
    for i in range(n):
        if i and flags[i - 1]:  # river resets at the top of the update after a detection
            count, mean, s_inc, s_dec = 0, 0.0, 0.0, 0.0
            min_inc, max_dec = math.inf, -1.0
        v = float(x[i])
        if not math.isfinite(v):
            v = mean
        count += 1
        mean += (v - mean) / count
        dev = v - mean
        s_inc = alpha * s_inc + dev - delta
        s_dec = alpha * s_dec + dev + delta
        min_inc = min(min_inc, s_inc)
        max_dec = max(max_dec, s_dec)
        inc, dec = s_inc - min_inc, max_dec - s_dec
        stat = inc if mode == "up" else dec if mode == "down" else max(inc, dec)
        if count < min_instances:
            stat = 0.0  # river emits no evidence before min_instances samples; neither do we
        det.update(v)
        flags[i] = bool(det.drift_detected)
        out[i] = bonus + (stat if math.isfinite(stat) else 0.0)
        if flags[i] and accumulate:
            bonus += thr
    return (running_max(out) if monotone else out), flags


def adwin_statistic(
    x: np.ndarray,
    *,
    center: float = 0.0,
    scale: float = 1.0,
    accumulate: bool = True,
    monotone: bool = True,
    **params: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream ``x`` through ``river.drift.ADWIN``; return ``(drift magnitude, drift_flags)``.

    The magnitude is ``|estimation - center| / scale`` - how far ADWIN's adaptive-window
    mean has moved away from the healthy training level, in healthy sigmas - plus one unit
    per detection when ``accumulate``, and run through :func:`running_max` when ``monotone``
    (the window mean itself wanders, which is what made the emitted curve fall after a step).

    The first ``max(grace_period, min_window_length)`` samples are held at zero: ADWIN's
    ``estimation`` over a one-sample window is the sample itself, and a monotone emission
    would otherwise lock onto that warm-up spike for the rest of the stream.
    """
    det = drift.ADWIN(**params)
    n = x.size
    out = np.zeros(n, dtype=np.float64)
    flags = np.zeros(n, dtype=bool)
    sc = max(abs(float(scale)), _EPS)
    warmup = max(int(params.get("grace_period", 10)), int(params.get("min_window_length", 5)), 2)
    bonus = 0.0
    for i in range(n):
        det.update(float(x[i]))
        est = float(det.estimation)
        flags[i] = bool(det.drift_detected)
        mag = abs(est - float(center)) / sc if np.isfinite(est) else 0.0
        out[i] = bonus + (mag if i >= warmup else 0.0)
        if flags[i] and accumulate:
            bonus += 1.0
    return (running_max(out) if monotone else out), flags


def kswin_statistic(
    x: np.ndarray, *, accumulate: bool = True, monotone: bool = True, **params: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Stream ``x`` through ``river.drift.KSWIN``; return ``(-log10 p, drift_flags)``.

    KSWIN is distribution-free on real values (a two-sample KS test between the sliding
    window and a reference sample), so its p-value is already a continuous evidence measure:
    ``-log10(p)`` rises as the recent distribution separates from the reference. A single
    p-value is noisy sample to sample, so ``monotone`` emits :func:`running_max` of it; the
    first ``window_size`` samples are held at zero because the sliding window is not full yet.
    """
    det = drift.KSWIN(**params)
    n = x.size
    out = np.zeros(n, dtype=np.float64)
    flags = np.zeros(n, dtype=bool)
    warmup = max(int(params.get("window_size", 100)), 2)
    bonus = 0.0
    for i in range(n):
        det.update(float(x[i]))
        p = float(det.p_value)
        flags[i] = bool(det.drift_detected)
        ev = -np.log10(max(p, _EPS)) if np.isfinite(p) and p > 0 else 0.0
        out[i] = bonus + (ev if i >= warmup else 0.0)
        if flags[i] and accumulate:
            bonus += 1.0
    return (running_max(out) if monotone else out), flags


# ======================================================================================
# shared base
# ======================================================================================


class _DriftBase(AnomalyDetector):
    """Scalar stream -> river detector -> ``metrics.cpd_score``. Subclasses give the stream.

    ``SEQUENTIAL = True``: every row here carries a running statistic, so the runner scores
    one ``series`` at a time in ascending ``t`` (``nebulax.bench.base``). The row context
    (``feature_names`` / ``series`` / ``unit``) is threaded through ``_prepare`` / ``_stream``
    so the named column or channel resolves against the real table.
    """

    task = "cpd"
    SEQUENTIAL: ClassVar[bool] = True
    #: Emit the running maximum of the accumulated evidence - see "Monotone evidence".
    monotone: bool = True

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.center_: float = 0.0
        self.scale_: float = 1.0
        self.n_drifts_: int = 0

    # -- subclass hooks ------------------------------------------------------------------
    def _prepare(
        self,
        X: np.ndarray,
        t: np.ndarray | None,
        *,
        feature_names: Sequence[str] | None = None,
        series: Any | None = None,
        unit: Any | None = None,
    ) -> None:
        """Fit whatever the scalar stream needs (column choice, residual model, baselines)."""

    def _stream(
        self,
        X: np.ndarray,
        t: np.ndarray | None,
        *,
        series: Any | None = None,
        unit: Any | None = None,
    ) -> np.ndarray:
        """One float per row, in the caller's row order."""
        raise NotImplementedError

    def _statistic(self, x_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the detector over the time-ordered stream."""
        raise NotImplementedError

    # -- model ---------------------------------------------------------------------------
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
        self._prepare(X, t, feature_names=feature_names, series=series, unit=unit)
        self.center_, self.scale_ = robust_baseline(self._stream(X, t, series=series, unit=unit))

    def _score(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        *,
        series: Any | None = None,
        unit: Any | None = None,
    ) -> np.ndarray:
        x = sanitise(self._stream(X, t, series=series, unit=unit), fill=self.center_)
        order, inv = _order_by_time(times_seconds(t, X.shape[0]))
        stat, flags = self._statistic(x[order])
        self.n_drifts_ = int(flags.sum())
        stat = sanitise(stat)
        if self.monotone:  # belt and braces: monotone whatever the subclass statistic did
            stat = running_max(stat)
        return M.cpd_score(x.size, statistic=stat[inv])


class _PageHinkleyMixin:
    """Page-Hinkley hyper-parameters (river defaults) + the ``cumulative_statistic`` adapter."""

    def _ph_params(self) -> dict[str, Any]:
        return {
            "min_instances": self.min_instances,
            "delta": self.delta,
            "threshold": self.threshold,
            "alpha": self.alpha,
            "mode": self.mode,
        }

    def _statistic(self, x_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return page_hinkley_statistic(
            x_sorted, accumulate=self.accumulate, monotone=self.monotone, **self._ph_params()
        )


# ======================================================================================
# door - the cycle scalar
# ======================================================================================


@register("page_hinkley_cycle_scalar", input_kind="cycle_features", family="drift_detector", task="cpd")
class PageHinkleyCycleScalar(_PageHinkleyMixin, _DriftBase):
    """[R178, R180, R193] Page-Hinkley on one per-cycle scalar - the maintained CUSUM.

    The ladder demotes ``cusum_cycle_scalar`` to the baseline this row must match: same
    sequential-deviation idea, a maintained implementation, and an output the evaluation
    protocol can rank (the cumulative statistic, via ``metrics.cpd_score``).

    Defaults: ``feature=None`` (auto: a ``peak_current|stroke|travel|duration``-like column
    when ``feature_names`` is given, else the highest-variance column), ``feature_index=None``,
    ``feature_names=None``, ``standardise=True``, ``min_instances=30``, ``delta=0.005``,
    ``threshold=50.0``, ``alpha=0.9999``, ``mode="both"``, ``accumulate=True``.
    """

    _PATTERNS: Final[tuple[str, ...]] = (
        r"peak_current",
        r"i_?max",
        r"stroke",
        r"travel",
        r"duration",
        r"idle_run_ratio",
        r"duty",
    )

    def __init__(
        self,
        *,
        feature: str | int | None = None,
        feature_index: int | None = None,
        feature_names: Sequence[str] | None = None,
        standardise: bool = True,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
        mode: str = "both",
        accumulate: bool = True,
        monotone: bool = True,
    ) -> None:
        super().__init__(
            feature=feature,
            feature_index=feature_index,
            feature_names=list(feature_names) if feature_names is not None else None,
            standardise=standardise,
            min_instances=min_instances,
            delta=delta,
            threshold=threshold,
            alpha=alpha,
            mode=mode,
            accumulate=accumulate,
            monotone=monotone,
        )
        self.feature = feature
        self.feature_index = feature_index
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.standardise = bool(standardise)
        self.min_instances = int(min_instances)
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.mode = str(mode)
        self.accumulate = bool(accumulate)
        self.monotone = bool(monotone)
        self.feature_index_: int = 0
        self.feature_name_: str = ""
        self.raw_center_: float = 0.0
        self.raw_scale_: float = 1.0

    def _prepare(
        self,
        X: np.ndarray,
        t: np.ndarray | None,
        *,
        feature_names: Sequence[str] | None = None,
        series: Any | None = None,
        unit: Any | None = None,
    ) -> None:
        names = feature_names if feature_names is not None else self.feature_names
        known = names is not None and len(names) == X.shape[1]
        with np.errstate(invalid="ignore"):
            var = np.nanvar(sanitise(X, fill=np.nan), axis=0)
        if self.feature is None and self.feature_index is None and not known:
            # no names and no pin: the documented highest-variance fallback
            self.feature_index_ = int(np.argmax(sanitise(var, 0.0)))
        else:
            self.feature_index_ = resolve_feature_index(
                self.feature,
                index=self.feature_index,
                feature_names=names,
                n_features=X.shape[1],
                patterns=self._PATTERNS,
                what=f"{self.name}.feature",
            )
        self.feature_name_ = str(names[self.feature_index_]) if known else ""
        self.raw_center_, self.raw_scale_ = robust_baseline(X[:, self.feature_index_])

    def _stream(
        self,
        X: np.ndarray,
        t: np.ndarray | None,
        *,
        series: Any | None = None,
        unit: Any | None = None,
    ) -> np.ndarray:
        x = sanitise(X[:, self.feature_index_], fill=self.raw_center_)
        return (x - self.raw_center_) / self.raw_scale_ if self.standardise else x


@register("adwin_kswin_residual", input_kind="cycle_features", family="drift_detector", task="cpd")
class AdwinKswinResidual(_DriftBase):
    """[R178, R179, R181, R193] ADWIN + KSWIN on a per-cycle residual stream.

    The residual is the selected cycle feature standardised against its healthy training
    median/MAD (the "model" here is the healthy level itself), so the stream is zero-mean on
    healthy data and both detectors see a genuine distribution shift when the component
    degrades. ADWIN contributes an adaptive-window **drift magnitude**, KSWIN - which is
    distribution-free on real values and so needs no classifier underneath - contributes
    ``-log10 p``; each is min-max normalised and combined by ``weight``.

    Defaults (the ladder's ADWIN block verbatim): ``delta=0.002``, ``clock=32``,
    ``max_buckets=5``, ``min_window_length=5``, ``grace_period=10``; KSWIN ``alpha=0.005``,
    ``window_size=100``, ``stat_size=30``, ``seed=0``; plus ``feature=None`` (auto, as in
    ``page_hinkley_cycle_scalar``), ``feature_index=None``, ``feature_names=None``,
    ``weight=0.5`` (1.0 = ADWIN only, 0.0 = KSWIN only), ``accumulate=True``.
    """

    _PATTERNS = PageHinkleyCycleScalar._PATTERNS

    def __init__(
        self,
        *,
        feature: str | int | None = None,
        feature_index: int | None = None,
        feature_names: Sequence[str] | None = None,
        delta: float = 0.002,
        clock: int = 32,
        max_buckets: int = 5,
        min_window_length: int = 5,
        grace_period: int = 10,
        kswin_alpha: float = 0.005,
        kswin_window_size: int = 100,
        kswin_stat_size: int = 30,
        seed: int = 0,
        weight: float = 0.5,
        accumulate: bool = True,
        monotone: bool = True,
    ) -> None:
        super().__init__(
            feature=feature,
            feature_index=feature_index,
            feature_names=list(feature_names) if feature_names is not None else None,
            delta=delta,
            clock=clock,
            max_buckets=max_buckets,
            min_window_length=min_window_length,
            grace_period=grace_period,
            kswin_alpha=kswin_alpha,
            kswin_window_size=kswin_window_size,
            kswin_stat_size=kswin_stat_size,
            seed=seed,
            weight=weight,
            accumulate=accumulate,
            monotone=monotone,
        )
        if not 0.0 <= float(weight) <= 1.0:
            raise ValueError(f"adwin_kswin_residual: weight must be in [0, 1], got {weight}")
        self.feature = feature
        self.feature_index = feature_index
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.adwin_params = {
            "delta": float(delta),
            "clock": int(clock),
            "max_buckets": int(max_buckets),
            "min_window_length": int(min_window_length),
            "grace_period": int(grace_period),
        }
        self.kswin_params = {
            "alpha": float(kswin_alpha),
            "window_size": int(kswin_window_size),
            "stat_size": int(kswin_stat_size),
            "seed": int(seed),
        }
        self.weight = float(weight)
        self.accumulate = bool(accumulate)
        self.monotone = bool(monotone)
        self.feature_index_: int = 0
        self.feature_name_: str = ""
        self.raw_center_: float = 0.0
        self.raw_scale_: float = 1.0

    _prepare = PageHinkleyCycleScalar._prepare

    def _stream(
        self,
        X: np.ndarray,
        t: np.ndarray | None,
        *,
        series: Any | None = None,
        unit: Any | None = None,
    ) -> np.ndarray:
        x = sanitise(X[:, self.feature_index_], fill=self.raw_center_)
        return (x - self.raw_center_) / self.raw_scale_

    def _statistic(self, x_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a_stat, a_flags = adwin_statistic(
            x_sorted,
            center=0.0,
            scale=1.0,
            accumulate=self.accumulate,
            monotone=self.monotone,
            **self.adwin_params,
        )
        k_stat, k_flags = kswin_statistic(
            x_sorted, accumulate=self.accumulate, monotone=self.monotone, **self.kswin_params
        )
        a_norm = M.cpd_score(x_sorted.size, statistic=a_stat)
        k_norm = M.cpd_score(x_sorted.size, statistic=k_stat)
        return self.weight * a_norm + (1.0 - self.weight) * k_norm, (a_flags | k_flags)


# ======================================================================================
# pneumatic - the raw-window residual stream
# ======================================================================================


class _RawResidualStream(_DriftBase):
    """Per-window channel means -> robust z against healthy training -> ``max`` over channels.

    ``raw_window`` input is ``(n, L, c)`` and the ladder's ``targets`` (TP2, TP3,
    Motor_current) are **channels addressed by name**: the runner passes the channel names of
    a 3-D ``X`` as ``feature_names`` (``nebulax.bench.base.CONTEXT_KEYS``), so the three
    targets land on the shipped pneumatic raw layout's channels ``0, 1, 6`` and on MetroPT-3's
    ``TP2_mean`` / ``TP3_mean`` / ``Motor_current_mean``. They used to collapse to ``[0]``,
    which silently dropped TP3 and Motor_current from the residual entirely.

    A named target that the table does not carry raises ``ValueError`` naming it
    (``on_missing="drop"`` keeps the ones that did resolve); ``channels=[...]`` pins indices
    outright, and with **no** channel names at all every channel is used, as before.
    """

    _PATTERNS: Final[tuple[str, ...]] = (r"^TP2", r"^TP3", r"motor_current", r"oil_temperature")

    def _init_stream(
        self,
        *,
        targets: Sequence[str | int] | None,
        channels: Sequence[int] | None,
        channel_names: Sequence[str] | None,
        reduce: str,
        combine: str,
        on_missing: str = "error",
    ) -> None:
        if reduce not in ("mean", "std", "max", "min", "ptp"):
            raise ValueError(f"{self.name}: reduce must be mean|std|max|min|ptp, got {reduce!r}")
        if combine not in ("max", "mean", "sum"):
            raise ValueError(f"{self.name}: combine must be max|mean|sum, got {combine!r}")
        self.targets = list(targets) if targets is not None else None
        self.channels = list(channels) if channels is not None else None
        self.channel_names = list(channel_names) if channel_names is not None else None
        self.reduce = reduce
        self.combine = combine
        self.on_missing = on_missing
        self.channels_: list[int] = []
        self.channel_names_: list[str] = []
        self.ch_center_: np.ndarray = np.zeros(0)
        self.ch_scale_: np.ndarray = np.ones(0)

    def _reduced(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"{self.name}: expects raw_window (n, L, c), got {X.shape}")
        sig = sanitise(X[:, :, self.channels_], fill=np.nan)
        with np.errstate(invalid="ignore"):
            if self.reduce == "mean":
                red = np.nanmean(sig, axis=1)
            elif self.reduce == "std":
                red = np.nanstd(sig, axis=1)
            elif self.reduce == "max":
                red = np.nanmax(sig, axis=1)
            elif self.reduce == "min":
                red = np.nanmin(sig, axis=1)
            else:
                red = np.nanmax(sig, axis=1) - np.nanmin(sig, axis=1)
        return sanitise(red)

    def _prepare(
        self,
        X: np.ndarray,
        t: np.ndarray | None,
        *,
        feature_names: Sequence[str] | None = None,
        series: Any | None = None,
        unit: Any | None = None,
    ) -> None:
        if X.ndim != 3:
            raise ValueError(f"{self.name}: expects raw_window (n, L, c), got {X.shape}")
        n_c = X.shape[2]
        names = feature_names if feature_names is not None else self.channel_names
        known = names is not None and len(names) == n_c
        self.channels_ = resolve_feature_indices(
            self.targets if (known or self.channels is not None) else None,
            indices=self.channels,
            feature_names=names,
            n_features=n_c,
            patterns=self._PATTERNS,
            default_k=n_c,
            on_missing=self.on_missing,
            what=f"{self.name}.targets",
        )
        self.channel_names_ = [str(names[j]) for j in self.channels_] if known else []
        red = self._reduced(X)
        stats = [robust_baseline(red[:, j]) for j in range(red.shape[1])]
        self.ch_center_ = np.asarray([m for m, _ in stats])
        self.ch_scale_ = np.asarray([s for _, s in stats])

    def _stream(
        self,
        X: np.ndarray,
        t: np.ndarray | None,
        *,
        series: Any | None = None,
        unit: Any | None = None,
    ) -> np.ndarray:
        z = np.abs(sanitise((self._reduced(X) - self.ch_center_) / self.ch_scale_))
        if self.combine == "max":
            return z.max(axis=1)
        if self.combine == "sum":
            return z.sum(axis=1)
        return z.mean(axis=1)


@register("page_hinkley_residual", input_kind="raw_window", family="drift_detector", task="cpd")
class PageHinkleyResidual(_PageHinkleyMixin, _RawResidualStream):
    """[R178, R180, R193, R36] Page-Hinkley on the pneumatic residual stream.

    The V4 design change: performance-aware drift detection on a residual stream, with a
    maintained detector instead of a hand-rolled CUSUM. The residual is the per-window level
    of the ladder's ``targets`` (TP2, TP3, Motor_current) standardised against the healthy
    training level and combined across channels.

    Defaults: ``targets=("TP2", "TP3", "Motor_current")`` (resolved against the channel
    names the runner passes; all channels when there are none), ``channels=None``,
    ``channel_names=None``, ``reduce="mean"``, ``combine="max"``, ``on_missing="error"``,
    ``min_instances=30``, ``delta=0.005``, ``threshold=50.0``, ``alpha=0.9999``,
    ``mode="both"``, ``accumulate=True``, ``monotone=True``.
    """

    def __init__(
        self,
        *,
        targets: Sequence[str | int] | None = ("TP2", "TP3", "Motor_current"),
        channels: Sequence[int] | None = None,
        channel_names: Sequence[str] | None = None,
        reduce: str = "mean",
        combine: str = "max",
        on_missing: str = "error",
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
        mode: str = "both",
        accumulate: bool = True,
        monotone: bool = True,
    ) -> None:
        super().__init__(
            targets=list(targets) if targets is not None else None,
            channels=list(channels) if channels is not None else None,
            channel_names=list(channel_names) if channel_names is not None else None,
            reduce=reduce,
            combine=combine,
            on_missing=on_missing,
            min_instances=min_instances,
            delta=delta,
            threshold=threshold,
            alpha=alpha,
            mode=mode,
            accumulate=accumulate,
            monotone=monotone,
        )
        self._init_stream(
            targets=targets,
            channels=channels,
            channel_names=channel_names,
            reduce=reduce,
            combine=combine,
            on_missing=on_missing,
        )
        self.min_instances = int(min_instances)
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.mode = str(mode)
        self.accumulate = bool(accumulate)
        self.monotone = bool(monotone)


@register("adwin_residual", input_kind="raw_window", family="drift_detector", task="cpd")
class AdwinResidual(_RawResidualStream):
    """[R178, R179] ADWIN on the same pneumatic residual stream, zero new dependencies.

    ADWIN keeps an adaptive window whose mean tracks the current regime; the emitted
    ``drift_magnitude`` is how far that mean has moved from the healthy training level, in
    healthy sigmas, plus one unit per detection (``accumulate``).

    Defaults: the ladder's ADWIN block - ``delta=0.002``, ``clock=32``, ``max_buckets=5``,
    ``min_window_length=5``, ``grace_period=10`` - plus
    ``targets=("TP2", "TP3", "Motor_current")``, ``channels=None``, ``channel_names=None``,
    ``reduce="mean"``, ``combine="max"``, ``on_missing="error"``, ``accumulate=True``,
    ``monotone=True``.
    """

    def __init__(
        self,
        *,
        targets: Sequence[str | int] | None = ("TP2", "TP3", "Motor_current"),
        channels: Sequence[int] | None = None,
        channel_names: Sequence[str] | None = None,
        reduce: str = "mean",
        combine: str = "max",
        on_missing: str = "error",
        delta: float = 0.002,
        clock: int = 32,
        max_buckets: int = 5,
        min_window_length: int = 5,
        grace_period: int = 10,
        accumulate: bool = True,
        monotone: bool = True,
    ) -> None:
        super().__init__(
            targets=list(targets) if targets is not None else None,
            channels=list(channels) if channels is not None else None,
            channel_names=list(channel_names) if channel_names is not None else None,
            reduce=reduce,
            combine=combine,
            on_missing=on_missing,
            delta=delta,
            clock=clock,
            max_buckets=max_buckets,
            min_window_length=min_window_length,
            grace_period=grace_period,
            accumulate=accumulate,
            monotone=monotone,
        )
        self._init_stream(
            targets=targets,
            channels=channels,
            channel_names=channel_names,
            reduce=reduce,
            combine=combine,
            on_missing=on_missing,
        )
        self.adwin_params = {
            "delta": float(delta),
            "clock": int(clock),
            "max_buckets": int(max_buckets),
            "min_window_length": int(min_window_length),
            "grace_period": int(grace_period),
        }
        self.accumulate = bool(accumulate)
        self.monotone = bool(monotone)

    def _statistic(self, x_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return adwin_statistic(
            x_sorted,
            center=self.center_,
            scale=self.scale_,
            accumulate=self.accumulate,
            monotone=self.monotone,
            **self.adwin_params,
        )


# ======================================================================================
# bearing - the thermal residual
# ======================================================================================


@register("page_hinkley_thermal_residual", input_kind="window_stats", family="drift_detector", task="cpd")
class PageHinkleyThermalResidual(_PageHinkleyMixin, _DriftBase):
    """[R178, R180, R110, R193, R151] Page-Hinkley on ``T_box - T_box_pred``.

    The stream is the standardised residual of :class:`nebulax.models.physics
    .ThermalResidualModel` (ridge on speed / ambient / load / dwell), so the detector sees a
    temperature deviation that the operating point cannot explain. This is the row the
    report benchmarks *directly* against the Netherlands Railways slow-degradation rule -
    itself a sequential deviation test (3.5 sigma sustained over >= 10 measurements in 30
    days, R151), which Page-Hinkley generalises.

    The residual model is the physics row, unchanged: the covariates are the **operating
    context** (``speed`` / ``ambient`` / ``load`` / ``dwell`` columns of the table) and the
    **adjacent / opposite box temperature** taken from the ``series`` / ``unit`` row context,
    with every column derived from the target's own sensor excluded. A table that supplies
    neither raises ``ValueError`` naming them rather than regressing ``T_box_mean`` on
    ``T_box_max``.

    **One honest limitation.** This row is ``SEQUENTIAL``, so the runner scores it one axle
    box at a time: within a scored slice the *peer* boxes are not present, and those rows fall
    back to the fitted median of the peer columns (counted in
    ``residual_model_.n_peerless_rows_``). The peer term is then a constant offset and the
    stream is the box's own temperature against the healthy operating point it was fitted to -
    still the R110 residual, but without the live differential. Fit-time peers *are* present
    (the runner fits on the whole training slice), so the coefficients themselves are peer-
    conditioned. Use ``peer_delta_temperature`` / ``peer_normalisation_shared`` when the live
    differential is what matters.

    Defaults: ``target="T_box"``, ``target_index=None``, ``covariates=None``,
    ``covariate_indices=None``, ``feature_names=None``, ``peer_covariates=True``,
    ``require_physical=True``, ``alpha_ridge=1.0``, ``max_covariates=24``,
    ``min_instances=30``, ``delta=0.005``, ``threshold=50.0``, ``alpha=0.9999``,
    ``mode="up"`` (a bearing fails hot; ``"both"`` also catches a sensor dropping out),
    ``accumulate=True``, ``monotone=True``.
    """

    def __init__(
        self,
        *,
        target: str | int | None = "T_box",
        target_index: int | None = None,
        covariates: Sequence[str | int] | None = None,
        covariate_indices: Sequence[int] | None = None,
        feature_names: Sequence[str] | None = None,
        peer_covariates: bool = True,
        require_physical: bool = True,
        alpha_ridge: float = 1.0,
        max_covariates: int = 24,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
        mode: str = "up",
        accumulate: bool = True,
        monotone: bool = True,
    ) -> None:
        super().__init__(
            target=target,
            target_index=target_index,
            covariates=list(covariates) if covariates is not None else None,
            covariate_indices=list(covariate_indices) if covariate_indices is not None else None,
            feature_names=list(feature_names) if feature_names is not None else None,
            peer_covariates=peer_covariates,
            require_physical=require_physical,
            alpha_ridge=alpha_ridge,
            max_covariates=max_covariates,
            min_instances=min_instances,
            delta=delta,
            threshold=threshold,
            alpha=alpha,
            mode=mode,
            accumulate=accumulate,
            monotone=monotone,
        )
        self.residual_kwargs = {
            "target": target,
            "target_index": target_index,
            "covariates": covariates,
            "covariate_indices": covariate_indices,
            "feature_names": feature_names,
            "peer_covariates": bool(peer_covariates),
            "require_physical": bool(require_physical),
            "alpha": float(alpha_ridge),
            "max_covariates": int(max_covariates),
        }
        self.min_instances = int(min_instances)
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.mode = str(mode)
        self.accumulate = bool(accumulate)
        self.monotone = bool(monotone)
        self.residual_model_: ThermalResidualModel | None = None

    def _prepare(
        self,
        X: np.ndarray,
        t: np.ndarray | None,
        *,
        feature_names: Sequence[str] | None = None,
        series: Any | None = None,
        unit: Any | None = None,
    ) -> None:
        self.residual_model_ = ThermalResidualModel(**self.residual_kwargs)
        self.residual_model_.fit(X, t, feature_names=feature_names, series=series, unit=unit)

    def _stream(
        self,
        X: np.ndarray,
        t: np.ndarray | None,
        *,
        series: Any | None = None,
        unit: Any | None = None,
    ) -> np.ndarray:
        assert self.residual_model_ is not None  # set by _prepare, which _fit calls first
        return self.residual_model_.standardised_residual(
            np.asarray(X, dtype=np.float64), t, series, unit
        )
