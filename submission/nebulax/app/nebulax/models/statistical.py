"""Honest statistical baselines - the ``tier: statistical`` rows of ``configs/model_ladder.yaml``.

Every model here is an :class:`~nebulax.bench.base.AnomalyDetector` (``task="ad"``, the ladder's
default for rows that omit ``task:``), unsupervised or normal-only, and never reads labels.
They exist so every fancier model in the ladder has something honest to beat - a MetroPT-2
F1 near 1.0 is not a win, because ``single_feature_threshold`` alone already gets there
(ladder note on R93). Registered names/``input_kind``/``family`` match
``configs/model_ladder.yaml`` exactly:

======================== ============== ============= ==================================
name                     input_kind     family        source / reference
======================== ============== ============= ==================================
sensor_range_baseline    window_stats   trivial       QuoVadisTAD-style range baseline (R5)
trivial_baseline_set     window_stats   trivial       QuoVadisTAD-style trivia set (R5, R2)
l2_norm_channels         raw_window     trivial       QuoVadisTAD-style L2 norm (R5)
nn1_distance             raw_window     trivial       QuoVadisTAD-style 1-NN distance (R5)
random_score             window_stats   control       chance floor (R2, R10)
single_feature_threshold window_stats   control       one raw feature as score (R93)
per_channel_or_aggregate window_stats   control       N univariate detectors, OR'd (R21)
moving_window_variance   raw_window     one_liner     within-window variance (R35)
squared_difference       raw_window     one_liner     within-window roughness energy (R35)
robust_z_ewma_cycle      cycle_features threshold     robust z, EWMA-smoothed (R64)
robust_z_ewma_window     window_stats   threshold     robust z, EWMA-smoothed (R1)
cusum_cycle_scalar       cycle_features sequential_test two-sided CUSUM (R72)
pca_spe_t2               window_stats   subspace      PCA SPE + Hotelling T^2 (R1)
mahalanobis_mincovdet    window_stats   subspace      robust Mahalanobis / MinCovDet (R1)
======================== ============== ============= ==================================

Shared conventions
-------------------
* Every hyper-parameter is a keyword-only constructor argument with a default, recorded in
  the class docstring and passed through to ``super().__init__(...)`` so it lands in
  :attr:`~nebulax.bench.base.BaseModel.params` for the results row / config hash.
* ``eps`` guards a denominator; it never changes which row is most anomalous, only numerical
  stability on a (near-)constant feature.
* Models never see column names unless they declare the ``feature_names`` keyword on ``_fit``
  (the runner's row-context channel, ``base.CONTEXT_KEYS``); ``single_feature_threshold`` and
  ``per_channel_or_aggregate`` select a feature/features by *name* when either the caller
  passes ``feature_names`` (the column order of ``X``) as a constructor kwarg, or the runner
  passes it into ``_fit`` - the explicit constructor kwarg wins when both are given. The
  resolved index (or indices) is stored on the fitted model as ``feature_index_``. Otherwise
  ``feature``/``features`` is an int index (or list of them).
* ``robust_z_ewma_*`` and ``cusum_cycle_scalar`` use ``t`` (the per-row end-timestamp, when
  given) to process rows in time order and write scores back at their original row position;
  with ``t=None`` they trust row order. They set ``SEQUENTIAL = True`` so the runner scores
  one series (component) at a time in ascending ``t``, which is what keeps their running
  statistic from smearing across interleaved units/components; with a single series already
  isolated per call, ``t=None`` still trusts row order within that call.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from nebulax.bench.base import AnomalyDetector
from nebulax.bench.registry import register

__all__ = [
    "SensorRangeBaseline",
    "TrivialBaselineSet",
    "L2NormChannels",
    "Nn1Distance",
    "RandomScore",
    "SingleFeatureThreshold",
    "PerChannelOrAggregate",
    "MovingWindowVariance",
    "SquaredDifference",
    "RobustZEwmaCycle",
    "RobustZEwmaWindow",
    "CusumCycleScalar",
    "PcaSpeT2",
    "MahalanobisMinCovDet",
]

_AggMaxMean = Literal["max", "mean"]


def _check_agg(name: str, agg: str) -> None:
    if agg not in ("max", "mean"):
        raise ValueError(f"{name}: agg must be 'max' or 'mean', got {agg!r}")


def _resolve_feature_index(
    name: str, feature: int | str, feature_names: Sequence[str] | None, n_features: int
) -> int:
    """One column index: ``feature`` verbatim if int, else looked up in ``feature_names``."""
    if isinstance(feature, str):
        if feature_names is None:
            raise ValueError(
                f"{name}: feature={feature!r} is a name but feature_names was not provided "
                f"(models never see column names - pass feature_names=<BenchData.feature_names> "
                f"as a constructor kwarg)"
            )
        try:
            return list(feature_names).index(feature)
        except ValueError:
            raise ValueError(f"{name}: feature {feature!r} not found in feature_names") from None
    idx = int(feature)
    if not (0 <= idx < n_features):
        raise ValueError(f"{name}: feature index {idx} out of range for {n_features} features")
    return idx


def _time_order(t: np.ndarray | None, n: int) -> np.ndarray:
    """Row order to process sequentially in: ``argsort(t)`` when given, else identity."""
    if t is None:
        return np.arange(n)
    return np.argsort(np.asarray(t).reshape(-1), kind="stable")


# ----------------------------------------------------------------------------------------
# trivial (QuoVadisTAD-style: "simple baselines beat most deep TSAD models")
# ----------------------------------------------------------------------------------------


@register("sensor_range_baseline", input_kind="window_stats", family="trivial", tier="statistical", reference_id=["R5"])
class SensorRangeBaseline(AnomalyDetector):
    """Trivial per-feature range baseline (QuoVadisTAD-style "Sensor Range").

    Learns the ``[min, max]`` of every input feature on the training slice; scores a row by
    how far its worst-offending feature falls outside that learned range, in units of the
    feature's training inter-quartile range (a scale-free "outside-ness"). Higher = more
    anomalous.

    Hyper-parameters
    -----------------
    eps : float, default 1e-9
        IQR denominator guard for a near-constant feature.
    """

    def __init__(self, *, eps: float = 1e-9) -> None:
        super().__init__(eps=eps)
        self.eps = eps

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kw: Any) -> None:
        self.lo_ = X.min(axis=0)
        self.hi_ = X.max(axis=0)
        q75, q25 = np.percentile(X, [75.0, 25.0], axis=0)
        self.scale_ = (q75 - q25) + self.eps

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        below = np.maximum(self.lo_ - X, 0.0)
        above = np.maximum(X - self.hi_, 0.0)
        excess = np.maximum(below, above) / self.scale_
        return excess.max(axis=1)


_TRIVIAL_SET_COMPONENTS = ("range_frac", "l2_z", "max_z", "constant")


@register(
    "trivial_baseline_set",
    input_kind="window_stats",
    family="trivial",
    tier="statistical",
    reference_id=["R5", "R2"],
)
class TrivialBaselineSet(AnomalyDetector):
    """Small ensemble of independently-trivial score rules, combined by ``agg``
    (QuoVadisTAD-style sanity floor: any model worth reporting must beat this).

    Components (each already a valid, if naive, anomaly score on its own), all fit on the
    training slice only:

    * ``"range_frac"`` - fraction of features outside their training ``[min, max]``.
    * ``"l2_z"`` - L2 norm of the per-feature training z-score (mean/std).
    * ``"max_z"`` - max ``|per-feature training z-score|`` (single worst feature).
    * ``"constant"`` - always ``0.0``, the "do nothing" floor (AUROC/AUPRC == chance).

    Hyper-parameters
    -----------------
    components : tuple[str, ...], default ("range_frac", "l2_z", "max_z")
        Subset of the components above to combine.
    agg : {"max", "mean"}, default "max"
        How the selected components are combined into one score per row.
    eps : float, default 1e-9
    """

    def __init__(
        self,
        *,
        components: Sequence[str] = ("range_frac", "l2_z", "max_z"),
        agg: _AggMaxMean = "max",
        eps: float = 1e-9,
    ) -> None:
        canonical = tuple(components)
        if not canonical:
            raise ValueError("trivial_baseline_set: components must name at least one component")
        bad = sorted(set(canonical) - set(_TRIVIAL_SET_COMPONENTS))
        if bad:
            raise ValueError(
                f"trivial_baseline_set: unknown components {bad}; expected a subset of {_TRIVIAL_SET_COMPONENTS}"
            )
        _check_agg("trivial_baseline_set", agg)
        super().__init__(components=components, agg=agg, eps=eps)  # params keep the caller's value verbatim
        self.components = canonical
        self.agg = agg
        self.eps = eps

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kw: Any) -> None:
        self.lo_ = X.min(axis=0)
        self.hi_ = X.max(axis=0)
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0) + self.eps

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        parts: list[np.ndarray] = []
        if "constant" in self.components:
            parts.append(np.zeros(len(X), dtype=np.float64))
        if "range_frac" in self.components:
            outside = (X < self.lo_) | (X > self.hi_)
            parts.append(outside.mean(axis=1))
        if "l2_z" in self.components or "max_z" in self.components:
            z = (X - self.mu_) / self.sd_
            if "l2_z" in self.components:
                parts.append(np.linalg.norm(z, axis=1))
            if "max_z" in self.components:
                parts.append(np.abs(z).max(axis=1))
        stacked = np.stack(parts, axis=1)
        return stacked.max(axis=1) if self.agg == "max" else stacked.mean(axis=1)


@register("l2_norm_channels", input_kind="raw_window", family="trivial", tier="statistical", reference_id=["R5"])
class L2NormChannels(AnomalyDetector):
    """Trivial raw-window baseline (QuoVadisTAD-style "L2 Norm"): the per-timestep L2 norm
    across channels of the (train-standardised) window, aggregated over the window's time
    axis.

    Fits per-channel mean/std on the training slice (flattened over windows and time) so a
    channel with a large constant offset does not dominate the norm.

    Hyper-parameters
    -----------------
    agg : {"max", "mean"}, default "max"
        How the per-timestep norm is aggregated over the window's time axis.
    eps : float, default 1e-9
    """

    def __init__(self, *, agg: _AggMaxMean = "max", eps: float = 1e-9) -> None:
        _check_agg("l2_norm_channels", agg)
        super().__init__(agg=agg, eps=eps)
        self.agg = agg
        self.eps = eps

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kw: Any) -> None:
        flat = X.reshape(-1, X.shape[-1])
        self.mu_ = flat.mean(axis=0)
        self.sd_ = flat.std(axis=0) + self.eps

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        z = (X - self.mu_) / self.sd_
        norm_t = np.linalg.norm(z, axis=2)
        return norm_t.max(axis=1) if self.agg == "max" else norm_t.mean(axis=1)


@register("nn1_distance", input_kind="raw_window", family="trivial", tier="statistical", reference_id=["R5"])
class Nn1Distance(AnomalyDetector):
    """Trivial 1-nearest-neighbour distance to a reference subsample of training windows
    (QuoVadisTAD-style "1NN"), flattened across time x channel, the classic non-parametric
    outlier-distance baseline.

    Hyper-parameters
    -----------------
    max_reference : int, default 500
        Training windows are subsampled (without replacement, seeded) to at most this many
        before building the reference set, so scoring stays ``O(n_score * max_reference)``.
    metric : str, default "euclidean"
        Passed to ``sklearn.neighbors.NearestNeighbors``.
    seed : int, default 0
        Subsample seed.
    """

    def __init__(self, *, max_reference: int = 500, metric: str = "euclidean", seed: int = 0) -> None:
        super().__init__(max_reference=max_reference, metric=metric, seed=seed)
        self.max_reference = max_reference
        self.metric = metric
        self.seed = seed

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kw: Any) -> None:
        from sklearn.neighbors import NearestNeighbors

        flat = X.reshape(len(X), -1)
        if len(flat) > self.max_reference:
            idx = np.random.default_rng(self.seed).choice(len(flat), self.max_reference, replace=False)
            flat = flat[idx]
        self._nn = NearestNeighbors(n_neighbors=1, metric=self.metric).fit(flat)

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        flat = X.reshape(len(X), -1)
        dist, _ = self._nn.kneighbors(flat, n_neighbors=1)
        return dist[:, 0]


# ----------------------------------------------------------------------------------------
# control (the chance / naive floors other families must beat)
# ----------------------------------------------------------------------------------------


@register("random_score", input_kind="window_stats", family="control", tier="statistical", reference_id=["R2", "R10"])
class RandomScore(AnomalyDetector):
    """Control baseline: uniform random scores, independent of the input - the chance floor
    every real model must beat on every threshold-free metric (AUROC/AUPRC/VUS-PR ~ 0.5 /
    base rate).

    Hyper-parameters
    -----------------
    seed : int, default 0
        RNG seed; the RNG is (re)seeded at :meth:`fit`, so repeated :meth:`score` calls after
        one ``fit`` draw a fresh i.i.d. sequence each time (this model ignores ``X`` entirely).
    """

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.seed = seed

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kw: Any) -> None:
        self._rng = np.random.default_rng(self.seed)

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        return self._rng.uniform(size=len(X))


@register(
    "single_feature_threshold",
    input_kind="window_stats",
    family="control",
    tier="statistical",
    reference_id=["R93"],
    note="Flowmeter_max > 16.05 (air leak) / > 16.18 (oil leak) reaches F1=1.0 on MetroPT-2; every model must beat this",
)
class SingleFeatureThreshold(AnomalyDetector):
    """Control baseline: the (signed) raw value of one input feature, used directly as the
    anomaly score - the honest single-sensor floor (MetroPT-2's ``Flowmeter_max`` alone
    reaches F1=1.0 on the leak events, ladder note R93; thresholding itself happens
    downstream in :mod:`nebulax.bench.thresholds`, calibrated on validation only).

    Hyper-parameters
    -----------------
    feature : int | str, default 0
        Column index into ``X``, or a column name (requires ``feature_names``, from either the
        constructor kwarg below or the runner's row-context channel - see :attr:`feature_index_`).
    feature_names : list[str] | None, default None
        ``X``'s column names in order; only needed when ``feature`` is a string. Takes
        precedence over ``feature_names`` the runner passes into :meth:`_fit` when both are
        given.
    direction : {"above", "below"}, default "above"
        ``"above"``: a higher raw value is more anomalous (score = value). ``"below"``: a
        lower raw value is more anomalous (score = -value).

    Attributes (set by :meth:`_fit`)
    ---------------------------------
    feature_index_ : int
        The resolved column index.
    """

    def __init__(
        self,
        *,
        feature: int | str = 0,
        feature_names: Sequence[str] | None = None,
        direction: Literal["above", "below"] = "above",
    ) -> None:
        if direction not in ("above", "below"):
            raise ValueError(f"single_feature_threshold: direction must be 'above' or 'below', got {direction!r}")
        super().__init__(feature=feature, feature_names=feature_names, direction=direction)
        self.feature = feature
        self.feature_names = feature_names
        self.direction = direction

    def _fit(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        *,
        feature_names: Sequence[str] | None = None,
        **kw: Any,
    ) -> None:
        # explicit constructor kwarg wins; otherwise fall back to the runner's row context.
        names = self.feature_names if self.feature_names is not None else feature_names
        self.feature_index_ = _resolve_feature_index("single_feature_threshold", self.feature, names, X.shape[1])

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        v = X[:, self.feature_index_].astype(np.float64)
        return v if self.direction == "above" else -v


@register(
    "per_channel_or_aggregate",
    input_kind="window_stats",
    family="control",
    tier="statistical",
    reference_id=["R21"],
    note="N univariate detectors aggregated by max/OR; justifies any multivariate model",
)
class PerChannelOrAggregate(AnomalyDetector):
    """Control baseline: N independent univariate robust-z detectors (one per selected
    feature), aggregated by max ("OR" across channels) or mean - the honest multivariate
    floor any true multivariate model (PCA/Mahalanobis/autoencoder/...) must beat.

    Each selected feature gets its own robust z-score, ``|x - median| / (1.4826 * MAD)``,
    learned on the training slice; the per-row score is the ``agg`` over selected features of
    that per-feature robust z.

    Hyper-parameters
    -----------------
    features : list[int | str] | None, default None
        Which columns to use - indices, or names (requires ``feature_names``, from either the
        constructor kwarg below or the runner's row-context channel). ``None`` uses every
        column of ``X``.
    feature_names : list[str] | None, default None
        ``X``'s column names in order; only needed when ``features`` contains strings. Takes
        precedence over ``feature_names`` the runner passes into :meth:`_fit` when both are
        given.
    agg : {"max", "mean"}, default "max"
        ``"max"`` is the literal "OR" of per-channel alarms; ``"mean"`` is a softer vote.
    eps : float, default 1e-9

    Attributes (set by :meth:`_fit`)
    ---------------------------------
    idx_ : np.ndarray
        The resolved column indices (also aliased as :attr:`feature_index_`).
    """

    def __init__(
        self,
        *,
        features: Sequence[int | str] | None = None,
        feature_names: Sequence[str] | None = None,
        agg: _AggMaxMean = "max",
        eps: float = 1e-9,
    ) -> None:
        _check_agg("per_channel_or_aggregate", agg)
        super().__init__(features=features, feature_names=feature_names, agg=agg, eps=eps)
        self.features = features
        self.feature_names = feature_names
        self.agg = agg
        self.eps = eps

    def _fit(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        *,
        feature_names: Sequence[str] | None = None,
        **kw: Any,
    ) -> None:
        # explicit constructor kwarg wins; otherwise fall back to the runner's row context.
        names = self.feature_names if self.feature_names is not None else feature_names
        n_features = X.shape[1]
        if self.features is None:
            self.idx_ = np.arange(n_features, dtype=np.int64)
        else:
            self.idx_ = np.array(
                [_resolve_feature_index("per_channel_or_aggregate", f, names, n_features) for f in self.features],
                dtype=np.int64,
            )
        self.feature_index_ = self.idx_  # alias, see class docstring
        sub = X[:, self.idx_]
        self.median_ = np.median(sub, axis=0)
        mad = np.median(np.abs(sub - self.median_), axis=0)
        self.scale_ = 1.4826 * mad + self.eps

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        sub = X[:, self.idx_]
        z = np.abs(sub - self.median_) / self.scale_
        return z.max(axis=1) if self.agg == "max" else z.mean(axis=1)


# ----------------------------------------------------------------------------------------
# one_liner (raw-window, no learned reference beyond a per-channel scale)
# ----------------------------------------------------------------------------------------


@register("moving_window_variance", input_kind="raw_window", family="one_liner", tier="statistical", reference_id=["R35"])
class MovingWindowVariance(AnomalyDetector):
    """One-liner baseline: the variance of the raw window itself, over time, aggregated
    across channels - a high-variance window (chattering current, oscillating pressure)
    scores as anomalous without reference to any other window.

    Fits only a per-channel scale (median training variance + eps) so channels of very
    different native scale are comparable; the window's own data still drives the score.

    Hyper-parameters
    -----------------
    agg : {"max", "mean"}, default "max"
        How the per-channel variance is aggregated across channels.
    eps : float, default 1e-9
    """

    def __init__(self, *, agg: _AggMaxMean = "max", eps: float = 1e-9) -> None:
        _check_agg("moving_window_variance", agg)
        super().__init__(agg=agg, eps=eps)
        self.agg = agg
        self.eps = eps

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kw: Any) -> None:
        var_tc = X.var(axis=1)
        self.scale_ = np.median(var_tc, axis=0) + self.eps

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        norm = X.var(axis=1) / self.scale_
        return norm.max(axis=1) if self.agg == "max" else norm.mean(axis=1)


@register("squared_difference", input_kind="raw_window", family="one_liner", tier="statistical", reference_id=["R35"])
class SquaredDifference(AnomalyDetector):
    """One-liner baseline: mean squared first difference within the window ("jerk" /
    roughness energy), aggregated across channels - flags choppy/noisy windows without
    reference to any other window.

    Hyper-parameters
    -----------------
    agg : {"max", "mean"}, default "max"
        How the per-channel mean-squared-difference is aggregated across channels.
    eps : float, default 1e-9
    """

    def __init__(self, *, agg: _AggMaxMean = "max", eps: float = 1e-9) -> None:
        _check_agg("squared_difference", agg)
        super().__init__(agg=agg, eps=eps)
        self.agg = agg
        self.eps = eps

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kw: Any) -> None:
        if X.shape[1] < 2:
            raise ValueError(f"squared_difference: window length must be >= 2 to difference, got {X.shape[1]}")
        msd = self._msd(X)
        self.scale_ = np.median(msd, axis=0) + self.eps

    @staticmethod
    def _msd(X: np.ndarray) -> np.ndarray:
        d = np.diff(X, axis=1)
        return np.mean(d * d, axis=1)

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        norm = self._msd(X) / self.scale_
        return norm.max(axis=1) if self.agg == "max" else norm.mean(axis=1)


# ----------------------------------------------------------------------------------------
# threshold (robust z-score, EWMA-smoothed)
# ----------------------------------------------------------------------------------------


class _RobustZEwmaBase(AnomalyDetector):
    """Shared implementation for the two ``robust_z_ewma_*`` rows (not itself registered).

    ``score = max_f |EWMA_alpha(robust_z(x))_f|`` where ``robust_z = (x - median) / (1.4826 *
    MAD)`` is fit on the training slice and the EWMA (``pandas`` ``ewm(adjust=False)``
    convention) runs in time order (sorted by ``t`` when given, else row order).

    Hyper-parameters
    -----------------
    alpha : float, default 0.3
        EWMA smoothing factor in ``(0, 1]``; higher = less smoothing / faster reaction.
    eps : float, default 1e-9
        MAD-scale denominator guard.
    """

    #: The EWMA is a running statistic in time order; the runner scores one series at a time
    #: (see base.AnomalyDetector's class docstring) so it never smears across components.
    SEQUENTIAL = True

    def __init__(self, *, alpha: float = 0.3, eps: float = 1e-9) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"{self.name}: alpha must be in (0, 1], got {alpha}")
        super().__init__(alpha=alpha, eps=eps)
        self.alpha = alpha
        self.eps = eps

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kw: Any) -> None:
        self.median_ = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.median_), axis=0)
        self.scale_ = 1.4826 * mad + self.eps

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        z = np.abs(X - self.median_) / self.scale_
        order = _time_order(t, len(z))
        smoothed_sorted = pd.DataFrame(z[order]).ewm(alpha=self.alpha, adjust=False).mean().to_numpy()
        smoothed = np.empty_like(smoothed_sorted)
        smoothed[order] = smoothed_sorted
        return smoothed.max(axis=1)


@register("robust_z_ewma_cycle", input_kind="cycle_features", family="threshold", tier="statistical", reference_id=["R64"])
class RobustZEwmaCycle(_RobustZEwmaBase):
    """:class:`_RobustZEwmaBase` on per-cycle features (door/compressor cycles).

    Hyper-parameters
    -----------------
    alpha : float, default 0.3
    eps : float, default 1e-9
    """


@register("robust_z_ewma_window", input_kind="window_stats", family="threshold", tier="statistical", reference_id=["R1"])
class RobustZEwmaWindow(_RobustZEwmaBase):
    """:class:`_RobustZEwmaBase` on per-window summary-statistic features.

    Hyper-parameters
    -----------------
    alpha : float, default 0.3
    eps : float, default 1e-9
    """


# ----------------------------------------------------------------------------------------
# sequential_test
# ----------------------------------------------------------------------------------------


@register("cusum_cycle_scalar", input_kind="cycle_features", family="sequential_test", tier="statistical", reference_id=["R72"])
class CusumCycleScalar(AnomalyDetector):
    """Two-sided CUSUM (Page 1954) sequential-test baseline on a scalar reduced from the
    per-cycle feature vector.

    The scalar is the ``agg`` of the training-fit per-feature z-score (how far, on average or
    at worst, the cycle's features sit from their training mean in standard-deviation units).
    The classic two-sided CUSUM statistics then run on that scalar in time order (sorted by
    ``t`` when given, else row order)::

        S+_i = max(0, S+_{i-1} + z_i - k)
        S-_i = max(0, S-_{i-1} - z_i - k)
        score_i = max(S+_i, S-_i)

    Hyper-parameters
    -----------------
    k : float, default 0.5
        CUSUM slack, in standard-deviation units of the reduced scalar; larger ``k`` requires
        a bigger sustained shift before the statistic accumulates.
    agg : {"mean", "max"}, default "mean"
        How the per-feature z-scores are reduced to the scalar CUSUM input.
    eps : float, default 1e-9
    """

    #: CUSUM accumulates state across rows in time order; the runner scores one series at a
    #: time (see base.AnomalyDetector's class docstring) so it never smears across components.
    SEQUENTIAL = True

    def __init__(self, *, k: float = 0.5, agg: Literal["mean", "max"] = "mean", eps: float = 1e-9) -> None:
        if agg not in ("mean", "max"):
            raise ValueError(f"cusum_cycle_scalar: agg must be 'mean' or 'max', got {agg!r}")
        super().__init__(k=k, agg=agg, eps=eps)
        self.k = k
        self.agg = agg
        self.eps = eps

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kw: Any) -> None:
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0) + self.eps

    def _scalar(self, X: np.ndarray) -> np.ndarray:
        z = (X - self.mu_) / self.sd_
        return z.max(axis=1) if self.agg == "max" else z.mean(axis=1)

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        s = self._scalar(X)
        order = _time_order(t, len(s))
        s_sorted = s[order]
        pos = np.empty(s_sorted.shape, dtype=np.float64)
        neg = np.empty(s_sorted.shape, dtype=np.float64)
        sp = sn = 0.0
        for i in range(s_sorted.shape[0]):
            v = float(s_sorted[i])
            sp = max(0.0, sp + v - self.k)
            sn = max(0.0, sn - v - self.k)
            pos[i] = sp
            neg[i] = sn
        out_sorted = np.maximum(pos, neg)
        out = np.empty_like(out_sorted)
        out[order] = out_sorted
        return out


# ----------------------------------------------------------------------------------------
# subspace
# ----------------------------------------------------------------------------------------


@register("pca_spe_t2", input_kind="window_stats", family="subspace", tier="statistical", reference_id=["R1"])
class PcaSpeT2(AnomalyDetector):
    """PCA subspace baseline: combines SPE (squared prediction/reconstruction error in the
    residual subspace) with Hotelling's T^2 (Mahalanobis distance within the retained
    principal-component subspace); ``sklearn.decomposition.PCA`` fit on the (standardised)
    training slice only.

    Hyper-parameters
    -----------------
    n_components : int | float, default 0.9
        Passed to ``sklearn.decomposition.PCA`` - an int number of components (clipped to
        ``min(n_train, n_features)``), or a float in ``(0, 1)`` for the variance ratio kept.
    combine : {"sum", "max"}, default "sum"
        How the SPE and T^2 terms (each independently scaled by its training 99th
        percentile) are combined into one score.
    eps : float, default 1e-9
    """

    def __init__(self, *, n_components: int | float = 0.9, combine: Literal["sum", "max"] = "sum", eps: float = 1e-9) -> None:
        if combine not in ("sum", "max"):
            raise ValueError(f"pca_spe_t2: combine must be 'sum' or 'max', got {combine!r}")
        super().__init__(n_components=n_components, combine=combine, eps=eps)
        self.n_components = n_components
        self.combine = combine
        self.eps = eps

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kw: Any) -> None:
        from sklearn.decomposition import PCA

        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0) + self.eps
        Xs = (X - self.mu_) / self.sd_

        n_comp = self.n_components
        if isinstance(n_comp, (int, np.integer)):
            n_comp = int(min(n_comp, min(Xs.shape)))
        self._pca = PCA(n_components=n_comp, svd_solver="full", random_state=0).fit(Xs)
        self.eigvals_ = self._pca.explained_variance_ + self.eps

        spe_train, t2_train = self._raw_scores(Xs)
        self.spe_scale_ = float(np.quantile(spe_train, 0.99)) + self.eps
        self.t2_scale_ = float(np.quantile(t2_train, 0.99)) + self.eps

    def _raw_scores(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = self._pca.transform(Xs)
        recon = self._pca.inverse_transform(scores)
        spe = np.sum((Xs - recon) ** 2, axis=1)
        t2 = np.sum((scores**2) / self.eigvals_, axis=1)
        return spe, t2

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        Xs = (X - self.mu_) / self.sd_
        spe, t2 = self._raw_scores(Xs)
        spe_n = spe / self.spe_scale_
        t2_n = t2 / self.t2_scale_
        return spe_n + t2_n if self.combine == "sum" else np.maximum(spe_n, t2_n)


@register("mahalanobis_mincovdet", input_kind="window_stats", family="subspace", tier="statistical", reference_id=["R1"])
class MahalanobisMinCovDet(AnomalyDetector):
    """Robust Mahalanobis-distance baseline: fits ``sklearn.covariance.MinCovDet`` (Fast-MCD
    robust covariance) on the training slice, then scores by the robust Mahalanobis distance
    to that fitted robust center/covariance - resistant to the handful of already-anomalous
    training windows a plain (non-robust) covariance estimate would be dragged towards.

    Hyper-parameters
    -----------------
    support_fraction : float | None, default None
        Passed to ``MinCovDet`` (``None`` = sklearn's default, ``(n + p + 1) / (2n)``).
    assume_centered : bool, default False
    seed : int, default 0
        ``random_state`` for ``MinCovDet``.
    """

    def __init__(self, *, support_fraction: float | None = None, assume_centered: bool = False, seed: int = 0) -> None:
        super().__init__(support_fraction=support_fraction, assume_centered=assume_centered, seed=seed)
        self.support_fraction = support_fraction
        self.assume_centered = assume_centered
        self.seed = seed

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kw: Any) -> None:
        from sklearn.covariance import MinCovDet

        Xd = np.asarray(X, dtype=np.float64)
        self._mcd = MinCovDet(
            support_fraction=self.support_fraction,
            assume_centered=self.assume_centered,
            random_state=self.seed,
        ).fit(Xd)

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        Xd = np.asarray(X, dtype=np.float64)
        return self._mcd.mahalanobis(Xd)
