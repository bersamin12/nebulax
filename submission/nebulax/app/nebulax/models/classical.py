"""Classical (non-deep) anomaly detectors for the ``configs/model_ladder.yaml`` "classical_ml"
tier: PyOD one-liners/ensembles on ``window_stats``, a k-means discord baseline and a
librosa-feature one-class SVM on ``raw_window``, a river streaming detector, and a
stumpy matrix-profile discord score.

Every class below is a :class:`nebulax.bench.base.AnomalyDetector`: ``_fit(X, t=None)`` then
``_score(X, t=None) -> (n,) float`` with higher = more anomalous (see ``nebulax/bench/base.py``).
No class ever reads labels - all fitting is unsupervised/semi-supervised on ``X`` alone.
Importing this module registers every class with :func:`nebulax.bench.registry.register`;
``nebulax/models/__init__.py`` imports this module so ``import nebulax.models`` registers
everything.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import stumpy
from pyod.models.cblof import CBLOF as _PyODCBLOF
from pyod.models.copod import COPOD as _PyODCOPOD
from pyod.models.ecod import ECOD as _PyODECOD
from pyod.models.iforest import IForest as _PyODIForest
from pyod.models.knn import KNN as _PyODKNN
from pyod.models.ocsvm import OCSVM as _PyODOCSVM
from river import anomaly as _river_anomaly
from scipy.signal import hilbert
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM

from nebulax.bench.base import AnomalyDetector
from nebulax.bench.registry import register

try:
    import librosa
except ImportError as exc:  # pragma: no cover - librosa is a hard project dependency
    raise ImportError("nebulax.models.classical: librosa is required for ocsvm_mfcc_ams") from exc

__all__ = [
    "IsolationForestAD",
    "ECODAD",
    "COPODAD",
    "KNNOutlier",
    "CBLOFAD",
    "KMeansAD",
    "OCSVMAD",
    "OCSVMMfccAms",
    "HalfspaceTreesOCKNN",
    "MatrixProfileDiscord",
]


def _sanitize(X: np.ndarray) -> np.ndarray:
    """Replace non-finite ``window_stats``/``raw_window`` values with 0.0 as ``float64``.

    None of the wrapped estimators below (pyod, sklearn, river) are NaN-safe, and the
    benchmark's feature tables can carry occasional NaN/inf from zero-variance windows;
    this is a defensive numeric guard only, never a use of labels.
    """
    return np.where(np.isfinite(X), X, 0.0).astype(np.float64, copy=False)


# --------------------------------------------------------------------------------------
# PyOD wrappers on window_stats
# --------------------------------------------------------------------------------------


@register(
    "isolation_forest",
    input_kind="window_stats",
    family="ensemble_outlier",
    task="ad",
    tier="classical_ml",
    priority="must",
    source="pyod",
    reference_id=["R1"],
)
class IsolationForestAD(AnomalyDetector):
    """Isolation Forest (pyod.models.iforest.IForest) on window_stats [ladder: isolation_forest].

    Params (defaults): ``n_estimators=100``, ``max_samples="auto"``, ``contamination=0.1``,
    ``max_features=1.0``, ``bootstrap=False``, ``random_state=0``, ``n_jobs=1``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Any = "auto",
        contamination: float = 0.1,
        max_features: float = 1.0,
        bootstrap: bool = False,
        random_state: int = 0,
        n_jobs: int = 1,
        **params: Any,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs,
            **params,
        )

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kwargs: Any) -> None:
        self.model_ = _PyODIForest(
            n_estimators=self.params["n_estimators"],
            max_samples=self.params["max_samples"],
            contamination=self.params["contamination"],
            max_features=self.params["max_features"],
            bootstrap=self.params["bootstrap"],
            random_state=self.params["random_state"],
            n_jobs=self.params["n_jobs"],
        )
        self.model_.fit(_sanitize(X))

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(self.model_.decision_function(_sanitize(X)), dtype=np.float64)


@register(
    "ecod",
    input_kind="window_stats",
    family="ecdf_outlier",
    task="ad",
    tier="classical_ml",
    priority="must",
    source="pyod",
    reference_id=["R1"],
)
class ECODAD(AnomalyDetector):
    """Empirical-CDF outlier detector (pyod.models.ecod.ECOD) [ladder: ecod].

    Params (defaults): ``contamination=0.1``, ``n_jobs=1``.
    """

    def __init__(self, contamination: float = 0.1, n_jobs: int = 1, **params: Any) -> None:
        super().__init__(contamination=contamination, n_jobs=n_jobs, **params)

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kwargs: Any) -> None:
        self.model_ = _PyODECOD(contamination=self.params["contamination"], n_jobs=self.params["n_jobs"])
        self.model_.fit(_sanitize(X))

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(self.model_.decision_function(_sanitize(X)), dtype=np.float64)


@register(
    "copod",
    input_kind="window_stats",
    family="copula_outlier",
    task="ad",
    tier="classical_ml",
    priority="must",
    source="pyod",
    reference_id=["R1"],
)
class COPODAD(AnomalyDetector):
    """Copula-based outlier detector (pyod.models.copod.COPOD) [ladder: copod].

    Params (defaults): ``contamination=0.1``, ``n_jobs=1``.
    """

    def __init__(self, contamination: float = 0.1, n_jobs: int = 1, **params: Any) -> None:
        super().__init__(contamination=contamination, n_jobs=n_jobs, **params)

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kwargs: Any) -> None:
        self.model_ = _PyODCOPOD(contamination=self.params["contamination"], n_jobs=self.params["n_jobs"])
        self.model_.fit(_sanitize(X))

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(self.model_.decision_function(_sanitize(X)), dtype=np.float64)


@register(
    "knn_outlier",
    input_kind="window_stats",
    family="distance_outlier",
    task="ad",
    tier="classical_ml",
    priority="must",
    source="pyod",
    reference_id=["R1"],
)
class KNNOutlier(AnomalyDetector):
    """k-NN distance outlier score (pyod.models.knn.KNN) [ladder: knn_outlier].

    Params (defaults): ``n_neighbors=5``, ``method="largest"``, ``contamination=0.1``,
    ``metric="minkowski"``, ``n_jobs=1``.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        method: str = "largest",
        contamination: float = 0.1,
        metric: str = "minkowski",
        n_jobs: int = 1,
        **params: Any,
    ) -> None:
        super().__init__(
            n_neighbors=n_neighbors,
            method=method,
            contamination=contamination,
            metric=metric,
            n_jobs=n_jobs,
            **params,
        )

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kwargs: Any) -> None:
        self.model_ = _PyODKNN(
            n_neighbors=self.params["n_neighbors"],
            method=self.params["method"],
            contamination=self.params["contamination"],
            metric=self.params["metric"],
            n_jobs=self.params["n_jobs"],
        )
        self.model_.fit(_sanitize(X))

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(self.model_.decision_function(_sanitize(X)), dtype=np.float64)


@register(
    "cblof",
    input_kind="window_stats",
    family="cluster_outlier",
    task="ad",
    tier="classical_ml",
    priority="must",
    source="pyod",
    reference_id=["R1"],
)
class CBLOFAD(AnomalyDetector):
    """Cluster-based local outlier factor (pyod.models.cblof.CBLOF) [ladder: cblof].

    Params (defaults): ``n_clusters=8``, ``contamination=0.1``, ``alpha=0.9``, ``beta=5``,
    ``random_state=0``. CBLOF needs a genuine large/small cluster split (its
    ``_set_small_large_clusters`` heuristic on ``alpha``/``beta``) or pyod raises
    ``ValueError("Could not form valid cluster separation ...")`` at fit time; this is a
    pyod property of the *data*, not something this wrapper works around.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        contamination: float = 0.1,
        alpha: float = 0.9,
        beta: float = 5,
        random_state: int = 0,
        **params: Any,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            contamination=contamination,
            alpha=alpha,
            beta=beta,
            random_state=random_state,
            **params,
        )

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kwargs: Any) -> None:
        self.model_ = _PyODCBLOF(
            n_clusters=self.params["n_clusters"],
            contamination=self.params["contamination"],
            alpha=self.params["alpha"],
            beta=self.params["beta"],
            random_state=self.params["random_state"],
        )
        self.model_.fit(_sanitize(X))

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(self.model_.decision_function(_sanitize(X)), dtype=np.float64)


@register(
    "ocsvm",
    input_kind="window_stats",
    family="one_class",
    task="ad",
    tier="classical_ml",
    priority="must",
    source="pyod",
    reference_id=["R37", "R120"],
)
class OCSVMAD(AnomalyDetector):
    """One-Class SVM (pyod.models.ocsvm.OCSVM) on window_stats, fit on a
    ``<= max_train_samples``-row subsample for tractability [ladder: ocsvm].

    Params (defaults): ``kernel="rbf"``, ``nu=0.1`` (lower than pyod's own default of 0.5 -
    a more typical assumed-anomaly-fraction prior for a semi-supervised detector, exposed
    so callers can override), ``gamma="auto"``, ``max_train_samples=20000``,
    ``random_state=0``.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.1,
        gamma: Any = "auto",
        max_train_samples: int = 20_000,
        random_state: int = 0,
        **params: Any,
    ) -> None:
        super().__init__(
            kernel=kernel,
            nu=nu,
            gamma=gamma,
            max_train_samples=max_train_samples,
            random_state=random_state,
            **params,
        )

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kwargs: Any) -> None:
        Xs = _sanitize(X)
        cap = int(self.params["max_train_samples"])
        if len(Xs) > cap:
            rng = np.random.default_rng(self.params["random_state"])
            idx = rng.choice(len(Xs), size=cap, replace=False)
            Xs = Xs[idx]
        self.model_ = _PyODOCSVM(
            kernel=self.params["kernel"], nu=self.params["nu"], gamma=self.params["gamma"]
        )
        self.model_.fit(Xs)

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(self.model_.decision_function(_sanitize(X)), dtype=np.float64)


# --------------------------------------------------------------------------------------
# raw_window models
# --------------------------------------------------------------------------------------


@register(
    "kmeans_ad",
    input_kind="raw_window",
    family="cluster_outlier",
    task="ad",
    tier="classical_ml",
    priority="must",
    source="sklearn",
    reference_id=["R1"],
)
class KMeansAD(AnomalyDetector):
    """Distance to the nearest k-means centroid on flattened raw windows [ladder: kmeans_ad].

    Params (defaults): ``n_clusters=8``, ``n_init=10``, ``random_state=0``.
    """

    def __init__(self, n_clusters: int = 8, n_init: int = 10, random_state: int = 0, **params: Any) -> None:
        super().__init__(n_clusters=n_clusters, n_init=n_init, random_state=random_state, **params)

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return _sanitize(X.reshape(X.shape[0], -1))

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kwargs: Any) -> None:
        self.model_ = KMeans(
            n_clusters=min(self.params["n_clusters"], X.shape[0]),
            n_init=self.params["n_init"],
            random_state=self.params["random_state"],
        )
        self.model_.fit(self._flatten(X))

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        d = self.model_.transform(self._flatten(X))
        return d.min(axis=1).astype(np.float64)


@register(
    "ocsvm_mfcc_ams",
    input_kind="raw_window",
    family="one_class",
    task="ad",
    tier="classical_ml",
    priority="must",
    source="librosa+sklearn",
    reference_id=["R120", "R121"],
)
class OCSVMMfccAms(AnomalyDetector):
    """One-Class SVM (sklearn.svm.OneClassSVM) on MFCC + amplitude-modulation-spectrum
    (AMS) features extracted per raw_window channel with librosa/scipy.hilbert
    [ladder: ocsvm_mfcc_ams]. Healthy-only training is the caller's responsibility (feed
    ``train_regime=normal_only`` windows); this class itself never looks at labels.

    Feature vector per window: for every channel, ``n_mfcc`` mean-pooled MFCCs plus
    ``n_ams_bins`` mean-pooled bins of ``|rFFT(hilbert_envelope - mean)|`` (the amplitude
    modulation spectrum), concatenated across channels and z-scored using the training
    mean/std before the SVM.

    Params (defaults): ``sr=10500.0`` (bearing raw-window sample rate in Hz; default matches
    Ottawa's native 42 kHz decimated by ``nebulax.bench.data.load_ottawa``'s default
    ``decimate=4`` - override to match whatever adapter/window actually produced the
    windows, e.g. the sim bearing subsystem's ``fs_hz``), ``n_mfcc=13``, ``n_ams_bins=8``,
    ``kernel="rbf"``, ``nu=0.1``, ``gamma="scale"``, ``max_train_samples=20000``,
    ``random_state=0``.
    """

    def __init__(
        self,
        sr: float = 10_500.0,
        n_mfcc: int = 13,
        n_ams_bins: int = 8,
        kernel: str = "rbf",
        nu: float = 0.1,
        gamma: Any = "scale",
        max_train_samples: int = 20_000,
        random_state: int = 0,
        **params: Any,
    ) -> None:
        super().__init__(
            sr=sr,
            n_mfcc=n_mfcc,
            n_ams_bins=n_ams_bins,
            kernel=kernel,
            nu=nu,
            gamma=gamma,
            max_train_samples=max_train_samples,
            random_state=random_state,
            **params,
        )

    def _features(self, X: np.ndarray) -> np.ndarray:
        n, L, c = X.shape
        sr = float(self.params["sr"])
        n_mfcc = int(self.params["n_mfcc"])
        n_ams_bins = int(self.params["n_ams_bins"])
        n_fft = max(4, min(256, L))
        hop = max(1, n_fft // 4)
        n_mels = max(4, min(20, n_fft // 2))
        feats = np.zeros((n, c * (n_mfcc + n_ams_bins)), dtype=np.float64)
        for i in range(n):
            parts: list[np.ndarray] = []
            for ch in range(c):
                x = np.nan_to_num(X[i, :, ch].astype(np.float64)).astype(np.float32)
                mfcc = librosa.feature.mfcc(
                    y=x, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop, center=False, n_mels=n_mels
                )
                mfcc_mean = mfcc.mean(axis=1)
                # librosa silently returns fewer than n_mfcc coefficients when n_mels < n_mfcc
                # (which our n_mels formula can hit for short windows); pad/truncate to a
                # fixed width so every window's feature vector has the same shape.
                if mfcc_mean.shape[0] < n_mfcc:
                    mfcc_mean = np.pad(mfcc_mean, (0, n_mfcc - mfcc_mean.shape[0]))
                elif mfcc_mean.shape[0] > n_mfcc:
                    mfcc_mean = mfcc_mean[:n_mfcc]
                parts.append(mfcc_mean)
                env = np.abs(hilbert(x.astype(np.float64)))
                ams = np.abs(np.fft.rfft(env - env.mean()))
                bins = np.array_split(ams, n_ams_bins)
                parts.append(np.array([b.mean() if b.size else 0.0 for b in bins]))
            feats[i] = np.concatenate(parts)
        return np.nan_to_num(feats)

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kwargs: Any) -> None:
        Xf = self._features(X)
        cap = int(self.params["max_train_samples"])
        if len(Xf) > cap:
            rng = np.random.default_rng(self.params["random_state"])
            idx = rng.choice(len(Xf), size=cap, replace=False)
            Xf = Xf[idx]
        self.mu_ = Xf.mean(axis=0)
        self.sd_ = Xf.std(axis=0) + 1e-9
        self.model_ = OneClassSVM(kernel=self.params["kernel"], nu=self.params["nu"], gamma=self.params["gamma"])
        self.model_.fit((Xf - self.mu_) / self.sd_)

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        Xn = (self._features(X) - self.mu_) / self.sd_
        # sklearn's OneClassSVM.decision_function is the signed distance to the boundary,
        # positive = inlier; flip sign so higher = more anomalous (this contract).
        return -np.asarray(self.model_.decision_function(Xn), dtype=np.float64)


@register(
    "halfspace_trees_ocknn",
    input_kind="window_stats",
    family="streaming_outlier",
    task="ad",
    tier="classical_ml",
    priority="must",
    source="river",
    reference_id=["R99"],
)
class HalfspaceTreesOCKNN(AnomalyDetector):
    """Streaming Half-Space Trees + One-Class kNN hybrid [ladder: halfspace_trees_ocknn,
    R99: "highest value-per-hour item in the rail-pneumatic literature ... the combination
    gives far fewer type-I errors (much higher precision) than HS-Trees alone"].

    Two streaming components, combined every row:

    1. ``river.anomaly.HalfSpaceTrees`` - the ensemble half-space-tree score, already
       bounded in ``[0, 1]`` by river's own ``1 - score / max_score`` normalisation.
    2. A one-class kNN distance to a fixed-size sliding window of the ``knn_window`` most
       recently seen rows (the "recent normal buffer"), each row min-max scaled by the same
       per-feature ``limits`` derived at fit time (below) so the kNN metric lives on a
       comparable ``[0, 1]``-ish scale, then normalised by ``sqrt(n_features)`` (the diagonal
       of the unit hypercube) so it is directly comparable to the HST score.

    The two normalised scores are combined by geometric mean (``sqrt(hst * knn)``, each
    floored at a small epsilon before multiplying): unlike an arithmetic mean/sum, a
    geometric mean requires *both* views to agree that a row is unusual before the combined
    score gets large, which is exactly the "far fewer type-I errors" property the ladder row
    cites - a row that only one of the two unsupervised views flags stays damped.

    Both components are streaming/online, in the same score-then-learn cycle every row:
    ``fit`` warms up the tree ensemble and the kNN buffer over the training rows in order;
    ``score`` continues the identical cycle over the scored rows (score with the current
    state, then learn/append), which is what makes this a genuinely online detector rather
    than a static one. ``SEQUENTIAL = True`` so the runner scores one series at a time in
    ascending ``t``, keeping two interleaved components from sharing a running buffer/tree
    state.

    river's ``HalfSpaceTrees`` assumes every feature already lies in ``[0, 1]`` unless you
    pass explicit per-feature ``limits`` (its constructor otherwise defaults every feature's
    range to exactly ``(0.0, 1.0)`` - it does *not* auto-adapt to the data). Since
    ``window_stats`` features are not scaled to ``[0, 1]``, ``_fit`` derives ``limits`` from
    the training rows' per-feature ``(min, max)`` (padded 10% each side) and passes them
    through, and reuses the same ``(min, max)`` to scale rows for the kNN component; without
    this the tree splits are degenerate on real feature scales and the score stops tracking
    "more anomalous" at all.

    Params (defaults): ``n_trees=10``, ``height=8``, ``window_size=250``, ``seed=0``,
    ``knn_k=5`` (neighbours for the one-class kNN distance), ``knn_window=None`` (buffer
    size for the recent-normal-rows kNN reference; defaults to ``window_size`` so both
    streaming components share one notion of "recent").
    """

    SEQUENTIAL = True

    def __init__(
        self,
        n_trees: int = 10,
        height: int = 8,
        window_size: int = 250,
        seed: int = 0,
        knn_k: int = 5,
        knn_window: int | None = None,
        **params: Any,
    ) -> None:
        super().__init__(
            n_trees=n_trees,
            height=height,
            window_size=window_size,
            seed=seed,
            knn_k=knn_k,
            knn_window=knn_window,
            **params,
        )

    @staticmethod
    def _keys(f: int) -> list[str]:
        return [f"f{i}" for i in range(f)]

    def _scale(self, row: np.ndarray) -> np.ndarray:
        return (row - self._lo_) / self._scale_

    def _knn_distance(self, x_scaled: np.ndarray) -> float:
        """Mean distance to the ``knn_k`` nearest rows in the recent-normal buffer,
        normalised into a scale comparable to the ``[0, 1]`` HST score. An empty buffer
        (only possible before any row has ever been seen) scores 0 - "not yet anomalous",
        matching river's own ``_first_window`` convention for HalfSpaceTrees."""
        if not self._knn_buffer_:
            return 0.0
        diff = np.asarray(self._knn_buffer_, dtype=np.float64) - x_scaled
        dists = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        k = min(int(self.params["knn_k"]), dists.size)
        nearest = np.partition(dists, k - 1)[:k]
        return float(nearest.mean()) / float(np.sqrt(self._n_features_))

    @staticmethod
    def _combine(hst_score: float, knn_norm: float) -> float:
        eps = 1e-3
        return float(np.sqrt(max(hst_score, eps) * max(knn_norm, eps)))

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kwargs: Any) -> None:
        Xs = _sanitize(X)
        n, f = Xs.shape
        mins = Xs.min(axis=0)
        maxs = Xs.max(axis=0)
        pad = (maxs - mins) * 0.1 + 1e-6
        lo = mins - pad
        hi = maxs + pad
        limits = {f"f{i}": (float(lo[i]), float(hi[i])) for i in range(f)}
        self.model_ = _river_anomaly.HalfSpaceTrees(
            n_trees=self.params["n_trees"],
            height=self.params["height"],
            window_size=self.params["window_size"],
            limits=limits,
            seed=self.params["seed"],
        )
        self._lo_ = lo
        self._scale_ = np.maximum(hi - lo, 1e-9)
        self._n_features_ = f
        knn_window = int(self.params["knn_window"] or self.params["window_size"])
        self._knn_buffer_: deque[np.ndarray] = deque(maxlen=max(1, knn_window))
        keys = self._keys(f)
        for row in Xs:
            x = dict(zip(keys, row.tolist()))
            self.model_.score_one(x)
            self.model_.learn_one(x)
            self._knn_buffer_.append(self._scale(row))

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        Xs = _sanitize(X)
        keys = self._keys(self._n_features_)
        out = np.empty(Xs.shape[0], dtype=np.float64)
        for i, row in enumerate(Xs):
            x = dict(zip(keys, row.tolist()))
            hst_score = self.model_.score_one(x)
            knn_norm = self._knn_distance(self._scale(row))
            out[i] = self._combine(hst_score, knn_norm)
            self.model_.learn_one(x)
            self._knn_buffer_.append(self._scale(row))
        return out


def _exclusion_aware_knn_mean(dists: np.ndarray, idxs: np.ndarray, k: int, excl_zone: int) -> np.ndarray:
    """Reduce a per-row ``(n, k_search)`` block of stumpy top-k AB-join candidate
    distances/indices to the mean of the first ``k`` candidates whose matched positions in
    the reference series are mutually at least ``excl_zone`` samples apart.

    stumpy's own top-k retrieval only excludes *trivial* (near-identical-offset) matches for
    a self-join; for an AB-join it happily returns ``k`` near-duplicate neighbours that all
    sit inside one adjacent cluster of the reference series, which is not a genuine k-NN
    density estimate. Over-fetching ``k_search >= k`` candidates (sorted by distance,
    ascending) and greedily accepting only those spaced ``>= excl_zone`` apart from every
    already-accepted match gives the exclusion-zone-aware kNN retrieval the ladder's
    MMPAD-style ``matrix_profile_discord`` row names.
    """
    n, k_search = dists.shape
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        chosen: list[float] = []
        chosen_idx: list[float] = []
        for j in range(k_search):
            d_j = dists[i, j]
            idx_j = idxs[i, j]
            if not np.isfinite(d_j) or idx_j < 0:
                continue
            if all(abs(idx_j - ci) >= excl_zone for ci in chosen_idx):
                chosen.append(float(d_j))
                chosen_idx.append(idx_j)
                if len(chosen) >= k:
                    break
        if not chosen:
            finite = dists[i][np.isfinite(dists[i])]
            chosen = [float(finite[0])] if finite.size else [0.0]
        out[i] = float(np.mean(chosen))
    return out


def _causal_moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Causal (backward-looking) moving average of window ``w`` (``w <= 1`` is a no-op)."""
    w = max(1, int(w))
    if w <= 1:
        return x
    n = len(x)
    csum = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - w + 1)
    counts = (idx - lo + 1).astype(np.float64)
    return (csum[idx + 1] - csum[lo]) / counts


@register(
    "matrix_profile_discord",
    input_kind="raw_window",
    family="matrix_profile",
    task="ad",
    tier="classical_ml",
    priority="must",
    source="stumpy",
    reference_id=["R11"],
)
class MatrixProfileDiscord(AnomalyDetector):
    """MMPAD-style matrix-profile discord score [ladder: matrix_profile_discord, R11,
    https://arxiv.org/abs/2604.02445]. Each channel's training windows are flattened to one
    reference 1-D series ``train_series_[ch]``; the scored batch is flattened the same way
    but is **never concatenated with the training series and never self-joined against
    itself** - every scored subsequence is matched only against the fitted training
    reference via a stumpy AB-join (``stumpy.stump(query, m, reference, ignore_trivial=False,
    k=...)``). This is what stops a repeated/duplicated scored anomaly from becoming its own
    nearest neighbour and being suppressed toward zero (the bug this class previously had):
    a subsequence's neighbours can structurally only come from ``train_series_``.

    Per query subsequence, the method:

    1. **Exclusion-zone-aware kNN retrieval** (:func:`_exclusion_aware_knn_mean`) - fetches
       more than ``k`` candidate matches from the training reference, sorted by distance, and
       greedily keeps the first ``k`` whose matched positions are mutually more than
       ``m // 4`` reference samples apart, then averages their distances. This gives a
       genuine k-NN density estimate against the reference instead of ``k`` near-duplicate
       copies of one matching region.
    2. **Pre-sorted multidimensional aggregation** - the per-channel kNN profiles are
       stacked ``(c, n_subseq)``, sorted *descending* along the channel axis at every
       subsequence position, and the mean of the top ``dim_count`` channels is taken as that
       position's discord value. ``dim_count`` is driven by the ladder's ``d`` param
       (``dim_count = round(d * c)``, at least 1); ``d=None`` falls back to ``agg`` for
       backward compatibility (``"max"`` -> ``dim_count=1``, i.e. the single most-deviant
       channel; ``"mean"`` -> ``dim_count=c``, i.e. every channel). Sorting descending (most
       anomalous channel first) rather than ascending means a fault that only shows up in a
       minority of channels is not diluted away by the rest of the (normal-looking) sensors.
    3. Per-window aggregation: a window's score is the max discord value among the
       subsequence positions that start inside it (unchanged from before).
    4. **Score post-processing** - an optional causal moving average (``post_ma`` windows,
       default 1 = no smoothing) over the final per-window score sequence.

    Params (defaults): ``m=16`` (subsequence length in samples), ``k=5`` (kNN neighbours
    retrieved from the training reference), ``d=None`` (multidimensional-aggregation
    fraction of channels to keep, sorted most-anomalous-first; ``None`` defers to ``agg``),
    ``agg="max"`` (legacy cross-channel fallback used only when ``d`` is ``None``: ``"max"``
    or ``"mean"``), ``post_ma=1`` (causal moving-average window over the final per-window
    score; ``1`` = no smoothing), ``normalize=False`` (z-normalise subsequences before
    distance, stumpy's own default is ``True``; kept ``False`` here on purpose - z-normalised
    distance divides out amplitude/scale, so an amplitude fault, e.g. a bearing's vibration
    RMS rising or a pressure trace's DC level shifting - the dominant fault signature in this
    project's raw-window channels - becomes literally invisible to the discord score; raw
    Euclidean distance keeps it), ``use_gpu=False`` (``stumpy.gpu_stump`` only supports 1-NN,
    so it is only attempted when the resolved kNN retrieval collapses to a single candidate;
    every ``k > 1`` run - the default - always uses the CPU ``stumpy.stump(..., k=...)``,
    which is also the ladder's own deviation-noted CPU default for determinism/portability
    over ``stumpy.gpu_stump``).
    """

    def __init__(
        self,
        m: int = 16,
        agg: str = "max",
        use_gpu: bool = False,
        k: int = 5,
        d: float | None = None,
        post_ma: int = 1,
        normalize: bool = False,
        **params: Any,
    ) -> None:
        super().__init__(
            m=m, agg=agg, use_gpu=use_gpu, k=k, d=d, post_ma=post_ma, normalize=normalize, **params
        )

    @staticmethod
    def _flatten_per_channel(X: np.ndarray) -> tuple[list[np.ndarray], int]:
        Xs = _sanitize(X)
        n, L, c = Xs.shape
        return [np.ascontiguousarray(Xs[:, :, ch].reshape(-1)) for ch in range(c)], L

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kwargs: Any) -> None:
        series, L = self._flatten_per_channel(X)
        self.train_series_ = series
        self.L_ = L
        self.n_train_ = X.shape[0]
        self.c_ = X.shape[2]

    def _knn_profile(self, query: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Exclusion-zone-aware kNN distance profile of every subsequence in ``query``
        against ``reference`` ONLY - an AB-join, so a query subsequence can never be
        matched against another query subsequence (must-fix: scoring peers can never
        suppress each other, and repeated anomalies in the scored batch all score high
        independently)."""
        m = int(self.params["m"])
        k = max(1, int(self.params["k"]))
        n_ref_subseq = len(reference) - m + 1
        if n_ref_subseq <= 0:
            raise ValueError(
                f"{self.name}: training reference ({len(reference)} samples) shorter than m={m}"
            )
        k = min(k, n_ref_subseq)
        k_search = min(n_ref_subseq, max(k * 3, k))
        normalize = bool(self.params["normalize"])
        use_gpu = bool(self.params["use_gpu"]) and k_search == 1
        if use_gpu:
            try:
                mp = stumpy.gpu_stump(query, m=m, T_B=reference, ignore_trivial=False, normalize=normalize)
            except Exception:
                mp = stumpy.stump(query, m, reference, ignore_trivial=False, k=k_search, normalize=normalize)
        else:
            mp = stumpy.stump(query, m, reference, ignore_trivial=False, k=k_search, normalize=normalize)
        dists = np.asarray(mp[:, :k_search], dtype=np.float64)
        idxs = np.asarray(mp[:, k_search : 2 * k_search], dtype=np.float64)
        excl_zone = max(1, m // 4)
        return _exclusion_aware_knn_mean(dists, idxs, k, excl_zone)

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        if X.shape[2] != self.c_:
            raise ValueError(f"{self.name}.score: fitted on {self.c_} channels, got {X.shape[2]}")
        test_series, L = self._flatten_per_channel(X)
        n_test = X.shape[0]
        m = int(self.params["m"])
        c = self.c_

        profiles = [self._knn_profile(test_series[ch], self.train_series_[ch]) for ch in range(c)]
        plen = len(profiles[0])
        chan_profile = np.vstack(profiles)  # (c, plen)

        d = self.params["d"]
        if d is not None:
            dim_count = max(1, min(c, round(float(d) * c)))
        else:
            dim_count = c if self.params["agg"] == "mean" else 1
        sorted_desc = np.sort(chan_profile, axis=0)[::-1, :]  # most-anomalous channel first
        aggregated = sorted_desc[:dim_count, :].mean(axis=0)

        scores = np.empty(n_test, dtype=np.float64)
        for i in range(n_test):
            start = i * L
            end = start + L
            lo = max(0, start - m + 1)
            hi = min(plen, end)
            if lo < hi:
                scores[i] = aggregated[lo:hi].max()
            else:
                idx = min(max(start, 0), max(plen - 1, 0))
                scores[i] = aggregated[idx] if plen else 0.0

        return _causal_moving_average(scores, int(self.params["post_ma"]))
