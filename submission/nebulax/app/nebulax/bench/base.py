"""Abstract model interfaces for the benchmark. Every model in ``nebulax/models/*.py``
subclasses one of these and registers itself with :func:`nebulax.bench.registry.register`.

Two tasks, two shapes
---------------------
* :class:`AnomalyDetector` - ``fit(X, t=None)`` then ``score(X, t=None) -> (n,)`` where a
  **higher score means more anomalous**. Change-point (``task="cpd"``) models use the same
  interface: the runner converts segmentation output into a continuous per-timestamp score.
* :class:`Classifier` - ``fit(X, y)`` then ``predict(X)`` / ``predict_proba(X)``.
  Severity/RUL regressors (``task="rul"``) subclass :class:`Regressor`.

``input_kind`` - what ``X`` is
------------------------------
Taken verbatim from ``configs/model_ladder.yaml`` (every ladder row carries one):

``"window_stats"``    ``(n_windows, n_features)`` float32 - per-window summary statistics.
``"raw_window"``      ``(n_windows, L, n_channels)`` float32 - raw/downsampled windows.
``"cycle_features"``  ``(n_cycles, n_features)`` float32 - per door/compressor-cycle features.

The runner is responsible for producing the right ``X`` for a model's declared
``input_kind``; a model never reshapes across kinds itself.

Contract rules the runner relies on
-----------------------------------
* ``fit`` is **concrete here**: it times ``_fit`` and sets :attr:`fit_seconds`, then sets
  ``is_fitted``. Subclasses implement ``_fit``/``_score`` (or ``_predict``), never ``fit``.
* Every constructor keyword lands in :attr:`params` unchanged so the runner can hash the
  config and write it into the results row.
* ``score`` must return a finite ``float64`` array of length ``len(X)``; NaN is rejected here
  rather than silently poisoning threshold calibration.
* Normal-only ("semi-supervised") models fit on clean data; the runner never shows them test
  data and never calibrates thresholds on test scores.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Final, Literal

import numpy as np

__all__ = [
    "InputKind",
    "Task",
    "INPUT_KINDS",
    "TASKS",
    "BaseModel",
    "AnomalyDetector",
    "Classifier",
    "Regressor",
]

InputKind = Literal["window_stats", "raw_window", "cycle_features"]
Task = Literal["ad", "cls", "cpd", "rul"]

#: The three input kinds used by ``configs/model_ladder.yaml``.
INPUT_KINDS: Final[tuple[str, ...]] = ("window_stats", "raw_window", "cycle_features")
#: ``ad`` anomaly detection (the ladder's default), ``cls`` classification,
#: ``cpd`` change-point detection, ``rul`` remaining-useful-life / severity regression.
TASKS: Final[tuple[str, ...]] = ("ad", "cls", "cpd", "rul")


class BaseModel(ABC):
    """Common machinery: identity, params, fit timing.

    Class attributes ``name``, ``family``, ``input_kind`` and ``task`` are normally set by the
    :func:`nebulax.bench.registry.register` decorator; setting them by hand is fine too.
    """

    #: Registry key, e.g. ``"iforest"``. Unique across the whole registry.
    name: ClassVar[str] = "unnamed"
    #: Model family from ``configs/model_ladder.yaml`` (``"one_liner"``, ``"subspace"``,
    #: ``"gradient_boosting"``, ``"autoencoder"``, ``"tsfm_ad"``, ...).
    family: ClassVar[str] = "unknown"
    #: One of :data:`INPUT_KINDS`.
    input_kind: ClassVar[str] = "window_stats"
    #: One of :data:`TASKS`.
    task: ClassVar[str] = "ad"

    def __init__(self, **params: Any) -> None:
        #: Every constructor keyword, verbatim, for the results row and the config hash.
        self.params: dict[str, Any] = dict(params)
        #: Wall-clock seconds spent in the last :meth:`fit`.
        self.fit_seconds: float = 0.0
        #: Set by :meth:`fit`.
        self.is_fitted: bool = False

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"{type(self).__name__}(name={self.name!r}, task={self.task!r}, params={self.params})"

    def get_params(self) -> dict[str, Any]:
        """The params dict (copy), for the benchmark results row."""
        return dict(self.params)

    @staticmethod
    def _as_float32(X: Any, *, what: str) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim not in (2, 3):
            raise ValueError(f"{what}: X must be 2-D (n, f) or 3-D (n, L, c), got shape {arr.shape}")
        if arr.shape[0] == 0:
            raise ValueError(f"{what}: X has zero rows")
        return arr

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name}: call fit() before scoring/predicting")


#: Row-context keywords the runner passes to ``_fit`` / ``_score`` **when the method declares
#: them** (by name or via ``**kwargs``): ``feature_names`` (list[str], the columns of a 2-D X or
#: the channels of a 3-D one), ``series`` (object array, one component id per row - the
#: series an alarm episode is raised on), ``unit`` (object array, one train / bearing / rig id
#: per row), ``fs_hz`` (float, the sampling rate of a raw window's samples as the loader
#: delivered them - after any decimation - so a spectral model never assumes the sensor's
#: native rate). A model that never declares them is called exactly as before.
CONTEXT_KEYS: tuple[str, ...] = ("feature_names", "series", "unit", "fs_hz")


def accepts_keyword(fn: Any, name: str) -> bool:
    """True if ``fn`` takes ``name`` explicitly or through ``**kwargs``."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return False
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class AnomalyDetector(BaseModel):
    """Unsupervised / normal-only anomaly detector (``task="ad"`` or ``"cpd"``).

    Implement :meth:`_fit` and :meth:`_score`. ``t`` is the optional per-row end-timestamp
    array (``datetime64[ms]`` or float seconds) for models that need time (forecast-residual,
    change-point, drift detectors); most models ignore it.

    **Row context.** A model that needs column names or the identity of the series each row
    belongs to declares the keyword on ``_fit`` / ``_score`` (see :data:`CONTEXT_KEYS`); the
    runner then passes ``feature_names`` / ``series`` / ``unit`` and :meth:`score` forwards
    whatever ``_score`` accepts. A model whose state is sequential (EWMA, CUSUM, drift
    detectors) sets ``SEQUENTIAL = True``: the runner then scores it one series at a time in
    ascending ``t``, each series on a **fresh copy** of the fitted model, so two interleaved
    components never share a running statistic and the validation pass never changes the
    state the test pass is scored with. A model that is not ``SEQUENTIAL`` must not learn
    inside ``_score``.
    """

    task: ClassVar[str] = "ad"
    #: Score one series at a time, in time order (runner-side). See the class docstring.
    SEQUENTIAL: ClassVar[bool] = False

    @abstractmethod
    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kwargs: Any) -> None:
        """Fit on (normal or contaminated) training windows."""

    @abstractmethod
    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        """Per-row anomaly score, higher = more anomalous."""

    def fit(self, X: Any, t: np.ndarray | None = None, **kwargs: Any) -> "AnomalyDetector":
        """Fit and record :attr:`fit_seconds`. Returns ``self``."""
        arr = self._as_float32(X, what=f"{self.name}.fit")
        t0 = time.perf_counter()
        self._fit(arr, t, **kwargs)
        self.fit_seconds = time.perf_counter() - t0
        self.is_fitted = True
        return self

    def score(self, X: Any, t: np.ndarray | None = None, **context: Any) -> np.ndarray:
        """Anomaly scores, ``float64`` of length ``len(X)``, higher = more anomalous.

        ``context`` (:data:`CONTEXT_KEYS`) is forwarded to ``_score`` only for the keywords it
        declares, so a plain ``_score(X, t)`` keeps working unchanged.
        """
        self._check_fitted()
        arr = self._as_float32(X, what=f"{self.name}.score")
        kw = {k: v for k, v in context.items() if accepts_keyword(self._score, k)}
        out = np.asarray(self._score(arr, t, **kw), dtype=np.float64).reshape(-1)
        if out.size != arr.shape[0]:
            raise ValueError(
                f"{self.name}.score: returned {out.size} scores for {arr.shape[0]} rows"
            )
        if not np.isfinite(out).all():
            raise ValueError(
                f"{self.name}.score: returned {int((~np.isfinite(out)).sum())} non-finite score(s); "
                f"impute or clip inside the model, never leave NaN for the threshold calibrator"
            )
        return out

    def fit_score(self, X: Any, t: np.ndarray | None = None, **kwargs: Any) -> np.ndarray:
        """Transductive convenience: ``fit(X).score(X)``. Only for genuinely unsupervised runs."""
        return self.fit(X, t, **kwargs).score(X, t)


class Classifier(BaseModel):
    """Supervised classifier (``task="cls"``).

    Implement :meth:`_fit` and :meth:`_predict`; override :meth:`_predict_proba` when the
    model has calibrated probabilities (macro-F1 is the primary metric, PR-AUC needs proba).
    """

    task: ClassVar[str] = "cls"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        #: Sorted unique training labels, set by :meth:`fit`.
        self.classes_: np.ndarray = np.empty(0)

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        """Fit on labelled windows."""

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted labels, one per row."""

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.name}: predict_proba is not implemented for this model")

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "Classifier":
        """Fit and record :attr:`fit_seconds`. Returns ``self``."""
        arr = self._as_float32(X, what=f"{self.name}.fit")
        y_arr = np.asarray(y).reshape(-1)
        if y_arr.size != arr.shape[0]:
            raise ValueError(f"{self.name}.fit: X has {arr.shape[0]} rows but y has {y_arr.size}")
        self.classes_ = np.unique(y_arr)
        t0 = time.perf_counter()
        self._fit(arr, y_arr, **kwargs)
        self.fit_seconds = time.perf_counter() - t0
        self.is_fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predicted labels, length ``len(X)``."""
        self._check_fitted()
        arr = self._as_float32(X, what=f"{self.name}.predict")
        out = np.asarray(self._predict(arr)).reshape(-1)
        if out.size != arr.shape[0]:
            raise ValueError(f"{self.name}.predict: returned {out.size} labels for {arr.shape[0]} rows")
        return out

    def predict_proba(self, X: Any) -> np.ndarray:
        """Class probabilities, shape ``(len(X), len(classes_))``, columns ordered by
        :attr:`classes_`."""
        self._check_fitted()
        arr = self._as_float32(X, what=f"{self.name}.predict_proba")
        out = np.asarray(self._predict_proba(arr), dtype=np.float64)
        if out.ndim != 2 or out.shape[0] != arr.shape[0]:
            raise ValueError(
                f"{self.name}.predict_proba: expected shape ({arr.shape[0]}, n_classes), got {out.shape}"
            )
        return out


class Regressor(BaseModel):
    """Severity / RUL regressor (``task="rul"``). Implement :meth:`_fit` and :meth:`_predict`."""

    task: ClassVar[str] = "rul"

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        """Fit on targets (severity stage, or remaining life in seconds)."""

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Point predictions, one per row."""

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "Regressor":
        """Fit and record :attr:`fit_seconds`. Returns ``self``."""
        arr = self._as_float32(X, what=f"{self.name}.fit")
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if y_arr.size != arr.shape[0]:
            raise ValueError(f"{self.name}.fit: X has {arr.shape[0]} rows but y has {y_arr.size}")
        t0 = time.perf_counter()
        self._fit(arr, y_arr, **kwargs)
        self.fit_seconds = time.perf_counter() - t0
        self.is_fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Point predictions, length ``len(X)``."""
        self._check_fitted()
        arr = self._as_float32(X, what=f"{self.name}.predict")
        out = np.asarray(self._predict(arr), dtype=np.float64).reshape(-1)
        if out.size != arr.shape[0]:
            raise ValueError(f"{self.name}.predict: returned {out.size} values for {arr.shape[0]} rows")
        return out
