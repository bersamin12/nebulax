"""aeon-based time-series classification models, ``task="cls"``, ``input_kind="raw_window"``.

Registered here (names/input_kind/family/task copied verbatim from ``configs/model_ladder.yaml``):

* ``multirocket_ridge``  - family ``convolution_kernel`` - MultiRocket convolutional features
  scored by ``RidgeClassifierCV`` (``aeon.classification.convolution_based.
  MultiRocketClassifier``, whose own default estimator *is* ``RidgeClassifierCV``, so no
  separate estimator wiring is needed).
* ``multirocket_hydra``  - family ``convolution_kernel`` - concatenated MultiRocket + Hydra
  kernel features scored by a linear classifier (``aeon.classification.convolution_based.
  MultiRocketHydraClassifier``).
* ``quant``               - family ``interval`` - dyadic-interval quantile features scored by
  an ``ExtraTreesClassifier`` (``aeon.classification.interval_based.QUANTClassifier``).

aeon 1.5 classifiers are "collection" estimators expecting ``(n_cases, n_channels,
n_timepoints)``, while the benchmark's ``raw_window`` ``X`` is ``(n, L, n_channels)``
(``nebulax/bench/base.py``); :func:`_to_aeon_collection` transposes axes 1 and 2 and never
reshapes across ``input_kind``. Class names verified against the installed aeon 1.5.0 via
``dir(aeon.classification.convolution_based)`` / ``dir(aeon.classification.interval_based)``
and each constructor's ``inspect.signature`` before wiring the wrappers below.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from aeon.classification.convolution_based import MultiRocketClassifier, MultiRocketHydraClassifier
from aeon.classification.interval_based import QUANTClassifier

from nebulax.bench.base import Classifier
from nebulax.bench.registry import register

__all__ = ["MultiRocketRidge", "MultiRocketHydra", "Quant"]


def _to_aeon_collection(X: np.ndarray) -> np.ndarray:
    """``(n, L, c)`` raw_window -> aeon's ``(n, n_channels, n_timepoints)`` collection format."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"_to_aeon_collection: expected raw_window X with shape (n, L, c), got {arr.shape}")
    return np.ascontiguousarray(np.transpose(arr, (0, 2, 1)))


@register(
    "multirocket_ridge",
    input_kind="raw_window",
    family="convolution_kernel",
    task="cls",
    tier="classical_ml",
    reference_id=["R47", "R43", "R44"],
)
class MultiRocketRidge(Classifier):
    """MultiRocket convolutional features + ``RidgeClassifierCV`` (aeon's default estimator).

    Params (constructor kwargs, defaults copied from
    ``aeon.classification.convolution_based.MultiRocketClassifier``): ``n_kernels=10_000``,
    ``max_dilations_per_kernel=32``, ``n_features_per_kernel=4``, ``estimator=None`` (->
    ``RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))``), ``class_weight=None``,
    ``n_jobs=1``, ``random_state=0``.
    """

    def __init__(
        self,
        n_kernels: int = 10_000,
        max_dilations_per_kernel: int = 32,
        n_features_per_kernel: int = 4,
        estimator: Any = None,
        class_weight: Any = None,
        n_jobs: int = 1,
        random_state: int = 0,
    ) -> None:
        super().__init__(
            n_kernels=n_kernels,
            max_dilations_per_kernel=max_dilations_per_kernel,
            n_features_per_kernel=n_features_per_kernel,
            estimator=estimator,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self._clf = MultiRocketClassifier(
            n_kernels=n_kernels,
            max_dilations_per_kernel=max_dilations_per_kernel,
            n_features_per_kernel=n_features_per_kernel,
            estimator=estimator,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        self._clf.fit(_to_aeon_collection(X), y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict(_to_aeon_collection(X))

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(_to_aeon_collection(X))


@register(
    "multirocket_hydra",
    input_kind="raw_window",
    family="convolution_kernel",
    task="cls",
    tier="classical_ml",
    reference_id=["R48", "R43"],
)
class MultiRocketHydra(Classifier):
    """Combined MultiRocket + Hydra kernel features + a linear classifier.

    Params (constructor kwargs, defaults copied from
    ``aeon.classification.convolution_based.MultiRocketHydraClassifier``): ``n_kernels=8``
    (per Hydra group), ``n_groups=64``, ``class_weight=None``, ``n_jobs=1``,
    ``random_state=0``. aeon does not expose a separate ``estimator=`` kwarg for this class
    (the Ridge classifier stage is internal).
    """

    def __init__(
        self,
        n_kernels: int = 8,
        n_groups: int = 64,
        class_weight: Any = None,
        n_jobs: int = 1,
        random_state: int = 0,
    ) -> None:
        super().__init__(
            n_kernels=n_kernels,
            n_groups=n_groups,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self._clf = MultiRocketHydraClassifier(
            n_kernels=n_kernels,
            n_groups=n_groups,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        self._clf.fit(_to_aeon_collection(X), y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict(_to_aeon_collection(X))

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(_to_aeon_collection(X))


@register(
    "quant",
    input_kind="raw_window",
    family="interval",
    task="cls",
    tier="classical_ml",
    reference_id=["R49"],
    note="interval quantiles are directly interpretable in the twin",
)
class Quant(Classifier):
    """QUANT: dyadic-interval quantile features + ``ExtraTreesClassifier`` (aeon default).

    Params (constructor kwargs, defaults copied from
    ``aeon.classification.interval_based.QUANTClassifier``): ``interval_depth=6``,
    ``quantile_divisor=4``, ``estimator=None`` (-> ``ExtraTreesClassifier(n_estimators=200)``),
    ``class_weight=None``, ``random_state=0``.
    """

    def __init__(
        self,
        interval_depth: int = 6,
        quantile_divisor: int = 4,
        estimator: Any = None,
        class_weight: Any = None,
        random_state: int = 0,
    ) -> None:
        super().__init__(
            interval_depth=interval_depth,
            quantile_divisor=quantile_divisor,
            estimator=estimator,
            class_weight=class_weight,
            random_state=random_state,
        )
        self._clf = QUANTClassifier(
            interval_depth=interval_depth,
            quantile_divisor=quantile_divisor,
            estimator=estimator,
            class_weight=class_weight,
            random_state=random_state,
        )

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        self._clf.fit(_to_aeon_collection(X), y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict(_to_aeon_collection(X))

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(_to_aeon_collection(X))
