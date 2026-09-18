"""Contract tests for nebulax.bench: the abstract model interfaces and the registry."""

from __future__ import annotations

import numpy as np
import pytest

from nebulax.bench import base as B
from nebulax.bench import registry as R


@pytest.fixture
def clean_registry():
    """Register throwaway models without leaking them into other tests."""
    names: list[str] = []
    yield names
    for n in names:
        R.unregister(n)


def _make_detector(name: str, names: list[str], **kw):
    @R.register(name, input_kind=kw.pop("input_kind", "window_stats"), family=kw.pop("family", "one_liner"), **kw)
    class Detector(B.AnomalyDetector):
        def _fit(self, X, t=None, **kwargs):
            self.mu_ = X.mean(axis=0)
            self.sd_ = X.std(axis=0) + 1e-9

        def _score(self, X, t=None):
            return np.abs((X - self.mu_) / self.sd_).max(axis=1)

    names.append(name)
    return Detector


def test_input_kinds_match_the_model_ladder():
    assert B.INPUT_KINDS == ("window_stats", "raw_window", "cycle_features")
    assert B.TASKS == ("ad", "cls", "cpd", "rul")


def test_register_stamps_metadata_and_build_passes_params(clean_registry):
    _make_detector("t_zscore", clean_registry, tier="statistical", reference_id=["R35"])
    spec = R.get_spec("t_zscore")
    assert spec.input_kind == "window_stats" and spec.family == "one_liner" and spec.task == "ad"
    assert spec.meta == {"tier": "statistical", "reference_id": ["R35"]}
    model = R.build("t_zscore", k=3, ewma=0.1)
    assert model.params == {"k": 3, "ewma": 0.1}
    assert model.get_params() == model.params and model.get_params() is not model.params
    assert model.name == "t_zscore" and model.input_kind == "window_stats"
    assert R.is_registered("t_zscore")
    assert [s.name for s in R.list_models(input_kind="window_stats", task="ad", names=["t_zscore"])] == ["t_zscore"]


def test_fit_times_itself_and_score_shapes(clean_registry):
    _make_detector("t_zscore2", clean_registry)
    model = R.build("t_zscore2")
    X = np.random.default_rng(0).normal(size=(200, 5))
    assert not model.is_fitted
    with pytest.raises(RuntimeError, match="call fit"):
        model.score(X)
    out = model.fit(X).score(X)
    assert model.is_fitted and model.fit_seconds > 0.0
    assert out.shape == (200,) and out.dtype == np.float64
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, model.fit_score(X))


def test_score_rejects_nan_and_wrong_length(clean_registry):
    @R.register("t_broken", input_kind="window_stats", family="trivial")
    class Broken(B.AnomalyDetector):
        def _fit(self, X, t=None, **kw):
            pass

        def _score(self, X, t=None):
            out = np.zeros(len(X))
            out[0] = np.nan
            return out

    clean_registry.append("t_broken")

    @R.register("t_short", input_kind="window_stats", family="trivial")
    class Short(B.AnomalyDetector):
        def _fit(self, X, t=None, **kw):
            pass

        def _score(self, X, t=None):
            return np.zeros(len(X) - 1)

    clean_registry.append("t_short")

    X = np.zeros((10, 3))
    with pytest.raises(ValueError, match="non-finite score"):
        R.build("t_broken").fit(X).score(X)
    with pytest.raises(ValueError, match="returned 9 scores for 10 rows"):
        R.build("t_short").fit(X).score(X)


def test_raw_window_models_accept_3d_input(clean_registry):
    @R.register("t_raw", input_kind="raw_window", family="autoencoder")
    class RawAE(B.AnomalyDetector):
        def _fit(self, X, t=None, **kw):
            assert X.ndim == 3
            self.ref_ = X.mean(axis=0)

        def _score(self, X, t=None):
            return np.abs(X - self.ref_).mean(axis=(1, 2))

    clean_registry.append("t_raw")
    X = np.random.default_rng(1).normal(size=(40, 60, 3))
    scores = R.build("t_raw").fit(X).score(X)
    assert scores.shape == (40,)


def test_classifier_interface(clean_registry):
    @R.register("t_cls", input_kind="cycle_features", family="gradient_boosting", task="cls")
    class Nearest(B.Classifier):
        def _fit(self, X, y, **kw):
            self.means_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])

        def _predict(self, X):
            d = np.linalg.norm(X[:, None, :] - self.means_[None], axis=2)
            return self.classes_[np.argmin(d, axis=1)]

        def _predict_proba(self, X):
            d = np.linalg.norm(X[:, None, :] - self.means_[None], axis=2)
            w = 1.0 / (d + 1e-9)
            return w / w.sum(axis=1, keepdims=True)

    clean_registry.append("t_cls")
    rng = np.random.default_rng(2)
    X = np.concatenate([rng.normal(0, 1, (50, 4)), rng.normal(5, 1, (50, 4))])
    y = np.array([0] * 50 + [1] * 50)
    model = R.build("t_cls")
    assert model.task == "cls" and model.input_kind == "cycle_features"
    model.fit(X, y)
    assert (model.predict(X) == y).mean() > 0.95
    proba = model.predict_proba(X)
    assert proba.shape == (100, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    with pytest.raises(ValueError, match="X has 100 rows but y has 99"):
        model.fit(X, y[:-1])


def test_classifier_without_proba_raises(clean_registry):
    @R.register("t_cls_np", input_kind="window_stats", family="trivial", task="cls")
    class NoProba(B.Classifier):
        def _fit(self, X, y, **kw):
            pass

        def _predict(self, X):
            return np.zeros(len(X))

    clean_registry.append("t_cls_np")
    m = R.build("t_cls_np").fit(np.zeros((5, 2)), np.zeros(5))
    with pytest.raises(NotImplementedError, match="predict_proba is not implemented"):
        m.predict_proba(np.zeros((5, 2)))


def test_regressor_interface(clean_registry):
    @R.register("t_rul", input_kind="cycle_features", family="degradation_path", task="rul")
    class Mean(B.Regressor):
        def _fit(self, X, y, **kw):
            self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]

        def _predict(self, X):
            return X @ self.coef_

    clean_registry.append("t_rul")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 3))
    y = X @ np.array([1.0, -2.0, 0.5])
    m = R.build("t_rul").fit(X, y)
    np.testing.assert_allclose(m.predict(X), y, atol=1e-6)
    assert m.task == "rul"


def test_register_validation_errors(clean_registry):
    with pytest.raises(ValueError, match="unknown input_kind"):
        R.register("t_bad", input_kind="spectrogram", family="x")
    with pytest.raises(ValueError, match="unknown task"):
        R.register("t_bad", input_kind="window_stats", family="x", task="forecast")
    with pytest.raises(ValueError, match="must subclass"):
        R.register("t_bad2", input_kind="window_stats", family="x")(dict)
    _make_detector("t_dup", clean_registry)
    with pytest.raises(ValueError, match="already registered"):
        _make_detector("t_dup", clean_registry)


def test_build_unknown_model_lists_alternatives():
    with pytest.raises(ValueError, match="is not registered"):
        R.build("definitely_not_a_model")


def test_fit_rejects_empty_and_1d_input(clean_registry):
    _make_detector("t_shape", clean_registry)
    m = R.build("t_shape")
    with pytest.raises(ValueError, match="must be 2-D"):
        m.fit(np.zeros(10))
    with pytest.raises(ValueError, match="zero rows"):
        m.fit(np.zeros((0, 3)))
