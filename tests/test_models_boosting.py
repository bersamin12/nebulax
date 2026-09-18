"""Smoke tests for nebulax/models/boosting.py: the nine boosting-tier ladder rows
(lgbm_residual, lgbm_three_regime, lgbm_envelope, shallow_rule_learner,
stacking_rf_xgb_logreg, random_forest_cycle, logreg_cycle, logreg_envelope,
conformal_threshold). Each model is smoke-fit on ~1k fake rows/windows of its declared
input_kind's shape and must produce finite scores/predictions of the right length. Registry
metadata (name/input_kind/family/task) is checked against configs/model_ladder.yaml.
"""

from __future__ import annotations

import numpy as np
import pytest

# Importing the module (which imports the nebulax.models package) runs every
# @register decorator so build()/get_spec() below see all nine rows.
import nebulax.models.boosting as boosting  # noqa: F401
from nebulax.bench.base import AnomalyDetector
from nebulax.bench.registry import build, get_spec

N = 1000

# name -> (input_kind, family, task), verbatim from configs/model_ladder.yaml.
EXPECTED_SPEC = {
    "lgbm_residual": ("raw_window", "residual_model", "ad"),
    "lgbm_three_regime": ("cycle_features", "gradient_boosting", "cls"),
    "lgbm_envelope": ("window_stats", "gradient_boosting", "cls"),
    "shallow_rule_learner": ("cycle_features", "interpretable_rule", "ad"),
    "stacking_rf_xgb_logreg": ("cycle_features", "stacking", "cls"),
    "random_forest_cycle": ("cycle_features", "tree_ensemble", "cls"),
    "logreg_cycle": ("cycle_features", "linear", "cls"),
    "logreg_envelope": ("window_stats", "linear", "cls"),
    "conformal_threshold": ("window_stats", "calibration", "ad"),
}


class _ZScoreAD(AnomalyDetector):
    """Trivial, unregistered AnomalyDetector: max |z-score| across features. Used only as a
    self-contained ``inner`` for testing ``conformal_threshold`` without depending on any
    other agent's registered AD model (e.g. ``isolation_forest``) being importable yet."""

    def _fit(self, X: np.ndarray, t=None, **kwargs) -> None:
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0) + 1e-9

    def _score(self, X: np.ndarray, t=None) -> np.ndarray:
        return np.abs((X - self.mu_) / self.sd_).max(axis=1)


@pytest.fixture
def window_stats(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(N, 12)).astype(np.float32)


@pytest.fixture
def raw_window(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(N, 20, 4)).astype(np.float32)


@pytest.fixture
def cycle_features(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(N, 15)).astype(np.float32)


@pytest.fixture
def y3(rng: np.random.Generator) -> np.ndarray:
    """Three balanced-ish classes, correlated with feature 0 so fitting is not degenerate."""
    return rng.integers(0, 3, size=N).astype(np.int64)


# --------------------------------------------------------------------------------------
# Registry metadata
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(EXPECTED_SPEC))
def test_registered_matches_ladder(name: str) -> None:
    spec = get_spec(name)
    input_kind, family, task = EXPECTED_SPEC[name]
    assert spec.input_kind == input_kind
    assert spec.family == family
    assert spec.task == task


# --------------------------------------------------------------------------------------
# AD models (never see labels)
# --------------------------------------------------------------------------------------


def test_lgbm_residual_smoke(raw_window: np.ndarray) -> None:
    model = build("lgbm_residual", n_estimators=20, max_depth=3, n_jobs=1)
    model.fit(raw_window)
    assert model.fit_seconds >= 0.0
    scores = model.score(raw_window)
    assert scores.shape == (N,)
    assert np.isfinite(scores).all()


def test_shallow_rule_learner_smoke(cycle_features: np.ndarray) -> None:
    model = build("shallow_rule_learner", max_depth=2, tail_frac=0.05)
    model.fit(cycle_features)
    scores = model.score(cycle_features)
    assert scores.shape == (N,)
    assert np.isfinite(scores).all()
    assert (scores >= 0.0).all()  # -log of a fraction in (0, 1]
    assert len(model.rules_) >= 1
    assert all(line.startswith("IF ") for line in model.rules_)


def test_conformal_threshold_smoke(window_stats: np.ndarray) -> None:
    model = build("conformal_threshold", inner=_ZScoreAD(), calibration_frac=0.2)
    model.fit(window_stats)
    assert model.n_calibration_ == pytest.approx(N * 0.2, abs=1)
    scores = model.score(window_stats)
    assert scores.shape == (N,)
    assert np.isfinite(scores).all()
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_conformal_threshold_rejects_non_ad_inner(window_stats: np.ndarray) -> None:
    from nebulax.bench.base import Classifier

    class _NotAD(Classifier):
        def _fit(self, X, y, **kw):
            pass

        def _predict(self, X):
            return np.zeros(len(X))

    model = build("conformal_threshold", inner=_NotAD())
    with pytest.raises(TypeError):
        model.fit(window_stats)


# --------------------------------------------------------------------------------------
# CLS models
# --------------------------------------------------------------------------------------


def _check_classifier(name: str, X: np.ndarray, y: np.ndarray, **params) -> None:
    model = build(name, **params)
    model.fit(X, y)
    assert model.fit_seconds >= 0.0
    pred = model.predict(X)
    assert pred.shape == (N,)
    assert set(np.unique(pred)) <= set(np.unique(y))
    proba = model.predict_proba(X)
    assert proba.shape == (N, len(np.unique(y)))
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_lgbm_three_regime_smoke(cycle_features: np.ndarray, y3: np.ndarray) -> None:
    _check_classifier(
        "lgbm_three_regime", cycle_features, y3, n_estimators=20, max_depth=3, n_jobs=1
    )


# Real synthetic-door cycle_features names (from
# D.load_bench('sim', subsystem='door', input_kind='cycle_features', ...).feature_names) that
# mark a regime by name, plus a few context columns that do not.
_DOOR_LIKE_REGIME_NAMES = [
    "closing_time", "opening_time", "i_peak", "i_mean_cruise", "i_rms_cruise", "i_end",
    "i_start_peak", "energy_J", "pwm_mean", "pos_open_max", "pos_close_max",
]


def test_lgbm_three_regime_groups_by_feature_name_when_names_mark_a_regime(
    cycle_features: np.ndarray, y3: np.ndarray
) -> None:
    f = cycle_features.shape[1]
    names = _DOOR_LIKE_REGIME_NAMES + [f"extra_ctx_{i}" for i in range(f - len(_DOOR_LIKE_REGIME_NAMES))]
    assert len(names) == f

    model = build("lgbm_three_regime", n_estimators=20, max_depth=3, n_jobs=1)
    model.fit(cycle_features, y3, feature_names=names)
    assert model.regime_source_ == "feature_names"
    opening, cruise, closing = model.regime_indices_
    assert names.index("opening_time") in opening.tolist()
    assert names.index("pos_open_max") in opening.tolist()
    assert names.index("i_mean_cruise") in cruise.tolist()
    assert names.index("i_rms_cruise") in cruise.tolist()
    assert names.index("closing_time") in closing.tolist()
    assert names.index("pos_close_max") in closing.tolist()
    # an unmatched/context column (no open/cruise/close marker) is shared by every regime
    ctx_idx = names.index("extra_ctx_0")
    assert ctx_idx in opening.tolist() and ctx_idx in cruise.tolist() and ctx_idx in closing.tolist()

    pred = model.predict(cycle_features)
    assert pred.shape == (N,)


def test_lgbm_three_regime_falls_back_to_equal_slices_without_regime_markers(
    cycle_features: np.ndarray, y3: np.ndarray
) -> None:
    f = cycle_features.shape[1]
    # Cranfield-cycle_features-style names: stroke-level mean/std/min/max, no open/cruise/close markers.
    names = [f"i_mean_{stat}" for stat in ("mean", "std", "min", "max")] + [f"stat_{i}" for i in range(f - 4)]

    named = build("lgbm_three_regime", n_estimators=20, max_depth=3, n_jobs=1)
    named.fit(cycle_features, y3, feature_names=names)
    assert named.regime_source_ == "equal_slices"
    assert sum(idx.size for idx in named.regime_indices_) == f

    unnamed = build("lgbm_three_regime", n_estimators=20, max_depth=3, n_jobs=1)
    unnamed.fit(cycle_features, y3)  # no feature_names context at all -> same fallback as before
    assert unnamed.regime_source_ == "equal_slices"
    for got, expect in zip(unnamed.regime_indices_, named.regime_indices_):
        np.testing.assert_array_equal(got, expect)


def test_lgbm_three_regime_explicit_regime_slices_override_feature_names(
    cycle_features: np.ndarray, y3: np.ndarray
) -> None:
    f = cycle_features.shape[1]
    model = build(
        "lgbm_three_regime",
        regime_slices=[(0, 5), (5, 10), (10, f)],
        n_estimators=20,
        max_depth=3,
        n_jobs=1,
    )
    # every name says "open" - if the override did not win, this would be all-opening.
    model.fit(cycle_features, y3, feature_names=[f"open_{i}" for i in range(f)])
    assert model.regime_source_ == "regime_slices"
    np.testing.assert_array_equal(model.regime_indices_[0], np.arange(0, 5))
    np.testing.assert_array_equal(model.regime_indices_[1], np.arange(5, 10))
    np.testing.assert_array_equal(model.regime_indices_[2], np.arange(10, f))


def test_lgbm_envelope_smoke(window_stats: np.ndarray, y3: np.ndarray) -> None:
    _check_classifier(
        "lgbm_envelope", window_stats, y3, n_estimators=30, num_leaves=7, n_jobs=1
    )


def test_stacking_rf_xgb_logreg_smoke(cycle_features: np.ndarray, y3: np.ndarray) -> None:
    _check_classifier(
        "stacking_rf_xgb_logreg",
        cycle_features,
        y3,
        rf_n_estimators=20,
        xgb_n_estimators=20,
        stack_cv=3,
        n_jobs=1,  # avoid thread-pool oversubscription overhead in a shared-CPU smoke test
    )


def test_random_forest_cycle_smoke(cycle_features: np.ndarray, y3: np.ndarray) -> None:
    _check_classifier("random_forest_cycle", cycle_features, y3, n_estimators=30, n_jobs=1)


def test_logreg_cycle_smoke(cycle_features: np.ndarray, y3: np.ndarray) -> None:
    _check_classifier("logreg_cycle", cycle_features, y3)


def test_logreg_envelope_smoke(window_stats: np.ndarray, y3: np.ndarray) -> None:
    _check_classifier("logreg_envelope", window_stats, y3)


# --------------------------------------------------------------------------------------
# AD contract: fit() structurally cannot see labels
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["lgbm_residual", "shallow_rule_learner", "conformal_threshold"])
def test_ad_models_fit_signature_has_no_label_argument(name: str) -> None:
    import inspect

    sig = inspect.signature(AnomalyDetector.fit)
    assert "y" not in sig.parameters
    assert get_spec(name).task == "ad"
