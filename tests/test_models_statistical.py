"""Smoke tests for nebulax/models/statistical.py.

Every model registered there is fit-and-scored on tiny fake data of the input_kind it
declares (1k rows/windows, the shape its @register row promises), asserting finite scores
of the right length - the minimum bar for "this model runs end to end". A few extra tests
pin down the name-resolution and aggregation behaviour of the two "control" models that take
a feature/channel name as a constructor kwarg.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from nebulax.bench import registry as R
from nebulax.models import statistical as S  # noqa: F401  (import triggers @register)

#: Registered name -> input_kind, taken verbatim from configs/model_ladder.yaml (see the
#: table in nebulax/models/statistical.py's module docstring).
LADDER_ROWS = {
    "sensor_range_baseline": "window_stats",
    "trivial_baseline_set": "window_stats",
    "l2_norm_channels": "raw_window",
    "nn1_distance": "raw_window",
    "random_score": "window_stats",
    "single_feature_threshold": "window_stats",
    "per_channel_or_aggregate": "window_stats",
    "moving_window_variance": "raw_window",
    "squared_difference": "raw_window",
    "robust_z_ewma_cycle": "cycle_features",
    "robust_z_ewma_window": "window_stats",
    "cusum_cycle_scalar": "cycle_features",
    "pca_spe_t2": "window_stats",
    "mahalanobis_mincovdet": "window_stats",
}


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260916)


def _fake_2d(rng: np.random.Generator, n: int = 1000, f: int = 6) -> np.ndarray:
    """1k fake window_stats/cycle_features rows: correlated-but-full-rank so MinCovDet/PCA
    do not hit a singular covariance."""
    base = rng.normal(size=(n, f))
    mix = rng.normal(size=(f, f)) * 0.3 + np.eye(f)
    return (base @ mix).astype(np.float32)


def _fake_3d(rng: np.random.Generator, n: int = 1000, length: int = 32, c: int = 3) -> np.ndarray:
    return rng.normal(size=(n, length, c)).astype(np.float32)


def _fake_t(n: int) -> np.ndarray:
    return pd.date_range("2026-01-01", periods=n, freq="10s").values


def _fake_X(name: str, rng: np.random.Generator, n: int = 1000) -> np.ndarray:
    kind = LADDER_ROWS[name]
    return _fake_3d(rng, n=n) if kind == "raw_window" else _fake_2d(rng, n=n)


def test_every_ladder_row_is_registered_with_the_right_input_kind_family_task() -> None:
    for name, input_kind in LADDER_ROWS.items():
        assert R.is_registered(name), f"{name} is not registered"
        spec = R.get_spec(name)
        assert spec.input_kind == input_kind, f"{name}: input_kind {spec.input_kind!r} != {input_kind!r}"
        assert spec.task == "ad", f"{name}: task {spec.task!r} != 'ad'"


@pytest.mark.parametrize("name", sorted(LADDER_ROWS))
def test_smoke_fit_and_score(name: str, rng: np.random.Generator) -> None:
    X = _fake_X(name, rng, n=1000)
    t = _fake_t(len(X))

    model = R.build(name)
    t0 = time.perf_counter()
    model.fit(X, t)
    fit_s = time.perf_counter() - t0
    assert model.is_fitted
    assert fit_s < 30.0, f"{name}: smoke fit took {fit_s:.2f}s, expected well under the 60s test budget"

    scores = model.score(X, t)
    assert scores.shape == (len(X),)
    assert scores.dtype == np.float64
    assert np.isfinite(scores).all()

    # also exercise the t=None path (row-order fallback for the sequential models)
    model2 = R.build(name)
    model2.fit(X)
    scores2 = model2.score(X)
    assert scores2.shape == (len(X),)
    assert np.isfinite(scores2).all()


def test_single_feature_threshold_resolves_name_and_direction(rng: np.random.Generator) -> None:
    X = _fake_2d(rng, n=200, f=4)
    names = ["a", "b", "c", "d"]

    above = R.build("single_feature_threshold", feature="c", feature_names=names)
    above.fit(X)
    np.testing.assert_allclose(above.score(X), X[:, 2].astype(np.float64))

    below = R.build("single_feature_threshold", feature="c", feature_names=names, direction="below")
    below.fit(X)
    np.testing.assert_allclose(below.score(X), -X[:, 2].astype(np.float64))

    with pytest.raises(ValueError, match="feature_names"):
        R.build("single_feature_threshold", feature="c").fit(X)


def test_per_channel_or_aggregate_default_uses_all_columns(rng: np.random.Generator) -> None:
    X = _fake_2d(rng, n=300, f=5)
    model = R.build("per_channel_or_aggregate")
    model.fit(X)
    assert model.idx_.tolist() == list(range(5))

    subset = R.build("per_channel_or_aggregate", features=["y", "z"], feature_names=["w", "x", "y", "z"])
    subset.fit(_fake_2d(rng, n=300, f=4))
    assert subset.idx_.tolist() == [2, 3]


def test_random_score_ignores_input_and_is_stochastic(rng: np.random.Generator) -> None:
    X = np.zeros((50, 3), dtype=np.float32)
    model = R.build("random_score", seed=1)
    model.fit(X)
    s1 = model.score(X)
    s2 = model.score(X)
    assert s1.shape == (50,) and s2.shape == (50,)
    assert s1.std() > 0.0
    assert not np.allclose(s1, s2)  # fresh draw each score() call, per the docstring


def test_trivial_baseline_set_rejects_unknown_component() -> None:
    with pytest.raises(ValueError, match="unknown components"):
        R.build("trivial_baseline_set", components=("nonsense",))


def test_sensor_range_baseline_flags_out_of_range_rows(rng: np.random.Generator) -> None:
    X_train = _fake_2d(rng, n=500, f=3)
    model = R.build("sensor_range_baseline")
    model.fit(X_train)
    X_test = X_train[:10].copy()
    X_test[0, 0] = X_train[:, 0].max() + 50.0  # blow one feature far outside the training range
    scores = model.score(X_test)
    assert scores[0] == scores.max()


def test_cusum_cycle_scalar_reacts_to_a_sustained_shift(rng: np.random.Generator) -> None:
    X_train = _fake_2d(rng, n=500, f=3)
    model = R.build("cusum_cycle_scalar", k=0.2)
    model.fit(X_train)
    n = 100
    X_test = rng.normal(size=(n, 3)).astype(np.float32)
    X_test[50:] += 5.0  # sustained shift halfway through
    t = _fake_t(n)
    scores = model.score(X_test, t)
    assert scores[-1] > scores[10]


def test_mahalanobis_and_pca_fit_on_training_slice_only(rng: np.random.Generator) -> None:
    X_train = _fake_2d(rng, n=500, f=5)
    for name in ("mahalanobis_mincovdet", "pca_spe_t2"):
        model = R.build(name)
        model.fit(X_train)
        X_test = _fake_2d(rng, n=50, f=5) * 10.0  # far-out-of-distribution test rows
        scores = model.score(X_test)
        assert np.isfinite(scores).all()
        assert scores.max() > 0.0


def test_ewma_and_cusum_models_are_sequential_moving_window_variance_is_not() -> None:
    """robust_z_ewma_* and cusum_cycle_scalar carry a running statistic across rows in time
    order, so the runner must score them one series at a time (SEQUENTIAL=True);
    moving_window_variance's score depends only on its own row's within-window variance and a
    fit-time scale, so it stays SEQUENTIAL=False."""
    assert R.build("robust_z_ewma_cycle").SEQUENTIAL is True
    assert R.build("robust_z_ewma_window").SEQUENTIAL is True
    assert R.build("cusum_cycle_scalar").SEQUENTIAL is True
    assert R.build("moving_window_variance").SEQUENTIAL is False


def test_single_feature_and_per_channel_resolve_feature_names_from_fit_context(rng: np.random.Generator) -> None:
    """The runner passes feature_names into _fit (never the constructor) when a model
    declares the keyword - see base.CONTEXT_KEYS / runner._context. Both control models must
    resolve a string feature/features from that as well as from the constructor kwarg, with
    the constructor kwarg taking precedence, and store the resolved index as feature_index_."""
    X = _fake_2d(rng, n=200, f=4)
    names = ["a", "b", "c", "d"]

    single = R.build("single_feature_threshold", feature="c")
    single.fit(X, feature_names=names)  # runner-style context, not a constructor kwarg
    assert single.feature_index_ == 2
    np.testing.assert_allclose(single.score(X), X[:, 2].astype(np.float64))

    per_channel = R.build("per_channel_or_aggregate", features=["b", "d"])
    per_channel.fit(X, feature_names=names)
    assert per_channel.idx_.tolist() == [1, 3]
    assert per_channel.feature_index_.tolist() == [1, 3]  # feature_index_ alias, see docstring

    # explicit constructor feature_names still wins when both are given
    override = R.build("single_feature_threshold", feature="c", feature_names=["w", "x", "c", "z"])
    override.fit(X, feature_names=["p", "q", "r", "c"])
    assert override.feature_index_ == 2  # resolved against the constructor's names, not fit's

    # neither given -> the original "feature_names is a name but not provided" error
    with pytest.raises(ValueError, match="feature_names"):
        R.build("single_feature_threshold", feature="c").fit(X)


def test_trivial_baseline_set_keeps_params_verbatim_and_rejects_empty():
    import pytest
    from nebulax.bench.registry import build

    m = build("trivial_baseline_set", components=["range_frac", "constant"])
    assert m.params["components"] == ["range_frac", "constant"]
    with pytest.raises(ValueError, match="at least one"):
        build("trivial_baseline_set", components=())
