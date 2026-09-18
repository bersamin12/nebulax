"""Smoke tests for nebulax/models/classical.py.

Every model registered there is fit-and-scored on fake data of the input_kind/shape its
``@register`` row promises, asserting finite scores of the right length - the minimum bar
for "this model runs end to end". ``matrix_profile_discord`` gets its own, much smaller
smoke test: its stumpy self-join is O((n*L)^2) per channel, so the shared 1k-row harness
used for the other nine models would blow well past the file's 60s budget.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from nebulax.bench import registry as R
from nebulax.models import classical as C  # noqa: F401  (import triggers @register)

#: Registered name -> (input_kind, family), taken verbatim from configs/model_ladder.yaml
#: (see nebulax/models/classical.py's per-class docstrings for the full row detail).
LADDER_ROWS = {
    "isolation_forest": ("window_stats", "ensemble_outlier"),
    "ecod": ("window_stats", "ecdf_outlier"),
    "copod": ("window_stats", "copula_outlier"),
    "knn_outlier": ("window_stats", "distance_outlier"),
    "cblof": ("window_stats", "cluster_outlier"),
    "kmeans_ad": ("raw_window", "cluster_outlier"),
    "ocsvm": ("window_stats", "one_class"),
    "ocsvm_mfcc_ams": ("raw_window", "one_class"),
    "halfspace_trees_ocknn": ("window_stats", "streaming_outlier"),
    "matrix_profile_discord": ("raw_window", "matrix_profile"),
}

#: Models whose smoke test needs the generic 1k-row harness excluded (handled separately).
_SPECIAL = {"matrix_profile_discord"}


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260916)


def _fake_window_stats(rng: np.random.Generator, n: int = 1000, f: int = 8) -> np.ndarray:
    """A majority/minority two-blob mixture (80/20, well separated): CBLOF's
    ``_set_small_large_clusters`` heuristic needs a genuine large/small cluster split, not
    pure iid Gaussian, or pyod raises ``ValueError("Could not form valid cluster
    separation ...")`` at fit time; the other window_stats models are indifferent to this
    and work fine on it too."""
    n_minor = max(1, int(n * 0.2))
    n_major = n - n_minor
    major = rng.normal(0.0, 1.0, size=(n_major, f))
    minor = rng.normal(6.0, 1.0, size=(n_minor, f))
    X = np.vstack([major, minor]).astype(np.float32)
    rng.shuffle(X)
    return X


def _fake_raw_window(rng: np.random.Generator, n: int = 1000, length: int = 32, c: int = 3) -> np.ndarray:
    return rng.normal(size=(n, length, c)).astype(np.float32)


def _fake_t(n: int) -> np.ndarray:
    return pd.date_range("2026-01-01", periods=n, freq="10s").values


def _fake_X(name: str, rng: np.random.Generator, n: int = 1000) -> np.ndarray:
    input_kind, _family = LADDER_ROWS[name]
    return _fake_raw_window(rng, n=n) if input_kind == "raw_window" else _fake_window_stats(rng, n=n)


def test_every_ladder_row_is_registered_with_the_right_input_kind_family_task() -> None:
    for name, (input_kind, family) in LADDER_ROWS.items():
        assert R.is_registered(name), f"{name} is not registered"
        spec = R.get_spec(name)
        assert spec.input_kind == input_kind, f"{name}: input_kind {spec.input_kind!r} != {input_kind!r}"
        assert spec.family == family, f"{name}: family {spec.family!r} != {family!r}"
        assert spec.task == "ad", f"{name}: task {spec.task!r} != 'ad'"


@pytest.mark.parametrize("name", sorted(set(LADDER_ROWS) - _SPECIAL))
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

    # also exercise the t=None path - none of these models require timestamps
    model2 = R.build(name)
    model2.fit(X)
    scores2 = model2.score(X)
    assert scores2.shape == (len(X),)
    assert np.isfinite(scores2).all()


def test_matrix_profile_discord_smoke(rng: np.random.Generator) -> None:
    """Small train/test batches (distinct arrays, unlike the transductive check above) so
    the AB-join's cost stays trivial; stumpy's first call also pays a one-off numba
    JIT-compile tax (roughly ten seconds) that every other call in this file reuses."""
    n, length, c = 120, 16, 2
    X_train = _fake_raw_window(rng, n=n, length=length, c=c)
    X_test = _fake_raw_window(rng, n=n, length=length, c=c)

    model = R.build("matrix_profile_discord", m=6)
    t0 = time.perf_counter()
    model.fit(X_train)
    fit_s = time.perf_counter() - t0
    assert model.is_fitted

    t0 = time.perf_counter()
    scores = model.score(X_test)
    score_s = time.perf_counter() - t0
    assert scores.shape == (n,)
    assert scores.dtype == np.float64
    assert np.isfinite(scores).all()
    assert score_s < 45.0, f"matrix_profile_discord: smoke score took {score_s:.2f}s"
    assert fit_s < 5.0

    # a wrong channel count is rejected rather than silently mis-scored
    with pytest.raises(ValueError, match="fitted on"):
        model.score(_fake_raw_window(rng, n=5, length=length, c=c + 1))


def test_matrix_profile_discord_scoring_peers_never_suppress_each_other(rng: np.random.Generator) -> None:
    """must_fix (verifier check #6): the old `_score` concatenated training with the whole
    scoring batch and self-joined everything, so a repeated scored anomaly could become its
    own nearest neighbour and be suppressed toward zero. The fixed version is a strict
    AB-join against `train_series_` only, so three identical injected copies inside the same
    scoring batch must score identically (never 0) and must rank at the very top of the
    batch (must_fix / verifier check #3: a conspicuous injected outlier must land in the
    top 5%)."""
    n_train, n_test, length, c = 60, 40, 12, 2
    X_train = _fake_raw_window(rng, n=n_train, length=length, c=c)
    X_test = _fake_raw_window(rng, n=n_test, length=length, c=c)
    outlier = (_fake_raw_window(rng, n=1, length=length, c=c)[0] + 40.0).astype(np.float32)
    injected = (3, 17, 33)
    for i in injected:
        X_test[i] = outlier

    model = R.build("matrix_profile_discord", m=6, k=5)
    model.fit(X_train)
    scores = model.score(X_test)

    injected_scores = scores[list(injected)]
    assert np.isfinite(injected_scores).all()
    assert injected_scores.min() > 0.0
    # identical injected windows must score (near-)identically - none may suppress another
    assert np.ptp(injected_scores) < 1e-6 * max(1.0, injected_scores.max())

    order = np.argsort(-scores)
    top_n = max(len(injected), round(0.05 * len(scores)))
    top5pct = set(order[:top_n])
    assert set(injected).issubset(top5pct), (
        f"injected outliers must rank in the top 5%; ranks were "
        f"{[int(np.where(order == i)[0][0]) for i in injected]} of {len(scores)}"
    )


def test_matrix_profile_discord_k_d_and_post_ma_are_consumed(rng: np.random.Generator) -> None:
    """must_fix (verifier check #5): `k`/`d` were stored in `self.params` but never used by
    the score. Now `k` drives exclusion-zone-aware kNN retrieval, `d` (or its `agg`
    fallback) drives the pre-sorted multidimensional aggregation, and `post_ma` drives a
    causal moving-average post-process - vary each independently and check the resulting
    scores actually change."""
    n_train, n_test, length, c = 50, 20, 10, 3
    X_train = _fake_raw_window(rng, n=n_train, length=length, c=c)
    X_test = _fake_raw_window(rng, n=n_test, length=length, c=c)
    X_test[5, :, 0] += 15.0  # single-channel anomaly

    base = R.build("matrix_profile_discord", m=5, k=1, d=None, agg="max", post_ma=1)
    base.fit(X_train)
    s_base = base.score(X_test)

    more_k = R.build("matrix_profile_discord", m=5, k=6, d=None, agg="max", post_ma=1)
    more_k.fit(X_train)
    s_more_k = more_k.score(X_test)
    assert not np.allclose(s_base, s_more_k), "k must influence the discord score"

    mean_agg = R.build("matrix_profile_discord", m=5, k=1, d=None, agg="mean", post_ma=1)
    mean_agg.fit(X_train)
    s_mean_agg = mean_agg.score(X_test)
    assert not np.allclose(s_base, s_mean_agg), "d/agg (multidimensional aggregation) must influence the score"

    # a single-channel anomaly stands out most clearly under max-style aggregation
    # (dim_count=1, the most-deviant channel alone), which is exactly agg="max" / d=1/c
    order = np.argsort(-s_base)
    assert order[0] == 5

    smoothed = R.build("matrix_profile_discord", m=5, k=1, d=None, agg="max", post_ma=4)
    smoothed.fit(X_train)
    s_smoothed = smoothed.score(X_test)
    assert not np.allclose(s_base, s_smoothed), "post_ma must influence the score"
    assert s_smoothed.max() <= s_base.max() + 1e-9  # a causal moving average cannot raise the peak


def test_ocsvm_subsamples_large_training_sets(rng: np.random.Generator) -> None:
    """ocsvm and ocsvm_mfcc_ams must cap training rows at max_train_samples (default
    20000) rather than fitting the SVM on the whole array; use a small cap here so the
    behaviour is checkable without an actually-large array."""
    X = _fake_window_stats(rng, n=500, f=4)
    model = R.build("ocsvm", max_train_samples=100)
    model.fit(X)
    assert model.model_.decision_scores_.shape[0] == 100

    Xr = _fake_raw_window(rng, n=200, length=16, c=1)
    model2 = R.build("ocsvm_mfcc_ams", max_train_samples=50)
    model2.fit(Xr)
    assert model2.model_.support_vectors_.shape[0] <= 50


def test_halfspace_trees_scores_streaming_outliers_higher(rng: np.random.Generator) -> None:
    """A point far outside the training range should score higher than an in-distribution
    point. Each side uses its own freshly-fit model scored once, rather than scoring a
    whole outlier batch through one model: this is a genuinely *online* detector (score
    then learn_one, even inside score()), so a multi-row outlier batch would adapt its own
    limits within a couple of rows and the signal would wash out - that is expected
    streaming behaviour, not something this single-shot check needs to exercise."""
    X_train = _fake_window_stats(rng, n=800, f=5)
    x_normal = rng.normal(0.0, 1.0, size=(1, 5)).astype(np.float32)
    x_outlier = rng.normal(25.0, 1.0, size=(1, 5)).astype(np.float32)

    model_a = R.build("halfspace_trees_ocknn", seed=0)
    model_a.fit(X_train)
    model_b = R.build("halfspace_trees_ocknn", seed=0)
    model_b.fit(X_train)

    s_normal = model_a.score(x_normal)
    s_outlier = model_b.score(x_outlier)
    assert np.isfinite(s_normal).all() and np.isfinite(s_outlier).all()
    assert s_outlier[0] > s_normal[0]


def test_halfspace_trees_ocknn_combines_hst_and_oneclass_knn(rng: np.random.Generator) -> None:
    """must_fix: `halfspace_trees_ocknn` implemented only River HalfSpaceTrees; the ladder
    (R99) requires the Half-Space Trees + One-Class kNN hybrid specifically. Check the kNN
    side is genuinely present (a sized recent-normal buffer, respecting `knn_window`) and
    genuinely contributes to the combined score (a point identical to a just-seen training
    row scores lower than one that is far from every buffered row in one feature, holding
    everything else about the two models - seed, training data - identical)."""
    assert C.HalfspaceTreesOCKNN.SEQUENTIAL is True

    X_train = _fake_window_stats(rng, n=400, f=4)

    model_default = R.build("halfspace_trees_ocknn", seed=0)
    model_default.fit(X_train)
    assert len(model_default._knn_buffer_) == 250  # default knn_window falls back to window_size

    model_small_buf = R.build("halfspace_trees_ocknn", seed=0, knn_window=50)
    model_small_buf.fit(X_train)
    assert len(model_small_buf._knn_buffer_) == 50

    recent_row = X_train[-1:].copy()
    far_row = recent_row.copy()
    far_row[0, -1] += 12.0  # push one feature well outside the recent buffer's spread

    model_a = R.build("halfspace_trees_ocknn", seed=0)
    model_a.fit(X_train)
    s_recent = model_a.score(recent_row)

    model_b = R.build("halfspace_trees_ocknn", seed=0)
    model_b.fit(X_train)
    s_far = model_b.score(far_row)

    assert np.isfinite(s_recent).all() and np.isfinite(s_far).all()
    assert s_far[0] > s_recent[0]


def test_kmeans_ad_scores_far_windows_higher(rng: np.random.Generator) -> None:
    X_train = _fake_raw_window(rng, n=500, length=16, c=2)
    X_test = np.concatenate(
        [
            _fake_raw_window(rng, n=20, length=16, c=2),
            _fake_raw_window(rng, n=5, length=16, c=2) + 20.0,
        ]
    )
    model = R.build("kmeans_ad", n_clusters=4)
    model.fit(X_train)
    scores = model.score(X_test)
    assert scores[20:].mean() > scores[:20].mean()
