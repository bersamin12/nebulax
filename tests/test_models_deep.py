"""Tests for ``nebulax.models.deep`` - the six torch rows of the ladder.

The default suite uses tiny architectures and short budgets (whole file well under a minute
on the GPU, a few minutes on a CPU-only box); the GPU timing test that the brief asks for
(2 000 fake windows, < 60 s per model) is gated on CUDA being present. Every test frees its
modules and empties the CUDA cache afterwards, because the A4500 is shared with other agents'
tests; the architectures here are a few tens of thousands of parameters, so the footprint is
tens of MB.
``--strict-markers`` is on and no custom markers are registered in ``pyproject.toml``, so the
GPU gate is a ``skipif`` (named ``_gpu`` below) rather than a ``gpu`` marker.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from nebulax.bench.registry import build, get_spec
from nebulax.models import deep as D  # noqa: F401  (import registers the six rows)

REPO = Path(__file__).resolve().parents[1]
LADDER = REPO / "configs/model_ladder.yaml"

AD_MODELS = ("lstm_ae", "tcn_ae", "usad", "sparse_autoencoder")
CLS_MODELS = ("resnet1d", "litetime")
ALL_MODELS = AD_MODELS + CLS_MODELS

_CUDA = torch.cuda.is_available()
_gpu = pytest.mark.skipif(not _CUDA, reason="no CUDA device (GPU test)")

#: Tiny-but-real configurations so the default suite stays fast.
FAST: dict[str, dict[str, Any]] = {
    "lstm_ae": dict(hidden_size=16, latent_size=8, max_seq_len=32, batch_size=128),
    "tcn_ae": dict(n_filters=8, n_levels=2, latent_channels=4, max_seq_len=32, batch_size=128),
    "usad": dict(hidden_sizes=(64,), latent_size=8, max_seq_len=16, batch_size=128),
    "sparse_autoencoder": dict(hidden_sizes=(8,), batch_size=256),
    "resnet1d": dict(filters=(8, 8), kernel_sizes=(5, 3), max_seq_len=32, batch_size=128),
    "litetime": dict(n_models=2, n_filters=8, max_seq_len=32, dwsc_kernel_size=8, batch_size=128),
}


def fast_model(name: str, **overrides: Any) -> Any:
    """Build ``name`` with a small architecture and a short budget (GPU when one is free)."""
    params = dict(FAST[name])
    params.update(dict(device=None, seed=0, max_minutes=0.5, max_epochs=4))
    params.update(overrides)
    return build(name, **params)


# --------------------------------------------------------------------------------------
# fake data helpers (the "helper in tests/" the brief asks for)
# --------------------------------------------------------------------------------------


def fake_raw_windows(
    n: int = 1000, length: int = 64, channels: int = 4, *, seed: int = 0, anomaly_frac: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """``((n, L, c) float32, (n,) bool)``: noisy sinusoids, the last ``anomaly_frac`` of which
    have a different amplitude and frequency."""
    rng = np.random.default_rng(seed)
    tt = np.linspace(0.0, 4.0 * np.pi, length, dtype=np.float32)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1, 1)).astype(np.float32)
    base = np.sin(tt[None, :, None] + phase) + 0.05 * rng.normal(size=(n, length, channels)).astype(np.float32)
    is_anom = np.zeros(n, dtype=bool)
    k = int(round(anomaly_frac * n))
    if k:
        is_anom[-k:] = True
        base[-k:] = 3.0 * np.sin(3.0 * tt[None, :, None] + phase[-k:]) + 0.05 * rng.normal(
            size=(k, length, channels)
        ).astype(np.float32)
    return base.astype(np.float32), is_anom


def fake_window_stats(n: int = 1000, n_features: int = 24, *, seed: int = 0, anomaly_frac: float = 0.0):
    """``((n, f) float32, (n,) bool)``: correlated Gaussian features with a shifted tail."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 3)).astype(np.float32)
    mix = rng.normal(size=(3, n_features)).astype(np.float32)
    X = latent @ mix + 0.1 * rng.normal(size=(n, n_features)).astype(np.float32)
    is_anom = np.zeros(n, dtype=bool)
    k = int(round(anomaly_frac * n))
    if k:
        is_anom[-k:] = True
        X[-k:] += 6.0 * rng.normal(size=(k, n_features)).astype(np.float32)
    return X.astype(np.float32), is_anom


def fake_labelled_windows(n: int = 900, length: int = 64, channels: int = 3, *, seed: int = 0):
    """``((n, L, c) float32, (n,) <U labels)`` with three well-separated classes."""
    rng = np.random.default_rng(seed)
    tt = np.linspace(0.0, 4.0 * np.pi, length, dtype=np.float32)
    per = n // 3
    names = np.array(["healthy", "friction", "leak"])
    X = np.empty((per * 3, length, channels), dtype=np.float32)
    y = np.empty(per * 3, dtype=names.dtype)
    for c, (freq, amp, offset) in enumerate(((1.0, 1.0, 0.0), (3.0, 1.0, 0.0), (1.0, 0.5, 2.0))):
        sl = slice(c * per, (c + 1) * per)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(per, 1, 1)).astype(np.float32)
        X[sl] = amp * np.sin(freq * tt[None, :, None] + phase) + offset
        X[sl] += 0.1 * rng.normal(size=(per, length, channels)).astype(np.float32)
        y[sl] = names[c]
    order = rng.permutation(per * 3)
    return X[order], y[order]


def data_for(name: str, **kw: Any):
    """The right fake input for ``name``'s registered ``input_kind``."""
    if get_spec(name).input_kind == "window_stats":
        return fake_window_stats(**kw)
    return fake_raw_windows(**kw)


def free(model: Any) -> None:
    if hasattr(model, "free"):
        model.free()
    if _CUDA:
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# registration matches configs/model_ladder.yaml
# --------------------------------------------------------------------------------------


def _ladder_rows() -> dict[str, dict[str, Any]]:
    doc = yaml.safe_load(LADDER.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = {}
    stack = [doc]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if isinstance(node.get("name"), str) and "input_kind" in node and "family" in node:
                rows.setdefault(node["name"], node)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return rows


@pytest.mark.parametrize("name", ALL_MODELS)
def test_registration_matches_ladder_row(name: str) -> None:
    row = _ladder_rows()[name]
    spec = get_spec(name)
    assert spec.name == name
    assert spec.input_kind == row["input_kind"]
    assert spec.family == row["family"]
    assert spec.task == row.get("task", "ad")
    # the class carries the same stamps, which is what the runner reads before instantiating
    assert (spec.cls.name, spec.cls.input_kind, spec.cls.family, spec.cls.task) == (
        name,
        row["input_kind"],
        row["family"],
        row.get("task", "ad"),
    )


def test_ladder_rows_agree_with_themselves() -> None:
    """Every duplicate of our six rows in the ladder declares the same family/input_kind."""
    doc = yaml.safe_load(LADDER.read_text(encoding="utf-8"))
    seen: dict[str, set[tuple[str, str, str]]] = {n: set() for n in ALL_MODELS}
    stack = [doc]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("name") in seen and "input_kind" in node:
                seen[node["name"]].add((node["input_kind"], node["family"], node.get("task", "ad")))
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    for name, variants in seen.items():
        assert len(variants) == 1, f"{name} appears with conflicting ladder metadata: {variants}"


# --------------------------------------------------------------------------------------
# smoke fits on 1k fake rows
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", AD_MODELS)
def test_ad_smoke_fit_and_score(name: str) -> None:
    X, _ = data_for(name, n=1000, seed=1)
    t = np.arange(len(X), dtype=np.float64) * 600.0
    model = fast_model(name)
    model.fit(X, t)
    assert model.is_fitted and model.fit_seconds > 0.0
    scores = model.score(X, t)
    assert scores.shape == (len(X),)
    assert scores.dtype == np.float64
    assert np.isfinite(scores).all()
    assert model.n_params > 0
    assert model.train_info_["stop_reason"] in {"max_epochs", "early_stopping", "budget"}
    assert model.get_params()["seed"] == 0
    free(model)


@pytest.mark.parametrize("name", CLS_MODELS)
def test_cls_smoke_fit_predict(name: str) -> None:
    X, y = fake_labelled_windows(n=900, seed=1)
    model = fast_model(name)
    model.fit(X, y)
    pred = model.predict(X)
    proba = model.predict_proba(X)
    assert pred.shape == (len(X),)
    assert set(np.unique(pred)) <= set(np.unique(y))
    assert proba.shape == (len(X), model.classes_.size)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert model.n_params > 0
    free(model)


@pytest.mark.parametrize("name", ALL_MODELS)
def test_scoring_before_fit_raises(name: str) -> None:
    model = fast_model(name)
    X, _ = data_for(name, n=32)
    with pytest.raises(RuntimeError):
        if get_spec(name).task == "cls":
            model.predict(X)
        else:
            model.score(X)


@pytest.mark.parametrize("name", ("lstm_ae", "tcn_ae", "usad"))
def test_channel_mismatch_raises(name: str) -> None:
    X, _ = fake_raw_windows(n=200, channels=4, seed=2)
    model = fast_model(name)
    model.fit(X, None)
    other, _ = fake_raw_windows(n=50, channels=5, seed=3)
    with pytest.raises(ValueError):
        model.score(other)
    free(model)


@pytest.mark.parametrize("name", ("lstm_ae", "tcn_ae", "usad"))
def test_scores_a_different_window_length(name: str) -> None:
    """Windows are pooled to the fitted length, so a shorter/longer slice still scores."""
    X, _ = fake_raw_windows(n=200, length=64, channels=3, seed=4)
    model = fast_model(name)
    model.fit(X, None)
    for length in (32, 128):
        other, _ = fake_raw_windows(n=40, length=length, channels=3, seed=5)
        s = model.score(other)
        assert s.shape == (40,) and np.isfinite(s).all()
    free(model)


def test_non_finite_input_is_survivable() -> None:
    X, _ = fake_window_stats(n=300, seed=6)
    X[3, 2] = np.nan
    X[7, 5] = np.inf
    model = fast_model("sparse_autoencoder")
    model.fit(X, None)
    assert np.isfinite(model.score(X)).all()
    free(model)


# --------------------------------------------------------------------------------------
# behaviour: the models actually learn something
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", AD_MODELS)
def test_reconstruction_error_separates_anomalies(name: str) -> None:
    """Fit on normal rows only (no labels are ever passed in), score a contaminated slice."""
    X_train, _ = data_for(name, n=600, seed=7)
    X_test, is_anom = data_for(name, n=400, seed=8, anomaly_frac=0.2)
    model = fast_model(name, max_epochs=12, max_minutes=0.5)
    model.fit(X_train, None)
    scores = model.score(X_test)
    assert scores[is_anom].mean() > scores[~is_anom].mean()
    free(model)


@pytest.mark.parametrize("name", CLS_MODELS)
def test_classifier_beats_chance(name: str) -> None:
    X, y = fake_labelled_windows(n=600, seed=9)
    model = fast_model(name, max_epochs=12, max_minutes=0.5)
    model.fit(X, y)
    acc = float((model.predict(X) == y).mean())
    assert acc > 0.6, f"{name} train accuracy {acc:.2f} is at chance"
    free(model)


# --------------------------------------------------------------------------------------
# budget, caps, determinism
# --------------------------------------------------------------------------------------


def test_budget_s_stops_training() -> None:
    X, _ = fake_raw_windows(n=1000, length=64, channels=4, seed=10)
    model = fast_model("tcn_ae", max_epochs=10_000, batch_size=16)
    t0 = time.perf_counter()
    model.fit(X, None, budget_s=3.0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 20.0
    assert model.train_info_["stop_reason"] == "budget"
    assert np.isfinite(model.score(X[:64])).all()
    free(model)


def test_fit_accepts_budget_s_like_the_runner_expects() -> None:
    """``runner._accepts(model, 'budget_s')`` must be true for every deep row, otherwise the
    per-run wall-clock cap is enforced by killing the worker instead of by early stopping."""
    import inspect

    for name in ALL_MODELS:
        model = fast_model(name)
        assert "budget_s" in inspect.signature(model._fit).parameters, name


def test_train_window_cap_is_applied() -> None:
    X, _ = fake_window_stats(n=800, seed=11)
    model = fast_model("sparse_autoencoder", max_windows=100, max_epochs=2)
    model.fit(X, None)
    assert model.score(X).shape == (800,)
    free(model)


def test_seed_is_deterministic() -> None:
    X, _ = fake_raw_windows(n=400, length=32, channels=3, seed=12)
    a = fast_model("lstm_ae", seed=3)
    b = fast_model("lstm_ae", seed=3)
    a.fit(X, None)
    b.fit(X, None)
    np.testing.assert_allclose(a.score(X), b.score(X), rtol=1e-5, atol=1e-6)
    free(a)
    free(b)


def test_usad_alpha_beta_change_the_score() -> None:
    X, _ = fake_raw_windows(n=300, length=32, channels=3, seed=13)
    model = fast_model("usad")
    model.fit(X, None)
    model.alpha, model.beta = 1.0, 0.0
    s_rec = model.score(X)
    model.alpha, model.beta = 0.0, 1.0
    s_adv = model.score(X)
    assert not np.allclose(s_rec, s_adv)
    free(model)


def test_sparse_autoencoder_smoothing() -> None:
    """The ladder note asks for a smoothed error signal; ``smooth_window`` does it in t-order."""
    X, _ = fake_window_stats(n=300, seed=14, anomaly_frac=0.1)
    t = np.arange(len(X), dtype=np.float64) * 600.0
    rough = fast_model("sparse_autoencoder", smooth_window=1)
    smooth = fast_model("sparse_autoencoder", smooth_window=9)
    rough.fit(X, t)
    smooth.fit(X, t)
    s_rough, s_smooth = rough.score(X, t), smooth.score(X, t)
    assert np.std(np.diff(s_smooth)) < np.std(np.diff(s_rough))
    # order-invariance: shuffling the rows and their timestamps together shuffles the scores
    perm = np.random.default_rng(0).permutation(len(X))
    np.testing.assert_allclose(smooth.score(X[perm], t[perm]), s_smooth[perm], rtol=1e-6, atol=1e-8)
    free(rough)
    free(smooth)


def test_sparse_autoencoder_kl_penalty_sparsifies() -> None:
    X, _ = fake_window_stats(n=400, n_features=16, seed=15)
    dense = fast_model("sparse_autoencoder", hidden_sizes=(16,), sparsity_weight=0.0, max_epochs=8)
    sparse = fast_model("sparse_autoencoder", hidden_sizes=(16,), sparsity_weight=1.0, max_epochs=8)
    dense.fit(X, None)
    sparse.fit(X, None)
    with torch.no_grad():
        a_dense = float(dense.net_(dense._prep_flat(X, fit=False).to(dense.device))[1].mean())
        a_sparse = float(sparse.net_(sparse._prep_flat(X, fit=False).to(sparse.device))[1].mean())
    assert a_sparse < a_dense, f"mean hidden activation {a_sparse:.3f} (KL) vs {a_dense:.3f} (plain)"
    free(dense)
    free(sparse)


def test_litetime_is_an_ensemble() -> None:
    X, y = fake_labelled_windows(n=300, seed=16)
    model = fast_model("litetime", n_models=3)
    model.fit(X, y)
    assert len(model.nets_) == 3
    assert model.train_info_["n_members"] == 3
    free(model)


# --------------------------------------------------------------------------------------
# GPU: 2 000 fake windows per model in well under a minute
# --------------------------------------------------------------------------------------


@_gpu
@pytest.mark.parametrize("name", ALL_MODELS)
def test_gpu_trains_2k_windows_under_60s(name: str) -> None:
    if get_spec(name).input_kind == "window_stats":
        X, _ = fake_window_stats(n=2000, n_features=32, seed=17)
    else:
        X, _ = fake_raw_windows(n=2000, length=128, channels=6, seed=17)
    params = dict(FAST[name])
    params.pop("max_seq_len", None)
    model = build(name, device="cuda", seed=0, max_minutes=0.8, max_epochs=10, **params)
    t0 = time.perf_counter()
    if get_spec(name).task == "cls":
        y = np.array(["healthy", "friction", "leak"] * (len(X) // 3 + 1))[: len(X)]
        model.fit(X, y)
        out = model.predict_proba(X)
    else:
        model.fit(X, np.arange(len(X), dtype=np.float64))
        out = model.score(X)
    elapsed = time.perf_counter() - t0
    assert np.isfinite(out).all()
    assert elapsed < 60.0, f"{name}: {elapsed:.1f}s on GPU"
    assert torch.cuda.max_memory_allocated() < 4e9
    free(model)
    torch.cuda.reset_peak_memory_stats()
