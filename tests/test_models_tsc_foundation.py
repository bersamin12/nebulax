"""Smoke tests for nebulax/models/tsc.py and nebulax/models/foundation.py: the four
``raw_window`` ladder rows multirocket_ridge, multirocket_hydra, quant (task="cls", aeon) and
tspulse_zeroshot (task="ad", zero-shot IBM Granite TSPulse-r1). Each model is smoke-fit on
~1k fake raw windows and must produce finite predictions/scores of the right length. Registry
metadata (name/input_kind/family/task) is checked against configs/model_ladder.yaml.

Note on the test module's filename: the brief that requested this file named it
``tests/test_models_tsc-foundation.py``, which is not a valid Python module name (a hyphen is
not a legal identifier character, so pytest cannot import it); this file is
``tests/test_models_tsc_foundation.py`` (underscore) instead.

TSPulse loads real pretrained weights from ``~/.cache/huggingface`` with
``local_files_only=True`` - never a network call - and its smoke test is skipped outright if
the weights are not already cached, per this task's hard rule.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

# Importing the modules (which import the nebulax.models package) runs every
# @register decorator so build()/get_spec() below see all four rows.
import nebulax.models.foundation as foundation  # noqa: F401
import nebulax.models.tsc as tsc  # noqa: F401
from nebulax.bench.base import AnomalyDetector
from nebulax.bench.registry import build, get_spec

N = 1000
_CUDA = torch.cuda.is_available()

try:
    from huggingface_hub import try_to_load_from_cache

    _TSPULSE_CACHED = (
        try_to_load_from_cache(foundation._MODEL_ID, "config.json") is not None
        and try_to_load_from_cache(foundation._MODEL_ID, "model.safetensors") is not None
    )
except Exception:  # pragma: no cover - defensive: never let a cache-probe crash collection
    _TSPULSE_CACHED = False

_needs_tspulse_weights = pytest.mark.skipif(
    not _TSPULSE_CACHED,
    reason=f"{foundation._MODEL_ID} not found in the local huggingface cache (no network calls in tests)",
)

# name -> (input_kind, family, task), verbatim from configs/model_ladder.yaml.
EXPECTED_SPEC = {
    "multirocket_ridge": ("raw_window", "convolution_kernel", "cls"),
    "multirocket_hydra": ("raw_window", "convolution_kernel", "cls"),
    "quant": ("raw_window", "interval", "cls"),
    "tspulse_zeroshot": ("raw_window", "tsfm_ad", "ad"),
}


@pytest.fixture
def raw_window(rng: np.random.Generator) -> np.ndarray:
    """~1k fake raw_window rows, shape (N, L, c): a low-frequency sine per channel plus noise,
    the same fixture shape convention as tests/test_models_boosting.py's raw_window."""
    L, c = 20, 4
    t = np.linspace(0.0, 4.0 * np.pi, L)
    base = np.stack([np.sin(t + phase) for phase in np.linspace(0.0, np.pi, c)], axis=-1)
    X = base[None, :, :] + 0.1 * rng.normal(size=(N, L, c))
    return X.astype(np.float32)


@pytest.fixture
def y2(rng: np.random.Generator, raw_window: np.ndarray) -> np.ndarray:
    """Two classes correlated with channel-0 energy, so fitting is not degenerate."""
    energy = raw_window[:, :, 0].mean(axis=1)
    return (energy > np.median(energy)).astype(np.int64)


# --------------------------------------------------------------------------------------
# Registry metadata: name/input_kind/family/task match configs/model_ladder.yaml
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(EXPECTED_SPEC))
def test_registered_matches_ladder(name: str) -> None:
    spec = get_spec(name)
    input_kind, family, task = EXPECTED_SPEC[name]
    assert spec.input_kind == input_kind
    assert spec.family == family
    assert spec.task == task


# --------------------------------------------------------------------------------------
# CLS models (aeon): multirocket_ridge, multirocket_hydra, quant
# --------------------------------------------------------------------------------------


def _check_classifier(name: str, X: np.ndarray, y: np.ndarray, **params) -> float:
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
    return model.fit_seconds


def test_multirocket_ridge_smoke(raw_window: np.ndarray, y2: np.ndarray) -> None:
    # n_kernels reduced from the class default (10_000) purely so this smoke test stays
    # well under the 60 s module budget; the default lives in the constructor/docstring.
    fit_seconds = _check_classifier("multirocket_ridge", raw_window, y2, n_kernels=84, random_state=0)
    assert fit_seconds < 30.0


def test_multirocket_hydra_smoke(raw_window: np.ndarray, y2: np.ndarray) -> None:
    fit_seconds = _check_classifier(
        "multirocket_hydra", raw_window, y2, n_kernels=8, n_groups=8, random_state=0
    )
    assert fit_seconds < 30.0


def test_quant_smoke(raw_window: np.ndarray, y2: np.ndarray) -> None:
    fit_seconds = _check_classifier("quant", raw_window, y2, interval_depth=3, random_state=0)
    assert fit_seconds < 30.0


def test_cls_models_use_aeon_collection_layout(raw_window: np.ndarray) -> None:
    """(n, L, c) raw_window -> aeon's (n, c, L); never reshaped in place."""
    out = tsc._to_aeon_collection(raw_window)
    assert out.shape == (raw_window.shape[0], raw_window.shape[2], raw_window.shape[1])
    np.testing.assert_allclose(out[3, 1, :], raw_window[3, :, 1])


# --------------------------------------------------------------------------------------
# AD model (zero-shot foundation model): tspulse_zeroshot
# --------------------------------------------------------------------------------------


def test_linear_resize_is_identity_at_target_length(raw_window: np.ndarray) -> None:
    out = foundation._linear_resize(raw_window, raw_window.shape[1])
    np.testing.assert_allclose(out, raw_window, atol=1e-6)


def test_linear_resize_shape_and_endpoints(raw_window: np.ndarray) -> None:
    out = foundation._linear_resize(raw_window, 512)
    assert out.shape == (raw_window.shape[0], 512, raw_window.shape[2])
    assert np.isfinite(out).all()
    # align_corners=True semantics: the first/last sample of every window is preserved.
    np.testing.assert_allclose(out[:, 0, :], raw_window[:, 0, :], atol=1e-5)
    np.testing.assert_allclose(out[:, -1, :], raw_window[:, -1, :], atol=1e-5)


@_needs_tspulse_weights
def test_tspulse_zeroshot_smoke(raw_window: np.ndarray) -> None:
    assert get_spec("tspulse_zeroshot").task == "ad"
    model = build("tspulse_zeroshot", batch_size=128)
    try:
        model.fit(raw_window)  # zero-shot: no labels, no gradient step
        assert model.fit_seconds >= 0.0
        assert model._context_length == 512  # the TSPulse-r1 checkpoint's fixed context
        scores = model.score(raw_window)
        assert scores.shape == (N,)
        assert np.isfinite(scores).all()
        assert (scores >= 0.0).all()  # mean squared reconstruction error
        assert scores.std() > 0.0  # not a degenerate constant score
        if _CUDA:
            assert torch.cuda.max_memory_allocated() < 2e9, "tspulse_zeroshot used > 2 GB on the shared GPU"
    finally:
        # Hard rule: the RTX A4500 is shared with other agents' tests - always release it.
        del model
        gc.collect()
        if _CUDA:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


@_needs_tspulse_weights
def test_tspulse_zeroshot_rejects_mismatched_context_points(raw_window: np.ndarray) -> None:
    model = build("tspulse_zeroshot", context_points=64)  # checkpoint context is fixed at 512
    with pytest.raises(ValueError, match="context_length"):
        model.fit(raw_window)


def test_ad_model_fit_signature_has_no_label_argument() -> None:
    import inspect

    sig = inspect.signature(AnomalyDetector.fit)
    assert "y" not in sig.parameters
    assert get_spec("tspulse_zeroshot").task == "ad"
