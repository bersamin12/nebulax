"""Foundation-model (zero-shot) anomaly detector, ``task="ad"``, ``input_kind="raw_window"``.

Registered here (name/input_kind/family/task copied verbatim from ``configs/model_ladder.yaml``):

* ``tspulse_zeroshot`` - family ``tsfm_ad`` - IBM Granite TSPulse-r1 reconstruction-error
  anomaly score, zero-shot (no fine-tuning). The ladder row carries no ``task:`` key (every
  ``tsfm_ad`` row omits it for anomaly detection), which
  :func:`nebulax.bench.registry.register` defaults to ``"ad"``.

TSPulse (``tsfm_public.models.tspulse.TSPulseForReconstruction``,
``ibm-granite/granite-timeseries-tspulse-r1``) is a small (``d_model=24``) pretrained,
patch-based time-series foundation model whose checkpoint has a **fixed context length of
512 samples** (see its cached ``config.json``: ``context_length: 512``, ``patch_length: 8``,
``num_patches: 128``) - the patch-embedding weights are architecture-fixed to that length, so
every raw window is linearly resampled to 512 samples before scoring (see
:func:`_linear_resize`), never padded: zero/edge padding would inject an artificial flat
segment that the model would flag as a discontinuity, biasing the reconstruction-error score
upward for short windows for reasons unrelated to any real fault.

Zero-shot scoring pattern: load the pretrained model with ``mask_type="user"`` and, at score
time, pass an all-ones ``past_observed_mask`` so ``TSPulseMasking.mask_with_past_observed``
masks nothing (``mask = ~past_observed_mask`` is all ``False``); ``reconstruction_outputs`` is
then the model's own unmasked self-reconstruction of the whole window rather than a
masked-region fill-in. The anomaly score is the per-window mean squared error between the
(resampled) input and its reconstruction, averaged over time and channels - higher error is
more anomalous, matching :class:`nebulax.bench.base.AnomalyDetector`'s contract. No gradient
step is ever taken (``torch.no_grad()`` throughout); ``_fit`` only loads the pretrained
weights and reads the channel count off ``X``.

No network calls: ``from_pretrained(..., local_files_only=True)`` by default, so this model
(and its test) only works when ``ibm-granite/granite-timeseries-tspulse-r1`` is already cached
under ``~/.cache/huggingface`` and raises ``OSError`` instead of downloading otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nebulax.bench.base import AnomalyDetector
from nebulax.bench.registry import register

__all__ = ["TSPulseZeroShot"]

_MODEL_ID = "ibm-granite/granite-timeseries-tspulse-r1"


def _linear_resize(X: np.ndarray, target_len: int) -> np.ndarray:
    """Resample each ``(L, c)`` window in ``X`` (``n, L, c``) to ``target_len`` samples.

    One fixed ``(target_len, L)`` linear-interpolation weight matrix is shared by every
    window in the batch (``align_corners=True`` semantics, matching
    ``torch.nn.functional.interpolate(mode="linear", align_corners=True)``), so the resize is
    a single ``einsum`` rather than a per-window Python loop. A no-op (only a dtype/contiguity
    cast) when ``L == target_len``.
    """
    n, L, c = X.shape
    if L == target_len:
        return np.ascontiguousarray(X, dtype=np.float32)
    if L < 2:
        raise ValueError(f"_linear_resize: window length {L} too short to resample to {target_len}")
    src = np.linspace(0.0, L - 1, num=L)
    dst = np.linspace(0.0, L - 1, num=target_len)
    lo = np.floor(dst).astype(np.int64)
    hi = np.clip(lo + 1, 0, L - 1)
    frac = (dst - lo).astype(np.float32)
    W = np.zeros((target_len, L), dtype=np.float32)
    rows = np.arange(target_len)
    np.add.at(W, (rows, lo), 1.0 - frac)
    np.add.at(W, (rows, hi), frac)
    del src
    return np.einsum("tl,nlc->ntc", W, X.astype(np.float32), optimize=True)


@register(
    "tspulse_zeroshot",
    input_kind="raw_window",
    family="tsfm_ad",
    task="ad",
    tier="foundation",
    reference_id=["R30"],
    needs_gpu=False,
)
class TSPulseZeroShot(AnomalyDetector):
    """Zero-shot TSPulse reconstruction-error anomaly score (no fine-tuning).

    Params (constructor kwargs): ``model_id="ibm-granite/granite-timeseries-tspulse-r1"``,
    ``context_points=None`` (the checkpoint's fixed context length; ``None`` reads it off the
    loaded model's own ``config.context_length``, i.e. 512 for TSPulse-r1 - an explicit value
    that disagrees with the checkpoint raises ``ValueError`` at fit time rather than corrupting
    a forward pass, since the patch-embedding weights are architecture-fixed to that length),
    ``batch_size=64`` (forward-pass batching only, does not affect scores), ``device=None``
    (``"cuda"`` if available else ``"cpu"``), ``local_files_only=True`` (never calls the
    network).
    """

    def __init__(
        self,
        model_id: str = _MODEL_ID,
        context_points: int | None = None,
        batch_size: int = 64,
        device: str | None = None,
        local_files_only: bool = True,
    ) -> None:
        super().__init__(
            model_id=model_id,
            context_points=context_points,
            batch_size=batch_size,
            device=device,
            local_files_only=local_files_only,
        )
        self._model: Any = None
        self._torch_device: str = "cpu"
        self._n_channels: int | None = None
        self._context_length: int | None = None

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kwargs: Any) -> None:
        import torch
        from tsfm_public.models.tspulse import TSPulseForReconstruction

        n_channels = int(X.shape[-1])
        device = self.params["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
        model = TSPulseForReconstruction.from_pretrained(
            self.params["model_id"],
            num_input_channels=n_channels,
            mask_type="user",
            local_files_only=self.params["local_files_only"],
        )
        checkpoint_context = int(model.config.context_length)
        requested = self.params["context_points"]
        if requested is not None and int(requested) != checkpoint_context:
            raise ValueError(
                f"{self.name}: context_points={requested} does not match the "
                f"{self.params['model_id']} checkpoint's fixed context_length="
                f"{checkpoint_context}; the patch-embedding weights cannot be resized"
            )
        model.to(device)
        model.eval()
        self._model = model
        self._torch_device = device
        self._n_channels = n_channels
        self._context_length = checkpoint_context

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        import torch

        if X.shape[-1] != self._n_channels:
            raise ValueError(f"{self.name}.score: fitted on {self._n_channels} channels, got {X.shape[-1]}")
        resized = _linear_resize(X, self._context_length)
        batch_size = int(self.params["batch_size"])
        scores = np.empty(resized.shape[0], dtype=np.float64)
        with torch.no_grad():
            for start in range(0, resized.shape[0], batch_size):
                chunk = resized[start : start + batch_size]
                past_values = torch.from_numpy(chunk).to(self._torch_device, dtype=torch.float32)
                observed_mask = torch.ones_like(past_values)
                out = self._model(past_values=past_values, past_observed_mask=observed_mask, return_loss=False)
                err = (out.reconstruction_outputs - past_values).pow(2).mean(dim=(1, 2))
                scores[start : start + chunk.shape[0]] = err.detach().to("cpu").numpy().astype(np.float64)
                del past_values, observed_mask, out, err
        if self._torch_device == "cuda":
            torch.cuda.empty_cache()
        return scores
