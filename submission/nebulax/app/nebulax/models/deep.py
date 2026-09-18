"""Deep (torch) rows of the model ladder: reconstruction anomaly detectors and
raw-window classifiers.

Registered here (name / family / input_kind / task, verbatim from
``configs/model_ladder.yaml``)::

    lstm_ae             autoencoder     raw_window      ad
    tcn_ae              autoencoder     raw_window      ad
    usad                adversarial_ae  raw_window      ad
    sparse_autoencoder  autoencoder     window_stats    ad
    resnet1d            deep_conv       raw_window      cls
    litetime            deep_conv       raw_window      cls

All six share one budgeted training loop (:func:`train_loop`):

* **train cap** - at most ``max_windows`` (default 50 000, the runner's
  ``MAX_DEEP_TRAIN_WINDOWS``) rows, seeded subsample when there are more;
* **early stopping** - a validation slice carved out of the *training* rows only
  (``val_frac``, seeded permutation). No model in this module ever sees a label during
  anomaly-detection training, and no model ever sees the test slice;
* **wall-clock budget** - ``_fit`` accepts ``budget_s`` (the runner passes the run's
  remaining budget; :meth:`_accepts` in ``nebulax.bench.runner`` detects it). Without it the
  constructor's ``max_minutes`` applies. Training stops at the budget mid-epoch and keeps the
  best validation weights;
* **fixed seed** - ``seed`` seeds weight init, the train/val carve and batch shuffling.

Memory discipline (the GPU is shared): training tensors stay in CPU RAM and are moved to the
device one mini-batch at a time; long raw windows are average-pooled to ``max_seq_len`` steps
before anything else; :meth:`free` drops the modules and empties the CUDA cache.

Deviations from the ladder rows are listed in each class docstring; the two that apply
repo-wide are (a) ``litetime``'s ``source: aeon`` - ``aeon``'s deep learners need TensorFlow,
which is not installed in the ``nebulax`` env, so LITE/LITETime is implemented here in torch;
(b) every architecture pools raw windows longer than ``max_seq_len`` (Ottawa windows are
~10 k samples), which is an efficiency choice, not a modelling one.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, ClassVar, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nebulax.bench.base import AnomalyDetector, Classifier
from nebulax.bench.registry import register

__all__ = [
    "MAX_TRAIN_WINDOWS",
    "resolve_device",
    "train_loop",
    "LSTMAutoencoder",
    "TCNAutoencoder",
    "USAD",
    "SparseAutoencoder",
    "ResNet1D",
    "LITETime",
]

#: Mirrors ``nebulax.bench.runner.MAX_DEEP_TRAIN_WINDOWS`` (imported lazily to keep this
#: module importable without the runner).
MAX_TRAIN_WINDOWS: int = 50_000

_BIG = 1e12


# --------------------------------------------------------------------------------------
# device / seeding / array hygiene
# --------------------------------------------------------------------------------------


def resolve_device(device: str | None = None) -> torch.device:
    """``"cuda"`` when available (or the explicit ``device``), else CPU."""
    if device is not None and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():  # pragma: no cover - GPU only
        torch.cuda.manual_seed_all(int(seed))


def _clean(arr: np.ndarray) -> np.ndarray:
    """float32, no NaN/inf (the runner rejects non-finite scores, so clean at the door)."""
    out = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=_BIG, neginf=-_BIG).astype(np.float32, copy=False)
    return out


def _pool_to(x: Tensor, length: int) -> Tensor:
    """Average-pool ``(n, c, L)`` to ``(n, c, length)``; identity when ``L == length``."""
    if x.shape[-1] == length:
        return x
    return F.adaptive_avg_pool1d(x, length)


# --------------------------------------------------------------------------------------
# the shared budgeted training loop
# --------------------------------------------------------------------------------------


def train_loop(
    modules: Sequence[nn.Module],
    tensors: Sequence[Tensor],
    *,
    train_step: Callable[[tuple[Tensor, ...], int], float],
    eval_step: Callable[[tuple[Tensor, ...], int], float],
    device: torch.device,
    budget_s: float,
    max_epochs: int = 200,
    batch_size: int = 128,
    patience: int = 8,
    val_frac: float = 0.15,
    seed: int = 0,
    min_val_rows: int = 16,
    min_delta: float = 1e-6,
) -> dict[str, Any]:
    """Budgeted mini-batch training with early stopping on a carved-out validation slice.

    ``tensors`` live in CPU RAM and are sliced/moved per batch. ``train_step(batch, epoch)``
    does the optimiser work and returns its loss; ``eval_step(batch, epoch)`` returns the
    validation criterion (lower is better) and is called under ``torch.no_grad``.
    ``epoch`` is 1-based (USAD's loss weights depend on it).

    Returns ``{"epochs", "best_loss", "stop_reason", "val_rows"}``. The best-validation state
    of every module in ``modules`` is restored before returning.
    """
    t_start = time.perf_counter()
    n = int(tensors[0].shape[0])
    gen = np.random.default_rng(int(seed))
    perm = gen.permutation(n)
    n_val = int(round(float(val_frac) * n))
    if n < min_val_rows * 2:
        n_val = 0
    n_val = min(max(n_val, 0), max(n - 1, 0))
    val_idx = torch.from_numpy(np.sort(perm[:n_val]).astype(np.int64))
    tr_idx = torch.from_numpy(np.sort(perm[n_val:]).astype(np.int64))
    bs = max(1, int(batch_size))

    def _batches(idx: Tensor) -> Iterable[tuple[Tensor, ...]]:
        for start in range(0, int(idx.numel()), bs):
            sel = idx[start : start + bs]
            yield tuple(t.index_select(0, sel).to(device, non_blocking=True) for t in tensors)

    best = float("inf")
    best_state: list[dict[str, Tensor]] | None = None
    bad_epochs = 0
    stop_reason = "max_epochs"
    epoch = 0
    for epoch in range(1, int(max_epochs) + 1):
        for m in modules:
            m.train()
        order = torch.from_numpy(gen.permutation(int(tr_idx.numel())).astype(np.int64))
        shuffled = tr_idx.index_select(0, order)
        train_loss, n_batches = 0.0, 0
        out_of_time = False
        for batch in _batches(shuffled):
            train_loss += float(train_step(batch, epoch))
            n_batches += 1
            if time.perf_counter() - t_start > budget_s:
                out_of_time = True
                break
        train_loss = train_loss / max(n_batches, 1)

        for m in modules:
            m.eval()
        if n_val:
            total, rows = 0.0, 0
            with torch.no_grad():
                for batch in _batches(val_idx):
                    k = int(batch[0].shape[0])
                    total += float(eval_step(batch, epoch)) * k
                    rows += k
            val_loss = total / max(rows, 1)
        else:
            val_loss = train_loss
        if not np.isfinite(val_loss):
            val_loss = _BIG

        if val_loss < best - min_delta:
            best = val_loss
            best_state = [copy.deepcopy({k: v.detach().cpu() for k, v in m.state_dict().items()}) for m in modules]
            bad_epochs = 0
        else:
            bad_epochs += 1

        if out_of_time:
            stop_reason = "budget"
            break
        if bad_epochs >= int(patience):
            stop_reason = "early_stopping"
            break

    if best_state is not None:
        for m, state in zip(modules, best_state):
            m.load_state_dict(state)
    for m in modules:
        m.eval()
    return {
        "epochs": int(epoch),
        "best_loss": float(best),
        "stop_reason": stop_reason,
        "val_rows": int(n_val),
    }


# --------------------------------------------------------------------------------------
# shared model machinery
# --------------------------------------------------------------------------------------


class _TorchMixin:
    """Preprocessing, budget bookkeeping and parameter counting shared by all six rows."""

    _modules_: list[nn.Module]

    def _setup(
        self,
        *,
        seed: int,
        device: str | None,
        max_minutes: float,
        max_windows: int,
        max_seq_len: int,
        batch_size: int,
    ) -> None:
        self.seed = int(seed)
        self.device = resolve_device(device)
        self.max_minutes = float(max_minutes)
        self.max_windows = int(max_windows)
        self.max_seq_len = int(max_seq_len)
        self.batch_size = int(batch_size)
        self._modules_ = []
        self._mean_: np.ndarray | None = None
        self._std_: np.ndarray | None = None
        self._seq_len_: int = 0
        self._n_channels_: int = 0
        self.train_info_: dict[str, Any] = {}

    # -- budget -------------------------------------------------------------------
    def _budget(self, budget_s: float | None, *, reserve: float = 0.85) -> float:
        """Seconds available for training: the runner's ``budget_s`` if given, else
        ``max_minutes`` * 60, minus a reserve for scoring/serialisation."""
        total = float(budget_s) if budget_s is not None else self.max_minutes * 60.0
        return max(1.0, float(reserve) * total)

    # -- subsampling --------------------------------------------------------------
    def _cap_rows(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] <= self.max_windows:
            return X
        idx = np.sort(np.random.default_rng(self.seed).choice(X.shape[0], self.max_windows, replace=False))
        return X[idx]

    # -- raw windows (n, L, c) -> (n, c, L') --------------------------------------
    def _prep_raw(self, X: np.ndarray, *, fit: bool) -> Tensor:
        arr = _clean(X)
        if arr.ndim != 3:
            raise ValueError(f"{self.name}: raw_window models need X of shape (n, L, c), got {arr.shape}")
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 2, 1)))  # (n, c, L)
        if fit:
            self._n_channels_ = int(x.shape[1])
            self._seq_len_ = int(min(x.shape[2], self.max_seq_len))
        elif int(x.shape[1]) != self._n_channels_:
            raise ValueError(
                f"{self.name}: fitted on {self._n_channels_} channels, got {int(x.shape[1])}"
            )
        x = _pool_to(x, self._seq_len_)
        if fit:
            mean = x.mean(dim=(0, 2), keepdim=True)
            std = x.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
            self._mean_, self._std_ = mean.numpy(), std.numpy()
        assert self._mean_ is not None and self._std_ is not None
        x = (x - torch.from_numpy(self._mean_)) / torch.from_numpy(self._std_)
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).contiguous()

    # -- tabular (n, f) -----------------------------------------------------------
    def _prep_flat(self, X: np.ndarray, *, fit: bool) -> Tensor:
        arr = _clean(X)
        if arr.ndim != 2:
            raise ValueError(f"{self.name}: window_stats models need X of shape (n, f), got {arr.shape}")
        x = torch.from_numpy(np.ascontiguousarray(arr))
        if fit:
            self._n_channels_ = int(x.shape[1])
            self._mean_ = x.mean(dim=0, keepdim=True).numpy()
            self._std_ = x.std(dim=0, keepdim=True).clamp_min(1e-6).numpy()
        elif int(x.shape[1]) != self._n_channels_:
            raise ValueError(f"{self.name}: fitted on {self._n_channels_} features, got {int(x.shape[1])}")
        assert self._mean_ is not None and self._std_ is not None
        x = (x - torch.from_numpy(self._mean_)) / torch.from_numpy(self._std_)
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).contiguous()

    # -- batched inference --------------------------------------------------------
    def _batched(self, x: Tensor, fn: Callable[[Tensor], Tensor], *, batch_size: int | None = None) -> np.ndarray:
        bs = int(batch_size or max(1, self.batch_size * 2))
        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, int(x.shape[0]), bs):
                chunk = x[start : start + bs].to(self.device, non_blocking=True)
                out.append(np.asarray(fn(chunk).detach().cpu().numpy(), dtype=np.float64))
        return np.concatenate(out, axis=0) if out else np.zeros((0,), dtype=np.float64)

    # -- reporting ----------------------------------------------------------------
    @property
    def n_params(self) -> int:
        """Total trainable + buffered parameter count (the runner writes it to the row)."""
        return int(sum(int(p.numel()) for m in self._modules_ for p in m.parameters()))

    def free(self) -> None:
        """Drop the torch modules and empty the CUDA cache (shared-GPU hygiene)."""
        self._modules_ = []
        for attr in ("net_", "encoder_", "decoder_", "decoder2_", "nets_"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        if torch.cuda.is_available():  # pragma: no cover - GPU only
            torch.cuda.empty_cache()

    @staticmethod
    def _finite(scores: np.ndarray) -> np.ndarray:
        out = np.asarray(scores, dtype=np.float64).reshape(-1)
        return np.nan_to_num(out, nan=0.0, posinf=_BIG, neginf=-_BIG)


def _row_mse(recon: Tensor, target: Tensor) -> Tensor:
    """Per-row mean squared error over every non-batch axis."""
    diff = (recon - target) ** 2
    return diff.flatten(1).mean(dim=1)


# --------------------------------------------------------------------------------------
# lstm_ae
# --------------------------------------------------------------------------------------


class _LSTMAENet(nn.Module):
    def __init__(self, n_channels: int, hidden_size: int, latent_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            n_channels, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.decoder = nn.LSTM(
            latent_size, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, n_channels)

    def forward(self, x: Tensor) -> Tensor:  # x: (n, c, L)
        seq = x.transpose(1, 2)  # (n, L, c)
        _, (h_n, _) = self.encoder(seq)
        z = self.to_latent(h_n[-1])  # (n, latent)
        rep = z.unsqueeze(1).expand(-1, seq.shape[1], -1)
        dec, _ = self.decoder(rep)
        return self.head(dec).transpose(1, 2)  # (n, c, L)


@register("lstm_ae", input_kind="raw_window", family="autoencoder", task="ad", needs_gpu=True, reference_id=("R92", "R1"))
class LSTMAutoencoder(_TorchMixin, AnomalyDetector):
    """Sequence-to-sequence LSTM autoencoder; score = per-window reconstruction MSE.

    The encoder LSTM's last hidden state is projected to a ``latent_size`` code, repeated over
    the window and decoded by a second LSTM (Malhotra et al.'s LSTM-ED, ladder refs R92/R1).

    Defaults: ``hidden_size=64, latent_size=32, num_layers=1, dropout=0.0, lr=1e-3,
    weight_decay=0.0, batch_size=128, max_epochs=200, patience=8, val_frac=0.15,
    max_seq_len=256, max_windows=50000, max_minutes=10.0, seed=0, device=None`` (None = cuda
    when available).

    Deviation: windows longer than ``max_seq_len`` are average-pooled to that length before
    training (efficiency; Ottawa raw windows are ~10 k samples).
    """

    def __init__(
        self,
        hidden_size: int = 64,
        latent_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 128,
        max_epochs: int = 200,
        patience: int = 8,
        val_frac: float = 0.15,
        max_seq_len: int = 256,
        max_windows: int = MAX_TRAIN_WINDOWS,
        max_minutes: float = 10.0,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size, latent_size=latent_size, num_layers=num_layers, dropout=dropout,
            lr=lr, weight_decay=weight_decay, batch_size=batch_size, max_epochs=max_epochs,
            patience=patience, val_frac=val_frac, max_seq_len=max_seq_len, max_windows=max_windows,
            max_minutes=max_minutes, seed=seed, device=device,
        )
        self._setup(seed=seed, device=device, max_minutes=max_minutes, max_windows=max_windows,
                    max_seq_len=max_seq_len, batch_size=batch_size)
        self.hidden_size, self.latent_size = int(hidden_size), int(latent_size)
        self.num_layers, self.dropout = int(num_layers), float(dropout)
        self.lr, self.weight_decay = float(lr), float(weight_decay)
        self.max_epochs, self.patience, self.val_frac = int(max_epochs), int(patience), float(val_frac)
        self.net_: _LSTMAENet | None = None

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, *, budget_s: float | None = None, **kwargs: Any) -> None:
        _seed_everything(self.seed)
        x = self._prep_raw(self._cap_rows(X), fit=True)
        net = _LSTMAENet(self._n_channels_, self.hidden_size, self.latent_size, self.num_layers, self.dropout)
        self.net_ = net.to(self.device)
        self._modules_ = [self.net_]
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def train_step(batch: tuple[Tensor, ...], epoch: int) -> float:
            (xb,) = batch
            opt.zero_grad(set_to_none=True)
            loss = _row_mse(net(xb), xb).mean()
            loss.backward()
            opt.step()
            return float(loss.detach())

        def eval_step(batch: tuple[Tensor, ...], epoch: int) -> float:
            (xb,) = batch
            return float(_row_mse(net(xb), xb).mean())

        self.train_info_ = train_loop(
            [net], [x], train_step=train_step, eval_step=eval_step, device=self.device,
            budget_s=self._budget(budget_s), max_epochs=self.max_epochs, batch_size=self.batch_size,
            patience=self.patience, val_frac=self.val_frac, seed=self.seed,
        )

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        assert self.net_ is not None
        x = self._prep_raw(X, fit=False)
        return self._finite(self._batched(x, lambda b: _row_mse(self.net_(b), b)))


# --------------------------------------------------------------------------------------
# tcn_ae
# --------------------------------------------------------------------------------------


class _TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding="same")
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding="same")
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class _TCNAENet(nn.Module):
    def __init__(
        self, n_channels: int, n_filters: int, kernel_size: int, n_levels: int,
        latent_channels: int, pool: int, dropout: float,
    ) -> None:
        super().__init__()
        self.pool = int(pool)
        self.stem = nn.Conv1d(n_channels, n_filters, 1)
        self.enc = nn.Sequential(*[_TCNBlock(n_filters, kernel_size, 2**i, dropout) for i in range(n_levels)])
        self.to_latent = nn.Conv1d(n_filters, latent_channels, 1)
        self.from_latent = nn.Conv1d(latent_channels, n_filters, 1)
        self.dec = nn.Sequential(*[_TCNBlock(n_filters, kernel_size, 2 ** (n_levels - 1 - i), dropout) for i in range(n_levels)])
        self.head = nn.Conv1d(n_filters, n_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        length = x.shape[-1]
        h = self.enc(self.stem(x))
        z = self.to_latent(h)
        if self.pool > 1 and z.shape[-1] > self.pool:
            z = F.avg_pool1d(z, kernel_size=self.pool, stride=self.pool, ceil_mode=True)
            z = F.interpolate(z, size=length, mode="linear", align_corners=False)
        return self.head(self.dec(self.from_latent(z)))


@register("tcn_ae", input_kind="raw_window", family="autoencoder", task="ad", needs_gpu=True, reference_id=("R37",))
class TCNAutoencoder(_TorchMixin, AnomalyDetector):
    """Temporal-convolutional autoencoder (dilated residual encoder, temporally compressed
    bottleneck, dilated decoder); score = per-window reconstruction MSE. Ladder ref R37.

    Defaults: ``n_filters=32, kernel_size=3, n_levels=4, latent_channels=8, pool=4,
    dropout=0.0, lr=1e-3, weight_decay=0.0, batch_size=128, max_epochs=200, patience=8,
    val_frac=0.15, max_seq_len=256, max_windows=50000, max_minutes=10.0, seed=0, device=None``.

    Deviation: the bottleneck compresses in time by ``pool`` and is resampled back with linear
    interpolation instead of Thill et al.'s learned up-sampling decoder (simpler, same
    behaviour for scoring); long windows are pooled to ``max_seq_len`` as above.
    """

    def __init__(
        self,
        n_filters: int = 32,
        kernel_size: int = 3,
        n_levels: int = 4,
        latent_channels: int = 8,
        pool: int = 4,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 128,
        max_epochs: int = 200,
        patience: int = 8,
        val_frac: float = 0.15,
        max_seq_len: int = 256,
        max_windows: int = MAX_TRAIN_WINDOWS,
        max_minutes: float = 10.0,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        super().__init__(
            n_filters=n_filters, kernel_size=kernel_size, n_levels=n_levels,
            latent_channels=latent_channels, pool=pool, dropout=dropout, lr=lr,
            weight_decay=weight_decay, batch_size=batch_size, max_epochs=max_epochs,
            patience=patience, val_frac=val_frac, max_seq_len=max_seq_len,
            max_windows=max_windows, max_minutes=max_minutes, seed=seed, device=device,
        )
        self._setup(seed=seed, device=device, max_minutes=max_minutes, max_windows=max_windows,
                    max_seq_len=max_seq_len, batch_size=batch_size)
        self.n_filters, self.kernel_size, self.n_levels = int(n_filters), int(kernel_size), int(n_levels)
        self.latent_channels, self.pool, self.dropout = int(latent_channels), int(pool), float(dropout)
        self.lr, self.weight_decay = float(lr), float(weight_decay)
        self.max_epochs, self.patience, self.val_frac = int(max_epochs), int(patience), float(val_frac)
        self.net_: _TCNAENet | None = None

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, *, budget_s: float | None = None, **kwargs: Any) -> None:
        _seed_everything(self.seed)
        x = self._prep_raw(self._cap_rows(X), fit=True)
        net = _TCNAENet(
            self._n_channels_, self.n_filters, self.kernel_size, self.n_levels,
            self.latent_channels, self.pool, self.dropout,
        )
        self.net_ = net.to(self.device)
        self._modules_ = [self.net_]
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def train_step(batch: tuple[Tensor, ...], epoch: int) -> float:
            (xb,) = batch
            opt.zero_grad(set_to_none=True)
            loss = _row_mse(net(xb), xb).mean()
            loss.backward()
            opt.step()
            return float(loss.detach())

        def eval_step(batch: tuple[Tensor, ...], epoch: int) -> float:
            (xb,) = batch
            return float(_row_mse(net(xb), xb).mean())

        self.train_info_ = train_loop(
            [net], [x], train_step=train_step, eval_step=eval_step, device=self.device,
            budget_s=self._budget(budget_s), max_epochs=self.max_epochs, batch_size=self.batch_size,
            patience=self.patience, val_frac=self.val_frac, seed=self.seed,
        )

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        assert self.net_ is not None
        x = self._prep_raw(X, fit=False)
        return self._finite(self._batched(x, lambda b: _row_mse(self.net_(b), b)))


# --------------------------------------------------------------------------------------
# usad
# --------------------------------------------------------------------------------------


def _mlp(sizes: Sequence[int], *, final_activation: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or final_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


@register("usad", input_kind="raw_window", family="adversarial_ae", task="ad", needs_gpu=True, reference_id=("R16", "R1"))
class USAD(_TorchMixin, AnomalyDetector):
    """UnSupervised Anomaly Detection (Audibert et al. 2020, ladder refs R16/R1): one shared
    encoder ``E`` and two decoders ``D1``/``D2`` trained in the paper's two-phase adversarial
    scheme, with the paper's epoch-dependent weights (``n`` = 1-based epoch)::

        L_AE1 = 1/n * ||W - D1(E(W))||^2 + (1 - 1/n) * ||W - D2(E(D1(E(W))))||^2
        L_AE2 = 1/n * ||W - D2(E(W))||^2 - (1 - 1/n) * ||W - D2(E(D1(E(W))))||^2

    Score (paper eq. 6): ``alpha * ||W - D1(E(W))||^2 + beta * ||W - D2(E(D1(E(W))))||^2``.

    Defaults: ``hidden_sizes=(256, 128), latent_size=32, alpha=0.5, beta=0.5, lr=1e-3,
    weight_decay=0.0, batch_size=128, max_epochs=100, patience=8, val_frac=0.15,
    max_seq_len=128, max_windows=50000, max_minutes=10.0, seed=0, device=None``.

    Deviations: (a) the window is flattened to ``channels * max_seq_len`` and the encoder uses
    explicit ``hidden_sizes`` instead of the paper's ``w/2, w/4, w/8`` cascade - on a
    15-channel 128-step window the paper's widths are a ~30 M-parameter first layer;
    (b) inputs are z-scored (not min-maxed) and the decoder output layer is linear, not
    sigmoid, because the benchmark's test slices routinely exceed the training range.
    """

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (256, 128),
        latent_size: int = 32,
        alpha: float = 0.5,
        beta: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 128,
        max_epochs: int = 100,
        patience: int = 8,
        val_frac: float = 0.15,
        max_seq_len: int = 128,
        max_windows: int = MAX_TRAIN_WINDOWS,
        max_minutes: float = 10.0,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        super().__init__(
            hidden_sizes=tuple(int(h) for h in hidden_sizes), latent_size=latent_size, alpha=alpha,
            beta=beta, lr=lr, weight_decay=weight_decay, batch_size=batch_size, max_epochs=max_epochs,
            patience=patience, val_frac=val_frac, max_seq_len=max_seq_len, max_windows=max_windows,
            max_minutes=max_minutes, seed=seed, device=device,
        )
        self._setup(seed=seed, device=device, max_minutes=max_minutes, max_windows=max_windows,
                    max_seq_len=max_seq_len, batch_size=batch_size)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.latent_size, self.alpha, self.beta = int(latent_size), float(alpha), float(beta)
        self.lr, self.weight_decay = float(lr), float(weight_decay)
        self.max_epochs, self.patience, self.val_frac = int(max_epochs), int(patience), float(val_frac)
        self.encoder_: nn.Module | None = None
        self.decoder_: nn.Module | None = None
        self.decoder2_: nn.Module | None = None

    # -- internals ----------------------------------------------------------------
    def _errors(self, xb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``(||W - D1(E)||^2, ||W - D2(E)||^2, ||W - D2(E(D1(E)))||^2)`` per row."""
        flat = xb.flatten(1)
        z = self.encoder_(flat)
        w1 = self.decoder_(z)
        w2 = self.decoder2_(z)
        w3 = self.decoder2_(self.encoder_(w1))
        return (
            ((w1 - flat) ** 2).mean(dim=1),
            ((w2 - flat) ** 2).mean(dim=1),
            ((w3 - flat) ** 2).mean(dim=1),
        )

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, *, budget_s: float | None = None, **kwargs: Any) -> None:
        _seed_everything(self.seed)
        x = self._prep_raw(self._cap_rows(X), fit=True)
        w = int(x.shape[1] * x.shape[2])
        sizes = [w, *self.hidden_sizes, self.latent_size]
        self.encoder_ = _mlp(sizes, final_activation=True).to(self.device)
        self.decoder_ = _mlp(list(reversed(sizes)), final_activation=False).to(self.device)
        self.decoder2_ = _mlp(list(reversed(sizes)), final_activation=False).to(self.device)
        self._modules_ = [self.encoder_, self.decoder_, self.decoder2_]
        opt1 = torch.optim.Adam(
            list(self.encoder_.parameters()) + list(self.decoder_.parameters()),
            lr=self.lr, weight_decay=self.weight_decay,
        )
        opt2 = torch.optim.Adam(
            list(self.encoder_.parameters()) + list(self.decoder2_.parameters()),
            lr=self.lr, weight_decay=self.weight_decay,
        )

        def train_step(batch: tuple[Tensor, ...], epoch: int) -> float:
            (xb,) = batch
            inv = 1.0 / float(epoch)
            opt1.zero_grad(set_to_none=True)
            e1, _, e3 = self._errors(xb)
            loss1 = inv * e1.mean() + (1.0 - inv) * e3.mean()
            loss1.backward()
            opt1.step()

            opt2.zero_grad(set_to_none=True)
            _, e2, e3b = self._errors(xb)
            loss2 = inv * e2.mean() - (1.0 - inv) * e3b.mean()
            loss2.backward()
            opt2.step()
            return float(loss1.detach())

        def eval_step(batch: tuple[Tensor, ...], epoch: int) -> float:
            (xb,) = batch
            e1, _, e3 = self._errors(xb)
            return float((self.alpha * e1 + self.beta * e3).mean())

        self.train_info_ = train_loop(
            self._modules_, [x], train_step=train_step, eval_step=eval_step, device=self.device,
            budget_s=self._budget(budget_s), max_epochs=self.max_epochs, batch_size=self.batch_size,
            patience=self.patience, val_frac=self.val_frac, seed=self.seed,
        )

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        assert self.encoder_ is not None
        x = self._prep_raw(X, fit=False)

        def fn(b: Tensor) -> Tensor:
            e1, _, e3 = self._errors(b)
            return self.alpha * e1 + self.beta * e3

        return self._finite(self._batched(x, fn))


# --------------------------------------------------------------------------------------
# sparse_autoencoder
# --------------------------------------------------------------------------------------


class _SparseAENet(nn.Module):
    def __init__(self, n_features: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        enc: list[nn.Module] = []
        sizes = [n_features, *hidden_sizes]
        for i in range(len(sizes) - 1):
            enc += [nn.Linear(sizes[i], sizes[i + 1]), nn.Sigmoid()]
        self.encoder = nn.Sequential(*enc)
        dec: list[nn.Module] = []
        rev = list(reversed(sizes))
        for i in range(len(rev) - 1):
            dec.append(nn.Linear(rev[i], rev[i + 1]))
            if i < len(rev) - 2:
                dec.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self.decoder(h), h


def _causal_moving_average(values: np.ndarray, k: int) -> np.ndarray:
    """Causal (trailing) moving average of width ``k``, warm-started on the prefix."""
    if k <= 1 or values.size == 0:
        return values
    k = min(int(k), values.size)
    csum = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    idx = np.arange(values.size)
    lo = np.maximum(idx - k + 1, 0)
    return (csum[idx + 1] - csum[lo]) / (idx - lo + 1)


@register(
    "sparse_autoencoder", input_kind="window_stats", family="autoencoder", task="ad",
    needs_gpu=True, reference_id=("R91", "R88"),
)
class SparseAutoencoder(_TorchMixin, AnomalyDetector):
    """KL-sparse dense autoencoder on window statistics - the reference MetroPT-3 baseline
    (ladder refs R91/R88, ``note: reference MetroPT-3 baseline; smooth the error signal``).

    Loss = reconstruction MSE + ``sparsity_weight`` * KL(``sparsity_target`` || mean hidden
    activation) + ``l1_weight`` * mean |h|. Score = per-row reconstruction MSE passed through
    a causal moving average of width ``smooth_window`` (the ladder note), applied in ascending
    ``t`` order when the runner supplies timestamps, else in row order.

    Defaults: ``hidden_sizes=(32,), sparsity_target=0.05, sparsity_weight=1e-3, l1_weight=0.0,
    smooth_window=5, lr=1e-3, weight_decay=0.0, batch_size=256, max_epochs=300, patience=15,
    val_frac=0.15, max_windows=50000, max_minutes=10.0, seed=0, device=None``.

    Note: the smoother assumes the scored slice is one component's series (MetroPT-3 is); with
    several interleaved series it still only mixes rows that are adjacent in time, and
    ``smooth_window=1`` turns it off.
    """

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (32,),
        sparsity_target: float = 0.05,
        sparsity_weight: float = 1e-3,
        l1_weight: float = 0.0,
        smooth_window: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        max_epochs: int = 300,
        patience: int = 15,
        val_frac: float = 0.15,
        max_windows: int = MAX_TRAIN_WINDOWS,
        max_minutes: float = 10.0,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        super().__init__(
            hidden_sizes=tuple(int(h) for h in hidden_sizes), sparsity_target=sparsity_target,
            sparsity_weight=sparsity_weight, l1_weight=l1_weight, smooth_window=smooth_window,
            lr=lr, weight_decay=weight_decay, batch_size=batch_size, max_epochs=max_epochs,
            patience=patience, val_frac=val_frac, max_windows=max_windows, max_minutes=max_minutes,
            seed=seed, device=device,
        )
        self._setup(seed=seed, device=device, max_minutes=max_minutes, max_windows=max_windows,
                    max_seq_len=0, batch_size=batch_size)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.sparsity_target, self.sparsity_weight = float(sparsity_target), float(sparsity_weight)
        self.l1_weight, self.smooth_window = float(l1_weight), int(smooth_window)
        self.lr, self.weight_decay = float(lr), float(weight_decay)
        self.max_epochs, self.patience, self.val_frac = int(max_epochs), int(patience), float(val_frac)
        self.net_: _SparseAENet | None = None

    def _kl(self, h: Tensor) -> Tensor:
        rho = torch.as_tensor(self.sparsity_target, dtype=h.dtype, device=h.device)
        rho_hat = h.mean(dim=0).clamp(1e-6, 1.0 - 1e-6)
        return (rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))).sum()

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, *, budget_s: float | None = None, **kwargs: Any) -> None:
        _seed_everything(self.seed)
        x = self._prep_flat(self._cap_rows(X), fit=True)
        net = _SparseAENet(self._n_channels_, self.hidden_sizes)
        self.net_ = net.to(self.device)
        self._modules_ = [self.net_]
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def train_step(batch: tuple[Tensor, ...], epoch: int) -> float:
            (xb,) = batch
            opt.zero_grad(set_to_none=True)
            recon, h = net(xb)
            loss = _row_mse(recon, xb).mean() + self.sparsity_weight * self._kl(h) + self.l1_weight * h.abs().mean()
            loss.backward()
            opt.step()
            return float(loss.detach())

        def eval_step(batch: tuple[Tensor, ...], epoch: int) -> float:
            (xb,) = batch
            recon, _ = net(xb)
            return float(_row_mse(recon, xb).mean())

        self.train_info_ = train_loop(
            [net], [x], train_step=train_step, eval_step=eval_step, device=self.device,
            budget_s=self._budget(budget_s), max_epochs=self.max_epochs, batch_size=self.batch_size,
            patience=self.patience, val_frac=self.val_frac, seed=self.seed,
        )

    def _smooth(self, err: np.ndarray, t: np.ndarray | None) -> np.ndarray:
        if self.smooth_window <= 1 or err.size < 2:
            return err
        if t is None:
            return _causal_moving_average(err, self.smooth_window)
        order = np.argsort(np.asarray(t).astype("datetime64[ms]").astype(np.int64)
                           if np.asarray(t).dtype.kind == "M" else np.asarray(t, dtype=np.float64),
                           kind="stable")
        out = np.empty_like(err)
        out[order] = _causal_moving_average(err[order], self.smooth_window)
        return out

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        assert self.net_ is not None
        x = self._prep_flat(X, fit=False)
        err = self._batched(x, lambda b: _row_mse(self.net_(b)[0], b))
        return self._finite(self._smooth(err, t))


# --------------------------------------------------------------------------------------
# classifiers: shared plumbing
# --------------------------------------------------------------------------------------


class _TorchClassifier(_TorchMixin, Classifier):
    """Shared fit/predict for the two raw-window classifiers (cross-entropy, optional
    balanced class weights, macro-F1-friendly)."""

    task: ClassVar[str] = "cls"

    def _class_weights(self, y_idx: np.ndarray, n_classes: int) -> Tensor | None:
        if self.class_weight != "balanced":
            return None
        counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
        w = np.where(counts > 0, len(y_idx) / (n_classes * np.maximum(counts, 1.0)), 1.0)
        return torch.as_tensor(w, dtype=torch.float32, device=self.device)

    def _make_nets(self, n_channels: int, n_classes: int) -> list[nn.Module]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _logits(self, net: nn.Module, xb: Tensor) -> Tensor:
        return net(xb)

    def _fit(self, X: np.ndarray, y: np.ndarray, *, budget_s: float | None = None, **kwargs: Any) -> None:
        _seed_everything(self.seed)
        arr, y_arr = np.asarray(X), np.asarray(y).reshape(-1)
        if arr.shape[0] > self.max_windows:
            idx = np.sort(np.random.default_rng(self.seed).choice(arr.shape[0], self.max_windows, replace=False))
            arr, y_arr = arr[idx], y_arr[idx]
        x = self._prep_raw(arr, fit=True)
        lookup = {c: i for i, c in enumerate(self.classes_.tolist())}
        y_idx = np.asarray([lookup[c] for c in y_arr.tolist()], dtype=np.int64)
        yt = torch.from_numpy(y_idx)
        n_classes = int(self.classes_.size)
        weights = self._class_weights(y_idx, n_classes)
        loss_fn = nn.CrossEntropyLoss(weight=weights)

        nets = [net.to(self.device) for net in self._make_nets(self._n_channels_, n_classes)]
        self.nets_ = nets
        self._modules_ = list(nets)
        budget = self._budget(budget_s) / max(len(nets), 1)
        infos: list[dict[str, Any]] = []
        for k, net in enumerate(nets):
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            def train_step(batch: tuple[Tensor, ...], epoch: int, _net: nn.Module = net, _opt: Any = opt) -> float:
                xb, yb = batch
                _opt.zero_grad(set_to_none=True)
                loss = loss_fn(self._logits(_net, xb), yb)
                loss.backward()
                _opt.step()
                return float(loss.detach())

            def eval_step(batch: tuple[Tensor, ...], epoch: int, _net: nn.Module = net) -> float:
                xb, yb = batch
                return float(loss_fn(self._logits(_net, xb), yb))

            infos.append(
                train_loop(
                    [net], [x, yt], train_step=train_step, eval_step=eval_step, device=self.device,
                    budget_s=budget, max_epochs=self.max_epochs, batch_size=self.batch_size,
                    patience=self.patience, val_frac=self.val_frac, seed=self.seed + k,
                )
            )
        self.train_info_ = {"members": infos, "n_members": len(nets)}

    def _proba(self, X: np.ndarray) -> np.ndarray:
        x = self._prep_raw(np.asarray(X), fit=False)

        def fn(b: Tensor) -> Tensor:
            probs = torch.stack([F.softmax(self._logits(net, b), dim=1) for net in self.nets_], dim=0)
            return probs.mean(dim=0)

        out = self._batched(x, fn)
        out = np.nan_to_num(out, nan=1.0 / max(int(self.classes_.size), 1))
        return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self._proba(X), axis=1)]

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._proba(X)


# --------------------------------------------------------------------------------------
# resnet1d
# --------------------------------------------------------------------------------------


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_sizes: Sequence[int]) -> None:
        super().__init__()
        chans = [in_ch, *([out_ch] * len(kernel_sizes))]
        self.convs = nn.ModuleList(
            [nn.Conv1d(chans[i], chans[i + 1], k, padding="same") for i, k in enumerate(kernel_sizes)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(out_ch) for _ in kernel_sizes])
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.bn_short = nn.BatchNorm1d(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = bn(conv(h))
            if i < len(self.convs) - 1:
                h = F.relu(h)
        return F.relu(h + self.bn_short(self.shortcut(x)))


class _ResNet1DNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, filters: Sequence[int], kernel_sizes: Sequence[int], dropout: float) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = n_channels
        for f in filters:
            blocks.append(_ResBlock(in_ch, f, kernel_sizes))
            in_ch = f
        self.blocks = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(in_ch, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        h = self.blocks(x)
        return self.head(self.drop(h.mean(dim=2)))


@register("resnet1d", input_kind="raw_window", family="deep_conv", task="cls", needs_gpu=True, reference_id=("R60",))
class ResNet1D(_TorchClassifier):
    """1-D ResNet time-series classifier (Wang et al. 2017 / ladder ref R60): three residual
    blocks of 64/128/128 filters with kernels 8/5/3, batch norm, global average pooling.

    Defaults: ``filters=(64, 128, 128), kernel_sizes=(8, 5, 3), dropout=0.0,
    class_weight="balanced", lr=1e-3, weight_decay=0.0, batch_size=64, max_epochs=200,
    patience=15, val_frac=0.15, max_seq_len=512, max_windows=50000, max_minutes=10.0, seed=0,
    device=None``.

    Deviation: ``class_weight="balanced"`` (inverse-frequency cross-entropy) is on by default
    because the ladder's CLS metric is macro-F1 on very imbalanced fault types; pass
    ``class_weight=None`` for the paper's plain cross-entropy. Windows longer than
    ``max_seq_len`` are average-pooled.
    """

    def __init__(
        self,
        filters: Sequence[int] = (64, 128, 128),
        kernel_sizes: Sequence[int] = (8, 5, 3),
        dropout: float = 0.0,
        class_weight: str | None = "balanced",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 15,
        val_frac: float = 0.15,
        max_seq_len: int = 512,
        max_windows: int = MAX_TRAIN_WINDOWS,
        max_minutes: float = 10.0,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        super().__init__(
            filters=tuple(int(f) for f in filters), kernel_sizes=tuple(int(k) for k in kernel_sizes),
            dropout=dropout, class_weight=class_weight, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, max_epochs=max_epochs, patience=patience, val_frac=val_frac,
            max_seq_len=max_seq_len, max_windows=max_windows, max_minutes=max_minutes, seed=seed,
            device=device,
        )
        self._setup(seed=seed, device=device, max_minutes=max_minutes, max_windows=max_windows,
                    max_seq_len=max_seq_len, batch_size=batch_size)
        self.filters = tuple(int(f) for f in filters)
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.dropout, self.class_weight = float(dropout), class_weight
        self.lr, self.weight_decay = float(lr), float(weight_decay)
        self.max_epochs, self.patience, self.val_frac = int(max_epochs), int(patience), float(val_frac)
        self.nets_: list[nn.Module] = []

    def _make_nets(self, n_channels: int, n_classes: int) -> list[nn.Module]:
        return [_ResNet1DNet(n_channels, n_classes, self.filters, self.kernel_sizes, self.dropout)]


# --------------------------------------------------------------------------------------
# litetime
# --------------------------------------------------------------------------------------


def _handcrafted_filters(n_channels: int, kernel_sizes: Sequence[int]) -> tuple[Tensor, int]:
    """LITE's fixed (non-trainable) increase / decrease / peak detectors.

    Returns a ``(out_channels, n_channels, max_k)`` weight tensor (short kernels are
    zero-padded so one conv covers every kernel size) and ``max_k``.
    """
    max_k = int(max(kernel_sizes))
    rows: list[np.ndarray] = []
    for k in kernel_sizes:
        k = int(k)
        half = max(k // 2, 1)
        inc = np.concatenate([-np.ones(half), np.ones(k - half)])
        dec = -inc
        q = max(k // 4, 1)
        peak = np.concatenate([-np.ones(q), 2.0 * np.ones(k - 2 * q), -np.ones(q)])[:k]
        valley = -peak
        for f in (inc, dec, peak, valley):
            w = np.zeros(max_k, dtype=np.float32)
            pad = (max_k - k) // 2
            w[pad : pad + k] = f.astype(np.float32) / float(k)
            rows.append(w)
    weight = np.stack(rows)[:, None, :].repeat(n_channels, axis=1) / float(n_channels)
    return torch.from_numpy(weight.astype(np.float32)), max_k


class _DWSCBlock(nn.Module):
    """Depthwise-separable convolution block (LITE's body)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, dilation=dilation, groups=in_ch, padding="same")
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn(self.pointwise(self.depthwise(x))))


class _LITENet(nn.Module):
    """One LITE network: multiplexing conv + fixed hand-crafted filters, then DWSC blocks."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        n_filters: int,
        kernel_sizes: Sequence[int],
        custom_kernel_sizes: Sequence[int],
        n_dwsc: int,
        dwsc_kernel_size: int,
        use_custom_filters: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        per = max(1, int(n_filters) // max(len(kernel_sizes), 1))
        self.branches = nn.ModuleList(
            [nn.Conv1d(n_channels, per, int(k), padding="same", bias=False) for k in kernel_sizes]
        )
        out_ch = per * len(kernel_sizes)
        self.use_custom_filters = bool(use_custom_filters)
        if self.use_custom_filters:
            weight, _ = _handcrafted_filters(n_channels, custom_kernel_sizes)
            self.register_buffer("custom_weight", weight)
            out_ch += int(weight.shape[0])
        self.bn = nn.BatchNorm1d(out_ch)
        blocks: list[nn.Module] = []
        in_ch = out_ch
        for i in range(int(n_dwsc)):
            blocks.append(_DWSCBlock(in_ch, int(n_filters), int(dwsc_kernel_size), 2 ** (i + 1)))
            in_ch = int(n_filters)
        self.blocks = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(in_ch, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        parts = [branch(x) for branch in self.branches]
        if self.use_custom_filters:
            parts.append(F.conv1d(x, self.custom_weight, padding="same"))
        h = F.relu(self.bn(torch.cat(parts, dim=1)))
        h = self.blocks(h)
        return self.head(self.drop(h.mean(dim=2)))


@register("litetime", input_kind="raw_window", family="deep_conv", task="cls", needs_gpu=True, reference_id=("R53", "R60"))
class LITETime(_TorchClassifier):
    """LITETime (Ismail-Fawaz et al. 2023, ladder refs R53/R60): an ensemble of ``n_models``
    LITE networks - the lightweight InceptionTime variant. Each LITE network is a multiplexing
    convolution layer (kernels 2/4/8) concatenated with fixed, non-trainable hand-crafted
    increase / decrease / peak detectors, followed by dilated depthwise-separable convolution
    blocks, global average pooling and a softmax head. ``predict_proba`` averages the members'
    softmax outputs (that averaging is what makes LITE an *ensemble*, i.e. LITETime).

    Defaults: ``n_models=5, n_filters=32, kernel_sizes=(2, 4, 8), custom_kernel_sizes=(2, 4, 8,
    16, 32, 64), n_dwsc=2, dwsc_kernel_size=40, use_custom_filters=True, dropout=0.0,
    class_weight="balanced", lr=1e-3, weight_decay=0.0, batch_size=64, max_epochs=200,
    patience=15, val_frac=0.15, max_seq_len=512, max_windows=50000, max_minutes=10.0, seed=0,
    device=None``. The wall-clock budget is split evenly across the ensemble members.

    Deviations: the ladder row says ``source: aeon``, but ``aeon``'s deep learners require
    TensorFlow, which is not installed in the ``nebulax`` env - this is a faithful torch
    re-implementation instead. ``class_weight="balanced"`` is on by default (see
    :class:`ResNet1D`), and windows longer than ``max_seq_len`` are average-pooled.
    """

    def __init__(
        self,
        n_models: int = 5,
        n_filters: int = 32,
        kernel_sizes: Sequence[int] = (2, 4, 8),
        custom_kernel_sizes: Sequence[int] = (2, 4, 8, 16, 32, 64),
        n_dwsc: int = 2,
        dwsc_kernel_size: int = 40,
        use_custom_filters: bool = True,
        dropout: float = 0.0,
        class_weight: str | None = "balanced",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 15,
        val_frac: float = 0.15,
        max_seq_len: int = 512,
        max_windows: int = MAX_TRAIN_WINDOWS,
        max_minutes: float = 10.0,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        super().__init__(
            n_models=n_models, n_filters=n_filters, kernel_sizes=tuple(int(k) for k in kernel_sizes),
            custom_kernel_sizes=tuple(int(k) for k in custom_kernel_sizes), n_dwsc=n_dwsc,
            dwsc_kernel_size=dwsc_kernel_size, use_custom_filters=use_custom_filters, dropout=dropout,
            class_weight=class_weight, lr=lr, weight_decay=weight_decay, batch_size=batch_size,
            max_epochs=max_epochs, patience=patience, val_frac=val_frac, max_seq_len=max_seq_len,
            max_windows=max_windows, max_minutes=max_minutes, seed=seed, device=device,
        )
        self._setup(seed=seed, device=device, max_minutes=max_minutes, max_windows=max_windows,
                    max_seq_len=max_seq_len, batch_size=batch_size)
        self.n_models, self.n_filters = int(n_models), int(n_filters)
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.custom_kernel_sizes = tuple(int(k) for k in custom_kernel_sizes)
        self.n_dwsc, self.dwsc_kernel_size = int(n_dwsc), int(dwsc_kernel_size)
        self.use_custom_filters, self.dropout = bool(use_custom_filters), float(dropout)
        self.class_weight = class_weight
        self.lr, self.weight_decay = float(lr), float(weight_decay)
        self.max_epochs, self.patience, self.val_frac = int(max_epochs), int(patience), float(val_frac)
        self.nets_: list[nn.Module] = []

    def _make_nets(self, n_channels: int, n_classes: int) -> list[nn.Module]:
        nets: list[nn.Module] = []
        for k in range(max(1, self.n_models)):
            torch.manual_seed(self.seed + k)
            nets.append(
                _LITENet(
                    n_channels, n_classes, self.n_filters, self.kernel_sizes, self.custom_kernel_sizes,
                    self.n_dwsc, self.dwsc_kernel_size, self.use_custom_filters, self.dropout,
                )
            )
        return nets
