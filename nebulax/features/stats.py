"""Vectorised per-window, per-channel summary statistics.

:func:`window_stats` is the feature extractor behind the ``"window_stats"``
:data:`nebulax.bench.base.InputKind`: it turns :class:`nebulax.features.windows.Windows`'s
``(n, L, c)`` array into the ``(n, f)`` matrix models actually consume. Every statistic is
computed with plain numpy reductions over the window axis (``axis=1``) across all ``(n, c)``
at once - no Python loop over windows or channels.

DC coupling
-----------
``rms``/``crest`` are **DC-coupled by default**: ``rms = sqrt(mean(x**2))`` and
``crest = max|x| / rms`` both include any constant sensor bias in the window. That is the
right default for a process signal (a pressure, a current) where the absolute level *is*
the quantity, and it is what every existing consumer of this function was calibrated
against. It is the *wrong* thing for a vibration channel with a biased accelerometer:
21 of the 60 University of Ottawa UORED-VAFCLS records carry a window mean larger than
their own AC RMS (worst ``H_2_0``: mean 1606 m/s2 against an AC RMS of 56), which inflates
``rms`` by 1.5-44x on those records (worst ``H_17_0``, 43.9x) and crushes ``crest``
towards 1 (``H_17_0``: 1.09 DC-coupled vs 4.19 AC-coupled) - see ``docs/parameters.md``,
bearing section 4.

The fix here is the **keyword argument** :paramref:`window_stats.ac_couple`, not two extra
columns appended to :data:`STAT_NAMES`. Reasons, in order:

* it keeps the emitted matrix **exactly** ``c * len(STAT_NAMES)`` columns wide in the same
  channel-major order, so every already-registered ``window_stats`` bench model keeps the
  feature vector (and the width) it was benchmarked on - appending ``rms_ac``/``crest_ac``
  to :data:`STAT_NAMES` would have silently widened the input of every such model;
* the DC offset is not lost when you opt in: the ``mean`` column *is* the per-window DC
  offset, and ``min``/``max`` stay DC-coupled on purpose, so the artefact remains visible
  rather than being quietly subtracted away;
* the emitted column *names* still tell the truth, because :func:`feature_names` takes the
  same flag and renames the two affected slots to ``rms_ac``/``crest_ac``
  (:data:`AC_STAT_NAMES`, via :func:`stat_names`). Pass the same ``ac_couple`` value to
  both calls - that is the one rule this design asks of a caller.

``kurtosis`` is central (mean-removed) by definition and therefore identical either way;
``std`` is likewise already an AC quantity (``rms`` of the centred window).
"""

from __future__ import annotations

from typing import Final, Sequence

import numpy as np

__all__ = ["STAT_NAMES", "AC_STAT_NAMES", "stat_names", "window_stats", "feature_names"]

#: Per-channel statistic names, in the order they are stacked by :func:`window_stats`.
STAT_NAMES: Final[tuple[str, ...]] = ("mean", "std", "min", "max", "slope", "rms", "kurtosis", "crest")
#: Same slots, same order, as emitted by ``window_stats(..., ac_couple=True)``: only the two
#: statistics whose value actually changes are renamed (``rms`` -> ``rms_ac``,
#: ``crest`` -> ``crest_ac``); ``mean`` still carries the DC offset that was removed.
AC_STAT_NAMES: Final[tuple[str, ...]] = tuple(
    {"rms": "rms_ac", "crest": "crest_ac"}.get(s, s) for s in STAT_NAMES
)


def stat_names(ac_couple: bool = False) -> tuple[str, ...]:
    """:data:`STAT_NAMES`, or :data:`AC_STAT_NAMES` when ``ac_couple`` - the per-channel
    statistic names emitted by ``window_stats(..., ac_couple=ac_couple)``."""
    return AC_STAT_NAMES if ac_couple else STAT_NAMES


def _slope(arr: np.ndarray) -> np.ndarray:
    """Least-squares slope of each ``(n, c)`` series against the sample index ``0..L-1``.

    NaN-aware: samples with a NaN value simply drop out of the per-window regression (a
    window that is all-NaN for a channel gets NaN back).
    """
    n, L, c = arr.shape
    t = np.arange(L, dtype=np.float64).reshape(1, L, 1)
    mask = np.isfinite(arr)
    x0 = np.where(mask, arr, 0.0)
    cnt = mask.sum(axis=1).astype(np.float64)
    sum_t = np.where(mask, t, 0.0).sum(axis=1)
    sum_t2 = np.where(mask, t * t, 0.0).sum(axis=1)
    sum_x = x0.sum(axis=1)
    sum_tx = (t * x0).sum(axis=1)
    denom = cnt * sum_t2 - sum_t * sum_t
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(denom > 0, (cnt * sum_tx - sum_t * sum_x) / np.where(denom == 0, np.nan, denom), np.nan)
    return slope


def window_stats(X: np.ndarray, *, ac_couple: bool = False) -> np.ndarray:
    """``(n, L, c)`` -> ``(n, c * len(STAT_NAMES))`` float32, channel-major column order.

    Column ``j`` of channel ``k`` is ``out[:, k * len(STAT_NAMES) + j]`` -
    :func:`feature_names` gives the matching names when the caller has channel names.
    NaN samples are ignored per statistic (``np.nan*`` reductions); a channel with fewer
    than 2 finite samples in a window returns NaN for every statistic of that window/channel.

    Parameters
    ----------
    X : ``(n, L, c)`` window stack.
    ac_couple : mean-remove each window/channel *before* ``rms`` and the peak behind
        ``crest``, making both invariant to a constant per-window offset (an accelerometer's
        DC bias). Default ``False`` - the historical DC-coupled behaviour, unchanged for
        every existing caller. The shape, the column order and the other six statistics are
        identical either way; ``mean`` still reports the offset that was removed and
        ``min``/``max`` stay DC-coupled, so the offset remains visible in the feature row.
        Pass the same flag to :func:`feature_names`, which then names the two affected slots
        ``rms_ac``/``crest_ac``. See the module docstring for why this is a flag rather than
        two extra columns.
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"window_stats: X must be 3-D (n, L, c), got shape {arr.shape}")
    n, L, c = arr.shape
    if n == 0:
        return np.empty((0, c * len(STAT_NAMES)), dtype=np.float32)
    if L < 2:
        raise ValueError(f"window_stats: window length L={L} must be >= 2 (need a slope)")

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(arr, axis=1)
        std = np.nanstd(arr, axis=1)
        mn = np.nanmin(arr, axis=1)
        mx = np.nanmax(arr, axis=1)

        centered = arr - mean[:, None, :]
        # The only difference AC coupling makes: which signal rms and the crest peak see.
        amp = centered if ac_couple else arr
        rms = np.sqrt(np.nanmean(amp * amp, axis=1))
        peak = np.nanmax(np.abs(amp), axis=1)
        crest = np.where(rms > 0, peak / np.where(rms == 0, np.nan, rms), np.nan)

        var = std * std
        m4 = np.nanmean(centered**4, axis=1)
        kurtosis = np.where(var > 0, m4 / np.where(var == 0, np.nan, var**2), np.nan)

    slope = _slope(arr)

    stats = np.stack([mean, std, mn, mx, slope, rms, kurtosis, crest], axis=-1)  # (n, c, 8)
    # Documented rule: fewer than 2 finite samples in a window/channel -> every statistic NaN
    # (a lone sample would otherwise read as mean=value, std=0, crest=1, which looks valid).
    too_few = np.sum(np.isfinite(arr), axis=1) < 2  # (n, c)
    stats[too_few] = np.nan
    return stats.reshape(n, c * len(STAT_NAMES)).astype(np.float32)


def feature_names(signals: Sequence[str], *, ac_couple: bool = False) -> list[str]:
    """``["<signal>_<stat>", ...]`` in the exact column order :func:`window_stats` emits.

    ``ac_couple`` must match the flag passed to :func:`window_stats`: it swaps the two
    renamed slots (``<signal>_rms_ac`` / ``<signal>_crest_ac``, see :data:`AC_STAT_NAMES`).
    """
    names = stat_names(ac_couple)
    return [f"{sig}_{stat}" for sig in signals for stat in names]
