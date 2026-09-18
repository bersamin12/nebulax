"""Feature extraction for the PS3 SHM task (cumulative fatigue damage regression).

One dynamic-stress channel, 581,120 samples per file, 64 labelled + 16 test files. The label was
produced by the organisers with rainflow counting + Palmgren-Miner + an S-N curve, but on *more*
than this channel (a plain rainflow/Miner fit on the published channel reproduces no label at any
exponent - measured LOO MAPE >= 0.78). The physics that does survive is the power law: for a
stationary process the Miner damage intensity is

    D = (nu0 * T / C) * (sqrt(2) * sigma)^k * Gamma(1 + k/2)          (Bendat, narrow band)

so ``log D`` is affine in ``log`` of any amplitude-scale statistic [R269]. Everything here is therefore
built to be read in log space: the feature families are amplitude scales, cycle counts and
spectral-shape numbers, and :data:`LOG_FEATURES` names the ones that are positive **by
construction** (not by inspection of the data - that would be a fold-local violation).

Families
--------
``stats``     20 cheap amplitude / count statistics (the baseline set).
``rainflow``  own ASTM turning-point extraction + 3-point (E1049 [R290]) and 4-point rainflow,
              residue conventions [R270], range percentiles, ``sum n*sigma^m`` over an exponent
              grid, damage equivalent loads and Goodman / SWT mean-stress corrected variants.
``spectral``  Welch PSD -> spectral moments, nu0+, nup, irregularity, Vanmarcke, alpha_0.75, and
              narrow-band / Dirlik / Tovo-Benasciutti damage at a grid of exponents, all
              hand-implemented from [R269][R275][R276] (no installs, so no FLife / fatpack).
``fds``       fatigue damage spectrum [R274]: SDOF filter bank (Q = 10, octave spaced) evaluated in
              the frequency domain, band RMS and band damage.

References are the ``[Rnnn]`` ids of ``docs/research/references.md`` (R269-R291, R299 are the SHM
entries; the addendum is ``docs/research/ps3_addendum.md`` section 4).

The sample rate is not published, so every frequency here is **normalised** (fs = 1, i.e. cycles
per sample) and every time constant is folded into the fitted multiplicative constant. That is
harmless: an unknown ``C`` and an unknown ``T`` are one unknown scale factor, which the regression
fits as an intercept in log space.

Caching: :func:`feature_table` writes one parquet per file under
``data/ps3_cache/shm/<FEATURE_VERSION>/`` keyed by file name, so re-runs are seconds.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from nebulax.ps3.common import CACHE_DIR

__all__ = [
    "FEATURE_VERSION",
    "FAMILIES",
    "LOG_FEATURES",
    "SN_EXPONENTS",
    "DAMAGE_EXPONENTS",
    "SIGMA_U",
    "load_signal",
    "turning_points",
    "rainflow_4point",
    "rainflow_3point",
    "rainflow_cycles",
    "stats_features",
    "rainflow_features",
    "spectral_features",
    "fds_features",
    "features_for_signal",
    "features_for_file",
    "feature_table",
    "to_log_space",
    "family_columns",
    "block_maxima",
]

#: Bump when a feature definition changes; it keys the parquet cache.
FEATURE_VERSION = "shm-f2"

FAMILIES: tuple[str, ...] = ("stats", "rainflow", "spectral", "fds")

#: S-N exponent grid for the ``sum n * sigma^m`` rainflow features.
SN_EXPONENTS: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
#: Smaller grid for the (more expensive) damage-style features.
DAMAGE_EXPONENTS: tuple[int, ...] = (3, 5, 7, 9, 12)

#: Reference ultimate stress for the Goodman correction. The channel is in arbitrary stress units
#: spanning roughly +-40, so this is a fixed, data-independent constant (never fitted on the data,
#: which would leak across folds); the exponent grid covers the sensitivity.
SIGMA_U = 200.0

_TINY = 1e-12
_LOG_FEATURES: set[str] = set()


def _log_names(*names: str) -> tuple[str, ...]:
    _LOG_FEATURES.update(names)
    return names


# --------------------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------------------


def load_signal(path: Path | str) -> np.ndarray:
    """Read one headerless single-column SHM csv into a float64 array.

    The organisers' files have no header (the first line is the first sample) and CRLF line
    endings. An empty file, or one whose first line is a header word, is a clear error.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SHM signal not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{path.name}: empty SHM file (expected one stress sample per line)")
    try:
        frame = pd.read_csv(
            path, header=None, dtype="float64", skip_blank_lines=False, na_values=["", "NA", "NaN", "nan"]
        )
    except (ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"{path.name}: invalid SHM stress file ({exc})") from exc
    if frame.shape[1] != 1:
        raise ValueError(f"{path.name}: expected one column of stress samples, got {frame.shape[1]}")
    x = frame.to_numpy(dtype="float64").ravel()
    if x.size == 0:
        raise ValueError(f"{path.name}: no samples")
    if not np.isfinite(x).all():
        n_bad = int((~np.isfinite(x)).sum())
        if n_bad > x.size // 100:
            raise ValueError(f"{path.name}: {n_bad} non-finite samples ({100 * n_bad / x.size:.1f}%)")
        x = _interpolate_nans(x)
    return x


def _interpolate_nans(x: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(x)
    idx = np.arange(x.size)
    x = x.copy()
    x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


# --------------------------------------------------------------------------------------
# Rainflow counting (own implementation; no new dependencies)
# --------------------------------------------------------------------------------------


def turning_points(x: np.ndarray) -> np.ndarray:
    """Local extrema of ``x`` with the first and last sample kept (ASTM E1049 step 1)."""
    x = np.asarray(x, dtype="float64")
    if x.size < 3:
        return x.copy()
    d = np.diff(x)
    nz = np.flatnonzero(d)
    if nz.size == 0:
        return x[[0, -1]].copy()
    sign = np.sign(d[nz])
    change = np.flatnonzero(np.diff(sign))
    keep = np.concatenate(([0], nz[change] + 1, [x.size - 1]))
    return x[keep]


def rainflow_4point(tp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Four-point rainflow on a turning-point series.

    Returns ``(ranges, means, residue)``: full closed cycles (count 1 each) and the unclosed
    residue, which the caller resolves with a residue convention (see :func:`rainflow_cycles`).
    """
    stack: list[float] = []
    ranges: list[float] = []
    means: list[float] = []
    for value in tp.tolist():
        stack.append(value)
        while len(stack) >= 4:
            s1, s2, s3, s4 = stack[-4], stack[-3], stack[-2], stack[-1]
            r_inner = abs(s2 - s3)
            if r_inner <= abs(s1 - s2) and r_inner <= abs(s3 - s4):
                ranges.append(r_inner)
                means.append(0.5 * (s2 + s3))
                del stack[-3:-1]
            else:
                break
    return (
        np.asarray(ranges, dtype="float64"),
        np.asarray(means, dtype="float64"),
        np.asarray(stack, dtype="float64"),
    )


def rainflow_3point(tp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Three-point (ASTM E1049) rainflow on a turning-point series.

    Returns ``(ranges, means, counts, residue)`` where ``counts`` is 1.0 for a closed cycle and
    0.5 for a half cycle extracted from the head of the history, exactly as E1049 prescribes.
    """
    stack: list[float] = []
    ranges: list[float] = []
    means: list[float] = []
    counts: list[float] = []
    for value in tp.tolist():
        stack.append(value)
        while len(stack) >= 3:
            s1, s2, s3 = stack[-3], stack[-2], stack[-1]
            r_prev = abs(s1 - s2)
            r_next = abs(s2 - s3)
            if r_prev > r_next:
                break
            if len(stack) == 3:
                # the range contains the start of the history -> half cycle, drop the first point
                ranges.append(r_prev)
                means.append(0.5 * (s1 + s2))
                counts.append(0.5)
                del stack[0]
            else:
                ranges.append(r_prev)
                means.append(0.5 * (s1 + s2))
                counts.append(1.0)
                del stack[-3:-1]
    return (
        np.asarray(ranges, dtype="float64"),
        np.asarray(means, dtype="float64"),
        np.asarray(counts, dtype="float64"),
        np.asarray(stack, dtype="float64"),
    )


def _residue_cycles(residue: np.ndarray, convention: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn an unclosed rainflow residue into (ranges, means, counts) per a residue convention."""
    convention = str(convention)
    if residue.size < 2 or convention == "discard":
        empty = np.zeros(0, dtype="float64")
        return empty, empty, empty
    if convention == "half":
        r = np.abs(np.diff(residue))
        m = 0.5 * (residue[:-1] + residue[1:])
        return r, m, np.full(r.size, 0.5)
    if convention == "full":
        r = np.abs(np.diff(residue))
        m = 0.5 * (residue[:-1] + residue[1:])
        return r, m, np.ones(r.size)
    if convention == "close":
        # "repeat history": rainflow the residue concatenated with itself, keep the new cycles.
        doubled = np.concatenate([residue, residue])
        r, m, _ = rainflow_4point(turning_points(doubled))
        return r, m, np.full(r.size, 0.5)
    raise ValueError(f"unknown residue convention {convention!r} (discard|half|full|close)")


def rainflow_cycles(
    x: np.ndarray,
    *,
    method: str = "4point",
    residue: str = "half",
    tp: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rainflow cycles of a signal: ``(ranges, means, counts)``.

    ``method`` is ``4point`` (fatpack-style) or ``3point`` (ASTM E1049 [R290]); ``residue`` is one
    of ``discard | half | full | close``. The residue convention materially changes the damage sum
    and the common conventions are non-conservative [R270], which is why it is a switch rather
    than a constant - and on this dataset it is the single lever that makes the label reproduce.
    """
    tp = turning_points(x) if tp is None else tp
    if method == "4point":
        r, m, res = rainflow_4point(tp)
        c = np.ones(r.size, dtype="float64")
    elif method == "3point":
        r, m, c, res = rainflow_3point(tp)
    else:
        raise ValueError(f"unknown rainflow method {method!r} (4point|3point)")
    r_res, m_res, c_res = _residue_cycles(res, residue)
    if r_res.size:
        r = np.concatenate([r, r_res])
        m = np.concatenate([m, m_res])
        c = np.concatenate([c, c_res])
    return r, m, c


def miner_damage(
    ranges: np.ndarray,
    counts: np.ndarray,
    *,
    exponent: float,
    coefficient: float = 1.0,
    endurance: float = 0.0,
    haibach: bool = False,
) -> float:
    """Palmgren-Miner sum for amplitude-based S-N ``sigma_a^m * N = C``.

    ``endurance`` is the amplitude knee: below it cycles are dropped (original Miner) or, with
    ``haibach``, kept on the shallower slope ``k* = 2k - 1``. **The Haibach rule is [R291], flagged
    SECONDARY-SOURCE in the reference list**, so it is swept in the diagnostic only and must not
    reach a shipped model until someone reads a primary source; nothing in the ladder uses it.
    """
    amp = 0.5 * np.asarray(ranges, dtype="float64")
    n = np.asarray(counts, dtype="float64")
    if endurance <= 0:
        return float(np.sum(n * np.power(amp, exponent)) / coefficient)
    above = amp >= endurance
    total = float(np.sum(n[above] * np.power(amp[above], exponent)))
    if haibach:
        k2 = 2.0 * exponent - 1.0
        below = ~above
        total += float(np.sum(n[below] * np.power(endurance, exponent - k2) * np.power(amp[below], k2)))
    return total / coefficient


# --------------------------------------------------------------------------------------
# Family 1: cheap amplitude / count statistics (the baseline 20)
# --------------------------------------------------------------------------------------

_BLOCK_COUNTS: tuple[int, ...] = (5, 10, 20, 50, 100, 200)
_LEVELS: tuple[int, ...] = (1, 2, 3)


def block_maxima(x: np.ndarray, n_blocks: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-block max and min of ``x`` split into ``n_blocks`` contiguous equal blocks."""
    n = x.size // n_blocks
    if n < 1:
        return np.asarray([x.max()]), np.asarray([x.min()])
    view = x[: n * n_blocks].reshape(n_blocks, n)
    return view.max(axis=1), view.min(axis=1)


def stats_features(x: np.ndarray) -> dict[str, float]:
    """20 amplitude / count statistics. Everything here is positive by construction except skew."""
    med = float(np.median(x))
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    d = np.diff(x)
    centred = x - float(x.mean())
    var = float(np.mean(centred**2))
    kurt = float(np.mean(centred**4) / var**2) if var > 0 else 0.0
    skew = float(np.mean(centred**3) / var**1.5) if var > 0 else 0.0
    tp = turning_points(x)
    out: dict[str, float] = {
        "st_p2p": float(x.max() - x.min()),
        "st_max_dev": float(x.max() - med),
        "st_min_dev": float(med - x.min()),
        "st_std": std,
        "st_kurtosis": kurt,
        "st_roughness": float(np.mean(np.abs(d))) if d.size else 0.0,
        "st_diff_std": float(d.std(ddof=1)) if d.size > 1 else 0.0,
        "st_turning_points": float(max(tp.size, 1)),
        "st_p99_p01": float(np.quantile(x, 0.99) - np.quantile(x, 0.01)),
        "st_p995_p005": float(np.quantile(x, 0.995) - np.quantile(x, 0.005)),
    }
    _log_names(
        "st_p2p", "st_max_dev", "st_min_dev", "st_std", "st_kurtosis", "st_roughness",
        "st_diff_std", "st_turning_points", "st_p99_p01", "st_p995_p005",
    )
    for nb in _BLOCK_COUNTS:
        hi, lo = block_maxima(x, nb)
        out[f"st_blockmax_p2p_{nb}"] = float(np.mean(hi - lo))
        _log_names(f"st_blockmax_p2p_{nb}")
    for c in _LEVELS:
        level = med + c * std
        up = int(np.count_nonzero((x[:-1] < level) & (x[1:] >= level))) if x.size > 1 else 0
        out[f"st_upcross_{c}sd"] = float(max(up, 1))
        _log_names(f"st_upcross_{c}sd")
    out["st_skew"] = skew  # signed: never log-transformed
    return out


# --------------------------------------------------------------------------------------
# Family 2: rainflow
# --------------------------------------------------------------------------------------

_RANGE_QUANTILES: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 0.99)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w)
    if cum[-1] <= 0:
        return 0.0
    return float(np.interp(q * cum[-1], cum, v))


def rainflow_features(
    x: np.ndarray,
    *,
    method: str = "4point",
    residue: str = "half",
    tp: np.ndarray | None = None,
) -> dict[str, float]:
    """Rainflow range statistics, ``sum n*sigma^m`` over the exponent grid, DELs, Goodman/SWT."""
    ranges, means, counts = rainflow_cycles(x, method=method, residue=residue, tp=tp)
    out: dict[str, float] = {}
    if ranges.size == 0:
        ranges = np.zeros(1)
        means = np.zeros(1)
        counts = np.zeros(1)
    amp = 0.5 * ranges
    n_tot = float(counts.sum())
    out["rf_n_cycles"] = max(n_tot, 1.0)
    out["rf_range_max"] = float(ranges.max())
    out["rf_amp_rms"] = float(np.sqrt(np.sum(counts * amp**2) / max(n_tot, _TINY)))
    _log_names("rf_n_cycles", "rf_range_max", "rf_amp_rms")
    for q in _RANGE_QUANTILES:
        name = f"rf_range_p{int(q * 100)}"
        out[name] = max(_weighted_quantile(ranges, counts, q), _TINY)
        _log_names(name)
    # sum n * sigma_a^m over the S-N exponent grid (stored as the log, they span 40 decades)
    for m in SN_EXPONENTS:
        s = float(np.sum(counts * np.power(amp, float(m))))
        out[f"rf_logsum_m{m}"] = math.log(max(s, _TINY))          # already a log: never re-logged
        out[f"rf_del_m{m}"] = float((s / max(n_tot, _TINY)) ** (1.0 / m))
        _log_names(f"rf_del_m{m}")
    # mean-stress corrections. Goodman uses the fixed reference SIGMA_U (data independent);
    # SWT is parameter free.
    sigma_max = means + amp
    good = amp / np.clip(1.0 - means / SIGMA_U, 0.05, None)
    swt = np.sqrt(np.clip(sigma_max, 0.0, None) * amp)
    for m in DAMAGE_EXPONENTS:
        out[f"rf_goodman_logsum_m{m}"] = math.log(max(float(np.sum(counts * np.power(good, float(m)))), _TINY))
        out[f"rf_swt_logsum_m{m}"] = math.log(max(float(np.sum(counts * np.power(swt, float(m)))), _TINY))
    out["rf_mean_stress_mean"] = float(np.average(means, weights=np.clip(counts, _TINY, None)))
    out["rf_mean_stress_std"] = float(means.std())
    _log_names("rf_mean_stress_std")
    return out


# --------------------------------------------------------------------------------------
# Family 3: spectral moments and frequency-domain damage
# --------------------------------------------------------------------------------------


def spectral_moments(freq: np.ndarray, psd: np.ndarray, orders: Sequence[float]) -> dict[float, float]:
    keep = freq > 0
    f = freq[keep]
    g = psd[keep]
    return {float(o): float(np.trapezoid(g * np.power(f, float(o)), f)) for o in orders}


def _dirlik_moment(m0: float, m1: float, m2: float, m4: float, k: float) -> float:
    """E[S^k] under the Dirlik rainflow-range pdf (S = range); hand-implemented from [R269][R276]."""
    if m0 <= 0 or m2 <= 0 or m4 <= 0:
        return 0.0
    gamma = m2 / math.sqrt(m0 * m4)
    xm = (m1 / m0) * math.sqrt(m2 / m4)
    denom = 1.0 + gamma**2
    d1 = 2.0 * (xm - gamma**2) / denom if denom else 0.0
    d1 = min(max(d1, 1e-6), 1.0)
    r_den = 1.0 - gamma - d1 + d1**2
    r = (gamma - xm - d1**2) / r_den if abs(r_den) > 1e-12 else 0.5
    r = min(max(r, 1e-6), 0.999999)
    d2 = (1.0 - gamma - d1 + d1**2) / (1.0 - r)
    d2 = min(max(d2, 0.0), 1.0)
    d3 = max(1.0 - d1 - d2, 0.0)
    q = 1.25 * (gamma - d3 - d2 * r) / d1 if d1 > 0 else 1.0
    q = max(q, 1e-6)
    scale = 2.0 * math.sqrt(m0)
    term = (
        d1 * q**k * math.gamma(1.0 + k)
        + (math.sqrt(2.0) ** k) * math.gamma(1.0 + 0.5 * k) * (d2 * abs(r) ** k + d3)
    )
    return (scale**k) * term


def _tb_correction(alpha1: float, alpha2: float, k: float) -> float:
    """Tovo-Benasciutti (TB2) bandwidth factor on narrow-band damage; from [R275][R276][R269]."""
    den = (alpha2 - 1.0) ** 2
    if den < 1e-12:
        return 1.0
    b = ((alpha1 - alpha2) * (1.112 * (1.0 + alpha1 * alpha2 - (alpha1 + alpha2)) * math.exp(2.11 * alpha2) + (alpha1 - alpha2))) / den
    b = min(max(b, 0.0), 1.0)
    return b + (1.0 - b) * alpha2 ** (k - 1.0)


def spectral_features(x: np.ndarray, *, fs: float = 1.0, nperseg: int = 8192) -> dict[str, float]:
    """Welch PSD -> moments, bandwidth parameters and narrow-band / Dirlik / TB damage [R269]."""
    from scipy import signal as sp_signal

    nperseg = int(min(nperseg, x.size))
    freq, psd = sp_signal.welch(x - x.mean(), fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
    mom = spectral_moments(freq, psd, (0.0, 0.75, 1.0, 1.5, 2.0, 4.0))
    m0, m075, m1, m15, m2, m4 = (mom[0.0], mom[0.75], mom[1.0], mom[1.5], mom[2.0], mom[4.0])
    m0 = max(m0, _TINY)
    out: dict[str, float] = {
        "sp_m0": m0,
        "sp_m1": max(m1, _TINY),
        "sp_m2": max(m2, _TINY),
        "sp_m4": max(m4, _TINY),
        "sp_rms": math.sqrt(m0),
    }
    _log_names("sp_m0", "sp_m1", "sp_m2", "sp_m4", "sp_rms")
    nu0 = math.sqrt(max(m2, _TINY) / m0)
    nup = math.sqrt(max(m4, _TINY) / max(m2, _TINY))
    alpha2 = m2 / math.sqrt(m0 * max(m4, _TINY))
    alpha1 = m1 / math.sqrt(m0 * max(m2, _TINY))
    alpha075 = m075 / math.sqrt(m0 * max(m15, _TINY))
    out["sp_nu0"] = max(nu0, _TINY)
    out["sp_nup"] = max(nup, _TINY)
    out["sp_alpha2"] = min(max(alpha2, _TINY), 1.0)
    out["sp_alpha1"] = min(max(alpha1, _TINY), 1.0)
    out["sp_alpha075"] = min(max(alpha075, _TINY), 1.0)
    out["sp_vanmarcke"] = max(math.sqrt(max(1.0 - alpha1**2, 0.0)), _TINY)
    out["sp_centroid"] = max(float(np.trapezoid(psd * freq, freq) / max(float(np.trapezoid(psd, freq)), _TINY)), _TINY)
    _log_names("sp_nu0", "sp_nup", "sp_alpha2", "sp_alpha1", "sp_alpha075", "sp_vanmarcke", "sp_centroid")
    duration = x.size / fs
    for k in DAMAGE_EXPONENTS:
        kf = float(k)
        nb = nu0 * duration * (math.sqrt(2.0) * math.sqrt(m0)) ** kf * math.gamma(1.0 + 0.5 * kf)
        out[f"sp_nb_logdam_k{k}"] = math.log(max(nb, _TINY))
        es_k = _dirlik_moment(m0, m1, m2, m4, kf)
        dirlik = nup * duration * es_k / (2.0**kf)      # E[sigma_a^k] = E[S^k] / 2^k
        out[f"sp_dirlik_logdam_k{k}"] = math.log(max(dirlik, _TINY))
        out[f"sp_tb_logdam_k{k}"] = math.log(max(nb * _tb_correction(alpha1, alpha2, kf), _TINY))
    return out


# --------------------------------------------------------------------------------------
# Family 4: fatigue damage spectrum (SDOF filter bank, evaluated on the PSD)
# --------------------------------------------------------------------------------------

#: Octave-spaced SDOF natural frequencies as a fraction of the sample rate (fs unknown).
FDS_BANDS: tuple[float, ...] = tuple(0.25 / (2.0**j) for j in range(11))
FDS_Q = 10.0


def fds_features(x: np.ndarray, *, fs: float = 1.0, nperseg: int = 8192, exponent: float = 5.0) -> dict[str, float]:
    """Fatigue damage spectrum: band RMS and narrow-band band damage for an SDOF bank (Q = 10).

    The response of a single-degree-of-freedom oscillator is evaluated in the frequency domain
    (``sigma_j^2 = integral |H_j(f)|^2 G(f) df``), so no per-band rainflow is needed - the whole
    bank costs one Welch PSD. This is the FDS construction of [R274], the standard fix when the
    label was computed on an SDOF-transformed channel rather than the raw one. The addendum makes
    it a MUST here because the diagnostic residual does track the irregularity factor
    (``results/ps3/shm_diagnostic.md``).
    """
    from scipy import signal as sp_signal

    nperseg = int(min(nperseg, x.size))
    freq, psd = sp_signal.welch(x - x.mean(), fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
    zeta = 1.0 / (2.0 * FDS_Q)
    duration = x.size / fs
    out: dict[str, float] = {}
    for j, ratio in enumerate(FDS_BANDS):
        fn = ratio * fs
        r = np.divide(freq, fn, out=np.zeros_like(freq), where=fn > 0)
        h2 = 1.0 / ((1.0 - r**2) ** 2 + (2.0 * zeta * r) ** 2)
        band_psd = psd * h2
        m0 = max(float(np.trapezoid(band_psd, freq)), _TINY)
        m2 = max(float(np.trapezoid(band_psd * freq**2, freq)), _TINY)
        nu0 = math.sqrt(m2 / m0)
        out[f"fds_b{j}_rms"] = math.sqrt(m0)
        dam = nu0 * duration * (math.sqrt(2.0 * m0)) ** exponent * math.gamma(1.0 + 0.5 * exponent)
        out[f"fds_b{j}_logdam"] = math.log(max(dam, _TINY))
        _log_names(f"fds_b{j}_rms")
    return out


# --------------------------------------------------------------------------------------
# Assembly, caching, log space
# --------------------------------------------------------------------------------------


def features_for_signal(x: np.ndarray, families: Sequence[str] = FAMILIES) -> dict[str, float]:
    """All requested feature families for one already-loaded signal."""
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown SHM feature families {sorted(unknown)}; known: {list(FAMILIES)}")
    out: dict[str, float] = {}
    tp = turning_points(x) if "rainflow" in families else None
    if "stats" in families:
        out.update(stats_features(x))
    if "rainflow" in families:
        out.update(rainflow_features(x, tp=tp))
    if "spectral" in families:
        out.update(spectral_features(x))
    if "fds" in families:
        out.update(fds_features(x))
    return out


def _cache_dir(cache_dir: Path | str | None = None) -> Path:
    base = Path(cache_dir) if cache_dir is not None else Path(CACHE_DIR) / "shm"
    return base / FEATURE_VERSION


def features_for_file(
    path: Path | str,
    families: Sequence[str] = FAMILIES,
    *,
    cache_dir: Path | str | None = None,
    use_cache: bool = True,
) -> dict[str, float]:
    """Features for one file, memoised to ``data/ps3_cache/shm/<FEATURE_VERSION>/<name>.parquet``."""
    path = Path(path)
    cache = _cache_dir(cache_dir) / f"{path.name}.parquet"
    if use_cache and cache.exists():
        try:
            row = pd.read_parquet(cache).iloc[0].to_dict()
            if all(c in row for c in _expected_columns(families)):
                return {"file_id": path.name, **{k: v for k, v in row.items() if k != "file_id"}}
        except Exception:  # pragma: no cover - a corrupt cache entry just gets recomputed
            pass
    x = load_signal(path)
    feats = features_for_signal(x, FAMILIES)
    row = {"file_id": path.name, **feats}
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_parquet(cache, index=False)
    return row


_EXPECTED: dict[tuple[str, ...], list[str]] = {}


def _expected_columns(families: Sequence[str]) -> list[str]:
    key = tuple(families)
    if key not in _EXPECTED:
        probe = np.sin(np.linspace(0, 40 * np.pi, 4096)) + 0.1 * np.cos(np.linspace(0, 700 * np.pi, 4096))
        _EXPECTED[key] = list(features_for_signal(probe, families))
    return _EXPECTED[key]


def family_columns(columns: Iterable[str], families: Sequence[str]) -> list[str]:
    """Subset of ``columns`` belonging to the given families (prefix based)."""
    prefixes = {"stats": "st_", "rainflow": "rf_", "spectral": "sp_", "fds": "fds_"}
    wanted = tuple(prefixes[f] for f in families)
    return [c for c in columns if c.startswith(wanted)]


def feature_table(
    paths: Sequence[Path | str],
    families: Sequence[str] = FAMILIES,
    *,
    n_jobs: int = 4,
    cache_dir: Path | str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Feature rows for a list of files, one row per file, ``file_id`` first."""
    from joblib import Parallel, delayed

    paths = [Path(p) for p in paths]
    if n_jobs == 1 or len(paths) == 1:
        rows = [features_for_file(p, families, cache_dir=cache_dir, use_cache=use_cache) for p in paths]
    else:
        rows = Parallel(n_jobs=min(n_jobs, len(paths)), backend="loky")(
            delayed(features_for_file)(p, families, cache_dir=cache_dir, use_cache=use_cache) for p in paths
        )
    frame = pd.DataFrame(list(rows))
    cols = ["file_id"] + [c for c in frame.columns if c != "file_id"]
    return frame[cols]


def log_feature_names() -> frozenset[str]:
    """Names that are positive by construction and therefore log-transformed."""
    _expected_columns(FAMILIES)
    return frozenset(_LOG_FEATURES)


def to_log_space(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace every positive-by-construction column with its natural log.

    The choice of which column is logged is fixed by the *definition* of the feature, not by the
    values in a fold, so this transform carries no fold-local information.
    """
    logs = log_feature_names()
    out = frame.copy()
    for col in out.columns:
        if col in logs:
            out[col] = np.log(np.clip(out[col].to_numpy(dtype="float64"), _TINY, None))
    return out


LOG_FEATURES = log_feature_names()
