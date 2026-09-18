"""Envelope-spectrum bearing features: the canonical order-tracking-free recipe [rail_phm
4.3.2/4.3.4, R117/R122] - kurtogram band selection -> bandpass -> Hilbert envelope -> FFT of
the envelope -> read off the single-sided amplitude (``2|E|/n``, m/s2) at the BPFO/BPFI/BSF/FTF
characteristic frequencies and their harmonics.

This is deliberately a *simple* fast kurtogram (whole-signal FFT masking per candidate band,
not a proper recursive FIR filter-bank), traded for being short and auditable at our window
lengths (seconds, not the minutes-long acquisitions the original kurtogram targets). It is
the same module for Ottawa raw vibration and for the synthetic bearing waveform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np
from scipy.signal import hilbert

__all__ = ["BearingGeometry", "fast_kurtogram", "envelope_spectrum_feats"]

#: Fault-frequency name -> number of harmonics to read off, per [rail_phm 4.3.4].
_HARMONICS: Final[dict[str, int]] = {"bpfo": 3, "bpfi": 3, "bsf": 2, "ftf": 1}


@dataclass(frozen=True, slots=True)
class BearingGeometry:
    """Rolling-element bearing geometry needed for the four characteristic-frequency factors.

    ``n_elements`` = Z (rolling element count), ``ball_diameter_m`` = Bd, ``pitch_diameter_m``
    = Dp, ``contact_angle_deg`` = the ball/roller contact angle (0 for a pure radial bearing).
    """

    n_elements: int
    ball_diameter_m: float
    pitch_diameter_m: float
    contact_angle_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.n_elements <= 0:
            raise ValueError(f"BearingGeometry: n_elements must be > 0, got {self.n_elements}")
        if self.ball_diameter_m <= 0 or self.pitch_diameter_m <= 0:
            raise ValueError("BearingGeometry: ball_diameter_m and pitch_diameter_m must be > 0")

    def factors(self) -> dict[str, float]:
        """Per-shaft-revolution multipliers: multiply by the shaft frequency ``fr`` (Hz) to
        get BPFO/BPFI/BSF/FTF in Hz."""
        ratio = self.ball_diameter_m / self.pitch_diameter_m
        cos_phi = math.cos(math.radians(self.contact_angle_deg))
        z = float(self.n_elements)
        return {
            "bpfo": 0.5 * z * (1.0 - ratio * cos_phi),
            "bpfi": 0.5 * z * (1.0 + ratio * cos_phi),
            "bsf": (1.0 / (2.0 * ratio)) * (1.0 - (ratio * cos_phi) ** 2),
            "ftf": 0.5 * (1.0 - ratio * cos_phi),
        }


def _clean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, np.nan)
    if np.isnan(x).any():
        x = np.where(np.isnan(x), np.nanmean(x) if np.isfinite(np.nanmean(x)) else 0.0, x)
    return x - x.mean()


def _bandpass_fft(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> np.ndarray:
    """Zero-phase FFT-mask bandpass: cheap and exact-enough for picking a demodulation band."""
    n = x.size
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return np.fft.irfft(np.where(mask, X, 0.0), n=n)


def _kurtosis_1d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    d = x - x.mean()
    var = float(np.mean(d * d))
    if var <= 0:
        return float("nan")
    return float(np.mean(d**4) / (var * var))


def fast_kurtogram(
    x: np.ndarray, fs: float, *, max_level: int = 4, min_bw_hz: float = 5.0
) -> tuple[float, float, float]:
    """Simplified fast kurtogram: returns ``(center_freq_hz, bandwidth_hz, kurtosis)`` of the
    dyadic sub-band whose Hilbert-envelope kurtosis is highest (the most impulsive band).

    Searches levels ``1..max_level`` (``2**level`` equal-width bands per level over
    ``[0, fs/2]``); stops early once a band would be narrower than ``min_bw_hz``.
    """
    if fs <= 0:
        raise ValueError(f"fast_kurtogram: fs must be > 0, got {fs}")
    xc = _clean(x)
    nyq = fs / 2.0
    best = (nyq / 2.0, nyq, _kurtosis_1d(np.abs(hilbert(xc))))
    for level in range(1, max_level + 1):
        n_bands = 2**level
        bw = nyq / n_bands
        if bw < min_bw_hz:
            break
        for i in range(n_bands):
            f_lo = i * bw
            f_hi = f_lo + bw
            band = _bandpass_fft(xc, fs, f_lo, f_hi)
            k = _kurtosis_1d(np.abs(hilbert(band)))
            if np.isfinite(k) and k > best[2]:
                best = (f_lo + bw / 2.0, bw, k)
    return best


def envelope_spectrum_feats(
    x: np.ndarray,
    fs: float,
    rpm_or_speed: float,
    geometry: BearingGeometry,
    *,
    is_speed_ms: bool = False,
    wheel_radius_m: float = 0.425,
    harmonics: Mapping[str, int] | None = None,
    max_level: int = 4,
) -> dict[str, float]:
    """Envelope-spectrum bearing features for one raw vibration segment ``x``.

    Parameters
    ----------
    x : 1-D raw vibration signal (one window/segment).
    fs : sample rate, Hz.
    rpm_or_speed : shaft rotational speed in RPM (default), or train speed in m/s when
        ``is_speed_ms=True`` - converted to shaft Hz via ``v / (2*pi*wheel_radius_m)``
        (``wheel_radius_m=0.425`` per the plan's bogie geometry).
    geometry : :class:`BearingGeometry`.
    harmonics : override of :data:`_HARMONICS` (fault-frequency name -> harmonic count).
    max_level : passed to :func:`fast_kurtogram`.

    Returns a flat ``dict`` with ``env_<name>_h<k>`` (envelope-spectrum **amplitude** at each
    characteristic-frequency harmonic, in m/s2), ``env_<name>_energy`` (sum of that family's
    harmonic amplitudes - the key name is historical, the quantity is a summed amplitude, and
    it is the one ``docs/parameters.md`` calls "BPFO amp"), ``env_<name>_freq_hz``,
    time-domain ``rms``/``kurtosis``/``crest`` on the raw signal, and the kurtogram-selected
    band ``sk_band_fc``/``sk_band_bw``/``sk_max``.

    **Scaling.** The harmonic readings are single-sided amplitudes ``2|E_k|/n`` of the
    mean-removed Hilbert envelope of the bandpassed signal (DC/Nyquist bins keep ``|E_k|/n``),
    so they carry the envelope's own units - m/s2 for an acceleration input - and are
    independent of window length. They are *not* ``|E_k|**2/n`` FFT power, which is what this
    function used to return under the same names while every document called them amplitudes;
    a length-independent amplitude is also what makes windows of different duration (Ottawa's
    0.25 s and 1 s tiles) comparable in one feature table.

    The *envelope* features are immune to a constant sensor bias (:func:`_clean` mean-removes
    before the kurtogram and the bandpass), but the three time-domain extras ``rms``/``crest``
    are **DC-coupled** - ``sqrt(mean(x**2))`` and ``max|x|/rms`` on the raw window - so on a
    DC-biased accelerometer they carry the bias (``kurtosis`` is central and does not). Use
    ``nebulax.features.stats.window_stats(..., ac_couple=True)`` when you need the AC-coupled
    pair; that is what :mod:`nebulax.adapters.ottawa` does, skipping these three keys.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 4:
        raise ValueError(f"envelope_spectrum_feats: x has {x.size} samples, need >= 4")
    fr_hz = (rpm_or_speed / (2.0 * math.pi * wheel_radius_m)) if is_speed_ms else (rpm_or_speed / 60.0)
    if fr_hz <= 0:
        raise ValueError(f"envelope_spectrum_feats: derived shaft frequency {fr_hz} Hz must be > 0")

    harm = dict(_HARMONICS if harmonics is None else harmonics)
    factors = geometry.factors()
    char_freq_hz = {name: factors[name] * fr_hz for name in factors}

    fc, bw, sk = fast_kurtogram(x, fs, max_level=max_level)
    xc = _clean(x)
    band = _bandpass_fft(xc, fs, max(fc - bw / 2.0, 0.0), fc + bw / 2.0)
    env = np.abs(hilbert(band))
    env = env - env.mean()

    n = env.size
    E = np.fft.rfft(env)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # Single-sided amplitude spectrum in m/s2: a sinusoid of amplitude A in ``env`` reads A
    # here, whatever the window length. The DC and (for even n) Nyquist bins are not mirrored
    # by a negative-frequency twin, so they keep the 1/n scaling instead of 2/n.
    amp = 2.0 * np.abs(E) / n
    amp[0] = np.abs(E[0]) / n
    if n % 2 == 0:
        amp[-1] = np.abs(E[-1]) / n
    bin_hz = fs / n
    tol_hz = max(fr_hz * 0.05, bin_hz)

    def _peak_near(f0: float) -> float:
        if f0 <= 0 or f0 >= freqs[-1]:
            return 0.0
        mask = np.abs(freqs - f0) <= tol_hz
        return float(amp[mask].max()) if mask.any() else 0.0

    out: dict[str, float] = {}
    for name, f0 in char_freq_hz.items():
        n_h = harm.get(name, 1)
        for h in range(1, n_h + 1):
            out[f"env_{name}_h{h}"] = _peak_near(f0 * h)
        out[f"env_{name}_energy"] = float(sum(out[f"env_{name}_h{h}"] for h in range(1, n_h + 1)))
        out[f"env_{name}_freq_hz"] = float(f0)

    finite = x[np.isfinite(x)]
    rms = float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")
    out["rms"] = rms
    out["kurtosis"] = _kurtosis_1d(finite) if finite.size else float("nan")
    out["crest"] = float(np.max(np.abs(finite)) / rms) if finite.size and rms > 0 else float("nan")
    out["sk_band_fc"] = float(fc)
    out["sk_band_bw"] = float(bw)
    out["sk_max"] = float(sk)
    out["shaft_hz"] = float(fr_hz)
    return out
