"""Rail corrugation feature extraction: tacho -> distance -> wavelength bands.

One expensive pass over the organisers' 5.5 GB of 10 kHz axle-box recordings produces, per file:

* file scalars - speed from the 90-tooth pulse train, its spread over 0.1 s windows, a duplicate
  fingerprint;
* per-channel scalars (128 channels = 64 axle boxes x {vibration, shock}) - log RMS, kurtosis,
  log peak, crest factor, seven log-spaced Hz band levels (20-5000 Hz, the *counter-design*
  arm), and four impulsive-vs-sustained discriminators;
* a per-channel **wavelength** PSD on a fixed log-spaced lambda grid (8-500 mm), obtained by
  resampling every channel from time to distance with the tacho as a keyphasor and taking Welch
  in the spatial-frequency domain.

Why wavelength and not Hz (`docs/research/ps3_addendum.md` section 2.2): the wheel is 0.85 m
across, so pi*0.85 / 90 = **29.67 mm of travel per tacho pulse** and the cumulative pulse count
*is* a distance encoder. Short-pitch metro corrugation (lambda = 25-80 mm on metro tangent track
[R248]) therefore sweeps 35-744 Hz over the dataset's 0-67 km/h, so a fixed Hz band cannot
isolate it, while a fixed *wavelength* band can [R231][R233][R239][R245][R250]. Axle-box
acceleration for fixed roughness scales as v^2 [R237], so every level is stored in dB and
`v2_normalise=True` simply subtracts `40*log10(v)` at aggregation time - the with/without-v^2
ablation costs nothing.

Everything downstream (side aggregation, the Side I - Side II contrast, the exact mirror
augmentation, band integration, every ablation arm) is a cheap function of the cached per-channel
arrays, so the 5.5 GB is read exactly once per feature version.

Layout of the cache (`data/ps3_cache/rail/<version>/`, gitignored):

    <name>.parquet   one row: file scalars + 128 x 15 flattened per-channel scalars
    <name>.spec.npy  float32 (128, N_LAMBDA) wavelength PSD in dB

plus the combined `feats_<version>.parquet` / `spec_<version>.npy` written by
:func:`build_feature_cache`.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import signal

from nebulax.ps3.common import CACHE_DIR, natural_key

__all__ = [
    "FEATURE_VERSION",
    "FS_HZ",
    "WHEEL_DIAMETER_M",
    "TEETH",
    "PULSE_DISTANCE_M",
    "N_CHANNELS",
    "N_BOXES",
    "LAMBDA_GRID_MM",
    "THIRD_OCTAVE_CENTRES_MM",
    "HZ_BAND_EDGES",
    "CHANNEL_SCALARS",
    "MIRROR_PERM",
    "side_mask",
    "read_rail_csv",
    "speed_from_pulse",
    "pulse_distance",
    "resample_to_distance",
    "wavelength_psd_db",
    "extract_file",
    "extract_array",
    "cache_dir",
    "build_feature_cache",
    "load_feature_cache",
    "RailFeatures",
    "band_levels_db",
    "aggregate",
    "mirror",
    "perturb_fault_channels",
    "AGG_DEFAULT",
    "coherence_for_array",
    "coherence_for_path",
    "mirror_coherence",
]

# --------------------------------------------------------------------------------------
# Physics and grid constants (Rail Info Kit section 2.1; `docs/research/ps3_addendum.md` 2.2)
# --------------------------------------------------------------------------------------

#: Bump when the cached arrays change shape or meaning; the cache path carries it.
FEATURE_VERSION = "v3"

FS_HZ: float = 10_000.0
WHEEL_DIAMETER_M: float = 0.85
TEETH: int = 90
#: pi * 0.85 / 90 = 29.67 mm of travel per tacho pulse.
PULSE_DISTANCE_M: float = float(np.pi * WHEEL_DIAMETER_M / TEETH)

N_BOXES: int = 64  # 8 cars x 8 positions
N_CHANNELS: int = 128  # each box contributes a vibration and a shock channel

#: Uniform spatial sampling step for the distance-resampled signal (metres).
DX_M: float = 0.001

#: Fixed log-spaced wavelength grid the per-channel PSD is interpolated onto (mm).
LAMBDA_MIN_MM: float = 8.0
LAMBDA_MAX_MM: float = 500.0
N_LAMBDA: int = 192
LAMBDA_GRID_MM: np.ndarray = np.geomspace(LAMBDA_MIN_MM, LAMBDA_MAX_MM, N_LAMBDA)

#: 1/3-octave wavelength band centres over 8-500 mm (IEC 61373 / EN 15610 presentation [R233][R250]).
THIRD_OCTAVE_CENTRES_MM: np.ndarray = LAMBDA_MIN_MM * 2.0 ** (np.arange(18) / 3.0)

#: Seven log-spaced Hz bands - the fixed-frequency counter-design arm only [R234].
HZ_BAND_EDGES: np.ndarray = np.geomspace(20.0, 5000.0, 8)

#: Short-pitch corrugation band for the envelope duty-cycle discriminator [R235][R246][R248].
SHORT_PITCH_MM: tuple[float, float] = (25.0, 80.0)

#: Per-channel scalar features, in the order they are flattened into the parquet.
CHANNEL_SCALARS: tuple[str, ...] = (
    "logrms",
    "logrms_clip95",
    "logrms_win_p10",
    "kurtosis",
    "logpeak",
    "crest",
    "skew",
    "margin",
    "pulse",
    "waveform",
    *[f"hz{i}" for i in range(len(HZ_BAND_EDGES) - 1)],
    "spec_centroid",
    "spec_sd",
    "wl_flatness",
    "wl_peak_prom",
    "wl_peak_lambda",
    "env_duty",
)

#: Channel index -> mirrored channel index. Positions pair (1,2), (3,4), (5,6), (7,8) and odd
#: positions are Side I, so ``ch ^ 1`` is the exact left-right mirror of the sensor layout.
MIRROR_PERM: np.ndarray = np.arange(N_BOXES) ^ 1


def side_mask(side: str) -> np.ndarray:
    """Boolean mask over the 64 axle boxes for ``"I"`` (positions 1,3,5,7) or ``"II"``."""
    pos0 = np.arange(N_BOXES) % 8  # 0-based position within the car
    if str(side) == "I":
        return pos0 % 2 == 0
    if str(side) == "II":
        return pos0 % 2 == 1
    raise ValueError(f"side must be 'I' or 'II', got {side!r}")


# --------------------------------------------------------------------------------------
# Loading and the tachometer
# --------------------------------------------------------------------------------------


def read_rail_csv(path: Path | str) -> np.ndarray:
    """Read one rail csv into ``float32 (n_samples, 129)``; column 0 is the 0/1 tacho pulse.

    Raises a clear error for an empty file or one with the wrong column count, which is what the
    upload path in the app hits when someone drops the wrong csv.
    """
    p = Path(path)
    try:
        df = pd.read_csv(p, dtype=np.float32)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{p.name}: empty rail file (expected a header and 10,000 rows)") from exc
    if df.shape[1] != N_CHANNELS + 1:
        raise ValueError(
            f"{p.name}: rail files have {N_CHANNELS + 1} columns "
            f"(rotating speed + 64 axle boxes x vibration/shock), got {df.shape[1]}"
        )
    if len(df) < 100:
        raise ValueError(f"{p.name}: only {len(df)} rows; a rail recording is 1 s at 10 kHz")
    arr = df.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _pulse_edges(pulse: np.ndarray) -> np.ndarray:
    """Sample indices where the 0/1 tacho toggles (each tooth gives two)."""
    b = (np.asarray(pulse, dtype=np.float32) > 0.5).astype(np.int8)
    return np.flatnonzero(np.diff(b) != 0) + 1


def speed_from_pulse(pulse: np.ndarray, *, fs: float = FS_HZ) -> float:
    """Mean speed in km/h over the record: ``transitions / 2 / 90 * pi * 0.85 * 3.6``."""
    edges = _pulse_edges(pulse)
    seconds = len(pulse) / float(fs)
    if seconds <= 0:
        return 0.0
    return float(len(edges) / 2.0 * PULSE_DISTANCE_M / seconds * 3.6)


def _windowed_speed(pulse: np.ndarray, *, fs: float = FS_HZ, window_s: float = 0.1) -> np.ndarray:
    """Speed (km/h) in consecutive ``window_s`` windows - the acceleration detector."""
    n = int(round(window_s * fs))
    if n <= 0 or len(pulse) < n:
        return np.array([speed_from_pulse(pulse, fs=fs)], dtype=float)
    nb = len(pulse) // n
    b = (np.asarray(pulse, dtype=np.float32)[: nb * n] > 0.5).astype(np.int8).reshape(nb, n)
    trans = np.abs(np.diff(b, axis=1)).sum(axis=1)
    return trans / 2.0 * PULSE_DISTANCE_M / window_s * 3.6


def pulse_distance(pulse: np.ndarray, *, fs: float = FS_HZ) -> np.ndarray:
    """Cumulative travelled distance (m) per sample, interpolated between tacho edges.

    The pulse count is *integrated*, never differentiated (`ps3_addendum.md` 2.2): each toggle marks half
    a tooth, i.e. ``PULSE_DISTANCE_M / 2`` of travel, and distance between edges is linear in time.
    Outside the first and last edge the mean speed of the record extends the ramp.
    """
    pulse = np.asarray(pulse)
    n = len(pulse)
    t = np.arange(n, dtype=np.float64) / float(fs)
    edges = _pulse_edges(pulse)
    if len(edges) < 2:
        v = speed_from_pulse(pulse, fs=fs) / 3.6
        return t * v
    d_edge = np.arange(len(edges), dtype=np.float64) * (PULSE_DISTANCE_M / 2.0)
    t_edge = edges.astype(np.float64) / float(fs)
    dist = np.interp(t, t_edge, d_edge)
    # Extrapolate the ends at the local edge rate rather than holding them flat.
    v_start = (d_edge[1] - d_edge[0]) / max(t_edge[1] - t_edge[0], 1e-9)
    v_end = (d_edge[-1] - d_edge[-2]) / max(t_edge[-1] - t_edge[-2], 1e-9)
    head = t < t_edge[0]
    tail = t > t_edge[-1]
    dist[head] = d_edge[0] - (t_edge[0] - t[head]) * v_start
    dist[tail] = d_edge[-1] + (t[tail] - t_edge[-1]) * v_end
    return dist - dist[0]


def resample_to_distance(
    sig: np.ndarray, dist: np.ndarray, *, dx: float = DX_M, fs: float = FS_HZ
) -> np.ndarray:
    """Computed order tracking: resample ``sig`` (…, n_samples) from time onto a uniform Δx grid.

    Computed order tracking with a real keyphasor [R245][R239]; speed variation otherwise smears
    the spectrum. An anti-alias low-pass at ``0.8 * v / (2*dx)`` runs first whenever the native spatial step is
    finer than ``dx`` (true at low speed), so content below the 2 mm Nyquist wavelength cannot fold
    into the 8-500 mm band of interest.
    """
    sig = np.atleast_2d(np.asarray(sig, dtype=np.float64))
    dist = np.asarray(dist, dtype=np.float64)
    total = float(dist[-1] - dist[0])
    if total <= 4 * dx or sig.shape[-1] < 8:
        return np.zeros((sig.shape[0], 0), dtype=np.float32)
    v = total / (len(dist) / fs)  # mean m/s
    f_cut = 0.8 * v / (2.0 * dx)
    nyq = fs / 2.0
    if f_cut < 0.95 * nyq:
        sos = signal.butter(4, max(f_cut / nyq, 1e-3), btype="low", output="sos")
        sig = signal.sosfiltfilt(sos, sig, axis=-1)
    grid = np.arange(0.0, total, dx)
    out = np.empty((sig.shape[0], len(grid)), dtype=np.float32)
    for i in range(sig.shape[0]):
        out[i] = np.interp(grid, dist, sig[i])
    return out


# --------------------------------------------------------------------------------------
# Spectra
# --------------------------------------------------------------------------------------


def _welch_db(x: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    nperseg = int(min(nperseg, x.shape[-1]))
    if nperseg < 16:
        return np.zeros(0), np.zeros((x.shape[0], 0))
    f, pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    return f, pxx


def wavelength_psd_db(
    resampled: np.ndarray, *, dx: float = DX_M, grid_mm: np.ndarray = LAMBDA_GRID_MM
) -> np.ndarray:
    """Welch PSD of the distance-resampled signal, in dB, on the fixed wavelength grid.

    ``resampled`` is ``(n_channels, n_x)``; the spatial sampling rate is ``1/dx`` cycles per metre,
    so ``lambda = 1 / k``. Bins below two frequency resolutions are unresolved (a 1 s record at
    6 km/h is only 1.7 m of track), so the returned curve is edge-held rather than extrapolated
    there, and the caller can tell from ``lambda_max_resolved_mm`` how much to trust the long end.
    """
    n_ch = resampled.shape[0]
    out = np.full((n_ch, len(grid_mm)), -200.0, dtype=np.float32)
    if resampled.shape[-1] < 32:
        return out
    nperseg = int(min(2048, max(64, resampled.shape[-1] // 4 * 2)))
    k, pxx = _welch_db(resampled.astype(np.float64), fs=1.0 / dx, nperseg=nperseg)
    if k.size == 0:
        return out
    dk = k[1] - k[0]
    valid = k >= 2 * dk
    if not valid.any():
        return out
    k_v = k[valid]
    p_v = np.maximum(pxx[:, valid], 1e-20)
    lam_v = 1000.0 / k_v  # mm, descending
    order = np.argsort(lam_v)
    lam_s = lam_v[order]
    db = 10.0 * np.log10(p_v[:, order])
    for i in range(n_ch):
        out[i] = np.interp(grid_mm, lam_s, db[i])  # np.interp edge-holds outside the range
    return out


def _band_index(grid_mm: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (grid_mm >= lo) & (grid_mm < hi)


def band_levels_db(spec_db: np.ndarray, *, centres_mm: np.ndarray = THIRD_OCTAVE_CENTRES_MM,
                   grid_mm: np.ndarray = LAMBDA_GRID_MM) -> np.ndarray:
    """Integrate a dB wavelength spectrum into 1/3-octave wavelength bands (still dB).

    ``spec_db`` is ``(..., n_lambda)``; returns ``(..., n_bands)``. Energy is summed in linear
    power inside each band edge pair ``centre * 2**(-+1/6)`` and converted back to dB.
    """
    lin = np.power(10.0, np.asarray(spec_db, dtype=np.float64) / 10.0)
    out = np.empty(lin.shape[:-1] + (len(centres_mm),), dtype=np.float32)
    for j, c in enumerate(centres_mm):
        m = _band_index(grid_mm, c * 2 ** (-1 / 6), c * 2 ** (1 / 6))
        if not m.any():
            m = np.argmin(np.abs(grid_mm - c))
            band = lin[..., m]
        else:
            band = lin[..., m].mean(axis=-1)
        out[..., j] = 10.0 * np.log10(np.maximum(band, 1e-20))
    return out


# --------------------------------------------------------------------------------------
# Per-file extraction
# --------------------------------------------------------------------------------------


def _fingerprint(arr: np.ndarray) -> tuple[str, str]:
    """(first-channel, speed-column) hashes of the first 1,000 samples - the duplicate check."""
    a = np.ascontiguousarray(arr[:1000, 1].astype(np.float32))
    s = np.ascontiguousarray(arr[:1000, 0].astype(np.float32))
    return hashlib.sha1(a.tobytes()).hexdigest()[:16], hashlib.sha1(s.tobytes()).hexdigest()[:16]


def extract_file(path: Path | str) -> tuple[dict[str, Any], np.ndarray]:
    """Per-file scalars and the ``(128, N_LAMBDA)`` wavelength PSD in dB, from one csv.

    Pure function of the file: no fold, no dataset statistic, nothing fitted. Everything that
    *is* fitted (scalers, thresholds, the choice of aggregation) happens inside a training fold in
    :mod:`nebulax.ps3.rail`.
    """
    p = Path(path)
    return extract_array(read_rail_csv(p), name=p.name)


def extract_array(arr: np.ndarray, *, name: str = "<array>") -> tuple[dict[str, Any], np.ndarray]:
    """:func:`extract_file` on an already-loaded ``(n_samples, 129)`` array (tests synthesise one)."""
    p = Path(name)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != N_CHANNELS + 1:
        raise ValueError(f"{name}: expected (n_samples, {N_CHANNELS + 1}), got {arr.shape}")
    pulse = arr[:, 0]
    chans = arr[:, 1:].T.astype(np.float64)  # (128, n)
    n = arr.shape[0]

    speed_kmh = speed_from_pulse(pulse)
    win = _windowed_speed(pulse)
    dist = pulse_distance(pulse)
    a_hash, s_hash = _fingerprint(arr)

    scalars: dict[str, Any] = {
        "speed_kmh": speed_kmh,
        "speed_std_kmh": float(np.std(win)),
        "speed_min_kmh": float(np.min(win)),
        "speed_max_kmh": float(np.max(win)),
        "n_pulses": float(len(_pulse_edges(pulse)) / 2.0),
        "distance_m": float(dist[-1] - dist[0]),
        "n_samples": float(n),
        "hash_ch1": a_hash,
        "hash_speed": s_hash,
    }

    # -- time domain -------------------------------------------------------------------
    mean = chans.mean(axis=1, keepdims=True)
    cen = chans - mean
    rms = np.sqrt(np.mean(cen**2, axis=1))
    peak = np.max(np.abs(cen), axis=1)
    var = np.maximum(rms**2, 1e-20)
    kurt = np.mean(cen**4, axis=1) / var**2
    mean_abs = np.maximum(np.mean(np.abs(cen), axis=1), 1e-12)
    mean_sqrt = np.maximum(np.mean(np.sqrt(np.abs(cen)), axis=1), 1e-12)
    # Shape factors from the 26-feature metro corrugation set [finder_rail #2 / addendum R232];
    # its fixed 250-2000 Hz wavelet bands are replaced by the wavelength bands above.
    ch = {
        "logrms": 20.0 * np.log10(np.maximum(rms, 1e-10)),
        "logrms_clip95": _clipped_rms_db(cen),
        "logrms_win_p10": _window_rms_percentile_db(chans),
        "kurtosis": kurt,
        "logpeak": 20.0 * np.log10(np.maximum(peak, 1e-10)),
        "crest": peak / np.maximum(rms, 1e-10),
        "skew": np.mean(cen**3, axis=1) / np.maximum(rms, 1e-12) ** 3,
        "margin": peak / mean_sqrt**2,
        "pulse": peak / mean_abs,
        "waveform": rms / mean_abs,
    }

    # -- fixed Hz bands: the counter-design arm only [R234] -----------------------------
    f, pxx = _welch_db(chans, fs=FS_HZ, nperseg=min(2048, n))
    for i in range(len(HZ_BAND_EDGES) - 1):
        m = (f >= HZ_BAND_EDGES[i]) & (f < HZ_BAND_EDGES[i + 1])
        band = pxx[:, m].mean(axis=1) if m.any() else np.full(chans.shape[0], 1e-20)
        ch[f"hz{i}"] = 10.0 * np.log10(np.maximum(band, 1e-20))
    tot = np.maximum(pxx.sum(axis=1), 1e-20)
    centroid = (pxx * f[None, :]).sum(axis=1) / tot
    ch["spec_centroid"] = centroid
    ch["spec_sd"] = np.sqrt(np.maximum((pxx * (f[None, :] - centroid[:, None]) ** 2).sum(axis=1) / tot, 0.0))

    # -- distance resampling and the wavelength PSD ------------------------------------
    resampled = resample_to_distance(chans, dist)
    spec_db = wavelength_psd_db(resampled)
    scalars["n_x"] = float(resampled.shape[-1])
    scalars["lambda_max_resolved_mm"] = (
        float(1000.0 * min(2048, max(64, resampled.shape[-1] // 4 * 2)) * DX_M / 2.0)
        if resampled.shape[-1] >= 32
        else 0.0
    )

    # -- impulsive vs sustained discriminators [R235][R246] -----------------------------
    lin = np.power(10.0, spec_db.astype(np.float64) / 10.0)
    gm = np.exp(np.mean(np.log(np.maximum(lin, 1e-20)), axis=1))
    am = np.maximum(np.mean(lin, axis=1), 1e-20)
    ch["wl_flatness"] = gm / am
    med_db = np.median(spec_db, axis=1)
    peak_idx = np.argmax(spec_db, axis=1)
    ch["wl_peak_prom"] = spec_db[np.arange(spec_db.shape[0]), peak_idx] - med_db
    ch["wl_peak_lambda"] = LAMBDA_GRID_MM[peak_idx]
    ch["env_duty"] = _envelope_duty(resampled)

    flat = np.concatenate([np.asarray(ch[k], dtype=np.float64).reshape(-1, 1) for k in CHANNEL_SCALARS], axis=1)
    if flat.shape != (N_CHANNELS, len(CHANNEL_SCALARS)):  # pragma: no cover - defensive
        raise RuntimeError(f"{p.name}: per-channel block is {flat.shape}, expected {(N_CHANNELS, len(CHANNEL_SCALARS))}")
    for j, name in enumerate(CHANNEL_SCALARS):
        for i in range(N_CHANNELS):
            scalars[f"c{i}_{name}"] = float(flat[i, j])
    return scalars, spec_db.astype(np.float32)


def _clipped_rms_db(centered: np.ndarray) -> np.ndarray:
    """RMS after clipping each channel at its own 95th absolute percentile."""
    limit = np.percentile(np.abs(centered), 95, axis=1, keepdims=True)
    clipped = np.clip(centered, -limit, limit)
    return 20.0 * np.log10(np.maximum(np.sqrt(np.mean(clipped**2, axis=1)), 1e-10))


def _window_rms_percentile_db(chans: np.ndarray) -> np.ndarray:
    """10th percentile of centred RMS in nonoverlapping 0.1 s windows."""
    width = min(int(FS_HZ // 10), chans.shape[1])
    count = chans.shape[1] // width
    windows = chans[:, :count * width].reshape(chans.shape[0], count, width)
    centered = windows - windows.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=-1))
    return 20.0 * np.log10(np.maximum(np.percentile(rms, 10, axis=1), 1e-10))


def _envelope_duty(resampled: np.ndarray) -> np.ndarray:
    """Duty cycle of the short-pitch band-passed envelope: sustained ~ high, impulsive ~ low.

    Corrugation is sustained, periodic and side-wide; a wheel flat, a switch or a structural
    resonance is impulsive and usually one channel, yet occupies the same wavelength band - the
    documented false-positive source [R235][R246].
    """
    n_ch = resampled.shape[0] if resampled.ndim == 2 else 1
    if resampled.shape[-1] < 256:
        return np.zeros(n_ch)
    fs_k = 1.0 / DX_M  # cycles per metre
    lo = 1000.0 / SHORT_PITCH_MM[1]  # long lambda -> low spatial frequency
    hi = 1000.0 / SHORT_PITCH_MM[0]
    nyq = fs_k / 2.0
    sos = signal.butter(4, [max(lo / nyq, 1e-4), min(hi / nyq, 0.99)], btype="band", output="sos")
    band = signal.sosfiltfilt(sos, resampled.astype(np.float64), axis=-1)
    env = np.abs(signal.hilbert(band, axis=-1))
    med = np.median(env, axis=-1, keepdims=True)
    return (env > 2.0 * np.maximum(med, 1e-12)).mean(axis=-1)


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------


def cache_dir(version: str = FEATURE_VERSION) -> Path:
    return CACHE_DIR / "rail" / version


def _extract_one(args: tuple[str, str]) -> str:
    path, version = args
    p = Path(path)
    out = cache_dir(version)
    shard = out / f"{p.stem}.parquet"
    spec_path = out / f"{p.stem}.spec.npy"
    if shard.exists() and spec_path.exists():
        return p.name
    scalars, spec = extract_file(p)
    scalars["file_id"] = p.name
    out.mkdir(parents=True, exist_ok=True)
    np.save(spec_path, spec)
    pd.DataFrame([scalars]).to_parquet(shard, index=False)
    return p.name


def build_feature_cache(
    paths: Sequence[Path | str],
    *,
    version: str = FEATURE_VERSION,
    n_jobs: int = 4,
    progress: bool = True,
) -> None:
    """Extract every file that is not cached yet, at most ``n_jobs`` files in flight.

    Streaming, file by file: one worker holds one 17 MB recording at a time, never the 5.5 GB.
    """
    todo = [Path(p) for p in paths]
    out = cache_dir(version)
    out.mkdir(parents=True, exist_ok=True)
    pending = [p for p in todo if not ((out / f"{p.stem}.parquet").exists() and (out / f"{p.stem}.spec.npy").exists())]
    if not pending:
        return
    args = [(str(p), version) for p in pending]
    done = 0
    if n_jobs <= 1:
        for a in args:
            _extract_one(a)
            done += 1
            if progress and done % 20 == 0:
                print(f"  {done}/{len(args)}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
        for _ in ex.map(_extract_one, args, chunksize=1):
            done += 1
            if progress and done % 20 == 0:
                print(f"  {done}/{len(args)}", flush=True)


@dataclass
class RailFeatures:
    """The cached feature block for a set of files, aligned row-for-row.

    ``scalars`` is one row per file (file scalars + the flattened per-channel block);
    ``spectra`` is ``(n_files, 128, N_LAMBDA)`` dB; ``file_ids`` is the row order.
    """

    file_ids: list[str]
    scalars: pd.DataFrame
    spectra: np.ndarray
    raw_path: str | None = None
    source_paths: list[str] | None = None
    coherence: pd.DataFrame | None = None
    mirrored: bool = False

    def __len__(self) -> int:
        return len(self.file_ids)

    def channel_block(self, name: str) -> np.ndarray:
        """``(n_files, 128)`` view of one per-channel scalar."""
        cols = [f"c{i}_{name}" for i in range(N_CHANNELS)]
        return self.scalars[cols].to_numpy(dtype=np.float64)

    def subset(self, idx: Sequence[int]) -> "RailFeatures":
        idx = [int(i) for i in np.asarray(idx).reshape(-1)]
        return RailFeatures(
            file_ids=[self.file_ids[i] for i in idx],
            scalars=self.scalars.iloc[idx].reset_index(drop=True),
            spectra=self.spectra[idx],
            raw_path=self.raw_path,
            source_paths=[self.source_paths[i] for i in idx] if self.source_paths is not None else None,
            coherence=self.coherence.iloc[idx].reset_index(drop=True) if self.coherence is not None else None,
            mirrored=self.mirrored,
        )


def load_feature_cache(
    paths: Sequence[Path | str], *, version: str = FEATURE_VERSION, n_jobs: int = 4
) -> RailFeatures:
    """Build (if needed) and read the cache for ``paths``, in natural file-name order."""
    build_feature_cache(paths, version=version, n_jobs=n_jobs)
    out = cache_dir(version)
    ordered = sorted((Path(p) for p in paths), key=lambda p: natural_key(p.name))
    frames = [pd.read_parquet(out / f"{p.stem}.parquet") for p in ordered]
    spec = np.stack([np.load(out / f"{p.stem}.spec.npy") for p in ordered]).astype(np.float32)
    df = pd.concat(frames, ignore_index=True)
    return RailFeatures(file_ids=[p.name for p in ordered], scalars=df, spectra=spec,
                        source_paths=[str(p) for p in ordered])


def coherence_for_array(raw: np.ndarray) -> dict[str, float]:
    """Welch magnitude-squared coherence across same-side boxes within each car.

    The summary is phase independent and per-recording: no fitted data statistics
    or labels enter it.  Six box pairs per side and car are averaged in seven Hz
    bands, then mirrored by exchanging the two side columns.
    """
    x = np.asarray(raw[:, 1::2], dtype=np.float32).T
    if x.shape[0] != N_BOXES or x.shape[1] < 2:
        raise ValueError("rail coherence needs 64 vibration boxes and at least two samples")
    x = x - x.mean(axis=1, keepdims=True)
    nperseg = min(1024, x.shape[1])
    hop = max(1, nperseg // 2)
    starts = np.arange(0, x.shape[1] - nperseg + 1, hop)
    segments = np.stack([x[:, start:start + nperseg] for start in starts], axis=1)
    fft = np.fft.rfft(segments * np.hanning(nperseg).astype(np.float32), axis=-1)
    power = np.mean(np.abs(fft) ** 2, axis=1)
    freq = np.fft.rfftfreq(nperseg, 1 / FS_HZ)
    side_curves = {}
    for side, positions in (("I", (0, 2, 4, 6)), ("II", (1, 3, 5, 7))):
        pairs = []
        for car in range(8):
            boxes = [car * 8 + p for p in positions]
            for j in range(4):
                for k in range(j + 1, 4):
                    a, b = boxes[j], boxes[k]
                    cross = np.mean(fft[a] * np.conj(fft[b]), axis=0)
                    pairs.append(np.abs(cross) ** 2 / np.maximum(power[a] * power[b], 1e-12))
        side_curves[side] = np.mean(pairs, axis=0)
    values = {}
    for band in range(len(HZ_BAND_EDGES) - 1):
        mask = (freq >= HZ_BAND_EDGES[band]) & (freq < HZ_BAND_EDGES[band + 1])
        if not mask.any():
            mask[np.argmin(abs(freq - np.sqrt(HZ_BAND_EDGES[band] * HZ_BAND_EDGES[band + 1])))] = True
        i_val = float(side_curves["I"][mask].mean())
        ii_val = float(side_curves["II"][mask].mean())
        values[f"coh_I_hz{band}"] = i_val
        values[f"coh_II_hz{band}"] = ii_val
        values[f"coh_contrast_hz{band}"] = i_val - ii_val
    return values


def coherence_for_path(path: Path | str) -> dict[str, float]:
    """Load an immutable source-keyed cache entry or compute one recording's coherence."""
    path = Path(path)
    stat = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()
    cache = CACHE_DIR / "rail_coherence" / "c1" / f"{key}.parquet"
    if cache.exists():
        try:
            return pd.read_parquet(cache).iloc[0].to_dict()
        except (OSError, ValueError):
            pass
    row = coherence_for_array(read_rail_csv(path))
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        pd.DataFrame([row]).to_parquet(tmp, index=False)
        tmp.replace(cache)
    except OSError:
        pass  # a read-only packaged app can still predict without caching
    return row


def mirror_coherence(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for band in range(len(HZ_BAND_EDGES) - 1):
        i_col, ii_col, d_col = f"coh_I_hz{band}", f"coh_II_hz{band}", f"coh_contrast_hz{band}"
        out[i_col] = frame[ii_col].to_numpy()
        out[ii_col] = frame[i_col].to_numpy()
        out[d_col] = -frame[d_col].to_numpy()
    return out


def _ensure_coherence(feats: RailFeatures) -> pd.DataFrame:
    if feats.coherence is None:
        paths = feats.source_paths or ([feats.raw_path] if feats.raw_path and len(feats) == 1 else None)
        if paths is None or len(paths) != len(feats):
            raise ValueError("rail coherence requires a source path for every recording")
        frame = pd.DataFrame([coherence_for_path(p) for p in paths])
        feats.coherence = mirror_coherence(frame) if feats.mirrored else frame
    return feats.coherence


# --------------------------------------------------------------------------------------
# Aggregation: 128 channels -> per-side levels, contrasts, votes
# --------------------------------------------------------------------------------------

#: Default aggregation options. Every flag here is an ablation arm in the ladder.
AGG_DEFAULT: dict[str, Any] = {
    "wavelength": True,   # 1/3-octave wavelength bands from the distance-resampled signal
    "hz": False,          # fixed 20-5000 Hz bands (the counter-design arm)
    "v2_normalise": True, # subtract 40*log10(v) from every dB level
    "contrast": True,     # Side I - Side II difference columns
    "speed": True,        # speed features as explicit covariates
    "time": True,         # per-channel log RMS / kurtosis / log peak / crest
    "discriminators": True,
    "votes": True,        # per-car "which side is louder in the short-pitch band" votes
    "spectra_series": False,  # emit the per-side spectra as a series block (the ROCKET arm)
    "shock": True,        # include the shock-channel blocks
    "robust_time": False, # clipped and low-window RMS for vibration channels
    "hz_p10": False,      # low-side sensor response in each vibration Hz band
    "coherence": False,   # same-side Welch coherence across boxes within each car
}

_VIB = np.arange(0, N_CHANNELS, 2)   # channel index of the vibration signal of box i
_SHOCK = np.arange(1, N_CHANNELS, 2)


def _box_channels(kind: str) -> np.ndarray:
    return _VIB if kind == "vib" else _SHOCK


def _aggs(block: np.ndarray) -> dict[str, np.ndarray]:
    """median / p90 / max over the channel axis of ``(n_files, n_ch[, n_band])``."""
    return {
        "med": np.median(block, axis=1),
        "p90": np.percentile(block, 90, axis=1),
        "max": np.max(block, axis=1),
    }


def mirror(feats: RailFeatures) -> RailFeatures:
    """Exact left-right mirror: swap positions (1,2), (3,4), (5,6), (7,8) on every car.

    Maps a Side I record onto a valid Side II record and vice versa, imposing the equivariance the
    sensor layout demands. Free and label-noise-free - `ps3_addendum.md` section 6 ranks it the
    must-do rail augmentation ("exact, free, turns 14/24 into 38/38"); the caller swaps the
    labels. Used **inside the training fold only**.
    """
    box_perm = MIRROR_PERM
    ch_perm = np.empty(N_CHANNELS, dtype=int)
    ch_perm[_VIB] = 2 * box_perm
    ch_perm[_SHOCK] = 2 * box_perm + 1
    sc = feats.scalars.copy()
    for name in CHANNEL_SCALARS:
        if f"c0_{name}" not in sc.columns:
            if name in {"logrms_clip95", "logrms_win_p10"}:
                continue  # older window caches do not contain optional robust summaries
            raise ValueError(f"rail feature cache is missing required scalar {name!r}")
        cols = [f"c{i}_{name}" for i in range(N_CHANNELS)]
        sc[cols] = sc[[f"c{ch_perm[i]}_{name}" for i in range(N_CHANNELS)]].to_numpy()
    return RailFeatures(
        file_ids=[f"mirror::{fid}" for fid in feats.file_ids],
        scalars=sc,
        spectra=feats.spectra[:, ch_perm, :],
        raw_path=feats.raw_path,
        source_paths=feats.source_paths,
        coherence=mirror_coherence(feats.coherence) if feats.coherence is not None else None,
        mirrored=not feats.mirrored,
    )


def perturb_fault_channels(feats: RailFeatures, mode: str, *, seed: int) -> RailFeatures:
    """One reproducible, physically coherent training copy of each fault recording.

    ``gain`` simulates small, independent vibration sensor calibration errors: every level
    derived from a channel receives the same dB offset. Dimensionless shape statistics and the
    tachometer are unchanged. ``mask`` replaces one vibration sensor on *each* side with its
    same-side median, so a missing sensor cannot create a side label. The caller is responsible
    for restricting ``feats`` to the training fold and fault classes.
    """
    if mode not in {"gain", "mask"}:
        raise ValueError(f"unknown rail sensor augmentation {mode!r}")
    sc = feats.scalars.copy()
    spec = feats.spectra.copy()
    rng = np.random.default_rng(seed)
    level_names = ("logrms", "logrms_clip95", "logrms_win_p10", "logpeak",
                   *[f"hz{i}" for i in range(len(HZ_BAND_EDGES) - 1)])
    available_names = [name for name in CHANNEL_SCALARS if f"c0_{name}" in sc.columns]
    for row in range(len(feats)):
        if mode == "gain":
            # +/- 0.75 dB is a small calibration perturbation, not an artificial speed change.
            offsets = rng.uniform(-0.75, 0.75, size=N_BOXES)
            for name in level_names:
                if name in available_names:
                    cols = [f"c{2 * box}_{name}" for box in range(N_BOXES)]
                    sc.loc[sc.index[row], cols] = sc.loc[sc.index[row], cols].to_numpy(dtype=float) + offsets
            spec[row, _VIB, :] += offsets[:, None].astype(spec.dtype)
        else:
            for side in ("I", "II"):
                boxes = np.flatnonzero(side_mask(side))
                chosen = int(rng.choice(boxes))
                donors = boxes[boxes != chosen]
                channel, donor_channels = 2 * chosen, 2 * donors
                for name in available_names:
                    cols = [f"c{int(ch)}_{name}" for ch in donor_channels]
                    sc.at[sc.index[row], f"c{channel}_{name}"] = float(
                        np.median(sc.loc[sc.index[row], cols].to_numpy(dtype=float))
                    )
                spec[row, channel] = np.median(spec[row, donor_channels], axis=0)
    ids = [f"{mode}::{fid}" for fid in feats.file_ids]
    if "file_id" in sc:
        sc["file_id"] = ids
    return RailFeatures(file_ids=ids, scalars=sc, spectra=spec, raw_path=feats.raw_path)


def aggregate(feats: RailFeatures, opts: dict[str, Any] | None = None) -> pd.DataFrame:
    """Per-side aggregation of the cached per-channel block into the model's design matrix.

    Pure function of the cache and ``opts`` - no dataset statistic is used, so calling it on a
    held-out fold cannot leak. The Side I - Side II contrast is the feature the label actually
    describes and removes speed, track and sensor gain as common-mode nuisances
    (`ps3_addendum.md` 2.2; supported by [R235]).
    """
    o = {**AGG_DEFAULT, **(opts or {})}
    n = len(feats)
    v_mps = np.maximum(feats.scalars["speed_kmh"].to_numpy(dtype=float) / 3.6, 0.05)
    v2_db = 40.0 * np.log10(v_mps) if o["v2_normalise"] else np.zeros(n)
    cols: dict[str, np.ndarray] = {}

    masks = {"I": side_mask("I"), "II": side_mask("II")}

    def add_side_block(prefix: str, block: np.ndarray, names: Sequence[str], is_db: bool) -> None:
        """``block`` is (n_files, 64 boxes, n_feat) for one channel kind."""
        per_side = {}
        for side, m in masks.items():
            a = _aggs(block[:, m, :])
            per_side[side] = a
            for agg, val in a.items():
                for j, nm in enumerate(names):
                    cols[f"{prefix}_{side}_{agg}_{nm}"] = val[:, j]
        if o["contrast"]:
            for agg in ("med", "p90"):
                d = per_side["I"][agg] - per_side["II"][agg]
                for j, nm in enumerate(names):
                    cols[f"{prefix}_contrast_{agg}_{nm}"] = d[:, j]
        if o.get("hz_p10", False) and prefix == "vib_hz":
            low = {side: np.percentile(block[:, m, :], 10, axis=1) for side, m in masks.items()}
            for side in ("I", "II"):
                for j, nm in enumerate(names):
                    cols[f"{prefix}_{side}_p10_{nm}"] = low[side][:, j]
            if o["contrast"]:
                for j, nm in enumerate(names):
                    cols[f"{prefix}_contrast_p10_{nm}"] = low["I"][:, j] - low["II"][:, j]
        _ = is_db

    kinds = ("vib", "shock") if o.get("shock", True) else ("vib",)
    for kind in kinds:
        boxes = _box_channels(kind)
        if o["wavelength"]:
            bands = band_levels_db(feats.spectra[:, boxes, :])  # (n, 64, 18)
            bands = bands - v2_db[:, None, None]
            names = [f"wl{j}" for j in range(bands.shape[-1])]
            add_side_block(f"{kind}_wl", bands, names, True)
        if o["hz"]:
            blk = np.stack([feats.channel_block(f"hz{i}")[:, boxes] for i in range(len(HZ_BAND_EDGES) - 1)], axis=-1)
            blk = blk - v2_db[:, None, None]
            add_side_block(f"{kind}_hz", blk, [f"hz{i}" for i in range(blk.shape[-1])], True)
        if o["time"]:
            names = ["logrms", "kurtosis", "logpeak", "crest", "skew", "margin", "pulse", "waveform"]
            blk = np.stack([feats.channel_block(nm)[:, boxes] for nm in names], axis=-1)
            blk[..., 0] -= v2_db[:, None]  # log RMS is a level; the shape factors are dimensionless
            blk[..., 2] -= v2_db[:, None]
            add_side_block(f"{kind}_t", blk, names, True)
        if kind == "vib" and o["robust_time"]:
            names = ["logrms_clip95", "logrms_win_p10"]
            blk = np.stack([feats.channel_block(nm)[:, boxes] for nm in names], axis=-1)
            blk -= v2_db[:, None, None]
            add_side_block(f"{kind}_robust", blk, names, True)
        if o["discriminators"]:
            names = ["wl_flatness", "wl_peak_prom", "wl_peak_lambda", "env_duty", "spec_centroid", "spec_sd"]
            blk = np.stack([feats.channel_block(nm)[:, boxes] for nm in names], axis=-1)
            add_side_block(f"{kind}_d", blk, names, False)
            # cross-channel agreement within a side: a wheel defect is one channel, corrugation a side
            lam = feats.channel_block("wl_peak_lambda")[:, boxes]
            for side, m in masks.items():
                sub = lam[:, m]
                med = np.median(sub, axis=1, keepdims=True)
                cols[f"{kind}_agree_{side}"] = (np.abs(sub - med) <= 0.2 * np.maximum(med, 1e-9)).mean(axis=1)
                cols[f"{kind}_lamspread_{side}"] = np.std(np.log(np.maximum(sub, 1e-6)), axis=1)

    if o["spectra_series"]:
        series_pairs = (("vib", "I"), ("vib", "II"), ("shock", "I"), ("shock", "II")) if o.get("shock", True) else (("vib", "I"), ("vib", "II"))
        for c, (kind, side) in enumerate(series_pairs):
            boxes = _box_channels(kind)[masks[side]]
            ser = np.median(feats.spectra[:, boxes, :], axis=1) - v2_db[:, None]
            for j in range(ser.shape[1]):
                cols[f"ser{c}_{j}"] = ser[:, j]

    if o["votes"]:
        # per-car side votes: how many cars hear the louder short-pitch band on Side I
        bands = band_levels_db(feats.spectra[:, _VIB, :])
        short = (THIRD_OCTAVE_CENTRES_MM >= SHORT_PITCH_MM[0]) & (THIRD_OCTAVE_CENTRES_MM <= SHORT_PITCH_MM[1])
        level = bands[:, :, short].mean(axis=-1)  # (n, 64)
        per_car = level.reshape(n, 8, 8)
        side_i = per_car[:, :, ::2].mean(axis=-1)
        side_ii = per_car[:, :, 1::2].mean(axis=-1)
        cols["car_vote_i"] = (side_i > side_ii).mean(axis=1)
        cols["car_vote_margin"] = (side_i - side_ii).mean(axis=1)
        cols["car_vote_margin_sd"] = (side_i - side_ii).std(axis=1)

    if o["speed"]:
        for c in ("speed_kmh", "speed_std_kmh", "speed_min_kmh", "speed_max_kmh"):
            cols[c] = feats.scalars[c].to_numpy(dtype=float)
        cols["log_speed"] = np.log(np.maximum(feats.scalars["speed_kmh"].to_numpy(dtype=float), 0.1))

    if o.get("coherence", False):
        for name, values in _ensure_coherence(feats).items():
            cols[name] = values.to_numpy(dtype=float)

    X = pd.DataFrame(cols)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
