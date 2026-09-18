"""Tests for nebulax.features.vibration: kurtogram band selection and envelope-spectrum
BPFO/BPFI/BSF/FTF feature extraction on a synthetic impulsive signal."""

from __future__ import annotations

import numpy as np
import pytest

from nebulax.features import vibration as V

# A generic deep-groove ball bearing geometry (illustrative, not a specific SKF part number):
# Z=9 elements, Bd=7.94 mm, Dp=39.0 mm, contact angle 0.
GEOM = V.BearingGeometry(n_elements=9, ball_diameter_m=7.94e-3, pitch_diameter_m=39.0e-3)


def _synthetic_bpfo_signal(fs, duration_s, shaft_hz, geometry, *, amp=1.0, noise=0.05, seed=0):
    """Amplitude-modulated impulse train at the BPFO characteristic frequency, ringing a
    resonance well above it - the textbook synthetic bearing-fault signal."""
    rng = np.random.default_rng(seed)
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    bpfo_hz = geometry.factors()["bpfo"] * shaft_hz
    resonance_hz = fs / 8.0
    # impulse train at 1/bpfo_hz period, each impulse a decaying sinusoid at resonance_hz
    period = 1.0 / bpfo_hz
    impulse_times = np.arange(0, duration_s, period)
    x = np.zeros(n)
    for it in impulse_times:
        idx0 = int(it * fs)
        tail = np.arange(0, min(int(fs * 0.01), n - idx0))
        x[idx0 : idx0 + tail.size] += amp * np.exp(-tail / (fs * 0.001)) * np.sin(2 * np.pi * resonance_hz * tail / fs)
    x += rng.normal(scale=noise, size=n)
    return x, bpfo_hz


def test_bearing_geometry_factors_are_positive_and_ordered():
    f = GEOM.factors()
    assert set(f) == {"bpfo", "bpfi", "bsf", "ftf"}
    assert all(v > 0 for v in f.values())
    # BPFI (inner race, moves with the shaft it's mounted on) exceeds BPFO for a fixed outer race
    assert f["bpfi"] > f["bpfo"]


def test_fast_kurtogram_finds_resonance_band():
    fs = 20_000.0
    x, _ = _synthetic_bpfo_signal(fs, duration_s=1.0, shaft_hz=25.0, geometry=GEOM, seed=1)
    fc, bw, sk = V.fast_kurtogram(x, fs, max_level=4)
    resonance_hz = fs / 8.0
    assert abs(fc - resonance_hz) < bw  # the selected band should straddle the true resonance
    assert sk > 3.0  # impulsive content -> kurtosis well above the Gaussian value


def test_envelope_spectrum_feats_detects_bpfo_peak():
    fs = 20_000.0
    shaft_hz = 25.0
    x, bpfo_hz = _synthetic_bpfo_signal(fs, duration_s=2.0, shaft_hz=shaft_hz, geometry=GEOM, seed=2)
    out = V.envelope_spectrum_feats(x, fs, shaft_hz * 60.0, GEOM)  # rpm = shaft_hz * 60
    assert out["env_bpfo_freq_hz"] == pytest.approx(bpfo_hz, rel=1e-6)
    # the injected fault frequency must dominate every other characteristic frequency's energy
    other_energies = [out[f"env_{name}_energy"] for name in ("bpfi", "bsf", "ftf")]
    assert out["env_bpfo_energy"] > max(other_energies) * 3.0


def test_envelope_spectrum_feats_healthy_signal_has_no_dominant_peak():
    fs = 20_000.0
    shaft_hz = 25.0
    rng = np.random.default_rng(3)
    x = rng.normal(scale=0.05, size=int(fs * 2.0))
    out_healthy = V.envelope_spectrum_feats(x, fs, shaft_hz * 60.0, GEOM)

    x_fault, _ = _synthetic_bpfo_signal(fs, duration_s=2.0, shaft_hz=shaft_hz, geometry=GEOM, seed=2)
    out_fault = V.envelope_spectrum_feats(x_fault, fs, shaft_hz * 60.0, GEOM)

    assert out_fault["env_bpfo_energy"] > out_healthy["env_bpfo_energy"] * 5.0


def test_envelope_spectrum_feats_speed_in_ms_matches_rpm():
    fs = 20_000.0
    shaft_hz = 10.0
    wheel_r = 0.425
    speed_ms = shaft_hz * 2 * np.pi * wheel_r
    x, bpfo_hz = _synthetic_bpfo_signal(fs, duration_s=1.5, shaft_hz=shaft_hz, geometry=GEOM, seed=4)
    out_rpm = V.envelope_spectrum_feats(x, fs, shaft_hz * 60.0, GEOM)
    out_speed = V.envelope_spectrum_feats(x, fs, speed_ms, GEOM, is_speed_ms=True, wheel_radius_m=wheel_r)
    assert out_rpm["shaft_hz"] == pytest.approx(out_speed["shaft_hz"], rel=1e-6)
    assert out_rpm["env_bpfo_freq_hz"] == pytest.approx(bpfo_hz, rel=1e-6)


def test_envelope_spectrum_feats_time_domain_stats_present():
    fs = 5_000.0
    x, _ = _synthetic_bpfo_signal(fs, duration_s=1.0, shaft_hz=20.0, geometry=GEOM, seed=5)
    out = V.envelope_spectrum_feats(x, fs, 20.0 * 60.0, GEOM)
    for key in ("rms", "kurtosis", "crest", "sk_band_fc", "sk_band_bw", "sk_max"):
        assert key in out
        assert np.isfinite(out[key])


def test_envelope_spectrum_feats_rejects_short_signal():
    with pytest.raises(ValueError):
        V.envelope_spectrum_feats(np.array([1.0, 2.0]), 100.0, 600.0, GEOM)


def test_bearing_geometry_rejects_bad_params():
    with pytest.raises(ValueError):
        V.BearingGeometry(n_elements=0, ball_diameter_m=1e-3, pitch_diameter_m=1e-2)
    with pytest.raises(ValueError):
        V.BearingGeometry(n_elements=9, ball_diameter_m=0.0, pitch_diameter_m=1e-2)


# --------------------------------------------- amplitude scaling (not FFT power)


def _am_signal(fs, duration_s, carrier_hz, mod_hz, *, carrier_amp=1.0, depth=0.4):
    """``A*(1 + m*cos(2*pi*f_mod*t))*cos(2*pi*f_c*t)`` - envelope ``A*(1 + m*cos(...))``,
    whose single-sided amplitude at ``f_mod`` is exactly ``A*m`` after mean removal."""
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    return carrier_amp * (1.0 + depth * np.cos(2 * np.pi * mod_hz * t)) * np.cos(2 * np.pi * carrier_hz * t)


@pytest.mark.parametrize("carrier_amp, depth", [(1.0, 0.4), (3.0, 0.25), (0.5, 0.8)])
def test_envelope_harmonics_are_amplitudes_in_signal_units(carrier_amp, depth):
    """``env_*_h<k>`` must read the envelope's *amplitude* in the input's units (m/s2).

    ``max_level=0`` pins the kurtogram to the full band, so the demodulation is exact and the
    expected answer is known in closed form: the envelope's BPFO component has amplitude
    ``carrier_amp * depth``. The old ``|E|**2/n`` power definition fails this by orders of
    magnitude (it grows with the window length, see the next test).
    """
    fs = 20_000.0
    shaft_hz = 25.0
    bpfo_hz = GEOM.factors()["bpfo"] * shaft_hz
    x = _am_signal(fs, 2.0, carrier_hz=fs / 8.0, mod_hz=bpfo_hz, carrier_amp=carrier_amp, depth=depth)
    out = V.envelope_spectrum_feats(x, fs, shaft_hz * 60.0, GEOM, max_level=0)
    assert out["env_bpfo_h1"] == pytest.approx(carrier_amp * depth, rel=0.15)


def test_envelope_amplitudes_do_not_scale_with_window_length():
    """The defining property of an amplitude: it is a property of the signal, not of ``n``.

    FFT power ``|E|**2/n`` at a coherent tone grows linearly with the window length, so this
    is the test that separates the two definitions - 1 s vs 4 s of the same signal would have
    differed 4x.
    """
    fs = 20_000.0
    shaft_hz = 25.0
    short, _ = _synthetic_bpfo_signal(fs, duration_s=1.0, shaft_hz=shaft_hz, geometry=GEOM, seed=7)
    long, _ = _synthetic_bpfo_signal(fs, duration_s=4.0, shaft_hz=shaft_hz, geometry=GEOM, seed=7)
    e_short = V.envelope_spectrum_feats(short, fs, shaft_hz * 60.0, GEOM)["env_bpfo_energy"]
    e_long = V.envelope_spectrum_feats(long, fs, shaft_hz * 60.0, GEOM)["env_bpfo_energy"]
    assert e_long == pytest.approx(e_short, rel=0.10)


def test_envelope_amplitudes_scale_linearly_with_the_signal():
    # doubling the acceleration doubles the amplitude (a power feature would quadruple)
    fs = 20_000.0
    shaft_hz = 25.0
    x, _ = _synthetic_bpfo_signal(fs, duration_s=1.0, shaft_hz=shaft_hz, geometry=GEOM, seed=8)
    one = V.envelope_spectrum_feats(x, fs, shaft_hz * 60.0, GEOM)
    two = V.envelope_spectrum_feats(2.0 * x, fs, shaft_hz * 60.0, GEOM)
    assert two["env_bpfo_energy"] == pytest.approx(2.0 * one["env_bpfo_energy"], rel=1e-6)
    assert two["sk_band_fc"] == pytest.approx(one["sk_band_fc"])
