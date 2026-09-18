"""Tests for scripts/calibrate_bearing.py - the Ottawa -> bearing-simulator calibration.

Everything except one small integration test runs on synthetic arrays, so the file stays
far inside the 60 s budget; the integration test reads a single real Ottawa recording and
is skipped when ``data/raw/ottawa`` is absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "calibrate_bearing.py"
RAW_DIR = REPO_ROOT / "data" / "raw" / "ottawa"


def _load_module():
    spec = importlib.util.spec_from_file_location("calibrate_bearing", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["calibrate_bearing"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cb():
    return _load_module()


# ------------------------------------------------------------------ names and IO guards


def test_parse_record_name_reads_class_bearing_and_state(cb):
    for stem, health, bid, state in (
        ("H_1_0", "healthy", 1, 0),
        ("I_5_2", "inner_race", 5, 2),
        ("O_10_1", "outer_race", 10, 1),
        ("B_11_2", "ball", 11, 2),
        ("C_20_1", "cage", 20, 1),
    ):
        rec = cb.parse_record_name(stem)
        assert (rec.health, rec.bearing_id, rec.state, rec.stem) == (health, bid, state, stem)


@pytest.mark.parametrize("bad", ["X_1_0", "H_1_3", "H_1", "h_1_0", ""])
def test_parse_record_name_rejects_anything_else(cb, bad):
    with pytest.raises(ValueError, match="unrecognised record name"):
        cb.parse_record_name(bad)


def test_extract_table_names_the_expected_layout_when_the_dir_is_wrong(cb, tmp_path):
    with pytest.raises(FileNotFoundError, match="H_1_0.csv"):
        cb.extract_table(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no \\*.csv records"):
        cb.extract_table(tmp_path / "empty")


# ------------------------------------------------------------------------ AC statistics


def test_window_stats_ac_is_invariant_to_a_dc_offset(cb, rng):
    """The whole calibration rests on this: 21 of 60 Ottawa records carry a DC offset
    larger than their own AC RMS, and a DC-coupled crest would read ~1 on those."""
    x = rng.standard_normal((3, 4096))
    base = cb.window_stats_ac(x)
    shifted = cb.window_stats_ac(x + 500.0)
    for key in ("rms", "kurt", "crest"):
        np.testing.assert_allclose(base[key], shifted[key], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(shifted["dc"] - base["dc"], 500.0, atol=1e-6)
    # ... and the DC-coupled version really is wrecked, which is why we do not use it.
    dc_rms = np.sqrt(np.mean((x + 500.0) ** 2, axis=1))
    dc_crest = np.max(np.abs(x + 500.0), axis=1) / dc_rms
    assert (dc_crest < 1.2).all()
    assert (base["crest"] > 3.0).all()


def test_window_stats_ac_matches_closed_form_values(cb):
    """A pure sine has RMS a/sqrt(2), crest sqrt(2) and kurtosis 1.5."""
    t = np.arange(0, 8192) / 8192.0
    x = (7.0 * np.sin(2 * np.pi * 16.0 * t))[None, :]
    s = cb.window_stats_ac(x)
    assert s["rms"][0] == pytest.approx(7.0 / np.sqrt(2.0), rel=1e-3)
    assert s["crest"][0] == pytest.approx(np.sqrt(2.0), rel=1e-3)
    assert s["kurt"][0] == pytest.approx(1.5, rel=1e-3)


def test_window_stats_ac_rejects_a_1d_array(cb):
    with pytest.raises(ValueError, match="2-D"):
        cb.window_stats_ac(np.zeros(16))


# --------------------------------------------------------------------------- features


def test_record_features_separates_an_impulsive_record_from_a_smooth_one(cb, rng):
    """An impulse train at BPFO must raise bpfo_amp, kurtosis and crest above broadband noise."""
    fs = 4200.0
    n = int(fs)
    rpm = 1800.0
    fr = rpm / 60.0
    bpfo_hz = cb.GEOMETRY.factors()["bpfo"] * fr
    t = np.arange(n) / fs
    healthy = rng.standard_normal(n)
    impulses = np.zeros(n)
    idx = np.unique((np.arange(0, int(n / fs * bpfo_hz)) / bpfo_hz * fs).astype(int))
    impulses[idx[idx < n]] = 1.0
    ring = np.exp(-np.arange(80) / 12.0) * np.sin(2 * np.pi * 900.0 * np.arange(80) / fs)
    faulty = healthy + 20.0 * np.convolve(impulses, ring)[:n]

    kw = dict(fs=fs, window_s=1.0, max_level=2)
    fh = cb.record_features(healthy, rpm, **kw)
    ff = cb.record_features(faulty, rpm, **kw)
    assert len(fh) == len(ff) == 1
    assert set(["rms", "kurt", "crest", "dc", "bpfo_amp", "bpfi_amp", "sk_max"]) <= set(fh.columns)
    assert ff["bpfo_amp"][0] > 5.0 * fh["bpfo_amp"][0]
    assert ff["kurt"][0] > 1.5 * fh["kurt"][0]
    assert ff["crest"][0] > fh["crest"][0]
    assert fh["shaft_hz"][0] == pytest.approx(fr)
    assert t.size == n  # (keeps the sample grid explicit)


def test_record_features_refuses_a_signal_shorter_than_one_window(cb):
    with pytest.raises(ValueError, match="shorter than one"):
        cb.record_features(np.zeros(100), 1800.0, fs=4200.0, window_s=1.0)


# ------------------------------------------------------------------------- power laws


def test_fit_power_law_recovers_a_known_exponent(cb):
    speed = np.array([10.0, 14.0, 20.0, 28.0, 40.0])
    fit = cb.fit_power_law(speed, 3.0 * speed**1.7)
    assert fit.alpha == pytest.approx(1.7, rel=1e-6)
    assert fit.r2 == pytest.approx(1.0, abs=1e-9)
    assert fit.covers(1.7) and not fit.covers(0.0)
    assert fit.speed_span == pytest.approx(4.0)
    np.testing.assert_allclose(fit.at(speed), 3.0 * speed**1.7, rtol=1e-6)
    assert "alpha" in fit.describe()


def test_fit_power_law_widens_its_ci_when_the_speed_span_is_narrow(cb, rng):
    """Ottawa's problem in one test: the same noise over a 1.1x span is uninformative."""
    noise = rng.normal(0.0, 0.4, 12)
    wide = np.linspace(10.0, 40.0, 12)
    narrow = np.linspace(1700.0, 1900.0, 12)
    f_wide = cb.fit_power_law(wide, np.exp(np.log(wide) * 2.0 + noise))
    f_narrow = cb.fit_power_law(narrow, np.exp(np.log(narrow) * 2.0 + noise))
    assert f_narrow.stderr > 5.0 * f_wide.stderr
    assert (f_narrow.ci_hi - f_narrow.ci_lo) > (f_wide.ci_hi - f_wide.ci_lo)


@pytest.mark.parametrize(
    "speed, value, msg",
    [
        (np.ones(5), np.arange(5.0) + 1.0, "unidentifiable"),
        (np.arange(2.0) + 1.0, np.arange(2.0) + 1.0, ">= 3 points"),
        (np.arange(4.0) + 1.0, np.arange(3.0) + 1.0, "points"),
    ],
)
def test_fit_power_law_rejects_degenerate_input(cb, speed, value, msg):
    with pytest.raises(ValueError, match=msg):
        cb.fit_power_law(speed, value)


# -------------------------------------------------------------- summary -> constants


def _synthetic_table(cb, **medians: float) -> pd.DataFrame:
    """A per-window table whose per-record medians are exactly the requested values."""
    rows = []
    for health, vals in medians.items():
        for i in range(4):
            for w in range(3):
                rows.append(
                    {
                        "file": f"{health}_{i}",
                        "health": health,
                        "state": 0 if health == "healthy" else 2,
                        "bearing_id": i + 1,
                        "rpm": 1700.0 + 50.0 * i,
                        "load": 400.0,
                        "win": w,
                        "dc": 0.0,
                        **vals,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic(cb):
    return _synthetic_table(
        cb,
        healthy=dict(rms=10.0, kurt=3.0, crest=4.5, bpfo_amp=1.0, bpfi_amp=1.0, sk_max=5.0),
        inner_race=dict(rms=60.0, kurt=9.0, crest=8.0, bpfo_amp=4.0, bpfi_amp=9.0, sk_max=30.0),
        outer_race=dict(rms=40.0, kurt=15.0, crest=12.0, bpfo_amp=8.0, bpfi_amp=6.0, sk_max=30.0),
    )


def test_summarise_reduces_windows_to_records_then_classes(cb, synthetic):
    cal = cb.summarise(synthetic)
    assert cal.n_windows == len(synthetic)
    assert cal.n_records == 12
    assert cal.med("healthy", "rms") == pytest.approx(10.0)
    assert cal.ratios["rms_pooled"] == pytest.approx(5.0)  # median of {40 x4, 60 x4} = 50
    assert cal.ratios["rms_outer_race"] == pytest.approx(4.0)
    assert cal.ratios["bpfo_amp_outer_race"] == pytest.approx(8.0)
    assert cal.ratios["bpfi_amp_inner_race"] == pytest.approx(9.0)
    assert set(cal.fits) >= {"healthy", "pooled_faulty"}
    assert cal.dc_dominated == 0


def test_recommend_constants_applies_the_documented_mapping_rules(cb, synthetic):
    from nebulax.sim.bearing import BearingParams

    p = BearingParams()
    cal = cb.summarise(synthetic)
    by_name = {u.name: u for u in cb.recommend_constants(cal, p)}

    # A_d = A_h * (ratio - 1) with the pooled faulty/healthy RMS ratio of 5.
    assert by_name["vib_rms_defect"].new == pytest.approx(round(p.vib_rms_healthy * 4.0, 2))
    # The measured class median is a LOWER BOUND on the bump peak: kurtosis 15 < the
    # incumbent peak of 22, so the gain is kept; crest 12 > the peak the gain would give,
    # so it is raised until the peak reaches 12.
    assert by_name["vib_kurt_gain"].new == p.vib_kurt_gain
    assert by_name["vib_kurt_gain"].verdict == "confirmed"
    assert by_name["vib_crest_gain"].new == pytest.approx(12.0 - p.vib_crest_healthy)
    # The s = 1 endpoint is held; only the measured 8x contrast is imposed.
    endpoint = p.vib_bpfo_healthy + p.vib_bpfo_gain
    assert by_name["vib_bpfo_healthy"].new == pytest.approx(round(endpoint / 8.0, 3))
    assert by_name["vib_bpfo_healthy"].new + by_name["vib_bpfo_gain"].new == pytest.approx(
        endpoint, abs=1e-3
    )
    # Healthy anchors are confirmations, never rewrites.
    for name in ("vib_kurt_healthy", "vib_crest_healthy"):
        assert by_name[name].verdict == "confirmed"
        assert by_name[name].new == by_name[name].old
    assert all(u.evidence for u in by_name.values())


def test_the_report_renders_every_constant(cb, synthetic):
    cal = cb.summarise(synthetic)
    text = cb.report(cal, cb.recommend_constants(cal))
    for name in ("vib_rms_defect", "vib_crest_gain", "vib_bpfo_healthy", "vib_speed_exp_healthy"):
        assert name in text
    assert "AC-coupled" in text


# ------------------------------------------- the live simulator defaults are the calibration


def test_bearing_params_still_carry_the_recorded_ottawa_calibration():
    """Pins ``nebulax.sim.bearing`` to the numbers recorded in ``docs/parameters.md``.

    These came out of a full run of ``scripts/calibrate_bearing.py`` over the 60-record
    ``data/raw/ottawa`` set. If a later change moves one of them, either the calibration was
    re-run (update this table and the doc) or the change is unexplained.
    """
    from nebulax.sim.bearing import BearingParams

    p = BearingParams()
    assert p.vib_rms_defect == pytest.approx(2.94)      # pooled faulty/healthy AC RMS 5.74x
    assert p.vib_crest_gain == pytest.approx(7.32)      # outer-race crest 11.82, healthy 4.62
    assert p.vib_bpfo_healthy == pytest.approx(0.149)   # BPFO floor: outer race is only 10.05x
    assert p.vib_bpfo_gain == pytest.approx(1.351)      # endpoint held at 1.50 m/s2
    assert p.vib_kurt_gain == pytest.approx(19.0)       # [R157] peak 22 > Ottawa's 16.75
    assert p.vib_kurt_healthy == pytest.approx(3.0)     # Ottawa healthy 3.15 confirms
    assert p.vib_crest_healthy == pytest.approx(4.5)    # Ottawa healthy 4.62 confirms
    assert p.vib_speed_exp_healthy == pytest.approx(2.0)   # Ottawa CI [-16.8, +4.2]: unrefuted
    assert p.vib_speed_exp_defect == pytest.approx(1.2)


def test_the_bpfo_law_now_has_a_healthy_floor():
    """``vib_bpfo`` must not be exactly zero on a healthy box - see rail_phm 4.3.2."""
    from nebulax.sim.bearing import BearingParams

    p = BearingParams()
    healthy = p.vib_bpfo_healthy
    faulty = p.vib_bpfo_healthy + p.vib_bpfo_gain
    assert healthy > 0.0
    assert 8.0 < faulty / healthy < 13.0


# ------------------------------------------------------------------------- integration


@pytest.mark.skipif(not (RAW_DIR / "H_1_0.csv").exists(), reason="data/raw/ottawa not downloaded")
def test_one_real_healthy_record_lands_on_the_anchors(cb):
    table = cb.extract_table(RAW_DIR, files=[RAW_DIR / "H_1_0.csv"])
    assert len(table) == 10  # a 10 s recording, 1 s windows
    assert table["health"].unique().tolist() == ["healthy"]
    assert table["rpm"].nunique() == 1 and 1500.0 < table["rpm"].iloc[0] < 2300.0
    assert np.isfinite(table[["rms", "kurt", "crest", "bpfo_amp"]].to_numpy()).all()
    assert 2.5 < table["kurt"].median() < 4.5     # the ~3 anchor
    assert 3.5 < table["crest"].median() < 6.0    # the ~4.5 anchor
    assert (table["rms"] > 0).all()


def test_window_stats_ac_agrees_with_the_features_module_ac_couple(cb, rng):
    """The calibration's float64 reference and nebulax.features.stats.window_stats(
    ac_couple=True) - what nebulax.adapters.ottawa now writes as vib_rms/vib_crest - must be
    the same statistics, so the adapter columns and these constants stay comparable."""
    from nebulax.features import stats as ST

    x = rng.standard_normal((5, 4096)) * 30.0 + 1600.0  # H_2_0-like: DC bias >> AC RMS
    ref = cb.window_stats_ac(x)
    got = ST.window_stats(x[:, :, None], ac_couple=True)
    idx = {name: i for i, name in enumerate(ST.AC_STAT_NAMES)}
    np.testing.assert_allclose(got[:, idx["rms_ac"]], ref["rms"], rtol=1e-5)
    np.testing.assert_allclose(got[:, idx["crest_ac"]], ref["crest"], rtol=1e-5)
    np.testing.assert_allclose(got[:, idx["kurtosis"]], ref["kurt"], rtol=1e-4)
    np.testing.assert_allclose(got[:, idx["mean"]], ref["dc"], rtol=1e-5)
