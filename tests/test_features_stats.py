"""Tests for nebulax.features.stats.window_stats."""

from __future__ import annotations

import numpy as np
import pytest

from nebulax.features import stats as ST


def test_window_stats_shape_and_names():
    n, L, c = 5, 20, 3
    X = np.random.default_rng(0).normal(size=(n, L, c)).astype(np.float32)
    out = ST.window_stats(X)
    assert out.shape == (n, c * len(ST.STAT_NAMES))
    names = ST.feature_names(["a", "b", "c"])
    assert len(names) == out.shape[1]
    assert names[: len(ST.STAT_NAMES)] == [f"a_{s}" for s in ST.STAT_NAMES]


def test_window_stats_constant_channel():
    n, L, c = 4, 16, 1
    X = np.full((n, L, c), 7.0, dtype=np.float32)
    out = ST.window_stats(X)
    idx = {name: i for i, name in enumerate(ST.STAT_NAMES)}
    np.testing.assert_allclose(out[:, idx["mean"]], 7.0)
    np.testing.assert_allclose(out[:, idx["std"]], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[:, idx["min"]], 7.0)
    np.testing.assert_allclose(out[:, idx["max"]], 7.0)
    np.testing.assert_allclose(out[:, idx["slope"]], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[:, idx["rms"]], 7.0)
    # crest = peak/rms = 1 for a nonzero constant signal
    np.testing.assert_allclose(out[:, idx["crest"]], 1.0, atol=1e-5)


def test_window_stats_linear_ramp_slope():
    n, L, c = 3, 50, 1
    t = np.arange(L, dtype=np.float64)
    slopes_true = np.array([0.5, -2.0, 3.0])
    X = np.stack([slopes_true[i] * t + 10.0 for i in range(n)], axis=0)[:, :, None].astype(np.float32)
    out = ST.window_stats(X)
    idx = list(ST.STAT_NAMES).index("slope")
    np.testing.assert_allclose(out[:, idx], slopes_true, rtol=1e-4)


def test_window_stats_sine_rms_and_crest():
    L = 4096
    t = np.arange(L)
    x = 2.0 * np.sin(2 * np.pi * 5 * t / L)
    X = x[None, :, None].astype(np.float32)
    out = ST.window_stats(X)
    idx = {name: i for i, name in enumerate(ST.STAT_NAMES)}
    # RMS of a sine with amplitude A is A/sqrt(2); crest factor is sqrt(2)
    assert out[0, idx["rms"]] == pytest.approx(2.0 / np.sqrt(2), rel=1e-2)
    assert out[0, idx["crest"]] == pytest.approx(np.sqrt(2), rel=1e-2)
    # a sine's kurtosis is 1.5, well below the Gaussian value of 3
    assert out[0, idx["kurtosis"]] == pytest.approx(1.5, rel=5e-2)


def test_window_stats_ignores_nan_samples():
    L = 32
    x = np.arange(L, dtype=np.float64)
    x[5] = np.nan
    x[10] = np.nan
    X = x[None, :, None].astype(np.float32)
    out = ST.window_stats(X)
    idx = list(ST.STAT_NAMES).index("mean")
    expected = np.nanmean(x)
    assert out[0, idx] == pytest.approx(expected, rel=1e-4)


def test_window_stats_rejects_bad_shape():
    with pytest.raises(ValueError):
        ST.window_stats(np.zeros((5, 5)))
    with pytest.raises(ValueError):
        ST.window_stats(np.zeros((5, 1, 3)))


def test_window_stats_empty_input():
    out = ST.window_stats(np.zeros((0, 10, 2), dtype=np.float32))
    assert out.shape == (0, 2 * len(ST.STAT_NAMES))


# ----------------------------------------------------------------- AC coupling (ac_couple=True)


def test_ac_stat_names_rename_only_the_two_affected_slots():
    assert len(ST.AC_STAT_NAMES) == len(ST.STAT_NAMES)
    assert ST.stat_names() == ST.STAT_NAMES
    assert ST.stat_names(True) == ST.AC_STAT_NAMES
    renamed = {dc: ac for dc, ac in zip(ST.STAT_NAMES, ST.AC_STAT_NAMES) if dc != ac}
    assert renamed == {"rms": "rms_ac", "crest": "crest_ac"}
    names = ST.feature_names(["v"], ac_couple=True)
    assert names == [f"v_{s}" for s in ST.AC_STAT_NAMES]
    assert len(names) == ST.window_stats(np.zeros((2, 8, 1)), ac_couple=True).shape[1]


def test_window_stats_ac_couple_is_invariant_to_a_dc_offset(rng):
    """The Ottawa fix: 21 of 60 records carry a DC bias larger than their own AC RMS, and a
    DC-coupled crest reads ~1 on those (docs/parameters.md, bearing section 4)."""
    X = rng.standard_normal((4, 2048, 2))
    idx = {name: i for i, name in enumerate(ST.AC_STAT_NAMES)}
    base = ST.window_stats(X, ac_couple=True)
    shifted = ST.window_stats(X + 500.0, ac_couple=True)
    for stat in ("rms_ac", "crest_ac", "std", "kurtosis"):
        cols = [k * len(ST.STAT_NAMES) + idx[stat] for k in range(2)]
        np.testing.assert_allclose(base[:, cols], shifted[:, cols], rtol=1e-4)
    # the offset is not hidden: mean still reports it, and the DC-coupled call is still wrecked
    mean_cols = [k * len(ST.STAT_NAMES) + idx["mean"] for k in range(2)]
    np.testing.assert_allclose(shifted[:, mean_cols] - base[:, mean_cols], 500.0, rtol=1e-4)
    dc = ST.window_stats(X + 500.0)
    crest_cols = [k * len(ST.STAT_NAMES) + idx["crest_ac"] for k in range(2)]
    assert (dc[:, crest_cols] < 1.2).all()
    assert (base[:, crest_cols] > 3.0).all()


def test_window_stats_ac_couple_changes_only_rms_and_crest(rng):
    X = rng.standard_normal((3, 512, 2)) + 17.0
    dc = ST.window_stats(X)
    ac = ST.window_stats(X, ac_couple=True)
    assert ac.shape == dc.shape == (3, 2 * len(ST.STAT_NAMES))
    idx = {name: i for i, name in enumerate(ST.STAT_NAMES)}
    for stat in ("mean", "std", "min", "max", "slope", "kurtosis"):
        cols = [k * len(ST.STAT_NAMES) + idx[stat] for k in range(2)]
        np.testing.assert_allclose(dc[:, cols], ac[:, cols], rtol=1e-5, atol=1e-5)
    rms_cols = [k * len(ST.STAT_NAMES) + idx["rms"] for k in range(2)]
    # sqrt(rms_dc^2 - mean^2) is the AC RMS; and removing the DC can only lower the RMS
    mean_cols = [k * len(ST.STAT_NAMES) + idx["mean"] for k in range(2)]
    np.testing.assert_allclose(
        ac[:, rms_cols], np.sqrt(dc[:, rms_cols] ** 2 - dc[:, mean_cols] ** 2), rtol=1e-3
    )
    assert (ac[:, rms_cols] < dc[:, rms_cols]).all()


def test_window_stats_ac_couple_sine_on_a_pedestal():
    """A sine on a big pedestal: AC-coupled it is still A/sqrt(2) RMS and sqrt(2) crest."""
    L = 4096
    x = 2.0 * np.sin(2 * np.pi * 5 * np.arange(L) / L) + 900.0
    X = x[None, :, None]
    out = ST.window_stats(X, ac_couple=True)
    idx = {name: i for i, name in enumerate(ST.AC_STAT_NAMES)}
    assert out[0, idx["rms_ac"]] == pytest.approx(2.0 / np.sqrt(2), rel=1e-3)
    assert out[0, idx["crest_ac"]] == pytest.approx(np.sqrt(2), rel=1e-3)
    assert out[0, idx["mean"]] == pytest.approx(900.0, rel=1e-6)


def test_window_stats_ac_couple_ignores_nan_samples():
    L = 64
    x = np.sin(np.arange(L) / 3.0) + 50.0
    x[7] = np.nan
    X = x[None, :, None]
    out = ST.window_stats(X, ac_couple=True)
    idx = {name: i for i, name in enumerate(ST.AC_STAT_NAMES)}
    finite = x[np.isfinite(x)]
    d = finite - finite.mean()
    assert out[0, idx["rms_ac"]] == pytest.approx(np.sqrt(np.mean(d * d)), rel=1e-5)
    assert out[0, idx["crest_ac"]] == pytest.approx(
        np.max(np.abs(d)) / np.sqrt(np.mean(d * d)), rel=1e-5
    )


def test_window_stats_ac_couple_constant_channel_has_no_crest():
    """A constant window has zero AC amplitude - crest_ac is undefined (NaN), not 1."""
    out = ST.window_stats(np.full((2, 16, 1), 7.0), ac_couple=True)
    idx = {name: i for i, name in enumerate(ST.AC_STAT_NAMES)}
    np.testing.assert_allclose(out[:, idx["rms_ac"]], 0.0, atol=1e-6)
    assert np.isnan(out[:, idx["crest_ac"]]).all()
    np.testing.assert_allclose(out[:, idx["mean"]], 7.0)


def test_fewer_than_two_finite_samples_gives_all_nan():
    """Regression (Codex audit 15 Sep 2026): one finite sample must not read as a valid row."""
    import numpy as np
    from nebulax.features.stats import STAT_NAMES, window_stats

    X = np.array([[[7.0, 1.0], [np.nan, 2.0], [np.nan, 3.0], [np.nan, 4.0]]])  # (1, 4, 2)
    out = window_stats(X)
    k = len(STAT_NAMES)
    assert np.all(np.isnan(out[0, :k])), out[0, :k]  # channel 0: one finite sample
    assert np.all(np.isfinite(out[0, k:2 * k])), out[0, k:]  # channel 1 untouched
    X2 = np.array([[[7.0], [8.0], [np.nan]]])  # two finite samples: still computed
    assert np.isfinite(window_stats(X2)[0, 0])
