"""Tests for nebulax.features.windows.make_windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nebulax.features import windows as W


def _long_two_components(n=50, freq="10ms"):
    ts = pd.date_range("2026-09-18T09:00:00Z", periods=n, freq=freq, tz="UTC").as_unit("ms")
    frames = []
    for cid, offset in (("door_L1", 0.0), ("door_L2", 100.0)):
        for sig in ("pos", "current"):
            base = np.arange(n, dtype=np.float64) + offset + (0.0 if sig == "pos" else 1000.0)
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": ts,
                        "run_id": "run0",
                        "train_id": "T01",
                        "component_id": cid,
                        "signal": sig,
                        "value": base,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def test_make_windows_shapes_and_channel_order():
    df = _long_two_components(n=50)
    win = W.make_windows(df, signals=["pos", "current"], L=10, stride=5, fs=100.0)
    # each component has 50 samples -> (50 - 10)//5 + 1 = 9 windows; 2 components -> 18
    assert win.X.shape == (18, 10, 2)
    assert len(win) == 18
    assert win.signals == ("pos", "current")
    assert set(win.train_id) == {"T01"}
    assert set(win.component_id) == {"door_L1", "door_L2"}
    assert win.run_id is not None and set(win.run_id) == {"run0"}


def test_make_windows_values_are_contiguous_and_correctly_placed():
    df = _long_two_components(n=50)
    win = W.make_windows(df, signals=["pos", "current"], L=10, stride=5, fs=100.0)
    mask = win.component_id == "door_L1"
    x_pos = win.X[mask, :, 0]
    # door_L1's 'pos' channel is exactly arange(50); first window must be [0..9]
    first_window = x_pos[np.argsort(win.t_end[mask])][0]
    np.testing.assert_allclose(first_window, np.arange(10, dtype=np.float32))


def test_make_windows_t_end_matches_last_sample_timestamp():
    df = _long_two_components(n=30)
    win = W.make_windows(df, signals=["pos"], L=10, stride=10, fs=100.0)
    # tiling stride==L -> exactly 3 windows per component, t_end must be strictly increasing
    # within a component.
    mask = win.component_id == "door_L1"
    t_end_sorted = pd.Series(win.t_end[mask]).sort_values().reset_index(drop=True)
    assert len(t_end_sorted) == 3
    assert t_end_sorted.is_monotonic_increasing and t_end_sorted.nunique() == 3


def test_make_windows_too_short_group_is_dropped():
    df = _long_two_components(n=5)
    win = W.make_windows(df, signals=["pos"], L=10, stride=5, fs=100.0)
    assert len(win) == 0
    assert win.X.shape == (0, 10, 1)


def test_make_windows_missing_signal_filled_with_nan():
    df = _long_two_components(n=20)
    win = W.make_windows(df, signals=["pos", "does_not_exist"], L=10, stride=10, fs=100.0)
    assert win.X.shape[-1] == 2
    assert np.isnan(win.X[:, :, 1]).all()
    assert not np.isnan(win.X[:, :, 0]).any()


def test_make_windows_accepts_wide_input():
    ts = pd.date_range("2026-09-18T09:00:00Z", periods=20, freq="1s", tz="UTC").as_unit("ms")
    wide = pd.DataFrame(
        {
            "timestamp": ts,
            "train_id": "T01",
            "component_id": "apu_1",
            "TP2": np.linspace(8.0, 10.0, 20),
        }
    )
    win = W.make_windows(wide, signals=["TP2"], L=5, stride=5, fs=1.0)
    assert win.X.shape == (4, 5, 1)
    np.testing.assert_allclose(win.X[0, :, 0], wide["TP2"].to_numpy()[:5], rtol=1e-5)


@pytest.mark.parametrize("bad_kwargs", [dict(L=0, stride=1), dict(L=1, stride=0)])
def test_make_windows_rejects_non_positive_L_or_stride(bad_kwargs):
    df = _long_two_components(n=20)
    with pytest.raises(ValueError):
        W.make_windows(df, signals=["pos"], fs=100.0, **bad_kwargs)


def test_make_windows_rejects_missing_key_columns():
    df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=5, tz="UTC"), "value": range(5)})
    with pytest.raises(ValueError):
        W.make_windows(df, signals=["x"], L=2, stride=1, fs=1.0)


# ------------------------------------------------- acquisition gaps (MetroPT-3 outages)


def _wide_with_hole(n_before=25, n_after=25, hole_s=3600.0, period_s=1.0):
    """A 1 Hz MetroPT-like analogue trace with a ``hole_s`` outage in the middle.

    The channel value is *seconds since the first sample*, so any window that bridged the
    hole would show a jump of ``hole_s`` between two adjacent samples.
    """
    t0 = pd.Timestamp("2020-02-01T09:00:00Z")
    secs = np.concatenate(
        [
            np.arange(n_before, dtype=np.float64) * period_s,
            (n_before - 1) * period_s + hole_s + np.arange(1, n_after + 1, dtype=np.float64) * period_s,
        ]
    )
    return pd.DataFrame(
        {
            "timestamp": (t0 + pd.to_timedelta(secs, unit="s")).as_unit("ms"),
            "train_id": "metro_1",
            "component_id": "apu_1",
            "TP2": secs,
        }
    )


def test_make_windows_does_not_bridge_a_one_hour_logger_outage():
    df = _wide_with_hole(n_before=25, n_after=25, hole_s=3600.0)
    win = W.make_windows(df, signals=["TP2"], L=10, stride=5, fs=1.0)
    # two gap-free runs of 25 samples -> (25 - 10)//5 + 1 = 4 windows each, never 9 as a
    # slice-by-row-position over the concatenated 50 rows would give.
    assert len(win) == 8
    assert win.max_gap_s == pytest.approx(3.0)
    # no window may contain the 1 h jump: the channel is seconds-since-start, so every
    # within-window step must be one sample period.
    steps = np.diff(win.X[:, :, 0], axis=1)
    assert float(np.abs(steps).max()) == pytest.approx(1.0, abs=1e-3)
    # and every window ends inside one of the two runs, never in the hole
    hole_start = df["timestamp"].iloc[24]
    hole_end = df["timestamp"].iloc[25]
    t_end = pd.DatetimeIndex(win.t_end)
    assert not ((t_end > hole_start) & (t_end < hole_end)).any()


def test_make_windows_max_gap_inf_restores_slice_by_row_position():
    df = _wide_with_hole(n_before=25, n_after=25, hole_s=3600.0)
    win = W.make_windows(df, signals=["TP2"], L=10, stride=5, fs=1.0, max_gap_s=float("inf"))
    assert len(win) == 9  # the old, gap-blind count
    assert float(np.abs(np.diff(win.X[:, :, 0], axis=1)).max()) > 3000.0


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("tz", [None, "UTC"])
def test_make_windows_gap_guard_is_resolution_and_tz_independent(tz, unit):
    """The 1 h outage must be seen at every datetime64 resolution, tz-aware or not.

    A tz-aware index carries a pandas ``DatetimeTZDtype`` (which exposes ``.unit``); a
    tz-naive one carries a bare numpy ``datetime64[unit]`` dtype (which does not). Reading
    the resolution off the dtype therefore silently defaulted to nanoseconds for tz-naive
    non-ns traces and measured the 3600 s hole as 0.0036 s, so no cut was made.
    """
    df = _wide_with_hole(n_before=25, n_after=25, hole_s=3600.0)
    ts = pd.DatetimeIndex(df["timestamp"])
    df["timestamp"] = (ts.tz_convert(None) if tz is None else ts).as_unit(unit)
    win = W.make_windows(df, signals=["TP2"], L=10, stride=5, fs=1.0)
    assert len(win) == 8
    assert float(np.abs(np.diff(win.X[:, :, 0], axis=1)).max()) == pytest.approx(1.0, abs=1e-3)


def test_make_windows_gap_threshold_is_three_sample_periods_by_default():
    # a 2-period hiccup is jitter and stays inside a window; a 4-period one is an outage
    tolerated = _wide_with_hole(n_before=10, n_after=10, hole_s=2.0)
    assert len(W.make_windows(tolerated, signals=["TP2"], L=10, stride=10, fs=1.0)) == 2
    split = _wide_with_hole(n_before=10, n_after=10, hole_s=4.0)
    assert len(W.make_windows(split, signals=["TP2"], L=10, stride=10, fs=1.0)) == 2
    # with L covering both runs at once, only the tolerated hiccup can produce a window
    assert len(W.make_windows(tolerated, signals=["TP2"], L=20, stride=20, fs=1.0)) == 1
    assert len(W.make_windows(split, signals=["TP2"], L=20, stride=20, fs=1.0)) == 0


def test_make_windows_explicit_max_gap_overrides_the_fs_default():
    df = _wide_with_hole(n_before=10, n_after=10, hole_s=60.0)
    assert len(W.make_windows(df, signals=["TP2"], L=20, stride=20, fs=1.0, max_gap_s=120.0)) == 1
    assert len(W.make_windows(df, signals=["TP2"], L=20, stride=20, fs=1.0, max_gap_s=30.0)) == 0


def test_make_windows_nat_timestamp_never_lands_inside_a_window():
    df = _wide_with_hole(n_before=10, n_after=10, hole_s=1.0)  # otherwise a regular grid
    df.loc[10, "timestamp"] = pd.NaT  # one unusable timestamp out of 20
    # the NaT row sorts to the end and is a gap on both sides, so it is never inside a window:
    # 19 usable rows at L=5/stride=5 -> 3 windows, and no window ends on a NaT.
    win = W.make_windows(df, signals=["TP2"], L=5, stride=5, fs=1.0)
    assert len(win) == 3
    assert not pd.DatetimeIndex(win.t_end).isna().any()
    # a window long enough to need the NaT row cannot be formed at all
    assert len(W.make_windows(df, signals=["TP2"], L=20, stride=20, fs=1.0)) == 0


def test_make_windows_rejects_non_positive_max_gap():
    df = _long_two_components(n=20)
    with pytest.raises(ValueError):
        W.make_windows(df, signals=["pos"], L=10, stride=5, fs=100.0, max_gap_s=0.0)
