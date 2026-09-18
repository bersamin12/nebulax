"""Thresholds: the false-alarm budget is respected, quantiles are reported beside it, and the
API refuses anything that is not the validation slice."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nebulax.bench import metrics as M
from nebulax.bench import thresholds as TH

HOUR = 3600.0


def _val(n=2000, seed=0, n_spikes=6):
    rng = np.random.default_rng(seed)
    t = (pd.Timestamp("2020-04-01") + pd.to_timedelta(np.arange(n) * 600.0, unit="s")).to_numpy().astype("datetime64[ms]")
    s = rng.normal(0.0, 1.0, n)
    for k in range(n_spikes):
        start = 100 + k * 250
        s[start : start + 4] += 8.0
    units = np.full(n, "porto_apu", dtype=object)
    return TH.ValidationScores(scores=s, times=t, units=units, index=np.arange(n))


def test_validation_scores_refuses_a_non_val_part():
    with pytest.raises(TH.TestScoresLeakedError, match="validation slice"):
        TH.ValidationScores(
            scores=np.zeros(3), times=np.arange(3), units=np.array(["a"] * 3, dtype=object),
            index=np.arange(3), part="test",
        )


def test_calibrate_refuses_a_bare_array():
    with pytest.raises(TypeError, match="ValidationScores"):
        TH.calibrate(np.zeros(10))  # type: ignore[arg-type]


def test_budget_threshold_respects_the_false_alarm_budget():
    vs = _val()
    ts = TH.calibrate(vs, window_seconds=600.0)
    budget = ts.primary
    # the slice is (2000-1)*600 s ~ 13.9 train-days -> <= 1/7 episodes/day allows ~1.98
    assert budget.val_train_days == pytest.approx((2000 - 1) * 600.0 / 86400.0)
    assert budget.val_far_per_train_day <= TH.DEFAULT_FAR_BUDGET + 1e-9
    assert budget.val_episodes <= TH.DEFAULT_FAR_BUDGET * budget.val_train_days + 1e-9

    # and it is not needlessly conservative: dropping just below it breaks the budget
    lower = float(np.nextafter(budget.value, -np.inf))
    n_eps = len(M.episodes(vs.scores, vs.times, lower, k=3, merge_gap_s=HOUR, units=vs.units))
    assert n_eps >= budget.val_episodes


def test_tighter_budget_gives_a_higher_threshold():
    vs = _val()
    loose = TH.calibrate(vs, far_budget_episodes_per_train_day=1.0).primary.value
    tight = TH.calibrate(vs, far_budget_episodes_per_train_day=1 / 30).primary.value
    assert tight >= loose


def test_quantile_thresholds_are_reported_beside_the_budget():
    vs = _val()
    ts = TH.calibrate(vs, window_seconds=600.0)
    assert set(ts.thresholds) == {"budget", "q995", "q999"}
    assert ts["q995"].value == pytest.approx(float(np.quantile(vs.scores, 0.995)))
    assert ts["q999"].value == pytest.approx(float(np.quantile(vs.scores, 0.999)))
    assert ts["q999"].value >= ts["q995"].value
    row = ts.as_dict()
    assert row["threshold_budget"] == ts.primary.value
    assert row["n_val_rows"] == len(vs)


def test_zero_budget_gives_a_silent_threshold_rather_than_crashing():
    # a zero false-alarm budget can only be met by never firing; say so, do not crash
    vs = _val(n=400)
    ts = TH.calibrate(vs, far_budget_episodes_per_train_day=0.0)
    assert ts.primary.val_episodes == 0
    assert ts.primary.meta["silent"] is True
    assert ts.primary.value > float(np.quantile(vs.scores, 0.99))


def test_non_binding_budget_is_flagged():
    # an all-quiet score: even the most sensitive candidate raises nothing, so the budget is
    # not what sets the threshold.
    vs = _val(n=400, n_spikes=0)
    ts = TH.calibrate(vs, far_budget_episodes_per_train_day=10.0)
    assert ts.primary.meta["budget_binding"] is False


def test_calibration_records_its_own_index_for_audit():
    vs = _val(n=500)
    ts = TH.calibrate(vs)
    assert np.array_equal(ts.calibrated_on_index, np.arange(500))
    assert ts.n_val_rows == 500


def test_empty_or_non_finite_scores_raise():
    t = np.arange(4).astype("float64")
    with pytest.raises(ValueError, match="non-finite"):
        TH.calibrate(
            TH.ValidationScores(
                scores=np.full(4, np.nan), times=t, units=np.array(["u"] * 4, dtype=object), index=np.arange(4)
            )
        )


def test_calibration_counts_episodes_in_row_pitch_not_window_duration():
    """A per-cycle validation slice (36 s cycles every 155 s): the spikes are 4 consecutive
    rows and must be visible to the budget scan, so the chosen threshold sits above the
    spikes' shoulders and is not the mute, non-binding median a window-derived budget gives."""
    rng = np.random.default_rng(1)
    n = 3000
    t = (pd.Timestamp("2020-04-01") + pd.to_timedelta(np.arange(n) * 155.0, unit="s")).to_numpy().astype("datetime64[ms]")
    s = rng.normal(0.0, 1.0, n)
    for k in range(8):
        s[200 + k * 300 : 204 + k * 300] += 8.0
    vs = TH.ValidationScores(scores=s, times=t, units=np.full(n, "T01", dtype=object), index=np.arange(n))
    ts = TH.calibrate(vs, window_seconds=36.1)
    budget = ts.primary
    n_days = (n - 1) * 155.0 / 86400.0  # ~5.4 train-days -> budget allows 0 whole episodes
    assert budget.val_train_days == pytest.approx(n_days)
    assert budget.val_episodes <= TH.DEFAULT_FAR_BUDGET * n_days + 1e-9
    assert budget.value > 4.0, "spikes of 4 consecutive cycles are episodes; the threshold must clear them"
    assert budget.meta["budget_binding"] is True
    assert budget.meta["max_step_s"] == pytest.approx(155.0 * 1.5)
    # at a threshold inside the spikes the 8 episodes are visible and break the budget ...
    step = M.max_step_seconds(36.1, M.row_pitch_seconds(vs.times, vs.units))
    assert len(M.episodes(vs.scores, vs.times, 4.0, k=3, merge_gap_s=HOUR, units=vs.units, max_step_s=step)) == 8
    # ... whereas a window-derived step (96 s < 155 s pitch) could never see any of them
    assert M.episodes(vs.scores, vs.times, 4.0, k=3, merge_gap_s=HOUR, units=vs.units, max_step_s=M.max_step_seconds(36.1)) == []


def test_budget_not_applicable_on_a_fabricated_timeline():
    """Ottawa-like: 60 records 20 s apart. The budget cannot be counted in train-days, so the
    'budget' threshold is the 99.5 % quantile with NaN train-days and rate, never a 'no alarm
    at all' threshold that reads as silent."""
    rng = np.random.default_rng(2)
    n = 600
    t = (pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n) * 1.0 + (np.arange(n) // 10) * 10.0, unit="s")).to_numpy().astype("datetime64[ms]")
    s = rng.normal(size=n)
    s[100:110] = 12.0 + 3.0 * np.arange(10)  # the top-3 scores are consecutive rows of one record
    series = np.repeat([f"rec{i}" for i in range(60)], 10).astype(object)
    vs = TH.ValidationScores(scores=s, times=t, units=np.repeat([f"b{i}" for i in range(20)], 30).astype(object), index=np.arange(n), series=series)
    ts = TH.calibrate(vs, window_seconds=1.0, budget_applicable=False)
    b = ts.primary
    assert b.value == pytest.approx(float(np.quantile(s, 0.995)))
    assert np.isnan(b.val_train_days) and np.isnan(b.val_far_per_train_day)
    assert b.meta["budget_applicable"] is False and "not applicable" in b.method
    assert b.val_episodes >= 1 and b.meta["silent"] is False
    assert set(ts.thresholds) >= {"budget", "q995", "q999"} or len(ts.thresholds) == 3
