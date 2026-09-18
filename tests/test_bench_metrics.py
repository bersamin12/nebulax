"""Metrics: episodes/merging, event recall with H, false alarms per train-day, a hand-computed
VUS-PR case, and the CPD metrics + score adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nebulax.bench import metrics as M

HOUR = 3600.0


def _times(n: int, step_s: float = 600.0, start: str = "2020-04-01T00:00:00") -> np.ndarray:
    return (pd.Timestamp(start) + pd.to_timedelta(np.arange(n) * step_s, unit="s")).to_numpy().astype("datetime64[ms]")


# --------------------------------------------------------------------------------------
# episodes
# --------------------------------------------------------------------------------------


def test_episode_needs_k_consecutive_windows():
    scores = np.zeros(20)
    scores[[2, 3]] = 5.0  # a 2-window blip
    scores[[10, 11, 12]] = 5.0  # a real 3-window run
    eps = M.episodes(scores, _times(20), threshold=1.0, k=3, merge_gap_s=0.0)
    assert len(eps) == 1
    assert (eps[0].i_start, eps[0].i_end, eps[0].n_windows) == (10, 12, 3)
    assert eps[0].peak_score == 5.0


def test_threshold_is_strictly_greater():
    scores = np.full(6, 1.0)
    assert M.episodes(scores, _times(6), threshold=1.0, k=3) == []
    assert len(M.episodes(scores, _times(6), threshold=0.999, k=3)) == 1


def test_merging_under_the_gap_and_not_over_it():
    # windows are 10 min apart; two 3-window runs separated by 3 windows (30 min) -> merged at
    # the 1 h protocol gap, separate when the gap rule is tightened to 10 min.
    scores = np.zeros(20)
    scores[0:3] = 9.0
    scores[6:9] = 9.0
    merged = M.episodes(scores, _times(20), threshold=1.0, k=3, merge_gap_s=HOUR)
    assert len(merged) == 1 and merged[0].n_windows == 9  # the union spans indices 0..8
    split = M.episodes(scores, _times(20), threshold=1.0, k=3, merge_gap_s=600.0)
    assert len(split) == 2


def test_episodes_never_cross_units():
    scores = np.r_[np.full(4, 9.0), np.full(4, 9.0)]
    units = np.array(["T01"] * 4 + ["T02"] * 4, dtype=object)
    t = np.r_[_times(4), _times(4)]  # identical timestamps on both units
    eps = M.episodes(scores, t, threshold=1.0, k=3, merge_gap_s=HOUR, units=units)
    assert sorted(e.unit for e in eps) == ["T01", "T02"]


def test_episodes_sort_unsorted_input_by_time():
    scores = np.zeros(10)
    scores[5:8] = 9.0
    t = _times(10)
    perm = np.array([9, 0, 5, 6, 7, 1, 2, 3, 4, 8])
    eps = M.episodes(scores[perm], t[perm], threshold=1.0, k=3)
    assert len(eps) == 1 and eps[0].n_windows == 3


# --------------------------------------------------------------------------------------
# event metrics
# --------------------------------------------------------------------------------------


def test_event_recall_respects_H_and_counts_false_alarms():
    t = _times(300)  # 300 windows x 10 min ~ 50 h on one unit
    scores = np.zeros(300)
    scores[100:104] = 9.0  # inside the horizon of the event below
    scores[10:14] = 9.0  # 15 h earlier -> outside a 6 h horizon -> false alarm
    t_s = M.to_epoch_seconds(t)
    units = np.full(300, "T01", dtype=object)
    ev = M.Event("T01", t_onset=float(t_s[120]), t_failure=float(t_s[140]))
    eps = M.episodes(scores, t, threshold=1.0, k=3, merge_gap_s=HOUR, units=units)
    assert len(eps) == 2

    tight = M.event_metrics(eps, [ev], H=6 * HOUR, times=t, units=units)
    assert tight["n_events"] == 1 and tight["n_events_detected"] == 1
    assert tight["event_recall"] == 1.0
    assert tight["n_false_alarms"] == 1
    # the detecting episode starts at index 100, failure at 140 -> 40 windows x 600 s
    assert tight["median_lead_time_s"] == pytest.approx(40 * 600.0)

    # With a horizon so short that no episode falls inside it, recall is 0 and BOTH episodes
    # become false alarms.
    none = M.event_metrics(eps, [ev], H=60.0, times=t, units=units)
    assert none["event_recall"] == 0.0 and none["n_false_alarms"] == 2


def test_false_alarms_per_train_day_uses_observed_span():
    t = _times(145, step_s=HOUR)  # 144 h = 6 days on one unit
    assert M.train_days(t) == pytest.approx(6.0)
    scores = np.zeros(145)
    scores[10:13] = 9.0
    eps = M.episodes(scores, t, threshold=1.0, k=3, merge_gap_s=HOUR)
    out = M.event_metrics(eps, [], H=HOUR, times=t)
    assert out["n_false_alarms"] == 1
    assert out["false_alarms_per_train_day"] == pytest.approx(1 / 6.0)
    assert np.isnan(out["event_recall"])  # no events: undefined, not zero


def test_train_days_sums_over_units():
    t = np.r_[_times(25, step_s=HOUR), _times(49, step_s=HOUR)]
    units = np.array(["A"] * 25 + ["B"] * 49, dtype=object)
    assert M.train_days(t, units) == pytest.approx(1.0 + 2.0)


# --------------------------------------------------------------------------------------
# VUS-PR
# --------------------------------------------------------------------------------------


#: Reference values produced by running the **upstream** TSB-AD code
#: (``TSB_AD/evaluation/basic_metrics.py`` at commit 6beac72e…, fetched from GitHub) on these
#: exact inputs: ``generate_curve(y, s, w, "opt", 250)[-1]`` for VUS-PR and
#: ``RangeAUC(labels=y, score=s, window=w, plot_ROC=True)[1]`` for R-AUC-PR. They are pasted
#: here so the test needs no network. ``None`` = upstream raises ZeroDivisionError on that
#: input (its buffered-label segmenter finds no segment); our port returns NaN there.
TSB_AD_REFERENCE = [
    ("six_point", [0, 0, 1, 1, 0, 0], [0.1, 0.95, 0.9, 0.8, 0.3, 0.0], 0, 0.5833333333333333, 0.5833333333333333),
    ("six_point_w4", [0, 0, 1, 1, 0, 0], [0.1, 0.95, 0.9, 0.8, 0.3, 0.0], 4, 0.757731469770386, None),
    ("perfect", [0, 0, 0, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 9, 9, 9, 0, 0, 0, 0], 2, 1.0, 1.0),
    ("inverted", [0, 0, 0, 1, 1, 1, 0, 0, 0, 0], [9, 9, 9, 0, 0, 0, 9, 9, 9, 9], 2,
     0.31669894220984585, 0.31384458990836517),
]


@pytest.mark.parametrize("name,y,s,w,vus,rauc", TSB_AD_REFERENCE, ids=[r[0] for r in TSB_AD_REFERENCE])
def test_vus_pr_matches_upstream_tsb_ad_fixtures(name, y, s, w, vus, rauc):
    """VUS-PR must equal TSB-AD's, to the last bit.

    The protocol pins VUS-PR as the primary metric and the ladder's sanity anchors (0.354
    multivariate / 0.440 univariate) are TSB-AD leaderboard numbers, so a definition that is
    merely *inspired by* the paper makes every comparison meaningless. An earlier revision
    re-implemented it from the published formulas and returned 0.41667 on the first case where
    upstream returns 0.58333; ``nebulax/bench/vus_tsb_ad.py`` is now a port of the upstream
    code itself, and this test is what keeps it honest.
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    assert M.vus_pr(y, s, max_window=w) == pytest.approx(vus, abs=1e-12)
    got = M.r_auc_pr(y, s, window=w)
    if rauc is None:
        assert np.isnan(got)
    else:
        assert got == pytest.approx(rauc, abs=1e-12)


def test_vus_pr_is_the_mean_over_buffer_sizes_and_records_its_provenance():
    from nebulax.bench import vus_tsb_ad as V

    assert V.TSB_AD_LICENCE == "Apache-2.0"
    assert V.TSB_AD_COMMIT in V.TSB_AD_SOURCE
    y = np.r_[np.zeros(20, int), np.ones(5, int), np.zeros(20, int)]
    s = np.r_[np.zeros(20), np.full(5, 3.0), np.zeros(20)] + 0.01 * np.arange(45)
    assert M.vus_pr(y, s, max_window=0) == pytest.approx(M.r_auc_pr(y, s, window=0), abs=1e-12)
    assert 0.0 <= M.vus_pr(y, s, max_window=8) <= 1.0


def test_vus_pr_is_nan_on_a_single_class_slice():
    s = np.linspace(0, 1, 20)
    assert np.isnan(M.vus_pr(np.zeros(20, int), s))
    assert np.isnan(M.vus_pr(np.ones(20, int), s))
    assert np.isnan(M.r_auc_pr(np.zeros(20, int), s))


def test_r_auc_pr_perfect_and_degenerate():
    y = np.array([0, 0, 1, 1, 0, 0])
    assert M.r_auc_pr(y, np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.0]), window=0) == pytest.approx(1.0)
    assert np.isnan(M.r_auc_pr(np.zeros(6), np.arange(6.0), window=0))  # one class -> undefined
    assert np.isnan(M.vus_pr(np.ones(6), np.arange(6.0)))


def test_vus_buffer_rewards_a_near_miss():
    """A detector that fires one window early scores better as the buffer widens - the whole
    point of the volume-under-the-surface construction."""
    y = np.zeros(16, dtype=int)
    y[8] = 1
    s = np.zeros(16)
    s[7] = 1.0  # fires one window early
    assert M.r_auc_pr(y, s, window=0) < M.r_auc_pr(y, s, window=3)


def test_auroc_auprc_basic():
    y = np.array([0, 0, 1, 1])
    assert M.auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert M.auprc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert np.isnan(M.auroc(np.zeros(4), np.arange(4.0)))


# --------------------------------------------------------------------------------------
# classification / ordinal
# --------------------------------------------------------------------------------------


def test_cls_metrics_shapes_and_values():
    y = np.array(["a", "a", "b", "c"])
    out = M.cls_metrics(y, np.array(["a", "b", "b", "c"]))
    assert out["labels"] == ["a", "b", "c"]
    assert np.array(out["confusion_matrix"]).shape == (3, 3)
    assert 0.0 < out["macro_f1"] < 1.0
    assert out["accuracy"] == pytest.approx(0.75)
    # a class never predicted still gets a row/column
    out2 = M.cls_metrics(y, np.array(["a", "a", "a", "a"]), labels=["a", "b", "c", "d"])
    assert np.array(out2["confusion_matrix"]).shape == (4, 4)


def test_monotonicity():
    stage = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=float)
    assert M.monotonicity(stage * 2.0 + 0.1, stage) == pytest.approx(1.0)
    assert M.monotonicity(-stage, stage) == pytest.approx(-1.0)
    assert np.isnan(M.monotonicity(np.ones(8), stage))


# --------------------------------------------------------------------------------------
# CPD
# --------------------------------------------------------------------------------------


def test_segments_and_covering():
    assert M.segments_from_cps([5], 10) == [set(range(5)), set(range(5, 10))]
    assert M.covering([5], [5], 10) == pytest.approx(1.0)
    # one predicted segment: each true half has Jaccard 5/10 with the whole series
    assert M.covering([5], [], 10) == pytest.approx(0.5)


def test_f1_at_margin_matching_is_greedy_and_no_double_count():
    assert M.f1_at_margin([10, 50], [10, 50], margin=5)["f1"] == pytest.approx(1.0)
    # two predictions inside the margin of one true cp: only one can match
    out = M.f1_at_margin([10], [9, 11], margin=5)
    assert out["recall"] == pytest.approx(1.0) and out["precision"] == pytest.approx(0.5)
    assert M.f1_at_margin([10], [], margin=5)["f1"] == 0.0
    # outside the margin -> nothing matches
    assert M.f1_at_margin([10], [30], margin=5)["f1"] == 0.0


def test_f1_over_margins_and_cpd_metrics_bundle():
    over = M.f1_over_margins([10, 50], [12, 48], margins=[1, 5, 10])
    assert over["f1_curve"][0] == 0.0 and over["f1_curve"][-1] == pytest.approx(1.0)
    bundle = M.cpd_metrics([10, 50], [12, 48], n=100)
    assert set(["covering", "cpd_f1", "n_pred_cps", "n_true_cps"]).issubset(bundle)


def test_cpd_score_adapter_all_three_modes():
    s = M.cpd_score(20, breakpoints=[10], tau_windows=2.0)
    assert s.shape == (20,) and s[10] == pytest.approx(1.0) and s[0] < s[8] < s[10]
    assert np.all(M.cpd_score(10, breakpoints=[]) == 0.0)

    prof = M.cpd_score(5, profile=np.array([1.0, 3.0, 5.0, 3.0, 1.0]))
    assert prof.min() == 0.0 and prof.max() == 1.0 and np.argmax(prof) == 2

    flags = np.array([False, False, True, False, True])
    boolean = M.cpd_score(5, statistic=flags)
    assert np.all(np.diff(boolean) >= 0) and boolean[-1] == pytest.approx(1.0)

    with pytest.raises(ValueError):
        M.cpd_score(5)


# --------------------------------------------------------------------------------------
# "consecutive" means consecutive IN TIME, not merely adjacent in the array
# --------------------------------------------------------------------------------------


def test_max_step_seconds_scales_with_the_window():
    assert M.max_step_seconds(600.0) == pytest.approx(900.0)  # 1.5 x
    assert M.max_step_seconds(10.0) == pytest.approx(70.0)  # floored at window + 60 s
    assert M.max_step_seconds(0.0) == float("inf")  # unknown step -> no contiguity check
    assert M.max_step_seconds(None) == float("inf")


def test_three_isolated_windows_far_apart_are_not_an_episode():
    """The attack this rule exists for: three above-threshold rows that are array-adjacent
    because everything between them was dropped, but two months apart in time. Without the
    step budget they collapse into one Episode(n_windows=3) spanning Jan-Mar, which overlaps
    essentially any event window and reports recall 1.0 with zero false alarms."""
    t = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]).to_numpy().astype("datetime64[ms]")
    scores = np.full(3, 9.0)
    naive = M.episodes(scores, t, threshold=1.0, k=3, merge_gap_s=HOUR)
    assert len(naive) == 1 and naive[0].n_windows == 3  # the old, wrong behaviour

    guarded = M.episodes(scores, t, threshold=1.0, k=3, merge_gap_s=HOUR, max_step_s=M.max_step_seconds(600.0))
    assert guarded == [], "three points two months apart are not three consecutive windows"


def test_a_run_is_cut_at_a_time_gap_before_the_k_rule_and_before_merging():
    # 8 rows, 10 min apart, except a 24 h hole between index 3 and 4 (a repair blanking).
    t = np.r_[_times(4), _times(4, start="2020-04-02T00:00:00")]
    scores = np.full(8, 9.0)
    step = M.max_step_seconds(600.0)

    bridged = M.episodes(scores, t, threshold=1.0, k=3, merge_gap_s=HOUR)
    assert len(bridged) == 1 and bridged[0].n_windows == 8

    cut = M.episodes(scores, t, threshold=1.0, k=3, merge_gap_s=HOUR, max_step_s=step)
    assert len(cut) == 2 and [e.n_windows for e in cut] == [4, 4]

    # ... and the surviving pieces must each clear k on their own: 2 + 2 is nothing.
    scores2 = np.zeros(8)
    scores2[[2, 3, 4, 5]] = 9.0  # 2 rows either side of the hole
    assert M.episodes(scores2, t, threshold=1.0, k=3, merge_gap_s=HOUR, max_step_s=step) == []


def test_a_gap_smaller_than_the_merge_window_still_merges_back():
    """Cutting on the step budget must not break the protocol's 1 h merge rule: two runs that
    each clear k and sit 30 min apart are still one episode."""
    t = np.r_[_times(3), _times(3, start="2020-04-01T00:30:00")]
    scores = np.full(6, 9.0)
    eps = M.episodes(scores, t, threshold=1.0, k=3, merge_gap_s=HOUR, max_step_s=M.max_step_seconds(600.0))
    assert len(eps) == 1 and eps[0].n_windows == 6


def test_episodes_rejects_a_nonpositive_max_step():
    with pytest.raises(ValueError, match="max_step_s"):
        M.episodes(np.ones(5), _times(5), threshold=0.0, max_step_s=0.0)


# --------------------------------------------------------------------------------------
# row pitch: the contiguity budget is counted in row spacing, not window duration
# --------------------------------------------------------------------------------------


def test_row_pitch_is_measured_within_each_unit():
    """Ten units interleaved on one timeline must not report a tenth of the true pitch."""
    t_one = _times(50, step_s=155.0)
    t = np.concatenate([t_one + np.timedelta64(int(k * 15_500), "ms") for k in range(10)])
    u = np.repeat([f"T{k:02d}" for k in range(10)], 50).astype(object)
    assert M.row_pitch_seconds(t, u) == pytest.approx(155.0)
    assert M.row_pitch_seconds(t) == pytest.approx(15.5)  # the interleaved, wrong answer
    assert M.row_pitch_seconds(t[:1], u[:1], fallback=36.1) == pytest.approx(36.1)


def test_max_step_uses_the_larger_of_window_and_row_pitch():
    # synthetic door cycle table: 36.1 s cycles every ~155 s
    assert M.max_step_seconds(36.1) == pytest.approx(96.1)  # window alone: never 3-consecutive
    assert M.max_step_seconds(36.1, 154.9) == pytest.approx(154.9 * 1.5)
    # overlapping MetroPT windows: 600 s windows every 300 s -> the window still wins
    assert M.max_step_seconds(600.0, 300.0) == pytest.approx(900.0)
    assert M.max_step_seconds(None, 155.0) == pytest.approx(232.5)
    assert M.max_step_seconds(None, None) == float("inf")
    assert M.max_step_seconds(0.0, float("nan")) == float("inf")


def test_three_consecutive_cycles_of_a_sparse_table_do_form_an_episode():
    """The reviewer's case: a per-cycle table whose rows are sparser than the window. With the
    budget derived from the window alone the >= 3-consecutive rule could never fire."""
    t = _times(12, step_s=155.0)
    scores = np.zeros(12)
    scores[4:8] = 9.0
    u = np.full(12, "T01", dtype=object)
    step = M.max_step_seconds(36.1, M.row_pitch_seconds(t, u))
    eps = M.episodes(scores, t, threshold=1.0, k=3, merge_gap_s=HOUR, max_step_s=step)
    assert len(eps) == 1 and eps[0].n_windows == 4
    assert M.episodes(scores, t, threshold=1.0, k=3, merge_gap_s=HOUR, max_step_s=M.max_step_seconds(36.1)) == []


def test_nat_becomes_nan_not_a_finite_epoch():
    t = np.array(["2020-04-01", "NaT"], dtype="datetime64[ms]")
    out = M.to_epoch_seconds(t)
    assert np.isfinite(out[0]) and np.isnan(out[1])
    assert np.isnan(M._scalar_seconds(np.datetime64("NaT")))
    ev = M.Event.from_any("u", np.datetime64("2020-04-01"), pd.NaT, "x")
    assert np.isnan(ev.t_failure), "an open-ended event must stay open-ended"
    # train-days over two real days plus a NaT row is 1 day, not 125,015
    tt = np.array(["2020-04-01", "2020-04-02", "NaT"], dtype="datetime64[ms]")
    assert M.train_days(tt, np.array(["a", "a", "a"], dtype=object)) == pytest.approx(1.0)


def test_open_ended_event_is_detectable_by_a_later_episode():
    t = _times(6, step_s=600.0)
    ev = M.Event.from_any("u", t[1], pd.NaT, "x")
    s = np.array([0.0, 0.0, 9.0, 9.0, 9.0, 0.0])
    u = np.full(6, "u", dtype=object)
    eps = M.episodes(s, t, 1.0, k=3, merge_gap_s=3600.0, units=u, max_step_s=900.0)
    em = M.event_metrics(eps, [ev], H=0.0, n_train_days=1.0)
    assert em["n_events"] == 1 and em["event_recall"] == 1.0 and em["n_false_alarms"] == 0
