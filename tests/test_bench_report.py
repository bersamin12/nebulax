"""Report: the leaderboard markdown and the plotly ablation heatmap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nebulax.bench import report as RP


def _runs() -> pd.DataFrame:
    rows = [
        dict(
            dataset="metropt3", subsystem="pneumatic", task="ad", model="iforest", split="metropt_temporal",
            input_kind="window_stats", window=60, feature_set="all", peer_norm=False, contamination=0.0,
            train_regime="normal_only", status="ok", git_rev="abc1234", vus_pr=0.41, auprc=0.5, auroc=0.8,
            budget_event_recall=1.0, budget_n_events=3, budget_n_events_detected=3,
            budget_false_alarms_per_train_day=0.05, budget_median_lead_time_s=7200.0,
            threshold_budget_val_far_per_train_day=0.06,
            val_vus_pr=0.38, val_auprc=0.44, val_auroc=0.79,
            q995_event_recall=0.66, q999_event_recall=0.33, fit_seconds=1.2,
            score_seconds_per_10k=0.4, peak_rss_mb=900.0, peak_vram_mb=0.0, config_hash="h1",
        ),
        dict(
            # Same window and split as iforest: `window` and `split` are bold-group keys (the
            # positive rate and the scored-row count differ across them), so a lof at window 360
            # would be its own population and would be bolded too, correctly. The feature_set
            # differs instead, to keep one varying ablation axis for the heatmap tests.
            dataset="metropt3", subsystem="pneumatic", task="ad", model="lof", split="metropt_temporal",
            input_kind="window_stats", window=60, feature_set="stats", peer_norm=False, contamination=0.0,
            train_regime="normal_only", status="ok", git_rev="abc1234", vus_pr=0.22, auprc=0.3, auroc=0.6,
            budget_event_recall=0.3333, budget_n_events=3, budget_n_events_detected=1,
            budget_false_alarms_per_train_day=0.9, budget_median_lead_time_s=600.0,
            threshold_budget_val_far_per_train_day=0.9,
            val_vus_pr=0.19, val_auprc=0.21, val_auroc=0.58,
            q995_event_recall=0.33, q999_event_recall=0.0, fit_seconds=8.0,
            score_seconds_per_10k=9.0, peak_rss_mb=1500.0, peak_vram_mb=0.0, config_hash="h2",
        ),
        dict(
            dataset="cranfield", subsystem="door", task="cls", model="lgbm", split="cranfield_loo_load",
            input_kind="cycle_features", window=None, feature_set="test", peer_norm=False, contamination=0.0,
            train_regime="all", status="ok", git_rev="abc1234", macro_f1=0.60, balanced_accuracy=0.62,
            fit_seconds=2.0, score_seconds_per_10k=0.2, peak_rss_mb=400.0, peak_vram_mb=0.0,
            params="{}", seed=0, config_hash="h3",
        ),
        dict(
            dataset="cranfield", subsystem="door", task="cls", model="lgbm", split="cranfield_random_rep",
            input_kind="cycle_features", window=None, feature_set="test", peer_norm=False, contamination=0.0,
            train_regime="all", status="ok", git_rev="abc1234", macro_f1=0.90, balanced_accuracy=0.91,
            fit_seconds=2.0, score_seconds_per_10k=0.2, peak_rss_mb=400.0, peak_vram_mb=0.0,
            params="{}", seed=0, config_hash="h4",
        ),
        dict(
            dataset="ottawa", subsystem="bearing", task="ad", model="boom", split="ottawa_sgkf5",
            input_kind="window_stats", window=1.0, feature_set="time", peer_norm=False, contamination=0.0,
            train_regime="normal_only", status="failed", git_rev="abc1234",
            traceback="Traceback...\nRuntimeError: exploded", config_hash="h5",
        ),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def results_dir(tmp_path) -> Path:
    _runs().to_parquet(tmp_path / "runs.parquet", engine="pyarrow", index=False)
    return tmp_path


def test_load_results_prefers_the_combined_parquet_and_falls_back(tmp_path):
    assert RP.load_results(tmp_path).empty
    runs = tmp_path / "runs"
    runs.mkdir()
    _runs().iloc[:1].to_parquet(runs / "h1.parquet", engine="pyarrow", index=False)
    assert len(RP.load_results(tmp_path)) == 1
    _runs().to_parquet(tmp_path / "runs.parquet", engine="pyarrow", index=False)
    assert len(RP.load_results(tmp_path)) == 5


def test_leaderboard_has_one_table_per_task_and_bolds_the_best(results_dir):
    path = RP.leaderboard(results_dir)
    md = path.read_text()
    assert path.name == "leaderboard.md"
    assert "## Task: `ad`" in md and "## Task: `cls`" in md
    assert "## Selected per subsystem" in md
    # best metropt row is iforest (higher VUS-PR) -> bolded
    assert "| **iforest** |" in md and "**0.410**" in md
    assert "| **lof** |" not in md


def test_every_metropt_recall_carries_its_event_count(results_dir):
    md = RP.leaderboard(results_dir).read_text()
    assert "1.000 (3/3)" in md, "recall must be printed with events detected / events present"
    assert "0.333 (1/3)" in md
    assert "N = [3] event(s)" in md


def test_failures_are_shown_not_dropped(results_dir):
    md = RP.leaderboard(results_dir).read_text()
    assert "## Runs that did not finish" in md
    assert "RuntimeError: exploded" in md
    assert "1 failed" in md


def test_generalisation_gap_pairs_random_against_held_out_group(results_dir):
    df = RP._add_generalisation_gap(RP.load_results(results_dir))
    loo = df[df["split"] == "cranfield_loo_load"].iloc[0]
    assert loo["generalisation_gap"] == pytest.approx(0.90 - 0.60)
    rnd = df[df["split"] == "cranfield_random_rep"].iloc[0]
    assert np.isnan(rnd["generalisation_gap"])  # the control has no control


def test_selected_per_subsystem_prefers_a_model_inside_the_budget(results_dir):
    md = RP.leaderboard(results_dir).read_text()
    block = md.split("## Selected per subsystem")[1]
    assert "**iforest**" in block  # lof is over the 1-per-7-train-days budget ON VALIDATION
    assert "**lof**" not in block


def test_the_budget_filter_reads_a_validation_column(tmp_path):
    """Filtering candidates on the test-slice false-alarm rate is selection by test
    performance just as much as ranking on it is."""
    assert RP.BUDGET_FILTER_METRIC.startswith("threshold_budget_val_")
    df = _runs().copy()
    # lof looks fine on the test slice but blew its budget on validation: it must stay out,
    # and iforest (worse on test, inside budget on validation) must be the pick.
    df["val_vus_pr"] = [0.10, 0.99, np.nan, np.nan, np.nan]
    df.loc[df["model"] == "lof", "budget_false_alarms_per_train_day"] = 0.01
    df.loc[df["model"] == "lof", RP.BUDGET_FILTER_METRIC] = 5.0
    df.to_parquet(tmp_path / "runs.parquet", engine="pyarrow", index=False)
    block = RP.leaderboard(tmp_path).read_text().split("## Selected per subsystem")[1]
    assert "**iforest**" in block and "**lof**" not in block


def test_leaderboard_on_empty_results(tmp_path):
    md = RP.leaderboard(tmp_path).read_text()
    assert "No runs found" in md


def test_ablation_heatmap_writes_plotly_html(results_dir):
    path = RP.ablation_heatmap(results_dir)
    html = path.read_text()
    assert path.name == "ablation_heatmap.html"
    assert "plotly" in html.lower()
    assert "Heatmap" in html or "heatmap" in html


def test_ablation_heatmap_without_varying_axes(tmp_path):
    one = _runs().iloc[[0]]
    one.to_parquet(tmp_path / "runs.parquet", engine="pyarrow", index=False)
    html = RP.ablation_heatmap(tmp_path).read_text()
    assert "Nothing to ablate" in html


def test_ablation_heatmap_on_empty_results(tmp_path):
    assert "No successful runs" in RP.ablation_heatmap(tmp_path).read_text()


# --------------------------------------------------------------------------------------
# what may be selected as the shipped model
# --------------------------------------------------------------------------------------


def _uncalibrated_row(**kw) -> dict:
    """An AD run that finished but never got an operating point: high VUS-PR, no threshold."""
    base = dict(
        dataset="metropt3", subsystem="pneumatic", task="ad", model="never_calibrated",
        split="metropt_temporal", input_kind="window_stats", window=60, feature_set="all",
        peer_norm=False, contamination=0.0, train_regime="normal_only", status="ok",
        git_rev="abc1234", vus_pr=0.95, auprc=0.9, auroc=0.99,
        budget_event_recall=float("nan"), budget_n_events=float("nan"),
        budget_n_events_detected=float("nan"),
        budget_false_alarms_per_train_day=float("nan"),
        threshold_budget_val_far_per_train_day=float("nan"), fit_seconds=0.5,
        uncalibrated=True, uncalibrated_reason="empty validation slice", config_hash="h9",
    )
    base.update(kw)
    return base


def test_a_run_with_no_operating_point_cannot_be_the_selected_model(tmp_path):
    df = pd.concat([_runs(), pd.DataFrame([_uncalibrated_row()])], ignore_index=True)
    df.to_parquet(tmp_path / "runs.parquet", engine="pyarrow", index=False)
    block = RP.leaderboard(tmp_path).read_text().split("## Selected per subsystem")[1]
    assert "never_calibrated" not in block, "a run with NaN FA/train-day is not shippable"
    assert "**iforest**" in block


def test_an_uncalibrated_run_without_the_flag_column_is_still_excluded(tmp_path):
    """Belt and braces: the missing false-alarm rate alone must disqualify the row, even on an
    older results table that predates the ``uncalibrated`` column."""
    row = _uncalibrated_row()
    row.pop("uncalibrated")
    row.pop("uncalibrated_reason")
    df = pd.concat([_runs(), pd.DataFrame([row])], ignore_index=True)
    df.to_parquet(tmp_path / "runs.parquet", engine="pyarrow", index=False)
    block = RP.leaderboard(tmp_path).read_text().split("## Selected per subsystem")[1]
    assert "never_calibrated" not in block


def test_selection_uses_the_validation_metric_not_the_test_one(tmp_path):
    """iforest wins on the test slice (VUS-PR 0.41 vs 0.22), lof wins on validation. The table
    must follow validation: maximising the test column over a whole ablation ladder is
    best-of-N on the test slice, which configs/model_ladder.yaml forbids."""
    df = _runs().copy()
    df["val_vus_pr"] = [0.10, 0.80, np.nan, np.nan, np.nan]
    df.loc[df["model"] == "lof", RP.BUDGET_FILTER_METRIC] = 0.05  # bring lof inside budget
    df.to_parquet(tmp_path / "runs.parquet", engine="pyarrow", index=False)
    block = RP.leaderboard(tmp_path).read_text().split("## Selected per subsystem")[1]
    assert "**lof**" in block and "**iforest**" not in block
    assert "val_vus_pr" in block


def test_selection_defers_rather_than_falling_back_to_a_test_metric(tmp_path):
    """MetroPT's validation slice holds no positive rows, so val_vus_pr / val_auprc are
    necessarily NaN there. The table must then select NOTHING and say so - a "test_vus_pr"
    fallback would be the forbidden best-of-N wearing a label."""
    df = _runs().copy()
    for col in RP.SELECTION_METRICS:
        df[col] = np.nan
    df.to_parquet(tmp_path / "runs.parquet", engine="pyarrow", index=False)
    block = RP.leaderboard(tmp_path).read_text().split("## Selected per subsystem")[1]
    assert "deferred" in block and "no validation-slice metric" in block
    assert "test_vus_pr" not in block
    assert "**iforest**" not in block and "**lof**" not in block
    # every eligible candidate is listed, ranked on the test metric and labelled as such
    assert "ranked on TEST VUS-PR for information only" in block
    assert "`iforest`" in block  # named in full, not truncated to an alphabetical head
    assert "1 eligible candidate" in block  # lof is out on the validation budget, not on a cut


def test_a_silent_threshold_cannot_be_selected(tmp_path):
    """A threshold that never fired once on validation has a perfect false-alarm rate for the
    worst possible reason."""
    df = _runs().copy()
    df["val_vus_pr"] = [0.10, 0.99, np.nan, np.nan, np.nan]
    df["threshold_budget_silent"] = [False, True, False, False, False]
    df.loc[df["model"] == "lof", RP.BUDGET_FILTER_METRIC] = 0.0
    df.to_parquet(tmp_path / "runs.parquet", engine="pyarrow", index=False)
    block = RP.leaderboard(tmp_path).read_text().split("## Selected per subsystem")[1]
    assert "**lof**" not in block
    assert "**iforest**" in block


def test_selection_metrics_are_all_validation_side():
    assert all(m.startswith("val_") for m in RP.SELECTION_METRICS)
    assert "vus_pr" not in RP.SELECTION_METRICS


def test_no_ad_run_has_an_operating_point(tmp_path):
    df = pd.DataFrame([_uncalibrated_row()])
    df.to_parquet(tmp_path / "runs.parquet", engine="pyarrow", index=False)
    block = RP.leaderboard(tmp_path).read_text().split("## Selected per subsystem")[1]
    assert "no AD run produced an eligible operating point" in block


def test_generalisation_gap_pairs_only_identical_configurations(tmp_path):
    """The gap is F1(random) - F1(held-out group) for otherwise IDENTICAL runs. If the join key
    omits an ablation axis, a control from one configuration gets subtracted from a treatment
    in another and the number measures the ablation instead of the grouping."""
    base = dict(
        dataset="cranfield", subsystem="door", task="cls", model="lgbm", input_kind="cycle_features",
        params="{}", feature_set="test", train_regime="all", H=259200.0, peer_norm=False,
        contamination=0.0, seed=0, data_kwargs="{}", status="ok", git_rev="a",
    )
    rows = [
        {**base, "split": "cranfield_loo_load", "window": 60, "macro_f1": 0.60, "config_hash": "a1"},
        {**base, "split": "cranfield_random_rep", "window": 60, "macro_f1": 0.90, "config_hash": "a2"},
        # a control at a DIFFERENT window: must not be paired with the window=60 treatment
        {**base, "split": "cranfield_loo_load", "window": 360, "macro_f1": 0.50, "config_hash": "a3"},
    ]
    df = pd.DataFrame(rows)
    out = RP._add_generalisation_gap(df)
    assert out.loc[0, "generalisation_gap"] == pytest.approx(0.30)
    assert np.isnan(out.loc[2, "generalisation_gap"]), "no control exists at window=360"
    assert np.isnan(out.loc[1, "generalisation_gap"])  # the control has no control


def test_generalisation_gap_key_pins_every_ablation_axis():
    """The two arms of the gap must be identical in everything but the split, so the join key
    has to be the whole of RunSpec's identity minus `split`. Deriving it from RunSpec here
    means a new field cannot be added to the spec and silently left out of the join."""
    from nebulax.bench.runner import ABLATION_KEYS, RunSpec

    identity = set(RunSpec(dataset="metropt3", model="m", split="metropt_temporal").identity)
    missing = (identity - {"split"}) - set(RP.GEN_GAP_KEY)
    assert not missing, f"the gen-gap join ignores RunSpec field(s) {sorted(missing)}"
    missing_axes = {k for k in ABLATION_KEYS if k != "split"} - set(RP.GEN_GAP_KEY)
    assert not missing_axes, f"the gen-gap join ignores ablation axes {sorted(missing_axes)}"
    assert "split" not in RP.GEN_GAP_KEY  # the split is the thing that differs
    assert "max_minutes" in RP.GEN_GAP_KEY  # a budget is an axis, not a scheduling detail


def test_the_leaderboard_never_bolds_a_run_without_an_operating_point(tmp_path):
    """The headline row a reader acts on must be a run that actually has a threshold.

    Selection already excluded uncalibrated / silent runs, but `_table` bolded the best TEST
    primary metric with no filter, so a run with `vus_pr=1.0`, `uncalibrated=True`, no
    threshold, no event recall and no false-alarm rate printed as the bolded best AD model.
    """
    base = dict(
        task="ad", status="ok", dataset="metropt3", subsystem="pneumatic", split="metropt_temporal",
        input_kind="window_stats", window=60, feature_set="default", train_regime="normal_only",
        H=259200.0, peer_norm=False, contamination=0.0, seed=0, params="{}", data_kwargs="{}",
        max_minutes=20.0, git_rev="a", fit_seconds=1.0,
    )
    df = pd.DataFrame(
        [
            {**base, "model": "ghost", "vus_pr": 1.00, "val_vus_pr": float("nan"),
             "uncalibrated": True, "threshold_budget_silent": False,
             "threshold_budget_val_far_per_train_day": float("nan")},
            {**base, "model": "mute", "vus_pr": 0.90, "val_vus_pr": 0.50,
             "uncalibrated": False, "threshold_budget_silent": True,
             "threshold_budget_val_far_per_train_day": 0.0},
            {**base, "model": "honest", "vus_pr": 0.20, "val_vus_pr": 0.80,
             "uncalibrated": False, "threshold_budget_silent": False,
             "threshold_budget_val_far_per_train_day": 0.1,
             "budget_event_recall": 0.5, "budget_n_events": 4, "budget_n_events_detected": 2},
        ]
    )
    out = tmp_path / "runs"
    out.mkdir()
    df.to_parquet(out.parent / "runs.parquet", engine="pyarrow")
    md = RP.leaderboard(out.parent).read_text()
    assert "**honest**" in md
    assert "**ghost**" not in md and "**mute**" not in md
    assert "**1.000**" not in md and "**0.900**" not in md
    # and the reason is visible in the table, not only in the code
    assert "op. point" in md and "uncalibrated" in md and "silent" in md


def test_every_operational_axis_is_printed_in_the_table(tmp_path):
    """Two runs that differ only in H (or peer_norm, contamination, train_regime) have different
    false-alarm and recall numbers; the table must show which is which."""
    a = _runs().iloc[0].to_dict()
    b = dict(a, H=7 * 86400.0, budget_false_alarms_per_train_day=0.15, config_hash="h1b")
    a["H"] = 48 * 3600.0
    df = pd.DataFrame([a, b])
    df.to_parquet(tmp_path / "runs.parquet", index=False)
    md = RP.leaderboard(tmp_path).read_text()
    header = next(line for line in md.splitlines() if line.startswith("| dataset"))
    for col in ("train_regime", "H", "peer_norm", "contamination"):
        assert f" {col} " in header, col
    rows = [line for line in md.splitlines() if line.startswith("| metropt3") and "iforest" in line]
    assert len(rows) == 2 and rows[0] != rows[1]
    assert any("172800" in r for r in rows) and any("604800" in r for r in rows)


def test_fabricated_timeline_rows_stay_eligible_and_impossible_rows_say_why(tmp_path):
    a = _runs().iloc[0].to_dict()
    ott = dict(a, dataset="ottawa", subsystem="bearing", model="zc", far_not_applicable_reason="fabricated timeline",
               threshold_budget_val_far_per_train_day=np.nan, budget_false_alarms_per_train_day=np.nan, config_hash="ho")
    cra = dict(a, dataset="cranfield", subsystem="door", model="zc", episodes_impossible=True, budget_event_recall=np.nan,
               budget_false_alarms_per_train_day=np.nan, threshold_budget_silent=False, config_hash="hc")
    df = pd.DataFrame([a, ott, cra])
    df.to_parquet(tmp_path / "runs.parquet", index=False)
    md = RP.leaderboard(tmp_path).read_text()
    assert "budget n/a: fabricated timeline" in md
    assert "episodes impossible" in md
    sel = md.split("## Selected per subsystem")[1]
    assert "bearing" in sel and "| zc |" in sel or "**zc**" in sel


def test_selection_groups_on_the_loader_subsystem_and_prefers_measured_false_alarm_evidence(tmp_path):
    a = _runs().iloc[0].to_dict()
    metro = dict(a, subsystem=None, dataset_subsystem="pneumatic", model="m_metro", val_vus_pr=0.60,
                 threshold_budget_val_far_per_train_day=0.05, config_hash="hm")
    cran = dict(a, subsystem=None, dataset="cranfield", dataset_subsystem="door", model="m_cran", val_vus_pr=0.90,
                far_not_applicable_reason="fabricated timeline", threshold_budget_val_far_per_train_day=np.nan,
                budget_false_alarms_per_train_day=np.nan, config_hash="hc")
    cran2 = dict(cran, dataset_subsystem="pneumatic", model="m_cran_pneu", config_hash="hc2")
    df = pd.DataFrame([metro, cran, cran2])
    df.to_parquet(tmp_path / "runs.parquet", index=False)
    sel = RP.leaderboard(tmp_path).read_text().split("## Selected per subsystem")[1]
    pneu = [l for l in sel.splitlines() if l.startswith("| pneumatic")]
    door = [l for l in sel.splitlines() if l.startswith("| door")]
    # one pick per (subsystem, dataset): the MetroPT pick has measured evidence, the Cranfield
    # picks are marked as having none; a fabricated row never displaces a measured one
    metro_line = [l for l in pneu if "| metropt3 |" in l]
    assert len(metro_line) == 1 and "**m_metro**" in metro_line[0] and "no false-alarm evidence" not in metro_line[0]
    assert all("no false-alarm evidence" in l for l in pneu + door if "| cranfield |" in l)
    assert len(door) == 1 and "**m_cran**" in door[0]


def test_a_silent_fabricated_timeline_row_stays_eligible_and_says_so(tmp_path):
    """Ottawa's real output: the q0.995 stand-in raises no episode on 10-row validation records,
    so threshold_budget_silent is True - that must not disqualify the row."""
    a = _runs().iloc[0].to_dict()
    ott = dict(a, dataset="ottawa", subsystem=None, dataset_subsystem="bearing", model="normz",
               far_not_applicable_reason="fabricated timeline", threshold_budget_silent=True,
               threshold_budget_val_far_per_train_day=np.nan, budget_false_alarms_per_train_day=np.nan,
               budget_event_recall=0.275, budget_n_events=40, budget_n_events_detected=11, val_vus_pr=0.98, config_hash="ho")
    df = pd.DataFrame([ott])
    assert RP._eligible(df).iloc[0]
    assert "silent on validation" in RP._operating_point(df).iloc[0]
    df.to_parquet(tmp_path / "runs.parquet", index=False)
    sel = RP.leaderboard(tmp_path).read_text().split("## Selected per subsystem")[1]
    assert "**normz** (no false-alarm evidence)" in sel
    # a real-timeline silent row is still excluded
    real = dict(a, threshold_budget_silent=True, config_hash="hs")
    assert not RP._eligible(pd.DataFrame([real])).iloc[0]


def test_base_rate_column_and_warning_and_per_dataset_selection(tmp_path):
    a = _runs().iloc[0].to_dict()
    metro = dict(a, dataset_subsystem="pneumatic", test_positive_rate=0.03, val_positive_rate=0.0, config_hash="h1")
    sim_noev = dict(a, dataset="sim", dataset_subsystem="bearing", model="sim_noevents", val_vus_pr=0.02,
                    budget_n_events=0, budget_n_false_alarms=3, budget_event_recall=np.nan, test_positive_rate=0.2, config_hash="h2")
    ott = dict(a, dataset="ottawa", dataset_subsystem="bearing", model="ott_det", val_vus_pr=0.95, budget_n_events=40,
               budget_n_events_detected=36, budget_event_recall=0.9, far_not_applicable_reason="fabricated timeline",
               threshold_budget_val_far_per_train_day=np.nan, budget_false_alarms_per_train_day=np.nan,
               test_positive_rate=0.667, config_hash="h3")
    df = pd.DataFrame([metro, sim_noev, ott])
    df.to_parquet(tmp_path / "runs.parquet", index=False)
    md = RP.leaderboard(tmp_path).read_text()
    assert " test +rate " in md.split("## Selected")[0]
    assert "Base-rate warning" in md and "ottawa" in md.split("Base-rate warning")[1].split("\n")[0]
    sel = md.split("## Selected per subsystem")[1]
    bearing = [l for l in sel.splitlines() if l.startswith("| bearing")]
    assert len(bearing) == 2, bearing  # one pick per (subsystem, dataset)
    assert any("**ott_det**" in l for l in bearing)
    assert any("**sim_noevents** (no false-alarm evidence)" in l for l in bearing)


def test_bolding_is_within_the_scored_population_and_the_class_target_is_printed(tmp_path):
    a = _runs().iloc[0].to_dict()
    rand = dict(a, dataset="sim", dataset_subsystem="door", input_kind="window_stats", model="random", vus_pr=0.78, auroc=0.5, val_vus_pr=0.7, test_positive_rate=0.75, config_hash="r")
    good = dict(a, dataset="sim", dataset_subsystem="bearing", input_kind="cycle_features", model="good", vus_pr=0.31, auroc=0.98, val_vus_pr=0.3, test_positive_rate=0.016, config_hash="g")
    c4 = dict(_runs().iloc[2].to_dict(), model="lgbm4", macro_f1=0.95, data_kwargs="{}", config_hash="c4")
    c13 = dict(_runs().iloc[2].to_dict(), model="lgbm13", macro_f1=0.55, data_kwargs='{"cls_target": "condition"}', config_hash="c13")
    df = pd.DataFrame([rand, good, c4, c13])
    df.to_parquet(tmp_path / "runs.parquet", index=False)
    md = RP.leaderboard(tmp_path).read_text()
    assert "| **random** |" in md and "| **good** |" in md, "one bold per scored population, not per dataset"
    assert "| **lgbm4** |" in md and "| **lgbm13** |" in md
    assert "cls_target" in md


# --------------------------------------------------------------------------------------
# the page must say what it is doing: populations, labels, lift, floors
# --------------------------------------------------------------------------------------


def test_bold_groups_include_the_split_and_the_window(tmp_path):
    """A bold group is a scored population. Two MetroPT regimes hold 4 and 2 events over very
    different row counts, and two windows on the synthetic door fleet differ 0.54 vs 0.75 in
    positive rate, so neither may be bolded against the other."""
    assert "split" in RP.BOLD_GROUP_KEYS and "window" in RP.BOLD_GROUP_KEYS
    a = _runs().iloc[0].to_dict()
    rows = [
        dict(a, model="temporal_best", split="metropt_temporal", window=60, vus_pr=0.30, config_hash="t1"),
        dict(a, model="temporal_worse", split="metropt_temporal", window=60, vus_pr=0.20, config_hash="t2"),
        # a different regime and a different window: each is its own population
        dict(a, model="contaminated_only", split="metropt_contaminated", window=60, vus_pr=0.10, config_hash="c1"),
        dict(a, model="w360_only", split="metropt_temporal", window=360, vus_pr=0.05, config_hash="w1"),
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "runs.parquet", index=False)
    md = RP.leaderboard(tmp_path).read_text().split("## Selected")[0]
    assert "**temporal_best**" in md
    assert "**temporal_worse**" not in md  # loses inside its own population
    # ... but the other two populations each keep their own bold, low as their numbers are
    assert "**contaminated_only**" in md and "**w360_only**" in md
    assert "**0.100**" in md and "**0.050**" in md


def test_the_false_alarm_column_is_labelled_per_scored_day_everywhere(tmp_path):
    """The denominator is the observed span of the SCORED slice, not of the training data."""
    assert RP.FA_RATE_LABEL == "FA/scored-day"
    a = _runs().iloc[0].to_dict()
    pd.DataFrame([a]).to_parquet(tmp_path / "runs.parquet", index=False)
    md = RP.leaderboard(tmp_path).read_text()
    assert "FA/train-day" not in md
    assert md.count("FA/scored-day") >= 3  # AD table header, definition, selection table
    assert "per day of the **scored** slice" in md


def test_lift_columns_are_printed_and_are_the_metric_over_the_base_rate(tmp_path):
    a = _runs().iloc[0].to_dict()
    row = dict(a, test_positive_rate=0.25, val_positive_rate=0.20, vus_pr=0.50, val_vus_pr=0.40, config_hash="l1")
    df = pd.DataFrame([row])
    lifted = RP._add_lift(df)
    assert lifted["vus_pr_lift"].iloc[0] == pytest.approx(2.0)
    assert lifted["val_lift"].iloc[0] == pytest.approx(2.0)
    df.to_parquet(tmp_path / "runs.parquet", index=False)
    md = RP.leaderboard(tmp_path).read_text()
    header = next(line for line in md.splitlines() if line.startswith("| dataset"))
    assert " lift " in header and " val lift " in header
    assert "2.00x" in md
    # a zero or missing base rate has no lift, and must not print one
    zero = pd.DataFrame([dict(row, test_positive_rate=0.0, val_positive_rate=np.nan, config_hash="l2")])
    assert not np.isfinite(RP._add_lift(zero)["vus_pr_lift"].iloc[0])
    assert not np.isfinite(RP._add_lift(zero)["val_lift"].iloc[0])


def test_a_high_base_rate_is_flagged_in_the_row_not_only_in_a_footer(tmp_path):
    a = _runs().iloc[0].to_dict()
    hi = dict(a, dataset="cranfield", dataset_subsystem="door", model="floorish",
              test_positive_rate=0.923, vus_pr=0.97, config_hash="hb")
    pd.DataFrame([hi]).to_parquet(tmp_path / "runs.parquet", index=False)
    md = RP.leaderboard(tmp_path).read_text()
    assert "floorish** (base rate 0.92)" in md or "floorish (base rate 0.92)" in md
    assert "Base-rate warning" in md  # the list is kept as well


def test_a_deferred_pick_lists_every_candidate_ranked_on_test(tmp_path):
    """The old list was `sorted(models)[:6]`: it hid the best row, hid the worst, and let the
    uniform-random control read as a ship candidate."""
    a = _runs().iloc[0].to_dict()
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "zulu_best"]
    rows = [
        dict(a, model=n, vus_pr=0.1 + 0.05 * i, val_vus_pr=np.nan, val_auprc=np.nan,
             val_auroc=np.nan, val_positive_rate=0.0, test_positive_rate=0.03, config_hash=f"d{i}")
        for i, n in enumerate(names)
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "runs.parquet", index=False)
    block = RP.leaderboard(tmp_path).read_text().split("## Selected per subsystem")[1]
    assert "deferred" in block
    assert "ranked on TEST VUS-PR for information only" in block
    for n in names:
        assert f"`{n}`" in block, n
    assert f"all {len(names)} eligible candidate(s)" in block
    # ranked, best first - zulu_best has the highest test VUS-PR and is listed as number 1
    assert "1. `zulu_best`" in block


def test_a_pick_that_never_alarmed_says_so_instead_of_reading_as_a_detector(tmp_path):
    """Every Cranfield AD row raises zero episodes at all three thresholds: recall is 0/144 by
    construction and the name alone would read as a shipped detector."""
    a = _runs().iloc[0].to_dict()
    mute = dict(a, dataset="cranfield", dataset_subsystem="door", model="nn1_distance",
                far_not_applicable_reason="fabricated timeline",
                threshold_budget_val_far_per_train_day=np.nan,
                budget_false_alarms_per_train_day=np.nan, budget_n_false_alarms=0.0,
                budget_n_episodes=0.0, q995_n_episodes=0.0, q999_n_episodes=0.0,
                budget_event_recall=0.0, budget_n_events=144, budget_n_events_detected=0,
                test_positive_rate=0.923, val_positive_rate=0.923, val_vus_pr=0.97,
                vus_pr=0.971, config_hash="mute")
    pd.DataFrame([mute]).to_parquet(tmp_path / "runs.parquet", index=False)
    block = RP.leaderboard(tmp_path).read_text().split("## Selected per subsystem")[1]
    assert "no alarm ever fired: AD ranking only, see the CLS table" in block
    # and a pick that did alarm does not carry the phrase
    live = dict(mute, model="alarms", budget_n_episodes=5.0, q995_n_episodes=3.0, q999_n_episodes=1.0,
                val_vus_pr=0.99, config_hash="live")
    pd.DataFrame([live]).to_parquet(tmp_path / "runs.parquet", index=False)
    block = RP.leaderboard(tmp_path).read_text().split("## Selected per subsystem")[1]
    assert "no alarm ever fired" not in block


def test_the_chance_floor_is_printed_under_every_pick(tmp_path):
    a = _runs().iloc[0].to_dict()
    real = dict(a, dataset="sim", dataset_subsystem="door", model="detector", val_vus_pr=0.80,
                val_positive_rate=0.20, test_positive_rate=0.20, vus_pr=0.70, config_hash="s1")
    rand = dict(real, model="random_score", val_vus_pr=0.21, vus_pr=0.22,
                budget_event_recall=0.6667, budget_n_events=3, budget_n_events_detected=2, config_hash="s2")
    lonely = dict(a, dataset="ottawa", dataset_subsystem="bearing", model="pca_spe_t2",
                  val_vus_pr=0.97, val_positive_rate=0.667, test_positive_rate=0.667, config_hash="o1")
    pd.DataFrame([real, rand, lonely]).to_parquet(tmp_path / "runs.parquet", index=False)
    block = RP.leaderboard(tmp_path).read_text().split("## Selected per subsystem")[1]
    assert "chance floor (`random_score`" in block and "0.667 (2/3)" in block
    assert "no `random_score` control was run on this population" in block


def test_selection_ranks_on_validation_lift_not_on_the_raw_validation_metric(tmp_path):
    """A 78.6 %-positive population hands any scorer a high val VUS-PR. Lift is what compares."""
    assert RP.SELECTION_METRICS[0] == "val_lift"
    a = _runs().iloc[0].to_dict()
    hi = dict(a, dataset="sim", dataset_subsystem="door", model="sensor_range_baseline",
              input_kind="window_stats", val_vus_pr=0.9788, val_positive_rate=0.7855,
              test_positive_rate=0.7529, vus_pr=0.947, config_hash="hi")
    lo = dict(a, dataset="sim", dataset_subsystem="door", model="cusum_cycle_scalar",
              input_kind="cycle_features", val_vus_pr=0.8793, val_positive_rate=0.2076,
              test_positive_rate=0.2698, vus_pr=0.722, config_hash="lo")
    pd.DataFrame([hi, lo]).to_parquet(tmp_path / "runs.parquet", index=False)
    block = RP.leaderboard(tmp_path).read_text().split("## Selected per subsystem")[1]
    assert "**cusum_cycle_scalar**" in block and "**sensor_range_baseline**" not in block
    assert "val_lift" in block and "4.24x" in block


def test_selection_falls_back_to_the_raw_validation_metric_when_no_lift_exists(tmp_path):
    """Lift is undefined on a validation slice with no positives; the pick must not vanish."""
    a = _runs().iloc[0].to_dict()
    rows = [
        dict(a, model="better", val_vus_pr=0.80, val_positive_rate=0.0, config_hash="b1"),
        dict(a, model="worse", val_vus_pr=0.30, val_positive_rate=0.0, config_hash="b2"),
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "runs.parquet", index=False)
    block = RP.leaderboard(tmp_path).read_text().split("## Selected per subsystem")[1]
    assert "**better**" in block and "**worse**" not in block
    assert "| val_vus_pr |" in block


def test_the_selection_table_shows_episodes_and_precision(tmp_path):
    a = _runs().iloc[0].to_dict()
    row = dict(a, dataset="sim", dataset_subsystem="bearing", model="cusum_cycle_scalar",
               val_vus_pr=0.79, val_positive_rate=0.0156, test_positive_rate=0.0165,
               budget_n_episodes=12.0, budget_n_events=12, budget_n_events_detected=10,
               budget_n_false_alarms=2.0, config_hash="p1")
    pd.DataFrame([row]).to_parquet(tmp_path / "runs.parquet", index=False)
    block = RP.leaderboard(tmp_path).read_text().split("## Selected per subsystem")[1]
    header = next(line for line in block.splitlines() if line.startswith("| subsystem"))
    for col in ("val lift", "test lift", "episodes (n)", "precision", RP.FA_RATE_LABEL):
        assert f" {col} " in header, col
    assert "0.833 (10/12)" in block  # precision = detected / (detected + false alarms)


def test_the_cls_table_carries_folds_voting_and_the_per_window_score(tmp_path):
    """0.988 is a 20-window majority vote on a random split whose per-window F1 is 0.840, shown
    in the same table as un-voted single-prediction rows."""
    base = _runs().iloc[2].to_dict()
    rnd = dict(base, model="resnet1d", split="cranfield_random_rep", macro_f1=0.988,
               window_macro_f1=0.840, voted=True, n_folds=1.0, config_hash="r1")
    grp = dict(base, model="random_forest_cycle", split="cranfield_loo_load", macro_f1=0.502,
               window_macro_f1=np.nan, voted=False, n_folds=3.0, config_hash="g1")
    pd.DataFrame([rnd, grp]).to_parquet(tmp_path / "runs.parquet", index=False)
    md = RP.leaderboard(tmp_path).read_text()
    header = next(line for line in md.splitlines() if line.startswith("| dataset") and "macro-F1" in line)
    for col in ("per-window macro-F1", "voted", "n_folds"):
        assert f" {col} " in header, col
    assert "| yes |" in md and "| no |" in md
    gap = next(line for line in md.splitlines() if "generalisation gap" in line)
    assert "0.486" in gap
    assert "`cranfield_loo_load` 3 fold(s)" in gap


def test_the_header_names_every_revision_and_the_seed(tmp_path):
    df = _runs().copy()
    df["git_rev"] = ["63d4e65", "18c4478", "63d4e65", "da84c60", "4c2802b"]
    df["seed"] = 0
    df["n_folds"] = 1.0
    df.to_parquet(tmp_path / "runs.parquet", index=False)
    head = RP.leaderboard(tmp_path).read_text().split("## Task")[0]
    assert "`63d4e65` x 2 row(s)" in head
    assert "`18c4478` x 1 row(s)" in head and "`da84c60` x 1 row(s)" in head
    assert "`4c2802b` x 1 row(s)" in head
    assert "fold aggregation" in head  # what changed between them
    assert "0 row(s) in this frame are stale" in head
    assert "Single seed (seed=0) everywhere" in head


def test_the_header_says_which_columns_the_horizon_governs(tmp_path):
    _runs().to_parquet(tmp_path / "runs.parquet", index=False)
    head = RP.leaderboard(tmp_path).read_text().split("## Task")[0]
    assert "do not use `H` at all" in head
    assert "lead to failure (h)" in head and "end** of the labelled failure interval" in head
    assert "A MetroPT F1 near 1.0" not in head  # the anchor that pointed at no column


def test_the_heatmap_facets_per_dataset_and_subsystem(tmp_path):
    """Pivoting over all datasets made every axis a proxy for the dataset."""
    a = _runs().iloc[0].to_dict()
    rows = [
        dict(a, dataset="metropt3", dataset_subsystem="pneumatic", model="m", window=60, vus_pr=0.30, config_hash="a"),
        dict(a, dataset="metropt3", dataset_subsystem="pneumatic", model="m", window=360, vus_pr=0.20, config_hash="b"),
        dict(a, dataset="ottawa", dataset_subsystem="bearing", model="m", window=1.0, vus_pr=0.99, config_hash="c"),
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "runs.parquet", index=False)
    html = RP.ablation_heatmap(tmp_path).read_text()
    assert "metropt3" in html and "pneumatic" in html
    # ottawa has a single window level of its own, so it is not drawn as a one-column strip
    # beside metropt3's - and it certainly is not a column of metropt3's window map
    assert "ottawa \\u002f bearing" not in html


def test_the_survivor_label_counts_at_one_scope(tmp_path):
    """"1 of N runs" marks a model whose ONLY finished run in the whole frame is that cell, out of
    N attempted across every dataset. A model with one finished run per facet and no failures is
    an ordinary row, not a survivor (the old per-facet count against a per-dataset denominator
    labelled such rows "1 of 3")."""
    a = _runs().iloc[0].to_dict()
    ok = dict(a, dataset="sim", status="ok")
    rows = [
        # ordinary: one finished run per subsystem facet, nothing failed
        dict(ok, dataset_subsystem="door", model="steady", window=60, vus_pr=0.3, config_hash="d1"),
        dict(ok, dataset_subsystem="door", model="steady", window=360, vus_pr=0.2, config_hash="d2"),
        dict(ok, dataset_subsystem="pneumatic", model="steady", window=60, vus_pr=0.3, config_hash="p1"),
        dict(ok, dataset_subsystem="pneumatic", model="steady", window=360, vus_pr=0.2, config_hash="p2"),
        # survivor: one finished run anywhere, four attempts (two on another dataset) timed out
        dict(ok, dataset_subsystem="door", model="mp", window=60, vus_pr=0.5, config_hash="s1"),
        dict(ok, dataset_subsystem="door", model="mp", window=360, vus_pr=np.nan, status="timeout", config_hash="s2"),
        dict(ok, dataset_subsystem="pneumatic", model="mp", window=60, vus_pr=np.nan, status="timeout", config_hash="s3"),
        dict(ok, dataset="metropt3", dataset_subsystem="pneumatic", model="mp", window=60, vus_pr=np.nan, status="timeout", config_hash="s4"),
        dict(ok, dataset="metropt3", dataset_subsystem="pneumatic", model="mp", window=360, vus_pr=np.nan, status="timeout", config_hash="s5"),
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "runs.parquet", index=False)
    html = RP.ablation_heatmap(tmp_path).read_text()
    assert "1 of 5 runs" in html
    assert "1 of 3 runs" not in html and "1 of 2 runs" not in html and "1 of 4 runs" not in html
