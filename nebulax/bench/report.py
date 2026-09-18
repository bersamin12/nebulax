"""Reporting: the leaderboard markdown and the ablation heatmap.

Two products, both read-only over ``results/``:

``leaderboard(results_dir)``
    ``results/leaderboard.md`` - one table per **task family** (ad, cpd, cls, rul), the best
    row per dataset in **bold**, a "Selected per subsystem" mini-table naming the model the
    demo should ship for brake air / bearings, and a failures table so a model that
    crashed or timed out is visible rather than absent.
``ablation_heatmap(results_dir)``
    ``results/ablation_heatmap.html`` - a plotly heatmap per ablation axis (window, feature
    set, peer normalisation, contamination, training regime), model x level, coloured by the
    task's primary metric.

Reporting rules this module enforces
------------------------------------
* **Every MetroPT recall figure carries its event count.** Recall over 4 events is not a
  percentage anyone should read without N, and a test slice usually contains fewer than 4.
  The leaderboard prints ``recall (k/N)`` and repeats N in a footnote.
* **No best-of-N.** Rows are shown as run; nothing is maximised over seeds or thresholds.
  The operational columns are always the ``budget`` threshold, with the 99.5 / 99.9 %
  quantile thresholds reported beside it, never instead of it.
* **A sanity anchor is printed**: TSB-AD's best VUS-PR is 0.354 (multivariate) / 0.440
  (univariate), and every VUS-PR is printed beside its **lift over the base rate**, because on
  a slice that is 92 % positive a VUS-PR of 0.97 is a lift of 1.05 and means nothing.
  The single-feature baseline actually run on MetroPT is quoted from the frame, not from a
  remembered F1.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "load_results",
    "leaderboard",
    "ablation_heatmap",
    "PRIMARY_METRIC",
    "GEN_GAP_PAIRS",
    "GEN_GAP_KEY",
    "SELECTION_METRICS",
    "BUDGET_FILTER_METRIC",
    "BOLD_GROUP_KEYS",
    "FA_RATE_LABEL",
    "GIT_REV_HISTORY",
]

LOGGER = logging.getLogger(__name__)

#: Primary metric per task, and whether bigger is better.
PRIMARY_METRIC: dict[str, tuple[str, bool]] = {
    "ad": ("vus_pr", True),
    "cpd": ("vus_pr", True),
    "cls": ("macro_f1", True),
    "rul": ("rmse", False),
}

#: ``generalisation_gap = F1(random split) - F1(held-out group split)``: which random-split
#: preset is the control for which held-out-group preset.
GEN_GAP_PAIRS: dict[str, str] = {
    "cranfield_loo_load": "cranfield_random_rep",
    "cranfield_loo_profile": "cranfield_random_rep",
    "cranfield_rep": "cranfield_random_rep",
    "sim_loo_unit": "sim_random_unit",
    "sim_run_kfold": "sim_random_unit",
}

#: The label used everywhere for ``budget_false_alarms_per_train_day``. The denominator is the
#: observed span of the **scored** slice (``metrics.train_days`` computes it from that slice),
#: not the span of the training data, so "FA/train-day" was a misnomer: the contaminated MetroPT
#: regime trains on twice as much data and shows a *smaller* denominator.
FA_RATE_LABEL: str = "FA/scored-day"

_AD_COLUMNS: list[tuple[str, str]] = [
    ("operating_point", "op. point"),
    ("vus_pr", "VUS-PR"),
    ("vus_pr_lift", "lift"),
    ("auprc", "AUPRC"),
    ("auroc", "AUROC"),
    ("test_positive_rate", "test +rate"),
    ("val_positive_rate", "val +rate"),
    ("val_lift", "val lift"),
    ("budget_event_recall", "recall@budget"),
    ("budget_false_alarms_per_train_day", FA_RATE_LABEL),
    ("budget_n_false_alarms", "FA (n)"),
    ("budget_n_episodes", "episodes (n)"),
    ("budget_median_lead_time_s", "lead to failure (h)"),
    ("q995_event_recall", "recall@99.5%"),
    ("q999_event_recall", "recall@99.9%"),
    ("fit_seconds", "fit (s)"),
    ("score_seconds_per_10k", "score s/10k"),
    ("peak_rss_mb", "RAM (MB)"),
    ("peak_vram_mb", "VRAM (MB)"),
]
_CLS_COLUMNS: list[tuple[str, str]] = [
    ("macro_f1", "macro-F1"),
    ("window_macro_f1", "per-window macro-F1"),
    ("voted", "voted"),
    ("n_folds", "n_folds"),
    ("balanced_accuracy", "bal.acc"),
    ("generalisation_gap", "gen. gap"),
    ("fit_seconds", "fit (s)"),
    ("score_seconds_per_10k", "score s/10k"),
    ("peak_rss_mb", "RAM (MB)"),
]
_RUL_COLUMNS: list[tuple[str, str]] = [
    ("rmse", "RMSE (s)"),
    ("mae", "MAE (s)"),
    ("monotonicity", "monotonicity"),
    ("fit_seconds", "fit (s)"),
    ("peak_rss_mb", "RAM (MB)"),
]

_ABLATION_AXES: tuple[str, ...] = ("window", "feature_set", "peer_norm", "contamination", "train_regime", "split")


def load_results(results_dir: str | Path) -> pd.DataFrame:
    """Read ``results/runs.parquet``, falling back to concatenating ``results/runs/*.parquet``.

    Returns an empty frame (not an error) when nothing has been run yet, so the report task
    is safe to wire into a Makefile before the sweep exists.
    """
    root = Path(results_dir)
    combined = root / "runs.parquet"
    if combined.exists():
        return pd.read_parquet(combined, engine="pyarrow")
    per_run = root / "runs"
    frames = []
    if per_run.is_dir():
        for p in sorted(per_run.glob("*.parquet")):
            try:
                frames.append(pd.read_parquet(p, engine="pyarrow"))
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("report: skipping unreadable %s (%s)", p, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _fmt(v: Any, col: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if bool(v) else "no"
    if isinstance(v, str):
        return v
    if col == "budget_median_lead_time_s":
        return f"{float(v) / 3600.0:.1f}"
    if col in ("n_folds", "window", "budget_n_episodes", "q995_n_episodes", "q999_n_episodes"):
        return str(int(round(float(v))))
    if col.endswith("_lift"):
        return f"{float(v):.2f}x"
    if col in ("fit_seconds", "score_seconds_per_10k", "peak_rss_mb", "peak_vram_mb", "rmse", "mae"):
        return f"{float(v):,.1f}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.3f}"


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """``num / den`` where ``den`` is a finite, strictly positive base rate, else NaN."""
    den = den.where(np.isfinite(den) & (den > 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def _add_lift(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``vus_pr_lift`` and ``val_lift``: VUS-PR divided by the positive rate of the slice
    it was measured on.

    VUS-PR (and AUPRC) floor at the base rate, so a raw 0.97 on a 92 %-positive slice is a lift
    of 1.05 - indistinguishable from a coin. Every place this module prints or ranks on a
    VUS-PR now has the lift beside it, and selection ranks on the **validation** lift.
    """
    out = df.copy()
    out["vus_pr_lift"] = _ratio(_num(out, "vus_pr"), _num(out, "test_positive_rate"))
    out["val_lift"] = _ratio(_num(out, "val_vus_pr"), _num(out, "val_positive_rate"))
    return out


def _recall_cell(row: pd.Series, col: str) -> str:
    """Recall with its event count - the "N behind the figure" rule."""
    val = row.get(col)
    prefix = col.rsplit("_event_recall", 1)[0]
    n = row.get(f"{prefix}_n_events", row.get("n_events"))
    k = row.get(f"{prefix}_n_events_detected")
    base = _fmt(val, col)
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return base
    if k is None or (isinstance(k, float) and not np.isfinite(k)):
        return f"{base} (N={int(round(float(n)))})"
    return f"{base} ({int(round(float(k)))}/{int(round(float(n)))})"


#: Every ``RunSpec`` identity field except ``split`` itself - i.e. everything ``config_hash``
#: is taken over, minus the one axis the two arms are supposed to differ in. Pinned against
#: ``RunSpec`` by ``tests/test_bench_report.py::test_generalisation_gap_key_pins_every_ablation_axis``. The generalisation gap is
#: ``F1(random split) - F1(held-out group split)`` for *otherwise identical* runs, so the join
#: has to pin every axis that changes what the model sees. Leaving out ``window``,
#: ``train_regime``, ``contamination``, ``peer_norm``, ``H`` or ``data_kwargs`` lets a control
#: from one configuration be subtracted from a treatment in another, and the difference then
#: measures the ablation rather than the grouping.
GEN_GAP_KEY: tuple[str, ...] = (
    "dataset",
    "subsystem",
    "task",
    "model",
    "params",
    "input_kind",
    "window",
    "feature_set",
    "train_regime",
    "H",
    "peer_norm",
    "contamination",
    "seed",
    "data_kwargs",
    # A budget is an experimental axis, not a scheduling detail: a deep model stops early on it
    # and any model can finish at one cap and time out at another. A control trained for 99
    # minutes is not the control for a treatment trained for 1.
    "max_minutes",
)


def _add_generalisation_gap(df: pd.DataFrame) -> pd.DataFrame:
    """``F1(random) - F1(held-out group)`` joined onto every CLS row that has a control.

    The two arms must be identical in everything but the split - see :data:`GEN_GAP_KEY`.
    """
    out = df.copy()
    out["generalisation_gap"] = np.nan
    if "macro_f1" not in out.columns or "split" not in out.columns:
        return out
    cls = out[out.get("task", "") == "cls"]
    if cls.empty:
        return out
    key = [k for k in GEN_GAP_KEY if k in out.columns]

    def _k(row: pd.Series, split: str) -> tuple:
        # NaN never equals itself, so a missing/NULL axis (window is None for cycle_features)
        # would make every row its own island. Normalise to strings with a NaN sentinel.
        return (split, *(("~" if pd.isna(row[c]) else str(row[c])) for c in key))

    ctrl_by_split: dict[tuple, float] = {}
    for i, r in cls.iterrows():
        if r["split"] in set(GEN_GAP_PAIRS.values()):
            ctrl_by_split[_k(r, str(r["split"]))] = float(_num(cls, "macro_f1").loc[i])
    for i, r in cls.iterrows():
        control_split = GEN_GAP_PAIRS.get(str(r["split"]))
        if control_split is None:
            continue
        ck = _k(r, control_split)
        if ck in ctrl_by_split:
            out.loc[i, "generalisation_gap"] = ctrl_by_split[ck] - float(_num(cls, "macro_f1").loc[i])
    return out


def _table(df: pd.DataFrame, columns: Sequence[tuple[str, str]], metric: str, bigger: bool) -> list[str]:
    """One markdown table, best row per dataset bolded."""
    # Every ablation axis that changes an operational number is printed, so two rows that
    # differ only in H / peer_norm / contamination / train_regime never read as one setting.
    head = [
        "dataset", "subsystem", "model", "split", "input_kind", "window", "feature_set",
        "train_regime", "H", "peer_norm", "contamination", "data_kwargs",
    ]
    head = [h for h in head if h in df.columns and df[h].notna().any()]
    if "data_kwargs" in head and (df["data_kwargs"].fillna("{}").astype(str).isin(("{}", "")).all()):
        head.remove("data_kwargs")  # nothing to distinguish
    lines = ["| " + " | ".join([*head, *(label for _, label in columns)]) + " |"]
    lines.append("|" + "|".join(["---"] * (len(head) + len(columns))) + "|")

    best_idx: set[Any] = set()
    if metric in df.columns:
        # Only rows with a real operating point may be bolded. Ranking is still on the test
        # primary metric (that is what the table is), but an uncalibrated or silent run has no
        # threshold, no event recall and no false-alarm rate, so calling it "best" misreads the
        # page. It is still printed, with its reason in the "op. point" column.
        vals = _num(df, metric).where(_eligible(df))
        # Bold within the POPULATION the metric was measured on - see BOLD_GROUP_KEYS. Across
        # those the no-skill floor of AUPRC / VUS-PR differs by up to 46x on the synthetic
        # fleet and a 4-class F1 is not a 13-class F1, so a dataset-wide bold would be won by
        # the base rate, not the model.
        for _, grp in df.groupby(_bold_group_keys(df), dropna=False):
            g = vals.loc[grp.index].dropna()
            if len(g):
                best_idx.add(g.idxmax() if bigger else g.idxmin())

    for i, row in df.iterrows():
        cells = [str(row.get(h, "")) for h in head]
        for col, _label in columns:
            cell = _recall_cell(row, col) if col.endswith("_event_recall") else _fmt(row.get(col), col)
            if i in best_idx and col == metric:
                cell = f"**{cell}**"
            cells.append(cell)
        if "model" in head:
            j = head.index("model")
            if i in best_idx:
                cells[j] = f"**{cells[j]}**"
            # The base-rate caveat belongs IN the row it qualifies, not 300 lines below it in a
            # warning block: a reader who looks only at the bolded cell must still see that the
            # metric floors at the slice's positive rate.
            rate = row.get("test_positive_rate")
            if isinstance(rate, (int, float, np.number)) and np.isfinite(float(rate)) and float(rate) > 0.5:
                cells[j] += f" (base rate {float(rate):.2f})"
        lines.append("| " + " | ".join(cells) + " |")
    return lines


#: Metrics a shipped model may be selected on, best first. Every one of them is computed on
#: the **validation** slice. ``vus_pr`` and the other test columns are deliberately absent:
#: picking the ladder's test-slice maximum is best-of-N, which the protocol forbids.
#:
#: ``val_lift`` (``val_vus_pr / val_positive_rate``) comes first because the raw validation
#: VUS-PR is won by whichever population happens to be most positive: on the synthetic door
#: fleet ``sensor_range_baseline`` scores 0.979 on a 78.6 %-positive slice (lift 1.25) where
#: ``random_score`` on the *same* slice already scores 0.787 (lift 1.00), while
#: ``cusum_cycle_scalar`` scores 0.879 on a 20.8 %-positive slice (lift 4.24). Ranking on the
#: raw number picks the base rate. ``val_vus_pr`` remains the fallback for a population whose
#: validation positive rate is missing or zero, where a lift is not defined.
SELECTION_METRICS: tuple[str, ...] = ("val_lift", "val_vus_pr", "val_auprc", "val_auroc")

#: The false-alarm column the eligibility filter uses - also validation-side, for the same
#: reason. ``budget_false_alarms_per_train_day`` (the test slice) is reported in the table but
#: never decides who is in it.
BUDGET_FILTER_METRIC: str = "threshold_budget_val_far_per_train_day"


def _flag(df: pd.DataFrame, col: str) -> pd.Series:
    """A boolean results column as a real bool Series (missing / object dtype -> ``False``)."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(bool)


def _eligible(df: pd.DataFrame) -> pd.Series:
    """Rows that have a real operating point: calibrated, and not silently never-firing.

    The single predicate behind both the bolding in :func:`_table` and the pick in
    :func:`_selected_per_subsystem`. They disagreed once: selection filtered these out, but the
    leaderboard's headline still bolded the best *test* metric with no filter at all, so a run
    with ``uncalibrated=True`` - no threshold, no event recall, no false-alarm rate - printed as
    the bolded best AD model on the page a reader actually looks at. Tables that carry neither
    flag (``cls``, ``rul``) get an all-True mask and bold as before.
    """
    # A fabricated-timeline row's "budget" threshold is a q0.995 stand-in: whether it fired on
    # a 10-row validation record says nothing about the model, so "silent" does not disqualify
    # it there. It is still ranked below any row with measured false-alarm evidence, its FA
    # count is printed, and its op. point says the budget was not applicable.
    silent = _flag(df, "threshold_budget_silent") & ~_budget_not_applicable(df)
    return ~_flag(df, "uncalibrated") & ~silent & ~_flag(df, "episodes_impossible")


def _measured_evidence(df: pd.DataFrame) -> pd.Series:
    """Rows whose operational numbers are measurements: a real-timeline budget AND at least
    one event in the scored slice AND a finite false-alarm count."""
    n_ev = _num(df, "budget_n_events") if "budget_n_events" in df.columns else pd.Series(np.nan, index=df.index)
    n_fa = _num(df, "budget_n_false_alarms") if "budget_n_false_alarms" in df.columns else pd.Series(np.nan, index=df.index)
    far = _num(df, "budget_false_alarms_per_train_day") if "budget_false_alarms_per_train_day" in df.columns else pd.Series(np.nan, index=df.index)
    return ~_budget_not_applicable(df) & (n_ev > 0) & (n_fa.notna() | far.notna())


#: The axes that define one **scored population**. Two rows may only be compared - and only one
#: of them bolded - when they agree on all of these, because each of them changes the no-skill
#: floor of the primary metric or the number of rows it is measured over:
#:
#: * ``dataset`` / ``dataset_subsystem`` - different fleets, different base rates (0.031 on
#:   MetroPT, 0.923 on Cranfield);
#: * ``input_kind`` / ``data_kwargs`` - cycle features and window statistics are different row
#:   populations, and a 4-class F1 is not a 13-class F1;
#: * ``split`` - MetroPT's temporal split scores 4 events over 31,923 rows and the contaminated
#:   split 2 events over 3,298-20,586; bolding one against the other compares regimes;
#: * ``window`` - on the synthetic door fleet the test positive rate is 0.544 at one window and
#:   0.753 at another, and the scored row count differs by 500x.
BOLD_GROUP_KEYS: tuple[str, ...] = (
    "dataset",
    "dataset_subsystem",
    "input_kind",
    "data_kwargs",
    "split",
    "window",
)


def _bold_group_keys(df: pd.DataFrame) -> list[pd.Series]:
    """Grouping keys for the per-table bold: the scored population, not the dataset alone."""
    keys: list[pd.Series] = []
    for col in BOLD_GROUP_KEYS:
        if col in df.columns:
            keys.append(df[col].fillna("").astype(str))
    return keys or [pd.Series("", index=df.index)]


def _budget_not_applicable(df: pd.DataFrame) -> pd.Series:
    """Rows whose false-alarm budget is not a quantity (fabricated timeline)."""
    if "far_not_applicable_reason" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["far_not_applicable_reason"].fillna("").astype(str).str.len() > 0


def _operating_point(df: pd.DataFrame) -> pd.Series:
    """A human-readable reason a row is or is not eligible, for the table's ``op. point``
    column - so "why is that row not bold?" is answerable from the leaderboard itself."""
    unc = _flag(df, "uncalibrated")
    silent = _flag(df, "threshold_budget_silent")
    impossible = _flag(df, "episodes_impossible")
    out = pd.Series("calibrated", index=df.index, dtype=object)
    na = _budget_not_applicable(df)
    out[silent & ~na] = "silent"
    out[na & ~silent] = "q0.995 (budget n/a: fabricated timeline)"
    out[na & silent] = "q0.995 (budget n/a: fabricated timeline; silent on validation)"
    out[impossible] = "episodes impossible (< 3 rows per series)"
    out[unc] = "uncalibrated"
    return out


def _selected_per_subsystem(df: pd.DataFrame) -> list[str]:
    """The model to ship per subsystem, **per dataset** - the plan's mini-table: "pneumatics ->
    best on A + D-pneumatic; doors -> best on B + D-door; bearings -> best on C + D-bearing".
    A subsystem therefore gets one pick per dataset that covers it (Cranfield's rig doors and
    the synthetic fleet's doors are different populations with non-comparable validation
    metrics), and inside each pick rows with measured false-alarm evidence outrank rows from
    a fabricated timeline, which are marked "(no false-alarm evidence)".

    Rules, all of which exist because this table is the one a reader will act on:

    1. **Only calibrated runs are eligible, judged on validation.** A run with no
       *validation* false-alarm rate never got an operating point (empty validation slice, or
       a validation slice with no scoreable rows) and is dropped, not waved through. The
       budget filter reads :data:`BUDGET_FILTER_METRIC` rather than the test-slice
       ``budget_false_alarms_per_train_day``: filtering on a test number is selection by test
       performance exactly as much as ranking on one is.
    2. **Silent thresholds are excluded.** ``threshold_budget_silent`` means the calibrated
       threshold never fired once on validation. Such a run has a perfect false-alarm rate
       for the worst possible reason and must not out-rank a detector that actually alarms.
    3. **Selection never touches a test-slice metric.** ``configs/model_ladder.yaml`` forbids
       best-of-N, and maximising ``vus_pr`` over an ablation ladder is best-of-N on the test
       slice whatever the column is labelled. Selection ranks on :data:`SELECTION_METRICS` -
       ``val_lift`` first, then ``val_vus_pr`` / ``val_auprc`` / ``val_auroc`` - all written by
       ``runner._calibrate`` from validation scores only. If none exists (MetroPT's validation
       slice holds no positive rows, so both are necessarily NaN there), **no model is
       selected**: the table says the choice is deferred and lists every eligible candidate,
       because the honest answer is that this ladder cannot pick a winner without looking at
       the test slice.
    4. **A pick that never alarmed is said so in words.** A row whose budget, q0.995 *and*
       q0.999 episode counts are all zero raised no alarm at any threshold; its event recall is
       0 by construction and it is not a detector, whatever its VUS-PR.
    5. **The chance floor is printed under the pick** where a ``random_score`` control was run
       on the same population, and its absence is printed where it was not.
    """
    header = [
        "subsystem", "dataset", "model", "split", "input_kind", "window", "selected on",
        "val lift", "VUS-PR (test)", "test lift", "episodes (n)", "recall@budget", "precision",
        FA_RATE_LABEL, "fit (s)",
    ]
    n_cols = len(header)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * n_cols) + "|"]

    def _filler(text: str) -> str:
        return "| " + text + " |" + " |" * (n_cols - 1)

    def _note(text: str) -> str:
        return "| | | " + text + " |" + " |" * (n_cols - 4)

    notes: list[str] = []
    all_ad = df[(df.get("task", "") == "ad") & (df.get("status", "") == "ok")]
    ad = all_ad
    if ad.empty:
        return lines + [_filler("_no successful AD runs yet_")]
    ad = ad[_eligible(ad)]
    # The operational filter must also be a VALIDATION number: budget_false_alarms_per_train_day
    # is measured on the test slice, so filtering on it is selection by test performance just
    # as much as ranking on it is. threshold_budget_val_far_per_train_day is what the threshold
    # was actually calibrated to achieve, and it is what an operator commits to in advance.
    far = _num(ad, BUDGET_FILTER_METRIC)
    # A fabricated-timeline dataset has no false-alarm rate to filter on (NaN with a reason);
    # its rows stay eligible and are ranked on the validation metric alone.
    eligible = ad[far.notna() | _budget_not_applicable(ad)]
    if eligible.empty:
        return lines + [_filler("_no AD run produced an eligible operating point_")]
    far = _num(eligible, BUDGET_FILTER_METRIC)
    # NOT a selection step: thresholds.calibrate scans the threshold down until the budget
    # breaks, so every calibrated row meets it on validation by construction (0 of 321 rows in
    # the published frame exceed it). The line is kept as the guarantee it is.
    ok_budget = eligible[(far <= 1.0 / 7.0 + 1e-9) | _budget_not_applicable(eligible)]
    pool = ok_budget if not ok_budget.empty else eligible
    # Group on the subsystem the LOADER declares (every row carries dataset_subsystem), not on
    # the spec axis, which a config may leave empty for the real datasets. Inside a group a
    # row with measured false-alarm evidence always outranks a fabricated-timeline row: the
    # latter is only picked when nothing with a measured rate exists, and is marked as such.
    # One pick per (subsystem, dataset): "door" on Cranfield and "door" on the synthetic
    # fleet are different populations with non-comparable validation metrics.
    key = ["dataset_subsystem", "dataset"] if "dataset_subsystem" in pool.columns else ["subsystem", "dataset"]
    for (subsystem, dataset), grp in pool.groupby(key, dropna=False):
        with_far = grp[_measured_evidence(grp)]
        no_far_evidence = with_far.empty
        grp = grp if no_far_evidence else with_far
        criterion, vals = "", pd.Series(dtype=float)
        for col in SELECTION_METRICS:
            if col in grp.columns:
                cand = _num(grp, col).dropna()
                if len(cand):
                    criterion, vals = col, cand
                    break
        floor = _chance_floor(
            all_ad, subsystem, dataset, key, {str(s) for s in grp.get("split", pd.Series(dtype=object))}
        )
        if not len(vals):
            # No validation-slice metric exists for this subsystem. Falling back to the test
            # column here is exactly the forbidden best-of-N, so the table defers instead - but
            # every eligible candidate is listed, ranked on the test metric and labelled as
            # information only, because an alphabetical truncation to six names hid the best
            # row and left the uniform-random control looking like a ship candidate.
            ranked = _rank_for_information(grp)
            lines.append(
                f"| {subsystem or '-'} | {dataset if dataset is not None else '-'} | "
                f"_deferred: no validation-slice metric_ | - | - | - | none | - | - | - | - | - | - | - | - |"
            )
            lines.append(_note(f"_all {len(ranked)} eligible candidate(s) listed below the table_"))
            if floor:
                lines.append(_note(floor))
            notes += [
                "",
                f"**Deferred pick - {subsystem or '-'} / {dataset if dataset is not None else '-'}: "
                f"all {len(ranked)} eligible candidate(s), ranked on TEST VUS-PR for information only.** "
                "This ranking is *not* a selection and must not be read as one: the maximum of "
                f"{len(ranked)} test-slice value(s) over one seed is an upper bound, not an estimate.",
                "",
                *[f"{i}. {line}" for i, line in enumerate(ranked, start=1)],
            ]
            continue
        row = grp.loc[vals.idxmax()]
        never_fired = _never_alarmed(row)
        model_cell = f"**{row.get('model', '-')}**"
        if no_far_evidence:
            model_cell += " (no false-alarm evidence)"
        if never_fired:
            model_cell += " - no alarm ever fired: AD ranking only, see the CLS table"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(subsystem or "-"),
                    str(row.get("dataset", "-")),
                    model_cell,
                    str(row.get("split", "-")),
                    str(row.get("input_kind", "-")),
                    _fmt(row.get("window"), "window"),
                    criterion,
                    _fmt(row.get("val_lift"), "val_lift"),
                    _fmt(row.get("vus_pr"), "vus_pr"),
                    _fmt(row.get("vus_pr_lift"), "vus_pr_lift"),
                    _fmt(row.get("budget_n_episodes"), "budget_n_episodes"),
                    _recall_cell(row, "budget_event_recall"),
                    _precision_cell(row),
                    _fmt(row.get("budget_false_alarms_per_train_day"), "far"),
                    _fmt(row.get("fit_seconds"), "fit_seconds"),
                ]
            )
            + " |"
        )
        if floor:
            lines.append(_note(floor))
    if len(lines) == 2:
        return lines + [_filler("_no eligible AD run_")]
    return lines + notes


def _never_alarmed(row: pd.Series) -> bool:
    """True when the row raised no alarm episode at *any* of the three thresholds."""
    counts = [row.get(f"{p}_n_episodes") for p in ("budget", "q995", "q999")]
    vals = [float(c) for c in counts if isinstance(c, (int, float, np.number)) and np.isfinite(float(c))]
    return bool(vals) and len(vals) == len(counts) and all(v == 0 for v in vals)


def _precision_cell(row: pd.Series) -> str:
    """``detected / (detected + false alarms)`` at the budget threshold, with its counts."""
    det, fa = row.get("budget_n_events_detected"), row.get("budget_n_false_alarms")
    try:
        d, f = float(det), float(fa)
    except (TypeError, ValueError):
        return "-"
    if not (np.isfinite(d) and np.isfinite(f)) or (d + f) <= 0:
        return "-"
    return f"{d / (d + f):.3f} ({int(round(d))}/{int(round(d + f))})"


def _chance_floor(ad: pd.DataFrame, subsystem: Any, dataset: Any, key: Sequence[str], splits: set[str]) -> str:
    """The ``random_score`` control's numbers on this exact population, or a statement that no
    chance control was run on it - the event-recall column is unreadable without one.

    Restricted to the **splits the candidate pool actually uses**: MetroPT's contaminated regime
    scores 2 events and its temporal regime 4, so a floor taken from the wrong regime would be
    quoted beside a recall it is not comparable to.
    """
    sub_col = key[0]
    if "model" not in ad.columns or sub_col not in ad.columns or "dataset" not in ad.columns:
        return ""
    same = ad[(ad[sub_col].astype(object) == subsystem) & (ad["dataset"].astype(object) == dataset)]
    rnd = same[same["model"].astype(str) == "random_score"]
    if splits and "split" in rnd.columns:
        rnd = rnd[rnd["split"].astype(str).isin(splits)]
    if rnd.empty:
        return (
            "_chance floor: **no `random_score` control was run on this population**, so the "
            "recall and VUS-PR above have no printed floor beside them_"
        )
    vus = _num(rnd, "vus_pr")
    best = rnd.loc[vus.idxmax()] if vus.notna().any() else rnd.iloc[0]
    return (
        f"_chance floor (`random_score`, best of {len(rnd)} run(s) on this population, "
        f"{best.get('input_kind', '-')} w={_fmt(best.get('window'), 'window')}, "
        f"split {best.get('split', '-')}): "
        f"VUS-PR {_fmt(best.get('vus_pr'), 'vus_pr')} (lift {_fmt(best.get('vus_pr_lift'), 'vus_pr_lift')}), "
        f"recall@budget {_recall_cell(best, 'budget_event_recall')}, "
        f"{FA_RATE_LABEL} {_fmt(best.get('budget_false_alarms_per_train_day'), 'far')}_"
    )


def _rank_for_information(grp: pd.DataFrame) -> list[str]:
    """Every eligible candidate of a deferred pick, ranked on the TEST primary metric."""
    vals = _num(grp, "vus_pr")
    order = vals.sort_values(ascending=False, na_position="last").index
    out: list[str] = []
    for i in order:
        r = grp.loc[i]
        out.append(
            f"`{r.get('model', '-')}` - VUS-PR (test) {_fmt(r.get('vus_pr'), 'vus_pr')} "
            f"(lift {_fmt(r.get('vus_pr_lift'), 'vus_pr_lift')}), "
            f"recall@budget {_recall_cell(r, 'budget_event_recall')}, "
            f"{FA_RATE_LABEL} {_fmt(r.get('budget_false_alarms_per_train_day'), 'far')}, "
            f"{r.get('input_kind', '-')} w={_fmt(r.get('window'), 'window')}, split {r.get('split', '-')}"
        )
    return out


#: The revisions that produced the published rows, **oldest first**, with what changed at each
#: one and whether that change can alter a number already in the frame. A configs-only commit
#: changes which runs exist, never what an existing row says; only a metric-affecting commit can
#: make an earlier row stale, and the header says how many earlier rows it could touch.
GIT_REV_HISTORY: tuple[tuple[str, str, bool], ...] = (
    (
        "18c4478",
        "bench core round-8 fixes: bold within the scored population, sparse-positive warning, "
        "opaque series ids for fabricated-timeline data, metropt3 `cycle_features` refused for cpd",
        False,
    ),
    (
        "63d4e65",
        "fold aggregation: positive-row counts are summed, so the published positive rate is "
        "micro-averaged - **this is the only commit here that changes an existing number, and "
        "it can only change a run with more than one fold**",
        True,
    ),
    ("da84c60", "configs only: `k_of_n_corroboration` member channels, `lgbm_residual` targets, thread caps", False),
    ("4c2802b", "configs only: the sim bearing block drops `dwt_w8w9_startpeak`", False),
)


def _git_rev_note(df: pd.DataFrame) -> str:
    """Every git revision in the frame with its row count, what changed at it, and whether any
    row in the frame predates a change that could have altered its numbers."""
    if "git_rev" not in df.columns:
        return "**Revisions.** No `git_rev` column in this frame."
    counts = df["git_rev"].astype(str).value_counts().to_dict()
    known = [r for r, _why, _m in GIT_REV_HISTORY]
    order = [r for r in known if r in counts] + sorted(r for r in counts if r not in known)
    why = {r: w for r, w, _m in GIT_REV_HISTORY}
    parts = [f"**Revisions.** The {len(df)} rows were produced at {len(order)} revision(s), oldest first:"]
    for r in order:
        parts.append(f"`{r}` x {counts[r]} row(s) - {why.get(r, 'not in the recorded history of this report')};")
    # Staleness: a row is only stale if it was produced BEFORE a metric-affecting commit and is
    # of a kind that commit could touch. The one metric-affecting commit here changes multi-fold
    # aggregation only, so single-fold rows produced before it are unaffected.
    last_metric = max((i for i, (_r, _w, m) in enumerate(GIT_REV_HISTORY) if m), default=-1)
    stale, older = 0, 0
    if last_metric >= 0:
        before = {r for r, _w, _m in GIT_REV_HISTORY[:last_metric]}
        pre = df[df["git_rev"].astype(str).isin(before)]
        older = len(pre)
        folds = _num(pre, "n_folds")
        stale = int((folds > 1).sum())
    parts.append(
        f"{older} row(s) predate the one commit that changes an existing number, and {stale} of "
        f"them have more than one fold, so **{stale} row(s) in this frame are stale**."
    )
    return " ".join(parts)


def _seed_note(df: pd.DataFrame) -> str:
    """The seeds behind the frame: a single seed means no error bar anywhere on this page."""
    if "seed" not in df.columns:
        return "Seed not recorded."
    seeds = sorted({int(s) for s in _num(df, "seed").dropna().tolist()})
    if len(seeds) == 1:
        return (
            f"**Single seed (seed={seeds[0]}) everywhere**: every number on this page is one "
            "draw, and no interval on this page is a confidence interval over seeds."
        )
    return f"Seeds: {seeds}."


def _sanity_anchor(ok: pd.DataFrame) -> str:
    """TSB-AD's published numbers, plus the single-feature baseline actually run here.

    The old anchor quoted "a MetroPT F1 near 1.0" - there is no F1 column in the AD table and
    the single-feature baseline does not reach 1.0 on anything, so it anchored nothing.
    """
    text = (
        "**Sanity anchors.** TSB-AD's best VUS-PR is 0.354 (multivariate) / 0.440 (univariate), "
        "on slices whose positive rate is a few percent. A VUS-PR is only comparable to those "
        "figures through its **lift** over the base rate, which is printed beside every one of "
        "them in the `lift` column."
    )
    ad = ok[(ok.get("task", "") == "ad") & (ok.get("dataset", "") == "metropt3")]
    if ad.empty or "model" not in ad.columns:
        return text
    base = ad[ad["model"].astype(str) == "single_feature_threshold"]
    vus = _num(base, "vus_pr")
    if base.empty or not vus.notna().any():
        return text
    b = base.loc[vus.idxmax()]
    best_all = _num(ad, "vus_pr").where(_eligible(ad))
    extra = ""
    if best_all.notna().any():
        w = ad.loc[best_all.idxmax()]
        extra = (
            f" against `{w.get('model', '-')}` at {_fmt(w.get('vus_pr'), 'vus_pr')} "
            f"(lift {_fmt(w.get('vus_pr_lift'), 'vus_pr_lift')}) for the best eligible MetroPT row"
        )
    return (
        text
        + f" On MetroPT-3 the single-feature baseline `single_feature_threshold` reaches VUS-PR "
        f"{_fmt(b.get('vus_pr'), 'vus_pr')} (lift {_fmt(b.get('vus_pr_lift'), 'vus_pr_lift')}, "
        f"AUROC {_fmt(b.get('auroc'), 'auroc')}) on `{b.get('input_kind', '-')}`{extra}: a "
        "MetroPT VUS-PR in the 0.4 range is a one-feature result, not a win."
    )


def _cls_generalisation_statement(sub: pd.DataFrame) -> list[str]:
    """One line per dataset: best random-split macro-F1 vs best held-out-group macro-F1, with
    the fold count of every split behind it - a 0.72 gap measured over 2 folds needs its N."""
    if "split" not in sub.columns or "macro_f1" not in sub.columns:
        return []
    controls = set(GEN_GAP_PAIRS.values())
    out: list[str] = []
    for dataset, grp in sub.groupby("dataset", dropna=False):
        rnd = grp[grp["split"].astype(str).isin(controls)]
        grouped = grp[grp["split"].astype(str).isin(set(GEN_GAP_PAIRS) - controls)]
        folds = ", ".join(
            f"`{s}` {int(round(float(_num(g, 'n_folds').max())))} fold(s)"
            for s, g in sorted(grp.groupby("split", dropna=False), key=lambda kv: str(kv[0]))
            if _num(g, "n_folds").notna().any()
        )
        if rnd.empty or grouped.empty:
            out.append(
                f"> **{dataset}: no generalisation gap can be computed** - "
                f"{'no random-split control was run' if rnd.empty else 'no held-out-group split was run'}. "
                f"Fold counts: {folds or 'not recorded'}."
            )
            continue
        r_best, g_best = _num(rnd, "macro_f1").max(), _num(grouped, "macro_f1").max()
        r_row = rnd.loc[_num(rnd, "macro_f1").idxmax()]
        g_row = grouped.loc[_num(grouped, "macro_f1").idxmax()]
        out.append(
            f"> **{dataset} generalisation gap: {r_best - g_best:.3f}.** Best random-split "
            f"macro-F1 {r_best:.3f} (`{r_row.get('model', '-')}`, `{r_row.get('split', '-')}`, "
            f"voted={_fmt(r_row.get('voted'), 'voted')}, per-window macro-F1 "
            f"{_fmt(r_row.get('window_macro_f1'), 'window_macro_f1')}) vs best held-out-group "
            f"macro-F1 {g_best:.3f} (`{g_row.get('model', '-')}`, `{g_row.get('split', '-')}`). "
            f"A random split over recordings of the same load and motion profile is not a "
            f"generalisation test; read the held-out-group number. Fold counts: {folds}."
        )
    return [line for pair in ((o, "") for o in out) for line in pair]


def leaderboard(results_dir: str | Path = "results", out_path: str | Path | None = None) -> Path:
    """Write ``results/leaderboard.md`` and return its path.

    One table per task family, the best row per dataset in bold, the "Selected per subsystem"
    mini-table, a failures table, and the event counts behind every MetroPT recall figure.
    """
    root = Path(results_dir)
    out = Path(out_path) if out_path else root / "leaderboard.md"
    df = load_results(root)
    out.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        out.write_text("# NEBULA X benchmark leaderboard\n\n_No runs found._\n")
        return out

    df = _add_lift(_add_generalisation_gap(df))
    ok = df[df.get("status", "ok") == "ok"].copy()
    ok["operating_point"] = _operating_point(ok)
    lines: list[str] = [
        "# NEBULA X benchmark leaderboard",
        "",
        f"_{len(df)} run(s); {len(ok)} ok, {int((df.get('status', '') == 'failed').sum())} failed, "
        f"{int((df.get('status', '') == 'timeout').sum())} timed out. {_seed_note(df)}_",
        "",
        _git_rev_note(df),
        "",
        "**Protocol.** No point adjustment, thresholds calibrated on the validation slice only. "
        "Alarm episodes are `score > threshold` for >= 3 windows that are consecutive **in "
        "time** (a gap wider than the nominal window step ends the run before the 3-of-3 rule "
        "is applied), merged under a 1 h gap. Model selection is on a validation-slice metric "
        "where one exists - see the note on \"Selected per subsystem\". Operational columns "
        "use the false-alarm-budget threshold (calibrated to <= 1 alarm episode per 7 days of "
        "scored operation **on the validation slice**); the 99.5 % and 99.9 % quantile "
        "thresholds are reported beside it, never instead of it.",
        "",
        "**What the bold means.** Exactly one row is bolded per **scored population** - "
        "`(dataset, dataset_subsystem, input_kind, data_kwargs, split, window)` - and it is the "
        "best *eligible* row inside that population, not the best row in the dataset. The AD "
        "table therefore carries many bolds, one per population: the no-skill floor of VUS-PR "
        "and AUPRC is the slice's positive rate, which differs by up to 50x between these "
        "populations, so a dataset-wide bold would be won by the base rate rather than by a "
        "model. A run whose `op. point` is `uncalibrated` (no threshold) or `silent` (the "
        "calibrated threshold never fired on validation) is printed but never bolded and never "
        "selected. **Exception - fabricated timelines** (Ottawa): that loader "
        "synthesises the timestamps, so its budget threshold is a q0.995 stand-in and `silent` "
        "there carries no information about the model. Such rows stay eligible, are bolded and "
        "ranked on validation numbers alone, are marked *(no false-alarm evidence)* where they "
        "are picked, and rank below any row with measured false-alarm evidence in the same pick.",
        "",
        f"**`{FA_RATE_LABEL}`.** False-alarm *episodes* per day of the **scored** slice. The "
        "denominator is the observed span of the rows that were scored (`metrics.train_days` "
        "computes it from that slice), not the span of the training data: the contaminated "
        "MetroPT regime trains on twice as much data and has the *smaller* denominator, because "
        "its test slice is shorter. The same definition is used in the AD table, in the "
        "selection table and in this prose.",
        "",
        "**`H` and `lead to failure (h)`.** `H` is the detection horizon: an alarm counts as a "
        "detection when it falls in `[onset - H, failure end]`. **`H` governs only "
        "`recall@budget`, `recall@99.5%`, `recall@99.9%`, the false-alarm columns and the lead "
        "column** - `VUS-PR`, `AUPRC` and `AUROC` are scored against the pointwise `is_faulty` "
        "labels and do not use `H` at all. `lead to failure (h)` is the time from the first "
        "alarm episode to the **end** of the labelled failure interval, not to its onset; with "
        "H = 48 h a lead >= 48 h therefore means the alarm was raised before the labelled "
        "onset. (Onset-relative leads are computed per fold but are not in this frame; adding "
        "the column needs a rerun.)",
        "",
        _sanity_anchor(ok),
        "",
    ]

    for task, (metric, bigger) in PRIMARY_METRIC.items():
        sub = ok[ok.get("task", "") == task]
        if sub.empty:
            continue
        cols = _AD_COLUMNS if task in ("ad", "cpd") else (_CLS_COLUMNS if task == "cls" else _RUL_COLUMNS)
        sub = sub.sort_values(["dataset", metric], ascending=[True, not bigger]) if metric in sub else sub
        lines += [
            f"## Task: `{task}` (primary metric: `{metric}`)",
            "",
            *_table(sub, cols, metric, bigger),
            "",
        ]
        if task in ("ad", "cpd") and "test_positive_rate" in sub.columns:
            hi = _num(sub, "test_positive_rate")
            lo = hi[(hi < 0.005) & (hi >= 0)]
            if len(lo):
                names = ", ".join(sorted({str(d) for d in sub.loc[lo.index, "dataset"]}))
                lines += [
                    f"> **Sparse-positive warning** ({names}): fewer than 0.5 % of the scored rows are positive on "
                    "some rows (see `test +rate`); AUPRC / VUS-PR there rest on a handful of rows and single-feature "
                    "rules can look perfect. Read the event recall and the false-alarm rate instead.",
                    "",
                ]
            if (hi > 0.5).any():
                names = ", ".join(sorted({str(d) for d in sub.loc[hi > 0.5, "dataset"]}))
                lines += [
                    f"> **Base-rate warning** ({names}): the scored slice is more than half positive, so AUPRC "
                    "and VUS-PR floor at the `test +rate` column - a random scorer reaches it. Judge those rows on "
                    "AUROC, recall and the monotonicity column, not on VUS-PR.",
                    "",
                ]
        if task in ("ad", "cpd"):
            n_events = sorted(
                {int(round(v)) for v in _num(sub[sub["dataset"] == "metropt3"], "budget_n_events").dropna().tolist()}
            )
            if n_events:
                lines += [
                    f"> MetroPT-3 recall figures above are over **N = {n_events} event(s)** in the "
                    f"scored test slice (the dataset has 4 air-leak episodes in total; a temporal "
                    f"test slice contains only the ones after its cut). `k/N` in each cell is "
                    f"events detected / events present.",
                    "",
                ]
        if task == "cls":
            lines += _cls_generalisation_statement(sub)

    lines += [
        "## Selected per subsystem",
        "",
        "One pick per **subsystem and dataset** (the plan's mini-table: best on the real dataset "
        "*and* best on the synthetic fleet for that subsystem). "
        "Eligibility: `status=ok`, an AD task, and a **calibrated, non-silent** operating point "
        "(a run with no validation false-alarm rate never got a threshold, and a silent "
        "threshold never fired at all; both are excluded, not defaulted in). "
        "**The validation false-alarm budget is a guarantee, not a filter:** "
        "`thresholds.calibrate` scans the threshold down until the budget breaks, so every "
        "calibrated row meets `<= 1 episode / 7 days` on validation by construction and the "
        "check drops nobody. On the **test** slice the budget is a real constraint and rows do "
        "exceed it - read the `" + FA_RATE_LABEL + "` column of the AD table. "
        "**Exception - fabricated timelines** (Ottawa): "
        "its `budget` threshold is a q0.995 stand-in with no scored-days behind it, so `silent` "
        "there says nothing about the model; such rows stay eligible, are ranked on validation "
        "numbers only, never displace a row with measured false-alarm evidence in the same pick, "
        "and are marked *(no false-alarm evidence)* when chosen. "
        "**Ranking is on validation LIFT** (`val_vus_pr / val_positive_rate`), falling back to "
        "raw `val_vus_pr` (then `val_auprc`, `val_auroc`) for a population whose validation "
        "positive rate is missing or zero and where a lift is therefore undefined. Ranking on "
        "raw validation VUS-PR can favour a population with a high positive rate. The "
        "`selected on` column names the metric that chose the "
        "row, and every metric that may appear there is computed on the **validation** slice. "
        "Where no validation-slice metric exists - MetroPT's validation window contains no "
        "faulty rows, so its val AUPRC/VUS-PR are necessarily undefined - the choice is "
        "**deferred** and *every* eligible candidate is listed below the table, ranked on the "
        "test metric and labelled as information only. Picking the ladder's best test-slice "
        "VUS-PR would be best-of-N on the test slice, which the protocol forbids.",
        "",
        *_selected_per_subsystem(df),
        "",
    ]

    bad = df[df.get("status", "ok") != "ok"]
    if len(bad):
        lines += ["## Runs that did not finish", "", "| dataset | model | split | status | why |", "|---|---|---|---|---|"]
        for _, r in bad.iterrows():
            why = str(r.get("traceback", "")).strip().splitlines()
            lines.append(
                f"| {r.get('dataset', '-')} | {r.get('model', '-')} | {r.get('split', '-')} | "
                f"{r.get('status', '-')} | `{why[-1][:160] if why else '-'}` |"
            )
        lines.append("")

    out.write_text("\n".join(lines))
    LOGGER.info("report: wrote %s", out)
    return out


def _facet_keys(df: pd.DataFrame) -> tuple[str, ...]:
    """The columns that identify one scored population for the heatmap facets."""
    sub = "dataset_subsystem" if "dataset_subsystem" in df.columns and df["dataset_subsystem"].notna().any() else "subsystem"
    return tuple(c for c in ("dataset", sub) if c in df.columns) or ("dataset",)


def ablation_heatmap(results_dir: str | Path = "results", out_path: str | Path | None = None) -> Path:
    """Write ``results/ablation_heatmap.html`` (plotly) and return its path.

    **One facet per (dataset, subsystem)**, and inside a facet one heatmap per ablation axis
    that actually varies there: rows are models, columns are the axis levels, colour is the
    task's primary metric averaged over everything else *within that facet*.

    Faceting is not cosmetic. Pivoting the metric over all datasets at once made every axis a
    proxy for the dataset: the brightest column of the ``window`` map was ``window = 1.0``,
    which is the Ottawa column and nothing else, on a slice that is 66.7 % positive, while
    ``window = 360`` mixed MetroPT (3.1 % positive) with the synthetic fleet. A cell that
    averages populations with a 50x difference in no-skill floor measures the base rate.

    A cell that is a model's **only finished run in the whole task frame** is labelled
    ``1 of N runs`` (N attempted across every dataset), because one survivor of five attempts
    is survivorship, not an ablation result.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    root = Path(results_dir)
    out = Path(out_path) if out_path else root / "ablation_heatmap.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = load_results(root)
    df = raw[raw.get("status", "ok") == "ok"] if len(raw) else raw
    if df.empty:
        out.write_text("<!doctype html><title>Ablation heatmap</title><p>No successful runs found.</p>")
        return out

    task = str(df["task"].mode().iloc[0]) if "task" in df else "ad"
    metric, bigger = PRIMARY_METRIC.get(task, ("vus_pr", True))
    df = df[df.get("task", task) == task]
    keys = _facet_keys(df)
    # Survivorship is judged at ONE scope, the whole task frame: a model whose only finished
    # run (any dataset, any facet) is this cell, out of more than one attempted, is a survivor.
    # A timed-out run has no subsystem (the loader never returned), so a finer scope cannot
    # count failures consistently, and a per-facet count of finished runs against a per-dataset
    # count of attempts labelled ordinary one-run-per-facet models as if their siblings failed.
    attempted = raw[raw.get("task", task) == task] if "task" in raw.columns else raw
    n_attempted = attempted.groupby("model", dropna=False).size() if len(attempted) else None
    n_finished = df.groupby("model", dropna=False).size() if len(df) else None

    panels: list[tuple[str, str, pd.DataFrame]] = []
    for facet, grp in df.groupby(list(keys), dropna=False):
        facet = facet if isinstance(facet, tuple) else (facet,)
        name = " / ".join("-" if pd.isna(v) else str(v) for v in facet)
        for axis in _ABLATION_AXES:
            if axis in grp.columns and grp[axis].astype(str).nunique() > 1:
                panels.append((name, axis, grp))
    if not panels or metric not in df.columns:
        out.write_text(
            f"<!doctype html><title>Ablation heatmap</title><p>Nothing to ablate: "
            f"no (dataset, subsystem) facet has more than one level on any of "
            f"{list(_ABLATION_AXES)}, or the metric `{metric}` is absent "
            f"(present={metric in df.columns}).</p>"
        )
        return out

    titles = [f"{name} - {axis} -> {metric}" for name, axis, _ in panels]
    fig = make_subplots(rows=len(panels), cols=1, subplot_titles=titles, vertical_spacing=0.12 / max(1, len(panels) / 3))
    for r, (name, axis, grp) in enumerate(panels, start=1):
        lvl = grp[axis].astype(str)
        piv = grp.pivot_table(index="model", columns=lvl, values=metric, aggfunc="mean")
        cnt = grp.pivot_table(index="model", columns=lvl, values=metric, aggfunc="count").reindex_like(piv)
        labels = np.full(piv.shape, "", dtype=object)
        for iy, model in enumerate(piv.index):
            if n_attempted is None or n_finished is None:
                continue
            total = int(n_attempted.get(model, 0))
            sole = int(n_finished.get(model, 0)) == 1
            if not sole or total <= 1:
                continue
            for ix in range(piv.shape[1]):
                if np.isfinite(float(cnt.to_numpy(dtype=float)[iy, ix] if cnt.size else np.nan)):
                    labels[iy, ix] = f"1 of {total} runs"
        fig.add_trace(
            go.Heatmap(
                z=piv.to_numpy(dtype=float),
                x=[str(c) for c in piv.columns],
                y=[str(i) for i in piv.index],
                text=labels,
                texttemplate="%{text}",
                textfont=dict(size=9, color="crimson"),
                colorscale="Viridis" if bigger else "Viridis_r",
                colorbar=dict(title=metric, len=0.9 / len(panels), y=1 - (r - 0.5) / len(panels)),
                hovertemplate=(
                    f"{name}<br>model=%{{y}}<br>{axis}=%{{x}}<br>{metric}=%{{z:.3f}}"
                    "<br>%{text}<extra></extra>"
                ),
            ),
            row=r,
            col=1,
        )
    fig.update_layout(
        title=(
            f"NEBULA X ablations - task `{task}`, metric `{metric}`, one facet per "
            f"(dataset, subsystem) so no axis mixes datasets or base rates "
            f"(mean over the other axes inside the facet).<br>"
            f"<sub>A cell labelled \"1 of N runs\" is the model's only finished run in the whole "
            f"benchmark, out of N attempted across every dataset - the rest failed or timed out, "
            f"so the cell is a survivor, not an ablation result.</sub>"
        ),
        height=320 * len(panels) + 120,
        template="plotly_white",
    )
    fig.write_html(out, include_plotlyjs="cdn")
    LOGGER.info("report: wrote %s", out)
    return out
