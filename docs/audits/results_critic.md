# Results critic — NEBULA X Track 3 leaderboard

Adversarial audit of `results/leaderboard.md` + `results/ablation_heatmap.html` against
`results/runs.parquet` (410 rows, 402 ok, 8 timeout). Every number below is a pandas query
result on that parquet; queries are quoted inline.

Reproducibility note: `report.leaderboard('results')` regenerates `results/leaderboard.md`
**byte-identically** (`difflib.unified_diff` → 0 diff lines). So there is no transcription
error anywhere in the file: every number printed is what the parquet holds. Everything below
is therefore about *claims, framing, grouping and protocol*, not arithmetic.

---

## (A) Verdict summary

1. **Two of the three headline AD numbers are base-rate artefacts.** Cranfield `nn1_distance`
   VUS-PR **0.971** sits on a test positive rate of **0.923** (lift 1.05) and fires **zero**
   alarm episodes at *every* threshold; Ottawa `pca_spe_t2` **0.992** sits on **0.667**
   (lift 1.49) with recall 7/40. Neither is evidence of detection.
2. **A uniform-random scorer reaches the operational headline.** `random_score` gets
   **3/4 MetroPT events**, **3/3 sim-pneumatic events** and **5/7 sim-door events** at the
   calibrated budget threshold. The bolded "1.000 (4/4)" is one event better than noise.
3. **The false-alarm budget is enforced only where it cannot bind and is blown where it
   matters.** 0 / 321 rows exceed 1/7 per day on validation (the calibrator guarantees it);
   **66 / 321 (21 %) exceed it on test**, including 26 of the 34 MetroPT rows that report 4/4.
   The column is also mislabelled: `FA/train-day` is false alarms per **scored-slice** day.
4. **The run set is a mixture of four git revisions** (63d4e65 ×273, 18c4478 ×76, da84c60 ×53,
   4c2802b ×8) and the header quotes only the first. The two bolded MetroPT rows come from
   *different* revs than the one printed, and rows with different fold counts and base rates
   are bolded against each other.
5. The protocol paragraph contradicts the table it introduces (silent rows *are* bolded), the
   CLS headline 0.988 is a random-split, 20-window-majority-vote number from a model never run
   on the held-out-group splits, and the "eligible candidates" list for MetroPT is silently
   truncated to 6 alphabetical names out of 28 — one of which is `random_score`.

---

## (B) Errors — report text or number contradicted by the parquet

### B1. `leaderboard.md:5` — "the best **eligible** run … `silent` … is printed but never bolded and never selected"

Contradicted by the table two lines below. `results/leaderboard.md:13` bolds
`nn1_distance` whose own `op. point` cell reads *"q0.995 (budget n/a: fabricated timeline;
**silent on validation**)"*. Same for Ottawa `pca_spe_t2`, and both fabricated-timeline rows in
the "Selected per subsystem" table (`:438`, `:440`).

```
crosstab(dataset, _operating_point(ok)):
             calibrated  q0.995(...silent on validation)  silent
cranfield            49                                8       0
metropt3             51                                0      66
ottawa               16                                8       0
sim                  95                                0     109
→ ALL 8 cranfield AD rows and ALL 8 ottawa AD rows are silent; all 16 are eligible via the
  fabricated-timeline exception, and 2 of them are bolded.
```

The exception is documented in `report._eligible` (`nebulax/bench/report.py:277-284`) and in the
"Selected per subsystem" prose (`:434`), but the *Protocol* paragraph a judge reads first states
the opposite without qualification. **Correct statement:** "…never bolded, **except on a
fabricated timeline (Cranfield, Ottawa), where `silent` carries no information and such rows are
bolded and ranked on validation VUS-PR alone.**"

### B2. `leaderboard.md:5` — "The bolded row **per dataset**"

The bold key is `_bold_group_keys` = `dataset × dataset_subsystem × input_kind × data_kwargs`
(`nebulax/bench/report.py:318-325`). The AD table has **13** bolded rows, not 4:

```
ad bolds: (cranfield,door,raw_window,{}) (metropt3,pneumatic,raw_window,{})
          (metropt3,pneumatic,window_stats,{}) (ottawa,bearing,window_stats,{})
          + 9 sim groups   →  13 bolds in one "per dataset" table
cls bolds: 8.  cpd bolds: 1.
```

**Correct statement:** "one bolded row per *scored population* — dataset × subsystem × input
kind × class target — so the AD table carries 13 bolds."

### B3. `leaderboard.md:3` — "git rev `63d4e65`"

`df['git_rev'].value_counts()` → `63d4e65: 273, 18c4478: 76, da84c60: 53, 4c2802b: 8`.
Only 67 % of rows were produced at the quoted commit, and those commits changed
*substantive* semantics (per the log: fold aggregation of positive-row counts at 63d4e65,
opaque series ids + cycle_features refusal at 18c4478, k_of_n member channels + thread caps at
da84c60). The two bolded MetroPT rows are **not** from 63d4e65:

```
k_of_n_corroboration / metropt_temporal → {'18c4478': 2}
lgbm_residual        / metropt_temporal → {'da84c60': 2}
sparse_autoencoder   / sim_loo_unit     → {'da84c60': 3}   (the sim pneumatic pick)
resnet1d             / cranfield_random_rep → {'da84c60': 1} (the CLS headline)
```

**Correct statement:** name all four revs, or re-run the 137 stale rows. As it stands the page
claims a single-commit sweep that did not happen.

### B4. Bold groups are not homogeneous populations

The bold exists to avoid comparing across different no-skill floors, yet six of the AD bold
groups mix two base rates and two scored-row counts, because `window` and `split` are **not**
grouping keys:

```
(sim, door, window_stats, {"max_runs":5}) : 35 rows
    test +rate ∈ {0.5439, 0.7529}   n_scored ∈ {344, 172647}   (500× row count)
(sim, door, raw_window,  {"max_runs":5}) : +rate ∈ {0.5439, 0.7529}
(sim, bearing, *)                        : +rate ∈ {0.0595, 0.1147}
(sim, pneumatic, *)                      : +rate ∈ {0.3611, 0.3918}
(metropt3, pneumatic, raw_window, {})    : 36 rows, splits temporal (4 events, 31923 scored)
                                            AND contaminated (2 events, 3298-20586 scored)
```

Consequence: in 5 of 6 heterogeneous groups the bold went to the sub-population with the
**higher** base rate (sim door window_stats → `mahalanobis_mincovdet` at +rate 0.753; sim door
raw_window → `ocsvm_mfcc_ams` at 0.753; sim pneumatic ×2 → 0.392). **Fix:** add `window` and
`split` to `_bold_group_keys`.

### B5. `leaderboard.md:434` — "inside the false-alarm budget **on validation**" is vacuous

```
threshold_budget_val_far_per_train_day.describe() → max = 0.133475
1/7 = 0.142857  →  rows above budget on validation: 0 of 321
_selected_per_subsystem: eligible 158 → ok_budget 158 (0 dropped)
```
`thresholds.py:236-245` calibrates the budget threshold by *scanning down until the budget
breaks*, so meeting it on validation is a construction guarantee, not a test. The clause reads
as a quality gate and filters nothing. On test it is a real filter and it fails:
`budget_false_alarms_per_train_day > 1/7` in **66 of 321** rows.

### B6. `FA/train-day` is not per train-day

```
(budget_n_false_alarms / train_days - budget_false_alarms_per_train_day).abs().max() = 0.0  (309/309 rows)
metropt_temporal     train_days = 142.9–143.2   (actual train slice: 1 Feb–31 Mar = 60 d)
metropt_contaminated train_days =  89.1– 89.2   (actual train slice: 1 Feb–31 May = 121 d)
```
The contaminated regime trains on **twice** as much data yet shows a **smaller** "train_days" —
proof the denominator is the observed span of the **scored slice** (11 Apr→1 Sep = 143 d;
4 Jun→1 Sep = 89 d), exactly as `metrics.train_days` documents ("computed from the scored slice
itself"). **Rename to `FA / monitored-day`** and restate the budget as "≤ 1 alarm episode per
7 days of monitored operation".

### B7. `lead (h)` exceeds the stated horizon H = 48 h in half the MetroPT rows

`metrics.event_metrics:408` sets `lead = t_failure - first_episode.t_start`, i.e. to the
**end** of the failure interval, while detection is credited over `[t_onset - H, t_failure]`.

```
metropt_temporal rows with a finite median lead: 52
median lead > H (172800 s): 24 rows;  max = 92.55 h;  mean = 45.1 h
e.g. leaderboard.md:55 sensor_range_baseline "lead (h) 92.6" (recall 3/4)
```
A 92.6 h "lead" on a 48 h horizon is arithmetically impossible as advance warning. The column
must either publish `lead_times_to_onset_s` (already computed, never written to the table) or be
relabelled "time from first alarm to end of failure".

### B8. The `H` column sits next to metrics that do not use `H`

The AD binary target is pointwise `is_faulty`, not the H-alarm window:
```
metropt_temporal w=60: scored rows 31923, test_positive_rate 0.0308 → 983 positives
  is_faulty rows in that slice = 984;  alarm_window rows = 2914 (0.0913)
→ AUROC/AUPRC/VUS-PR use is_faulty; H only enters recall/FA/lead.
```
Printing `H = 172800.0` in the same row as VUS-PR invites the reading that VUS-PR is scored
against a 48 h horizon. Say which columns H governs.

### B9. The "eligible candidates" list for MetroPT is truncated to 6 of 28, alphabetically, and includes the chance control

`report.py:412` does `sorted({...})[:6]`. `leaderboard.md:443` prints
`cblof, conformal_threshold, copod, ecod, halfspace_trees_ocknn, isolation_forest`.
```
distinct eligible models in the metropt pool = 28
the alphabetical cut excludes k_of_n_corroboration (the bolded best, VUS-PR 0.783),
lgbm_residual (0.587), moving_window_variance (0.572) …
and the full list contains 'random_score' — the uniform-random control — as a ship candidate.
```

### B10. The sparse-positive warning is dead code

`report.py:497` fires at `test_positive_rate < 0.005`; the minimum in the AD/CPD tables is
**0.016465** (sim bearing cycle_features), so the warning has never printed. The 63d4e65 commit
message advertises it as a shipped safeguard.

### B11. `leaderboard.md:7` — the MetroPT sanity anchor references a metric the table does not contain

"A MetroPT F1 near 1.0 is not a win — a single-feature threshold already achieves it."
There is **no F1 column** in the AD table, and by the table's own primary metric
`single_feature_threshold` peaks at VUS-PR **0.443** vs 0.783 for the best eligible row. The
anchor is unsupported by these results in either direction.

### B12. Ablation heatmap: every axis is confounded with dataset

`report.ablation_heatmap` pivots `mean(vus_pr)` by model × axis level over **all** datasets.
```
window level "1.0"  → 8 rows, ALL ottawa      (base rate 0.667)
window level "nan"  → 8 cranfield (0.923) + 36 sim cycle_features (0.016–0.270)
window "360"        → 55 metropt3 (0.031) + 94 sim
base rate by dataset: metropt3 0.031 | sim 0.016–0.769 | ottawa 0.667 | cranfield 0.923
```
The brightest column of the `window` heatmap (`pca_spe_t2` 0.992, `isolation_forest` 0.983,
`copod` 0.956 at "window = 1.0") is not a window effect — it is the only Ottawa column.
`matrix_profile_discord` shows 0.677 at window 360 from its **single** surviving run while 4 of
its 5 runs timed out (survivorship). `split` as an "ablation axis" likewise conflates datasets.
Either facet the heatmap by dataset or plot VUS-PR **lift over base rate**.

### B13. CLS headline is a random-split, majority-voted number from a model with no held-out-group run

```
crosstab(model, split) on cranfield cls:
  resnet1d : random_rep 1, loo_load 0, loo_profile 0
  litetime : random_rep 1, loo_load 0, loo_profile 0
every model that DID run the grouped splits loses 0.23–0.72 macro-F1
  (best loo_load = random_forest_cycle 0.502; best loo_profile = quant 0.437)
resnet1d: macro_f1 0.988 but window_macro_f1 0.840  (voted=True, 20 windows → 156 recordings)
lgbm_envelope 0.873 is voted=False, one row per recording
```
So `leaderboard.md:373` bolds 0.988 for a model that (a) was never tested on a split that holds
out load or motion profile, and (b) is a 20-window majority vote whose per-window F1 is 0.840,
compared in the same table against un-voted single-prediction rows. The table shows neither
`voted` nor `window_macro_f1`.

---

## (C) Sentences to soften, quoted, with a rewrite

**C1 —** *"Operational columns use the false-alarm-budget threshold (<= 1 episode / 7
train-days)"* (`:5`)
> **Rewrite:** "Operational columns use the threshold **calibrated to** ≤ 1 alarm episode per
> 7 days on the *validation* slice. That target is met on validation by construction (0 of 321
> rows exceed it). On the test slice **66 of 321 rows exceed it**, and of the 34 MetroPT rows
> reporting 4/4 event recall only **8** stay inside the budget. `FA/monitored-day` is false
> alarms per day of scored operation, not per day of training data."

**C2 —** *"The bolded row per dataset is the best **eligible** run: a run whose `op. point` is
`uncalibrated` … or `silent` … is printed but never bolded and never selected."* (`:5`)
> **Rewrite:** "One row is bolded per scored population (dataset × subsystem × input kind ×
> class target) — 13 in the AD table. Uncalibrated and silent runs are never bolded **on a real
> timeline**. Cranfield and Ottawa have fabricated timelines where `silent` carries no
> information, so their rows are bolded despite firing on no validation window; every Cranfield
> AD row in fact fires **zero** alarm episodes at all three thresholds."

**C3 —** Cranfield's bolded `| **0.971** | 0.970 | 0.773 | 0.923 |` (`:13`)
> **Rewrite (row footnote):** "Cranfield AD: the scored slice is **92.3 % positive**, so VUS-PR
> floors at 0.923 and the whole 8-model ladder spans 0.931–0.971 — a lift of 1.01–1.05 over
> chance. Read AUROC (0.528–0.773) instead. **No Cranfield model produced a single alarm
> episode** at the budget, q0.995 or q0.999 threshold
> (`budget_n_episodes = q995_n_episodes = q999_n_episodes = 0` for all 8 rows), so recall is
> 0/144 by construction: a 20-row recording cannot easily satisfy the 3-consecutive-window
> rule. Cranfield is reported as a *classification* result; its AD numbers are not evidence of
> detection."

**C4 —** Ottawa's bolded `| **0.992** | … | 0.175 (7/40) | - |` (`:438`, AD table)
> **Rewrite:** "Ottawa AD: 2/3 of the scored slice is positive, so VUS-PR floors at 0.667
> (lift 1.49). At the operating point `pca_spe_t2` raises **7 alarm episodes across 5 folds,
> all 7 inside a faulty recording (0 false alarms) — precision 7/7, recall 7/40 (17.5 %)**.
> A high-precision, low-recall screener, not a detector. There is no `random_score` control run
> on Ottawa or Cranfield, so neither has a printed chance floor."

**C5 —** *"MetroPT-3 recall figures above are over N = [2, 4] event(s)"* (`:341`) together with
the bolded `1.000 (4/4)`
> **Rewrite:** "`1.000 (4/4)` means **4 air-leak events out of 4**, all in one dataset, one
> seed, one split. The 95 % Clopper–Pearson interval on 4/4 is [0.47, 1.00] (and on 2/2, [0.22, 1.00]). It is **not**
> comparable to the `1.000 (2/2)` rows: those come from the *contaminated* regime, whose test
> slice holds only the 5 Jun and 15 Jul events, whose 2.93-day validation slice permits at most
> **0.42** alarm episodes under the budget, and **every one of whose 56 rows is therefore
> `silent` and ineligible** (`_eligible` → 0 of 56). The two regimes must never be read as
> 4/4 > 2/2. For calibration, `random_score` on the same MetroPT temporal slice reaches
> **3/4** events (`budget_event_recall = 0.75`, 0.272 FA/monitored-day)."

**C6 —** *"Judge those rows on AUROC, recall and the monotonicity column, not on VUS-PR."*
(`:339`, the base-rate warning)
> Two problems: the warning sits at line 339, **326 lines below** the bolded 0.971 it is about;
> and the bold *and* the "Selected per subsystem" pick are both made on (val\_)VUS-PR, the
> metric the warning tells the reader to ignore.
> **Rewrite:** move it directly under the AD table header, and add: "the bolded cells and the
> per-subsystem picks on Cranfield, Ottawa and the high-base-rate sim populations are chosen by
> a metric that floors at the base rate; they should be read together with the AUROC and
> event-recall columns, and with the VUS-PR **lift over base rate** printed beside them."

**C7 —** *"Sanity anchors. TSB-AD's best VUS-PR is 0.354 / 0.440."* (`:7`)
> **Rewrite:** "TSB-AD's best VUS-PR is 0.354 (multivariate) / 0.440 (univariate) on slices
> whose positive rate is a few percent. Comparing that to a 0.97 on a 92 %-positive slice is
> meaningless. The comparable figure here is the **lift over base rate**: MetroPT best
> 25.4× (0.783 / 0.0308), sim bearing 39.9× (0.658 / 0.0165), sim door 1.26× (0.947 / 0.753),
> Ottawa 1.49×, Cranfield 1.05×."

**C8 —** the "Runs that did not finish" table (`:448-457`)
> **Rewrite (add a line):** "All 8 were killed at their cap (`wall_seconds` = 1200.0 / 600.0
> exactly). `matrix_profile_discord` timed out on **both** MetroPT splits and on sim
> (4 of its 5 runs), so it is untested on MetroPT and its single surviving cell in the ablation
> heatmap is survivorship-biased. `ocsvm_mfcc_ams` timed out at window = 60 on sim (3 runs) but
> completed at 360, so its bolded sim-door raw_window row has no window = 60 counterpart. The
> caps (10 min / 20 min wall-clock) are an experimental axis: a model that finishes at one cap
> and dies at another is not the same model."

**C9 —** *"Selection uses `val_vus_pr`, then `val_auprc`"* (`report.py:369`, prose at `:434`)
> Selection maximises validation VUS-PR **across input kinds and windows**, whose validation
> base rates differ by 50×, even though `_table` refuses to bold across exactly those axes for
> exactly that reason. Concretely, for sim door:
> ```
> sensor_range_baseline  window_stats w=360  val_vus_pr 0.9788  val +rate 0.7855  lift 1.25  ← picked
> cusum_cycle_scalar     cycle_features      val_vus_pr 0.8793  val +rate 0.2076  lift 4.24
> random_score           window_stats w=360  val_vus_pr 0.7866  val +rate 0.7855  lift 1.00
> ```
> **Rewrite:** "…and the pick is made *within one input kind and window*, or on validation
> VUS-PR **lift over the validation base rate**, so that a population with a higher positive
> rate cannot win the comparison."

---

## (D) MetroPT selection — recommendation

### The facts

```
metropt_temporal  val slice 1–10 Apr:  n_val_rows 362–2291, val_faulty_rows 0,
                  val_alarm_window_rows 0  →  val_vus_pr / val_auprc / val_auroc NaN in 61/61 rows
The four events: 18 Apr | 30 May | 5–7 Jun | 15 Jul  (all after the 11 Apr cut → 4 test events)
metropt_contaminated val 1–3 Jun: val_faulty_rows 0 but val_alarm_window_rows 23–142
                  → val metrics NaN in 56/56 rows as well; 56/56 silent; 0/56 eligible
eligible metropt pool for selection: 47 rows, 28 distinct models, test vus_pr 0.017–0.783
```

### Cost of each option

**(a) Keep deferred, pick by hand with a stated caveat — statistical cost 0.**
Nothing is spent. The deliverable loses its flagship pick (pneumatics is the only subsystem
with *real* labelled failures), and "deferred" in the shipped table reads as an unfinished
benchmark rather than as a principled refusal.

**(b) Select on test VUS-PR with an "selected on test, 4 events" disclosure — cost: a
best-of-47 optimism you cannot bound.**
The pool's test VUS-PR is `mean 0.193, sd 0.195, max 0.783`; the winner sits **3.0 sd** above
the pool mean and 0.180 above the runner-up (`conformal_threshold` 0.603). With 47 draws and no
held-out re-estimate, that maximum is not an unbiased estimate of the winner's performance, and
there is **no second event set left** to correct it on — all four events are in the slice you
selected on. This is the exact best-of-N the ladder config forbids. It also cross-contaminates
the *operational* claim: the winner `k_of_n_corroboration` blows the FA budget on the same
slice (0.154/day, 22 false alarms), while the only 4/4 row inside budget is `lgbm_residual`
(0.105/day, 15 FA, VUS-PR 0.587) — so "best on test" and "meets the budget on test" disagree,
and you would be choosing which test number to honour.

**(c) Move validation to 1–25 Apr / test 26 Apr onward, rerun ~120 rows (~1 h) — cost: one of
four events, and a validation metric estimated on a single event.**
You gain a *defined* `val_vus_pr` and a genuine, protocol-clean selection. You pay:
3 test events instead of 4 (recall denominators become `k/3`; a single miss now costs 33 pp);
validation VUS-PR estimated over exactly **one** event (18 Apr, 285 faulty rows at 10 s pitch,
755 rows inside its 48 h alarm window) — a noisy criterion that will happily rank 28 models on
one leak; and the training slice grows from 60 to ~60 days unchanged but the *calibration*
slice grows from 9.95 to ~25 days, which changes the budget's episode allowance from 1.43 to
3.6 and therefore changes every threshold in the table (all 120 rows must be rerun together —
you cannot mix). Honest, but it buys a 1-event selection signal at the price of 25 % of the
evidence.

**(d) Recommended — nested "select on the earliest event, report on the rest", stated as
such.**
Keep the published `metropt_temporal` numbers exactly as they are (4 events, deferred pick),
**and add one extra selection-only split**: train Feb–Mar, *selection* slice 11–25 Apr (the
18 Apr event), *report* slice 26 Apr onward (30 May, 5–7 Jun, 15 Jul). Choose the model on the
selection slice; report its 3-event numbers from the report slice **and** its 4-event numbers
from the existing table, clearly labelled as the in-sample-for-selection view. This costs the
same ~1 h of compute as (c), but it does **not** replace the published table: the 4/4 figures
survive as descriptive, the shipped pick gets a clean out-of-sample number, and the report can
say precisely which event paid for the choice.

If there is no time for any rerun, take **(a) + a named default**: ship
`lgbm_residual / metropt_temporal / raw_window / w=360` as a **hand-chosen** default
(VUS-PR 0.587, AUROC 0.986, 4/4, **0.105 FA/monitored-day — the only 4/4 row inside the
budget**, 15 false alarms over 143 days, median 17.3 h to end-of-failure, fit 33.9 s) and state
in one sentence that it was chosen by a human on the operational constraint (budget compliance
on test), not by the ladder, and that its VUS-PR is therefore not an out-of-sample estimate.
That is more useful to a judge than a "deferred" cell and more honest than pretending 0.783 was
selected without looking.

**Do not take (b) as written.** If you do take it, the disclosure must say "chosen as the
maximum of 47 test-slice values over 4 events at one seed; the reported 0.783 is an upper bound,
not an estimate", and it must note that the chosen row is 8 % over the false-alarm budget.

---

## (E) Proposed winner per subsystem per dataset

| subsystem | dataset | proposed winner | one-line justification | caveat that must accompany it |
|---|---|---|---|---|
| pneumatic | metropt3 | `lgbm_residual` / metropt_temporal / raw_window / w=360 | VUS-PR 0.587 (**19× the 0.0313 base rate**), AUROC 0.986, 4/4 events, and the **only** 4/4 row whose test FA rate (0.105/day, 15 episodes in 143 days) stays inside the ≤1/7 budget | Hand-chosen on an operational constraint, not selected by the ladder — the validation slice (1–10 Apr) holds 0 faulty rows so no validation metric exists. 4 events, 1 seed. Row produced at rev `da84c60`, not the `63d4e65` in the header. |
| pneumatic | sim | `sparse_autoencoder` / window_stats / w=360 | Highest val_vus_pr (0.4995) **and** highest val lift (6.93× the 0.0720 val base rate) in its pool; 3/3 events, 0.110 FA/day | Only **3** events; `random_score` also reaches 3/3 on this population (0.203 FA/day), so the margin is the false-alarm rate, not the recall. Rev `da84c60`. |
| bearing | sim | `cusum_cycle_scalar` / cycle_features | Wins on both rankings: val_vus_pr 0.7920 *and* val lift **50.8×** (base rate 0.0156); 10/12 events at **0.008 FA/day** (1 false alarm) | 12 events over the leave-one-unit-out folds; simulated fleet, so this is a sanity check of the pipeline, not field evidence. |
| door | sim | `cusum_cycle_scalar` / cycle_features — **in place of** the published `sensor_range_baseline` | The published pick wins only because its population is 78.6 % positive (val_vus_pr 0.9788 = lift **1.25**, where `random_score` on the identical population scores 0.7866 = lift 1.00). `cusum_cycle_scalar` scores 0.8793 on a 20.8 %-positive population = lift **4.24** | Lower raw recall (4/12 vs 4/7) and a non-zero FA rate (0.016 vs 0.000). If you keep `sensor_range_baseline`, say plainly that the shipped door detector is a **sensor-range check** that beats chance by 0.19 VUS-PR on a slice where chance already scores 0.79. |
| bearing | ottawa | `pca_spe_t2` / window_stats — keep, **demote the framing** | Best val_vus_pr (0.9776) and best test AUROC (0.983); at the operating point it raises 7 episodes across 5 folds and **all 7 land in faulty recordings — 0 false alarms, precision 7/7** | Recall **7/40 = 17.5 %**. VUS-PR 0.992 on a 66.7 %-positive slice is lift 1.49, not a detection result. Fabricated timeline: no false-alarm rate exists and the `-` in the table should read "0 false alarms / 7 episodes". No chance control was run on Ottawa. |
| door | cranfield | **No AD winner. Withdraw the pick.** | All 8 Cranfield AD rows produce `budget_n_episodes = q995_n_episodes = q999_n_episodes = 0` — not one model raises a single alarm at any threshold; recall is 0/144 for every row | The published `nn1_distance` "0.971" is a 92.3 % base rate plus 0.048. Ship the Cranfield **classification** result instead — and then only the grouped-split numbers (best loo_load `random_forest_cycle` 0.502, best loo_profile `quant` 0.437), never the 0.988 random-split figure. |

---

## (F) Open questions for the humans

1. **Rerun or relabel the 137 stale rows?** 33 % of the parquet predates `63d4e65`, including
   4 of the 6 headline rows. Either rerun them at HEAD or print the rev per row.
2. **Which MetroPT option (§D)?** (d) and (c) both cost ~1 h; (a) ships a "deferred" cell.
   Decision needed before the table is frozen.
3. **Is the 3-consecutive-window episode rule applicable to Cranfield/Ottawa at all?**
   `max_rows_per_series = 20` (Cranfield) and `10` (Ottawa). Requiring 3 in a row inside a
   10-row recording is why Cranfield fires zero episodes and Ottawa fires 7. Either relax `k`
   for short-recording datasets and say so, or stop publishing AD on them.
4. **Add `random_score` to Cranfield and Ottawa.** It exists only for sim and MetroPT
   (`input_kind=window_stats`), so the two datasets with the highest headline numbers have no
   printed chance floor. On the populations where it *does* run it reaches 3/4 and 3/3 events —
   the event-recall column needs that floor beside it everywhere.
5. **Single seed (all 410 rows `seed = 0`).** No error bar anywhere. At minimum run 3 seeds for
   the 6 shipped picks; a 0.015 val_vus_pr gap between the top 4 sim-door candidates on a
   345-row validation slice is not resolvable at one seed.
6. **Fold counts.** `cranfield_loo_profile` is **2** folds, `cranfield_loo_load` **3**,
   `sim_loo_unit` 2–8. A "generalisation gap" of 0.72 measured over 2 folds needs that N
   printed next to it.
7. **Record epochs and best val loss for deep runs.** 28 deep rows, no `epoch`/`loss` column
   exists in the schema. `usad` on sim door fits in **1.68 s** and scores AUROC **0.506** (base
   rate 0.753) — indistinguishable from an untrained network, and unfalsifiable without the
   epoch count. Fit times for the *same* model on the same split span 100× (lstm_ae
   2.4 / 34.0 / 282.3 s; tcn_ae 14.3 / 37.3 / 443.5 s), which tracks train-set size but does not
   by itself rule out patience-8 firing at epoch 1. **Yes, this gap is worth closing** — it is
   two extra result columns and it is the first thing a reviewer will ask about the
   "deep models underperform" claim.
8. **Deep models were only ever run at window = 360 on MetroPT** (5 rows, all
   `metropt_temporal`, none contaminated). The "deep ≤ 0.94 vs 0.99" comparison is partly a
   window comparison: `moving_window_variance` reaches AUROC 0.987 at w=60 and only 0.771 at
   w=360. The like-for-like statement is "at w=360, best deep `tcn_ae` 0.935 vs best classical
   `lgbm_residual` 0.986".
9. **Below-chance rows are real, not a sign bug — decide how to say so.** Verified: score
   orientation is correct in every implicated model (`KMeansAD._score` returns
   `transform(...).min(axis=1)`, `Nn1Distance._score` returns the 1-NN distance, TSPulse returns
   reconstruction MSE — all higher = more anomalous, `statistical.py:293`, `classical.py:369`,
   `foundation.py:24`). It is a property of the data, reproduced directly from the feature cache:
   ```
   metropt3 raw_window w=60,  1-NN to 500 train windows: AUROC 0.479
       mean within-window std   positive 0.592  negative 0.848
       mean 1-NN distance       positive 26.41  negative 26.79
   metropt3 raw_window w=360:                       AUROC 0.352
       mean within-window std   positive 0.818  negative 1.149
       mean 1-NN distance       positive 118.45 negative 143.77   ← positives are CLOSER
   ```
   During an air leak the APU runs in a near-constant regime, so failure windows are **flatter**
   and their flattened vectors land nearer the dense centre of the training cloud than healthy
   windows, which contain the full compressor duty cycle. Every affected model is a
   distance-to-training-manifold score (`nn1_distance`, `kmeans_ad`, `knn_outlier`,
   `halfspace_trees_ocknn`, `tspulse_zeroshot`); the same models are above chance on sim raw
   windows (0.56–0.85) and on Cranfield (0.55–0.77), so it is MetroPT-specific.
   **Suggested wording:** "On MetroPT raw windows, distance- and reconstruction-based scores run
   *below* chance (AUROC 0.044–0.19). This is not a sign error — orientation was verified in
   each model and the same models are above chance elsewhere. The air-leak period is a
   low-variance, near-constant operating regime, so its windows sit closer to the training
   manifold than healthy duty-cycling windows do. Unsupervised distance scores are the wrong
   family for this failure mode; the residual and variance families (`lgbm_residual` 0.986,
   `moving_window_variance` 0.987) are the right one." Reported this way it is a finding, not an
   embarrassment — but it must not be left standing as an unexplained 0.044.
10. **`k_of_n_corroboration`'s member channels were changed at `da84c60`** for the sim rows.
    The bolded MetroPT `k_of_n` row is at `18c4478` (before that change). Confirm the MetroPT
    configuration was not affected, or rerun it.
