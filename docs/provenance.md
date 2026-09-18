# Data provenance

The active research benchmarks retain Ottawa axle bearing, MetroPT brake air supply, and
synthetic bearing/pneumatic data. Older Cranfield and synthetic Door sections below document
historical work; their datasets and benchmark results are no longer in this checkout. The
scored PS3 Door task remains part of the submission.

One section per real dataset adapter under `nebulax/adapters/`: where it comes from, its
licence, and exactly what we extract from it. See `docs/research/datasets.md` and
`docs/research/rail_phm.md` for the underlying literature review; this file is the short,
adapter-facing summary plus anything specific to how `nebulax` reads the files.

---

## Problem Statement 3 released datasets

These four datasets were supplied directly by the NEBULA X/LTA organisers under
`readingmaterials/problem_statement/PS3/02_Datasets/`. Their authoritative schemas, acquisition
notes and metrics are the adjacent organiser Info Kits in `03_References/`; no separate public
licence or permission to redistribute the raw files is stated in the supplied materials, so the
submission contains models/code/predictions but not raw data.

| dataset | released material and use | handling/provenance note |
|---|---|---|
| **PS3 Door** | `Door/Train.csv`, `Train_Segments_Answer.csv`, `Test.csv`; 50 Hz motor-current/voltage/back-EMF/position stream used for cycle segmentation and Normal/Abnormal-resistance classification | Original filenames and millisecond timestamps are preserved. Derived features/caches stay outside `readingmaterials/`; no raw file is rewritten. |
| **PS3 ACV** | six labelled `.xlsx` train cases and one unlabelled Test workbook; eight-car, 30 s ACV control/temperature telemetry used to rank refrigerant-leak location | Headers vary by case and are parsed from each workbook. Ranked car IDs echo their source spelling. Missing/unmapped channels are reported rather than silently borrowed across cases. |
| **PS3 Rail Corrugation** | 272 labelled and 68 unlabelled one-second CSVs at 10 kHz; speed pulse and 64 axle-box vibration/shock pairs used for Normal/Side I/Side II classification | File-level SHA-256/fingerprints are used only for duplicate grouping and split audit. Feature cache is derived; the organiser CSVs remain read-only. |
| **PS3 SHM** | 64 labelled and 16 unlabelled headerless single-stress-channel CSVs, 581,120 samples each; cumulative fatigue-damage regression | The Info Kit does not publish sample rate. Spectral features therefore use normalised cycles/sample; no physical-Hz interpretation is claimed. |

---

## The `meta_` convention (repo-wide, feature tables)

A feature-table column that **describes the sample but is not a legitimate model input** is
prefixed **`meta_`** (`nebulax.schema.METADATA_PREFIX`). That covers dataset class labels,
condition ids, severity stages, latent ground-truth counters and the bearing/test ids used only
for grouping. `nebulax.schema.feature_columns(df)` returns every column that is not in
`FEATURE_KEY_COLUMNS`, not in `LABEL_COLUMNS` and does not start with `meta_`, and the benchmark
builds `X` from exactly that list; group ids for leave-one-out splits are read from `meta_*` or
key columns, never from `X`. `meta_*` columns may carry **any** dtype — `validate_features`
accepts them as-is and never counts them as features.

The convention exists because the W1 audit (`docs/audits/w1_audit.json`) found three columns
that were silently training-visible targets:

| was | now | why it is not a feature |
| --- | --- | --- |
| `cranfield_class`, `level` | `meta_class`, `meta_level` | a relabelling of `fault_type` / a bijection of `severity` |
| `ottawa_class`, `ottawa_state`, `ottawa_state_label` | `meta_class`, `meta_state`, `meta_state_label` | the file name *is* the ground truth |
| `shock_count` (`nebulax.sim.door`) | `meta_shock_count` | written straight from the latent `ShockSeries`, which no fleet DCU observes |

The renames keep the columns (they are useful for analysis, stratification and grouping) and
only change what a model is allowed to see. Other identity columns moved with them:
`ottawa_bearing_id → meta_bearing_id`, `motion_profile → meta_motion_profile`,
`rep → meta_rep`, plus the new `meta_test_id` on Cranfield, so a leave-one-bearing-out or
leave-one-test-out split never has to re-parse a `run_id` string.

**The door simulator gained an observable replacement, not just a rename.**
`nebulax.sim.door.simulate` now also emits `shock_jump_count_est` — a running count of shock-like
**steps** in the per-cycle statistics a DCU really has (`pos_close_max`, `i_peak`,
`pos_err_max`), from `nebulax.sim.door.estimate_shock_jumps`: each series is median-filtered
over the trailing 9 cycles (so a one-cycle passenger obstruction cannot read as damage), then
compared with the median of the trailing 50 cycles in units of that window's robust sigma
(`IQR/1.349`, floored at 0.1 % of the median); `|z| > 6` on any series counts, rising edge only.
Every window is **trailing**, so the column carries no look-ahead. On the day-long shock
fixtures it tracks the truth (cumulative correlation ~0.9) and resolves roughly half the
shocks — once the door is jamming on the walked-in stop, the reversal scatter is the size of one
shock's step. That gap is the honest cost of not having the latent process; the two columns are
correlated, never equal, and only `shock_jump_count_est` reaches the model.

The simulators' other feature tables were audited for the same defect and are clean: the
pneumatic cycle features come from the sensed channels and the compressor state machine (which
is itself observable as `COMP`/`DV_eletric`/`Towers`/`Pressure_switch`), and the bearing window
features come from the sensed `T_box`/vibration channels, their peer/thermal residuals and the
alarm rules computed on them. `severity`, `fault_type`, `rul_s`, `is_faulty` and
`alarm_window_3d` stay **label** columns (`LABEL_COLUMNS`), which `feature_columns()` already
excludes.

---

## Cranfield linear actuator (door proxy)

- **Citation / DOI**: C. Ruiz-Carcel and A. Starr, *"Data set for 'Data-based Detection and
  Diagnosis of Faults in Linear Actuators'"*, Through-Life Engineering Services Institute,
  Cranfield University, Cranfield Online Research Data (CORD),
  https://doi.org/10.17862/cranfield.rd.5097649.
- **Licence**: **UNVERIFIED.** No licence statement appears anywhere in the delivered files —
  not in `Data description.pdf`, not in any `.mat` header, and there is no sidecar. Only the
  CORD landing page would carry it, and that page is still behind a Cloudflare managed JS
  challenge that no HTTP client can pass. **Do not redistribute the raw files** until someone
  opens the DOI in a real browser and reads the licence there. (The `.mat` headers say only
  `MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Wed Feb 28 ... 2018`.)
- **Access status (2026-09-15)**: the release is **present** — 13 `.mat` files plus
  `Data description.pdf` under `data/raw/cranfield/`, obtained by hand because
  `scripts/download_data.py --dataset cranfield` still cannot reach the DOI (Cloudflare
  challenge on the DSpace instance, AWS WAF on the `cranfield.figshare.com` mirror, and the
  figshare public API no longer indexes the record; the HTTP evidence is kept in the
  download script). The script now detects a manual drop-in, checksums it and writes
  `data/raw/cranfield/MANIFEST.json` with `status: "ok"`, a `sha256` and size per file, and
  **no network request at all** — that manifest is the checksum record for these files.
- **The real layout, verified against the files and against the release's own PDF** (which is
  the primary source for everything below — read it with `pdftotext -layout`):
  - 13 `.mat` files, one per condition: `Normal.mat`, `LackLubrication{1,2}.mat`,
    `Backlash{1,2}.mat`, `Spalling{1..8}.mat`.
  - Each holds **60 matrices** — 2 motion profiles × 3 loads × 10 repetitions —
    **except `Backlash1.mat`, which holds 59**: `backtrap1st40kg` has only 9 repetitions
    (`backtrap1st40kg2` is absent from the release). 779 test recordings in total.
  - Every matrix is `(2000, 3) float64` = **80 s at 25 Hz** (PDF §2 "All the data was acquired
    at 25 Hz"; §4 "0.04 s intervals"). The sample rate is now **measured, not assumed**.
  - Columns are `[position set point (mm), position error (mm), motor current (A)]` (PDF §4 and
    Fig. 6). The **error** is set point minus measurement — there is no measured-position
    column, and **no voltage column at all**.
  - Variable names encode `<class><profile><level><load><rep>` (PDF Fig. 8): class in
    `train` (normal), `back`, `lub`, `point` (spalling — the PDF's Fig. 6 legend calls it
    "point defect"); profile `sin` or `trap`; level `1st`..`8th`, absent for `train`; load
    `20kg`, `40kg` or `neg40kg` (kgf, `neg` = opposing); rep `1..10`. E.g. `trainsin20kg3`,
    `lubtrap2ndneg40kg7`, `pointtrap8thneg40kg10`.
  - **What changed against the old reconstruction**: the previously documented layout
    (`<fault_class>/<test>.csv`, fuzzy column-name matching, a 4-channel
    reference/position/error/current file, an optional `cranfield_labels.csv` override) came
    from secondary sources and was **wrong in every structural detail**. It is gone;
    `nebulax/adapters/cranfield.py` now parses the real layout only.
- **Rig and faults (PDF §2-3)**: ball screw with anti-backlash ball nut **RM1605-C7, 5 mm
  lead**; **Nema 34 stepper** motor, 4.6 N·m holding torque; 120 mm stroke; ±40 kgf external
  load applied by a second actuator through a load cell; position by a Vishay REC 115L linear
  potentiometer, current by a Honeywell CSLA2CD Hall-effect sensor; ~30 min of warm-up before
  every test. Trapezoidal profile = 120 mm in 5 s with 3 s waits at both ends, sinusoidal =
  6 s with 2 s waits; the full out-and-back sequence is repeated **5 times per recording**, so
  each matrix holds 10 strokes. Faults: lack of lubrication stage 1 = degreased, stage 2 = nut
  seals bolted tighter to create friction; spalling **8 stages**, a 1 mm raceway defect grown
  to 2/3/4 mm, then replicated into neighbouring channels and finally through the sidewall
  between them; backlash 2 stages, the original 3.15 mm balls replaced by 3.0 mm then 2.5 mm.
- **THE MOTOR IS A STEPPER — this bounds every current-derived number.**
  `nebulax/sim/door.py` models a **PMDC** drive, where quasi-static current is proportional to
  load force. A chopper-regulated stepper holds a commanded phase current instead, and the
  measurement shows it: the healthy rig draws **0.40 A standing still against 0.86 A while
  travelling**, so roughly half the cruise current carries no load information. Consequences,
  all recorded in `docs/parameters.md`: a seeded lubrication fault moves the mean current by
  only 17-25 %; the mechanical least-squares fit
  `i = (m_eff/c_i)·a + (F_c0/c_i)·sgn(v) + (b0/c_i)·v` scores **R² = 0.03** and is reported as
  *rejected*, not fitted; and every faulty/healthy current ratio is treated as a **proxy**, so
  gains read off a PMDC inversion curve with a mean-current ratio are **lower bounds**.
- **What `nebulax.adapters.cranfield.load()` extracts:**
  - Discovery is by the release's own 13 filenames (case-insensitively). Any other `.mat` in
    the directory is reported in `meta["unexpected_variables"]`/skipped rather than guessed at,
    because its condition and stage would be unknown.
  - **A missing release file is a logged `WARNING`, never an error**: `load()` parses whatever
    is present, lists the rest in `meta["missing_files"]`, and adds a line to
    `meta["unverified"]`. It raises `FileNotFoundError` only for a `raw_dir` that does not
    exist, or one holding neither an expected `.mat` nor a `MANIFEST.json` explaining their
    absence; a manifest-only directory still returns an honest, schema-valid, **empty**
    `Dataset` so the validate CLI can report the situation instead of crashing.
  - **One run per (file, `.mat` variable)** — one 80 s recording = one `run_id`
    (`cranfield_<Stem>_<variable>`), 779 runs. Channels emitted: `pos_ref` (mm → m), `pos`
    (reconstructed as `pos_ref − pos_err`, since the rig logs the error), `current` (A) and
    `vel` (**derived**, `np.gradient(pos)·25 Hz`). 6.23 M telemetry rows.
  - **One feature row per stroke** (extend or retract half-cycle of the set point, with the
    2-3 s end waits dropped): 7 790 rows, 10 per recording. Features are the per-stroke signal
    statistics `duration_s, n_samples, i_mean, i_rms, i_peak, i_start_peak,
    pos_ref_start/end/range, direction, v_ref_mps, pos_err_max, pos_err_rms` plus **`load_kg`**
    (kgf, negative = opposing). The test matrix rides along as **metadata** under the reserved
    `meta_` prefix — `meta_class`, `meta_level`, `meta_test_id` (`<Stem>:<variable>`),
    `meta_motion_profile`, `meta_rep` — and `nebulax.schema.feature_columns()` keeps every
    `meta_*` column out of the model's `X`, because `meta_level` is a bijection of the
    `severity` label and `meta_class` of `fault_type`: as plain feature columns they were a
    **leakage path** (found by the W1 audit, `docs/audits/w1_audit.json`), and a classifier
    reading them would only be reading the file name back.
  - **Why `load_kg` stays a feature**: the opposing/aiding load is an operating condition a
    real door controller knows at inference time, it is applied across *every* class of the
    test matrix (20 kg, 40 kg, −40 kg for all 13 conditions), so it confounds the signal rather
    than encoding the target — exactly the confounder a load-aware model must handle.
    `meta_motion_profile` is an operating condition too, but it is a string describing the test
    matrix (and the calibration stratifies by it), so it is carried as metadata.
  - Class mapping onto `nebulax.schema.FAULT_TYPES["door"]`, per the plan's door↔Cranfield
    note: `back → backlash`, `lub → friction` (lack of lubrication *is* a friction fault —
    rail_phm 4.1 groups them), `point`/spalling `→ misalignment` (the spalling ripple is
    carried through the simulator's position-periodic term), `train → healthy`.
  - Ordinal **ML label**: `0.0` for healthy, else `min(1.0, 0.2 + 0.2·(level − 1))`. Spalling
    has **8** stages, so stages 5-8 all saturate at `1.0` on this scale — deliberate, and
    *not* the map the physics calibration uses: `scripts/calibrate_door.py` places the spalling
    stages linearly at `s = level/8`, which keeps the plan's anchors (stage 4 → `s = 0.5`,
    stage 8 → `s = 1.0`). The two maps must not be unified.
  - One fault-log row per run, `shape="step"` (a seeded bench fixture holds its severity for
    the whole 80 s test; there is no run-to-failure trajectory), `t_failure` /
    `t_functional_failure` unset, `params_json` carrying the source file, variable, class,
    level, profile, load and repetition.
  - Timestamps are **synthetic** (the source has no wall clock): each run gets its own
    1-hour-spaced slot from `2000-01-01T00:00:00Z`, recorded in `meta`, purely to keep runs
    from overlapping — no calendar meaning. `component_id` is fixed to `door_L1`: a generic
    bench rig, not a specific fleet door.
- **Tests**: `tests/test_adapter_cranfield.py` builds tiny `.mat` fixtures **in the real
  layout** (real filenames, real `<class><profile><level><load><rep>` variable names, real
  `[set point mm, error mm, current A]` columns) and covers variable parsing including
  `neg40kg` and the level-less `train` case, mm→m conversion and the `pos = pos_ref − pos_err`
  reconstruction, stroke segmentation, discovery, the missing-file WARNING path, both
  `FileNotFoundError` cases, the manifest-only fallback, the metadata columns, the severity
  saturation at spalling stage 5, and the validate CLI. Two tests run against the real
  `data/raw/cranfield/` and assert its actual counts (13 files, 779 runs, 7 790 strokes,
  60/119/120/480 per class, 120 mm stroke) — they skip if the release is not in the checkout.
  `python -m nebulax.adapters.validate --source cranfield --raw data/raw/cranfield` exits 0.

---

## MetroPT-3 (pneumatic / APU air-supply proxy)

- **URL / DOI**: https://doi.org/10.24432/C5VW3R -- "MetroPT-3 (Air Compressor)", UCI Machine
  Learning Repository, dataset id 791. Downloaded via the static zip
  `https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip` (the `ucimlrepo` package's
  `fetch_ucirepo(id=791)` raises `DatasetNotFoundError` for this id -- it is a raw time-series
  release, not a registered tabular import -- so the static zip is the working path).
- **Licence**: CC BY 4.0 (stated on the UCI dataset page), citation:
  `Veloso, B., Ribeiro, R.P., Gama, J., Pereira, P.M. (2022). MetroPT-3 Dataset. UCI Machine
  Learning Repository. https://doi.org/10.24432/C5VW3R`.
  Cite alongside `docs/research/rail_phm.md` [R87][R88] for the physical-constant provenance.
- **File**: one CSV, `MetroPT3(AirCompressor).csv`, 1,516,948 rows, 2020-02-01 -> 2020-09-01. 7
  analogue channels (`TP2, TP3, H1, DV_pressure, Reservoirs, Oil_temperature, Motor_current`) + 8
  digital (`COMP, DV_eletric, Towers, MPG, LPS, Pressure_switch, Oil_level, Caudal_impulses`) --
  matches `nebulax.schema.METROPT3_SIGNALS` exactly, verbatim names including the `DV_eletric`
  typo. **No `Flowmeter`, no GPS** -- those are MetroPT-1/2 only [rail_phm 2.1]; the adapter
  correctly emits only the 15 columns this release actually has (`validate --strict` passes with
  zero extra/missing columns).
- **Sampling is NOT a clean 1 Hz** despite the dataset card's headline rate: the row-to-row
  timestamp delta is 9-10 s for >99.9% of rows (nominal ~10 s cadence), with 331 gaps over 60 s
  (largest ~2 days) presumably from acquisition downtime. `nebulax.adapters.metropt3` caps any
  single inter-sample duration at 60 s before summing it into a compressor-cycle duration, so a
  multi-day logger outage cannot masquerade as multi-day compressor-off time; the cap count is
  recorded in `meta["n_gaps_over_60s"]`.
- **Units**: no conversion needed. `TP2/TP3/H1/Reservoirs/DV_pressure` are already bar,
  `Oil_temperature` already degC, `Motor_current` already A -- all match
  `nebulax.schema.SIGNAL_SPECS` units for the pneumatic subsystem directly.
- **Timestamp convention**: the source states no timezone. Treated as UTC with no shift, so the
  telemetry and the hard-coded fault-report timestamps below (same clock) stay mutually aligned
  regardless of the true offset (Portugal, so most likely WET/WEST, UTC+0/+1 seasonally) -- see
  the docstring of `nebulax/adapters/metropt3.py` for the reasoning.
- **Component mapping**: one APU per file -> `component_id="apu_1"`, `car=0` (unit-level,
  matching `nebulax.schema.APU_COMPONENT_IDS`), `train_id="porto_apu"`, `run_id="metropt3"`.
- **Compressor state segmentation (ours, not from the paper)**: `Motor_current` is trimodal in
  this file -- ~0 A off (54.6% of rows), ~3.4-4.0 A idling/unloaded (29.4%), ~5.3-6.5 A loaded
  (15.9%, matching `DV_eletric`'s own 16.1% "working" mean almost exactly). The adapter classifies
  each sample as off / unloaded / loaded by thresholding at 2.0 A and 5.0 A
  (`meta["compressor_state_thresholds"]`) and defines one **compressor cycle** as
  `[start of an off->active transition, start of the next such transition)`. 10,395 cycles fall
  out of the full file this way.
- **What `nebulax.adapters.metropt3.load()` extracts:**
  - `long`: all 15 signals verbatim, no resampling, no unit conversion (~22.75M long rows for the
    full file; ~450 MB is the size of the resulting DataFrames alone -- the measured peak resident memory of a full `load()` + validation pass is **~5.85 GiB** (Codex audit, 15 Sep 2026: 6,129,648 KB max RSS), so budget 8 GB for the MetroPT-3 adapter, not 0.5 GB -- no
    chunking was needed for this file size).
  - `features`: **the compressor-cycle table only** (plan: "compressor-cycle table keyed on COMP
    transitions"), one row per cycle above, columns `t_loaded, t_unloaded, t_off, duty_ratio,
    idle_run_ratio` (the last two per rail_phm 4.2/2.3b's DOE-cited leak signature),
    `dP_dt_loaded, dP_dt_off` (reservoir pressure slope during the loaded/off phases),
    `I_loaded_mean, I_start_peak, T_oil_max, TP2_minus_TP3_mean, H1_loaded_mean, tower_switches,
    purge_count, LPS_any, hour_of_day, transition_frac`, plus `is_faulty, fault_type,
    alarm_window_3d` joined from the fault log below (a cycle is `alarm_window_3d=True` if it
    overlaps `[t_onset - 3 days, t_failure]` of any episode). The plan's separate
    **10-second-aggregate** window-stats table is deliberately **not** produced here: it is
    generic fixed-window aggregation over `long` with no MetroPT-specific logic (mean/max per
    analogue channel, duty fraction per digital channel), so it is left to `nebulax.features` /
    built by the `window_stats` input_kind models directly from `to_wide(long, "pneumatic")` at
    bench time -- duplicating that here would just be `features/windows.py` copy-pasted into an
    adapter. Noted in `meta["features_note"]`.
    - **`I_loaded_mean` / `H1_loaded_mean` / `TP2_minus_TP3_mean` are averaged over `state == 2`
      (loaded) samples only** -- the same 3-way off/unloaded/loaded cut on `Motor_current` that
      `scripts/calibrate_pneumatic.py` uses (`_classify_state`; off < 2 A, loaded >= 5 A).
      **Fixed 15 Sep 2026**: `TP2_minus_TP3_mean` previously averaged `TP2 - TP3` over the
      *whole* cycle (~-7.9 bar on real data -- `TP2` reads ~0 while off/unloaded, which is most
      of a cycle), a different quantity from `nebulax.sim.pneumatic`'s identically-named column
      (loaded-only, ~+0.55 bar simulated / +0.304 bar measured, docs/parameters.md pneumatic
      section 5). `H1_loaded_mean` was already loaded-only and did not need this fix.
      **Also fixed 15 Sep 2026 (same-day follow-up, after an audit ran the adapter and the
      simulator's own `_cycle_features` side by side and found two more mismatches the first
      pass missed):** `I_loaded_mean` now also drops the first 4 samples of the cycle before
      averaging, matching `nebulax.sim.pneumatic._cycle_features`'s own starting-current-transient
      exclusion (previously a compressor inrush spike at the start of a loaded run could pull the
      whole cycle's `I_loaded_mean` up); and `dP_dt_loaded` / `dP_dt_off` now average each
      sample's own `d(Reservoirs)/dt` against the *preceding* sample
      (`nebulax.adapters.metropt3._backward_diff_rate`) instead of the phase's
      `(last - first) / duration`, which silently dropped the transition-in sample's pressure
      delta (a fencepost -- a run of *n* samples has *n - 1* internal diffs) and so understated
      the rate; the two formulas agree for a uniform-cadence trace but not for this file's
      irregular ~10 s cadence, which is why the mismatch was invisible until the cross-check ran
      against real `_cycle_features` output rather than a hand-picked helper. The shared
      definition table (which columns are loaded-only / off-only / whole-cycle, and how) lives in
      `nebulax/adapters/metropt3.py::_build_cycle_features`'s docstring, is cross-checked in
      `tests/test_adapter_metropt3.py::test_loaded_off_features_match_simulator` -- which calls
      `nebulax.sim.pneumatic._cycle_features` itself on a shared synthetic 1 Hz, 3-cycle trace
      (with a starting-current inrush) and compares every column both sides claim in common, not
      just `TP2_minus_TP3_mean` / `H1_loaded_mean` on a hand-picked helper as the first pass did.
      **Correction (this fix run, after an audit found the earlier claim false):** this table is
      **not** echoed in `nebulax/sim/pneumatic.py::_cycle_features`'s own docstring -- that file
      is out of scope for this fix task (it is not in the declared file list and is untouched;
      its docstring documents only `TP2_minus_TP3_mean` / `H1_loaded_mean` semantics, not the
      full column table). `nebulax/adapters/metropt3.py::_build_cycle_features`'s docstring is
      the single source of truth for the table, kept honest by being cross-checked against the
      simulator's real output rather than by a claimed (and, until now, incorrect) mirror.
      **Also fixed this run (latent, never triggers on the real file):** `_backward_diff_rate`'s
      `rate[0]` is now `0.0`, not `NaN` -- matching the simulator's `np.diff(P, prepend=P[0])`
      convention (`dP[0] = 0`) exactly, instead of silently excluding a record's first sample
      from the loaded/off mean on the adapter side while the simulator includes it as a zero. On
      the real file this never fires (row 0 is OFF at 0.04 A, and only a record whose very first
      sample is already loaded/off-at-start would see it); a regression test
      (`test_backward_diff_rate_first_sample_matches_simulator_prepend_convention` and
      `test_loaded_off_features_match_simulator_when_record_starts_mid_cycle`) now covers it.
      `hour_of_day` differs by construction (the adapter truncates to whole minutes, the
      simulator keeps fractional seconds) and is deliberately **not** claimed identical in either
      docstring table or asserted in the test.
      **A second follow-up fix, same day:** the same-day cross-check test's own synthetic trace
      happened to put the current's global maximum in the first two samples of every cycle and
      its one `Towers` flip / purge pulse away from every cycle boundary, so it could not see two
      further mismatches an independent audit trace exposed: `I_start_peak` restricted the max to
      the cycle's first 3 samples (`nebulax.sim.pneumatic._cycle_features` takes `fmax` over the
      *whole* cycle -- on the real file the two disagreed on 97.9 % of the 10,395 cycles, median
      0.37 A, max 2.5 A), and `tower_switches` / `purge_count` dropped a flip / rising edge
      landing exactly on a cycle's own first sample (the simulator's plain `add.reduceat` sum
      counts it -- 7.8 % of the real file's 64,580 `Towers` flips land exactly there). Both are
      now plain whole-cycle reductions with no first-sample exclusion, matching the simulator
      exactly (verified: 0 mismatching cycles on the full real file for `I_start_peak`, and the
      adapter's summed `tower_switches` now equals the file's true total flip count, 64,580). The
      docstring table in `_build_cycle_features` now lists all three columns; the trace in
      `tests/test_adapter_metropt3.py::_multi_cycle_trace` now carries a later, higher current
      spike and a boundary-coincident `Towers` flip / purge pulse in its second cycle so the
      cross-check (and two new hand-computed assertions on that cycle) actually exercises this.
    - **`transition_frac`** (added 15 Sep 2026, `docs/research/rail_phm.md` "Detector design
      additions": "Emit an `is_transition` mask and exclude +/-5 s around `Towers` flips and
      `COMP` load/offload edges from point scoring") -- the fraction of a cycle's samples within
      `nebulax.adapters.metropt3._TRANSITION_WINDOW_S` (**5.0 s, rail_phm's own number, CONFIRMED
      not UNVERIFIED**) of a state change (off/unloaded/loaded), a `Towers` flip, or a `COMP`
      edge, from `nebulax.adapters.metropt3.transition_mask()`.
      **Follow-up fix, same run (audit found `COMP` was never read):** the mask used to be built
      only from `Motor_current`-derived state changes and `Towers` flips, on the theory those
      subsume `COMP`'s own load/offload edges -- they do not exactly: 703 of the real file's
      33,846 `COMP` digital edges (2.08 %) lag the nearest `Motor_current`/`Towers` event by
      one 10 s sample. `transition_mask()` now also reads `COMP` directly (when present; it is
      optional so the function still works on a caller-built frame without it) and ORs its edges
      in, giving exact "COMP load/offload edges" coverage as rail_phm.md names it, at a cost of
      +0.05 pp masked samples on the real file (6.164 % -> 6.210 %).
      **Documented cadence caveat (not changed, since 5 s is rail_phm's own literal number):**
      this file's native cadence is ~10 s, so a +/-5 s window is below one sample period and in
      practice flags essentially only the edge sample itself, not a temporal neighbourhood either
      side of it (measured: the preceding sample is flagged only ~21 % of the time, the next one
      ~40 %). Both `transition_mask()`'s docstring and `_TRANSITION_WINDOW_S`'s comment now say
      so explicitly, and `meta["transition_window_note"]` carries it into `load()`'s output; a
      caller wanting an actual multi-sample exclusion zone at this cadence should pass a larger
      `window_s` (`window_s=15` masks ~15.5 % of the real file).
      **Scope note -- deferred, NOT self-accepted (a prior audit round flagged an earlier
      version of this note, then headed "Acceptance note", for reading as the adapter agent
      unilaterally accepting its own deviation; retitled and reworded here for that reason --
      the substance below is unchanged, and closing this gap is still an open `must_fix`, not a
      resolved item):** the task asked for an `is_transition` *column on a 10-second aggregate
      table*. No such aggregate table exists anywhere in this codebase (`nebulax.features` has
      no table-shaped window-stats builder yet, only the array-level `window_stats` extractor in
      `nebulax/features/stats.py`), and building one from scratch is out of scope for an adapter
      fix task that names only `nebulax/adapters/metropt3.py` (+ its tests, this file, and one
      `docs/parameters.md` bullet) as the files to touch -- `nebulax/features/*` is owned by a
      different, parallel task this run and is explicitly off-limits here. What is shipped
      instead is the per-sample `transition_mask()` function (the exact ingredient such a table
      would need for its `is_transition` column, now including `COMP` edges, see above) plus a
      per-cycle `transition_frac` column on the table this adapter *does* emit, both new. The
      per-sample `is_transition` boolean itself is **not** materialised as a `long` column:
      `nebulax.schema.SIGNALS["pneumatic"]` is a closed vocabulary of physical channels and this
      fix does not touch `schema.py`. Whoever builds the real 10-second-aggregate /
      `window_stats` table (from `to_wide(long, "pneumatic")` at bench time) should call
      `transition_mask()` on that wide frame to get its `is_transition` column -- that table
      itself remains unbuilt, and is the one piece of the (b) deliverable this run does not
      ship, by scope necessity rather than oversight. `nebulax.sim.pneumatic` emits no separate
      aggregate table either (only `long` + the per-cycle `features` table, same as this
      adapter), so nothing there needed a matching column; the mask there would be computed the
      same way, downstream, from `COMP`/the state machine's own state array -- left undone (and
      `nebulax/sim/pneumatic.py` untouched) because no aggregate table exists yet to hang it on.
      **Resolution path (not this adapter's to take, per the audit that reviewed this fix
      round):** either the `nebulax/features` workstream owner adds the plan's MetroPT-3
      10-second aggregate builder and wires `nebulax.adapters.metropt3.transition_mask()` in as
      its `is_transition` column, or the orchestrator/user explicitly accepts the deferral and
      records that acceptance themselves (outside this file's adapter-authored provenance
      narrative) -- this note only describes the gap, it does not close it.
      **Accepted deferral (15 Sep 2026, orchestrator decision):** the gap above is accepted, not
      closed here either. The 10-s aggregate table (analogue mean/max per channel, digital duty
      fraction, and `is_transition` from `transition_mask()`) is built by the benchmark data
      loader `nebulax/bench/data.py` in the W2 stage, not by this adapter -- `nebulax/bench/`
      (`base.py`, `registry.py`) exists but `data.py` does not yet, so this is a forward
      assignment of ownership, not a claim that the table is built. When it is, it should call
      `nebulax.adapters.metropt3.transition_mask()` on `to_wide(long, "pneumatic")` for its
      `is_transition` column exactly as the "Resolution path" above already specifies.
  - **MetroPT-3 is an anomaly-detection dataset, not a supervised one, on the compressor-cycle
    table above.** Of the 10,395 cycles `load()` emits, only **6** carry `is_faulty=True` --
    every UCI-reported air-leak episode after the first is long enough (days, not the ~109 s
    median `t_loaded`) that the leak collapses its own compressor cycle into one multi-day
    cycle rather than producing many short faulty ones, so per-cycle supervised counts are
    structurally tiny and not a usable positive class (6 positives out of 10,395 rows is not a
    classification problem, it is six events to detect). Models trained against this adapter's
    `features` table must be scored on events/alarm windows (`fault_log`, `alarm_window_3d`) --
    the anomaly-detection framing the rest of this project already uses -- never on `is_faulty`
    as a per-cycle supervised label.
  - `fault_log`: the 4 air-leak episodes transcribed **verbatim** from the UCI page's free-text
    `additional_info.summary` failure-report table (`data/raw/metropt3/MANIFEST.json`):
    2020-04-18 00:00-23:59, 2020-05-29 23:30 -> 05-30 06:00, 2020-06-05 10:00 -> 06-07 14:30,
    2020-07-15 14:30-19:00, all `fault_type="air_leak"` -- **MetroPT-3 has air-leak failures only**,
    no oil leak (that is MetroPT-1's F3 / MetroPT-2's second episode) [rail_phm 2.2, 4.2].
    `gamma`/`shape`/`t_functional_failure` are left unset (NaN/None) -- the report gives only a
    window, not a degradation trajectory or a functional-failure criterion.
    **`meta["fault_log_note"]` flags these dates as UCI's own report text, not independently
    re-verified against the underlying paper** -- re-check before quoting as ground truth, per
    the task's own instruction to do so.
  - `events`: `comp_load`/`comp_off` at compressor start/stop, `tower_switch` on `Towers`
    transitions, `purge` on `Pressure_switch` rising edges, `lps` on `LPS` rising edges, `oil_low`
    on `Oil_level` rising edges -- all in `nebulax.schema.EVENT_TYPES` (~105k rows on the full
    file, dominated by `tower_switch`).
- **Tests**: `tests/test_adapter_metropt3.py` copies the first 6,000 real rows (~40 compressor
  cycles) into a temp `raw_dir` so the whole module runs in under a second, and skips entirely if
  `data/raw/metropt3/MetroPT3(AirCompressor).csv` is absent. Covers both `FileNotFoundError` paths,
  full schema validation (including `--strict`), the verbatim signal set, the hard-coded fault log,
  the cycle-feature table's shape/ranges, and the event vocabulary. The adapter was also run and
  validated end-to-end against the full real file (`python -m nebulax.adapters.validate --source
  metropt3 --raw data/raw/metropt3`, ~45 s, 22.75M telemetry rows, 10,395 cycles, `OK`) as part of
  building it, but that full run is not part of the automated test suite.

---

## University of Ottawa UORED-VAFCLS (bearing / axle-box proxy)

- **URL / DOI**: https://data.mendeley.com/datasets/y2px5tg92h/5 -- "UORED-VAFCLS" (University of
  Ottawa Rolling Element Dataset - Variable Angular Fault Classification and Localization Set),
  Mendeley Data, DOI `10.17632/y2px5tg92h.5` (version 5).
- **Licence**: CC BY 4.0 (stated on the Mendeley record).
- **Citation**: cite the Mendeley DOI above; cite `docs/research/datasets.md` [R119] alongside
  it -- this is the *constant-speed ablation* member of the Ottawa bearing family, demoted from
  primary in favour of the encoder-equipped variable-speed set [R118] (DOI
  `10.17632/v43hmbwxpm.2`, not downloaded to this repo -- see the family-attribution note below).
- **Files**: 60 raw recordings under `data/raw/ottawa/`, named `<Class>_<bearingId>_<state>.csv`
  (`.mat` copies also downloaded but not read - byte-identical to the `.csv` per the download
  manifest, so reading both would double the work for nothing). `Class in {H, I, O, B, C}`
  (healthy / inner race / outer race / ball / cage), `state in {0, 1, 2}` (healthy / developing /
  faulty). Every one of the 20 physical bearings (ids 1-20) has one healthy recording
  (`H_<id>_0`); ids 1-5/6-10/11-15/16-20 additionally get inner-race/outer-race/ball/cage
  recordings at both fault states -- 20 healthy + 20 x 2 faulty = 60. Each file: 420,000 rows,
  5 columns, one 10 s recording at **42 kHz**.
- **This is UORED-VAFCLS (`y2px5tg92h`), the constant-speed ablation member of the Ottawa
  bearing family -- NOT the encoder-equipped variable-speed set `rail_phm.md` SS4.3.4 promotes
  to primary.** `docs/research/rail_phm.md` SS4.3.4's description of "a 1024-CPR encoder channel
  at 200 kHz" with speed that "sweeps 13.7-28.9 Hz shaft speed WITHIN a single 10 s record" is
  about a **different, separate** Ottawa dataset -- "uOttawa bearing vibration under
  time-varying rotational speed", DOI `10.17632/v43hmbwxpm.2` [R118] -- which `rail_phm.md`
  explicitly promotes to primary and `configs/model_ladder.yaml`'s `bearing.datasets` list
  carries as its own id (`ottawa_variable_speed`), separate from this dataset's id
  (`ottawa_uored_vafcls`). `docs/research/datasets.md` calls UORED-VAFCLS the "constant-speed
  ablation" [R119] -- `rail_phm.md` is internally consistent about this, not self-contradictory.
  **`v43hmbwxpm` (`ottawa_variable_speed`) has not been downloaded to this repo** -- there is no
  `data/raw/ottawa_variable_speed/` directory and no adapter for it; flagged here for whoever
  owns data acquisition next, since the bearing subsystem's promoted-primary dataset is
  currently missing entirely, not just under-documented.
  Correctly attributed to *this* dataset (UORED-VAFCLS, the one actually on disk): the columns
  present are `Accelerometer(g), Acoustic(raw mic), Speed(nominal RPM), Load(nominal load
  units), Temperature Difference(degC)` -- no encoder column -- and `Speed`/`Load` are **not
  per-sample**: only row 0 carries the nominal value, the remaining 419,999 rows are exactly
  0.0 (matches the download agent's own MANIFEST note). Speed is therefore a **per-recording
  constant** that varies *across* the 60 files (the "variable-speed" part of the family name,
  confusingly shared with `v43hmbwxpm`), not *within* one -- every window in this adapter is
  necessarily "tacholess" for this dataset; the tacho-vs-tacholess ablation
  `configs/model_ladder.yaml` describes needs `v43hmbwxpm`, not this set.
  Also correctly attributed to UORED-VAFCLS: **a temperature channel does exist**
  (`Temperature Difference`, degC) -- consistent with `rail_phm.md` SS4.3.3's "no temperature
  channel" note being about `v43hmbwxpm`, not this set. Its reference point (difference from
  what?) is undocumented in anything reachable offline this session, so it is carried into the
  long table as `T_box` with that caveat -- a coarse proxy, **not** a calibrated absolute
  axle-box temperature; the real bearing-thermal ground truth still comes from [R149]/[R150] via
  `nebulax/sim/bearing.py`, not from this dataset.
- **Accelerometer units: g's in the raw file, converted to m/s2 in the adapter.** The
  `Accelerometer` column is in g's (`data/raw/ottawa/MANIFEST.json`'s own inspection note:
  "Accelerometer (g)"). `nebulax.adapters.ottawa.load()` multiplies by `G_MS2 = 9.80665`
  immediately on read, before any feature or long-table signal is derived, so `vib_rms`/
  `vib_bpfo` land in the same unit `nebulax.schema.SIGNAL_SPECS` declares (`unit='m/s2'`) and
  that `nebulax.sim.bearing`'s own calibration uses for the identical signal names. Absolute
  magnitudes still differ sharply from the simulator's healthy floor (~0.6-2.6 m/s2) even after
  the unit fix -- this lab rig's accelerometer is mounted directly on a small bearing housing at
  42 kHz broadband, not on a 12 t axle box through a heavier structural path, so a real-vs-
  simulated magnitude gap is expected on top of the unit conversion, not a remaining unit bug.
- **`T_box` uses a robust, outlier-rejecting window mean, not a plain mean.** A handful of raw
  files (`C_16_1.csv`, `C_16_2.csv`, `C_17_1.csv`, `C_17_2.csv`) contain isolated single-sample
  sensor-dropout glitches in `Temperature Difference` reaching magnitudes up to `-8e11` degC,
  which would otherwise poison an entire window's mean into a physically impossible value.
  `nebulax.adapters.ottawa._robust_temp_mean` rejects samples outside a physically plausible
  range (`_TEMP_PLAUSIBLE_C = (-50, 150)` degC) before averaging, falling back to the window's
  raw median for the rare window where every sample is rejected; the total rejected-sample count
  is recorded in `meta["temp_glitches_rejected"]`.
- **Bearing geometry is an assumption, stated as one.** Envelope-spectrum harmonics (BPFO/BPFI/
  BSF/FTF) need rolling-element geometry that UORED-VAFCLS's own descriptor does not confirm
  offline. `nebulax.adapters.ottawa.GEOMETRY` assumes an **ER16K-class deep-groove ball bearing**
  (`n_elements=9, ball_diameter_m=7.94e-3, pitch_diameter_m=39.0e-3`) -- the same geometry already
  used in `tests/test_features_vibration.py`, reused here for one consistent assumption across the
  repo rather than a second, different guess. Treat the harmonic *frequency assignment*
  (`env_bpfo_*` etc.) as approximate; the underlying envelope statistics themselves
  (`sk_band_fc/bw/max`, `rms/kurtosis/crest`) do not depend on this assumption.
- **What `nebulax.adapters.ottawa.load()` extracts:**
  - `long`: five engineered channels at a synthetic **1 Hz** cadence (matching
    `nebulax.schema.SIGNAL_SPECS`'s documented `fs_hz` for `bearing`) -- `vib_rms, vib_kurt,
    vib_crest` (time-domain, via `nebulax.features.stats.window_stats` over each 1 s /
    42,000-sample tile of `Accelerometer`, **AC-coupled** -- `ac_couple=True`, i.e. computed on
    `x - mean(x)`; `vib_kurt` is central by definition and unaffected -- because 21 of the 60
    records carry a DC bias larger than their own AC RMS, see below), `vib_bpfo` (the summed
    BPFO-harmonic envelope-
    spectrum energy from `nebulax.features.vibration.envelope_spectrum_feats` under the assumed
    geometry above), and `T_box` (the window-mean `Temperature Difference`, with the caveat
    above). 10 rows/signal/file x 5 signals x 60 files = 3,000 rows. `component_id` cycles
    through `nebulax.schema.AXLEBOX_COMPONENT_IDS` as `(bearing_id - 1) % 8` (the registry only
    names 8 axle-box slots per train and this rig has 20 physical units); the real identity is
    never lost, it lives in `train_id` (`bearing_01`..`bearing_20`, **the CV group -- bearing-wise
    partitioning is required per rail_phm.md 3.4/4.3.4 to avoid the leakage that segment-wise
    splits are documented to cause on this exact dataset**) and `run_id` (the file stem).
  - **DC coupling / the accelerometer bias.** 21 of the 60 records carry a window mean larger
    than their own AC RMS (worst `H_2_0`: mean 1606 m/s2 against an AC RMS of 56). A constant
    offset is a sensor bias, not vibration: DC-coupled it inflates the RMS of those records
    by 1.5-44x (worst `H_17_0`, 43.9x) and crushes the crest factor towards 1 (`H_17_0`:
    **1.09** DC-coupled vs **4.19** AC-coupled), which would
    have falsified the ~4.5 healthy crest anchor for instrumental reasons. `vib_rms`/`vib_crest`
    are therefore AC-coupled, matching `scripts/calibrate_bearing.py`, which derived the bearing
    simulator's constants on `x - mean(x)` from the start (re-running it after the change moved
    nothing: all seven measured constants still CONFIRMED). The bias is **kept visible, not
    hidden**: the feature table carries the DC-coupled `rms`/`crest` beside the AC-coupled
    `rms_ac`/`crest_ac` and `dc_offset_ms2` (= the window `mean`), `min`/`max` stay DC-coupled,
    and `meta["dc_offset_ms2"]` records the whole-record offset per file (`meta["ac_coupling"]`
    states the choice). The envelope-spectrum features never had the problem --
    `nebulax.features.vibration._clean` mean-removes before the kurtogram and the bandpass.
    See `docs/parameters.md` bearing section 4.
  - `features`: **0.25 s (10,500-sample) and 1 s (42,000-sample) tiled windows** of
    `Accelerometer`, both scales, tagged by a `window_s` column -- 10 + 40 = 50 windows/file x 60
    files = 3,000 rows. Per window: `mean, std, min, max, slope, rms, kurtosis, crest` (
    `nebulax.features.stats.window_stats`) plus its AC-coupled pair `rms_ac, crest_ac` and the
    `dc_offset_ms2` bias behind them, plus `skew, p2p, shape_factor, impulse_factor`
    (computed directly, not covered by `window_stats`), the full envelope-spectrum feature set
    from `nebulax.features.vibration.envelope_spectrum_feats` (`env_bpfo_h1..h3`,
    `env_bpfi_h1..h3`, `env_bsf_h1..h2`, `env_ftf_h1`, each family's `_energy`/`_freq_hz`,
    `sk_band_fc, sk_band_bw, sk_max, shaft_hz`), the operating conditions the rig sets
    (`nominal_speed_rpm, nominal_load_units`, both constant across classes by design of the
    test matrix) and the measured `temp_diff_mean_c`, the **metadata** `meta_class`,
    `meta_bearing_id`, `meta_state`, `meta_state_label` — the file name's own ground truth and
    the grouping id, held out of `X` by `nebulax.schema.feature_columns()` because
    `meta_state` is a bijection of the `severity` label and `meta_class` of `fault_type`
    (the W1 audit's second leakage path, `docs/audits/w1_audit.json`) — and the labels
    below. The kurtogram search depth is capped
    (`max_level=3` at 1 s, `max_level=2` at 0.25 s) so the full 60-file adapter run finishes in
    ~40 s rather than several minutes -- timed empirically on the real data, documented in
    `_KURTOGRAM_MAX_LEVEL`.
  - **Labels = condition**, exactly per the file name: `fault_type` from `Class` (`H -> healthy`,
    `I -> inner_race`, `O -> outer_race`, `B -> ball`, `C -> cage`, all legal
    `nebulax.schema.FAULT_TYPES["bearing"]` values), `severity` from `state`
    (`0 -> 0.0, 1 -> 0.5, 2 -> 1.0`), `is_faulty = state != 0`. `rul_s` is left `NaN` -- there is
    no run-to-failure trace, only a seeded static condition per recording.
  - `fault_log`: one row per non-healthy recording (40 rows: 5 bearing ids x 4 fault classes x 2
    states), health state hard-coded from the file name. `shape="step"` (the seeded defect is
    present for the whole recording, not evolving within it), `gamma=1.0`, `t_onset` = the
    recording's synthetic anchor time, `t_failure`/`t_functional_failure` left `NaT` -- unknown,
    there is no failure event to date. `params_json` carries the raw `Class`/`bearingId`/`state`
    plus the nominal RPM/load for that recording.
  - `events`: empty -- no discrete event concept in this dataset.
  - Timestamps are entirely synthetic (Ottawa ships no wall-clock time): each recording is
    anchored 20 s apart from a fixed `2020-01-01T00:00:00Z` epoch purely to give every row a
    legal, sortable `datetime64[ms, UTC]`; recorded in `meta["timestamp_epoch"]`, carries no
    calendar meaning.
  - Peak RAM: one file (~8 MB of the 4 columns actually read, float32) is processed at a time and
    discarded before the next; the full 60-file run measured **~270 MB** resident (`/usr/bin/time
    -v`), far under the 8 GB budget -- no chunking beyond one-file-at-a-time was needed.
- **Tests**: `tests/test_adapter_ottawa.py` copies 3 real files (one healthy + both fault states
  of the same physical bearing) into a temp `raw_dir`, skips entirely if
  `data/raw/ottawa/{H_1_0,I_1_1,I_1_2}.csv` are absent, and runs in ~5 s. Covers full schema
  validation, both `FileNotFoundError` paths plus the bad-filename `ValueError`, purity (raw files
  untouched, no new files created), the long-table signal set and cadence, the feature table's
  shape/columns for both window scales, the DC/AC coupling contract (`dc_offset_ms2` is the
  window mean, `rms_ac = sqrt(rms^2 - mean^2) <= rms`, the long `vib_rms`/`vib_crest` are the
  AC pair and `vib_kurt` the kurtosis, and on the windows whose bias exceeds their AC RMS the
  DC-coupled crest collapses below 2 while the AC one does not), bearing-wise grouping via
  `train_id`, the fault-log's
  health-state mapping (including that a healthy recording gets no fault-log row), and the
  provenance/correction notes in `meta`. The adapter was also run end-to-end against the full raw
  directory (`python -m nebulax.adapters.validate --source ottawa --raw data/raw/ottawa`, ~44 s,
  3,000 telemetry rows, 3,000 feature rows, 40 fault-log rows, `OK`) as part of building it; that
  full run is not part of the automated test suite.
