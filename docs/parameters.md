# Simulator parameters — where every number comes from

One row per constant that a NEBULA X simulator actually uses, with its **provenance tag**:

| Tag | Meaning |
|---|---|
| **measured** | read off a published table, or measured by us on real data we hold |
| **derived** | our arithmetic on a published correlation or table |
| **calibrated** | fitted by us on a real dataset in `data/raw/`, script named |
| **ours** | a modelling assumption with no source — labelled as such on the slide too |

Where the approved plan and `docs/research/rail_phm.md` disagree on a physical constant,
**rail_phm.md wins** and the disagreement is stated in the section below.

---

## Bearing — `nebulax/sim/bearing.py`

Reproduce everything in this section with:

```
python scripts/calibrate_bearing.py --table results/sim_checks/cal_bearing_windows.csv
```

(~35 s over the 60 records; writes `results/sim_checks/cal_bearing_*.png`). Re-running it
after this calibration reports **CONFIRMED** on the seven constants Ottawa can measure and
**UNREFUTED** on the two speed exponents `vib_speed_exp_healthy` / `vib_speed_exp_defect`,
which this dataset cannot test at all (§2) — nine rows, `7 CONFIRMED + 2 UNREFUTED`, and that
is the audit. A run that printed CONFIRMED on all nine would mean the script had started
claiming evidence it does not have. **Acceptance criterion (orchestrator decision, 15 Sep 2026):** `7 CONFIRMED + 2 UNREFUTED` is the accepted W1 verdict for this table; confirming the two exponents requires the variable-speed Ottawa release (`ottawa_variable_speed`, DOI 10.17632/v43hmbwxpm.2), which is an optional later download, not a W1 gate.

### 1. Vibration — calibrated against the University of Ottawa set

**Data.** UORED-VAFCLS, Mendeley `y2px5tg92h` v5, CC BY 4.0 — 60 records × 10 s at 42 kHz,
20 physical bearings, classes `H`/`I`/`O`/`B`/`C` (healthy / inner race / outer race / ball /
cage) at two seeded fault states. Loaded exactly as `nebulax.adapters.ottawa` loads it
(`Accelerometer` × 9.80665 → m/s²), cut into **1 s windows**, and reduced with
`nebulax.features.stats` + `nebulax.features.vibration` (kurtogram band → Hilbert envelope →
harmonic amplitudes, geometry `n=9, Bd=7.94 mm, Dp=39.0 mm`).

**"BPFO amp" means amplitude.** `env_bpfo_*`/`env_bpfi_*` are **single-sided envelope-spectrum
amplitudes** `2|E_k|/n` of the mean-removed Hilbert envelope, in m/s² — the same units as the
acceleration they are demodulated from, and independent of the window length. `bpfo_amp` below
is the sum over the first three harmonics. Until this run `envelope_spectrum_feats` returned
FFT *power* `|E_k|²/n` under those names (a quantity that grows linearly with window length)
while this document called them amplitudes; the function was fixed rather than the wording, so
the ratios and the two `vib_bpfo_*` constants in this section were re-measured with it.

**Measured, per-record medians (median over records of the per-window median):**

| Class | records | rpm span | AC RMS (m/s²) | kurtosis | crest | BPFO amp | BPFI amp |
|---|---|---|---|---|---|---|---|
| healthy | 20 | 1709–1948 | **32.1** | **3.15** | **4.62** | 1.00× | 1.00× |
| inner race | 10 | 1789–1819 | 285.1 (8.9×) | 10.08 (3.2×) | 9.02 (2.0×) | 7.8× | **9.8×** |
| outer race | 10 | 1796–2190 | 160.1 (5.0×) | 16.75 (5.3×) | **11.82** (2.6×) | **10.1×** | 9.7× |
| pooled inner+outer | 20 | 1789–2190 | 184.2 (**5.74×**, bootstrap CI 2.9–7.9×) | 12.02 (3.8×) | 9.64 (2.1×) | 7.9× | 9.8× |
| ball | 10 | 1709–1736 | 6.6× | 1.3× | 1.3× | 5.9× | 5.4× |
| cage | 10 | 1700–1800 | 5.8× | 1.3× | 1.2× | 7.4× | 5.9× |

The BPFO/BPFI columns are ratios to the healthy median; in absolute terms the healthy medians
are **0.40 m/s² BPFO** and **0.38 m/s² BPFI** (summed over three harmonics), against 4.04 /
3.72 m/s² on outer-race and 3.13 / 3.74 m/s² on inner-race records.

Healthy per-record spread (p10–p90): kurtosis 3.00–3.89, crest 4.22–6.08, RMS 25.4–79.1.
Ball and cage records are reported for completeness only — the `B_*` records are the only
ones logged at **zero applied load**, so their ratios confound damage with load.

**Constants changed.** Plots: `results/sim_checks/cal_bearing_severity.png` (before/after over
the measured bands), `cal_bearing_rms_speed.png`, `cal_bearing_anchors.png`,
`cal_bearing_envelope.png`.

| Constant | Old | New | Tag | Evidence and mapping rule |
|---|---|---|---|---|
| `vib_rms_defect` | 2.00 | **2.94** | calibrated | `A_d = A_h·(ratio−1)` with the pooled faulty/healthy AC-RMS ratio **5.74×** (inner 8.9×, outer 5.0×; file-level bootstrap CI 2.9–7.9×). Old value was "ours", implying only 4.2×. |
| `vib_crest_gain` | 4.0 | **7.32** | calibrated | The old bump peak `4.5 + 4.0 = 8.5` sits **below** the measured outer-race crest **11.82**, so it is raised until the peak reaches the measurement. |
| `vib_bpfo_healthy` | *(did not exist; floor was exactly 0)* | **0.149** | calibrated | Healthy records are **not** BPFO-silent: outer-race records carry only **10.05×** more BPFO envelope amplitude. A floor of exactly 0 made the channel trivially separable — the same error rail_phm §4.3.2 caught on temperature. (0.146 before the amplitude fix above: the ratio was 10.3× when the feature was `sqrt` of FFT power.) |
| `vib_bpfo_gain` | 1.5 | **1.351** | ours + calibrated | The `s = 1` endpoint `B_h + B_d` is **held at 1.50 m/s²** (the absolute scale of this channel has no source either way); only the measured 10.05× contrast is imposed. (1.354 before the amplitude fix.) |

**Constants confirmed (unchanged, now backed by our own measurement rather than a citation):**

| Constant | Value | Evidence |
|---|---|---|
| `vib_kurt_healthy` | 3.0 | Ottawa healthy kurtosis **3.15** (p10–p90 3.00–3.89 over 20 bearings) vs [R157] 2.76–2.96. |
| `vib_crest_healthy` | 4.5 | Ottawa healthy crest **4.62** (p10–p90 4.22–6.08) vs [R157] 4.22–5.59. The plan's `crest = 3 + 4s` intercept stays falsified. |
| `vib_kurt_gain` | 19.0 | Bump peak `3 + 19 = 22` already clears the most impulsive measured class (outer race 16.75), so [R157]'s measured 21.69 peak stands. |
| `vib_speed_exp_healthy` | 2.0 | See §2 — the fit cannot refute it. |
| `vib_speed_exp_defect` | 1.2 | See §2 — the fit cannot refute it. |

**The mapping rules, stated once** (they are also in `recommend_constants`' docstring, and the
test `tests/test_calibrate_bearing.py::test_recommend_constants_applies_the_documented_mapping_rules`
pins them):

1. Only **dimensionless ratios** cross from Ottawa into the simulator. Ottawa's healthy AC RMS
   is ~32 m/s² against our 0.62 m/s²; that ~50× gap is a real physical difference (a bench rig
   accelerometer on a small ER16K-class housing at 42 kHz broadband vs a 12 t axle box through
   a heavy structural path), already documented in `nebulax/adapters/ottawa.py`, not a unit bug.
2. Ottawa's fault states are `developing`/`faulty` with **no documented defect size**, and they
   are not even monotone (inner-race RMS is 9.7× at state 1 but 5.8× at state 2). So the whole
   defective population is treated as one class and used as the **`s = 1` endpoint for RMS** and
   as a **lower bound on the bump peak** for kurtosis/crest — never as a point on the severity
   axis. Where the literature peak already clears that bound it stands (kurtosis); where it does
   not, it is raised to it (crest).
3. The most impulsive measured class (outer race) sets the bound in both bump cases.

### 2. What the Ottawa data could **not** calibrate

**The speed exponents.** `rail_phm.md §4.3.3` proposes fitting `rms ~ v^α` per health state on
Ottawa "in an afternoon". That is only possible on the *variable-speed* member of the Ottawa
family (`ottawa_variable_speed`, DOI 10.17632/v43hmbwxpm.2 [R118], 13.7–28.9 Hz **within** one
10 s record). What is downloaded here is **UORED-VAFCLS** [R119], the *constant-speed* ablation:
speed is a per-recording constant and the whole set spans 1700–2190 rpm, i.e. **1.14× within the
healthy class**. Clustered on the record (one point per recording — the 10 windows of a record
share a bearing, a speed and a mounting):

| Class | α | 95 % CI | R² | n | span |
|---|---|---|---|---|---|
| healthy | −6.30 ± 4.97 | [−16.75, +4.15] | 0.08 | 20 | 1.14× |
| inner race | −3.44 ± 24.52 | [−59.99, +53.11] | 0.00 | 10 | 1.02× |
| outer race | −0.43 ± 2.19 | [−5.48, +4.62] | 0.01 | 10 | 1.22× |
| pooled faulty | −3.12 ± 2.35 | [−8.05, +1.81] | 0.09 | 20 | 1.22× |

Between-bearing scatter swamps the speed term: every CI contains both the incumbent exponent and
zero. **So the [R158]-derived 2.0 (healthy) and 1.2 (defect) stand unchanged** — not confirmed,
*unrefuted*, which is the honest word.

> Fitting **per window** instead would have given healthy `α = −6.27 ± 1.50`, CI [−9.23, −3.31] —
> excluding 2.0 and apparently "refuting" it at p < 0.05. That is pseudo-replication: 200 windows
> from 20 bearings are not 200 samples. `scripts/calibrate_bearing.py` prints both fits side by
> side so the trap stays visible.

To actually fit these exponents, download `ottawa_variable_speed` (`v43hmbwxpm`) — no adapter and
no download exists for it in this repo yet, and `configs/model_ladder.yaml` already lists it as a
separate dataset id.

**A severity axis** (no defect size or run-to-failure trace in Ottawa) and **anything thermal**
(see §3).

### 3. Thermal — derived, because no public data can calibrate it

No public bearing dataset we found pairs a documented thermal model or an ambient log with
vibration under a permissive licence [rail_phm §4.3.3]. Ottawa's `Temperature Difference` channel
has an undocumented reference point, so it grounds nothing. **These constants are therefore
derived from published correlations, and the derivation is the evidence:**

| Quantity | Value | Tag | Provenance |
|---|---|---|---|
| Healthy rise above ambient at 80 km/h, 12 t axle | **20–26 K** (`BearingParams.steady_rise(22.2, 0.5) = 21.0 K`) | derived | [R149] Table 1 measures **30–38 K at 200 km/h** on a real 8-car high-speed fleet. With `Q ∝ v` (Palmgren, [R150] Eqs. 7–11) and `h_a ∝ v^0.57` ([R149] Eq. 25), the steady rise scales as `v^0.43`; `(80/200)^0.43 = 0.674` → **20–26 K**. The plan's asserted 21–25 K lands inside this, so **rail_phm.md's derivation is used and the plan's bare number is dropped**. It ignores the metro's lower axle load (12 t vs ~17 t), so it is an upper bound. |
| Speed law | `ΔT ∝ v^0.43`, implemented as `P ∝ v` over `hA = hA_0 + hA_1·v^0.57` | derived | [R149] Eq. 25. The *implemented* log-log exponent is **0.466** over 8–22 m/s because of the viscous `k_visc·ω^(2/3)` torque and the natural-convection floor `hA_0` (both "ours", both must stay non-zero); `BearingParams.speed_exponent()` reports it rather than hiding it. |
| Severity → `T_box` | `dT_per_severity_k = 2.0` K over the full sweep | measured-scaled | [R149]'s wheel-flat sweep: 30→60 mm raises the rollers **+15.60 K** but the **axle box only +1.02 K** — ~15× less sensitive. The plan's law, which drove `T_box` hard, is **falsified**. |
| Healthy inter-box spread | 8.4 K | measured | [R149] Table 1. |
| Functional-failure thresholds | NS operator levels (30 K / 50 K differential, 80 °C / 115 °C absolute) | measured | [R151]. **rail_phm.md's own `T_box > 90 °C` line is dropped** in favour of these — the deviation is documented in the `nebulax/sim/bearing.py` docstring. |
| Ambient envelope | 24.3–32.4 °C routine, 19–37 °C stress | measured | rail_phm §0 (Singapore), [R152][R153]. |

### 4. Data-quality findings while calibrating (worth a slide)

* **DC offsets in the Ottawa accelerometer.** **21 of the 60 records** carry a mean larger than
  their own AC RMS (worst: `H_2_0`, mean 1606 m/s² against an AC RMS of 56). DC-coupled, those
  records read crest ≈ 1.1 and an RMS inflated **1.5–44×** (worst `H_17_0`, 43.9×), which
  would have *falsified* the ~4.5 healthy crest anchor for purely instrumental reasons (`H_17_0`: crest **1.09** DC-coupled vs
  **4.19** AC-coupled). Every statistic in this section is computed on `x − mean(x)`.
  **Follow-up: done.** `nebulax.features.stats.window_stats` still computes
  `rms = sqrt(mean(x²))` and `crest = max|x|/rms` **DC-coupled by default** — correct for a
  process signal, and the existing `window_stats` bench models keep exactly the feature vector
  they were benchmarked on — but it now takes **`ac_couple=True`**, which mean-removes each
  window before `rms` and the crest peak. The matrix keeps its width and its column order; only
  the two affected slots change value, and `feature_names(..., ac_couple=True)` renames them
  `rms_ac` / `crest_ac` (`stats.AC_STAT_NAMES`) so the emitted names stay truthful. `kurtosis`
  is central by definition and identical either way. `nebulax.adapters.ottawa` uses the flag for
  the `vib_rms` / `vib_crest` channels it writes into the long table (`vib_kurt` unchanged), and
  **keeps the artefact visible rather than hiding it**: the feature table carries the DC-coupled
  `rms` / `crest` *and* the new `rms_ac` / `crest_ac` / `dc_offset_ms2` (= the window mean)
  columns side by side, `min`/`max` stay DC-coupled, and `meta["dc_offset_ms2"]` records the
  whole-record bias per file (`meta["ac_coupling"]` states the choice). The envelope-spectrum
  features were never affected — `nebulax.features.vibration._clean` already mean-removes before
  the kurtogram and the bandpass. Re-running this calibration after the change moved **nothing**
  (it was AC-coupled from the start; `window_stats_ac` there stays the float64 reference the
  constants are measured with, pinned against `window_stats(ac_couple=True)` in
  `tests/test_calibrate_bearing.py`): all seven measured constants still read CONFIRMED and the
  two speed exponents UNREFUTED.
* **Temperature dropout sentinels** (down to −8 × 10¹¹ °C) in `C_16..C_17` are already rejected
  by `nebulax.adapters.ottawa._robust_temp_mean`; this calibration does not use temperature.
* **Bearing geometry is assumed** (ER16K-class), so BPFO/BPFI *frequency assignment* is
  approximate. It survives its own sanity check — inner-race records raise BPFI above BPFO
  (9.8× vs 7.8×) and outer-race records raise BPFO above BPFI (10.1× vs 9.7×) — but the
  outer-race margin is thin (it was 10.3× vs 9.1× under the old power-valued features) and the
  outer-race state-2 subgroup inverts, so treat individual harmonic assignments, not the band
  statistics, as the soft part.

---

## Pneumatic (APU) — `nebulax/sim/pneumatic.py`

Reproduce everything in this section with:

```
python scripts/calibrate_pneumatic.py --table results/sim_checks/cal_pneumatic_cycles.csv
```

(~19 s end to end: reading the 218 MB CSV and cutting 3,141 cycles, a 21-day simulator run for
the KS table, and four overlay plots into `results/sim_checks/cal_pneumatic_*.png`). Re-running
it after this calibration reports **CONFIRMED on all 31 rows** — that is the audit. The `--table`
flag is optional and dumps the real per-cycle table; `--no-sim` skips the simulator and the KS
section, `--no-plots` the figures, and `--leak-ramp` adds §3's lead-time experiment (~6 min more:
4 healthy baselines + 12 ramps).

**Data.** MetroPT-3 (UCI 791, DOI 10.24432/C5VW3R, CC BY 4.0), loaded exactly as
`nebulax.adapters.metropt3` loads it. The calibration window is **1 Feb – 31 Mar 2020**, the
longest failure-free stretch in the file: 445,298 rows, 3,141 complete compressor cycles. A
±3 day guard band around each of the four UCI failure episodes (18 Apr, 29–30 May, 5–7 Jun,
15 Jul 2020) is applied and **removes 0 rows**, because every episode is later than 31 March;
the guard is implemented and reported anyway so the window is auditable rather than asserted.

**Sampling.** The file is **10 s**, not the 1 Hz its dataset card claims (median inter-sample
gap 10.0 s; 190 gaps longer than 30 min). Every duration is capped at 60 s per sample before
being summed, and every rate is computed only across intervals of 8–15 s.

### 1. What was measured

| Quantity | Measured | Method |
|---|---|---|
| Load / unload switch pressure | **8.066 / 10.124 bar** (cycle min / max), **8.06 / 10.21** after correcting for the 10 s sampling lag | median over 3,141 cycles |
| Charge slope (loaded) | **+0.01894 bar/s** | endpoint: band ÷ `t_loaded`, per cycle |
| Fall rate (unloaded + off) | **+0.00157 bar/s**; unloaded alone −0.00165, off alone −0.00119 | endpoint, per phase, per cycle |
| `t_loaded` / `t_unloaded` / `t_off` | **109 / 416 / 903 s** (medians) | per cycle; `t_unloaded` p25–p75 is 416–426 s, i.e. a **fixed timer** |
| Duty (loaded fraction) | **0.089** over whole cycles, 0.114 over the raw stream | air balance |
| `idle_run_ratio` | **7.77** | DOE band ≥ 9.0 is "well maintained" [R155] |
| `Motor_current` per state | **0.040 / 3.785 / 5.918 A** (off / unloaded / loaded); max sample 9.29 A | three clean clusters, gaps at 0.25–3.5 A and 4.25–4.75 A |
| Loaded current model | `I = 5.736 + 0.2231·(P − 8.06) − 0.01139·(T − 58)` | OLS, R² 0.57, residual sd 0.16 A |
| Twin-tower period | **59 s of loaded time** (p25–p75 50–60 s), 80 % of flips while loaded | 9,050 flips on the `TOWERS` channel |
| Oil node | `hA/C` = **0.001282 /s** (τ **780 s**), R² 0.29; steady 52.0 / 62.0 / 93.3 °C off / unloaded / loaded | regression of `dT/dt` on `T` with a per-state intercept over 434,936 clean intervals |
| Outdoor ambient at Porto | **15.4–18.5 °C** | oil temperature at the record's own cold starts (charging an empty reservoir from < 2 bar, 2020-03-07 and 2020-08-17) |
| `T_oil` vs duty | **+22.4 K per unit duty**, r = 0.72 | hourly, duration-weighted |
| Measurement chain | pressures σ 0.0113 bar / 2 mbar; oil σ 0.065 K / 25 mK; current σ 0.049 A / 2.5 mA | second difference: `var(d²x) = 6σ²`, pooled over states by duration |
| 29 May – 7 Jun episode | `t_off` **624 s = 69 %** of healthy; equilibrium **8.38 bar** during its 7,781 s continuous run; `LPS` on 0.95 % of samples; reservoir to 2.81 bar | the `s = 1` acceptance case |

### 2. Constants changed

Plots: `cal_pneumatic_band.png` (control band + both rate distributions),
`cal_pneumatic_cycles.png` (the four KS panels), `cal_pneumatic_oil.png` (thermal fit, duty
coupling, duty per hour band), `cal_pneumatic_leak.png` (the episode trace, the `t_off`
shrinkage, the leak law vs endpoint choice).

| Constant | Old | New | Tag | Evidence |
|---|---|---|---|---|
| `V_res_l` | 600.0 | **290.0** | calibrated | Inside one cycle the same demand is subtracted from the charge and added to the coast-down, so the per-cycle **sum** of the two endpoint rates cancels it: `band/t_loaded + band/(t_unloaded+t_off) = Q_comp·(1−purge)·P_atm/V` = 0.02149 bar/s. DOE's receiver relation [R155] read backwards. |
| `P_start_bar` | 8.2 | **8.06** | measured | Median cycle minimum 8.066 bar, minus half a sample of OFF decay. [R87] only bounds it ("starts below 8.2"). |
| `t_hold_s` | 45.0 | **376.0** | measured | The unloaded run is 416 s and tight (p25–p75 416–426), minus the 40 s [R155] blowdown. The 45 s was invented. |
| `unloaded_vent_nls` | *(new)* | **0.098** | calibrated | The unloaded phase falls at −0.00165 bar/s, faster than leak + auxiliaries at that (higher) pressure. Without it the reservoir enters the OFF phase ~0.3 bar high and `t_off` runs long. |
| `tower_period_s` | 90.0 | **59.0** | measured | Median loaded-seconds between `TOWERS` flips. rail_phm 2.3b explicitly says to measure this rather than invent it; 90 s was invented. |
| `I_unloaded_a` | 4.0 | **3.785** | measured | Median of the unloaded current cluster. |
| `I_loaded_a` | 6.65 | **5.618** | calibrated | Regression intercept 5.736 A at `P = P_start`, referred back through the 0.53 bar discharge drop the module adds. **The loaded plateau is 5.92 A, not [R87]'s nominal 7 A.** |
| `I_loaded_kp` | 0.35 | **0.2231** | calibrated | Same regression. |
| `I_loaded_kT` | +0.02 | **−0.0114** | calibrated | **Sign was wrong.** Current *falls* with oil temperature (thinner oil, less viscous drag). |
| `T_oil_ref_c` | 60.0 | **58.0** | measured | Median `Oil_temperature`; the point the current regression is centred on. |
| `hA_oil_w_per_k` | 12.0 | **32.0** | calibrated | `hA/C` = 0.001282 /s × the pinned `C`. The first cut's sump was 2.7× too sluggish (τ 2083 s vs the measured 780 s). |
| `P_heat_loaded_w` | 900.0 | **1323.0** | calibrated | Loaded steady state sits 41.3 K above the OFF asymptote, × `hA`. |
| `P_heat_unloaded_w` | 350.0 | **320.0** | calibrated | Unloaded steady state 10.0 K above the OFF asymptote, × `hA`. |
| `T_sump_offset_c` | *(new)* | **33.5** | calibrated | OFF asymptote 52.0 °C minus the record's own cold-start ambient 18.5 °C. See §4. |
| `T_oil_init_c` | 60.0 | **66.0** | derived | The Singapore operating point implied by the new sink and duty; avoids a warm-up transient at `t = 0`. |
| `aux_nls` | 0.6 | **0.076** | calibrated | OFF decay −0.00119 bar/s over 290 L = **0.342 NL/s** quiescent draw, of which **0.266** is the healthy leak itself. |
| `brake_nl_per_stop` | 120.0 | **22.0** | calibrated | Cumulative burst scale **0.183** against the demo_runs first cut, from a least-squares fit on the 24 measured hour bands with the auxiliaries already pinned by the OFF decay. (The script reports the scale *relative to the current defaults*, so a converged re-run prints 1.00 and `CONFIRMED`.) |
| `spring_nl_per_dwell` | 30.0 | **5.5** | calibrated | Same burst scale. |
| `spring_nl_per_unit_load` | 360.0 | **66.0** | calibrated | Same burst scale; the brake ↔ air-spring ratio is untouched. |
| `leak_d1_mm` | 2.40 | **2.88** | calibrated | See §3. |
| `sensors['Reservoirs'].noise_sigma` (all 5 pressure channels) | 0.010 | **0.0113** | measured | Second-difference σ pooled over states (off 0.0061, unloaded 0.0129, loaded 0.0254 bar). The first cut's 0.010 was already inside the estimator's tolerance; the value is updated to the measurement anyway. |
| `sensors['Reservoirs'].quantum` (all 5 pressure channels) | 0.001 | **0.002** | measured | Smallest gap between distinct reported values. |
| `sensors['Oil_temperature'].noise_sigma` / `.quantum` | 0.10 / 0.01 | **0.065 / 0.025** | measured | Second-difference method. |
| `sensors['Motor_current'].noise_sigma` | 0.050 | **0.049** | measured | Same method. The spread is **not** instrument noise: the floor with the motor off is 1.1 mA and the 0.147 A while loaded is real ripple; a `SensorSpec` carries one additive σ, so the duration-pooled figure is used — and it confirms the first cut to 2 %. |
| `sensors['Motor_current'].quantum` | 0.01 | **0.0025** | measured | Smallest gap between distinct reported values. |

**Constants confirmed (unchanged, now backed by measurement rather than a citation):**

| Constant | Value | Evidence |
|---|---|---|
| `P_stop_bar` | 10.2 | Median cycle maximum 10.124 bar + half a sample of charge slope = **10.21**. [R87]'s "stops above 10.2" is confirmed to 0.1 %. |
| `P_lps_bar` | 7.0 | [R87]. **40 of 3,141 healthy cycles touch it**, so it is a real, rarely-crossed floor even on a healthy unit — not a fault-only signal. |
| `t_unload_s`, `t_reload_s` | 40.0, 3.0 | [R155]. Unresolvable at 10 s sampling, so the citation stands. |
| `I_start_a` | 9.0 | The largest `Motor_current` sample anywhere in the window is **9.29 A**. A direct-on-line peak is badly under-sampled at 10 s, so 9.0 A is a **floor**, not a fit. |
| `leak_d0_mm`, `leak_ref_nls` | 0.55 mm, 0.2 NL/s | Attributing the **whole** quiescent OFF draw to the leak gives an equivalent **0.62 mm** orifice, so the plan's 0.55 mm fits inside the measurement with 0.076 NL/s left over for auxiliaries. |

**Pinned, not fitted** — MetroPT-3 cannot identify these, and everything derived moves with them:

| Constant | Value | Why it cannot be fitted |
|---|---|---|
| `Q_comp_nls` | 7.0 NL/s | MetroPT-3 has **no flow channel** (`Flowmeter` is a MetroPT-1/2 signal, rail_phm 2.1), so pressure data fixes only the ratio `Q_comp/V_res`. Pinning `V_res = 600 L` instead gives `Q_comp = 14.5 NL/s` and leaves every pressure rate, duty and cycle feature **bit-identical**. Both are physical for a metro APU. |
| `C_oil_j_per_k` | 25 000 J/K | The thermal regression identifies `hA/C` and `P_state/hA`, never the three separately. 25 kJ/K ≈ 13 L of compressor oil (1.9 kJ/kg/K, 870 kg/m³) plus the separator shell. |

### 3. The leak endpoint — how `s = 1` is defined

rail_phm 4.2's acceptance test is that **`s = 1` must reproduce the `t_off` shrinkage of late
May – early June 2020**. Measured: healthy `t_off` 903 s → episode median **624 s (69 %)**, with
the acute phase (5–7 Jun) at `t_off = 0`, `duty = 0.944`, `t_unloaded` collapsed from 416 s to
10 s, `LPS` on 0.95 % of samples and the reservoir down to 2.81 bar.

The episodes bound the endpoint **from below**, because each was repaired before functional
failure. During the 29 May – 7 Jun episode's 7,781 s continuous run the reservoir settled at
**8.38 bar**, so the leak there exactly balanced the net delivery at that pressure — an
equivalent **2.66 mm** sharp orifice; 18 Apr settled at 8.86 bar (**2.57 mm**). We therefore
define `s = 1` as **the leak that just beats the compressor at the 7.0 bar `LPS` trip**:

```
A(1)/A(0) = (Q_comp·(1−purge) − aux) · (P_ref + P_atm) / (leak_ref · (P_LPS + P_atm)) = 27.4
d1        = 0.55 · sqrt(27.4) = 2.88 mm
```

Worst case **8.51 NL/s** at 10.2 bar — **3.3× below** the [R156] freight bound of 60 cfm =
28.3 NL/s. The plan's smaller endpoint claimed 10×; the margin narrows because the measurement
is larger than the guess, and the test bound moved from `28.3/4` to `28.3/3` accordingly.
rail_phm 4.2's "roughly 2.2 mm" is **superseded**: it was a back-calculation from the plan's own
discarded constant-draw law `Q = 0.2 + 3.0·s`, interpolated between DOE rows, not a measurement.

**What this costs, stated plainly.** The healthy APU is leak-dominated (0.266 NL/s of the
0.342 NL/s quiescent draw is the leak itself), so a *reservoir* leak takes over the demand budget
early. In service `t_off` reaches zero at **s ≈ 0.13**, and the compressor stops cycling
altogether — hence **no feature rows at all** — at **s ≈ 0.73**. The same arithmetic applied to
MetroPT-3 says the real unit loses its OFF phase at a total draw only ~1.6× healthy, and 5–7 Jun
confirms it, so this is the physics rather than a tuning artefact. The practical consequence is
that **the detection-relevant band is `s ≲ 0.15`**, which makes `gamma` the constant that
controls how long the unit spends inside it.

**The lead-time table was regenerated on 15 Sep 2026, and the rule behind it had to be
restated.** Until then this section quoted 6.3 / 9.2 / 10.8 d at γ = 1 / 2 / 3 for [R101]'s
literal rule — a 6 h rolling median of `idle_run_ratio` below **the healthy 5th percentile of
the raw feature**. That rule is not implementable on this unit and never was: `idle_run_ratio`
is **zero-inflated**. MetroPT-3's own 3,141 healthy cycles have `t_off = 0` in **17.3 %** of
cycles, so the real record's healthy p1, p5 and p10 are all exactly **0.000** and no median can
fall below them. The old numbers were readable only because the pre-15-Sep simulator's flat
demand schedule produced **no zeros at all** (`t_off = 0` in 0.000 of 2,070 healthy cycles) and
hence a healthy p5 of **≈ 5.0** (5.009 at seed 3, 4.991 at seed 5, re-measured against the
pre-fix module) — an artefact of the schedule, not a property of the APU. Fixing the demand (§4)
put 13–19 % of healthy cycles at `t_off = 0`, the healthy p5 went to 0.000 exactly as in the
real record, and the old rule stopped firing at every γ and every seed. **The retraction is a finding, not a regression.**

The restatement keeps [R101]'s statistic and its 5th percentile but makes three things explicit
that the citation leaves implicit — and each is forced by a measurement, not chosen for effect:

* the threshold is the percentile **of the rolling statistic**, not of the raw feature (the raw
  feature's lower tail is the zeros);
* the baseline is **the unit's own first 8 days** — a post-overhaul reference period, not a
  fleet constant, because the healthy level of the statistic varies by 2.6× across service
  patterns (p5 of 1.20–3.11 across the four seeds below);
* the crossing must **hold for 24 h**. A rolling median is heavily autocorrelated, so a bare
  percentile crossing chatters: with no hold at all *both* detectors fire on **2 of the 4
  healthy runs** (seeds 3 and 11) on the first day they are allowed to, and a 6–12 h hold still
  fires on both of those runs (days 8.3–16.1) with **no leak present at all**. 18 h is the
  shortest hold in the grid that is clean on all four runs and both detectors; 24 h is used for
  margin, and it costs exactly the 24 h it holds.

Measured on a 30-day ramp (onset day 8, nominal `s = 1` on day 26), **4 seeds (3/5/7/11) per
γ**, with `duty_ratio` reported beside `idle_run_ratio` because it needs no OFF phase and so
keeps working past `s ≈ 0.13`. The alarm is dated at the **end** of the 24 h hold, not at the
first sample of the excursion — a detector cannot declare before it has the evidence, and dating
it at the start would hand every row a free day. Median [min, max] over the four seeds:

| `gamma` | detector | alarm | functional failure | **lead** | fired |
|---|---|---|---|---|---|
| 1.0 | `idle_run_ratio` | day 10.33 [10.10, 10.65] | day 12.14 [11.47, 13.48] | **2.0 d [1.1, 2.8]** | 4/4 |
| 2.0 | `idle_run_ratio` | day 13.56 [13.33, 14.61] | day 16.18 [15.56, 16.59] | **2.3 d [2.0, 2.9]** | 4/4 |
| 3.0 | `idle_run_ratio` | day 16.04 [14.68, 17.24] | day 18.52 [17.66, 18.79] | **2.5 d [1.6, 3.0]** | 4/4 |
| 1.0 | `duty_ratio` | day 10.34 [10.09, 10.51] | day 12.14 [11.47, 13.48] | **2.0 d [1.1, 3.0]** | 4/4 |
| 2.0 | `duty_ratio` | day 13.54 [13.32, 14.48] | day 16.18 [15.56, 16.59] | **2.3 d [2.1, 2.9]** | 4/4 |
| 3.0 | `duty_ratio` | day 15.96 [14.68, 16.76] | day 18.52 [17.66, 18.79] | **2.6 d [2.0, 3.0]** | 4/4 |
| — | both | **never fires** on the matched **healthy** run | — | — | 0/4 |

Reproduce with `python scripts/calibrate_pneumatic.py --leak-ramp`
(`calibrate_pneumatic.leak_ramp_lead_time`; ~6 min, 4 healthy baselines + 12 ramps). The
healthy row is the false-alarm audit and is part of the experiment, not a footnote: a rule that
fires early on a healthy unit can be given any lead time you like.

**What this costs us, stated plainly.** The honest lead is **2–3 days, not 1–4 weeks**, and
[R101]'s band is **not** met on this unit. Three numbers say why, and none of them is a knob:
the functional-failure day itself moved (γ = 2: day 16.2 [15.6, 16.6] now against the 18.57
quoted before, read off `t_functional_failure` the same way); the healthy
APU is leak-dominated (§2: 0.266 of the 0.342 NL/s quiescent draw *is* the leak), so `t_off`
reaches zero at `s ≈ 0.13` and the whole detectable ramp is short; and the two detectors agree
to within 0.1 d on every γ, so this is the physics rather than one badly chosen statistic.
**`gamma` buys almost no extra warning** — 2.0 → 2.5 d from γ = 1 to γ = 3 — it moves the alarm
and the failure together, stretching only the *quiet* part of the ramp. `sample_trajectory`'s
default `gamma_range=(1, 3)` therefore stays, but the earlier recommendation
"**γ ≥ 2 for air leaks**" is **withdrawn**: it was justified by a lead time that no longer holds.

A full day of that lead is spent on the hold itself. What would buy it back is not a different
threshold but a different *estimator*: the 24 h is paid purely to out-live the autocorrelation
of a rolling median, and an accumulating test — CUSUM on the same statistic, which
`docs/research/model_ladder.md` §2a already lists as a MUST — spends that budget on evidence
instead of on delay. That is a detector-side task, not a simulator constant, so it is recorded
here as the open item rather than tuned into this table.

### 4. Validation — KS distance against MetroPT-3 healthy cycles

Simulator: 21 days, seed 3, healthy, cycles extracted by the *same* code as the real ones.

| Feature | MetroPT-3 median | Sim median | sim/real | **KS** | KS before 15 Sep 2026 |
|---|---|---|---|---|---|
| `t_loaded` | 109 s | 110 s | 1.01 | **0.324** | 0.472 |
| `t_off` | 903 s | 952 s | 1.05 | **0.091** | 0.480 |
| `dP_dt_off` | −0.001228 bar/s | −0.001267 bar/s | 1.03 | **0.226** | 0.620 |
| `I_loaded_mean` | 6.005 A | 5.855 A | 0.98 | **0.747** | 0.748 |
| `T_oil_max` | 66.4 °C | 72.3 °C | 1.09 | **0.454** | 0.458 |

n = 3,141 real cycles, 1,310 simulated. Reproduce with
`python scripts/calibrate_pneumatic.py` (19 s end to end, KS table and phase table printed).

**What changed: the burst schedule is heavy-tailed and the phases are no longer alike.**
Until 15 Sep 2026 every `run` segment drew exactly `brake_nl_per_stop` and every `dwell` exactly
its air-spring share, so every simulated cycle landed on the mean and **the medians agreed while
the distributions did not** — every KS above 0.45. MetroPT-3 says the demand is not one number:
`t_off` spreads **0–1,814 s** (sd 600 s, 17.3 % of cycles at zero), `duty_ratio` spreads
0.015–0.99 with 13.3 % of cycles above 0.5, the per-cycle *quiescent* OFF draw spreads
**0.205 – 0.351 – 0.97 NL/s** (p95 – median – p5), and the draw per phase is
**0.733 (loaded) / 0.473 (unloaded) / 0.342 (off) NL/s** — a burst big enough to matter pulls the
pressure down fast enough to start the compressor, so it lands inside a loaded window *by
construction*, while ours drew air regardless of state and so polluted the OFF decay.

The replacement, in `nebulax/sim/pneumatic.py::_consumption`, **changes no calibrated mean**.
Per service event the air is `mean × M` with `M` a unit-mean three-component mixture — nothing at
all (a blended ED brake took the stop, or the load step was inside the levelling valve's
deadband) / an ordinary application / a rare large event (door + brake test, coupling,
crush-load levelling, brake-pipe recharge) — each spread by a unit-mean gamma
(`_burst_multipliers`). The auxiliaries get the same treatment on a 30 min block, because the
measured quiescent spread above cannot come from the bursts. Because `E[M] = 1` **exactly**, the
air balance, the duty and all 31 audited constants are untouched: the re-run above still reports
**CONFIRMED on all 31 rows**, with `burst_scale` 0.999. (`fit_consumption` asks `_consumption`
for the schedule with `rng=None`, i.e. the *expected* schedule, precisely so the hour-band fit
does not chase one noisy realisation.) The five new knobs are modelling assumptions — tagged
**ours**, set by the shape of the distributions in this section rather than by a source — and
live with their evidence in `PneumaticParams`:

| Constant | Value | Tag | What it is |
|---|---|---|---|
| `burst_draw_prob` | 0.06 | ours | fraction of stops / dwells that draw reservoir air at all |
| `burst_shape_k` | 0.8 | ours | gamma shape of one drawing event (`sd/mean = 1/√k`) |
| `big_event_prob` | 0.006 | ours | rate of a large event per service event ≈ 5 per day |
| `big_event_mult` | 120 | ours | its mean size in units of the per-event mean, drawn from the **same** budget |
| `burst_max_nls` | 2.0 NL/s | ours | capacity of the one pipe feeding brake + springs; simultaneous demands queue (`_choke`) |
| `aux_active_frac`, `aux_block_s` | 0.70, 1800 s | ours | the auxiliaries are intermittent, in blocks of ~2 compressor cycles |

Two mechanisms then do the work and both are emergent, not imposed: a large event is *not* an
impulse (the line is choked, so it becomes a multi-minute episode, which is what cuts `t_off` to
zero), and the event that matters *starts the compressor*, so its air is spent inside a loaded
window and the OFF windows that survive are the quiet ones. `_run_core` is told none of this; it
falls out of the pressure feedback.

**Result, per phase** (median over cycles, endpoint rate, same estimator both sides —
`sim_phase_demand_nls`):

| Phase | MetroPT-3 | Sim (was, flat schedule) | sim/real |
|---|---|---|---|
| loaded | 0.733 NL/s | **0.599** (0.661) | 0.82 |
| unloaded | 0.473 NL/s | **0.516** (0.688) | 1.09 |
| off | 0.342 NL/s | **0.348** (0.543) | 1.02 |

The ordering and the OFF leg are now right; the loaded leg is still 18 % light for the same
reason `t_loaded`'s KS is stuck (below). Simulated `t_off` spreads 0–1,651 s with **sd 546 s**
(real 600 s) and **16.3 %** of cycles at zero (real 17.3 %); `dP_dt_off` p5/p25/p50/p75 is
−0.00321/−0.00169/−0.00127/−0.00104 against the real −0.00338/−0.00166/−0.00123/−0.00095.

**What is left, and why each one is a floor rather than a knob:**

* **`dP_dt_off` 0.226 is set by `leak_d0_mm`, not by the demand schedule.** The whole residual
  sits at the shallow end: 25 % of real cycles decay slower than **−0.000946 bar/s**, and the
  simulator cannot, because with the auxiliaries at zero the healthy leak alone draws
  0.2717 NL/s at the OFF window's mean pressure = **−0.000949 bar/s**. Only 3.6 % of simulated
  cycles get below that (the ones whose OFF window sits low in the band, where the orifice
  flows less). Reaching the real p95 (−0.000715 = 0.205 NL/s) with a non-negative auxiliary
  draw needs `leak_d0_mm ≤ 0.55·√(0.205/0.2717) = 0.48 mm`. That is a section-2 constant and a
  *pinned* one, so it was not touched here; it is the next thing to move if this KS matters, and
  it would move `aux_nls` with it (the two are identified only by their sum, see §2).
* **`t_loaded` 0.324 is arithmetic.** MetroPT-3 puts 11.1 % of healthy cycles at `t_loaded ≤ 30 s`
  and 13.9 % at ≤ 60 s; with a 2.15 bar band and 6.16 NL/s of net delivery a charge cannot take
  less than **107 s**, so those are either unloader short-cycling or a 10 s sampling artefact and
  no 1 Hz state machine will reproduce them. Ours span 107–189 s.
* **No `duty_ratio → 1` population, on purpose.** 13.3 % of real cycles sit above 0.5 duty; ours
  effectively never do — **0.00–0.11 %** of cycles, with the maximum duty anywhere on a 30-day
  healthy run at **0.470 / 0.486 / 0.510 / 0.481** for seeds 3/5/7/11 (0.467–0.483 over 21 days
  at seeds 3/11/101) — because the supply line is choked at `burst_max_nls = 2.0` NL/s. That is a
  deliberate trade: the real unit is at Porto (outdoor 15–19 °C), we run Singapore's 24–33 °C,
  and a sustained duty above ~0.75 walks the sump through the **95 °C functional-failure line**
  on a *healthy* run. Before the choke was made a property of the line rather than of each
  event, two overlapping large events did exactly that on 3 of 4 seeds of a 30-day healthy run.
  With it, the worst sump temperature over 30-day healthy runs at seeds 3/11/29/101 is
  **90.1–92.4 °C** (90.8 / 92.4 / 91.2 / 90.1) and `LPS` never asserts on any of them.
* **`I_loaded_mean` 0.747 and `T_oil_max` 0.454 are unchanged by all of this** (0.748 / 0.458
  before), and both are explained elsewhere: the loaded current is referred back through our
  0.55 bar `TP2 − TP3` against the measured 0.304 (§5), and the sump runs 6 K hot because
  **Singapore is not Porto** — a genuine finding, since it means the 95 °C line has far less
  margin here, and on a 30-day leak ramp the 105 °C thermal-protection ceiling actually binds.

### 5. Data-quality and model findings (worth a slide)

* **The oil node has two time constants and the simulator models one.** The cycle-scale
  regression gives τ = 780 s and an OFF asymptote of **52 °C**, but the record's own cold starts
  put the true outdoor ambient at **15–19 °C**. A single node cannot relax to 52 °C on a 780 s
  constant *and* sit 34 K above outdoor air — there is a slow, massive warm body (the compressor
  block and its bay) that the fast oil node relaxes toward. We keep the fast node and give its
  sink the measured offset `T_sump_offset_c = 33.5 K`. Fitting the slow node from the logger's
  own multi-hour gaps is **not identifiable**: the compressor state during a gap is unknown, and
  the fit lands at rms 9.5 K with `T_amb` running from −4 °C to +24 °C depending on which gaps
  are admitted. Recorded as a limitation, not a number. R² of the single-node fit is **0.29**.
* **`TP2` / `H1` placement — FIXED on 15 Sep 2026.** `H1` was emitted inverted (high while
  loaded, ~0 otherwise). The previous note said *both* channels were inverted; the measurement
  says only `H1` was — `TP2`'s state shape was already right and only its 0.02 bar idle floor
  moved (to 0.0, the SensorSpec clip floor). Measured on the failure-free window
  1 Feb – 31 Mar 2020 (445,298 rows,
  states from `Motor_current` exactly as `scripts/calibrate_pneumatic.py` cuts them: off < 2 A,
  loaded > 5 A; 279,701 / 114,672 / 50,925 samples), the medians are

  | state | `TP2` | `H1` | `TP3` |
  |---|---|---|---|
  | off | **−0.012 bar** | **8.706 bar** | 8.718 bar |
  | unloaded | **−0.012 bar** | **9.662 bar** | 9.678 bar |
  | loaded | **9.294 bar** | **−0.012 bar** | 9.162 bar |

  and while loaded `TP2 − TP3 = +0.304 bar` (p25–p75 0.156–0.396) against `H1 − TP3 = −9.168 bar`.
  Recompute in three lines with the calibration script's own helpers, so the table above is not a
  one-off: `f = load_metropt3(Path("data/raw/metropt3"))`, `f = f[normal_window_mask(f["timestamp"])[0]]`,
  `st = segment_states(f["Motor_current"].to_numpy())`, then median `f["TP2"]` / `f["H1"]` per `st`.
  The `COMP` digital alone gives the same split (`COMP = 0`, i.e. working: `TP2` 9.234 / `H1`
  −0.012; `COMP = 1`: `TP2` −0.012 / `H1` 8.958) and agrees with the current cut on 99.4 % of
  samples, so the shape is not an artefact of the thresholds. −0.012 bar is the transducers'
  common zero offset, below the 0.0113 bar noise σ, so the simulator's clean floor is 0.0.
  `TP2` is therefore the **compressor discharge** tap (upstream of separator, air/oil filter and
  dryer, vented with the unloader) and `H1` the **cyclonic-separator discharge** tap, which reads
  the panel whenever the separator is not discharging and is dumped while loaded — the two are
  *not* two points of one chain. `nebulax/sim/pneumatic.py` now emits exactly this shape; the
  anchors above are asserted in `tests/test_sim_pneumatic.py`. The `clogged_filter` fault was
  re-routed accordingly: it separates on **`TP2 − TP3` while loaded** (discharge minus panel
  spans the filter, so clogging opens it from ~0.55 to ~1.4 bar) with **`H1` as the negative
  control** (reservoir side of a closed check valve — it must not move). That pair beats the old
  single `TP2 − H1` difference, because an *air leak* moves `H1` (it follows the panel) without
  opening `TP2 − TP3`. The one number still open: our loaded `TP2 − TP3` is **0.55 bar** against
  the measured 0.304, because the drop is the sum of three handbook [R155] drops
  (`dp_sep` 0.10 + `dp_filter` 0.15 + `dp_dryer` 0.28) rather than a fit. Fitting it is a
  *constants-table* change, not a signal-shape one — `I_loaded_a` is defined as the current
  regression's intercept referred back through exactly that sum — so the two must move together
  in the next calibration pass, and every audited row was left CONFIRMED by this fix.
* **`H1_loaded_mean` is now ~0 by construction, in the simulator and in the real file.** The
  column's semantics are unchanged (mean `H1` over the cycle's loaded samples); its healthy value
  moved from ~9.6 bar to ~0.004 bar, which is what MetroPT-3 itself gives (−0.012 bar), so the
  simulator and `nebulax.adapters.metropt3` finally agree on that column instead of differing by
  9.5 bar. It is retained as a vent-integrity check and as the `clogged_filter` negative control.
  **`TP2_minus_TP3_mean` mismatch — FIXED 15 Sep 2026** (was the remaining item here): the
  adapter used to average `TP2 − TP3` over the **whole** cycle while the simulator averages over
  the **loaded** samples only — the same column name, two incompatible quantities. `TP2` is zero
  for most of a cycle (off + unloaded), so the old adapter figure was ≈ **−7.9 bar** against the
  simulator's/real loaded-only **+0.304 bar** median (this section's `TP2` table above).
  `nebulax/adapters/metropt3.py::_build_cycle_features` now restricts `TP2_minus_TP3_mean` (like
  `I_loaded_mean` and `H1_loaded_mean` already did) to `state == 2` (loaded) samples — the same
  `Motor_current`-derived cut `scripts/calibrate_pneumatic.py` uses — so the adapter and the
  simulator finally compute the identical quantity; the full-file adapter run now gives a median
  of **+0.188 bar** (p25–p75 0.032–0.321 bar) per cycle, broadly consistent with the pooled
  +0.304 bar above (per-cycle averaging over short, sometimes noisy loaded runs widens the tails
  relative to the pooled-sample statistic).
  **The negative tail is real and measured, not "a handful of 1–2-sample cycles" (that
  explanation, written before the numbers behind it were checked, is false and is corrected
  here):** on the Feb–Mar 2020 healthy window, 27.1 % of cycles (n = 2,621) are negative, median
  +0.153 bar. Per-*sample* loaded `TP2 − TP3` in the same window is left-skewed (mean +0.005 bar,
  18.2 % of loaded samples negative, 10.3 % below −1 bar) even though its *median* is the
  pooled +0.304 bar quoted above — a mean-vs-median gap, not a small-sample artefact. The skew is
  concentrated in a specific position: the **first** 10 s sample of every loaded *run* (a duty
  cycle can re-load more than once, e.g. after a brief unload) lands mid-ramp, with `TP2` still
  rising toward `TP3` — mean **−1.194 bar**, 58.3 % negative at that one position, against
  **+0.28 bar** mean and almost no negatives at positions 1–4 of the same run. Because the
  per-cycle mean pools every loaded run a cycle contains, cycles that reload more often carry
  *more* of these negative first-of-run samples, so **more loaded samples means more negative,
  not less** — **corrected 15 Sep 2026** (the two counts below were carried over from an earlier
  draft of this note without being re-measured against the shipped adapter; re-measured here with
  the adapter's own cycle segmentation and its own `state == 2` sample count per cycle, on the
  same Feb–Mar 2020 healthy window as above): cycles with exactly one loaded sample (no ramp
  sample to land on) are trivially **100 %** positive (median +0.380) — but there is only **n = 1**
  such cycle in this window, too few to generalise from; the pattern is clearer at the other end,
  where cycles with ≥16 loaded samples (n = 383, several reloads) are **51.2 %** negative (median
  −0.006). This is consistent with, and is the same
  population `is_transition` is built to flag: averaging over `(state == 2) & ~is_transition`
  instead of plain `state == 2` on the same Feb–Mar window gives a **per-sample median of
  +0.350 bar** (39,601 of the window's 50,925 loaded samples survive the `transition_mask()`
  exclusion) — **corrected 15 Sep 2026**, re-measured with
  `nebulax.adapters.metropt3.transition_mask` on the calibration script's own state cut
  (`scripts/calibrate_pneumatic.segment_states`, `Motor_current < 2 A` off / `< 5 A` unloaded /
  else loaded) over `normal_window_mask`'s `NORMAL_WINDOW = ("2020-02-01", "2020-04-01")` minus
  the failure guard bands (no rows excluded in this window); the previous **+0.357 bar** figure
  here was not reproducible against the shipped `transition_mask()` and is replaced. This is the
  median of the raw per-sample `TP2 − TP3` differences pooled over all 39,601 selected samples
  (not a per-cycle mean of per-cycle statistics), within noise of the pooled +0.304 bar
  plain-`state == 2` figure above. The shipped `TP2_minus_TP3_mean` definition is
  deliberately left as the plain `state == 2` mean regardless — narrowing it to exclude
  `is_transition` would reintroduce a mismatch with `nebulax.sim.pneumatic`'s identically-named
  column (the simulator does not exclude transition samples from any `*_loaded_*` column either;
  see (a)'s "identical definitions" requirement) — so the per-cycle figure is expected to run
  lower and more scattered than the pooled-sample statistic, and that is what the data show.
  The shared loaded/off feature-definition table lives in
  `nebulax/adapters/metropt3.py::_build_cycle_features`'s docstring — **not** echoed in
  `nebulax/sim/pneumatic.py::_cycle_features`'s own docstring (that file is out of this fix
  task's scope and is untouched; an earlier draft of this note claimed the echo and was wrong —
  corrected 15 Sep 2026) — and is cross-checked in
  `tests/test_adapter_metropt3.py::test_loaded_off_features_match_simulator`, which now calls
  `nebulax.sim.pneumatic._cycle_features` itself (not just its private `_nan_mean_by` helper) on
  a shared synthetic 1 Hz trace and compares every `*_loaded_*` / `*_off_*` column the two sides
  claim in common. That rewrite caught two further definitional mismatches the original fix
  missed, both now aligned: **`I_loaded_mean`** now excludes the first 4 samples of the cycle
  (matching the simulator's own starting-current-transient exclusion — it was previously the
  plain `state == 2` mean, which a 2-sample 12 A inrush at the start of a loaded run could pull
  up by several tenths of an amp), and **`dP_dt_loaded` / `dP_dt_off`** now average each sample's
  own `d(Reservoirs)/dt` against its *preceding* sample (`_backward_diff_rate`) instead of
  `(last − first) / duration`, which drops the phase's own transition-in delta (a fencepost: a
  contiguous run of *n* samples has *n − 1* internal diffs, so dividing by an *n*-sample duration
  understates the rate) — the two formulas coincide for a uniform-cadence run but not for this
  file's variable ~10 s cadence.
* **The dataset card's "1 Hz" is wrong** — the file is 10 s, with 190 gaps longer than 30 min and
  one apparent 6 h "cooldown" that is actually a logger outage. Already flagged by
  `nebulax.adapters.metropt3`; repeated here because it changes every rate estimate.
* **`SIGNAL_SPECS` and the module now agree about `H1`** (reconciled 15 Sep 2026, with the
  inversion above). The spec said "downstream of the cyclonic separator" while the module placed
  `H1` downstream of the air/oil filter; the file says it is neither — it is the separator's
  *discharge* tap, alive only while the compressor is not delivering. Both descriptions were
  rewritten to that, and `TP2`'s to "compressor discharge pressure, upstream of
  separator/filter/dryer". Text only: no `SIGNAL_SPECS` API change.
* **`Flowmeter` does not exist in MetroPT-3**, so neither its scaling nor its sensor chain could
  be calibrated, and the [R93] `Flowmeter_max > 16.05` air-leak rule **cannot be transferred
  numerically** to our units: over a 21-day healthy run the peak `Flowmeter_max` of any cycle is
  **2.99–3.60 NL/s** (seeds 3/11/101) with a per-cycle median of **2.34 NL/s**, against the
  rule's 16.05 — and ~16 was our own peak only before the consumption was calibrated down.
  Reproducing that rule needs MetroPT-1 or MetroPT-2.

---

<!-- BEGIN: calibrate_door -->
## Door — `nebulax/sim/door.py`

Reproduce everything in this section with:

```
python scripts/calibrate_door.py              # exits 2 only if the data is missing
python scripts/calibrate_door.py --self-check  # recovery test of the estimator
```

**Calibration status: CALIBRATED.**

Measured on the real CORD release in `data/raw/cranfield` — **7,790 strokes** from **779 test recordings** across 13 condition files (`Normal`, `LackLubrication1-2`, `Backlash1-2`, `Spalling1-8`); each recording is 80 s at 25.0 Hz holding 5 out-and-back sequences, and the design is 2 motion profiles × 3 loads (20, 40, −40 kgf) × 10 repetitions per condition. Channels: position set point, position **error**, motor current — **no voltage**, which is what bounds §2.

> **Read §3 before any current ratio below.** The rig's motor is a *stepper*, not the PMDC drive `nebulax/sim/door.py` models, so every faulty/healthy **current** ratio here is a **proxy** for the simulator's torque-proportional observable and the gains read off it with a mean-current ratio are **lower bounds**.

### 1. Cranfield ratio vs sim ratio

Both columns are produced by the **same estimator** (`scripts/calibrate_door.py::stroke_table`) — segment strokes on the reference derivative, take the constant-speed segment per [R64], then form the faulty/healthy median ratio. Simulator side: `DoorParams.cranfield()` on a one-day service window at constant severity.

| Cranfield class | level | sim severity | metric | Cranfield ratio | sim ratio (current default) | gain | default | gain read off the curve |
|---|---|---|---|---|---|---|---|---|
| `lack_of_lubrication` | 1 | 0.50 | `i_cruise_mean` | 1.169 | 1.859 | `k_friction_c` | 2.000 | 0.390 |
| `lack_of_lubrication` | 2 | 1.00 | `i_cruise_mean` | 1.248 | 2.709 | `k_friction_c` | 2.000 | 0.287 |
| `lack_of_lubrication` | 1 | 0.50 | `v_cruise` *(diagnostic)* | 1.034 | 0.999 | `k_friction_c` | 2.000 | PENDING |
| `lack_of_lubrication` | 2 | 1.00 | `v_cruise` *(diagnostic)* | 1.027 | 0.996 | `k_friction_c` | 2.000 | PENDING |
| `backlash` | 1 | 0.50 | `err_spike_width_s` *(diagnostic)* | 0.807 | 1.389 | `backlash_m` | 0.008 | PENDING |
| `backlash` | 2 | 1.00 | `err_spike_width_s` *(diagnostic)* | 0.807 | 1.222 | `backlash_m` | 0.008 | PENDING |
| `backlash` | 1 | 0.50 | `err_spike_peak_m` | 0.977 | 1.644 | `backlash_m` | 0.008 | PENDING |
| `backlash` | 2 | 1.00 | `err_spike_peak_m` | 1.029 | 2.584 | `backlash_m` | 0.008 | 0.001 |
| `spalling` | 1 | 0.12 | `ripple_frac` | 1.040 | 1.015 | `k_mis` | 0.150 | PENDING |
| `spalling` | 2 | 0.25 | `ripple_frac` | 1.257 | 1.055 | `k_mis` | 0.150 | PENDING |
| `spalling` | 3 | 0.38 | `ripple_frac` | 1.341 | 1.111 | `k_mis` | 0.150 | 0.275 |
| `spalling` | 4 | 0.50 | `ripple_frac` | 1.336 | 1.232 | `k_mis` | 0.150 | 0.185 |
| `spalling` | 5 | 0.62 | `ripple_frac` | 1.218 | 1.339 | `k_mis` | 0.150 | PENDING |
| `spalling` | 6 | 0.75 | `ripple_frac` | 1.240 | 1.441 | `k_mis` | 0.150 | PENDING |
| `spalling` | 7 | 0.88 | `ripple_frac` | 1.350 | 1.564 | `k_mis` | 0.150 | 0.113 |
| `spalling` | 8 | 1.00 | `ripple_frac` | 1.448 | 1.697 | `k_mis` | 0.150 | 0.115 |

Level→severity map: lubrication and backlash keep the plan's anchors (`s = 0.5`, `s = 1.0` for stages 1 and 2). Spalling has **8** stages in the real release — the plan never saw them — so they are placed linearly at `s = level/8`, the placement that keeps the plan's anchors: **stage 4 → `s = 0.5`, stage 8 → `s = 1.0`**. All eight are measured and reported; the gain is read off whichever of them the sweep can invert. This is deliberately **not** the `severity` column `nebulax.adapters.cranfield` writes (`min(1, 0.2 + 0.2·(level−1))`, which saturates at spalling stage 5); that one is an ordinal ML label, this one is the physics target. Both are correct on their own axis and must not be unified.

### 2. Healthy least-squares fit — what is identifiable

600 healthy strokes, 66367 moving samples, mechanical R² = 0.030, electrical R² = PENDING.

| quantity | fitted | unit | status |
|---|---|---|---|
| `stroke_m` | 0.1217 | m | measured |
| `v_ref` | 0.0268 | m/s | measured |
| `a_ref` | 0.0619 | m/s^2 | measured |
| `r_ohm` | PENDING | ohm | NOT identifiable |
| `k_t` | 0.0500 | N.m/A | assumed (not identifiable) |
| `m_eff` | -11.5290 | kg | REJECTED (mechanical R2 = 0.030 < 0.5) |
| `f_c0` | 0.2596 | N | REJECTED (mechanical R2 = 0.030 < 0.5) |
| `b0` | -59.1689 | N.s/m | REJECTED (mechanical R2 = 0.030 < 0.5) |
| `F_c0 + b0*v_ref (cruise force)` | -1.3240 | N | REJECTED (mechanical R2 = 0.030 < 0.5) |
| `i_standing (drive current at rest)` | 0.4081 | A | measured |
| `i_moving (drive current while travelling)` | 0.8659 | A | measured |

**The mechanical regression is rejected outright: R² = 0.030.** It is not a noisy fit of the rig's mechanics, it is the wrong model for the rig's *drive* — `i = (m_eff/c_i)·a + (F_c0/c_i)·sgn(v) + (b0/c_i)·v` assumes current follows force, and a chopper-regulated stepper holds current almost flat while it travels (§3). The fitted `m_eff`, `F_c0` and `b0` above come out negative and physically meaningless, and they are printed **only** so that nobody re-derives them and believes them. Nothing downstream uses them: the fault gains are read off *ratios*, which is the whole reason the calibration is built on ratios rather than on absolute constants.

**Identifiability — the real files settle it, and the answer is narrow.** The mechanical regression `i = (m_eff/c_i)·a + (F_c0/c_i)·sgn(v) + (b0/c_i)·v + i₀` contains `k_t` only inside `c_i = η·k_t·G/r`, so from position and current alone **only the three ratios** `m_eff/c_i`, `F_c0/c_i`, `b0/c_i` are identifiable — a heavy leaf and a weak motor are the same dataset. `R` and `k_e` need the **voltage** channel (`V = R·i + k_e·ω + V₀`), and then `k_t = k_e` in SI closes the system. **The release has no voltage channel** — each matrix is exactly `[set point (mm), error (mm), current (A)]` (PDF §4) — so:

* **`R`, `k_e`, `k_t`: NOT identifiable.** Not "not yet": there is no observable in this dataset that separates them, and `fit_healthy` reports `k_t` as *assumed*, never fitted.
* **`m_eff`, `F_c0`, `b0`: identifiable only as ratios to `c_i`**, i.e. conditional on the assumed `k_t`. The physical columns in the table above are that conditional rescaling, which is why their status says so.
* **`b0`: not separable from `F_c0` either.** The rig runs two motion profiles whose cruise speeds differ by only ≈1.18× (needs ≥ 1.5), so only the combination `F_c0 + b0·v_ref` is well posed — and that combination *is* reported.
* **`stroke_m`, `v_ref`, `a_ref`: always identified** — they are read straight off the commanded profile, which the rig logs exactly.
* And the regression's own premise — current ∝ force — is not this drive's physics at all (§3), which is the deeper reason the mechanical numbers above are reported as a *fit of the rig's current*, not as the rig's mechanics.

### 3. The stepper caveat, quantified

The rig's motor is a **Nema 34 stepper** with 4.6 N·m holding torque, its current read by a Honeywell CSLA2CD Hall-effect sensor on the drive (PDF §2). `nebulax/sim/door.py` models a **PMDC** drive, where the quasi-static current is proportional to load force. A chopper-regulated stepper holds a commanded phase current instead, so a large part of what the sensor reads does not depend on the load at all.

That part is measurable, and it is big: the healthy rig draws **0.408 A standing still** against **0.866 A while travelling**, so only **53%** of the cruise current carries any load information. A faulty/healthy ratio of *mean* currents is therefore compressed towards 1 by that offset, and a gain read off the simulator's PMDC inversion curve with it is a **lower bound**.

Removing the standing current as an offset brackets the truth from the other side — reported, never adopted, because "the rest of the current is proportional to force" is itself an assumption about a drive this project does not model:

| lubrication stage | sim severity | raw ratio | offset-corrected ratio | `k_friction_c` from raw | from corrected |
|---|---|---|---|---|---|
| 1 | 0.50 | 1.164 | 1.321 | 0.379 | 0.743 |
| 2 | 1.00 | 1.241 | 1.472 | 0.279 | 0.547 |

Both ends of the bracket sit far below the plan's `k_friction_c = 2.0`, so the *direction* of the finding is robust even though its magnitude is not identifiable from this drive: **a seeded lubrication fault on a ball screw is a much milder force change than the plan's map assumes**. The release's own PDF says the same in words — "No dramatic changes were observed in the signals, mainly due to the inherent low friction of the ball-screw architecture" — which is why stage 2 exists at all (the nut seals were then bolted tighter to *create* friction).

**Which observables survive.** A ratio of *ripple fractions* does: `ripple_frac` is measured after detrending the current against **position**, which removes the standing component from the ripple itself, and the offset then survives only in the two means, where it cancels between faulty and healthy to within a few per cent. That is why `k_mis` is adopted below and `k_friction_c` is not. Position-derived observables (`err_spike_peak_m`, `pos_err_*`) are untouched by the drive question entirely — they just happen not to be invertible here (§5).

### 4. What the estimator can actually recover

`--self-check` generates a rig dataset **from the simulator at known gains**, writes it as `.mat` files **in the real release layout** (`Normal.mat`, `LackLubrication{1,2}.mat`, `Backlash{1,2}.mat`, `Spalling{4,8}.mat`, the release's own `<class><profile><level><load><rep>` variable names, resampled to the rig's 25 Hz), and runs this exact pipeline on it — discovery, channel extraction, segmentation, inversion. It is a recovery test of the estimator, never evidence about the real actuator, but it is the only thing that says how much of a measured ratio survives the trip. The six spalling stages it does not write also exercise the adapter's *missing file is a WARNING* path.

| gain | level | known | recovered | error |
|---|---|---|---|---|
| `k_friction_c` | 1 | 2.0000 | 1.9800 | −1 % |
| `k_friction_c` | 2 | 2.0000 | 1.9990 | −0 % |
| `backlash_m` | 1 | 0.0080 | 0.0086 | +7 % |
| `k_mis` | 4 | 0.1500 | 0.0854 | −43 % |
| `k_mis` | 8 | 0.1500 | 0.1019 | −32 % |

*(last `--self-check` run; re-run `python scripts/calibrate_door.py --self-check --no-doc` to refresh.)*

So the **friction gain is recoverable to ~2 %**, the **backlash dead band to ~7 %**, and the **misalignment amplitude only to ~40 %** at the rig's 25 Hz — the ripple fraction is the noisiest of the three observables and its map is the most curved, so `k_mis` should be read as *order 0.15*, not as three significant figures. Rows the sweep cannot bracket come back `PENDING` rather than extrapolated, which is exactly what happens to every real `backlash_m` row in §1.

### 5. Constants changed

| Constant | Old | New | Tag | Evidence |
|---|---|---|---|---|
| `DoorParams.obs_base_seed` | *(did not exist; baselines started at 0)* | **True** | calibrated | A zero-seeded asymmetric EWMA needs `obs_base_tau_s·ln(i_cruise/i_obs_delta_a)` to climb onto the cruise current, and until it arrives the actuator's **own** steady current reads as an abrupt rise. On `DoorParams.cranfield()` that climb is 0.42 s against a 0.25 s acceleration ramp, so over 69 cycles at constant severity the phantom-reversal rate was **100 % at friction s = 0.5 and s = 1.0** against a 1.4 % healthy baseline, and `closing_time` went **non-monotone** — 2.69 / **6.32** / **6.30** s at s = 0/0.5/1.0 — which makes it useless as a degradation feature. Seeding both baselines with the operating point at the instant the current path arms gives **1.4 % / 13.0 %** (the residual at s = 1.0 is real, not phantom) and a monotone **2.69 / 3.59 / 4.49 s**, with the current channel untouched (0.7615 → 0.7581 A, 1.1020 → 1.0994 A). On the 110 V door the change is a no-op: **0 of 69 cycles differ** at s = 0 and s = 0.5 and **1 of 69** at s = 1.0, with every median identical — it escaped only because its 0.50 s ramp happens to outlast its 0.34 s climb, a 0.16 s accident this removes. Pinned by `tests/test_sim_door.py::test_obs_base_seed_kills_the_phantom_reversal_on_the_cranfield_variant`. |
| `DoorParams.cranfield().mis_lambda_m` | 0.30 (inherited from the 0.725 m door leaf) | **0.005 (`CRANFIELD_SCREW_LEAD_M`, the ball-screw lead)** | **derived → measured** | λ = 0.30 m is **3× the rig's entire 0.10 m stroke**, so the position-periodic term degenerated into a monotone friction ramp: recovering the dominant wavelength from the simulated current gave **0.0976 m ≈ the stroke itself** and a ripple of 0.014 A (3.7 % of healthy cruise) — no periodic signature to match spalling against, and it also tripped the obstruction detector on 3 of 3 retries. At the screw lead the same estimator recovers **0.0050 m exactly**, 20 periods across the stroke, 0 reversals. The lead is already implicit in `pulley_r_m = lead/2π`; `CRANFIELD_SCREW_LEAD_M` now drives both so they cannot drift apart. **The real files upgrade this from a guess to a measurement**: the release's own 'Data description.pdf' section 2 names the screw as an *RM1605-C7 with 5 mm lead*, which is exactly the assumed value — the `UNVERIFIED` tag this row used to carry is retired. |
| `DoorParams.cranfield().pos_err_obs_m` | 0.020 (inherited) | **0.00276 (`OBS_POS_ERR_STROKE_FRAC × 0.10`)** | derived | The plan's 20 mm tracking-error trip is 2.76 % of the 0.725 m door leaf; inherited unscaled it is **20 % of the rig's 0.10 m stroke**, i.e. unreachable — the *simulated* rig's `pos_err_max` is 0.9 mm healthy and 2.6 mm at friction s = 1, so the third detection path was simply dead. Rescaling by the same stroke fraction puts it at 2.76 mm, above the worst simulated wear-driven error with ≈1.5× margin. **The real rig is not a check on this number and must not be read as one**: its own tracking error is far larger (median `pos_err_max` **3.3 mm healthy**, 3.5 mm at lubrication stage 1, 5.8 mm worst case over 7 790 strokes) because its set point steps straight onto full commanded speed with no acceleration ramp, so that error is the controller catching up with a discontinuity, not a fault. The simulator commands a trapezoid and has no such transient; comparing the two directly would be comparing a rig artefact with a fault threshold. (The rig's stroke is also 120 mm, not the 100 mm this variant models — see §6.) |
| `k_friction_c` | 2 | **unchanged (2); curve says 0.339** | measured, NOT adopted | The inversion is well posed: 2 of 2 stages (levels [1, 2]) land inside the monotone prefix of the `k_friction_c` sweep and read **0.339** off `i_cruise_mean`, 0.17× the plan's 2. It is **not adopted**: observable fails gate 2: `i_cruise_mean` is a ratio of mean drive currents and the rig's stepper puts a large standing current underneath both, so the inverted gain is a lower bound on the force gain, not a measurement of it. Removing the measured standing current as an offset (see §3) lifts the same inversion only to 0.547–0.743, still far under the plan's 2, so the *direction* of the finding is robust even though its magnitude is not identifiable from this drive — a seeded lubrication fault on a ball screw is a far milder force change than the plan's map assumes, which the dataset's own PDF anticipates ("No dramatic changes were observed in the signals, mainly due to the inherent low friction of the ball-screw architecture"). Adopting a lower bound would also drop the rig variant's cruise force at s = 0.5 to ≈ 19 N against its own 30 N obstruction trip, so `DoorParams.cranfield()` would stop exercising the obstruction path at all — the path `tests/test_sim_door.py::test_obs_base_seed_kills_the_phantom_reversal_on_the_cranfield_variant` exists to pin. |
| `k_friction_b` | 1.5 | **unchanged (1.5); curve says 0.254** | measured, NOT adopted | The inversion is well posed: 2 of 2 stages (levels [1, 2]) land inside the monotone prefix of the `k_friction_c` sweep and read **0.254** off `i_cruise_mean`, 0.17× the plan's 1.5. It is **not adopted**: tied to `k_friction_c` by the plan's b:c ratio and blocked with it; `b0` is not separately identifiable on this rig anyway (one cruise speed per profile). Removing the measured standing current as an offset (see §3) lifts the same inversion only to 0.41–0.557, still far under the plan's 1.5, so the *direction* of the finding is robust even though its magnitude is not identifiable from this drive — a seeded lubrication fault on a ball screw is a far milder force change than the plan's map assumes, which the dataset's own PDF anticipates ("No dramatic changes were observed in the signals, mainly due to the inherent low friction of the ball-screw architecture"). Adopting a lower bound would also drop the rig variant's cruise force at s = 0.5 to ≈ 19 N against its own 30 N obstruction trip, so `DoorParams.cranfield()` would stop exercising the obstruction path at all — the path `tests/test_sim_door.py::test_obs_base_seed_kills_the_phantom_reversal_on_the_cranfield_variant` exists to pin. |
| `backlash_m` | 0.008 | **unchanged (0.008)** | measurement rejects the map | **The observable moves the wrong way.** The simulator's `backlash_m` map can only *raise* `err_spike_peak_m`, but the measured ratio is 0.977 at stage 1 — below 1, i.e. the fault made the rig *better* by this measure — while another stage sits just above it (1.029). Reading a gain off a 2–3 % excursion whose sign flips between the two stages would be inverting noise, so the sign gate refuses it and the plan's value stands. Physically this is credible rather than surprising: the rig's balls were swapped into an **anti-backlash** nut, so a smaller ball adds play *and* relieves preload, and the reversal error here is dominated by the controller's response to a step-onto-ramp set point rather than by the dead band. This dataset cannot set δ through this observable — a finding, not a gap. (The curve would have returned 0.000785 from the one stage that happened to land inside its monotone prefix.) |
| `DoorParams.cranfield().k_mis` | 3 (class default, inherited) | **0.15** | calibrated | Read off the `k_mis` inversion curve at `ripple_frac`: 4 of 8 spalling stages (levels [3, 4, 7, 8]) fall inside the curve's monotone prefix and their median gain is **0.15**, 0.050× the plan's 3. The measured `ripple_frac` ratio grows 1.04 → 1.45 from stage 1 to stage 8, while the plan's 3 would have put it at **11×** — the plan's position-periodic amplitude is an order of magnitude too large for a seeded screw spall. This observable **survives the stepper caveat** (§3): `ripple_frac` is measured after detrending the current against position, so the drive's 0.41 A standing current is removed from the ripple itself and survives only in the two means, where it cancels between faulty and healthy to within a few per cent. **Applied** to `DoorParams.cranfield().k_mis = 0.15`; this run re-read 0.15 off the curve with that value already in place, so the calibration is a fixed point rather than a one-way edit. Written onto `DoorParams.cranfield()` **only** — the shipped 110 V door was not fitted on a ball-screw rig and does not move (pinned by `tests/test_sim_door.py::test_the_shipped_door_is_barely_touched_by_the_cranfield_calibration`). |

### 6. Notes

* Parsed 13 of the 13 release files (779 test recordings, 7,790 strokes).
* `Backlash1.mat` holds **59** matrices, not 60: `backtrap1st40kg` has 9 repetitions in the release (`backtrap1st40kg2` is absent). The adapter parses what is there and records the count in `meta`; no ratio is sensitive to one missing repetition out of 590 strokes.
* The rig's stroke is **120 mm** (PDF §2) while `DoorParams.cranfield()` models 100 mm, and its commanded cruise speed is 26.8 mm/s against the variant's 50 mm/s. Both are *geometry*, not fault-map gains, so they are outside what this calibration is allowed to change; they are recorded here because every absolute (not ratio) comparison with the rig inherits them.
* Timing carries no fault information on this rig: with the ratios formed **within a motion profile** the stroke-duration ratio is 1.000 at every fault and stage and the cruise-speed ratio stays inside ±6 %. It is a position-commanded stepper following a fixed profile with force headroom, so the whole fault signal is in the current and the tracking error — the experimental form of rail_phm 1.1 [R64][R72].
* The **mean current does not rank the faults the way the door model would**. The strongest current signature in the release is `backlash` stage 2 at **1.339×** healthy — above every lubrication stage — because smaller balls in a ball screw add rolling friction as well as play; while spalling *lowers* the mean current (down to 0.870× at `misalignment` stage 7) at the same time as it raises the position-periodic ripple that `k_mis` is read off. Neither is a defect in the data: it is why each fault's constant is read off its **own** observable and the others are printed as diagnostics only.
* Pooling the two motion profiles was a real trap and `class_summary` now avoids it: because `Backlash1.mat` is one repetition short of 60, a pooled median tipped off the 5 s trapezoidal profile onto the 6 s sinusoidal one and reported a spurious **1.077** duration ratio for backlash stage 1. Every ratio is now formed against the healthy strokes of the *same* profile and then median-combined.

Plots: `results/sim_checks/cal_door_healthy_fit.png`, `cal_door_fault_ratios.png`, `cal_door_backlash_width.png`, `cal_door_spalling_ripple.png`.

<!-- END: calibrate_door -->
