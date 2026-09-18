# PS3 research addendum — door, rail corrugation, ACV, SHM

Companion to `model_ladder.md` (which covers the door *classifier* family and nothing else in
PS3's world), `references.md` (every `[Rnnn]` resolves there; this pass appends **R231–R314**) and
`configs/model_ladder.yaml` (`ps3:` block, machine-readable mirror of the tables below).

**How to read a row.** `MUST` = in the ladder table by the ladder milestone, implementable by one
person in the stated hours. `NICE` = only when the MUST rows are green. `SKIP` = named and excluded
on the record, so "did you consider X?" is answered by a sentence. `hrs` is implement-hours for
*this* pipeline, not to reproduce the paper.

**Verification status.** Every citation passed the W4 citation-verifier sweep (~90 identifiers via
Crossref/OpenAlex/Semantic Scholar/arXiv). Paywalled numbers carry `NUMBERS UNVERIFIED` in
`references.md` and are not quoted here as results. **One item was refuted and must never be
cited**: the *Applied Energy* 402 (2026) few-shot HVAC-FDD DOI and its 73.77 % / 67.22 % F1 figures
— the DOI resolves to an unrelated fuel-cell paper. It is absent from `references.md` by design.

**The rule that governs every row.** Scalers, thresholds, templates, band edges, rule weights,
signs, bias corrections, feature selection and augmentation are fitted **inside the training
fold**; model selection is nested or on a frozen outer split; augmented children stay in their
parent's fold; held-out files are never augmented. Augmenting before the split is a *named*
leakage class [R303], of the same family that inflates a genuine 20–60 % to 99.9 % on identical
data [R220], and the group constraint is the same one the bearing literature already forces
[R115].

---

## 1. Door — segment + classify, IoU-weighted F1

### 1.1 What practitioners build first
Segment, featurise per cycle, classify with a tree ensemble on interpretable features — in that
order, most of the effort in the features. The published door pipeline works **only after** the
current trace is split into accelerate / constant / decelerate regimes, and its authors argue the
per-fault health index that falls out beats a CNN's accuracy operationally [R64]. The 2026
subway-door reference pipeline is a stacking ensemble (RF + XGBoost → calibrated logistic
regression) with Spearman+VIF screening, an F1-max threshold chosen on validation and **PR-AUC
reported alongside ROC-AUC** under 7.3:1 imbalance [R68]. A DWT detail-coefficient detector on the
same hardware family reports >96 % with three features [R66][R67]. Peer normalisation across a
vehicle's doors is the published answer to "distributions differ among doors", the info kit's core
problem 2 [R78].

### 1.2 Physics-grounded features
Abnormal resistance is mechanical work the motor must supply: at fixed commanded profile it shows
as **excess current at a given position**, not as a global mean shift. So: per-regime current
statistics on the accelerate / cruise / decelerate split [R64]; `i_mid` over 15–85 % travel;
**back-EMF residual** (the controller publishes back-EMF and position, so `V − i·R − k·ω` is a
physically-typed residual, not a learned one); mechanical energy proxy `Σ i·Δpos`; position of the
current peak; distance from a fold-fitted current-vs-position template; cycle duration against the
commanded open/close time registers; DWT detail-band L1 norms at the two coarsest levels plus the
starting-current peak [R66]. Separate Open and Close models — they are different mechanisms with
different gravity and latch terms.

### 1.3 Ladder
| Tier | Row | Why + cite | hrs | Verdict |
|---|---|---|---|---|
| Segmenter | Gap segmenter (`dt > 0.5 s`), ms-exact, never trimmed | 109/110 train cycle starts coincide with a gap; the metric charges for boundaries | 0.5 | **MUST** |
| Segmenter | Command/position state machine with hysteresis, fallback | Gaps are a dataset artefact, not a controller guarantee; must be tested on Train with gaps removed | 2 | **MUST** |
| Segmenter | ClaSP binary segmentation of the cycle interior | Hyper-parameter-free, so nothing to calibrate under the fold-local rule; gives the learned alternative to the fixed three-regime split [R176][R177][R64] | 2 | NICE |
| Features | Per-operation, fold-fitted baseline + robust peer z | Cancels per-door distribution shift without a labelled normal period [R78] | 1 | **MUST** |
| Model | Logistic regression on physics features | Honest floor; also the calibrated head of the stack | 1 | **MUST** |
| Model | `LGBMThreeRegime` on regime-named features | The published traditional door pipeline, and the only arm that yields a trendable per-fault index [R64] | 2 | **MUST** |
| Model | Stacking RF + XGBoost → calibrated logreg, PR-AUC reported | The 2026 subway-door reference result under comparable imbalance [R68] | 3 | **MUST** |
| Model | `multirocket_ridge` on the raw 50 Hz cycle | MultiRocket+Hydra is level with HC2 at a fraction of the compute [R43][R47][R48] | 1 | **MUST** |
| Model | QUANT | SOTA interval method in <15 min on one core; quantile-of-a-stroke-quarter is judge-legible [R49] | 1 | NICE |
| Model | LITETime | InceptionTime accuracy at 2.34 % of the parameters; the single deep row the plan allows [R53] | 3 | NICE |
| Model | HIVE-COTE 2.0 | **SKIP** — ~340 h to train over the benchmark; cite as the ceiling not bought [R52] | — | **SKIP** |
| Model | Audio MNPE + SVM | **SKIP** — no microphone channel in the door controller stream [R75] | — | **SKIP** |

### 1.4 Validation pitfalls
The metric is **end-to-end**: a correctly timed segment with the wrong label scores exactly as a
miss *plus* a false positive, and sloppy boundaries shrink both soft-recall and soft-precision.
(i) Tune the threshold on **fold IoU-F1 of the full pipeline**, never on cycle-level F1 — they
disagree, because over-segmentation is free under one and charged under the other. (ii) Matching is
one-to-one and greedy by IoU, so never emit overlapping candidates hoping one lands. (iii)
Oracle-segment F1 is a **diagnostic**, not a result. Blocks must be contiguous in time and the
block-edge cycle handled explicitly. The "within-block batch normalisation" variant is
**transductive** — it reads the held-out block's own statistics — so it is an ablation row with
that word attached, never the headline [R303].

---

## 2. Rail corrugation — 3-class, macro F1

### 2.1 What practitioners build first
Resample from time to **distance** with the tacho, take the PSD in the **wavelength** domain,
integrate into 1/3-octave wavelength bands, normalise by `v²`, then classify with something cheap.
The most useful data point in the sweep: on a balanced 639-sample metro corrugation problem with 26
hand features, BPNN (95.31 % acc / 96.17 % F1), SVM, LSTM and RF land within 1.6 points of each
other [R232] — **the representation, not the classifier, is where the accuracy is**. The closest
published setting to ours is continuous subway-line corrugation monitoring from in-service axlebox
acceleration [R231]; the band structure is the IEC 61373 1/3-octave ASD layout [R233], i.e.
EN 15610's wavelength presentation [R250].

### 2.2 Physics-grounded features
Wheel Ø 0.85 m → circumference **2.670 m**; 90 teeth → **29.67 mm per pulse**, so the **cumulative
pulse count is a distance encoder — integrate it, never differentiate it**. At 10 kHz and 67 km/h
the spatial sample is 1.86 mm, so short-pitch corrugation (λ = 25–80 mm, 30–60 mm on metro tangent
track [R248]) is resolved at every speed in range. Its passing frequency `f = v/λ` sweeps
**35–744 Hz** over 0–67 km/h: **a fixed-Hz band cannot track it**. ABA amplitude scales as `v²` for
fixed roughness, so normalise by `v²` [R237] or the classifier learns speed. Feature set:
1/3-octave wavelength bands over ~8–500 mm per channel [R233][R239][R250]; per-side aggregates
(median and 90th percentile over 32 odd = Side I, 32 even = Side II); the **Side I − Side II
contrast in dB**, which cancels speed, track type and sensor gain as common mode; vibration and
shock channel types kept separate, since different axes carry complementary mechanisms [R236]; and
**impulsive-vs-sustained discriminators** — spectral flatness, harmonic-peak prominence, envelope
duty cycle, within-side channel agreement — because wheel flats, switches and structural resonances
occupy the *same* wavelength range and are the documented false-positive source [R235][R246].
Corrugation is sustained, periodic and side-wide; a wheel defect is impulsive and one channel.

### 2.3 Ladder
| Tier | Row | Why + cite | hrs | Verdict |
|---|---|---|---|---|
| Transform | Tacho-driven resampling to constant Δx ≈ 1 mm | Computed order tracking with a real keyphasor; speed variation otherwise smears the spectrum [R245][R239] | 3 | **MUST** |
| Transform | Welch PSD → 1/3-octave **wavelength** bands, 8–500 mm | The domain's own presentation; non-arbitrary band edges [R233][R250] | 1.5 | **MUST** |
| Transform | `v²` normalisation | Makes the roughness→acceleration relation speed-independent [R237] | 0.5 | **MUST** |
| Features | Side aggregates + Side I − Side II contrast (dB) | The label *is* a side; common-mode removal in one step (structural; supported by [R235]) | 1 | **MUST** |
| Features | Impulsive-vs-sustained discriminators | Stops a flat or a switch scoring as corrugation [R235][R246] | 1.5 | **MUST** |
| Features | 26-feature time/frequency/wavelet set, Hz bands replaced by wavelength bands | Steal the list, not its fixed 250–2000 Hz bands [R232] | 1 | **MUST** |
| Model | Hierarchical: (a) present/absent, (b) which side | 38 positives support one binary decision far better than three classes | 0.5 | **MUST** |
| Model | Logistic / LightGBM on ~60 band features | Honest baseline; classifier choice is worth ~1.6 points here [R232] | 1 | **MUST** |
| Model | `multirocket_ridge` on the resampled, `v²`-normalised, side-aggregated 2-channel signal | ROCKET+RidgeClassifierCV is the measured default on short accelerometer windows (0.77 balanced acc vs 0.65 hand-crafted) [R249][R43][R47] | 1.5 | **MUST** |
| Ablation | Tree on **Hz** bands + speed as an explicit input | The published counter-design: give the model speed instead of normalising it away [R234] | 1 | **MUST** |
| Model | Comb filter at multiples of 1/2.670 m⁻¹ (wheel vs rail roughness) | The rigorous "is it the rail or the wheel?" answer [R238] | 4 | NICE |
| Model | 1D-CNN with speed as auxiliary input | Cheapest on-task deep arm; note its >95 % is a scaled rig [R240] | 4 | NICE |
| Model | log-`v` concatenated to the winning feature vector | 2-line borrowing from the condition-on-speed school [R242] | 0.25 | NICE |
| Model | VMD / CEEMDAN / EWT / SPWVD decompositions | **SKIP** — per-record, parameter-heavy, slow; buy nothing a wavelength PSD does not, at n=272 [R247] | — | **SKIP** |
| Model | Full model-based roughness inversion | **SKIP** — needs track receptance / rail-pad stiffness we do not have; a 20 % pad-stiffness error moves the answer by ~3.5 dB (`NUMBERS UNVERIFIED`) [R238][R239] | — | **SKIP** |
| Model | MoCo / contrastive pre-training on the 234 Normals | **SKIP** — right shape for 14 labels, but 234 unlabelled records are far too few and it is a full day [R244] | — | **SKIP** |
| Model | Simulator pre-train → few-label fine-tune | **SKIP** — no corrugation simulator exists here and building a vehicle–track model is not a 12 h job [R243] | — | **SKIP** |

### 2.4 Validation pitfalls
**The speed confound is the whole problem**: Normal spans 0–65 km/h, faults 35–67, so a speed-only
classifier scores well and means nothing. Report all-files macro F1 **and** speed-matched
(v ≥ 35 km/h) macro F1, plus a no-speed-feature ablation; `v < 20 km/h → Normal` is a documented
dataset shortcut, not a model. With 14 Side I examples, 5-fold leaves ~2.8 per fold and macro F1 is
dominated by that class: report mean ± sd over 3 seeds, never a point estimate. Add the
contiguous-filename-block and leave-speed-range-out stress splits — session-leaking splits inflate
exactly this kind of result [R58] — and fingerprint files for duplicates. Never normalise per file
with statistics pooled across folds. **Calibrate expectations**: a 2026 railway-vibration study with
a proper held-out split reports 81.5 % accuracy and ROC-AUC ≈ 0.94 over 21 classifiers, feature
standardisation decisive [R241]; the 95–98 % figures elsewhere are rigs, simulations or
sim-pretrained transfer [R240][R243]. **0.70–0.85 macro F1 with visible spread is the honest
target.**

---

## 3. ACV — rank the leaking car, linear rank decay

### 3.1 What practitioners build first
Peer-fleet residuals with a signed fault index. The blueprint is an electric-bus AC fleet study
(38 units, 12 technician-verified faults including 5 refrigerant undercharges): assume most peers
are healthy, regress each feature on operating conditions **per peer**, predict the target unit from
its peers' models, take the **median** of peer predictions (robust up to 1/3 faulty training units),
then aggregate signed residuals as `I = Σ_k sign_k · (R_k / IQR(R_k))² · IQR(SND)`, where `sign_k`
is the *known thermodynamic direction* of that feature under undercharge; 11 of 12 faults were
correctly named [R251]. Here the peer group is free and perfectly matched: 8 cars, one train, same
timestamp, weather and schedule. The method descends from the founding temperature-only residual +
directional-rule work, which detected **≈5 % refrigerant loss** from six temperatures [R252] — the
upper bound of what rich instrumentation buys. Domain framing: the only metro-train-AC FDD paper
found is simulation-based with a richer sensor set [R258]; fleet information beats a unit's own
short history [R261].

### 3.2 Physics-grounded features
PS3 exposes **no pressures, no superheat/subcooling, no power and no supply-air temperature**, so
the whole virtual-charge-sensor branch is inapplicable [R262]. What survives, in the order ORNL's 12
field tests over five undercharge levels put them: **runtime fraction rises monotonically with
undercharge**, supply-air temperature rises (unavailable to us), and **indoor air temperature alone
barely sees low-intensity undercharge — unmet hours appear only at 40–50 % charge loss** [R253]. So
build, per car on gated rows: share of cooling time in Full and Half Cooling; mode-transition count
per hour and mean on-time per cooling episode [R254]; pull-down slope `d(T_indoor)/dt` over the
first minutes of each Full-Cooling episode and time-to-setpoint; steady-state shortfall
`T_indoor − T_control` and unmet degree-minutes in the high-ambient tail [R253]; and the
`T_indoor − T_outdoor` lift, the only capacity proxy left. Write the **fault × feature direction
matrix** before modelling and check no other cause shares the signature — `load_halved`, a stuck
door, a blocked filter and a solar-exposed end car all push runtime fraction the same way, which is
precisely the decoupling objection [R263][R260].

### 3.3 Ladder
| Tier | Row | Why + cite | hrs | Verdict |
|---|---|---|---|---|
| Gate | Validity flag, mode mask, `load_halved` split, rolling-σ steady/pull-down gate | Without a transient gate, cars starting to cool at different times generate residuals unrelated to refrigerant; gate on *all* features, not a favourite pair [R259] | 1 | **MUST** |
| Features | Duty-cycle / runtime-fraction family | The one ORNL "key variable" PS3 still exposes [R253] | 1.5 | **MUST** |
| Features | Pull-down slope, time-to-setpoint, cycle count | Cooling-cycle duration is the workhorse when you have only thermostat-grade signals [R254] | 1.5 | **MUST** |
| Features | Unmet degree-minutes above setpoint, high-ambient tail | The confirming, late-stage feature — never the primary one [R253] | 0.5 | **MUST** |
| Model | Robust peer z (median/MAD of the 7 siblings within ambient bins) | Cheap form of the peer-median residual; the existing `features.peer_normalise()` already does it [R251][R78] | 1 | **MUST** |
| Model | Per-peer ridge/GPR on `(T_outdoor, T_control, load_halved)` → median peer prediction → residual | The published form; median, not mean, because one peer may be the faulty car [R251] | 2 | **MUST** |
| Model | Signed-squared-IQR fault index + direction matrix | Turns 8 residual vectors into one ranking with a stated physical sign per term [R251][R252] | 1 | **MUST** |
| Model | Per-position offset subtraction (end vs middle cars) | End cars carry more solar and door load; if the faulty-car index is not uniform across the 6 cases, peer exchangeability is violated [R256] | 0.5 | **MUST** |
| Model | Modified (autocorrelation-corrected) Mann-Kendall on each car's peer-normalised index | A cheap tie-breaker between two cars at similar levels; 30 s telemetry is heavily autocorrelated, hence *modified* [R266] | 1 | NICE |
| Model | Within-system-over-time comparison alongside between-system | The population-benchmarking method's second half, which partly compensates for having 7 peers rather than thousands [R256] | 1.5 | NICE |
| Model | Dry-run on the LBNL labelled FDD datasets | The only open ground-truthed HVAC fault data; validates the direction matrix before touching 6 precious cases [R264] | 2 | NICE |
| Model | Virtual refrigerant charge sensor | **SKIP** — needs liquid-line/suction-line temperatures and subcooling that this telemetry does not contain [R262][R257] | — | **SKIP** |
| Model | 3R2C grey-box + EKF identification | **SKIP** — non-convex with many local optima, needs a model-selection stage per vehicle; the 7 peers do the same job for free [R254][R255] | — | **SKIP** |
| Model | Supervised classifier / few-shot metric learning on 48 rows | **SKIP** — 6 cases × 8 cars with one positive each; metric learning presupposes more labelled episodes than exist, and targets fault *typing* rather than localisation [R268] | — | **SKIP** |
| Model | Physics-simulation hybrid (rail HVAC FDD + RUL) | **SKIP** — right vehicle, wrong fault (air filter) and a physics model of the ACV cycle is well beyond budget; cite for *why* train-HVAC fault data is scarce [R267] | — | **SKIP** |

### 3.4 Validation pitfalls
**Six cases is the entire CV.** Leave-one-case-out with `rank_decay` is both the selection set and
the reported number, so anything chosen on it is optimistic: freeze the rule, its weights and its
signs on the 5 training cases before touching the 6th, and **label the six-case result
exploratory**. The chance floor is exact and must appear in the table: a uniformly random ranking of
8 cars scores **0.5625**; correct-first is 1.000, second 0.875, third 0.750. At or below 0.5625 is a
bug, not a result. Never use a car-ID prior. One nuance the fold-local rule does *not* forbid:
**peer normalisation at test time is legitimate**, because it reads only the held-out file's own 8
cars and no labels — what must be frozen is the feature set, the signs and the weights. A leak over
3-4 days is a **soft** fault: expect a level difference against peers, not a step [R265]. Expect
to catch advanced leaks and to be near chance on early ones: thermostat-grade methods detect a 40 %
undercharge at ~82 % in simulation and **70.6 % on real faulted-equipment data** [R254][R255],
against ~5 % with refrigerant-side instrumentation [R252]. That asymmetry is why rank decay, not
top-1, is the thing to optimise.

---

## 4. SHM — cumulative fatigue damage regression, max(0, 1 − MAPE)

### 4.1 What practitioners build first
Not a model — a diagnostic. The blocker is that plain rainflow + Miner on the given channel does
not reproduce the organiser label at any exponent, while peak-to-peak correlates at Spearman 0.96.
That pattern is physics, not triviality: for a narrow-band Gaussian process the closed-form
Palmgren–Miner damage is `D = (ν0⁺·T/C)·(√2σ)^k·Γ(1+k/2)` [R269], so **`log D` is affine in
`log`(any amplitude scale)** — ranking well is guaranteed, hitting MAPE is not. Spend the first 90
minutes plotting damage against the label on log–log axes for all 64 files, across rainflow
variants, residue conventions and an exponent grid. The shape of that plot separates the published
mechanisms by which a rainflow+Miner label fails to reproduce: **residue handling** (discard /
half-cycles / full / repeat-history — the choice materially changes the sum and common conventions
are non-conservative [R270]); the wrong exponent; an **SDOF/FDS transform** applied before counting
[R274]; a knee or two-slope S-N curve (the Haibach `k* = 2k−1` rule is `SECONDARY-SOURCE`
and must not reach a deliverable unverified [R291]); a **mean-stress correction** (Goodman / Morrow / SWT) the
organisers applied and you did not [R269]; or damage reported at a **different location** than the
published gauge [R286][R287]. Parallel-but-offset means a free constant (fit it); a different slope
means the wrong `k`; scatter tracking the irregularity factor points at FDS; scatter tracking
kurtosis points at a non-Gaussian correction [R277].

### 4.2 Physics-grounded features
`1 − MAPE` is a relative-error metric, so **fit `log D` by least squares** — nearly the right loss
and far better conditioned than fitting `D` across a 30× range with a bimodal label. Features:
RMS (√m0), p2p, p99−p01, rainflow-range percentiles; kurtosis and skewness [R277]; spectral
moments m0/m1/m2/m4 with `ν0⁺`, `νp`, the irregularity factor and the Vanmarcke bandwidth
parameter; closed-form spectral damage at 3–4 candidate `k` from Dirlik, Tovo–Benasciutti and
narrow-band [R269][R275][R276]; and an 8–12 band **fatigue damage spectrum** (SDOF filter bank,
Q ≈ 10, octave-spaced, rainflow per band) [R274]. Damage depends on **phase**, not only on the
amplitude spectrum, so PSD-only features are known to be insufficient on their own [R284].

### 4.3 Ladder
| Tier | Row | Why + cite | hrs | Verdict |
|---|---|---|---|---|
| Diagnostic | 3-point (ASTM E1049 [R290]) vs 4-point rainflow × residue conventions × exponent grid × {original, elementary, knee} Miner × {none, Goodman, SWT} | The highest-yield 90 minutes in the task; separates six published failure mechanisms [R270][R269] | 1.5 | **MUST** |
| Physics | Fold-calibrated rainflow + S-N (scale and exponent fitted on the training fold) | The honest physics row the ML must beat; the free `C`/`T` constant is harmless, so fit it | 1.5 | **MUST** |
| Physics | Dirlik and Tovo–Benasciutti spectral damage, hand-implemented | Best spectral methods sit within ~7 % relative life error for steel (k = 3.324); no installs, and the formulas are short [R269][R275][R276] | 2 | **MUST** |
| Features | ~30 log-space amplitude + spectral-shape features | `log D` is affine in `log`(amplitude scale) [R269] | 2 | **MUST** |
| Model | Ridge / Lasso on `log` damage, LOO over 64 files | At n = 64 anything deeper overfits; the wind-DEL literature gets R² = 0.988 from **three scalars**, and a TCN on the raw series adds ~0.003 [R272] | 2 | **MUST** |
| Diagnostic | Noise floor: bootstrap damage over sub-windows of one file | Damage from one realisation is a draw from a distribution [R271]; RFC damage converged within 2 % of the 1 h value only after ~2 s (k = 3.3), ~150 s (k = 7.3), ~2600 s (k = 11.8) [R269]; window choice alone moves long-term mean damage by up to 30 % in the closest published study [R289]. If the floor is 8 %, `1 − MAPE = 0.92` is a win and 0.98 is chasing noise | 1 | **MUST** |
| Features | 8–12 band FDS (SDOF filter bank) | **MUST if** the diagnostic shows scatter tracking the irregularity factor — the single most likely fix if the label was SDOF-transformed [R274] | 3 | **MUST\*** |
| Features | Kurtosis / non-Gaussian correction | One feature, and the diagnostic tells you whether it matters [R277] | 1 | NICE |
| Model | Small NN / GBM on the same ~30 features as an ablation row; GP variant if cheap | The surrogate-for-rainflow family [R273][R272][R280], and the route to a calibrated error bar on a damage number [R281] | 3 | NICE |
| Framing | "Damage-consistency" calibration coefficients per load amplitude | The closest published precedent to fitting a correction so a spectrum reproduces a reference damage; also the rail-engineering vocabulary (EN 13749 / EN 12663) for the write-up [R283][R282][R285] | 2 | NICE |
| Model | 1D-CNN / LSTM on 581k raw samples | **SKIP** — 64 training examples, and the time-series network buys ~0.3 % over three scalars in the closest published comparison [R272] | — | **SKIP** |
| Model | Multiaxial / critical-plane criteria | **SKIP** — one channel, so there is no second channel to be multiaxial with | — | **SKIP** |
| Model | Rainflow-matrix extrapolation to longer histories | **SKIP** — we are not extrapolating to a longer life; the record *is* the label's domain [R279] | — | **SKIP** |
| Library | `ffpack` | **SKIP** — GPL-3.0 would infect a shipped deliverable (and no installs are permitted anyway) [R269] | — | **SKIP** |

### 4.4 Validation pitfalls
**MAPE on small damages is the trap.** The labels run 0.029–0.928, bimodal with median 0.10, so an
absolute error of 0.03 is a 100 % relative error on the smallest file and 3 % on the largest: the
score is decided almost entirely by the low-damage mode. Report MAPE split by mode
(< 0.25 / ≥ 0.25) alongside the headline, optimise in log space, and clip predictions to positive
finite values. Bias correction — including the log→linear retransformation correction — is a fitted
quantity and belongs **inside** the fold, as do the S-N exponent and the scale constant. Use LOO
plus repeated 5×10-fold with nested inner selection, and pick the winner on the nested CV, not on
LOO. Quote the bootstrap noise floor next to the score, so "we stopped here" is measured rather
than a shrug.

---

## 5. Augmentation — must / nice / skipped, per subsystem

Set expectations first. Every headline augmentation number in the literature is measured on deep
nets trained from scratch; through a **ROCKET + ridge** pipeline — which is what three of our four
subsystems actually run — the measured gain is **≈ +1.55 % mean relative accuracy on 10 of 13 UEA
datasets** [R296], and **+0.11 % to +1.92 %** on a real operational dataset [R294]. The gain is
largest exactly at our sample sizes (the correlation between gain and training-set size is
negative [R292]), and it is still small. Class weighting is probably a **no-op** on a tuned
boosting stack [R302] — the CNN-oversampling result that says otherwise is image evidence and does
not govern ROCKET+ridge or LightGBM [R301] — and that ablation is worth publishing as a negative
result rather than quietly running SMOTE. **The leakage-safe split is worth more than every
augmenter combined** [R303][R220][R115].

| Subsystem | MUST | NICE | SKIP (reason) |
|---|---|---|---|
| Door | Window warping and magnitude warping / jitter on the **raw-cycle rows only**; window warping is the top-ranked general-purpose transform for VGG/ResNet/LSTM and slicing second [R292][R293]. Report the with/without ablation either way. | wDBA barycentre synthesis for the sparse abnormal class (110 cycles is genuinely sparse), noting it generates low-diversity samples [R297]. Physical-consistency pass rate on injected cycles, the way the door reference pipeline reports 98.8 % [R68] — physics-parameterised augmentation with a measured consistency rate is the accepted PHM answer to imbalance [R306] and is the differentiating contribution, not any generic transform. Cutout/cutmix/mixup are cheap extra arms, but quote the *median* of their 1–45 % range, not the maximum [R309]. | **Rotation/flip** (door current has a physical sign; decreased accuracy for all six architectures tested) and **permutation** (cycle phases are ordered: unlock → open → dwell → close → lock) [R292]; the permutation-is-good result [R295] is the smaller study and loses on our ordered data. **Window slicing** — the fault signature lives in one phase, so slicing is label noise [R293]. |
| Rail | **Mirror (odd↔even side swap with label swap)** — exact, free, ~30 lines, turns 14/24 into 38/38 and imposes the left-right equivariance the sensor layout demands. It is not from a paper; it follows from the geometry, and it dominates everything generic. | Sub-crops 0.6–0.8 s of the distance-resampled signal (the label *is* sub-window stationary here) [R293]; SpecAugment-style band masking on the wavelength spectra, with mask widths kept small relative to the corrugation band [R298]; per-side sensor dropout, which doubles as the "can we drop a sensor?" ablation [R57]. On a scalogram route, online generalised-Morse wavelet parameter variation is the no-extra-generator alternative [R305]. | SMOTE on spectra and all GAN/diffusion synthesis [R307][R308] — with 14 positives a generator memorises, and the field needed a benchmark precisely because generated-series quality is contested and rankings unstable across measures [R304]. Rotation, permutation [R292]. Self-supervised pre-training [R311] is the structurally correct answer to label scarcity but is a stretch row, not an augmenter, and degrades at very small label counts. |
| ACV | **None.** The label is one car per case; there is no transform of an 8-car train that preserves "car 3 is the leaking one" without also fabricating thermodynamics. | Peer-order shuffling as a *test-time* symmetry check (the ranker must be invariant to car ordering), not as training data. | Every generic transform: 6 cases is not a sample size an augmenter fixes, and synthetic cases would be scored by the same 6-fold CV that selects the rule [R303]. |
| SHM | **C-Mixup**: mix *whole files* in log-feature space with pair-sampling weighted by **label similarity**, so the interpolated label is meaningful; vanilla mixup on regression labels "can result in arbitrarily incorrect labels" and cropping does not preserve damage at all [R299]. One ablation row, generated inside the fold. | Anchor Data Augmentation as the alternative of comparable strength, **only if** C-Mixup's kernel bandwidth proves fiddly — they solve the same problem and running both wastes the budget [R300]. | Cropping / slicing (damage is cumulative over the record, so no sub-window carries the label) [R293]; TVAR-style generative synthesis [R313] (unverified venue, n = 5 effect sizes); all GAN/diffusion [R307][R308][R304]. |

**The fold-only rule, concretely.** Pass a **parent-id group vector** through the augmenter so
`GroupKFold` / `LeaveOneGroupOut` can never separate a parent from its children [R115][R303]; fit
the augmenter's own statistics (warp envelopes, mixup bandwidths, band-mask widths) on the
training fold; never augment a held-out file; and report every augmented row **beside** its
un-augmented twin so the table shows what augmentation bought.

---

## 6. What this addendum changes vs the W4 plan

The list the four implementers must read before the ladder milestone.

| # | Subsystem | Change | Why + cite |
|---|---|---|---|
| 1 | Rail | The plan's 7 log-spaced bands over **20–5000 Hz** stop being the feature path | The passing frequency sweeps 35–744 Hz over 0–67 km/h, so fixed-Hz bands track speed, not roughness. Keep them **only** as the counter-design ablation that hands speed to the model [R234]; distance → wavelength is the MUST path [R232][R233][R239][R250] |
| 2 | Rail | **Add `v²` normalisation** as its own MUST row and its own ablation | Not in the plan, and the highest-leverage single line in the task [R237] |
| 3 | Rail | **Hierarchical** present/absent → which-side, replacing the flat 3-class LightGBM | 38 positives support one binary decision far better than three classes |
| 4 | Rail | **Add impulsive-vs-sustained discriminators** | Wheel flats, switches and resonances occupy the same wavelength band and are the documented false-positive source [R235][R246] |
| 5 | Rail | Expectation band **0.70–0.85 macro F1 with visible spread**, stated up front | Anchored on a held-out railway-vibration result of 81.5 % [R241], not on 95 %+ rig/simulation figures [R240][R243] |
| 6 | ACV | **Reorder the feature priority**: duty-cycle and pull-down become primary, setpoint shortfall becomes confirming | The plan's baseline leads with `indoor − peer-median`, but field tests say indoor air temperature is the *weakest* undercharge observable while runtime fraction rises monotonically [R253] |
| 7 | ACV | **Upgrade the aggregation** from a mean of deltas to median-of-peer-predictions residual + signed-squared-IQR fault index, with a written feature-direction matrix | The published form, robust to a faulty peer, and the only version that states a physical sign per term [R251][R252][R263] |
| 8 | ACV | **Two numbers go in the ladder table**: the 0.5625 random-ranking floor, and the per-position (end vs middle car) offset check | Below the floor is a bug, not a result; the offset check tests the peer-exchangeability assumption the method rests on [R256] |
| 9 | Door | **Name the transductive ablation**: the "within-block batch normalisation" variant reads the held-out block's own statistics | Keep it, label it, never headline it [R303] |
| 10 | Door | **Tune the threshold on end-to-end IoU-F1**, not cycle-level F1 | Over-segmentation is free under one and charged under the other, and a wrong label costs a miss *and* a false positive (info kit §4) |
| 11 | SHM | **Promote the rainflow diagnostic to a gate** before any ladder row; FDS bands move from nice to MUST-if-the-diagnostic-points-there | The diagnostic is what separates the six published reasons a rainflow label fails to reproduce [R270][R274] |
| 12 | SHM | **Add a noise-floor row** (bootstrap damage over sub-windows) and report MAPE split by damage mode | Bounds the achievable `1 − MAPE` before you optimise against it [R269][R271]; the low-damage mode decides the score |
| 13 | SHM | **Hand-implement the spectral methods** — no installs | Dirlik and Tovo–Benasciutti are short closed forms [R269][R275][R276]; `ffpack` would have been skipped anyway (GPL-3.0 in a shipped deliverable) |
| 14 | Aug. | The plan's "SHM and ACV = none" is **half right**: ACV stays at none, SHM gains exactly one row (C-Mixup in log-feature space) | Cropping and vanilla mixup do not preserve a cumulative-damage label; C-Mixup is the published fix [R299] |
| 15 | Aug. | **Publish the negative results**: ≈ +1.5 % expected through ROCKET, class weighting expected to be a wash on the boosting stack | Both belong in the leaderboard as measured rows, each with the citation that predicted them [R296][R302] |
| 16 | All | **Do not cite the refuted item** — the *Applied Energy* 402 (2026) few-shot HVAC-FDD DOI and its F1 numbers | Refuted by the verifier and absent from `references.md`; [R268] is the verified alternative |
