# Rail PHM — what the literature and operators actually use

Per subsystem: the signals real systems carry, the features that work on them, the fault taxonomy,
the thresholds and standards, and what is actually deployed. Every claim carries a `[Rnn]` resolving
to `references.md`. Numbers marked `UNVERIFIED` come from abstracts or secondary sources and must not
be quoted as hard figures in the pitch.

Ends with **"Implications for our simulators and features"** — concrete feature names and threshold
values to put into `nebulax/sim/*.py` and `nebulax/features/*.py`.

**W1 update (18 Sep).** Three things changed in this pass and they change simulator parameters, not
just prose. (a) A new **section 0** fixes the operating envelope: every thermal number in this
document was previously written for a temperate climate, and Singapore's ambient band moves the
bearing baseline by ~15 K and pins the dryer at its rated inlet condition year-round. (b) The
axle-box **thermal and vibration laws in 4.3** are now derived from published correlations and
measured tables instead of asserted, and two of the old constants are **falsified** — the healthy
crest factor and the monotonic kurtosis law. (c) **Section 3.3 is rewritten**: the free EU/OTIF
regulatory text was read this pass, it contains **no axle-box temperature threshold at all**, and
EN 15437-1 explicitly excludes alarm criteria from its scope — so the "95 °C / 56 K" pair is retired
and replaced with a published operator rule set. The same applies to the door: **no freely available
regulatory text carries a newton or a second** for obstacle detection.

---

## 0. Operating environment — Singapore

Everything downstream of a temperature depends on this, and nothing in the W0 pass stated it.

| Quantity | Value | Source |
|---|---|---|
| 24-hour mean air temperature | **26.8 °C** (Dec/Jan) to **28.6 °C** (May) | [R152] |
| Mean daily **maximum** | **30.5 °C** (Dec) to **32.4 °C** (Apr) | [R152] |
| Mean daily **minimum** | **24.3 °C** (Jan/Dec) to **25.7 °C** (May/Jun) | [R152] |
| Record daily maximum / minimum | **37.0 °C** (13 May 2023) / **19.0 °C** (14 Feb 1989) | [R153] |
| Mean annual relative humidity | **≈82 %**; monthly daily-mean 80.7-85.5 % | [R152] |
| Daily **maximum** RH | **93.0 %** (Aug) to **96.5 %** (Nov) | [R152] |
| Daily **minimum** RH | **61.4 %** (Mar/Oct) to **68.0 %** (Dec) | [R152] |
| Rainfall | 2113.3 mm over 171 rain days | [R152] |
| Regulatory design envelope, zone **T1** | **−25 °C to +40 °C** nominal | [R163] cl. 4.2.6.1 |

**Two consequences that change simulator parameters.**

1. **Bearing thermal baseline.** A healthy axle box running **25 K above ambient** sits at **53 °C**
   at the 28 °C annual mean and **57 °C** on a 32 °C afternoon; on the 37 °C record day it reaches
   **62 °C**. Against the Dutch Railways level-4 line of **80 °C absolute** [R151] that leaves only
   18-27 K of headroom, and a bearing running a high-speed-rail-like **50 K** rise would sit at
   **78-82 °C healthy** — i.e. already alarming. **An absolute temperature threshold is far less
   useful in Singapore than in a temperate fleet; the peer-relative feature is not a nicety, it is
   the primary detector.** The simulator must sweep ambient over 24-33 °C routinely and 19-37 °C for
   the stress case, and the ambient driver must have a diurnal shape, not a constant.
2. **Dryer duty.** A twin-tower desiccant dryer is **rated on saturated inlet air**, and the DOE
   sourcebook states plainly that "because dryer ratings are based upon saturated air at inlet, the
   geographical location is not a concern… the dryer has a lower load in areas of lower relative
   humidity, but the pressure dew point is not affected" [R155]. Singapore's inlet is at or near
   saturation on most days of the year (daily-max RH 93-96.5 % [R152]), so the dryer sits **at its
   rated load year-round with no seasonal relief** — model tower period and purge loss at the
   worst rated condition, not at a temperate average, and drive dryer load from **inlet temperature**
   (moisture carrying capacity) rather than from RH alone.

**Not found this session, and therefore `UNVERIFIED`:** any published LTA or SMRT rolling-stock
environmental design specification, and any measured Singapore MRT **tunnel** temperature/humidity
figures. Tunnel conditions differ from the surface (station cooling, train waste heat) and we have no
source for them — say "surface climate normals" on the slide, not "tunnel conditions".

---

## 1. DOOR — electric passenger door

### 1.1 Signals that are actually available

| Source | Signal set | Note |
|---|---|---|
| Korean test-rig comparative study [R64] | **Motor current only** | The entire diagnosis is built on one channel. No added sensors. |
| Cranfield railway-asset PHM [R72] | **Motor current only** | Stated explicitly: current is already available from the controller / motor drive with **no added sensors**, and the method generalises to any electro-mechanical actuator. |
| Chinese rail-vehicle door telemetry [R78] | **Door position, motor speed, motor current**, collected from **multiple doors of the same vehicle simultaneously** | The multi-door aspect is the transferable trick: siblings are the reference population. |
| ScotRail Class 158 RCM analysis [R74] | door key switch, **door interlock switch**, door operating pressure, door opening pressure, door closing pressure, emergency passenger relay | The operator's own condition-monitoring shortlist. Cheap discrete/analogue channels, **not vibration**. Note: Class 158 doors are electrically controlled but **pneumatically operated** (torque cylinder driving linkages, mechanically locked over centre). |
| PHME 2026 door test bench [R69] | `Time, POS_REF, POS_FBK, VEL_REF, VEL_FBK, FBK_DIGHALL, FBK_DIGENC1, DRV_PROT_VBUS, MOT_PROT_TEMP, FBK_CUR_A/B/C, DRV_PROT_TEMP, FBK_VOL_A/B/C` | 16 channels, 600 samples per file, one file per opening or closing activity, two files = one cycle. Three-phase brushless servomotor, INGENIA EVEREST XCR controller. |
| Plug-door audio work [R75] | Airborne audio, 44.1 kHz portable recorder, CRH5A plug door | Retrofit modality where the DCU cannot be tapped. |

**Verdict for us:** motor current plus position is sufficient and is what the field uses. Limit-switch
and interlock discrete state deserve first-class status because that is what the operator monitors.

### 1.2 Features that work

- **Three velocity-regime segmentation** of the current signal — acceleration / constant-speed /
  deceleration — then time-domain features per segment, then Fisher discriminant feature selection,
  then kNN. The traditional pipeline reaches good accuracy **only after** this segmentation [R64].
- **DWT detail-coefficient L1-norms at levels W8 and W9 (Daubechies db10) plus the motor starting-
  current peak.** Three features, >96 % fault-detection accuracy, response time below 0.3 s, validated
  on a LabVIEW rig [R66].
- **DWT norm and peak of the closing current** for obstruction detection, thresholds derived from
  simulation then validated experimentally, discriminating **soft vs hard** trapped objects — an
  obstruction *severity* class, not a binary [R67].
- **Information-value weighting of features/segments** so the classifier stays valid across operating
  conditions (open vs close, load, speed regime) — the "condition-conditional threshold" idea instead
  of one global threshold [R65].
- **Spearman rank correlation + variance inflation factor screening, then XGBoost gain ranking**, then
  stacking, then dynamic threshold optimisation by F1 maximisation, then SHAP to check the decision
  logic matches physical failure mechanisms [R68].
- **Feature families that carry door degradation signal** (from a PHME 2026 challenge solution):
  electrical load characteristics, position behaviour, **shock events**, trend information, and
  source-model predictions [R71].
- **DTW distance between the current cycle and a typical normal cycle** as the degradation indicator,
  then k-means to assign fault-severity levels, then a representative dwell time per level [R72].
- **Cycle-level statistical features** matched against historical run-to-failure trajectories by
  similarity, producing a monotonic remaining-life countdown [R70].
- **EMD → multi-scale normalised permutation entropy → Fisher selection → IPSO-optimised multi-class
  SVM** on airborne audio [R75]; and fractional wavelet-packet energy entropy with a hybrid IMF
  selection criterion [R76][R77] (`NUMBERS UNVERIFIED`, paywalled).

### 1.3 Fault taxonomy, with real frequencies

**ScotRail Class 380, 38-unit fleet, 205 door defects — root causes ranked [R73]:**

| Root cause | Defects | Share |
|---|---|---|
| **No fault found (NFF)** | 87 | **42.4 %** |
| Faulty push buttons | 39 | 19 % |
| Faulty DCU | 20 | |
| Limit / micro-switch disengagement | 18 | |
| Light barrier | 7 | |
| Door drive (motor failure, **encoder** failure, faulty connections) | 6 | |
| Guard operating panel | 6 | |
| Limit switches | 6 | |
| Loose plugs | 6 | |
| Obstruction by dirt / debris in door tracks | 6 | |
| Door roller detachment | 2 | |
| Poor lubrication | 2 | |

Criticality (risk factor) per failure mode ranges 3-28 out of 100: ~3 % very low, ~15 % low, ~70 %
medium, 12 % high; nine high-critical failure modes (four at RF 27, five at RF 28). Their 5-whys
example traces a dead door to a tripped MCB caused by an **unsecured 40-pin motor plug with no
secondary locking** [R73].

**Injected faults in the published door simulator** [R66]: armature electrical fault, brush wear,
increased friction, lead-screw misalignment.

**Cranfield linear actuator classes** [R86]: normal, backlash, lack of lubrication, spalling.
(Class names and `.mat` format confirmed from `SECONDARY-SOURCE` users of the dataset, not the CORD
landing page — verify at the DOI.)

**PHME 2026 degradation model** [R69]: stochastic **shocks** reduce the maximum closing position;
failure when closing position falls below **10 % of optimal**; shock times drawn from e.g.
[3600, 7200] s, severity [2, 14] %.

### 1.4 Thresholds, standards and cost

**What the free regulatory text actually says — read this pass, and it is a correction.** The EU
LOC&PAS TSI (mirrored in full, free, by the OTIF UTP LOC&PAS) requires obstacle detection but
**contains no force in newtons and no reaction time in seconds**. Clause 4.2.5.5.3(5), verbatim:
"External passenger access doors shall incorporate devices that detect if they close on an obstacle
(e.g. a passenger). Where an obstacle is detected the doors shall automatically stop, and remain free
for a limited period of time or reopen. The sensitivity of the system shall be such as to detect an
obstacle according to the specification referenced in Appendix J-1, index [17], with a maximum force
on the obstacle according to the specification referenced in Appendix J-1, index [17]." [R163]

Appendix J-1 index [17] resolves to **EN 14752:2019+A1:2021** — note the edition, **not** the 2025
edition this document previously named — with **[17.1] sensitivity → clause 5.2.1.4.1** and
**[17.2] maximum force → clause 5.2.1.4.2.2** [R163]. That gives us the exact citable clause
structure without owning the standard. The standard itself remains paywalled; BS EN 14752:2025 is
94 pp., published 2025-06-23, and its free scope confirms it applies to **metro** as well as
main-line stock [R168], but it is not the TSI-mandated edition.

**Therefore the 100-200 N / 0.3 s operating point is a design assumption, not a compliance
reference.** It comes from the two Shiao papers, which benchmark obstruction detection in that band
and claim EN 14752 compliance [R67][R66]. We will quote it as "the operating point used by [R67]"
and report detection latency against 0.3 s as *our* acceptance metric — never as "EN 14752 requires".
**We cannot make a numeric door compliance claim to LTA judges from freely available text.**

**The only door force and timing numbers in free regulatory text are these, and they are not
obstruction limits** — they are PRM human-interface and passenger-warning requirements [R165]:

| Quantity | Value | Clause |
|---|---|---|
| Palm force to operate a public door control device | **≤ 20 N** | PRM 4.2.2.3.1(2) |
| Force to open or close a **manual** door | **≤ 60 N** | PRM 4.2.2.3.3(3) |
| Door **opening** signal duration | **≥ 5 s** (may cease after 3 s if the door is operated) | PRM 4.2.2.3.2 |
| Remote / automatic opening signal | **≥ 3 s** from the start of opening | PRM 4.2.2.3.2 |
| Door **closing** signal | starts **≥ 2 s before the door starts to close**, continues until closed | PRM 4.2.2.3.2 |
| Clear usable door width | **≥ 800 mm** (≥ 1000 mm for wheelchair-access level-access doors, < 250 km/h) | PRM 4.2.2.3.2 |
| Internal emergency-opening device active below | **10 km/h** | LOC&PAS 4.2.5.5.9(1) [R163] |

There is **no cycle-time (open-to-close duration) requirement anywhere in the PRM TSI** — so any
closing-time threshold in our simulator is ours, not a regulator's. The ≥ 2 s closing warning is
useful though: it pins the simulated door event timeline to a regulator-consistent shape, so an
injected obstruction always sits after a legitimate warning phase.
- **Class 158 RCM** [R74]: 48-unit DMU fleet, 8 doors per unit, **over 100 inter-dependent components
  and over 345 failure modes**; doors have the **most technical incidents and the most delay minutes**
  of any safety-critical system on the fleet; a door failure typically causes a **5+ minute delay**,
  part-cancellation or full cancellation. Their CBM feasibility test: is the P-F curve predictable,
  is the monitoring interval practicable, is the P-F interval long enough to act.
- **Class 380 cost** [R73]: 518 total delay minutes attributable to door defects at **£50/min**
  penalty.
- **"30-40 % of operating train failures occur in the train door systems"** — Bombardier experience
  feedback, as cited in [R64]; the manufacturer-coauthored provenance is [R82].

### 1.5 What is deployed

- **Online predictive diagnosis of electrical train door systems**, IFSTTAR with **Bombardier Transport
  France** [R82] — the target architecture is onboard/online, not offline batch. `NUMBERS UNVERIFIED`.
- Operators' own CBM shortlists are **switches and pressures**, not vibration [R74].
- The 2019 comparative study's practical conclusion is the one to carry into the pitch: **a CNN on raw
  current beats the traditional feature pipeline on accuracy, but the feature pipeline is more useful
  in service because features give a per-fault health index you can trend over time — the CNN does
  not** [R64]. That trade-off is our headline ablation axis.

---

## 2. PNEUMATIC — brake air supply (APU: compressor, main reservoir, twin-tower dryer)

### 2.1 Signals

| Release | Signals | Rate | Window |
|---|---|---|---|
| **MetroPT-3** [R88] (ours) | 7 analogue: `TP2, TP3, H1, DV_pressure, Reservoirs, Motor_current, Oil_temperature`; 8 digital: `COMP, DV_electric, TOWERS, MPG, LPS, Pressure_Switch, Oil_Level, Caudal_Impulses`. **No GPS, no Flowmeter.** | 1 Hz | Feb-Aug 2020, 1,516,948 rows |
| **MetroPT-1** [R87][R132] | 8 analogue: adds **`Flowmeter`**; 8 digital; **plus `gpsLong/gpsLat/gpsSpeed/gpsQuality`** | 1 Hz | Jan-Jun 2022, 10,979,547 rows, 20 variables |
| **MetroPT-2** [R89] | 16 sensor signals + control signals + GPS, 21 attributes | 1 Hz | 28 Apr - 28 Jul 2022, 7,116,940 records |

**Naming trap for our repo:** the three releases are different datasets with different signal sets and
different failure modes. MetroPT-3 has **air leaks only** [R88]; a classifier trained on it cannot
separate air-leak from oil-leak or dryer faults.

### 2.2 Ground-truth failure windows (cite these exactly)

**MetroPT-1** [R87]: F1 **Air Leak on Clients**, 28-02-22 21:53 → 01-03-22 02:00 (14,820 rows);
F2 **Air Leak on Air Dryer**, 23-03-22 14:54 → 15:24 (1,800 rows); F3 **Oil Leak on Compressor**,
30-05-22 12:00 → 02-06-22 06:18 (281,800 rows).

**MetroPT-3** [R88]: four high-severity **air-leak** windows — 2020-04-18, 2020-05-29/30,
2020-06-05/07, 2020-07-15.

**MetroPT-2** [R89] (failure timestamps taken from [R93], not the Zenodo landing page): **air leak**
2022-06-04 10:19:24 - 14:22:39; **oil leak** 2022-07-11 10:10:18 → 2022-07-14 10:22:08. `LPS` (low
pressure signal) fires **late inside** each window, giving a natural "alarm must precede LPS" deadline.

Severe event-count imbalance: **3-4 episodes per release**. Episode-level metrics with 3 positives
are statistically fragile — say so on the slide.

### 2.3 Features and thresholds that work

- **The one-feature rule.** On MetroPT-2, training on failure-free data up to 2022-06-01, testing
  from 2022-06-01, 30-minute windows with 5-minute stride (L=1800, d=300), declaring failure when
  p>0.5: **`Flowmeter_max > 16.05` (air leak) and `Flowmeter_max > 16.18` (oil leak)** give
  **F1 = 1.0** — both failures caught, zero false positives — detecting the air leak **~150 min before
  the LPS signal** and the oil leak **>2 days ahead**, beating or matching the deep baselines
  (WAE-GAN, LSTM-AE) and AMRules [R93].
- **Compressor duty-cycle asymmetry.** Using **only compressor on/off logs** on a Dutch Railways
  fleet: air leakage manifests as **compressor idle time becoming shorter than compressor run time**
  (consumption outpaces generation). Per-train logistic regression separates the two regimes;
  density-based clustering with a dynamic threshold grades severity; a logistic function on compressor
  run time plus leak duration yields a severity model from which RUL of the brake pipe to a given
  severity level is estimated. Most air leaks were detected **one to four weeks** before braking
  failure, with contextual pre-filtering suppressing false alarms [R101].
- **Reconstruction-error thresholding with smoothing.** Train a sparse autoencoder on nominal periods
  only, threshold the reconstruction error, and low-pass the error signal to cut false alarms — the
  reference baseline the UCI record asks you to cite [R91][R88].
- **Multi-signal corroboration.** Forecast APU signals over a horizon, detect anomalies in the
  **forecast** rather than the observation, and only raise a fault if **several signals go anomalous
  simultaneously** [R95]. ~20 lines, large false-alarm reduction.
- **Streaming.** Half-Space Trees combined with One-Class kNN, adapted for data streams: far fewer
  type-I errors (much higher precision) than HS-Trees alone while still catching most catastrophic
  APU failures [R99].
- **Change points, not anomalies.** Dryer tower switching and compressor load/offload transitions are
  change points; treating them as such avoids flagging every normal `COMP` transition [R105].
  **W1 addition:** this is no longer only a framing. Offline multiple-change-point detection over
  multivariate series is a one-line library call — `ruptures` (BSD-2) ships Pelt, BinSeg, BottomUp,
  Window, Dynp and KernelCPD with ten cost functions [R173][R174]; `changeforest` (BSD, JMLR 2023) is
  purpose-built for the multivariate/high-dimensional case that MetroPT's 15 channels represent
  [R187]; `claspy` (BSD-3) is **hyper-parameter-free**, so there is nothing to calibrate [R176][R177];
  and streaming drift detectors (ADWIN, Page-Hinkley, KSWIN) are already installed, because the
  pneumatic ladder imports `river` for Half-Space Trees [R178][R179][R180][R181]. See
  `model_ladder.md` §2c for the rows and the honest caveat that all of these emit **discrete**
  breakpoints or booleans and therefore need a score adapter before they can enter a VUS-PR table.
- **Compressor fault vocabulary** from a real balanced multi-class acoustic dataset: Healthy, Leakage
  Inlet Valve, Leakage Outlet Valve, Non-Return Valve, Piston Ring, Flywheel, Rider Belt, Bearing
  [R135].

### 2.3b First-principles constants — the APU, the leak and the dryer

*Added in the W1 pass. Every number the pneumatic simulator uses was previously uncited; these are
the primary sources. Three are rail-specific, the rest are industrial compressed-air engineering and
transfer at the same pressure class (5-9 barg) as a rail main reservoir.*

**Compressor control and instrument constants — rail-specific, from the dataset descriptor itself
[R87].** These are the four numbers the simulator must not invent, because the APU we are modelling
is the one this paper describes:

| Constant | Value | Source |
|---|---|---|
| Compressor **start** pressure | below **8.2 bar** | [R87] |
| Compressor **stop** pressure | above **10.2 bar** (operating pressure) | [R87] |
| `LPS` low-pressure switch trip | below **7 bar** | [R87] |
| `Motor_current` three-state expectation | ≈ **0 A** off, ≈ **4 A** offloaded, ≈ **7 A** under load | [R87] |
| `TOWERS` semantics | inactive = tower one drying, active = tower two drying | [R87] |
| Acquisition rate | 1 Hz | [R87] |

The twin-tower **switching period does not need to be invented either** — measure it directly from
the `TOWERS` channel in MetroPT-2/3 [R88][R89]. That is empirical ground truth from a real metro
dryer in service and is strictly better evidence than any industrial rule of thumb.

**Leak magnitude — a leak is a pressure-dependent orifice, not a constant draw.** The DOE / Compressed
Air Challenge leak table gives flow in cfm by orifice diameter and supply pressure, with correction
factors ×0.97 for well-rounded and **×0.61 for sharp-edged** orifices, and states that leakage scales
with the **square of orifice diameter** and rises with supply pressure [R154]. Converting their 90
psig row to NL/s for sharp-edged orifices (1 cfm = 0.4719 NL/s; *our arithmetic on their published
table*):

| Orifice | Sharp-edged leak at 90 psig ≈ 6.2 barg |
|---|---|
| 1/64 in (0.40 mm) | **0.10 NL/s** |
| 1/32 in (0.79 mm) | **0.42 NL/s** |
| 1/16 in (1.59 mm) | **1.65 NL/s** |
| 1/8 in (3.18 mm) | **6.65 NL/s** |

So our planned `Q_leak = 0.2 + 3.0·s` NL/s spans roughly a **0.55 mm** sharp orifice at `s = 0` to a
**2.2 mm** sharp orifice at `s = 1` — both physically sensible, the large end a plausible fitting or
hose leak, and now citable. **But the functional form must change**: a fixed NL/s draw is wrong during
charge and discharge transients, because the real leak flow follows `Q ∝ d²·f(P)`. See 4.2.

**Regulatory upper bound.** The only regulatory leak figure found is US freight: brake-pipe leakage
"shall not exceed **5 psi per minute** or air flow shall not exceed **60 cubic feet per minute**",
measured after a 20 psi service reduction with the brake valve in neutral for 45-60 s [R156]. 60 cfm
= **28.3 NL/s** (*our conversion*) — that is a whole freight consist's brake pipe, roughly 10× our
`s = 1` severity, so use it as the **catastrophic end** of the severity scale, not as a per-unit
value. The **5 psi/min = 0.34 bar/min** form is directly usable as a pressure-decay fault-injection
spec and as a labelled "failed the acceptance test" threshold.

**Duty ratio has a published absolute band, not only a percentile.** With all end uses off, DOE gives
`Leakage(%) = T·100/(T + t)` where `T` = on-load minutes and `t` = off-load minutes, with **< 10 %**
in a well-maintained system and **20-30 %** when poorly maintained [R155]. Our `idle_run_ratio =
t_off / t_loaded` is exactly `t/T`, so `Leakage(%) = 100/(1 + idle_run_ratio)` and (*our algebra on
their formula*):

| System state | DOE leakage % | Equivalent `idle_run_ratio` |
|---|---|---|
| Well maintained | ≤ 10 % | **≥ 9.0** |
| Degraded | 20 % | **4.0** |
| Poorly maintained | 30 % | **2.33** |

**Honest caveat that must be stated with it:** DOE measures with all end uses *off*. A train in
revenue service has genuine consumption (door operations, brake applications, air suspension
levelling), so the absolute band applies cleanly only to a **stabled or idle** train. In service, keep
the validation-percentile rule and report the absolute band as a second axis. This is still a large
improvement on the W0 text, which had only a percentile and no citation.

**Twin-tower dryer constants** [R155][R160]:

| Quantity | Value | Source |
|---|---|---|
| Rated pressure dew point, twin-tower desiccant | **−40 °F = −40 °C** | [R155] |
| Purge air, pressure-swing regenerative | **10-18 %** of the dryer's rating | [R155] |
| Purge air, heater-purge twin tower (measured, 13 sites) | **12 % of flow** | [R160] |
| Pressure drop through the dryer | **3-5 psi** | [R155] |
| Heater run-time fraction (heated types) | 14-57 %, depending on dew-point setpoint and load | [R160] |
| Regeneration mechanism | depressurise the tower, pass previously dried purge air through the bed | [R155] |
| Regeneration control | built-in cycle based on **time, dew point, or both**; in practice a **timed cycle adjusted by season** | [R155][R160] |
| Coupled failure mode | **low supply pressure → higher volumetric flow → reduced purge → incomplete regeneration → degraded performance and possible failure** | [R160] |

That last row is a genuinely useful *cascading* fault to inject: a main-reservoir leak degrades the
dryer, which is a plausible real failure chain and a much better demo than two independent faults.

**Compressor load/unload timing** [R155]: relieving the sump/separator on unload "typically takes
about **40 seconds**", and about **3 seconds** is needed to repressurise on reload. Use these as the
transition durations in the state machine rather than instantaneous switching.

### 2.4 Operational requirement and deployed practice

- The APU has **no redundancy**; failure forces **immediate train withdrawal**. The operator's
  requirement is detection **at least ~2 h before removal** with as few false alarms as possible
  [R90]. A WAE-GAN is reported to meet exactly that — ≥2 h ahead, zero false alarms [R94]
  (`NUMBERS UNVERIFIED`, publisher abstract).
- **LSTM-AE + SHAP** attributes each flagged anomaly to specific sensors, reported to beat a sparse
  autoencoder on F1/recall/precision [R92] — the closest published analogue to our deliverable.
  `NUMBERS UNVERIFIED`; the ≈90.8 % F1 figure circulating in secondary sources was not confirmed.
- **Failure mode to avoid:** detector-plus-rule-explainer pipelines can produce rules that cover only
  a sliver of the failure episode [R98], which is the explicit target of the [R93] critique. If we
  ship rule explanations in the UI, **report rule support over the whole episode**.
- **Explanation UIs already exist** for this data: NL + visual explanations over the MetroPT stream
  [R97], and a multi-task ANN with LIME plus a delivered dashboard [R106]. Cite them for the UI, not
  for their numbers.

---

## 3. BEARING — bogie axle-box

### 3.1 Sensing modalities and what each can see

| Modality | Where | What it catches | Source |
|---|---|---|---|
| Wayside HABD (hot axle box detector, IR) | Trackside | End-of-life thermal runaway. **Late-stage indicator.** | [R109][R113] |
| Wayside acoustic (RailBAM / TADS) | Trackside microphone array | End-of-life bearings; **misses incipient defects**. Requires Doppler-effect removal by resampling before envelope/order analysis, because the signal is Doppler-distorted and buried in wheel-rail noise. | [R109][R127] |
| Onboard vibration | Axle box | **Damage far earlier than temperature.** | [R109] |
| Onboard temperature | Axle box | Continuous, cheap, mandated above 250 km/h. | [R113][R114] |
| Acoustic emission | Axle box | Earliest of all; specialist. | [R109] |

The central argument for our pitch: **temperature-only detection is a late-stage indicator while
vibration and acoustic emission detect damage far earlier** [R109].

### 3.2 Features and methods

- **The canonical recipe** [R117]: order tracking / discrete-random separation to remove deterministic
  gear and shaft components → **spectral kurtosis (kurtogram)** to pick the demodulation band →
  **envelope analysis** → **BPFO / BPFI / BSF / FTF** harmonic identification. Written explicitly for
  signals masked by other machine components — the bogie situation.
- **Variable speed, no tacho** [R122]: order-tracking suffers resampling error and harmonic
  interference; time-frequency ridge methods hit resolution limits; identify the **fault
  characteristic frequency band directly in the envelope spectrum** instead.
- **Physics-hybrid thermal residual** [R110]: a lumped thermal/physical model of the axle box
  hybridised with BP-NN and LSTM models sharing the same inputs — **speed, ambient temperature, load,
  adjacent-bearing temperatures** — with the residual between predicted healthy temperature and
  measured temperature as the fault indicator, calibrated on real fleet operation data (~2.25 million
  km, `UNVERIFIED` abstract-level figure).
- **Messy in-service records** [R111]: time-delay low-rank + sparse decomposition separates normal
  thermal trend, sparse anomalies and sensor noise, then an attention-augmented BP-LSTM predicts the
  healthy temperature for residual-based alarms. Directly addresses missing-value and spiky-sensor
  problems. `NUMBERS UNVERIFIED`.
- **Audio-derived features on real rail** [R120]: **MFCC** and **amplitude modulation spectrogram**
  features with a **One-Class SVM trained on healthy signals only**, on a state-of-the-art commuter
  railway engine fed by an industrial power converter and coupled to a load machine. Significantly
  improved classification over conventional bearing features; the one-class setup is explicitly
  motivated by real-world class imbalance.
- **Generalisation to unseen damage types** [R121]: MFCC + a simple MLP detects bearing faults from
  airborne sound on real commuter railway vehicles and **generalises to damage types not present in
  training**.
- **Learned health index for railway axle boxes** [R123]: deep residual shrinkage network (learned
  soft-thresholding denoising) + LSTM, motivated by axle-box signals being noise-dominated and
  distorted by the transfer path; validated on artificially induced defects **and** accelerated
  fatigue run-to-failure tests; reported sensitive to early degradation.

### 3.3 Thresholds and standards — **rewritten in the W1 pass**

**Headline finding, and it is a negative one: there is no harmonised European axle-box temperature
threshold.** The W0 text carried an alarm pair of "absolute > 95 °C, differential > 56 K" attributed
to EN 15437 via secondary sources. That attribution is **wrong in kind, not merely unverified**, and
both numbers are now withdrawn from this document. Three independent free sources establish why:

1. **The TSI / UTP text contains no temperature value at all.** LOC&PAS clause 4.2.3.3.2 requires
   axle-bearing condition monitoring and says only that "the bearing condition shall be evaluated
   either by monitoring its temperature, or its dynamic frequencies or some other suitable bearing
   condition characteristic". A search of the whole 248-page in-force text finds **no °C value
   anywhere in the bearing clauses** [R163]. The freight-wagon UTP repeats the identical wording and
   is equally silent [R164]. The only number the regulation fixes is a **speed**: units with a maximum
   design speed **≥ 250 km/h shall be equipped with on-board detection equipment**; below that, either
   on-board or track-side monitoring is acceptable [R163].
2. **EN 15437-1 explicitly excludes alarm criteria from its scope.** Its free published scope states
   the standard "does not cover Hot Wheel Detectors, **temperature measurement methodologies,
   operational procedures for responding to detector alerts**, or maintenance requirements for
   detector systems" [R167]. It defines the trackside/axlebox **interface and target-zone geometry**
   (rolling stock in Clause 5, trackside equipment in Clause 6) — and the TSI reproduces those
   geometry numbers for 1668 mm gauge: target area `YTA 1176 ± 10 mm, WTA ≥ 55 mm, LTA ≥ 100 mm`;
   prohibitive zone `YPZ 1176 ± 10 mm, WPZ ≥ 110 mm, LPZ ≥ 500 mm` [R163]; the WAG UTP tabulates the
   1524 mm and 1600 mm gauges too [R164]. **Geometry, never temperature.**
3. **ERA declares on-board axle-bearing monitoring an OPEN POINT.** The official ERA application guide
   lists EN 15437-2:2012+A1:2022 against TSI 4.2.3.3.2 with the annotation "Track side system /
   On-board system **(open point)**" [R166]. In TSI language an open point means **no harmonised
   European requirement exists** and each Member State or operator sets its own rule.

**EN 15437-2 (the on-board part) is 20 pages and its free abstract mentions no alarm level.** The
catalogue entry confirms the exact title, a publication date of 2023-05-18 and a 20-page extent —
consistent with a short performance/design document specifying measurement accuracy and system design
rather than a harmonised alarm value [R169]. Whether the body contains any numeric threshold is
`UNVERIFIED` and must not be asserted. If one standard purchase is ever judged worthwhile, this is the
single document most likely to contain an on-board alarm structure.

Two further corrections to the W0 text: the **EKE-Electronics page publishes no numeric threshold** —
it states only that temperature "is assessed to determine if it has crossed one of the **four
thresholds**" and that "**configurable alarm levels** support tailored safety responses" [R171], so
the 95 °C / 56 K pair is not attributable to that vendor either; and **RSSB RIS-2714-RST Issue 1**
does exist (published 03 June 2023, live, covers "condition monitoring of axle bearings, whether by
trackside or onboard detection systems") but its PDF is **login-gated** and no threshold was read
[R170]. RSSB offers free individual registration — one team member should read the real values before
the pitch; until then any number attributed to it stays `UNVERIFIED`.

**What replaces them: a published operator rule set.** Netherlands Railways published its actual
in-service axle-bearing temperature decision rules, derived from ~3000 carriages, 131 trains of one
series, **60 million wayside HotBox measurements over 2 years** across 27 measurement locations
[R151]. This is a far better citation than a paywalled standard because it is an operator saying what
it really does:

| Rule | Value | Note |
|---|---|---|
| Wayside **immediate-stop** limit | measured axle-box temperature **> 115 °C** → driver ordered to stop immediately | [R151] |
| Peer feature, defined exactly | `dT` = absolute bearing temperature **− median of the other bearing temperatures on the same SIDE of the train** | not all peers — the same-side restriction matters, see below |
| Differential alarm | `dT > 30 °C` → level 1/2/3 depending on frequency; `dT > 50 °C` → level 3 immediately | [R151] |
| Absolute alarm | **> 80 °C** → level 4 immediately | [R151] |
| Slow-degradation rule | per bearing, take the deviation of its side-difference from the **median of all side-differences on that train**; if it exceeds **3.5 standard deviations for ≥ 10 measurements within 30 days** → level 1; double that frequency → level 2 | [R151] |
| Data-quality guard | if **≥ 4 bearings on one train in one measurement read > 50 °C**, attribute the cause to the **measurement station** and suppress the alarms | [R151] |
| Achieved lead time | alarms **one to three months** before existing detection methods in every true-positive case; > 90 days in the illustrated case | [R151] |
| Normal variation is driven by | ambient temperature, sunshine, and **duty cycle** — "the bearing cools significantly by the wind while driving and warms up while standing still" | [R151] |

**Why "same side" rather than all peers.** Axle boxes on one wheelset are not thermally identical when
the car is powered: on a motor car, bearing 1 carries **13.5 % (200 km/h) and 10.5 % (300 km/h)**
higher roller-outer-raceway contact force than bearing 2 and runs measurably hotter, while on a
trailer car the two bearings of a wheelset are thermally identical [R150]. So a peer feature must
compare **like-for-like positions**, not pool all eight boxes of a car. This is a concrete correction
to our `features.peer_normalise()` design.

**Absolute in-service temperatures actually measured.** Across 5 trains and 128 sensors per train over
three months, the **maximum** axle-box bearing temperature observed was **82 °C on a motor car** and
**69.8 °C on a trailer car**, with temperature tracking ambient and higher at 300 km/h than at
200 km/h [R150]. That is a high-speed fleet; a metro at 80-100 km/h will run cooler for the same
ambient — but Singapore's ambient is ~13 K warmer than a temperate mean (§0), which partly cancels it.

**HABD readings are noisy and biased.** A field + lab study found many HBD-flagged bearings have **no
discernible defect**, and the **IR scanning location on the bearing cup dominates** the measured
temperature [R112]. Independent corroboration of inter-box spread: at one matched condition
(200 km/h, ambient ≈15 °C) measured axle-box temperatures across 8 cars of one train spanned
**44.64-53.08 °C**, an **8.4 K** spread on nominally identical healthy bearings [R149]. This is the
empirical case for treating raw HBD temperature as a noisy, biased sensor, for event-level precision
rather than raw alarm counts, and for the ±5 K scan-location term in our wayside model (§4.3).

**Alarm architecture worth keeping from the vendor page**: four tiers, operator-configurable [R171].
Our detector should expose a multi-level, tunable threshold, not a single hard-coded °C.

**Deployed practice confirms the thresholds live with the infrastructure manager, not the standard.**
Trafikverket's freely published Network Statement says its network carries detectors for overheating,
unintended brake application, wheel damage and acoustic wheel-bearing detection, and that in the event
of an alarm the process is defined by its own internal document **TDOK 2020:0074** — not by a European
standard [R172]. The same Network Statement notes that access to extended traffic information via
detectors (API Järnväg, `data.trafikverket.se`) is **free of charge**, which is a candidate source of
real axle-box temperature measurements if we want one.

- Our own LTA reading materials give a wayside Temperature Measuring System alarming on axle, wheel
  and gearbox temperature in three categories — consistent with the multi-tier alarm structure above.

**What we can honestly say to LTA judges.** "There is no harmonised European threshold for on-board
axle-bearing monitoring — ERA marks it an open point [R166], and EN 15437-1 excludes alarm criteria
from its scope [R167]. We therefore calibrate on validation data only, expose the threshold as a
configurable parameter, and benchmark our rule against the published Netherlands Railways operator
rule set [R151]." That is a stronger position than a fabricated compliance claim, and it is true.

### 3.4 Evaluation practice specific to bearings

- **Bearing-wise partitioning is required.** The standard segment-wise and condition-wise splits used
  on CWRU, Paderborn, Ottawa UORED-VAFCLS and HUST **leak information and inflate accuracy**;
  bearing-wise partitioning means no physical bearing is shared between train and test, and the
  number of unique **training bearings**, not the number of segments, decides robustness. The paper
  also proposes a multi-label reformulation for co-occurring faults [R115].
- **Recording-level cross-validation** with Hilbert envelope demodulation: an interpretability-
  constrained shapelet ensemble matches ROCKET on CWRU and MFPT with a difference smaller than
  fold-to-fold SD — interpretability costs nothing here — and most bearing papers report inflated
  accuracy from segment-level splits [R58]. `NUMBERS UNVERIFIED`.
- **Artificial vs natural damage is a documented domain shift.** The Paderborn set separates 12
  artificially damaged bearings (drilling, EDM, electric engraving) from 14 naturally damaged by
  accelerated lifetime tests — the cleanest published ablation for "does my model only work on seeded
  faults?" [R125].
- **CWRU is near-saturated**; its near-perfect accuracies are a known artefact and do not transfer
  [R142][R115]. Include it only as an obligatory sanity baseline alongside a variable-speed set.

---

## 4. Implications for our simulators and features

Concrete. Feature names go straight into `nebulax/features/*.py`; numeric values into
`nebulax/sim/*.py` and `configs/`.

### 4.1 Door simulator (`sim/door.py`)

**Fault priors — re-weight by observed frequency [R73].** Our current fault map is
friction / brush wear / backlash / misalignment / obstruction / limit-switch / DCU dropouts. The real
distribution is dominated by **electrical, connector and switch** faults, not glamorous mechanical
wear. Set the scenario prior in `sample_scenarios()` to:

```
p(limit_switch | DCU | connector | push_button class) = 0.45
p(obstruction / debris in track)                      = 0.15
p(friction / lack of lubrication)                     = 0.15
p(backlash)                                           = 0.10
p(brush wear / armature electrical)                   = 0.10
p(misalignment / roller)                              = 0.05
```

and emit an explicit **`nff`** label for cycles that raise an alarm with no injected fault. The
**42.4 % no-fault-found** figure [R73] is the strongest single number in our pitch: continuous
telemetry plus anomaly detection is exactly what NFF is for.

**New signals to expose** (first-class, per [R74]): `interlock`, `door_key`, `emergency_relay`,
alongside the existing `ls_open`, `ls_closed`.

**New fault mode — shock-driven closing-position degradation** [R69][R71]. Add to
`DegradationTrajectory` a shock process, which makes our sim directly comparable to a public
run-to-failure benchmark:

```
shock inter-arrival  ~ U(3600, 7200) s
shock severity       ~ U(0.02, 0.14)   # fractional reduction of max closing position
functional failure   : max_closing_position < 0.10 * optimal
emitted per cycle    : shock_count, shock_peak (current-derivative spike amplitude), pos_close_max
```

**Obstruction operating point** [R67][R85]: keep the simulated obstruction force in the **100-200 N**
band and require detection within **0.3 s**; emit a soft/hard object class rather than a binary
`obstruction` flag. Report **detection latency against 0.3 s** as a first-class metric, not only F1.
(EN 14752 clause values `UNVERIFIED` — we quote them as "the operating point used by [R67]".)

**`cycle_features` additions.** Current list keeps `closing_time, opening_time, i_peak, i_mean_cruise,
i_rms_cruise, i_end, energy_J, pwm_mean, pos_err_max, pos_err_rms, reversal_count, obstruction,
ls_timeout, dropout_frac, T_motor, current_profile_50`. Add:

| Feature | Definition | Source |
|---|---|---|
| `dwt_l1_w8`, `dwt_l1_w9` | L1 norm of db10 DWT detail coefficients at levels 8 and 9 of the closing current | [R66] |
| `i_start_peak` | motor starting-current peak | [R66] |
| `i_mean_accel`, `i_mean_cruise`, `i_mean_decel` | mean current per velocity regime | [R64] |
| `i_rms_accel`, `i_rms_decel`, `t_accel`, `t_cruise`, `t_decel` | regime-wise RMS and durations | [R64] |
| `dtw_to_ref` | DTW distance of the (current vs position) profile to the fleet-median healthy cycle | [R72] |
| `peer_z_<feat>` | z-score of each feature against the 7 sibling doors of the unit at the same timestamp | [R78] |
| `shock_count`, `shock_peak`, `pos_close_max` | shock events and closing-position ceiling | [R69][R71] |
| `severity_stage` | k-means cluster index over `dtw_to_ref` (ordinal 0..4) | [R72][R81] |

**Feature screening and reporting** [R68]: Spearman rank correlation + **variance inflation factor**
screening → LightGBM gain ranking → stacking → threshold tuned by **F1 max on validation only** →
SHAP audit that the decision logic matches physical failure mechanisms. **Report PR-AUC alongside
ROC-AUC**; the published door imbalance is ~7.3:1 and PR-AUC 0.913 vs ROC-AUC 0.977 shows how much
the two differ [R68].

**Cost framing for the slides:** door failure → 5+ min delay, part- or full cancellation [R74];
518 delay minutes at **£50/min** on one 38-unit fleet [R73]; **30-40 % of operating train failures
are door failures** (Bombardier feedback via [R64]).

### 4.2 Pneumatic simulator (`sim/pneumatic.py`)

**Emit both flow channels.** MetroPT-3 has `Caudal_impulses` but **no `Flowmeter`**; MetroPT-1/2 have
`Flowmeter` [R87][R88][R89]. Our simulator should emit **both**, so the [R93] rule
(`Flowmeter_max > 16.05` air leak, `> 16.18` oil leak) is reproducible as a control arm on synthetic
data and the MetroPT-3 pipeline still runs.

**Duty-cycle ratio as a first-class output** [R101]. Per compressor cycle emit:

```
duty_ratio      = t_loaded / (t_loaded + t_unloaded + t_off)
idle_run_ratio  = t_off / t_loaded           # leak signature: this SHRINKS
```

Detector rule: alarm when the **6-hour rolling median of `idle_run_ratio` drops below its healthy
validation 5th percentile**, **and** report the DOE absolute band as a second axis — `idle_run_ratio
≥ 9.0` is DOE's "well maintained" (< 10 % leakage), `4.0` is 20 % and `2.33` is 30 % [R155] (our
algebra on their `Leakage(%) = T·100/(T+t)` formula, §2.3b). Caveat to state: DOE's band assumes all
end uses off, so it is exact only for a stabled train; in revenue service the validation percentile
governs and the absolute band is context. The 6-hour window and the 5th-percentile choice remain
**ours** and are labelled as such — [R101] establishes the *feature*, not our window length.

**Extend the fault map with the real compressor fault vocabulary** [R135]. Current map: air leak, oil
leak, compressor wear, dryer valve stuck, clogged filter. Minimum addition — `valve_leak_inlet` and
`valve_leak_outlet` as **distinct from reservoir-side air leak**: a valve leak shortens the charge
slope (`dP_dt_loaded` falls) **without** raising the `Reservoirs` decay rate during OFF, so the two
are separable by exactly the pair of features we already extract. Optional further classes for
multi-class work: `non_return_valve`, `piston_ring`, `rider_belt`, `flywheel`, `bearing`.

**Treat transitions as change points, not anomalies** [R105]. Emit an `is_transition` mask and
exclude ±5 s around `Towers` flips and `COMP` load/offload edges from point scoring; score those
regions with a change-point detector instead.

**Leak magnitude is now cited, and the functional form is CORRECTED.** Our plan set
`Q_leak = 0.2 + 3.0·s` NL/s on a 0.2 NL/s baseline (16× at `s = 1`). Against the DOE leak table those
endpoints correspond to a **0.55 mm** sharp-edged orifice healthy and a **2.2 mm** sharp-edged
orifice at `s = 1` (§2.3b, [R154]) — both physically sensible, so **keep the magnitudes**. But the
DOE table makes leak flow scale as **d²** and **rise with supply pressure** [R154], so a constant
NL/s draw misbehaves during charge and discharge transients. Replace it with a pressure-dependent
orifice:

```
A_leak(s)  = A0 * (1 + 15*s)                      # A0 fixed by Q(s=0)=0.2 NL/s at 6.2 barg
Q_leak     = Cd * A_leak * f(P_res, P_atm)        # Cd = 0.61 for a sharp-edged orifice [R154]
                                                  # f = choked/subsonic orifice flow; choked while
                                                  #     P_res/P_atm > ~1.9, which holds over the whole
                                                  #     8.2-10.2 bar control band [R87]
```

**Reservoir dynamics in closed form.** DOE's receiver sizing relation `V = T·C·Pa/(P1 − P2)` [R155]
rearranges directly into the ODE the simulator needs:

```
dP_res/dt = (S_supply - C_demand - Q_leak) * Pa / V_res
```

with `S_supply` the compressor free-air delivery while loaded, `C_demand` the genuine consumption
(doors, brakes, air suspension), and `V_res` the reservoir volume. Control band, LPS trip and motor
current come from [R87] (§2.3b): start below 8.2 bar, stop above 10.2 bar, LPS below 7 bar, motor
current 0 / 4 / 7 A for off / offloaded / loaded. Load-unload transitions are not instantaneous —
sump relief on unload ≈ **40 s**, repressurise on reload ≈ **3 s** [R155].

**Dryer model** (§2.3b, [R155][R160]): purge loss **10-18 %** of rated flow (measured **12 %** for
heater-purge twin towers), rated PDP **−40 °C**, dryer pressure drop **3-5 psi**, regeneration on a
**timed cycle** with a moisture-dependent correction rather than continuous dew-point control. Take
the **tower period from the `TOWERS` channel of MetroPT-2/3** [R88][R89] rather than inventing one —
that is a measurement from a real metro dryer in service. Per §0, drive dryer load from **inlet
temperature** at a Singapore-saturated inlet, and hold it at the rated condition year-round [R152].

**Add the coupled dryer failure chain** [R160]: low supply pressure → higher volumetric flow →
reduced purge → **incomplete regeneration** → degraded dew point. A reservoir leak therefore degrades
the dryer. This is a far better demo than two independent faults, and it is cited.

**Acceptance tests (unchanged in spirit, now with an upper bound).** `s = 1` must reproduce the
**`t_off` shrinkage** seen in MetroPT-3 late May - early June 2020 [R88], and `idle_run_ratio` must
cross its healthy 5th percentile with 1-4 weeks of lead [R101]. Additionally the pressure-decay rate
at the top of our severity scale should stay **well below** the US freight regulatory limit of
**5 psi/min (0.34 bar/min) / 60 cfm (28.3 NL/s)** [R156] — that limit is a whole-consist brake pipe,
roughly 10× our `s = 1`, and is the right label for a "catastrophic" class if we want one.

**Lead-time targets to state as acceptance criteria:**

| Fault | Target lead | Source |
|---|---|---|
| Any APU failure | **≥ 2 h before removal**, minimal false alarms (operator requirement) | [R90][R94] |
| Air leak | **≥ 150 min before `LPS` fires** | [R93] |
| Oil leak | **≥ 2 days** | [R93] |
| Brake-pipe air leak (fleet, from on/off logs only) | **1-4 weeks** | [R101] |

**False-alarm suppression rule** [R95]: raise a fault only when **≥ 3 of
`{TP2, TP3, Motor_current, Oil_temperature, idle_run_ratio}`** are simultaneously anomalous. ~25 lines.

**Do not overclaim multi-class.** MetroPT-3 contains **air leaks only** [R88]; multi-class pneumatic
fault identification must be demonstrated on MetroPT-1 [R132] / MetroPT-2 [R89] or on our simulator,
and the write-up must say which.

### 4.3 Bearing simulator (`sim/bearing.py`) — **substantially rewritten in the W1 pass**

Every numeric constant in this section was previously "plan value, keep" with no citation. Below,
each is either **derived from a published correlation**, **anchored to a measured table**, or
**explicitly labelled as our assumption**. Two of the old constants are **falsified** and must change.

#### 4.3.1 Thermal model — first principles

**Heat generation** uses the standard Palmgren friction-torque model, given explicitly for a railway
axle box [R150]:

```
P      = M * omega,  omega = 2*pi*n/60                              (Eq. 7)
M      = M0 + M1 + M2                                               (Eq. 8)
M0     = 160e-7 * f0 * Dm^3                       for nu*n <  2000  (Eq. 9)
M0     = 1e-7  * f0 * (nu*n)^(2/3) * Dm^3         for nu*n >= 2000
M1     = 2 * f1 * Y * F_r * Dm                                      (Eq. 10)   # radial load term
M2     = f2 * F_a * Dm                                              (Eq. 11)   # axial load term
```

**Heat rejection** uses the axle-box convection correlations given in full in [R149]. The one that
matters for us is the external axle-box-to-air forced convection, Eq. 25:

```
h_a = 0.03 * (k_air / (2*R_a)) * (2*u_air*R_a / nu_air)^0.57        =>  h_a  proportional to  v^0.57
```

with inner-ring/lubricant `h_i = 0.19*(k_l/(2*R_iex))*(Re_Di^2 + Gr_Di)^(1/3)` (Eq. 20), roller
`h_r = 0.33*(k_l/D_r)*(omega_r*D_r/nu_l)^0.57` (Eq. 23), fixed outer ring natural convection
`h_o = 0.53*(k_l/(2*R_oin))*(Gr_Do*Pr_Do)^0.25` (Eq. 24) and radiation per Eq. 26 [R149].

**Validation anchors from a real fleet** [R149] Table 1 (8 cars of a high-speed train, measured vs
simulated axle-box bearing temperature; max simulated-vs-mean-measured error ≈ 2.4 K):

| Case | Speed | Ambient | Measured range | Simulated |
|---|---|---|---|---|
| 1 | 200 km/h | ≈15 °C | 44.64-53.08 °C | 47.62 °C |
| 2 | 300 km/h | ≈25 °C | 50.08-60.43 °C | 52.84 °C |
| 3 | 200 km/h | ≈15 °C | 53.83-64.73 °C | 58.93 °C |
| 4 | 300 km/h | ≈25 °C | 59.57-71.39 °C | 63.76 °C |

**The healthy 21-25 K anchor is now derived, not asserted.** Take the case-1/case-2 rise above
ambient: ≈**30-38 K at 200 km/h**. Under `Q ∝ v` (friction power is torque × speed, and the
load-dependent M1 term dominates at constant load) and `h ∝ v^0.57` [R149], the steady rise scales as
`ΔT ∝ v / v^0.57 = v^0.43`. Scaling 200 → 80 km/h gives `(0.4)^0.43 = 0.674`, i.e. **20-26 K** — which
lands exactly on the plan's 21-25 K at 80 km/h. *This scaling is our arithmetic on their published
numbers and correlation*, not a figure from either paper, and it ignores the lower axle load of a
metro (12 t vs a high-speed EMU's ~17 t), which pushes it lower still. **Keep 21-25 K, but state it
as a derivation from [R149] + [R150], and implement `ΔT ∝ v^0.43` rather than a lookup.**

Conditions caveat to state on the slide: both sources are **Chinese high-speed EMUs at 200-350 km/h,
ambient 15-25 °C** — 2.5-4× our speed and ~13 K cooler than Singapore. The absolute rises are an
**upper bound** for a metro; the correlations themselves are speed-generic.

#### 4.3.2 The severity law that was wrong — the axle box barely feels the defect

**This is the single most important correction in the W1 pass.** [R149] swept wheel-flat length 0-60
mm (flat depth `D_f = L_f²/(16R)`). Going from a 30 mm to a 60 mm flat raises:

| Node | Temperature rise |
|---|---|
| Roller, large end | **+15.60 K** |
| Roller, small end | +16.84 K |
| Cage | +15.74 K |
| Inner ring | +15.38 K |
| Outer ring | +13.23 K |
| **AXLE BOX** (where our sensor is) | **+1.02 K** |

The axle box is roughly **15× less sensitive to defect severity than the internal rings**. Speed
sensitivity under a fixed 30 mm flat, 300 → 350 km/h, shows the same pattern: axle box **+3.75 K**
versus roller +11.25 K, cage +10.38 K, inner ring +9.27 K, outer ring +8.59 K [R149].

**Consequence for the simulator:** a severity law that drives `T_box` hard is physically wrong and
will make the severe class trivially separable, inflating every detector's score. `T_box` should move
by **~1 K per severity step**, buried in an inter-box spread of **8.4 K** on healthy bearings [R149]
and a diurnal ambient swing of ~8 K (§0). **That is exactly why vibration is the early indicator and
temperature is a late one [R109] — we now have the number that proves it, and it becomes a headline
slide rather than an assertion.**

#### 4.3.3 Vibration severity laws — two of ours are falsified

The W0 laws were `vib_rms ∝ (v/22)^1.2·(1+3s)`, `vib_kurt = 3 + 6s²`, `crest = 3 + 4s`,
`bpfo_band ∝ s·(v/22)²`. Measured evidence:

| Constant | W0 value | Measured | Verdict |
|---|---|---|---|
| Healthy **kurtosis** | 3 | **2.76-2.96** (four speeds) [R157] | **CONFIRMED** |
| Healthy **crest factor** | 3 (`crest = 3+4s` at `s=0`) | **4.22-5.59** [R157] | **FALSIFIED — raise the intercept to ≈4.5** |
| **Kurtosis monotonic in severity** (`3 + 6s²`) | monotonic rise | inner race by defect size: **5.56 → 21.69 → 8.06 → 3.29**; outer race **7.85 → 3.02 → 23.16** [R157] | **FALSIFIED — must rise then COLLAPSE** |
| Speed exponent for **defective** bearings | 1.2 | fitted 1.0-1.3 for inner/outer-race defects (*our power-law fit to [R158] Table 3*, speed ratio 41/17 Hz) | **CONFIRMED for the defect term** |
| Speed exponent for the **healthy** floor | 1.2 | fitted **≈2.3** healthy (no load 2.30, loaded 2.28) [R158] | **TOO LOW — use ≈2 for the broadband healthy floor** |
| Healthy axle-box acceleration magnitude | unstated | **8.8 m/s² RMS vertical at 300 km/h** (sim 7.6, 15.6 % low); bogie frame 1.6 m/s²; peak near 100 m/s² [R150] | **NEW anchor — our healthy `vib_rms` must be the right order of magnitude** |

**The kurtosis correction matters most.** A monotonic kurtosis law makes the severe class trivially
separable; the real physics is spall widening — a large distributed defect stops being impulsive, and
at the largest inner-race defect measured kurtosis (3.29) is back at the **healthy** value [R157].
Replace `vib_kurt = 3 + 6s²` with a rise-then-collapse shape, e.g.

```
vib_kurt  = 3 + K * s * exp(-(s/s_peak)^2)      # peaks near s_peak ~ 0.35, returns toward 3 at s->1
crest     = 4.5 + 4*s * exp(-(s/s_peak)^2)      # same shape; healthy intercept 4.5 per [R157]
vib_rms   = A_h * (v/22)^2.0  +  A_d * s * (v/22)^1.2       # healthy floor ^2, defect term ^1.2
bpfo_band = B * s * (v/22)^2                                 # unchanged; no source found either way
```

`bpfo_band ∝ s·(v/22)²` is the one law we could **not** source; keep it and label it **our
assumption**, not a citation.

**Caveats to state honestly.** [R157] is a 35 mm-bore lab motor bearing at ≈29 Hz on CWRU data (which
has documented labelling caveats [R142][R115]), not a 180 mm-pitch-diameter axle box at 6-8 Hz — use
it for the **shape** of the severity law, not for absolute g values; and its 1730-1797 rpm span is
only 4 %, so it **cannot** source a speed exponent. [R158] reports spectrum **peak** amplitudes
(gPk), not band RMS, under a 5 kg lab load rather than a 12 t axle, and its loaded outer-race case is
non-monotonic in speed (fitted exponent −0.18), so no single exponent fits everything — treat the
exponents as order-of-magnitude guidance.

**Better still, fit the exponents on data we already hold.** The Ottawa variable-speed set sweeps
**13.7-28.9 Hz shaft speed within single 10 s records** at 200 kHz [R159][R118]. So we can fit
RMS / kurtosis / BPFO-band-energy versus speed **per health state, on our own dataset, in an
afternoon** — a stronger and self-contained justification than any literature exponent, and it also
gives the healthy-vs-defective contrast at matched speed. **Do this before the hackathon.** Note the
shortfall, though: [R159] states **no defect size and no applied load and carries no temperature
channel**, so Ottawa cannot ground a severity axis or a thermal model. No public bearing dataset we
found pairs a **documented thermal model or ambient log** with vibration under a permissive licence;
FEMTO/PRONOSTIA carries temperature alongside vibration [R143] and is the closest available, and the
PoliTo set is the best sensor match but is under a restricted-use agreement [R139]. **The bearing
dataset choice therefore does not change** — Ottawa variable-speed stays primary, FEMTO is added for
the thermal + RUL arm.

#### 4.3.4 Thresholds to encode — updated

| Quantity | Value | Status |
|---|---|---|
| Functional failure (our sim), absolute | `T_box > 90 °C` | **our definition**, and it now sits defensibly *between* the two published operator lines below |
| Functional failure (our sim), differential | `T_box − median(same-side peers) > 30 K` | **aligned with [R151]**, which uses `dT > 30 °C` for level 1-3 — note it is the median of peers on the **same side**, and the published persistence criterion is **10 measurements within 30 days**, not 5 minutes. Our 5-minute persistence is an *on-board continuous-monitoring* choice; state it as ours. |
| Operator absolute, level 4 (immediate) | **> 80 °C** | [R151] — a real operator rule, published |
| Operator differential, level 3 (immediate) | **> 50 °C** | [R151] |
| Wayside immediate-stop | **> 115 °C** | [R151] |
| Slow-degradation rule | side-difference deviating **> 3.5 σ from the train median for ≥ 10 measurements in 30 days** | [R151] |
| Data-quality guard | **≥ 4 boxes on one train > 50 °C in one pass ⇒ blame the station, suppress** | [R151] |
| Healthy rise above ambient at 80 km/h, 12 t axle | **21-25 K**, implemented as `ΔT ∝ v^0.43` | **derived** from [R149] Eq. 25 + Table 1 and [R150] Eqs. 7-11 (our scaling arithmetic) |
| Healthy absolute at Singapore ambient | **53 °C** at 28 °C mean, **57 °C** at 32 °C daily max, **62 °C** on the 37 °C record day | §0, [R152][R153] |
| Max observed in service (high-speed fleet) | **82 °C** motor car, **69.8 °C** trailer car | [R150] |
| Inter-box spread, healthy, matched condition | **8.4 K** across 8 cars | [R149] Table 1 |
| Wayside scan-location bias | **±5 K** | **our modelling assumption**, motivated by [R112] (IR scan location dominates the reading) and bounded by the 8.4 K healthy spread in [R149]. Not a published figure — label it. |
| Onboard monitoring mandated above | **250 km/h** design speed | [R163] cl. 4.2.3.3.2 — **irrelevant to an LTA metro**, which falls in the either/or band, so on-board bearing monitoring is a voluntary engineering choice. Pitch it that way. |
| EN-standard alarm pair | ~~`> 95 °C` / `> 56 K`~~ | **WITHDRAWN** — not in any EN standard, not on the vendor page, not in the TSI. See 3.3. |
| Regulatory ambient design envelope | zone T1 **−25 to +40 °C** | [R163] cl. 4.2.6.1 — a citable ambient ceiling for the sweep |

Report the 30 K / 90 °C pair as **our** functional-failure definition, benchmark it against the
**published NS operator rules** [R151], and say plainly that no harmonised European threshold exists
[R166][R167].

**Model wayside measurement noise explicitly** [R112][R151]. Keep the `T_box_wayside` variant:
sampled only at "detector passes" (a few times per day), with the ±5 K scan-location bias above, a
miss/false-flag rate, and — new — the **station-fault mode** from [R151]: occasionally bias *all*
boxes of a passing train upward, which is exactly what the `≥ 4 boxes > 50 °C` guard exists to catch.
The demo then contrasts continuous onboard monitoring against wayside snapshots *and* shows our
detector surviving a bad detector station.

**Detector is a residual, not a threshold** [R110]. Fit a healthy thermal model with inputs
`{v, T_amb, load_frac, T_box of the same-side peers and the opposite box}` and score
`thermal_residual = T_box − T_box_pred`. Per [R151], normal variation is driven by ambient, sunshine
and **duty cycle** ("the bearing cools significantly by the wind while driving and warms up while
standing still"), so `dwell_fraction_last_hour` must be a model input too — a stationary train at a
terminus is the confounder that will generate our false alarms in Singapore.

**Vibration features — add explicit fault-frequency harmonics** so the same `features/vibration.py`
runs on Ottawa raw data and on our synthetic waveform:

```
env_bpfo_h1..h3, env_bpfi_h1..h3, env_bsf_h1..h2, env_ftf_h1     # envelope-spectrum harmonic amplitudes
sk_band_fc, sk_band_bw, sk_max                                    # kurtogram-selected demodulation band (auditable)
mfcc_1..13, ams_band_1..8                                         # MFCC + amplitude modulation spectrogram
rms, kurtosis, crest, skew, p2p, shape_factor, impulse_factor     # time-domain
```

[R117] for the kurtogram/envelope chain, [R120][R121] for MFCC/AMS, [R158] for the BPFO/BPFI/BSF
per-revolution factors.

**Promote the variable-speed Ottawa set to primary** [R118][R159]. A metro axle box never runs at
constant speed. The variable-speed set ships a **1024-CPR encoder channel at 200 kHz** alongside the
accelerometer, across four speed profiles, which makes **true computed order tracking** possible — and
therefore a **tacho vs tacholess ablation** [R122]. UORED-VAFCLS (`y2px5tg92h`) [R119] becomes the
**constant-speed ablation**.

**Splits** [R115][R58]: bearing-wise for Ottawa (no physical bearing in both train and test),
recording-level CV with envelope preprocessing, seed/unit-wise for our simulator. **Report the gap
between window-random and bearing-wise accuracy as the headline honesty number.**

**Unseen-fault-type ablation** [R121]: hold out one damage type entirely and test generalisation —
a published result shows MFCC features do generalise to unseen damage types, so this is a fair,
winnable ablation rather than a gotcha.

### 4.4 Cross-subsystem

- **Peer normalisation everywhere.** Doors compare against the 7 siblings of the unit [R78]; axle
  boxes against the other 7 boxes of the bogie/car and against the opposite box on the same axle
  [R110][R112]; compressors against the other trains in the fleet [R101]. One
  `features.peer_normalise()` serves all three, and it is the single strongest UI story we have.
- **Severity staging, not binary health.** Two independent door groups [R72][R81], the brake-pipe leak
  work [R101] and the axle-box health-index work [R123] all converge on: build a degradation
  indicator, cluster it into ordinal severity levels, then estimate time-to-next-level. Use ordinal
  `severity_stage` as the shared health representation across all three subsystems, feeding both the
  CBM step display and the RUL arm.
- **k-of-n corroboration** [R95] and **contextual pre-filtering** [R101] are the two cheap false-alarm
  suppressors that every subsystem inherits.

### 4.5 Prognostics and RUL — making the three subsystems symmetric

*Added in the W1 pass. The W0 ladder had a RUL arm for the **door only**, while `datasets.md` was
ingesting three bearing run-to-failure sets for a capability nothing delivered. That is now fixed
from the method side; the datasets stay, and here is what they are for.*

**The shared representation is already chosen: ordinal severity stage + time-to-next-level (§4.4).**
Four independent lines converge on it — two door groups [R72][R81], the brake-pipe leak work [R101]
and the axle-box health-index work [R123]. The **General Path Model** is the published name for this
pattern: identify a degradation measure characterising progression to failure, fit a functional form
to it, extrapolate to a threshold, and update the fit as data arrives [R199]. It needs **only a
monotonic health index and a threshold — no run-to-failure labels** — so it ports to the pneumatic
and bearing subsystems at essentially zero marginal cost, and it is ~100 lines with
`scipy.optimize.curve_fit`.

**Choosing the health index before fitting anything.** Score candidate HIs (`vib_rms`, kurtosis,
band energy, `idle_run_ratio`, `dtw_to_ref`) on **prognostic suitability** — monotonicity,
prognosability, trendability — rather than on downstream RUL error [R218]. This is a label-free way
to pick between competing indices before spending GPU time. (Exact metric definitions `UNVERIFIED`;
read the PDF before quoting formulas.)

**Stochastic degradation is the right family at our sample sizes.** Generalization bounds for RUL
give learning rates in `n` = the number of **complete run-to-failure trajectories**, and show that
embedding degradation physics can cut data requirements by up to **two orders of magnitude** for deep
networks [R221]. XJTU-SY has **15** bearings [R124][R227] and FEMTO ~17 [R143][R228]. That is
squarely the small-`n` regime where a deep RUL net is not sample-efficient and a parametric
degradation model is. Concretely:

| Model | When it applies | Cost | Source |
|---|---|---|---|
| **Wiener process** (drift-diffusion), RUL by first hitting time | HI may fluctuate up and down | ~40 lines: MLE of `μ`, `σ` from HI increments; inverse-Gaussian first-passage gives a free predictive interval, mean `(w − x_t)/μ` | [R196][R195] |
| **Gamma process** | HI strictly monotone increasing — bearing spall growth, leak growth, fouling | ~40 lines: MLE of shape/scale on increments, RUL by gamma quantiles | [R197] |
| **Inverse-Gaussian process** | monotone but gamma fits poorly | ~30 lines with `scipy.stats.invgauss` | [R198] |
| **General Path Model** | any ordinal severity index + level threshold | ~100 lines, `curve_fit` | [R199] |
| **Similarity-based RUL** | a library of complete run-to-failure trajectories exists | ~100 lines, DTW or Euclidean matching over a smoothed HI. **Requires run-to-failure data** — this is what XJTU-SY and FEMTO are for | [R200] |
| **Particle / Kalman filter state tracking** with uncertainty propagation | a state-space degradation model exists (which our simulator gives us for free) | `progpy` state estimators + `MonteCarlo` predictor [R201]; or `filterpy` (MIT) for Kalman only [R202] | [R201][R202] |

**Licence note that decides a dependency:** ProgPy is **NASA-1.3**, not an OSI-standard permissive
licence [R201] — check the NEBULA X IP rules before it ships. FilterPy is MIT but has no particle
filter and is unmaintained since 2018 [R202]; a bootstrap particle filter is ~60 lines.

**Survival analysis is the arm that fits the pneumatic subsystem, because we have no run-to-failure
compressors.** Cox proportional hazards [R205] and random survival forests [R206] need only
**censored time-to-event data** — per-unit duration, an event indicator and covariates — which is
exactly what a fleet of compressors with occasional maintenance records looks like, and exactly the
shape of SCANIA Component X [R134][R209]. A published PHM paper puts RUL regression and survival
analysis on the **same ladder** and compares them fairly on C-MAPSS and a Volvo truck fleet [R207]
(`NUMBERS UNVERIFIED` — which family won is not on the landing page). The decisive practical result:
including data from assets that **did not fail** improves prediction [R210], and `SurvLoss` makes
that a ~30-line change to an ordinary regression loss rather than a new model [R208].

**Library choice, on licence grounds:** `lifelines` (**MIT**) for Cox and Weibull-AFT [R211];
`xgbse` (**Apache-2.0**) for gradient-boosted survival with calibrated survival curves — which map
directly onto "probability of reaching severity level k within h days" for the twin UI [R213];
`SurPyval` (**MIT**) or `reliability` (LGPL-3.0) for Weibull/exponential population fitting with
censoring [R204][R203]. **`scikit-survival` is GPL-3.0-or-later** [R212] — importing it makes the
deliverable a derived work, so use it only if RSF is essential and the rules permit.

**Honest gap, stated so we do not overclaim.** Searching for prognostics (as opposed to detection) on
railway brake air supply returned exactly **one** paper with a severity-to-RUL formulation — the
Dutch Railways brake-pipe work we already cite [R101] — and a 2023 PRISMA systematic review of the
entire compressed-air ML literature is scoped **entirely to anomaly detection with no RUL section**
[R226]. So the pneumatic RUL arm cannot be filled by citing domain literature; it is filled by porting
a generic degradation model (General Path Model or gamma process) onto the `idle_run_ratio` severity
index we already have. Say that on the slide — it is a defensible contribution, not a gap.

**RUL evaluation — the prognostics analogue of our no-point-adjustment rule.**

- **Metrics**: α-λ accuracy, prognostic horizon, cumulative relative accuracy and monotonicity, all
  incorporating probabilistic uncertainty rather than a point estimate [R215][R216]; `progpy`
  implements exactly these four names on a `ToEPredictionProfile` [R201]. α-λ needs a **predictive
  distribution**, which is another reason to prefer the stochastic-degradation family.
- **The C-MAPSS asymmetric score has a pitfall we must state.** It penalises late (optimistic)
  predictions exponentially more than early ones [R217], but it is unnormalised, so a single badly
  late unit dominates the fleet total and the score is **not comparable across datasets with
  different unit counts or life lengths**. Always report it beside RMSE and a per-unit error
  distribution, never alone.
- **Leakage control is worth as much as model choice.** Device-level leakage optimistically biases
  RUL benchmarks; the fix is **leave-one-device-out** evaluation with a leakage-safe preprocessing
  pipeline, and under it plain SVR was the most consistent model [R219]. An independent 2026 study
  reports that naive splitting inflates accuracy from a genuine **20-60 % to 99.9 %** on the same
  data, and prescribes chunk-based leakage-audited splits with 5 seeds and significance testing
  [R220]. **Transplanted rule for us: leave-one-bearing-out on XJTU-SY / FEMTO, never random window
  splits, and fit scalers and HI normalisation on training bearings only.** This is the direct
  prognostics analogue of the bearing-wise rule in §3.4.
- **Do not assume deep wins.** On C-MAPSS FD003, XGBoost (RMSE 13.36) beats both a 1D CNN (15.68)
  and an LSTM (14.20) [R222]; and tabular foundation models with in-context learning report the best
  average ranks across prognostic and diagnostic PHM tasks specifically in **low-data regimes**
  [R223] (`UNVERIFIED` which datasets). Both point the same way as V1 for anomaly detection.

**What this buys each subsystem:**

| Subsystem | RUL arm | Needs run-to-failure? | Dataset it justifies |
|---|---|---|---|
| **Door** | DTW-to-reference → k-means severity → dwell time (already MUST) [R72][R81]; similarity ensemble [R70][R200]; gated EOL [R71] | similarity arm: yes | PHME 2026 [R69] |
| **Pneumatic** | General Path Model on `idle_run_ratio` severity [R199][R101]; gamma process [R197]; Cox / RSF on fleet censored data [R205][R206] | **no** | SCANIA Component X [R134] for the survival demo |
| **Bearing** | Gamma or Wiener process on the envelope-band HI [R197][R196]; similarity-based RUL [R200]; particle-filter state tracking [R201] | similarity arm: yes | **XJTU-SY [R124], FEMTO [R143], Paderborn R2F [R138] — now earning their ingest** |

---

## 5. Background reading for the related-work slide (cited, not implemented)

These are verified but carry no implementation task. Use them for the related-work slide and to show
the field was surveyed, not cherry-picked.

- **Rail PdM taxonomy by task / method / metric / equipment / dataset** [R104] — structured exactly
  like the benchmark table we have to produce, from the MetroPT authors, CC BY 4.0. Use its taxonomy
  as the skeleton so reviewers can map our rows onto the existing literature.
- **Deep learning for anomaly detection in railway systems: a structured survey** [R103] — the most
  recent structured survey; cites the MetroPT descriptor, so its scope covers rolling-stock telemetry
  rather than only infrastructure. `NUMBERS UNVERIFIED` — confirm it covers doors and bearings before
  citing it as such.
- **Time series analysis in compressor-based machines: a survey** [R105] — compressor-specific rather
  than railway-specific; its **change-point detection** section is the part we mined.
- **Fault Diagnosis of Bogie Bearings in High-Speed Train: A Review** [R126] — scoped to bogie
  bearings; names the open problems we inherit (weak early fault signatures under wheel-rail
  excitation, scarce field fault labels). `NUMBERS UNVERIFIED` (Springer auth redirect).
- **Deep Learning for TSC and Extrinsic Regression: A Current Survey** [R61] — justifies our deep
  architecture choices (LITE, ConvTran) without re-deriving the landscape during the 48 h.
- **A two-stage framework for early failure detection on metro trains** [R96] — the most recent
  journal treatment from the group that owns the MetroPT data, extending the forecast-then-detect
  line of [R95]. Crossref lists CC BY, so figures may be reusable in the pitch deck — worth one human
  click. `NUMBERS UNVERIFIED`.
- **ChronosAD** [R39] — a TSFM used as a frozen zero-shot feature extractor plus a BiLSTM +
  multi-head-attention temporal block, reporting +4.72 % AUC and +6.60 % AP on average over prior
  methods across 11 benchmarks spanning industrial, medical, cyber-physical and automotive systems.
  `NUMBERS UNVERIFIED` beyond the abstract. This is the *frozen-embedding + small supervised head*
  pattern we adopt in the bearing ladder, rather than raw residual thresholding.
- **Sound-source-localisation gas leakage detection in railway pneumatic brake systems** [R107] —
  answers "**where** is the leak", not "is there a leak". Needs a mic array, so it is out of scope
  for a telemetry-only build; one line on the sensor-fusion / future-work slide. `NUMBERS UNVERIFIED`.
- **Change-point and drift surveys** — [R174] (offline CPD organised as cost function × search method
  × constraint, explicitly multivariate), [R190] (supervised and unsupervised CPD with comparison
  criteria, free via PMC), [R191] and [R192] (concept drift adaptation), and [R193]
  (**performance-aware** drift detectors — the established-family citation for firing on a model's
  residual degradation rather than on instantaneous magnitude, which is exactly verdict V4).
- **Critical review of ML approaches to RUL** [R229] — `CONTENT UNVERIFIED` (null abstract in
  OpenAlex, Springer page not opened), so we cannot confirm whether it critiques evaluation practice,
  leakage or simple-vs-deep comparisons. Listed only as a lead if a **peer-reviewed** "limitations"
  citation is wanted in place of the two arXiv leakage preprints [R219][R220].
- **Prognostics reviews** — [R195] (statistical data-driven RUL, the umbrella reference), [R224]
  (bearing RUL organised as data → health indicator → algorithm → evaluation, with the dataset
  catalogue; it surveys rather than adjudicates), [R225] (transfer learning for bearing RUL from an
  industrial-deployment perspective, `CONTENT UNVERIFIED`) and [R226] (PRISMA review of compressed-air
  anomaly detection — cited for what it *lacks*: no RUL section at all).
- **TimeEval** [R189] — peer-reviewed evaluation of **71** detectors over **976** datasets reporting
  **runtime alongside accuracy**; the one verified source for the computational-cost half of a ladder
  decision. `NUMBERS UNVERIFIED` — do not quote per-algorithm winners.
- **Dive into Time-Series Anomaly Detection: A Decade Review** [R194] — the current taxonomy from the
  group that produced VUS and TSB-AD, so it is consistent with our primary metric. `NOT PEER-REVIEWED`;
  framing only.
- **Industrial fault diagnosis: pneumatic train door case study** [R83] — the long-standing IMechE
  Part F reference for **pneumatically operated** doors, which matters because a large share of real
  fleets (and the C151A after refurb) use electrically controlled, pneumatically operated doors. It
  also bridges our door and brake-air-supply subsystems, since both then share pressure-channel
  diagnostics. Outside the 2015-2026 window; lineage only. `NUMBERS UNVERIFIED`.
