# Finder 2 — refrigerant leak / undercharge diagnosis in HVAC and train air-conditioning

Sweep for the **PS3 ACV** subsystem: 6 labelled cases + 1 held-out test, each a train of 8 cars,
30 s telemetry over 3–4 days, per car `setting mode`, `running mode` (strings: Automatic Cooling /
Full Cooling / Half Cooling / Stop …), cooling and heating control temperature, indoor and outdoor
average temperature, load-halved flag, information-valid flag (one case carries ~60 params/car).
Exactly one car leaks; output is a **ranking of the 8 cars**, scored `(n − (r − 1)) / n`.

Existing team research checked first: `docs/research/model_ladder.md`, `rail_phm.md`, `datasets.md`,
`references.md` contain **nothing on HVAC, refrigerant or vapour-compression FDD** (grepped for
`hvac|air.condition|refriger|charge|chiller|superheat|subcool|RP-1043|ASHRAE`; only hits are about
*compressed-air* charge/discharge in the pneumatic subsystem and Trafikverket detector access being
"free of charge"). So this whole area is new to the corpus. The one thing that *does* carry over is
the **peer-normalisation machinery already built for doors and bearings** —
`features.peer_normalise()`, cited to [R78] in `model_ladder.md` §1a and `rail_phm.md` §"Peer
normalisation everywhere" — and it is exactly the right primitive here.

---

## Practitioner defaults — what a rail/PHM engineer builds first with ~12 h

The decisive constraint is that **the ACV telemetry has no pressures, no superheat/subcooling, no
power and no supply-air temperature**. Almost the entire refrigerant-charge FDD literature (virtual
charge sensors, decoupling features, RP-1043-style residual rules) is built on exactly those signals,
so it cannot be transplanted — but it tells you *what the fault does*, and the smart-thermostat and
peer-fleet branches of the same literature tell you how to see it with temperatures and on/off state
alone. Undercharge reduces cooling capacity; a capacity-starved unit therefore (a) spends **more time
in the hardest cooling mode** and cycles differently, (b) **pulls the cabin down more slowly**, and
(c) at high ambient, **fails to reach its control temperature at all**. ORNL's 2024 field tests [C6]
are blunt about the ordering: indoor air temperature alone barely sees low-intensity undercharge —
unmet hours only appear at 40–50 % charge loss — while **runtime fraction and supply-air temperature
rise monotonically with undercharge**. So build duty-cycle and pull-down-rate features first and
treat setpoint shortfall as the confirming, late-stage feature, not the primary one.

The method to copy almost verbatim is **Guo, Chen & Xiao 2024 [C1]** (electric-bus AC fleet, 38 units,
Shenzhen): assume the majority of peers are healthy, learn per-peer regressions of each feature on the
operating conditions, predict the target unit from *its peers'* models, take the **median** of peer
predictions (robust to a faulty peer — they show it survives up to 1/3 faulty training units),
residual = measured − predicted, then aggregate signed residuals into a scalar **fault index**
`I = Σ_k sign_k · (R_k / IQR(R_k))² · IQR(SND)` where `sign_k` is the *known thermodynamic direction*
of that feature under undercharge. It detected 11 of 12 technician-verified faults including all
refrigerant-undercharge cases. Here the "peer group" is free and perfectly matched: the 8 cars of the
same train, same timestamp, same weather, same schedule. Concretely, for ~12 h:

1. **Load and gate (2 h).** Read each file by its own headers (the 60-param file must not break the
   loader). Drop rows where `information valid = 0`; drop cars/rows in `Stop`; keep only cooling-mode
   rows; flag `load-halved` rows separately (halved load is a legitimate capacity reduction that will
   masquerade as a leak if you pool it). Apply a Kim et al. [C13] style moving-window steady-state
   gate (σ over a 5–10 min window below a threshold) to separate pull-down transients from steady
   operation — you want both, but scored as different features.
2. **Per-car features (3 h), all computed on the same gated rows.** (i) fraction of cooling time in
   Full Cooling and in Half Cooling; (ii) mode-transition/cycle count per hour and mean on-time per
   cooling episode (Chintala's "thermostat drive cycles" [C4]); (iii) pull-down slope
   `d(T_indoor)/dt` over the first N minutes of each Full-Cooling episode, and time-to-setpoint;
   (iv) steady-state shortfall `T_indoor − T_control` and its high-ambient tail (unmet degree-minutes
   above the setpoint, i.e. ORNL's "unmet hours" [C6]); (v) `T_indoor − T_outdoor` lift achieved,
   which is the only capacity proxy available without supply-air temperature.
3. **Peer residual + fault index (3 h).** For each feature, either (cheap) robust z of the car against
   the median/MAD of its 7 siblings *within ambient bins*, or (better, still cheap) fit a per-peer
   ridge/GPR of the feature on `(T_outdoor, T_control, load_halved)` on the other 7 cars, predict the
   target, take the median peer prediction, residual. Combine with the signed-squared-IQR index of
   [C1]. Rank the 8 cars by the index. This is ~60–100 lines on top of the existing
   `features.peer_normalise()`.
4. **Validate honestly (2 h).** Leave-one-*case*-out over the 6 labelled cases, reporting the actual
   competition metric. **Sanity anchor: a uniformly random ranking of 8 cars scores 0.5625**
   (mean of `(8−(r−1))/8` over r = 1..8 = 36/64), and always ranking car 01 first scores the same in
   expectation if the faulty car is uniform. Any pipeline not clearly above 0.5625 on LOO-CV is not
   working. Also check the label distribution: if the faulty car index is *not* uniform across the 6
   cases you have a position confound (end cars carry more solar and door load than middle cars), and
   peer exchangeability — the assumption the whole method rests on — is violated; correct it by
   subtracting a per-position offset estimated across cases before ranking.
5. **Do not (0 h saved).** Do not train a supervised classifier or any deep model on 6 cases × 8 cars
   = 48 rows with one positive each; do not attempt a virtual-charge-sensor regression ([C7], [C8])
   — the required subcooling/superheat inputs do not exist in this telemetry; do not use the 60-param
   file's extra columns as features unless they also exist in the test file (they probably do not).

Expected ceiling, calibrated from the literature: thermostat-only methods detect ~40 % undercharge at
~70–82 % accuracy [C4][C5], and pressure/temperature-rich methods detect ~5 % charge loss [C2]. With
temperatures + modes only, expect to reliably catch *advanced* leaks and to be near-chance on early
ones — which is precisely why the rank-decay metric, not top-1, is the thing to optimise, and why
committing a confident top-2 is worth more than a confident top-1 with a bad tail.

---

## Ranked candidates

Ranked by relevance × feasibility. `subsystem_relevance` 0–3 where 3 = directly transferable to
"rank 8 identical peer AC units, temperature + mode telemetry only".

| # | Short name | Rel. | Impl. h | One-line reason |
|---|---|---|---|---|
| C1 | Guo/Chen/Xiao — peer-fleet GPR FDD for bus AC | 3 | 3–4 | The blueprint: peer-median residuals → signed fault index, finds undercharge unsupervised |
| C2 | Rossi & Braun — statistical rule-based FDD | 3 | 2 | Founding temperature-only residual + directional-rule method; ~5 % charge loss detectable |
| C6 | ORNL field study of under/overcharge | 3 | 0 (evidence) | Says which observable signals move under undercharge and which do not |
| C4 | Chintala/Winkler/Jin — thermostat drive cycles | 3 | 4–6 | Cooling-cycle features from thermostat-only data; 82 % on 40 % undercharge |
| C3 | Guo & Rasmussen — population benchmarking | 3 | 3 | Population-as-benchmark + within/between comparison; explicitly detects refrigerant leaks |
| C9 | Yoo/Hong/Kim — leak detection with limited sensors | 3 | 2 | Leak detection when subcooling is *not* measurable — the exact constraint here |
| C10 | Jiang/Chen/Yang — metro train AC XAI FDD | 3 | n/a | Only metro-train-AC FDD paper found; domain framing + interpretability |
| C13 | Kim et al. — steady-state detector | 2 | 1 | The transient/steady gate every FDD pipeline needs; moving-window σ |
| C11 | Tormos et al. — physics-informed XAI, bus fleet | 2 | 2 | Unsupervised fleet anomaly detection on imperfect HVAC telemetry, CC BY |
| C5 | Chintala et al. — sensitivity analysis on real data | 2 | 2 | Same algorithm on *real* faulted-home data: 70.6 % undercharge — the reality check |
| C12 | Michau & Fink — fleet-based unsupervised FD | 2 | 4 | Formalises "borrow health from the fleet"; 112 units |
| C7 | Kim & Braun — virtual refrigerant charge sensor | 2 | 6 | Charge from surface temperatures within 10 %; needs sensors we lack — read for direction signs |
| C8 | Li & Braun — decoupling features | 2 | 6 | Why a feature must be *uniquely* tied to one fault; multi-fault contamination |
| C14 | Granderson et al. — LBNL FDD datasets | 2 | 2 | Open labelled HVAC fault data to dry-run the pipeline on |
| C15 | Llopis-Mengual/Yuill/Navarro-Peris — soft faults in field data | 2 | 3 | Time-series degradation assessment of soft faults from field data |
| C16 | Guo & Rasmussen — modified Mann-Kendall | 1 | 1 | Trend test on a health index; 3–4 days is short but a monotone drift check is ~free |
| C17 | Gálvez et al. — railway HVAC hybrid FDD + RUL | 1 | 6 | Rail-carriage HVAC, CC BY, but physics-simulation-driven and air-filter-focused |
| C18 | Chen/Xiao/Guo — similarity learning, limited labels | 1 | 5 | Few-labelled-case framing; heavier than the budget justifies |

---

### C1. Fault detection and diagnosis of electric bus air conditioning systems incorporating domain knowledge and probabilistic artificial intelligence
- **venue / year**: *Energy and AI* 16:100364, 2024
- **doi_or_url**: https://doi.org/10.1016/j.egyai.2024.100364 — OA PDF mirror: https://ira.lib.polyu.edu.hk/bitstream/10397/108221/1/1-s2.0-S2666546824000302-main.pdf
- **authors**: Fangzhou Guo, Zhijie Chen, Fu Xiao
- **task**: unsupervised fault detection **and localisation to a unit within a peer fleet**
- **code_url**: none found (`unverified` — no code statement located)
- **licence**: gold open access (Energy and AI is fully OA; OpenAlex `oa_status: gold`)
- **reported_metric**: 38 electric-bus AC units (R410A, Shenzhen, 1–24 Aug 2022, 80,860 one-minute
  averaged points ≈ 90 points/system/day); 12 technician-verified faulty systems (5 refrigerant
  undercharge from leaks, 5 outdoor-fan/condenser, 1 indoor-fan/evaporator, 3 high-power compressor);
  **11 of 12 correctly labelled, 1 false negative**, thresholds at the 99.9th percentile with a
  "≥2 of 3 consecutive days" persistence rule; robust with **up to 1/3 of the training peers faulty**
- **dataset**: proprietary bus-fleet telemetry (not released)
- **subsystem_relevance**: **3** — closest analogue found to the PS3 task: a fleet of identical
  vehicle AC units, dynamic operation, no fault-free baseline, one faulty unit to be named
- **implement_hours**: 3–4 (the peer-median-residual + signed fault index core; GPR via
  `sklearn.gaussian_process` or replaced by ridge/quantile regression with no loss of the idea)
- **what_it_preserves_or_assumes**: assumes peers are the *same model under similar conditions* and
  that the **majority are healthy**; assumes a feature's deviation direction under a given fault is
  known a priori (domain knowledge supplies `sign_k`); preserves robustness by (i) median over peer
  predictions rather than mean, (ii) discarding predictions whose GPR variance exceeds Q3, (iii)
  scale-free normalisation by `IQR(R_k)`
- **notes**: their six features (evaporating pressure, superheat, condensing pressure, subcooling,
  discharge temperature, power) are all unavailable in PS3 — **transfer the architecture, not the
  feature list**. Substitute the mode/duty-cycle and pull-down features listed in the practitioner
  paragraph, keeping the signed-index form. Their Table 1 (fault × feature direction matrix) is the
  template for the direction table you must write for your own features.

### C2. A Statistical, Rule-Based Fault Detection and Diagnostic Method for Vapor Compression Air Conditioners
- **venue / year**: *HVAC&R Research* 3(1):19–37, 1997
- **doi_or_url**: https://doi.org/10.1080/10789669.1997.10391359
- **authors**: Todd M. Rossi, James E. Braun
- **task**: detection + diagnosis (fault classification) from residuals
- **code_url**: none
- **licence**: paywalled (Taylor & Francis)
- **reported_metric**: "capable of detecting about a **5 % loss of refrigerant**"; good performance on
  five faults using **only six temperatures** (2 input, 4 output) and linear models, improving ~2× with
  ten measurements and higher-order models. `NUMBERS PARTIALLY UNVERIFIED` — taken from secondary
  sources and the abstract, full text not retrieved this session
- **dataset**: laboratory rooftop unit
- **subsystem_relevance**: **3** — the canonical demonstration that **temperature-only residuals
  against a normal-performance model, classified by the *directional pattern* of residuals**, suffice
  to separate refrigerant leak from fouling and restrictions. Everything [C1] does is a fleet-flavoured
  descendant of this
- **implement_hours**: 2 (the residual + sign-rule idea; no need to reimplement their models)
- **what_it_preserves_or_assumes**: assumes steady-state operation and a *normal-performance model*
  that must come from somewhere — the PS3 twist is that the 7 sibling cars supply it for free, with
  no healthy-baseline period required
- **notes**: read it for the **rule table** (which residual goes up, which goes down, per fault), not
  for the statistics. The 5 % figure is the upper bound of what rich instrumentation buys, and sets
  the contrast with the ~40 % needed by thermostat-only methods [C4][C6].

### C3. Performance benchmarking of residential air conditioning systems using smart thermostat data
- **venue / year**: *Applied Thermal Engineering* 225:120195, 2023
- **doi_or_url**: https://doi.org/10.1016/j.applthermaleng.2023.120195
- **authors**: Fangzhou Guo, Bryan Rasmussen
- **task**: population-level performance benchmarking → anomaly identification and repair verification
- **code_url**: none found
- **licence**: paywalled (Elsevier)
- **reported_metric**: applied across **~9,000 residential units**; the average performance of a large
  population is the benchmark, a per-system metric compares against it, and faults are identified by
  comparing the metric **both between and within systems**; states it detects degradation,
  **especially refrigerant leaks**, and can verify a repair. `NUMBERS UNVERIFIED` — abstract not
  retrievable (Crossref/OpenAlex/S2 all return no abstract); figures above are from secondary
  descriptions, so a verifier should confirm the 9,000 count and the leak claim from the PDF
- **dataset**: proprietary smart-thermostat fleet (indoor temperature, setpoint, runtime, weather)
- **subsystem_relevance**: **3** — same signal poverty as PS3 (temperature + on/off state + ambient)
  and the same "compare a unit to its population" logic, with the added and directly useful idea of
  **within-system** comparison over time alongside between-system comparison
- **implement_hours**: 3
- **what_it_preserves_or_assumes**: assumes a large enough population for the mean to be a clean
  benchmark — **PS3 has 7 peers, not 9,000**, so use median/MAD, not mean/σ, and expect the peer
  estimate to be noisy; the *within-system over time* half of the method partly compensates
- **notes**: pair with [C16] (same authors, trend testing) and [C4] (same data modality, different
  modelling stance). Together with [C1] these three are the "temperature + runtime is enough" case.

### C4. Automated fault detection of residential air-conditioning systems using thermostat drive cycles
- **venue / year**: *Energy and Buildings* 236:110691, 2021
- **doi_or_url**: https://doi.org/10.1016/j.enbuild.2020.110691 — accepted manuscript (free):
  https://www.sciencedirect.com/science/article/am/pii/S0378778820334770
- **authors**: Rohit Chintala, Jon Winkler, Xin Jin
- **task**: automated fault detection using **only the thermostat and outdoor air temperature**
- **code_url**: none found (`unverified`)
- **licence**: paywalled; accepted manuscript publicly posted under the DOE Public Access Plan
- **reported_metric**: EnergyPlus model of a typical Orlando home; 3R2C grey-box thermal model
  identified by **extended Kalman filter**, used to predict cooling times over a series of thermostat
  drive-cycle experiments. Accuracy: duct-leak **70 %**, 40 % airflow fault **77 %**, **40 %
  refrigerant undercharge 82 %**, no-fault **87 %**
- **dataset**: EnergyPlus simulation (a lab-home follow-up is [C5])
- **subsystem_relevance**: **3** — establishes that **cooling-cycle duration is the workhorse
  feature** when you have nothing but indoor temperature, setpoint and ambient. PS3's running-mode
  string is exactly a drive-cycle record
- **implement_hours**: 4–6 for the full 3R2C+EKF; **1–2 h** for the cheap version that actually fits
  the budget — measured pull-down slope and time-to-setpoint per cooling episode, peer-normalised,
  with no state-space identification at all
- **what_it_preserves_or_assumes**: assumes you can observe complete on-cycles and that the thermal
  envelope parameters are identifiable; the RC identification is explicitly **non-convex with many
  local optima** (the whole point of [C5])
- **notes**: the honest read is that the grey-box arm is over-budget and fragile; take the *feature
  definition* (cooling-cycle duration / pull-down rate) and let the 7 peer cars do the job the RC
  model does in their setting. Note the accuracy ordering: even at a **40 %** undercharge, 82 % —
  early leaks are simply not visible in this modality.

### C5. Sensitivity analysis of an automated fault detection algorithm for residential air-conditioning systems
- **venue / year**: *Applied Thermal Engineering* 238:121895, 2024 (preprint dated 2020)
- **doi_or_url**: https://doi.org/10.1016/j.applthermaleng.2023.121895 — free accepted manuscript:
  https://www.osti.gov/pages/servlets/purl/2274818
- **authors**: Rohit Chintala, Jon Winkler, Sugirdhalakshmi Ramaraj, Xin Jin (NREL)
- **task**: robustness study of [C4] plus first application to **real** faulted-equipment data
- **code_url**: none found
- **licence**: paywalled; DOE-public-access accepted manuscript free at OSTI
- **reported_metric**: on Florida Solar Energy Center lab-home data with faults intentionally imposed
  over **seven months**: **undercharge 70.6 %**, concurrent duct-leak + undercharge **85.2 %**,
  duct leak **69.1 %**. Sensitivity study over EnergyPlus models of **nine house constructions**:
  average **71 %** no-fault, **77 %** at 40 % undercharge, **76 %** duct leak
- **dataset**: FSEC lab home (real) + 9 EnergyPlus models
- **subsystem_relevance**: **2** — not a peer-fleet method, but it is the best available *calibration
  of expectations*: the same algorithm drops from 82 % (simulation) to 70.6 % (real) on undercharge
- **implement_hours**: 2 (read-only; the model-selection trick is not needed if you skip the RC arm)
- **what_it_preserves_or_assumes**: makes explicit that the grey-box parameter identification is
  **highly non-convex with several local optima** and needs a model-selection step to be usable
  across different buildings — a direct argument for *not* putting an identified physical model on
  the critical path in a 12 h budget
- **notes**: cite this in the addendum as the "why we did not build the RC/physics arm" evidence.

### C6. Residential HVAC Fault Detection: Field Data Analysis and Interviews with Smart Thermostat Manufacturers
- **venue / year**: Oak Ridge National Laboratory technical report **ORNL/TM-2024/3661**, October 2024
- **doi_or_url**: https://info.ornl.gov/sites/publications/Files/Pub224924.pdf
- **authors**: Yeobeom Yoon, Young Jae Choi, Sungkyun Jung, Piljae Im, Junjie Luo, Vishaldeep Sharma,
  Brian Kolar, Islam Safir
- **task**: characterisation — which measurable quantities respond to refrigerant charge faults
- **code_url**: n/a
- **licence**: US DOE contractor report, publicly available (not a formal open licence — treat as
  public-domain-ish government work; `licence UNVERIFIED`)
- **reported_metric**: 12 cooling-season field tests: **five undercharge levels (−10 %, −20 %, −30 %,
  −40 %, −50 %)**, five overcharge (+5 % … +25 %), two baselines. Verbatim findings:
  "**Low-intensity refrigerant undercharge faults are difficult to detect using only indoor air
  temperature data**"; "**Unmet hours, during which the indoor air temperature exceeds the cooling set
  point, were observed during 40 % and 50 % refrigerant undercharge faults**"; "**Supply-air
  temperature increased as the refrigerant undercharge increased**"; "**System runtime fraction
  increased as the refrigerant undercharge increased**"; "Compressor energy consumption decreased as
  the refrigerant undercharge increased"; overcharge is *not* visible in supply-air temperature or
  runtime fraction. Also quotes "even a slight undercharge can decrease cooling capacity by nearly
  **13 %** and reduce energy efficiency by **7.6 %**"
- **dataset**: ORNL field test house, cooling season 2024; also interviews with 6 representatives from
  4 smart-thermostat manufacturers
- **subsystem_relevance**: **3** — this is the *evidence table* for the PS3 feature-direction matrix,
  and the single strongest justification for ranking duty-cycle features above setpoint-shortfall
  features. Verified directly from the report PDF this session
- **implement_hours**: 0 (evidence, not a method)
- **what_it_preserves_or_assumes**: residential split system, single unit, thermostat control — the
  physics (capacity loss → longer runtime → eventual setpoint miss) carries to a train ACV unit, the
  absolute magnitudes do not
- **notes**: PS3 has no supply-air temperature and no power, so of ORNL's three "key variables" only
  **runtime fraction** survives — which is precisely the running-mode string. The `load-halved` flag
  is a confound that mimics the same direction and must be conditioned on.

### C7. Performance evaluation of a virtual refrigerant charge sensor / Extension of a virtual refrigerant charge sensor
- **venue / year**: *International Journal of Refrigeration* 36(3), 2013 and 51, 2015
- **doi_or_url**: https://doi.org/10.1016/j.ijrefrig.2012.11.004 ; https://doi.org/10.1016/j.ijrefrig.2014.09.015
- **authors**: Woohyun Kim, James E. Braun
- **task**: regression of refrigerant charge level from low-cost measurements (virtual sensor)
- **code_url**: none
- **licence**: paywalled
- **reported_metric**: estimates of refrigerant charge **within 10 % of actual** over a wide range of
  operating conditions for several systems, using **surface-mounted temperature measurements only**;
  three ways to set the empirical parameters (defaults, simulation, regression on measurements); a
  piecewise-linear structure segmented by subcooling improves accuracy at low subcooling.
  `NUMBERS PARTIALLY UNVERIFIED` — abstracts not retrievable from Crossref/S2/OpenAlex; figures from
  the Purdue/secondary summaries
- **dataset**: laboratory + field packaged AC / split systems
- **subsystem_relevance**: **2** — "temperature-only" here still means *refrigerant-side* surface
  temperatures at the liquid line and suction line, which PS3 does not have. Value is conceptual:
  it names **subcooling as the dominant charge indicator** and shows charge is a smooth, monotone
  function of a couple of temperatures — i.e. a leak is a *slow drift*, not a step
- **implement_hours**: 6+ if attempted (it cannot be, for lack of inputs) — recommend **skip with
  reason**
- **what_it_preserves_or_assumes**: assumes liquid-line/suction-line temperature access and a TXV or
  fixed-orifice expansion device of known type; assumes near-steady operation
- **notes**: the clean "skipped-with-reason" entry for the ladder: *the entire virtual-charge-sensor
  family is inapplicable because PS3 exposes no refrigerant-side temperatures.* Say so explicitly
  rather than silently omitting it — it is the first thing an HVAC reviewer will look for.

### C8. Decoupling features and virtual sensors for diagnosis of faults in vapor compression air conditioners
- **venue / year**: *International Journal of Refrigeration* 30(3):546–564, 2007 (companion:
  *HVAC&R Research* 13(3):369–395, 2007, "A Methodology for Diagnosing Multiple Simultaneous Faults")
- **doi_or_url**: https://doi.org/10.1016/j.ijrefrig.2006.07.024 ; https://doi.org/10.1080/10789669.2007.10390959 —
  free PDF mirror: https://ncesr.unl.edu/wordpress/wp-content/uploads/2013/08/li-decoupling-features-and-virtual-sensors.pdf
- **authors**: Haorong Li, James E. Braun
- **task**: multiple-simultaneous-fault diagnosis via features uniquely tied to one fault
- **code_url**: none
- **licence**: paywalled (free author/host mirror above, terms `unverified`)
- **reported_metric**: `NUMBERS UNVERIFIED` — the decoupling framework is qualitative in the abstract;
  effectiveness on refrigerant under- and overcharge is asserted in secondary sources
- **dataset**: laboratory rooftop units
- **subsystem_relevance**: **2** — the methodological warning that matters here: a feature is only
  diagnostic if it is **uniquely dependent on one fault and independent of driving conditions and
  other faults**. PS3's candidate features fail this badly — `load-halved`, a stuck door, a blocked
  filter and a solar-exposed end car all push runtime fraction the same way as a leak
- **implement_hours**: 6+ for the real method; **0 h** to apply the principle, which is to build the
  fault × feature direction matrix and check that no *other* plausible cause shares the signature
- **what_it_preserves_or_assumes**: assumes enough sensors to construct decoupled features; the whole
  point is that coupled features give confounded diagnoses
- **notes**: use this to justify reporting a **ranking with a confounder caveat** rather than a
  confident diagnosis — the PS3 metric rewards exactly that hedge.

### C9. Refrigerant leakage detection in an EEV installed residential air conditioner with limited sensor installations
- **venue / year**: *International Journal of Refrigeration* 78:157–165, 2017
- **doi_or_url**: https://doi.org/10.1016/j.ijrefrig.2017.03.001
- **authors**: J. Yoo, Sung-Bin Hong, Min Soo Kim
- **task**: refrigerant leak detection under a deliberately impoverished sensor set
- **code_url**: none
- **licence**: paywalled
- **reported_metric**: `NUMBERS UNVERIFIED` — abstract not retrievable from Crossref/S2/OpenAlex this
  session; secondary summary states that pressure sensors and mass-flow meters cost several times more
  than temperature sensors so most residential units carry **temperature sensors only**, that
  differential subcooling (DSC) is the usual leak indicator but **cannot be obtained** from such units,
  and that a method for limited sensor information was proposed and **experimentally verified**
- **dataset**: laboratory EEV-equipped residential AC
- **subsystem_relevance**: **3** on framing — it is the paper that asks precisely the PS3 question
  ("what can you still detect when the informative sensor is missing?"), though its "limited" set is
  still richer than PS3's
- **implement_hours**: 2 (read + lift the indicator construction)
- **what_it_preserves_or_assumes**: EEV (electronic expansion valve) control, which changes how a leak
  manifests — an EEV compensates and hides superheat drift, pushing the signature into capacity and
  runtime instead. Train ACV units are likely TXV or EEV; **`unverified` for this fleet**
- **notes**: worth a verifier pass to pull the actual indicator and numbers out of the PDF; flagged
  as the highest-value unverified entry in this list.

### C10. A metro train air conditioning system fault diagnosis method based on explainable artificial intelligence: considering interpretability and generalization
- **venue / year**: *International Journal of Refrigeration* 174:47–59, June 2025
- **doi_or_url**: https://doi.org/10.1016/j.ijrefrig.2025.03.001
- **authors**: Minhui Jiang, Huanxin Chen, Chuang Yang
- **task**: fault diagnosis (multi-fault) for **metro train** AC, with interpretability
- **code_url**: none found
- **licence**: paywalled
- **reported_metric**: `NUMBERS UNVERIFIED` — abstract not retrievable; secondary summary says most
  HVAC fault-diagnosis models are poorly interpretable and rarely applied to metro-train AC, that
  metro-train AC takes ~**70 %** of carriage energy consumption, and that the method is verified on
  **simulation** data for single and simultaneous faults under different operating conditions
- **dataset**: simulation of a metro-train AC system (not released)
- **subsystem_relevance**: **3** by domain — the only metro/train-AC-specific FDD paper this sweep
  found, so it is the citation that makes the addendum's domain framing credible even if the method
  is not adopted
- **implement_hours**: n/a (likely not reimplementable in budget; simulation-trained)
- **what_it_preserves_or_assumes**: assumes a simulation model of the specific AC unit and a richer
  sensor set than PS3 provides (`unverified`)
- **notes**: **verifier priority** — confirm the sensor list and whether any real telemetry is used.
  If it is simulation-only with pressure sensors, it belongs in "cited for context, not adopted".

### C11. A Physics-Informed Explainable AI Framework for HVAC Anomaly Detection and Maintenance-Oriented Analysis in Urban Bus Fleets
- **venue / year**: *Algorithms* 19(7):586, 2026
- **doi_or_url**: https://doi.org/10.3390/a19070586
- **authors**: Bernardo Tormos, Ramón Sánchez-Márquez, Jorge Alvis-Sánchez, Vicente Bermudez
- **task**: unsupervised anomaly detection + explanation on fleet HVAC telemetry
- **code_url**: none found (`unverified`)
- **licence**: **CC BY 4.0**
- **reported_metric**: `NUMBERS UNVERIFIED` — MDPI blocks automated fetching; abstract obtained via
  Crossref states the framework begins with **sensor contextualisation and physical mapping of the
  available measurements onto the vapour-compression cycle**, then unsupervised anomaly detection,
  then XAI, for data that lack "reliable fault labels, complete contextual information, or a clear
  physical interpretation of the monitored variables"
- **dataset**: urban bus fleet telematics (proprietary)
- **subsystem_relevance**: **2** — the *process* it prescribes (map each available channel onto the
  refrigeration cycle before modelling anything) is exactly the first hour of PS3 work, and the
  paper's premise — imperfect, unlabelled fleet HVAC data — matches
- **implement_hours**: 2
- **what_it_preserves_or_assumes**: assumes a fleet with heterogeneous routes/duty (PS3's 8 cars are
  far more homogeneous, which is an advantage); assumes no reliable labels (PS3 has 6)
- **notes**: CC BY and recent — cheap to cite, and the "physical mapping" table is a good model for
  documenting *why* each PS3 channel was or was not used.

### C12. Unsupervised Fault Detection in Varying Operating Conditions
- **venue / year**: 2019 IEEE International Conference on Prognostics and Health Management (ICPHM)
  (preprint arXiv:1907.06481); related: *Knowledge-Based Systems* 216:106816, 2021
- **doi_or_url**: https://doi.org/10.1109/ICPHM.2019.8819383 — preprint https://arxiv.org/abs/1907.06481
  ; https://doi.org/10.1016/j.knosys.2021.106816
- **authors**: Gabriel Michau, Olga Fink
- **task**: unsupervised fault detection when the training data do not cover all normal conditions,
  with and without a fleet
- **code_url**: `unverified` (the authors' group publishes code irregularly; nothing located)
- **licence**: IEEE paywall; arXiv preprint free
- **reported_metric**: **five approaches** compared — two using only the target unit (early-life
  baseline; incremental learning) and **three exploiting a fleet of similar units**, including an
  Unsupervised Feature Alignment Network; tested on a fleet of **112 units over one year**; all
  proposed approaches improve on a baseline trained with only two months of data
- **dataset**: proprietary industrial fleet
- **subsystem_relevance**: **2** — the general PHM statement of the PS3 premise. Its taxonomy
  (own-unit history vs fleet transfer) is a useful way to frame the addendum's ladder, and its
  finding that fleet information beats a short own-history baseline directly supports spending the
  budget on peer normalisation rather than on per-car change-point detection over 3–4 days
- **implement_hours**: 4 for the ELM/alignment machinery — **not recommended**; 0 to use the framing
- **what_it_preserves_or_assumes**: assumes the fleet units are comparable after feature alignment;
  assumes enough data per unit to learn a representation (PS3 has ~8,600–11,500 rows per car at 30 s
  over 3–4 days, which is enough for simple features but not for representation learning on 8 units)
- **notes**: complements the door/bearing peer-normalisation citation [R78] already in the corpus
  with a *fleet-transfer* citation, at the cost of being from a different industry.

### C13. Design of a steady-state detector for fault detection and diagnosis of a residential air conditioner
- **venue / year**: *International Journal of Refrigeration* 31(5):790–799, 2008
- **doi_or_url**: https://doi.org/10.1016/j.ijrefrig.2007.11.008 — also
  https://www.nist.gov/publications/design-steady-state-detector-fault-detection-and-diagnosis-residential-air-conditioner
- **authors**: Minsung Kim, Seok Ho Yoon, Piotr A. Domanski, W. Vance Payne (NIST)
- **task**: transient/steady-state gating as a preprocessing stage for FDD
- **code_url**: none
- **licence**: paywalled (NIST author copy likely available)
- **reported_metric**: moving window over **standard deviations of seven selected measurements**;
  window size and thresholds tuned on steady no-fault and start-up transient tests; **evaporator
  superheat and condenser subcooling alone sufficed for start-up transients but misidentified
  steady-state during indoor-temperature-change tests**, where evaporator saturation temperature and
  air-temperature change across the evaporator were needed — hence the recommendation to **include all
  FDD features in the steady-state detector**. A worked instance uses the mean and standard deviation
  over the **previous 5 minutes**. `NUMBERS PARTIALLY UNVERIFIED` (from abstract/secondary text)
- **dataset**: NIST laboratory residential AC
- **subsystem_relevance**: **2** — mundane but load-bearing: without a transient gate, a train that
  starts cooling at different times per car will generate residuals that have nothing to do with
  refrigerant. Its specific lesson (gate on *all* features, not a favourite pair) is worth 10 lines
- **implement_hours**: 1 (rolling σ over a 5–10 min window on the indoor temperature and mode, plus a
  "same mode for the whole window" condition, is ~15 lines on top of pandas `rolling`)
- **what_it_preserves_or_assumes**: assumes a sampling rate fast enough to resolve transients — PS3's
  30 s is comfortable; assumes transients are a nuisance, whereas here the **pull-down transient is
  itself a capacity feature**, so gate to *separate* the two regimes rather than to discard one
- **notes**: recommend a two-regime split (steady / pull-down) rather than the paper's discard-the-
  transient default. State the deviation explicitly.

### C14. LBNL Fault Detection and Diagnostics Datasets (+ Scientific Data papers)
- **venue / year**: Granderson, Lin, Harding, Im, Chen, *Scientific Data* 7, 2020; Granderson, Lin,
  Chen, Casillas, Wen, Chen, Im, Huang, Ling, *Scientific Data* 10, 2023; dataset release 2022
- **doi_or_url**: https://doi.org/10.1038/s41597-020-0398-6 ; https://doi.org/10.1038/s41597-023-02197-w ;
  data https://faultdetection.lbl.gov/data/ , https://data.openei.org/submissions/5763 ,
  https://www.osti.gov/biblio/1881324
- **task**: benchmark data — labelled faulted / fault-free HVAC operation with ground truth
- **code_url**: n/a (data)
- **licence**: US DOE open data (OEDI / data.gov); **exact terms `unverified`** — the LBNL landing page
  states no licence
- **reported_metric**: seven system types (rooftop units, single- and dual-duct AHUs, VAV boxes, fan
  coil units, chiller plants, boiler plants); **20 to 100+ points per dataset**; simulation, laboratory
  and field sources; each period labelled with which fault is present at which severity. **Whether the
  RTU subset includes refrigerant undercharge specifically is `unverified`** — the data landing page
  does not enumerate faults and could not be fetched further this session
- **dataset**: itself
- **subsystem_relevance**: **2** — no train data and no 8-identical-peers structure, but it is the only
  open, labelled, ground-truthed HVAC fault data found, so it is the honest way to dry-run a pipeline
  and to sanity-check that the feature-direction matrix is right before touching 6 precious cases
- **implement_hours**: 2 to download and run one RTU subset through the feature extractor
- **what_it_preserves_or_assumes**: building HVAC, single units, far richer instrumentation
- **notes**: also relevant is the 2026 follow-up, "Labeled Datasets for Air Handling Units Operating in
  Faulted and Fault-free States", *Scientific Data* 13, https://doi.org/10.1038/s41597-025-06179-y —
  AHU-only, so lower priority. Cross-reference `docs/research/datasets.md`, which does not list any of
  these.

### C15. Time series analysis of field data for soft faults detection and degradation assessment in residential air conditioning systems
- **venue / year**: *Applied Thermal Engineering* 269:126104, 2025
- **doi_or_url**: https://doi.org/10.1016/j.applthermaleng.2025.126104
- **authors**: Belén Llopis-Mengual, David P. Yuill, Emilio Navarro-Peris
- **task**: detection and degradation assessment of **soft faults** (gradual, e.g. charge loss and
  fouling) from field time series
- **code_url**: none found
- **licence**: Crossref lists a **CC BY-NC-ND 4.0** entry alongside the Elsevier TDM licence — likely
  open access, `licence PARTIALLY UNVERIFIED`
- **reported_metric**: `NUMBERS UNVERIFIED` — abstract not retrieved; secondary description points to
  virtual-refrigerant-charge-style indicators correlated with the charge the system *should* have
- **dataset**: field data from split-system air conditioners (normal operation plus condenser/evaporator
  inlet fouling and compressor-capacitor degradation)
- **subsystem_relevance**: **2** — a refrigerant leak over 3–4 days is a *soft* fault by definition and
  will not show a step; this is the framing for "look at level relative to peers, and slope within the
  window", not "look for a change point"
- **implement_hours**: 3
- **notes**: Yuill's group is also the source of the standard FDD-evaluation critique literature; worth
  a verifier pass for an evaluation-protocol citation that would strengthen `model_ladder.md` §6.

### C16. Predictive maintenance for residential air conditioning systems with smart thermostat data using modified Mann-Kendall tests
- **venue / year**: *Applied Thermal Engineering* 222:119955, 2023
- **doi_or_url**: https://doi.org/10.1016/j.applthermaleng.2022.119955
- **authors**: Fangzhou Guo, Bryan Rasmussen
- **task**: trend detection on a thermostat-derived health index for predictive maintenance
- **code_url**: none found
- **licence**: paywalled
- **reported_metric**: `NUMBERS UNVERIFIED` — no abstract retrievable
- **dataset**: smart-thermostat fleet
- **subsystem_relevance**: **1** — 3–4 days is short for a trend test, and a leak that developed before
  the recording window shows as a *level* difference, not a trend. But a **modified Mann-Kendall test
  (autocorrelation-corrected) on each car's peer-normalised index** is ~5 lines with `pymannkendall`
  or ~30 hand-written, and it is a cheap tie-breaker between two cars with similar levels
- **implement_hours**: 1
- **what_it_preserves_or_assumes**: assumes 30 s telemetry is heavily autocorrelated — which it is,
  hence the *modified* (variance-corrected) form rather than plain Mann-Kendall
- **notes**: propose as a **NICE**, explicitly as a tie-breaker, not as a primary ranker.

### C17. Fault Detection and RUL Estimation for Railway HVAC Systems Using a Hybrid Model-Based Approach
- **venue / year**: *Sustainability* 13(12):6828, 2021
- **doi_or_url**: https://doi.org/10.3390/su13126828 (verified via Crossref + Semantic Scholar;
  MDPI blocks automated fetching) — OA PDF https://www.mdpi.com/2071-1050/13/12/6828/pdf
- **authors**: Antonio Gálvez, Alberto Diez-Olivan, Dammika Seneviratne, Diego Galar
- **task**: FDD + RUL for HVAC installed in a **passenger train carriage**
- **code_url**: none
- **licence**: **CC BY 4.0** (confirmed via Semantic Scholar openAccessPdf: GOLD / CCBY)
- **reported_metric**: hybrid model-based approach (HyMA) where a **physics-based model generates
  healthy and faulty data at several degradation levels**, individually and in combination, then
  synthetic data are **fused with measured data** to train/validate/test. FDD accuracy **92.60 %**;
  air-filter RUL accuracy **95.21–97.80 %**. Abstract verified verbatim via Semantic Scholar
- **dataset**: physics simulation + measured train-carriage HVAC data (neither released)
- **subsystem_relevance**: **1–2** — right vehicle, right subsystem, wrong fault (air filter, not
  refrigerant) and wrong method for the budget (building a physics model of the ACV cycle is well
  beyond 12 h). Its genuinely transferable point is the **framing of the data problem**: railway HVAC
  components are replaced early, so advanced-degradation data essentially do not exist — which is
  exactly why PS3 gives you 6 cases and why a supervised approach is the wrong instinct
- **implement_hours**: 6+ (skip)
- **notes**: strong candidate for the **skipped-with-reason** ladder, and the best single citation for
  "why the ACV training set is tiny and will stay tiny". CC BY, so quotable.

### C18. Similarity learning-based fault detection and diagnosis in building HVAC systems with limited labeled data
- **venue / year**: *Renewable and Sustainable Energy Reviews* 185:113612, 2023
- **doi_or_url**: https://doi.org/10.1016/j.rser.2023.113612 — OA copy
  http://hdl.handle.net/10397/108217 (**CC BY-NC-ND**)
- **authors**: Zhe Chen, Fu Xiao, Fangzhou Guo
- **task**: FDD when very few labelled fault examples exist (similarity/metric learning)
- **code_url**: none found
- **licence**: paywalled at Elsevier; CC BY-NC-ND institutional copy at PolyU
- **reported_metric**: `NUMBERS UNVERIFIED` — no abstract retrievable via Crossref/S2/OpenAlex
- **dataset**: building HVAC benchmarks (likely ASHRAE RP-1043 / RP-1312; `unverified`)
- **subsystem_relevance**: **1** — the "limited labelled data" framing matches (6 cases), but metric
  learning across 48 rows with 6 positives is not a defensible use of the budget, and the method
  targets *classification into fault types*, whereas PS3 needs *localisation of one known fault type*
- **implement_hours**: 5 (not recommended)
- **notes**: include in the ladder as **skipped-with-reason**: few-shot/metric-learning approaches
  presuppose more labelled episodes than PS3 provides, and the peer structure gives a stronger,
  cheaper inductive bias. Adjacent same-framing work if a stronger citation is wanted: a few-shot
  (semi-supervised adaptive weighted prototype network) framework reporting **mean F1 73.77 % on
  ASHRAE RP-1043 severity level 1** and **67.22 % on RP-1312 AHU summer**, *Applied Energy* 402,
  2026, https://doi.org/10.1016/j.apenergy.2025.126786 — **DOI `unverified`**, located only via
  IDEAS/RePEc listing `v402y2026ipcs0306261925017866`; those F1 numbers are also the fairest
  statement of how hard few-shot HVAC FDD is even with rich chiller instrumentation.

---

## Background anchors (not ranked candidates, cite only if the synthesis needs them)

- **ASHRAE RP-1043** — the centrifugal-chiller fault dataset (1999) behind most chiller FDD papers:
  64 parameters of which 48 directly measured; faults include **refrigerant leakage**, excess oil,
  condenser fouling and non-condensable gas at several severity levels. Its sibling projects are
  **RP-1139, RP-1275 and RP-1486**. It is a *chiller* dataset with full refrigerant-side
  instrumentation, so it is background, not a resource for PS3. `Access terms UNVERIFIED` (ASHRAE
  sells the research report).
- **AI in HVAC fault detection and diagnosis: a systematic review**, *Energy and AI* / Elsevier 2024,
  https://www.sciencedirect.com/science/article/pii/S277297022400004X — a recent survey to cite in
  one line if the addendum needs a "the field at large" sentence. `Metadata UNVERIFIED` (not resolved
  to a DOI this session).
- **PS3 metric arithmetic** (not a citation, but the number every reviewer will want): with n = 8 and
  the true faulty car uniform over cars, a random ranking scores **0.5625**; correct-first is 1.000,
  second 0.875, third 0.750. The realistic target band for a peer-residual ranker on 6 LOO-CV cases is
  0.75–0.95; anything at or below 0.5625 is a bug, not a result.

---

## Verifier notes — what to check first

1. **[C9] Yoo et al. 2017** — highest-value unverified entry. Get the actual limited-sensor indicator
   and its detection threshold from the PDF; if it needs only suction-line and air temperatures it may
   partially transfer.
2. **[C10] Jiang et al. 2025** — confirm sensor list and whether any real metro telemetry is used. This
   determines whether it is "the domain citation" or "adopted method".
3. **[C3] Guo & Rasmussen 2023** — confirm the ~9,000-unit figure and the explicit refrigerant-leak
   claim; both came from a secondary summary, not the abstract.
4. **[C14] LBNL datasets** — confirm whether the **RTU** subset contains a refrigerant-undercharge
   fault, and pin down the licence. If yes, this becomes a MUST-tier dry-run resource.
5. **[C2] Rossi & Braun 1997** — confirm the "≈5 % refrigerant loss" figure from the paper itself; it
   is quoted widely but was not read first-hand here.
6. **[C18] appendix** — the *Applied Energy* 402 (2026) few-shot paper DOI is a guess derived from the
   RePEc identifier; either verify or drop the DOI and cite by title.
7. **[C6] ORNL** — verified first-hand from the report PDF (quotes are verbatim); the only entry in
   this list whose headline findings were read in full.
8. **[C1] Guo et al. 2024** — verified first-hand from the PolyU open-access PDF (fleet size, fault
   counts, 11/12 detection, median-of-peers aggregation, fault-index formula, 1/3-faulty robustness).

Everything else in this file is Crossref/OpenAlex/Semantic Scholar metadata (title, venue, volume,
pages, year, DOI, authors, licence) that **was** resolved programmatically and can be treated as
verified, plus abstracts or secondary summaries as marked.
