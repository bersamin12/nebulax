# Model ladder — NEBULA X Track 3

Companion to `configs/model_ladder.yaml` (machine-readable), `rail_phm.md` (domain evidence) and
`references.md` (every `[Rnn]` resolves there). This file **confirms, corrects and extends** the
"Model benchmark and ablations (P2)" section of the plan.

**How to read a row.** `MUST` = in the minimum leaderboard that must exist by Sat 13:00 and must be
implementable by one person in the stated hours. `NICE` = extension slot, only started when the core
matrix is green. `SKIP` = deliberately excluded, with the reason stated in the row; every SKIP is a
sentence we can say to a judge.

`implement_hours` is **per model, written once**, and a model written once is reused across all three
subsystems. The per-subsystem tables repeat a model only where the input, features or hyperparameters
genuinely differ; the shared effort budget is totalled at the end.

---

**W1 update (14 Sep).** This pass added two whole families the W0 ladder was missing and made a
third symmetric. (a) **Change-point detection and concept drift** — verdict V4 made change-point
scoring the central design decision, yet the only change-point machinery in the ladder was a
hand-written CUSUM on a scalar. There are now `§1d / §2c / §3c` change-point sections with MUST, NICE
and reasoned SKIP rows, plus a segmentation-specific evaluation subsection (§6.6). (b) **Prognostics
and RUL** — the W0 yaml carried `task: rul` rows for the **door only**, while `datasets.md` was
ingesting three bearing run-to-failure sets for a capability nothing delivered. `§2d / §3d` now fill
the pneumatic and bearing RUL arms, and §6.7 adds prognostics metrics and the leave-one-unit-out
rule. (c) Verdicts **V6** and **V7** below state the two new cross-cutting positions.

---

## 0. Seven cross-cutting verdicts that reshape the ladder

**V1 — Simple beats deep, and this is now the benchmark consensus, not a hunch.** On TSB-AD (1,070
curated series, 40 algorithms, point-adjustment *removed*, ranked by VUS-PR), the TSB-AD-M top-10 is
CNN 0.3130, OmniAnomaly 0.3122, PCA 0.3096, LSTMAD 0.3066, USAD 0.3041, AutoEncoder 0.2950, KMeansAD
0.2949, CBLOF 0.2731, MCD 0.2711; TSB-AD-U is led by Sub-PCA 0.4234, KShapeAD 0.4008, POLY 0.3898,
Series2Graph 0.3881 [R1]. **No transformer detector reaches the multivariate top-10.** The current #1
on both tracks is a matrix-profile method (MMPAD: 0.4399 U, 0.3539 M) with no neural network at all
[R11]. So: statistical and classical tiers are MUST everywhere, deep is a small controlled set, and
transformers are ablation rows rather than headline models.

**V2 — Matrix profile is the single highest-value addition, and it is best exactly where we live.**
MMPAD's multivariate advantage over the best neural baseline is **+0.2517 VUS-PR at 2-3 channels,
+0.0798 at 4-19 channels, but −0.0801 at 20-31 and −0.1724 at 32-248** [R11]. Our bearing subsystem
has 2-5 channels, our door ~6-10, MetroPT-3 15. All three sit in the regime where matrix profile
dominates. It was **absent from the pre-research plan**; it is now MUST for all three subsystems.
One hyperparameter (window length `m`) plus `k`; `stumpy.gpu_stump` on the A4500.

**V3 — Foundation models earn their slot only if purpose-built for AD.** Three independent 2026
results say naive residual/reconstruction thresholding on a TSFM is not a win: five TSFM families
(MOMENT, Chronos, TimesFM, Time-MoE, TSPulse) are statistically indistinguishable from **moving-window
variance and squared difference** [R35]; TimesFM on SWaT beat neither baseline family and the authors
conclude "naive zero-shot FMs are unsuitable for MTSAD but promising for change-point detection"
[R36]; a cost-aware study finds a small fitted TCN-AE and an OCSVM **outperform** MOMENT's
reconstructions while MOMENT costs more latency, VRAM and disk [R37]. The exception is **TSPulse**:
1M parameters, Apache-2.0, CPU-capable, dual time+frequency masked reconstruction, and IBM reports
it #1 on **TSB-AD** — the benchmark that removed point adjustment [R30]. Correction to the plan:
TSPulse becomes the foundation MUST; Chronos-2 drops to NICE (forecasting evidence, not AD [R24]);
MOMENT fine-tuning is dropped entirely (FT adds only +0.007 VUS-PR over zero-shot [R1]).

**V4 — Residual magnitude is the wrong score for persistent faults.** A TSFM tracks anomalous
dynamics *too well*, so forecast error stays **low during the interior of a long anomaly and peaks
only at its boundaries** [R36]. MetroPT air-leak and oil-leak events are multi-hour to multi-day
persistent regimes [R87][R88][R89]. Every forecast-residual detector we build therefore scores
**change points** (CUSUM on the residual, or the residual's derivative) and fills the episode with a
hysteresis state machine, rather than thresholding residual magnitude per sample. This is a design
change, not a new model, and it costs ~1 h.

**V5 — Multivariate machinery must be justified against a univariate control.** Across eight public
multivariate benchmarks, no cross-channel rupture occurs without an accompanying univariate
deviation, and most labelled anomalies deviate univariately on 89-100 % of their timesteps [R21].
So every multivariate model in our table is compared against **N independent univariate detectors
aggregated by max/OR**. If the multivariate model does not beat it, we say so. Running this
diagnostic on our own simulator output is also a genuinely novel ablation: it lets us claim our
injected faults *are* multivariate, if they are.

**V6 — Change-point detection is a first-class family, and the cheap offline method wins under our
own rules.** V4 says every forecast-residual detector must score change points rather than residual
magnitude [R36]. That was a design change with no models attached. Three findings settle the models.
(i) On the only change-point benchmark we verified — 37 annotated real series, 5 human annotators,
14 algorithms — **with default hyperparameters binary segmentation had the highest average
performance**, and only with *oracle* hyperparameter tuning did Bayesian online change-point
detection beat everything [R175]. Under our validation-only-threshold rule we are permanently in the
**default** column, so the cheap method is the right MUST and BOCPD is NICE. (ii) The libraries are
one-line calls under permissive licences: `ruptures` BSD-2 [R173], `changeforest` BSD with a JMLR
2023 paper and explicit multivariate design [R187], `claspy` BSD-3 and **hyper-parameter-free** so
there is nothing to calibrate [R176][R177], and `river` — **already a dependency**, because the
pneumatic ladder imports it for Half-Space Trees — ships ADWIN, Page-Hinkley and KSWIN [R178].
(iii) TSB-AD's own verdict that simple statistical methods beat advanced neural architectures [R1]
points the same way. **The one real cost is an adapter**: Pelt/BinSeg/ClaSP/changeforest emit
**discrete breakpoints** and river drift detectors emit **booleans**, whereas VUS-PR requires a
**continuous per-timestamp score** [R7]. Budget that adapter once (§6.6) and reuse it on every row.

**V7 — The RUL ladder must be symmetric, and at our sample sizes it is parametric, not deep.**
Generalization bounds for RUL give learning rates in `n` = the number of **complete run-to-failure
trajectories**, and show that embedding degradation physics can cut data requirements by up to **two
orders of magnitude** for deep networks [R221]. XJTU-SY has 15 bearings, FEMTO ~17. That is the
small-`n` regime where stochastic degradation models (Wiener [R196], gamma [R197], inverse-Gaussian
[R198]) and the General Path Model [R199] are the sample-efficient choice and a deep RUL net is not.
Two further results agree: on C-MAPSS FD003 **XGBoost (RMSE 13.36) beats a 1D CNN (15.68) and an LSTM
(14.20)** [R222], and tabular foundation models with in-context learning rank best specifically in
low-data PHM regimes [R223]. Separately, where we have **no** run-to-failure data but do have a fleet
with maintenance records — the pneumatic case — the correct family is **survival analysis on censored
data** (Cox [R205], random survival forests [R206]), for which there is a published precedent putting
RUL regression and survival models on the same ladder [R207] and a ~30-line loss change that lets a
plain regressor consume censored samples [R208]. Finally, the prognostics analogue of our
no-point-adjustment rule is **leave-one-unit-out**: naive splitting inflates RUL-adjacent accuracy
from a genuine 20-60 % to 99.9 % on the same data [R220], and device-level leakage optimistically
biases every published RUL benchmark [R219].

---

## 1. DOOR — electric passenger door (PMDC motor + gearbox/belt)

Signals: `pos_ref, pos, vel, current, voltage, pwm, ls_open, ls_closed, obstruction, T_motor`
(~6-10 channels). Unit of analysis: **one door cycle**. Datasets: Cranfield linear actuator [R86],
our simulator, and (recommended addition) the PHME 2026 subway-door run-to-failure set [R69].

### 1a. Door — anomaly detection (normal-only training on cycle features)

| Tier | Model | Why it is on the ladder | Code source | Hrs | Verdict |
|---|---|---|---|---|---|
| Statistical | **Moving-window variance** and **squared difference** | The two one-line baselines that five TSFM families failed to beat [R35]. Non-negotiable control row. | hand-write, ~25 lines numpy | 0.5 | **MUST** |
| Statistical | **Trivial-baseline set**: sensor-range, L2-norm-of-channels, 1-NN distance, random score | ICML 2024 position paper: these match or exceed deep TSAD once flawed metrics are removed [R5]; the random-score arm is also the point-adjustment demonstrator [R2]. | `QuoVadisTAD` repo or hand-write, ~60 lines | 1 | **MUST** |
| Statistical | **Robust z-score + EWMA**, max over cycle features | Cheapest per-feature health trend; the operator-facing "health index you can plot" that the door literature stresses over black-box accuracy [R64]. | hand-write, ~40 lines | 1 | **MUST** |
| Statistical | **CUSUM on `closing_time`, `i_mean_cruise`, `dtw_to_ref`** | Door degradation is slow drift in a per-cycle scalar; CUSUM is the standard sequential test and gives lead time directly [R72]. | hand-write, ~30 lines | 1 | **MUST** |
| Statistical | **Peer normalisation wrapper** (z against the 7 sibling doors of the unit at the same timestamp) | Real door telemetry is collected from *multiple doors of the same vehicle simultaneously*; the sibling population removes the need for a labelled normal baseline and cancels ambient/load confounders [R78]. Also the strongest UI story (8 doors, one drifting). | hand-write, ~40 lines | 2 | **MUST** |
| Statistical | **DWT detector**: db10 detail-coefficient L1-norms at levels W8, W9 + starting-current peak | Published door-actuation monitor with >96 % accuracy and <0.3 s response on a LabVIEW rig, on a DC motor + gearbox + lead-screw drive with injected armature fault, brush wear, friction and misalignment [R66]; the same feature family detects obstruction at 100-200 N within 0.3 s [R67]. Three features, trivially interpretable. | `pywt` + threshold, ~60 lines | 2 | **MUST** |
| Statistical | **PCA-SPE / T²** and **Mahalanobis (MinCovDet)** | PCA is 3rd on TSB-AD-M (0.3096) and MCD is 9th (0.2711) — both ahead of every transformer [R1]. | `sklearn` | 1 | **MUST** |
| Statistical | **Sub-PCA** (subsequence PCA) | #2 on TSB-AD-U at 0.4234, behind only matrix profile [R1][R11]. Cheap variant of the row above. | hand-write on top of PCA, ~20 lines | 1 | NICE |
| Classical ML | **Matrix profile discord (MMPAD-style, k=5)** | #1 on both TSB-AD tracks; the multivariate gain is **+0.2517 VUS-PR at 2-3 channels** and +0.0798 at 4-19, i.e. exactly the door's dimensionality [R11]. Unsupervised, one real hyperparameter. | `stumpy` (`gpu_stump` on the A4500) | 2 | **MUST** |
| Classical ML | **IsolationForest, ECOD, COPOD, kNN, CBLOF, KMeansAD** | KMeansAD (0.2949) and CBLOF (0.2731) are in the TSB-AD-M top-10; IForest/ECOD/COPOD are the standard cheap bank [R1]. | `pyod` (+ ~30 lines for KMeansAD) | 1.5 | **MUST** |
| Classical ML | **One-Class SVM** on a ≤20k subsample | A cost-aware industrial study found OCSVM **beat** a foundation model's reconstructions on AD at a fraction of the cost [R37]; on real rail bearing data a healthy-only OC-SVM is the published recipe [R120]. | `sklearn` | 1 | **MUST** |
| Classical ML | **LOF** | Same family as kNN, different locality assumption; cheap extra row. | `pyod` | 0.5 | NICE |
| Classical ML | **Density-peak clustering / adaptive mean-shift on cycle features** | Two independent rail-door papers use exactly this on real door position/speed/current telemetry, unsupervised, and a third extends it to degradation-stage analysis [R78][R79][R81]. Hyperparameter-light, seconds to run. | hand-write, ~80 lines | 2 | NICE |
| Boosting | **LightGBM residual model** — predict `current` from `pwm`, `pos`, `vel` and the sibling doors; score = max standardised residual | The covariate-informed framing is the door's natural structure (PWM and limit-switch state are *known* inputs, current is the response) [R32]; gradient-boosting gain ranking is the published door feature-screening route [R68]. | `lightgbm` | 3 | **MUST** |
| Deep | **LSTM-AE** | The reference deep detector for rail PHM telemetry: an LSTM-AE beat a sparse AE on F1/recall/precision on the Metro do Porto APU, and its SHAP attributions are what our detail panel shows [R92]. LSTMAD is 4th on TSB-AD-M (0.3066) [R1]. | torch, hand-write ~120 lines | 3 | **MUST** |
| Deep | **TCN-AE** | The specific small fitted model that outperformed MOMENT on AD with lower latency, VRAM and model size in a cost-aware industrial study [R37]. | torch, hand-write ~110 lines | 3 | **MUST** |
| Deep | **USAD** | The one deep detector that holds a top-10 place on **both** TSB-AD tracks under a non-point-adjusted metric (0.3041 M, 0.3612 U) and trains in minutes [R16][R1]. Best deep value-per-hour in the literature. | torch, hand-write ~130 lines (or lift from TSB-AD/TimeSeAD [R1][R4]) | 3 | **MUST** |
| Deep | **DeepSVDD** | One-class deep boundary, different inductive bias from reconstruction; free in pyod. | `pyod` | 1 | NICE |
| Deep | **TranAD** | Genuine, protocol-independent selling point is *speed* (very fast training, clean repo); its accuracy claims are point-adjusted and do not reproduce as a leaderboard win, and it is absent from the TSB-AD-M top-10 [R17][R1]. Worth one ablation row, not a headline. | https://github.com/imperial-qore/TranAD | 3 | NICE |
| Deep | **DCdetector** | Kept **only** as the protocol demonstrator: run it with and without point adjustment in the same table to make the argument visual — SOTA PA-F1, absent from the VUS-PR leaderboard [R15][R2][R1]. | https://github.com/DAMO-DI-ML/KDD2023-DCdetector | 4 | NICE |
| Deep | Anomaly Transformer | **SKIP** — headline results are point-adjusted, which our protocol forbids, and it does not appear in the TSB-AD-M top-10; DCdetector already fills the one demonstration slot [R2][R1]. | — | — | **SKIP** |
| Deep | CATCH (frequency patching, channel fusion) | **SKIP** — bi-level multi-objective optimisation makes it the most training-unstable item in the candidate set; it needs a full-time owner we do not have, and we could not verify its headline numbers or any TSB-AD/VUS-PR evaluation [R12]. | — | — | **SKIP** |
| Deep | KAN-AD | **SKIP** — no verified headline numbers and no TSB-AD entry; a lightweight but unproven forecasting-residual detector, and V4 says residual magnitude is the wrong score for our fault shapes anyway [R13][R36]. | — | — | **SKIP** |
| Deep | ImDiffusion | **SKIP** — diffusion inference is too slow for a 48 h build and for the live replay loop, even with the Microsoft production result [R18]. | — | — | **SKIP** |
| Foundation | **TSPulse zero-shot AD** | Strongest verified FM-for-AD evidence in the whole sweep and the cheapest: 1M params, Apache-2.0, GPU-free, and its headline is on **TSB-AD** — the benchmark that removed point adjustment [R30]. Dual time+frequency masked reconstruction suits door motor current. Needs ~1536-2048 points of context: budget door windows accordingly. | `granite-tsfm` (`pip install granite-tsfm`) | 3 | **MUST** |
| Foundation | **Adaptive conformal threshold wrapper** around any frozen scorer | Post-hoc, training-free weighted-quantile conformal calibration that adapts to distribution shift while controlling the false-alarm rate [R42]. This is exactly our "thresholds from validation only" rule, made defensible to a judge. | code in the IBM Granite repo [R42] | 2 | **MUST** |
| Foundation | **Chronos-2 covariate-informed forecast → change-point residual** | The only open TSFM we verified with native multivariate input **plus past and known-future covariates** — PWM and limit-switch state are exactly known-future covariates [R24]. Scored as change points per V4 [R36]. Apache-2.0, 120M params. | `chronos-forecasting>=2.0` | 3 | NICE |
| Foundation | **TabPFN-TS covariate-informed residual** | Training-free, 11M params, and specifically strongest in the **covariate-informed** setting, which is the door verbatim [R32]. Licence caveat: Prior Labs weights need an accepted licence, not a plain OSI grant. | `tabpfn-time-series` | 2 | NICE |
| Foundation | **DADA zero-shot** | The main *purpose-built* AD foundation model (adaptive bottlenecks + dual adversarial decoders explicitly widen the normal/abnormal reconstruction gap), competitive zero-shot vs per-dataset models [R14] — a direct attack on the "anomalies are not harder to reconstruct" failure [R35]. One model across all three subsystems is a strong demo. | https://github.com/iambowen/DADA | 3 | NICE |
| Foundation | MOMENT (zero-shot or fine-tuned) | **SKIP for door** — MOMENT's verified AD standing is *univariate* (TSB-AD-U 6th/7th) [R1], the door is multivariate, TSPulse covers the FM slot at 1/385th the parameters with better verified AD evidence [R30], and MOMENT was found indistinguishable from a one-liner [R35] and more expensive than a TCN-AE [R37]. | — | — | **SKIP** |
| Foundation | TimesFM, Time-MoE, Moirai-2, TiRex, Sundial, Toto | **SKIP** — TimesFM has *direct negative* evidence for multivariate AD [R36]; Time-MoE is a One-Liners failure case [R35]; Moirai-2 is CC-BY-NC (non-commercial) [R27]; TiRex is under a non-OSI community licence [R28]; Sundial has no verified AD evaluation [R33]; Toto's AD claim is vendor blog framing of residual thresholding [R29]. | — | — | **SKIP** |

### 1b. Door — supervised classification and severity (Cranfield `B`, simulator `D`)

| Tier | Model | Why | Code source | Hrs | Verdict |
|---|---|---|---|---|---|
| Statistical | Logistic regression / Random Forest on cycle features | Baseline; tree ensembles are the pragmatic default for door telemetry across three independent rail-door papers [R80][R68]. | `sklearn` | 1 | **MUST** |
| Boosting | **LightGBM on three-velocity-regime features** (accel / constant / decel segmentation of the current trace) | The published traditional door pipeline reaches good accuracy **only after** this segmentation; the authors stress it yields a per-fault health index you can trend, which a CNN does not [R64]. That trade-off is our headline ablation axis. | `lightgbm` | 3 | **MUST** |
| Boosting | **Stacking: RF + XGBoost → calibrated logistic regression**, with Spearman+VIF screening, F1-max threshold on validation, SHAP audit, **PR-AUC reported** | The 2026 subway-door reference pipeline under 7.3:1 class imbalance: ROC-AUC 0.977, PR-AUC 0.913, accuracy 0.937, precision 0.815, recall 0.810, F1 0.812 [R68]. PR-AUC alongside ROC-AUC is the right pair under imbalance. | `sklearn` + `lightgbm`/`xgboost` | 4 | **MUST** |
| Convolution | **MultiRocket + RidgeClassifierCV** | Bake-off redux: MultiRocket+Hydra and HIVE-COTE 2.0 are the only two in the top clique, and ROCKET-family reaches that accuracy an order of magnitude faster on multivariate panels [R43][R47][R44]. Confirmed transfer from UCR to real condition-monitoring signals [R59]. | `aeon` | 1 | **MUST** |
| Convolution | **Hydra / MultiRocketHydra** | Hydra alone loses to HC2 (27 wins / 80 losses) but **MultiRocket+Hydra is level with HC2** (48/57) at a fraction of the compute, and Hydra has a torch/GPU implementation for the A4500 [R48][R43]. Best accuracy-per-hour headline. | `aeon` (`MultiRocketHydraClassifier`) | 2 | **MUST** |
| Interval | **QUANT** | State-of-the-art among interval methods on 142 UCR datasets in **<15 min total on one CPU core** [R49]. Interval quantile features are directly interpretable in the twin ("the 3rd quarter of the closing stroke has a shifted current quantile") — worth more to judges than a rank. | `aeon` (`QUANTClassifier`) | 1 | **MUST** |
| Deep | **LITETime** | InceptionTime-level accuracy with **2.34 % of the trainable parameters**, 2.78× faster, ~2.79× less energy; LITEMV handles the multivariate case [R53]. Replaces the plan's ResNet1D/InceptionTime slot; minutes per fold on the A4500. | `aeon` deep module | 3 | **MUST** |
| Dictionary | **WEASEL 2.0** | Dilated dictionary transform with fixed memory; highest median accuracy over UCR in its comparison and a genuinely different error profile from ROCKET, so it earns an ablation slot as the third representation family; SFA words are human-readable for the door [R51][R43]. | `aeon` (`WEASEL_V2`) | 1 | NICE |
| Deep | **ConvTran** | tAPE + eRPE position encodings; significantly more accurate than SOTA convolution and transformer models on 32 multivariate problems, and one of the four deep baselines carried into MONSTER [R54][R46]. The strongest transformer option for our multivariate door panel. | https://github.com/Navidfoumani/ConvTran | 3 | NICE |
| Foundation | **RocketPFN** (ROCKET features → TabPFN v2.5, in-context, training-free) | **0.900 mean accuracy on 92 UCR datasets, matching HC2, at ~30 s median per fold**, and significantly beats MOMENT, Mantis and MantisV2 (p<0.001 each) with no learned parameters [R56]. This is the foundation-model arm that is actually worth running, and it kills the "we should use a TSFM" instinct with evidence. Check TabPFN's context limit against our window counts and its licence [R32]. | `aeon` Rocket + `tabpfn` | 4 | NICE |
| Feature selection | **Detach-ROCKET** | Sequential feature detachment prunes the redundant ROCKET feature set without sensitive tuning, preserving or improving generalisation [R57]. Two uses: an edge-deployment story, and a cheap interpretability ablation — which kernels/channels survive tells you which sensor carries the fault. | https://github.com/gon-uri/detach_rocket | 2 | NICE |
| Audio | MNPE (EMD → multi-scale normalised permutation entropy) + SVM on airborne door audio | **SKIP for the core**, NICE only if a mic is added — a phone-grade mic beating current-only on some classes is a deployable retrofit story, and permutation entropy is ~20 lines of numpy [R75]. Out of scope because our simulator emits no audio. | hand-write, ~120 lines | 4 | **SKIP** (no audio channel in our schema; revisit as future work) |
| Hybrid | HIVE-COTE 2.0 | **SKIP** — nominally the accuracy ceiling on UCR but reported at ~340 h to train over the benchmark; compute-bounded, not 48 h-compatible. Cite as the ceiling we consciously did not buy [R52][R43]. | — | — | **SKIP** |
| Foundation | Mantis, MOMENT-embeddings→LogReg | **SKIP** — RocketPFN beats both at p<0.001 with fewer features and no learned parameters, and MOMENT's pretraining corpus includes UCR training data, so UCR comparisons against it are contaminated [R56][R55][R19]. | — | — | **SKIP** |
| Interval | MomentQuant | **SKIP** — ten-day-old preprint with no head-to-head against Hydra or HC2; only relevant if inference latency becomes a judged criterion [R50]. | — | — | **SKIP** |
| Classical TF | Fractional wavelet-packet energy entropy + IPSO-SVM (plug doors) | **SKIP** — three papers in this line are paywalled with unverified numbers, and the DWT W8/W9 detector above already gives us an interpretable time-frequency arm at a third of the effort [R76][R77][R66]. | — | — | **SKIP** |

### 1d. Door — change-point detection and drift *(new in W1)*

Door degradation is slow drift in a per-cycle scalar, and a door cycle is *natively segmentable*
(accelerate / cruise / decelerate / latch), which is exactly what a segmentation algorithm is for —
and it is the same three-regime split the published door pipeline needs before its features work
[R64]. So this family earns rows on the door track for two distinct jobs: **segmenting the cycle**
and **detecting the drift across cycles**.

| Tier | Model | Why it is on the ladder | Code source | Hrs | Verdict |
|---|---|---|---|---|---|
| Statistical | **ClaSP segmentation of the cycle** (`BinaryClaSPSegmentation`) | **Hyper-parameter-free** — it determines the number of change points itself, so under our validation-only rule there is nothing to calibrate, which is a genuine advantage rather than a convenience [R176][R177]. Peer-reviewed on a 107-dataset benchmark [R177]. Replaces hand-tuned velocity-regime cut points with a learned segmentation, giving an ablation against the published fixed-regime split [R64]. | `claspy` (BSD-3) | 2 | **MUST** |
| Statistical | **Page-Hinkley on `closing_time`, `i_mean_cruise`, `dtw_to_ref`** | The maintained, tested version of the hand-written CUSUM already in §1a. Same 1954 Biometrika pedigree [R180], but with a library stopping rule and a continuous cumulative statistic that can be scored [R178]. Demote the hand-written CUSUM to a baseline it must match. | `river.drift.PageHinkley` | 0.5 | **MUST** |
| Statistical | **Pelt / BinSeg on the per-cycle scalar trend** | With **default** hyperparameters binary segmentation had the highest average performance across 14 algorithms on 37 annotated real series [R175]; Pelt is the exact-penalised counterpart [R173][R174]. One line: `rpt.Pelt(model="rbf").fit(X).predict(pen=p)`, with `pen` calibrated on validation only. | `ruptures` (BSD-2) | 1.5 (+2 shared adapter) | **MUST** |
| Statistical | **ADWIN / KSWIN on the door residual stream** | KSWIN is distribution-free and consumes **real values**, so it drops straight onto a forecast residual with no classifier underneath [R181]; ADWIN gives a `drift_detected` flag plus an `estimation` attribute usable as a score [R178][R179]. This is the principled replacement for scalar CUSUM that V4 asked for, and the family has a survey behind it [R193]. Zero new dependencies. | `river.drift` | 0.5 | **MUST** |
| Statistical | **aeon `RandomSegmenter` control** | The random baseline that Kim et al. [R2] and Huet et al. [R8] both argue any event-based metric must be compared against, but for segmentation. Free: it sits behind the same `BaseSegmenter` interface as eight real segmenters [R188]. | `aeon.segmentation` | 0 (shared) | **MUST** |
| Classical ML | **FLUSS / aeon segmentation bank** (`FLUSSSegmenter`, `InformationGainSegmenter`, `GreedyGaussianSegmenter`, `EAggloSegmenter`, `HMMSegmenter`) | Nine segmenters plus the random control behind one sklearn-style interface for ~2 h is the best hours-to-ladder-rows ratio available anywhere in this document [R188]. FLUSS also shares the matrix-profile machinery we already install for §1a [R11]. | `aeon` (BSD-3) | 2 | NICE |
| Classical ML | **changeforest** | Peer-reviewed (JMLR 2023) multivariate nonparametric CPD via a random-forest classifier log-likelihood ratio, explicitly designed for the multivariate case [R187]. Overkill for the ~6-10 channel door — it earns its MUST on the pneumatic track instead. | `changeforest` (BSD) | 2 (booked in §2c) | NICE |
| Classical ML | **BOCPD** | The **only** change-point method whose output is natively a continuous per-timestamp quantity (the run-length posterior), so it enters a VUS-PR table with **no adapter** [R185]. But it won the TCPD benchmark only under *oracle* hyperparameter tuning, not defaults [R175] — and defaults are the regime our rules put us in. | lift ~150 lines from TCPDBench (MIT) [R175], or implement [R185] | 4 | NICE |
| Statistical | DDM / EDDM / FHDDM / HDDM | **SKIP** — these consume a **binary error stream**, i.e. they require a supervised classifier underneath and cannot act as unsupervised triggers on a raw sensor or an unsupervised residual [R182][R183][R178]. KSWIN and Page-Hinkley cover the same job on real-valued input. (EDDM's bibliographic details are `SECONDARY-SOURCE` via `river` — no DOI retrievable [R183].) | — | — | **SKIP** |
| Classical ML | `bayesian-changepoint-detection` package | **SKIP** — **no licence declared on PyPI** (the license field is empty) and unmaintained since 2019 on a `.dev` version [R186]. A licence-less dependency is a legal problem in a submitted artefact. If we want BOCPD, lift it from the MIT-licensed TCPDBench repo instead [R175]. | — | — | **SKIP** |

---

### 1c. Door — RUL / degradation staging (no run-to-failure labels)

| Tier | Model | Why | Code source | Hrs | Verdict |
|---|---|---|---|---|---|
| Statistical | **DTW-to-reference-cycle degradation index → k-means severity levels → per-level dwell time → time-to-critical-severity** | Solves exactly our blocker (no run-to-failure door data): builds a degradation indicator from **motor current alone**, which is already available from the drive with no added sensors, and the severity threshold updates as operational data accumulates [R72]. A second group converges on the same "cluster cycles into severity stages" formulation [R81]. ~150 lines of Python. | hand-write | 4 | **MUST** |
| Classical | **Similarity-based ensemble RUL** (cycle statistical features → match partial trajectories against historical run-to-failure cases → monotonic countdown) | Winning-tier solution on the public PHME 2026 subway-door RUL challenge: validation 0.9467, leaderboard 0.9963, no deep net [R70]. Cycle-as-unit-of-analysis matches our event/episode metric rule. Reproducible against a public dataset we can cite **and** match. | hand-write | 5 | NICE |
| Classical | **Gated residual end-of-life** (predict failure cycle, subtract current cycle; gated blend of a similar-prefix specialist and a shock/position/current specialist) | Second independent solution on the same public dataset; names the feature families that carry door degradation signal — electrical load, position behaviour, **shock events**, trend, source-model predictions — and handles variable-length prefixes, which is our exact problem [R71]. | hand-write | 6 | NICE |
| Deep | Transfer learning from elevator doors | **SKIP** — nearest-neighbour domain with more public work and the same physics, and the sim-to-real framing matches ours, but the numbers are unverified and the elevator datasets are a detour from our three MUST datasets [R84]. Keep as the citation that simulator-first is standard practice when fault samples are scarce. | — | — | **SKIP** |

---

## 2. PNEUMATIC — brake air supply (compressor, main reservoir, twin-tower dryer)

Signals: MetroPT-3 names verbatim, 15 channels. Unit of analysis: **one compressor cycle** plus
fixed windows. Datasets: MetroPT-3 [R88] (air leaks only), MetroPT-2 [R89] and MetroPT-1 [R132]
(air + oil + dryer), our simulator.

### 2a. Pneumatic — anomaly detection

| Tier | Model | Why | Code source | Hrs | Verdict |
|---|---|---|---|---|---|
| Statistical | **Single-best-univariate-threshold control** | The decisive correction to the plan. On MetroPT-2 a **one-feature rule** — `Flowmeter_max > 16.05` (air leak), `> 16.18` (oil leak) — achieves **F1 = 1.0** on both failures with zero false positives, detecting the air leak ~150 min before the LPS signal and the oil leak >2 days ahead, beating the published deep baselines [R93]. If our deep models do not beat a one-feature threshold we must say so. Protocol reused verbatim: train on failure-free data up to a cutoff, 30-min windows with 5-min stride (L=1800, d=300), declare failure when p>0.5. | hand-write, ~40 lines | 1 | **MUST** |
| Statistical | **Compressor duty-cycle ratio** CUSUM: `idle_run_ratio = t_off / t_loaded` per cycle | The physically motivated, leak-specific univariate feature. On Dutch Railways fleet data, using **only compressor on/off logs**, air leakage manifests as idle time becoming shorter than run time; severity grading from density clustering gives RUL, and leaks were caught **1-4 weeks** before braking failure [R101]. Computable on MetroPT-3 in minutes from `COMP`/`Motor_current`. | hand-write, ~60 lines | 2 | **MUST** |
| Statistical | **Per-channel univariate detector + max/OR aggregation** | The V5 control. On eight public multivariate benchmarks, essentially every labelled anomaly is already visible in a single channel, so cross-channel modelling cannot be validated without this baseline [R21]. | hand-write, ~30 lines | 1 | **MUST** |
| Statistical | Moving-window variance, squared difference, sensor-range, L2-norm, 1-NN, random | Same mandatory control set as the door [R35][R5][R2]. | shared with §1a | 0 | **MUST** |
| Statistical | **Robust z + EWMA, PCA-SPE/T², Mahalanobis (MinCovDet)** | PCA 3rd and MCD 9th on TSB-AD-M, both ahead of every transformer [R1]. | `sklearn` | 1.5 | **MUST** |
| Classical ML | **Matrix profile (MMPAD-style, d=0.7, k=15)** | #1 on TSB-AD-M at 0.3539 vs CNN 0.3130; MetroPT-3's 15 channels sit in the **+0.0798 VUS-PR** band [R11]. | `stumpy` | shared | **MUST** |
| Classical ML | **Half-Space Trees + One-Class kNN (streaming)** | Highest value-per-hour item in the rail-pneumatic literature: the combination gives far fewer type-I errors (much higher precision) than HS-Trees alone while still catching most catastrophic APU failures, and it is online, CPU-only, and naturally respects our temporal-split and validation-calibrated-threshold rules [R99]. | `river` (`anomaly.HalfSpaceTrees`) + ~40 lines | 2 | **MUST** |
| Classical ML | **IForest, ECOD, COPOD, kNN, CBLOF, MCD, KMeansAD, OC-SVM** | The TSB-AD-M top-10 bank [R1]; OCSVM specifically beat a TSFM on industrial AD [R37]. | `pyod` / `sklearn` | shared | **MUST** |
| Boosting | **LightGBM residual model** across the 7 analogue channels (predict each from the others + lags; score = max standardised residual) | The workhorse multivariate detector the plan already expected to win; now it must clear the one-feature control [R93] and the univariate-aggregation control [R21] before we call it a winner. | `lightgbm` | shared | **MUST** |
| Boosting | **Interpretable rule learner** (shallow tree / decision stump on cycle features) | The published state of the art on this data is an interpretable rule, not a network [R93]; and the failure mode to avoid is detector+rule-explainer pipelines whose rules cover only a sliver of the failure episode — so **report rule support over the whole episode** [R98][R93]. | `sklearn` tree, depth ≤3 | 2 | **MUST** |
| Deep | **LSTM-AE** (+ SHAP attribution for the detail panel) | Reported to beat a sparse AE on F1/recall/precision on this exact APU, with SHAP attributing each flagged anomaly to specific sensors — the closest published analogue to our deliverable [R92]. Exact numbers unverified; cite the method, not the figure. | shared with §1a | shared | **MUST** |
| Deep | **Sparse autoencoder (SAE) with reconstruction-error smoothing** | The reference baseline the UCI MetroPT-3 record asks you to cite: train on nominal periods only, threshold reconstruction error, low-pass the error signal to cut false alarms [R91][R88]. Half a day to reimplement. | torch, ~90 lines | 2 | **MUST** |
| Deep | **TCN-AE**, **USAD** | TCN-AE beat a TSFM on cost-aware industrial AD [R37]; USAD is the only deep detector in both TSB-AD top-10s [R16][R1]. | shared with §1a | shared | **MUST** |
| Deep | **k-of-n corroboration voting** on top of any detector | Raise a fault only when several signals go anomalous simultaneously — a ~20-line false-alarm suppressor from a forecast-then-detect APU architecture [R95], directly reusable on door and bearing too. | hand-write, ~25 lines | 1 | **MUST** |
| Deep | **GDN / MTAD-GAT** (graph attention over the 15 channels) | Bought for **interpretability, not VUS-PR**: a learned sensor-dependency graph drawn straight into the digital twin shows *which* relationship broke (Motor_current vs TP2 vs Oil_temperature), and attention-based variants are recommended when the true topology is unknown — which it is [R22]. Honest synthesis: read alongside [R21], which argues the opposite about raw accuracy. TimeSeAD ships GDN [R4]. | `TimeSeAD` | 6 | NICE |
| Deep | **TranAD**, **DCdetector** | Ablation / protocol-demonstration rows only, as in §1a [R17][R15][R2]. | shared | shared | NICE |
| Deep | WAE-GAN (adversarial Wasserstein autoencoder) | **SKIP** — reported to detect APU failures ≥2 h ahead with no false alarms [R94], but it is the most expensive item to reimplement (~8 h) and a **one-feature threshold already reaches F1 = 1.0** on the same family of failures [R93]. Poorest expected return per hour in the set. | — | — | **SKIP** |
| Deep | Multiscale CNN-Mamba-Transformer for railway APU | **SKIP** — only Crossref metadata verified, dataset and numbers unconfirmed, ~10 h to build; given [R93] it is certainly overkill. List in related work [R100]. | — | — | **SKIP** |
| Deep | ImDiffusion, CATCH, KAN-AD, Anomaly Transformer | **SKIP** — same reasons as §1a [R18][R12][R13][R2]. | — | — | **SKIP** |
| Foundation | **TSPulse zero-shot AD** | Purpose-built for AD, #1 on TSB-AD, Apache-2.0, 1M params, CPU-capable [R30]. Context requirement 1536-2048 points — at MetroPT's 1 Hz that is ~25-34 min, which conveniently brackets the 30-min window used by [R93]. | `granite-tsfm` | shared | **MUST** |
| Foundation | **Adaptive conformal threshold wrapper** | Training-free weighted-quantile conformal calibration on the validation slice, adapting to drift while controlling the false-alarm rate [R42]. | shared with §1a | shared | **MUST** |
| Foundation | **Chronos-2 with known-future covariates** (`COMP`, `DV_eletric`, `Towers` as controls; `TP2`, `TP3`, `Motor_current` as targets) → **change-point** residual scoring | The only verified open TSFM with native multivariate + past/known-future covariate support, Apache-2.0, 120M params [R24]. Must be scored as change points, not residual magnitude, because MetroPT faults are long persistent regimes [R36] — see V4. | `chronos-forecasting>=2.0` | 3 | NICE |
| Foundation | **DADA zero-shot** | Purpose-built AD foundation model; one detector across three subsystems is a strong twin demo [R14]. | shared with §1a | shared | NICE |
| Foundation | TimesFM | **SKIP** — the most directly relevant negative result we found: on the SWaT industrial multivariate benchmark, neither per-feature forecasting nor frozen embeddings matched established baselines, and the failure mechanism (low error *during* persistent anomalies) is exactly MetroPT's fault shape [R36][R25]. | — | — | **SKIP** |
| Foundation | Time-MoE, Moirai-2, Toto, Sundial, MOMENT, TTM | **SKIP** — One-Liners failure case [R35][R26]; non-commercial licence [R27]; vendor-blog AD claim only [R29]; no verified AD evaluation [R33]; superseded by TSPulse at 1/385th the size [R19][R30]; TTM is a forecaster not an AD model [R31]. | — | — | **SKIP** |

### 2c. Pneumatic — change-point detection and drift *(new in W1)*

**This is the subsystem where the change-point family is decisive, not optional.** Three facts
converge. (i) V4: a TSFM tracks anomalous dynamics too well, so forecast error stays **low during the
interior of a long anomaly and peaks only at its boundaries**, and the SWaT authors conclude verbatim
that "naive zero-shot FMs are unsuitable for MTSAD but **promising for change-point detection**"
[R36]. (ii) MetroPT air-leak and oil-leak events are multi-hour to multi-day **persistent regimes**
[R87][R88][R89] — precisely the shape that defeats residual-magnitude scoring. (iii) Dryer tower
switches and compressor load/offload edges are themselves change points that must be *excluded* from
point scoring rather than flagged [R105]. The compressor-survey literature has a change-point
section for exactly this reason [R105].

Note also the honest framing point: the rail-compressor literature we could reach defaults to
**reconstruction/deep AD**, not change-point detection [R91][R92][R94][R95] — so a change-point arm
is a genuine differentiator for our submission rather than a reproduction of prior work.

| Tier | Model | Why it is on the ladder | Code source | Hrs | Verdict |
|---|---|---|---|---|---|
| Statistical | **Page-Hinkley on the forecast residual of `TP2`, `TP3`, `Motor_current`** | The V4 design change, implemented with a maintained detector instead of hand-written CUSUM. Its cumulative statistic is the continuous score the VUS-PR table needs [R178][R180]; **performance-aware drift detection on a model's residual stream is an established family with its own survey** [R193], so this is not an ad hoc choice. | `river.drift.PageHinkley` | 0 (shared with §1d) | **MUST** |
| Statistical | **ADWIN on the residual stream** | `update(x)` accepts arbitrary real values, exposes `drift_detected` and an `estimation` attribute; constructor `delta=0.002, clock=32, max_buckets=5, min_window_length=5, grace_period=10` [R178][R179]. Zero new dependencies — `river` is already imported for Half-Space Trees [R99]. | `river.drift.ADWIN` | 0 (shared) | **MUST** |
| Statistical | **KSWIN on raw `idle_run_ratio` and `Flowmeter_max`** | Distribution-free two-sample test on **real values**, so it needs no classifier underneath [R181]. Applied to the duty-cycle ratio it is a direct, principled competitor to the hand-written CUSUM row in §2a — and the duty-ratio feature is the one with 1-4 week lead time on fleet data [R101]. | `river.drift.KSWIN` | 0 (shared) | **MUST** |
| Statistical | **Pelt / BinSeg over the 15-channel panel, and over the residual panel** | Default-hyperparameter binary segmentation was the top performer across 14 algorithms on 37 annotated real series [R175]; `ruptures` gives Pelt, BinSeg, BottomUp, Window, Dynp and KernelCPD with ten cost functions and multivariate support, BSD-2 [R173][R174]. Penalty `pen` calibrated on validation only. | `ruptures` | 0 (booked in §1d) + 2 shared adapter | **MUST** |
| Classical ML | **changeforest** | **The best-evidenced multivariate option and the right MUST here.** Peer-reviewed in JMLR 2023, multivariate nonparametric CPD via a random-forest classifier log-likelihood ratio from out-of-bag probabilities, explicitly designed for the high-dimensional case and documented on 5-dimensional examples [R187]. This matters because MetroPT-3 is **15-channel**: `ruptures`' `CostRbf` on 15 channels is a kernel method with distance-concentration risk, whereas changeforest is built for it. Runs against `ruptures` Pelt as the cheap baseline it must beat. Caveat to state: the reported evidence is a **simulation** study, not industrial data. | `changeforest` (BSD) | 2 | **MUST** |
| Statistical | **Transition mask from `COMP` and `TOWERS` edges** | Not a detector — the exclusion rule. Emit `is_transition` and exclude ±5 s around `Towers` flips and `COMP` load/offload edges from point scoring, scoring those regions with a change-point detector instead [R105]. Load/unload is not instantaneous: sump relief ≈40 s, reload repressurise ≈3 s [R155], so the mask width is physically grounded. | hand-write, ~30 lines | 1 | **MUST** |
| Classical ML | **StreamingClaSP** | Hyper-parameter-free online segmentation [R176][R177] — a direct competitor to the `river` Half-Space Trees row already in §2a, and the comparison is free once `claspy` is installed for the door. | `claspy` | 0 (shared with §1d) | NICE |
| Classical ML | **BOCPD** | Run-length posterior is natively continuous, so no score adapter [R185]; but it wins only under oracle tuning [R175] and we are in the default regime. | TCPDBench (MIT) [R175] | 0 (booked in §1d) | NICE |
| Statistical | DDM / EDDM / FHDDM / HDDM | **SKIP** — binary-error-stream detectors requiring a supervised classifier underneath; unusable as unsupervised triggers on APU telemetry [R182][R183][R178]. | — | — | **SKIP** |
| Classical ML | `bayesian-changepoint-detection` | **SKIP** — no licence declared on PyPI, unmaintained since 2019 [R186]. | — | — | **SKIP** |
| Statistical | Ranking the change-point arm on **detection-position accuracy alone** | **SKIP as a methodology.** A large-scale comparison of 14 drift detectors explicitly "verif[ies] and challenge[s] a common belief… that the best drift detection methods are necessarily those that detect all the existing drifts closer to their correct positions, and only them" [R184]. Localisation accuracy is not the same objective as downstream utility. Material caveat: that study used **artificial, fully-labelled** streams, so it is not rail evidence — we cite it for the methodological warning only. Consequence: report the covering metric and affiliation alongside F1-at-margin, never F1-at-margin alone (§6.6). | — | — | **SKIP** |

### 2d. Pneumatic — RUL and prognostics *(new in W1 — the arm the W0 ladder omitted entirely)*

**Honest starting point.** A search for prognostics (as opposed to detection) on railway brake air
supply returns exactly one paper with a severity-to-RUL formulation — the Dutch Railways brake-pipe
work already cited [R101] — and a 2023 PRISMA systematic review of the whole compressed-air ML
literature is scoped **entirely to anomaly detection with no RUL section at all** [R226]. So this arm
is filled by porting a generic degradation model onto the severity index we already have, and we say
so on the slide. That is a contribution, not a gap.

| Tier | Model | Why it is on the ladder | Code source | Hrs | Verdict |
|---|---|---|---|---|---|
| Statistical | **General Path Model on the `idle_run_ratio` severity index → time-to-next-severity-level** | The published name for the pattern §4.4 of `rail_phm.md` already specifies: fit a functional form to a degradation measure, extrapolate to a threshold, update dynamically as data arrives [R199]. **Needs only a monotonic health index and a threshold — no run-to-failure labels**, which is exactly our situation. The brake-pipe work does the domain-specific version of this and reports 1-4 weeks of lead [R101]. ~100 lines with `scipy.optimize.curve_fit`. | hand-write | 2 | **MUST** |
| Statistical | **Gamma process on the same index** | A gamma process **forbids the health index from improving**, which is physically right for cumulative leak growth and fouling [R197]. ~40 lines: MLE of shape/scale on HI increments, RUL by gamma quantiles. Gives a predictive **distribution**, which α-λ accuracy requires [R215]. | hand-write, `scipy.stats.gamma` | 1.5 | **MUST** |
| Statistical | **Health-indicator suitability screening** (monotonicity / prognosability / trendability) | Pick between candidate indices (`idle_run_ratio`, `dP_dt_loaded`, `Flowmeter_max`, reservoir decay rate) on **prognostic suitability before fitting any RUL model** [R218] — label-free and cheap. Metric definitions `UNVERIFIED`; read the PDF before quoting formulas. | hand-write, ~60 lines | 1 | **MUST** |
| Classical ML | **Cox proportional hazards / Weibull AFT on fleet censored data** | The family that fits when there are **no run-to-failure trajectories but there are maintenance records**: needs only per-unit duration, an event indicator and covariates [R205]. This is the arm that makes SCANIA Component X earn its place in `datasets.md` — the dataset is explicitly positioned for survival analysis [R134][R209]. `lifelines` is **MIT** [R211], so no licence question. One call. | `lifelines` | 1 | **MUST** |
| Statistical | **Wiener process** on a non-monotone index | When the index fluctuates both ways, first-hitting-time RUL has a closed-form inverse-Gaussian density with mean `(w − x_t)/μ`, giving a free predictive interval [R196][R195]. ~40 lines. | hand-write | 1.5 | NICE |
| Classical ML | **Random survival forest / gradient-boosted survival** | Non-parametric alternative to Cox [R206]; `xgbse` is **Apache-2.0** and returns **calibrated survival curves with confidence intervals**, which map directly onto "probability of reaching severity level k within h days" in the twin UI [R213]. Prefer `xgbse` over `scikit-survival`, which is **GPL-3.0-or-later** and would make the deliverable a derived work [R212]. | `xgbse` | 1 | NICE |
| Deep | **SurvLoss** — asymmetric loss letting a plain regressor consume censored samples | ~30 lines on an existing regression loss, not a new model; evaluated on C-MAPSS and the SCANIA component dataset, and using censored samples alongside event samples improved RUL over baselines [R208]. Independently, including data from assets that **did not fail** improves prediction on a SCANIA fleet [R210]. The cheapest way to make the Component X ingest pay off. | hand-write on the existing torch head | 1.5 | NICE |
| Deep | WTTE-RNN (Weibull time-to-event RNN) | **SKIP** — the right idea (a network optimising a Weibull survival function handles censoring natively) and the closest published fleet-prognostics precedent [R210], but the reference implementation was last pushed 2020-08-07 so the Weibull log-likelihood must be ported by hand, and V7 says we are in the small-`n` regime where a deep RUL net is not sample-efficient [R221]. SurvLoss buys the same censoring capability for a third of the effort. | — | — | **SKIP** |
| Deep | `pycox` discrete-time hazard head | **SKIP** — the discrete-time hazard is the cleanest probabilistic form of "time to next severity level" and composes with our ordinal staging, but the model list on the retrieved page is `UNVERIFIED` and a hand-written discrete-time hazard head is ~60 lines, cheaper than the dependency [R214]. Revisit only if the Cox arm underperforms. | — | — | **SKIP** |
| Classical | Bayesian-network brake-pipe leak early warning | **SKIP** — `CONTENT UNVERIFIED` (title/venue/DOI only, zero recorded citations, low-profile venue) and it is **detection, not RUL**, so it does not fill this gap [R230]. | — | — | **SKIP** |

---

## 3. BEARING — bogie axle-box (temperature + vibration)

Signals: `T_box, vib_rms, vib_kurt, vib_crest, vib_bpfo` plus context (2-5 channels — the regime where
matrix profile wins by **+0.2517 VUS-PR** [R11]). Datasets: Ottawa UORED-VAFCLS [R119] (already in
use), Ottawa variable-speed [R118] (recommended promotion to primary), our simulator.

### 3a. Bearing — physics-grounded features and detection

| Tier | Model | Why | Code source | Hrs | Verdict |
|---|---|---|---|---|---|
| Statistical | **Kurtogram (spectral kurtosis) band selection → Hilbert envelope → BPFO/BPFI/BSF/FTF harmonic amplitudes** | The canonical recipe, explicitly written for signals masked by other machine components — exactly the bogie situation [R117]. A deep model that cannot beat kurtogram+envelope is not worth showing. Emit `sk_band_fc`, `sk_band_bw`, `sk_max` so band selection is auditable. | hand-write, ~200 lines numpy/scipy | 5 | **MUST** |
| Statistical | **Thermal residual vs a healthy physics model** driven by speed, ambient, load and **adjacent-bearing temperatures** | The closest published analogue to our bearing plan: a lumped thermal model hybridised with BP-NN/LSTM sharing the same inputs, residual = fault indicator, calibrated on real fleet operation [R110]. Makes the detector a residual, not a raw threshold. | hand-write + `sklearn` | 3 | **MUST** |
| Statistical | **Peer / opposite-axlebox differential temperature** (`dT_peer`, `dT_opposite`) | Differential temperature is the alarm structure real axlebox monitoring uses [R113][R114], and wayside HBD absolute readings are demonstrably noisy and biased by IR scan location [R112]. | hand-write, ~30 lines | 1 | **MUST** |
| Statistical | Moving variance, squared difference, sensor-range, L2-norm, 1-NN, random | Mandatory controls [R35][R5]. | shared | 0 | **MUST** |
| Statistical | **Envelope-spectrum fault characteristic band identification (tacholess)** | Order-tracking suffers resampling error and harmonic interference; time-frequency ridge methods hit resolution limits; identifying the fault band directly in the envelope spectrum avoids needing a clean tacho [R122]. Metro bearings never run at constant speed. | hand-write, ~120 lines (open-access algorithm) | 4 | NICE |
| Statistical | **Computed order tracking with the encoder channel** (tacho vs tacholess ablation) | The variable-speed Ottawa set ships a 1024-CPR encoder at 200 kHz, which makes true order tracking possible [R118] — a cheap, strong ablation against the tacholess route [R122]. | hand-write, ~80 lines | 3 | NICE |
| Classical ML | **Matrix profile on `vib_rms` / `T_box`** | 2-5 channels is the **+0.2517 VUS-PR** regime; #1 on TSB-AD-U at 0.4399 [R11]. | `stumpy` | shared | **MUST** |
| Classical ML | **One-Class SVM on MFCC + amplitude-modulation-spectrogram features, trained on healthy only** | Verified on a real commuter railway engine fed by an industrial power converter: audio-processing features beat conventional time/frequency bearing indicators, and the one-class setup is explicitly motivated by real-world class imbalance — our exact regime [R120]. A companion result shows MFCC+MLP generalises to **damage types not seen in training** [R121], giving us an unseen-fault-type ablation. | `librosa`/`scipy` + `sklearn` | 4 | **MUST** |
| Classical ML | **IForest, ECOD, kNN, KMeansAD on envelope features** | The TSB-AD-U/M classical bank [R1]. | `pyod` | shared | **MUST** |
| Boosting | **LightGBM on envelope-spectrum + time-domain features** | Envelope + band powers + kurtosis/crest into a tree ensemble is the standard strong bearing baseline, and the ROCKET family confirmed to transfer to real condition-monitoring signals sits beside it [R58][R59]. | `lightgbm` | 2 | **MUST** |
| Deep | **LITETime / ResNet1D on decimated raw windows** | LITE gives InceptionTime accuracy at 2.34 % of the parameters [R53]; a convolution ensemble (ROCKET + 1D-CNN ResNet + FCN) reached >98.8 % on multivariate gearbox/bearing vibration [R60]. | `aeon` / torch | shared | **MUST** |
| Deep | **ROCKET + 1D-CNN ensemble** | The published template for vibration fault detection: a heterogeneous ensemble beats any single convolution model [R60]. | `aeon` + torch | 3 | NICE |
| Deep | **DRSN-LSTM health index** | End-to-end health index for **railway axle-box** bearings using learned soft-thresholding denoising, motivated by axle-box signals being noise-dominated and distorted by the transfer path; validated on artificial defects **and** accelerated fatigue run-to-failure, reported sensitive to early degradation [R123]. The DRSN block is a small self-contained PyTorch module. | torch, ~150 lines | 6 | NICE |
| Deep | **Boundary-adjacent pseudo-anomaly generation + contrastive separation** (TPA-AD *idea*, not reimplementation) | The one candidate directly on bogie axle-box bearings **and** rail rolling stock; two-stage design generating pseudo-anomalous windows just outside the normal boundary [R23]. This is our fault-injection simulator, but learned — so borrow the augmentation, do not chase the paper (no code, no verified numbers). Its emphasis on degradation *evolution* over binary detection matches the RUL framing judges expect. | hand-write augmentation, ~100 lines | 5 | NICE |
| Foundation | **TSPulse zero-shot AD** | Dual time+frequency masked reconstruction is the right inductive bias for bearing vibration; #1 on TSB-AD; 1M params [R30]. | `granite-tsfm` | shared | **MUST** |
| Foundation | **MOMENT zero-shot reconstruction** | The FM result that holds up on the *univariate* track, which is the bearing's natural shape: 0.3790 VUS-PR zero-shot (7th on TSB-AD-U), ahead of USAD (0.3612) and Sub-KNN (0.3501), with fine-tuning adding only +0.007 — so **skip fine-tuning** [R1][R19]. Credible detector with no training data and no labels: the cold-start case for a new rolling-stock class. MIT licence, `pip install momentfm`. | `momentfm` | 3 | NICE |
| Foundation | **Frozen-forecaster embeddings + small supervised head**, with the **spectrum-as-pseudo-time-series** trick | Frozen forecasting models match or surpass classification-specific pretrained models, with a positive correlation between forecasting and classification skill [R28]. The bearing-specific ESANN study converts the vibration **spectrum** into a pseudo time series so a time-domain TSFM can consume it, sidestepping the kHz-vs-low-frequency sampling mismatch [R38]. Cheapest credible FM arm for Ottawa. | `chronos-forecasting` + `sklearn` head | 3 | NICE |
| Foundation | **TimeRep** — score anomalies from **intermediate layer** representations plus a core-set reference bank and drift adaptation | The cheapest way to make a frozen TSFM competitive, claimed to beat non-DL, DL and FM baselines on the 250-series UCR Anomaly Archive [R41]. Good ablation axis against raw reconstruction error, which [R35] shows is a dead end. | hand-write on top of `momentfm` | 3 | NICE |
| Foundation | **Chronos-2 frozen embeddings + Wide & Deep RUL head** | Template verified for prognostics: a **frozen** Chronos-2 backbone with a Wide & Deep head reports average RMSE 10.32 on full C-MAPSS with no backbone fine-tuning [R40]. Directly transplantable as our bearing/door RUL head. | `chronos-forecasting` + torch head | 3 | NICE |
| Foundation | TiRex | **SKIP** — smallest genuinely top-of-leaderboard forecaster (35M, one-line pip) but the **NXAI Community Licence is not OSI**; we may need to hand this code over [R28]. Its companion feature-extractor result is still used, via Chronos-2 instead. | — | — | **SKIP** |
| Foundation | ESANN in-context-learning bearing classifier (reproduction) | **SKIP as a reproduction** — the abstract names neither the TSFM nor the bearing datasets and reports no numbers, so there is nothing to reproduce; we keep only the spectrum-as-pseudo-series trick above [R38]. | — | — | **SKIP** |
| Foundation | BearingFM | **SKIP** — `UNVERIFIED` this session; not in the citable set [R38 notes]. | — | — | **SKIP** |

### 3b. Bearing — supervised classification (Ottawa `C`)

| Tier | Model | Why | Code source | Hrs | Verdict |
|---|---|---|---|---|---|
| Statistical | LogReg / RF on time-domain + envelope features | Baseline. | `sklearn` | shared | **MUST** |
| Boosting | LightGBM-envelope | See §3a. | `lightgbm` | shared | **MUST** |
| Convolution | **MultiRocket + Ridge**, **Hydra** | Top clique of the 2024 bake-off redux [R43][R47][R48]; SelF-Rocket confirms the family transfers to real machinery fault signals [R59]. | `aeon` | shared | **MUST** |
| Interval | **QUANT** | Interpretable interval quantiles at <15 min on one core [R49]. | `aeon` | shared | **MUST** |
| Deep | **LITETime / ResNet1D** | See §3a [R53][R60]. | `aeon` / torch | shared | **MUST** |
| Shapelet | **Interpretability-constrained shapelet ensemble** with Hilbert envelope demodulation, **recording-level CV** | Reported to **match ROCKET** on CWRU and MFPT with the difference smaller than fold-to-fold SD, i.e. interpretability costs nothing here; and the recording-level protocol is the methodological point — most bearing papers inflate accuracy with segment-level splits [R58]. Exact F1 values are `UNVERIFIED` (MDPI 403). | `aeon` shapelet transform | 4 | NICE |
| Convolution | **SelF-Rocket** (multivariate extension) | Best overall accuracy-latency trade-off for multi-class machinery faults; highest accuracy on MaFaulDa, competitive on the harder stator inter-turn-short set — and the ITSC half is relevant to our **PMDC door motor** winding faults from current signature [R59]. | https://arxiv.org/abs/2608.18716 (code UNVERIFIED) | 3 | NICE |
| Transfer | **UDTL unsupervised deep transfer benchmark** | A unified, reproducible PyTorch benchmark for domain adaptation across standard bearing datasets, with an extended interface for custom datasets — so we can plug our simulator in as the source domain and Ottawa as the target, instead of reimplementing a dozen DA methods [R116]. This is our lab→field transfer ablation. | https://github.com/ZhaoZhibin/UDTL | 6 | NICE |
| Hybrid | HIVE-COTE 2.0 | **SKIP** — ~340 h compute [R52][R43]. | — | — | **SKIP** |
| Wayside | Doppler-corrected wayside acoustic diagnosis | **SKIP** — needs a microphone array and Doppler-removal resampling; a wayside sensing modality out of scope for a telemetry-only build, and wayside acoustics reliably catch end-of-life bearings but miss incipient defects [R127][R109]. Keep for the sensor-fusion slide. | — | — | **SKIP** |

### 3c. Bearing — change-point detection *(new in W1)*

The bearing's change-point job is different from the other two: a metro axle box runs at
**continuously varying speed** and the thermal signal is driven by duty cycle — "the bearing cools
significantly by the wind while driving and warms up while standing still" [R151]. So most apparent
"changes" in `T_box` are **operational regime changes, not faults**, and the honest use of this family
here is largely as a **confounder segmenter**: segment the run into speed/dwell regimes first, then
detect within regime.

| Tier | Model | Why it is on the ladder | Code source | Hrs | Verdict |
|---|---|---|---|---|---|
| Statistical | **Pelt / BinSeg on `(speed, dwell_fraction)` to segment operating regimes** | Segment first, detect within regime. This is the bearing-specific answer to the [R151] observation that normal `T_box` variation is driven by ambient, sunshine and duty cycle; and to [R159], whose records sweep 13.7-28.9 Hz shaft speed **within** a single 10 s record so a global threshold is meaningless. Default-hyperparameter BinSeg is the top performer on real annotated series [R175][R173]. | `ruptures` | 0 (shared) | **MUST** |
| Statistical | **Page-Hinkley / ADWIN on the thermal residual** | The residual-stream drift detector, applied to `thermal_residual = T_box − T_box_pred` from §3a [R110]. Justified as a family by the performance-aware drift survey [R193]. Matches the structure of the published Netherlands Railways slow-degradation rule, which is itself a sequential deviation test (3.5 σ over ≥10 measurements in 30 days) [R151] — so we can benchmark a library drift detector directly against a real operator rule. That comparison is a strong slide. | `river.drift` | 0 (shared) | **MUST** |
| Classical ML | **FLUSS (matrix-profile segmentation)** | Reuses the matrix-profile machinery already installed for §3a at essentially zero marginal cost [R11][R188], and is the natural segmenter for vibration where the regime boundary is a change in *shape* rather than in mean. | `aeon.segmentation.FLUSSSegmenter` | 0 (shared with §1d) | NICE |
| Classical ML | **ClaSP on the vibration envelope** | Hyper-parameter-free, so nothing to calibrate [R176][R177]. | `claspy` | 0 (shared) | NICE |
| Classical ML | changeforest | **SKIP for bearing** — its advantage is the high-dimensional multivariate case [R187], and the bearing panel is 2-5 channels, which is the regime where matrix profile leads by **+0.2517 VUS-PR** [R11]. It is MUST on the pneumatic track instead. | — | — | **SKIP** |
| Statistical | DDM / EDDM / FHDDM / HDDM | **SKIP** — binary-error-stream detectors; no unsupervised classifier error to feed them [R182][R183][R178]. | — | — | **SKIP** |

### 3d. Bearing — RUL and prognostics *(new in W1 — this is what the run-to-failure datasets are for)*

The W0 `datasets.md` added XJTU-SY [R124], FEMTO/PRONOSTIA [R143][R228] and the Paderborn
run-to-failure set [R138] on the explicit grounds that they are "the standard public RUL benchmark
judges will expect if we claim a prognostics result" and "the rare combination we actually need for a
credible bogie RUL story". **Nothing in the W0 ladder consumed them.** These rows fix that.

| Tier | Model | Why it is on the ladder | Code source | Hrs | Verdict |
|---|---|---|---|---|---|
| Statistical | **Health-indicator construction + suitability screening**, then **gamma process → RUL to the next severity level** | The bearing HI is the envelope-band energy / RMS trend; score candidate HIs on monotonicity, prognosability and trendability *before* fitting [R218], then fit a **gamma process** because spall growth is strictly monotone [R197]. Gives a predictive distribution, which α-λ accuracy needs [R215]. ~40 lines on top of features we already compute. V7: at `n` ≈ 15 trajectories this is the sample-efficient family, not a deep net [R221]. | hand-write, `scipy.stats.gamma` | 1.5 (shared with §2d) | **MUST** |
| Classical | **Similarity-based RUL** — match the partial trajectory against the XJTU-SY / FEMTO library, aggregate matched remaining lives | The canonical formulation [R200], and the published ancestor of the door row `similarity_ensemble_rul` we already carry [R70] — so promoting it to bearing is mostly reuse. **Requires run-to-failure trajectories**, which is exactly why XJTU-SY and FEMTO are ingested. ~100 lines with DTW or Euclidean matching over a smoothed HI. | hand-write | 2 | **MUST** |
| Statistical | **General Path Model** on the same HI | Threshold-extrapolation form of the same idea, needing **no** run-to-failure labels — so it also runs on our simulator and on the CITEF F0-F4 severity ladder [R140] where no trajectory library exists [R199]. | shared with §2d | 0 | **MUST** |
| Statistical | **Wiener process** on a fluctuating HI | Vibration HIs are noisy and non-monotone before the spall stabilises; first-hitting-time gives a closed-form inverse-Gaussian RUL density [R196][R195]. | hand-write | 0 (shared) | NICE |
| Classical | **Inverse-Gaussian process** | The third member of the standard trio, for when the gamma fit is poor [R198]. ~30 lines with `scipy.stats.invgauss`. | hand-write | 1 | NICE |
| Classical | **Particle-filter / Kalman degradation-state tracking with uncertainty propagation to RUL** | Our simulator *is* a state-space degradation model, so wrapping it as a `PrognosticsModel` gives particle-filter RUL with uncertainty bands **plus the judge-recognised metrics for free** — `alpha_lambda`, `prognostic_horizon`, `cumulative_relative_accuracy`, `monotonicity` on a `ToEPredictionProfile` [R201]. Strong digital-twin demo: the twin shows a widening RUL band, not a point estimate. **Licence caution: ProgPy is NASA-1.3, not OSI-permissive** — check the NEBULA X IP rules first; `filterpy` (MIT) covers Kalman only and has no particle filter [R202], and a bootstrap particle filter is ~60 lines. | `progpy` (or hand-write) | 3 | NICE |
| Boosting | **LightGBM / XGBoost RUL regression on HI features** | Evidence, not habit: on C-MAPSS FD003 **XGBoost RMSE 13.36 beats a 1D CNN (15.68) and an LSTM (14.20)** [R222]. The cheap supervised RUL row that the deep arm must clear. | `lightgbm` | 2 | NICE |
| Classical | **Weibull / exponential population life fitting with right-censored suspensions** | Gives a **population baseline RUL** to quote against the conditional model — the "what would you predict knowing nothing about this bearing?" control, which is the RUL analogue of our random-score row. `SurPyval` is MIT and uniquely also handles **truncation**, which matters because fleet observation windows are left-truncated [R204]; `reliability` (LGPL-3.0) is the alternative [R203]. One call. | `surpyval` | 0.5 | NICE |
| Foundation | **Chronos-2 frozen embeddings + Wide & Deep RUL head** | Already NICE in §3a; kept, with V7's caveat attached — frozen-backbone RMSE 10.32 on full C-MAPSS is a real result [R40], but C-MAPSS has far more trajectories than XJTU-SY's 15, so do not assume it transfers to `n` = 15 [R221]. | `chronos-forecasting` + torch | 0 (already booked) | NICE |
| Foundation | **Tabular foundation model with in-context learning for RUL** | Needs **no training run** on the A4500 and reports the best average ranks across prognostic and diagnostic PHM tasks specifically in **low-data regimes** — which is our regime [R223]. `UNVERIFIED`: the benchmark datasets are not named on the abstract page, so it is unknown whether bearing run-to-failure sets are among them. NICE, never MUST. | `tabpfn` | 3 | NICE |
| Deep | Deep sequence RUL nets (LSTM/CNN regression on raw windows) | **SKIP as a headline, keep only as the §3a `drsn_lstm_health_index` NICE row.** V7: with `n` ≈ 15 run-to-failure trajectories the generalization bound `O(B²√(p/n))` puts a high-`p` model far outside its sample-efficient regime, and embedding degradation physics instead can cut data requirements by up to two orders of magnitude [R221]; a boosted tree already beats both a CNN and an LSTM on C-MAPSS FD003 [R222], which has *more* data than we do. | — | — | **SKIP** |

---

## 4. Cross-subsystem

| Model | Why | Code source | Hrs | Verdict |
|---|---|---|---|---|
| **MSAD model selection** (train a cheap time-series classifier that maps series features to the best detector) + **Averaging Ensemble** fallback | Since TSB-AD proves no single detector wins everywhere [R1], the winning move is to *select per series*: model selection outperforms **every** individual detector while staying within the same order of magnitude of execution time, and the Averaging Ensemble also beats all 12 individual detectors (but requires running every method) [R20]. We have three heterogeneous subsystems with different anomaly morphologies — door: short transient; pneumatic: slow persistent drift; bearing: spectral — so a per-subsystem selector is both a real gain and a strong twin demo. | https://github.com/boniolp/MSAD | 4 | NICE |
| **Detach-ROCKET channel/kernel pruning as per-signal ablation** | Which kernels and channels survive pruning tells you which sensor carries the fault, feeding both the UI and the "can we drop a sensor?" question an operator will ask [R57]. | see §1b | shared | NICE |

---

## 5. Effort budget for the MUST set

Written once, shared across subsystems. One owner (P2) plus overflow.

| Block | Hours |
|---|---|
| One-liners + trivial baselines + random control + PA%K | 2.0 |
| Robust z + EWMA + CUSUM + duty-cycle ratio + single-feature control | 4.0 |
| PCA-SPE/T² + Mahalanobis + peer normalisation + per-channel-OR aggregation + k-of-n voting | 5.0 |
| pyod/sklearn bank (IForest, ECOD, COPOD, kNN, CBLOF, MCD, OC-SVM, KMeansAD) | 2.0 |
| Matrix profile (stumpy, U and M variants) | 2.0 |
| Half-Space Trees + One-Class kNN (river) | 2.0 |
| LightGBM residual + LightGBM supervised + stacking/SHAP | 7.0 |
| Deep bank sharing one budgeted `_train_loop`: LSTM-AE, TCN-AE, USAD, SAE | 8.0 |
| aeon classification bank: MultiRocket, Hydra, QUANT, LITETime | 4.0 |
| TSPulse + conformal threshold wrapper | 5.0 |
| Door features: DWT W8/W9 + start-current peak + 3-regime segmentation + DTW-to-reference + k-means severity | 7.0 |
| Bearing features: kurtogram + envelope + BPFO/BPFI/BSF/FTF + MFCC/AMS + thermal residual + peer ΔT | 13.0 |
| Pneumatic-specific: single-feature control, duty-cycle ratio, per-channel-OR, sparse AE, shallow rule learner | 8.0 |
| Bearing-specific extras: ResNet1D, LightGBM-envelope | 4.0 |
| Metrics harness: vendored VUS-PR, affiliation-F, range-based P/R, event/episode metrics, tsadmetrics wiring | 4.0 |
| **Change-point family (new, W1)**: `ruptures` Pelt/BinSeg + `claspy` ClaSP + `river` Page-Hinkley/ADWIN/KSWIN + `aeon` RandomSegmenter control + `changeforest` + the `COMP`/`TOWERS` transition mask | 7.0 |
| **Breakpoint→score adapter (new, W1)**: turn discrete breakpoints and booleans into a continuous per-timestamp score so CPD rows can enter the VUS-PR table; written once, reused by every CPD row | 2.0 |
| **CPD evaluation (new, W1)**: covering metric + F1-at-margin ported from the MIT-licensed TCPDBench repo | 2.0 |
| **Prognostics family (new, W1)**: General Path Model + gamma process + Wiener + HI suitability screening + Cox/Weibull-AFT via `lifelines` + similarity-based RUL on XJTU-SY/FEMTO | 9.0 |
| **Prognostics metrics (new, W1)**: α-λ accuracy, prognostic horizon, cumulative relative accuracy, monotonicity, C-MAPSS asymmetric score (reported beside RMSE), leave-one-unit-out split auditor | 2.5 |
| **Total MUST implementation** | **~99.5 h** (was ~77 h before the W1 pass; door 49 + pneumatic 11 + bearing 17 new-work hours plus 22.5 h of change-point and prognostics work, per `configs/model_ladder.yaml`) |

**Is the enlarged MUST set still feasible for 3 people?** Yes, but only because the new work is
unusually cheap per row and is deliberately front-loadable. Of the 22.5 h added: **7 h is library
wiring** (`ruptures`, `claspy`, `changeforest` and `aeon.segmentation` are `fit().predict()`;
`river.drift` is already a dependency and adds **zero** new packages), **4.5 h is shared
infrastructure** written once (the breakpoint→score adapter and the CPD metrics), and **9 h of the
prognostics block is four models of 30-100 lines each** (`scipy` MLE fits and one `lifelines` call),
not training runs. Roughly **18 of the 22.5 h lands in the 14-17 Sep prep window**, because none of it
depends on the simulators being finished — the adapters and metrics can be built and unit-tested
against synthetic step functions today. Nothing added here requires a GPU.

**If the budget still slips, cut in this order** (all four remain defensible SKIPs to a judge):
`changeforest` → `ruptures` Pelt already covers offline CPD; similarity-based bearing RUL → the gamma
process covers the RUL arm without a trajectory library; Cox/Weibull-AFT → the pneumatic RUL story
survives on the General Path Model alone; HI suitability screening → pick the HI by inspection and
say so.

Feasible: ~25 h of it lands in the 14-17 Sep prep window (features, metrics harness, classical bank),
the remainder inside the 48 h with one owner plus overflow to a second person. Every MUST model is a
library call or ≤150 hand-written lines. Nothing on the MUST list requires a training run longer than
the deep bank's shared budget.

---

## 6. Evaluation protocol evidence

### 6.1 Why we refuse point adjustment

Point adjustment (PA) marks an entire ground-truth anomaly segment as detected if the detector fires
on any single point inside it. Kim et al. prove the protocol is catastrophically permissive: **a
random anomaly score attains state-of-the-art PA-F1** on the standard benchmarks, and an untrained
model is competitive with published methods even when PA is forbidden [R2]. Every published PA-F1
number in the TSAD literature — including Anomaly Transformer's and DCdetector's headline results —
is therefore unusable for ranking [R2][R15].

This is not a fringe position. The current SOTA benchmark, TSB-AD (NeurIPS 2024), **completely
removes point adjustment** from its protocol and states plainly that point adjustment plus threshold
tuning "misleadingly inflate scores and hinder fair comparison" [R1]. TimeSeAD reaches the same
conclusion for deep *multivariate* detectors: labels are erroneous, there is no accepted standard
metric, protocols are inconsistent, and cross-paper comparison is meaningless [R4]. And the ICML 2024
position paper shows trivial baselines — 1-NN distance, sensor range, L2-norm of channels — match or
exceed deep TSAD models once flawed metrics are removed, arguing deep TSAD models typically learn
near-linear mappings [R5].

**Concrete consequence for our tables.** We will run DCdetector both ways in one table — PA-F1 next
to VUS-PR and event-F1 — so the argument is visual rather than asserted [R15][R2]. And we include a
random-score row in every AD ablation; if a judge asks "how do you know your numbers are real", the
random row is the answer.

### 6.2 What we use instead

| Metric | Role | Why | Source |
|---|---|---|---|
| **VUS-PR** | **Primary** threshold-free score | Parameter-free and threshold-independent (labels relaxed from {0,1} to [0,1] with a decay, ROC/PR surface integrated over a range of buffer sizes), so it cannot be gamed by threshold tuning, and tolerant of small temporal misalignment — which matters because our simulator's injected-fault onsets are approximate. TSB-AD identifies it as the most reliable measure and ranks by it, so our numbers are leaderboard-comparable [R7][R1]. | vendored from TSB-AD |
| **Event recall within [onset − H, failure]**, **false alarms per train-day**, **lead-time distribution per event** | **Primary** operational metrics | This is how a depot maintainer thinks. The operational requirement on this exact APU is detection ≥2 h before removal with as few false alarms as possible [R90]; air-leak severity work reports 1-4 weeks of lead [R101]; the interpretable-rules paper reports ~150 min before the LPS signal and >2 days for oil [R93]. | hand-written |
| **Affiliation precision / recall / F1** | **Secondary** event metric | Parameter-free event-level extension that scores each ground-truth event by the temporal distance between predicted and actual events, with theoretical robustness against random-score strategies [R8]. Reported as Affiliation-F in TSB-AD, so it is leaderboard-comparable [R1]. **Secondary, not primary** — see 6.3. | TSB-AD implementation |
| **Range-based precision / recall** | **Secondary** | The original formalisation of episode-level scoring; its *existence reward* is the knob that says "detecting the compressor degradation episode at all is worth most of the credit, detecting every second of it is not" — the right bias for maintenance alerting [R9]. **Fix the position/cardinality weights once and state them**, or the metric itself becomes a tuning knob. | TSB-AD (R-based-F1) |
| **PA%K** | Reported as a second axis | The less-gameable replacement Kim et al. propose; ~1 h to implement and it stays near-flat under repeated trials [R2][R10]. | `tadpak` [R2] |
| AUROC / AUPRC | Reported, not ranked on | Point-level; kept for comparability with the rail literature only. | `sklearn` |
| **Latency, peak VRAM, serialized model size, fit/score seconds** | Reported for every row | The template from the cost-aware industrial study, which found a foundation model incurred higher latency, VRAM and state-dict size than a TCN-AE while losing on accuracy [R37]. A judge can then see the cost side, not just the score. | harness |

Implementation: `tsadmetrics` (`pip install tsadmetrics`, GPL-3.0 — keep it in the evaluation harness
only, not linked into anything we may want to relicense) gives 34 metrics under an explicit taxonomy
— SPM (point-wise, temporally blind) vs TEM (temporal), with TEM split into TPDM (partial detection
inside a true event counts; **this class contains point-adjusted F-score**), PTDM (detection must
cover a significant fraction of the event) and TMEM (alignment of true and predicted events) [R102].
**We state in the write-up which taxonomy class each reported metric belongs to, and that we exclude
the TPDM / point-adjusted variants.**

### 6.3 The 2026 correction: report single runs

A 2026 adversarial stress-test of the five post-PA metrics (PA%K, range-based P/R, affiliation P/R,
VUS-ROC, VUS-PR) against random detectors on UCR, SMD, SMAP, MSL, NAB and PSM finds that under
**single honest runs (N=1) all five resist gaming**, but under best-of-N reporting they split
sharply: **affiliation becomes gameable at N≈3, ROC-based metrics at N≈9-11, while PR-based metrics
and PA%K stay near-flat** [R10]. Therefore:

- **VUS-PR is primary**, VUS-ROC is not reported as a ranking metric.
- **Affiliation-F is secondary only.**
- **Seeds are fixed and N is stated explicitly** in the write-up. No best-of-N, no seed shopping.
- The paper is a single-author preprint with low citation count — treat the specific N thresholds as
  indicative, but the qualitative ranking is free to honour and costs us nothing.

### 6.4 Splits, thresholds and leakage

- **Temporal splits are mandatory on MetroPT.** The benchmarks-are-flawed paper names **run-to-failure
  bias** as one of four flaws afflicting the majority of exemplars in Yahoo, NAB and NASA SMAP/MSL
  [R3]. MetroPT-3 has run-to-failure structure, so a random split lets a model learn "later = broken".
- **Thresholds are calibrated on a validation slice only**, never on test. TSB-AD's Tuning/Eval split
  is exactly this discipline [R1], and the adaptive conformal wrapper makes it principled and
  drift-aware rather than a fixed quantile [R42].
- **Bearing-wise splits.** Segment-wise and condition-wise splits on CWRU/Paderborn/Ottawa **leak
  information and inflate accuracy**; bearing-wise partitioning (no physical bearing shared between
  train and test) is required, and the number of unique **training bearings**, not segments, decides
  robustness [R115]. Recording-level CV is the same point from the shapelet study [R58]. We report
  the **gap** between window-random and bearing-wise as our honesty headline.
- **Door: leave-one-load-out and leave-one-motion-profile-out**, because the same door fault looks
  different under different operating conditions and needs condition-aware weighting rather than one
  global threshold [R65].
- **Our own simulator is a feature, not a shortcut.** Much of the apparent progress in TSAD is
  illusory because public benchmarks are trivially solvable or mislabelled [R3]; and the door
  literature's "physically constrained data augmentation that respects kinematic consistency" is the
  published precedent that physics-respecting synthetic cycles are a legitimate contribution [R68].

### 6.5 Sanity anchors — numbers that mean we have a bug

- Best average VUS-PR on TSB-AD-M is ≈**0.31-0.35**; on TSB-AD-U ≈**0.42-0.44** [R1][R11]. **If our
  MetroPT-3 pipeline reports 0.95 F1 we are almost certainly leaking or point-adjusting.**
- A single-feature threshold gets **F1 = 1.0** on MetroPT-2's two failures [R93] — so a high F1 on
  MetroPT proves nothing about model quality, and with **2-4 failure events** an episode-level F1 has
  essentially no statistical power. Report lead-time distributions and false alarms per train-day
  instead, and say on the slide that event recall over 4 events is coarse.
- Published MetroPT results of ">98 % F-measure, >99 % accuracy" [R97] and "98.89-99.24 % accuracy"
  [R106] are **point-level supervised classification on a randomly split, heavily imbalanced stream
  where the positive class is a handful of long contiguous windows**. That is precisely the
  evaluation our rules forbid. We cite those two papers for their **explanation UI and multi-task
  framing**, never as a benchmark to beat.

### 6.6 Scoring the change-point arm *(new in W1)*

The W0 document made change-point scoring the central design decision (V4) but gave no way to
**measure** whether a change-point detector was any good. This closes that.

**The hard constraint.** VUS-PR requires a **continuous per-timestamp anomaly score** [R7]. But
`ruptures` Pelt/BinSeg, `claspy` ClaSP, `changeforest` and `aeon`'s segmenters all emit **discrete
breakpoint sets**, and `river`'s drift detectors emit **booleans** [R173][R176][R187][R188][R178].
**None of them can enter our VUS-PR table unmodified.** Every change-point row therefore passes
through one shared adapter, written once:

| CPD row | Continuous score used |
|---|---|
| Pelt / BinSeg / changeforest / aeon segmenters | exponential kernel on distance to the nearest breakpoint, or the cost-improvement curve exposed directly |
| ClaSP | the **ClaSP profile** itself (already a per-timestamp curve) |
| Page-Hinkley | the running cumulative statistic |
| ADWIN | the `estimation` drift magnitude |
| BOCPD | the **run-length posterior** — natively continuous, **no adapter needed** [R185] |

**The two accepted change-point metrics**, both from the only CPD benchmark we verified [R175], and
both portable out of its **MIT-licensed** repo in ~2 h:

- **Covering metric** (adapted from Arbeláez et al. 2010 image segmentation):
  `C(G,G') = (1/T)·Σ_{A∈G} |A|·max_{A'∈G'} J(A,A')`, with `J` the Jaccard/IoU of segments. Scores the
  **partition**, which is the right object when the question is "did you find the regime boundary".
- **F-measure with an annotation margin `M`** (`M = 5` time steps in TCPD). Note that the margin is
  conceptually the **same device** as VUS's tolerance buffer — VUS's whole argument is that you
  should **integrate over the tolerance** rather than fix one value [R7], which makes integrating over
  `M` a strict improvement on reporting F1 at a single margin. We report both.

**And one metric that bridges the two families with no adapter at all.** Affiliation precision/recall
is defined by the **temporal distance between predicted and true event sets** and is parameter-free
[R8], so it scores discrete change-point outputs and discrete anomaly-event outputs on **identical
footing**. It also normalises against a random baseline, quantifying how much better than random a
result is — valuable given our no-point-adjustment stance [R2]. **Reporting plan: VUS-PR primary for
score-based rows, affiliation P/R as the cross-family column that lets CPD rows compete, covering and
F1-at-margin as the segmentation-specific pair.** Affiliation stays **secondary**, per §6.3, because
it is the first of the post-PA metrics to become gameable under best-of-N (N≈3) [R10] — which costs
us nothing, since we report N=1.

**One methodological warning we must honour.** A large-scale comparison of 14 drift detectors
explicitly challenges "a common belief in the area, namely that the best drift detection methods are
necessarily those that detect all the existing drifts closer to their correct positions, and only
them" [R184] — i.e. **localisation accuracy is not the same objective as downstream utility**, which
is exactly what an F1-at-margin-5 metric optimises. That is why the covering metric and affiliation
are reported alongside it rather than after it. Material caveat to state: [R184] used **artificial,
fully-labelled** streams, so it is a methodological warning, not rail evidence.

**Cost, not just accuracy.** TimeEval is the one peer-reviewed source that systematically reports
**runtime alongside accuracy** across 71 detectors and 976 datasets [R189] — the template for the
cost columns we already carry [R37]. `NUMBERS UNVERIFIED`: we did not retrieve its per-algorithm
table, so no specific winner is quoted.

**Design rationale citations for the arm as a whole**: offline CPD organised as cost function ×
search method × constraint, explicitly multivariate [R174]; supervised/unsupervised CPD and
comparison criteria [R190]; concept drift adaptation [R191][R192]; and — most directly — the
**performance-aware drift detector** survey, which reviews precisely the pattern of firing on a
model's residual/error degradation rather than on instantaneous magnitude [R193]. That is verdict V4
as an established family, not an ad hoc choice.

**Honest limit on the evidence.** We could not find a change-point benchmark on **rail or industrial
compressor** telemetry. TCPD is 37 general real-world series [R175], the ClaSP benchmark is 107
segmentation datasets [R177], and changeforest's evidence is a **simulation** study [R187]. The one
rail-domain result that speaks to this is indirect: TimesFM on SWaT failed at multivariate anomaly
detection but the authors concluded it is "promising for **change-point detection**" [R36]. We say
that plainly rather than implying a rail benchmark exists.

### 6.7 Scoring the RUL arm *(new in W1)*

- **Primary metrics**: α-λ accuracy, prognostic horizon, cumulative relative accuracy and
  monotonicity — the NASA offline prognostics suite, which incorporates **probabilistic uncertainty
  estimates** rather than scoring a point estimate [R215][R216]. `progpy` implements exactly these
  four names on a `ToEPredictionProfile` [R201], so the harness is a wrapper, not an implementation.
  `NUMBERS UNVERIFIED`: the [R215] landing-page abstract does not itself enumerate the metric names —
  our mapping rests on ProgPy's implementation, so **read the PDF before quoting definitions
  verbatim**. α-λ needs a predictive **distribution**, which is a further reason the stochastic
  degradation family (V7) is the right MUST.
- **The C-MAPSS asymmetric score, with its pitfall stated.** It penalises late (optimistic) RUL
  predictions exponentially more than early ones [R217] — the right bias for maintenance. But it is
  **unnormalised and exponential**, so one badly-late unit dominates the fleet total and the score is
  **not comparable across datasets with different unit counts or life lengths**. Report it **beside
  RMSE and a per-unit error distribution, never alone**. This is the RUL-side analogue of our
  no-point-adjustment rule. ~15 lines.
- **Leave-one-unit-out is mandatory, and it is the prognostics twin of §6.4's bearing-wise rule.**
  Device-level leakage optimistically biases RUL benchmarks; the fix is a leakage-safe preprocessing
  pipeline plus **leave-one-device-out** evaluation, under which plain SVR was the most consistent
  model across devices [R219]. A second 2026 study reports that naive splitting inflates accuracy from
  a genuine **20-60 % to 99.9 %** on the same data and prescribes chunk-based leakage-audited splits
  with **5 seeds** and significance testing [R220] — note that both are `NOT PEER-REVIEWED` preprints,
  so we cite them for the protocol, not for a number. **Concretely: leave-one-bearing-out on XJTU-SY
  and FEMTO, never random window splits, and fit scalers and HI normalisation on training bearings
  only.**
- **Choose the health index before fitting.** Score candidate HIs on monotonicity, prognosability and
  trendability — prognostic suitability, not downstream RUL error [R218]. Label-free, and it stops us
  burning GPU time on an index that was never going to trend. Exact definitions `UNVERIFIED`.
- **Report the population baseline.** A Weibull fit to the fleet's lives with right-censored
  suspensions [R204][R203] answers "what would you predict knowing nothing about this unit?" — the
  RUL analogue of the random-score row in every AD table.
- **Fair comparison across families.** There is a published precedent for putting RUL regression and
  survival analysis on the **same ladder** and comparing them on run-to-failure data [R207], which is
  what §2d and §3d do. `NUMBERS UNVERIFIED` — the landing page does not say which family won.

---

## 7. What changed vs the pre-research plan

The plan's ladder was sound in shape — tiers, no point adjustment, temporal splits, thresholds from
validation. Eighteen substantive changes were made in the W0 pass; the **W1 pass adds six more,
numbered 19-24 at the end of this section**.

**Added (were absent from the plan)**

1. **Matrix profile is now MUST for all three subsystems.** It is the current #1 on both TSB-AD
   tracks, and its advantage is concentrated at exactly our channel counts [R11]. The plan had no
   matrix-profile row at all. `stumpy.gpu_stump`, ~2 h. *Biggest single change.*
2. **Mandatory one-liner controls**: moving-window variance and squared difference, because five TSFM
   families are statistically indistinguishable from them [R35]; plus the QuoVadisTAD trivial set
   (sensor-range, L2-norm, 1-NN) and a **random-score** row [R5][R2]. The plan had robust z-score but
   none of these named controls.
3. **Per-channel-univariate + max/OR aggregation control**, without which no multivariate model in our
   table is justified [R21]. Also a novel ablation on our own simulator: does our injected fault
   actually need multiple channels?
4. **Single-best-univariate-threshold control on the pneumatic track.** A one-feature rule reaches
   F1 = 1.0 on MetroPT-2 [R93]. The plan named LightGBM-residual as the "expected MetroPT winner"
   without a control that can beat it.
5. **Compressor duty-cycle ratio** (`t_off / t_loaded`) as a first-class physically motivated leak
   feature and univariate detector, with 1-4 week lead time on fleet data from compressor on/off logs
   alone [R101].
6. **Half-Space Trees + One-Class kNN** streaming detector — highest value-per-hour item in the
   rail-pneumatic literature, online, CPU-only, large precision gain [R99].
7. **k-of-n multi-signal corroboration voting** (~20 lines) as a false-alarm suppressor on every
   detector [R95].
8. **Adaptive conformal threshold wrapper** — training-free, drift-aware, validation-only calibration
   [R42]. Replaces a bare quantile as the defensible threshold story.
9. **Door DWT features** (db10 detail-coefficient L1-norms at W8/W9 + starting-current peak, >96 %
   accuracy and <0.3 s in the published rig) and **three-velocity-regime segmentation** of the
   current trace [R66][R64]. The plan's `cycle_features` list had neither.
10. **Door DTW-to-reference degradation index + k-means severity staging** as the RUL arm that needs
    no run-to-failure labels, from motor current alone [R72][R81].
11. **Bearing kurtogram (spectral kurtosis) band selection** ahead of envelope analysis, with the
    selected band emitted as an auditable feature [R117]; and **MFCC + amplitude-modulation-spectrogram
    features with healthy-only OC-SVM**, verified on real commuter-railway bearings [R120][R121].
12. **Bearing-wise / recording-level splits**, reporting the window-random vs bearing-wise gap
    explicitly [R115][R58]. The plan had GroupKFold by bearing — correct, now with a citation and an
    explicit honesty metric attached.

**Corrected (the plan had it, the evidence changes it)**

13. **Foundation-model slot re-cast.** Plan: "Chronos-Bolt-small MUST, MOMENT-1-small NICE". New:
    **TSPulse MUST** (purpose-built for AD, #1 on TSB-AD, 1M params, Apache-2.0, CPU-capable [R30]);
    Chronos-2 NICE and only as a covariate-informed forecaster [R24]; MOMENT NICE **for the bearing
    track only** (its verified standing is univariate) and **zero-shot only**, since fine-tuning adds
    +0.007 VUS-PR [R1][R19].
14. **Residual scoring → change-point scoring** for persistent faults. Forecast error stays low
    *during* long anomalies and peaks at their boundaries [R36]; MetroPT leaks are exactly that shape.
    Score onsets, then fill the episode with hysteresis. ~1 h, affects every forecast-residual arm.
15. **Classification headline changed from MiniRocket/ResNet1D to MultiRocket + Hydra + QUANT +
    LITETime.** MultiRocket+Hydra is in the top clique with HC2 [R43][R47][R48]; QUANT gets
    interval-level interpretability for the UI at <15 min on one core [R49]; LITETime replaces
    ResNet1D/InceptionTime at 2.34 % of InceptionTime's parameters [R53]. All four are one `aeon`
    call each [R62].
16. **VUS-PR promoted from secondary to primary**, VUS-ROC demoted out of ranking, affiliation
    explicitly secondary, single runs with fixed stated seeds [R7][R1][R10]. The plan listed VUS-PR
    under "secondary threshold-free".
17. **Cost columns added to every leaderboard row** — latency, peak VRAM, serialized size — following
    the cost-aware industrial template [R37]. Plan had fit/score seconds and RAM; VRAM and model size
    are added because a foundation model losing on cost *and* accuracy is a result worth showing.
18. **`aeon` scope widened.** Plan: "`aeon` only for MiniRocket/MultiRocket (no TF-based
    classifiers)". aeon is BSD-3-Clause, sklearn-compatible on `(n_cases, n_channels, n_timepoints)`
    arrays, and the newest classifiers (QUANT, Hydra, WEASEL 2.0, LITETime) land there first [R62].
    Use it for the whole classification bank, and use
    `load_classification_bake_off_2023_results()` + `plot_critical_difference()` to emit a **critical
    difference diagram** rather than bar charts of a single split — a 1 h win and the cheapest way to
    look methodologically serious [R63][R43].

**Confirmed unchanged**

- No point adjustment, event/episode metrics, temporal splits, thresholds from validation only —
  all four now carry primary citations [R2][R1][R3][R9][R8].
- The plan's SKIP reasoning for Anomaly Transformer / TimesNet-AD ("point-adjustment") and for
  TimesFM / Time-MoE ("too heavy") is confirmed, and strengthened: TimesFM also has a *direct
  negative result* for multivariate AD [R36], and Time-MoE is a One-Liners failure case [R35].
- LSTM-AE, TCN-AE and USAD stay MUST: USAD is the only deep detector in both TSB-AD top-10s [R16],
  and TCN-AE is the specific model that beat a foundation model on cost-aware industrial AD [R37].
- LightGBM residual stays MUST — but must now clear two controls before it can be called a winner.
- "Deep models that time out stay in the table as `status=timeout`" is confirmed as defensible by
  [R1] and [R5].

**New candidate dataset the plan should absorb:** the PHME 2026 subway-door servomotor run-to-failure
set [R69] — real position/velocity/current/voltage/hall/encoder traces from an electric door drive,
48 run-to-failure training experiments across 3 operating conditions, with two published baseline
solutions to match [R70][R71]. See `datasets.md`.

---

### Added in the W1 pass (14 Sep)

19. **A change-point / concept-drift family now exists** (§1d, §2c, §3c). The W0 ladder had made
    change-point scoring the central design decision in V4 but the only change-point machinery
    anywhere was a hand-written CUSUM on a scalar — no offline segmentation, no Bayesian online CPD,
    no streaming drift detector, not even a SKIP row with a reason. Now: `ruptures` Pelt/BinSeg
    (BSD-2) [R173], `changeforest` (BSD, JMLR 2023, purpose-built multivariate — **MUST on the
    pneumatic track** because MetroPT-3 is 15-channel) [R187], `claspy` ClaSP (BSD-3,
    **hyper-parameter-free**) [R176][R177], `river` Page-Hinkley / ADWIN / KSWIN (**zero new
    dependencies** — `river` is already imported for Half-Space Trees) [R178][R179][R180][R181], and
    `aeon`'s `RandomSegmenter` as the mandatory random control [R188]. The hand-written CUSUM is
    **demoted to a baseline** that `river.drift.PageHinkley` must match — not discarded, since it has
    a 1954 Biometrika pedigree [R180]. **Reasoned SKIPs now exist** where none did: DDM/EDDM/FHDDM/HDDM
    consume a **binary error stream** and need a classifier underneath, so they cannot act as
    unsupervised triggers [R182][R183]; and `bayesian-changepoint-detection` has **no licence declared
    on PyPI** and is unmaintained since 2019 [R186].
20. **The cheap change-point method is the MUST and the expensive one is NICE — on evidence.** Under
    **default** hyperparameters binary segmentation had the highest average performance across 14
    algorithms on 37 annotated real series; BOCPD only won under **oracle** tuning [R175]. Our
    validation-only-threshold rule puts us permanently in the default column, so this is a rule-driven
    choice, not a budget-driven one.
21. **A change-point evaluation method now exists** (§6.6), which V4 completely lacked. The covering
    metric and F1-at-annotation-margin come from the MIT-licensed TCPDBench [R175]; **affiliation
    P/R is promoted to the cross-family bridging column** because it is defined by temporal distance
    between event sets and therefore scores discrete CPD output and discrete AD output on identical
    footing with no adapter [R8]. **The one real cost is named and budgeted**: every CPD row emits
    discrete breakpoints or booleans while VUS-PR needs a continuous per-timestamp score [R7], so a
    2 h shared adapter is now an explicit line item. We also honour [R184]'s warning that localisation
    accuracy is not the same objective as downstream utility, by never reporting F1-at-margin alone.
22. **The RUL ladder is now symmetric across all three subsystems** (§2d, §3d). The W0 yaml had
    `task: rul` rows for the **door only**, while `datasets.md` justified ingesting XJTU-SY [R124],
    FEMTO [R143] and the Paderborn run-to-failure set [R138] for a prognostics capability nothing
    delivered. Added: **General Path Model** [R199] (needs only a monotonic index and a threshold —
    **no run-to-failure labels**, so it ports to pneumatic and bearing for free and plugs straight into
    the ordinal severity staging §4.4 of `rail_phm.md` already specifies), **gamma / Wiener /
    inverse-Gaussian degradation processes** [R197][R196][R198], **similarity-based RUL** [R200]
    (the published ancestor of our existing door row, and what the trajectory libraries are *for*),
    **particle-filter state tracking with uncertainty propagation** [R201], and **health-indicator
    suitability screening before any fit** [R218]. **The three run-to-failure bearing datasets now
    earn their ingest.**
23. **Survival analysis fills the pneumatic RUL gap, where run-to-failure data does not exist.** Cox
    proportional hazards and Weibull AFT need only **censored** duration + event + covariates
    [R205][R211], which makes SCANIA Component X earn its place in `datasets.md` [R134][R209];
    `xgbse` (Apache-2.0) returns calibrated survival curves that map directly onto "probability of
    reaching severity level k within h days" in the twin UI [R213]; and `SurvLoss` turns censored
    samples into training signal with a ~30-line loss change [R208], supported by the finding that
    including non-failed assets improves fleet prediction [R210]. **Licence note added**:
    `scikit-survival` is GPL-3.0-or-later [R212] and ProgPy is NASA-1.3 [R201] — neither is
    OSI-permissive, so both are flagged before they can enter the deliverable.
24. **Deep RUL is now a reasoned SKIP, and RUL evaluation gets its own protocol** (§6.7). V7's
    generalization bound `O(B²√(p/n))` in the number of **run-to-failure trajectories** puts `n` ≈ 15
    (XJTU-SY) far outside a deep net's sample-efficient regime, while embedding degradation physics
    can cut data requirements by up to two orders of magnitude [R221]; and on C-MAPSS FD003 —
    which has *more* data than we do — **XGBoost beats both a 1D CNN and an LSTM** [R222]. Protocol
    additions: α-λ accuracy / prognostic horizon / cumulative relative accuracy / monotonicity
    [R215][R216][R201]; the C-MAPSS asymmetric score **reported beside RMSE and a per-unit error
    distribution, never alone**, because it is unnormalised and one late unit dominates [R217]; and
    **leave-one-bearing-out**, the prognostics twin of our bearing-wise rule, because naive splitting
    inflates accuracy from a genuine 20-60 % to 99.9 % on the same data [R220][R219].

**What did NOT change in the W1 pass, and why that matters.** The dataset choice for the bearing
track is unchanged: we looked specifically for a bearing set with a **documented thermal model or an
ambient log**, and none exists under a usable licence. FEMTO/PRONOSTIA carries temperature alongside
vibration [R143] and is already an ADD; the PoliTo set is the best sensor match (vibration +
temperature + speed + axial load on large bearings) but is under a **Restricted Use Agreement**
forbidding redistribution [R139]; and the Ottawa variable-speed set we promoted to primary states
**no defect size, no applied load and carries no temperature channel** [R159], so it cannot ground a
severity axis or a thermal model — but it *can* be used to fit our vibration-versus-speed exponents
directly, which is a better justification than any literature value (see `rail_phm.md` §4.3.3).

## Scope decision (14 Sep 2026)

The team trimmed the ladder to the anomaly-detection and classification families as MUST. Of the change-point family only `page_hinkley_*`, `adwin_*` and the pneumatic `transition_mask` stay MUST (they share the residual stream and cost about 1 h). Every RUL, survival and remaining change-point row is demoted to NICE in `configs/model_ladder.yaml` (field `demoted_by`). The physics corrections and evaluation protocol in this document are kept in full.
