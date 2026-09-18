# Finder 1 — Rail corrugation detection from axle-box acceleration (PS3 addendum)

**Task framing.** 272 labelled files × 1 s × 10 kHz × 64 channels (8 cars × 8 positions; odd =
Side I, even = Side II), plus a 90-tooth tacho, wheel Ø 0.85 m, 0–67 km/h. Three classes,
**234 Normal / 14 Side I / 24 Side II**, scored by **macro F1**. So the minority class has 14
examples: with 5-fold stratified CV that is ~2.8 per fold, and macro F1 is dominated by it.
Budget ~12 h.

**Nothing in `docs/research/` covers this.** `rail_phm.md` §3 is the axle *box bearing*
(thermal + envelope demodulation); `datasets.md` §3 is bearing datasets. Rail *corrugation*
from ABA — the track-side use of the same sensor — appears nowhere, and none of R109–R123,
R138–R146 are about track condition. `model_ladder.md` §1b (MultiRocket/Hydra/QUANT/LITETime,
[R43][R47][R48][R59]) and the recording-level-CV warning in §3.4 [R58] *do* transfer and should
be cited rather than re-found; this sweep only adds what is ABA/corrugation-specific.

---

## Physics that fixes the feature design (derive once, before any model)

| Quantity | Value | Consequence |
|---|---|---|
| Wheel circumference | π × 0.85 = **2.670 m** | 90 teeth → **29.67 mm of travel per tacho pulse** |
| Pulse count over 1 s at 67 km/h | 18.61 m / 29.67 mm = **627 pulses** (wheel 6.97 rev/s) | cumulative pulse count *is* a distance encoder; integrate it, do not differentiate it |
| Spatial sampling at 10 kHz, 67 km/h | **1.86 mm** → Nyquist wavelength 3.7 mm | 10 kHz is comfortably enough for short-pitch corrugation at every speed in range |
| Short-pitch metro corrugation | **λ = 25–80 mm** (30–60 mm most common on metro tangent and curved track) | this is the target band |
| Passing frequency f = v/λ | 25 mm: **744 Hz @ 67 km/h → 111 Hz @ 10 km/h**; 80 mm: 233 Hz → 35 Hz | **a fixed Hz band cannot work.** Over 0–67 km/h the corrugation band sweeps 35–744 Hz and overlaps everything else |
| ABA amplitude vs speed | a ∝ (2πf)²·r(λ) ⇒ **∝ v²** for fixed roughness | normalise by v² (published practice, Yu et al. 2025 / Carrigan & Talbot 2023) or the classifier learns speed, not corrugation |

Two structural facts specific to *this* dataset, which no paper will give you:

1. **The label is a side, so the feature must be a side contrast.** Reduce 64 channels to two
   side-aggregates (32 odd, 32 even) and feed the classifier the per-band **difference/ratio**
   Side I − Side II alongside the pooled level. This removes speed, track type and car-to-car
   sensor gain as nuisance variables in one step, because they are common-mode.
2. **Mirror augmentation is free and exact.** Swapping the odd↔even channel index map maps a
   Side I record to a valid Side II record. That turns 14/24 into 38/38 and imposes the
   left-right equivariance the physics demands. This is the highest-value 30 lines in the whole
   task and costs nothing in label noise. (Not from a paper — it follows from the sensor layout.)

---

## Practitioner defaults — what a rail/PHM practitioner builds first with 12 h

Resample every channel from time to **distance** using the cumulative tacho pulse count
(constant Δx ≈ 1 mm, linear interpolation of pulse edges — this is computed order tracking with
a real keyphasor, ~60 lines of numpy, no library), then take a Welch PSD **in the wavelength
domain** and integrate it into **1/3-octave wavelength bands** over roughly 8–500 mm, the
EN 15610 presentation that every rail engineer already reads. Normalise by v² (or add log v as a
covariate). That single transform converts a 64 × 10 000 float array into ~20 band levels per
channel that are *speed-invariant by construction*, which is the entire difficulty of the
problem. Aggregate to per-side band levels (median over the 32 channels of a side, plus a
90th-percentile to keep localised sections) and form the Side I − Side II contrast in dB.
Then: **logistic regression / LightGBM on ~60 features** as the honest baseline, and
**MultiRocket + RidgeClassifierCV on the distance-resampled, v²-normalised, side-aggregated
2-channel signal** (`aeon`, already in `model_ladder.md` §1b) as the strong arm. Decompose the
3-class problem into (a) corrugation present/absent and (b) which side, because 38 positives
support one binary decision far better than three. Handle imbalance with mirror augmentation +
class weights, not SMOTE on spectra. Evaluate with **repeated stratified CV grouped by
recording/site**, never segment-level splits — the inflation warning in `model_ladder.md` §3.4
[R58] applies with full force at n=14. Expect the deep-learning literature's 95–98 % numbers to
be unreachable here: those are 10³–10⁴-sample field campaigns or simulations, and several
explicitly train on simulated data and fine-tune. Budget: 4 h resampling + wavelength PSD, 2 h
feature/side-contrast, 3 h models + CV, 3 h ablations (with/without distance resampling,
with/without v² normalisation) — the ablations *are* the result worth showing.

---

## Ranked candidates

Ranked by relevance × feasibility. `subsystem_relevance` 0–3; `implement_hours` = hours to get
the *usable idea* into this pipeline, not to reproduce the paper.

### 1. Faccini, Karaki, Di Gialleonardo, Somaschini, Bocciolone, Collina — *A Methodology for Continuous Monitoring of Rail Corrugation on Subway Lines Based on Axlebox Acceleration Measurements*
- venue/year: **Applied Sciences** 13(6):3773, **2023** · doi:10.3390/app13063773 · **CC BY 4.0**
- task: continuous corrugation monitoring on a **subway line** from in-service ABA
- code: none · dataset: proprietary metro line · reported_metric: agreement with trolley
  roughness measurements (exact values `unverified` — MDPI blocked WebFetch with 403; abstract
  and Crossref metadata verified)
- relevance **3** · implement_hours **3**
- preserves/assumes: assumes a tacho or GPS for position; assumes corrugation lives in a
  bounded wavelength band and that a band-limited RMS index tracks its severity
- notes: **the closest published setting to ours** — metro, in-service, axlebox, continuous. The
  reference design for the band-limited RMS index and the positioning/alignment step. Read this
  first even though only the abstract is machine-readable.

### 2. Lian, Zhang, Gao, Liu — *A feature extraction framework for metro rail corrugation detection using onboard vibration and noise monitoring data*
- venue/year: **Intelligent Transportation Infrastructure** (Oxford) 4, **2025** ·
  doi:10.1093/iti/liaf010 · **CC BY**
- task: binary corrugation / non-corrugation from onboard MEMS vibration + noise
- code: none · dataset: 639 samples (352 non-corrugation / 287 corrugation), metro, roughness
  ground truth
- reported_metric: **BPNN 95.31 % acc / 96.17 % F1**; SVM 94.79/95.73; LSTM 94.27/95.40;
  RF 93.75/94.83 (verified from the OA full text)
- relevance **3** · implement_hours **3**
- preserves/assumes: 26 hand features — 14 time-domain (RMS, crest, kurtosis, skewness, margin,
  pulse, waveform factors…), 6 frequency-domain (peak frequency, spectral centroid, spectral
  energy SD, total energy), 6 wavelet-band energies from L3–L5 (**1000–2000, 500–1000,
  250–500 Hz**). Assumes roughly constant metro speed — the fixed Hz wavelet bands are exactly
  what breaks over our 0–67 km/h range.
- notes: **steal the feature list, replace its fixed Hz bands with wavelength bands.** Also the
  single most useful data point in this sweep: on a balanced 639-sample metro problem, four very
  different classifiers land within 1.6 points of each other — i.e. **the representation, not
  the classifier, is where the accuracy is**. Directly justifies spending 6 of 12 h on the
  distance/wavelength transform and 3 on models.

### 3. Liu, Wu, Chi, Wen, He — *Determination of rail corrugation maintenance limit based on axle box acceleration spectrum defined in IEC 61373*
- venue/year: **Vehicle System Dynamics** 61(11):2936–2952, **2023** (online 2022) ·
  doi:10.1080/00423114.2022.2151920 · paywalled
- task: map ABA spectrum → corrugation severity → maintenance limit
- code: none · reported_metric: threshold values `unverified` (paywall)
- relevance **3** · implement_hours **2**
- preserves/assumes: works in a **standardised 1/3-octave ASD band structure (IEC 61373)** and
  assumes the corrugation contribution is separable within it
- notes: the octave-band framing is the reusable part — it gives a defensible, non-arbitrary
  band layout and a vocabulary ("maintenance limit") that reads as rail engineering rather than
  Kaggle. Cite for the band choice even if the numbers stay unverified.

### 4. Hassanieh, Chehade, Facchinetti, Carman, Bocciolone, Somaschini — *Leveraging machine learning to predict rail corrugation level from axle-box acceleration measurements on commercial vehicles*
- venue/year: **International Journal of Rail Transportation** 12(4):604–625, **2024**
  (online 2023) · doi:10.1080/23248378.2023.2220112 · paywalled (T&F 403)
- task: **regression** of corrugation level from ABA on in-service commercial vehicles
- code: none · reported_metric: R² / MAE `unverified` (paywall)
- relevance **3** · implement_hours **3**
- preserves/assumes: tuned **Random Forest** on accelerometer features **plus static/offline
  covariates** (speed, curve radius, track type) — i.e. it treats speed as an explicit input
  rather than normalising it away
- notes: the **counter-design to our default**: instead of removing speed by resampling, give it
  to the model. Cheap and worth running as an ablation arm (tree model on Hz-band features +
  speed). Same Politecnico di Milano group as #1.

### 5. De Rosa, Luber, Müller, Fuchs — *Methodology to Detect Rail Corrugation from Vehicle On-Board Measurements by Isolating Effects from Other Sources of Excitation*
- venue/year: **Applied Sciences** 14(19):8920, **2024** · doi:10.3390/app14198920 · **CC BY 4.0**
- task: corrugation detection from in-service ABA while **rejecting confounders**
- code: none · dataset: in-service high-speed vehicle · reported_metric: qualitative /
  `unverified` (MDPI 403)
- relevance **3** · implement_hours **2**
- preserves/assumes: other excitations — **structural modes and resonances, bridges, switches
  and crossings, wheel defects** — occupy the *same wavelength range* as corrugation and must be
  identified by their own signatures
- notes: **this is the paper that explains our false positives.** A wheel flat or a switch
  produces a broadband burst in the same band; corrugation is *sustained and periodic*. Encode
  that difference directly: add spectral flatness / harmonic-peak prominence / duty-cycle of the
  band-passed envelope, so an impulsive event does not score as corrugation. Also the argument
  for why *cross-channel agreement within a side* (§Physics point 1) is diagnostic: a wheel
  defect is one channel, corrugation is a side.

### 6. Li, S. et al. — *Monitoring of rail short pitch corrugation using the time-frequency features of both vertical and longitudinal axle box accelerations*
- venue/year: **Measurement** 255:118064, **2025** · doi:10.1016/j.measurement.2025.118064 ·
  paywalled
- task: short-pitch (25–80 mm) corrugation monitoring from ABA
- code: none · reported_metric: `unverified` (paywall)
- relevance **3** · implement_hours **3**
- preserves/assumes: **two axes carry complementary information** — longitudinal ABA responds to
  the creep/stick-slip mechanism, vertical to the contact geometry
- notes: our dataset has *vertical + shock* channels rather than vertical + longitudinal, but the
  transferable claim is the same: **do not collapse the channel types**; keep per-axis band
  levels and let the model weight them. Cheap to honour, plausible accuracy gain.

### 7. Yu, X. et al. — *Experimental study of the use of a transfer function to find rail corrugation from axle-box accelerations*
- venue/year: **Measurement** 249:117058, **2025** · doi:10.1016/j.measurement.2025.117058 ·
  paywalled
- task: invert ABA → corrugation amplitude via a calibrated roughness→acceleration transfer
  function, on a scaled vehicle–track rig
- code: none · reported_metric: `unverified` (paywall)
- relevance **3** · implement_hours **2** (only the normalisation, not the inversion)
- preserves/assumes: **normalise measured acceleration by the square of forward velocity** so the
  transfer function becomes speed-independent; assumes speed roughly constant within a record
  (true for us — 1 s windows) and a calibratable track transfer function (not true for us)
- notes: **the citation for the v² normalisation**, which is the single highest-leverage line of
  code in this task. Take the normalisation, skip the inversion (needs a reference track).

### 8. Carrigan & Talbot — *A new method to derive rail roughness from axle-box vibration accounting for track stiffness variations and wheel-to-wheel coupling*
- venue/year: **Mechanical Systems and Signal Processing** 192:110232, **2023** ·
  doi:10.1016/j.ymssp.2023.110232 · paywalled. Companion: *Use of Flexible Wheelset Model, Comb
  Filter and Track Identification…*, LNME (IWRN 14), 2024, doi:10.1007/978-981-99-7852-6_25
- task: rail roughness spectra from ABA **in the presence of wheel roughness**
- code: none · reported_metric: 1/3-octave band deviation **< 1 dB** over λ = 5 mm–0.5 m for
  known track dynamics; **rail-pad stiffness is the dominant sensitivity — a 20 % deviation moves
  estimated roughness by up to 3.5 dB** (from abstracts/search snippets; `unverified` at full-text
  level)
- relevance **2** · implement_hours **4** (comb filter only; full inversion is out of budget)
- preserves/assumes: the wheel and the rail both contribute roughness; a **comb filter keyed to
  the wheel circumference** separates the wheel-fixed component from the track-fixed one
- notes: the rigorous answer to "is this the rail or the wheel?". The comb filter (notch at
  multiples of 1/2.670 m⁻¹) is implementable in ~40 lines on the distance-resampled signal and
  is a genuinely differentiating ablation — but it is optional at 12 h, and the side-contrast
  feature gets most of the benefit for a tenth of the work.

### 9. Pieringer & Kropp — *Model-based estimation of rail roughness from axle box acceleration*
- venue/year: **Applied Acoustics** 193:108760, **2022** · doi:10.1016/j.apacoust.2022.108760 ·
  paywalled
- task: frequency-domain model-based inversion ABA → roughness wavelength spectrum
- code: none · reported_metric: accuracy over λ = 5 mm–0.5 m, values `unverified`
- relevance **2** · implement_hours **4** (reference only; do not implement)
- preserves/assumes: known/modelled track receptance and contact filter; compensates explicitly
  for vehicle speed and track dynamics
- notes: the canonical statement of **why the PSD must be translated into the wavelength domain
  before anything else**. Cite as the theoretical backing for the distance-resampling step; the
  inversion itself needs track parameters we do not have.

### 10. Haghbin, Chiachío, Muñoz, Escalona, Guillén, Crespo Marquez, Cantero-Chinchilla — *Predicting Rail Corrugation Based on Convolutional Neural Networks Using Vehicle's Acceleration Measurements*
- venue/year: **Sensors** 24(14):4627, **2024** · doi:10.3390/s24144627 · **CC BY 4.0**
- task: corrugation prediction from vehicle acceleration **+ speed** with a **1D-CNN**
- code: none stated · dataset: scaled railway test rig, multiple speeds
- reported_metric: **> 95 % accuracy across speeds** (from the Crossref abstract; per-class
  values `unverified`)
- relevance **3** · implement_hours **4**
- preserves/assumes: raw/near-raw 1D windows, **speed supplied as an explicit auxiliary input**;
  saliency-style visualisation to localise the damaged zone
- notes: the cheapest deep arm that is actually on-task, and it is CC BY so the figures are
  quotable. Note the caveat to state honestly: **scaled rig, not a metro line** — do not let the
  95 % set expectations for our 14-sample class.

### 11. Amin, Najeh, Ghoul — *AI-driven vibration-based event classification in railway switches and crossings*
- venue/year: **Scientific Reports** 16, **2026** · doi:10.1038/s41598-026-58967-0 · **CC BY 4.0**
- task: multi-class event classification from railway vibration under **small, imbalanced** data
- code/data: availability statement `unverified` (Nature IDP redirect blocked WebFetch)
- reported_metric: **81.5 % held-out accuracy, ROC-AUC ≈ 0.94**, best with ensembles, over
  **21 classifiers** benchmarked; autoencoder-based synthetic augmentation; **feature
  standardisation was decisive — without it neural nets fell below chance**
- relevance **2** · implement_hours **2**
- preserves/assumes: strict partitioning to avoid leakage; macro-F1 as the imbalance-aware metric
- notes: the **realism anchor**. A 2026 Scientific Reports paper on railway vibration with a
  proper held-out split reports 81.5 %, not 98 %. Use it to set the expected operating point and
  to justify (a) ensembles over a single deep model, (b) standardising features, (c) reporting
  macro F1 with CV spread rather than a point estimate.

### 12. Samani, Núñez, De Schutter — *WaveletInception Networks for on-board Vibration-Based Infrastructure Health Monitoring*
- venue/year: **arXiv:2507.12969**, 2025 (rev. Jan 2026); under review at *Engineering
  Applications of AI* — venue `unverified`
- task: on-board vibration → track stiffness regression and transition-zone classification
- code: none found · reported_metric: "significantly outperforms state of the art"; numbers
  `unverified` (not in the abstract)
- relevance **2** · implement_hours **8**
- preserves/assumes: **learnable wavelet packet transform** front end + 1D Inception-ResNet +
  **BiGRU that ingests measurement speed as an operating condition**, explicitly to *avoid*
  hand-built preprocessing of speed
- notes: the most modern statement of the alternative philosophy — **condition on speed instead
  of normalising it**. At n=272 a learnable front end will overfit, so this is a *nice-to-have
  citation for the related-work slide*, or a 2-line borrowing: concatenate log-speed to the
  feature vector of whatever model wins.

### 13. Wang, Xiao, Ma, Zhang, Cui, Xu — *On-board detection of rail corrugation using improved convolutional block attention mechanism* (TBVA-Net)
- venue/year: **Engineering Applications of Artificial Intelligence** 146:110349, **2025** ·
  doi:10.1016/j.engappai.2025.110349 · paywalled
- task: corrugation detection from **car-body** vertical acceleration (not axlebox)
- code: none · dataset: simulated + limited field labels
- reported_metric: **test accuracy > 95 %, mean 98.6 % on the simulated set; 98.5 % transfer
  accuracy after fine-tuning on a small labelled field subset**
- relevance **2** · implement_hours **6**
- preserves/assumes: 1D residual CNN + channel/spatial attention; **pre-train on simulation,
  fine-tune on few field labels**
- notes: two things to take, neither of them the architecture. (a) The **sim-pretrain →
  few-label fine-tune** recipe is the textbook answer to 14 labelled examples — but we have no
  simulator for this and building one is not a 12 h job, so it is a *skipped-with-reason*
  candidate worth naming explicitly. (b) Car-body sensing is a *weaker* signal than our axlebox
  data, so their numbers are an upper bound achieved with far more data, not a target.

### 14. Wang, Xiao, Nadakatti, Zhang, Chi, Liu — *A metro rail corrugation detection framework based on car body vibration signals and unsupervised learning*
- venue/year: **Engineering Applications of Artificial Intelligence** 153:110976, **2025** ·
  doi:10.1016/j.engappai.2025.110976 · paywalled *(the search index also exposes PII
  S0952197625009765 for this title — treat the DOI as Crossref-verified and the PII as
  `unverified`)*
- task: self-supervised pre-training then fine-tune for corrugation **wavelength classification
  + amplitude assessment**
- code: none
- reported_metric: wavelength classification **95–100 %**, amplitude assessment **> 95 %**
  (from abstract; `unverified` at full-text level)
- relevance **2** · implement_hours **10** → **skip**
- preserves/assumes: **synchrosqueezed wave-packet transform** spectrograms + **momentum
  contrastive (MoCo) pre-training on unlabelled data**, then a small labelled fine-tune
- notes: exactly the right *shape* of solution for 14 labels — pre-train on the 234 Normal
  records without labels, fine-tune on the 38 positives. But MoCo on spectrograms is a full day
  on its own and the 234 unlabelled records are far too few for contrastive pre-training to pay.
  **Recommend skipping with this reason stated**; it is the strongest "what we would do with
  more time" line in the write-up.

### 15. Vold–Kalman order tracking of axle-box accelerations for railway stiffness assessment
- venue/year: **arXiv:2209.12899**, 2022 · venue/journal version `unverified`
- task: order-tracked ABA for track property assessment under varying speed
- code: none found · reported_metric: `unverified`
- relevance **2** · implement_hours **4** (Vold–Kalman) / **1** (plain resampling)
- preserves/assumes: Vold–Kalman filtering extracts smooth order components under speed
  variation without the spectral smearing that a time-domain STFT suffers
- notes: names the problem correctly — **speed variation smears the spectrum, and order tracking
  is the fix**. For us the cheap version (tacho-driven resampling to constant Δx) captures the
  benefit; Vold–Kalman is over-engineering at this budget. Cite alongside
  `model_ladder.md` §3 R118/R122 (the tacho vs tacholess ablation the team already planned for
  the bearing) — **the same ablation structure works here and reuses code**.

### 16. Jahan, Lähns, Baasch, Heusel, Roth — *Rail Surface Defect Detection and Severity Analysis Using CNNs on Camera and Axle Box Acceleration Data*
- venue/year: **LNME**, Int. Congress & Workshop on Industrial AI and eMaintenance 2023, pp.
  423–435, **2024** · doi:10.1007/978-3-031-39619-9_31 · OA copy at
  https://elib.dlr.de/201722/ (PDF did not parse — contents `unverified`)
- task: joint camera + ABA CNN for **squat and corrugation**, with severity grading
- code: none · dataset: time-synchronised ABA + camera with labelled defect instances (DLR)
- reported_metric: `unverified`
- relevance **2** · implement_hours **3**
- preserves/assumes: corrugation and squat are the two dominant ABA-visible surface defects and
  are confusable; camera gives the labels
- notes: useful mainly as the DLR-side confirmation that **corrugation vs impulsive squat is the
  hard confusion**, reinforcing #5. Also a lead on a labelled ABA source if the team ever wants
  external data. Fetch the OA PDF by hand — the automated parse failed.

### 17. Yang, Huo, Yao — *Rail corrugation detection based on optimal position window and weighted-bandwidth mode decomposition*
- venue/year: **Measurement** 255:117888, **2025** · doi:10.1016/j.measurement.2025.117888 ·
  paywalled (preprint doi:10.2139/ssrn.5009696)
- task: adaptive mode decomposition to isolate the corrugation component from ABA
- code: none · reported_metric: `unverified`
- relevance **1** · implement_hours **8** → **skip**
- preserves/assumes: the corrugation component is a narrowband mode recoverable by adaptive
  bandwidth selection; a position window localises where to look
- notes: representative of a large VMD/CEEMDAN/EWT/SPWVD literature on this exact problem (see
  also *Identification Method for Railway Rail Corrugation Utilizing CEEMDAN-PE-SPWVD*, Sensors
  2024, PMC11680013, and *Identification of rail corrugation in high-speed railway using
  VMD-SPWVD*, KSCE J. Civ. Eng. 2025). **Recommend skipping the whole family with reason**:
  these are per-record adaptive decompositions with parameters to tune, they are slow, and on a
  272-file classification task with macro F1 they buy nothing that a wavelength-band PSD does
  not already give. Naming the family and saying why you skipped it is worth more than
  implementing one.

### 18. Wang, Huang, Wang, Ni, Ran, Li, Zhang — *Concise Historic Overview of Rail Corrugation Studies: From Formation Mechanisms to Detection Methods*
- venue/year: **Buildings** 14(4):968, **2024** · doi:10.3390/buildings14040968 · **CC BY 4.0**
- task: review — formation mechanisms through to detection
- code: n/a · reported_metric: n/a
- relevance **2** · implement_hours **1**
- notes: the one-stop citation for the **wavelength taxonomy** (short-pitch 25–80 mm, long-pitch,
  roaring rails) and for the mechanism sentence in the slide deck. Open access, so quotable.
  Pair with *Formation mechanism of short-pitch rail corrugation on metro tangent tracks with
  resilient fasteners*, Vehicle System Dynamics 61(6), 2023, doi:10.1080/00423114.2022.2086143,
  for the metro-specific mechanism and the 30–60 mm figure.

### 19. Dissanayake, McPherson, Allyndree, Kennedy, Cunningham, Riaboff — *Evaluating ROCKET and Catch22 features for calf behaviour classification from accelerometer data*
- venue/year: **arXiv:2404.18159**, 2024 (journal version stated but unnamed — `unverified`)
- task: multi-class classification of short accelerometer windows (3 s), off-domain
- code: none stated · dataset: 27.4 h annotated, 6 classes, imbalanced
- reported_metric: balanced accuracy **ROCKET 0.70 ± 0.07 > catch22 0.69 ± 0.05 > hand-crafted
  0.65 ± 0.03**; best combination **ROCKET + RidgeClassifierCV = 0.77**
- relevance **2** · implement_hours **1**
- preserves/assumes: raw short accelerometer windows, no domain preprocessing
- notes: not rail, but the **cleanest head-to-head evidence that ROCKET + RidgeClassifierCV is
  the right default on short accelerometer windows**, and that hand-crafted features trail it by
  ~5 points. Supports running `aeon`'s MultiRocket arm ([R43][R47] in `references.md`) *on the
  distance-resampled signal* rather than instead of the physics. Also a warning: absolute
  numbers on small imbalanced accelerometer problems sit near 0.7, not 0.95.

### 20. EN 15610 (2019+A1:2025) *Railway acoustics — rail and wheel roughness measurement related to noise generation*, and EN ISO 3095 limit spectrum
- venue/year: **CEN standards**, 2019/2025 · standard text **not read — `unverified`**
- task: the standard presentation of railhead roughness as a **1/3-octave band wavelength
  spectrum**, with a defined upper-limit spectrum
- code: n/a · licence: purchase only
- relevance **2** · implement_hours **1**
- notes: the reason to present features as 1/3-octave **wavelength** bands rather than an
  arbitrary FFT binning — it is what the domain already uses, and reprofiling literature reports
  against it (grinding is effective over 30–500 mm, and can *leave* content below 30 mm).
  Same treatment as EN 15437 in `references.md` R113/R114: **cite by number, do not paraphrase
  clauses.**

---

## Skipped deliberately, with reasons (for the addendum's third ladder)

| Candidate family | Why skipped |
|---|---|
| VMD / CEEMDAN / EWT / SPWVD adaptive decompositions (#17 and relatives) | Per-record, parameter-heavy, slow; no evidence they beat a wavelength-band PSD for a 3-class macro-F1 task at n=272. |
| Full model-based roughness inversion (#8, #9) | Needs track receptance / rail-pad stiffness we do not have; the paper's own sensitivity analysis says a 20 % pad-stiffness error moves the answer 3.5 dB. |
| Self-supervised / contrastive pre-training (#14) | Right shape for 14 labels, but 234 unlabelled records are far too few for MoCo to pay, and it is a full day. |
| Sim-pretrain → fine-tune (#13) | No corrugation simulator exists in the repo and building a vehicle–track model is not a 12 h task. |
| GAN / diffusion augmentation for imbalance | The **mirror (side-swap) augmentation is exact and free**; generative augmentation on 14 spectra is strictly worse and adds a failure mode. |
| Camera/vision rail defect models | No image channel in this dataset. |

## Verification status

Crossref-verified metadata (title, venue, year, DOI, authors, volume/pages): #1–#5, #6, #7, #8,
#9, #10, #11, #13, #14, #17, #18. Full text read only for **#2** (OA). Abstract-level only for
the rest. Marked `unverified` above: all Elsevier/T&F reported metrics (403 on ScienceDirect,
tandfonline, MDPI HTML and PMC), the Nature data-availability statement, the #16 PDF contents
(binary parse failed), #12 and #15 venue status, #19's journal version, and the EN 15610 /
IEC 61373 / ISO 3095 clause text (standards not purchased).
