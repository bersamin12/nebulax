# NEBULA X — Problem Statement 3

## Scope and data

We attempt all four condition-monitoring subsystems with one offline-capable web app. The app
uses the same prediction and CSV-rendering functions as the CLI; the submitted CSVs therefore
come from the displayed pipeline, not a separate notebook.

| subsystem | released training data | task and organiser metric |
|---|---|---|
| Door | one 18,036-row, 50 Hz stream; 110 labelled cycles (80 Normal, 30 Abnormal resistance) | segment the raw stream and classify each non-overlapping cycle; greedy same-label IoU-weighted F1 |
| ACV | 6 cases, 8 cars per case, 30 s telemetry; the parameter set varies by workbook and four cars in case 04 are empty | rank every car by leak likelihood; linear rank decay |
| Rail corrugation | 272 one-second, 10 kHz files (234 Normal, 14 Side I, 24 Side II); speed pulse plus 64 vibration/shock sensor pairs | classify Normal/Side I/Side II; fixed-vocabulary macro F1 |
| SHM | 64 single-channel stress records of 581,120 samples; labels 0.029–0.928 | regress cumulative fatigue damage; `max(0, 1 − MAPE)` |

The distributed Test inputs are unlabelled: one Door stream (38 inferred cycles), one ACV
workbook, 68 Rail files and 16 SHM files. We never use an organiser Test label because none is
available.

## Approach

### Door

A gap larger than 0.5 s starts a cycle; a command/position hysteresis state machine is the
no-gap fallback. Both preserve source timestamps to the millisecond and never emit overlapping
segments. Per-cycle features describe direction, duration, middle-travel current, current peak,
voltage, back-EMF residual, energy and distance from a normal position/current profile. Separate
Open/Close logistic models use a baseline fitted on training cycles only. Current and residual
features follow condition-monitoring practice [R64][R66][R67]; a stump, random forest, SVM,
LightGBM, stacking, Quant, MultiRocket and LITETime provide comparison rows [R43][R47][R49][R53].

### ACV

The loader discovers `Car N - parameter` columns rather than assuming a fixed workbook width,
normalises mode/validity aliases, and leaves cars with no usable values last. Because six cases
cannot support a credible supervised classifier, the deployed, pre-registered rule ranks a car
by its hot, cooling-mode indoor-temperature delta from its seven contemporaneous peers. This
uses the train itself as a matched weather/load reference, consistent with peer-fleet residual
diagnosis [R251] and temperature-direction rules [R252]. Runtime, control-residual, unmet-load,
recovery, persistence and robust-peer alternatives are reported but not promoted after viewing
the six cases [R253][R254]. No car-ID prior is used.

### Rail corrugation

The 90-tooth pulse is integrated as distance, and the axle-box vibration channels are summarised
in time and fixed-frequency bands. Side medians/maxima/top-three means, Side I−II contrasts,
peak prominence, spectral flatness and envelope statistics distinguish sustained side-wide
corrugation from isolated impulses [R233][R235][R239][R250]. The selected balanced LightGBM adds
Welch magnitude-squared coherence across the six same-side vibration-box pairs per car. It
averages each side's 48 pairs in seven Hz bands and supplies Side I, Side II and difference
summaries; mirror test-time averaging swaps the sides exactly. Odd axle-box positions 1/3/5/7
are Side I and even positions 2/4/6/8 are Side II. The model uses no shock channels. Its
fixed-Hz bands and explicit speed are useful on this release, but `speed < 20 km/h → Normal`
is a dataset shortcut, not physics. The wavelength/v² design remains in the ladder because
axle-box amplitude is speed-sensitive [R237]. Candidate duplicates are grouped before splitting.

### SHM

We implement 3- and 4-point rainflow, residue conventions, Miner sums over an S–N exponent grid,
spectral moments, Dirlik/Tovo–Benasciutti estimates, amplitude/count statistics and an octave
fatigue-damage spectrum [R269][R270][R274][R275][R276]. An all-data diagnostic finds that
4-point rainflow, half-cycle residue and `m=5` reproduce labels at MAPE 0.0254 after one global
scale; this is explicitly in-sample. Predictive rows refit every scale, transform, regularisation
and bias correction inside each fold. The shipped model is a log-damage Lasso on 109 statistics,
rainflow, spectral and FDS features, averaged 50/50 with a fold-calibrated m=5 rainflow estimate
only when the recording has positive stress skew. Log space matches the relative-error metric
and the fatigue power law [R269].

## Validation and the fold-local rule

Every scaler, threshold, template, feature selection, bias correction, augmentation and model
choice is fitted using the training partition only. A held-out file/block is transformed once
with those frozen quantities and is never augmented. Selection is nested where a ladder choice
is reported. Oracle segmentation, transductive Door batch normalisation and Rail speed-matched
subsets are diagnostics, never headlines.

| subsystem | frozen validation | honest estimate (mean ± fold SD) | authoritative JSON |
|---|---|---:|---|
| Door | five contiguous raw-stream outer blocks; inner contiguous selection; full segmentation + classification rerun | **0.9818 ± 0.0364 IoU-F1** | `door_cv.json:headline.iou_f1_mean/.iou_f1_sd` |
| ACV | leave one complete case out; deterministic, exploratory | **0.9792 ± 0.0510 rank decay** | `acv_ladder.json:selected.score/.sd` |
| Rail | grouped 5-fold × seeds [0,1,2], with grouped inner selection of 23 rows | **0.8051 ± 0.1291 macro F1** | `rail_ladder.json:nested.macro_f1_mean/.macro_f1_sd` |
| SHM | outer leave-one-file-out selection; repeated 5×10-fold ladder diagnostics | **MAPE 0.0187; score 0.9813** | `shm_ladder.json:nested.mape/.score` |

Door segmentation is boundary-exact on all 110 training cycles. Rail additionally reports
contiguous-filename (0.7817 ± 0.1349 for the selected row) and leave-speed-range-out
(0.7409 ± 0.1166) stress splits; the speed-matched subset removes the explicit low-speed branch
but cannot prove that all speed/label confounding is gone. SHM's positive-skew physics blend
was designed after exploratory LOO diagnostics; its six-row nested audit selected it in 62/64
outer folds and lowered nested MAPE from 0.0203 to 0.0187. This remains a Train-only estimate,
and its reported organiser Test score is 0.972795, versus 0.971752844618028 for the earlier Lasso.

## Model selection and benchmarking

The full, machine-sourced ladder is `results/ps3/leaderboard.md`; every displayed number names
its JSON key path. Selection-CV values are not substituted for outer estimates. Door's physics
logistic row scores 1.0000 on post-hoc selection CV, versus the 0.9818 nested headline. ACV's
best inspected rule scores 1.0000, but the fixed pre-registered 0.9792 rule remains the headline.
Rail's post-hoc coherence selection result is 0.8441 ± 0.1059, while the full 23-row nested
outer CV is 0.8051 ± 0.1291. The earlier 22-row nested result was 0.7571 ± 0.1232.
The revised rail round predeclared three further shock-free LightGBM rows: robust vibration RMS
(0.7872), mirror-aligned class-boost calibration (0.8242), and both together (0.7877), each on
the same grouped selection CV. None beat 0.8365 in that round, so W7 was retained then. The
separate `results/ps3/rail_revised_round.md` records their Side I F1 and the expanded nested
audit (0.7558 ± 0.1140 over 15 held-out folds and 25 candidates).
Two further training-only sensor augmentation rows were run with 12 parallel fold workers:
bounded vibration sensor gain jitter scored 0.8289 macro F1 (Side I 0.5863), while sparse
same-side-median sensor masking scored 0.8017 (Side I 0.5590), versus W7's 0.8365 (Side I
0.5975). Neither cleared the frozen selection gate, so no new nested audit or artifact change
was needed; `results/ps3/rail_sensor_augmentation_round.md` records the 15-fold results.
Frozen external encoders were also probed on the same duplicate-grouped folds, with sensor
embeddings pooled separately for the two rail sides. A 75% W7 / 12.5% short-window MantisV2 /
12.5% long-window MOMENT blend scored 0.8381 macro F1 and 0.6038 Side I F1, clearing the
historical selection gate by only 0.0017 and correcting two of 816 held-out decisions. A
conditional 13-row nested transfer audit on the reused outer splits scored 0.8363; contiguous
and held-speed-range stress scores were 0.7446 and 0.6997, compared with 0.7436 and 0.6865
for its refitted W7 reference. The encoder weights exceed the 5 MB rail artifact limit, and
the small local gain did not justify replacing W7 in that round. MOMENT, UniTS, SimMTM, Chronos, and
Moirai comparisons are reported in `results/ps3/rail_transfer_compare.md` and
`results/ps3/rail_forecast_transfer_compare.md`; no other frozen blend cleared the gate.
Fold-local MantisV2 fine-tuning of the first, last, or all three blocks scored at most 0.4810
standalone and 0.8258 with W7. Moirai-2 last-block fine-tuning scored 0.5459, or 0.5967
with fixed fault-class weighting; their W7 blends scored 0.8197 and 0.8176. A fold-local
Ti-MAE-style masked-autoencoder probe scored 0.4027. No fine-tuned large teacher beat W7,
so student distillation was not triggered. The full comparison and links to each fold report
are in `results/ps3/rail_foundation_round.md`.
The later same-side coherence row scored 0.8441 macro F1 and 0.6454 Side I F1 in frozen grouped
selection CV, against W7's 0.8365 and 0.5975. Full 23-row nested selection chose it in 6/15
outer folds and scored 0.8051 ± 0.1291. Its contiguous and speed-range stress macro F1 values
were 0.7817 and 0.7409. The 794 KB artefact replaced W7 after clearing the declared 0.8365
selection and 0.7471 nested gates. It changes two of 68 Rail Test predictions. The previous
organiser Rail score for the earlier W7 artefact was **0.7994152046783626**; the coherence CSV
subsequently scored **0.83104**, a gain of **0.03162480** macro F1. The portal supplied no per-file errors or confusion matrix. The
saved model and CSV hashes are in `results/ps3/portal_scores.json`, and
`results/ps3/rail_coherence_round.md` has the full validation audit.
Compact checks on the same grouped selection folds scored 0.8178 for a smaller regularized
LightGBM using the 201 coherence features and 0.7599 after removing time and vote blocks to
leave 134 features. Earlier 113-feature and 64-feature ladder rows scored 0.7720 and 0.7799,
respectively, on different starting feature sets. None displaced the selected row;
`results/ps3/rail_compact_coherence_round.md` records the new checks.
Blending W7 and coherence on the same 816 held-out grouped decisions scored 0.8420 with equal
probability weights and 0.8421 with 75% coherence, below coherence alone at 0.8441. The
aligned fold probabilities and five checked weights are in
`results/ps3/rail_coherence_w7_fusion.md`; no blend was promoted.
Adding frozen MantisV2 short-window and MOMENT long-window probe probabilities to coherence
at 6.25% each scored 0.8447 on the same selection folds, only 0.0007 above coherence alone.
It changed three of 816 held-out decisions, reduced Side I F1 from 0.6454 to 0.6406, and
requires encoder weights beyond the 5 MB Rail artefact limit. It has no full nested or stress
audit and was not distilled or promoted; `results/ps3/rail_coherence_teacher_fusion.md` records
the fold checks and one near-tie MOMENT probe difference from the older run.
SHM's full-feature Lasso reduces repeated-CV MAPE from the stats-only baseline 0.1197 to 0.0195.
The positive-skew physics blend lowers that to 0.0187, with nested LOO MAPE 0.0187; the prior
organiser score for the earlier Lasso was **0.971752844618028** (MAPE 0.02824716). Its
nested Train score rose from 0.97968234 to 0.98129996, a 0.00161762 absolute gain and 7.96%
relative MAPE reduction. The updated CSV subsequently scored **0.972795** on the organiser
portal, an absolute score gain of **0.00104216** over the earlier Lasso. This corresponds to
MAPE falling from 0.02824716 to 0.027205, a **3.69% relative reduction**.

Budget-based skips are deliberate. HIVE-COTE was estimated at roughly 340 h [R52]. Door has no
audio for an MNPE/SVM path [R75]. Rail adaptive decompositions and model-based roughness inversion
are slow or require unavailable receptance/pad parameters [R238][R247]; 272 files are too few for
contrastive pretraining [R244]. ACV virtual-charge sensing needs pressure/superheat channels that
are absent [R262], and 48 car-case rows do not justify few-shot classification [R268]. SHM raw
CNN/LSTM/TCN models are not credible at 64 records and add little in the nearest comparison
[R272]; multiaxial methods are impossible with one stress channel.

## Augmentation

Augmentation is always fold-local. It did not provide a reliable generalisation gain. Door raw
cycle jitter and window warping each scored 1.0000 on the same already-saturated selection CV as
the unaugmented raw models, so neither improved the honest headline [R296]. Rail's exact Side
I/Side II mirror is label-preserving; mirror test-time averaging was retained, but its gain is
inside fold spread and does not answer the speed confound. SHM C-Mixup worsened repeated-CV MAPE
from 0.01954 to 0.01993 for the comparable full-feature Lasso [R299]. ACV has no augmentation:
mixing six cases or cropping a multi-day leak can change the ranked fault label.
Time-series Mixup and CutMix have worked on some classification benchmarks [R309], but Rail
raw-signal mixing across speeds and rail sides can alter the fault signature or cancel phase.
We therefore tested fold-local feature Mixup on speed-near recordings: same-class fault mixing
at Beta(0.4,0.4) and Beta(2,2) scored 0.8409 and 0.8187 macro F1; fault/Normal mixing with
weighted soft targets scored 0.8157. All were below the selected coherence row's 0.8441.
A separate fold-local majority-class undersampling diagnostic kept 25% or 0% of low-speed
Normal training files. It scored 0.8410 and 0.8351 macro F1, also below the selected row.
Neither Mixup nor undersampling was promoted.

## Explainability and app

Each prediction returns organiser-schema rows plus a bounded trace (at most 2,000 points), named
physical quantities and a component/side/car target. The web page shows the exact downloadable
rows, the trace and a train tint. Rail files upload in batches of at most 32 and are processed
sequentially, keeping memory bounded. The server, models and production web bundle run without
network access.

## Assumptions and limitations

- ACV has only six labelled cases; its 0.9792 result is exploratory, not a confidence interval.
- Rail has only 14 Side I files. Fold SD and per-class results matter more than a point estimate.
- SHM's sample rate is not published. Frequencies are normalised to cycles/sample and unknown
  time scale is absorbed by a fold-local intercept; physical Hz claims are intentionally absent.
- Test inputs are visible, but all Test labels remain unseen. Validation estimates are not claims
  of held-out organiser performance.
- Door assumes the released inter-cycle gaps define boundaries. The state-machine fallback is
  separately checked after removing gaps.
- Rail's low-speed rule and fixed-Hz winner exploit this dataset's operating distribution and may
  not transfer to a route with low-speed faults.
- ACV parameter availability differs by case; unknown columns are reported and missing cars rank
  last rather than being imputed from another case.
- The compulsory specification requires a non-technical app and prediction ZIP, not a CLI.
  The Info Kits also show a simple `predict.py --input --output` convention. We provide that
  wrapper as compatibility insurance; it infers the subsystem, while `--task` is an optional
  disambiguator. The app remains the submission evidence and source of downloadable bytes.

Reference IDs resolve in `docs/research/references.md`; the task-specific synthesis and evidence
grading are in `docs/research/ps3_addendum.md`.
