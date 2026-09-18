# Problem Statement 3 model pipelines

This document describes the **current repository** pipelines in `nebulax/ps3/` and saved artifacts in `models/ps3/` as of 19 September 2026. Each task follows `load -> featurise -> predict -> organiser CSV` through `nebulax/api/ps3.py`; the app and CLI use the same task implementations. The input schemas below describe the released data, while the model feature lists describe what the saved artifact actually consumes. The source datasets are not stored in this checkout; the input header examples come from `tests/fixtures/ps3/` and the loaders. The already running Cloud Run image was built before the latest Rail and SHM artifact updates; it needs a rebuild and redeployment to use these repository versions.

| Task | Unit predicted | Saved model | Output CSV columns |
|---|---|---|---|
| Door | One detected door cycle | Separate Open/Close logistic classifiers | `start_time,end_time,prediction` |
| ACV | Ranking of all cars in one case workbook | Fixed hot cooling peer temperature rule | `file_id,ranked_cars` |
| Rail corrugation | One recording file | Three seed LightGBM ensemble with same-side coherence | `file_id,prediction` |
| SHM | One stress history file | Log-Lasso with a positive-skew rainflow blend | `file_id,prediction` |

The `.pkl` files hold fitted models and preprocessing state; adjacent `.json` files describe their saved configuration. Training labels are separate from prediction inputs. The Test data have no released labels.

## Door

**Input and columns.** One CSV is a 50 Hz time stream, with 17 released columns:

| Raw column | Use |
|---|---|
| `Datetime` | Required; parse/sort timestamps, find cycle gaps, and emit exact start/end timestamps. |
| `Motor current(mA)` | Required; current magnitude, phase means/peaks, charge, energy, current profile and wavelet features. |
| `Door leaf position` | Required; determine Open/Close, travel fraction, stroke phases, speed, and the fallback segmenter. |
| `Motor Voltage(10mV)` | Mean voltage, energy and resistance proxy. |
| `Motor electrodynamic force` | Back EMF and speed residual. The loader also accepts “Motor back electromotive force.” |
| `Door opening time(.1s)`, `Door closing time(.1s)` | Compare observed cycle duration with the appropriate nonzero command time. |
| `Open command`, `Close command` | Fallback cycle boundaries and direction evidence. |
| `Door is opening`, `Door is closing` | Direction fallback when position travel is degenerate. |
| `DCSR`, `DCSL`, `DLSR`, `DLSL` | Summarised as switch fractions during feature extraction; **not selected by the deployed classifier**. |
| `Door Opened`, `Door Locked` | Loaded but **not selected by the deployed classifier**. |

`Datetime`, current and position are mandatory. The other known columns become NaN if absent; unrecognised columns are reported and ignored. Training uses `Train.csv` plus `Train_Segments_Answer.csv` (`t_start`, `t_end`, `status` after parsing). The raw status is `Normal` or `Abnormal resistance`.

**Preprocessing and features.** Timestamps are parsed to millisecond precision and sorted if necessary. A gap greater than 0.5 s splits cycles. If the stream has no such gap, command edges and a hysteretic position sweep identify boundaries. Each emitted cycle spans its actual first through last sample; cycles do not overlap. Position travel sets Open/Close. Travel fraction partitions a stroke into opening (0–15%), cruise (15–85%) and closing (85–100%). Current is used as an absolute magnitude.

Per cycle, the code computes current levels, peaks and their positions, voltage, back EMF, duration, charge, energy, speed proxies, command duration ratio, two coarse wavelet bands and a 64-bin current versus position profile. A baseline fitted on **normal training cycles separately for Open and Close** supplies medians, robust scales, a current profile template and an EMF versus speed fit. Its transform produces relative ratios, robust z scores, profile distance and EMF residual. Missing ratios are filled with the neutral value 1, missing z scores with 0; ratios are clipped to 0–10 and z scores to ±25.

**Deployed model.** `models/ps3/door.json` selects `DoorClassifier(kind="logreg", separate=True, baseline_mode="fold", augment="none")` with the 24 `PHYSICS_FEATURES` in `door_features.py`: cruise/opening/closing current, current peak and variation, charge/energy, voltage/EMF, resistance and speed proxies, duration, wavelet bands, profile distance and EMF residual. The models and baseline use training data only. The classifier yields abnormal probability for each cycle; its saved threshold assigns `Abnormal resistance` or `Normal`. Training can tune penalty and threshold using inner contiguous splits. The output keeps native sample boundaries in `start_time,end_time,prediction`; the app also shows current traces and per-cycle explanations. No extra smoothing merges or relabels cycles.

**Validation.** Five contiguous blocks of the raw training stream are held out, with segmentation and classification rerun end to end. The nested headline is 0.9818 ± 0.0364 IoU-weighted F1 (`results/ps3/door_cv.json`). Stump, forest, SVM, LightGBM, stacking and raw time-series models occur in the experimental ladder; they are not the saved predictor.

## ACV

**Input and columns.** Each `.xlsx` case has `Time`, often `Car model` and `Train number`, and repeating headers of the form `Car N - <parameter>` for each of up to eight cars. The workbook width and available parameters vary by case. The loader preserves each car ID's spelling, accepts one or two digits, and discovers columns by pattern. The following names are mapped to canonical fields:

| Canonical field | Accepted raw parameter names | Deployed role |
|---|---|---|
| Indoor temperature | `Indoor Average Temperature`; `Passenger Cabin Temperature Detected Value`; `Observation Area Temperature Detected Value` | Primary car versus peer comparison. |
| Outdoor temperature | `Outdoor Average Temperature`; `Outside Temperature Sensor Reading`; `Fresh Air Temperature Detected Value` | Select the hotter half of timestamps; falls back to fleet median indoor temperature, then all timestamps. |
| Running mode | `ACV Running Mode` | Restrict primary comparison to cooling modes. |
| Validity | `ACV Information Valid` | Exclude explicitly invalid readings; absent or blank flags are accepted. |
| Cooling setpoint | `ACV Control Temperature (Cooling)`; `Target Temperature Value` | Tie break using indoor temperature minus setpoint. |
| Heating setpoint, setting mode, load halved | `ACV Control Temperature (Heating)`; `ACV Setting Mode`/`ACV Control Mode`; `Load Halved`/`Load Shedding` | Loaded for alternative features or diagnostics; not part of the primary score. |

Other raw car parameters are listed as unmapped rather than silently treated as model features. `Car model` and `Train number` are metadata only. Training labels identify the leaking car once per case; six labelled cases were released.

**Preprocessing and features.** Parse `Time`, discard rows with invalid timestamps, convert numeric strings and missing tokens, and normalise mode/validity text. A car reading is usable when it has a finite indoor temperature, an accepted validity flag and a cooling mode (`automatic cooling`, `full cooling`, `half cooling`, `cooling` or `auto cooling`). “Hot” means the fleet median outdoor temperature is at or above its within-case median; if outdoor temperature is unavailable, the same rule uses fleet median indoor temperature, then all rows if needed. At each timestamp, subtract the contemporaneous median of usable cars' indoor temperatures. The car's primary score, `peer_delta_hot`, is the mean of that difference over hot, usable rows. At least 30 usable rows and 15 hot usable rows are required; cars without enough data receive no score and sort last. Four cars in training case 04 are entirely empty.

**Deployed model and output.** `models/ps3/acv.json` saves the fixed `baseline_peer_delta_hot` rule, with hot quantile 0.5 and cooling-only filtering. Higher score means more likely to leak. Ties use a within-case relative control residual (indoor minus cooling setpoint), then car ID for deterministic ordering; the car ID is **not** a learned fault prior. Every car appears once in the descending `ranked_cars` string, separated by `|`, next to the source workbook's `file_id`. The app can show the peer temperature trace. Runtime, recovery, persistence and other residual features are computed for ladder comparisons, but the deployed ranking does not combine them.

**Validation.** Leave one complete case out, six times, gives mean rank decay 0.9792 ± 0.0510 (`results/ps3/acv_ladder.json`). With only six cases this is exploratory, not a reliable uncertainty interval for new fleets.

## Rail corrugation

**Input and columns.** Each CSV is roughly one second at 10 kHz and has 129 columns: `Rotating speed`, then `Vibration of bearing in position P of car C` and `Shock of bearing in position P of car C` for P=1–8 and C=1–8. The first column is a tachometer pulse, not a direct speed value. The loader reads all 128 sensor channels, but the current selected model's feature table uses vibration channels; shock features are disabled. Odd axlebox positions are Side I and even positions are Side II. `Train_Labels.csv` gives one `Normal`, `Side I`, or `Side II` label per training file.

**Preprocessing and features.** Nonfinite signal entries become zero. Pulse edges, a 90-tooth wheel and 0.85 m wheel diameter give speed and distance. Vibration signals are centred; per-channel statistics include RMS, peaks, skew, kurtosis and shape factors. Welch spectra supply fixed 20–5000 Hz bands. Distance-resampled wavelength features can be calculated for diagnostics, but wavelength band levels are not selected model columns. The current model also uses Welch magnitude-squared coherence between same-side vibration boxes within each car, summarised by frequency band for Side I, Side II and their difference. Side aggregates, contrasts, per-car votes and speed features complete the design. Missing or infinite aggregate values are filled with zero. `models/ps3/rail.json` selects 201 columns with `coherence=true`, `shock=false`, `wavelength=false`, `hz=true`, and `v2_normalise=false`.

**Current repository model and postprocessing.** Three balanced LightGBM classifiers (seeds 0, 1, 2) average class probabilities. Training adds an exact left/right mirror of each recording with Side I and Side II labels swapped. At inference the model also scores the mirrored input, swaps those two probability columns back and averages the two views. Side I and Side II probabilities receive saved multipliers of 1.0 and 1.25 before argmax. A final `speed < 20 km/h -> Normal` rule applies. This is a dataset shortcut, so transfer to routes with low-speed faults is uncertain. One `file_id,prediction` row is emitted per recording. The explanation trace can use wavelength spectra and sensor locations without changing the selected model columns.

**Validation.** Duplicate candidate recordings are grouped before stratified splitting. Nested grouped five-fold validation over three seeds gives macro F1 0.8051 ± 0.1291 (`models/ps3/rail.json:honest_headline` and `results/ps3/rail_ladder.json`). The 0.8441 value in `models/ps3/rail.json` is post-selection CV, not the nested estimate. Only 14 Side I training files were available. The selected coherence model later received organiser score 0.83104 (`results/ps3/portal_scores.md`).

## Structural health monitoring (SHM)

**Input and columns.** Each CSV has no header and one numeric stress sample per line; released records have 581,120 samples. There is no timestamp or second sensor column. The sample rate is not supplied, so frequency features use cycles per sample rather than physical Hz. Training labels give one cumulative fatigue damage number per file.

**Preprocessing and features.** Read as float64. If at most 1% of samples are nonfinite, interpolate them; otherwise reject the file. The stress history produces one feature row from four families:

| Family | Extracted information |
|---|---|
| `stats` | Amplitude and count statistics: ranges, standard deviation, skew/kurtosis, roughness, turning points, block maxima and level crossings. |
| `rainflow` | Four-point rainflow cycles with half-cycle residue, range quantiles, amplitude/count statistics, Miner-style sums for S–N exponents 3–12, damage-equivalent loads and mean-stress corrections. |
| `spectral` | Welch power-spectrum moments and bandwidth, plus narrow-band, Dirlik and Tovo–Benasciutti damage approximations. |
| `fds` | Eleven octave-spaced single-degree-of-freedom bands, with band RMS and damage estimates. |

The selected feature matrix has 109 columns. Positive scale features are log transformed; signed or already logged quantities stay on their original scale. Nonfinite feature values become zero before inference. `file_id` identifies the record and is excluded from the design matrix.

**Current repository model and postprocessing.** `models/ps3/shm.json` selects all four families, `lasso_log`, a log target, fitted multiplicative bias correction, no mixup and `physics_blend_pos_skew=0.5`. A `StandardScaler` and `LassoCV` fit inside each training fold. The regressor predicts log damage, which is exponentiated and multiplied by the saved training-only bias factor. If stress skew is positive, the prediction is averaged 50/50 with an `m=5` rainflow damage estimate whose log intercept was fitted on training data. Negative-skew signals keep the Lasso result. The final value is clipped to 1e-6 through 1e6; a nonfinite result falls back to training median damage. Output is one positive `file_id,prediction` value per file, formatted to nine significant digits. The app's downsampled trace and explanations do not alter the CSV.

**Validation.** Nested leave-one-file-out selection yields MAPE 0.0187, or score 0.9813 (`models/ps3/shm.json` and `results/ps3/shm_score_round.md`). Repeated 5×10-fold CV is also reported. The positive-skew gate was chosen after exploratory diagnostics, so these Train-only estimates are conditional evidence. Its later organiser score was 0.972795 (`results/ps3/portal_scores.md`). The near-exact four-point rainflow `m=5` reconstruction in a diagnostic report is in-sample and is not the saved estimator or a held-out score.

## Shared output and evidence

The API and CLI load the saved artifact through `nebulax/ps3/common.py`, run each task's loader, feature extractor and predictor, then render the organiser columns through `nebulax/api/ps3.py`. `nebulax/ps3/submission.py` validates exact headers, IDs, class vocabulary, nonoverlapping Door segments and positive finite SHM values. File IDs retain the source basename and extension. Train labels, model selection, scalers, baselines and corrections are fitted within training partitions during validation; held-out samples are transformed only.

For implementation details, see `nebulax/ps3/{door,door_features,acv,acv_features,rail,rail_features,shm,shm_features}.py`. For validation and model comparisons, see `docs/ps3_writeup.md` and `results/ps3/leaderboard.md`. The CSV schema is specified in `docs/ps3_contract.md`.
