# Rail corrugation - post-hoc selected artefact; nested outer headline CV

`python scripts/ps3_train.py --task rail`, git `1ecb3e7`, feature version `v2`, seeds [0, 1, 2], 11097 s wall. Every number here is reproducible from `rail_cv.json` in this directory.

Training set: 272 files, Normal 234, Side I 14, Side II 24. Metric: macro F1 over the fixed three-label vocabulary (`nebulax.ps3.scoring.macro_f1`), so a class the model never gets right scores 0.

## Frozen validation schemes

| scheme | folds | macro F1 (mean +- sd) | speed-matched v>=35 | Normal F1 | Side I F1 | Side II F1 |
|---|---|---|---|---|---|---|
| stratified | 15 | 0.731 +- 0.090 | 0.728 +- 0.100 | 0.967 | 0.395 | 0.831 |
| contiguous | 5 | 0.716 +- 0.126 | 0.709 +- 0.129 | 0.970 | 0.338 | 0.841 |
| speed_range | 3 | 0.542 +- 0.031 | 0.511 +- 0.075 | 0.844 | 0.190 | 0.592 |

These three rows show the original all-on baseline. The selected coherence row's selection score and two stress scores are in `rail_ladder.json:winner` and `rail_ladder.json:winner_schemes`.

**Honest headline (nested stratified grouped 5-fold x seeds; complete ladder selected by grouped inner 3-fold, 15 folds): macro F1 0.805 +- 0.129.**
Every declared ladder row is ranked only on grouped inner folds of each outer training partition. The shipped row was chosen after the full ladder was read and is labelled post-hoc; its selection-CV score is not this headline.

Pooled confusion over every held-out fold of the headline scheme (rows = truth):

| true \ predicted | Normal | Side I | Side II |
|---|---|---|---|
| Normal | 682 | 9 | 11 |
| Side I | 16 | 24 | 2 |
| Side II | 2 | 6 | 64 |

## Ablations (same scheme, same seeds, one flag flipped)

| arm | macro F1 (mean +- sd) | Side I F1 | n features |
|---|---|---|---|
| baseline (all on) | 0.731 +- 0.090 | 0.395 | 528 |
| no distance resampling at all (fixed 20-5000 Hz bands, no wavelength discriminators) | 0.691 +- 0.115 | 0.306 | 245 |
| wavelength bands -> Hz bands, wavelength discriminators kept | 0.698 +- 0.115 | 0.343 | 352 |
| no v^2 normalisation | 0.768 +- 0.099 | 0.474 | 528 |
| no Side I - Side II contrast | 0.678 +- 0.089 | 0.265 | 400 |
| no mirror augmentation | 0.667 +- 0.111 | 0.260 | 528 |
| no speed features | 0.722 +- 0.090 | 0.356 | 523 |
| no impulsive/sustained discriminators | 0.686 +- 0.147 | 0.300 | 424 |
| no low-speed rule | 0.731 +- 0.090 | 0.395 | 528 |

## Dataset checks

* **Duplicate fingerprint** (sha1 of the first 1,000 samples of channel 1 **and** the speed column): 2 candidate group(s) - `Train107.csv`, `Train115.csv` labelled ['Normal']; `Train165.csv`, `Train187.csv` labelled ['Normal']. Full-file SHA-256 confirms 2 byte-identical group(s). Every candidate fingerprint group is forced into the same fold for both stratified and contiguous schemes, so it cannot straddle train and test.
* **Speed confound**: Normal spans 0-70 km/h, Side I 35-67, Side II 42-67. 100 Normal files and 0 fault files run below 20 km/h, so `speed < 20 km/h -> Normal` is a **dataset shortcut, not physics** - the no-rule ablation above shows exactly what it is worth, and the speed-matched column (v >= 35 km/h, the faults' own range) shows the score with the confound removed.

## Method

Tacho pulse -> cumulative distance (pi*0.85/90 = 29.67 mm per pulse, integrated, never differentiated) -> anti-aliased resampling of all 128 channels onto a 1 mm spatial grid -> Welch PSD in the wavelength domain -> 1/3-octave wavelength bands over 8-500 mm (IEC 61373 / EN 15610 presentation [R233][R250]), v^2-normalised [R237] -> per-side aggregation (32 odd = Side I, 32 even = Side II; median / p90 / max) plus the Side I - Side II contrast in dB, which cancels speed, track type and sensor gain as common mode. Per channel the wavelength bands sit beside the 26-feature metro set's time-domain shape factors (log RMS, kurtosis, log peak, crest, skew, margin, pulse and waveform factors) and spectral moments, with its fixed 250-2000 Hz wavelet bands replaced by the wavelength bands [R232]; the seven 20-5000 Hz bands survive only as the counter-design ablation arm [R234]. Impulsive-vs-sustained discriminators (spectral flatness, harmonic peak prominence, envelope duty cycle, within-side cross-channel agreement) keep a wheel flat or a switch from scoring as corrugation [R235][R246]. Classifier: LightGBM, class-balanced, 3 seeds averaged, with the Side I / Side II probability boosts tuned on three repeats of a grouped inner 3-fold split of each training fold (a file and its mirror share a group), and mirror augmentation (odd<->even swap with the label swapped) applied inside the training fold only.

Research basis: `docs/research/ps3_addendum.md` section 2 and `references.md` R231-R250 - the physics (29.67 mm per tacho pulse; lambda 25-80 mm [R248] sweeping 35-744 Hz over 0-67 km/h, so fixed-Hz bands survive only as the counter-design arm [R234]), v^2 normalisation [R237], the 1/3-octave wavelength band presentation [R231][R233][R250], computed order tracking [R239][R245], impulsive-vs-sustained confounders [R235][R246], the 26-feature metro set with its Hz bands replaced by wavelength bands [R232], ROCKET + RidgeClassifierCV on short accelerometer windows [R249][R43][R47], and the realism anchor [R241]: a properly held-out railway-vibration classification lands near 0.8, not the 95 %+ of rigs and simulations [R240][R243].

## Selected-row configuration

The selected row uses the no-shock, fixed-Hz features and adds 21 Welch magnitude-squared coherence summaries. For each car, six vibration-box pairs are formed within each rail side (positions 1/3/5/7 for Side I and 2/4/6/8 for Side II). The 48 pairs per side are averaged in seven Hz bands, then Side I, Side II and their difference enter the classifier. The coherence calculation uses only the recording being predicted; all classifier fitting and class-boost calibration remain inside each training fold.
