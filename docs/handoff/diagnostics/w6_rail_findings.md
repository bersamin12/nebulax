# W6 rail diagnostic (orchestrator, 18 Sep 2026 10:00-10:50, read-only; scripts diag_rail*.py, outputs diag_rail*.txt)

Data: `results/ps3/rail_cv.json` headline folds (15 outer folds, per-file held-out predictions) and the
per-channel feature cache `data/ps3_cache/rail/v2` (no raw files re-read).

1. **Where the score is lost.** Pooled confusion (rows truth): Side I -> Normal 17/42 fold-instances,
   Side I -> Side II 4, Side II -> Side I 9, Side II -> Normal 3, Normal wrong 15/702.
   Per-class F1: Normal 0.975, Side II 0.82, Side I 0.49. Macro F1 is the plain mean.
2. **The side label is NOT a left/right level contrast.** Under the documented layout (info kit: odd
   positions = Side I) the sign of median(Side I boxes) - median(Side II boxes) in the 25-80 mm band agrees
   with the label on 20/38 fault files (chance). Same for broadband RMS (21/38) and every Hz band (17-26/38).
   Per-box gain correction from Normal files does not change this. Both sides show the same +3..+5 dB
   fault-vs-normal excess on every box. What differs between Side I and Side II files in-sample is a
   per-CAR pattern (car 1 lower, cars 5-6 higher for Side I), which looks like a section/session signature.
3. **Per-car profile features do not help out of fold** (disposable 5-fold x 3-seed LightGBM harness,
   mirror-swap augmentation, low-speed rule): winner feats 0.812 -> + car profile 0.799. Car-profile alone 0.548.
4. **Per-box multiple-instance model is dead**: 0.44 macro F1 (per-box gains dominate the instance signal).
5. **The five always-missed Side I files are confident Normals**, not near misses: Train121 (34.9 km/h, slowest
   fault), Train170 (47 km/h, the quietest fault file, -38 dB), Train180/185/202 (66-67 km/h, where 9 Normals
   sit at 65-71). OOF p(Normal) 0.93-0.99. No threshold or prior boost recovers them: a boost sweep on OOF
   probabilities peaks at the shipped (x1, x1) setting (0.838 vs <=0.840 within noise).
6. **What does help, cheaply: a two-view ensemble.** Average the probabilities of two separately fitted
   LightGBMs on the two representations already in the cache:
   A = Hz bands + speed, wavelength discriminators kept, no v^2 (the shipped winner's opts
       `{"wavelength": False, "hz": True, "v2_normalise": False}`),
   B = wavelength bands, no v^2 (`{"v2_normalise": False}`).
   Same harness: A 0.816 +- 0.009, B 0.813 +- 0.012, union design matrix 0.814, **A+B average 0.830 +- 0.027**
   (Side I 0.594 vs 0.559). Geometric mean identical. Adding a third view or all four views hurts.
   Mirror TTA on A alone: 0.814 (no change). Two-stage (binary fault/normal with mirror label kept, then
   side) 0.798; two-stage + flat average 0.825.
7. **Test set**: 68 files, predicted 60/2/6. Speed-matched border cases under the shipped model (p(Normal)
   0.44-0.68): Test27 (Side I 0.53), Test32, Test9, Test37 (loud, max logRMS 6.6 dB, but kurtosis 7.6 ->
   impulsive, plausibly a wheel flat). Test33 is Side I 0.41 vs Side II 0.57. An ensemble can flip 1-3 of
   these; each Side I test file is worth ~0.1 macro F1 on the final score.
8. **Path note**: `RailTask.predict` (app/CLI path) calls `mdl.proba(X)` without the mirror view even when
   `mdl.tta` is True; `predict_test_set` (results csv) uses `predict_labels(..., X_mirror)`. On the current
   Test set both give byte-identical CSVs (md5 59ebeb15...), so no submission change, but the app path
   should apply TTA when the model asks for it.
