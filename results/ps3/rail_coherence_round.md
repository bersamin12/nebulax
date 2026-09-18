# Rail same-side coherence round

The organiser reported **0.7994152046783626** for the shipped W7 rail CSV.
Only this aggregate score was available; no confusion matrix, per-file errors
or Test labels were used.

The train has eight cars with eight boxes each. Odd box positions (1, 3, 5,
7) contact Side I and even positions (2, 4, 6, 8) Side II. For each recording,
we compute Welch magnitude-squared coherence across the six within-side
vibration-box pairs in every car. We average the resulting curves across the
eight cars and seven fixed Hz bands, then supply Side I, Side II and their
difference to W7's existing no-shock feature table. The feature is phase
independent, uses one file at a time and mirrors exactly under the documented
box permutation.

| Train-only check | W7 reference | W7 + coherence |
|---|---:|---:|
| Frozen duplicate-grouped 5-fold × 3 selection CV, macro F1 | 0.8365 historical; 0.8341 current-code refit | **0.8441 ± 0.106** |
| Side I F1 on those folds | 0.5975 historical; 0.5930 current-code refit | **0.6454** |
| Contiguous stress macro F1 | 0.7488 historical | **0.7817** |
| Leave-speed-range-out macro F1 | 0.7258 historical | **0.7409** |
| Contiguous Side I F1 | 0.4032 historical | **0.4603** |
| Speed-range Side I F1 | 0.4583 historical | 0.4524 |
| Conditional two-row nested selection macro F1 | — | 0.8393 ± 0.103 |
| Full 23-row nested selection macro F1 | 0.7571 ± 0.1232 for earlier 22-row ladder | **0.8051 ± 0.1291** |
| Full nested Side I F1 | — | **0.5754** |

The conditional two-row nested audit chose coherence in 11 of 15 outer
folds. The full 23-row nested audit of the original ladder plus this
candidate chose coherence in 6 of 15 outer folds and cleared the declared
0.7471 gate. The two-row number is conditional and cannot be compared directly
with the original 22-row nested score of 0.7571.

Against the current-code W7 held predictions, coherence changed ten of 816
decisions: five corrections and five new errors. Four corrections were Side I,
including the same `Train150.csv` recording in three repeats. That dependence
is why the nested and stress checks matter more than the selection gain alone.

The promoted model refit on all Train files is 794,225 bytes and changes two of 68
Test predictions (`Test5.csv` to Normal and `Test56.csv` to Side II). The true
labels of those files remain unknown. The earlier W7 artefacts are backed up
under `data/ps3_backups/20260918_score_round_before_rail_coherence/`. The new
organiser score has not been measured.

Production and research feature calculations agree exactly on a real file.
The production pipeline reproduced **all 15 grouped folds and all held-out
predictions** from the research run exactly, and
production contiguous and speed-range stress scores reproduced the separate
audit to 1e-12. Synthetic mirror, short-recording and fold-isolation tests
passed.
