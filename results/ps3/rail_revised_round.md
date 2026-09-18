# Rail revised round

Three predeclared rows on the frozen grouped 5-fold x seeds [0, 1, 2] selection CV. W7 remains the reference at 0.8365. No Test labels were used.
Promotion required selection CV > 0.8365 and nested macro F1 >= 0.7471.

| row | macro F1 ± sd | Side I F1 | features |
|---|---:|---:|---:|
| W7 shock-free reference | 0.8365 ± 0.0960 | 0.5975 | 180 |
| no shock + robust vibration RMS | 0.7872 ± 0.1096 | 0.5327 | 196 |
| no shock + mirrored OOF boost calibration | 0.8242 ± 0.1071 | 0.5930 | 180 |
| no shock + robust RMS + mirrored OOF calibration | 0.7877 ± 0.0819 | 0.5213 | 196 |

Nested grouped outer macro F1: 0.7558 ± 0.1140 (15 folds, 25 candidates); Side I F1 0.4687.

Round runtime: 78.5 minutes (six-hour budget).

Decision: **retain W7**. Selected Side I F1: 0.5975.

- contiguous: 0.7488 ± 0.1406; Side I 0.4032
- speed_range: 0.7258 ± 0.1006; Side I 0.4583

## Verification

- Rail classification tests and Rail stream tests pass.
- All 15 nested folds match the frozen splits; each of 272 training files is held out once per seed.
- The packaged app prediction CSV matches both saved rail CSVs on all 68 Test inputs; no Test labels were read.
- The retained W7 model is 790,652 bytes, below the 5 MB limit.
