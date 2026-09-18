# Rail sensor augmentation round

Two predeclared W7 variants evaluated on frozen grouped 5-fold × three-seed CV. Only fault files in each training fold receive one extra transformed and mirrored copy. Gain jitter is ±0.75 dB per vibration sensor; masking imputes one vibration sensor on each side from the same-side median. No Test labels were used.

| row | selection macro F1 ± sd | Side I F1 |
|---|---:|---:|
| W7 reference | 0.8365 ± 0.0960 | 0.5975 |
| no shock + bounded sensor gain jitter | 0.8289 ± 0.1036 | 0.5863 |
| no shock + sparse sensor masking | 0.8017 ± 0.1003 | 0.5590 |

Nested audit: not run because neither row cleared the selection gate.

Decision: **retain W7: neither augmentation beat selection CV**.

Retained W7 stress splits:

- contiguous: 0.7488 ± 0.1406; Side I F1 0.4032
- speed_range: 0.7258 ± 0.1006; Side I F1 0.4583

Runtime: 1.9 minutes.

## Verification

- All 15 grouped selection folds completed; each of 272 training files was held out once per seed, and duplicate fingerprint groups remained intact.
- Full Rail classification test file passed, including gain coherence, side-median masking, and augmentation source isolation.
- Packaged app reproduced the saved W7 CSV byte for byte on all 68 Test inputs; no Test labels were read.
- The selected W7 model and CSV hashes were unchanged; the model is 790,652 bytes (<5 MB).
