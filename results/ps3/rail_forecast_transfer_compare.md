# Rail Chronos and Moirai transfer trial

Frozen released forecasting encoders. Chronos-2 processes all 64 sensors jointly; Moirai processes each car's eight axle boxes jointly. Side I and Side II statistics use odd and even box positions. Probes are fitted inside duplicate-grouped folds. No Test labels or files were used.

Historical W7 selection gate: 0.8365.

| row | macro F1 ± sd | Side I F1 | speed-matched macro F1 |
|---|---:|---:|---:|
| W7 current-code reference | 0.8341 ± 0.1008 | 0.5930 | 0.8387 |
| Chronos T5 tiny short | 0.5457 ± 0.0628 | 0.0222 | 0.5224 |
| Chronos Bolt tiny long | 0.5579 ± 0.1528 | 0.2089 | 0.5322 |
| Chronos-2 small joint | 0.6021 ± 0.0893 | 0.2224 | 0.5750 |
| Moirai-1 car joint | 0.7102 ± 0.0702 | 0.3674 | 0.7062 |
| Moirai-2 car joint | 0.7451 ± 0.1075 | 0.4811 | 0.7433 |
| W7 + 25% Chronos T5 tiny short | 0.8225 ± 0.0940 | 0.5660 | 0.8258 |
| W7 + 25% Chronos Bolt tiny long | 0.8324 ± 0.0993 | 0.5975 | 0.8396 |
| W7 + 25% Chronos-2 small joint | 0.8252 ± 0.1104 | 0.5975 | 0.8293 |
| W7 + 25% Moirai-1 car joint | 0.8285 ± 0.1044 | 0.5930 | 0.8321 |
| W7 + 25% Moirai-2 car joint | 0.8198 ± 0.1002 | 0.5752 | 0.8223 |
| W7 + 12.5% Mantis short + 12.5% Chronos-2 small joint | 0.8284 ± 0.1067 | 0.5975 | 0.8328 |
| W7 + 12.5% Chronos T5 tiny short + 12.5% Chronos-2 small joint | 0.8242 ± 0.0931 | 0.5752 | 0.8277 |
| W7 + 12.5% Mantis short + 12.5% Moirai-2 car joint | 0.8285 ± 0.1044 | 0.5930 | 0.8321 |

Decision: **retain W7: no forecasting encoder row beat selection gate**.

Nested audit: not run because no row cleared the selection gate.

Retained W7 stress splits:

- contiguous: 0.7488 ± 0.1406; Side I F1 0.4032
- speed_range: 0.7258 ± 0.1006; Side I F1 0.4583

Probe runtime: 1.2 minutes after frozen extraction and W7 probability caching.
