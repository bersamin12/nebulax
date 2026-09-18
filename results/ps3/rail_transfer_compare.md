# Rail external encoder and multiscale transfer trial

Frozen externally pretrained encoders. Eight windows per channel are pooled within each of 64 vibration sensors. Odd box positions form Side I and even positions form Side II. Probe scaling and fitting remain within each grouped fold. No Test files or labels were used.

MOMENT long uses a 2048-sample window averaged to its 512-point input. UniTS and Mantis long receive 2048 native samples. SimMTM uses its native 178 samples.

Historical W7 selection gate: 0.8365. Current-code W7 is refitted in the table and can vary slightly from the historical result.

| row | macro F1 ± sd | Side I F1 | speed-matched macro F1 |
|---|---:|---:|---:|
| W7 current-code reference | 0.8341 ± 0.1008 | 0.5930 | 0.8387 |
| MOMENT short | 0.6225 ± 0.0704 | 0.2542 | 0.6035 |
| SimMTM short | 0.4384 ± 0.0719 | 0.1815 | 0.3827 |
| UniTS long | 0.4934 ± 0.0931 | 0.1721 | 0.4389 |
| Mantis long | 0.7119 ± 0.1365 | 0.3984 | 0.7009 |
| MOMENT long | 0.6702 ± 0.1282 | 0.4858 | 0.6357 |
| W7 + 25% MOMENT short | 0.8248 ± 0.0926 | 0.5771 | 0.8284 |
| W7 + 25% SimMTM short | 0.8266 ± 0.1014 | 0.5883 | 0.8307 |
| W7 + 25% UniTS long | 0.8243 ± 0.0928 | 0.5708 | 0.8279 |
| W7 + 25% Mantis long | 0.8227 ± 0.1100 | 0.5867 | 0.8263 |
| W7 + 25% MOMENT long | 0.8350 ± 0.1032 | 0.6038 | 0.8398 |
| W7 + 12.5% Mantis short + 12.5% MOMENT long | 0.8381 ± 0.0988 | 0.6038 | 0.8433 |
| W7 + 12.5% Mantis short + 12.5% UniTS long | 0.8343 ± 0.1010 | 0.5994 | 0.8387 |

Decision: **promotion candidate: W7 + 12.5% Mantis short + 12.5% MOMENT long**.

Nested audit: completed; see rail_transfer_nested.md.

Deployment: W7 retained: encoder weights exceed the 5 MB model limit, and the selected fusion corrects only two of 816 held-out decisions.

Retained W7 stress splits:

- contiguous: 0.7488 ± 0.1406; Side I F1 0.4032
- speed_range: 0.7258 ± 0.1006; Side I F1 0.4583

Selection runtime: 1.3 minutes excluding frozen extraction.
