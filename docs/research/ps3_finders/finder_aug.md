# Finder 4 — Time-series data augmentation for small, imbalanced PHM datasets

Scope: what to do about (a) **110 door cycles @ 50 Hz, 30 positives**, (b) **272 × 1 s @ 10 kHz × 128-channel
vibration files, 14 and 24 positives**, (c) a **64-file regression whose label is not preserved by cropping**.
Written against the team's existing ladder: the door/bearing classifiers are **MultiRocket + Hydra + ridge**,
**LightGBM on envelope/band features**, and a **stacking ensemble** (`model_ladder.md` §1b, §1c, lines 160-171,
316-344), not a from-scratch deep net. That distinction changes almost every recommendation below, because
**every headline augmentation number in the literature is measured on deep nets**.

Existing refs reused: `[R68]` subway-door stacking (Sci Rep 2026), `[R115]` realistic bearing evaluation (MSSP
2026), `[R58]` recording-level shapelet CV, `[R23]` TPA-AD pseudo-anomalies, `[R220]` leakage-robust RUL
evaluation. New candidates are numbered **A1-A25** for the synthesiser to renumber (references.md currently
ends at R230).

---

## Practitioner defaults — what a rail/PHM practitioner builds first with ~12 h per subsystem

With 30 / 14 / 24 positives the binding constraint is **variance of the estimate, not capacity of the model**.
A PHM practitioner would therefore spend the augmentation budget in this order, and would spend most of it
*not* on augmentation:

1. **Hour 0-1: fix the split before touching augmentation.** Fit the augmenter *inside* each training fold and
   never let an augmented child sit in a different fold from its parent. For the bearing set this means
   augmenting *after* the bearing-wise partition `[R115]`; for the door set, after leave-one-load-out `[R65]`.
   This is the single highest-value hour in the whole brief: A15/A16 document that synthesis-before-split is
   a textbook leakage class, and `[R220]` already shows 20-60 % → 99.9 % inflation on exactly this family of
   data. Report the window-random-vs-group-wise gap as the honesty headline, as the team already plans to.
2. **Hour 1-2: class weighting and threshold moving before any synthesis.** `class_weight='balanced'` /
   `scale_pos_weight`, plus a PR-AUC-selected operating point. A13 (Buda, *Neural Networks* 2018) finds that
   for CNNs **oversampling is the best of the classical fixes and — unlike in classical ML — does not cause
   overfitting**, and that **thresholding by prior is essential**; A14 finds that for **strong, properly tuned
   learners (XGBoost/CatBoost) balancing gives no gain at all** once a proper metric and consistent
   hyper-parameter selection are used. With LightGBM + stacking as the door workhorse, A14 is the operative
   result: **expect class weighting to be a wash and say so on the slide, rather than claiming SMOTE saved us.**
3. **Hour 2-4: two label-preserving transforms only — window warping and window slicing.** A1 (Iwana &
   Uchida, PLOS ONE 2021, 12 methods × 128 UCR × 6 architectures, code) makes **window warping the single most
   recommended general-purpose method**, top-ranked for VGG, ResNet and LSTM, with **slicing second** and a
   **negative correlation between gain and training-set size** — i.e. the gain is largest exactly at our scale.
   A2 is the original 20 lines of code for both. A6 is the one study that ran augmentation **through ROCKET**
   (10/13 UEA datasets improved, ≈ **+1.55 % mean relative accuracy**) — a real but *small* effect, which is
   the number to quote when a judge asks "did augmentation help?".
4. **Hour 4-6: the domain-physics augmenter, which is the actual contribution.** The team already owns a
   fault-injection simulator. `[R68]` is the published precedent — physically constrained door-cycle synthesis
   under kinematic consistency, with a reported **98.8 % physical-consistency pass rate** on the augmented set.
   Reporting a physical-consistency pass rate for our own injected cycles is a cheap, defensible, judge-legible
   number that no generic augmenter can produce. A17/A18 are the bearing-side equivalents (digital-twin /
   simulation-to-measurement augmentation) if the vibration arm wants the same story.
5. **Hour 6-8 (bearing only): SpecAugment-style masking on the time-frequency representation.** A10 is 15
   lines on top of any spectrogram/scalogram pipeline and is the natural augmenter for 10 kHz × 128 channels,
   where time-domain warping is expensive and semantically dubious. Add **channel dropout** (mask a random
   subset of the 128 channels) as the multivariate analogue — it also doubles as the "can we drop a sensor?"
   ablation the team already wants from Detach-ROCKET `[R57]`.
6. **Hour 8-10 (regression arm): C-Mixup, not cropping.** A11 is the correct answer to "labels are not
   preserved by cropping": mix *pairs of whole files* with a sampling probability weighted by **label
   similarity**, so the interpolated label is meaningful. ~40 lines, code released, +5-6 % reported. A12 is the
   NeurIPS-2023 alternative of comparable strength if C-Mixup's bandwidth is fiddly.
7. **Skip all generative synthesis.** GAN and diffusion augmenters (A19, A22) are 8-20 h each, need their
   own train/eval loop, and A20 (TSGBench, PVLDB 2024, best-paper nomination) exists precisely because the
   field could not agree on whether the generated series are any good. With 14 positives there is not enough
   signal to fit a generator that is not memorising. Cite them as consciously-not-bought.

**The honest framing for the slide:** at n = 110 / 272 / 64, augmentation is a **variance-reduction and
class-balancing device worth low single-digit points**, and the leakage-safe split is worth more than every
augmenter combined. The differentiated contribution is the physics-constrained injector, not the generic
transforms.

---

## Candidates, ranked by relevance × feasibility

Columns: `name | venue | year | doi_or_url | task | code_url | licence | reported_metric | dataset |
subsystem_relevance (0-3) | implement_hours | what_it_preserves_or_assumes | notes`.

### Tier 1 — build these

**A1. Iwana & Uchida, "An empirical survey of data augmentation for time series classification with neural
networks"**
| field | value |
|---|---|
| venue / year | **PLOS ONE** 16(7):e0254841, **2021** (peer-reviewed) |
| doi | https://doi.org/10.1371/journal.pone.0254841 |
| task | TSC augmentation benchmark |
| code | https://github.com/uchidalab/time_series_augmentation (Keras + numpy reference implementations) |
| licence | paper CC BY 4.0; repo licence `UNVERIFIED` |
| reported metric | 12 augmentation methods × **128 UCR datasets** × 6 architectures (MLP, VGG, ResNet, LSTM, BLSTM, LSTM-FCN). **Window warping = highest average rank for VGG, ResNet and LSTM**; slicing second; **DGW best for BLSTM**. **Rotation/flipping decreased accuracy for all six architectures**; permutation degrades badly by breaking temporal order; **negative correlation between accuracy gain and training-set size**. |
| dataset | UCR 2018 archive |
| relevance | **3** (door and bearing) |
| implement_hours | **1-2** — lift `augmentation.py` wholesale, it is pure numpy |
| preserves / assumes | window warping and slicing preserve **class** but *not* duration or phase; rotation assumes sign-invariance, which vibration and door-current signals do **not** have |
| notes | The empirical backbone of this whole document. Its two warnings map straight onto our data: **do not flip** a door-current or vibration signal, and **do not permute** segments of a door cycle whose semantics are ordered (unlock → open → dwell → close → lock). Also warns that over-transforming pushes classes into overlap and that LSTM-FCN simply does not respond to augmentation. |

**A2. Le Guennec, Malinowski & Tavenard, "Data Augmentation for Time Series Classification using Convolutional
Neural Networks"**
| field | value |
|---|---|
| venue / year | **ECML/PKDD Workshop on Advanced Analytics and Learning on Temporal Data (AALTD)**, **2016** |
| url | https://www.semanticscholar.org/paper/e467404d68c8c2f45bae0e4bdfda12fc7df65cce (workshop proceedings; no stable DOI found) |
| task | TSC augmentation (origin of **window slicing** and **window warping**) |
| code | none official; reimplemented in A1's repo and in `tsaug`/`aeon` |
| licence | n/a |
| reported metric | `UNVERIFIED` — the venue page is a workshop proceedings without a stable DOI; the *methods* are universally reproduced, the original numbers were not re-read |
| dataset | UCR |
| relevance | **3** |
| implement_hours | **0.5** (≈ 20 lines each) |
| preserves / assumes | slicing assumes **the label is carried by every sub-window** — true for steady bearing vibration, **false for a door cycle whose fault signature lives in one phase**, and **false for the 64-file regression** |
| notes | Cite as the primitive; cite A1 for the evidence. The slicing caveat is the direct bridge to the regression sub-problem: if the label is not sub-window-stationary, slicing is label noise, not augmentation. |

**A3. Wen, Sun, Yang, Song, Gao, Wang & Xu, "Time Series Data Augmentation for Deep Learning: A Survey"**
| field | value |
|---|---|
| venue / year | **IJCAI 2021** (peer-reviewed) |
| doi | https://doi.org/10.24963/ijcai.2021/631 — preprint https://arxiv.org/abs/2002.12478 |
| task | survey + empirical comparison across **classification, anomaly detection, forecasting** |
| code | none |
| licence | n/a |
| reported metric | AD experiment on Yahoo with U-Net: raw **F1 0.403** → decomposition **0.662** → decomposition + augmentation **0.693**. Classification on 5,000 Alibaba Cloud samples: **+0.11 % to +1.92 %**. Forecasting: **mixed, from +76 % (M4-weekly) to −16 % (traffic)** with DeepAR/Transformer. |
| dataset | Yahoo AD, Alibaba Cloud monitoring, electricity/traffic/M4 |
| relevance | **3** — it is the only source here that covers the **anomaly-detection** arm, which is what the pneumatic/MetroPT subsystem actually is |
| implement_hours | **0** to read; **1** for the **label-expansion** trick alone |
| preserves / assumes | label expansion assumes **anomalies persist over a span**, so points adjacent to a labelled anomaly are also anomalous — exactly the episode structure the team already builds |
| notes | Two usable items: (i) **label expansion** is a legitimate, cited way to widen 3-4 episode positives without inventing signal; (ii) the **−16 % forecasting result** is the honest citation for "augmentation is not free" and pairs with the team's existing sanity-anchor section. The AD numbers are on a *univariate* benchmark; do not transfer them to MetroPT. |

**A4. Gao, Liu & Li, "Data Augmentation for Time-Series Classification: An Extensive Empirical Study and
Comprehensive Survey"**
| field | value |
|---|---|
| venue / year | **JAIR** (Journal of Artificial Intelligence Research), **2025** (arXiv 2023) |
| doi | https://doi.org/10.1613/jair.1.17084 — preprint https://arxiv.org/abs/2310.10060 |
| task | survey (100+ articles, 60+ techniques) + empirical study |
| code | `UNVERIFIED` — no repo found on the abstract page |
| licence | JAIR is open access |
| reported metric | ~20 strategies × **15 UCR datasets** × ResNet/LSTM. Baselines **84.98 ± 16.41 % (ResNet)**, **82.41 ± 18.71 % (LSTM)**. **Random Geometric Warps and Random Permutation gave significant improvement; EMD-based decomposition was ineffective.** Taxonomy: Transformation / Pattern / Generative / Decomposition / **Automated**. |
| dataset | UCR (15) |
| relevance | **2** |
| implement_hours | **0** to read |
| preserves / assumes | as per family |
| notes | **Flag the disagreement honestly:** A4 reports random **permutation** as a winner; A1 reports it as one of the worst. The resolution is architecture and dataset selection (ResNet/LSTM on 15 sets vs 6 architectures on 128 sets), and A1 is the larger, more careful study. For our ordered door cycles, follow A1 and do not permute. Citing both, with the conflict named, is stronger than citing either. |

**A6. Ilbert, Hoang & Zhang, "Data Augmentation for Multivariate Time Series Classification: An Experimental
Study"**
| field | value |
|---|---|
| venue / year | **MulTiSA workshop @ ICDE 2024** (workshop-reviewed) |
| url | https://arxiv.org/abs/2406.06518 |
| task | augmentation for **multivariate** TSC |
| code | `UNVERIFIED` |
| licence | arXiv default |
| reported metric | **ROCKET and InceptionTime** on **13 UEA/UCR multivariate datasets**; accuracy improved on **10/13**, **≈ +1.55 % mean relative improvement** for the best method vs the un-augmented ROCKET baseline; gains do **not** track baseline accuracy. |
| dataset | UEA multivariate archive |
| relevance | **3** — the *only* candidate that measures augmentation with **ROCKET**, which is the team's actual classifier |
| implement_hours | **0** to read; it validates A1/A2's transforms rather than adding new ones |
| preserves / assumes | as per underlying transforms |
| notes | This is the number to put on the slide: **augmentation through a ROCKET + ridge pipeline buys ~1.5 %, not the 10-45 % headline you see for deep nets.** It sets a realistic expectation and pre-empts the "why didn't augmentation help more?" question. Workshop venue — weaker evidence than A1, mark accordingly. |

**A9. Song, Tang, Xia, Zhang, Kang & Li — physically constrained door augmentation (= existing `[R68]`)**
| field | value |
|---|---|
| venue / year | **Scientific Reports**, **2026** |
| doi | https://doi.org/10.1038/s41598-026-43371-5 |
| task | subway-door fault prediction, stacking ensemble, **physically constrained augmentation under kinematic consistency** |
| code | none published |
| licence | CC BY-NC-ND |
| reported metric | **98.8 % physical-consistency pass rate** on the augmented sample set; downstream pipeline ROC-AUC 0.977 / PR-AUC 0.913 under 7.3:1 imbalance (already in `model_ladder.md`) |
| dataset | subway door records |
| relevance | **3** (door) |
| implement_hours | **0 extra** — the team's injector already exists; **1 h** to add and report the pass-rate metric |
| preserves / assumes | assumes a kinematic model of the door (travel, dwell, velocity/acceleration envelope) that synthetic cycles must satisfy |
| notes | The team cites this already, but **not for its augmentation contribution or its 98.8 % number**. Adding a *physical-consistency pass rate* for our own injected cycles converts the simulator from "a shortcut we have to defend" into "a measured, precedented contribution". Highest value-per-hour item in this whole list. |

**A10. Park, Chan, Zhang, Chiu, Zoph, Cubuk & Le, "SpecAugment"**
| field | value |
|---|---|
| venue / year | **Interspeech 2019** (peer-reviewed) |
| doi | https://doi.org/10.21437/Interspeech.2019-2680 — https://www.isca-archive.org/interspeech_2019/park19e_interspeech.pdf |
| task | augmentation on the **spectrogram** (time warping + **time masking** + **frequency masking**) |
| code | in `torchaudio` (`FrequencyMasking`, `TimeMasking`), Apache-2.0 |
| licence | torchaudio Apache-2.0 |
| reported metric | LibriSpeech test-other **6.8 % WER without an LM, 5.8 % with shallow fusion** — SOTA at the time, from augmentation alone on an unchanged network |
| dataset | LibriSpeech 960 h, Switchboard 300 h |
| relevance | **3** (bearing / 10 kHz vibration, wherever a spectrogram or scalogram is the input) |
| implement_hours | **1** |
| preserves / assumes | assumes the class is **redundantly encoded across time and frequency**, so masking a band still leaves the fault detectable. **True for broadband bearing defect signatures; check it does not mask the defect harmonic family itself** — mask widths must be small relative to BPFO/BPFI spacing |
| notes | Cheapest real augmenter for the vibration arm. Natural multivariate companion: **channel dropout** over the 128 channels, which is the same idea on the sensor axis and doubles as a sensor-redundancy ablation. |

**A11. Yao, Wang, Pan, Huang, Finn — "C-Mixup: Improving Generalization in Regression"**
| field | value |
|---|---|
| venue / year | **NeurIPS 2022** (peer-reviewed) |
| url | https://proceedings.neurips.cc/paper_files/paper/2022/hash/1626be0ab7f3d7b3c639fbfd5951bc40-Abstract-Conference.html — https://arxiv.org/abs/2210.05775 |
| task | **regression** augmentation — mixup with pair-sampling weighted by **label similarity** |
| code | https://github.com/huaxiuyao/C-Mixup |
| licence | `UNVERIFIED` (repo) |
| reported metric | **11 datasets** (tabular → video): **+6.56 % in-distribution generalisation, +4.76 % task generalisation, +5.82 % OOD robustness** over the best prior approach; theory shows lower MSE than vanilla mixup |
| dataset | 11, incl. tabular and time-series-adjacent |
| relevance | **3** — the direct answer to the **64-file regression whose label survives no crop** |
| implement_hours | **2-3** (the kernel bandwidth over labels is the only hyper-parameter) |
| preserves / assumes | assumes the **label is locally linear in the input**, which is why it only mixes *near-label* pairs; makes no assumption about temporal structure, so it works on whole files |
| notes | Vanilla mixup on regression labels "can result in arbitrarily incorrect labels" — that is precisely the failure mode the brief describes for cropping, and C-Mixup is the published fix. With 64 files, mixing whole files is also the only augmenter that does not need sub-window label stationarity. |

**A13. Buda, Maki & Mazurowski, "A systematic study of the class imbalance problem in convolutional neural
networks"**
| field | value |
|---|---|
| venue / year | **Neural Networks** 106:249-259, **2018** (peer-reviewed) |
| doi | https://doi.org/10.1016/j.neunet.2018.07.011 — preprint https://arxiv.org/abs/1710.05381 |
| task | imbalance remedies for CNNs: oversampling, undersampling, two-phase training, **thresholding** |
| code | `UNVERIFIED` |
| licence | n/a |
| reported metric | MNIST / CIFAR-10 / ImageNet at increasing imbalance: **oversampling is the dominant method and, unlike in classical ML, does not cause overfitting in CNNs**; **thresholding by class prior should almost always be applied**; undersampling is competitive only at extreme imbalance |
| dataset | image benchmarks (transfer to TS is by analogy, not measurement) |
| relevance | **2** |
| implement_hours | **0.5** |
| preserves / assumes | assumes a deep net trained by SGD; image-domain evidence |
| notes | The canonical "just oversample and move the threshold" citation, and the counterweight to A14. Honest caveat for the write-up: **this is image evidence applied to 1-D signals**, and the team's headline models (ROCKET+ridge, LightGBM) are *not* SGD-trained CNNs, so A14 governs them. |

**A14. Elor & Averbuch-Elor, "To SMOTE, or not to SMOTE?"**
| field | value |
|---|---|
| venue / year | **arXiv preprint, 2022** — `VENUE UNVERIFIED`, **not peer-reviewed** as far as could be confirmed |
| url | https://arxiv.org/abs/2201.08528 |
| task | does class balancing help? 73 datasets, many balancing methods |
| code | **https://github.com/aws/to-smote-or-not** (AWS, Apache-2.0 likely — `UNVERIFIED`) |
| licence | `UNVERIFIED` |
| reported metric | Balancing helps **weak** classifiers (as the older literature claimed), but **does not improve prediction performance for strong classifiers** (XGBoost/CatBoost class) once a proper metric and consistent hyper-parameter selection are used. Metric choice and consistency of the algorithm materially change the apparent effect of balancing. |
| dataset | 73 public tabular datasets |
| relevance | **3** — the door subsystem's workhorse is a **LightGBM/XGBoost stack**, i.e. exactly the "strong classifier" case |
| implement_hours | **0.5** (an ablation: with vs without SMOTE at fixed threshold policy) |
| preserves / assumes | tabular feature-space setting; assumes the tuning budget is honest |
| notes | The right result to run *against ourselves*: publish the "SMOTE changed nothing on our stack" ablation rather than quietly using SMOTE. It also explains why point-adjusted or accuracy-style metrics make balancing look better than it is — the same methodological point the team already makes for TSAD. Preprint status must be stated. |

**A15. Kapoor & Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science"**
| field | value |
|---|---|
| venue / year | **Patterns** (Cell Press), **2023** (peer-reviewed) |
| doi | https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9 (PubMed 37720327) — preprint https://arxiv.org/abs/2207.07048 |
| task | leakage taxonomy (**8 types**) + "model info sheets" |
| code | model-info-sheet templates via https://reproducible.cs.princeton.edu/ |
| licence | open access |
| reported metric | leakage found in **17 fields, 294 papers**, "in some cases leading to wildly overoptimistic conclusions" |
| dataset | literature survey |
| relevance | **3** (methodology, all subsystems) |
| implement_hours | **1** to fill in one model info sheet for the pipeline |
| preserves / assumes | n/a |
| notes | Augmenting before the split is a *named* leakage type (illegitimate sharing of information between train and test). Pair with `[R220]`'s **20-60 % → 99.9 %** inflation figure and `[R115]`'s bearing-wise split requirement. The specific rule for us: **the augmented children of a parent window must live in the parent's fold, and the augmenter's own statistics must be fitted on the training fold only.** |

### Tier 2 — cheap, worth an ablation slot

**A5. Yang & Desell, "Robust Augmentation for Multivariate Time Series Classification"** — arXiv:2201.11739,
**2022**, `VENUE UNVERIFIED / NOT PEER-REVIEWED`. https://arxiv.org/abs/2201.11739. Cutout, cutmix, mixup and
window warp on **26 UEA multivariate datasets** across convolutional, recurrent and self-attention models;
**InceptionTime + augmentation improves accuracy by 1-45 % on 18 of the datasets**. Code URL not found; licence
CC BY-NC-SA 4.0 on the preprint. Relevance **2**, implement **2 h**. Preserves: cutout/cutmix assume the class
is not destroyed by removing a contiguous chunk; mixup assumes linear label interpolation is meaningful (fine
for binary door fault, **not** for the regression arm — use A11 there). Note the 1-45 % range is enormous and
per-dataset; quote the *median*, not the maximum, or it looks like cherry-picking.

**A7. Forestier, Petitjean, Dau, Webb & Keogh, "Generating Synthetic Time Series to Augment Sparse Datasets"**
— **IEEE ICDM 2017**, pp. 865-870, IEEE ICDM 2017 pp. 865-870 (`DOI UNVERIFIED`), author PDF https://germain-forestier.info/publis/icdm2017.pdf, record https://research.monash.edu/en/publications/generating-synthetic-time-series-to-augment-sparse-datasets. Weighted **DTW Barycentre Averaging (wDBA)** with three
weighting schemes (Average All, Average Selected, Average Selected with Distance), evaluated with **1-NN DTW at
1-2 training samples per class**. Code: `tslearn.barycenters.dtw_barycenter_averaging` (BSD-3), plus A1's repo.
**Exact accuracy gains `UNVERIFIED`** — the PDF's result tables did not extract cleanly; the qualitative claim
("substantial improvement in the extreme few-shot regime") is what can be asserted. Relevance **3** for the
door (110 cycles is genuinely sparse), **1** for bearing (DTW on 10,000-sample windows is too slow — A1
measures pattern-mixing methods at **60+ s per generated sample**). Implement **3 h**. Preserves: the *DTW-space
average shape* of the class; assumes warping-invariance, which holds for door cycles at varying load/speed —
that is exactly the `[R65]` multiple-operating-conditions problem — and is dubious for bearing phase. A1 also
reports **wDBA generates low-diversity samples**, so treat it as a complement to window warping, not a
replacement.

**A8. Demirel & Holz, "Finding Order in Chaos: A Novel Data Augmentation Method for Time Series in Contrastive
Learning"** — **NeurIPS 2023**,
https://proceedings.neurips.cc/paper_files/paper/2023/hash/61c2c6338033da68885e0226881cbe71-Abstract-Conference.html,
arXiv:2309.13439. Code https://github.com/eth-siplab/Finding_Order_in_Chaos (licence `UNVERIFIED`). A mixup
variant that treats **phase and magnitude as separate features** to respect the **quasi-periodic,
non-stationary** structure of physiological/sensor signals, controlling "the degree of chaos" the augmentation
introduces. Relevance **2** — door cycles are quasi-periodic by construction and bearing vibration is
genuinely periodic, so this is the most physically appropriate mixup in the list; but its evaluation is in a
**contrastive-pretraining** setting, which the team is not running. Implement **4-5 h**. Preserves: periodic
structure and phase relationships that time-domain mixup destroys.

**A12. Schneider, Goshtasbpour & Perez-Cruz, "Anchor Data Augmentation"** — **NeurIPS 2023**,
https://proceedings.neurips.cc/paper_files/paper/2023/hash/ecc9b6dfdbe374c0a3364ff81cd28642-Abstract-Conference.html,
arXiv:2311.06965, code https://github.com/NoraSchneider/anchordataaugmentation. Extends **anchor regression**
(causality literature): cluster the data to define anchors, sample a regularisation strength γ, and emit
several modified replicas per sample. **Competitive with C-Mixup** on the same in-distribution and OOD
benchmarks. Relevance **3** for the 64-file regression, implement **3 h**. Preserves: the causal/invariant part
of the regression function; assumes a meaningful clustering of the 64 files exists (by unit? by operating
condition?). Run it **only if** C-Mixup underperforms — they solve the same problem and doing both is a waste
of the 12 h.

**A16. Vieira et al., "Towards a more realistic evaluation of machine learning models for bearing fault
diagnosis"** (= existing `[R115]`, MSSP vol. 258, 2026, arXiv:2509.22267) — re-cited here **for the
augmentation-specific corollary** the team has not yet drawn: if augmented copies of the same physical bearing
land on both sides of the split, the augmentation *is* the leak, and the inflation is of the same magnitude as
the segment-wise-split inflation the paper already documents. Relevance **3**, implement **0 h** (it is a rule,
not code): **group the augmented children by parent bearing / parent door unit, and pass that group vector to
the CV splitter.**

**A17. Kulevome, Wang, Cobbinah, Mawuli & Kumar, "Effective time-series Data Augmentation with Analytic
Wavelets for bearing fault diagnosis"** — **Expert Systems with Applications** 249(A):123536, **2024**,
https://www.sciencedirect.com/science/article/abs/pii/S0957417424004019 (ESWA 249 part A, art. 123536; `DOI UNVERIFIED`). Generates synthetic **scalograms** by varying the decay/compress
parameter of **generalised Morse wavelets** online, applied at the input stage with no auxiliary generator.
Code `UNVERIFIED`; numbers `UNVERIFIED` (ScienceDirect abstract only). Relevance **3** for bearing *if* the team
takes a CWT/scalogram route, **1** if it stays on envelope-spectrum features. Implement **4 h** (needs
`ssqueezepy` or MATLAB-equivalent Morse CWT). Preserves: the time-frequency ridge structure of the defect
impulses; assumes the wavelet family is the right basis. Its selling point over GANs is explicitly that it
needs **no separate generative training stage** — the same argument the team should make.

**A18. Zhang, Zhen, Feng, Cui, Zhang & Gu, "Resonance-aware digital twin-driven data augmentation for bearing
fault diagnosis under sample imbalance"** — **Structural Health Monitoring**, **2026**,
https://doi.org/10.1177/14759217261462579. Adaptively extracts the optimal **resonance band** from measured
fault signals to fit the dynamic-model parameters, minimising the resonance-characteristic gap between
simulated and measured data; framed as *interpretable* augmentation under imbalance. Numbers `UNVERIFIED`.
Relevance **2** (it is the bearing-side twin of `[R68]`'s door argument), implement **6-8 h** — too much for
12 h unless the bearing simulator already produces resonance-band-matched signals. Cite as the precedent that
**physics-parameterised augmentation is the accepted answer to sample imbalance in bearing PHM**, then point at
our own injector. Companion lead, same family: "Physics-enhanced simulation-to-measurement translation for
rolling bearing fault diagnosis under limited samples", *Engineering Applications of AI* 2025,
https://www.sciencedirect.com/science/article/abs/pii/S095219762503310X (`UNVERIFIED`).

**A21. Zhang, Zhao, Tsiligkaridis & Zitnik, "Self-Supervised Contrastive Pre-Training for Time Series via
Time-Frequency Consistency" (TF-C)** — **NeurIPS 2022**, https://proceedings.neurips.cc/paper_files/paper/2022/hash/194b8dac525581c346e30a2cebe9a369-Abstract-Conference.html — arXiv:2206.08496 — code https://github.com/mims-harvard/TFC-pretraining. Pretrain on unlabelled signal, fine-tune on the handful of
labels; reports **+15.4 % F1 on average over baselines** in one-to-one transfer, and the wider SSL literature
reports **fine-tuning on 1-10 % of labels matching fully supervised training**. Relevance **2** — the team has
far more *unlabelled* door cycles and vibration seconds than labelled ones, so this is the structurally correct
answer to label scarcity, and it is the **alternative to augmentation, not a form of it**. Implement **6-8 h**
(pretrain + fine-tune loop, GPU). **Caveat worth quoting:** the semi-supervised literature reports that TS-TCC
and TS2Vec *degrade* at very small label counts, so with 14 positives this is a real gamble. Recommend as
**NICE / stretch**, not MUST.

### Tier 3 — cite as consciously skipped

**A19. Yoon, Jarrett & van der Schaar, "Time-series Generative Adversarial Networks (TimeGAN)"** — **NeurIPS
2019**,
author PDF https://www.vanderschaar-lab.com/papers/NIPS2019_TGAN_Main.pdf (proceedings URL `UNVERIFIED`), code
https://github.com/jsyoon0823/TimeGAN. The canonical TS GAN (adversarial + stepwise supervised loss on an
embedded space). Relevance **1**, implement **8-12 h + tuning**. **SKIP** (see also A20's benchmark evidence): with 14-30 positives a GAN has
nothing to learn that is not memorisation, and the discriminator/generator balance eats the entire subsystem
budget.

**A20. Ang, Huang, Bao, Tung & Huang, "TSGBench: Time Series Generation Benchmark"** — **PVLDB 17(3):305-318,
2024** (Best Research Paper *nomination*), https://doi.org/10.14778/3632093.3632097, code
https://github.com/YihaoAng/TSGBench, arXiv:2309.03755. **10 generation methods × 10 real datasets × 12
measures**, plus a domain-adaptation-based generalisation test. Relevance **2 as evidence**, implement **0 h**.
This is the **citation that justifies skipping A19/A21-class synthesis**: the field needed a benchmark because
generated-series quality is contested and method rankings are unstable across measures. Quoting it is much
stronger than "we ran out of time".

**A22 (lead). ReF-DDPM and the diffusion-augmentation family** — e.g. "ReF-DDPM: A novel DDPM-based data
augmentation method for imbalanced rolling bearing fault diagnosis", *Reliability Engineering & System Safety*,
**2024**, https://www.sciencedirect.com/science/article/abs/pii/S0951832024004150; and "Denoising diffusion
probabilistic model-enabled data augmentation method for intelligent machine fault diagnosis", *Engineering
Applications of AI*, **2024**, https://doi.org/10.1016/j.engappai.2024.109520. Third-party comparisons report
**DDPM ≈ 93.9 % and DDIM ≈ 95.8 % accuracy with ResNet at imbalance ratio 0.98, above DCGAN/BAGAN/TransGAN
(≈ 94.8 %)** — all numbers `UNVERIFIED` (abstract/secondary sources only, and these are accuracy figures on
balanced-test protocols, i.e. exactly the metric the team has already ruled out). Relevance **1**, implement
**10-16 h**. **SKIP**, but cite as the current SOTA direction so the ladder shows we knew what we were not
building. Consistent with the team's existing ImDiffusion SKIP `[R18]`.

**A23 (lead). Qiu, Pfrommer, Kloft, Mandt & Rudolph, "Neural Transformation Learning for Deep Anomaly Detection
Beyond Images" (NeuTraL-AD)** — **ICML 2021**, https://proceedings.mlr.press/v139/qiu21a.html, code
https://github.com/boschresearch/NeuTraL-AD (Bosch, AGPL-3.0 `UNVERIFIED`). *Learns* the transformations
instead of hand-picking them, which is the principled answer to A1's "which augmentation?" problem; strong on
time-series one-vs-rest AD, competitive on n-vs-rest. Relevance **2**, implement **6 h**. **SKIP for W4** — it
is a detector, not an augmenter, and would have to displace an existing ladder entry rather than add to one.
Worth one line in the addendum as the "learned augmentation" frontier, alongside the team's existing `[R23]`
TPA-AD pseudo-anomaly note.

**A24 (lead). Chen, Xu, Zeng & Xu, "FrAug: Frequency Domain Augmentation for Time Series Forecasting"** —
arXiv:2302.09292, **2023**, `VENUE UNVERIFIED` (OpenReview submission `G0uzEweZB1`, acceptance not confirmed).
**FreqMask** and **FreqMix** preserve the semantic consistency of the data-label pair in forecasting, where
time-domain augmentation breaks the fine-grained temporal relationship; reports that models trained on **1 % of
the data reach near-full-data accuracy**, and that test-time training with FrAug helps under distribution
shift. Relevance **2** — only if the physics/drift arm uses a forecast-residual detector, where it would be the
correct augmenter; **0** for the classification arms. Implement **2 h**. Note it is the forecasting counterpart
of A10 (both are masking in the frequency domain) and the two should be cited together.

**A25 (lead). de Souza & Leao, "Data Augmentation of Multivariate Sensor Time Series using Autoregressive
Models and Application to Failure Prognostics"** — arXiv:2410.16419, **2024** (Siemens), `NOT PEER-REVIEWED`.
Time-varying AR (TVAR) with a "decoupling trick" that models mean and covariance dynamics as separate
sub-processes. On **C-MAPSS FD001/FD003 with only five real samples per experiment plus five synthetic ones**:
**RMSE −2 % / −6 %** and **scoring function −4 % / −20 %**. No code. Relevance **2** for the 64-file regression —
it is the closest published match to "tiny-sample multivariate sensor regression" and its whole premise is that
crop-style augmentation is unavailable. Implement **5 h** (`statsmodels` TVAR is fiddly). Listed as a lead
because the venue is unverified and n=5 makes the effect size statistically thin.

---

## Cross-cutting findings the synthesiser should carry into the ladder

1. **The magnitude of the prize is small and the literature agrees on that at our scale.** +1.55 % through
   ROCKET (A6), +0.11-1.92 % on a real ops dataset (A3), a median in the low single digits on UEA (A5). The
   45 % figures are per-dataset maxima on deep nets. Setting this expectation up front prevents the demo from
   over-claiming.
2. **Two transforms are actively harmful on our signals.** **Rotation/flipping** (hurt all six architectures,
   A1) — a door motor current or a vibration waveform has a physical sign. **Permutation** (A1; contested by
   A4) — a door cycle's phases are ordered, so permuting them creates a cycle that could not physically occur,
   which is the exact opposite of `[R68]`'s kinematic-consistency criterion.
3. **Label preservation is the decision rule, not method popularity.** Crop/slice needs a sub-window-stationary
   label (true: steady bearing vibration; false: door-phase faults, false: the 64-file regression). Mixup needs
   a linearly interpolable label (true: binary; false: regression → use A11/A12). Warping needs
   warping-invariance (true across load/speed conditions per `[R65]`; questionable for phase-sensitive bearing
   analysis).
4. **Augmentation and the split interact, and the split wins.** A15's taxonomy + `[R220]`'s 20-60 % → 99.9 % +
   `[R115]`'s bearing-wise requirement together give a three-citation argument that the augmentation-group
   constraint is not pedantry. One concrete implementation note: pass a **parent-id group vector** through the
   augmenter so `GroupKFold`/`LeaveOneGroupOut` can never separate a parent from its children.
5. **Class weighting is probably a no-op for this team's models, and that is a publishable ablation.** A14 vs
   A13 is a genuine, live disagreement whose resolution depends on classifier strength; the team's stack sits
   on A14's side of it. Reporting "we tried it, it did nothing, here is the reference that predicted that" is
   more credible than a silent SMOTE.
6. **The differentiating contribution is physics-constrained synthesis with a measured consistency rate**
   (A9, A18, A17), not any generic transform. That framing also converts the team's existing simulator — the
   thing judges are most likely to attack — into a cited methodological contribution.

---

## Verification status summary

| Item | Status |
|---|---|
| A1, A3, A7, A9, A10, A11, A12, A13, A15, A16, A17, A18, A19, A20, A23 | venue and DOI/URL confirmed peer-reviewed |
| A4 | JAIR DOI reported by the arXiv listing page; **JAIR page not opened** — `VENUE LIKELY, DOI UNVERIFIED` |
| A6 | MulTiSA @ ICDE 2024 **workshop** — weaker peer review than a main track |
| A2 | AALTD 2016 workshop; **original numbers UNVERIFIED** (methods universally reproduced) |
| A5, A14, A24, A25 | **arXiv preprints, not confirmed peer-reviewed** — flag in any citation |
| A7 accuracy gains, A17/A18/A22 numbers | **UNVERIFIED** — publisher paywall or abstract-only |
| Repo licences for A1, A8, A11, A12, A14, A23 | **UNVERIFIED** — not opened |
| A21 "+15.4 % F1" and "1 % labels ≈ full supervision" | from search snippets of TF-C and adjacent SSL papers; **numbers UNVERIFIED against the papers themselves** |
