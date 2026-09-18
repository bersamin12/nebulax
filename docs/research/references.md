# References

Single numbered reference list for `model_ladder.md`, `rail_phm.md`, `datasets.md` and
`configs/model_ladder.yaml`. Cite by id, e.g. `[R11]`.

**Verification status.** Every item below was returned by the W0 research sweep and passed the
citation-verifier stage; each carries the exact DOI or URL that was retrieved. Where the sweep could
confirm only metadata (title / venue / DOI via Crossref) and **not** the numbers or the body text,
the entry is tagged `NUMBERS UNVERIFIED`. Where a stated fact came from a secondary source rather
than the primary document, the entry is tagged `SECONDARY-SOURCE`. Nothing outside this list may be
cited in our write-up; if a new item is needed it must be verified first and appended here.

**W1 update pass (14 Sep).** Sections **H-K** were added to close four gaps the W0 sweep left open:
first-principles physics and the Singapore operating envelope (H, R149-R162), freely available
regulatory text (I, R163-R172), change-point detection and concept drift (J, R173-R194), and
prognostics / RUL / survival analysis (K, R195-R230). Section I in particular **retires** two numbers
previously carried in `rail_phm.md`: the "EN 15437 alarm pair 95 °C / 56 K" (EN 15437-1 explicitly
excludes temperature methodology and alert response [R167]; the vendor page cited for it publishes no
values [R171]) and the framing of "100-200 N within 0.3 s" as an EN 14752 compliance figure (the free
regulatory text delegates every number to the paywalled standard [R163]). Both are now labelled as
what they are.

**W4 PS3 pass (17 Sep).** Section **L** (R231-R314) was added for `ps3_addendum.md`, covering the
four Problem Statement 3 subsystems: rail corrugation from axle-box acceleration (L1),
refrigerant-leak / undercharge diagnosis in HVAC and vehicle air conditioning (L2), fatigue-damage
estimation from stress time series (L3) and time-series data augmentation for small, imbalanced
datasets (L4). One candidate was **refuted** by the verifier and is deliberately absent — see the
note at the head of section L.

---

## A. Evaluation protocol, benchmarks and TSAD method evidence

R1. Liu, Q. and Paparrizos, J. **The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark (TSB-AD).** NeurIPS 2024 Datasets & Benchmarks Track. https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html — OpenReview https://openreview.net/forum?id=R6kJtWsTGy — code https://github.com/TheDatumOrg/TSB-AD (Apache-2.0), `pip install TSB-AD`.

R2. Kim, S., Choi, K., Choi, H.-S., Lee, B. and Yoon, S. **Towards a Rigorous Evaluation of Time-series Anomaly Detection.** AAAI 2022. https://arxiv.org/abs/2109.05257 — published version DOI 10.1609/aaai.v36i7.20680 — code https://github.com/tuslkkk/tadpak (repo licence UNVERIFIED).

R3. Wu, R. and Keogh, E. **Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress.** IEEE TKDE, 2023. https://doi.org/10.1109/TKDE.2021.3112126 — preprint https://arxiv.org/abs/2009.13807.

R4. Wagner, D. et al. **TimeSeAD: Benchmarking Deep Multivariate Time-Series Anomaly Detection.** TMLR 2023. https://ml.cs.rptu.de/publications/2023/TimeSeAD.pdf — code https://github.com/wagner-d/TimeSeAD (repo licence UNVERIFIED).

R5. Sarfraz, M. S. et al. **Position: Quo Vadis, Unsupervised Time Series Anomaly Detection?** ICML 2024. https://proceedings.mlr.press/v235/sarfraz24a.html — preprint arXiv:2405.02678 — code https://github.com/ssarfraz/QuoVadisTAD (repo licence UNVERIFIED).

R6. Paparrizos, J. et al. **TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection.** PVLDB 15(8), 2022. https://doi.org/10.14778/3529337.3529354 — code https://github.com/TheDatumOrg/TSB-UAD — docs https://tsb-uad.readthedocs.io/.

R7. Paparrizos, J. et al. **Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection (VUS-ROC / VUS-PR).** PVLDB 15(11), 2022. https://doi.org/10.14778/3551793.3551830 — code https://github.com/TheDatumOrg/VUS — journal extension VLDB Journal 2025, https://doi.org/10.1007/s00778-025-00907-x.

R8. Huet, A., Navarro, J. M. and Rossi, D. **Local Evaluation of Time Series Anomaly Detection Algorithms (affiliation precision / recall).** KDD 2022. https://arxiv.org/abs/2206.13167 — author implementation https://github.com/ahstat/affiliation-metrics-py (repo not opened; prefer the Affiliation-F shipped inside TSB-AD [R1]).

R9. Tatbul, N. et al. **Precision and Recall for Time Series (range-based P/R).** NeurIPS 2018. https://arxiv.org/abs/1803.03639.

R10. Lyu, Z. **Did We Actually Fix It? An Independent Adversarial Stress-Test of Post-Point-Adjustment Evaluation Metrics for Time-Series Anomaly Detection.** arXiv preprint, 2026. https://arxiv.org/abs/2607.11969. *Single-author preprint; the specific N thresholds are indicative, the qualitative ranking is what we act on.*

R11. Yeh, C.-C. M. **MMPAD — Matrix Profile for Time-Series Anomaly Detection: A Reproducible Open-Source Benchmark on TSB-AD.** arXiv preprint v3, 2026. https://arxiv.org/abs/2604.02445 (paper CC BY 4.0) — code https://github.com/mcyeh/mmpad_tsb (repo licence UNVERIFIED).

R12. **CATCH: Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching.** ICLR 2025. https://arxiv.org/abs/2410.12261 — OpenReview https://openreview.net/forum?id=m08aK3xxdJ — code https://github.com/decisionintelligence/CATCH. `NUMBERS UNVERIFIED`.

R13. **KAN-AD: Time Series Anomaly Detection with Kolmogorov–Arnold Networks.** ICML 2025. https://arxiv.org/abs/2411.00278 — OpenReview PDF https://openreview.net/pdf?id=LWQ4zu9SdQ — code https://github.com/issaccv/KAN-AD. `NUMBERS UNVERIFIED`.

R14. Shentu, Q. et al. **DADA: Towards a General Time Series Anomaly Detector with Adaptive Bottlenecks and Dual Adversarial Decoders.** ICLR 2025. Poster https://iclr.cc/virtual/2025/poster/29176 — proceedings PDF https://proceedings.iclr.cc/paper_files/paper/2025/file/ca7998666c2e53cc1e882b7268414d8a-Paper-Conference.pdf — OpenReview https://openreview.net/forum?id=aKcd7ImG5e — preprint associated with arXiv:2405.15273 — code https://github.com/iambowen/DADA (official-mirror status UNVERIFIED).

R15. Yang, Y. et al. **DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection.** KDD 2023. https://doi.org/10.1145/3580305.3599295 — code https://github.com/DAMO-DI-ML/KDD2023-DCdetector — author PDF https://yyysjz1997.github.io/Files/Yang-DCdetector-KDD2023.pdf.

R16. Audibert, J. et al. **USAD: UnSupervised Anomaly Detection on Multivariate Time Series.** KDD 2020. https://doi.org/10.1145/3394486.3403392. *No canonical author repo confirmed; take the implementation from TSB-AD [R1] or TimeSeAD [R4].*

R17. Tuli, S., Casale, G. and Jennings, N. R. **TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data.** PVLDB 15(6), 2022. https://doi.org/10.14778/3514061.3514067 — preprint arXiv:2201.07284 — code https://github.com/imperial-qore/TranAD.

R18. **ImDiffusion: Imputed Diffusion Models for Multivariate Time Series Anomaly Detection.** PVLDB 17(3), 2024. https://doi.org/10.14778/3632093.3632101 — preprint arXiv:2307.00754 — code https://github.com/17000cyh/IMDiffusion.

R19. Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S. and Dubrawski, A. **MOMENT: A Family of Open Time-series Foundation Models.** ICML 2024 (PMLR v235). https://proceedings.mlr.press/v235/goswami24a.html — preprint https://arxiv.org/abs/2402.03885 — weights https://huggingface.co/AutonLab/MOMENT-1-large (MIT) — code https://github.com/moment-timeseries-foundation-model/moment, research repo https://github.com/moment-timeseries-foundation-model/moment-research, `pip install momentfm`.

R20. Sylligardos, E., Boniol, P. et al. **Choose Wisely: An Extensive Evaluation of Model Selection for Anomaly Detection in Time Series (MSAD).** PVLDB 16(11), 2023. https://doi.org/10.14778/3611479.3611536 — code https://github.com/boniolp/MSAD — author PDF https://www.paparrizos.org/papers/SylligardosVLDB23.pdf — VLDB 2025 PhD-workshop follow-up https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/PhD/PhD25_17.pdf — explorer https://github.com/boniolp/ADecimo.

R21. Pinet, P. et al. **Anomalies in Multivariate Time Series Benchmarks Are Mostly Univariate.** arXiv preprint, 2026. https://arxiv.org/abs/2606.02670.

R22. Bello, A. et al. **GNNs for Time Series Anomaly Detection: An Open-Source Framework and a Critical Evaluation.** arXiv preprint, 2026. https://arxiv.org/abs/2603.09675.

R23. **TPA-AD: A Two-Stage Pseudo Anomaly-Guided Method for Bearing Time-Series Anomaly Detection.** arXiv preprint, 2026. https://arxiv.org/abs/2606.04073. `NUMBERS UNVERIFIED`, no code found.

R35. Zhu, X., Carpentier, L. and Verbeke, M. **When Foundation Models are One-Liners: Limitations and Future Directions for Time Series Anomaly Detection.** ICLR 2026 (poster). https://openreview.net/forum?id=H27kvyG4qf — also https://iclr.cc/virtual/2026/poster/10010437.

R36. Uray, M., Messineo, S., Kwitt, R. and Huber, S. **Exploring Zero-Shot Foundation Models for Multivariate Time Series Anomaly Detection.** EUROCAST 2026, Springer LNCS. https://arxiv.org/abs/2607.12454.

R37. **Do Time-Series Foundation Models Pay Off for Industrial Monitoring? A Cost-Aware Empirical Study.** CIF26, poster P01013, 2026. https://arxiv.org/abs/2608.22968.

R102. Velasco, P. and Zafra, A. **TSADmetrics: A library for evaluating time series anomaly detection methods.** Neurocomputing, 2026. https://doi.org/10.1016/j.neucom.2026.134154 — `pip install tsadmetrics` https://pypi.org/project/tsadmetrics/ (GPL-3.0) — docs https://tsadmetrics.readthedocs.io/en/latest/.

## B. Time-series foundation models

R24. Amazon Science. **Chronos-2 / Chronos-Bolt.** Weights https://huggingface.co/amazon/chronos-2 and https://huggingface.co/amazon/chronos-bolt-base (Apache-2.0) — technical report https://arxiv.org/abs/2510.15821 — code https://github.com/amazon-science/chronos-forecasting.

R25. Google Research. **TimesFM 2.5.** https://huggingface.co/google/timesfm-2.5-200m-pytorch — also https://huggingface.co/google/timesfm-2.5-200m-transformers — code https://github.com/google-research/timesfm. Apache-2.0 up to v2.5; `SECONDARY-SOURCE` report that v3.0 restricts commercial use — pin 2.5 and read the licence file before use.

R26. Shi, X. et al. **Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts.** ICLR 2025 Spotlight. https://arxiv.org/abs/2409.16040 — weights https://huggingface.co/Maple728/TimeMoE-200M (Apache-2.0) — code https://github.com/Time-MoE/Time-MoE.

R27. Salesforce AI Research. **Moirai-2.0-R-small.** https://huggingface.co/Salesforce/moirai-2.0-R-small (**CC-BY-NC-4.0, non-commercial**) — code https://github.com/SalesforceAIResearch/uni2ts.

R28. NX-AI. **TiRex.** https://huggingface.co/NX-AI/TiRex (NXAI Community Licence, not OSI) — paper arXiv:2505.23719 — `pip install tirex-ts`. Companion: Auer, A., Klotz, D., Böck, S. and Hochreiter, S. **Pre-trained Forecasting Models: Strong Zero-Shot Feature Extractors for Time Series Classification.** NeurIPS 2025 Workshop on Recent Advances in Time Series Foundation Models. https://arxiv.org/abs/2510.26777.

R29. Datadog AI Research. **Toto and the BOOM benchmark.** https://arxiv.org/abs/2505.14766 — code https://github.com/DataDog/toto (Apache-2.0) — blog https://www.datadoghq.com/blog/ai/toto-boom-unleashed/. *The anomaly-detection claim is vendor framing of forecast-residual thresholding, not a peer-reviewed AD result.*

R30. IBM Research. **TSPulse (granite-timeseries-tspulse-r1).** https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1 (Apache-2.0) — paper https://arxiv.org/abs/2505.13033 — `pip install granite-tsfm` https://pypi.org/project/granite-tsfm/.

R31. IBM Research. **Tiny Time Mixers (granite-timeseries-ttm-r2).** NeurIPS 2024. https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2 (Apache-2.0 for the granite weights; the separate `ibm-research/ttm-research-r2` weights are research-only) — https://research.ibm.com/publications/tiny-time-mixers-ttms-fast-pre-trained-models-for-enhanced-zerofew-shot-forecasting-of-multivariate-time-series--1 — same `granite-tsfm` package as R30.

R32. Prior Labs. **TabPFN-TS (TabPFN-v2 for time series).** NeurIPS 2024 TRL / TSALM workshops. Paper https://arxiv.org/abs/2501.02945 — code https://github.com/PriorLabs/tabpfn-time-series — `pip install tabpfn-time-series` https://pypi.org/project/tabpfn-time-series/. Weights require accepting the Prior Labs licence at ux.priorlabs.ai — not a plain OSI grant.

R33. Tsinghua THUML. **Sundial.** ICML 2025 Oral. https://icml.cc/virtual/2025/oral/47235 — code/weights https://github.com/thuml/Sundial (licence UNVERIFIED).

R34. Aksu, T. et al. **GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation.** arXiv 2024. https://arxiv.org/abs/2410.10393 — leaderboard https://huggingface.co/spaces/Salesforce/GIFT-Eval — blog https://www.salesforce.com/blog/gift-eval-time-series-benchmark/. *Forecasting only; quote ranks only with the data-leakage caveat.*

R38. **TSFM in-context learning for time-series classification of bearing-health status.** ESANN 2026 preprint. https://arxiv.org/abs/2511.15447. `NUMBERS UNVERIFIED` — the abstract names neither the TSFM nor the bearing datasets.

R39. **ChronosAD.** IEEE INDIN 2026. https://arxiv.org/abs/2606.01300 — code https://github.com/intelligolabs/ChronosAD. `NUMBERS UNVERIFIED` beyond the abstract.

R40. Abdouni, A., Voisin, A. and Cerisara, C. **Leveraging Time Series Foundation Models Embeddings for Remaining Useful Life Prediction.** PHM Society European Conference, 2026. https://doi.org/10.36001/phme.2026.v9i1.4906. `NUMBERS UNVERIFIED` beyond the landing page.

R41. **TimeRep: anomaly scoring from intermediate representations of a time-series foundation model.** arXiv 2025. https://arxiv.org/abs/2509.12650.

R42. Martinez Gil, J., O'Donncha, F., Gifford, W. et al. **Adaptive Conformal Anomaly Detection with Time Series Foundation Models for Signal Monitoring.** arXiv, Apr 2026. https://arxiv.org/abs/2604.20122. Code stated to be in the IBM Granite repo.

## C. Time-series classification

R43. Middlehurst, M., Schäfer, P. and Bagnall, A. **Bake off redux: a review and experimental evaluation of recent time series classification algorithms.** Data Mining and Knowledge Discovery 38, 2024. https://arxiv.org/abs/2304.13029 — DOI 10.1007/s10618-024-01022-1 — correction https://link.springer.com/article/10.1007/s10618-024-01040-z.

R44. Ruiz, A. P., Flynn, M., Large, J., Middlehurst, M. and Bagnall, A. **The great multivariate time series classification bake off.** Data Mining and Knowledge Discovery, 2021. https://pubmed.ncbi.nlm.nih.gov/33679210/ (open access, PMC7897627).

R45. Middlehurst, M. et al. **The Multiverse of Time Series Machine Learning: an Archive for Multivariate Time Series Classification.** arXiv, Mar 2026. https://arxiv.org/abs/2603.20352 — code https://github.com/aeon-toolkit/multiverse. `NUMBERS UNVERIFIED` (the classifier list is not enumerated in the abstract).

R46. Dempster, A. et al. **MONSTER: Monash Scalable Time Series Evaluation Repository.** arXiv 2025, accepted at DMLR. https://arxiv.org/abs/2502.15122 — code https://github.com/Navidfoumani/monster — data https://huggingface.co/monster-monash.

R47. Tan, C. W., Dempster, A., Bergmeir, C. and Webb, G. I. **MultiRocket: multiple pooling operators and transformations for fast and effective time series classification.** DMKD 36:1623-1646, 2022. https://arxiv.org/abs/2102.00457 — DOI 10.1007/s10618-022-00844-1 — code https://github.com/ChangWeiTan/MultiRocket.

R48. Dempster, A., Schmidt, D. F. and Webb, G. I. **Hydra: competing convolutional kernels for fast and accurate time series classification.** DMKD 37:1779-1805, 2023. https://doi.org/10.1007/s10618-023-00939-3 — preprint arXiv:2203.13652 — code https://github.com/angus924/hydra.

R49. Dempster, A., Schmidt, D. F. and Webb, G. I. **QUANT: a minimalist interval method for time series classification.** DMKD 38:2377-2402, 2024. https://arxiv.org/abs/2308.00928 — code https://github.com/angus924/quant (repo not opened; QUANT ships in aeon as `QUANTClassifier`).

R50. Faouzi, J. **MomentQuant: an even more minimalist interval method with linear time complexity for time series classification.** arXiv, 4 Sep 2026. https://arxiv.org/abs/2609.05136. No head-to-head vs Hydra/HC2 in the abstract — `NUMBERS UNVERIFIED`.

R51. Schäfer, P. and Leser, U. **WEASEL 2.0: a random dilated dictionary transform for fast, accurate and memory constrained time series classification.** Machine Learning, 2023. https://doi.org/10.1007/s10994-023-06395-w — preprint arXiv:2301.10194 — code https://github.com/patrickzib/dictionary.

R52. Middlehurst, M., Large, J., Flynn, M., Lines, J., Bostrom, A. and Bagnall, A. **HIVE-COTE 2.0: a new meta ensemble for time series classification.** Machine Learning, 2021. https://doi.org/10.1007/s10994-021-06057-9 — preprint arXiv:2104.07551.

R53. Ismail-Fawaz, A., Devanne, M., Berretti, S., Forestier, G. et al. **LITE / LITETime — Light Inception with boosTing tEchniques.** IEEE DSAA 2023. https://germain-forestier.info/publis/dsaa2023.pdf — follow-up "Look Into the LITE" arXiv:2409.02869 — code https://github.com/MSD-IRIMAS/LITE (repo not opened; LITETime ships in aeon).

R54. Foumani, N. M., Tan, C. W., Webb, G. I. and Salehi, M. **ConvTran: Improving position encoding of transformers for multivariate time series classification.** DMKD 38(1):22-48, 2024. https://arxiv.org/abs/2305.16642 — code https://github.com/Navidfoumani/ConvTran.

R55. Feofanov, V. et al. **Mantis: Lightweight Foundation Model for Time Series Classification.** arXiv 2502.15637 (v1 Feb 2025, v2 Jun 2026). https://arxiv.org/abs/2502.15637 — code https://github.com/vfeofanov/mantis — `pip install mantis-tsfm`.

R56. O'Rourke, S., Trisovic, A. and Bertsimas, D. **RocketPFN: Accurate Time Series Classification via In-Context Learning.** arXiv, 19 Jun 2026. https://arxiv.org/abs/2606.21786. Code URL UNVERIFIED (reproducible from aeon Rocket + the `tabpfn` package).

R57. Uribarri, G., Barone, F., Ansuini, A. and Fransén, E. **Detach-ROCKET: sequential feature selection for time series classification with random convolutional kernels.** DMKD 2024. https://arxiv.org/abs/2309.14518 — code https://github.com/gon-uri/detach_rocket.

R58. **Shapelet-Based Bearing Fault Diagnosis Under Interpretability Constraints: A Recording-Level Evaluation.** Electronics (MDPI) 15(14):3035, 2026. https://doi.org/10.3390/electronics15143035. `NUMBERS UNVERIFIED` — full text returned HTTP 403; the reported F1 values come from the indexed abstract.

R59. Lo, A. et al. **Multi-Class Electrical and Mechanical Fault Classification Using Random Convolutional Kernels (SelF-Rocket).** arXiv 2608.18716, submitted to INTCEC 2026. https://arxiv.org/abs/2608.18716.

R60. Lee, W. J., Kumar, A., Vidyaratne, L., Rao, A. R., Farahat, A. and Gupta, C. **An ensemble of convolution-based methods for fault detection using vibration signals.** IEEE ICPHM 2023. https://arxiv.org/abs/2305.05532.

R61. Foumani, N. M., Miller, L., Tan, C. W., Webb, G. I., Forestier, G. and Salehi, M. **Deep Learning for Time Series Classification and Extrinsic Regression: A Current Survey.** ACM Computing Surveys 56(9), Article 217, 2024. https://dl.acm.org/doi/10.1145/3649448 — code https://github.com/Navidfoumani/TSC_Survey.

R62. Middlehurst, M., Ismail-Fawaz, A. et al. **aeon: a Python Toolkit for Learning from Time Series.** JMLR 25(289):1-10, 2024. http://jmlr.org/papers/v25/23-1444.html — preprint arXiv:2406.14231 — code https://github.com/aeon-toolkit/aeon (BSD-3-Clause).

R63. aeon documentation. **Benchmarking: retrieving and comparing against published results.** https://www.aeon-toolkit.org/en/latest/examples/benchmarking/published_results.html.

## D. Rail passenger doors

R64. Ham, S., Han, S.-Y., Kim, S., Park, H. J., Park, K.-J. and Choi, J.-H. **A Comparative Study of Fault Diagnosis for Train Door System: Traditional versus Deep Learning Approaches.** Sensors 19(23):5160, 2019. https://doi.org/10.3390/s19235160 (CC BY) — PDF https://www.mdpi.com/1424-8220/19/23/5160/pdf?version=1574931220.

R65. Kim, S., Kim, N. H. and Choi, J.-H. **Information Value-Based Fault Diagnosis of Train Door System under Multiple Operating Conditions.** Sensors 20(14):3952, 2020. https://doi.org/10.3390/s20143952 (CC BY).

R66. Shiao, Y., Gadde, S. and Bollepelly, V. **Wavelet-Based Health Monitoring Approach for Train Door Actuation Using Motor Current Analysis.** Sensors 26(9):2898, 2026. https://doi.org/10.3390/s26092898 (CC BY) — PDF https://www.mdpi.com/1424-8220/26/9/2898/pdf?version=1778047874.

R67. Shiao, Y., Gadde, S. and Liu, C.-Y. **Wavelet-Based Analysis of Motor Current Signals for Detecting Obstacles in Train Doors.** Applied Sciences 15(1):25, 2025. https://doi.org/10.3390/app15010025 (CC BY) — PDF https://www.mdpi.com/2076-3417/15/1/25/pdf?version=1735044232.

R68. Song, X., Tang, Y., Xia, Z., Zhang, Z., Kang, Y. and Li, X. **Subway door fault prediction employing stacking ensemble learning.** Scientific Reports, 2026. https://doi.org/10.1038/s41598-026-43371-5 (CC BY-NC-ND) — PDF https://www.nature.com/articles/s41598-026-43371-5.pdf — PubMed 41872356.

R69. PHM Society. **PHM Europe 2026 Conference Data Challenge — subway ticket-validation door servomotor run-to-failure (PIMSSIS test bench).** https://data.phmsociety.org/phm-europe-2026-conference-data-challenge/ — data https://phm-datasets.s3.us-east-1.amazonaws.com/Data_Challenge_PHME2026_dataset.zip (1.6 GB) — documentation https://data.phmsociety.org/wp-content/uploads/sites/9/2026/05/Data_Challenge_2026.pdf — leaderboard https://data.phmsociety.org/phm-europe-2026-data-challenge-leader-board/ — org https://github.com/PHME-Datachallenge. Licence not stated on the landing page — `UNVERIFIED`. Cite also Soualhi et al., Computers in Industry 144:103766 (2023), DOI 10.1016/j.compind.2022.103766.

R70. Yang, Ji and Li. **A Similarity-Based Ensemble Framework for Remaining Useful Life Prediction of a Subway Door System.** PHM Society European Conference 9(1), 2026. https://doi.org/10.36001/phme.2026.v9i1.4986.

R71. Mannone, M., Herrmann, ... and Dazer, M. **Gated Residual End-of-Life Prediction for Variable-Prefix RUL in the PHME2026 Data Challenge.** PHM Society European Conference 9(1), 2026. https://doi.org/10.36001/phme.2026.v9i1.4911. `NUMBERS UNVERIFIED`.

R72. Shimizu, S., Perinpanayagam, S., Namoano, B. and Starr, A. **Real-Time Prognostics and Health Management Without Run-to-Failure Data on Railway Assets.** IEEE Access 11:28724-28734, 2023. https://doi.org/10.1109/ACCESS.2023.3259221 — open copy https://eprints.whiterose.ac.uk/id/eprint/226980/.

R73. Dinmohammadi, F., Alkali, B., Shafiee, M., Bérenguer, C. and Labib, A. **Risk Evaluation of Railway Rolling Stock Failures Using FMECA Technique: A Case Study of Passenger Door System.** Urban Rail Transit 2(3-4):128-145, 2016. https://doi.org/10.1007/s40864-016-0043-z (CC BY) — PDF https://discovery.ucl.ac.uk/id/eprint/10107612/1/Dinmohammadi2016_Article_RiskEvaluationOfRailwayRolling.pdf.

R74. Alkali, B. (Glasgow Caledonian University) and Orsi (SNC-Lavalin). **Implementing Condition Based Maintenance for Rolling Stock System — ScotRail Class 158 door RCM case study.** IMechE Scottish R&D Centre presentation, 2018. https://nearyou.imeche.org/docs/default-source/scottish-rd-centre---past-presentations/180215-glasgow-caledonian-university---class-158-door-reliability.pdf?sfvrsn=4. *Presentation slides, not peer reviewed.*

R75. Sun, Y., Xie, G., Cao, Y. and Wen, T. **Strategy for Fault Diagnosis on Train Plug Doors Using Audio Sensors.** Sensors 19(1):3, 2019. https://doi.org/10.3390/s19010003 (CC BY) — PDF mirror https://pdfs.semanticscholar.org/f2e9/fffeb884a4b36d853d6d7e34d20be82056b2.pdf.

R76. Sun, Y. et al. **Fault Diagnosis of Train Plug Door Based on a Hybrid Criterion for IMFs Selection and Fractional Wavelet Package Energy Entropy.** IEEE Transactions on Vehicular Technology, 2019. https://doi.org/10.1109/TVT.2019.2925903. `NUMBERS UNVERIFIED` (paywalled; Crossref metadata only).

R77. Sun, Y. et al. **Fault diagnosis for train plug door using weighted fractional wavelet packet decomposition energy entropy.** Accident Analysis & Prevention 159:106549, 2022. https://doi.org/10.1016/j.aap.2021.106549. `NUMBERS UNVERIFIED` (paywalled).

R78. Shi, J., Lu, Y., Jiang, B., Zhi, P. and Xu, ... **An Unsupervised Anomaly Detection Method Based on Density Peak Clustering for Rail Vehicle Door System.** 2019 Chinese Control And Decision Conference (CCDC). https://doi.org/10.1109/CCDC.2019.8833427. `NUMBERS UNVERIFIED` (paywalled; Crossref metadata only).

R79. **Incipient anomaly detection for railway vehicle door system based on adaptive mean shift clustering.** 2017 Chinese Automation Congress (CAC). https://doi.org/10.1109/CAC.2017.8242967. `NUMBERS UNVERIFIED`.

R80. **Incipient Fault Diagnosis Method of Railway Vehicle Door System Based on Random Forest.** 2019 Chinese Control Conference (CCC). https://doi.org/10.23919/ChiCC.2019.8865741. `NUMBERS UNVERIFIED`.

R81. **Performance Degradation Analysis of Railway Vehicle Door System Based on Density Peak Clustering.** 2021 CAA SAFEPROCESS. https://doi.org/10.1109/SAFEPROCESS52771.2021.9693586. `NUMBERS UNVERIFIED`.

R82. Han, Z., Francois, M., Samé, A., Bouillaut, L., Oukhellou, L., Aknin, P. (IFSTTAR) and Branger, G. (Bombardier Transport France). **Online predictive diagnosis of electrical train door systems.** WCRR 2013, Sydney. https://hal.science/hal-00863798. *Outside the 2015-2026 window; included as manufacturer-coauthored provenance.* `NUMBERS UNVERIFIED`.

R83. **Industrial fault diagnosis: Pneumatic train door case study.** Proc. IMechE Part F: J. Rail and Rapid Transit, 2002. https://doi.org/10.1243/095440902760213602. *Outside the window; background/lineage only.* `NUMBERS UNVERIFIED`.

R84. **Research on Fault Prediction Method of Elevator Door System Based on Transfer Learning.** Sensors 24(7):2135, 2024. https://doi.org/10.3390/s24072135 (CC BY). Related verified items: https://doi.org/10.1109/CASE49997.2022.9926596 and https://doi.org/10.3390/app15137017. `NUMBERS UNVERIFIED`.

R85. CEN. **EN 14752 — Railway applications: bodyside entrance systems for rolling stock** (current edition EN 14752:2025). https://standards.iteh.ai/catalog/standards/cen/039a77e7-4665-46f5-bfaf-9b5c0cdbbe6b/en-14752-2025. **Standard text not read** — the obstruction-force band (100-200 N) and 0.3 s reaction time quoted in this repo come from R67, which claims compliance; treat clause numbers and exact force limits as `UNVERIFIED` until the standard is purchased.

R86. Cranfield University. **Detection and Diagnosis of Faults in Linear Actuators.** Cranfield Online Research Data (CORD). https://doi.org/10.17862/cranfield.rd.5097649. Fault classes (normal, backlash, lack of lubrication, spalling) and the `.mat` distribution were confirmed from `SECONDARY-SOURCE` pages that use the dataset (https://www.ti.com/technologies/edge-ai/use-cases/linear-actuator-fault-classification.html and an STMicroelectronics NanoEdge AI case study), **not** from the CORD landing page — verify classes and licence at the DOI before relying on them.

## E. Brake air supply and compressors

R87. Veloso, B., Ribeiro, R. P., Gama, J. and Pereira, P. M. **The MetroPT dataset for predictive maintenance.** Scientific Data 9, 2022. https://doi.org/10.1038/s41597-022-01877-3 (CC BY 4.0) — open text Europe PMC PMC9747912 — preprint arXiv:2207.05466.

R88. Davari, N., Veloso, B., Ribeiro, R. P. and Gama, J. **MetroPT-3 Dataset.** UCI Machine Learning Repository #791, 2023. https://doi.org/10.24432/C5VW3R — https://archive.ics.uci.edu/dataset/791/metropt+3+dataset (CC BY 4.0).

R89. Veloso, B., Gama, J., Ribeiro, R. P. and Pereira, P. M. **MetroPT2: A Benchmark dataset for predictive maintenance.** Zenodo, 2022. https://doi.org/10.5281/zenodo.7766691 (CC BY 4.0).

R90. Barros, M., Veloso, B., Pereira, P. M., Ribeiro, R. P. and Gama, J. **Failure Detection of an Air Production Unit in Operational Context.** IoT Streams for Data-Driven Predictive Maintenance / ITEM 2020, CCIS 1325, Springer. https://doi.org/10.1007/978-3-030-66770-2_5. `NUMBERS UNVERIFIED` (paywalled; Crossref metadata only).

R91. Davari, N., Veloso, B., Ribeiro, R. P., Pereira, P. M. and Gama, J. **Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry.** IEEE DSAA 2021. https://doi.org/10.1109/DSAA53316.2021.9564181. `NUMBERS UNVERIFIED` (paywalled).

R92. Davari, N., Veloso, B., Ribeiro, R. P. and Gama, J. **Detecting and Explaining Anomalies in the Air Production Unit of a Train.** ACM SAC 2024. https://doi.org/10.1145/3605098.3635906. `NUMBERS UNVERIFIED` — the F1 ≈ 90.8 % figure circulating in secondary sources was **not** confirmed; do not quote it.

R93. Jakobs, M., Veloso, B. and Gama, J. **Interpretable rules for online failure prediction: a case study on Metro do Porto datasets.** International Journal of Data Science and Analytics, 2026. https://doi.org/10.1007/s41060-026-01039-3 (CC BY 4.0) — preprint https://arxiv.org/abs/2502.07394 — code https://github.com/MatthiasJakobs/metro-xai.

R94. Silva, ..., Veloso, B. and Gama, J. **Predictive Maintenance, Adversarial Autoencoders and Explainability.** ECML PKDD 2023 Applied Data Science track, LNCS 14174. https://doi.org/10.1007/978-3-031-43430-3_16. `NUMBERS UNVERIFIED` (publisher abstract only).

R95. Zafra, A., Veloso, B. and Gama, J. **Early Failure Detection for Air Production Unit in Metro Trains.** HAIS 2024, LNCS 14857. https://doi.org/10.1007/978-3-031-74183-8_28. `NUMBERS UNVERIFIED`.

R96. Toribio, ..., Veloso, B., Gama, J. and Zafra, A. **A two-stage framework for early failure detection in predictive maintenance: a case study on metro trains.** Neurocomputing, 2026. https://doi.org/10.1016/j.neucom.2025.132506. `NUMBERS UNVERIFIED` (ScienceDirect 403; Crossref lists a CC BY licence).

R97. García-Méndez, S., de Arriba-Pérez, F., Leal, F., Veloso, B., Malheiro, B. and Burguillo-Rial, J. C. **An explainable machine learning framework for railway predictive maintenance using data streams from the metro operator of Portugal.** Scientific Reports, 2025. https://doi.org/10.1038/s41598-025-08084-1 (CC BY 4.0).

R98. Gama, J., Ribeiro, R. P., Mastelini, S., Davari, N. and Veloso, B. **From fault detection to anomaly explanation: a case study on predictive maintenance.** Journal of Web Semantics, 2024. https://doi.org/10.1016/j.websem.2024.100821. `NUMBERS UNVERIFIED` (paywalled).

R99. Meira, J., Veloso, B., Bolón-Canedo, V., Marreiros, G., Alonso-Betanzos, A. and Gama, J. **Data-driven predictive maintenance framework for railway systems.** Intelligent Data Analysis, 2023. https://doi.org/10.3233/IDA-226811.

R100. Chen and Liu. **Anomaly Detection of Railway Air Production Unit Based on Multiscale CNN-Mamba-Transformer.** Lecture Notes in Electrical Engineering, Springer, 2025. https://doi.org/10.1007/978-981-96-3977-9_45. `NUMBERS UNVERIFIED` (Crossref metadata only; dataset not confirmed).

R101. Lee, W.-J. (Dutch Railways / TU Delft). **Anomaly Detection and Severity Prediction of Air Leakage in Train Braking Pipes.** International Journal of Prognostics and Health Management 8(3), 2017. https://doi.org/10.36001/ijphm.2017.v8i3.2662.

R103. Bouketta, ..., Niar, S. and Ouarnoughi, H. **Deep learning for anomaly detection in railway systems: a structured survey.** Engineering Applications of Artificial Intelligence, 2026. https://doi.org/10.1016/j.engappai.2026.115776. `NUMBERS UNVERIFIED` (Crossref metadata only).

R104. Davari, N., Veloso, B., Costa, G. de A., Pereira, P. M., Ribeiro, R. P. and Gama, J. **A Survey on Data-Driven Predictive Maintenance for the Railway Industry.** Sensors 21(17):5739, 2021. https://doi.org/10.3390/s21175739 (CC BY 4.0).

R105. Forbicini, F., Pinciroli Vago, N. O. and Fraternali, P. **Time series analysis in compressor-based machines: a survey.** Neural Computing and Applications, 2025. https://doi.org/10.1007/s00521-025-11065-0 (CC BY 4.0).

R106. Jadhav, ..., Kotecha, K. and Choudhury, ... **A multi-task model for failure identification and GPS assessment in metro trains.** AIMS Environmental Science, 2024. https://doi.org/10.3934/environsci.2024048.

R107. Zhao, ..., Lin, ..., Pang, ... and Gong, ... **Sound Source Localization-Based Gas Leakage Detection in Railway Pneumatic Brake Systems.** Lecture Notes in Electrical Engineering, Springer, 2026. https://doi.org/10.1007/978-981-95-9350-7_14. `NUMBERS UNVERIFIED`.

R108. Steiner, ..., Abdelkader, ..., Helm, ... and Ansari, F. **MetroAT: A Benchmark Dataset for Data-Driven Maintenance in Metro Operations.** Procedia CIRP, 2026. https://doi.org/10.1016/j.procir.2026.03.165 (CC BY-NC-ND 4.0 per Crossref). `NUMBERS UNVERIFIED` — contents, signals and downloadability not confirmed.

## F. Axle-box bearings and bogies

R109. Entezami, M., Roberts, C., Weston, P., Stewart, E., Amini, A. and Papaelias, M. **Perspectives on railway axle bearing condition monitoring.** Proc. IMechE Part F 234(1):17-31, 2020. https://doi.org/10.1177/0954409719831822 — open AAM https://pure-oai.bham.ac.uk/ws/files/57377840/PERSPECTIVES_ON_RAILWAY_AXLE_BEARING_CONDITION_MONITORING_final_PartF_Revised_final_no_track_changes.pdf.

R110. Yang, ..., Wu, ..., Shao, ..., Lu, ..., Zhang, ..., Xu, ... and Chen, ... **Fault detection of high-speed train axle bearings based on a hybridized physical and data-driven temperature model.** Mechanical Systems and Signal Processing 208:111037, 2024. https://doi.org/10.1016/j.ymssp.2023.111037. `NUMBERS UNVERIFIED` (ScienceDirect 403; the ~2.25 million km calibration figure is abstract-level).

R111. **A Time-Delay Low-Rank Reconstruction and Attention-BP-LSTM Framework for Axle-Box Bearing Temperature Prediction in Railway Vehicles.** Sensors 26(16):5137, 2026. https://doi.org/10.3390/s26165137 (CC BY 4.0) — PubMed 42655446. `NUMBERS UNVERIFIED` (full text blocked).

R112. Tarawneh, C., Aranda, J., Hernandez, V., Crown, S. and Montalvo, J. **An investigation into wayside hot-box detector efficacy and optimization.** International Journal of Rail Transportation 8(3):264-284, 2020. https://doi.org/10.1080/23248378.2019.1636721 — author copy https://www.utrgv.edu/railwaysafety/_files/documents/research/mechanical/ijrt_wayside-hbd-investigation.pdf.

R113. CEN / BSI. **EN 15437-1:2009+A1:2022 — Railway applications. Axlebox condition monitoring. Interface and design requirements. Part 1: Track side equipment and rolling stock axlebox.** https://www.en-standard.eu/bs-en-15437-1-2009-a1-2022-railway-applications-axlebox-condition-monitoring-interface-and-design-requirements-track-side-equipment-and-rolling-stock-axlebox/. **Standard text not read.** The alarm values in circulation (absolute axlebox temperature > 95 °C, differential > 56 °C) are `SECONDARY-SOURCE` (EKE-Electronics; RSSB RIS-2714-RST draft) — mark `UNVERIFIED` wherever quoted.

R114. CEN. **EN 15437-2:2012+A1:2022 — Axlebox condition monitoring, Part 2: Performance and design requirements of on-board systems for temperature monitoring.** https://standards.iteh.ai/catalog/standards/cen/9b0a02fb-fad2-4fe9-b0ab-49a1f84550ee/en-15437-2-2012a1-2022. Catalogue page only; cite by number, do not paraphrase clauses.

R115. Vieira, ..., Bauler, ..., Rosa, ... and Silva, ... **Towards a more realistic evaluation of machine learning models for bearing fault diagnosis.** Mechanical Systems and Signal Processing vol. 258, 2026 — preprint https://arxiv.org/abs/2509.22267.

R116. Zhao, Z., Zhang, Q., Yu, X., Sun, C., Wang, S., Yan, R. and Chen, X. **Applications of Unsupervised Deep Transfer Learning to Intelligent Fault Diagnosis: A Survey and Comparative Study (UDTL benchmark).** IEEE TIM 2021 — preprint https://arxiv.org/abs/1912.12528 — code https://github.com/ZhaoZhibin/UDTL (repo licence UNVERIFIED).

R117. Randall, R. B. and Antoni, J. **Rolling element bearing diagnostics — A tutorial.** Mechanical Systems and Signal Processing 25(2):485-520, 2011. https://doi.org/10.1016/j.ymssp.2010.07.017. *Outside the window; non-negotiable physics baseline.*

R118. Huang, H. and Baddour, N. **Bearing Vibration Data under Time-varying Rotational Speed Conditions.** Mendeley Data, 2018/2019. https://doi.org/10.17632/v43hmbwxpm.2 — record https://data.mendeley.com/datasets/v43hmbwxpm/2 — data paper https://www.sciencedirect.com/science/article/pii/S2352340918314124.

R119. Sehri, M., Dumond, P. et al. **University of Ottawa constant load and speed rolling-element bearing vibration and acoustic fault signature datasets (UORED-VAFCLS).** Data in Brief, 2023. https://www.sciencedirect.com/science/article/pii/S2352340923004456 (CC BY) — Mendeley record https://data.mendeley.com/datasets/y2px5tg92h/5 — open mirror PMC10331275.

R120. Kreuzer, M., Schmidt, D. and Kellermann, W. **Novel features for the detection of bearing faults in railway vehicles.** Inter-Noise 2021 — preprint https://arxiv.org/abs/2304.08249.

R121. Kreuzer, M., Schmidt, D., Wokusch, ... and Kellermann, W. **Airborne Sound Analysis for the Detection of Bearing Faults in Railway Vehicles with Real-World Data.** IEEE ICPHM 2023 — preprint https://arxiv.org/abs/2304.07307.

R122. **A Novel Method for Bearing Fault Diagnosis under Variable Speed Based on Envelope Spectrum Fault Characteristic Frequency Band Identification.** Sensors 23(9):4338, 2023. https://www.mdpi.com/1424-8220/23/9/4338 (CC BY 4.0) — mirror PMC10181550.

R123. **Performance Degradation Assessment of Railway Axle Box Bearing Based on Combination of Denoising Features and Time Series Information (DRSN-LSTM).** Sensors 23(13):5910, 2023. https://doi.org/10.3390/s23135910 (CC BY 4.0) — mirror PMC10346339.

R124. Wang, B., Lei, Y., Li, N. and Li, N. **XJTU-SY Rolling Element Bearing Accelerated Life Test Datasets.** https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets — paper IEEE Trans. Reliability 69(1):401-412, 2020, DOI 10.1109/TR.2018.2882682 — preprocessed mirror https://data.mendeley.com/datasets/mpn45f4gxc/1.

R125. Paderborn University, Chair of Design and Drive Technology. **KAt-DataCenter bearing dataset.** https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download (CC BY-NC 4.0) — summary https://github.com/jonathanwvd/awesome-industrial-datasets/blob/master/markdown/paderborn_university_bearing_dataset.md.

R126. **Fault Diagnosis of Bogie Bearings in High-Speed Train: A Review.** Journal of Failure Analysis and Prevention 25(4):1539-1575, 2025. https://doi.org/10.1007/s11668-025-02225-4. `NUMBERS UNVERIFIED` (Springer page redirected to auth).

R127. **Wayside acoustic diagnosis of defective train bearings based on signal resampling and information enhancement.** Journal of Sound and Vibration, 2013. https://www.sciencedirect.com/science/article/abs/pii/S0022460X13004859. *Outside the window.* Related open item: PMC4610586. The "~39 North American acoustic detectors (20 RailBAM, 19 TADS), FRA 2019" figure is `SECONDARY-SOURCE` and `UNVERIFIED`.

## G. Additional datasets

R128. Mazzoleni, M., Scandella, M., Previdi, F. and Pispola, G. **First endurance activity of a Brushless DC motor for aerospace applications (REPRISE).** Mendeley Data V2, 2019. https://doi.org/10.17632/m58bdhy2df.2 (CC BY 4.0).

R129. PHM Society / GE / UTK. **Servomotor-Driven Ballscrew Mechanism Degradation Data Set (FMCRD).** https://data.phmsociety.org/servomotor_dataset/ — data https://phm-datasets.s3.amazonaws.com/GE-UTK/FMCRD_Data.zip (21.3 GB). Licence not stated — `UNVERIFIED`.

R130. Huawei Munich Research Center. **Predictive maintenance dataset (elevator door IoT).** Zenodo, 2020. https://doi.org/10.5281/zenodo.3653909.

R131. **Multimodal Signal Dataset for Fault Detection in PMSM-Driven Elevators Under Real Operating Conditions.** Zenodo, 2025. https://doi.org/10.5281/zenodo.15613954 (CC BY 4.0).

R132. Veloso, B. et al. **MetroPT (MetroPT-1) — Air Production Unit benchmark, Metro do Porto.** Zenodo, 2022. https://doi.org/10.5281/zenodo.6854240 (CC BY 4.0).

R133. **APS Failure at Scania Trucks.** UCI ML Repository #421 / IDA 2016 Industrial Challenge. https://doi.org/10.24432/C51S51 — https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks (CC BY 4.0).

R134. **SCANIA Component X dataset.** Scientific Data 12:493, 2025. https://doi.org/10.5878/bnh5-ka77 — catalogue https://researchdata.se/en/catalogue/dataset/2024-34 — paper DOI 10.1038/s41597-025-04802-6 — preprint arXiv:2401.15199 (CC BY 4.0).

R135. IIT Kanpur IDEA Lab. **Air Compressor acoustic health-state dataset.** https://www.iitk.ac.in/idea/datasets/ — companion paper Verma et al., IEEE Trans. Reliability 65(1):291-309, 2016. No formal licence stated; citation required. Sampling rate reported inconsistently in the literature — `UNVERIFIED` until read from the download readme.

R136. **Refinery Compressor Sensor Data, One-Year Dataset.** Zenodo, 2025. https://doi.org/10.5281/zenodo.14866092 (CC BY 4.0). No fault labels documented.

R137. Sehri, M. and Dumond, P. **University of Ottawa Electric Motor Dataset — Vibration and Acoustic Faults under Constant and Variable Speed (UOEMD-VAFCVS).** Mendeley Data, 2023. https://data.mendeley.com/datasets/msxs4vj48g/2 — DOI 10.17632/msxs4vj48g.1 — data paper DOI 10.1016/j.dib.2024.110144. Licence `UNVERIFIED`.

R138. Aimiyekagbon, O. K. et al. **Run-to-failure data set of ball bearings subjected to time-varying load and speed conditions.** Zenodo, 2024. https://doi.org/10.5281/zenodo.10805043 (CC BY 4.0).

R139. Politecnico di Torino. **Dataset of Vibration, Temperature and Speed Measurements for Multiple Types of Localized Defects on Spherical Roller Bearings across Multiple Operating Conditions.** Zenodo, 2024. https://doi.org/10.5281/zenodo.13913254. **Restricted Use Agreement** — written consent required, no third-party sharing.

R140. CITEF, Universidad Politécnica de Madrid. **Bearing Database series — railway axlebox test rig.** Zenodo: rolling-element defects https://doi.org/10.5281/zenodo.3898942 ; combined failure https://doi.org/10.5281/zenodo.5084405 ; isolated cases https://doi.org/10.5281/zenodo.8241764 (all CC BY 4.0) ; combined failure test 2 (restricted access) https://doi.org/10.5281/zenodo.20761127.

R141. **HUST bearing — a practical dataset for ball bearing fault diagnosis.** Mendeley Data V3, 2023. https://data.mendeley.com/datasets/cbv7jyx4p9/3 — BMC Research Notes DOI 10.1186/s13104-023-06400-4 — preprint arXiv:2302.12533. Licence `UNVERIFIED`.

R142. Case Western Reserve University. **CWRU Bearing Data Center.** https://engineering.case.edu/bearingdatacenter/download-data-file. No licence or usage terms stated — `UNVERIFIED`. (The `csegroups.case.edu` URL cited in most papers is dead.)

R143. NASA Prognostics Center of Excellence, mirrored by PHM Society. **IMS (Univ. of Cincinnati) bearing and FEMTO/PRONOSTIA IEEE PHM 2012 bearing datasets.** https://data.phmsociety.org/nasa/ — IMS https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip — FEMTO https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip — PCoE index https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ — mirror https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset.

R144. **MFPT bearing fault dataset.** Third-party copy https://figshare.com/articles/dataset/MFPT_zip/28606802. Licence `UNVERIFIED`.

R145. PHM Society. **PHM 2017 Data Challenge — rail vehicle bogie suspension fault detection.** https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2017/phm-data-challenge-5/. Published leaderboard: K2 0.762, Tony2017 0.706, KTX 0.654.

R146. **PHM-Beijing 2024 Data Challenge / BJTU-RAO bogie datasets — subway train transmission system.** https://arxiv.org/abs/2504.07155 (accepted at IEEE ICPHM 2025). **Download route unconfirmed** — the stated host `2024.globalphm.org` did not resolve (DNS failure), and the claimed companion DOI 10.1109/TII.2025.3553042 returned 404. Channel counts differ between sources (21 vs 24).

R147. **MCC5-THU gearbox benchmark datasets (variable working conditions).** https://github.com/liuzy0708/MCC5-THU-Gearbox-Benchmark-Datasets — https://ieee-dataport.org/documents/multi-mode-fault-diagnosis-datasets-gearbox-under-variable-working-conditions — preprint arXiv:2403.12521. Licence `UNVERIFIED`. Use version 2 (v1 had a torque sign-reversal error).

R148. PHM Society. **PHM North America 2026 Data Challenge — gearbox run-to-failure with tooth imaging.** https://data.phmsociety.org/phm-north-america-2026-conference-data-challenge/. Licence not specified — `UNVERIFIED`.

---

## H. First-principles physics, operating environment and simulator constants

*Added in the W1 update pass to close the "every numeric constant in the simulator is uncited" gap.*

R149. **Temperature Characteristics of Axle-Box Bearings Under Wheel Flat Excitation.** *Lubricants* 13(1):19, 2025. https://doi.org/10.3390/lubricants13010019 — full text retrieved from https://mdpi-res.com/d_attachment/lubricants/lubricants-13-00019/article_deploy/lubricants-13-00019.pdf. Complete lumped thermal-network model of a railway axle box with every convection correlation given explicitly (external axle-box-to-air forced convection `h_a = 0.03·(k_air/(2R_a))·(2·u_air·R_a/ν_air)^0.57`, Eq. 25 — i.e. `h ∝ v^0.57`; inner-ring/lubricant Eq. 20; roller Eq. 23; fixed outer ring Eq. 24; radiation Eq. 26). Field validation Table 1 (8 cars of a high-speed train) and a wheel-flat severity sweep. Conditions: Chinese high-speed EMU double-row tapered roller axle-box bearing, 200-350 km/h, ambient 15-25 °C.

R150. **Analysis of vibration and temperature on the axle box bearing of a high-speed train.** *Vehicle System Dynamics* 58(10), 2019. https://doi.org/10.1080/00423114.2019.1645340 — open accepted manuscript retrieved from https://pure.hud.ac.uk/ws/files/17580727/VSD_Manuscript_R3_PA.pdf. Palmgren friction-torque heat-generation model for an axle box (`P = M·ω`, `M = M0 + M1 + M2`, Eqs. 7-11); 128 temperature sensors per train across 5 trains, 11 Aug - 19 Nov 2016; measured healthy axle-box vertical acceleration RMS and bearing geometry.

R151. Peters, M. F. E. (Netherlands Railways). **Early Warnings for failing Train Axle Bearings based on Temperature.** Annual Conference of the PHM Society, vol. 9 no. 1, 2017. https://doi.org/10.36001/phmconf.2017.v9i1.2451 — PDF https://papers.phmsociety.org/index.php/phmconf/article/download/2451/1415. Operational in-service alarm rules from a real fleet (~3000 carriages, 131 trains of one series, 60 million wayside HotBox measurements over 2 years, 27 measurement locations): side-difference feature, four alarm levels, a slow-degradation 3.5σ rule, a measurement-station data-quality guard, and 1-3 months of lead time over existing methods.

R152. Meteorological Service Singapore / NEA. **Climate of Singapore** — climate normals 1991-2020. http://www.weather.gov.sg/climate-climate-of-singapore/. 24-hour mean air temperature 26.8 °C (Dec/Jan) to 28.6 °C (May); mean daily maximum 30.5-32.4 °C; mean daily minimum 24.3-25.7 °C; mean annual relative humidity ≈82 %, monthly daily-mean RH 80.7-85.5 %, daily maximum RH 93.0-96.5 %, daily minimum RH 61.4-68.0 %; mean annual rainfall 2113.3 mm over 171 rain days.

R153. Meteorological Service Singapore / NEA. **Historical Extremes — Temperature.** http://www.weather.gov.sg/climate-historical-extremes-temperature/. Highest daily maximum 37.0 °C (Ang Mo Kio Ave 5, 13 May 2023; Tengah, 17 Apr 1983); lowest daily maximum 21.2 °C (Mount Faber, 6 Jan 1934); highest daily minimum 30.0 °C (Pasir Panjang Terminal, 19 Jun 2013); lowest daily minimum 19.0 °C (Paya Lebar, 14 Feb 1989).

R154. U.S. Department of Energy, EERE / Compressed Air Challenge. **Compressed Air Tip Sheet #3: Minimize Compressed Air Leaks.** 2004. https://www.energy.gov/sites/prod/files/2014/05/f16/compressed_air3.pdf. Leak-flow table in cfm by orifice diameter and supply pressure (at 90 psig: 1/64 in 0.36, 1/32 in 1.46, 1/16 in 5.72, 1/8 in 23.10, 1/4 in 92, 3/8 in 206.6 cfm); correction factors ×0.97 well-rounded, ×0.61 sharp-edged; leakage scales with the **square of orifice diameter** and rises with supply pressure; leaks often waste 20-30 % of compressor output, with 5-10 % a realistic target.

R155. U.S. Department of Energy, EERE / Compressed Air Challenge. **Improving Compressed Air System Performance: A Sourcebook for Industry.** 2003. https://www1.eere.energy.gov/manufacturing/tech_assistance/pdfs/compressed_air_sourcebook.pdf. Twin-tower desiccant dryer: rated at −40 °F (−40 °C) pressure dew point, purge air **10-18 %** of the dryer's rating for pressure-swing regenerative types, 3-5 psi pressure drop, regeneration by depressurising the tower and passing previously dried purge air through the bed, built-in regeneration cycle based on time, dew point or both; **"Because dryer ratings are based upon saturated air at inlet, the geographical location is not a concern."** Compressor duty-cycle leak estimate `Leakage(%) = T·100/(T+t)` (T on-load minutes, t off-load minutes), <10 % well-maintained, 20-30 % poorly maintained. Pressure-decay leak estimate and **receiver charge/discharge sizing `V = T·C·Pa/(P1 − P2)`**. Load/unload detail: sump relief on unload ≈40 s, repressurise on reload ≈3 s.

R156. **49 CFR 232.205 — Class I brake test (brake pipe leakage limit).** US Code of Federal Regulations (FRA), via GovInfo, 2023. https://www.govinfo.gov/content/pkg/CFR-2023-title49-vol4/xml/CFR-2023-title49-vol4-sec232-205.xml. "Brake pipe leakage shall not exceed **5 psi per minute** or air flow shall not exceed **60 cubic feet per minute (CFM)**", measured after a 20 psi service reduction with the brake valve in neutral for 45-60 s. Conditions: US freight brake pipe, not a metro APU main reservoir.

R157. Jain, P. and Bhosle, S. **Analysis of vibration signals caused by ball bearing defects using time-domain statistical indicators.** *International Journal of Advanced Technology and Engineering Exploration* 9(90), 2022. http://dx.doi.org/10.19101/IJATEE.2021.875416 — PDF https://www.accentsjournals.org/PaperDirectory/Journal/IJATEE/2022/5/9.pdf. Measured RMS / crest factor / kurtosis versus defect size **and** shaft speed (SKF 6205-2RS, CWRU drive-end, 1730-1797 rpm, defects 0.007-0.028 in). Healthy: RMS 0.064-0.074 g, kurtosis **2.76-2.96**, crest factor **4.22-5.59**. Inner-race kurtosis by defect size 5.56 → 21.69 → 8.06 → **3.29** (strongly non-monotonic). Conditions: 35 mm-bore lab motor bearing at ≈29 Hz, narrow 4 % speed span — usable for the *shape* of a severity law, not for absolute g values and not for a speed exponent.

R158. Saruhan, H., Saridemir, S., Cicek, A. and Uygur, I. **Vibration Analysis of Rolling Element Bearings Defects.** *Journal of Applied Research and Technology* 12(3):384-395, 2014. https://doi.org/10.1016/S1665-6423(14)71620-7 — PDF https://jart.icat.unam.mx/index.php/jart/article/download/201/198. Maximum vibration-spectrum peak (gPk) at four shaft speeds (17, 25, 33, 41 Hz), unloaded and with a 5.04 kg loader, for healthy / outer-race / inner-race / ball / combined defects; BPFO, BPFI and BSF factors for the test bearing. The only source in this set with a measured amplitude-vs-speed trend over a factor-2.4 speed range. Conditions: small lab rig, spectrum *peak* amplitudes not band RMS.

R159. Huang, H. and Baddour, N. **Bearing vibration data collected under time-varying rotational speed conditions.** *Data in Brief* 21:1745-1749, 2018. https://doi.org/10.1016/j.dib.2018.11.019 — full text via Europe PMC https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6249544/fullTextXML. ER16K ball bearings, **200 kHz**, 36 datasets = 3 health states × 4 speed-variation patterns × 3 trials, 10 s each; speed varies continuously *within* a record (e.g. 14.1 → 23.8 Hz, 28.9 → 13.7 Hz). **Defect dimensions, applied load and any temperature channel are absent** — this dataset cannot ground a severity axis or a thermal model.

R160. Botts, A. et al. **Energy efficiency of blower heater non-purge compressed air dryers.** *International Journal of Energy Technology and Policy* 17(3), 2021. https://doi.org/10.1504/IJETP.2021.116321 — PDF https://www.osti.gov/servlets/purl/1863533. Field energy assessments of twin-tower regenerative desiccant dryers at 13 sites: measured/manufacturer purge rate **12 % of flow** for heater-purge twin towers (Table 12), achieved dew-point setpoints −10 to −40 °C, heater run-time fraction 14-57 %; regeneration run on a **timed cycle manually adjusted by season**; low supply pressure reduces purge flow and causes incomplete regeneration. Conditions: US hospitals and manufacturing plants, not rail.

R161. **Theoretical and experimental investigation on the thermal characteristics of double-row tapered roller bearings of high speed locomotive.** *International Journal of Heat and Mass Transfer* 84:1119-1130, 2015. https://doi.org/10.1016/j.ijheatmasstransfer.2014.11.057. **`UNVERIFIED` — paywalled, full text not opened; title and DOI confirmed from the OpenAlex record only. No number from this paper is quoted anywhere in our documents.** Listed as the highest-value remaining target for the axle-box thermal model if a team member has institutional access.

R162. **Analysis of the Temperature Characteristics of High-speed Train Bearings Based on a Dynamics Model and Thermal Network Method.** *Chinese Journal of Mechanical Engineering* 35, 104, 2022. https://doi.org/10.1186/s10033-022-00789-y. **`UNVERIFIED` — gold open access, but every fetch route redirected to a Springer identity endpoint this session; title, venue and DOI confirmed from OpenAlex only. No number from this paper is quoted anywhere in our documents.** Natural cross-check on R149's `h ∝ v^0.57` correlation; ten minutes in a normal browser will open it.

---

## I. Regulatory and standards text (freely available sources only)

*Added in the W1 update pass. Every EU-level item below is a free, full-text source: the OTIF Uniform Technical Prescriptions mirror the EU TSI clause structure and are published in full as PDFs. The paywalled EN standards themselves were **not** read; where a clause number is given it comes from the free TSI/UTP standard-mapping appendices.*

R163. OTIF (APTU Uniform Rules, Appendix F to COTIF 1999). **UTP LOC&PAS — Uniform Technical Prescription, Rolling stock: Locomotives and Passenger Rolling Stock**, in force 01.01.2026. https://otif.org/fileadmin/docs/LegalTexts/COTIF/TechnicalInteroperability/PrescriptionsOtherRules/BasedonAPTU/2026_UTP_LOCPAS-e-in_force.pdf (248 pp., downloaded and text-extracted). Free mirror of EU LOC&PAS TSI Reg. (EU) 1302/2014 clause structure. Clauses used: **4.2.5.5.3(5)** door obstacle detection (qualitative; sensitivity and maximum force delegated to Appendix J-1 index [17]); **Appendix J-1 index [17]** = *EN 14752:2019+A1:2021*, with [17.1] sensitivity → cl. **5.2.1.4.1**, [17.2] maximum force → cl. **5.2.1.4.2.2**, [17.3] emergency-opening manual force → cl. 5.5.1.5; **4.2.3.3.2 / 4.2.3.3.2.1 / 4.2.3.3.2.2** axle bearing condition monitoring (on-board detection mandatory at ≥250 km/h design speed; below that, on-board **or** track-side; "the bearing condition shall be evaluated either by monitoring its temperature, or its dynamic frequencies or some other suitable bearing condition characteristic"; **no temperature threshold of any kind appears in the text**); **Table 0** + Appendix J-1 index [8] = EN 15437-1:2009, trackside target-zone geometry; **4.2.5.5.9(1)** internal emergency-opening device active below **10 km/h**; **4.2.6.1** ambient temperature zones **T1 −25 to +40 °C** (nominal), T2 −40 to +35 °C, T3 −25 to +45 °C.

R164. OTIF. **UTP WAG — Uniform Technical Prescription, Freight Wagons**, in force 2025. https://otif.org/fileadmin/new/3-Reference-Text/3D-Technical-Interoperability/3D1-Prescriptions-and-other-rules/2025_UTP_WAG-e-in_force.pdf. Clause **4.2.3.4** axle bearing condition monitoring with the identical on-board wording and, again, **no temperature alarm value**; **Table 2** trackside target/prohibitive zone geometry per gauge; **Appendix D index [6]** = EN 15437-1:2009+A1:2022, mandatory points 5.1 and 5.2.

R165. OTIF. **UTP PRM — Uniform Technical Prescription, Persons with Reduced Mobility**, in force 01.01.2026. https://otif.org/fileadmin/docs/LegalTexts/COTIF/TechnicalInteroperability/PrescriptionsOtherRules/BasedonAPTU/2026_UTP_PRM-e-in_force.pdf (86 pp., downloaded and text-extracted). Free mirror of EU PRM TSI Reg. (EU) 1300/2014. Clause **4.2.2.3.2** exterior doors: door-opening signal ≥5 s (may cease after 3 s if the door is operated); remote/automatic opening signal ≥3 s from the start of opening; **closing signal starts ≥2 s before the door starts to close** and continues until closed; clear usable width ≥800 mm (≥1000 mm for wheelchair-access level-access doors on trains <250 km/h); control-centre heights 800-1200 mm exterior, 800-1100 mm interior. Clause **4.2.2.3.1(2)** control device operable by the palm at ≤**20 N**; clause **4.2.2.3.3(3)** force to open or close a **manual** door ≤**60 N**; clause 4.2.2.3.3(1) internal automatic/semi-automatic doors "shall incorporate devices that prevent passengers becoming trapped" — qualitative, no number.

R166. European Union Agency for Railways (ERA). **Guide for the application of the LOC&PAS TSI (GUI/LOC&PAS TSI/2023).** https://www.era.europa.eu/system/files/2024-02/LOC-PAS_Guide-2023.pdf (2.7 MB, downloaded and text-extracted). Against TSI point **4.2.3.3.2 Axle bearing condition monitoring** the standards table lists *EN 15437-2:2012+A1:2022* annotated **"Track side system / On-board system (open point)"** — i.e. **no harmonised European requirement exists for on-board axle-bearing monitoring**; each Member State / operator sets its own rule. ERA application guides are non-binding but are official ERA publications.

R167. Estonian Centre for Standardisation and Accreditation (EVS), CEN national member — free catalogue scope text. **EVS-EN 15437-1:2009 — Railway applications. Axlebox condition monitoring. Interface and design requirements. Part 1: Track side equipment and rolling stock axlebox.** https://www.evs.ee/en/evs-en-15437-1-2009. Scope: minimum interface characteristics between trackside Hot Axlebox Detectors and rolling stock, 1435 mm gauge, outboard bearings, speeds up to 250 km/h; rolling-stock requirements in Clause 5, trackside detector equipment in Clause 6. **Decisive exclusion stated in the free scope text: the standard "does not cover Hot Wheel Detectors, temperature measurement methodologies, operational procedures for responding to detector alerts, or maintenance requirements for detector systems."** Status: withdrawn 15.02.2023, superseded by EVS-EN 15437-1:2009+A1:2023.

R168. BSI/CEN reseller catalogue (free scope preview). **BS EN 14752:2025 — Railway applications. Bodyside entrance systems for rolling stock.** https://www.en-standard.eu/bs-en-14752-2025-railway-applications-bodyside-entrance-systems-for-rolling-stock/. 94 pp., published 2025-06-23, ICS 45.060.20 / 45.060.01 / 45.140. Scope applies to passenger bodyside entrance systems of all newly designed railway vehicles "such as **tram, metro**, suburban, main-line and high-speed trains"; excludes crew/equipment-access doors, freight wagon doors and emergency escape hatches. **The numeric obstacle-detection sensitivity and force limits remain behind the paywall** (cl. 5.2.1.4.1 and 5.2.1.4.2.2 per the R163 mapping). Note: the **TSI/UTP-mandated edition is EN 14752:2019+A1:2021**, not the 2025 edition.

R169. BSI/CEN reseller catalogue (free preview). **BS EN 15437-2:2012+A1:2022 — Railway applications. Axlebox condition monitoring. Interface and design requirements. Performance and design requirements of on-board systems for temperature monitoring.** https://www.en-standard.eu/bs-en-15437-2-2012-a1-2022-railway-applications-axlebox-condition-monitoring-interface-and-design-requirements-performance-and-design-requirements-of-on-board-systems-for-temperature-monitoring/. 20 pp., publication 2023-05-18. The free abstract makes **no mention of alarm levels or temperature thresholds**; whether the body contains any numeric threshold is `UNVERIFIED` and must not be asserted.

R170. RSSB. **RIS-2714-RST Issue 1, "Axle Bearing Condition Monitoring".** Standards catalogue entry, published 03 June 2023, status Registered/Live, last reviewed 05 Sep 2024, replaced GERT8014 Iss 2 and GEGN8614 Iss 1. https://www.rssb.co.uk/standards-catalogue/CatalogueItem/RIS-2714-RST-iss-1. Synopsis: "This document sets out requirements and guidance for condition monitoring of axle bearings, whether by trackside or onboard detection systems." **The PDF is login-gated — no threshold value was read. Any numeric alarm attributed to this document is `UNVERIFIED`.** RSSB offers free individual registration, so one team member can read the real values before the pitch.

R171. EKE-Electronics Ltd. **Hot Axle Box Detector (HABD)** product page. https://www.eke-electronics.com/hot-axle-box-detector-habd/. States that the measured temperature "is assessed to determine if it has crossed one of the **four thresholds**" and that "**Configurable alarm levels** support tailored safety responses"; references EN 15437-2, EN 50126, EN 50716, EN 50129. **It publishes no absolute °C and no differential K value** — the "95 °C / 56 K" pair previously attributed to this vendor is not on the page and is withdrawn from our documents.

R172. Trafikverket (Swedish national infrastructure manager). **Network Statement 2026**, sections 2.3.14 and 6.4.1. https://bransch.trafikverket.se/contentassets/69c4090dadc44e6c95782493337a9e5e/ns-2026-version-2026-05-22.pdf (244 pp., text-extracted). Confirms that detector alarm handling is governed by the infrastructure manager's own internal rule **TDOK 2020:0074** ("Detectors: Handling of alarms and measures after established damage", v4.0), not by a European standard; detectors cover overheating, unintended brake application, wheel damage with weighing, and acoustic wheel-bearing detection. NS §5.5.2.1 also states that access to extended traffic information via detectors (API Järnväg, data.trafikverket.se) is **free of charge** — a candidate source of real axle-box temperature measurements.

---

## J. Change-point detection and concept drift

*Added in the W1 update pass to give cross-cutting verdict V4 (score change points, not residual magnitude) an actual model family, a library, and an evaluation method.*

R173. **ruptures** (Python package), v1.1.10. https://pypi.org/pypi/ruptures/json — https://github.com/deepcharles/ruptures/ — user guide https://centre-borelli.github.io/ruptures-docs/user-guide/. Licence **BSD-2-Clause**. Offline change-point search methods `Pelt`, `Binseg`, `BottomUp`, `Window`, `Dynp`, `KernelCPD`, combinable with cost functions `CostL1`, `CostL2`, `CostNormal`, `CostRbf`, `CostCosine`, `CostLinear`, `CostCLinear`, `CostRank`, `CostMl`, `CostAR`; multivariate input supported. Ships `ruptures.metrics` with precision/recall under an annotation margin, Hausdorff metric and Rand index.

R174. Truong, C., Oudre, L. and Vayatis, N. **Selective review of offline change point detection methods.** *Signal Processing* 167, 2020. https://doi.org/10.1016/j.sigpro.2019.107299 (arXiv:1801.00718). Organises all offline CPD into three orthogonal choices — cost function, search method, constraint on the number of changes — explicitly scoped to **multivariate** time series; states that implementations of the main algorithms are provided in `ruptures`.

R175. van den Burg, G. J. J. and Williams, C. K. I. **An Evaluation of Change Point Detection Algorithms (TCPD / TCPDBench).** arXiv:2003.06222 v3 (Feb 2022) — https://doi.org/10.48550/arXiv.2003.06222 — benchmark code https://github.com/alan-turing-institute/TCPDBench (**MIT**). **`NOT PEER-REVIEWED`** (arXiv abs page lists no journal reference). 14 algorithms on 37 annotated real-world series with 5 human annotators each. Metrics: the **covering metric** (adapted from Arbeláez et al. 2010: `C(G,G') = (1/T)·Σ_{A∈G} |A|·max_{A'∈G'} J(A,A')`, J = Jaccard/IoU) and **F-measure with an annotation margin M = 5 time steps**. Headline: with **default** hyperparameters, **binary segmentation** had the highest average performance on univariate series; only with **oracle** hyperparameter tuning did Bayesian online change-point detection beat everything on both univariate and multivariate series.

R176. **claspy** (ClaSPy Python package), v0.2.8. https://pypi.org/pypi/claspy/json — https://github.com/ermshaua/claspy. Licence **BSD 3-Clause**. Exposes `BinaryClaSPSegmentation` (univariate and multivariate), `StreamingClaSPSegmentation` (online) and `AgglomerativeCLaPDetection`. **Hyper-parameter-free**: it determines the number of change points itself, so there is no penalty to calibrate.

R177. Ermshaus, A., Schäfer, P. and Leser, U. **ClaSP: parameter-free time series segmentation.** *Data Mining and Knowledge Discovery* 37:1262-1300, 2023. https://doi.org/10.1007/s10618-023-00923-x (arXiv:2207.13987). Argues that current time-series segmentation algorithms all require domain-dependent hyper-parameters that are hard to tune per dataset; on a benchmark of **107 datasets** ClaSP "outperforms the state of the art in terms of accuracy and is fast and scalable". Peer-reviewed segmentation benchmark — not an industrial multivariate fault benchmark.

R178. **river** (Python package), v0.26.1. https://pypi.org/pypi/river/json — https://github.com/online-ml/river — API overview https://riverml.xyz/latest/api/overview/. Licence **BSD-3-Clause**. `river.drift`: **ADWIN, KSWIN, PageHinkley**, DriftRetrainingClassifier, DummyDriftDetector, NoDrift. `river.drift.binary`: **DDM, EDDM, FHDDM, HDDMA, HDDMW** (these consume a *binary error stream*, i.e. they need a classifier underneath). `river.anomaly`: PredictiveAnomalyDetection, GaussianScorer, **HalfSpaceTrees**, LODA, LocalOutlierFactor, OneClassSVM, QuantileFilter, StandardAbsoluteDeviation, ThresholdFilter. ADWIN API: `update(x: int|float)`, `drift_detected` boolean, `estimation` attribute; constructor `delta=0.002, clock=32, max_buckets=5, min_window_length=5, grace_period=10`.

R179. Bifet, A. and Gavaldà, R. **Learning from Time-Changing Data with Adaptive Windowing (ADWIN).** SIAM International Conference on Data Mining (SDM) 2007. https://doi.org/10.1137/1.9781611972771.42. Primary source for ADWIN; the reference `river` itself cites.

R180. Page, E. S. **Continuous Inspection Schemes.** *Biometrika* 41(1-2):100-115, 1954. https://doi.org/10.1093/biomet/41.1-2.100. Original source of the CUSUM statistic and of `river.drift.PageHinkley`.

R181. Raab, C., Heusinger, M. and Schleif, F.-M. **Reactive Soft Prototype Computing for Concept Drift Streams (KSWIN).** *Neurocomputing* 416:340-351, 2020. https://doi.org/10.1016/j.neucom.2019.11.111. Source for the Kolmogorov-Smirnov Windowing drift detector; distribution-free and operates on **real values**, so unlike DDM/EDDM it needs no supervised classifier underneath.

R182. Gama, J., Medas, P., Castillo, G. and Rodrigues, P. **Learning with Drift Detection (DDM).** *Advances in Artificial Intelligence — SBIA 2004*, LNCS, Springer, pp. 286-295. https://doi.org/10.1007/978-3-540-28645-5_29. DDM monitors a **binary error rate**, so it can only sit downstream of a classifier.

R183. Baena-García, M., del Campo-Ávila, J., Fidalgo, R., Bifet, A., Gavaldà, R. and Morales-Bueno, R. **Early Drift Detection Method (EDDM).** Fourth International Workshop on Knowledge Discovery from Data Streams, 2006. Citation as printed in the `river` documentation: https://riverml.xyz/latest/api/drift/binary/EDDM/. **`SECONDARY-SOURCE`** — no DOI or publisher landing page could be retrieved this session; the bibliographic details are second-hand via `river`. Same binary-error-stream limitation as DDM.

R184. Barros, R. S. M. and Santos, S. G. T. C. **A large-scale comparison of concept drift detectors.** *Information Sciences* 451-452:348-370, 2018. https://doi.org/10.1016/j.ins.2018.04.014. Compares 14 drift-detector configurations with two base classifiers. Headline: it explicitly "verif[ies] and challenge[s] a common belief in the area, namely that the best drift detection methods are necessarily those that detect all the existing drifts closer to their correct positions, and only them". **Material caveat: the comparison uses artificial datasets with fully labelled streams — it is not evidence about industrial, rail or compressor telemetry.**

R185. Adams, R. P. and MacKay, D. J. C. **Bayesian Online Changepoint Detection (BOCPD).** arXiv:0710.3742, 2007. https://arxiv.org/abs/0710.3742. **`NOT PEER-REVIEWED`** (no journal reference on the arXiv page). Online exact inference over the posterior "run length" (time since the last change) — the one change-point output that is natively a **continuous per-timestamp quantity**.

R186. **bayesian-changepoint-detection** (Python package), v0.2.dev1, last upload 12 Aug 2019. https://pypi.org/pypi/bayesian-changepoint-detection/json — http://github.com/hildensia/bayesian_changepoint_detection. **Licence NOT SPECIFIED (the PyPI license field is empty)** and unmaintained since 2019 — two disqualifiers for a submitted artefact.

R187. Londschien, M., Bühlmann, P. and Kovács, S. **Random Forests for Change Point Detection** (`changeforest`). *Journal of Machine Learning Research* 24(216):1-45, 2023. https://www.jmlr.org/papers/v24/22-0512.html — https://github.com/mlondschien/changeforest/ — PyPI v1.2.1, licence **BSD**. Multivariate **nonparametric** multiple-change-point detection using a classifier log-likelihood ratio from random-forest out-of-bag class probabilities; reports improved empirical performance against existing multivariate nonparametric CPD methods and is explicitly designed for the multivariate / high-dimensional case. Caveat: the reported evidence is a **simulation** study, not industrial data.

R188. **aeon** segmentation module, v1.5.0 (requires Python ≥3.11). https://pypi.org/pypi/aeon/json — API https://www.aeon-toolkit.org/en/stable/api_reference/segmentation.html. Licence **BSD 3-Clause**. `BinSegmenter`, `ClaSPSegmenter`, `FLUSSSegmenter`, `InformationGainSegmenter`, `GreedyGaussianSegmenter`, `EAggloSegmenter`, `HMMSegmenter`, `HidalgoSegmenter`, **`RandomSegmenter`**, all behind a common sklearn-style `BaseSegmenter`. `UNVERIFIED`: the API page does not state which of these accept multivariate input — verify per estimator before relying on it for 15-channel MetroPT.

R189. Schmidl, S., Wenig, P. and Papenbrock, T. **Anomaly detection in time series: a comprehensive evaluation (TimeEval).** *Proceedings of the VLDB Endowment* 15(9):1779-1797, 2022. https://doi.org/10.14778/3538598.3538602. Peer-reviewed evaluation of **71** anomaly detection algorithms across **976** time series datasets, reporting **runtime alongside accuracy**. `NUMBERS UNVERIFIED` — the per-algorithm ranking table was not retrieved; do not quote specific winners.

R190. Aminikhanghahi, S. and Cook, D. J. **A Survey of Methods for Time Series Change Point Detection.** *Knowledge and Information Systems*, 2017. https://doi.org/10.1007/s10115-016-0987-z. Supervised **and** unsupervised CPD methods, comparison criteria including evaluation criteria, and open challenges. Free full text via PubMed Central (PMC5464762).

R191. Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M. and Bouchachia, A. **A survey on concept drift adaptation.** *ACM Computing Surveys* 46:1-37, 2014. https://doi.org/10.1145/2523813. Standard umbrella reference for the drift-detector family.

R192. Lu, J., Liu, A., Dong, F., Gu, F., Gama, J. and Zhang, G. **Learning under Concept Drift: A Review.** *IEEE Transactions on Knowledge and Data Engineering*, 2018. https://doi.org/10.1109/TKDE.2018.2876857. Peer-reviewed review covering drift detection, understanding and adaptation.

R193. Bayram, F., Ahmed, B. S. and Kassler, A. **From concept drift to model degradation: An overview on performance-aware drift detectors.** *Knowledge-Based Systems*, 2022. https://doi.org/10.1016/j.knosys.2022.108632. Consolidated taxonomy of drift types by mathematical definition, then a hierarchical review of **performance-based** drift detection — methods that fire on a predictive model's error/residual degradation rather than on instantaneous magnitude. This is the established-family citation for "drift detector on the forecast residual".

R194. Boniol, P., Liu, Q., Huang, M., Palpanas, T. and Paparrizos, J. **Dive into Time-Series Anomaly Detection: A Decade Review.** arXiv:2412.20512, 2024. https://arxiv.org/abs/2412.20512. **`NOT PEER-REVIEWED`**. Process-centric taxonomy of TSAD plus a meta-analysis of the literature, from the group that produced VUS and TSB-AD — so its taxonomy is consistent with our primary metric. Framing only; the abstract does not discuss method-family performance.

---

## K. Prognostics, degradation models, survival analysis and RUL evaluation

*Added in the W1 update pass to make the RUL arm symmetric across door, pneumatic and bearing, and to justify (or retire) the run-to-failure datasets in `datasets.md`.*

R195. Si, X.-S., Wang, W., Hu, C.-H. and Zhou, D.-H. **Remaining useful life estimation — A review on the statistical data driven approaches.** *European Journal of Operational Research*, 2011. https://doi.org/10.1016/j.ejor.2010.11.018. The umbrella reference for statistical data-driven RUL: organises the family of stochastic degradation and filtering-based remaining-life models. This is the citation that legitimises adding a "stochastic degradation" family to the ladder at all.

R196. Zhang, Z., Si, X., Hu, C. and Lei, Y. **Degradation data analysis and remaining useful life estimation: A review on Wiener-process-based methods.** *European Journal of Operational Research*, 2018. https://doi.org/10.1016/j.ejor.2018.02.033. Canonical formulation and review of Wiener (drift-diffusion) degradation modelling and the resulting **first-hitting-time** RUL distribution (inverse-Gaussian first-passage density; mean `(w − x_t)/μ` to threshold `w`).

R197. van Noortwijk, J. M. **A survey of the application of gamma processes in maintenance.** *Reliability Engineering & System Safety*, 2009. https://doi.org/10.1016/j.ress.2007.03.019. Canonical survey of **gamma-process** degradation modelling — the right choice when degradation is strictly monotone increasing (wear, fouling, leak growth), because a gamma process forbids the health index from improving.

R198. Ye, Z.-S. and Chen, N. **The Inverse Gaussian Process as a Degradation Model.** *Technometrics*, 2014 (online 2013). https://doi.org/10.1080/00401706.2013.830074. Third member of the standard stochastic-degradation trio alongside Wiener and gamma; the natural model when degradation is monotone but a gamma process fits poorly. `scipy.stats.invgauss` gives the fit and the RUL quantiles.

R199. Coble, J. and Hines, J. W. **Applying the General Path Model to Estimation of Remaining Useful Life.** *International Journal of Prognostics and Health Management* 2(1), 2011. https://doi.org/10.36001/ijphm.2011.v2i1.1352. Open access. General Path Model: identify a degradation measure characterising progression to failure, fit a functional form to it, extrapolate to a failure threshold, and update the fit dynamically (Bayesian) as new data arrives. Needs only a monotonic health index plus a threshold — **no run-to-failure labels**.

R200. Wang, T., Yu, J., Siegel, D. and Lee, J. **A similarity-based prognostics approach for Remaining Useful Life estimation of engineered systems.** IEEE International Conference on Prognostics and Health Management, 2008. https://doi.org/10.1109/phm.2008.4711421. Canonical formulation of similarity-based RUL: match the current degradation trajectory against a library of historical run-to-failure trajectories and aggregate the matched remaining lives. **Requires run-to-failure trajectories**; cannot run on censored-only data.

R201. NASA. **ProgPy — NASA Prognostics Python Packages**, PyPI 1.7.1 (24 Apr 2025); NASA 2024 Software of the Year. https://github.com/nasa/progpy. Licence **NASA-1.3 (NASA Open Source Agreement) — not an OSI-standard permissive licence; check hackathon IP rules before shipping.** State estimators `KalmanFilter`, `UnscentedKalmanFilter`, `ParticleFilter`; predictors `MonteCarlo`, `UnscentedTransformPredictor`; `ToEPredictionProfile` metrics **`alpha_lambda`, `prognostic_horizon`, `cumulative_relative_accuracy`, `monotonicity`**.

R202. **FilterPy**, PyPI 1.4.5 (10 Oct 2018). https://github.com/rlabbe/filterpy. Licence **MIT**. Kalman, Extended Kalman, Unscented Kalman filters, Kalman smoothers, least-squares, fading-memory and g-h filters. The PyPI description does **not** list a particle filter. Permissive fallback if ProgPy's licence is a problem; effectively unmaintained since 2018.

R203. Reid, M. **reliability** (Python), PyPI 0.9.0 (7 Mar 2025). https://reliability.readthedocs.io/en/latest/ — https://github.com/MatthewReid854/reliability. Licence **LGPL-3.0**. Fits Weibull, exponential, gamma, lognormal, normal, beta, Gumbel and loglogistic with **right-censored** support; mixture / competing-risks models; Kaplan-Meier, Nelson-Aalen, rank adjustment; 24 accelerated-life-test models; repairable-system MCF/ROCOF.

R204. **SurPyval**, PyPI 0.19.0 (4 Aug 2026). https://surpyval.readthedocs.io/en/latest/ — https://github.com/derrynknife/SurPyval. Licence **MIT**. Parametric and non-parametric survival/reliability fitting with a scipy-style `.fit()`; handles observed, right-censored, left-censored **and truncated** observations simultaneously; MLE/MPP/MSE/MOM/MPS; Kaplan-Meier, Nelson-Aalen, Fleming-Harrington, Turnbull.

R205. Cox, D. R. **Regression Models and Life-Tables.** *Journal of the Royal Statistical Society Series B*, 1972. https://doi.org/10.1111/j.2517-6161.1972.tb00899.x. Primary source for the Cox proportional-hazards model. Requires **censored time-to-event data** (duration + event indicator + covariates), not a health index and not full run-to-failure trajectories.

R206. Ishwaran, H., Kogalur, U. B., Blackstone, E. H. and Lauer, M. S. **Random survival forests.** *The Annals of Applied Statistics*, 2008. https://doi.org/10.1214/08-aoas169. Non-parametric tree-ensemble alternative to Cox regression for right-censored data; one call as `sksurv.ensemble.RandomSurvivalForest`.

R207. Rahat, M., Kharazian, Z., Mashhadi, P. S., Rögnvaldsson, T. and Choudhury, S. **Bridging the Gap: A Comparative Analysis of Regressive Remaining Useful Life Prediction and Survival Analysis Methods for Predictive Maintenance.** PHM Society Asia-Pacific Conference, 2023. https://doi.org/10.36001/phmap.2023.v4i1.3646. Open access. Explicit framework for comparing **RUL-regression against survival analysis** on run-to-failure datasets: three degradation models × two learning algorithms (six models), including random survival forests; evaluated on C-MAPSS and on real condition-monitoring data from turbocharger devices in a Volvo truck fleet. **`NUMBERS UNVERIFIED`** — the landing page does not state which family won.

R208. Rahat, M. and Kharazian, Z. **SurvLoss: A New Survival Loss Function for Neural Networks to Process Censored Data.** PHM Society European Conference, 2024. https://doi.org/10.36001/phme.2024.v8i1.4052. Open access. An asymmetric loss letting an ordinary regression network consume **censored** samples by penalising predictions outside the censoring region and skipping them otherwise; evaluated on C-MAPSS and the SCANIA truck component dataset, reporting improved RUL over baselines when censored samples are used alongside event samples. ≈30 lines on top of an existing regression loss.

R209. Kharazian, Z., Lindgren, T., Magnússon, S., Steinert, O. and Andersson Reyna, O. **SCANIA Component X Dataset: A Real-World Multivariate Time Series Dataset for Predictive Maintenance.** arXiv:2401.15199 (v1 Jan 2024, revised Mar 2025). https://doi.org/10.48550/arXiv.2401.15199. The authors state the dataset is intended for classification, regression, **survival analysis** and anomaly detection. `UNVERIFIED`: the abstract page does not state the download location or the exact time-to-event / censoring column layout. (Dataset record itself is **R134**.)

R210. Dhada, M., Parlikad, A. K., Steinert, O. and Lindgren, T. **Weibull recurrent neural networks for failure prognosis using histogram data.** *Neural Computing and Applications*, 2022. https://doi.org/10.1007/s00521-022-07667-7. WTTE-RNN (a network optimising a Weibull survival function, so censoring is handled natively) applied to turbocharger condition monitoring in a heavy-duty SCANIA truck fleet recorded as time-series histograms. Two findings: **including data from assets that did not fail improves prediction**, and sub-fleet clustering helps only when the training set is large enough. Reference implementation `ragulpr/wtte-rnn` (MIT) last pushed 2020-08-07 — expect to port the Weibull log-likelihood yourself (≈50 lines in torch).

R211. Davidson-Pilon, C. **lifelines**, PyPI 0.30.3 (5 Mar 2026). https://lifelines.readthedocs.io/en/latest/ — https://github.com/CamDavidsonPilon/lifelines — citable via Zenodo https://doi.org/10.5281/zenodo.805993. Licence **MIT**. Kaplan-Meier, Nelson-Aalen, `CoxPHFitter`, `WeibullAFTFitter`. First choice for the survival row on licence grounds.

R212. Pölsterl, S. **scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn.** *Journal of Machine Learning Research* 21(212):1-6, 2020. http://jmlr.org/papers/v21/20-729.html — https://github.com/sebp/scikit-survival — PyPI 0.28.0 (5 Jul 2026). Licence **GPL-3.0-or-later — importing it makes the deliverable a derived work under GPL terms.** Home of `RandomSurvivalForest` and gradient-boosted survival models.

R213. **xgbse (XGBoost Survival Embeddings)**, PyPI 0.3.3, repo last pushed 3 Oct 2024. https://github.com/loft-br/xgboost-survival-embeddings. Licence **Apache-2.0**. XGBoost-based survival analysis producing **calibrated survival curves with confidence intervals** plus prototype-based explainability — the permissive-licence route to a gradient-boosted survival model, and the curve maps directly onto "probability of reaching severity level k within h days" for the twin UI.

R214. Kvamme, H. **pycox**, PyPI 0.3.0. https://github.com/havakv/pycox. Licence **BSD**. Survival analysis with PyTorch; the standard home for **discrete-time hazard** models. `UNVERIFIED`: the specific model list (DeepSurv / DeepHit / logistic-hazard) was not confirmed from the retrieved page. A hand-written discrete-time hazard head is ≈60 lines and may be cheaper than the dependency.

R215. Saxena, A., Celaya, J., Saha, B., Saha, S. and Goebel, K. **Metrics for Offline Evaluation of Prognostic Performance.** *International Journal of Prognostics and Health Management* 1(1), 2010. https://doi.org/10.36001/ijphm.2010.v1i1.1336 — PDF https://papers.phmsociety.org/index.php/ijphm/article/download/1336/324. Open access. Presents evaluation metrics tailored for prognostics that incorporate probabilistic uncertainty estimates, giving both a quantitative score and a visual perspective. **`NUMBERS UNVERIFIED`**: the landing-page abstract does not itself enumerate the metric names; our mapping of α-λ accuracy, prognostic horizon, relative accuracy and convergence to this paper rests on ProgPy [R201] implementing exactly those names — **read the PDF before quoting definitions verbatim**.

R216. Saxena, A., Celaya, J., Balaban, E., Goebel, K., Saha, B., Saha, S. and Schwabacher, M. **Metrics for evaluating performance of prognostic techniques.** IEEE International Conference on Prognostics and Health Management, 2008. https://doi.org/10.1109/phm.2008.4711436. Earlier conference source of the same NASA metric programme. **`NUMBERS UNVERIFIED`** — IEEE full text not opened; cite alongside the open-access R215.

R217. Saxena, A., Goebel, K., Simon, D. and Eklund, N. **Damage propagation modeling for aircraft engine run-to-failure simulation.** IEEE International Conference on Prognostics and Health Management, 2008. https://doi.org/10.1109/phm.2008.4711414. Primary source for the **C-MAPSS** benchmark and for the **asymmetric exponential PHM scoring function** that penalises late (optimistic) RUL predictions more than early ones. **`NUMBERS UNVERIFIED`** — IEEE full text not opened; the asymmetric-score attribution is standard but was not checked against the PDF. Known pitfall: the score is unnormalised and exponential, so one badly-late unit can dominate the fleet total and it is not comparable across datasets with different unit counts or life lengths.

R218. Coble, J. and Hines, J. W. **Identifying Optimal Prognostic Parameters from Data: A Genetic Algorithms Approach.** Annual Conference of the PHM Society, 2009. https://papers.phmsociety.org/index.php/phmconf/article/view/1404. Open access. The standard citation for scoring a candidate **health indicator** on prognostic suitability (the monotonicity / prognosability / trendability family) rather than on downstream RUL error — i.e. how to choose the HI before fitting any RUL model. `UNVERIFIED`: OpenAlex lists no DOI for this record, the year comes from OpenAlex, and the exact metric definitions were not read from the PDF — verify before quoting formulas.

R219. **Leakage-safe benchmark of machine learning models for IGBT remaining useful life prediction.** *IET Conference Proceedings*, 2026. https://doi.org/10.1049/icp.2026.2213. Argues RUL benchmarks are optimistically biased by device-level leakage; proposes a leakage-safe preprocessing pipeline plus **leave-one-device-out** evaluation on the NASA IGBT run-to-failure set. Support vector regression was the most consistent across devices (MAE ≈0.019, R² 99.3 %); the stated conclusion is that leakage control matters as much as model complexity. `UNVERIFIED` beyond the OpenAlex abstract — the IET publisher page was not opened.

R220. **Leakage-Robust Evaluation and Data-Scale Sensitivity of Attention-Enhanced Multi-Task Learning for Joint Fault Diagnosis and Remaining Useful Life Estimation.** arXiv:2607.16493, 2026. https://arxiv.org/abs/2607.16493. **`NOT PEER-REVIEWED`**. Reports that naive splitting can inflate classification accuracy from a genuine **20-60 % to 99.9 %** on the same data (or collapse it to 0 %); introduces a chunk-based, leakage-audited splitting protocol with 5 seeds, one-way ANOVA and Tukey HSD. Evaluated on C-MAPSS, NASA **IMS bearing** data and the UCI hydraulic set; the attention-enhanced multi-task net only matches single-task CNN-LSTM baselines on C-MAPSS and multi-task training is unstable on the smaller datasets. No code URL on the abstract page.

R221. **Generalization bounds and sample complexity for remaining useful life prediction from complete degradation trajectories.** arXiv:2607.23454, 2026 (stated as accepted for *Measurement Science and Technology*). https://arxiv.org/abs/2607.23454. Uniform deviation of MSE decreasing as `O(B²·√(p/n))` with `p` = model complexity and `n` = number of **run-to-failure trajectories**; embedding degradation physics can cut data requirements by up to **two orders of magnitude** for deep networks; fleet-to-fleet variability creates irreducible trade-offs and right-censored observations incur an efficiency loss. Validated against published turbofan, battery and bearing benchmarks to within a factor 2-3.

R222. **Remaining Useful Life Estimation for Turbofan Engines: A Comparative Study of Classical, CNN, and LSTM Approaches.** arXiv:2604.27234, 2026. https://arxiv.org/abs/2604.27234. **`NOT PEER-REVIEWED`**. C-MAPSS FD001/FD003 RMSE: LSTM 14.93 / 14.20, 1D CNN 16.97 / 15.68, **XGBoost 13.36 on FD003** — i.e. gradient boosting beats both deep models on FD003 and a single-layer LSTM beats the deeper published LSTM. C-MAPSS, not bearings.

R223. **Towards Unified and Data-Efficient Prognostics and Health Management with Tabular Foundation Models.** arXiv:2606.05481, 2026. https://arxiv.org/abs/2606.05481. **`NOT PEER-REVIEWED`**. Tabular foundation models with in-context learning on industrial time series report the best average ranks across prognostic and diagnostic PHM tasks versus sequence models, transformer baselines and gradient-boosted trees, with the advantage concentrated in **low-data regimes**. `UNVERIFIED`: the benchmark datasets used are not stated on the abstract page, so whether bearing run-to-failure sets are among them is unknown.

R224. **Advancements in bearing remaining useful life prediction methods: a comprehensive review.** *Measurement Science and Technology*, 2024. https://doi.org/10.1088/1361-6501/ad5223. Organises bearing RUL into four stages — data acquisition, health-indicator construction, algorithm, evaluation — and splits methods into **physics-based, statistical-based and data-driven** families; catalogues the public bearing run-to-failure datasets used for validation. It surveys rather than adjudicates: it does **not** declare a winning family. `UNVERIFIED` beyond the OpenAlex abstract.

R225. **Transfer learning algorithms for bearing remaining useful life prediction: A comprehensive review from an industrial application perspective.** *Mechanical Systems and Signal Processing*, 2023. https://doi.org/10.1016/j.ymssp.2023.110239. **`CONTENT UNVERIFIED`** — only title, venue, year, DOI and citation count (260) retrieved from OpenAlex. Listed because lab-to-field transfer is the objection a judge will raise against a bogie RUL result trained on XJTU-SY.

R226. **Intelligent Approaches for Anomaly Detection in Compressed Air Systems: A Systematic Review.** *Machines* (MDPI) 11(7):750, 2023. https://doi.org/10.3390/machines11070750. PRISMA review (Scopus + Web of Science to Nov 2022, 37 eligible papers) of mathematical, machine-learning, neural-network, time-series and hybrid methods for compressed-air systems — **scoped entirely to anomaly detection, with no prognostics or RUL section.** `UNVERIFIED` beyond the OpenAlex abstract. Cited as positive evidence for the gap: the compressed-air ML literature has no RUL arm to borrow.

R227. **XJTU-SY Rolling Element Bearing Accelerated Life Test Datasets: A Tutorial.** *Journal of Mechanical Engineering*, 2019. https://doi.org/10.3901/jme.2019.16.001. The description paper for the XJTU-SY datasets (dataset record itself is **R124**). **`CONTENT UNVERIFIED`** — title/venue/DOI from OpenAlex only; the paper is in Chinese.

R228. Nectoux, P. et al. **PRONOSTIA: An experimental platform for bearings accelerated degradation tests.** IEEE International Conference on Prognostics and Health Management, PHM'12, Denver, 2012. https://hal.science/hal-00719503v1/document. Canonical source paper for the FEMTO/PRONOSTIA bearing run-to-failure benchmark (dataset record is **R143**). **No DOI exists in the OpenAlex record**; the HAL URL above is the retrieved full-text location.

R229. **Advances and limitations in machine learning approaches applied to remaining useful life predictions: a critical review.** *The International Journal of Advanced Manufacturing Technology*, 2024. https://doi.org/10.1007/s00170-024-14000-0. **`CONTENT UNVERIFIED`** — the OpenAlex record has a null abstract and the Springer page was not opened, so whether it critiques evaluation practice, leakage or simple-vs-deep comparisons cannot be confirmed. Listed only as a lead.

R230. **Train Brake System Pipe Leakage Detection and Early Warning Method Based on Bayesian Networks.** *Academic Journal of Science and Technology*, 2023. https://doi.org/10.54097/ajst.v7i1.10988. **`CONTENT UNVERIFIED`** — title/year/venue/DOI from OpenAlex only, zero recorded citations, low-profile venue. Detection and early warning, **not** RUL. Listed for completeness of the pneumatic-prognostics search, **not recommended for citation**.

---

## L. PS3 subsystems (W4 pass, 17 Sep 2026) — rail corrugation, HVAC/refrigerant FDD, fatigue damage, time-series augmentation

Added for `docs/research/ps3_addendum.md`. Four parallel finders produced ~90 candidate
identifiers; a citation verifier resolved every one against Crossref, OpenAlex, Semantic Scholar,
the arXiv Atom API and (for code) GitHub. Tags follow the conventions above: `NUMBERS UNVERIFIED`
means the identifier and metadata resolve but the quoted figures sit behind a paywall,
`NOT PEER-REVIEWED` means preprint or workshop, `SECONDARY-SOURCE` means the fact came from a
summary rather than the primary document.

**One candidate was refuted and is deliberately absent from this list**: the *Applied Energy* 402
(2026) "few-shot adaptive weighted prototype network" HVAC-FDD paper cited in the ACV finder's
appendix by the guessed DOI `10.1016/j.apenergy.2025.126786`. That DOI resolves to an unrelated
fuel-cell paper; the real work at the quoted RePEc/pii identifier is 10.1016/j.apenergy.2025.127056
("A few-shot learning framework for HVAC fault diagnosis in data centers with minimal data
required", Yan et al.), a different title and author list whose content was never verified, and the
quoted F1 figures (73.77 % on ASHRAE RP-1043 severity 1, 67.22 % on RP-1312 AHU summer) could not
be located in any paper. **Do not cite either the DOI or the numbers.** Use R268 if a
limited-labels HVAC-FDD citation is wanted.

### L1. Rail corrugation from axle-box acceleration

R231. Faccini, L., Karaki, J., Di Gialleonardo, E., Somaschini, C., Bocciolone, M. and Collina, A. **A Methodology for Continuous Monitoring of Rail Corrugation on Subway Lines Based on Axlebox Acceleration Measurements.** *Applied Sciences* 13(6):3773, 2023. https://doi.org/10.3390/app13063773. **CC BY 4.0**. The closest published setting to PS3 — metro, in-service, axlebox, continuous — and the reference design for a band-limited RMS corrugation index with a positioning/alignment step. `NUMBERS UNVERIFIED` (MDPI HTML returned 403; abstract and Crossref metadata verified).

R232. Lian, Q., Zhang, H., Gao, Y. and Liu, J. **A feature extraction framework for metro rail corrugation detection using onboard vibration and noise monitoring data.** *Intelligent Transportation Infrastructure* 4, 2025. https://doi.org/10.1093/iti/liaf010. **CC BY**, full text read. 639 samples (352 non-corrugation / 287 corrugation), 26 hand features (14 time-domain, 6 frequency-domain, 6 wavelet-band energies at 250–500, 500–1000, 1000–2000 Hz). **BPNN 95.31 % acc / 96.17 % F1; SVM 94.79/95.73; LSTM 94.27/95.40; RF 93.75/94.83** — four very different classifiers within 1.6 points, i.e. the representation carries the accuracy. Its fixed-Hz wavelet bands assume near-constant metro speed and are exactly what breaks over PS3's 0–67 km/h range.

R233. Liu, W., Wu, T., Chi, M., Wen, Z. and He, C. **Determination of rail corrugation maintenance limit based on axle box acceleration spectrum defined in IEC 61373.** *Vehicle System Dynamics* 61(11):2936-2952, 2023. https://doi.org/10.1080/00423114.2022.2151920. Works in a standardised **1/3-octave ASD band structure (IEC 61373)** and maps the ABA spectrum to a maintenance limit — the defensible, non-arbitrary band layout. `NUMBERS UNVERIFIED` (T&F paywall).

R234. Hassanieh, W., Chehade, A., Facchinetti, A., Carman, G., Bocciolone, M. and Somaschini, C. **Leveraging machine learning to predict rail corrugation level from axle-box acceleration measurements on commercial vehicles.** *International Journal of Rail Transportation* 12(4):604-625, 2024. https://doi.org/10.1080/23248378.2023.2220112. Tuned Random Forest on accelerometer features **plus speed, curve radius and track type as explicit inputs** — the counter-design to normalising speed away. `NUMBERS UNVERIFIED` (T&F paywall).

R235. De Rosa, A., Luber, B., Müller, G. and Fuchs, J. **Methodology to Detect Rail Corrugation from Vehicle On-Board Measurements by Isolating Effects from Other Sources of Excitation.** *Applied Sciences* 14(19):8920, 2024. https://doi.org/10.3390/app14198920. **CC BY 4.0**. Structural modes, bridges, switches and crossings and wheel defects occupy the *same* wavelength range as corrugation and must be separated by their own signatures — the paper that explains corrugation false positives. `NUMBERS UNVERIFIED` (MDPI 403).

R236. Li, S. et al. **Monitoring of rail short pitch corrugation using the time-frequency features of both vertical and longitudinal axle box accelerations.** *Measurement* 255:118064, 2025. https://doi.org/10.1016/j.measurement.2025.118064. Two axes carry complementary information (longitudinal responds to the creep/stick-slip mechanism, vertical to contact geometry) — the argument for not collapsing channel types. `NUMBERS UNVERIFIED` (Elsevier paywall).

R237. Yu, X. et al. **Experimental study of the use of a transfer function to find rail corrugation from axle-box accelerations.** *Measurement* 249:117058, 2025. https://doi.org/10.1016/j.measurement.2025.117058. **The citation for normalising measured acceleration by the square of forward velocity** so the roughness→acceleration relation becomes speed-independent. The inversion itself needs a calibrated track transfer function we do not have. `NUMBERS UNVERIFIED` (Elsevier paywall).

R238. Carrigan, T. D. and Talbot, J. P. **A new method to derive rail roughness from axle-box vibration accounting for track stiffness variations and wheel-to-wheel coupling.** *Mechanical Systems and Signal Processing* 192:110232, 2023. https://doi.org/10.1016/j.ymssp.2023.110232. Companion: **Use of Flexible Wheelset Model, Comb Filter and Track Identification…**, LNME (IWRN 14), 2024, https://doi.org/10.1007/978-981-99-7852-6_25. A **comb filter keyed to the wheel circumference** separates wheel-fixed from track-fixed roughness. Reported 1/3-octave band deviation <1 dB over λ = 5 mm-0.5 m for known track dynamics, with rail-pad stiffness the dominant sensitivity (a 20 % deviation moves estimated roughness by up to 3.5 dB) — **`NUMBERS UNVERIFIED`**, from abstracts/snippets only.

R239. Pieringer, A. and Kropp, W. **Model-based estimation of rail roughness from axle box acceleration.** *Applied Acoustics* 193:108760, 2022. https://doi.org/10.1016/j.apacoust.2022.108760. The canonical statement of why the PSD must be translated into the **wavelength domain** before anything else, with explicit compensation for vehicle speed and track dynamics. `NUMBERS UNVERIFIED` (Elsevier paywall).

R240. Haghbin, M., Chiachío, J., Muñoz, S., Escalona, J. L., Guillén, A., Crespo Marquez, A. and Cantero-Chinchilla, S. **Predicting Rail Corrugation Based on Convolutional Neural Networks Using Vehicle's Acceleration Measurements.** *Sensors* 24(14):4627, 2024. https://doi.org/10.3390/s24144627. **CC BY 4.0**. 1D-CNN on near-raw windows with **speed as an explicit auxiliary input**; >95 % accuracy across speeds — but on a **scaled railway test rig**, not a metro line. Per-class values `NUMBERS UNVERIFIED`.

R241. Amin, A., Najeh, T. and Ghoul, N. **AI-driven vibration-based event classification in railway switches and crossings.** *Scientific Reports* 16, 2026. https://doi.org/10.1038/s41598-026-58967-0. **CC BY 4.0**. **The realism anchor**: 21 classifiers benchmarked on small, imbalanced railway vibration data with a strict held-out split give **81.5 % accuracy and ROC-AUC ≈ 0.94**, best with ensembles, and **feature standardisation was decisive** (without it neural nets fell below chance). Macro F1 used as the imbalance-aware metric. Data-availability statement `UNVERIFIED` (Nature IDP redirect blocked fetch).

R242. Samani, F. S., Núñez, A. and De Schutter, B. **WaveletInception Networks for on-board Vibration-Based Infrastructure Health Monitoring.** arXiv:2507.12969, 2025 (rev. Jan 2026). https://arxiv.org/abs/2507.12969. **NOT PEER-REVIEWED** — the arXiv title matches exactly, but the stated *Engineering Applications of AI* placement could not be confirmed and stays `UNVERIFIED`. Learnable wavelet-packet front end + 1D Inception-ResNet + BiGRU that **ingests measurement speed as an operating condition** — the "condition on speed instead of normalising it" school. Headline numbers `UNVERIFIED`.

R243. Wang, Y., Xiao, X., Ma, T., Zhang, S., Cui, X. and Xu, Z. **On-board detection of rail corrugation using improved convolutional block attention mechanism (TBVA-Net).** *Engineering Applications of Artificial Intelligence* 146:110349, 2025. https://doi.org/10.1016/j.engappai.2025.110349. Car-body (not axlebox) acceleration; **pre-train on simulation, fine-tune on few field labels**; >95 % test accuracy, mean 98.6 % on the simulated set and 98.5 % transfer accuracy after fine-tuning. `NUMBERS UNVERIFIED` (Elsevier paywall). Cited in PS3 as a reasoned SKIP: no corrugation simulator exists here.

R244. Wang, Y., Xiao, X., Nadakatti, M., Zhang, S., Chi, M. and Liu, J. **A metro rail corrugation detection framework based on car body vibration signals and unsupervised learning.** *Engineering Applications of Artificial Intelligence* 153:110976, 2025. https://doi.org/10.1016/j.engappai.2025.110976. Synchrosqueezed wave-packet spectrograms + **MoCo contrastive pre-training** then a small labelled fine-tune; wavelength classification 95-100 %, amplitude assessment >95 %. `NUMBERS UNVERIFIED` (Elsevier paywall). Cited as a reasoned SKIP: 234 unlabelled records are far too few for contrastive pre-training to pay.

R245. **Vold-Kalman Filter Order tracking of Axle Box Accelerations for Railway Stiffness Assessment.** arXiv:2209.12899, 2022. https://arxiv.org/abs/2209.12899. **NOT PEER-REVIEWED**; journal version `UNVERIFIED`. Names the problem correctly — speed variation smears the spectrum and **order tracking is the fix**. PS3 takes the cheap version (tacho-driven resampling to constant Δx); Vold-Kalman itself is over-engineering at this budget. `NUMBERS UNVERIFIED`.

R246. Jahan, K., Lähns, F., Baasch, B., Heusel, J. and Roth, M. **Rail Surface Defect Detection and Severity Analysis Using CNNs on Camera and Axle Box Acceleration Data.** LNME, Int. Congress & Workshop on Industrial AI and eMaintenance 2023, pp. 423-435, 2024. https://doi.org/10.1007/978-3-031-39619-9_31 — OA copy https://elib.dlr.de/201722/. DLR-side confirmation that **corrugation vs impulsive squat is the hard confusion** on ABA. `CONTENT UNVERIFIED` — the OA PDF did not parse; fetch by hand before quoting anything.

R247. Yang, J., Huo, B. and Yao, D. **Rail corrugation detection based on optimal position window and weighted-bandwidth mode decomposition.** *Measurement* 255:117888, 2025. https://doi.org/10.1016/j.measurement.2025.117888 — preprint https://doi.org/10.2139/ssrn.5009696. Representative of the whole **VMD / CEEMDAN / EWT / SPWVD** adaptive-decomposition family for this problem; cited in PS3 as the named SKIP for that family (per-record, parameter-heavy, slow, and no evidence of beating a wavelength-band PSD at n=272). `NUMBERS UNVERIFIED` (Elsevier paywall).

R248. Wang, P., Huang, J., Wang, J., Ni, Y., Ran, Y., Li, X. and Zhang, Z. **Concise Historic Overview of Rail Corrugation Studies: From Formation Mechanisms to Detection Methods.** *Buildings* 14(4):968, 2024. https://doi.org/10.3390/buildings14040968. **CC BY 4.0**. The wavelength taxonomy (short-pitch 25-80 mm, long-pitch, roaring rails). Pair with **Formation mechanism of short-pitch rail corrugation on metro tangent tracks with resilient fasteners**, *Vehicle System Dynamics* 61(6), 2023, https://doi.org/10.1080/00423114.2022.2086143, for the metro-specific 30-60 mm figure.

R249. Dissanayake, O., McPherson, S., Allyndree, J., Kennedy, E., Cunningham, P. and Riaboff, L. **Evaluating ROCKET and Catch22 features for calf behaviour classification from accelerometer data.** arXiv:2404.18159, 2024. https://arxiv.org/abs/2404.18159. **NOT PEER-REVIEWED** (a journal version is stated but unnamed). Off-domain but the cleanest head-to-head on short accelerometer windows: balanced accuracy **ROCKET 0.70 ± 0.07 > catch22 0.69 ± 0.05 > hand-crafted 0.65 ± 0.03**, best combination **ROCKET + RidgeClassifierCV 0.77**. Also a warning that absolute numbers on small imbalanced accelerometer problems sit near 0.7, not 0.95.

R250. **EN 15610 (2019+A1:2025)** *Railway acoustics — rail and wheel roughness measurement related to noise generation*, and **EN ISO 3095** limit spectrum; **IEC 61373** ASD band structure (see R233). CEN/IEC standards, **purchase only — text not read, `UNVERIFIED`**. The reason to present features as **1/3-octave wavelength bands** rather than arbitrary FFT bins: it is what the domain already uses and what grinding effectiveness is reported against. Same treatment as EN 15437 (R113/R114): **cite by number, never paraphrase clauses.**

### L2. ACV — refrigerant leak / undercharge diagnosis in HVAC and vehicle air conditioning

R251. Guo, F., Chen, Z. and Xiao, F. **Fault detection and diagnosis of electric bus air conditioning systems incorporating domain knowledge and probabilistic artificial intelligence.** *Energy and AI* 16:100364, 2024. https://doi.org/10.1016/j.egyai.2024.100364 — OA PDF https://ira.lib.polyu.edu.hk/bitstream/10397/108221/1/1-s2.0-S2666546824000302-main.pdf. **Gold OA, read first-hand.** The PS3 ACV blueprint: 38 electric-bus AC units, 12 technician-verified faults (5 refrigerant undercharge), **11 of 12 correctly labelled**; per-peer regressions, **median** of peer predictions (robust with up to **1/3 of training peers faulty**), residual = measured − predicted, aggregated as `I = Σ_k sign_k·(R_k/IQR(R_k))²·IQR(SND)` with `sign_k` from a fault × feature direction matrix. Its own six features (pressures, superheat, subcooling, discharge temperature, power) are all unavailable in PS3 — **transfer the architecture, not the feature list**.

R252. Rossi, T. M. and Braun, J. E. **A Statistical, Rule-Based Fault Detection and Diagnostic Method for Vapor Compression Air Conditioners.** *HVAC&R Research* 3(1):19-37, 1997. https://doi.org/10.1080/10789669.1997.10391359. The founding temperature-only residual + directional-rule method. **Verified from the OpenAlex abstract**: "…is capable of detecting about a **5 % loss of refrigerant**, and can distinguish between refrigerant leaks, condenser fouling, evaporator fouling, liquid line restrictions, and compressor valve leakage." That 5 % is the upper bound of what rich instrumentation buys, and the contrast against the ~40 % needed by thermostat-grade methods (R254, R255, R253).

R253. Yoon, Y., Choi, Y. J., Jung, S., Im, P., Luo, J., Sharma, V., Kolar, B. and Safir, I. **Residential HVAC Fault Detection: Field Data Analysis and Interviews with Smart Thermostat Manufacturers.** Oak Ridge National Laboratory technical report **ORNL/TM-2024/3661**, October 2024. https://info.ornl.gov/sites/publications/Files/Pub224924.pdf (PDF resolves live; read first-hand, quotes verbatim). US DOE contractor report, publicly available (`licence UNVERIFIED`). 12 cooling-season field tests at five undercharge levels (−10 % … −50 %): "**Low-intensity refrigerant undercharge faults are difficult to detect using only indoor air temperature data**"; "**Unmet hours … were observed during 40 % and 50 % refrigerant undercharge faults**"; "**Supply-air temperature increased as the refrigerant undercharge increased**"; "**System runtime fraction increased as the refrigerant undercharge increased**". Of ORNL's three key variables only **runtime fraction** survives into PS3's telemetry — which is exactly the running-mode string. The `load_halved` flag is a confound in the same direction.

R254. Chintala, R., Winkler, J. and Jin, X. **Automated fault detection of residential air-conditioning systems using thermostat drive cycles.** *Energy and Buildings* 236:110691, 2021. https://doi.org/10.1016/j.enbuild.2020.110691 — free accepted manuscript https://www.sciencedirect.com/science/article/am/pii/S0378778820334770. 3R2C grey-box identified by extended Kalman filter, predicting cooling times over thermostat drive cycles. Accuracy: duct leak 70 %, 40 % airflow fault 77 %, **40 % refrigerant undercharge 82 %**, no-fault 87 %, on an EnergyPlus model. Establishes **cooling-cycle duration / pull-down rate as the workhorse feature** when only indoor temperature, setpoint and ambient are available; PS3 takes the feature definition and lets the 7 peer cars do the RC model's job.

R255. Chintala, R., Winkler, J., Ramaraj, S. and Jin, X. **Sensitivity analysis of an automated fault detection algorithm for residential air-conditioning systems.** *Applied Thermal Engineering* 238:121895, 2024. https://doi.org/10.1016/j.applthermaleng.2023.121895 — free accepted manuscript https://www.osti.gov/pages/servlets/purl/2274818. The same algorithm on **real** faulted-equipment data (FSEC lab home, seven months): **undercharge 70.6 %**, duct leak + undercharge 85.2 %, duct leak 69.1 %; across nine EnergyPlus house constructions, 71 % no-fault / 77 % at 40 % undercharge. Also shows the grey-box parameter identification is **highly non-convex with several local optima** — the evidence for not putting an identified physical model on a 12 h critical path.

R256. Guo, F. and Rasmussen, B. **Performance benchmarking of residential air conditioning systems using smart thermostat data.** *Applied Thermal Engineering* 225:120195, 2023. https://doi.org/10.1016/j.applthermaleng.2023.120195. Population-as-benchmark anomaly identification with **both between-system and within-system comparison**, stated to detect degradation and **especially refrigerant leaks**, and to verify a repair. `NUMBERS UNVERIFIED` — no abstract retrievable from Crossref/OpenAlex/S2, so the ~9,000-unit population figure and the leak claim come from secondary descriptions. PS3 has **7 peers, not 9,000**: use median/MAD, not mean/σ.

R257. Yoo, J., Hong, S.-B. and Kim, M. S. **Refrigerant leakage detection in an EEV installed residential air conditioner with limited sensor installations.** *International Journal of Refrigeration* 78:157-165, 2017. https://doi.org/10.1016/j.ijrefrig.2017.03.001. Asks precisely the PS3 question — what can still be detected when the informative sensor is missing — and notes that an EEV compensates and hides superheat drift, pushing the signature into capacity and runtime instead. `NUMBERS UNVERIFIED` (Elsevier paywall; abstract elided even in the Semantic Scholar record).

R258. Jiang, M., Chen, H. and Yang, C. **A metro train air conditioning system fault diagnosis method based on explainable artificial intelligence: considering interpretability and generalization.** *International Journal of Refrigeration* 174:47-59, 2025. https://doi.org/10.1016/j.ijrefrig.2025.03.001. The only metro-train-AC FDD paper the sweep found — the domain-framing citation, not an adopted method (it is verified on **simulation** data with a richer sensor set). `NUMBERS UNVERIFIED` (Elsevier paywall).

R259. Kim, M., Yoon, S. H., Domanski, P. A. and Payne, W. V. **Design of a steady-state detector for fault detection and diagnosis of a residential air conditioner.** *International Journal of Refrigeration* 31(5):790-799, 2008. https://doi.org/10.1016/j.ijrefrig.2007.11.008. Moving-window standard deviations over the previous ~5 minutes; its specific lesson is to **include all FDD features in the steady-state detector**, not a favourite pair. PS3 deviates deliberately: gate to *separate* pull-down from steady operation rather than discard the transient, because the pull-down slope is itself a capacity feature. `NUMBERS PARTIALLY UNVERIFIED` (abstract/secondary).

R260. Tormos, B., Sánchez-Márquez, R., Alvis-Sánchez, J. and Bermudez, V. **A Physics-Informed Explainable AI Framework for HVAC Anomaly Detection and Maintenance-Oriented Analysis in Urban Bus Fleets.** *Algorithms* 19(7):586, 2026. https://doi.org/10.3390/a19070586. **CC BY 4.0**. Prescribes **sensor contextualisation and physical mapping of the available measurements onto the vapour-compression cycle** before any modelling — the first hour of PS3 ACV work, and the model for documenting why each channel was or was not used. `NUMBERS UNVERIFIED` (MDPI blocks automated fetching; abstract via Crossref).

R261. Michau, G. and Fink, O. **Unsupervised Fault Detection in Varying Operating Conditions.** IEEE ICPHM 2019. https://doi.org/10.1109/ICPHM.2019.8819383 — preprint https://arxiv.org/abs/1907.06481; related *Knowledge-Based Systems* 216:106816, 2021, https://doi.org/10.1016/j.knosys.2021.106816. Five approaches compared on a fleet of **112 units over one year**; the three fleet-exploiting methods beat a baseline trained on the target unit's own short history. The general PHM statement of the PS3 premise, and the justification for spending the budget on peer normalisation rather than per-car change-point detection over 3-4 days. Code `UNVERIFIED`.

R262. Kim, W. and Braun, J. E. **Performance evaluation of a virtual refrigerant charge sensor** / **Extension of a virtual refrigerant charge sensor.** *International Journal of Refrigeration* 36(3), 2013 and 51, 2015. https://doi.org/10.1016/j.ijrefrig.2012.11.004 ; https://doi.org/10.1016/j.ijrefrig.2014.09.015. Charge estimated within 10 % of actual from **surface-mounted temperature measurements only** — but those are liquid-line and suction-line refrigerant-side temperatures, which PS3 does not expose. The clean **skipped-with-reason** entry: the entire virtual-charge-sensor family is inapplicable here, and an HVAC reviewer will look for it first. `NUMBERS PARTIALLY UNVERIFIED` (abstracts not retrievable; figures from Purdue/secondary summaries).

R263. Li, H. and Braun, J. E. **Decoupling features and virtual sensors for diagnosis of faults in vapor compression air conditioners.** *International Journal of Refrigeration* 30(3):546-564, 2007. https://doi.org/10.1016/j.ijrefrig.2006.07.024 — companion **A Methodology for Diagnosing Multiple Simultaneous Faults**, *HVAC&R Research* 13(3):369-395, 2007, https://doi.org/10.1080/10789669.2007.10390959 — free mirror https://ncesr.unl.edu/wordpress/wp-content/uploads/2013/08/li-decoupling-features-and-virtual-sensors.pdf (terms `unverified`). A feature is diagnostic only if it is **uniquely dependent on one fault and independent of driving conditions and other faults**. PS3's candidate features fail this badly, which is why the deliverable is a *ranking with a stated confounder caveat* rather than a confident diagnosis. `NUMBERS UNVERIFIED`.

R264. Granderson, J., Lin, G., Harding, A., Im, P. and Chen, Y. **Building fault detection data to aid diagnostic algorithm creation and performance testing.** *Scientific Data* 7, 2020. https://doi.org/10.1038/s41597-020-0398-6. Follow-ups: *Scientific Data* 10, 2023, https://doi.org/10.1038/s41597-023-02197-w ; *Scientific Data* 13, 2026 (AHU-only), https://doi.org/10.1038/s41597-025-06179-y. Data at https://faultdetection.lbl.gov/data/. Seven system types, 20 to 100+ points per dataset, each period labelled with which fault at which severity. The only open, labelled, ground-truthed HVAC fault data found — the honest way to dry-run a feature-direction matrix before touching 6 precious cases. **`UNVERIFIED`**: whether the RTU subset contains refrigerant undercharge specifically, and the exact licence (the LBNL landing page states none).

R265. Llopis-Mengual, B., Yuill, D. P. and Navarro-Peris, E. **Time series analysis of field data for soft faults detection and degradation assessment in residential air conditioning systems.** *Applied Thermal Engineering* 269:126104, 2025. https://doi.org/10.1016/j.applthermaleng.2025.126104. A refrigerant leak over 3-4 days is a **soft** fault and will not show a step — the framing for "look at level relative to peers, and slope within the window", not "look for a change point". Licence `PARTIALLY UNVERIFIED` (Crossref lists CC BY-NC-ND 4.0 alongside the Elsevier TDM licence); `NUMBERS UNVERIFIED`.

R266. Guo, F. and Rasmussen, B. **Predictive maintenance for residential air conditioning systems with smart thermostat data using modified Mann-Kendall tests.** *Applied Thermal Engineering* 222:119955, 2023. https://doi.org/10.1016/j.applthermaleng.2022.119955. An autocorrelation-corrected trend test on a health index; ~5 lines with `pymannkendall`. Proposed for PS3 as a **tie-breaker** between two cars at similar levels, not a primary ranker — 3-4 days is short for a trend test and a pre-existing leak shows as a level, not a slope. `NUMBERS UNVERIFIED`.

R267. Gálvez, A., Diez-Olivan, A., Seneviratne, D. and Galar, D. **Fault Detection and RUL Estimation for Railway HVAC Systems Using a Hybrid Model-Based Approach.** *Sustainability* 13(12):6828, 2021. https://doi.org/10.3390/su13126828. **CC BY 4.0**, abstract verified verbatim via Semantic Scholar. Physics-based model generates healthy and faulty data at several degradation levels, fused with measured train-carriage HVAC data: FDD accuracy **92.60 %**, air-filter RUL accuracy **95.21-97.80 %**. Right vehicle and subsystem, wrong fault (air filter) and out of budget — but the best single citation for **why train-HVAC fault data is scarce and will stay scarce** (components are replaced early), which is why PS3 gives 6 cases and why supervised learning is the wrong instinct.

R268. Chen, Z., Xiao, F. and Guo, F. **Similarity learning-based fault detection and diagnosis in building HVAC systems with limited labeled data.** *Renewable and Sustainable Energy Reviews* 185:113612, 2023. https://doi.org/10.1016/j.rser.2023.113612 — OA copy http://hdl.handle.net/10397/108217 (**CC BY-NC-ND**). The limited-labels framing matches PS3, but metric learning across 48 rows with 6 positives is not a defensible use of the budget and the method targets fault *typing* rather than localisation of one known fault. `NUMBERS UNVERIFIED`. **This is the verified paper to cite for few-shot/limited-label HVAC FDD** — see the refuted-item note at the head of this section.

### L3. SHM — fatigue damage estimation from stress time series

R269. Zorman, A., Slavič, J. and Boltežar, M. **Vibration fatigue by spectral methods: a review with open-source support.** *Mechanical Systems and Signal Processing* 190:110149, 2023. https://doi.org/10.1016/j.ymssp.2023.110149. Paper **CC BY 4.0**; code **`FLife`**, MIT, https://github.com/ladisk/FLife (repo confirmed live). 20+ spectral damage methods (narrow-band, Wirsching-Light, Ortiz-Chen, α0.75, Tovo-Benasciutti, Dirlik, Zhao-Baker, Park, Jiao-Moan, Huang-Moan, …) under one API, plus the closed-form narrow-band damage `D = (ν0⁺·T/C)·(√2σ)^k·Γ(1+k/2)` that makes `log D` affine in `log`(amplitude scale). Reported: best methods within **<7 % relative life error** for steel (k = 3.324); for k = 11.76 only Ortiz-Chen, α0.75, Park, Jun-Park and Huang-Moan stay within 25 %; and RFC damage from a fragment converged within 2 % of the 1 h value only after **~2 s (k=3.3), ~150 s (k=7.3), ~2600 s (k=11.8)**. Note for PS3: **no installs are permitted, so the Dirlik and Tovo-Benasciutti formulas are hand-implemented** from this review and R275/R276.

R270. Marsh, G., Wignall, C., Thies, P. R., Barltrop, N. et al. **Review and application of rainflow residue processing techniques for accurate fatigue damage estimation.** *International Journal of Fatigue* 82:757-765, 2016. https://doi.org/10.1016/j.ijfatigue.2015.10.007. After rainflow closes all hysteresis loops a **residue** of unclosed half-cycles remains, and there is no single convention (discard / half cycles / full cycles / repeat-history). The choice materially changes the damage sum and the common conventions are shown to be **non-conservative**. The cheapest, highest-yield experiment for the PS3 label mismatch. `NUMBERS UNVERIFIED` (Elsevier paywall).

R271. Marques, J. M. E., Benasciutti, D. and Tovo, R. **Variability of the fatigue damage due to the randomness of a stationary vibration load.** *International Journal of Fatigue* 141:105891, 2020. https://doi.org/10.1016/j.ijfatigue.2020.105891 — corrigendum https://doi.org/10.1016/j.ijfatigue.2021.106265. Closed-form damage variance / coefficient of variation as a function of record length and the S-N exponent: damage from a single time-history is **one draw from a distribution**. The basis for the PS3 noise-floor bootstrap. `NUMBERS UNVERIFIED` (abstract only).

R272. Haghi, R. and Crawford, C. **Data-driven surrogate model for wind turbine damage equivalent load.** *Wind Energy Science* 9:2039-2062, 2024. https://doi.org/10.5194/wes-9-2039-2024. **CC BY 4.0**. A fully connected net on **three scalars** (mean wind, turbulence intensity, shear) reaches **R² = 0.988 / NRMSE 2.34 %** for blade-root edgewise DEL, and a temporal-convolution network on the raw time series adds only **+0.003 R²**. The independent confirmation of verdict V1 for this subsystem, and the reason a 1D-CNN on 581k raw samples with 64 training files is a SKIP.

R273. Guo, X., Li, Y., Gui, W. and Hu, J. **Neural network approaches for real-time fatigue life estimation by surrogating the rainflow counting method.** *International Journal of Fatigue* 197:108941, 2025. https://doi.org/10.1016/j.ijfatigue.2025.108941. NN surrogate for RFC+Miner on a load history. `NUMBERS UNVERIFIED` (abstract only).

R274. Proner, E. and Mucchi, E. **A multi-axial Fatigue Damage Spectrum for the evaluation of the fatigue damage potential of multi-axis random vibration environments.** *Mechanical Systems and Signal Processing* 226:112362, 2025. https://doi.org/10.1016/j.ymssp.2025.112362. The **Fatigue Damage Spectrum** — damage as a function of SDOF natural frequency, computed by pushing the signal through a bank of SDOF filters (Q ≈ 10) and rainflow-counting each output. If the PS3 label was computed on an SDOF-transformed channel rather than the raw one, raw-channel rainflow will never match at any exponent but FDS band features will. ~40 lines. `NUMBERS UNVERIFIED` (Elsevier paywall). Free but **not peer-reviewed** step-by-step alternative: Halfpenny, *Accelerated vibration testing based on fatigue damage spectra*, https://www.vibrationdata.com/tutorials_alt/fatigue_damage_spectra.pdf — cite R274, not the tutorial.

R275. Benasciutti, D. and Tovo, R. **Spectral methods for lifetime prediction under wide-band stationary random processes.** *International Journal of Fatigue* 27(8):867-877, 2005. https://doi.org/10.1016/j.ijfatigue.2004.10.007. Companion benchmark: **Comparison of spectral methods for fatigue analysis of broad-band Gaussian random processes**, *Probabilistic Engineering Mechanics* 21(4):287-299, 2006, https://doi.org/10.1016/j.probengmech.2005.10.003. The Tovo-Benasciutti method itself, and the head-to-head against narrow-band, Wirsching-Light, Dirlik and Zhao-Baker. `NUMBERS UNVERIFIED` (Elsevier paywall).

R276. Dirlik, T. and Benasciutti, D. **Dirlik and Tovo-Benasciutti spectral methods in vibration fatigue: a review with a historical perspective.** *Metals* 11(9):1333, 2021. https://doi.org/10.3390/met11091333. **CC BY 4.0**. The authors' own retrospective on the two dominant spectral methods — the source to hand-implement them from. `NUMBERS UNVERIFIED` (MDPI 403 on fetch).

R277. Wang, X. and Serra, R. **Vibration fatigue damage estimation by new stress correction based on kurtosis control of random excitation loadings.** *Sensors* 21(13):4518, 2021. https://doi.org/10.3390/s21134518. **CC BY 4.0**. Non-Gaussian (kurtosis/skewness) correction to Gaussian damage estimates — one cheap feature, and the diagnostic tells you whether it matters. `NUMBERS UNVERIFIED`.

R278. Yuan, S., Peng, W. and Sun, J. **An artificial neural network model for fatigue damage analysis of wide-band non-Gaussian random processes.** *Applied Ocean Research* 144:103896, 2024. https://doi.org/10.1016/j.apor.2024.103896. ANN mapping spectral + higher-moment features to a damage correction. `NUMBERS UNVERIFIED`.

R279. Johannesson, P. **Extrapolation of load histories and spectra.** *Fatigue & Fracture of Engineering Materials & Structures* 29(3):209-217, 2006. https://doi.org/10.1111/j.1460-2695.2006.00982.x. Rainflow-matrix / load-spectrum extrapolation from a short record (implemented in WAFO, GPL). Cited in PS3 as a reasoned SKIP — we are not extrapolating to a longer life; the record *is* the label's domain. `NUMBERS UNVERIFIED`.

R280. Farid, M. **Data-driven method for real-time prediction and uncertainty quantification of fatigue failure under stochastic loading using artificial neural networks and Gaussian process regression.** *International Journal of Fatigue* 155:106415, 2022. https://doi.org/10.1016/j.ijfatigue.2021.106415 — preprint https://arxiv.org/abs/2103.08349. ANN + GPR for damage with calibrated uncertainty. `NUMBERS UNVERIFIED`.

R281. Gibson, S. J., Rogers, T. J. and Cross, E. J. **Distributions of fatigue damage from data-driven strain prediction using Gaussian process regression.** *Structural Health Monitoring* 22(4), 2023. https://doi.org/10.1177/14759217221140080. Propagates GP strain-prediction uncertainty *through* rainflow into a damage distribution — the principled route to an error bar on a damage number. `NUMBERS UNVERIFIED`.

R282. Xiu, R., Spiryagin, M., Wu, Q., Yang, S. et al. **Fatigue life assessment methods for railway vehicle bogie frames.** *Engineering Failure Analysis* 116:104725, 2020. https://doi.org/10.1016/j.engfailanal.2020.104725. Names the governing standards (**EN 13749, UIC 615-4, JIS E 4207**) and the four-part validation route (FEM, static, fatigue-spectrum test, line test) — what "cumulative damage per file" means to a rail engineer. `NUMBERS UNVERIFIED`.

R283. Ji, Y., Sun, S., Li, Q. and Ren, Z. **Realistic fatigue damage assessment of a high-speed train bogie frame by damage-consistency load spectra based on measured field load.** *Measurement* 166:108164, 2020. https://doi.org/10.1016/j.measurement.2020.108164. "Damage-consistency" calibration coefficients per load amplitude — the published precedent for **fitting a correction so that a spectrum reproduces a reference damage**, which is structurally what the PS3 SHM task requires. `NUMBERS UNVERIFIED`.

R284. Li, Q., Ren, Z., Wu, S. and An, Q. **Fatigue damage assessment of high-speed train bogie frame load spectra based on phase reconstruction.** *Engineering Failure Analysis* 159:108008, 2024. https://doi.org/10.1016/j.engfailanal.2024.108008. Damage depends on signal **phase**, not only on the amplitude spectrum — a direct warning against pure-PSD feature sets. `NUMBERS UNVERIFIED`.

R285. Kraft, S., Blum, M. and Gomes Alves, D. **Calibration and validation of fatigue design models for railway car bodies considering uncertainty.** *Fatigue & Fracture of Engineering Materials & Structures* 46(11), 2023. https://doi.org/10.1111/ffe.14160. **EN 12663-1** car-body design is static-equivalent; real fatigue design needs operating loads. The uncertainty framing for PS3's error bars. `NUMBERS UNVERIFIED`.

R286. Lagerblad, U., Wentzel, H. and Kulachenko, A. **A methodology for strain-based fatigue damage prediction by combining finite element modelling with vibration measurements.** *Engineering Failure Analysis* 121:105130, 2021. https://doi.org/10.1016/j.engfailanal.2020.105130. Measurement-point → critical-point damage transfer, lab-validated. Supports the hypothesis that a PS3 label may be damage at a hot spot rather than at the published gauge. `NUMBERS UNVERIFIED`.

R287. Gulgec, N. S., Takáč, M. and Pakzad, S. N. **Structural sensing with deep learning: strain estimation from acceleration data for fatigue assessment.** *Computer-Aided Civil and Infrastructure Engineering* 35(12), 2020. https://doi.org/10.1111/mice.12565. Learned proxy-channel → strain mapping for fatigue; the precedent for "the published channel is not the damaging channel". `NUMBERS UNVERIFIED`.

R288. Maghsood, R. and Wallin, J. **Online estimation of driving events and fatigue damage on vehicles.** arXiv:1603.06455, 2016. https://arxiv.org/abs/1603.06455. **NOT PEER-REVIEWED** — stat.AP only; no journal version found. HMM + online EM for on-board damage estimation from a mixture of driving events — a candidate explanation for a **bimodal** damage-label distribution (two operating regimes). `NUMBERS UNVERIFIED`.

R289. **Interannual variability in fatigue damage estimation from short-term strain monitoring of offshore wind turbines.** *Wind Energy Science Discussions*, wes-2026-65. https://doi.org/10.5194/wes-2026-65. **NOT PEER-REVIEWED** — preprint under review, as cited. Reports up to 30 % deviation in long-term mean damage from window choice but <1 % within-year bootstrap uncertainty; the closest published treatment of "how much of the error is irreducible". Numbers indicative only.

R290. **ASTM E1049-85 (2017)** *Standard Practices for Cycle Counting in Fatigue Analysis*. **Paywalled standard, not opened — `UNVERIFIED`.** The normative definition of 3-point rainflow, level-crossing, peak and range-pair counting; cite by number, never paraphrase clauses. (`rainflow` on PyPI claims to implement it; `fatpack` implements the 4-point variant — the two disagree on residues, which is the point of R270.)

R291. **Haibach two-slope S-N modification** (`k* = 2k − 1` below the knee). **`SECONDARY-SOURCE` — no primary document was opened**; the rule was taken from search summaries and topic pages, not from Haibach's own text or a paper stating it. Listed because the knee/two-slope choice is a large lever on damage when the small-amplitude tail is huge (581k samples per file), **but it must not appear in a deliverable until someone reads a primary source.** The reviews that were read (R269, R282) use single-slope curves.

### L4. Time-series data augmentation for small, imbalanced datasets

R292. Iwana, B. K. and Uchida, S. **An empirical survey of data augmentation for time series classification with neural networks.** *PLOS ONE* 16(7):e0254841, 2021. https://doi.org/10.1371/journal.pone.0254841 — code https://github.com/uchidalab/time_series_augmentation (confirmed live; repo licence `UNVERIFIED`). Paper **CC BY 4.0**. 12 augmentation methods × **128 UCR datasets** × 6 architectures. **Window warping has the highest average rank for VGG, ResNet and LSTM; slicing is second**; **rotation/flipping decreased accuracy for all six architectures**; permutation degrades badly by breaking temporal order; and there is a **negative correlation between accuracy gain and training-set size** — the gain is largest exactly at PS3's scale. The empirical backbone of the addendum's augmentation section.

R293. Le Guennec, A., Malinowski, S. and Tavenard, R. **Data Augmentation for Time Series Classification using Convolutional Neural Networks.** ECML/PKDD Workshop on Advanced Analytics and Learning on Temporal Data (AALTD), 2016. **No DOI exists** — the verifier confirmed the absence is a real gap in the record, not a wrong guess; cite by title and venue. The origin of **window slicing** and **window warping** (≈20 lines each). Its key caveat is the decision rule PS3 uses throughout: slicing assumes **the label is carried by every sub-window** — true for steady vibration, false for a door cycle whose signature lives in one phase, false for a cumulative-damage regression. Original numbers `UNVERIFIED`.

R294. Wen, Q., Sun, L., Yang, F., Song, X., Gao, J., Wang, X. and Xu, H. **Time Series Data Augmentation for Deep Learning: A Survey.** IJCAI 2021. https://doi.org/10.24963/ijcai.2021/631 — preprint https://arxiv.org/abs/2002.12478. Survey plus experiments across classification, anomaly detection and forecasting: on 5,000 Alibaba Cloud samples augmentation gives **+0.11 % to +1.92 %**; on Yahoo AD with a U-Net, raw F1 0.403 → decomposition 0.662 → decomposition + augmentation 0.693; forecasting ranges from **+76 % to −16 %**. The honest citation for "augmentation is not free", and the source of the **label-expansion** trick for episode-structured anomalies.

R295. Gao, Z., Liu, H. and Li, L. **Data Augmentation for Time-Series Classification: An Extensive Empirical Study and Comprehensive Survey.** *Journal of Artificial Intelligence Research* 83, 2025. https://doi.org/10.1613/jair.1.17084 — preprint https://arxiv.org/abs/2310.10060. **Corrected**: the finder flagged this DOI as unverified; it resolves cleanly. ~20 strategies × 15 UCR datasets × ResNet/LSTM; reports **random geometric warps and random permutation** as significant improvements and EMD-based decomposition as ineffective. **Note the live disagreement with R292**, which ranks permutation among the worst over 128 datasets and 6 architectures; R292 is the larger, more careful study and governs PS3's ordered door cycles. Citing both with the conflict named is stronger than citing either.

R296. Ilbert, R., Hoang, T. V. and Zhang, Z. **Data Augmentation for Multivariate Time Series Classification: An Experimental Study.** MulTiSA workshop @ ICDE 2024. https://arxiv.org/abs/2406.06518. **Workshop-reviewed — weaker evidence than a main track.** The only study that measures augmentation **through ROCKET** (and InceptionTime) on 13 UEA multivariate datasets: accuracy improved on **10/13** with **≈ +1.55 % mean relative improvement** over the un-augmented ROCKET baseline, and gains do not track baseline accuracy. **This is the number to quote when a judge asks whether augmentation helped.**

R297. Forestier, G., Petitjean, F., Dau, H. A., Webb, G. I. and Keogh, E. **Generating Synthetic Time Series to Augment Sparse Datasets.** IEEE ICDM 2017, pp. 865-870. https://doi.org/10.1109/icdm.2017.106 — author PDF https://germain-forestier.info/publis/icdm2017.pdf. **Corrected**: the finder flagged the DOI as unverified; it exists as given here. Weighted DTW Barycentre Averaging with three weighting schemes, evaluated with 1-NN DTW at 1-2 training samples per class. Code: `tslearn.barycenters.dtw_barycenter_averaging` (BSD-3). **Exact gains `UNVERIFIED`** (result tables did not extract); the assertable claim is substantial improvement in the extreme few-shot regime. R292 reports wDBA generates **low-diversity** samples, so treat it as a complement to window warping, and note pattern-mixing methods cost 60+ s per generated sample.

R298. Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D. and Le, Q. V. **SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.** Interspeech 2019. https://doi.org/10.21437/Interspeech.2019-2680. Time warping + **time masking** + **frequency masking** on the spectrogram; implemented in `torchaudio` (Apache-2.0). The cheapest real augmenter wherever a spectrogram or wavelength spectrum is the input, ~1 h. Assumes the class is **redundantly encoded across time and frequency**, so mask widths must stay small relative to the defect/corrugation band spacing.

R299. Yao, H., Wang, Y., Pan, L., Huang, Z. and Finn, C. **C-Mixup: Improving Generalization in Regression.** NeurIPS 2022. https://arxiv.org/abs/2210.05775 — code https://github.com/huaxiuyao/C-Mixup (confirmed live; licence `UNVERIFIED`). Mixup with pair-sampling weighted by **label similarity**: +6.56 % in-distribution generalisation, +4.76 % task generalisation, +5.82 % OOD robustness over the best prior approach across 11 datasets. **The published fix for "vanilla mixup on regression labels can result in arbitrarily incorrect labels"** — i.e. the PS3 SHM case, where cropping preserves nothing and whole-file mixing is the only option that needs no sub-window label stationarity.

R300. Schneider, N., Goshtasbpour, S. and Perez-Cruz, F. **Anchor Data Augmentation.** NeurIPS 2023. https://arxiv.org/abs/2311.06965 — code https://github.com/NoraSchneider/anchordataaugmentation (confirmed live). Extends anchor regression: cluster to define anchors, sample a regularisation strength γ, emit several modified replicas per sample. Competitive with R299 on the same in-distribution and OOD benchmarks. Run it **only if** C-Mixup's bandwidth proves fiddly — they solve the same problem.

R301. Buda, M., Maki, A. and Mazurowski, M. A. **A systematic study of the class imbalance problem in convolutional neural networks.** *Neural Networks* 106:249-259, 2018. https://doi.org/10.1016/j.neunet.2018.07.011 — preprint https://arxiv.org/abs/1710.05381. **Oversampling is the dominant remedy and, unlike in classical ML, does not cause overfitting in CNNs**, and **thresholding by class prior should almost always be applied**. Honest caveat: this is **image** evidence applied by analogy to 1-D signals, and it does not govern ROCKET+ridge or LightGBM — R302 does.

R302. Elor, Y. and Averbuch-Elor, H. **To SMOTE, or not to SMOTE?** arXiv:2201.08528, 2022. https://arxiv.org/abs/2201.08528 — code https://github.com/aws/to-smote-or-not. **NOT PEER-REVIEWED** (venue could not be confirmed). Across 73 tabular datasets, balancing helps **weak** classifiers but **gives no improvement for strong, properly tuned classifiers** (XGBoost/CatBoost class) once a proper metric and consistent hyper-parameter selection are used. The operative result for a LightGBM/stacking workhorse: expect class weighting to be a wash and **publish that ablation** rather than quietly using SMOTE.

R303. Kapoor, S. and Narayanan, A. **Leakage and the reproducibility crisis in machine-learning-based science.** *Patterns* 4(9):100804, 2023. https://doi.org/10.1016/j.patter.2023.100804 — preprint https://arxiv.org/abs/2207.07048. **Corrected**: the finder supplied a cell.com URL and PubMed id but no DOI; the DOI is as given. A taxonomy of **8 leakage types** found across 17 fields and 294 papers, plus model info sheets (https://reproducible.cs.princeton.edu/). Augmenting before the split, and any transductive use of held-out statistics, are *named* leakage types. Pair with R220's 20-60 % → 99.9 % inflation figure and R115's group-split requirement.

R304. Ang, Y., Huang, Q., Bao, Y., Tung, A. K. H. and Huang, Z. **TSGBench: Time Series Generation Benchmark.** *PVLDB* 17(3):305-318, 2024 (Best Research Paper nomination). https://doi.org/10.14778/3632093.3632097 — code https://github.com/YihaoAng/TSGBench (confirmed live). 10 generation methods × 10 real datasets × 12 measures plus a domain-adaptation generalisation test. **The citation that justifies skipping GAN/diffusion synthesis**: the field needed a benchmark precisely because generated-series quality is contested and method rankings are unstable across measures. (Crossref's indexed date reads 2023; PVLDB vol. 17 is the 2024 VLDB cycle, so 2024 is the correct citation year.)

R305. Kulevome, D. K. B., Wang, H., Cobbinah, B. M., Mawuli, E. S. and Kumar, R. **Effective time-series Data Augmentation with Analytic Wavelets for bearing fault diagnosis.** *Expert Systems with Applications* 249(A):123536, 2024. https://doi.org/10.1016/j.eswa.2024.123536. **Corrected**: the finder flagged the DOI as unverified; it is as given. Generates synthetic **scalograms** online by varying the decay/compress parameter of generalised Morse wavelets, with **no separate generative training stage** — the same argument to make against GANs. Relevant only on a CWT/scalogram route. `NUMBERS UNVERIFIED` (ScienceDirect abstract only).

R306. Zhang, X., Zhen, D., Feng, G., Cui, L., Zhang, H. and Gu, F. **Resonance-aware digital twin-driven data augmentation for bearing fault diagnosis under sample imbalance.** *Structural Health Monitoring*, 2026. https://doi.org/10.1177/14759217261462579. Adaptively extracts the optimal resonance band from measured fault signals to fit dynamic-model parameters, minimising the simulated-vs-measured resonance gap; framed as *interpretable* augmentation under imbalance. The precedent that **physics-parameterised augmentation is the accepted answer to sample imbalance in PHM** — the bearing-side twin of R68's door argument. `NUMBERS UNVERIFIED`.

R307. Yoon, J., Jarrett, D. and van der Schaar, M. **Time-series Generative Adversarial Networks (TimeGAN).** NeurIPS 2019. https://www.vanderschaar-lab.com/papers/NIPS2019_TGAN_Main.pdf — code https://github.com/jsyoon0823/TimeGAN (confirmed live); proceedings URL `UNVERIFIED`. The canonical time-series GAN. Cited in PS3 as a consciously-not-bought SKIP: with 14-30 positives a generator has nothing to learn that is not memorisation, and the generator/discriminator balance eats the whole subsystem budget.

R308. **ReF-DDPM: A novel DDPM-based data augmentation method for imbalanced rolling bearing fault diagnosis.** *Reliability Engineering & System Safety*, 2024. https://doi.org/10.1016/j.ress.2024.110343. Companion: **Denoising diffusion probabilistic model-enabled data augmentation method for intelligent machine fault diagnosis.** *Engineering Applications of Artificial Intelligence*, 2024/2025. https://doi.org/10.1016/j.engappai.2024.109520. The current SOTA direction for generative PHM augmentation, cited so the ladder shows what was not built. Third-party comparisons report DDPM ≈ 93.9 % and DDIM ≈ 95.8 % with ResNet at imbalance ratio 0.98 versus DCGAN/BAGAN/TransGAN ≈ 94.8 % — **all `NUMBERS UNVERIFIED`**, and they are accuracy figures on balanced-test protocols, i.e. the metric family this project already rejects. Consistent with the existing ImDiffusion SKIP (R18).

R309. Yang, H. and Desell, T. **Robust Augmentation for Multivariate Time Series Classification.** arXiv:2201.11739, 2022. https://arxiv.org/abs/2201.11739. **NOT PEER-REVIEWED**; preprint CC BY-NC-SA 4.0. Cutout, cutmix, mixup and window warp on 26 UEA multivariate datasets; **InceptionTime + augmentation improves accuracy by 1-45 % on 18 of them**. The 1-45 % range is enormous and per-dataset — **quote the median, not the maximum**.

R310. Demirel, B. U. and Holz, C. **Finding Order in Chaos: A Novel Data Augmentation Method for Time Series in Contrastive Learning.** NeurIPS 2023. https://arxiv.org/abs/2309.13439 — code https://github.com/eth-siplab/Finding_Order_in_Chaos (confirmed live; licence `UNVERIFIED`). A mixup variant treating **phase and magnitude as separate features** to respect quasi-periodic, non-stationary structure — the most physically appropriate mixup for cyclic machine signals, but evaluated in a contrastive-pretraining setting this project does not run.

R311. Zhang, X., Zhao, Z., Tsiligkaridis, T. and Zitnik, M. **Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency (TF-C).** NeurIPS 2022. https://arxiv.org/abs/2206.08496 — code https://github.com/mims-harvard/TFC-pretraining (confirmed live). Pretrain on unlabelled signal, fine-tune on the handful of labels — the structurally correct answer to label scarcity, and **the alternative to augmentation rather than a form of it**. The quoted "+15.4 % F1" and "1-10 % of labels matches full supervision" figures are **`NUMBERS UNVERIFIED`** against the papers themselves; the semi-supervised literature also reports TS-TCC/TS2Vec *degrading* at very small label counts, so at 14 positives this is a gamble, not a plan.

R312. Chen, M., Xu, Z., Zeng, A. and Xu, Q. **FrAug: Frequency Domain Augmentation for Time Series Forecasting.** arXiv:2302.09292, 2023. https://arxiv.org/abs/2302.09292. **NOT PEER-REVIEWED** (OpenReview submission `G0uzEweZB1`, acceptance unconfirmed). FreqMask and FreqMix preserve the semantic consistency of the data-label pair in forecasting, where time-domain augmentation breaks fine-grained temporal relationships. The forecasting counterpart of R298; relevant only to a forecast-residual detector arm. `NUMBERS UNVERIFIED`.

R313. de Souza, D. and Leao, B. **Data Augmentation of Multivariate Sensor Time Series using Autoregressive Models and Application to Failure Prognostics.** arXiv:2410.16419, 2024 (Siemens). https://arxiv.org/abs/2410.16419. **NOT PEER-REVIEWED.** Time-varying AR with a decoupling trick for mean and covariance dynamics; on C-MAPSS FD001/FD003 with five real plus five synthetic samples, RMSE −2 % / −6 % and scoring function −4 % / −20 %. The closest published match to tiny-sample multivariate sensor regression, but n=5 makes the effect size statistically thin. No code.

R314. Qiu, C., Pfrommer, T., Kloft, M., Mandt, S. and Rudolph, M. **Neural Transformation Learning for Deep Anomaly Detection Beyond Images (NeuTraL-AD).** ICML 2021. https://proceedings.mlr.press/v139/qiu21a.html — code https://github.com/boschresearch/NeuTraL-AD (confirmed live; AGPL-3.0 `UNVERIFIED`). *Learns* the transformations instead of hand-picking them — the principled answer to "which augmentation?". Cited as the learned-augmentation frontier, alongside the existing TPA-AD pseudo-anomaly note (R23); it is a detector, not an augmenter, so adopting it would displace a ladder entry rather than add to one.


---

## Items deliberately NOT cited

- **TimeRCD** (arXiv 2509.21190) — marked WITHDRAWN on arXiv (v4). Do not cite.
- **STAR** (arXiv 2510.16014), **RATFM** (arXiv 2506.02081), **THEMIS** (arXiv 2510.03911), **VAN-AD**, **BearingFM** (ScienceDirect S0925527324001762), **"Revisiting WEASEL 2.0"** (arXiv 2608.18021), **"Rethinking Evaluation in the Era of Time Series Foundation Models"** (arXiv 2510.13654), **"Zero-shot Multivariate Time Series Forecasting Using Tabular Prior Fitted Networks"** (arXiv 2604.08400) — seen in search results but **not verified** this session. `UNVERIFIED` — do not cite without opening them first.
