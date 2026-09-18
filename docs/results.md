# NEBULA X research results — brake air supply and axle bearings

The pre-release research retained in this project covers pneumatic air supply and axle bearings. The four scored Problem Statement 3 tasks, including the organiser's Door data and model, are documented separately in [the PS3 write-up](ps3_writeup.md).

The [research leaderboard](../results/leaderboard.md) is generated from 285 retained benchmark rows in `results/runs.parquet`: MetroPT-3, Ottawa, and pneumatic/bearing synthetic fleet experiments. The generated report records each population, split, validation selection rule, test metric and failure count. `results/ablation_heatmap.html` shows the retained ablations.

| Research population | Selected example | Test result | Operational context |
|---|---|---|---|
| Synthetic bearing fleet | `cusum_cycle_scalar` | VUS-PR 0.658; 10/12 events detected | 0.008 false alarm episodes per scored day |
| Synthetic pneumatic fleet | `sparse_autoencoder` | VUS-PR 0.780; 3/3 events detected | 0.110 false alarm episodes per scored day |
| MetroPT-3 compressor | `lgbm_residual` shipped default | VUS-PR 0.587; 4/4 events detected | 0.105 false alarm episodes per scored day; selection was deferred because the validation slice lacked faults |

The Ottawa bearing recordings are a laboratory proxy, with bearing-grouped evaluation and fabricated timeline; use the leaderboard's classification results rather than interpreting its synthetic false-alarm clock as field evidence. The benchmark is a research demonstration, separate from the four organiser-scored PS3 CSV outputs. Raw MetroPT and Ottawa recordings and the corresponding derived feature caches are retained locally for reproduction; Git tracks the source, reports, model artifacts and a compact fleet replay sample.
