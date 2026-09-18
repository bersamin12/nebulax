# Benchmark configurations

`core.yaml` combines the MetroPT-3, Ottawa bearing and synthetic pneumatic/bearing blocks. The legacy Cranfield and synthetic Door research blocks have been removed; PS3 Door remains a separate scored task.

The source presets are `metropt_core.yaml`, `ottawa_core.yaml` and `synth_core.yaml`. Run `python scripts/run_bench.py --config configs/core.yaml --dry-run` to inspect the current experiment matrix. `model_ladder.yaml` retains the brake, bearing and four PS3 requirements.
