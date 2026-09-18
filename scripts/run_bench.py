#!/usr/bin/env python
"""Run the NEBULA X model benchmark from a config yaml.

    python scripts/run_bench.py --config configs/core.yaml
    python scripts/run_bench.py --config configs/core.yaml --dry-run
    python scripts/run_bench.py --config configs/core.yaml --limit 5 --max-workers 2
    python scripts/run_bench.py --config configs/core.yaml --report-only

Every completed run writes ``<out>/<config_hash>.parquet`` and the sweep writes the combined
``<out>/../runs.parquet``, the per-run split audit under ``<out>/../splits/`` and, unless
``--no-report``, ``results/leaderboard.md`` + ``results/ablation_heatmap.html``.

Re-running the same config is safe and cheap: specs whose ``<config_hash>.parquet`` already
exists are skipped (``--no-skip-existing`` forces a redo), and the feature cache under
``data/features/`` means the datasets are parsed once, not once per model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebulax.bench import report as report_mod  # noqa: E402
from nebulax.bench import runner as runner_mod  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/core.yaml", help="benchmark config yaml (see runner.expand)")
    p.add_argument("--limit", type=int, default=None, help="run only the first N expanded specs")
    p.add_argument("--dry-run", action="store_true", help="expand and print the specs, run nothing")
    p.add_argument("--max-workers", type=int, default=4, help="classical worker processes; 0 = run inline")
    p.add_argument("--out", default="results/runs", help="per-run parquet directory")
    p.add_argument(
        "--max-minutes-per-run", type=float, default=None, help="override every spec's wall-clock cap (minutes)"
    )
    p.add_argument("--no-skip-existing", action="store_true", help="re-run specs that already have a parquet")
    p.add_argument("--cache-dir", default="data/features", help="feature cache directory ('' disables it)")
    p.add_argument("--no-report", action="store_true", help="skip the leaderboard / heatmap step")
    p.add_argument("--report-only", action="store_true", help="only rebuild the leaderboard and heatmap")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _bound_thread_pools(max_workers: int) -> None:
    """Give each worker process its share of the cores, not all of them.

    LightGBM / XGBoost / sklearn default to every core (``n_jobs=-1``, OpenMP); with four
    workers that is 4x oversubscription and single fits that should take a minute time out at
    the cap. The env vars are inherited by the subprocess workers. An explicit value already in
    the environment is respected.
    """
    import os

    n_cpu = os.cpu_count() or 4
    share = max(1, n_cpu // max(1, int(max_workers)))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(share))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    results_dir = Path(args.out).parent

    if args.report_only:
        print(report_mod.leaderboard(results_dir))
        print(report_mod.ablation_heatmap(results_dir))
        return 0

    _bound_thread_pools(args.max_workers)
    df = runner_mod.run(
        args.config,
        out_dir=args.out,
        max_workers=args.max_workers,
        max_minutes_per_run=args.max_minutes_per_run,
        skip_existing=not args.no_skip_existing,
        limit=args.limit,
        dry_run=args.dry_run,
        cache_dir=args.cache_dir or None,
    )

    if args.dry_run:
        cols = [c for c in ("block", "dataset", "subsystem", "task", "model", "split", "window", "config_hash") if c in df]
        print(df[cols].to_string(index=False) if len(df) else "(no specs)")
        print(f"\n{len(df)} spec(s) would run.")
        return 0

    n_ok = int((df["status"] == "ok").sum()) if len(df) else 0
    print(f"{len(df)} row(s): {n_ok} ok, {len(df) - n_ok} failed/timeout -> {results_dir / 'runs.parquet'}")
    if not args.no_report and len(df):
        print(report_mod.leaderboard(results_dir))
        print(report_mod.ablation_heatmap(results_dir))
    return 0 if (len(df) == 0 or n_ok > 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
