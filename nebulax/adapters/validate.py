"""CLI: run one adapter and validate every table it returns against ``nebulax.schema``.

::

    /home/administrator/miniconda3/envs/nebulax/bin/python -m nebulax.adapters.validate \\
        --source metropt3 --raw data/raw/metropt3

Prints row counts, date range, signals present, components and fault-log rows, then either
``OK`` (exit 0) or the first schema violation (exit 1). The adapter module is imported
lazily, so a missing adapter is a clear message rather than an ImportError at startup.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from nebulax.adapters import ADAPTER_SUBSYSTEM, DEFAULT_RAW_DIRS, resolve_adapter
from nebulax.schema import (
    Dataset,
    SIGNALS,
    validate_events,
    validate_fault_log,
    validate_features,
    validate_long,
)

__all__ = ["main", "describe", "validate_dataset"]


def _fmt_ts(ts: Any) -> str:
    return "-" if ts is None or pd.isna(ts) else pd.Timestamp(ts).isoformat()


def describe(ds: Dataset, source: str) -> str:
    """Human-readable report: row counts, date range, signals present, fault-log rows."""
    s = ds.summary()
    lines: list[str] = []
    lines.append(f"source            : {source}  (proxy for subsystem: {ADAPTER_SUBSYSTEM.get(source, '?')})")
    lines.append(f"telemetry rows    : {s['n_long']:,}")
    lines.append(f"feature rows      : {s['n_features']:,}")
    lines.append(f"fault-log rows    : {s['n_fault_log']:,}")
    lines.append(f"event rows        : {s['n_events']:,}")
    lines.append(f"date range (UTC)  : {_fmt_ts(s['t_min'])}  ->  {_fmt_ts(s['t_max'])}")
    if s["t_min"] is not None and s["t_max"] is not None and s["n_long"]:
        span = pd.Timestamp(s["t_max"]) - pd.Timestamp(s["t_min"])
        lines.append(f"span              : {span}")
    lines.append(f"sources in table  : {s['sources']}")
    lines.append(f"subsystems        : {s['subsystems']}")
    lines.append(f"components        : {s['components'][:16]}{' ...' if len(s['components']) > 16 else ''}")

    if s["n_long"]:
        for sub in s["subsystems"]:
            present = sorted(
                set(
                    ds.long.loc[ds.long["subsystem"].astype("string") == sub, "signal"]
                    .astype("string")
                    .dropna()
                    .unique()
                    .tolist()
                )
            )
            expected = list(SIGNALS.get(sub, ()))
            missing = [x for x in expected if x not in present]
            lines.append(f"signals [{sub}]    : {len(present)}/{len(expected)} present -> {present}")
            if missing:
                lines.append(f"  not emitted     : {missing}")
        counts = (
            ds.long.groupby(["subsystem", "signal"], observed=True)["value"]
            .agg(["size", lambda v: float(v.isna().mean())])
            .rename(columns={"size": "rows", "<lambda_0>": "nan_frac"})
        )
        lines.append("rows per signal   :")
        for (sub, sig), row in counts.iterrows():
            lines.append(f"  {sub:<10} {sig:<18} {int(row['rows']):>10,}  NaN {row['nan_frac'] * 100:5.2f} %")
    if s["n_fault_log"]:
        lines.append("fault log         :")
        cols = ["component_id", "fault_type", "t_onset", "t_failure", "t_functional_failure"]
        for _, row in ds.fault_log[cols].iterrows():
            lines.append(
                f"  {row['component_id']:<12} {row['fault_type']:<20} "
                f"{_fmt_ts(row['t_onset'])} -> {_fmt_ts(row['t_failure'])}"
            )
    if ds.meta:
        keys = sorted(k for k in ds.meta if k != "partitions")
        lines.append(f"meta keys         : {keys}")
    return "\n".join(lines)


def validate_dataset(ds: Dataset, *, require_labels: bool = False, strict: bool = False) -> list[str]:
    """Validate all four tables, returning the list of failures (empty == all good)."""
    problems: list[str] = []
    checks = (
        ("long", lambda: validate_long(ds.long, strict=strict)),
        ("features", lambda: validate_features(ds.features, require_labels=require_labels) if len(ds.features) else None),
        ("fault_log", lambda: validate_fault_log(ds.fault_log, strict=strict)),
        ("events", lambda: validate_events(ds.events, strict=strict)),
    )
    for table, check in checks:
        try:
            check()
        except ValueError as exc:
            problems.append(f"[{table}] {exc}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit code (0 ok, 1 validation/adapter failure)."""
    parser = argparse.ArgumentParser(
        prog="python -m nebulax.adapters.validate",
        description="Run one dataset adapter and validate its tables against nebulax.schema.",
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=sorted(DEFAULT_RAW_DIRS),
        help="which adapter to run (metropt3 | metropt2 | cranfield | ottawa | sim)",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="raw data directory; default = data/raw/<source>",
    )
    parser.add_argument("--strict", action="store_true", help="reject columns outside the schema too")
    parser.add_argument("--require-labels", action="store_true", help="require the label columns on the feature table")
    parser.add_argument("--quiet", action="store_true", help="print only the verdict")
    args = parser.parse_args(argv)

    raw_dir = args.raw if args.raw is not None else Path(DEFAULT_RAW_DIRS[args.source])
    try:
        module = resolve_adapter(args.source)
    except (ValueError, ModuleNotFoundError, AttributeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if not raw_dir.exists():
        print(f"FAIL: raw directory {raw_dir} does not exist (pass --raw, or run scripts/download_data.py)", file=sys.stderr)
        return 1

    print(f"adapter           : {module.__name__}.load({raw_dir})")
    try:
        ds = module.load(raw_dir)
    except Exception as exc:  # noqa: BLE001 - the CLI reports, it does not swallow
        print(f"FAIL: {module.__name__}.load raised {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    if not isinstance(ds, Dataset):
        print(
            f"FAIL: {module.__name__}.load returned {type(ds).__name__}, expected nebulax.schema.Dataset",
            file=sys.stderr,
        )
        return 1

    if not args.quiet:
        print(describe(ds, args.source))

    problems = validate_dataset(ds, require_labels=args.require_labels, strict=args.strict)
    if problems:
        print(f"\nFAIL: {len(problems)} schema violation(s):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"\nOK: {args.source} passes the nebulax schema.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
