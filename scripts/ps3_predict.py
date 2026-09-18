#!/usr/bin/env python
"""Run one PS3 subsystem over an input file or folder and write its `*_predictions.csv`.

    python scripts/ps3_predict.py --task door --input <PS3>/02_Datasets/Door/Test.csv \
                                  --output door_predictions.csv
    python scripts/ps3_predict.py --task rail --input <PS3>/02_Datasets/Rail_Corrugation/Test \
                                  --output rail_predictions.csv
    python scripts/ps3_predict.py --task shm  --input <PS3>/02_Datasets/SHM/Test --output shm.csv
    python scripts/ps3_predict.py --task acv  --input <PS3>/02_Datasets/ACV/Test  --output acv.csv

Same code path as the app: :mod:`nebulax.api.ps3` resolves the registered ``Task``, loads its
committed artefact once, runs the inputs **sequentially** (one file in memory at a time - the
rail folder is 1.1 GB) and renders the CSV with the one shared writer, so the bytes here and the
bytes the download button serves are identical.

The CSV is validated with :func:`nebulax.ps3.submission.validate_csv` before the script returns;
``--expect-ids`` additionally demands set equality with the file ids in the input folder.
Exit status: 0 valid, 1 invalid or every input failed, 2 the task cannot run yet (no module, no
fitted artefact) - the same distinction the API draws between 4xx and 503.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebulax.api.ps3 import (  # noqa: E402
    TaskUnavailable,
    collect_inputs,
    infer_task,
    predict_paths,
    write_csv,
)
from nebulax.ps3.common import TASK_NAMES, acv_car_ids, natural_key  # noqa: E402
from nebulax.ps3.submission import validate_csv  # noqa: E402

__all__ = ["main", "run"]


def _parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog or "ps3_predict.py",
        description="Run a Problem Statement 3 subsystem over an input file or folder.",
    )
    p.add_argument("--task", choices=list(TASK_NAMES), default=None, help="subsystem (default: inferred from --input)")
    p.add_argument("--input", "-i", required=True, help="input file, or a folder of input files")
    p.add_argument("--output", "-o", required=True, help="where to write the predictions CSV")
    p.add_argument("--expect-ids", action="store_true", help="require one row per input file (set equality)")
    p.add_argument("--quiet", "-q", action="store_true", help="only print errors")
    return p


def run(
    task: str | None,
    input_path: str | Path,
    output_path: str | Path,
    *,
    expect_ids: bool = False,
    quiet: bool = False,
) -> int:
    """The body of :func:`main`; returns the process exit status."""
    target = Path(input_path)
    if not target.exists():
        print(f"error: no such input: {target}", file=sys.stderr)
        return 1
    try:
        name = task or infer_task(target)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        paths = collect_inputs(name, target)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not quiet:
        print(f"[{name}] {len(paths)} input file(s) from {target}")
    t0 = time.perf_counter()
    try:
        rows, explanations, errors = predict_paths(
            name,
            paths,
            on_file=None if quiet else (lambda p, _r: print(f"  {p.name}", flush=True)),
        )
    except TaskUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    took = time.perf_counter() - t0

    for err in errors:
        print(f"error: {err['file']}: {err['message']}", file=sys.stderr)
    if not rows:
        print(f"error: [{name}] no predictions were produced", file=sys.stderr)
        return 1

    out = write_csv(name, rows, output_path)
    expected = None
    expected_cars = None
    if expect_ids and name != "door":
        expected = sorted((p.name for p in paths), key=natural_key)
    if name == "acv":
        expected_cars = {}
        for p in paths:
            try:
                expected_cars[p.name] = acv_car_ids(p)
            except Exception:  # a car-id sniff must never fail the run
                pass
    report = validate_csv(name, out, expected, expected_cars=expected_cars or None)
    for warning in report.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if not quiet:
        print(
            f"[{name}] {len(rows)} row(s), {len(explanations)} file(s) predicted, "
            f"{len(errors)} failed, {took:.1f}s -> {out}"
        )
    if not report.ok:
        print(f"error: {out} is not a valid {name} submission:", file=sys.stderr)
        for e in report.errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    if errors:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run(
        args.task,
        args.input,
        args.output,
        expect_ids=args.expect_ids,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    raise SystemExit(main())
