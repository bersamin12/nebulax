#!/usr/bin/env python
"""Build the Problem Statement 3 deliverable: every distributed Test input -> `predictions.zip`.

    python scripts/ps3_submission.py                        # all four, submission/<team>/
    python scripts/ps3_submission.py --tag baseline         # -> predictions_baseline.zip
    python scripts/ps3_submission.py --tasks rail,shm       # only what is ready
    python scripts/ps3_submission.py --team "NEBULA X"      # the folder the judges look for

Runs the organisers' held-out inputs (``02_Datasets/<sub>/Test``, door: ``Door/Test.csv``)
through **the same** ``Task`` code the app and ``scripts/ps3_predict.py`` use - one file at a
time - writes each ``<task>_predictions.csv`` next to the zip, validates every CSV against
``04_Example_Submission`` (header order, case-sensitive basenames with extension, set equality
with the distributed file ids, no duplicates, no NaN/Inf, label vocabularies, ``ranked_cars`` =
every car in that file exactly once), packs the valid ones at the archive root with
:func:`nebulax.ps3.submission.pack` (byte-deterministic, so a re-run is diffable) and re-validates
the archive.

A subsystem whose task module or artefact is not ready yet is **skipped with a reason**, not a
crash: the specification's Overall/Average split is exactly what a partial submission is for.
Exit status: 0 every attempted subsystem is valid, 1 something is invalid or failed,
2 nothing could be attempted.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebulax.api.ps3 import TaskUnavailable, collect_inputs, predict_paths, write_csv  # noqa: E402
from nebulax.ps3.common import OUTPUT_FILENAMES, TASK_NAMES, acv_car_ids, test_dir  # noqa: E402
from nebulax.ps3.submission import expected_ids_for, pack, validate_csv, validate_zip  # noqa: E402

__all__ = ["main", "build", "TaskOutcome"]

#: The door "Test set" is one continuous stream, not a folder.
_DOOR_TEST_FILE = "Test.csv"


@dataclass
class TaskOutcome:
    """What happened to one subsystem: enough to print the table and decide the exit status."""

    task: str
    status: str = "skipped"  # ok | invalid | failed | skipped
    reason: str = ""
    csv: Path | None = None
    n_files: int = 0
    n_rows: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def attempted(self) -> bool:
        return self.status != "skipped"


def _test_input(task: str) -> Path:
    d = test_dir(task)
    return d / _DOOR_TEST_FILE if task == "door" else d


def build(
    tasks: list[str],
    out_dir: Path,
    *,
    tag: str | None = None,
    quiet: bool = False,
) -> tuple[list[TaskOutcome], Path | None]:
    """Predict, validate and pack. Returns the per-task outcomes and the zip (``None`` if empty)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[TaskOutcome] = []
    for task in tasks:
        out = TaskOutcome(task=task)
        outcomes.append(out)
        target = _test_input(task)
        if not target.exists():
            out.reason = f"no distributed test input at {target} (set NEBULAX_PS3_DATA)"
            continue
        try:
            paths = collect_inputs(task, target)
        except FileNotFoundError as exc:
            out.reason = str(exc)
            continue
        if not quiet:
            print(f"[{task}] {len(paths)} input file(s) from {target}", flush=True)
        t0 = time.perf_counter()
        try:
            rows, _explanations, errors = predict_paths(task, paths)
        except TaskUnavailable as exc:
            out.reason = str(exc)
            continue
        except Exception as exc:  # a broken task module must not lose the other three
            out.status, out.reason = "failed", f"{type(exc).__name__}: {exc}"
            out.seconds = time.perf_counter() - t0
            continue
        out.seconds = time.perf_counter() - t0
        out.n_files = len(paths)
        out.n_rows = len(rows)
        out.errors = [f"{e['file']}: {e['message']}" for e in errors]
        if not rows:
            out.status, out.reason = "failed", "no predictions were produced"
            continue

        csv_path = write_csv(task, rows, out_dir / OUTPUT_FILENAMES[task])
        out.csv = csv_path
        expected_cars = None
        if task == "acv":
            expected_cars = {}
            for p in paths:
                try:
                    expected_cars[p.name] = acv_car_ids(p)
                except Exception:  # a car-id sniff must never fail the run
                    pass
        try:
            expected = expected_ids_for(task, None if task == "door" else target)
        except (FileNotFoundError, KeyError) as exc:  # pragma: no cover - target exists above
            expected, out.warnings = None, out.warnings + [f"expected ids unavailable: {exc}"]
        report = validate_csv(task, csv_path, expected, expected_cars=expected_cars or None)
        out.warnings += list(report.warnings)
        out.errors += list(report.errors)
        out.status = "ok" if (report.ok and not errors) else "invalid"

    valid = [o.csv for o in outcomes if o.status == "ok" and o.csv is not None]
    if not valid:
        return outcomes, None
    zip_path = out_dir / (f"predictions_{tag}.zip" if tag else "predictions.zip")
    pack(valid, zip_path)
    zip_report = validate_zip(
        zip_path,
        {o.task: (expected_ids_for(o.task, None if o.task == "door" else _test_input(o.task))) for o in outcomes if o.status == "ok"},
    )
    if not zip_report.ok:
        for o in outcomes:
            if o.status == "ok":
                o.status = "invalid"
                o.errors += [f"{zip_path.name}: {e}" for e in zip_report.errors]
    return outcomes, zip_path


def _table(outcomes: list[TaskOutcome]) -> str:
    head = f"{'subsystem':<10} {'status':<8} {'files':>6} {'rows':>7} {'seconds':>8}  {'output / reason'}"
    lines = [head, "-" * max(len(head), 78)]
    for o in outcomes:
        what = o.csv.name if o.csv is not None else (o.reason or "")
        lines.append(f"{o.task:<10} {o.status:<8} {o.n_files:>6} {o.n_rows:>7} {o.seconds:>8.1f}  {what}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ps3_submission.py",
        description="Run every distributed PS3 Test input and pack submission/<team>/predictions.zip.",
    )
    p.add_argument("--team", default=os.environ.get("NEBULAX_TEAM", "nebulax"), help="team folder name")
    p.add_argument("--tag", default=None, help="suffix for the zip, e.g. baseline -> predictions_baseline.zip")
    p.add_argument("--tasks", default="all", help="comma-separated subsystems (default: all)")
    p.add_argument("--out-dir", default=None, help="override submission/<team>")
    p.add_argument("--quiet", "-q", action="store_true", help="print the table only")
    args = p.parse_args(argv)

    if args.tasks.strip().lower() in ("", "all"):
        tasks = list(TASK_NAMES)
    else:
        tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
        unknown = [t for t in tasks if t not in TASK_NAMES]
        if unknown:
            print(f"error: unknown subsystem(s) {unknown}; expected {list(TASK_NAMES)}", file=sys.stderr)
            return 1

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "submission" / args.team
    outcomes, zip_path = build(tasks, out_dir, tag=args.tag, quiet=args.quiet)

    print()
    print(_table(outcomes))
    for o in outcomes:
        for w in o.warnings:
            print(f"warning: [{o.task}] {w}", file=sys.stderr)
        for e in o.errors:
            print(f"error: [{o.task}] {e}", file=sys.stderr)
        if o.status == "skipped" and o.reason:
            print(f"note: [{o.task}] skipped - {o.reason}", file=sys.stderr)
    if zip_path is not None:
        n_ok = sum(1 for o in outcomes if o.status == "ok")
        print(f"\npacked {n_ok} subsystem(s) -> {zip_path}")

    attempted = [o for o in outcomes if o.attempted]
    if not attempted:
        print("error: no subsystem could be attempted", file=sys.stderr)
        return 2
    return 0 if all(o.status == "ok" for o in attempted) else 1


if __name__ == "__main__":
    raise SystemExit(main())
