#!/usr/bin/env python
"""Fit, cross-validate and save one Problem Statement 3 subsystem model.

    python scripts/ps3_train.py --task door [--seeds 0 1 2] [--ladder] [--n-jobs 4]
    python scripts/ps3_train.py --task all

Each subsystem owns its own trainer; this script only dispatches on ``--task`` and prints the
summary. **Adding a task is one ``elif`` branch** in :func:`dispatch` calling
``nebulax.ps3.<task>.train(args)`` - keep it that way so four agents can merge trivially.

Every trainer writes ``results/ps3/<task>_cv.json`` + ``.md`` (scheme, folds, seeds, git rev,
feature version, wall seconds) and ``models/ps3/<task>.pkl`` + ``.json``; with ``--ladder`` it
also writes ``results/ps3/<task>_ladder.json`` + ``.md`` and replaces the artefact only if a
ladder row beats the baseline on the frozen outer CV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/ps3_train.py` with no install
    sys.path.insert(0, str(REPO_ROOT))

from nebulax.ps3.common import TASK_NAMES  # noqa: E402


def dispatch(task: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run one subsystem's trainer. One ``elif`` per task, nothing else."""
    if task == "door":
        from nebulax.ps3.door import train as door_train

        return door_train(args)
    elif task == "acv":
        from nebulax.ps3.acv import train as acv_train

        return acv_train(args)
    elif task == "rail":
        from nebulax.ps3.rail import train as rail_train

        return rail_train(args)
    elif task == "shm":
        from nebulax.ps3.shm import train as shm_train

        return shm_train(args)
    raise SystemExit(f"unknown task {task!r}; expected one of {list(TASK_NAMES)} or 'all'")


def _brief(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop the bulky per-fold arrays so the console summary stays readable."""
    drop = {"headline", "folds", "ablation_folds", "ablations", "rows"}
    return {k: v for k, v in payload.items() if k not in drop}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=[*TASK_NAMES, "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="random seeds (default: 0)")
    parser.add_argument("--n-jobs", dest="n_jobs", type=int, default=4, help="worker cap (box rule: 4)")
    parser.add_argument("--ladder", action="store_true", help="also run the model ladder")
    parser.add_argument("--tag", default="baseline", help="suffix for the predictions CSV")
    parser.add_argument("--no-predict", dest="predict", action="store_false", help="skip the Test CSV")
    parser.add_argument("--data-root", dest="data_root", default=None, help="override NEBULAX_PS3_DATA")
    parser.add_argument("--out-dir", dest="out_dir", default=None, help="override results/ps3")
    parser.add_argument("--model-dir", dest="model_dir", default=None, help="override models/ps3")
    args = parser.parse_args(argv)

    tasks = list(TASK_NAMES) if args.task == "all" else [args.task]
    failures: list[str] = []
    for task in tasks:
        print(f"== {task} ==", flush=True)
        try:
            payload = dispatch(task, args)
        except Exception as exc:  # pragma: no cover - --task all must not die on one subsystem
            if args.task != "all":
                raise
            failures.append(f"{task}: {type(exc).__name__}: {exc}")
            print(f"   FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        print(json.dumps(_brief(payload), indent=2, default=str), flush=True)
    if failures:
        print("\n".join(["", "failed:", *failures]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
