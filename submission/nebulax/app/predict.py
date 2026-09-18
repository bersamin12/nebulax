#!/usr/bin/env python
"""`python predict.py --input <file|folder> --output <predictions.csv>` - the info kits' interface.

The four Problem Statement 3 Info Kits all describe the same two-flag command, with no ``--task``:
give it an input and a destination and it works out which subsystem the input belongs to.

    python predict.py --input Door/Test.csv             --output door_predictions.csv
    python predict.py --input ACV/Test/acv_test_case.xlsx --output acv_predictions.csv
    python predict.py --input Rail_Corrugation/Test     --output rail_predictions.csv
    python predict.py --input SHM/Test                  --output shm_predictions.csv

How the subsystem is inferred (``nebulax.api.ps3.infer_task``; ``--task`` always wins):

* ``.xlsx`` / ``.xls``, or a folder of them          -> **acv**
* a csv whose header starts ``Datetime,Motor current`` -> **door**
* a wide csv (the released rail files are 129 columns) -> **rail**
* a headerless single-column numeric csv              -> **shm**

Everything else - loading, prediction, the CSV bytes and the submission validation - is
``scripts/ps3_predict.py``, which is the same code the app's upload page runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ps3_predict import run  # noqa: E402
from nebulax.ps3.common import TASK_NAMES  # noqa: E402

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="predict.py",
        description="Predict one Problem Statement 3 input (subsystem inferred from the input).",
        epilog="Inference: .xlsx -> acv, the door header -> door, a wide csv -> rail, "
        "a single-column csv -> shm. Pass --task to override.",
    )
    p.add_argument("--input", "-i", required=True, help="input file, or a folder of input files")
    p.add_argument("--output", "-o", required=True, help="where to write the predictions CSV")
    p.add_argument("--task", choices=list(TASK_NAMES), default=None, help="skip inference")
    p.add_argument("--quiet", "-q", action="store_true", help="only print errors")
    args = p.parse_args(argv)
    return run(args.task, args.input, args.output, quiet=args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
