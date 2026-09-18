"""Train-only diagnostic: drop easy low-speed Normal files inside each Rail fold.

The shipped coherence row and its 15 grouped held-out folds are the reference.
No Test labels or Test predictions are used in this experiment.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

from nebulax.ps3 import rail


OUT = Path("results/ps3/rail_low_speed_undersample_round.json")
FRACTIONS = (0.25, 0.0)


def worker(fraction: float):
    feats, y = rail._load_training(n_jobs=1)
    kwargs = rail.COHERENCE_ROWS[0][2]
    report = rail.cross_validate(
        feats, y, scheme="stratified", seeds=(0, 1, 2), n_jobs=1,
        include_predictions=True, low_speed_normal_keep=fraction, **kwargs,
    )
    return fraction, report


def main():
    start = time.time()
    reference = json.loads(Path("results/ps3/rail_coherence_production_selection.json").read_text())
    with ProcessPoolExecutor(max_workers=len(FRACTIONS)) as pool:
        jobs = {pool.submit(worker, f): f for f in FRACTIONS}
        rows = []
        for future in as_completed(jobs):
            fraction, report = future.result()
            rows.append({"low_speed_normal_keep": fraction, "report": report})
            print(fraction, report["macro_f1_mean"], report["class_f1_mean"], flush=True)
    rows.sort(key=lambda x: -x["low_speed_normal_keep"])
    payload = {
        "source": "Train only; duplicate-grouped 5-fold x 3; undersampling inside each training fold",
        "reference_macro_f1_mean": reference["macro_f1_mean"],
        "reference_side_i_f1_mean": reference["class_f1_mean"]["Side I"],
        "rows": rows,
        "wall_seconds": time.time() - start,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
