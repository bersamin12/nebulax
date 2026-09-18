"""Fold-local feature Mixup on Rail faults, compared with the promoted coherence row."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

from nebulax.ps3 import rail


OUT = Path("results/ps3/rail_mixup_round.json")
ALPHAS = (0.4, 2.0)


def worker(alpha: float):
    feats, y = rail._load_training(n_jobs=1)
    report = rail.cross_validate(
        feats, y, scheme="stratified", seeds=(0, 1, 2), n_jobs=1,
        include_predictions=True, feature_mixup_alpha=alpha,
        **rail.COHERENCE_ROWS[0][2],
    )
    return alpha, report


def main():
    start = time.time()
    reference = json.loads(Path("results/ps3/rail_coherence_production_selection.json").read_text())
    with ProcessPoolExecutor(max_workers=len(ALPHAS)) as pool:
        jobs = {pool.submit(worker, alpha): alpha for alpha in ALPHAS}
        rows = []
        for future in as_completed(jobs):
            alpha, report = future.result()
            rows.append({"alpha": alpha, "report": report})
            print(alpha, report["macro_f1_mean"], report["class_f1_mean"], flush=True)
    rows.sort(key=lambda row: row["alpha"])
    payload = {
        "source": "Train only; grouped 5-fold x 3; feature Mixup generated in each fitting partition",
        "reference_macro_f1_mean": reference["macro_f1_mean"],
        "reference_macro_f1_sd": reference["macro_f1_sd"],
        "reference_class_f1_mean": reference["class_f1_mean"],
        "method": "one within-class, speed-near convex feature interpolation per fault row; "
                  "original plus mirrored training rows; inner calibration regenerates Mixup "
                  "from its own training split and scores only original/mirrored validation rows",
        "rows": rows,
        "wall_seconds": time.time() - start,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
