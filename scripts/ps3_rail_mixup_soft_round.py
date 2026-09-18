"""Full grouped Rail CV for cross-class, weighted-target feature Mixup."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import numpy as np

from nebulax.ps3 import rail
from nebulax.ps3.common import RAIL_LABELS
from nebulax.ps3.scoring import class_f1_report


OUT = Path("results/ps3/rail_mixup_soft_round.json")
CHECKPOINT = Path("results/ps3/rail_mixup_soft_checkpoints")
ALPHA = 0.4


def worker(index: int):
    feats, y = rail._load_training(n_jobs=1)
    report = rail.cross_validate(
        feats, y, scheme="stratified", seeds=(0, 1, 2), fold_indices=(index,),
        n_jobs=1, include_predictions=True,
        feature_mixup_alpha=ALPHA, feature_mixup_soft=True,
        **rail.COHERENCE_ROWS[0][2],
    )
    return index, report["folds"][0]


def main():
    start = time.time()
    reference = json.loads(Path("results/ps3/rail_coherence_production_selection.json").read_text())
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    folds: list[dict | None] = [None] * 15
    for i in range(15):
        p = CHECKPOINT / f"fold{i:02d}.json"
        if p.exists():
            folds[i] = json.loads(p.read_text())
    pending = [i for i, fold in enumerate(folds) if fold is None]
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 1)) as pool:
        jobs = {pool.submit(worker, i): i for i in pending}
        for future in as_completed(jobs):
            i, fold = future.result()
            folds[i] = fold
            (CHECKPOINT / f"fold{i:02d}.json").write_text(json.dumps(fold, indent=2) + "\n")
            print(i, fold["macro_f1"], f"{sum(x is not None for x in folds)}/15", flush=True)
    assert all(fold is not None for fold in folds)
    done = [fold for fold in folds if fold is not None]
    held = [row for fold in done for row in fold["held_predictions"]]
    pooled = class_f1_report([row["truth"] for row in held], [row["prediction"] for row in held])
    report = {
        "source": "Train only; grouped 5-fold x 3; Mixup only within each fitting partition",
        "method": "one fault/Normal speed-near feature Mixup pair per fault training row; "
                  "two weighted copies approximate a soft label; inner calibration mixes "
                  "only its training split and evaluates unmixed validation rows",
        "alpha": ALPHA,
        "reference_macro_f1_mean": reference["macro_f1_mean"],
        "reference_macro_f1_sd": reference["macro_f1_sd"],
        "n_folds": len(done),
        "macro_f1_mean": float(np.mean([f["macro_f1"] for f in done])),
        "macro_f1_sd": float(np.std([f["macro_f1"] for f in done])),
        "class_f1_mean": {c: float(np.mean([f["class_f1"][c] for f in done])) for c in RAIL_LABELS},
        "pooled_macro_f1": float(pooled["macro_f1"]),
        "confusion": rail._confusion(np.asarray([r["truth"] for r in held], dtype=object),
                                     np.asarray([r["prediction"] for r in held], dtype=object)),
        "folds": done,
        "wall_seconds": time.time() - start,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(OUT, report["macro_f1_mean"], report["class_f1_mean"], flush=True)


if __name__ == "__main__":
    main()
