"""Frozen grouped CV for low-quantile and regularized rail candidates.

This research script writes only a report.  It does not replace the W7 model.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from nebulax.ps3 import rail
from nebulax.ps3.common import RAIL_LABELS


BASE_OPTS = {"wavelength": False, "hz": True, "v2_normalise": False, "shock": False}
SEEDS = (0, 1, 2)


def worker(i, variant):
    feats, y = rail._load_training(n_jobs=1)
    opts = {**BASE_OPTS, "hz_p10": variant == "hz_p10"}
    kind = "lgbm_regularized" if variant == "regularized" else "lgbm"
    result = rail.cross_validate(
        feats, y, scheme="stratified", kind=kind, opts=opts, seeds=SEEDS,
        augment=True, tta=True, n_jobs=1, fold_indices=(i,), include_predictions=True,
    )
    return result["folds"][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("hz_p10", "regularized"), default="hz_p10")
    args = parser.parse_args()
    t0 = time.time()
    reports = [None] * 15
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(worker, i, args.variant): i for i in range(15)}
        for future in as_completed(futures):
            i = futures[future]
            reports[i] = future.result()
            print(i, reports[i]["macro_f1"], flush=True)
    vals = np.asarray([v["macro_f1"] for v in reports])
    side_i = np.asarray([v["class_f1"]["Side I"] for v in reports])
    payload = {
        "source": "Train only, duplicate-grouped 5-fold x 3 seeds; no Test labels",
        "reference_selection_f1": 0.8365,
        "variant": args.variant,
        "macro_f1_mean": float(vals.mean()),
        "macro_f1_sd": float(vals.std()),
        "side_i_f1_mean": float(side_i.mean()),
        "speed_matched_f1_mean": float(np.mean([v["macro_f1_speed_matched"] for v in reports])),
        "seconds": time.time() - t0,
        "folds": reports,
    }
    out = Path(f"results/ps3/rail_{args.variant}_round.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(out, payload["macro_f1_mean"], payload["macro_f1_sd"], payload["side_i_f1_mean"])


if __name__ == "__main__":
    main()
