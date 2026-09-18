"""Check whether a larger W7 seed ensemble stabilises rare rail labels.

The 15 frozen grouped validation splits remain the original three split seeds.
Only the number of fitted LightGBM seeds changes from three to nine.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import numpy as np

from nebulax.ps3 import rail, rail_features as rf
from nebulax.ps3.common import RAIL_LABELS
from nebulax.ps3.scoring import class_f1_report, macro_f1


SPLIT_SEEDS = (0, 1, 2)
MODEL_SEEDS = tuple(range(9))
OPTS = {"wavelength": False, "hz": True, "v2_normalise": False, "shock": False}


def worker(i):
    feats, y = rail._load_training(n_jobs=1)
    splits = list(rail._splits_stratified(y, rail._duplicate_groups(feats), SPLIT_SEEDS, n_splits=5))
    name, tr, te = splits[i]
    mirrored = rf.mirror(feats)
    model = rail.fit_rail(feats.subset(tr), y[tr], kind="lgbm", opts=OPTS, seeds=MODEL_SEEDS,
                          augment=True, tta=True, n_jobs=1, mirrored=mirrored.subset(tr))
    X = rf.aggregate(feats.subset(te), OPTS)
    Xm = rf.aggregate(mirrored.subset(te), OPTS)
    speed = feats.scalars.speed_kmh.to_numpy(float)[te]
    pred = model.predict_labels(X, speed, Xm)
    rep = class_f1_report(y[te], pred)
    matched = speed >= 35
    return {
        "fold": name,
        "macro_f1": float(rep["macro_f1"]),
        "side_i_f1": float(rep["per_class"]["Side I"]["f1"]),
        "speed_matched_f1": float(macro_f1(y[te][matched], pred[matched])),
        "boosts": list(map(float, model.boosts)),
        "held": [{"file_id": feats.file_ids[int(j)], "truth": str(y[j]), "prediction": str(p)}
                 for j, p in zip(te, pred)],
    }


def main():
    t0 = time.time()
    reports = [None] * 15
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(worker, i): i for i in range(15)}
        for future in as_completed(futures):
            i = futures[future]
            reports[i] = future.result()
            print(i, reports[i]["macro_f1"], flush=True)
    vals = np.asarray([v["macro_f1"] for v in reports])
    side_i = np.asarray([v["side_i_f1"] for v in reports])
    payload = {
        "source": "Train only; frozen duplicate-grouped 5-fold x 3 split seeds; no Test labels",
        "model_seeds": list(MODEL_SEEDS),
        "reference_selection_f1": 0.8365,
        "macro_f1_mean": float(vals.mean()),
        "macro_f1_sd": float(vals.std()),
        "side_i_f1_mean": float(side_i.mean()),
        "speed_matched_f1_mean": float(np.mean([v["speed_matched_f1"] for v in reports])),
        "seconds": time.time() - t0,
        "folds": reports,
    }
    out = Path("results/ps3/rail_multiseed_round.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(out, payload["macro_f1_mean"], payload["macro_f1_sd"], payload["side_i_f1_mean"])


if __name__ == "__main__":
    main()
