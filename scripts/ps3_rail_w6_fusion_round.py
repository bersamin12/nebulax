"""Check whether the prior shock-inclusive rail view complements shipped W7.

The 25% W6 blend is the declared candidate.  Other weights are shown only as
diagnostics.  All component probabilities come from held-out grouped Train folds.
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
from nebulax.ps3.scoring import class_f1_report


SEEDS = (0, 1, 2)
OPTS = {"wavelength": False, "hz": True, "v2_normalise": False}


def worker(i):
    feats, y = rail._load_training(n_jobs=1)
    splits = list(rail._splits_stratified(y, rail._duplicate_groups(feats), SEEDS, n_splits=5))
    name, tr, te = splits[i]
    mirrored = rf.mirror(feats)
    model = rail.fit_rail(feats.subset(tr), y[tr], kind="lgbm", opts=OPTS, seeds=SEEDS,
                          augment=True, tta=True, n_jobs=1, mirrored=mirrored.subset(tr))
    X = rf.aggregate(feats.subset(te), OPTS)
    Xm = rf.aggregate(mirrored.subset(te), OPTS)
    p = model.proba_tta(X, Xm)
    p[:, 1] *= model.boosts[0]
    p[:, 2] *= model.boosts[1]
    p /= p.sum(axis=1, keepdims=True)
    speed = feats.scalars.speed_kmh.to_numpy(float)[te]
    return {
        "fold": name,
        "file_ids": [feats.file_ids[int(j)] for j in te],
        "truth": list(map(str, y[te])),
        "boosts": list(map(float, model.boosts)),
        "speed": list(map(float, speed)),
        "boosted_proba": p.tolist(),
    }


def main():
    t0 = time.time()
    old = json.loads(Path("results/ps3/rail_w7_fold_proba.json").read_text())["folds"]
    reports = [None] * 15
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(worker, i): i for i in range(15)}
        for future in as_completed(futures):
            i = futures[future]
            reports[i] = future.result()
            print(i, flush=True)
    rows = []
    for weight in (0, 0.25, 0.5, 0.75, 1):
        scores, side_i = [], []
        for i, newer in enumerate(reports):
            prior = old[str(i)]
            if newer["file_ids"] != prior["file_ids"] or newer["truth"] != prior["truth"]:
                raise ValueError(f"W6 and W7 held-out rows differ in fold {i}")
            p = (1 - weight) * np.asarray(prior["boosted_proba"]) + weight * np.asarray(newer["boosted_proba"])
            pred = np.asarray(RAIL_LABELS, dtype=object)[p.argmax(axis=1)]
            pred = np.where(np.asarray(newer["speed"]) < 20, RAIL_LABELS[0], pred)
            report = class_f1_report(newer["truth"], pred)
            scores.append(float(report["macro_f1"]))
            side_i.append(float(report["per_class"]["Side I"]["f1"]))
        rows.append({"w6_weight": weight, "macro_f1_mean": float(np.mean(scores)),
                     "macro_f1_sd": float(np.std(scores)), "side_i_f1_mean": float(np.mean(side_i))})
    payload = {
        "source": "Train only, duplicate-grouped 5-fold x 3 seeds; no Test labels",
        "declared_candidate_w6_weight": 0.25,
        "reference_selection_f1": 0.8365,
        "rows": rows,
        "folds": reports,
        "seconds": time.time() - t0,
    }
    out = Path("results/ps3/rail_w6_fusion_round.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(out, rows)


if __name__ == "__main__":
    main()
