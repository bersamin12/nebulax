"""Train-only XGBoost check against the shipped W7 rail representation.

Uses the same duplicate-grouped 5-fold x 3-seed selection partitions.  A mirror
copy of each training recording stays in its source fold; validation predictions
are averaged with the mirror view.  This script does not overwrite submissions.
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


SEEDS = (0, 1, 2)
OPTS = {"wavelength": False, "hz": True, "v2_normalise": False, "shock": False}
NAME_TO_INT = {name: i for i, name in enumerate(RAIL_LABELS)}


def fit_xgb(x, y, seed):
    from xgboost import XGBClassifier

    yi = np.asarray([NAME_TO_INT[v] for v in y], dtype=int)
    counts = np.bincount(yi, minlength=3)
    weights = len(yi) / (3.0 * np.maximum(counts, 1))
    model = XGBClassifier(
        objective="multi:softprob", num_class=3, tree_method="hist",
        n_estimators=300, max_depth=3, learning_rate=0.05,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.4,
        reg_lambda=3.0, reg_alpha=0.1, random_state=seed, n_jobs=1,
        eval_metric="mlogloss",
    )
    model.fit(x, yi, sample_weight=weights[yi], verbose=False)
    return model


def worker(i):
    feats, y = rail._load_training(n_jobs=1)
    splits = list(rail._splits_stratified(y, rail._duplicate_groups(feats), SEEDS, n_splits=5))
    name, tr, te = splits[i]
    X = rf.aggregate(feats, OPTS).to_numpy(float)
    Xm = rf.aggregate(rf.mirror(feats), OPTS).to_numpy(float)
    xtr = np.concatenate([X[tr], Xm[tr]])
    ytr = np.concatenate([y[tr], rail._mirror_labels(y[tr])])
    proba = np.zeros((len(te), 3))
    for seed in SEEDS:
        model = fit_xgb(xtr, ytr, seed)
        proba += 0.5 * (model.predict_proba(X[te]) + model.predict_proba(Xm[te])[:, [0, 2, 1]]) / len(SEEDS)
    speed = feats.scalars.speed_kmh.to_numpy(float)
    preds = rail._apply_rules(proba, (1.0, 1.0), speed[te])
    rep = class_f1_report(y[te], preds)
    return {
        "fold": name,
        "macro_f1": float(rep["macro_f1"]),
        "side_i_f1": float(rep["per_class"]["Side I"]["f1"]),
        "speed_matched_f1": float(macro_f1(y[te][speed[te] >= 35], preds[speed[te] >= 35])),
        "held": [{"file_id": feats.file_ids[int(j)], "truth": str(y[j]), "prediction": str(p),
                  "probability": list(map(float, z))} for j, p, z in zip(te, preds, proba)],
    }


def main():
    t0 = time.time()
    workers = min(12, os.cpu_count() or 1)
    reports = [None] * 15
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, i): i for i in range(15)}
        for future in as_completed(futures):
            i = futures[future]
            reports[i] = future.result()
            print(i, reports[i]["macro_f1"], flush=True)
    vals = np.asarray([v["macro_f1"] for v in reports])
    i_vals = np.asarray([v["side_i_f1"] for v in reports])
    payload = {
        "source": "Train only, duplicate-grouped frozen 5-fold x 3 seeds; no Test labels",
        "reference_selection_f1": 0.8365,
        "macro_f1_mean": float(vals.mean()),
        "macro_f1_sd": float(vals.std()),
        "side_i_f1_mean": float(i_vals.mean()),
        "speed_matched_f1_mean": float(np.mean([v["speed_matched_f1"] for v in reports])),
        "seconds": time.time() - t0,
        "folds": reports,
    }
    out = Path("results/ps3/rail_xgb_round.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(out, payload["macro_f1_mean"], payload["macro_f1_sd"], payload["side_i_f1_mean"],
          payload["seconds"])


if __name__ == "__main__":
    main()
