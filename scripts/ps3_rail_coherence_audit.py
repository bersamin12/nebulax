"""Nested W7/coherence selection and rail stress partitions on Train only."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from nebulax.ps3 import rail, rail_features as rf
from nebulax.ps3.common import RAIL_LABELS, train_dir
from nebulax.ps3.scoring import class_f1_report, macro_f1
from scripts.ps3_rail_coherence_round import coherence_features, mirror_coherence, OPTS, SEEDS


def fit_predict(x, xm, y, speed, groups, tr, va):
    """Same file-level W7 pipeline as the selection round, fitted inside tr."""
    xtr = np.concatenate([x[tr], xm[tr]])
    ytr = np.concatenate([y[tr], rail._mirror_labels(y[tr])])
    str_ = np.concatenate([speed[tr], speed[tr]])
    gtr = np.concatenate([groups[tr], groups[tr]])
    models = [rail._make_estimator("lgbm", seed, n_jobs=1).fit(xtr, ytr) for seed in SEEDS]
    oofs = []
    try:
        for rep in range(3):
            oof = np.zeros((len(ytr), 3))
            splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=101 * rep)
            for itr, iva in splitter.split(xtr, ytr, gtr):
                est = rail._make_estimator("lgbm", SEEDS[0], n_jobs=1).fit(xtr[itr], ytr[itr])
                order = [list(est.classes_).index(label) for label in RAIL_LABELS]
                oof[iva] = est.predict_proba(xtr[iva])[:, order]
            oofs.append(oof)
        boosts = rail._tune_boosts(oofs, ytr, str_)
    except ValueError:
        boosts = (1.0, 1.0)
    p = np.zeros((len(va), 3))
    for model in models:
        order = [list(model.classes_).index(label) for label in RAIL_LABELS]
        p += 0.5 * (model.predict_proba(x[va])[:, order] +
                    model.predict_proba(xm[va])[:, order][:, [0, 2, 1]]) / len(models)
    return rail._apply_rules(p, boosts, speed[va])


def nested_worker(index, base, base_m, coh, coh_m, y, speed, groups, ids):
    splits = list(rail._splits_stratified(y, groups, SEEDS, n_splits=5))
    name, tr, te = splits[index]
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1000 + index)
    scores = {}
    for candidate, x, xm in (("W7", base, base_m), ("coherence", coh, coh_m)):
        fold_scores = []
        for itr, iva in inner.split(base[tr], y[tr], groups[tr]):
            pred = fit_predict(x, xm, y, speed, groups, tr[itr], tr[iva])
            fold_scores.append(float(macro_f1(y[tr[iva]], pred)))
        scores[candidate] = float(np.mean(fold_scores))
    selected = "coherence" if scores["coherence"] > scores["W7"] else "W7"
    return {"fold": name, "selected": selected, "inner_macro_f1": scores,
            "held_file_ids": [ids[int(j)] for j in te]}


def stress_worker(scheme, index, coh, coh_m, y, speed, groups, ids):
    if scheme == "contiguous":
        splits = list(rail._splits_contiguous(ids, groups, n_blocks=5))
    else:
        splits = list(rail._splits_speed_range(speed, y))
    name, tr, te = splits[index]
    pred = fit_predict(coh, coh_m, y, speed, groups, tr, te)
    report = class_f1_report(y[te], pred)
    return {"scheme": scheme, "fold": name, "macro_f1": float(report["macro_f1"]),
            "side_i_f1": float(report["per_class"]["Side I"]["f1"]),
            "held": [{"file_id": ids[int(j)], "truth": str(y[j]), "prediction": str(p)}
                     for j, p in zip(te, pred)]}


def main():
    start = time.time()
    feats, y = rail._load_training(n_jobs=8)
    paths = [Path(train_dir("rail")) / fid for fid in feats.file_ids]
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 1)) as pool:
        coherent = list(pool.map(coherence_features, paths))
    cf = pd.DataFrame(coherent)
    base_df = rf.aggregate(feats, OPTS)
    base_m_df = rf.aggregate(rf.mirror(feats), OPTS)
    base, base_m = base_df.to_numpy(float), base_m_df.to_numpy(float)
    coh = pd.concat([base_df, cf], axis=1).to_numpy(float)
    coh_m = pd.concat([base_m_df, mirror_coherence(cf)], axis=1).to_numpy(float)
    speed = feats.scalars.speed_kmh.to_numpy(float)
    groups = rail._duplicate_groups(feats)
    ids = feats.file_ids
    old = json.loads(Path("results/ps3/rail_w7_fold_proba.json").read_text())["folds"]
    candidate = json.loads(Path("results/ps3/rail_coherence_round.json").read_text())["folds"]
    nested = [None] * 15
    stress = {"contiguous": [None] * 5, "speed_range": [None] * 3}
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 1)) as pool:
        jobs = {pool.submit(nested_worker, i, base, base_m, coh, coh_m, y, speed, groups, ids): ("nested", i)
                for i in range(15)}
        for scheme, rows in stress.items():
            for i in range(len(rows)):
                jobs[pool.submit(stress_worker, scheme, i, coh, coh_m, y, speed, groups, ids)] = (scheme, i)
        for future in as_completed(jobs):
            scheme, i = jobs[future]
            if scheme == "nested":
                nested[i] = future.result()
                print("nested", i, nested[i]["selected"], flush=True)
            else:
                stress[scheme][i] = future.result()
                print(scheme, i, stress[scheme][i]["macro_f1"], flush=True)
    nested_values, nested_i = [], []
    for i, choice in enumerate(nested):
        prior = old[str(i)]
        latest = candidate[i]
        if choice["held_file_ids"] != prior["file_ids"] or choice["held_file_ids"] != [r["file_id"] for r in latest["held"]]:
            raise ValueError(f"nested fold {i} held-out alignment failed")
        truth = prior["truth"]
        pred = prior["predictions"] if choice["selected"] == "W7" else [r["prediction"] for r in latest["held"]]
        report = class_f1_report(truth, pred)
        choice["outer_macro_f1"] = float(report["macro_f1"])
        choice["outer_side_i_f1"] = float(report["per_class"]["Side I"]["f1"])
        nested_values.append(choice["outer_macro_f1"])
        nested_i.append(choice["outer_side_i_f1"])
    stress_summary = {}
    for scheme, rows in stress.items():
        values = [r["macro_f1"] for r in rows]
        stress_summary[scheme] = {"macro_f1_mean": float(np.mean(values)), "macro_f1_sd": float(np.std(values)),
                                  "side_i_f1_mean": float(np.mean([r["side_i_f1"] for r in rows])), "folds": rows}
    payload = {
        "source": "Train only; outer duplicate-grouped 5-fold x 3 seeds; inner grouped 3-fold selects W7 or coherence; no Test labels",
        "reference_historical_nested_f1": 0.7571,
        "nested_macro_f1_mean": float(np.mean(nested_values)),
        "nested_macro_f1_sd": float(np.std(nested_values)),
        "nested_side_i_f1_mean": float(np.mean(nested_i)),
        "nested_selected_coherence_folds": sum(r["selected"] == "coherence" for r in nested),
        "nested_folds": nested,
        "stress": stress_summary,
        "seconds": time.time() - start,
    }
    out = Path("results/ps3/rail_coherence_audit.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(out, payload["nested_macro_f1_mean"], stress_summary["contiguous"]["macro_f1_mean"],
          stress_summary["speed_range"]["macro_f1_mean"], flush=True)


if __name__ == "__main__":
    main()
