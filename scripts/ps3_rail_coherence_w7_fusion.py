"""Train-only probability blends of the two strongest deployable Rail rows."""

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
CACHE = Path("results/ps3/rail_coherence_fold_proba.json")
OUT = Path("results/ps3/rail_coherence_w7_fusion.json")


def worker(index: int):
    feats, y = rail._load_training(n_jobs=1)
    splits = list(rail._splits_stratified(y, rail._duplicate_groups(feats), SEEDS, n_splits=5))
    name, tr, te = splits[index]
    opts = rail.COHERENCE_ROWS[0][2]
    mirrored = rf.mirror(feats)
    model = rail.fit_rail(feats.subset(tr), y[tr], seeds=SEEDS, n_jobs=1,
                          mirrored=mirrored.subset(tr), **opts)
    X = rf.aggregate(feats.subset(te), model.opts)
    Xm = rf.aggregate(mirrored.subset(te), model.opts)
    p = model.proba_tta(X, Xm)
    speed = feats.scalars.speed_kmh.to_numpy(dtype=float)[te]
    pred = rail._apply_rules(p, model.boosts, speed)
    boosted = p.copy()
    boosted[:, 1] *= model.boosts[0]
    boosted[:, 2] *= model.boosts[1]
    boosted /= boosted.sum(axis=1, keepdims=True)
    return index, {
        "fold": name,
        "file_ids": [feats.file_ids[int(j)] for j in te],
        "truth": list(map(str, y[te])),
        "predictions": list(map(str, pred)),
        "boosts": list(map(float, model.boosts)),
        "speed": list(map(float, speed)),
        "boosted_proba": boosted.tolist(),
    }


def main():
    start = time.time()
    saved = json.loads(CACHE.read_text()) if CACHE.exists() else {
        "model": "coherence LightGBM", "options": rail.COHERENCE_ROWS[0][2],
        "seeds": list(SEEDS), "folds": {},
    }
    if saved["options"] != rail.COHERENCE_ROWS[0][2]:
        raise ValueError("coherence probability cache uses another recipe")
    pending = [i for i in range(15) if str(i) not in saved["folds"]]
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 1)) as pool:
        jobs = {pool.submit(worker, i): i for i in pending}
        for future in as_completed(jobs):
            i, row = future.result()
            saved["folds"][str(i)] = row
            CACHE.write_text(json.dumps(saved, indent=2) + "\n")
            print(f"coherence proba {len(saved['folds'])}/15: {row['fold']}", flush=True)
    reference = json.loads(Path("results/ps3/rail_coherence_production_selection.json").read_text())
    w7 = json.loads(Path("results/ps3/rail_w7_fold_proba.json").read_text())["folds"]
    for i in range(15):
        a = saved["folds"][str(i)]
        b = w7[str(i)]
        c = reference["folds"][i]["held_predictions"]
        if a["file_ids"] != b["file_ids"] or a["truth"] != b["truth"]:
            raise ValueError(f"component held-out files differ in fold {i}")
        if list(zip(a["file_ids"], a["predictions"])) != [
            (row["file_id"], row["prediction"]) for row in c
        ]:
            raise ValueError(f"coherence probabilities do not reproduce production CV fold {i}")
    rows = []
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        scores = []
        class_scores = {c: [] for c in RAIL_LABELS}
        changes_vs_coherence = 0
        for i in range(15):
            a = saved["folds"][str(i)]
            b = w7[str(i)]
            p = weight * np.asarray(a["boosted_proba"]) + (1.0 - weight) * np.asarray(b["boosted_proba"])
            pred = np.asarray(RAIL_LABELS, dtype=object)[p.argmax(axis=1)]
            pred = np.where(np.asarray(a["speed"]) < rail.LOW_SPEED_KMH, RAIL_LABELS[0], pred)
            report = class_f1_report(a["truth"], pred)
            scores.append(float(report["macro_f1"]))
            for c in RAIL_LABELS:
                class_scores[c].append(float(report["per_class"][c]["f1"]))
            changes_vs_coherence += int(np.sum(pred != np.asarray(a["predictions"], dtype=object)))
        rows.append({
            "coherence_weight": weight,
            "macro_f1_mean": float(np.mean(scores)),
            "macro_f1_sd": float(np.std(scores)),
            "class_f1_mean": {c: float(np.mean(v)) for c, v in class_scores.items()},
            "changed_decisions_vs_coherence": changes_vs_coherence,
        })
    payload = {
        "source": "Train only; matched duplicate-grouped 5-fold x 3 held-out probabilities; no Test labels",
        "method": "blend each fold's independently fitted, mirror-averaged, class-boosted probabilities; "
                  "apply the shared low-speed rule after blending",
        "weights_examined": [r["coherence_weight"] for r in rows],
        "reference_selection_macro_f1_mean": reference["macro_f1_mean"],
        "n_held_decisions": sum(len(saved["folds"][str(i)]["truth"]) for i in range(15)),
        "rows": rows,
        "wall_seconds": time.time() - start,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUT, rows, flush=True)


if __name__ == "__main__":
    main()
