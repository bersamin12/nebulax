"""Train-only fusion of coherence with frozen Mantis-short and MOMENT-long probes."""

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
from ps3_rail_transfer_compare import CACHES, _load, _probe_proba, _side_design


SEEDS = (0, 1, 2)
OUT = Path("results/ps3/rail_coherence_teacher_fusion.json")
CACHE = Path("results/ps3/rail_coherence_teacher_fold_proba.json")
ENCODERS = ("Mantis short", "MOMENT long")


def worker(index: int):
    feats, y = rail._load_training(n_jobs=1)
    splits = list(rail._splits_stratified(y, rail._duplicate_groups(feats), SEEDS, n_splits=5))
    name, tr, te = splits[index]
    speed = feats.scalars.speed_kmh.to_numpy(dtype=float)[te]
    probs = {}
    for encoder in ENCODERS:
        embedded = _load(CACHES[encoder], feats.file_ids)
        X, Xm = _side_design(embedded)
        probs[encoder] = _probe_proba(X[tr], Xm[tr], y[tr], X[te], Xm[te]).tolist()
    return index, {
        "fold": name,
        "file_ids": [feats.file_ids[int(j)] for j in te],
        "truth": list(map(str, y[te])),
        "speed": list(map(float, speed)),
        "proba": probs,
    }


def main():
    start = time.time()
    saved = json.loads(CACHE.read_text()) if CACHE.exists() else {
        "encoders": list(ENCODERS), "seeds": list(SEEDS), "folds": {},
    }
    if saved["encoders"] != list(ENCODERS):
        raise ValueError("teacher probe cache uses another encoder recipe")
    pending = [i for i in range(15) if str(i) not in saved["folds"]]
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 1)) as pool:
        jobs = {pool.submit(worker, i): i for i in pending}
        for future in as_completed(jobs):
            i, row = future.result()
            saved["folds"][str(i)] = row
            CACHE.write_text(json.dumps(saved, indent=2) + "\n")
            print(f"teacher probes {len(saved['folds'])}/15: {row['fold']}", flush=True)
    coherence = json.loads(Path("results/ps3/rail_coherence_fold_proba.json").read_text())["folds"]
    w7 = json.loads(Path("results/ps3/rail_w7_fold_proba.json").read_text())["folds"]
    old = json.loads(Path("results/ps3/rail_transfer_compare.json").read_text())["selection_folds"]
    moment_mismatches = []
    for i in range(15):
        p = saved["folds"][str(i)]
        c = coherence[str(i)]
        w = w7[str(i)]
        if p["file_ids"] != c["file_ids"] or p["truth"] != c["truth"] or p["file_ids"] != w["file_ids"]:
            raise ValueError(f"held-out files differ in fold {i}")
        speed = np.asarray(p["speed"])
        moment = np.asarray(p["proba"]["MOMENT long"])
        moment_pred = rail._apply_rules(moment, (1.0, 1.0), speed)
        historical_moment = old[str(i)]["rows"]["MOMENT long"]["held_predictions"]
        if [(r["file_id"], r["truth"]) for r in historical_moment] != list(zip(p["file_ids"], p["truth"])):
            raise ValueError(f"MOMENT held-out files do not align in fold {i}")
        for j, (new, row) in enumerate(zip(moment_pred, historical_moment)):
            if str(new) != row["prediction"]:
                moment_mismatches.append({"fold": i, "file_id": p["file_ids"][j],
                                          "historical": row["prediction"], "rerun": str(new),
                                          "probability": moment[j].tolist()})
        mantis = np.asarray(p["proba"]["Mantis short"])
        w7_fused = 0.75 * np.asarray(w["boosted_proba"]) + 0.125 * mantis + 0.125 * moment
        fused_pred = rail._apply_rules(w7_fused, (1.0, 1.0), speed)
        expected_fused = [r["prediction"] for r in old[str(i)]["rows"]["W7 + 12.5% Mantis short + 12.5% MOMENT long"]["held_predictions"]]
        if list(fused_pred) != expected_fused:
            raise ValueError(f"W7 teacher blend does not reproduce historical fold {i}")
    if len(moment_mismatches) > 1:
        raise ValueError(f"MOMENT standalone probe drifted on {len(moment_mismatches)} held-out decisions")
    rows = []
    for coherence_weight in (0.5, 0.75, 0.875, 1.0):
        scores = []
        class_scores = {c: [] for c in RAIL_LABELS}
        changed = 0
        for i in range(15):
            p = saved["folds"][str(i)]
            c = coherence[str(i)]
            teacher = 0.5 * np.asarray(p["proba"]["Mantis short"]) + 0.5 * np.asarray(p["proba"]["MOMENT long"])
            blend = coherence_weight * np.asarray(c["boosted_proba"]) + (1.0 - coherence_weight) * teacher
            pred = rail._apply_rules(blend, (1.0, 1.0), np.asarray(p["speed"]))
            report = class_f1_report(p["truth"], pred)
            scores.append(float(report["macro_f1"]))
            for label in RAIL_LABELS:
                class_scores[label].append(float(report["per_class"][label]["f1"]))
            changed += int(np.sum(pred != np.asarray(c["predictions"], dtype=object)))
        rows.append({
            "coherence_weight": coherence_weight,
            "macro_f1_mean": float(np.mean(scores)),
            "macro_f1_sd": float(np.std(scores)),
            "class_f1_mean": {c: float(np.mean(v)) for c, v in class_scores.items()},
            "changed_decisions_vs_coherence": changed,
        })
    result = {
        "source": "Train only; same duplicate-grouped 5-fold x 3 held-out files; no Test labels",
        "method": "coherence probability plus equal Mantis-short and MOMENT-long frozen-probe probabilities; "
                  "each probe is fitted only on the respective outer training fold",
        "historical_probe_reproduction": "W7+Mantis+MOMENT exact on all 816 held-out decisions; "
                                           "MOMENT standalone has one near-tie mismatch",
        "moment_standalone_mismatches": moment_mismatches,
        "rows": rows,
        "wall_seconds": time.time() - start,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(OUT, rows, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
