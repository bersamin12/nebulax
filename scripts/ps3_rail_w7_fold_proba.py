#!/usr/bin/env python
"""Cache fold-local W7 probabilities for external transfer comparisons."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nebulax.ps3 import rail, rail_features as rf  # noqa: E402
from ps3_rail_mantis_transfer_round import W7_KWARGS  # noqa: E402

OUT = ROOT / "results/ps3/rail_w7_fold_proba.json"


def _fold(index: int) -> tuple[int, dict]:
    feats, y = rail._load_training(n_jobs=1)
    splits = list(rail._splits_stratified(y, rail._duplicate_groups(feats),
                                          (0, 1, 2), n_splits=5))
    name, tr, te = splits[index]
    model = rail.fit_rail(feats.subset(tr), y[tr], seeds=(0, 1, 2),
                          n_jobs=2, **W7_KWARGS)
    held = feats.subset(te)
    speed = feats.scalars["speed_kmh"].to_numpy(dtype=float)[te]
    p = model.proba_tta(rf.aggregate(held, model.opts),
                        rf.aggregate(rf.mirror(held), model.opts))
    pred = rail._apply_rules(p, model.boosts, speed)
    boosted = p.copy()
    boosted[:, 1] *= model.boosts[0]
    boosted[:, 2] *= model.boosts[1]
    boosted /= boosted.sum(axis=1, keepdims=True)
    return index, {"fold": name, "file_ids": [feats.file_ids[int(i)] for i in te],
                   "truth": [str(v) for v in y[te]], "predictions": [str(v) for v in pred],
                   "boosts": list(model.boosts), "boosted_proba": boosted.tolist()}


def main() -> None:
    payload = json.loads(OUT.read_text()) if OUT.exists() else {
        "model": "W7 current-code reference", "w7_options": W7_KWARGS,
        "seeds": [0, 1, 2], "folds": {},
    }
    if payload["w7_options"] != W7_KWARGS:
        raise ValueError("W7 cache uses another recipe")
    pending = [i for i in range(15) if str(i) not in payload["folds"]]
    if pending:
        with ProcessPoolExecutor(max_workers=min(12, len(pending))) as pool:
            jobs = {pool.submit(_fold, i): i for i in pending}
            for future in as_completed(jobs):
                i, result = future.result()
                payload["folds"][str(i)] = result
                temp = OUT.with_suffix(".tmp")
                temp.write_text(json.dumps(payload, indent=2) + "\n")
                os.replace(temp, OUT)
                print(f"W7 proba {len(payload['folds'])}/15: {result['fold']}", flush=True)
    selection = json.loads((ROOT / "results/ps3/rail_transfer_compare.json").read_text())
    for i in range(15):
        a = payload["folds"][str(i)]
        b = selection["selection_folds"][str(i)]["rows"]["W7 current-code reference"]["held_predictions"]
        if [(x, y) for x, y in zip(a["file_ids"], a["predictions"])] != [
            (r["file_id"], r["prediction"]) for r in b]:
            raise ValueError(f"W7 cached probability prediction mismatch in fold {i}")
    print("W7 probabilities reproduce all 816 reference decisions", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
