"""Train-only compact-architecture checks against the promoted coherence row."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

from nebulax.ps3 import rail


OUT = Path("results/ps3/rail_compact_coherence_round.json")
BASE = rail.COHERENCE_ROWS[0][2]
ROWS = (
    ("smaller, regularized LightGBM; same 201 features", "lgbm_regularized", BASE),
    ("remove time and vote blocks; keep coherence", "lgbm",
     {"opts": {**BASE["opts"], "time": False, "votes": False}, "tta": True}),
)


def worker(row):
    name, kind, kwargs = row
    feats, y = rail._load_training(n_jobs=1)
    report = rail.cross_validate(feats, y, kind=kind, seeds=(0, 1, 2), n_jobs=1,
                                 include_predictions=True, **kwargs)
    return name, report


def main():
    start = time.time()
    reference = json.loads(Path("results/ps3/rail_coherence_production_selection.json").read_text())
    with ProcessPoolExecutor(max_workers=len(ROWS)) as pool:
        rows = []
        for future in as_completed([pool.submit(worker, row) for row in ROWS]):
            name, report = future.result()
            rows.append({"name": name, "report": report})
            print(name, report["macro_f1_mean"], report["class_f1_mean"],
                  report["n_features"], flush=True)
    rows.sort(key=lambda row: [r[0] for r in ROWS].index(row["name"]))
    payload = {
        "source": "Train only; duplicate-grouped 5-fold x 3; no Test labels",
        "reference_macro_f1_mean": reference["macro_f1_mean"],
        "reference_side_i_f1_mean": reference["class_f1_mean"]["Side I"],
        "reference_n_features": reference["n_features"],
        "rows": rows, "wall_seconds": time.time() - start,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
