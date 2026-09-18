"""Full 23-row rail nested audit after adding the coherence candidate.

Each outer fold runs the original 22 ladder rows plus coherence on grouped
inner folds.  Checkpoints allow a stopped run to resume without repeating
finished folds.  Nothing under models/ or submission/ is changed.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

from nebulax.ps3 import rail


SEEDS = (0, 1, 2)
ROWS = (*rail.LADDER_ROWS, *rail.COHERENCE_ROWS)
CHECKPOINT_DIR = Path("results/ps3/coherence_nested_checkpoints")


def worker(index):
    feats, y = rail._load_training(n_jobs=1)
    checkpoint = CHECKPOINT_DIR / f"fold{index:02d}.json"
    report = rail.nested_cross_validate(
        feats, y, seeds=SEEDS, n_jobs=1, rows=ROWS,
        fold_indices=(index,), checkpoint_path=checkpoint,
    )
    return index, report["folds"][0]


def main():
    start = time.time()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    folds = [None] * 15
    partial = Path("results/ps3/rail_coherence_full_nested_partial.json")
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(worker, i): i for i in range(15)}
        for future in as_completed(futures):
            i, fold = future.result()
            folds[i] = fold
            partial.write_text(json.dumps({"done": sum(v is not None for v in folds),
                                           "folds": folds}, indent=2) + "\n")
            print(i, fold["macro_f1"], fold["selected_arm"], flush=True)
    report = rail.summarise_nested_folds(folds, seeds=SEEDS, n_candidates=len(ROWS),
                                         wall_seconds=time.time() - start)
    report["source"] = "Train only; 22 historical rows plus one coherence row; no Test labels"
    report["prior_nested_macro_f1_mean"] = 0.7571
    out = Path("results/ps3/rail_coherence_full_nested.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out, report["macro_f1_mean"], report["macro_f1_sd"], report["selection_counts"], flush=True)


if __name__ == "__main__":
    main()
