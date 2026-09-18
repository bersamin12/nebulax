#!/usr/bin/env python
"""Contiguous and speed-range stress for the short/long transfer blend."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nebulax.ps3 import rail  # noqa: E402
from nebulax.ps3.common import RAIL_LABELS  # noqa: E402
from nebulax.ps3.scoring import class_f1_report  # noqa: E402
from ps3_rail_mantis_transfer_round import _side_design  # noqa: E402
from ps3_rail_transfer_compare import _load, CACHES, W7, FUSIONS  # noqa: E402
from ps3_rail_transfer_nested import _all_predictions  # noqa: E402

OUT = ROOT / "results/ps3/rail_transfer_stress.json"
ARMS = (W7, FUSIONS[0])


def _fold(scheme: str, index: int) -> tuple[str, int, dict]:
    feats, y = rail._load_training(n_jobs=1)
    groups = rail._duplicate_groups(feats)
    speed = feats.scalars["speed_kmh"].to_numpy(dtype=float)
    splits = (list(rail._splits_contiguous(feats.file_ids, groups)) if scheme == "contiguous"
              else list(rail._splits_speed_range(speed, y)))
    name, tr, te = splits[index]
    embeds = {encoder: _side_design(_load(cache, feats.file_ids))
              for encoder, cache in CACHES.items()}
    predictions = _all_predictions(feats, y, embeds, tr, te, 2)
    out = {"fold": name, "n_train": len(tr), "n_test": len(te), "rows": {}}
    for arm in ARMS:
        pred = predictions[arm]
        rep = class_f1_report(y[te], pred)
        out["rows"][arm] = {"macro_f1": float(rep["macro_f1"]),
                            "class_f1": {c: float(rep["per_class"][c]["f1"]) for c in RAIL_LABELS},
                            "held_predictions": [{"file_id": feats.file_ids[int(i)], "truth": str(t),
                                                  "prediction": str(p)} for i, t, p in zip(te, y[te], pred)]}
    return scheme, index, out


def _write(payload: dict) -> None:
    temp = OUT.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, OUT)
    lines = ["# Rail transfer stress trial", "",
             "The same fixed short-Mantis/long-MOMENT blend is evaluated on contiguous "
             "filename blocks and held speed ranges. Classifier fitting remains inside each "
             "training split. No Test files or labels were used.", "",
             "| scheme | row | macro F1 ± sd | Side I F1 |",
             "|---|---|---:|---:|"]
    for scheme, arms in payload.get("summary", {}).items():
        for arm, r in arms.items():
            lines.append(f"| {scheme} | {arm} | {r['macro_f1_mean']:.4f} ± {r['macro_f1_sd']:.4f} | "
                         f"{r['class_f1_mean']['Side I']:.4f} |")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    payload = json.loads(OUT.read_text()) if OUT.exists() else {"arms": list(ARMS), "folds": {}}
    if payload["arms"] != list(ARMS):
        raise ValueError("stress checkpoint row mismatch")
    jobs_to_run = [("contiguous", i) for i in range(5)] + [("speed_range", i) for i in range(3)]
    pending = [(s, i) for s, i in jobs_to_run if f"{s}:{i}" not in payload["folds"]]
    if pending:
        with ProcessPoolExecutor(max_workers=min(8, len(pending))) as pool:
            jobs = {pool.submit(_fold, s, i): (s, i) for s, i in pending}
            for future in as_completed(jobs):
                s, i, result = future.result()
                payload["folds"][f"{s}:{i}"] = result
                _write(payload)
                print(f"stress {len(payload['folds'])}/8: {s} {result['fold']}", flush=True)
    summary = {}
    for scheme, count in (("contiguous", 5), ("speed_range", 3)):
        summary[scheme] = {}
        folds = [payload["folds"][f"{scheme}:{i}"] for i in range(count)]
        for arm in ARMS:
            reps = [f["rows"][arm] for f in folds]
            summary[scheme][arm] = {
                "macro_f1_mean": float(np.mean([r["macro_f1"] for r in reps])),
                "macro_f1_sd": float(np.std([r["macro_f1"] for r in reps])),
                "class_f1_mean": {c: float(np.mean([r["class_f1"][c] for r in reps])) for c in RAIL_LABELS},
                "n_folds": count,
            }
    payload["summary"] = summary
    _write(payload)
    for scheme, arms in summary.items():
        for arm, r in arms.items():
            print(f"{scheme} {arm}: {r['macro_f1_mean']:.4f}, Side I {r['class_f1_mean']['Side I']:.4f}")


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
