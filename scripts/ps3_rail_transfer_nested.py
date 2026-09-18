#!/usr/bin/env python
"""Nested grouped selection among W7 and every declared transfer comparison row."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nebulax.ps3 import rail, rail_features as rf  # noqa: E402
from nebulax.ps3.common import RAIL_LABELS  # noqa: E402
from nebulax.ps3.scoring import class_f1_report, macro_f1  # noqa: E402
from ps3_rail_mantis_transfer_round import _probe_proba, _side_design, W7_KWARGS  # noqa: E402
from ps3_rail_transfer_compare import CACHES, W7, PROBES, FUSIONS, ROWS, _load  # noqa: E402

OUT = ROOT / "results/ps3/rail_transfer_nested.json"
SELECTION = ROOT / "results/ps3/rail_transfer_compare.json"


def _all_predictions(feats, y, embeds, tr, te, threads: int) -> dict[str, np.ndarray]:
    speed = feats.scalars["speed_kmh"].to_numpy(dtype=float)
    probs = {}
    for encoder, embedded in embeds.items():
        X, Xm = embedded
        probs[encoder] = _probe_proba(X[tr], Xm[tr], y[tr], X[te], Xm[te])
    model = rail.fit_rail(feats.subset(tr), y[tr], seeds=(0, 1, 2),
                          n_jobs=threads, **W7_KWARGS)
    held = feats.subset(te)
    p = model.proba_tta(rf.aggregate(held, model.opts),
                        rf.aggregate(rf.mirror(held), model.opts))
    predictions = {W7: rail._apply_rules(p, model.boosts, speed[te])}
    p[:, 1] *= model.boosts[0]
    p[:, 2] *= model.boosts[1]
    p /= p.sum(axis=1, keepdims=True)
    for encoder in PROBES:
        predictions[encoder] = rail._apply_rules(probs[encoder], (1, 1), speed[te])
        predictions[f"W7 + 25% {encoder}"] = rail._apply_rules(
            0.75*p + 0.25*probs[encoder], (1, 1), speed[te])
    predictions[FUSIONS[0]] = rail._apply_rules(
        0.75*p + 0.125*probs["Mantis short"] + 0.125*probs["MOMENT long"],
        (1, 1), speed[te])
    predictions[FUSIONS[1]] = rail._apply_rules(
        0.75*p + 0.125*probs["Mantis short"] + 0.125*probs["UniTS long"],
        (1, 1), speed[te])
    return predictions


def _outer(index: int, threads: int, selection_fold: dict) -> tuple[int, dict]:
    feats, y = rail._load_training(n_jobs=1)
    groups = rail._duplicate_groups(feats)
    outer = list(rail._splits_stratified(y, groups, (0, 1, 2), n_splits=5))
    name, outer_tr, outer_te = outer[index]
    if name != selection_fold["fold"]:
        raise ValueError("outer fold mismatch")
    embeds = {encoder: _side_design(_load(cache, feats.file_ids))
              for encoder, cache in CACHES.items()}
    inner = list(rail._splits_stratified(
        y[outer_tr], groups[outer_tr], (10_000 + index,), n_splits=3))
    scores = {arm: [] for arm in ROWS}
    for _, inner_tr, inner_te in inner:
        tr, te = outer_tr[inner_tr], outer_tr[inner_te]
        predictions = _all_predictions(feats, y, embeds, tr, te, threads)
        for arm in ROWS:
            scores[arm].append(float(macro_f1(y[te], predictions[arm])))
    means = {arm: float(np.mean(scores[arm])) for arm in ROWS}
    selected = max(ROWS, key=lambda arm: (means[arm], -ROWS.index(arm)))
    held = selection_fold["rows"][selected]["held_predictions"]
    if [r["file_id"] for r in held] != [feats.file_ids[int(i)] for i in outer_te]:
        raise ValueError("outer held prediction order mismatch")
    truth = np.asarray([r["truth"] for r in held], dtype=object)
    pred = np.asarray([r["prediction"] for r in held], dtype=object)
    rep = class_f1_report(truth, pred)
    speed = feats.scalars["speed_kmh"].to_numpy(dtype=float)[outer_te]
    matched = speed >= rail.SPEED_MATCHED_KMH
    fold = {"fold": name, "n_train": len(outer_tr), "n_test": len(outer_te),
            "selected_row_index": ROWS.index(selected), "selected_arm": selected,
            "selected_model": "W7" if selected == W7 else "frozen transfer",
            "inner_macro_f1_mean": means[selected],
            "inner_macro_f1_sd": float(np.std(scores[selected])),
            "inner_candidate_scores": means,
            "macro_f1": float(rep["macro_f1"]),
            "macro_f1_speed_matched": float(macro_f1(truth[matched], pred[matched])) if matched.any() else None,
            "class_f1": {c: float(rep["per_class"][c]["f1"]) for c in RAIL_LABELS},
            "held_predictions": held}
    return index, fold


def _write(payload: dict) -> None:
    temp = OUT.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, OUT)
    lines = ["# Rail transfer nested audit", "",
             "All 13 declared W7 and transfer rows are ranked inside each outer training "
             "partition using grouped 3-fold CV. Outer held predictions come from the "
             "same fixed recipes in the completed selection run. All external encoders are frozen.", ""]
    if payload.get("summary"):
        r = payload["summary"]
        lines += [f"Nested macro F1: **{r['macro_f1_mean']:.4f} ± {r['macro_f1_sd']:.4f}**; "
                  f"Side I F1 {r['class_f1_mean']['Side I']:.4f}; "
                  f"speed-matched macro F1 {r['macro_f1_speed_matched_mean']:.4f}.", "",
                  f"Nested gate: {payload['nested_gate']:.4f}. "
                  f"Status: **{payload['status']}**.", "", "Selection counts:", ""]
        lines += [f"- {k}: {v}" for k, v in r["selection_counts"].items()]
    if payload.get("two_candidate_summary"):
        two = payload["two_candidate_summary"]
        lines += ["", "W7 versus the short-Mantis/long-MOMENT fusion only:", "",
                  f"- Nested macro F1 {two['macro_f1_mean']:.4f} ± {two['macro_f1_sd']:.4f}; "
                  f"Side I F1 {two['class_f1_mean']['Side I']:.4f}."]
        lines += [f"- {k}: {v} outer folds" for k, v in two["selection_counts"].items()]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    started = time.time()
    sel = json.loads(SELECTION.read_text())
    payload = json.loads(OUT.read_text()) if OUT.exists() else {
        "selection_signature": sel["signature"], "nested_gate": sel["nested_gate"],
        "selection_gate": sel["selection_gate"], "rows": list(ROWS), "folds": {},
    }
    if payload["selection_signature"] != sel["signature"] or payload["rows"] != list(ROWS):
        raise ValueError("nested checkpoint does not match selection rows")
    pending = [i for i in range(15) if str(i) not in payload["folds"]]
    if pending:
        with ProcessPoolExecutor(max_workers=min(12, len(pending))) as pool:
            jobs = {pool.submit(_outer, i, 2, sel["selection_folds"][str(i)]): i
                    for i in pending}
            for future in as_completed(jobs):
                i, result = future.result()
                payload["folds"][str(i)] = result
                _write(payload)
                print(f"nested {len(payload['folds'])}/15: {result['fold']} "
                      f"{result['selected_arm']} {result['macro_f1']:.3f}", flush=True)
    folds = [payload["folds"][str(i)] for i in range(15)]
    summary = rail.summarise_nested_folds(
        folds, seeds=(0, 1, 2), n_candidates=len(ROWS), wall_seconds=time.time()-started)
    payload["summary"] = summary
    payload["status"] = ("passes nested gate" if summary["macro_f1_mean"] >= payload["nested_gate"]
                         else "fails nested gate")
    _write(payload)
    print(f"nested macro {summary['macro_f1_mean']:.4f}, "
          f"Side I {summary['class_f1_mean']['Side I']:.4f}, {payload['status']}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
