#!/usr/bin/env python
"""Evaluate Chronos and Moirai frozen representations against rail W7."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
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
from ps3_rail_mantis_transfer_round import _probe_proba, _side_design  # noqa: E402
from ps3_rail_transfer_compare import _load  # noqa: E402

W7 = "W7 current-code reference"
CACHES = {
    "Chronos T5 tiny short": ROOT / "data/ps3_cache/rail/chronos_t5_tiny_short",
    "Chronos Bolt tiny long": ROOT / "data/ps3_cache/rail/chronos_bolt_tiny_long",
    "Chronos-2 small joint": ROOT / "data/ps3_cache/rail/chronos2_small_joint",
    "Moirai-1 car joint": ROOT / "data/ps3_cache/rail/moirai1_car_joint",
    "Moirai-2 car joint": ROOT / "data/ps3_cache/rail/moirai2_car_joint",
    "Mantis short": ROOT / "data/ps3_cache/rail/mantis_v2_raw512_8win",
}
PROBES = tuple(k for k in CACHES if k != "Mantis short")
FUSIONS = (
    "W7 + 12.5% Mantis short + 12.5% Chronos-2 small joint",
    "W7 + 12.5% Chronos T5 tiny short + 12.5% Chronos-2 small joint",
    "W7 + 12.5% Mantis short + 12.5% Moirai-2 car joint",
)
ROWS = (W7,) + PROBES + tuple(f"W7 + 25% {k}" for k in PROBES) + FUSIONS
OUT = ROOT / "results/ps3/rail_forecast_transfer_compare.json"
W7_PROBA = ROOT / "results/ps3/rail_w7_fold_proba.json"


def _fold(index: int, reference: dict) -> tuple[int, dict]:
    feats, y = rail._load_training(n_jobs=1)
    splits = list(rail._splits_stratified(y, rail._duplicate_groups(feats),
                                          (0, 1, 2), n_splits=5))
    name, tr, te = splits[index]
    if name != reference["fold"] or reference["file_ids"] != [feats.file_ids[int(i)] for i in te]:
        raise ValueError("W7 reference fold mismatch")
    speed = feats.scalars["speed_kmh"].to_numpy(dtype=float)[te]
    base = np.asarray(reference["boosted_proba"], dtype=float)
    probs = {}
    for encoder, cache in CACHES.items():
        embedded = _load(cache, feats.file_ids)
        X, Xm = _side_design(embedded)
        probs[encoder] = _probe_proba(X[tr], Xm[tr], y[tr], X[te], Xm[te])
    predictions = {W7: np.asarray(reference["predictions"], dtype=object)}
    for encoder in PROBES:
        predictions[encoder] = rail._apply_rules(probs[encoder], (1, 1), speed)
        predictions[f"W7 + 25% {encoder}"] = rail._apply_rules(
            0.75*base + 0.25*probs[encoder], (1, 1), speed)
    predictions[FUSIONS[0]] = rail._apply_rules(
        0.75*base + 0.125*probs["Mantis short"] + 0.125*probs["Chronos-2 small joint"],
        (1, 1), speed)
    predictions[FUSIONS[1]] = rail._apply_rules(
        0.75*base + 0.125*probs["Chronos T5 tiny short"] + 0.125*probs["Chronos-2 small joint"],
        (1, 1), speed)
    predictions[FUSIONS[2]] = rail._apply_rules(
        0.75*base + 0.125*probs["Mantis short"] + 0.125*probs["Moirai-2 car joint"],
        (1, 1), speed)
    matched = speed >= rail.SPEED_MATCHED_KMH
    out = {"fold": name, "n_train": len(tr), "n_test": len(te), "rows": {}}
    for arm, pred in predictions.items():
        rep = class_f1_report(y[te], pred)
        out["rows"][arm] = {
            "macro_f1": float(rep["macro_f1"]),
            "class_f1": {c: float(rep["per_class"][c]["f1"]) for c in RAIL_LABELS},
            "macro_f1_speed_matched": float(macro_f1(y[te][matched], pred[matched])) if matched.any() else None,
            "held_predictions": [{"file_id": feats.file_ids[int(i)], "truth": str(t),
                                  "prediction": str(p)} for i, t, p in zip(te, y[te], pred)],
        }
    return index, out


def _summarise(folds: list[dict], arm: str) -> dict:
    row_folds = [{"fold": f["fold"], **f["rows"][arm]} for f in folds]
    values = np.asarray([f["macro_f1"] for f in row_folds])
    matched = np.asarray([f["macro_f1_speed_matched"] for f in row_folds])
    held = [r for f in row_folds for r in f["held_predictions"]]
    truth = np.asarray([r["truth"] for r in held], dtype=object)
    pred = np.asarray([r["prediction"] for r in held], dtype=object)
    return {"macro_f1_mean": float(values.mean()), "macro_f1_sd": float(values.std(ddof=0)),
            "class_f1_mean": {c: float(np.mean([f["class_f1"][c] for f in row_folds])) for c in RAIL_LABELS},
            "macro_f1_speed_matched_mean": float(matched.mean()),
            "confusion": rail._confusion(truth, pred), "n_folds": len(row_folds), "folds": row_folds}


def _write(payload: dict) -> None:
    temp = OUT.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, OUT)
    lines = ["# Rail Chronos and Moirai transfer trial", "",
             "Frozen released forecasting encoders. Chronos-2 processes all 64 sensors jointly; "
             "Moirai processes each car's eight axle boxes jointly. Side I and Side II statistics "
             "use odd and even box positions. Probes are fitted inside duplicate-grouped folds. "
             "No Test labels or files were used.", "",
             f"Historical W7 selection gate: {payload['selection_gate']:.4f}.", "",
             "| row | macro F1 ± sd | Side I F1 | speed-matched macro F1 |",
             "|---|---:|---:|---:|"]
    for arm in ROWS:
        if arm in payload.get("rows", {}):
            r = payload["rows"][arm]
            lines.append(f"| {arm} | {r['macro_f1_mean']:.4f} ± {r['macro_f1_sd']:.4f} | "
                         f"{r['class_f1_mean']['Side I']:.4f} | "
                         f"{r['macro_f1_speed_matched_mean']:.4f} |")
    if payload.get("decision"):
        lines += ["", f"Decision: **{payload['decision']}**."]
    if payload.get("nested_status"):
        lines += ["", f"Nested audit: {payload['nested_status']}."]
    if payload.get("reference_stress"):
        lines += ["", "Retained W7 stress splits:", ""]
        for rep in payload["reference_stress"]:
            lines.append(f"- {rep['scheme']}: {rep['macro_f1_mean']:.4f} ± "
                         f"{rep['macro_f1_sd']:.4f}; Side I F1 "
                         f"{rep['class_f1_mean']['Side I']:.4f}")
    if payload.get("runtime_seconds"):
        lines += ["", f"Probe runtime: {payload['runtime_seconds']/60:.1f} minutes "
                  "after frozen extraction and W7 probability caching."]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    started = time.time()
    manifests = {k: json.loads((v / "manifest.json").read_text()) for k, v in CACHES.items()}
    reference = json.loads(W7_PROBA.read_text())
    signature = hashlib.sha1(json.dumps({"manifests": manifests, "rows": ROWS,
                                         "w7": reference["w7_options"]}, sort_keys=True).encode()).hexdigest()
    payload = json.loads(OUT.read_text()) if OUT.exists() else {
        "signature": signature, "manifests": manifests, "selection_gate": 0.8364557785677434,
        "nested_gate": 0.7471, "selection_folds": {},
    }
    if payload["signature"] != signature:
        raise ValueError("checkpoint does not match declared forecast transfer recipe")
    pending = [i for i in range(15) if str(i) not in payload["selection_folds"]]
    if pending:
        with ProcessPoolExecutor(max_workers=min(12, len(pending))) as pool:
            jobs = {pool.submit(_fold, i, reference["folds"][str(i)]): i for i in pending}
            for future in as_completed(jobs):
                i, result = future.result()
                payload["selection_folds"][str(i)] = result
                _write(payload)
                print(f"completed {len(payload['selection_folds'])}/15: {result['fold']}", flush=True)
    folds = [payload["selection_folds"][str(i)] for i in range(15)]
    payload["rows"] = {arm: _summarise(folds, arm) for arm in ROWS}
    eligible = [arm for arm in ROWS[1:] if payload["rows"][arm]["macro_f1_mean"] > payload["selection_gate"]]
    payload["decision"] = ("promotion candidate: " + max(eligible, key=lambda arm: payload["rows"][arm]["macro_f1_mean"])
                           if eligible else "retain W7: no forecasting encoder row beat selection gate")
    payload["nested_status"] = ("required for promotion candidate" if eligible else
                                "not run because no row cleared the selection gate")
    payload["reference_stress"] = json.loads((ROOT / "results/ps3/rail_ladder.json").read_text())["winner_schemes"]
    payload["runtime_seconds"] = round(time.time()-started, 2)
    _write(payload)
    for arm in ROWS:
        r = payload["rows"][arm]
        print(f"{arm}: {r['macro_f1_mean']:.4f} Side I {r['class_f1_mean']['Side I']:.4f}")
    print(payload["decision"], flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
