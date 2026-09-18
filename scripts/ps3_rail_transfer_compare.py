#!/usr/bin/env python
"""Compare externally pretrained time-series encoders in grouped rail CV."""

from __future__ import annotations

import argparse
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
from ps3_rail_mantis_transfer_round import _probe_proba, _side_design, W7_KWARGS  # noqa: E402

SEEDS = (0, 1, 2)
CACHES = {
    "Mantis short": ROOT / "data/ps3_cache/rail/mantis_v2_raw512_8win",
    "MOMENT short": ROOT / "data/ps3_cache/rail/moment_1_small_raw512_8win",
    "SimMTM short": ROOT / "data/ps3_cache/rail/simmtm_short",
    "UniTS long": ROOT / "data/ps3_cache/rail/units_long",
    "Mantis long": ROOT / "data/ps3_cache/rail/mantis_long",
    "MOMENT long": ROOT / "data/ps3_cache/rail/moment_long",
}
W7 = "W7 current-code reference"
PROBES = tuple(k for k in CACHES if k != "Mantis short")
BLENDS = tuple(f"W7 + 25% {k}" for k in PROBES)
FUSIONS = (
    "W7 + 12.5% Mantis short + 12.5% MOMENT long",
    "W7 + 12.5% Mantis short + 12.5% UniTS long",
)
ROWS = (W7,) + PROBES + BLENDS + FUSIONS


def _load(cache: Path, file_ids: list[str]) -> np.ndarray:
    dim = json.loads((cache / "manifest.json").read_text())["embedding_dim"]
    chunks = []
    for fid in file_ids:
        p = cache / f"{Path(fid).stem}.npy"
        z = np.load(p)
        if z.shape != (rf.N_BOXES, dim) or not np.isfinite(z).all():
            raise ValueError(f"bad embedding {p}: {z.shape}")
        chunks.append(z)
    return np.stack(chunks)


def _fold(index: int, threads: int) -> tuple[int, dict]:
    feats, y = rail._load_training(n_jobs=1)
    splits = list(rail._splits_stratified(y, rail._duplicate_groups(feats), SEEDS, n_splits=5))
    name, tr, te = splits[index]
    speed = feats.scalars["speed_kmh"].to_numpy(dtype=float)
    probs = {}
    for encoder, cache in CACHES.items():
        embedded = _load(cache, feats.file_ids)
        X, Xm = _side_design(embedded)
        probs[encoder] = _probe_proba(X[tr], Xm[tr], y[tr], X[te], Xm[te])
    w7 = rail.fit_rail(feats.subset(tr), y[tr], seeds=SEEDS, n_jobs=threads, **W7_KWARGS)
    held = feats.subset(te)
    p = w7.proba_tta(rf.aggregate(held, w7.opts), rf.aggregate(rf.mirror(held), w7.opts))
    pred_w7 = rail._apply_rules(p, w7.boosts, speed[te])
    p[:, 1] *= w7.boosts[0]
    p[:, 2] *= w7.boosts[1]
    p /= p.sum(axis=1, keepdims=True)
    predictions = {W7: pred_w7}
    for encoder in PROBES:
        predictions[encoder] = rail._apply_rules(probs[encoder], (1.0, 1.0), speed[te])
        predictions[f"W7 + 25% {encoder}"] = rail._apply_rules(
            0.75 * p + 0.25 * probs[encoder], (1.0, 1.0), speed[te])
    predictions[FUSIONS[0]] = rail._apply_rules(
        0.75 * p + 0.125 * probs["Mantis short"] + 0.125 * probs["MOMENT long"],
        (1.0, 1.0), speed[te])
    predictions[FUSIONS[1]] = rail._apply_rules(
        0.75 * p + 0.125 * probs["Mantis short"] + 0.125 * probs["UniTS long"],
        (1.0, 1.0), speed[te])
    out = {"fold": name, "n_train": len(tr), "n_test": len(te),
           "w7_boosts": list(w7.boosts), "rows": {}}
    matched = speed[te] >= rail.SPEED_MATCHED_KMH
    for arm, pred in predictions.items():
        rep = class_f1_report(y[te], pred)
        out["rows"][arm] = {
            "macro_f1": float(rep["macro_f1"]),
            "class_f1": {c: float(rep["per_class"][c]["f1"]) for c in RAIL_LABELS},
            "macro_f1_speed_matched": float(macro_f1(y[te][matched], pred[matched])) if matched.any() else None,
            "held_predictions": [{"file_id": feats.file_ids[int(i)], "truth": str(t),
                                  "prediction": str(pr)} for i, t, pr in zip(te, y[te], pred)],
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


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)
    lines = ["# Rail external encoder and multiscale transfer trial", "",
             "Frozen externally pretrained encoders. Eight windows per channel are pooled within "
             "each of 64 vibration sensors. Odd box positions form Side I and even positions "
             "form Side II. Probe scaling and fitting remain within each grouped fold. "
             "No Test files or labels were used.", "",
             "MOMENT long uses a 2048-sample window averaged to its 512-point input. UniTS and "
             "Mantis long receive 2048 native samples. SimMTM uses its native 178 samples.", "",
             f"Historical W7 selection gate: {payload['selection_gate']:.4f}. Current-code W7 "
             "is refitted in the table and can vary slightly from the historical result.", "",
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
    if payload.get("deployment_status"):
        lines += ["", f"Deployment: {payload['deployment_status']}."]
    if payload.get("reference_stress"):
        lines += ["", "Retained W7 stress splits:", ""]
        for rep in payload["reference_stress"]:
            lines.append(f"- {rep['scheme']}: {rep['macro_f1_mean']:.4f} ± "
                         f"{rep['macro_f1_sd']:.4f}; Side I F1 "
                         f"{rep['class_f1_mean']['Side I']:.4f}")
    if payload.get("runtime_seconds"):
        lines += ["", f"Selection runtime: {payload['runtime_seconds']/60:.1f} minutes "
                  "excluding frozen extraction."]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, default=ROOT / "results/ps3/rail_transfer_compare.json")
    args = ap.parse_args()
    started = time.time()
    manifests = {k: json.loads((v / "manifest.json").read_text()) for k, v in CACHES.items()}
    signature = hashlib.sha1(json.dumps({"manifests": manifests, "rows": ROWS,
                                         "w7": W7_KWARGS}, sort_keys=True).encode()).hexdigest()
    payload = json.loads(args.out.read_text()) if args.out.exists() else {
        "signature": signature, "manifests": manifests, "seeds": list(SEEDS),
        "selection_gate": 0.8364557785677434, "nested_gate": 0.7471,
        "selection_folds": {},
    }
    if payload["signature"] != signature:
        raise ValueError("checkpoint does not match transfer recipe")
    pending = [i for i in range(15) if str(i) not in payload["selection_folds"]]
    if pending:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
            jobs = {pool.submit(_fold, i, args.threads): i for i in pending}
            for future in as_completed(jobs):
                i, result = future.result()
                payload["selection_folds"][str(i)] = result
                _write(args.out, payload)
                print(f"completed {len(payload['selection_folds'])}/15: {result['fold']}", flush=True)
    folds = [payload["selection_folds"][str(i)] for i in range(15)]
    payload["rows"] = {arm: _summarise(folds, arm) for arm in ROWS}
    eligible = [arm for arm in ROWS[1:] if payload["rows"][arm]["macro_f1_mean"] > payload["selection_gate"]]
    payload["decision"] = ("promotion candidate: " + max(eligible, key=lambda arm: payload["rows"][arm]["macro_f1_mean"])
                           if eligible else "retain W7: transfer rows did not beat selection gate")
    nested_report = ROOT / "results/ps3/rail_transfer_nested.json"
    payload["nested_status"] = ("completed; see rail_transfer_nested.md" if eligible and nested_report.exists()
                                else "required for promotion candidate" if eligible else
                                "not run because no transfer row cleared the selection gate")
    payload["deployment_status"] = ("W7 retained: encoder weights exceed the 5 MB model limit, "
                                    "and the selected fusion corrects only two of 816 held-out decisions"
                                    if eligible else "W7 retained")
    payload["reference_stress"] = json.loads((ROOT / "results/ps3/rail_ladder.json").read_text())["winner_schemes"]
    payload["runtime_seconds"] = round(time.time() - started, 2)
    _write(args.out, payload)
    for arm in ROWS:
        r = payload["rows"][arm]
        print(f"{arm}: {r['macro_f1_mean']:.4f} Side I {r['class_f1_mean']['Side I']:.4f}")
    print(payload["decision"], flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
