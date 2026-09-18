#!/usr/bin/env python
"""Compare frozen MantisV2 rail embeddings and a fixed W7 blend on grouped CV.

The pretrained encoder is never fitted to organiser data. Per-file embeddings are cached by
``ps3_rail_mantis_embed.py``; classifier/scaler fitting and W7 fitting remain inside each fold.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nebulax.ps3 import rail, rail_features as rf  # noqa: E402
from nebulax.ps3.common import RAIL_LABELS  # noqa: E402
from nebulax.ps3.scoring import class_f1_report, macro_f1  # noqa: E402
from ps3_rail_mantis_embed import CACHE  # noqa: E402

SEEDS = (0, 1, 2)
BLEND_WEIGHT = 0.25
W7_KWARGS = {"kind": "lgbm", "opts": {"wavelength": False, "hz": True,
             "v2_normalise": False, "shock": False}, "tta": True, "augment": True,
             "boost_repeats": 3}
ROWS = ("W7 current-code reference", "MantisV2 frozen linear probe",
        "W7 + 25% MantisV2 probability blend")


def _load_embeddings(file_ids: list[str]) -> np.ndarray:
    chunks = []
    for fid in file_ids:
        p = CACHE / f"{Path(fid).stem}.npy"
        if not p.exists():
            raise FileNotFoundError(f"missing frozen Mantis embedding {p}")
        z = np.load(p)
        if z.shape != (rf.N_BOXES, 512) or not np.isfinite(z).all():
            raise ValueError(f"bad Mantis embedding {p}: {z.shape}")
        chunks.append(z)
    return np.stack(chunks)


def _side_design(embedded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and channel spread for each rail; mirror swaps the two complete side blocks."""
    side_i = embedded[:, rf.side_mask("I")]
    side_ii = embedded[:, rf.side_mask("II")]
    mi, mii = side_i.mean(axis=1), side_ii.mean(axis=1)
    si, sii = side_i.std(axis=1), side_ii.std(axis=1)
    return np.concatenate([mi, mii, si, sii], axis=1), np.concatenate([mii, mi, sii, si], axis=1)


def _probe_proba(x_train: np.ndarray, x_train_m: np.ndarray, y_train: np.ndarray,
                 x_test: np.ndarray, x_test_m: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    xa = np.concatenate([x_train, x_train_m])
    ya = np.concatenate([y_train, rail._mirror_labels(y_train)])
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=0.05, class_weight="balanced", max_iter=500, solver="lbfgs",
    ))
    model.fit(xa, ya)
    classes = list(model.classes_)
    idx = [classes.index(c) for c in RAIL_LABELS]
    p = model.predict_proba(x_test)[:, idx]
    pm = model.predict_proba(x_test_m)[:, idx][:, [0, 2, 1]]
    return 0.5 * (p + pm)


def _fold(index: int, threads: int) -> tuple[int, dict]:
    feats, y = rail._load_training(n_jobs=1)
    embedded = _load_embeddings(feats.file_ids)
    X, Xm = _side_design(embedded)
    splits = list(rail._splits_stratified(y, rail._duplicate_groups(feats), SEEDS, n_splits=5))
    name, tr, te = splits[index]
    speed = feats.scalars["speed_kmh"].to_numpy(dtype=float)

    p_probe = _probe_proba(X[tr], Xm[tr], y[tr], X[te], Xm[te])
    y_probe = rail._apply_rules(p_probe, (1.0, 1.0), speed[te])

    w7 = rail.fit_rail(feats.subset(tr), y[tr], seeds=SEEDS, n_jobs=threads, **W7_KWARGS)
    held = feats.subset(te)
    x_w7 = rf.aggregate(held, w7.opts)
    xm_w7 = rf.aggregate(rf.mirror(held), w7.opts)
    p_w7 = w7.proba_tta(x_w7, xm_w7)
    y_w7 = rail._apply_rules(p_w7, w7.boosts, speed[te])
    p_w7_boosted = p_w7.copy()
    p_w7_boosted[:, 1] *= w7.boosts[0]
    p_w7_boosted[:, 2] *= w7.boosts[1]
    p_w7_boosted /= p_w7_boosted.sum(axis=1, keepdims=True)
    p_blend = (1.0 - BLEND_WEIGHT) * p_w7_boosted + BLEND_WEIGHT * p_probe
    y_blend = rail._apply_rules(p_blend, (1.0, 1.0), speed[te])

    out = {"fold": name, "n_train": len(tr), "n_test": len(te), "w7_boosts": list(w7.boosts),
           "rows": {}}
    matched = speed[te] >= rail.SPEED_MATCHED_KMH
    for arm, pred in zip(ROWS, (y_w7, y_probe, y_blend)):
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
    return {
        "macro_f1_mean": float(values.mean()),
        "macro_f1_sd": float(values.std(ddof=0)),
        "class_f1_mean": {c: float(np.mean([f["class_f1"][c] for f in row_folds])) for c in RAIL_LABELS},
        "macro_f1_speed_matched_mean": float(matched.mean()),
        "confusion": rail._confusion(truth, pred),
        "n_folds": len(row_folds),
        "folds": row_folds,
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    lines = ["# Rail frozen MantisV2 transfer trial", "",
             "Frozen external MantisV2 embeddings of eight 512-sample windows from each of 64 "
             "vibration sensors. Encoder weights are unchanged. Mean and spread of the 32 sensors "
             "on each side form a side-aware linear-probe input. All scaling and classifier fitting "
             "is inside each grouped fold. No Test files or labels were used.", "",
             f"Historical W7 selection gate: {payload['selection_gate']:.4f}. The W7 reference "
             "below is refitted with the current duplicate-grouped inner splits; it can differ "
             "slightly from the historical 0.8365 result.", "",
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
    if payload.get("paired_changes"):
        changes = payload["paired_changes"]
        lines += ["", f"The blend changed {len(changes)} of 816 held-out decisions "
                  "relative to the current-code W7 reference:", ""]
        for ch in changes:
            lines.append(f"- {ch['fold']} {ch['file_id']} ({ch['truth']}): "
                         f"{ch['w7_prediction']} → {ch['blend_prediction']}")
    if payload.get("nested_status"):
        lines += ["", f"Nested audit: {payload['nested_status']}."]
    if payload.get("reference_stress"):
        lines += ["", "Retained W7 stress splits:", ""]
        for rep in payload["reference_stress"]:
            lines.append(f"- {rep['scheme']}: {rep['macro_f1_mean']:.4f} ± "
                         f"{rep['macro_f1_sd']:.4f}; Side I F1 "
                         f"{rep['class_f1_mean']['Side I']:.4f}")
    if payload.get("runtime_seconds"):
        lines += ["", f"Runtime: {payload['runtime_seconds']/60:.1f} minutes, "
                  "excluding the one-time frozen embedding extraction."]
    if payload.get("verification"):
        lines += ["", "## Verification", ""]
        lines += [f"- {item}" for item in payload["verification"]]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, default=ROOT / "results/ps3/rail_mantis_transfer_round.json")
    args = ap.parse_args()
    started = time.time()
    manifest = json.loads((CACHE / "manifest.json").read_text())
    signature = hashlib.sha1(json.dumps({"manifest": manifest, "rows": ROWS, "blend": BLEND_WEIGHT,
                                         "w7": W7_KWARGS}, sort_keys=True).encode()).hexdigest()
    payload = json.loads(args.out.read_text()) if args.out.exists() else {
        "signature": signature, "manifest": manifest, "seeds": list(SEEDS),
        "blend_weight": BLEND_WEIGHT, "w7_options": W7_KWARGS,
        "selection_gate": 0.8364557785677434, "nested_gate": 0.7471,
        "selection_folds": {},
    }
    if payload["signature"] != signature:
        raise ValueError("checkpoint does not match frozen Mantis recipe")
    pending = [i for i in range(15) if str(i) not in payload["selection_folds"]]
    if pending:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
            jobs = {pool.submit(_fold, i, args.threads): i for i in pending}
            for future in as_completed(jobs):
                i, result = future.result()
                payload["selection_folds"][str(i)] = result
                _write(args.out, payload)
                print(f"completed {len(payload['selection_folds'])}/15: {result['fold']} "
                      f"probe {result['rows'][ROWS[1]]['macro_f1']:.3f} "
                      f"blend {result['rows'][ROWS[2]]['macro_f1']:.3f}", flush=True)
    folds = [payload["selection_folds"][str(i)] for i in range(15)]
    payload["rows"] = {arm: _summarise(folds, arm) for arm in ROWS}
    payload["paired_changes"] = []
    for fold in folds:
        original = fold["rows"][ROWS[0]]["held_predictions"]
        blended = fold["rows"][ROWS[2]]["held_predictions"]
        for a, b in zip(original, blended):
            if a["prediction"] != b["prediction"]:
                payload["paired_changes"].append({
                    "fold": fold["fold"], "file_id": a["file_id"], "truth": a["truth"],
                    "w7_prediction": a["prediction"], "blend_prediction": b["prediction"],
                })
    eligible = [arm for arm in ROWS[1:] if payload["rows"][arm]["macro_f1_mean"] > payload["selection_gate"]]
    payload["decision"] = ("promotion candidate: " + max(eligible, key=lambda arm: payload["rows"][arm]["macro_f1_mean"])
                           if eligible else "retain W7: transfer rows did not beat selection gate")
    if not eligible:
        payload["nested_status"] = "not run because neither transfer row cleared the selection gate"
    payload["runtime_seconds"] = round(time.time()-started, 2)
    _write(args.out, payload)
    for arm in ROWS:
        rep = payload["rows"][arm]
        print(f"{arm}: {rep['macro_f1_mean']:.4f} Side I {rep['class_f1_mean']['Side I']:.4f}", flush=True)
    print(payload["decision"], flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
