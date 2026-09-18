#!/usr/bin/env python
"""Run two declared W9 sensor augmentation rows on frozen grouped rail folds.

Workers own independent held-out folds. Checkpoints are separate from the selected W7 artefacts.
The expanded nested audit runs only when a new row exceeds W7 selection CV.
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
from nebulax.ps3.scoring import class_f1_report  # noqa: E402

ROWS = rail.SENSOR_AUGMENT_ROWS
SEEDS = (0, 1, 2)
NESTED_ROWS = (*rail.LADDER_ROWS, *rail.REVISED_ROWS, *ROWS)


def _selection_worker(index: int, threads: int) -> tuple[int, list[dict]]:
    feats, y = rail._load_training(n_jobs=1)
    reports = []
    for arm, kind, kwargs in ROWS:
        rep = rail.cross_validate(
            feats, y, scheme="stratified", kind=kind, seeds=SEEDS,
            fold_indices=(index,), include_predictions=True, n_jobs=threads, **kwargs,
        )
        reports.append({"arm": arm, "model": kind, "kwargs": kwargs, "fold": rep["folds"][0]})
    return index, reports


def _nested_worker(index: int, threads: int, checkpoint: Path) -> tuple[int, dict]:
    feats, y = rail._load_training(n_jobs=1)
    rep = rail.nested_cross_validate(
        feats, y, seeds=SEEDS, n_jobs=threads, rows=NESTED_ROWS,
        fold_indices=(index,), checkpoint_path=checkpoint,
    )
    return index, rep["folds"][0]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    lines = [
        "# Rail sensor augmentation round",
        "",
        "Two predeclared W7 variants evaluated on frozen grouped 5-fold × three-seed CV. "
        "Only fault files in each training fold receive one extra transformed and mirrored copy. "
        "Gain jitter is ±0.75 dB per vibration sensor; masking imputes one vibration sensor "
        "on each side from the same-side median. No Test labels were used.",
        "",
        "| row | selection macro F1 ± sd | Side I F1 |",
        "|---|---:|---:|",
        f"| W7 reference | {payload['reference']['macro_f1_mean']:.4f} ± "
        f"{payload['reference']['macro_f1_sd']:.4f} | "
        f"{payload['reference']['class_f1_mean']['Side I']:.4f} |",
    ]
    for row in payload["rows"]:
        if "selection" in row:
            rep = row["selection"]
            lines.append(f"| {row['arm']} | {rep['macro_f1_mean']:.4f} ± "
                         f"{rep['macro_f1_sd']:.4f} | {rep['class_f1_mean']['Side I']:.4f} |")
    if payload.get("nested"):
        nested = payload["nested"]
        lines += ["", f"Expanded nested macro F1: {nested['macro_f1_mean']:.4f} ± "
                  f"{nested['macro_f1_sd']:.4f}; Side I F1 "
                  f"{nested['class_f1_mean']['Side I']:.4f} ({nested['n_folds']} held-out folds, "
                  f"{nested['n_candidates']} candidate rows)."]
    elif payload.get("nested_status"):
        lines += ["", f"Nested audit: {payload['nested_status']}."]
    if payload.get("decision"):
        lines += ["", f"Decision: **{payload['decision']}**."]
    if payload.get("reference_stress"):
        lines += ["", "Retained W7 stress splits:", ""]
        for rep in payload["reference_stress"]:
            lines.append(f"- {rep['scheme']}: {rep['macro_f1_mean']:.4f} ± "
                         f"{rep['macro_f1_sd']:.4f}; Side I F1 "
                         f"{rep['class_f1_mean']['Side I']:.4f}")
    if payload.get("runtime_seconds"):
        lines += ["", f"Runtime: {payload['runtime_seconds'] / 60:.1f} minutes."]
    if payload.get("verification"):
        lines += ["", "## Verification", ""]
        lines += [f"- {item}" for item in payload["verification"]]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarise_selection(folds: list[dict]) -> dict:
    values = np.asarray([f["macro_f1"] for f in folds], dtype=float)
    matched = np.asarray([f["macro_f1_speed_matched"] for f in folds], dtype=float)
    held = [h for f in folds for h in f["held_predictions"]]
    truth = np.asarray([h["truth"] for h in held], dtype=object)
    pred = np.asarray([h["prediction"] for h in held], dtype=object)
    pooled = class_f1_report(truth, pred)
    return {
        "macro_f1_mean": float(values.mean()),
        "macro_f1_sd": float(values.std(ddof=0)),
        "macro_f1_speed_matched_mean": float(matched.mean()),
        "class_f1_mean": {c: float(np.mean([f["class_f1"][c] for f in folds])) for c in RAIL_LABELS},
        "pooled_macro_f1": float(pooled["macro_f1"]),
        "confusion": rail._confusion(truth, pred),
        "fit_seconds_mean": float(np.mean([f["fit_seconds"] for f in folds])),
        "n_folds": len(folds),
        "folds": folds,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/ps3/rail_sensor_augmentation_round.json")
    args = ap.parse_args()
    started = time.time()
    old = json.loads((ROOT / "results/ps3/rail_ladder.json").read_text())
    reference = next(r for r in old["rows"] if r["arm"] == old["winner"]["arm"])
    signature = hashlib.sha1(json.dumps(ROWS, sort_keys=True, default=str).encode()).hexdigest()
    payload = json.loads(args.out.read_text()) if args.out.exists() else {
        "feature_version": rf.FEATURE_VERSION,
        "signature": signature,
        "seeds": list(SEEDS),
        "reference": {k: reference[k] for k in
                      ("arm", "macro_f1_mean", "macro_f1_sd", "class_f1_mean")},
        "selection_gate": float(reference["macro_f1_mean"]),
        "nested_gate": 0.7471,
        "rows": [{"arm": arm, "model": kind, "kwargs": kwargs} for arm, kind, kwargs in ROWS],
        "selection_folds": {},
    }
    if payload["feature_version"] != rf.FEATURE_VERSION or payload["signature"] != signature:
        raise ValueError("checkpoint does not match the declared augmentation rows")
    payload["reference_stress"] = old["winner_schemes"]
    feats, y = rail._load_training(n_jobs=1)
    splits = list(rail._splits_stratified(y, rail._duplicate_groups(feats), SEEDS, n_splits=5))
    pending = [i for i in range(len(splits)) if str(i) not in payload["selection_folds"]]
    if pending:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
            jobs = {pool.submit(_selection_worker, i, args.threads): i for i in pending}
            for future in as_completed(jobs):
                i, reports = future.result()
                if any(r["fold"]["fold"] != splits[i][0] for r in reports):
                    raise ValueError("selection fold mismatch")
                payload["selection_folds"][str(i)] = reports
                _write(args.out, payload)
                print(f"selection {len(payload['selection_folds'])}/{len(splits)}: "
                      f"{splits[i][0]}", flush=True)
    for row in payload["rows"]:
        folds = [next(r["fold"] for r in payload["selection_folds"][str(i)]
                      if r["arm"] == row["arm"]) for i in range(len(splits))]
        row["selection"] = _summarise_selection(folds)
        print(f"{row['arm']}: {row['selection']['macro_f1_mean']:.4f} "
              f"Side I {row['selection']['class_f1_mean']['Side I']:.4f}", flush=True)
    eligible = [r for r in payload["rows"] if
                r["selection"]["macro_f1_mean"] > payload["selection_gate"] + 1e-9]
    if eligible:
        nested_folds = payload.setdefault("nested_folds", {})
        pending = [i for i in range(len(splits)) if str(i) not in nested_folds]
        if pending:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
                jobs = {pool.submit(_nested_worker, i, args.threads,
                                    args.out.with_name(f"rail_sensor_nested_fold{i}.json")): i
                        for i in pending}
                for future in as_completed(jobs):
                    i, fold = future.result()
                    if fold["fold"] != splits[i][0]:
                        raise ValueError("nested fold mismatch")
                    nested_folds[str(i)] = fold
                    _write(args.out, payload)
                    print(f"nested {len(nested_folds)}/{len(splits)}: {fold['fold']}", flush=True)
        payload["nested"] = rail.summarise_nested_folds(
            [nested_folds[str(i)] for i in range(len(splits))],
            seeds=SEEDS, n_candidates=len(NESTED_ROWS), wall_seconds=time.time() - started,
        )
        best = max(eligible, key=lambda r: r["selection"]["macro_f1_mean"])
        payload["decision"] = ("promotion gate passed for " + best["arm"]
                               if payload["nested"]["macro_f1_mean"] >= payload["nested_gate"]
                               else "retain W7: nested gate failed")
    else:
        payload["decision"] = "retain W7: neither augmentation beat selection CV"
        payload["nested_status"] = "not run because neither row cleared the selection gate"
    payload["runtime_seconds"] = round(time.time() - started, 2)
    _write(args.out, payload)
    print(payload["decision"], flush=True)


if __name__ == "__main__":
    # Fold workers control LightGBM threading; BLAS should not start extra worker pools.
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
