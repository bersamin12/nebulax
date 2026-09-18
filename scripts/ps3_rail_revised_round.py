#!/usr/bin/env python
"""Evaluate the three declared W8 rail candidates against the preserved W7 ladder.

Writes a separate checkpoint/report without changing the W7 model or prediction artefacts.
The nested estimate is run once over the original ladder plus the three new rows.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nebulax.ps3 import rail, rail_features as rf  # noqa: E402


def _nested_worker(index: int, n_jobs: int, out: Path) -> tuple[int, dict]:
    feats, y = rail._load_training(n_jobs=1)
    report = rail.nested_cross_validate(
        feats, y, seeds=(0, 1, 2), n_jobs=n_jobs,
        rows=(*rail.LADDER_ROWS, *rail.REVISED_ROWS),
        checkpoint_path=out.with_name(f"rail_revised_nested_fold{index}.json"),
        fold_indices=(index,),
    )
    return index, report["folds"][0]


def _parallel_nested(feats, y, out: Path, workers: int) -> dict:
    partial = out.with_name("rail_revised_nested_partial.json")
    base = json.loads(partial.read_text()) if partial.exists() else {"folds": []}
    splits = list(rail._splits_stratified(y, rail._duplicate_groups(feats), (0, 1, 2), n_splits=5))
    by_name = {f["fold"]: f for f in base["folds"]}
    done = {i: by_name[name] for i, (name, _, _) in enumerate(splits) if name in by_name}
    pending = [i for i in range(len(splits)) if i not in done]
    if pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = {pool.submit(_nested_worker, i, 2, out): i for i in pending}
            for future in as_completed(futures):
                i, fold = future.result()
                if fold["fold"] != splits[i][0]:
                    raise ValueError("parallel nested worker returned the wrong held-out fold")
                done[i] = fold
                print(f"parallel nested {len(done)}/{len(splits)}: {fold['fold']} "
                      f"{fold['macro_f1']:.3f} ({fold['selected_arm']})", flush=True)
    folds = [done[i] for i in range(len(splits))]
    return rail.summarise_nested_folds(folds, seeds=(0, 1, 2),
                                       n_candidates=len(rail.LADDER_ROWS) + len(rail.REVISED_ROWS),
                                       wall_seconds=0.0)


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [
        "# Rail revised round",
        "",
        "Three predeclared rows on the frozen grouped 5-fold x seeds [0, 1, 2] selection CV. "
        "W7 remains the reference at 0.8365. No Test labels were used.",
        f"Promotion required selection CV > {payload['reference_selection_cv']:.4f} and "
        f"nested macro F1 >= {payload['nested_gate']:.4f}.",
        "",
        "| row | macro F1 ± sd | Side I F1 | features |",
        "|---|---:|---:|---:|",
    ]
    if payload.get("reference_row"):
        r = payload["reference_row"]
        lines.append(f"| W7 shock-free reference | {r['macro_f1_mean']:.4f} ± "
                     f"{r['macro_f1_sd']:.4f} | {r['class_f1_mean']['Side I']:.4f} | "
                     f"{r['n_features']} |")
    for r in payload["rows"]:
        lines.append(
            f"| {r['arm']} | {r['macro_f1_mean']:.4f} ± {r['macro_f1_sd']:.4f} | "
            f"{r['class_f1_mean']['Side I']:.4f} | {r['n_features']} |"
        )
    if payload.get("nested"):
        n = payload["nested"]
        lines += ["", f"Nested grouped outer macro F1: {n['macro_f1_mean']:.4f} ± {n['macro_f1_sd']:.4f} "
                  f"({n['n_folds']} folds, {n['n_candidates']} candidates); "
                  f"Side I F1 {n['class_f1_mean']['Side I']:.4f}."]
    if payload.get("wall_seconds"):
        lines += ["", f"Round runtime: {payload['wall_seconds'] / 60:.1f} minutes "
                  "(six-hour budget)."]
    if payload.get("decision"):
        lines += ["", f"Decision: **{payload['decision']}**. Selected Side I F1: "
                  f"{payload['selected_side_i_f1']:.4f}.", ""]
        for s in payload.get("stress", []):
            lines.append(f"- {s['scheme']}: {s['macro_f1_mean']:.4f} ± {s['macro_f1_sd']:.4f}; "
                         f"Side I {s['class_f1_mean']['Side I']:.4f}")
    if payload.get("verification"):
        lines += ["", "## Verification", ""]
        lines += [f"- {item}" for item in payload["verification"]]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--parallel-workers", type=int, default=9)
    ap.add_argument("--out", type=Path, default=ROOT / "results/ps3/rail_revised_round.json")
    args = ap.parse_args()
    started = time.time()
    old = json.loads((ROOT / "results/ps3/rail_ladder.json").read_text())
    reference = float(old["winner"]["macro_f1_mean"])
    feats, y = rail._load_training(n_jobs=args.n_jobs)
    payload = json.loads(args.out.read_text()) if args.out.exists() else {
        "feature_version": rf.FEATURE_VERSION,
        "seeds": [0, 1, 2],
        "reference_selection_cv": reference,
        "nested_gate": 0.7471,
        "rows": [],
    }
    ref = next(r for r in old["rows"] if r["arm"] == old["winner"]["arm"])
    payload["reference_row"] = {k: ref[k] for k in
                                ("arm", "macro_f1_mean", "macro_f1_sd", "class_f1_mean", "n_features")}
    if payload["feature_version"] != rf.FEATURE_VERSION:
        raise ValueError("checkpoint feature version does not match the current extractor")
    if "started_at_epoch" not in payload and "wall_seconds" not in payload:
        payload["started_at_epoch"] = started
    write_report(args.out, payload)
    for arm, kind, kwargs in rail.REVISED_ROWS:
        if any(row["arm"] == arm for row in payload["rows"]):
            continue
        rep = rail.cross_validate(feats, y, scheme="stratified", kind=kind,
                                  seeds=(0, 1, 2), n_jobs=args.n_jobs, **kwargs)
        payload["rows"].append({"arm": arm, "model": kind, "kwargs": kwargs,
                                **{k: rep[k] for k in ("macro_f1_mean", "macro_f1_sd",
                                    "class_f1_mean", "n_features", "fit_seconds_mean",
                                    "macro_f1_speed_matched_mean")}, "report": rep})
        write_report(args.out, payload)
        print(f"{arm}: {rep['macro_f1_mean']:.4f} Side I {rep['class_f1_mean']['Side I']:.4f}", flush=True)

    nested = payload.get("nested") or _parallel_nested(feats, y, args.out, args.parallel_workers)
    if nested["wall_seconds"] == 0.0:
        nested["wall_seconds"] = round(time.time() - started, 2)
    payload["nested"] = nested
    eligible = [r for r in payload["rows"] if r["macro_f1_mean"] > reference + 1e-9]
    best = max(eligible, key=lambda r: r["macro_f1_mean"], default=None)
    if best and nested["macro_f1_mean"] >= payload["nested_gate"]:
        payload["decision"] = "promotion gate passed for " + best["arm"]
        payload["selected"] = best["arm"]
        payload["selected_side_i_f1"] = best["class_f1_mean"]["Side I"]
        payload["stress"] = [rail.cross_validate(feats, y, scheme=scheme, kind=best["model"],
                             opts=best["kwargs"]["opts"], seeds=(0, 1, 2), n_jobs=args.n_jobs,
                             tta=True, aligned_boosts=best["kwargs"].get("aligned_boosts", False))
                             for scheme in ("contiguous", "speed_range")]
    else:
        payload["decision"] = "retain W7"
        payload["selected"] = old["winner"]["arm"]
        payload["selected_side_i_f1"] = next(r["class_f1_mean"]["Side I"]
            for r in old["rows"] if r["arm"] == old["winner"]["arm"])
        payload["stress"] = old["winner_schemes"]
    if "started_at_epoch" in payload:
        payload["wall_seconds"] = round(time.time() - payload["started_at_epoch"], 2)
    else:
        payload.setdefault("wall_seconds", round(time.time() - started, 2))
    write_report(args.out, payload)
    print(payload["decision"], flush=True)


if __name__ == "__main__":
    main()
