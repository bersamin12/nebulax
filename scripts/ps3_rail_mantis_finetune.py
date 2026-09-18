#!/usr/bin/env python
"""Fold-local supervised MantisV2 partial and full backbone fine-tuning trial."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nebulax.ps3 import rail, rail_features as rf  # noqa: E402
from nebulax.ps3.common import RAIL_LABELS, train_dir  # noqa: E402
from nebulax.ps3.scoring import class_f1_report, macro_f1  # noqa: E402
from ps3_rail_mantis_transfer_round import W7_KWARGS  # noqa: E402

RAW_CACHE = ROOT / "data/ps3_cache/rail/mantis_finetune_raw512_8win.npy"
RAW_IDS = RAW_CACHE.with_suffix(".json")
N_WINDOWS = 8
WINDOW = 512
SEEDS = (0, 1, 2)
MODES = ("first_block", "last_block", "all_three_blocks")
W7_PROBA = ROOT / "results/ps3/rail_w7_fold_proba.json"


def _read(path: Path) -> np.ndarray:
    arr = rf.read_rail_csv(path)
    vibration = arr[:, 1::2].T
    if vibration.shape != (rf.N_BOXES, len(arr)):
        raise ValueError(f"{path.name}: unexpected vibration layout {vibration.shape}")
    starts = np.linspace(0, len(arr) - WINDOW, N_WINDOWS, dtype=int)
    return np.stack([vibration[:, s:s + WINDOW] for s in starts], axis=1).astype(np.float32)


def _ensure_raw(file_ids: list[str]) -> None:
    if RAW_CACHE.exists() and RAW_IDS.exists():
        if json.loads(RAW_IDS.read_text()) != file_ids:
            raise ValueError("raw window cache uses a different file order")
        x = np.load(RAW_CACHE, mmap_mode="r")
        if x.shape != (len(file_ids), rf.N_BOXES, N_WINDOWS, WINDOW):
            raise ValueError(f"unexpected raw cache shape {x.shape}")
        return
    RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temp = RAW_CACHE.with_suffix(".tmp.npy")
    x = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32,
                                   shape=(len(file_ids), rf.N_BOXES, N_WINDOWS, WINDOW))
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, row in enumerate(pool.map(_read, [train_dir("rail") / fid for fid in file_ids])):
            x[i] = row
            if (i + 1) % 40 == 0:
                print(f"cached raw windows {i+1}/{len(file_ids)}", flush=True)
    del x
    os.replace(temp, RAW_CACHE)
    RAW_IDS.write_text(json.dumps(file_ids) + "\n")


def _forward(model, head, windows):
    import torch
    batch, _, nwin, _ = windows.shape
    z = model(windows.reshape(batch * rf.N_BOXES * nwin, 1, WINDOW))
    z = z.reshape(batch, rf.N_BOXES, nwin, 512).mean(dim=2)
    a, b = z[:, 0::2], z[:, 1::2]
    features = torch.cat([a.mean(1), b.mean(1),
                          a.std(1, unbiased=False), b.std(1, unbiased=False)], dim=1)
    mirrored = torch.cat([features[:, 512:1024], features[:, :512],
                          features[:, 1536:], features[:, 1024:1536]], dim=1)
    return head(features), head(mirrored)


def _train_fold(fold_no: int, mode: str, epochs: int, raw: np.ndarray,
                feats, y, splits: list) -> dict:
    import torch
    from mantis.architecture import MantisV2

    torch.manual_seed(2026 + fold_no)
    np.random.seed(2026 + fold_no)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    name, tr, te = splits[fold_no]
    model = MantisV2(device="cuda", return_transf_layer=2, output_token="combined")
    model = model.from_pretrained("paris-noah/MantisV2")
    model.remove_transf_layers()
    if mode in ("first_block", "last_block"):
        for param in model.parameters():
            param.requires_grad = False
        block = 0 if mode == "first_block" else -1
        for param in model.transf_unit.transformer.layers[block].parameters():
            param.requires_grad = True
        if mode == "first_block":
            for param in model.tokgen_unit.parameters():
                param.requires_grad = True
    elif mode != "all_three_blocks":
        raise ValueError(mode)
    head = torch.nn.Linear(2048, 3).to("cuda")
    optim = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-5},
        {"params": head.parameters(), "lr": 5e-4},
    ], weight_decay=1e-3)
    lookup = {label: i for i, label in enumerate(RAIL_LABELS)}
    labels = np.array([lookup[v] for v in y])
    mirrored_labels = np.array([0, 2, 1], dtype=np.int64)
    # Train on every fold-training file, with one extra view of each rare side case.
    extra = tr[labels[tr] != 0]
    rng = np.random.default_rng(2026 + fold_no)
    losses = []
    started = time.time()
    for epoch in range(epochs):
        order = np.concatenate([tr, extra])
        rng.shuffle(order)
        model.train()
        head.train()
        epoch_losses = []
        for start in range(0, len(order), 2):
            idx = order[start:start + 2]
            picked = rng.choice(N_WINDOWS, size=2, replace=False)
            # np.take keeps [batch, box, selected window, sample] ordering.
            windows = np.take(raw[idx], picked, axis=2).copy()
            x = torch.from_numpy(windows).to("cuda")
            target = torch.as_tensor(labels[idx], device="cuda")
            model.zero_grad(set_to_none=True)
            head.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, logits_m = _forward(model, head, x)
                loss = 0.5 * (torch.nn.functional.cross_entropy(logits.float(), target) +
                              torch.nn.functional.cross_entropy(
                                  logits_m.float(), torch.as_tensor(mirrored_labels[labels[idx]], device="cuda")))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)))
        print(f"{mode} {name} epoch {epoch+1}/{epochs} train loss {losses[-1]:.4f}", flush=True)
    model.eval()
    head.eval()
    probs = []
    with torch.inference_mode():
        for i in te:
            x = torch.from_numpy(np.asarray(raw[i:i+1]).copy()).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, logits_m = _forward(model, head, x)
            p = torch.softmax(logits.float(), dim=-1).cpu().numpy()[0]
            pm = torch.softmax(logits_m.float(), dim=-1).cpu().numpy()[0][[0, 2, 1]]
            probs.append(0.5 * (p + pm))
    probs = np.asarray(probs)
    speed = feats.scalars["speed_kmh"].to_numpy(dtype=float)[te]
    pred = rail._apply_rules(probs, (1.0, 1.0), speed)
    rep = class_f1_report(y[te], pred)
    matched = speed >= rail.SPEED_MATCHED_KMH
    return {"fold": name, "mode": mode, "epochs": epochs, "n_train": len(tr), "n_test": len(te),
            "train_loss": losses, "runtime_seconds": round(time.time()-started, 2),
            "macro_f1": float(rep["macro_f1"]),
            "class_f1": {c: float(rep["per_class"][c]["f1"]) for c in RAIL_LABELS},
            "macro_f1_speed_matched": float(macro_f1(y[te][matched], pred[matched])) if matched.any() else None,
            "held_predictions": [{"file_id": feats.file_ids[int(i)], "truth": str(t),
                                  "prediction": str(pr), "proba": p.tolist()}
                                 for i, t, pr, p in zip(te, y[te], pred, probs)]}


def _finish(payload: dict, feats) -> None:
    if not all(f"{mode}:{i}" in payload["folds"] for mode in MODES for i in range(15)):
        payload.pop("summary", None)
        payload.pop("decision", None)
        payload.pop("nested_status", None)
        return
    reference = json.loads(W7_PROBA.read_text())
    speed_all = feats.scalars["speed_kmh"].to_numpy(dtype=float)
    summaries = {}
    for mode in MODES:
        for variant in ("fine-tuned only", "W7 + 25% fine-tuned"):
            reports = []
            for i in range(15):
                fold = payload["folds"][f"{mode}:{i}"]
                ref = reference["folds"][str(i)]
                held = fold["held_predictions"]
                if [r["file_id"] for r in held] != ref["file_ids"] or [r["truth"] for r in held] != ref["truth"]:
                    raise ValueError(f"Mantis/W7 fold mismatch at {i}")
                truth = np.asarray(ref["truth"], dtype=object)
                if variant == "fine-tuned only":
                    pred = np.asarray([r["prediction"] for r in held], dtype=object)
                else:
                    p = 0.75*np.asarray(ref["boosted_proba"]) + 0.25*np.asarray([r["proba"] for r in held])
                    speed = np.asarray([speed_all[feats.file_ids.index(fid)] for fid in ref["file_ids"]])
                    pred = rail._apply_rules(p, (1.0, 1.0), speed)
                rep = class_f1_report(truth, pred)
                reports.append({"fold": fold["fold"], "macro_f1": float(rep["macro_f1"]),
                                "class_f1": {c: float(rep["per_class"][c]["f1"]) for c in RAIL_LABELS}})
            key = f"{mode} / {variant}"
            summaries[key] = {"macro_f1_mean": float(np.mean([r["macro_f1"] for r in reports])),
                              "macro_f1_sd": float(np.std([r["macro_f1"] for r in reports])),
                              "class_f1_mean": {c: float(np.mean([r["class_f1"][c] for r in reports]))
                                                for c in RAIL_LABELS}, "n_folds": len(reports), "folds": reports}
    payload["summary"] = summaries
    eligible = [k for k, r in summaries.items() if r["macro_f1_mean"] > payload["selection_gate"]]
    payload["decision"] = ("promotion candidate: " + max(eligible, key=lambda k: summaries[k]["macro_f1_mean"])
                           if eligible else "retain W7: fine-tuning rows did not beat selection gate")
    payload["nested_status"] = ("required for promotion candidate" if eligible else
                                "not run because no fine-tuning row cleared the selection gate")


def _write_report(payload: dict, out: Path) -> None:
    lines = ["# Rail MantisV2 backbone fine-tuning trial", "",
             "MantisV2's first three pretrained transformer blocks process 512-sample vibration "
             "windows. The input and first block, last block, and all-three-block modes are trained separately inside "
             "each of 15 duplicate-grouped folds. The 64 sensors are pooled by rail side. "
             "Two windows per sensor are sampled at each training step; all eight cached "
             "windows are averaged for held-out prediction. No Test files or labels were used.", "",
             f"Historical W7 selection gate: {payload['selection_gate']:.4f}.", "",
             "| row | macro F1 ± sd | Side I F1 |",
             "|---|---:|---:|"]
    for key, rep in payload.get("summary", {}).items():
        lines.append(f"| {key} | {rep['macro_f1_mean']:.4f} ± {rep['macro_f1_sd']:.4f} | "
                     f"{rep['class_f1_mean']['Side I']:.4f} |")
    if payload.get("decision"):
        lines += ["", f"Decision: **{payload['decision']}**.", "",
                  f"Nested audit: {payload['nested_status']}."]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", nargs="*", type=int, default=[0])
    ap.add_argument("--modes", nargs="*", choices=MODES, default=list(MODES))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--out", type=Path, default=ROOT / "results/ps3/rail_mantis_finetune.json")
    args = ap.parse_args()
    feats, y = rail._load_training(n_jobs=12)
    _ensure_raw(feats.file_ids)
    raw = np.load(RAW_CACHE, mmap_mode="r")
    splits = list(rail._splits_stratified(y, rail._duplicate_groups(feats), SEEDS, n_splits=5))
    payload = json.loads(args.out.read_text()) if args.out.exists() else {
        "model": "MantisV2 pretrained 3-layer encoder", "window_samples": WINDOW,
        "windows_cached": N_WINDOWS, "windows_per_training_step": 2,
        "epochs": args.epochs, "selection_gate": 0.8364557785677434,
        "nested_gate": 0.7471, "folds": {},
    }
    if payload["epochs"] != args.epochs:
        raise ValueError("checkpoint uses another epoch recipe")
    for mode in args.modes:
        for fold_no in args.folds:
            if not 0 <= fold_no < 15:
                raise ValueError(fold_no)
            key = f"{mode}:{fold_no}"
            if key in payload["folds"]:
                continue
            payload["folds"][key] = _train_fold(fold_no, mode, args.epochs, raw, feats, y, splits)
            temp = args.out.with_suffix(".tmp")
            temp.write_text(json.dumps(payload, indent=2) + "\n")
            os.replace(temp, args.out)
            r = payload["folds"][key]
            print(f"completed {key}: macro {r['macro_f1']:.4f}, "
                  f"Side I {r['class_f1']['Side I']:.4f}", flush=True)
    _finish(payload, feats)
    if payload.get("summary"):
        temp = args.out.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temp, args.out)
        _write_report(payload, args.out)
        for key, rep in payload["summary"].items():
            print(f"{key}: {rep['macro_f1_mean']:.4f}, Side I {rep['class_f1_mean']['Side I']:.4f}")
        print(payload["decision"], flush=True)


if __name__ == "__main__":
    main()
