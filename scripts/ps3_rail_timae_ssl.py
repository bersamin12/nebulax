#!/usr/bin/env python
"""Fold-local Ti-MAE-style masked pretraining and rail linear-probe trial.

No externally pretrained Ti-MAE checkpoint is released by the available implementation,
so this is an in-domain self-supervised architecture test rather than a transfer test.
"""

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
from ps3_rail_mantis_transfer_round import _probe_proba, _side_design  # noqa: E402

SOURCE = ROOT / "data/ps3_cache/rail/transfer_repos/ti-mae"
RAW = ROOT / "data/ps3_cache/rail/timae_raw128_8win.npy"
IDS = RAW.with_suffix(".json")
OUT = ROOT / "results/ps3/rail_timae_ssl.json"
W7 = ROOT / "results/ps3/rail_w7_fold_proba.json"
WINDOW = 128
N_WINDOWS = 8
EPOCHS = 5


def _read(path: Path) -> np.ndarray:
    arr = rf.read_rail_csv(path)
    vib = arr[:, 1::2].T
    if vib.shape != (rf.N_BOXES, len(arr)):
        raise ValueError(f"{path.name}: unexpected vibration layout {vib.shape}")
    starts = np.linspace(0, len(arr)-WINDOW, N_WINDOWS, dtype=int)
    windows = np.stack([vib[:, s:s+WINDOW] for s in starts], axis=1)
    mean = windows.mean(axis=-1, keepdims=True)
    std = windows.std(axis=-1, keepdims=True).clip(min=1e-6)
    return ((windows-mean)/std).astype(np.float32)


def _ensure_raw(file_ids: list[str]) -> None:
    if RAW.exists() and IDS.exists():
        if json.loads(IDS.read_text()) != file_ids:
            raise ValueError("Ti-MAE raw cache file order changed")
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    tmp = RAW.with_suffix(".tmp.npy")
    x = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                   shape=(len(file_ids), rf.N_BOXES, N_WINDOWS, WINDOW))
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, row in enumerate(pool.map(_read, [train_dir("rail") / fid for fid in file_ids])):
            x[i] = row
    del x
    os.replace(tmp, RAW)
    IDS.write_text(json.dumps(file_ids) + "\n")


def _fold(index: int, raw: np.ndarray, feats, y, reference: dict) -> dict:
    import torch
    sys.path.insert(0, str(SOURCE))
    from src.nn.model import MaskedAutoencoder

    torch.manual_seed(2026+index)
    torch.set_num_threads(2)
    rng = np.random.default_rng(2026+index)
    splits = list(rail._splits_stratified(y, rail._duplicate_groups(feats),
                                          (0, 1, 2), n_splits=5))
    name, tr, te = splits[index]
    model = MaskedAutoencoder(
        in_chans=1, seq_len=WINDOW, embed_dim=64, depth=2, num_heads=4,
        decoder_embed_dim=32, decoder_depth=2, decoder_num_heads=4,
        mask_ratio=0.75, forecast_ratio=0, cls_embed=False,
    ).to("cuda")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    losses = []
    started = time.time()
    for epoch in range(EPOCHS):
        pick = rng.integers(0, N_WINDOWS, size=(len(tr), rf.N_BOXES, 1, 1))
        samples = np.take_along_axis(raw[tr], pick, axis=2).reshape(-1, WINDOW).copy()
        order = rng.permutation(len(samples))
        model.train()
        epoch_losses = []
        for start in range(0, len(order), 256):
            x = torch.from_numpy(samples[order[start:start+256], None, :]).to("cuda")
            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                reconstruction_losses = model(x, torch.zeros(len(x), device="cuda", dtype=torch.long))[0]
                loss = reconstruction_losses[0]
            loss.backward()
            optim.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)))
        print(f"Ti-MAE {name} epoch {epoch+1}/{EPOCHS} masked MAE {losses[-1]:.4f}", flush=True)
    model.eval()
    embedded = np.empty((len(feats.file_ids), rf.N_BOXES, 32), dtype=np.float32)
    with torch.inference_mode():
        for i in range(len(feats.file_ids)):
            windows = np.asarray(raw[i]).reshape(rf.N_BOXES*N_WINDOWS, WINDOW)
            chunks = []
            for start in range(0, len(windows), 256):
                x = torch.from_numpy(windows[start:start+256].copy()[:, None, :]).to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = model.forward_encoder(x, 0)[0]
                chunks.append(z.float().mean(dim=1).cpu().numpy())
            embedded[i] = np.concatenate(chunks).reshape(rf.N_BOXES, N_WINDOWS, 32).mean(axis=1)
    if not np.isfinite(embedded).all():
        raise ValueError("nonfinite Ti-MAE embedding")
    X, Xm = _side_design(embedded)
    p = _probe_proba(X[tr], Xm[tr], y[tr], X[te], Xm[te])
    speed = feats.scalars["speed_kmh"].to_numpy(dtype=float)[te]
    pred = rail._apply_rules(p, (1, 1), speed)
    base = np.asarray(reference["folds"][str(index)]["boosted_proba"])
    blend = rail._apply_rules(0.75*base + 0.25*p, (1, 1), speed)
    matched = speed >= rail.SPEED_MATCHED_KMH
    out = {"fold": name, "n_train": len(tr), "n_test": len(te),
           "masked_mae": losses, "runtime_seconds": round(time.time()-started, 2), "rows": {}}
    for arm, predictions in (("Ti-MAE SSL probe", pred), ("W7 + 25% Ti-MAE SSL", blend)):
        rep = class_f1_report(y[te], predictions)
        out["rows"][arm] = {
            "macro_f1": float(rep["macro_f1"]),
            "class_f1": {c: float(rep["per_class"][c]["f1"]) for c in RAIL_LABELS},
            "macro_f1_speed_matched": float(macro_f1(y[te][matched], predictions[matched])) if matched.any() else None,
            "held_predictions": [{"file_id": feats.file_ids[int(i)], "truth": str(t),
                                  "prediction": str(pr)} for i, t, pr in zip(te, y[te], predictions)],
        }
    return out


def _write(payload: dict) -> None:
    temp = OUT.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, OUT)
    lines = ["# Rail Ti-MAE architecture trial", "",
             "An unofficial Ti-MAE implementation has no released general pretrained checkpoint. "
             "This trial trains masked reconstruction from scratch on each outer training fold's "
             "unlabelled vibration windows, then fits a side-aware logistic probe within that "
             "fold. Held-out recordings never enter pretraining. No Test files or labels were used.", "",
             "| row | macro F1 ± sd | Side I F1 | speed-matched macro F1 |",
             "|---|---:|---:|---:|"]
    for arm, r in payload.get("summary", {}).items():
        lines.append(f"| {arm} | {r['macro_f1_mean']:.4f} ± {r['macro_f1_sd']:.4f} | "
                     f"{r['class_f1_mean']['Side I']:.4f} | {r['macro_f1_speed_matched_mean']:.4f} |")
    if payload.get("decision"):
        lines += ["", f"Decision: **{payload['decision']}**."]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", nargs="*", type=int, default=[0])
    args = ap.parse_args()
    feats, y = rail._load_training(n_jobs=12)
    _ensure_raw(feats.file_ids)
    raw = np.load(RAW, mmap_mode="r")
    reference = json.loads(W7.read_text())
    payload = json.loads(OUT.read_text()) if OUT.exists() else {
        "source": "asmodaay/ti-mae (unofficial, no released weights)",
        "epochs": EPOCHS, "window_samples": WINDOW, "selection_gate": 0.8364557785677434,
        "nested_gate": 0.7471, "folds": {},
    }
    for i in args.folds:
        if not 0 <= i < 15:
            raise ValueError(i)
        if str(i) in payload["folds"]:
            continue
        payload["folds"][str(i)] = _fold(i, raw, feats, y, reference)
        _write(payload)
        print(f"Ti-MAE completed fold {i}: "
              f"{payload['folds'][str(i)]['rows']['Ti-MAE SSL probe']['macro_f1']:.4f}", flush=True)
    if len(payload["folds"]) == 15:
        summary = {}
        for arm in ("Ti-MAE SSL probe", "W7 + 25% Ti-MAE SSL"):
            reports = [payload["folds"][str(i)]["rows"][arm] for i in range(15)]
            summary[arm] = {
                "macro_f1_mean": float(np.mean([r["macro_f1"] for r in reports])),
                "macro_f1_sd": float(np.std([r["macro_f1"] for r in reports])),
                "class_f1_mean": {c: float(np.mean([r["class_f1"][c] for r in reports])) for c in RAIL_LABELS},
                "macro_f1_speed_matched_mean": float(np.mean([r["macro_f1_speed_matched"] for r in reports])),
            }
        payload["summary"] = summary
        eligible = [arm for arm in summary if summary[arm]["macro_f1_mean"] > payload["selection_gate"]]
        payload["decision"] = ("promotion candidate: " + max(eligible, key=lambda a: summary[a]["macro_f1_mean"])
                               if eligible else "retain W7: Ti-MAE rows did not beat selection gate")
        _write(payload)
        for arm, r in summary.items():
            print(f"{arm}: {r['macro_f1_mean']:.4f}, Side I {r['class_f1_mean']['Side I']:.4f}")
        print(payload["decision"])


if __name__ == "__main__":
    main()
