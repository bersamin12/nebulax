#!/usr/bin/env python
"""Fold-local supervised Moirai-2 last-layer teacher trial on rail recordings."""

from __future__ import annotations

import argparse
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

RAW = ROOT / "data/ps3_cache/rail/mantis_finetune_raw512_8win.npy"
IDS = RAW.with_suffix(".json")
W7 = ROOT / "results/ps3/rail_w7_fold_proba.json"
OUT = ROOT / "results/ps3/rail_moirai2_finetune.json"
EPOCHS = 5


def _encode(model, windows):
    import torch
    from uni2ts.common.torch_util import packed_causal_attention_mask
    batch = len(windows)
    groups = windows.reshape(batch*8, 8, 512)
    target = groups.reshape(batch*8, 8, 32, 16).reshape(batch*8, 256, 16)
    observed = torch.ones_like(target, dtype=torch.bool)
    sample_id = torch.zeros(batch*8, 256, device=target.device, dtype=torch.long)
    time_id = torch.arange(32, device=target.device).repeat(8).repeat(batch*8, 1)
    variate_id = torch.arange(8, device=target.device).repeat_interleave(32).repeat(batch*8, 1)
    loc, scale = model.scaler(target, observed, sample_id, variate_id)
    scaled = (target-loc)/scale
    inp = model.in_proj(torch.cat([scaled, observed.float()], dim=-1))
    z = model.encoder(inp, packed_causal_attention_mask(sample_id, time_id),
                      time_id=time_id, var_id=variate_id)
    return z.reshape(batch, 8, 8, 32, 384).mean(dim=3).reshape(batch, rf.N_BOXES, 384)


def _classify(head, z):
    import torch
    a, b = z[:, 0::2], z[:, 1::2]
    x = torch.cat([a.mean(1), b.mean(1), a.std(1, unbiased=False),
                   b.std(1, unbiased=False)], dim=1)
    xm = torch.cat([x[:, 384:768], x[:, :384], x[:, 1152:], x[:, 768:1152]], dim=1)
    return head(x), head(xm)


def _fold(index: int, raw: np.ndarray, feats, y, reference: dict,
          rare_weight: float) -> dict:
    import torch
    from uni2ts.model.moirai2 import Moirai2Module
    torch.manual_seed(2026+index)
    torch.set_num_threads(2)
    rng = np.random.default_rng(2026+index)
    splits = list(rail._splits_stratified(y, rail._duplicate_groups(feats),
                                          (0, 1, 2), n_splits=5))
    name, tr, te = splits[index]
    held_ids = [feats.file_ids[int(i)] for i in te]
    reference_fold = reference["folds"][str(index)]
    if name != reference_fold["fold"] or held_ids != reference_fold["file_ids"]:
        raise ValueError(f"Moirai-2/W7 fold mismatch at {index}")
    model = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small").to("cuda")
    for p in model.parameters():
        p.requires_grad = False
    for p in model.encoder.layers[-1].parameters():
        p.requires_grad = True
    head = torch.nn.Linear(1536, 3).to("cuda")
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.layers[-1].parameters(), "lr": 1e-5},
        {"params": head.parameters(), "lr": 5e-4},
    ], weight_decay=1e-3)
    label_index = {label: i for i, label in enumerate(RAIL_LABELS)}
    labels = np.asarray([label_index[v] for v in y])
    mirror_idx = np.asarray([0, 2, 1])
    weights = torch.tensor([1.0, rare_weight, rare_weight], device="cuda")
    extra = tr[labels[tr] != 0]
    losses = []
    started = time.time()
    for epoch in range(EPOCHS):
        order = np.concatenate([tr, extra])
        rng.shuffle(order)
        model.train()
        head.train()
        epoch_losses = []
        for start in range(0, len(order), 2):
            idx = order[start:start+2]
            w = int(rng.integers(0, 8))
            x = torch.from_numpy(np.asarray(raw[idx, :, w, :]).copy()).to("cuda")
            target = torch.as_tensor(labels[idx], device="cuda")
            target_m = torch.as_tensor(mirror_idx[labels[idx]], device="cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = _encode(model, x)
                logits, logits_m = _classify(head, z)
                loss = 0.5*(torch.nn.functional.cross_entropy(logits.float(), target, weight=weights) +
                            torch.nn.functional.cross_entropy(logits_m.float(), target_m, weight=weights))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.layers[-1].parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)))
        print(f"Moirai-2 fine-tune {name} epoch {epoch+1}/{EPOCHS} loss {losses[-1]:.4f}", flush=True)
    model.eval()
    head.eval()
    probs = []
    with torch.inference_mode():
        for i in te:
            # Average all eight frozen sampling positions before the classification head.
            x = torch.from_numpy(np.asarray(raw[i]).transpose(1, 0, 2).copy()).to("cuda")
            chunks = []
            for start in range(0, 8, 2):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    chunks.append(_encode(model, x[start:start+2]).float())
            z = torch.cat(chunks).mean(dim=0, keepdim=True)
            logits, logits_m = _classify(head, z)
            p = torch.softmax(logits.float(), dim=-1).cpu().numpy()[0]
            pm = torch.softmax(logits_m.float(), dim=-1).cpu().numpy()[0][[0, 2, 1]]
            probs.append(0.5*(p+pm))
    probs = np.asarray(probs)
    speed = feats.scalars["speed_kmh"].to_numpy(dtype=float)[te]
    pred = rail._apply_rules(probs, (1, 1), speed)
    base = np.asarray(reference_fold["boosted_proba"])
    blend = rail._apply_rules(0.75*base+0.25*probs, (1, 1), speed)
    matched = speed >= rail.SPEED_MATCHED_KMH
    out = {"fold": name, "n_train": len(tr), "n_test": len(te),
           "train_loss": losses, "runtime_seconds": round(time.time()-started, 2), "rows": {}}
    for arm, predictions in (("Moirai-2 last-layer fine-tune", pred),
                             ("W7 + 25% fine-tuned Moirai-2", blend)):
        rep = class_f1_report(y[te], predictions)
        out["rows"][arm] = {
            "macro_f1": float(rep["macro_f1"]),
            "class_f1": {c: float(rep["per_class"][c]["f1"]) for c in RAIL_LABELS},
            "macro_f1_speed_matched": float(macro_f1(y[te][matched], predictions[matched])) if matched.any() else None,
            "held_predictions": [{"file_id": feats.file_ids[int(i)], "truth": str(t),
                                  "prediction": str(pr), "proba": p.tolist()}
                                 for i, t, pr, p in zip(te, y[te], predictions, probs)],
        }
    return out


def _write(payload: dict, out: Path = OUT) -> None:
    temp = out.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, out)
    reference = json.loads(W7.read_text())
    baseline = [class_f1_report(np.asarray(f["truth"], dtype=object),
                                np.asarray(f["predictions"], dtype=object))
                for f in reference["folds"].values()]
    base_macro = np.mean([r["macro_f1"] for r in baseline])
    base_side_i = np.mean([r["per_class"]["Side I"]["f1"] for r in baseline])
    lines = ["# Rail Moirai-2 last-layer fine-tuning trial", "",
             "The released Moirai-2 small encoder is adapted inside each training fold. "
             "Its last transformer layer and a side-aware head are trained on eight-box car "
             "groups. Validation files are excluded from parameter updates and head fitting. "
             "No Test files or labels were used. "
             f"Fault-class loss weight: {payload.get('rare_weight', 1.0):g}.", "",
             f"Historical W7 selection gate: {payload['selection_gate']:.4f}. "
             f"Current-code W7 reference on these folds: {base_macro:.4f} macro F1; "
             f"Side I F1 {base_side_i:.4f}.", "",
             "| row | macro F1 ± sd | Side I F1 |",
             "|---|---:|---:|"]
    for arm, r in payload.get("summary", {}).items():
        lines.append(f"| {arm} | {r['macro_f1_mean']:.4f} ± {r['macro_f1_sd']:.4f} | "
                     f"{r['class_f1_mean']['Side I']:.4f} |")
    if payload.get("decision"):
        lines += ["", f"Decision: **{payload['decision']}**."]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", nargs="*", type=int, default=[0])
    ap.add_argument("--rare-weight", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.rare_weight <= 0:
        raise ValueError("rare weight must be positive")
    feats, y = rail._load_training(n_jobs=12)
    if json.loads(IDS.read_text()) != feats.file_ids:
        raise ValueError("Moirai fine-tuning raw cache file order changed")
    raw = np.load(RAW, mmap_mode="r")
    reference = json.loads(W7.read_text())
    payload = json.loads(args.out.read_text()) if args.out.exists() else {
        "model_id": "Salesforce/moirai-2.0-R-small", "mode": "last_encoder_layer",
        "epochs": EPOCHS, "rare_weight": args.rare_weight,
        "selection_gate": 0.8364557785677434,
        "nested_gate": 0.7471, "folds": {},
    }
    if payload.get("rare_weight", 1.0) != args.rare_weight:
        raise ValueError("checkpoint uses another class weight")
    for i in args.folds:
        if not 0 <= i < 15:
            raise ValueError(i)
        if str(i) in payload["folds"]:
            continue
        payload["folds"][str(i)] = _fold(i, raw, feats, y, reference, args.rare_weight)
        _write(payload, args.out)
        print(f"Moirai-2 fine-tuned fold {i}: "
              f"{payload['folds'][str(i)]['rows']['Moirai-2 last-layer fine-tune']['macro_f1']:.4f}", flush=True)
    if len(payload["folds"]) == 15:
        summary = {}
        for arm in ("Moirai-2 last-layer fine-tune", "W7 + 25% fine-tuned Moirai-2"):
            reports = [payload["folds"][str(i)]["rows"][arm] for i in range(15)]
            summary[arm] = {
                "macro_f1_mean": float(np.mean([r["macro_f1"] for r in reports])),
                "macro_f1_sd": float(np.std([r["macro_f1"] for r in reports])),
                "class_f1_mean": {c: float(np.mean([r["class_f1"][c] for r in reports])) for c in RAIL_LABELS},
            }
        payload["summary"] = summary
        eligible = [arm for arm in summary if summary[arm]["macro_f1_mean"] > payload["selection_gate"]]
        payload["decision"] = ("teacher candidate: " + max(eligible, key=lambda a: summary[a]["macro_f1_mean"])
                               if eligible else "retain W7: fine-tuned teacher did not beat selection gate")
        _write(payload, args.out)
        for arm, r in summary.items():
            print(f"{arm}: {r['macro_f1_mean']:.4f}, Side I {r['class_f1_mean']['Side I']:.4f}")
        print(payload["decision"])


if __name__ == "__main__":
    main()
