#!/usr/bin/env python
"""Cache frozen UniTS, SimMTM, and longer-context rail vibration embeddings."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nebulax.ps3 import rail_features as rf  # noqa: E402
from nebulax.ps3.common import natural_key, train_dir  # noqa: E402

REPOS = ROOT / "data/ps3_cache/rail/transfer_repos"
WEIGHTS = ROOT / "data/ps3_cache/rail/transfer_weights"
SPECS = {
    "units_long": (2048, 8, 64),
    "simmtm_short": (178, 8, 128),
    "mantis_long": (2048, 8, 512),
    "moment_long": (2048, 8, 512),
}


def _prepare(item: tuple[Path, int, int, str]) -> tuple[str, np.ndarray]:
    path, window_samples, n_windows, encoder = item
    arr = rf.read_rail_csv(path)
    vibration = arr[:, 1::2].T
    if vibration.shape != (rf.N_BOXES, len(arr)):
        raise ValueError(f"{path.name}: unexpected vibration layout {vibration.shape}")
    starts = np.linspace(0, len(arr) - window_samples, n_windows, dtype=int)
    windows = np.stack([vibration[:, s:s + window_samples] for s in starts], axis=1)
    windows = windows.reshape(rf.N_BOXES * n_windows, 1, window_samples).copy()
    if encoder == "moment_long":
        # Low-pass average before decimation, preserving a 0.205 s context in 512 points.
        windows = windows.reshape(-1, 1, 512, 4).mean(axis=-1)
    elif encoder == "simmtm_short":
        means = windows.mean(axis=-1, keepdims=True)
        stds = windows.std(axis=-1, keepdims=True).clip(min=1e-6)
        windows = (windows - means) / stds
    return path.name, windows.astype(np.float32, copy=False)


def _load_model(encoder: str):
    import torch
    if encoder == "mantis_long":
        from mantis.architecture import MantisV2
        model = MantisV2(device="cuda", return_transf_layer=2, output_token="combined")
        model = model.from_pretrained("paris-noah/MantisV2")
        def embed(x):
            return model(x)
        return model.eval(), embed
    if encoder == "moment_long":
        from momentfm import MOMENTPipeline
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small", model_kwargs={"task_name": "embedding"})
        model.init()
        model.to("cuda").eval()
        def embed(x):
            return model(x_enc=x).embeddings
        return model, embed
    if encoder == "units_long":
        sys.path.insert(0, str(REPOS / "UniTS"))
        from models.UniTS import Model
        ckpt = torch.load(WEIGHTS / "units_x32_pretrain_checkpoint.pth",
                          map_location="cpu", weights_only=False)
        args = ckpt["args"]
        cfg = [("CLS_ECG5000", {"dataset": "ECG5000", "task_name": "classification",
                                "enc_in": 1, "num_class": 5, "seq_len": 2048, "pred_len": 0})]
        model = Model(args, cfg, pretrain=True)
        weights = {k.removeprefix("module."): v for k, v in ckpt["student"].items()}
        missing, _ = model.load_state_dict(weights, strict=False)
        if missing != ["category_tokens.CLS_ECG5000"]:
            raise ValueError(f"unexpected missing UniTS pretrained weights: {missing}")
        model.to("cuda").eval()
        def embed(x):
            x = x.transpose(1, 2)  # UniTS expects [batch, time, channel].
            tokens, _, _, n_vars, _ = model.tokenize(x)
            n_patches = tokens.shape[-2]
            tokens = model.prepare_prompt(
                tokens, n_vars, model.prompt_tokens["ECG5000"],
                model.cls_tokens["CLS_ECG5000"], 1, task_name="classification")
            tokens = model.backbone(tokens, args.prompt_num, n_patches)
            # Last token is the pretrained ECG class token; patches provide generic context.
            return torch.cat([tokens[:, 0, -1],
                              tokens[:, 0, args.prompt_num:-1].mean(dim=1)], dim=1)
        return model, embed
    if encoder == "simmtm_short":
        sys.path.insert(0, str(REPOS / "SimMTM/SimMTM_Classification/code"))
        from config_files.SleepEEG_Configs import Config
        from model import TFC
        args = SimpleNamespace(training_mode="pre_train", temperature=0.2, positive_nums=3)
        model = TFC(Config(), args)
        with zipfile.ZipFile(WEIGHTS / "simmtm_checkpoints.zip") as archive:
            name = "checkpoints/classification/SleepEEG_2_Epilepsy/ckpt_best.pt"
            ckpt = torch.load(io.BytesIO(archive.read(name)), map_location="cpu",
                              weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to("cuda").eval()
        def embed(x):
            _, z = model(x)
            return z
        return model, embed
    raise ValueError(encoder)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("encoder", choices=SPECS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("frozen embedding extraction requires CUDA")
    window_samples, n_windows, dim = SPECS[args.encoder]
    cache = ROOT / "data/ps3_cache/rail" / args.encoder
    cache.mkdir(parents=True, exist_ok=True)
    source = {
        "units_long": WEIGHTS / "units_x32_pretrain_checkpoint.pth",
        "simmtm_short": WEIGHTS / "simmtm_checkpoints.zip",
        "mantis_long": None,
        "moment_long": None,
    }[args.encoder]
    manifest = {
        "encoder": args.encoder, "source_checkpoint": str(source.relative_to(ROOT)) if source else None,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
        "window_samples": window_samples, "windows_per_channel": n_windows,
        "effective_input_points": 512 if args.encoder == "moment_long" else window_samples,
        "embedding_dim": dim, "channels": rf.N_BOXES, "labels_used": False,
    }
    manifest_path = cache / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("existing embedding cache has another extraction recipe")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    paths = sorted(train_dir("rail").glob("Train*.csv"), key=lambda p: natural_key(p.name))
    if args.limit:
        paths = paths[:args.limit]
    pending = [p for p in paths if not (cache / f"{p.stem}.npy").exists()]
    print(f"{args.encoder}: {len(pending)}/{len(paths)} pending", flush=True)
    if not pending:
        return
    model, embed = _load_model(args.encoder)
    started = time.time()
    jobs = [(p, window_samples, n_windows, args.encoder) for p in pending]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for number, (file_id, windows) in enumerate(pool.map(_prepare, jobs), 1):
            chunks = []
            for start in range(0, len(windows), args.batch_size):
                x = torch.from_numpy(windows[start:start + args.batch_size]).to("cuda")
                with torch.inference_mode():
                    z = embed(x)
                chunks.append(z.float().cpu().numpy())
            pooled = np.concatenate(chunks).reshape(rf.N_BOXES, n_windows, dim).mean(axis=1)
            if not np.isfinite(pooled).all():
                raise ValueError(f"{file_id}: nonfinite embedding")
            np.save(cache / f"{Path(file_id).stem}.npy", pooled.astype(np.float32))
            if number % 10 == 0 or number == len(pending):
                print(f"{args.encoder}: {number}/{len(pending)} in "
                      f"{(time.time()-started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
