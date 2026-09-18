#!/usr/bin/env python
"""Cache joint eight-axle-box-per-car Moirai vibration embeddings."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nebulax.ps3 import rail_features as rf  # noqa: E402
from nebulax.ps3.common import natural_key, train_dir  # noqa: E402

SPECS = {
    "moirai1_car_joint": ("Salesforce/moirai-1.1-R-small", 32),
    "moirai2_car_joint": ("Salesforce/moirai-2.0-R-small", 16),
}
WINDOW = 512
N_WINDOWS = 8
DIM = 384


def _prepare(path: Path) -> tuple[str, np.ndarray]:
    arr = rf.read_rail_csv(path)
    vibration = arr[:, 1::2].T
    if vibration.shape != (rf.N_BOXES, len(arr)):
        raise ValueError(f"{path.name}: unexpected vibration layout {vibration.shape}")
    starts = np.linspace(0, len(arr)-WINDOW, N_WINDOWS, dtype=int)
    windows = np.stack([vibration[:, s:s+WINDOW] for s in starts], axis=0)
    return path.name, windows.reshape(N_WINDOWS*8, 8, WINDOW).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("encoder", choices=SPECS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import torch
    from huggingface_hub import model_info
    from uni2ts.model.moirai import MoiraiModule
    from uni2ts.model.moirai2 import Moirai2Module
    from uni2ts.common.torch_util import packed_attention_mask, packed_causal_attention_mask

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("Moirai extraction requires CUDA for this recording set")
    model_id, patch = SPECS[args.encoder]
    is_v1 = args.encoder.startswith("moirai1")
    cache = ROOT / "data/ps3_cache/rail" / args.encoder
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_id": model_id, "revision": model_info(model_id).sha,
        "package": "uni2ts==2.0.0", "window_samples": WINDOW,
        "windows_per_channel": N_WINDOWS, "patch_size": patch,
        "joint_group": "eight axle boxes within each car",
        "embedding_dim": DIM, "channels": rf.N_BOXES, "labels_used": False,
    }
    manifest_path = cache / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("existing Moirai cache has another extraction recipe")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    paths = sorted(train_dir("rail").glob("Train*.csv"), key=lambda p: natural_key(p.name))
    if args.limit:
        paths = paths[:args.limit]
    pending = [p for p in paths if not (cache / f"{p.stem}.npy").exists()]
    print(f"{args.encoder}: {len(pending)}/{len(paths)} pending", flush=True)
    if not pending:
        return
    cls = MoiraiModule if is_v1 else Moirai2Module
    model = cls.from_pretrained(model_id).to("cuda").eval()
    patch_count = WINDOW // patch
    tokens = 8*patch_count
    target_width = 128 if is_v1 else patch
    time_id = torch.arange(patch_count, device="cuda").repeat(8)
    var_id = torch.arange(8, device="cuda").repeat_interleave(patch_count)
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for number, (file_id, windows) in enumerate(pool.map(_prepare, pending), 1):
            chunks = []
            for start in range(0, len(windows), args.batch_size):
                batch = torch.from_numpy(windows[start:start+args.batch_size]).to("cuda")
                n = len(batch)
                target = batch.reshape(n, 8, patch_count, patch).reshape(n, tokens, patch)
                if is_v1:
                    target = torch.nn.functional.pad(target, (0, target_width-patch))
                observed = torch.zeros_like(target, dtype=torch.bool)
                observed[:, :, :patch] = True
                sample_id = torch.zeros(n, tokens, device="cuda", dtype=torch.long)
                tid = time_id.repeat(n, 1)
                vid = var_id.repeat(n, 1)
                with torch.inference_mode():
                    loc, scale = model.scaler(target, observed, sample_id, vid)
                    scaled = (target-loc)/scale
                    if is_v1:
                        patches = torch.full((n, tokens), patch, device="cuda", dtype=torch.long)
                        z = model.encoder(model.in_proj(scaled, patches),
                                          packed_attention_mask(sample_id),
                                          time_id=tid, var_id=vid)
                    else:
                        inp = model.in_proj(torch.cat([scaled, observed.float()], dim=-1))
                        z = model.encoder(inp, packed_causal_attention_mask(sample_id, tid),
                                          time_id=tid, var_id=vid)
                chunks.append(z.reshape(n, 8, patch_count, DIM).mean(dim=2).float().cpu().numpy())
            encoded = np.concatenate(chunks).reshape(N_WINDOWS, 8, 8, DIM)
            pooled = encoded.transpose(1, 2, 0, 3).reshape(rf.N_BOXES, N_WINDOWS, DIM).mean(axis=1)
            if pooled.shape != (rf.N_BOXES, DIM) or not np.isfinite(pooled).all():
                raise ValueError(f"{file_id}: bad Moirai embedding")
            np.save(cache / f"{Path(file_id).stem}.npy", pooled.astype(np.float32))
            if number % 10 == 0 or number == len(pending):
                print(f"{args.encoder}: {number}/{len(pending)} in "
                      f"{(time.time()-started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
