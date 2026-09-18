#!/usr/bin/env python
"""Cache frozen Chronos family embeddings for rail vibration recordings."""

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
    "chronos_t5_tiny_short": ("amazon/chronos-t5-tiny", 512, 8, 512, False),
    "chronos_bolt_tiny_long": ("amazon/chronos-bolt-tiny", 2048, 8, 512, False),
    "chronos2_small_joint": ("autogluon/chronos-2-small", 2048, 4, 1024, True),
}


def _prepare(item: tuple[Path, int, int, bool]) -> tuple[str, np.ndarray]:
    path, window, n_windows, joint = item
    arr = rf.read_rail_csv(path)
    vibration = arr[:, 1::2].T
    if vibration.shape != (rf.N_BOXES, len(arr)):
        raise ValueError(f"{path.name}: unexpected vibration layout {vibration.shape}")
    starts = np.linspace(0, len(arr) - window, n_windows, dtype=int)
    windows = np.stack([vibration[:, s:s + window] for s in starts], axis=0)
    if joint:
        return path.name, windows.astype(np.float32)
    return path.name, windows.transpose(1, 0, 2).reshape(rf.N_BOXES*n_windows, window).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("encoder", choices=SPECS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import torch
    from chronos import ChronosBoltPipeline, ChronosPipeline, Chronos2Pipeline
    from huggingface_hub import model_info

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("Chronos extraction requires CUDA for this recording set")
    model_id, window, n_windows, dim, joint = SPECS[args.encoder]
    cache = ROOT / "data/ps3_cache/rail" / args.encoder
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_id": model_id, "revision": model_info(model_id).sha,
        "package": "chronos-forecasting==2.3.2", "window_samples": window,
        "windows_per_channel": n_windows, "joint_64_channel_encoder": joint,
        "embedding_dim": dim, "channels": rf.N_BOXES, "labels_used": False,
    }
    manifest_path = cache / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("existing Chronos cache has another extraction recipe")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    paths = sorted(train_dir("rail").glob("Train*.csv"), key=lambda p: natural_key(p.name))
    if args.limit:
        paths = paths[:args.limit]
    pending = [p for p in paths if not (cache / f"{p.stem}.npy").exists()]
    print(f"{args.encoder}: {len(pending)}/{len(paths)} pending", flush=True)
    if not pending:
        return
    cls = (Chronos2Pipeline if joint else
           ChronosBoltPipeline if "bolt" in args.encoder else ChronosPipeline)
    pipeline = cls.from_pretrained(model_id, device_map="cuda", dtype=torch.float32)
    started = time.time()
    jobs = [(p, window, n_windows, joint) for p in pending]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for number, (file_id, windows) in enumerate(pool.map(_prepare, jobs), 1):
            if joint:
                embeddings, _ = pipeline.embed(windows, batch_size=64,
                                               context_length=window)
                if len(embeddings) != n_windows:
                    raise ValueError("Chronos-2 joint embedding count mismatch")
                # Each output is [64 sensors, context patches + REG + masked future, d_model].
                z = np.stack([torch.cat([e[:, :-2].mean(dim=1), e[:, -2]], dim=1).numpy()
                              for e in embeddings], axis=1)
            else:
                chunks = []
                for start in range(0, len(windows), args.batch_size):
                    e, _ = pipeline.embed(torch.from_numpy(windows[start:start+args.batch_size]))
                    if "bolt" in args.encoder:
                        v = torch.cat([e[:, :-1].mean(dim=1), e[:, -1]], dim=1)
                    else:
                        v = torch.cat([e[:, :-1].mean(dim=1), e[:, -1]], dim=1)
                    chunks.append(v.numpy())
                z = np.concatenate(chunks).reshape(rf.N_BOXES, n_windows, dim)
            if z.shape != (rf.N_BOXES, n_windows, dim) or not np.isfinite(z).all():
                raise ValueError(f"{file_id}: bad Chronos embedding {z.shape}")
            np.save(cache / f"{Path(file_id).stem}.npy", z.mean(axis=1).astype(np.float32))
            if number % 10 == 0 or number == len(pending):
                print(f"{args.encoder}: {number}/{len(pending)} in "
                      f"{(time.time()-started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
