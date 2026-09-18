#!/usr/bin/env python
"""Cache frozen MOMENT-1-small embeddings for the 64 rail vibration boxes."""

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

MODEL_ID = "AutonLab/MOMENT-1-small"
CACHE = ROOT / "data/ps3_cache/rail/moment_1_small_raw512_8win"
WINDOW_SAMPLES = 512
N_WINDOWS = 8


def _prepare(path: Path) -> tuple[str, np.ndarray]:
    arr = rf.read_rail_csv(path)
    vibration = arr[:, 1::2].T
    if vibration.shape != (rf.N_BOXES, len(arr)):
        raise ValueError(f"{path.name}: unexpected vibration layout {vibration.shape}")
    starts = np.linspace(0, len(arr) - WINDOW_SAMPLES, N_WINDOWS, dtype=int)
    windows = np.stack([vibration[:, s:s + WINDOW_SAMPLES] for s in starts], axis=1)
    return path.name, windows.reshape(rf.N_BOXES * N_WINDOWS, 1, WINDOW_SAMPLES).copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import torch
    from huggingface_hub import model_info
    from momentfm import MOMENTPipeline

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("MOMENT extraction requires CUDA for this recording set")
    info = model_info(MODEL_ID)
    manifest = {
        "model_id": MODEL_ID, "revision": info.sha, "package": "momentfm==0.1.4",
        "task": "embedding", "source": "raw vibration channels in the 64-box order",
        "sampling_hz": int(rf.FS_HZ), "window_samples": WINDOW_SAMPLES,
        "windows_per_channel": N_WINDOWS, "embedding_dim": 512,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("existing MOMENT cache uses another checkpoint or extraction recipe")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    paths = sorted(train_dir("rail").glob("Train*.csv"), key=lambda p: natural_key(p.name))
    if args.limit:
        paths = paths[:args.limit]
    pending = [p for p in paths if not (CACHE / f"{p.stem}.npy").exists()]
    print(f"MOMENT frozen embedding: {len(pending)}/{len(paths)} pending", flush=True)
    if not pending:
        return
    model = MOMENTPipeline.from_pretrained(MODEL_ID, model_kwargs={"task_name": "embedding"})
    model.init()
    model.to("cuda").eval()
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for number, (file_id, windows) in enumerate(pool.map(_prepare, pending), 1):
            chunks = []
            for start in range(0, len(windows), args.batch_size):
                x = torch.from_numpy(windows[start:start + args.batch_size]).to("cuda")
                with torch.inference_mode():
                    z = model(x_enc=x).embeddings
                chunks.append(z.float().cpu().numpy())
            embedded = np.concatenate(chunks).reshape(rf.N_BOXES, N_WINDOWS, 512)
            pooled = embedded.mean(axis=1).astype(np.float32)
            if not np.isfinite(pooled).all():
                raise ValueError(f"{file_id}: nonfinite MOMENT embedding")
            np.save(CACHE / f"{Path(file_id).stem}.npy", pooled)
            if number % 10 == 0 or number == len(pending):
                print(f"embedded {number}/{len(pending)} in {(time.time()-started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
