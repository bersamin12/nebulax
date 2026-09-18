"""Sub-window extraction and cache for rail corrugation [R249].

Slices each 1 s (10,000 sample) recording into K = 3 windows of 0.5 s (5,000 samples)
at 0.25 s (2,500 sample) hops:
  w0: [0, 5000)
  w1: [2500, 7500)
  w2: [5000, 10000)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from nebulax.ps3.common import CACHE_DIR, natural_key
from nebulax.ps3 import rail_features as rf

__all__ = [
    "WINDOW_VERSION",
    "WINDOW_SLICES",
    "window_cache_dir",
    "build_window_cache",
    "load_window_features",
    "extract_windows_from_array",
]

WINDOW_VERSION: str = "w1"

WINDOW_SLICES: tuple[tuple[int, slice], ...] = (
    (0, slice(0, 5000)),
    (1, slice(2500, 7500)),
    (2, slice(5000, 10000)),
)


def window_cache_dir(version: str = WINDOW_VERSION) -> Path:
    return CACHE_DIR / "rail_win" / version


def _extract_windows_one(args: tuple[str, str]) -> str:
    path_str, version = args
    p = Path(path_str)
    out = window_cache_dir(version)
    stem = p.stem
    all_exist = all(
        (out / f"{stem}#w{k}.parquet").exists() and (out / f"{stem}#w{k}.spec.npy").exists()
        for k, _ in WINDOW_SLICES
    )
    if all_exist:
        return p.name
    arr = rf.read_rail_csv(p)
    out.mkdir(parents=True, exist_ok=True)
    for k, sl in WINDOW_SLICES:
        shard = out / f"{stem}#w{k}.parquet"
        spec_path = out / f"{stem}#w{k}.spec.npy"
        if shard.exists() and spec_path.exists():
            continue
        window = arr[sl]
        scalars, spec = rf.extract_array(window, name=f"{stem}#w{k}.csv")
        scalars["file_id"] = f"{stem}#w{k}.csv"
        np.save(spec_path, spec)
        pd.DataFrame([scalars]).to_parquet(shard, index=False)
    return p.name


def build_window_cache(
    paths: Sequence[Path | str],
    *,
    version: str = WINDOW_VERSION,
    n_jobs: int = 4,
    progress: bool = True,
) -> None:
    """Extract 3 sub-windows per file that is not cached yet, at most ``n_jobs`` in flight."""
    todo = [Path(p) for p in paths]
    out = window_cache_dir(version)
    out.mkdir(parents=True, exist_ok=True)
    pending = [
        p for p in todo
        if not all((out / f"{p.stem}#w{k}.parquet").exists() and (out / f"{p.stem}#w{k}.spec.npy").exists() for k, _ in WINDOW_SLICES)
    ]
    if not pending:
        return
    args = [(str(p), version) for p in pending]
    done = 0
    if n_jobs <= 1:
        for a in args:
            _extract_windows_one(a)
            done += 1
            if progress and done % 20 == 0:
                print(f"  windows: {done}/{len(args)}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
        for _ in ex.map(_extract_windows_one, args, chunksize=1):
            done += 1
            if progress and done % 20 == 0:
                print(f"  windows: {done}/{len(args)}", flush=True)


def load_window_features(
    file_ids: Sequence[Path | str],
    *,
    version: str = WINDOW_VERSION,
) -> tuple[rf.RailFeatures, np.ndarray]:
    """Load cached features for sub-windows of parent files.

    Returns (RailFeatures of windows, parent index array).
    """
    out = window_cache_dir(version)
    frames: list[pd.DataFrame] = []
    specs: list[np.ndarray] = []
    win_ids: list[str] = []
    parent_indices: list[int] = []

    for i, fid in enumerate(file_ids):
        stem = Path(fid).stem
        for k, _ in WINDOW_SLICES:
            shard = out / f"{stem}#w{k}.parquet"
            spec_path = out / f"{stem}#w{k}.spec.npy"
            if not (shard.exists() and spec_path.exists()):
                raise FileNotFoundError(
                    f"window cache missing for {stem}#w{k} in {out}; run build_window_cache first"
                )
            frames.append(pd.read_parquet(shard))
            specs.append(np.load(spec_path))
            win_ids.append(f"{stem}#w{k}.csv")
            parent_indices.append(i)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    spec = np.stack(specs).astype(np.float32) if specs else np.zeros((0, rf.N_CHANNELS, rf.N_LAMBDA), dtype=np.float32)
    win_feats = rf.RailFeatures(file_ids=win_ids, scalars=df, spectra=spec)
    return win_feats, np.asarray(parent_indices, dtype=int)


def extract_windows_from_array(
    arr: np.ndarray,
    stem: str = "raw",
) -> tuple[rf.RailFeatures, np.ndarray]:
    """Window a raw (n_samples, 129) array on the fly into 3 sub-windows."""
    frames: list[pd.DataFrame] = []
    specs: list[np.ndarray] = []
    win_ids: list[str] = []
    parent_indices: list[int] = []

    for k, sl in WINDOW_SLICES:
        window = arr[sl]
        scalars, spec = rf.extract_array(window, name=f"{stem}#w{k}.csv")
        scalars["file_id"] = f"{stem}#w{k}.csv"
        frames.append(pd.DataFrame([scalars]))
        specs.append(spec)
        win_ids.append(f"{stem}#w{k}.csv")
        parent_indices.append(0)

    df = pd.concat(frames, ignore_index=True)
    spec_arr = np.stack(specs).astype(np.float32)
    win_feats = rf.RailFeatures(file_ids=win_ids, scalars=df, spectra=spec_arr)
    return win_feats, np.asarray(parent_indices, dtype=int)
