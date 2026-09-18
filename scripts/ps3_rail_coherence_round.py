"""Train-only rail coherence features on the frozen W7 grouped folds.

Same-side axle boxes in each car share rail excitation.  A Welch coherence
summary checks whether several boxes have the same frequency content after
allowing arbitrary phase delay.  Extraction uses only each recording's raw
vibration channels.  The existing W7 Hz features, mirror augmentation, class
boost calibration and split scheme are otherwise unchanged.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from nebulax.ps3 import rail, rail_features as rf
from nebulax.ps3.common import RAIL_LABELS, train_dir
from nebulax.ps3.scoring import class_f1_report, macro_f1


OPTS = {"wavelength": False, "hz": True, "v2_normalise": False, "shock": False}
SEEDS = (0, 1, 2)
CACHE = Path("data/ps3_cache/rail_coherence/c1")


def coherence_features(path):
    path = Path(path)
    cached = CACHE / f"{path.stem}.parquet"
    if cached.exists():
        return pd.read_parquet(cached).iloc[0].to_dict()
    raw = rf.read_rail_csv(path)
    x = raw[:, 1::2].T.astype(np.float32, copy=False)  # 64 vibration boxes x time
    x = x - x.mean(axis=1, keepdims=True)
    nperseg, hop = 1024, 512
    starts = np.arange(0, x.shape[1] - nperseg + 1, hop)
    segments = np.stack([x[:, start:start + nperseg] for start in starts], axis=1)
    fft = np.fft.rfft(segments * np.hanning(nperseg).astype(np.float32), axis=-1)
    power = np.mean(np.abs(fft) ** 2, axis=1)
    freq = np.fft.rfftfreq(nperseg, 1 / rf.FS_HZ)
    side_curves = {}
    for side, positions in (("I", (0, 2, 4, 6)), ("II", (1, 3, 5, 7))):
        pairs = []
        for car in range(8):
            boxes = [car * 8 + p for p in positions]
            for j in range(4):
                for k in range(j + 1, 4):
                    a, b = boxes[j], boxes[k]
                    cross = np.mean(fft[a] * np.conj(fft[b]), axis=0)
                    pairs.append(np.abs(cross) ** 2 / np.maximum(power[a] * power[b], 1e-12))
        side_curves[side] = np.mean(pairs, axis=0)
    values = {}
    for band in range(len(rf.HZ_BAND_EDGES) - 1):
        mask = (freq >= rf.HZ_BAND_EDGES[band]) & (freq < rf.HZ_BAND_EDGES[band + 1])
        if not mask.any():
            mask[np.argmin(abs(freq - np.sqrt(rf.HZ_BAND_EDGES[band] * rf.HZ_BAND_EDGES[band + 1])))] = True
        i_val = float(side_curves["I"][mask].mean())
        ii_val = float(side_curves["II"][mask].mean())
        values[f"coh_I_hz{band}"] = i_val
        values[f"coh_II_hz{band}"] = ii_val
        values[f"coh_contrast_hz{band}"] = i_val - ii_val
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([values]).to_parquet(cached, index=False)
    return values


def mirror_coherence(frame):
    out = frame.copy()
    for band in range(len(rf.HZ_BAND_EDGES) - 1):
        i_col, ii_col, d_col = f"coh_I_hz{band}", f"coh_II_hz{band}", f"coh_contrast_hz{band}"
        out[i_col] = frame[ii_col].to_numpy()
        out[ii_col] = frame[i_col].to_numpy()
        out[d_col] = -frame[d_col].to_numpy()
    return out


def cv_worker(index, x, xm, y, speed, groups, ids):
    splits = list(rail._splits_stratified(y, groups, SEEDS, n_splits=5))
    name, tr, te = splits[index]
    xtr = np.concatenate([x[tr], xm[tr]])
    ytr = np.concatenate([y[tr], rail._mirror_labels(y[tr])])
    str_ = np.concatenate([speed[tr], speed[tr]])
    gtr = np.concatenate([groups[tr], groups[tr]])
    models = [rail._make_estimator("lgbm", seed, n_jobs=1).fit(xtr, ytr) for seed in SEEDS]
    oofs = []
    for rep in range(3):
        oof = np.zeros((len(ytr), 3))
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=101 * rep)
        for itr, iva in splitter.split(xtr, ytr, gtr):
            est = rail._make_estimator("lgbm", SEEDS[0], n_jobs=1).fit(xtr[itr], ytr[itr])
            order = [list(est.classes_).index(label) for label in RAIL_LABELS]
            oof[iva] = est.predict_proba(xtr[iva])[:, order]
        oofs.append(oof)
    boosts = rail._tune_boosts(oofs, ytr, str_)
    p = np.zeros((len(te), 3))
    for model in models:
        order = [list(model.classes_).index(label) for label in RAIL_LABELS]
        p += 0.5 * (model.predict_proba(x[te])[:, order] +
                    model.predict_proba(xm[te])[:, order][:, [0, 2, 1]]) / len(models)
    pred = rail._apply_rules(p, boosts, speed[te])
    rep = class_f1_report(y[te], pred)
    matched = speed[te] >= 35
    return {"fold": name, "macro_f1": float(rep["macro_f1"]),
            "side_i_f1": float(rep["per_class"]["Side I"]["f1"]),
            "speed_matched_f1": float(macro_f1(y[te][matched], pred[matched])),
            "boosts": list(map(float, boosts)),
            "held": [{"file_id": ids[int(j)], "truth": str(y[j]), "prediction": str(pv)}
                     for j, pv in zip(te, pred)]}


def main():
    start = time.time()
    feats, y = rail._load_training(n_jobs=8)
    paths = [Path(train_dir("rail")) / fid for fid in feats.file_ids]
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 1)) as pool:
        coh = list(pool.map(coherence_features, paths))
    coh_frame = pd.DataFrame(coh)
    mirrored_coh = mirror_coherence(coh_frame)
    assert np.allclose(mirror_coherence(mirrored_coh), coh_frame)
    X = pd.concat([rf.aggregate(feats, OPTS), coh_frame], axis=1)
    Xm = pd.concat([rf.aggregate(rf.mirror(feats), OPTS), mirrored_coh], axis=1)
    x, xm = X.to_numpy(float), Xm.to_numpy(float)
    speed = feats.scalars.speed_kmh.to_numpy(float)
    groups = rail._duplicate_groups(feats)
    print("features", X.shape, "extraction seconds", round(time.time() - start, 1), flush=True)
    reports = [None] * 15
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(cv_worker, i, x, xm, y, speed, groups, feats.file_ids): i for i in range(15)}
        for future in as_completed(futures):
            i = futures[future]
            reports[i] = future.result()
            print(i, reports[i]["macro_f1"], flush=True)
    values = np.asarray([row["macro_f1"] for row in reports])
    payload = {"source": "Train only; duplicate-grouped 5-fold x 3 seeds; no Test labels",
               "reference_selection_f1": 0.8365,
               "macro_f1_mean": float(values.mean()), "macro_f1_sd": float(values.std()),
               "side_i_f1_mean": float(np.mean([r["side_i_f1"] for r in reports])),
               "speed_matched_f1_mean": float(np.mean([r["speed_matched_f1"] for r in reports])),
               "features": X.columns.tolist(), "folds": reports, "seconds": time.time() - start}
    out = Path("results/ps3/rail_coherence_round.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(out, payload["macro_f1_mean"], payload["side_i_f1_mean"], flush=True)


if __name__ == "__main__":
    main()
