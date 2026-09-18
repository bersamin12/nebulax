"""Quick CV: does a per-car level-profile block help Side I vs Side II? Same 5-fold x 3 seeds shape; disposable."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from nebulax.ps3 import rail_features as rf
from nebulax.ps3.scoring import macro_f1

ROOT = Path("readingmaterials/problem_statement/PS3/02_Datasets/Rail_Corrugation")
labels = pd.read_csv(ROOT / "Train_Labels.csv").set_index("filename")["label"]
paths = sorted((ROOT / "Train").glob("*.csv"), key=lambda p: rf.natural_key(p.name))
feats = rf.load_feature_cache(paths)
y = labels.loc[feats.file_ids].to_numpy()
CL = np.array(["Normal", "Side I", "Side II"])
WIN = {"wavelength": False, "hz": True, "v2_normalise": False}  # the shipped winner's representation

def car_profile(fe: rf.RailFeatures) -> pd.DataFrame:
    """Per car (8) x per band: mean over the 8 positions of the level relative to the file mean.
    Also per-position-within-car (8) x band, averaged over cars. Pure per-file function, no dataset statistic."""
    cols = {}
    n = len(fe)
    for kind, ch in (("vib", rf._VIB), ("shock", rf._SHOCK)):
        blocks = {}
        wl = rf.band_levels_db(fe.spectra[:, ch, :])
        for j in range(0, 18, 2):
            blocks[f"wl{j}"] = wl[:, :, j]
        for i in range(7):
            blocks[f"hz{i}"] = fe.channel_block(f"hz{i}")[:, ch]
        blocks["logrms"] = fe.channel_block("logrms")[:, ch]
        for nm, L in blocks.items():
            L = np.where(np.isfinite(L), L, np.nan)
            rel = L - np.nanmean(L, 1, keepdims=True)
            g = rel.reshape(n, 8, 8)
            car = np.nanmean(g, 2)      # (n, 8 cars)
            pos = np.nanmean(g, 1)      # (n, 8 positions)
            for c in range(8):
                cols[f"{kind}_{nm}_car{c+1}"] = car[:, c]
            for p in range(8):
                cols[f"{kind}_{nm}_pos{p+1}"] = pos[:, p]
    return pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)

X_win = rf.aggregate(feats, WIN)
X_win_m = rf.aggregate(rf.mirror(feats), WIN)
X_cp = car_profile(feats)
X_cp_m = car_profile(rf.mirror(feats))     # mirror keeps car profile, swaps pos parity
speed = feats.scalars["speed_kmh"].to_numpy(float)

def lgbm(seed):
    return LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=7, min_child_samples=5,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                          class_weight="balanced", random_state=seed, verbose=-1, n_jobs=4)

def run(name, X, Xm=None, mirror_mode="swap"):
    """mirror_mode: 'none' | 'swap' (label swapped) | 'keep' (label kept)."""
    swap = {"Normal": "Normal", "Side I": "Side II", "Side II": "Side I"}
    scores, sideI, sideII, per_file = [], [], [], {}
    for seed in (0, 1, 2):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        yhat = np.empty(len(y), dtype=object)
        for tr, te in skf.split(X, y):
            Xtr, ytr = X.iloc[tr], y[tr]
            if Xm is not None and mirror_mode != "none":
                ym = np.array([swap[v] for v in ytr]) if mirror_mode == "swap" else ytr
                Xtr = pd.concat([Xtr, Xm.iloc[tr]], ignore_index=True); ytr = np.r_[ytr, ym]
            ps = np.mean([lgbm(s).fit(Xtr, ytr).predict_proba(X.iloc[te]) for s in (0, 1, 2)], 0)
            pred = CL[ps.argmax(1)]
            pred = np.where(speed[te] < 20, "Normal", pred)
            yhat[te] = pred
        m = macro_f1(list(y), list(yhat))
        scores.append(m if not isinstance(m, dict) else m["macro_f1"])
        from sklearn.metrics import f1_score
        sideI.append(f1_score(y == "Side I", yhat == "Side I")); sideII.append(f1_score(y == "Side II", yhat == "Side II"))
        for f, t, p in zip(feats.file_ids, y, yhat):
            if t != "Normal" and p != t: per_file[f] = per_file.get(f, 0) + 1
    print(f"{name:55s} macroF1 {np.mean(scores):.3f} +- {np.std(scores):.3f} | Side I {np.mean(sideI):.3f} | Side II {np.mean(sideII):.3f} | fault files missed (seed-count): {dict(sorted(per_file.items(), key=lambda kv: -kv[1]))}")

run("winner feats, no mirror", X_win, None, "none")
run("winner feats + mirror(swap)  [~shipped]", X_win, X_win_m, "swap")
run("car-profile only, no mirror", X_cp, None, "none")
run("winner + car-profile, no mirror", pd.concat([X_win, X_cp], axis=1), None, "none")
run("winner + car-profile, mirror(swap)", pd.concat([X_win, X_cp], axis=1), pd.concat([X_win_m, X_cp_m], axis=1), "swap")
run("winner + car-profile, mirror(keep label)", pd.concat([X_win, X_cp], axis=1), pd.concat([X_win_m, X_cp_m], axis=1), "keep")
