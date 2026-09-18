"""Quick CV of (a) a per-box multiple-instance model and (b) OOF probabilities of the hard files under the shipped-like config."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
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
n = len(y)
CL = np.array(["Normal", "Side I", "Side II"])
speed = feats.scalars["speed_kmh"].to_numpy(float)
swap = {"Normal": "Normal", "Side I": "Side II", "Side II": "Side I"}

def lgbm(seed, **kw):
    p = dict(n_estimators=250, learning_rate=0.05, num_leaves=7, max_depth=4, min_child_samples=5, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, class_weight="balanced", random_state=seed, verbose=-1, n_jobs=4)
    p.update(kw); return LGBMClassifier(**p)

# ---------- (a) per-box instances: 64 rows per file ----------
def instances(fe: rf.RailFeatures, with_parity=True, with_car=False):
    m = len(fe)
    cols = {}
    for kind, ch in (("vib", rf._VIB), ("shock", rf._SHOCK)):
        wl = rf.band_levels_db(fe.spectra[:, ch, :])  # (m,64,18)
        for j in range(18): cols[f"{kind}_wl{j}"] = wl[:, :, j]
        for nm in rf.CHANNEL_SCALARS: cols[f"{kind}_{nm}"] = fe.channel_block(nm)[:, ch]
        # box level relative to the file's own mean over 64 boxes (a within-file, per-file quantity)
        lr = fe.channel_block("logrms")[:, ch]; cols[f"{kind}_logrms_rel"] = lr - np.nanmean(lr, 1, keepdims=True)
    sp = fe.scalars["speed_kmh"].to_numpy(float)
    cols["speed"] = np.repeat(sp[:, None], 64, 1); cols["log_speed"] = np.log(np.maximum(cols["speed"], 0.1))
    if with_parity: cols["side_I"] = np.repeat((np.arange(64) % 2 == 0)[None, :].astype(float), m, 0)
    if with_car: cols["car"] = np.repeat((np.arange(64) // 8)[None, :].astype(float), m, 0)
    X = pd.DataFrame({k: v.reshape(-1) for k, v in cols.items()}).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X

def run_mil(name, agg="mean", with_parity=True, with_car=False, mirror=True, seeds=(0, 1, 2)):
    X = instances(feats, with_parity, with_car); Xm = instances(rf.mirror(feats), with_parity, with_car)
    file_of = np.repeat(np.arange(n), 64)
    scores, sI, sII, missed = [], [], [], {}
    for seed in seeds:
        yhat = np.empty(n, dtype=object)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y):
            trm = np.isin(file_of, tr); tem = np.isin(file_of, te)
            Xtr = X[trm]; ytr = np.repeat(y[tr], 64)
            if mirror:
                Xtr = pd.concat([Xtr, Xm[trm]], ignore_index=True); ytr = np.r_[ytr, np.repeat([swap[v] for v in y[tr]], 64)]
            P = np.mean([lgbm(s, colsample_bytree=0.5).fit(Xtr, ytr).predict_proba(X[tem]) for s in (0, 1, 2)], 0)  # (len(te)*64, 3)
            P = P.reshape(len(te), 64, 3)
            pf = P.mean(1) if agg == "mean" else np.log(P + 1e-6).mean(1)
            pred = CL[pf.argmax(1)]; pred = np.where(speed[te] < 20, "Normal", pred)
            yhat[te] = pred
        scores.append(macro_f1(list(y), list(yhat)))
        sI.append(f1_score(y == "Side I", yhat == "Side I")); sII.append(f1_score(y == "Side II", yhat == "Side II"))
        for f, t, p in zip(feats.file_ids, y, yhat):
            if t != "Normal" and p != t: missed[f] = missed.get(f, 0) + 1
    print(f"MIL {name:45s} macroF1 {np.mean(scores):.3f} +- {np.std(scores):.3f} | Side I {np.mean(sI):.3f} | Side II {np.mean(sII):.3f} | missed: {dict(sorted(missed.items(), key=lambda kv: -kv[1]))}", flush=True)

run_mil("parity, mirror-swap, mean-prob", with_parity=True, mirror=True)
run_mil("parity+car, mirror-swap, mean-prob", with_parity=True, with_car=True, mirror=True)
run_mil("parity, mirror-swap, geo-mean", agg="geo", with_parity=True, mirror=True)
run_mil("no parity, no mirror (side-agnostic)", with_parity=False, mirror=False)

# ---------- (b) OOF probabilities under the shipped-like file-level config ----------
WIN = {"wavelength": False, "hz": True, "v2_normalise": False}
X = rf.aggregate(feats, WIN); Xm = rf.aggregate(rf.mirror(feats), WIN)
P_oof = np.zeros((n, 3)); cnt = 0
for seed in (0, 1, 2):
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        Xtr = pd.concat([X.iloc[tr], Xm.iloc[tr]], ignore_index=True); ytr = np.r_[y[tr], [swap[v] for v in y[tr]]]
        P_oof[te] += np.mean([lgbm(s).fit(Xtr, ytr).predict_proba(X.iloc[te]) for s in (0, 1, 2)], 0)
P_oof /= 3
df = pd.DataFrame(P_oof.round(3), columns=CL, index=feats.file_ids); df.insert(0, "speed", speed.round(1)); df.insert(0, "label", y)
print("\nOOF mean probabilities (file-level shipped-like config, mirror-swap), fault files:")
print(df[y != "Normal"].sort_values(["label", "Side I"]).to_string())
print("\nNormal files with p(Normal) < 0.6:")
print(df[(y == "Normal") & (df["Normal"] < 0.6)].to_string())
# how would prior boosts move macro F1? sweep a Side I multiplier and Side II multiplier on the OOF probabilities
print("\nBoost sweep on OOF probabilities (Side I x, Side II x) -> macro F1:")
for bI in (1, 1.5, 2, 3, 4, 6):
    row = []
    for bII in (1, 1.5, 2, 3):
        q = P_oof * np.array([1, bI, bII]); pred = CL[q.argmax(1)]; pred = np.where(speed < 20, "Normal", pred)
        row.append(f"{macro_f1(list(y), list(pred)):.3f}")
    print(f"  Side I x{bI}: " + "  ".join(row))
