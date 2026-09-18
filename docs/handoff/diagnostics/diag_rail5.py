"""Representation ensemble + two-stage check, same disposable 5-fold x 3 seeds harness."""
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
feats = rf.load_feature_cache(paths); mfeats = rf.mirror(feats)
y = labels.loc[feats.file_ids].to_numpy(); n = len(y)
CL = np.array(["Normal", "Side I", "Side II"]); speed = feats.scalars["speed_kmh"].to_numpy(float)
swap = {"Normal": "Normal", "Side I": "Side II", "Side II": "Side I"}
def lgbm(seed, **kw):
    p = dict(n_estimators=250, learning_rate=0.05, num_leaves=7, max_depth=4, min_child_samples=5, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, class_weight="balanced", random_state=seed, verbose=-1, n_jobs=4)
    p.update(kw); return LGBMClassifier(**p)
REPS = {
    "hz (winner)": {"wavelength": False, "hz": True, "v2_normalise": False},
    "wl no-v2": {"v2_normalise": False},
    "hz+wl no-v2": {"hz": True, "v2_normalise": False},
    "wl v2 (baseline)": {},
}
X = {k: rf.aggregate(feats, o) for k, o in REPS.items()}; Xm = {k: rf.aggregate(mfeats, o) for k, o in REPS.items()}
def oof(rep, tta=False, seeds=(0, 1, 2)):
    """(3, n, 3) OOF probabilities per seed with mirror-swap augmentation."""
    out = np.zeros((len(seeds), n, 3))
    for si, seed in enumerate(seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X[rep], y):
            Xtr = pd.concat([X[rep].iloc[tr], Xm[rep].iloc[tr]], ignore_index=True); ytr = np.r_[y[tr], [swap[v] for v in y[tr]]]
            ms = [lgbm(s).fit(Xtr, ytr) for s in (0, 1, 2)]
            P = np.mean([m.predict_proba(X[rep].iloc[te]) for m in ms], 0)
            if tta:
                Pm = np.mean([m.predict_proba(Xm[rep].iloc[te]) for m in ms], 0)[:, [0, 2, 1]]
                P = 0.5 * (P + Pm)
            out[si, te] = P
    return out
def report(name, P):
    sc, sI, sII = [], [], []
    for si in range(P.shape[0]):
        pred = CL[P[si].argmax(1)]; pred = np.where(speed < 20, "Normal", pred)
        sc.append(macro_f1(list(y), list(pred))); sI.append(f1_score(y == "Side I", pred == "Side I")); sII.append(f1_score(y == "Side II", pred == "Side II"))
    print(f"{name:40s} macroF1 {np.mean(sc):.3f} +- {np.std(sc):.3f} | Side I {np.mean(sI):.3f} | Side II {np.mean(sII):.3f}", flush=True)
P = {}
for rep in REPS:
    P[rep] = oof(rep); report(rep, P[rep])
P["hz (winner) +TTA"] = oof("hz (winner)", tta=True); report("hz (winner) +TTA", P["hz (winner) +TTA"])
report("ens: hz + wl no-v2", (P["hz (winner)"] + P["wl no-v2"]) / 2)
report("ens: hz + wl no-v2 + hz+wl", (P["hz (winner)"] + P["wl no-v2"] + P["hz+wl no-v2"]) / 3)
report("ens: all four", sum(P[r] for r in REPS) / 4)
report("ens: hz+TTA + wl no-v2", (P["hz (winner) +TTA"] + P["wl no-v2"]) / 2)
# geometric mean ensemble
report("geo-ens: hz + wl no-v2", np.exp((np.log(P["hz (winner)"] + 1e-6) + np.log(P["wl no-v2"] + 1e-6)) / 2))
# two-stage: binary fault/normal with mirror kept, then side among faults with mirror-swap
def two_stage(rep, seeds=(0, 1, 2)):
    out = np.zeros((len(seeds), n, 3))
    yb = (y != "Normal").astype(int)
    for si, seed in enumerate(seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X[rep], y):
            Xtr = pd.concat([X[rep].iloc[tr], Xm[rep].iloc[tr]], ignore_index=True)
            pb = np.mean([lgbm(s).fit(Xtr, np.r_[yb[tr], yb[tr]]).predict_proba(X[rep].iloc[te])[:, 1] for s in (0, 1, 2)], 0)
            f = tr[y[tr] != "Normal"]
            Xf = pd.concat([X[rep].iloc[f], Xm[rep].iloc[f]], ignore_index=True); yf = np.r_[y[f], [swap[v] for v in y[f]]]
            ps = np.mean([lgbm(s).fit(Xf, (yf == "Side I").astype(int)).predict_proba(X[rep].iloc[te])[:, 1] for s in (0, 1, 2)], 0)
            out[si, te, 0] = 1 - pb; out[si, te, 1] = pb * ps; out[si, te, 2] = pb * (1 - ps)
    return out
P2 = two_stage("hz (winner)"); report("two-stage hz (binary mirror-kept)", P2)
report("ens: two-stage + flat hz", (P2 + P["hz (winner)"]) / 2)
