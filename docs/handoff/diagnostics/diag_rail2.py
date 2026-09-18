"""Where does the side information live? Per-band, per-box, with a per-box gain baseline from Normal files."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from nebulax.ps3 import rail_features as rf

ROOT = Path("readingmaterials/problem_statement/PS3/02_Datasets/Rail_Corrugation")
labels = pd.read_csv(ROOT / "Train_Labels.csv").set_index("filename")["label"]
paths = sorted((ROOT / "Train").glob("*.csv"), key=lambda p: rf.natural_key(p.name))
feats = rf.load_feature_cache(paths)
y = labels.loc[feats.file_ids].to_numpy()
n = len(y)
isN, isI, isII = y == "Normal", y == "Side I", y == "Side II"
isF = ~isN
mI, mII = rf.side_mask("I"), rf.side_mask("II")
speed = feats.scalars["speed_kmh"].to_numpy(float)

def blocks(kind):
    ch = rf._VIB if kind == "vib" else rf._SHOCK
    out = {}
    wl = rf.band_levels_db(feats.spectra[:, ch, :])  # (n,64,18)
    for j in range(18):
        out[f"wl{j}({rf.THIRD_OCTAVE_CENTRES_MM[j]:.0f}mm)"] = wl[:, :, j]
    for i in range(7):
        out[f"hz{i}({rf.HZ_BAND_EDGES[i]:.0f}-{rf.HZ_BAND_EDGES[i+1]:.0f})"] = feats.channel_block(f"hz{i}")[:, ch]
    for nm in ("logrms", "logpeak", "kurtosis", "crest", "env_duty", "wl_peak_prom", "wl_flatness"):
        out[nm] = feats.channel_block(nm)[:, ch]
    return out

def safe_auc(mask_pos, mask_neg, s):
    yy = np.r_[np.ones(mask_pos.sum()), np.zeros(mask_neg.sum())]
    ss = np.r_[s[mask_pos], s[mask_neg]]
    if np.all(np.isfinite(ss)) and len(set(yy)) == 2:
        return roc_auc_score(yy, ss)
    return np.nan

for kind in ("vib", "shock"):
    print(f"\n################ {kind} ################")
    rows = []
    B = blocks(kind)
    for name, L in B.items():
        L = np.where(np.isfinite(L), L, np.nan)
        rel = L - np.nanmean(L, axis=1, keepdims=True)          # remove file-level level (speed etc.)
        base = np.nanmedian(rel[isN], axis=0)                    # per-box gain/geometry baseline from Normals
        corr = rel - base
        raw_con = np.nanmedian(L[:, mI], 1) - np.nanmedian(L[:, mII], 1)
        cor_con = np.nanmedian(corr[:, mI], 1) - np.nanmedian(corr[:, mII], 1)
        # side discrimination among fault files
        auc_raw = safe_auc(isI, isII, raw_con)
        auc_cor = safe_auc(isI, isII, cor_con)
        sign_raw = ((raw_con[isF] > 0) == isI[isF]).mean()
        sign_cor = ((cor_con[isF] > 0) == isI[isF]).mean()
        # fault vs normal, speed-matched (v>=35) using file max and median level
        sm = speed >= 35
        auc_fn_max = safe_auc(isF & sm, isN & sm, np.nanmax(L, 1))
        auc_fn_med = safe_auc(isF & sm, isN & sm, np.nanmedian(L, 1))
        auc_fn_abscon = safe_auc(isF & sm, isN & sm, np.abs(cor_con))
        rows.append((name, round(sign_raw, 2), round(auc_raw, 2), round(sign_cor, 2), round(auc_cor, 2), round(auc_fn_med, 2), round(auc_fn_max, 2), round(auc_fn_abscon, 2)))
    df = pd.DataFrame(rows, columns=["feature", "side_sign_raw", "side_auc_raw", "side_sign_gaincorr", "side_auc_gaincorr", "FvN_auc_med(v>=35)", "FvN_auc_max(v>=35)", "FvN_auc_|corr_contrast|"])
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))

# per-box pattern in the best few bands: t-stat Side I vs Side II of gain-corrected relative level, 8x8 grid
print("\n=== Per-box Side I minus Side II mean gain-corrected relative level (dB), vib, rows=car, cols=position ===")
B = blocks("vib")
for name in ("wl9(64mm)", "wl11(102mm)", "wl13(161mm)", "wl15(256mm)", "logrms", "hz3(213-469)"):
    L = B[name]; rel = L - L.mean(1, keepdims=True); corr = rel - np.median(rel[isN], 0)
    d = corr[isI].mean(0) - corr[isII].mean(0)
    sd = np.sqrt(corr[isI].var(0) / isI.sum() + corr[isII].var(0) / isII.sum())
    print(f"\n{name}: diff (dB) / t")
    print(pd.DataFrame(d.reshape(8, 8).round(1), index=[f"car{i+1}" for i in range(8)], columns=[f"p{i+1}" for i in range(8)]).to_string())
    print(pd.DataFrame((d / sd).reshape(8, 8).round(1), index=[f"car{i+1}" for i in range(8)], columns=[f"p{i+1}" for i in range(8)]).to_string())

# fault vs normal per box: which boxes carry corrugation at all (speed-matched, wl9)
print("\n=== Per-box Fault minus Normal (v>=35) mean level, vib wl9(64mm) and wl13(161mm) ===")
sm = speed >= 35
for name in ("wl9(64mm)", "wl13(161mm)", "logrms"):
    L = B[name]
    d = L[isF & sm].mean(0) - L[isN & sm].mean(0)
    print(name); print(pd.DataFrame(d.reshape(8, 8).round(1), index=[f"car{i+1}" for i in range(8)], columns=[f"p{i+1}" for i in range(8)]).to_string())

# Speed overlap: how many Normal files at each speed decile vs faults
print("\n=== Speed histogram (files) ===")
bins = [0, 20, 35, 42, 46, 50, 55, 60, 65, 71]
h = pd.DataFrame({lab: np.histogram(speed[y == lab], bins)[0] for lab in ("Normal", "Side I", "Side II")}, index=[f"{a}-{b}" for a, b in zip(bins[:-1], bins[1:])])
print(h.to_string())
