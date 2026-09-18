"""Read-only rail diagnostic: where are the Side I misses and does the side sign agree with the label."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from nebulax.ps3 import rail_features as rf

ROOT = Path("readingmaterials/problem_statement/PS3/02_Datasets/Rail_Corrugation")
labels = pd.read_csv(ROOT / "Train_Labels.csv").set_index("filename")["label"]
paths = sorted((ROOT / "Train").glob("*.csv"), key=lambda p: rf.natural_key(p.name))
feats = rf.load_feature_cache(paths)
fid = feats.file_ids
y = labels.loc[fid].to_numpy()
n = len(fid)

# ---- out-of-fold predictions from the headline scheme (15 folds = 5 x 3 seeds)
j = json.load(open("results/ps3/rail_cv.json"))
oof = {}
for f in j["headline"]["folds"]:
    for r in f["held_predictions"]:
        oof.setdefault(r["file_id"], []).append(r["prediction"])
pred_counts = pd.DataFrame({f: pd.Series(v).value_counts() for f, v in oof.items()}).T.fillna(0).astype(int)
pred_counts = pred_counts.reindex(columns=["Normal", "Side I", "Side II"], fill_value=0)

# ---- per-box short-pitch band level (vib channels), Side I vs Side II
bands = rf.band_levels_db(feats.spectra[:, rf._VIB, :])          # (n, 64, 18)
short = (rf.THIRD_OCTAVE_CENTRES_MM >= 25) & (rf.THIRD_OCTAVE_CENTRES_MM <= 80)
lvl = bands[:, :, short].mean(-1)                                 # (n, 64)
mI, mII = rf.side_mask("I"), rf.side_mask("II")
sideI, sideII = lvl[:, mI], lvl[:, mII]                           # (n, 32) each, paired by MIRROR_PERM
pair_diff = lvl[:, mI] - lvl[:, rf.MIRROR_PERM[mI]]               # box (odd) minus its mirror partner
contrast_med = np.median(sideI, 1) - np.median(sideII, 1)
frac_I_louder = (pair_diff > 0).mean(1)
per_car = lvl.reshape(n, 8, 8)
car_margin = per_car[:, :, ::2].mean(-1) - per_car[:, :, 1::2].mean(-1)   # (n, 8)
# same with broadband log RMS and with the Hz band 3 (~250-500 Hz)
logrms = feats.channel_block("logrms")[:, rf._VIB]
rms_contrast = np.median(logrms[:, mI], 1) - np.median(logrms[:, mII], 1)
hz = np.stack([feats.channel_block(f"hz{i}")[:, rf._VIB] for i in range(7)], -1)  # (n,64,7)
hz_contrast = np.median(hz[:, mI, :], 1) - np.median(hz[:, mII, :], 1)            # (n,7)
peak_lam = feats.channel_block("wl_peak_lambda")[:, rf._VIB]
speed = feats.scalars["speed_kmh"].to_numpy(float)
idx = np.array([int(''.join(ch for ch in f if ch.isdigit())) for f in fid])

df = pd.DataFrame({
    "file": fid, "idx": idx, "label": y, "speed": speed.round(1),
    "lvl_I": np.median(sideI, 1).round(1), "lvl_II": np.median(sideII, 1).round(1),
    "contrast_db": contrast_med.round(2), "frac_I_louder": frac_I_louder.round(2),
    "cars_I_louder": (car_margin > 0).sum(1), "rms_contrast": rms_contrast.round(2),
    "peak_lam_I": np.median(peak_lam[:, mI], 1).round(1), "peak_lam_II": np.median(peak_lam[:, mII], 1).round(1),
}).set_index("file")
df = df.join(pred_counts)
df["misses"] = 3 - df[["Normal", "Side I", "Side II"]].to_numpy()[np.arange(n), pd.Categorical(y, ["Normal", "Side I", "Side II"]).codes]

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print("=== Fault files (sorted by label, then idx) ===")
print(df[df.label != "Normal"].sort_values(["label", "idx"]).to_string())
print("\n=== Side sign check on fault files (contrast_db > 0 means Side I louder in 25-80 mm) ===")
f = df[df.label != "Normal"]
agree = ((f.contrast_db > 0) == (f.label == "Side I"))
print(f"agree: {agree.sum()}/{len(f)}; disagreements:", list(f.index[~agree]))
agree_rms = ((f.rms_contrast > 0) == (f.label == "Side I"))
print(f"broadband RMS sign agree: {agree_rms.sum()}/{len(f)}")
for b in range(7):
    a = ((hz_contrast[df.label != 'Normal', b] > 0) == (f.label == 'Side I').to_numpy())
    print(f"  hz band {b} ({rf.HZ_BAND_EDGES[b]:.0f}-{rf.HZ_BAND_EDGES[b+1]:.0f} Hz) sign agree: {a.sum()}/{len(f)}")

print("\n=== Normal files: contrast distribution ===")
nrm = df[df.label == "Normal"]
print(nrm[["contrast_db", "frac_I_louder", "cars_I_louder"]].describe().round(2).to_string())
print("Normal files with |contrast| > 1.5 dB:", (nrm.contrast_db.abs() > 1.5).sum(), " > 3 dB:", (nrm.contrast_db.abs() > 3).sum())
print("Fault files with |contrast| < 1.5 dB:", (f.contrast_db.abs() < 1.5).sum())

print("\n=== Level comparison: labelled-side short-pitch level, by class ===")
lab_side = np.where(df.label == "Side I", df.lvl_I, df.lvl_II)
df["lvl_lab"] = lab_side
print(df.groupby("label")[["lvl_I", "lvl_II", "speed"]].describe().round(1).T.to_string())

print("\n=== Misses summary ===")
for lab in ("Side I", "Side II"):
    sub = df[df.label == lab]
    print(lab, "files:", len(sub), "| total miss-folds:", int(sub.misses.sum()), "of", 3 * len(sub))
    print(sub.sort_values("misses", ascending=False)[["idx", "speed", "contrast_db", "frac_I_louder", "cars_I_louder", "Normal", "Side I", "Side II", "misses"]].to_string())
print("\nNormal files mis-predicted in >=1 fold:")
print(df[(df.label == "Normal") & (df.misses > 0)][["idx", "speed", "contrast_db", "frac_I_louder", "cars_I_louder", "Normal", "Side I", "Side II"]].to_string())

print("\n=== Filename index blocks of fault files ===")
print("Side I idx:", sorted(df[df.label == "Side I"].idx.tolist()))
print("Side II idx:", sorted(df[df.label == "Side II"].idx.tolist()))

# per-car margins for fault files: is corrugation on every car?
print("\n=== Per-car side margin (dB, Side I - Side II, short-pitch) for fault files ===")
fm = df.label != "Normal"
cm = pd.DataFrame(car_margin[fm].round(1), index=df.index[fm], columns=[f"car{i+1}" for i in range(8)])
cm.insert(0, "label", df.label[fm])
print(cm.sort_values("label").to_string())
df.to_csv("/tmp/claude-1000/-mnt-berstorage-nebulax/d79f5589-9884-4f57-9724-c7f3bdfebef3/scratchpad/w6/diag_rail.csv")
