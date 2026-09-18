"""Explore a fixed rainflow blend for positively skewed SHM recordings.

The rule uses only a file's own skew and fold-trained models.  It was suggested
by exploratory LOO diagnostics, so repeated CV here is supporting evidence,
not an independent confirmatory estimate.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import KFold

from nebulax.ps3 import shm


def predictions(train, y_train, held, spec):
    model = shm.fit_model(train, y_train, spec)
    lasso = model.predict(held)
    intercept = np.median(np.log(y_train) - train["rf_logsum_m5"].to_numpy(float))
    physics = np.exp(held["rf_logsum_m5"].to_numpy(float) + intercept)
    blend = lasso.copy()
    positive = held["st_skew"].to_numpy(float) > 0
    blend[positive] = 0.5 * (lasso[positive] + physics[positive])
    return lasso, blend


def metrics(y, pred):
    ape = np.abs(pred - y) / y
    low = y < shm.MODE_SPLIT
    return {"mape": float(ape.mean()),
            "low_mape": float(ape[low].mean()) if low.any() else None,
            "high_mape": float(ape[~low].mean()) if (~low).any() else None,
            "max_ape": float(ape.max())}


def main():
    t0 = time.time()
    train, y, test = shm.build_dataset(n_jobs=8)
    spec = json.loads(Path("models/ps3/shm.json").read_text())["spec"]
    n = len(y)
    loo_base = np.empty(n)
    loo_gate = np.empty(n)
    for i in range(n):
        tr = np.delete(np.arange(n), i)
        loo_base[i], loo_gate[i] = (v[0] for v in predictions(train.iloc[tr], y[tr], train.iloc[[i]], spec))
    repeats = []
    for seed in range(5):
        base = np.empty(n)
        gate = np.empty(n)
        for tr, va in KFold(n_splits=10, shuffle=True, random_state=seed).split(y):
            base[va], gate[va] = predictions(train.iloc[tr], y[tr], train.iloc[va], spec)
        repeats.append({"seed": seed, "base": metrics(y, base), "gate": metrics(y, gate)})
        print(seed, repeats[-1]["base"]["mape"], repeats[-1]["gate"]["mape"], flush=True)
    # Test-sized held-outs: inner folds choose whether to use the gate without
    # inspecting an outer validation label. This audits selection instability.
    nested_repeats = []
    for seed in range(5):
        base_outer = np.empty(n)
        gate_outer = np.empty(n)
        selected_outer = np.empty(n)
        choices = []
        for tr, va in KFold(n_splits=4, shuffle=True, random_state=100 + seed).split(y):
            inner_base = np.empty(len(tr))
            inner_gate = np.empty(len(tr))
            for itr, iva in KFold(n_splits=5, shuffle=True, random_state=200 + seed).split(tr):
                inner_base[iva], inner_gate[iva] = predictions(
                    train.iloc[tr[itr]], y[tr[itr]], train.iloc[tr[iva]], spec
                )
            use_gate = metrics(y[tr], inner_gate)["mape"] < metrics(y[tr], inner_base)["mape"]
            choices.append(bool(use_gate))
            base_outer[va], gate_outer[va] = predictions(train.iloc[tr], y[tr], train.iloc[va], spec)
            selected_outer[va] = gate_outer[va] if use_gate else base_outer[va]
        nested_repeats.append({"seed": seed, "base": metrics(y, base_outer),
                               "gate": metrics(y, gate_outer),
                               "selected": metrics(y, selected_outer),
                               "selected_gate_folds": int(sum(choices))})
        print("nested", seed, nested_repeats[-1]["base"]["mape"],
              nested_repeats[-1]["gate"]["mape"], nested_repeats[-1]["selected"]["mape"], flush=True)
    stresses = []
    for feature in ("st_skew", "sp_centroid"):
        order = np.argsort(train[feature].to_numpy(float))
        base = np.empty(n)
        gate = np.empty(n)
        per_quartile = []
        for q, va in enumerate(np.array_split(order, 4)):
            tr = np.setdiff1d(np.arange(n), va)
            base[va], gate[va] = predictions(train.iloc[tr], y[tr], train.iloc[va], spec)
            per_quartile.append({"quartile": q, "base": metrics(y[va], base[va]),
                                 "gate": metrics(y[va], gate[va])})
        stresses.append({"held_out_feature": feature, "base": metrics(y, base),
                         "gate": metrics(y, gate), "quartiles": per_quartile})
    base_test, gate_test = predictions(train, y, test, spec)
    payload = {
        "source": "Train-only labels; positive-skew gate chosen after exploratory LOO comparison",
        "rule": "if st_skew > 0, average fold-trained Lasso and fold-calibrated m=5 rainflow equally",
        "loo_base": metrics(y, loo_base), "loo_gate": metrics(y, loo_gate),
        "repeat_base_mean": float(np.mean([r["base"]["mape"] for r in repeats])),
        "repeat_gate_mean": float(np.mean([r["gate"]["mape"] for r in repeats])),
        "repeat_rows": repeats,
        "nested_4fold_rows": nested_repeats,
        "nested_4fold_base_mean": float(np.mean([r["base"]["mape"] for r in nested_repeats])),
        "nested_4fold_gate_mean": float(np.mean([r["gate"]["mape"] for r in nested_repeats])),
        "nested_4fold_selected_mean": float(np.mean([r["selected"]["mape"] for r in nested_repeats])),
        "stress_splits": stresses,
        "test_prediction_diagnostic": {str(fid): {"base": float(b), "gate": float(g)}
                                       for fid, b, g in zip(test.file_id, base_test, gate_test)},
        "seconds": time.time() - t0,
    }
    out = Path("results/ps3/shm_skew_gate_round.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(out, payload["loo_base"], payload["loo_gate"], payload["repeat_base_mean"], payload["repeat_gate_mean"])


if __name__ == "__main__":
    main()
