"""Small, fold-local corrections to the SHM rainflow m=5 estimate.

This is an exploratory Train-only round.  The existing shipped model is never
overwritten.  Predictions on Test are diagnostic only until a candidate wins
the prespecified LOO, repeated K-fold, and mode checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import HuberRegressor, LassoCV, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nebulax.ps3.shm import MODE_SPLIT, build_dataset
from nebulax.ps3.shm_features import to_log_space


FEATURE_SETS = {
    "skew_alpha": ("st_skew", "sp_alpha1"),
    "skew_alpha_kurt": ("st_skew", "sp_alpha1", "st_kurtosis"),
    "stress_shape": ("st_skew", "sp_alpha1", "st_kurtosis", "rf_mean_stress_mean", "rf_mean_stress_std"),
}
SPECS = (
    ("miner_m5", None, "intercept"),
    ("ridge_skew_alpha", "skew_alpha", "ridge"),
    ("huber_skew_alpha", "skew_alpha", "huber"),
    ("ridge_skew_alpha_kurt", "skew_alpha_kurt", "ridge"),
    ("ridge_stress_shape", "stress_shape", "ridge"),
    ("lasso_m5_offset", None, "lasso"),
)


def fit_predict(train, y_train, held, spec):
    """Fit log(label/rainflow_m5) on train only and predict held rows."""
    _, family, kind = spec
    m5_train = train["rf_logsum_m5"].to_numpy(float)
    m5_held = held["rf_logsum_m5"].to_numpy(float)
    residual = np.log(y_train) - m5_train
    if kind == "intercept":
        correction = np.full(len(held), np.median(residual))
    else:
        if kind == "lasso":
            columns = [c for c in train.columns if c != "file_id" and c != "rf_logsum_m5"]
            x_train = to_log_space(train)[columns].to_numpy(float)
            x_held = to_log_space(held)[columns].to_numpy(float)
            reg = LassoCV(cv=5, n_alphas=60, max_iter=20000, random_state=0)
        else:
            columns = FEATURE_SETS[family]
            x_train = train[list(columns)].to_numpy(float)
            x_held = held[list(columns)].to_numpy(float)
            reg = Ridge(alpha=10.0) if kind == "ridge" else HuberRegressor(alpha=0.1, max_iter=1000)
        model = make_pipeline(StandardScaler(), reg)
        model.fit(x_train, residual)
        correction = model.predict(x_held)
    return np.exp(m5_held + correction)


def evaluate(y, pred):
    ape = np.abs(pred - y) / y
    low = y < MODE_SPLIT
    return {
        "mape": float(ape.mean()),
        "low_mape": float(ape[low].mean()),
        "high_mape": float(ape[~low].mean()),
        "max_ape": float(ape.max()),
    }


def main():
    train, y, test = build_dataset(n_jobs=8)
    n = len(y)
    loo = {name: np.empty(n) for name, *_ in SPECS}
    for i in range(n):
        tr_idx = np.delete(np.arange(n), i)
        for spec in SPECS:
            loo[spec[0]][i] = fit_predict(train.iloc[tr_idx], y[tr_idx], train.iloc[[i]], spec)[0]

    repeated = {name: [] for name, *_ in SPECS}
    for seed in range(5):
        fold_preds = {name: np.empty(n) for name, *_ in SPECS}
        for tr_idx, va_idx in KFold(n_splits=10, shuffle=True, random_state=seed).split(y):
            for spec in SPECS:
                fold_preds[spec[0]][va_idx] = fit_predict(train.iloc[tr_idx], y[tr_idx], train.iloc[va_idx], spec)
        for name in repeated:
            repeated[name].append(evaluate(y, fold_preds[name])["mape"])

    ladder = json.loads(Path("results/ps3/shm_ladder.json").read_text())
    old_loo = ladder["rows"][ladder["winner"]["row_index"]]["loo"]
    payload = {
        "source": "Train only, 64 files; no Test labels",
        "shipped_reference": {
            "loo_mape": old_loo["mape"],
            "repeated_mape": ladder["rows"][ladder["winner"]["row_index"]]["rkf_mape_mean"],
            "organiser_score": 0.971752844618028,
        },
        "rows": [],
    }
    for spec in SPECS:
        name = spec[0]
        test_pred = fit_predict(train, y, test, spec)
        payload["rows"].append({
            "name": name,
            "features": list(FEATURE_SETS[spec[1]]) if spec[1] else (["all_except_rf_m5"] if spec[2] == "lasso" else []),
            "loo": evaluate(y, loo[name]),
            "repeat_mape_mean": float(np.mean(repeated[name])),
            "repeat_mape_sd": float(np.std(repeated[name], ddof=1)),
            "test_predictions": dict(zip(test.file_id, map(float, test_pred))),
            "loo_predictions": dict(zip(train.file_id, map(float, loo[name]))),
        })
    out = Path("results/ps3/shm_physics_residual_round.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("Shipped LOO MAPE", payload["shipped_reference"]["loo_mape"])
    print("Shipped 5x10 MAPE", payload["shipped_reference"]["repeated_mape"])
    for row in payload["rows"]:
        print(row["name"], row["loo"], "5x10", row["repeat_mape_mean"], "+-", row["repeat_mape_sd"])
    print(out)


if __name__ == "__main__":
    main()
