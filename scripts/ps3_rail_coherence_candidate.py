"""Fit a reviewable coherence candidate without changing the shipped W7 model."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nebulax.ps3 import rail
from nebulax.ps3.common import save_model


def main():
    result = json.loads(Path("results/ps3/rail_coherence_round.json").read_text())
    if result["macro_f1_mean"] <= result["reference_selection_f1"]:
        raise ValueError("coherence did not pass the frozen selection gate")
    feats, y = rail._load_training(n_jobs=8)
    kwargs = rail.COHERENCE_ROWS[0][2]
    model = rail.fit_rail(feats, y, seeds=(0, 1, 2), n_jobs=4, **kwargs)
    model.meta["selection_cv_macro_f1"] = result["macro_f1_mean"]
    folder = Path("data/ps3_candidates/rail_coherence")
    pkl = save_model("rail", model, {"stage": "candidate", "opts": model.opts,
                                     "selection_macro_f1_mean": result["macro_f1_mean"],
                                     "selection_side_i_f1_mean": result["side_i_f1_mean"],
                                     "feature_version": model.feature_version}, model_dir=folder)
    csv = rail.predict_test_set(model=model, n_jobs=8,
                                out="results/ps3/rail_coherence_candidate_predictions.csv")
    before = pd.read_csv("submission/nebulax/rail_predictions.csv")
    after = pd.read_csv(csv)
    if before.file_id.tolist() != after.file_id.tolist():
        raise ValueError("candidate Test file order changed")
    changed = after[after.prediction != before.prediction]
    print("model", pkl, pkl.stat().st_size, "bytes; Test changes", len(changed))
    print(changed.to_string(index=False))


if __name__ == "__main__":
    main()
