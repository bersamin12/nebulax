"""Promote the audited SHM physics gate and merge its row into the ladder.

The previous shipped artefacts should be backed up before running this script.
It writes the root model and results only; submission packaging is a separate
validated step so all four task CSVs are regenerated together.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nebulax.ps3 import shm
from nebulax.ps3.common import save_model


def main():
    audit = json.loads(Path("results/ps3/shm_gate_audit.json").read_text())
    old = json.loads(Path("results/ps3/shm_ladder.json").read_text())
    probe = json.loads(Path("results/ps3/shm_skew_gate_round.json").read_text())
    new = audit["new_row"]
    if audit["winner_spec_id"] != new["spec_id"]:
        raise ValueError("SHM physics gate did not win repeated selection CV")
    if new["rkf_mape_mean"] >= audit["prior_selection_mape"] or audit["nested"]["mape"] >= audit["prior_nested_mape"]:
        raise ValueError("SHM physics gate did not improve both selection and nested CV")
    if any(row.get("spec_id") == new["spec_id"] for row in old["rows"]):
        raise ValueError("SHM gate is already in the ladder")

    frame, y, test = shm.build_dataset(n_jobs=8)
    model = shm.fit_model(frame, y, new["spec"])
    proposed = model.predict(test)
    expected = np.asarray([probe["test_prediction_diagnostic"][fid]["gate"] for fid in test.file_id], dtype=float)
    if not np.allclose(proposed, expected, atol=1e-12, rtol=1e-12):
        raise ValueError("production SHM gate does not reproduce the independently measured candidate")
    model.cv = {"median_damage": float(np.median(y)), "mape": float(new["loo_mape"]),
                "score": float(new["loo_score"]), "nested_mape": float(audit["nested"]["mape"]),
                "scheme": "repeated 5x10-fold selection / nested LOO", "stage": "ladder"}
    save_model("shm", model, {"stage": "ladder", "spec": new["spec"],
                              "loo_mape": new["loo_mape"], "rkf_mape_mean": new["rkf_mape_mean"],
                              "nested_loo_mape": audit["nested"]["mape"],
                              "feature_version": shm.sfeat.FEATURE_VERSION})
    csv = shm._predictions_csv(model, test, Path("results/ps3/shm_predictions.csv"))

    old["rows"].append(new)
    old["n_rows"] = len(old["rows"])
    old["winner"] = {"spec_id": new["spec_id"], "spec": new["spec"], "n_features": model.n_features}
    old["nested"] = audit["nested"]
    old["winner_selection_status"] = "post-hoc row ranking; the headline is nested selection, not the winning row's repeated-CV score"
    old["round_note"] = "Historical 39 rows reused at shm-f2; one physics-gate row added; six-row nested LOO rerun."
    old["predictions_csv"] = str(csv)
    ladder_path = Path("results/ps3/shm_ladder.json")
    ladder_path.write_text(json.dumps(old, indent=2, default=str) + "\n")
    markdown = shm._ladder_markdown(old)
    markdown = markdown.replace("# SHM ladder - cumulative fatigue damage\n",
                                "# SHM ladder - cumulative fatigue damage\n\n"
                                "This round reuses the unchanged 39 historical shm-f2 rows, adds one "
                                "positive-skew physics blend, and reruns six-row nested LOO selection. "
                                "The organiser's prior SHM score was 0.971752844618028; the new Test score is unknown.\n")
    Path("results/ps3/shm_ladder.md").write_text(markdown)
    print(new["spec_id"], "selection MAPE", new["rkf_mape_mean"],
          "nested MAPE", audit["nested"]["mape"], "model bytes", Path("models/ps3/shm.pkl").stat().st_size,
          "csv", csv, flush=True)


if __name__ == "__main__":
    main()
