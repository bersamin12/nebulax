"""Promote coherence after the 23-row nested gate and preserve W7 artefacts.

This writes the root model, CSV and reports.  Rebuilding the four-task
submission and copying the app are separate verification steps.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from nebulax.ps3 import rail
from nebulax.ps3.common import load_model, model_meta, save_model
from nebulax.ps3.submission import expected_ids_for, validate_csv


BACKUP = Path("data/ps3_backups/20260918_score_round_before_rail_coherence")


def main():
    selection = json.loads(Path("results/ps3/rail_coherence_production_selection.json").read_text())
    nested = json.loads(Path("results/ps3/rail_coherence_full_nested.json").read_text())
    stress = json.loads(Path("results/ps3/rail_coherence_production_stress.json").read_text())
    old = json.loads(Path("results/ps3/rail_ladder.json").read_text())
    candidate = Path("results/ps3/rail_coherence_candidate_predictions.csv")
    if selection["macro_f1_mean"] <= 0.8365:
        raise ValueError("coherence did not clear the frozen selection-CV gate")
    if nested["macro_f1_mean"] < 0.7471 or nested["n_candidates"] != 23 or nested["n_folds"] != 15:
        raise ValueError("coherence did not clear the full 23-row nested gate")
    if len(stress) != 2 or [r["scheme"] for r in stress] != ["contiguous", "speed_range"]:
        raise ValueError("coherence stress reports are incomplete")
    if any(row["arm"] == rail.COHERENCE_ROWS[0][0] for row in old["rows"]):
        raise ValueError("coherence is already in the shipped ladder")

    files = [
        "models/ps3/rail.pkl", "models/ps3/rail.json",
        "results/ps3/rail_predictions.csv", "results/ps3/rail_ladder.json",
        "results/ps3/rail_ladder.md", "results/ps3/rail_cv.json", "results/ps3/rail_cv.md",
        "submission/nebulax/rail_predictions.csv", "submission/nebulax/predictions.zip",
        "submission/nebulax/app/models/ps3/rail.pkl", "submission/nebulax/app/models/ps3/rail.json",
    ]
    if BACKUP.exists():
        raise FileExistsError(f"W7 backup already exists: {BACKUP}")
    for name in files:
        target = BACKUP / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, target)

    model = load_model("rail", model_dir="data/ps3_candidates/rail_coherence")
    if not model.opts.get("coherence", False):
        raise ValueError("candidate model is missing coherence features")
    sidecar = rail._model_meta(model, selection, stage="coherence-ladder-winner", honest_report=nested)
    sidecar["prior_organiser_macro_f1"] = 0.7994152046783626
    save_model("rail", model, sidecar)
    csv = rail.predict_test_set(model=model, n_jobs=8, out="results/ps3/rail_predictions.csv")
    if csv.read_bytes() != candidate.read_bytes():
        raise ValueError("promoted rail model does not reproduce candidate Test CSV")
    validate_csv("rail", csv, expected_ids_for("rail")).raise_for_errors()

    arm, kind, _ = rail.COHERENCE_ROWS[0]
    row_index = len(old["rows"])
    new_row = {
        "row_index": row_index, "arm": arm, "model": kind,
        "macro_f1_mean": selection["macro_f1_mean"], "macro_f1_sd": selection["macro_f1_sd"],
        "macro_f1_speed_matched_mean": selection["macro_f1_speed_matched_mean"],
        "class_f1_mean": selection["class_f1_mean"], "n_features": selection["n_features"],
        "fit_seconds_mean": selection["fit_seconds_mean"], "wall_seconds": selection["wall_seconds"],
        "opts": selection["opts"], "tta": selection["tta"], "augment": selection["augment"],
        "boost_repeats": selection["boost_repeats"], "cite": "[R235][R246]", "report": selection,
    }
    old["rows"].append(new_row)
    old["winner"] = {k: new_row[k] for k in ("row_index", "arm", "model", "macro_f1_mean",
                                           "macro_f1_sd", "opts", "tta", "augment", "boost_repeats", "cite")}
    old["nested"] = nested
    old["winner_schemes"] = [
        {k: r[k] for k in ("scheme", "n_folds", "macro_f1_mean", "macro_f1_sd", "class_f1_mean", "confusion")}
        for r in stress
    ]
    old["replaced_artefact"] = True
    old["predictions_csv"] = str(csv)
    old["round_note"] = "Historical 22 rows reused at v3; coherence row added; full 23-row nested CV rerun."
    old["findings"].append(
        f"* **Same-side axle-box coherence**: the extra 21 phase-independent Hz-band features "
        f"raise selection macro F1 from the prior 0.8365 to {selection['macro_f1_mean']:.4f}, "
        f"with Side I F1 {selection['class_f1_mean']['Side I']:.4f}. "
        "They average six same-side box pairs per car across eight cars. "
        "The full 23-row nested estimate and both stress splits are reported above."
    )
    old["wall_seconds"] = round(float(old["wall_seconds"]) + float(selection["wall_seconds"]) + float(nested["wall_seconds"]), 2)
    ladder_path = Path("results/ps3/rail_ladder.json")
    ladder_path.write_text(json.dumps(old, indent=2, default=str) + "\n")
    markdown = rail._ladder_markdown(old).replace(
        "# Rail corrugation - model ladder\n",
        "# Rail corrugation - model ladder\n\n"
        "The unchanged 22 historical rows are reused. One coherence row was added and the "
        "full 23-row nested selection rerun. The prior organiser Rail score was 0.7994152; "
        "the new Test score is unknown.\n",
    )
    Path("results/ps3/rail_ladder.md").write_text(markdown)

    cv_path = Path("results/ps3/rail_cv.json")
    cv = json.loads(cv_path.read_text())
    cv["headline"] = nested
    cv["selection_cv"] = old["winner"]
    cv["winner_selection_status"] = old["winner_selection_status"]
    cv["ladder"] = {k: v for k, v in old.items() if k != "rows"}
    cv["model_meta"] = model_meta("rail")
    cv["predictions_csv"] = str(csv)
    cv["wall_seconds"] = round(float(cv["wall_seconds"]) + float(selection["wall_seconds"]) + float(nested["wall_seconds"]), 2)
    cv_path.write_text(json.dumps(cv, indent=2, default=str) + "\n")
    Path("results/ps3/rail_cv.md").write_text(rail._cv_markdown(cv))
    print("promoted", arm, "nested", nested["macro_f1_mean"],
          "model bytes", Path("models/ps3/rail.pkl").stat().st_size,
          "backup", BACKUP, flush=True)


if __name__ == "__main__":
    main()
