"""Official-scheme Train-only audit of the positive-skew SHM physics blend.

The historical 39 ladder rows are reused byte-for-byte because their feature
version and model code did not change.  The new row is evaluated on the same
LOO and repeated folds, then the six best rows are selected anew inside LOO.
No shipped artefact is written by this script.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

from nebulax.ps3 import shm


def main():
    start = time.time()
    frame, y, _ = shm.build_dataset(n_jobs=8)
    spec = dict(shm.LADDER_SPECS[-1])
    loo = shm.loo_cv(frame, y, spec)
    repeated = shm.repeated_kfold_cv(frame, y, spec)
    print("new row", shm.spec_id(spec), "LOO", loo["mape"],
          "5x10", repeated["mape_mean"], flush=True)
    previous = json.loads(Path("results/ps3/shm_ladder.json").read_text())
    rows = list(previous["rows"])
    if any(r.get("spec_id") == shm.spec_id(spec) for r in rows):
        raise ValueError("new SHM gate row is already in the historical ladder")
    new = {
        "spec_id": shm.spec_id(spec), "spec": spec,
        "n_features": len(shm.spec_columns(spec, frame.columns)),
        "loo_mape": loo["mape"], "loo_score": loo["score"],
        "loo_mape_low": loo["mape_low_mode"], "loo_mape_high": loo["mape_high_mode"],
        "rkf_mape_mean": repeated["mape_mean"], "rkf_mape_sd": repeated["mape_sd"],
        "rkf_score": repeated["score_mean"], "fit_seconds": loo["seconds"] + repeated["seconds"],
        "loo": loo,
    }
    rows.append(new)
    ordered = sorted((r for r in rows if "error" not in r), key=lambda r: r["rkf_mape_mean"])
    shortlist = [r["spec"] for r in ordered[:6]]
    print("shortlist", [shm.spec_id(s) for s in shortlist], flush=True)
    nested = shm.nested_cv(frame, y, shortlist)
    print("nested", nested["mape"], nested["chosen_counts"], flush=True)
    payload = {
        "source": "Train only; same feature version, reused historical 39 rows; no Test labels",
        "prior_nested_mape": previous["nested"]["mape"],
        "prior_selection_mape": previous["rows"][previous["winner"]["row_index"]]["rkf_mape_mean"],
        "new_row": new,
        "shortlist": [shm.spec_id(s) for s in shortlist],
        "nested": nested,
        "winner_spec_id": ordered[0]["spec_id"],
        "seconds": time.time() - start,
    }
    out = Path("results/ps3/shm_gate_audit.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(out, flush=True)


if __name__ == "__main__":
    main()
