"""Independent regression checks for the Rail verifier fixes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nebulax.ps3.common import load_model
from nebulax.ps3.rail import RailTask
from nebulax.ps3.scoring import macro_f1

ROOT = Path(__file__).resolve().parents[1]


def test_committed_headline_is_nested_and_recomputes_from_held_predictions():
    payload = json.loads((ROOT / "results/ps3/rail_cv.json").read_text(encoding="utf-8"))
    assert payload["headline"]["scheme"].startswith("nested stratified grouped")
    assert "post-hoc" in payload["winner_selection_status"]
    scores = []
    for fold in payload["headline"]["folds"]:
        held = fold["held_predictions"]
        score = macro_f1([row["truth"] for row in held], [row["prediction"] for row in held])
        assert score == fold["macro_f1"]
        scores.append(score)
    assert float(np.mean(scores)) == payload["headline"]["macro_f1_mean"]


def test_committed_cv_contains_all_frozen_stress_splits_and_speed_ablations():
    payload = json.loads((ROOT / "results/ps3/rail_cv.json").read_text(encoding="utf-8"))
    assert [row["scheme"] for row in payload["schemes"]] == ["stratified", "contiguous", "speed_range"]
    assert all(row["macro_f1_speed_matched_mean"] is not None for row in payload["schemes"])
    arms = {row["arm"] for row in payload["ablations"]}
    assert {"no speed features", "no low-speed rule"} <= arms


def test_duplicate_pairs_never_cross_any_committed_split():
    payload = json.loads((ROOT / "results/ps3/rail_cv.json").read_text(encoding="utf-8"))
    pairs = [set(group["files"]) for group in payload["duplicates"]]
    assert pairs and all(group["byte_identical"] for group in payload["duplicates"])
    for scheme in payload["schemes"]:
        for fold in scheme["folds"]:
            held = {row["file_id"] for row in fold["held_predictions"]}
            assert all(not (0 < len(pair & held) < len(pair)) for pair in pairs), scheme["scheme"]


def test_real_artefact_loads_fresh_and_reproduces_committed_csv(tmp_path):
    assert (ROOT / "models/ps3/rail.pkl").stat().st_size < 5_000_000
    out = tmp_path / "rail_predictions.csv"
    code = (
        "from pathlib import Path; "
        "from nebulax.ps3.rail import predict_test_set; "
        f"predict_test_set(out=Path({str(out)!r}), n_jobs=4)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    assert out.read_bytes() == (ROOT / "results/ps3/rail_predictions.csv").read_bytes()


def test_constant_speed_thousand_sample_file_has_valid_explanation(tmp_path):
    n = 1000
    time = np.arange(n, dtype=float)
    pulse = ((time // 10) % 2).astype(float)
    channels = np.stack([0.01 * np.sin(2 * np.pi * time / (20 + j % 17)) for j in range(128)], axis=1)
    arr = np.column_stack([pulse, channels])
    path = tmp_path / "constant_speed_1000.csv"
    pd.DataFrame(arr, columns=[f"c{i}" for i in range(129)]).to_csv(path, index=False)
    task = RailTask()
    result = task.run(path, load_model("rail"))
    payload = task.explain(result).as_dict()
    assert result.rows[0]["prediction"] in ("Normal", "Side I", "Side II")
    assert len(payload["trace"]["x"]) == len(payload["trace"]["y"]) <= 2000
    assert all(np.isfinite(list(payload["numbers"].values())))
