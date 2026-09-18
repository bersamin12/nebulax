"""Independent regression checks for the Door verifier fixes."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nebulax.ps3.common import dataset_dir, load_model, parse_door_timestamp
from nebulax.ps3.door import DoorClassifier, DoorTask, _eligible_ladder_row, _pick_winner
from nebulax.ps3.scoring import iou_f1

ROOT = Path(__file__).resolve().parents[1]


def test_classifier_contains_one_fit_implementation():
    assert inspect.getsource(DoorClassifier).count("def fit(") == 1


def test_transductive_and_diagnostic_rows_cannot_win_deployment_selection():
    leaked = {"name": "TRANSDUCTIVE", "iou_f1_mean": 1.0, "n_features": 1, "baseline_mode": "batch"}
    diagnostic = {"name": "stump", "iou_f1_mean": 1.0, "n_features": 1, "diagnostic": True}
    honest = {"name": "honest", "iou_f1_mean": 0.8, "n_features": 2, "baseline_mode": "fold"}
    assert not _eligible_ladder_row(leaked)
    assert not _eligible_ladder_row(diagnostic)
    assert _pick_winner([leaked, diagnostic, honest]) is honest


def test_committed_headline_is_nested_and_recomputes_from_held_predictions():
    payload = json.loads((ROOT / "results/ps3/door_cv.json").read_text(encoding="utf-8"))
    assert payload["headline"]["scheme"].startswith("nested model selection")
    assert "post-hoc" in payload["winner_selection_status"]
    scores = []
    for fold in payload["headline"]["folds"]:
        held = fold["held_predictions"]
        score = iou_f1(held["truth"], held["predicted"])["score"]
        assert score == fold["iou_f1"]
        scores.append(score)
    assert float(np.mean(scores)) == payload["headline"]["iou_f1_mean"]


def test_real_artefact_loads_fresh_and_reproduces_committed_csv(tmp_path):
    assert (ROOT / "models/ps3/door.pkl").stat().st_size < 5_000_000
    out = tmp_path / "door_predictions.csv"
    code = (
        "from pathlib import Path; "
        "from nebulax.ps3.door import predict_stream, write_predictions; "
        f"write_predictions(predict_stream(Path({str(dataset_dir('door') / 'Test.csv')!r})), Path({str(out)!r}))"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    assert out.read_bytes() == (ROOT / "results/ps3/door_predictions.csv").read_bytes()


def test_nan_contaminated_thousand_sample_stream_has_valid_explanation(tmp_path):
    source = pd.read_csv(dataset_dir("door") / "Test.csv").iloc[:1000].copy()
    source.iloc[10, 1] = np.nan
    source.iloc[20, 2] = np.nan
    path = tmp_path / "door_1000.csv"
    source.to_csv(path, index=False)
    task = DoorTask()
    result = task.run(path, load_model("door"))
    payload = task.explain(result).as_dict()
    assert result.rows
    assert len(payload["trace"]["x"]) == len(payload["trace"]["y"]) <= 2000
    starts = [parse_door_timestamp(row["start_time"]) for row in result.rows]
    ends = [parse_door_timestamp(row["end_time"]) for row in result.rows]
    assert all(a >= b for a, b in zip(starts[1:], ends[:-1]))
