"""PS3 leaderboard assembly on tiny fabricated ladder files."""

from __future__ import annotations

import json

from nebulax.ps3 import report


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixtures(tmp_path):
    common = {"seeds": [7], "skipped": ["expensive row [R999]"]}
    _write(
        tmp_path / "door_ladder.json",
        {
            **common,
            "task": "door",
            "scheme": "blocks",
            "rows": [{"name": "door row", "model": "logreg", "iou_f1_mean": 0.8, "iou_f1_sd": 0.1, "fit_seconds_mean": 1.2, "n_features": 3, "augment": "none", "cite": "[R64]"}],
            "baseline": {"row_index": 0},
            "winner": {"row_index": 0},
            "nested": {"scheme": "nested blocks", "iou_f1_mean": 0.75, "iou_f1_sd": 0.05, "n_folds": 5, "seeds": [7]},
            "chance_floor": {"name": "all Normal", "score": 0.4, "sd": 0.0},
        },
    )
    _write(
        tmp_path / "acv_ladder.json",
        {
            **common,
            "task": "acv",
            "scheme": "LOO exploratory",
            "n_cases": 6,
            "rows": [{"row": "peer", "score": 0.9, "sd": 0.03, "fit_seconds": 0.01, "n_features": 1}],
            "baseline": {"row_index": 0},
            "winner": {"row_index": 0},
            "selected": {"scheme": "fixed LOO", "score": 0.88, "sd": 0.04, "n_folds": 6, "seeds": [7]},
            "chance_floor": 0.6,
            "chance_floor_uniform": 0.56,
        },
    )
    _write(
        tmp_path / "rail_ladder.json",
        {
            **common,
            "task": "rail",
            "scheme": "grouped 5-fold",
            "rows": [{"row_index": 0, "arm": "bands", "model": "lgbm", "macro_f1_mean": 0.7, "macro_f1_sd": 0.08, "fit_seconds_mean": 2.5, "wall_seconds": 3.0, "n_features": 4, "augment": True, "cite": "[R232]"}],
            "baseline": {"row_index": 0},
            "winner": {"row_index": 0},
            "nested": {"scheme": "nested grouped", "macro_f1_mean": 0.65, "macro_f1_sd": 0.09, "n_folds": 5, "seeds": [7]},
            "chance_floor": {"name": "majority", "score": 0.3, "sd": 0.0},
            "winner_schemes": [{"scheme": "contiguous", "n_folds": 5, "macro_f1_mean": 0.5, "macro_f1_sd": 0.1}],
        },
    )
    _write(
        tmp_path / "shm_ladder.json",
        {
            **common,
            "task": "shm",
            "scheme": "5x10",
            "rows": [{"spec_id": "rainflow_sn/rainflow", "spec": {"model": "rainflow_sn", "mixup": 0}, "rkf_score": 0.85, "rkf_mape_sd": 0.02, "fit_seconds": 0.5, "n_features": 2}],
            "baseline": {"row_index": 0},
            "physics_row_index": 0,
            "winner": {"row_index": 0, "spec_id": "rainflow_sn/rainflow"},
            "nested": {"scheme": "nested LOO", "score": 0.84, "score_sd": None, "n_folds": 8, "seeds": [7]},
            "chance_floor": {"name": "median", "score": 0.1, "sd": None},
        },
    )


def test_assemble_normalises_all_four_without_metric_recomputation(tmp_path):
    _fixtures(tmp_path)
    payload = report.assemble(tmp_path)
    assert list(payload["subsystems"]) == ["door", "acv", "rail", "shm"]
    assert payload["subsystems"]["door"]["rows"][0]["source"].startswith("rows[0].iou_f1_mean")
    assert payload["subsystems"]["rail"]["stress_splits"][0]["macro_f1_mean"] == 0.5
    assert [row["score"] for row in payload["selected"]] == [0.75, 0.88, 0.65, 0.84]
    assert payload["subsystems"]["acv"]["chance_floor"]["uniform_score"] == 0.56


def test_write_report_emits_source_keyed_markdown_json_and_heatmap(tmp_path, monkeypatch):
    _fixtures(tmp_path)

    def fake_heatmap(payload, path):
        path.write_text("<html>four facets</html>", encoding="utf-8")

    monkeypatch.setattr(report, "_heatmap", fake_heatmap)
    report.write_report(tmp_path)
    text = (tmp_path / "leaderboard.md").read_text(encoding="utf-8")
    assert "Selected per subsystem" in text
    assert "`nested.iou_f1_mean/.iou_f1_sd`" in text
    assert "post-hoc" in text
    assert (tmp_path / "leaderboard.json").is_file()
    assert (tmp_path / "ablation_heatmap.html").read_text(encoding="utf-8") == "<html>four facets</html>"
