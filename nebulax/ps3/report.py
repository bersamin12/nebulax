"""Assemble the PS3 leaderboard from the four committed ladder JSON files.

No metric is recomputed here. The subsystem trainers/verifiers own every number, including chance
floors and honest nested/outer headlines; this module only normalises their heterogeneous schemas,
writes a source-keyed Markdown report and renders the same rows as a four-facet Plotly heatmap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from nebulax.ps3.common import RESULTS_DIR, TASK_NAMES

__all__ = ["assemble", "write_report", "main"]

_TITLES = {
    "door": "Door — segmentation + classification",
    "acv": "ACV — leaking-car ranking",
    "rail": "Rail corrugation — three-class classification",
    "shm": "SHM — cumulative fatigue damage",
}

_METRICS = {
    "door": "IoU-weighted F1",
    "acv": "rank decay",
    "rail": "macro F1",
    "shm": "1 − MAPE",
}

_ARTIFACTS = {task: f"models/ps3/{task}.pkl" for task in TASK_NAMES}


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read ladder JSON {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        raise ValueError(f"{path}: expected an object with a rows array")
    return value


def _require(data: dict[str, Any], key: str, source: Path) -> Any:
    if key not in data:
        raise ValueError(f"{source}: missing report metadata {key!r}")
    return data[key]


def _shm_cite(spec_id: str) -> str:
    if spec_id.startswith("rainflow"):
        return "[R269][R270]"
    if "spectral" in spec_id:
        return "[R269][R275][R276]"
    if "fds" in spec_id:
        return "[R274]"
    if "cmixup" in spec_id:
        return "[R299]"
    return "[R272][R273]"


def _normalise_rows(task: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    winner = data.get("winner") or {}
    winner_idx = winner.get("row_index")
    if winner_idx is None and task == "shm":
        target = winner.get("spec_id")
        winner_idx = next((i for i, row in enumerate(data["rows"]) if row.get("spec_id") == target), None)
    baseline = data.get("baseline") or {}
    baseline_idx = baseline.get("row_index")
    physics_idx = data.get("physics_row_index")
    rows: list[dict[str, Any]] = []
    for i, raw in enumerate(data["rows"]):
        if task == "door":
            name = str(raw.get("name", raw.get("model", f"row {i}")))
            model = str(raw.get("model", "unknown"))
            score, sd = raw.get("iou_f1_mean"), raw.get("iou_f1_sd")
            fit, n_features = raw.get("fit_seconds_mean"), raw.get("n_features")
            augment = "off" if str(raw.get("augment", "none")) == "none" else "on"
            cite = str(raw.get("cite", ""))
            cv = f"{len(raw.get('folds', []))} folds; seeds {raw.get('seeds', data.get('seeds', []))}"
            paths = f"rows[{i}].iou_f1_mean/.iou_f1_sd/.fit_seconds_mean/.n_features"
        elif task == "rail":
            name = str(raw.get("arm", f"row {i}"))
            model = str(raw.get("model", "unknown"))
            score, sd = raw.get("macro_f1_mean"), raw.get("macro_f1_sd")
            fit, n_features = raw.get("fit_seconds_mean", raw.get("wall_seconds")), raw.get("n_features")
            augment = "on" if raw.get("augment") else "off"
            cite = str(raw.get("cite", ""))
            detail = raw.get("report") or {}
            cv = f"{detail.get('n_folds', 0)} folds; seeds {detail.get('seeds', data.get('seeds', []))}"
            paths = f"rows[{i}].macro_f1_mean/.macro_f1_sd/.fit_seconds_mean/.n_features"
        elif task == "acv":
            name = str(raw.get("row", f"row {i}"))
            model = "fixed rule"
            score, sd = raw.get("score"), raw.get("sd")
            fit, n_features = raw.get("fit_seconds"), raw.get("n_features")
            augment, cite = "off", "[R251][R252][R253]"
            cv = f"{data.get('n_cases', 0)} cases; seeds {data.get('seeds', [])}"
            paths = f"rows[{i}].score/.sd/.fit_seconds/.n_features"
        else:
            name = str(raw.get("spec_id", f"row {i}"))
            model = str((raw.get("spec") or {}).get("model", "unknown"))
            score, sd = raw.get("rkf_score"), raw.get("rkf_mape_sd")
            fit, n_features = raw.get("fit_seconds"), raw.get("n_features")
            augment = "on" if int((raw.get("spec") or {}).get("mixup", 0) or 0) > 0 else "off"
            cite = _shm_cite(name)
            cv = (
                f"{int(data.get('n_splits', 0)) * int(data.get('n_repeats', 0))} folds; "
                f"seeds {data.get('seeds', [])}"
            )
            paths = f"rows[{i}].rkf_score/.rkf_mape_sd/.fit_seconds/.n_features"
        if score is None or sd is None:
            continue
        tags: list[str] = []
        if i == winner_idx:
            tags.append("winner")
        if i == baseline_idx:
            tags.append("baseline")
        if i == physics_idx:
            tags.append("physics")
        rows.append(
            {
                "row_index": i,
                "name": name,
                "model": model,
                "score": float(score),
                "sd": float(sd),
                "fit_seconds": None if fit is None else float(fit),
                "n_features": None if n_features is None else int(n_features),
                "augmentation": augment,
                "cv": cv,
                "citation": cite,
                "tags": tags,
                "source": paths,
            }
        )
    return rows


def _selected(task: str, data: dict[str, Any]) -> dict[str, Any]:
    if task == "door":
        headline = _require(data, "nested", Path("door_ladder.json"))
        score, sd = headline["iou_f1_mean"], headline["iou_f1_sd"]
        source = "nested.iou_f1_mean/.iou_f1_sd"
    elif task == "rail":
        headline = _require(data, "nested", Path("rail_ladder.json"))
        score, sd = headline["macro_f1_mean"], headline["macro_f1_sd"]
        source = "nested.macro_f1_mean/.macro_f1_sd"
    elif task == "acv":
        headline = _require(data, "selected", Path("acv_ladder.json"))
        score, sd = headline["score"], headline["sd"]
        source = "selected.score/.sd"
    else:
        headline = _require(data, "nested", Path("shm_ladder.json"))
        score, sd = headline["score"], headline.get("score_sd")
        source = "nested.score/.score_sd"
    return {
        "task": task,
        "metric": _METRICS[task],
        "score": float(score),
        "sd": None if sd is None else float(sd),
        "scheme": str(headline.get("scheme", data.get("scheme", ""))),
        "n_folds": int(headline.get("n_folds", headline.get("n", data.get("n_cases", 0))) or 0),
        "seeds": list(headline.get("seeds", data.get("seeds", []))),
        "artifact": _ARTIFACTS[task],
        "source": source,
    }


def assemble(results_dir: Path | str = RESULTS_DIR) -> dict[str, Any]:
    """Normalise four ladder files without fitting, predicting or scoring anything."""
    root = Path(results_dir)
    ladders = {task: _read(root / f"{task}_ladder.json") for task in TASK_NAMES}
    subsystems: dict[str, Any] = {}
    for task, data in ladders.items():
        chance = _require(data, "chance_floor", root / f"{task}_ladder.json")
        if not isinstance(chance, dict):
            chance = {
                "name": "empty-cars-last blind ranking" if task == "acv" else "chance floor",
                "score": float(chance),
                "sd": None,
                "uniform_score": data.get("chance_floor_uniform"),
                "source": "stored ladder metadata",
                "key_path": "chance_floor",
                "uniform_key_path": "chance_floor_uniform",
            }
        subsystems[task] = {
            "title": _TITLES[task],
            "metric": _METRICS[task],
            "scheme": data.get("scheme", ""),
            "seeds": list(data.get("seeds", [])),
            "rows": _normalise_rows(task, data),
            "chance_floor": chance,
            "skipped": list(_require(data, "skipped", root / f"{task}_ladder.json")),
            "winner": data.get("winner"),
            "winner_selection_status": data.get("winner_selection_status", ""),
            "stress_splits": data.get("winner_schemes", []) if task == "rail" else [],
            "source_file": f"{task}_ladder.json",
        }
    return {
        "source": "results/ps3/<task>_ladder.json only; no metrics recomputed",
        "subsystems": subsystems,
        "selected": [_selected(task, ladders[task]) for task in TASK_NAMES],
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PS3 model ladder leaderboard",
        "",
        "All metrics below are copied from the four committed ladder JSON files; this report does not "
        "refit, predict, or recompute a score. A bold row is the winner of its frozen **selection CV**, "
        "not automatically the honest headline. Door and Rail artifacts were chosen after the ladder "
        "was inspected, so their selection values are explicitly post-hoc; the final table uses the "
        "nested/outer estimates. ACV keeps its pre-registered fixed rule because the six-case pool is "
        "exploratory. Rail's `speed < 20 km/h -> Normal` rule is a dataset shortcut, not physics.",
        "",
    ]
    for task in TASK_NAMES:
        block = report["subsystems"][task]
        lines += [f"## {block['title']}", ""]
        selected = next(row for row in report["selected"] if row["task"] == task)
        lines += [
            f"Frozen scheme: {block['scheme']}; seeds {block['seeds']}. Honest headline uses "
            f"{selected['n_folds']} outer folds/cases ({selected['source']}).",
            "",
            "| model | ablation / feature row | metric mean ± sd | CV folds / seeds | fit s | n feat | augmentation | addendum cite | flags | JSON key path |",
            "|---|---|---:|---|---:|---:|---|---|---|---|",
        ]
        for row in block["rows"]:
            model = row["model"]
            name = row["name"]
            metric = f"{_fmt(row['score'])} ± {_fmt(row['sd'])}"
            if "winner" in row["tags"]:
                model, name, metric = f"**{model}**", f"**{name}**", f"**{metric}**"
            lines.append(
                f"| {model} | {name} | {metric} | {row['cv']} | {_fmt(row['fit_seconds'], 2)} | "
                f"{row['n_features'] if row['n_features'] is not None else 'n/a'} | {row['augmentation']} | "
                f"{row['citation']} | {', '.join(row['tags']) or '-'} | `{row['source']}` |"
            )
        chance = block["chance_floor"]
        lines.append(
            f"| chance | {chance['name']} | {_fmt(float(chance['score']))} ± "
            f"{_fmt(None if chance.get('sd') is None else float(chance['sd']))} | fixed vocabulary / released set | n/a | n/a | off | - | chance floor | "
            f"`{chance.get('key_path', 'chance_floor.score/.sd')}` |"
        )
        if task == "acv" and "uniform_score" in chance:
            lines.append(
                f"| chance | uniform 8-car ranking | {_fmt(float(chance['uniform_score']))} ± n/a | "
                f"6 released cases | n/a | n/a | off | - | chance floor | `{chance.get('uniform_key_path', 'chance_floor.uniform_score')}` |"
            )
        if task == "rail" and block["stress_splits"]:
            lines += ["", "Winner stress splits (beside the stratified selection number):", ""]
            for i, split in enumerate(block["stress_splits"]):
                lines.append(
                    f"- {split['scheme']}: {_fmt(float(split['macro_f1_mean']))} ± "
                    f"{_fmt(float(split['macro_f1_sd']))}, {split['n_folds']} folds "
                    f"(`winner_schemes[{i}].macro_f1_mean/.macro_f1_sd/.n_folds`)."
                )
        lines += ["", "Skipped from the addendum:", ""] + [f"- {item}" for item in block["skipped"]] + [""]

    lines += [
        "## Selected per subsystem",
        "",
        "| subsystem | metric | honest nested/outer headline | folds/cases | seeds | artifact | JSON key path |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for selected in report["selected"]:
        spread = "" if selected["sd"] is None else f" ± {_fmt(selected['sd'])}"
        lines.append(
            f"| {selected['task']} | {selected['metric']} | **{_fmt(selected['score'])}{spread}** | "
            f"{selected['n_folds']} | {selected['seeds']} | `{selected['artifact']}` | `{selected['source']}` |"
        )
    return "\n".join(lines) + "\n"


def _heatmap(report: dict[str, Any], out: Path) -> None:
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - app extra includes plotly
        raise RuntimeError("plotly is required to write the PS3 ablation heatmap") from exc

    fig = make_subplots(rows=2, cols=2, subplot_titles=[_TITLES[t] for t in TASK_NAMES])
    for panel, task in enumerate(TASK_NAMES):
        rows = report["subsystems"][task]["rows"]
        y = [row["name"] for row in rows]
        z = [[row["score"]] for row in rows]
        text = [[f"{row['score']:.4f} ± {row['sd']:.4f}<br>{row['source']}"] for row in rows]
        r, c = divmod(panel, 2)
        fig.add_trace(
            go.Heatmap(z=z, y=y, x=[_METRICS[task]], text=text, hovertemplate="%{y}<br>%{text}<extra></extra>", colorscale="Viridis", showscale=False),
            row=r + 1,
            col=c + 1,
        )
    fig.update_layout(title="PS3 model × ablation scores (copied from ladder JSON)", height=2200, width=1500)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs=True, full_html=True)


def write_report(results_dir: Path | str = RESULTS_DIR) -> dict[str, Any]:
    root = Path(results_dir)
    report = assemble(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "leaderboard.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (root / "leaderboard.md").write_text(_markdown(report), encoding="utf-8")
    _heatmap(report, root / "ablation_heatmap.html")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)
    write_report(args.results_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
