#!/usr/bin/env python3
"""Make the small, tracked bearing and brake replay from the local research fleet.

Run from the project root after the full ``data/sim`` and ``data/scores`` exist.
The output is a synthetic replay sample, not a training or validation dataset.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data"
DEST = ROOT / "demo_data"
TRAINS = ("T04", "T09")
START = pd.Timestamp("2026-09-12T00:00:00Z")
END = pd.Timestamp("2026-09-14T00:00:00Z")
SUBSYSTEMS = ("bearing", "pneumatic")


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _slice_parquet(source: Path, target: Path, column: str) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(
        source,
        filters=[(column, ">=", START.to_pydatetime()), (column, "<", END.to_pydatetime())],
    )
    pq.write_table(table, target, compression="zstd")
    return table.num_rows


def _overlapping_faults(frame: pd.DataFrame) -> pd.DataFrame:
    onset = pd.to_datetime(frame["t_onset"], utc=True)
    failure = pd.to_datetime(frame["t_failure"], utc=True)
    return frame.loc[(onset < END) & (failure.isna() | (failure >= START))].reset_index(drop=True)


def build() -> None:
    assert SOURCE.resolve() != DEST.resolve()
    if not (SOURCE / "sim" / "index.json").is_file():
        raise FileNotFoundError("full data/sim/index.json is required to build demo_data")
    if DEST.exists():
        shutil.rmtree(DEST)
    index = json.loads((SOURCE / "sim" / "index.json").read_text(encoding="utf-8"))
    selected = [
        run for run in index["runs"]
        if run.get("train_id") in TRAINS and run.get("subsystem") in SUBSYSTEMS
    ]
    if len(selected) != 6:
        raise RuntimeError(f"expected six T04/T09 bearing and pneumatic runs, found {len(selected)}")

    runs: list[dict] = []
    for run in selected:
        rel = Path(run["path"])
        src = SOURCE / "sim" / rel
        dst = DEST / "sim" / rel
        counts = {
            "long": _slice_parquet(src / "telemetry.parquet", dst / "telemetry.parquet", "timestamp"),
            "features": _slice_parquet(src / "features.parquet", dst / "features.parquet", "t_start"),
            "events": _slice_parquet(src / "events.parquet", dst / "events.parquet", "timestamp"),
        }
        faults = _overlapping_faults(pd.read_parquet(src / "fault_log.parquet"))
        faults.to_parquet(dst / "fault_log.parquet", engine="pyarrow", compression="zstd", index=False)
        counts["fault_log"] = len(faults)
        meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
        meta["rows"] = counts
        meta["sample_window"] = {"start": START.isoformat(), "end_exclusive": END.isoformat()}
        _save_json(dst / "meta.json", meta)
        row = dict(run)
        row.update(n_long=counts["long"], n_features=counts["features"],
                   n_events=counts["events"], n_fault_log=counts["fault_log"], days=2)
        runs.append(row)

    fault_log = pd.read_parquet(SOURCE / "sim" / "fault_log.parquet")
    fault_log = _overlapping_faults(fault_log.loc[fault_log["train_id"].isin(TRAINS)
                                                   & fault_log["subsystem"].isin(SUBSYSTEMS)])
    (DEST / "sim").mkdir(parents=True, exist_ok=True)
    fault_log.to_parquet(DEST / "sim" / "fault_log.parquet", engine="pyarrow", compression="zstd", index=False)
    demo_index = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source": "synthetic research fleet excerpt",
        "sample_window": {"start": START.isoformat(), "end_exclusive": END.isoformat()},
        "runs_per_train": {"pneumatic": 1, "bearing": 2},
        "totals": {"n_runs": len(runs), "n_long": sum(r["n_long"] for r in runs),
                   "n_features": sum(r["n_features"] for r in runs),
                   "n_events": sum(r["n_events"] for r in runs),
                   "n_fault_log": len(fault_log)},
        "runs": runs,
    }
    _save_json(DEST / "sim" / "index.json", demo_index)

    scores = pq.read_table(
        SOURCE / "scores" / "scores.parquet",
        filters=[("train_id", "in", list(TRAINS)),
                 ("timestamp", ">=", START.to_pydatetime()),
                 ("timestamp", "<", END.to_pydatetime())],
    )
    (DEST / "scores").mkdir(parents=True, exist_ok=True)
    pq.write_table(scores, DEST / "scores" / "scores.parquet", compression="zstd")
    episodes = pd.read_parquet(SOURCE / "scores" / "episodes.parquet")
    episodes = episodes.loc[episodes["train_id"].isin(TRAINS)
                            & (pd.to_datetime(episodes["t_start"], utc=True) < END)
                            & (pd.to_datetime(episodes["t_end"], utc=True) >= START)].copy()
    episodes["t_start"] = pd.to_datetime(episodes["t_start"], utc=True).clip(lower=START)
    episodes["t_end"] = pd.to_datetime(episodes["t_end"], utc=True).clip(upper=END - pd.Timedelta(seconds=1))
    episodes.to_parquet(DEST / "scores" / "episodes.parquet", engine="pyarrow", compression="zstd", index=False)
    manifest = json.loads((SOURCE / "scores" / "manifest.json").read_text(encoding="utf-8"))
    manifest["subsystems"] = {key: value for key, value in manifest["subsystems"].items()
                              if key in SUBSYSTEMS}
    manifest["sample_window"] = demo_index["sample_window"]
    manifest["sample_train_ids"] = list(TRAINS)
    _save_json(DEST / "scores" / "manifest.json", manifest)
    for sub in SUBSYSTEMS:
        for train in TRAINS:
            source_model = SOURCE / "scores" / "models" / sub / f"{train}.pkl"
            target_model = DEST / "scores" / "models" / sub / f"{train}.pkl"
            target_model.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_model, target_model)
    (DEST / "README.md").write_text(
        "# Compact synthetic fleet replay\n\n"
        "T04 and T09, 12–14 September 2026 UTC. This sample contains only brake air supply "
        "and axle bearing telemetry, scores, events and injection models. It lets a fresh "
        "checkout run the two research demos without the large ignored research datasets. "
        "Regenerate it with `python scripts/build_demo_data.py` after restoring the full local data.\n",
        encoding="utf-8",
    )
    print(f"demo_data: {len(runs)} runs, {scores.num_rows} score rows, {len(episodes)} episodes")


if __name__ == "__main__":
    build()
