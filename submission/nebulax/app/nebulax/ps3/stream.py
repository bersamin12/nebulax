"""Streaming runners for the four PS3 tasks.

Each task provides a streamer that produces a sequence of ``Frame`` objects:
- intermediate frames are causal previews on prefixes of the input;
- the final frame carries exactly the rows the batch predictor writes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from nebulax.ps3 import acv_features as af
from nebulax.ps3 import common
from nebulax.ps3 import door_features
from nebulax.ps3 import rail_features as rf
from nebulax.ps3 import shm_features as sfeat
from nebulax.ps3.acv import ACVRanker, ACVTask, _as_ranker
from nebulax.ps3.common import (
    DOOR_LABELS,
    TASK_NAMES,
    Viewport,
    get_task,
    to_ms,
)
from nebulax.ps3.door import DoorClassifier, DoorTask
from nebulax.ps3.rail import RailTask
from nebulax.ps3.shm import MODE_SPLIT, SHMTask

__all__ = [
    "Frame",
    "STREAMERS",
    "register_streamer",
    "stream_file",
    "stream_door",
    "stream_acv",
    "stream_shm",
    "stream_rail",
]

NORMAL, ABNORMAL = DOOR_LABELS

FRAME_KEYS = [
    "task",
    "file_id",
    "step",
    "n_steps",
    "t",
    "t_unit",
    "progress",
    "rows",
    "numbers",
    "viewport",
    "final",
]


@dataclass
class Frame:
    task: str
    file_id: str
    step: int
    n_steps: int
    t: float
    t_unit: str
    progress: float
    rows: list[dict[str, Any]]
    numbers: dict[str, float]
    viewport: dict[str, Any]
    final: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": str(self.task),
            "file_id": str(self.file_id),
            "step": int(self.step),
            "n_steps": int(self.n_steps),
            "t": float(self.t),
            "t_unit": str(self.t_unit),
            "progress": float(self.progress),
            "rows": [dict(r) for r in self.rows],
            "numbers": {str(k): float(v) for k, v in self.numbers.items()},
            "viewport": dict(self.viewport),
            "final": bool(self.final),
        }


def _clean_numbers(nums: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in nums.items():
        try:
            val = float(v)
            if math.isfinite(val):
                out[str(k)] = val
        except (TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------------------
# Streamers
# --------------------------------------------------------------------------------------


def stream_door(path: Path | str, model: Any = None, *, max_frames: int = 200) -> list[Frame]:
    task = DoorTask()
    raw = door_features.load_stream(path)
    spans = door_features.segment(raw)
    feats = door_features.cycle_features(raw, spans)
    result = task.predict(feats, model)
    batch_rows = task.to_rows(result)
    proba = result.extras.get("proba", [])
    n_steps = len(batch_rows)
    if n_steps == 0:
        vp = Viewport(car=1, side="L", health="ok", component="door_L1").as_dict()
        return [
            Frame(
                task="door",
                file_id="",
                step=0,
                n_steps=1,
                t=0.0,
                t_unit="s",
                progress=1.0,
                rows=[],
                numbers={},
                viewport=vp,
                final=True,
            )
        ]

    start_0 = to_ms(batch_rows[0]["start_time"])
    frames: list[Frame] = []
    for k in range(n_steps):
        end_k = to_ms(batch_rows[k]["end_time"])
        t = float(end_k - start_0) / 1000.0
        p = float(proba[k]) if k < len(proba) else 0.0
        health = "crit" if batch_rows[k]["prediction"] == ABNORMAL else "ok"
        vp = Viewport(car=1, side="L", health=health, component="door_L1").as_dict()
        f = Frame(
            task="door",
            file_id="",
            step=k,
            n_steps=n_steps,
            t=t,
            t_unit="s",
            progress=float(k + 1) / n_steps,
            rows=[dict(r) for r in batch_rows[: k + 1]],
            numbers={"p_abnormal": p},
            viewport=vp,
            final=(k == n_steps - 1),
        )
        frames.append(f)
    return frames


def stream_acv(path: Path | str, model: Any = None, *, max_frames: int = 200) -> list[Frame]:
    p = Path(path)
    task = ACVTask()
    batch_res = task.run(p, model)
    batch_rows = task.to_rows(batch_res)
    case = af.load_case(p)
    n_rows = len(case.time)
    n_steps = min(max_frames, n_rows)
    if n_steps == 0:
        vp = (batch_res.viewport or Viewport()).as_dict()
        return [
            Frame(
                task="acv",
                file_id=case.file_id,
                step=0,
                n_steps=1,
                t=0.0,
                t_unit="h",
                progress=1.0,
                rows=[dict(r) for r in batch_rows],
                numbers=_clean_numbers(batch_res.numbers),
                viewport=vp,
                final=True,
            )
        ]

    cutoffs = [int(round((i + 1) * n_rows / n_steps)) for i in range(n_steps)]
    ranker = _as_ranker(model)
    time_0 = case.time[0]
    frames: list[Frame] = []
    for i, cut in enumerate(cutoffs):
        is_final = i == n_steps - 1
        time_k = case.time[cut - 1]
        t = float((time_k - time_0).total_seconds() / 3600.0)
        if is_final:
            res = batch_res
            rows = [dict(r) for r in batch_rows]
        else:
            time_slice = case.time[:cut]
            panel_slice = {name: df.iloc[:cut] for name, df in case.panel.items()}
            case_k = af.ACVCase(case.file_id, case.cars, time_slice, panel_slice, case.unmapped, case.meta)
            feats_k = task.featurise(case_k, ranker)
            feats_k.attrs["file_id"] = case.file_id
            res = task.predict(feats_k, ranker)
            rows = [dict(r) for r in task.to_rows(res)]

        nums = _clean_numbers(res.numbers)
        vp = (res.viewport or Viewport()).as_dict()
        f = Frame(
            task="acv",
            file_id=case.file_id,
            step=i,
            n_steps=n_steps,
            t=t,
            t_unit="h",
            progress=float(i + 1) / n_steps,
            rows=rows,
            numbers=nums,
            viewport=vp,
            final=is_final,
        )
        frames.append(f)
    return frames


def stream_shm(path: Path | str, model: Any = None, *, max_frames: int = 200) -> list[Frame]:
    p = Path(path)
    name = p.name
    task = SHMTask()
    batch_res = task.run(p, model)
    batch_rows = task.to_rows(batch_res)
    final_pred = float(batch_rows[0]["prediction"])

    x = sfeat.load_signal(p)
    n = x.size
    n_steps = min(max_frames, n)
    if n_steps == 0:
        vp = Viewport(car=None, side=None, health="ok", component="bogie_frame").as_dict()
        return [
            Frame(
                task="shm",
                file_id=name,
                step=0,
                n_steps=1,
                t=0.0,
                t_unit="sample",
                progress=1.0,
                rows=[dict(r) for r in batch_rows],
                numbers={"damage_running": final_pred, "damage_final": final_pred, "cycles_so_far": 0.0},
                viewport=vp,
                final=True,
            )
        ]

    r_full, _, c_full = sfeat.rainflow_cycles(x, method="4point", residue="half")
    d_full = sfeat.miner_damage(r_full, c_full, exponent=5.0)
    cutoffs = [int(round((i + 1) * n / n_steps)) for i in range(n_steps)]
    frames: list[Frame] = []
    for i, cut in enumerate(cutoffs):
        is_final = i == n_steps - 1
        if is_final:
            r = r_full
            val = final_pred
            rows = [dict(r) for r in batch_rows]
        else:
            r, _, c = sfeat.rainflow_cycles(x[:cut], method="4point", residue="half")
            d = sfeat.miner_damage(r, c, exponent=5.0)
            val = final_pred * d / d_full if d_full > 0 else final_pred
            rows = [{"file_id": name, "prediction": f"{val:.9g}"}]

        health = "crit" if val >= 0.6 else ("warn" if val >= MODE_SPLIT else "ok")
        vp = Viewport(car=None, side=None, health=health, component="bogie_frame").as_dict()
        nums = {
            "damage_running": float(val),
            "damage_final": float(final_pred),
            "cycles_so_far": float(r.size),
        }
        f = Frame(
            task="shm",
            file_id=name,
            step=i,
            n_steps=n_steps,
            t=float(cut),
            t_unit="sample",
            progress=float(i + 1) / n_steps,
            rows=rows,
            numbers=nums,
            viewport=vp,
            final=is_final,
        )
        frames.append(f)
    return frames


def stream_rail(path: Path | str, model: Any = None, *, max_frames: int = 200) -> list[Frame]:
    p = Path(path)
    name = p.name
    task = RailTask()
    batch_res = task.run(p, model)
    batch_rows = task.to_rows(batch_res)
    arr = rf.read_rail_csv(p)
    n_steps = min(max_frames, 10)
    if n_steps == 0:
        vp = (batch_res.viewport or Viewport()).as_dict()
        return [
            Frame(
                task="rail",
                file_id=name,
                step=0,
                n_steps=1,
                t=1.0,
                t_unit="s",
                progress=1.0,
                rows=[dict(r) for r in batch_rows],
                numbers=_clean_numbers(batch_res.numbers),
                viewport=vp,
                final=True,
            )
        ]

    frames: list[Frame] = []
    for k in range(n_steps):
        is_final = k == n_steps - 1
        t = float(k + 1) / n_steps
        n_seen = int(round(t * rf.FS_HZ))
        speed = float(rf.speed_from_pulse(arr[:n_seen, 0]))
        speed_ready = float(len(rf._pulse_edges(arr[:n_seen, 0])) >= 2)
        if is_final:
            rows = [dict(r) for r in batch_rows]
            nums = _clean_numbers(batch_res.numbers)
            nums["speed_kmh_so_far"] = speed
            nums["speed_ready"] = speed_ready
            vp = (batch_res.viewport or Viewport()).as_dict()
        else:
            rows = []
            nums = {"speed_kmh_so_far": speed, "speed_ready": speed_ready}
            vp_raw = batch_res.viewport
            vp = Viewport(
                car=vp_raw.car if vp_raw else None,
                side=vp_raw.side if vp_raw else None,
                health="ok",
                component=vp_raw.component if vp_raw else "",
            ).as_dict()

        f = Frame(
            task="rail",
            file_id=name,
            step=k,
            n_steps=n_steps,
            t=t,
            t_unit="s",
            progress=float(k + 1) / n_steps,
            rows=rows,
            numbers=nums,
            viewport=vp,
            final=is_final,
        )
        frames.append(f)
    return frames


# --------------------------------------------------------------------------------------
# Registry & Dispatch
# --------------------------------------------------------------------------------------

STREAMERS: dict[str, Callable[..., list[Frame]]] = {}


def register_streamer(name: str, fn: Callable[..., list[Frame]]) -> None:
    STREAMERS[str(name).strip().lower()] = fn


register_streamer("door", stream_door)
register_streamer("acv", stream_acv)
register_streamer("shm", stream_shm)
register_streamer("rail", stream_rail)


def stream_file(task: str, path: Path | str, model: Any = None, *, max_frames: int = 200) -> list[Frame]:
    key = str(task).strip().lower()
    if key not in STREAMERS:
        raise KeyError(f"unknown PS3 task {task!r}; expected one of {list(STREAMERS)}")
    if model is None:
        from nebulax.api.ps3 import task_model

        model = task_model(key)
    return STREAMERS[key](Path(path), model, max_frames=max_frames)
