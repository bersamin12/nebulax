"""``nebulax.adapters.synthetic`` - thin ``Dataset`` wrapper over the fleet ``scripts/generate.py``
writes under ``data/sim`` (source token ``"sim"``, ``ADAPTER_SUBSYSTEM["sim"] == "all"``: one
generated run covers whichever of door/pneumatic/bearing it was simulated for).

Public contract: :func:`load(raw_dir) -> nebulax.schema.Dataset`, per
``nebulax/adapters/__init__.py``. Unlike the three real-data adapters this module does no
signal engineering at all - ``scripts/generate.py`` already writes schema-conformant
``telemetry.parquet`` / ``features.parquet`` / ``fault_log.parquet`` / ``events.parquet`` +
``meta.json`` per ``source=sim/run_id=<run>/`` partition (via
:func:`nebulax.schema.write_dataset`), so ``load`` only has to find the right partitions,
read them and concatenate.

Why this is not just ``nebulax.schema.read_dataset(raw_dir, source="sim")``
-----------------------------------------------------------------------------
``read_dataset`` reads every table of every matching partition unconditionally. A full
``--n-trains 10 --days 30`` fleet is 50 partitions and ~808 M telemetry rows (~15-20 GB as a
single in-memory frame with categorical columns) - reading all of it eagerly is exactly the
kind of thing that makes ``python -m nebulax.adapters.validate --source sim --raw data/sim``
hang or OOM. The feature/fault_log/event tables are cheap regardless (a few hundred MB at
most across the whole fleet: ~1.66 M feature rows, ~290 K event rows, dozens of fault-log
rows) and are always read in full for every *selected* partition; only ``long`` (telemetry)
is big enough to need care.

``load`` keyword arguments
---------------------------
``subsystem`` (``None`` | ``"door"`` | ``"pneumatic"`` | ``"bearing"``)
    Restrict to partitions whose own ``meta.json`` (written by ``scripts/generate.py``)
    records that subsystem. ``None`` (default) = every subsystem present under ``raw_dir``.
``run_ids`` (``None`` | iterable of str)
    Restrict to these exact ``run_id`` values (e.g. ``["door_0000", "door_0003"]``). Takes
    precedence over ``subsystem`` when both are given (each is still required to actually
    narrow, i.e. an unmatched id raises). ``None`` (default) = no id filter.
``include_long``
    Whether to read the ``telemetry.parquet`` table at all for the *selected* partitions.
    ``False`` is cheap by construction - the telemetry read is simply skipped and ``long``
    comes back as :func:`nebulax.schema.empty_long`. Default ``True``.

The CLI (``nebulax.adapters.validate``) calls every adapter as plain ``module.load(raw_dir)``
- it has no way to pass ``subsystem=``/``run_ids=``/``include_long=``. So the *default* call
with no keyword arguments (``subsystem=None`` **and** ``run_ids=None``, i.e. the caller did
not narrow the selection at all) does not honour ``include_long=True`` literally as "read
every selected partition's telemetry": instead it reads telemetry for only
:data:`SAMPLE_PARTITIONS_PER_SUBSYSTEM` partition(s) per subsystem actually present (chosen
deterministically, lowest ``run_id`` first) - a handful of partitions instead of up to 50 -
and records that it did so in ``meta["long_sampled"]`` / ``meta["warning"]`` plus a
``WARNING:`` line on stderr via :mod:`warnings`. This keeps the validate CLI path fast and
within a few GB of RAM while still exercising the real telemetry schema.

The **full-fleet telemetry path stays available**, just via keyword arguments: pass an
explicit ``subsystem="door"`` (all of that subsystem's partitions, no sampling) or an explicit
``run_ids=[...]`` (exactly those partitions, no sampling) - either counts as the caller having
deliberately narrowed the selection, so the automatic sampling does not apply. A caller that
wants literally everything can pass ``run_ids=[...]`` naming every partition (see
``meta["partitions"]`` on a first, ``include_long=False`` call for the full list), accepting
the memory cost that implies.

``features``/``fault_log``/``events`` are never sampled - they are read in full for every
*selected* (subsystem/run_ids-filtered) partition regardless of ``include_long``, since they
are cheap even at full-fleet scale.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Final, Sequence

import pandas as pd

from nebulax.schema import (
    Dataset,
    coerce_events,
    coerce_fault_log,
    coerce_features,
    coerce_long,
    empty_events,
    empty_fault_log,
    empty_features,
    empty_long,
)

__all__ = ["load", "SAMPLE_PARTITIONS_PER_SUBSYSTEM"]

SOURCE: Final[str] = "sim"

#: How many partitions' telemetry to read per subsystem when the caller did not narrow the
#: selection at all (the CLI's ``load(raw_dir)`` call) - see the module docstring.
SAMPLE_PARTITIONS_PER_SUBSYSTEM: Final[int] = 1

#: File names inside each ``source=sim/run_id=<run>/`` partition, matching
#: :func:`nebulax.schema.write_dataset`.
_TABLE_FILES: Final[dict[str, str]] = {
    "long": "telemetry.parquet",
    "features": "features.parquet",
    "fault_log": "fault_log.parquet",
    "events": "events.parquet",
}


def _partition_dirs(raw_dir: Path) -> list[Path]:
    """Every ``source=sim/run_id=*`` directory under ``raw_dir`` that actually holds a
    telemetry parquet, sorted for determinism."""
    return sorted(
        p for p in raw_dir.glob(f"source={SOURCE}/run_id=*") if (p / _TABLE_FILES["long"]).exists()
    )


def _read_meta(part: Path) -> dict[str, Any]:
    mpath = part / "meta.json"
    if not mpath.exists():
        return {}
    return json.loads(mpath.read_text(encoding="utf-8"))


def _concat(frames: list[pd.DataFrame], empty: pd.DataFrame, coerce) -> pd.DataFrame:
    """Concatenate one table's chunks across partitions and re-cast, exactly like
    :func:`nebulax.schema.read_dataset` does - ``pd.concat`` on ``category`` columns with
    per-partition category sets does not itself guarantee a clean union, so the coercion
    after concatenation (not before) is what actually preserves categorical dtypes."""
    chunks = [f for f in frames if len(f)]
    if not chunks:
        return empty
    return coerce(pd.concat(chunks, ignore_index=True))


def load(
    raw_dir: Path | str = Path("data/sim"),
    *,
    subsystem: str | None = None,
    run_ids: Sequence[str] | None = None,
    include_long: bool = True,
) -> Dataset:
    """Load the synthetic fleet under ``raw_dir`` (``scripts/generate.py``'s ``--out``) into a
    :class:`~nebulax.schema.Dataset`. See the module docstring for ``subsystem``/``run_ids``/
    ``include_long`` and the default sampling behaviour that keeps the plain
    ``load(raw_dir)`` (CLI) call cheap.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"synthetic adapter: {raw_dir} does not exist; expected the layout scripts/generate.py "
            f"writes: <raw_dir>/source=sim/run_id=<run>/{{telemetry,features,fault_log,events}}.parquet "
            f"+ meta.json, plus <raw_dir>/index.json (run scripts/generate.py --out {raw_dir} first)"
        )
    all_parts = _partition_dirs(raw_dir)
    if not all_parts:
        raise FileNotFoundError(
            f"synthetic adapter: no source=sim/run_id=*/telemetry.parquet partitions found under "
            f"{raw_dir}; run scripts/generate.py --out {raw_dir} first"
        )
    metas = {p: _read_meta(p) for p in all_parts}

    unrestricted = subsystem is None and run_ids is None
    if run_ids is not None:
        want = {str(r) for r in run_ids}
        selected = [p for p in all_parts if metas[p].get("run_id") in want]
        found = {metas[p].get("run_id") for p in selected}
        missing = sorted(want - found)
        if missing:
            raise ValueError(
                f"synthetic adapter: run_id(s) {missing} not found under {raw_dir}; available = "
                f"{sorted(metas[p].get('run_id') for p in all_parts)}"
            )
    elif subsystem is not None:
        selected = [p for p in all_parts if metas[p].get("subsystem") == subsystem]
        if not selected:
            available = sorted({metas[p].get("subsystem") for p in all_parts})
            raise ValueError(
                f"synthetic adapter: no partitions with subsystem={subsystem!r} under {raw_dir}; "
                f"available subsystems = {available}"
            )
    else:
        selected = all_parts

    long_sampled = False
    if not include_long:
        long_parts: list[Path] = []
    elif unrestricted:
        by_sub: dict[str, list[Path]] = {}
        for p in selected:
            by_sub.setdefault(str(metas[p].get("subsystem", "unknown")), []).append(p)
        long_parts = []
        for sub in sorted(by_sub):
            long_parts.extend(sorted(by_sub[sub], key=lambda p: metas[p].get("run_id", p.name))[:SAMPLE_PARTITIONS_PER_SUBSYSTEM])
        long_sampled = len(long_parts) < len(selected)
    else:
        long_parts = selected

    frames: dict[str, list[pd.DataFrame]] = {k: [] for k in _TABLE_FILES}
    for p in selected:
        for key in ("features", "fault_log", "events"):
            fp = p / _TABLE_FILES[key]
            if fp.exists():
                frames[key].append(pd.read_parquet(fp, engine="pyarrow"))
    for p in long_parts:
        fp = p / _TABLE_FILES["long"]
        if fp.exists():
            frames["long"].append(pd.read_parquet(fp, engine="pyarrow"))

    long = _concat(frames["long"], empty_long(), coerce_long)
    features = _concat(frames["features"], empty_features(), coerce_features)
    fault_log = _concat(frames["fault_log"], empty_fault_log(), coerce_fault_log)
    events = _concat(frames["events"], empty_events(), coerce_events)

    index_path = raw_dir / "index.json"
    index_content = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else None

    selected_ids = sorted(str(metas[p].get("run_id", p.name)) for p in selected)
    long_ids = sorted(str(metas[p].get("run_id", p.name)) for p in long_parts)
    meta: dict[str, Any] = {
        "source": SOURCE,
        "raw_dir": str(raw_dir),
        "index_json": index_content,
        "partitions": selected_ids,
        "partitions_with_long": long_ids,
        "n_partitions_total": len(all_parts),
        "n_partitions_selected": len(selected),
        "subsystem_filter": subsystem,
        "run_ids_filter": sorted(str(r) for r in run_ids) if run_ids is not None else None,
        "include_long": include_long,
        "long_sampled": long_sampled,
    }
    if long_sampled:
        msg = (
            f"nebulax.adapters.synthetic.load({raw_dir!s}): called with no subsystem=/run_ids= "
            f"filter, so telemetry ('long') is a sample of {SAMPLE_PARTITIONS_PER_SUBSYSTEM} "
            f"partition(s) per subsystem ({len(long_parts)}/{len(selected)} partitions read: "
            f"{long_ids}), not the full fleet - the ~808 M-row full telemetry table does not fit "
            f"in memory as one frame. Pass subsystem='door'|'pneumatic'|'bearing' or an explicit "
            f"run_ids=[...] to read that subsystem's/those partitions' full telemetry instead."
        )
        meta["warning"] = msg
        warnings.warn(msg, stacklevel=2)
        print(f"WARNING: {msg}", file=sys.stderr)

    return Dataset(long=long, features=features, fault_log=fault_log, events=events, meta=meta)
