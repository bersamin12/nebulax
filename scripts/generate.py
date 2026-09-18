#!/usr/bin/env python
"""Generate a synthetic Level-2 fleet dataset from the three subsystem simulators.

::

    python scripts/generate.py --subsystem all --n-trains 10 --days 30 --seed 0 \\
        --out data/sim --p-healthy 0.4

Writes one ``write_dataset`` partition per run under
``<out>/source=sim/run_id=<run_id>/`` (telemetry/features/fault_log/events parquet +
meta.json), plus two fleet-level rollups at the root of ``<out>``:

* ``fault_log.parquet``  - every injected fault across every run, concatenated (this is
  a convenience copy; each run's own ``fault_log.parquet`` partition already carries its
  own rows and is the source of truth).
* ``index.json``         - one manifest: the CLI args, the wall-clock/row-count report,
  and one entry per run (run_id, subsystem, train_id, car, component_id, healthy,
  injected fault types, row counts, sim seconds, partition path).

Runs are simulated in a multiprocessing pool (default: ``cpu_count() - 1`` workers, one
process per run) because the three ``simulate()`` calls are pure CPU-bound numpy work with
no shared state - the only IPC is a small results dict per run (row counts and the fault
log rows themselves; the large telemetry/features/events frames are written to parquet
*inside* the worker and never cross the process boundary).

How many runs per train, and why it is NOT ``len(COMPONENT_IDS[subsystem])``
-----------------------------------------------------------------------------
Every ``simulate()`` call independently re-emits the train's 1 Hz context rows
(``Service.context_long``) into its own ``long`` table, because each run's parquet
partition has to be usable standalone. For **door** those context rows are never thinned
by ``store_every`` (``context_dt=1.0`` s over the whole service, ~10.4 M rows for one
30-day run), so simulating all 8 door leaves of one train separately would multiply that
fixed cost by 8 for no informational gain - the within-run cycle/window count already
supplies most of the label diversity a model needs (one 30-day run yields >10,000 labelled
cycles/windows on its own). So ``RUNS_PER_TRAIN`` below is deliberately small
(``door: 2, pneumatic: 1, bearing: 2``) and the *fleet* still cycles through every
component id and car via :func:`nebulax.sim.common.sample_scenarios`'s own deterministic
cycling across the whole run list (not per train) - across ``n_trains`` trains every
component id gets hit multiple times.

Shared ``Service`` per train
-----------------------------
The plan requires every subsystem of one train to share one
:class:`~nebulax.sim.common.Service`. Rather than pickle a ``Service`` (whose
``segments`` frame is itself sizeable) across the process pool, each worker
**deterministically reconstructs** it from ``(days, train_seed(seed, train_id), train_id)``
- :func:`nebulax.sim.common.generate_service` is a pure function of its rng draws, so two
processes building the same train's service from the same seed produce bit-identical
segments, whatever subsystem or run asked for it.

Wall-clock projection
---------------------
Before committing to the full run, a short 2-day calibration run per requested subsystem
measures real seconds/simulated-day on this machine, projects the full job list's wall
clock at the chosen worker count, and - only if that projection exceeds ``--budget-min``
(default 30) - falls back from the requested ``--n-trains`` to 6, *loudly*: it prints the
projection, the fallback, and writes the same into ``index.json["projection"]``. It never
shrinks the run silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebulax import schema as S  # noqa: E402
from nebulax.sim.common import Scenario, generate_service, sample_scenarios  # noqa: E402

SUBSYSTEMS = ("door", "pneumatic", "bearing")

#: Simulated runs sampled per train, per subsystem - see the module docstring for why this
#: is small rather than ``len(COMPONENT_IDS[subsystem])``.
RUNS_PER_TRAIN: dict[str, int] = {"door": 2, "pneumatic": 1, "bearing": 2}

#: Legal ``car`` values to sample from, per subsystem (pneumatic's APU is unit-level).
CARS_PER_SUBSYSTEM: dict[str, list[int]] = {
    "door": list(range(1, S.MAX_CAR + 1)),
    "pneumatic": [0],
    "bearing": list(range(1, S.MAX_CAR + 1)),
}

DEFAULT_BUDGET_MIN = 30.0
FALLBACK_N_TRAINS = 6
CALIBRATION_DAYS = 2


def _stable_seed(*parts: object) -> int:
    """A 32-bit seed deterministic across processes and Python invocations - unlike the
    builtin ``hash()``, which is salted per-process unless ``PYTHONHASHSEED`` is pinned."""
    h = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def train_seed(base_seed: int, train_id: str) -> int:
    """Deterministic per-train rng seed, independent of subsystem/run order, so every
    subsystem of one train reconstructs the identical :class:`Service`."""
    return _stable_seed(int(base_seed), train_id)


def _train_ids(n_trains: int) -> list[str]:
    return [f"T{i + 1:02d}" for i in range(n_trains)]


def build_scenarios(
    subsystem: str, n_trains: int, days: int, seed: int, p_healthy: float, max_faults: int
) -> list[Scenario]:
    """Sample every scenario for one subsystem across the whole fleet in one call, so
    :func:`sample_scenarios`'s deterministic component/car cycling runs over the full run
    list rather than restarting per train."""
    n_runs = n_trains * RUNS_PER_TRAIN[subsystem]
    rng = np.random.default_rng(_stable_seed(seed, subsystem))
    return sample_scenarios(
        n_runs,
        subsystem,
        rng,
        p_healthy=p_healthy,
        days=days,
        train_ids=_train_ids(n_trains),
        cars=CARS_PER_SUBSYSTEM[subsystem],
        run_id_prefix=subsystem,
        max_faults=max_faults,
    )


def _simulate(subsystem: str, scenario: Scenario, service, rng, store_every: int):
    if subsystem == "door":
        from nebulax.sim import door as mod

        return mod.simulate(
            mod.DoorParams(),
            scenario.faults,
            service,
            rng,
            store_every,
            run_id=scenario.run_id,
            car=scenario.car,
            component_id=scenario.component_id,
        )
    if subsystem == "pneumatic":
        from nebulax.sim import pneumatic as mod

        return mod.simulate(
            mod.PneumaticParams(),
            scenario.faults,
            service,
            rng,
            store_every,
            run_id=scenario.run_id,
            car=scenario.car,
            component_id=scenario.component_id,
        )
    if subsystem == "bearing":
        from nebulax.sim import bearing as mod

        return mod.simulate(
            mod.BearingParams(),
            scenario.faults,
            service,
            rng,
            store_every,
            run_id=scenario.run_id,
            car=scenario.car,
        )
    raise ValueError(f"_simulate: unknown subsystem {subsystem!r}")


def _run_one(job: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point: rebuild the shared Service, run one scenario, write it, and
    return only small picklable summaries (never the long/features/events frames)."""
    scenario: Scenario = job["scenario"]
    subsystem: str = job["subsystem"]
    store_every: int = job["store_every"]
    out_dir: str = job["out_dir"]
    base_seed: int = job["base_seed"]

    service = generate_service(
        scenario.days, np.random.default_rng(train_seed(base_seed, scenario.train_id)), train_id=scenario.train_id
    )
    rng = np.random.default_rng(scenario.seed)

    t0 = time.perf_counter()
    long, feats, events = _simulate(subsystem, scenario, service, rng, store_every)
    elapsed_s = time.perf_counter() - t0

    fault_rows = scenario.fault_log_rows(service.t0)
    fault_log = pd.DataFrame(fault_rows) if fault_rows else S.empty_fault_log()

    meta = {
        "generator": "scripts/generate.py",
        "subsystem": subsystem,
        "train_id": scenario.train_id,
        "car": int(scenario.car),
        "component_id": scenario.component_id,
        "healthy": bool(scenario.healthy),
        "days": int(scenario.days),
        "scenario_seed": int(scenario.seed),
        "train_seed": train_seed(base_seed, scenario.train_id),
        "store_every": int(store_every),
        "fault_types": sorted({f.fault_type for f in scenario.faults}),
        "elapsed_sim_s": elapsed_s,
    }
    part = S.write_dataset(
        long, feats, fault_log, events, out_dir, source="sim", run_id=scenario.run_id, meta=meta
    )
    return {
        "run_id": scenario.run_id,
        "subsystem": subsystem,
        "train_id": scenario.train_id,
        "car": int(scenario.car),
        "component_id": scenario.component_id,
        "healthy": bool(scenario.healthy),
        "fault_types": meta["fault_types"],
        "days": int(scenario.days),
        "n_long": int(len(long)),
        "n_features": int(len(feats)),
        "n_events": int(len(events)),
        "n_fault_log": int(len(fault_log)),
        "elapsed_sim_s": elapsed_s,
        "path": str(part.relative_to(Path(out_dir))),
        "fault_log_rows": fault_rows,
    }


def _calibrate(subsystem: str, seed: int) -> float:
    """Seconds/simulated-day for one representative (healthy) run of this subsystem.

    Simulates directly (never through :func:`_run_one`/``write_dataset``) so calibration
    touches no disk and contributes nothing to the real dataset."""
    rng = np.random.default_rng(_stable_seed(seed, "calibration", subsystem))
    scenarios = sample_scenarios(
        1,
        subsystem,
        rng,
        p_healthy=1.0,
        days=CALIBRATION_DAYS,
        train_ids=["T01"],
        cars=CARS_PER_SUBSYSTEM[subsystem],
        run_id_prefix=f"calib_{subsystem}",
    )
    service = generate_service(
        CALIBRATION_DAYS, np.random.default_rng(train_seed(seed, "T01")), train_id="T01"
    )
    rng2 = np.random.default_rng(scenarios[0].seed)
    t0 = time.perf_counter()
    _simulate(subsystem, scenarios[0], service, rng2, 10)
    elapsed = time.perf_counter() - t0
    return elapsed / CALIBRATION_DAYS


def project_and_maybe_fallback(
    subsystems: list[str], n_trains: int, days: int, seed: int, p_healthy: float, max_faults: int, workers: int, budget_min: float
) -> tuple[int, dict[str, Any]]:
    """Calibrate seconds/day per subsystem, project total wall clock for the requested
    ``n_trains``, and fall back to :data:`FALLBACK_N_TRAINS` (reporting why) if the
    projection exceeds ``budget_min``. Returns ``(n_trains_to_use, projection_report)``."""
    per_day = {sub: _calibrate(sub, seed) for sub in subsystems}
    n_runs = {sub: n_trains * RUNS_PER_TRAIN[sub] for sub in subsystems}
    cpu_s = sum(per_day[sub] * days * n_runs[sub] for sub in subsystems)
    safety = 1.15
    projected_s = safety * cpu_s / max(1, workers)
    report: dict[str, Any] = {
        "seconds_per_sim_day": per_day,
        "requested_n_trains": n_trains,
        "requested_n_runs": n_runs,
        "workers": workers,
        "projected_cpu_seconds": cpu_s,
        "projected_wall_seconds": projected_s,
        "budget_seconds": budget_min * 60.0,
        "fell_back": False,
    }
    if projected_s > budget_min * 60.0 and n_trains > FALLBACK_N_TRAINS:
        used = FALLBACK_N_TRAINS
        n_runs2 = {sub: used * RUNS_PER_TRAIN[sub] for sub in subsystems}
        cpu_s2 = sum(per_day[sub] * days * n_runs2[sub] for sub in subsystems)
        projected_s2 = safety * cpu_s2 / max(1, workers)
        report.update(
            fell_back=True,
            used_n_trains=used,
            used_n_runs=n_runs2,
            projected_cpu_seconds_used=cpu_s2,
            projected_wall_seconds_used=projected_s2,
        )
        print(
            f"WARNING: projected wall-clock for --n-trains {n_trains} is "
            f"{projected_s / 60.0:.1f} min (budget {budget_min:.0f} min) - falling back to "
            f"--n-trains {used} (projected {projected_s2 / 60.0:.1f} min). "
            f"Pass --budget-min to override.",
            file=sys.stderr,
        )
        return used, report
    report["used_n_trains"] = n_trains
    return n_trains, report


def _merge_with_existing(
    out_dir: Path, subsystems: list[str], results: list[dict[str, Any]], new_fault_log: pd.DataFrame
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any] | None]:
    """Combine this invocation's ``results``/``new_fault_log`` (which cover only
    ``subsystems``) with whatever ``out_dir``'s existing ``index.json``/``fault_log.parquet``
    already recorded for the *other* subsystems.

    A plain ``--subsystem pneumatic`` run only touches the pneumatic partitions on disk, but
    without this merge the fleet-level rollups (``index.json``'s ``runs`` list and the root
    ``fault_log.parquet``) would still be overwritten with pneumatic-only content, silently
    dropping the door/bearing entries even though their partition directories are untouched.
    Returns ``(runs, fault_log, old_index_or_None)``; ``old_index`` is ``None`` when
    ``out_dir/index.json`` does not exist yet (a fresh dataset).
    """
    index_path = out_dir / "index.json"
    fault_path = out_dir / "fault_log.parquet"
    old_index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else None
    old_runs = old_index["runs"] if old_index else []
    kept_runs = [r for r in old_runs if r.get("subsystem") not in subsystems]
    runs = kept_runs + results

    if old_index is not None and fault_path.exists():
        old_fault_log = S.coerce_fault_log(pd.read_parquet(fault_path, engine="pyarrow"))
        kept_fault = old_fault_log[~old_fault_log["subsystem"].astype("string").isin(subsystems)]
    else:
        kept_fault = S.empty_fault_log()
    # pd.concat with a 0-row side warns/raises (FutureWarning on empty-or-all-NA columns);
    # skip it outright rather than feed concat an empty frame.
    if len(kept_fault) and len(new_fault_log):
        fault_log = S.coerce_fault_log(pd.concat([kept_fault, new_fault_log], ignore_index=True))
    elif len(kept_fault):
        fault_log = kept_fault.reset_index(drop=True)
    else:
        fault_log = new_fault_log
    return runs, fault_log, old_index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python scripts/generate.py", description=__doc__)
    ap.add_argument("--subsystem", default="all", choices=("all", *SUBSYSTEMS))
    ap.add_argument("--n-trains", type=int, default=10)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/sim"))
    ap.add_argument("--p-healthy", type=float, default=0.4)
    ap.add_argument("--max-faults", type=int, default=1)
    ap.add_argument("--store-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=None, help="default: cpu_count() - 1")
    ap.add_argument("--budget-min", type=float, default=DEFAULT_BUDGET_MIN)
    ap.add_argument("--no-calibrate", action="store_true", help="skip the wall-clock projection")
    args = ap.parse_args(argv)

    # The active research fleet covers brake air supply and axle bearings. The old
    # door simulator remains importable for compatibility, but ``all`` does not
    # recreate its retired dataset.
    subsystems = ["pneumatic", "bearing"] if args.subsystem == "all" else [args.subsystem]
    workers = args.workers or max(1, (os.cpu_count() or 4) - 1)

    n_trains = args.n_trains
    projection: dict[str, Any] = {}
    if not args.no_calibrate:
        print(f"calibrating {subsystems} on {CALIBRATION_DAYS} sim day(s)...")
        n_trains, projection = project_and_maybe_fallback(
            subsystems, args.n_trains, args.days, args.seed, args.p_healthy, args.max_faults, workers, args.budget_min
        )
        for sub, spd in projection["seconds_per_sim_day"].items():
            print(f"  {sub}: {spd:.3f} s/sim-day")
        print(
            f"  projected wall-clock for --n-trains {n_trains}: "
            f"{projection.get('projected_wall_seconds_used', projection['projected_wall_seconds']) / 60.0:.1f} min "
            f"on {workers} workers"
        )

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    scenario_counts: dict[str, int] = {}
    for sub in subsystems:
        scenarios = build_scenarios(sub, n_trains, args.days, args.seed, args.p_healthy, args.max_faults)
        scenario_counts[sub] = len(scenarios)
        for sc in scenarios:
            jobs.append(
                {"scenario": sc, "subsystem": sub, "store_every": args.store_every, "out_dir": str(out_dir), "base_seed": args.seed}
            )

    print(f"running {len(jobs)} run(s) across {n_trains} train(s), {args.days} day(s) each, on {workers} worker(s)...")
    for sub, n in scenario_counts.items():
        print(f"  {sub}: {n} run(s)")

    wall_t0 = time.perf_counter()
    if workers <= 1 or len(jobs) <= 1:
        results = [_run_one(j) for j in jobs]
    else:
        with mp.Pool(processes=min(workers, len(jobs))) as pool:
            results = pool.map(_run_one, jobs)
    wall_s = time.perf_counter() - wall_t0

    all_fault_rows: list[dict[str, Any]] = []
    for r in results:
        all_fault_rows.extend(r.pop("fault_log_rows"))
    new_fault_log = (
        S.coerce_fault_log(pd.DataFrame(all_fault_rows)) if all_fault_rows else S.empty_fault_log()
    )
    S.validate_fault_log(new_fault_log)

    # Merge with whatever index.json/fault_log.parquet already recorded for the other
    # subsystems, so a partial --subsystem run never drops their entries - see
    # _merge_with_existing's docstring.
    runs, fault_log, old_index = _merge_with_existing(out_dir, subsystems, results, new_fault_log)
    S.validate_fault_log(fault_log)
    fault_log.to_parquet(out_dir / "fault_log.parquet", engine="pyarrow", compression="zstd", index=False)

    totals = {
        "n_runs": len(runs),
        "n_long": int(sum(r["n_long"] for r in runs)),
        "n_features": int(sum(r["n_features"] for r in runs)),
        "n_events": int(sum(r["n_events"] for r in runs)),
        "n_fault_log": int(len(fault_log)),
        "sim_seconds_total": float(sum(r["elapsed_sim_s"] for r in runs)),
        "wall_seconds": wall_s,
    }
    index = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "args": {
            "subsystem": args.subsystem,
            "n_trains_requested": args.n_trains,
            "n_trains_used": n_trains,
            "days": args.days,
            "seed": args.seed,
            "out": str(out_dir),
            "p_healthy": args.p_healthy,
            "max_faults": args.max_faults,
            "store_every": args.store_every,
            "workers": workers,
        },
        "runs_per_train": {sub: RUNS_PER_TRAIN[sub] for sub in {r["subsystem"] for r in runs}},
        "projection": projection,
        "totals": totals,
        "runs": runs,
    }
    if old_index is not None and set(subsystems) != set(SUBSYSTEMS):
        index["merged_from"] = {
            "previous_generated_at": old_index.get("generated_at"),
            "regenerated_subsystems": subsystems,
            "kept_subsystems": sorted({r.get("subsystem") for r in runs if r.get("subsystem") not in subsystems}),
        }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")

    print(
        f"done: {totals['n_runs']} runs, {totals['n_long']:,} long rows, "
        f"{totals['n_features']:,} feature rows, {totals['n_fault_log']} fault-log rows, "
        f"wall-clock {wall_s:.1f} s -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
