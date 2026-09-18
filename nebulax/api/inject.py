"""``POST /api/sim/inject``: re-simulate one component with a fault and re-score it.

The route is the demo's "what if" button: pick a leaf, a fault and a ramp, and the same three
simulators that built ``data/sim`` regenerate *that one component* with the onset at the
current replay ``ts``. The fitted winner for the held-out train re-scores the result and the
rows go into :class:`~nebulax.api.state.Overlay`, which every GET route reads before the
files on disk. ``DELETE /api/sim/inject`` drops it again.

Why a window and not the whole clock: one 30-day door leaf costs ~100 s in
``scripts/generate.py`` and the contract gives this route 60 s. So the simulation runs from
``ts - 1 day`` to the end of the replay clock, capped at :func:`max_days_for` days (measured
cost per simulated day against the budget, at most :data:`MAX_INJECT_DAYS`), and the response
says exactly which window it covered and which cap applied. Rows scored *before* that window
are kept: the overlay patches the series, it does not replace it.

The scoring module (``nebulax.demo.scoring``, contract section 3) is imported **inside** the
call, so the API imports, boots and serves every other route whether or not that module
exists yet - and so tests can monkeypatch a fake in its place.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any

import numpy as np
import pandas as pd

from nebulax import schema as S
from nebulax.api.state import ComponentKey, FleetState, iso, parse_ts

__all__ = [
    "MAX_INJECT_DAYS",
    "INJECT_BUDGET_S",
    "SECONDS_PER_SIM_DAY",
    "InjectError",
    "InjectNotFound",
    "inject",
    "max_days_for",
    "plan_window",
    "build_scenario",
]

#: Longest span one injection may ever simulate: the replay clock is 30 days (contract s. 1).
MAX_INJECT_DAYS = 30

#: The contract's budget for this route, and the fraction of it we are willing to spend.
INJECT_BUDGET_S = 60.0
INJECT_SAFETY = 0.75

#: Wall-clock seconds per simulated day, measured on the demo machine (simulation + re-score,
#: ``store_every=10``): a 22-day door leaf costs 46.9 s end to end (2.13 s/day) and a 12-day
#: bogie 30.3 s (2.53 s/day - eight axle-box series to score), the pneumatic unit about half.
#: :func:`max_days_for` turns these into the per-request cap so the worst case stays inside
#: :data:`INJECT_BUDGET_S`: 20 days for a door leaf, 17 for a bogie, the whole clock for an APU.
SECONDS_PER_SIM_DAY: dict[str, float] = {"door": 2.2, "pneumatic": 1.0, "bearing": 2.6}

#: How far before the onset the re-simulation starts, so the charts show a healthy baseline.
LEAD_IN_DAYS = 1.0

#: ``store_every`` for the door waveform capture - the same value ``scripts/generate.py`` used.
STORE_EVERY = 10


class InjectError(ValueError):
    """Bad request: unknown component, fault type or ramp. The route turns it into a 400."""


class InjectNotFound(InjectError):
    """Nothing to inject into: the train is not in the fleet. The route turns it into a 404."""


def max_days_for(subsystem: str) -> int:
    """Longest window this subsystem may simulate inside the route's budget.

    Linear in the measured cost per simulated day (:data:`SECONDS_PER_SIM_DAY`), capped at the
    30-day replay clock: 22 days for a door leaf or a bogie, the whole clock for an APU.
    """
    per_day = SECONDS_PER_SIM_DAY.get(subsystem, max(SECONDS_PER_SIM_DAY.values()))
    budget = INJECT_BUDGET_S * INJECT_SAFETY
    return int(max(1, min(MAX_INJECT_DAYS, int(budget / per_day))))


def _stable_seed(*parts: object) -> int:
    """``scripts/generate.py``'s seed helper, repeated here so one injection is reproducible
    across processes (the builtin ``hash()`` is salted per process).

    It does **not** reproduce the stored run's ``Service``: that one was drawn for 30 days from
    the fleet's base seed at the fleet ``t0``, and this window starts elsewhere and is shorter.
    The timetable is therefore statistically the same and numerically different, which the
    response's ``note`` says out loud.
    """
    h = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def _subsystem_of(component_id: str) -> str | None:
    for sub, cids in S.COMPONENT_IDS.items():
        if component_id in cids:
            return sub
    return None


def plan_window(
    t_onset: pd.Timestamp, clock_end: pd.Timestamp, *, max_days: int = MAX_INJECT_DAYS
) -> tuple[pd.Timestamp, int]:
    """``(t0, days)`` of the span to simulate: one lead-in day before the onset, then as much
    of the remaining replay clock as ``max_days`` allows (always >= 1 day)."""
    t0 = t_onset - pd.Timedelta(days=LEAD_IN_DAYS)
    span_days = (clock_end - t0) / pd.Timedelta(days=1)
    days = int(min(max(math.ceil(span_days), 1), max_days))
    return t0, days


def build_scenario(
    *,
    subsystem: str,
    train_id: str,
    car: int,
    component_id: str,
    fault_type: str,
    t_onset_s: float,
    ramp_days: float,
    days: int,
    seed: int,
    run_id: str,
):
    """A one-fault :class:`~nebulax.sim.common.Scenario` with the requested onset and ramp."""
    from nebulax.sim.common import (
        SEC_PER_DAY,
        SHOCK_FAULTS,
        STEP_FAULTS,
        DegradationTrajectory,
        Scenario,
        ShockProcess,
    )

    rng = np.random.default_rng(seed)
    shape = "step" if fault_type in STEP_FAULTS else ("shock" if fault_type in SHOCK_FAULTS else "power")
    shocks = None
    if shape == "shock":
        shocks = ShockProcess().sample(days * SEC_PER_DAY, rng, t_start=t_onset_s)
    traj = DegradationTrajectory(
        fault_type=fault_type,
        subsystem=subsystem,
        component_id=component_id,
        t_onset=float(t_onset_s),
        t_failure=float(t_onset_s) + (0.0 if shape == "step" else float(ramp_days) * SEC_PER_DAY),
        gamma=2.0 if shape == "power" else 1.0,
        shape=shape,
        shocks=shocks,
    )
    return Scenario(
        run_id=run_id,
        subsystem=subsystem,
        train_id=train_id,
        car=int(car),
        component_id=component_id,
        days=int(days),
        healthy=False,
        faults=(traj,),
        seed=int(seed),
    )


def _simulate(subsystem: str, scenario, service, rng, store_every: int):
    """``scripts/generate.py::_simulate``, repeated (scripts/ is not an importable package)."""
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
    raise InjectError(f"inject: unknown subsystem {subsystem!r}")


def _scores_dir_kwarg(scoring: Any, state: FleetState) -> dict[str, Any]:
    """``scoring.load_fitted`` defaults its pickle directory to ``data/scores``; pass ours
    instead whenever the module accepts it, so ``NEBULAX_DATA_DIR`` is honoured end to end."""
    import inspect

    try:
        params = inspect.signature(scoring.load_fitted).parameters
    except (TypeError, ValueError):  # pragma: no cover - a builtin or C callable
        return {}
    return {"scores_dir": state.settings.scores_dir} if "scores_dir" in params else {}


def _episodes_from(scoring: Any, scores: pd.DataFrame, fault_log: pd.DataFrame) -> pd.DataFrame:
    """``scoring.episodes_from_scores`` if the module has it, else an empty frame - the route
    must not fail because the demo module is mid-flight."""
    fn = getattr(scoring, "episodes_from_scores", None)
    if fn is None:
        return pd.DataFrame()
    try:
        out = fn(scores, fault_log)
    except Exception:  # pragma: no cover - a half-written scorer must not 500 the demo
        return pd.DataFrame()
    return out if isinstance(out, pd.DataFrame) else pd.DataFrame()


def inject(
    state: FleetState,
    *,
    train_id: str,
    car: int,
    component_id: str,
    fault_type: str,
    severity_ramp_days: float,
    seed: int | None = None,
    t_onset: Any = None,
    now: Any = None,
) -> dict[str, Any]:
    """Simulate + re-score one component and install the result as the overlay.

    ``t_onset`` defaults to ``now`` (the current replay ``ts``). Raises :class:`InjectError`
    for an unknown component / fault type / non-positive ramp, which ``main.py`` maps to 400.
    """
    if train_id not in state.sim_train_ids():
        # Checked first: an unknown train is a 404 whether or not the fitted winners exist.
        raise InjectNotFound(
            f"inject: unknown train {train_id!r}; the sim fleet is {state.sim_train_ids()}"
        )
    subsystem = _subsystem_of(component_id)
    if subsystem is None or subsystem not in S.FAULT_SUBSYSTEMS:
        raise InjectError(
            f"inject: component_id {component_id!r} is not a door leaf, APU or axle box; "
            f"expected one of {sorted(c for cs in S.COMPONENT_IDS.values() for c in cs)}"
        )
    allowed = S.FAULT_TYPES[subsystem]
    if fault_type not in allowed or fault_type in ("healthy", "nff"):
        raise InjectError(
            f"inject: fault_type {fault_type!r} is not injectable for {subsystem!r}; expected one of "
            f"{[f for f in allowed if f not in ('healthy', 'nff')]}"
        )
    try:
        ramp = float(severity_ramp_days)
    except (TypeError, ValueError):
        raise InjectError(f"inject: severity_ramp_days {severity_ramp_days!r} is not a number") from None
    if not (ramp > 0.0):
        raise InjectError(f"inject: severity_ramp_days must be > 0, got {severity_ramp_days!r}")
    car = int(car)
    if subsystem == "pneumatic":
        car = 0
    elif not 1 <= car <= S.MAX_CAR:
        raise InjectError(f"inject: car must be 1..{S.MAX_CAR} for {subsystem!r}, got {car}")

    at = parse_ts(t_onset if t_onset is not None else (now if now is not None else state.clock_end), what="t_onset")
    cap = max_days_for(subsystem)
    t0, days = plan_window(at, state.clock_end, max_days=cap)
    base_seed = int(seed) if seed is not None else _stable_seed(train_id, component_id, fault_type, iso(at))

    from nebulax.sim.common import generate_service

    service = generate_service(
        days,
        np.random.default_rng(_stable_seed(base_seed, train_id)),
        t0=t0,
        train_id=train_id,
    )
    run_id = f"inject_{train_id}_{component_id}_{int(at.value) // 1_000_000_000}"
    scenario = build_scenario(
        subsystem=subsystem,
        train_id=train_id,
        car=car,
        component_id=component_id,
        fault_type=fault_type,
        t_onset_s=(at - t0).total_seconds(),
        ramp_days=ramp,
        days=days,
        seed=base_seed,
        run_id=run_id,
    )

    t_start = time.perf_counter()
    long, features, events = _simulate(subsystem, scenario, service, np.random.default_rng(base_seed), STORE_EVERY)
    sim_s = time.perf_counter() - t_start

    fault_rows = scenario.fault_log_rows(service.t0)
    fault_log = S.coerce_fault_log(pd.DataFrame(fault_rows)) if fault_rows else S.empty_fault_log()

    from nebulax.demo import scoring  # lazy: the module is written by a parallel agent

    fw = scoring.load_fitted(subsystem, held_out_train=train_id, **_scores_dir_kwarg(scoring, state))
    scores = scoring.score_frames(
        fw, long, features, train_id=train_id, car=car, component_id=component_id
    )
    episodes = _episodes_from(scoring, scores, fault_log)
    elapsed_s = time.perf_counter() - t_start

    key: ComponentKey = (train_id, car, subsystem, component_id)
    info = {
        "run_id": run_id,
        "train_id": train_id,
        "car": car,
        "subsystem": subsystem,
        "component_id": component_id,
        "fault_type": fault_type,
        "severity_ramp_days": ramp,
        "seed": base_seed,
        "t_onset": iso(at),
        "window": {
            "start": iso(t0),
            "end": iso(t0 + pd.Timedelta(days=days)),
            "days": days,
            "max_days": cap,
        },
        "sim_seconds": round(sim_s, 3),
    }
    recs = state.apply_overlay(
        key=key,
        scores=scores,
        episodes=episodes,
        long=long,
        features=features,
        fault_rows=[
            {
                "run_id": run_id,
                "train_id": train_id,
                "car": car,
                "subsystem": subsystem,
                "component_id": component_id,
                "fault_type": fault_type,
                "t_onset": iso(r.get("t_onset")),
                "t_failure": iso(r.get("t_failure")),
                "t_functional_failure": iso(r.get("t_functional_failure")),
                "gamma": float(r.get("gamma", 0.0)),
                "shape": str(r.get("shape", "")),
                "injected": True,
            }
            for r in fault_rows
        ],
        info=info,
    )

    ts_ns = int(parse_ts(now).value) if now is not None else int(state.clock_end.value)
    return {
        **info,
        "n_rows": int(len(scores)),
        "n_telemetry_rows": int(len(long)),
        "n_events": int(len(events)),
        "episodes": [state.alert_payload(r, max(ts_ns, r.t_start_ns)) for r in recs],
        "elapsed_s": round(elapsed_s, 3),
        "note": (
            f"re-simulated {days} day(s) from {iso(t0)} (cap {cap} day(s) for {subsystem}: about "
            f"{SECONDS_PER_SIM_DAY.get(subsystem, 2.0):.3g} s per simulated day against this "
            f"route's {INJECT_BUDGET_S:.0f} s budget); rows scored before {iso(t0)} are kept as "
            f"they were, so the charts keep their history. The service timetable for this window "
            f"is regenerated from the train seed, so it is statistically identical to but not "
            f"bit-identical with the stored run"
        ),
    }
