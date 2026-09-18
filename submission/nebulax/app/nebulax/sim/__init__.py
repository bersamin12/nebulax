"""Behavioural twin (Level 2): physics-lite simulators with fault injection.

``nebulax.sim.common`` is the frozen shared contract - degradation trajectories, the service
generator, the speed profile, the sensor model, scenario sampling and the ``simulate()``
protocol. The three subsystem modules (``door``, ``pneumatic``, ``bearing``) are written
against it and must not modify it.
"""

from __future__ import annotations

from nebulax.sim.common import (
    SEC_PER_DAY,
    V_MAX_MS,
    AmbientProfile,
    DegradationTrajectory,
    RunCycle,
    Scenario,
    SensorSpec,
    Service,
    ServiceParams,
    ShockProcess,
    ShockSeries,
    SimResult,
    SimulateFn,
    Timeline,
    ambient,
    apply_sensor,
    generate_service,
    passenger_load,
    sample_scenarios,
    sample_trajectory,
    speed_profile,
    to_timestamp,
)

__all__ = [
    "SEC_PER_DAY",
    "V_MAX_MS",
    "AmbientProfile",
    "DegradationTrajectory",
    "RunCycle",
    "Scenario",
    "SensorSpec",
    "Service",
    "ServiceParams",
    "ShockProcess",
    "ShockSeries",
    "SimResult",
    "SimulateFn",
    "Timeline",
    "ambient",
    "apply_sensor",
    "generate_service",
    "passenger_load",
    "sample_scenarios",
    "sample_trajectory",
    "speed_profile",
    "to_timestamp",
]
