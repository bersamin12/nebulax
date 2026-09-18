"""Shared machinery for the three behavioural-twin simulators (door, pneumatic, bearing).

Everything here is numpy-vectorised, discrete-time and ODE-solver-free.

What lives here
---------------
* :class:`DegradationTrajectory` - the single severity scalar ``s(t) in [0, 1]`` that every
  fault map keys off, with ``shape`` in ``{"power", "step", "shock"}`` and per-cycle jitter.
* :class:`ShockProcess` / :class:`ShockSeries` - discrete shock damage (door closing-position
  ceiling degradation) [rail_phm 4.1].
* :func:`generate_service` -> :class:`Service` - ONE service day shared by every subsystem of
  a train, as a table of run/dwell/depot segments with speed, passenger load and ambient.
* :func:`speed_profile` - trapezoidal speed to 22 m/s.
* :class:`AmbientProfile` - Singapore diurnal ambient (routine 24-33 C, stress 19-37 C).
* :class:`SensorSpec` / :func:`apply_sensor` - noise, quantisation, burst NaN dropouts, drift.
* :func:`sample_scenarios` - 30-day run scenarios with the rail-literature fault priors.
* :class:`SimulateFn` - the ``simulate(...)`` contract every ``sim/<subsystem>.py`` implements.

Physical constants: where the approved plan and ``docs/research/rail_phm.md`` disagree,
**rail_phm.md wins**. Two places where that bites here:

* **Ambient.** The plan said ``30 + 4 sin(.) -> 26-34 C``. rail_phm.md section 0 gives the
  Singapore normals (daily min 24.3-25.7 C, daily max 30.5-32.4 C, record 19/37 C) and
  ``configs/model_ladder.yaml:operating_environment`` fixes the simulator sweep at
  **24-33 C routine, 19-37 C stress**, with a *required diurnal shape*. Implemented as
  :meth:`AmbientProfile.singapore_routine` / :meth:`AmbientProfile.singapore_stress`.
* **Door fault priors.** The plan implied a uniform-ish fault mix; rail_phm 4.1 re-weights to
  the ScotRail Class 380 observed distribution (electrical/switch class 0.45, obstruction
  0.15, friction 0.15, backlash 0.10, brush wear 0.10, misalignment 0.05), and adds the
  ``nff`` (no fault found) label. Implemented in :data:`FAULT_PRIORS`.

Time convention
---------------
Simulator-internal time ``t`` is **float seconds since the run start** (``Service.t0``, a
UTC ``pd.Timestamp``). Convert once, at the edge, with :func:`to_timestamp`; the schema
stores ``datetime64[ms, UTC]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from nebulax.schema import (
    DEGRADATION_SHAPES,
    FAULT_SUBSYSTEMS,
    FAULT_TYPES,
    COMPONENT_IDS,
)

__all__ = [
    "SEC_PER_DAY",
    "V_MAX_MS",
    "DegradationTrajectory",
    "ShockProcess",
    "ShockSeries",
    "AmbientProfile",
    "ServiceParams",
    "Service",
    "Timeline",
    "RunCycle",
    "speed_profile",
    "passenger_load",
    "ambient",
    "generate_service",
    "SensorSpec",
    "apply_sensor",
    "Scenario",
    "FAULT_PRIORS",
    "STEP_FAULTS",
    "SHOCK_FAULTS",
    "sample_trajectory",
    "sample_scenarios",
    "to_timestamp",
    "SimulateFn",
    "SimResult",
]

SEC_PER_DAY: Final[float] = 86_400.0
V_MAX_MS: Final[float] = 22.0  # 22 m/s ~ 79 km/h, the plan's line speed

Shape = Literal["power", "step", "shock"]


def to_timestamp(t_s: np.ndarray | float, t0: pd.Timestamp) -> pd.DatetimeIndex | pd.Timestamp:
    """Seconds-since-``t0`` -> ``datetime64[ms, UTC]``. The only place seconds meet the schema."""
    ms = np.rint(np.asarray(t_s, dtype=np.float64) * 1000.0).astype("int64")
    if ms.ndim == 0:
        return pd.Timestamp(t0) + pd.Timedelta(int(ms), unit="ms")
    return pd.DatetimeIndex(pd.Timestamp(t0) + pd.to_timedelta(ms, unit="ms")).as_unit("ms")


# --------------------------------------------------------------------------------------
# Degradation
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class ShockSeries:
    """Sampled shock events: arrival times (s) and fractional damage magnitudes."""

    times: np.ndarray
    magnitudes: np.ndarray

    def cumulative(self, t: np.ndarray | float) -> np.ndarray:
        """Cumulative damage in ``[0, 1]`` at each ``t`` (right-continuous step function)."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
        if self.times.size == 0:
            return np.zeros_like(t_arr)
        cum = np.cumsum(self.magnitudes)
        idx = np.searchsorted(self.times, t_arr, side="right")
        out = np.where(idx > 0, cum[np.clip(idx - 1, 0, cum.size - 1)], 0.0)
        return np.clip(out, 0.0, 1.0)

    def count(self, t_start: float, t_end: float) -> int:
        """Number of shocks in ``[t_start, t_end)`` - the per-cycle ``shock_count`` feature."""
        return int(np.searchsorted(self.times, t_end, "left") - np.searchsorted(self.times, t_start, "left"))


@dataclass(frozen=True, slots=True)
class ShockProcess:
    """Shock damage process for the door closing-position ceiling [rail_phm 4.1, R69/R71].

    Inter-arrival ``U(3600, 7200) s``, magnitude ``U(0.02, 0.14)`` fractional reduction of the
    maximum closing position; functional failure when ``pos_close_max < 0.10 * optimal``.
    """

    inter_arrival_s: tuple[float, float] = (3600.0, 7200.0)
    magnitude: tuple[float, float] = (0.02, 0.14)

    def sample(self, t_end: float, rng: np.random.Generator, t_start: float = 0.0) -> ShockSeries:
        """Sample the whole shock series over ``[t_start, t_end)`` in one vectorised draw."""
        span = max(float(t_end) - float(t_start), 0.0)
        if span <= 0.0:
            return ShockSeries(np.empty(0), np.empty(0))
        mean_gap = 0.5 * (self.inter_arrival_s[0] + self.inter_arrival_s[1])
        n = int(span / max(mean_gap, 1e-9) * 1.5) + 8
        gaps = rng.uniform(self.inter_arrival_s[0], self.inter_arrival_s[1], size=n)
        times = float(t_start) + np.cumsum(gaps)
        keep = times < t_end
        times = times[keep]
        mags = rng.uniform(self.magnitude[0], self.magnitude[1], size=times.size)
        return ShockSeries(times, mags)


@dataclass(slots=True)
class DegradationTrajectory:
    """One injected fault and its severity law ``s(t) in [0, 1]``.

    ``shape``
        ``"power"``  ``s = ((t - t_onset) / (t_failure - t_onset)) ** gamma`` clipped to [0, 1]
        (gamma = 1 linear wear, 2-3 accelerating).
        ``"step"``   ``s = 1`` for ``t >= t_onset`` (stuck valve, failed limit switch).
        ``"shock"``  ``s`` = cumulative damage of :attr:`shocks` (door closing-position ceiling).

    ``jitter_sigma`` is the per-cycle multiplicative jitter ``x (1 + N(0, sigma))`` applied
    only when a generator is passed to :meth:`severity`; the underlying law stays monotone,
    which is what the tests assert.

    ``t_functional_failure`` is filled in by the simulator when its functional criterion trips
    (closing time > 5 s, LPS > 60 s in service, ``T_box`` > 90 C, ...). It is ground truth and
    goes straight into the fault log.
    """

    fault_type: str
    subsystem: str
    component_id: str
    t_onset: float
    t_failure: float
    gamma: float = 1.0
    shape: Shape = "power"
    jitter_sigma: float = 0.05
    shocks: ShockSeries | None = None
    t_functional_failure: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.subsystem not in FAULT_SUBSYSTEMS:
            raise ValueError(
                f"DegradationTrajectory: subsystem {self.subsystem!r} must be one of {list(FAULT_SUBSYSTEMS)}"
            )
        if self.fault_type not in FAULT_TYPES[self.subsystem]:
            raise ValueError(
                f"DegradationTrajectory: fault_type {self.fault_type!r} is not legal for "
                f"{self.subsystem!r}; expected one of {list(FAULT_TYPES[self.subsystem])}"
            )
        if self.component_id not in COMPONENT_IDS[self.subsystem]:
            raise ValueError(
                f"DegradationTrajectory: component_id {self.component_id!r} is not legal for "
                f"{self.subsystem!r}; expected one of {list(COMPONENT_IDS[self.subsystem])}"
            )
        if self.shape not in DEGRADATION_SHAPES:
            raise ValueError(
                f"DegradationTrajectory: shape {self.shape!r} must be one of {list(DEGRADATION_SHAPES)}"
            )
        if self.gamma <= 0.0:
            raise ValueError(f"DegradationTrajectory: gamma must be > 0, got {self.gamma}")
        if self.t_failure < self.t_onset:
            raise ValueError(
                f"DegradationTrajectory: t_failure ({self.t_failure}) < t_onset ({self.t_onset})"
            )
        if self.shape == "shock" and self.shocks is None:
            raise ValueError("DegradationTrajectory: shape='shock' requires a sampled ShockSeries")

    @property
    def duration_s(self) -> float:
        """Onset-to-failure span in seconds."""
        return float(self.t_failure - self.t_onset)

    def severity(self, t: np.ndarray | float, rng: np.random.Generator | None = None) -> np.ndarray:
        """Severity at ``t`` (seconds since run start). Pass ``rng`` to add per-sample jitter.

        Always returns a ``float64`` array (shape of ``t``, scalar -> shape ``()``), clipped to
        ``[0, 1]``, zero before ``t_onset``.
        """
        t_arr = np.asarray(t, dtype=np.float64)
        flat = np.atleast_1d(t_arr)
        if self.shape == "step":
            s = (flat >= self.t_onset).astype(np.float64)
        elif self.shape == "shock":
            assert self.shocks is not None
            s = self.shocks.cumulative(flat)
        else:
            span = max(self.duration_s, 1e-9)
            u = np.clip((flat - self.t_onset) / span, 0.0, 1.0)
            s = u**self.gamma
        if rng is not None and self.jitter_sigma > 0.0:
            s = s * (1.0 + rng.normal(0.0, self.jitter_sigma, size=s.shape))
        s = np.clip(s, 0.0, 1.0)
        return s.reshape(t_arr.shape) if t_arr.ndim else s[0]

    def is_active(self, t: np.ndarray | float) -> np.ndarray:
        """``t >= t_onset`` elementwise."""
        return np.asarray(t, dtype=np.float64) >= self.t_onset

    def to_fault_log_row(
        self,
        *,
        run_id: str,
        train_id: str,
        car: int,
        t0: pd.Timestamp,
        params_json: str | None = None,
    ) -> dict[str, Any]:
        """One row shaped for :data:`nebulax.schema.FAULT_LOG_COLUMNS`."""
        import json

        return {
            "run_id": run_id,
            "train_id": train_id,
            "car": np.int8(car),
            "subsystem": self.subsystem,
            "component_id": self.component_id,
            "fault_type": self.fault_type,
            "t_onset": to_timestamp(self.t_onset, t0),
            "t_failure": to_timestamp(self.t_failure, t0),
            "t_functional_failure": (
                to_timestamp(self.t_functional_failure, t0)
                if self.t_functional_failure is not None
                else pd.NaT
            ),
            "gamma": np.float32(self.gamma),
            "shape": self.shape,
            "params_json": params_json if params_json is not None else json.dumps(self.params, default=str),
        }


# --------------------------------------------------------------------------------------
# Ambient - Singapore envelope [rail_phm section 0]
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AmbientProfile:
    """Diurnal ambient-temperature driver.

    Defaults are the Singapore **routine** case: daily minimum 24.3 C (climatological Jan/Dec
    mean daily min) and daily maximum 32.4 C (Apr mean daily max), peaking mid-afternoon,
    with a per-day offset and small measurement noise, hard-clipped into the sweep envelope
    ``[24, 33] C`` from ``configs/model_ladder.yaml:operating_environment.simulator_sweep``.
    A constant ambient is forbidden: ``diurnal_shape_required: true`` [rail_phm 0].
    """

    daily_min_c: float = 24.3
    daily_max_c: float = 32.4
    peak_hour: float = 15.0
    day_offset_sigma_c: float = 0.8
    noise_sigma_c: float = 0.10
    envelope_c: tuple[float, float] = (24.0, 33.0)

    @classmethod
    def singapore_routine(cls) -> "AmbientProfile":
        """24-33 C routine sweep [rail_phm 0, R152]."""
        return cls()

    @classmethod
    def singapore_stress(cls) -> "AmbientProfile":
        """19-37 C stress sweep, the Singapore record min/max [rail_phm 0, R153]."""
        return cls(daily_min_c=21.5, daily_max_c=35.5, day_offset_sigma_c=1.5, envelope_c=(19.0, 37.0))

    @classmethod
    def regulatory_t1(cls) -> "AmbientProfile":
        """Zone T1 design envelope -25..+40 C [rail_phm 0, R163 cl. 4.2.6.1]. Not a climate."""
        return cls(daily_min_c=-25.0, daily_max_c=40.0, day_offset_sigma_c=0.0, envelope_c=(-25.0, 40.0))

    def day_offsets(self, days: int, rng: np.random.Generator | None) -> np.ndarray:
        """Per-day temperature offsets (one draw per calendar day)."""
        if rng is None or self.day_offset_sigma_c <= 0.0:
            return np.zeros(int(days) + 1)
        return rng.normal(0.0, self.day_offset_sigma_c, size=int(days) + 1)

    def temperature(
        self,
        t_s: np.ndarray | float,
        *,
        rng: np.random.Generator | None = None,
        offsets: np.ndarray | None = None,
    ) -> np.ndarray:
        """Ambient (deg C) at ``t_s`` seconds since run start, clipped into :attr:`envelope_c`.

        ``offsets`` (from :meth:`day_offsets`) keeps the per-day offset stable across repeated
        calls - simulators that evaluate the ambient in chunks MUST pass it.
        """
        t = np.atleast_1d(np.asarray(t_s, dtype=np.float64))
        hour = (t / 3600.0) % 24.0
        mean = 0.5 * (self.daily_max_c + self.daily_min_c)
        amp = 0.5 * (self.daily_max_c - self.daily_min_c)
        temp = mean + amp * np.cos(2.0 * np.pi * (hour - self.peak_hour) / 24.0)
        day = np.floor(t / SEC_PER_DAY).astype(np.int64)
        if offsets is None:
            offsets = self.day_offsets(int(day.max()) + 1 if day.size else 1, rng)
        temp = temp + offsets[np.clip(day, 0, offsets.size - 1)]
        if rng is not None and self.noise_sigma_c > 0.0:
            temp = temp + rng.normal(0.0, self.noise_sigma_c, size=temp.shape)
        temp = np.clip(temp, self.envelope_c[0], self.envelope_c[1])
        out = temp.astype(np.float64)
        return out if np.ndim(t_s) else out[0]


def ambient(
    t_s: np.ndarray | float,
    profile: AmbientProfile | None = None,
    rng: np.random.Generator | None = None,
    offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Convenience wrapper around :meth:`AmbientProfile.temperature`."""
    return (profile or AmbientProfile.singapore_routine()).temperature(t_s, rng=rng, offsets=offsets)


# --------------------------------------------------------------------------------------
# Speed profile and passenger load
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunCycle:
    """One station-to-station run: duration, cruise ceiling and the accel/brake rates."""

    duration_s: float
    v_max: float = V_MAX_MS
    accel: float = 1.0
    decel: float = 1.0

    @property
    def v_peak(self) -> float:
        """Peak speed actually reached (triangular profile when the run is too short)."""
        v_tri = self.duration_s / (1.0 / max(self.accel, 1e-9) + 1.0 / max(self.decel, 1e-9))
        return float(min(self.v_max, v_tri))


def speed_profile(cycle: RunCycle, dt: float) -> np.ndarray:
    """Trapezoidal speed profile (m/s) for one run, sampled every ``dt`` seconds.

    ``v(tau) = min(a*tau, v_peak, d*(D - tau))`` clipped at 0 - accelerate, cruise at the
    22 m/s ceiling, brake to a stand. Degenerates to a triangle when the run is too short to
    reach ``v_max``. Returns ``float64``, length ``max(1, round(duration/dt))``.
    """
    if dt <= 0.0:
        raise ValueError(f"speed_profile: dt must be > 0, got {dt}")
    n = max(int(round(cycle.duration_s / dt)), 1)
    tau = np.arange(n, dtype=np.float64) * dt
    v_peak = cycle.v_peak
    v = np.minimum(np.minimum(cycle.accel * tau, v_peak), cycle.decel * (cycle.duration_s - tau))
    return np.clip(v, 0.0, None)


def passenger_load(
    hour_of_day: np.ndarray | float,
    rng: np.random.Generator | None = None,
    *,
    base: float = 0.18,
    am_peak_hour: float = 8.0,
    pm_peak_hour: float = 18.25,
    am_amp: float = 0.62,
    pm_amp: float = 0.70,
    width_h: float = 1.4,
    noise_sigma: float = 0.05,
) -> np.ndarray:
    """Double-peak passenger load fraction in ``[0.02, 1.0]`` (0 = tare ~40 t, 1 = crush ~64 t).

    Two Gaussian humps at the morning and evening peaks on a flat off-peak base.
    """
    h = np.atleast_1d(np.asarray(hour_of_day, dtype=np.float64)) % 24.0
    d_am = np.minimum(np.abs(h - am_peak_hour), 24.0 - np.abs(h - am_peak_hour))
    d_pm = np.minimum(np.abs(h - pm_peak_hour), 24.0 - np.abs(h - pm_peak_hour))
    load = base + am_amp * np.exp(-0.5 * (d_am / width_h) ** 2) + pm_amp * np.exp(-0.5 * (d_pm / width_h) ** 2)
    if rng is not None and noise_sigma > 0.0:
        load = load + rng.normal(0.0, noise_sigma, size=load.shape)
    load = np.clip(load, 0.02, 1.0)
    return load if np.ndim(hour_of_day) else load[0]


# --------------------------------------------------------------------------------------
# Service generator
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ServiceParams:
    """Timetable parameters for one train-day (plan "Level 2 - Simulators")."""

    service_start_h: float = 5.5  # 05:30
    service_end_h: float = 24.5  # 00:30 the next morning
    run_s: tuple[float, float] = (90.0, 150.0)
    dwell_s: tuple[float, float] = (25.0, 45.0)
    v_max: float = V_MAX_MS
    accel: float = 1.0
    decel: float = 1.0


SEGMENT_KINDS: Final[tuple[str, ...]] = ("run", "dwell", "depot")

SEGMENT_COLUMNS: Final[tuple[str, ...]] = (
    "seg_id",
    "day",
    "kind",
    "t_start",
    "t_end",
    "duration_s",
    "v_peak",
    "distance_m",
    "load_frac",
    "T_amb",
    "hour_of_day",
    "in_service",
)


@dataclass(slots=True)
class Timeline:
    """Per-sample service context, all arrays the same length.

    ``t`` is seconds since :attr:`Service.t0`; ``speed`` m/s; ``load_frac`` 0-1; ``T_amb`` deg C;
    ``in_service`` bool; ``seg_id`` indexes :attr:`Service.segments`.
    """

    t: np.ndarray
    speed: np.ndarray
    load_frac: np.ndarray
    T_amb: np.ndarray
    in_service: np.ndarray
    seg_id: np.ndarray
    dt: float
    t0: pd.Timestamp

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def timestamp(self) -> pd.DatetimeIndex:
        """The ``datetime64[ms, UTC]`` index for these samples."""
        return to_timestamp(self.t, self.t0)  # type: ignore[return-value]

    def to_frame(self) -> pd.DataFrame:
        """Wide context frame (``timestamp, speed, load_frac, T_amb, in_service``)."""
        return pd.DataFrame(
            {
                "timestamp": self.timestamp,
                "speed": self.speed.astype("float32"),
                "load_frac": self.load_frac.astype("float32"),
                "T_amb": self.T_amb.astype("float32"),
                "in_service": self.in_service.astype("float32"),
            }
        )


@dataclass(slots=True)
class Service:
    """One train's service over ``days`` days: the segment table plus its samplers.

    ``segments`` columns: :data:`SEGMENT_COLUMNS`. ``kind`` is ``run`` (station to station),
    ``dwell`` (doors open at a platform) or ``depot`` (stabled, out of service). Doors,
    pneumatics and bearings of the same train MUST be simulated against the same
    :class:`Service` instance so their confounders agree.
    """

    segments: pd.DataFrame
    days: int
    t0: pd.Timestamp
    params: ServiceParams
    ambient_profile: AmbientProfile
    day_offsets: np.ndarray
    train_id: str = "T01"

    @property
    def duration_s(self) -> float:
        """Total simulated span in seconds."""
        return float(self.days) * SEC_PER_DAY

    def dwells(self) -> pd.DataFrame:
        """Dwell segments only - one door cycle happens per dwell."""
        return self.segments[self.segments["kind"] == "dwell"]

    def runs(self) -> pd.DataFrame:
        """Run segments only."""
        return self.segments[self.segments["kind"] == "run"]

    def timeline(self, dt: float = 1.0, t_start: float = 0.0, t_end: float | None = None) -> Timeline:
        """Sample the service context every ``dt`` seconds over ``[t_start, t_end)``.

        Fully vectorised. Intended for the 1 Hz simulators (pneumatic, bearing); the 100 Hz
        door simulator builds per-cycle arrays and reads the context per dwell instead of
        materialising 30 days at 100 Hz.
        """
        if dt <= 0.0:
            raise ValueError(f"Service.timeline: dt must be > 0, got {dt}")
        t_end = self.duration_s if t_end is None else float(t_end)
        n = max(int(round((t_end - t_start) / dt)), 0)
        t = t_start + np.arange(n, dtype=np.float64) * dt
        seg_start = self.segments["t_start"].to_numpy(dtype=np.float64)
        idx = np.clip(np.searchsorted(seg_start, t, side="right") - 1, 0, len(self.segments) - 1)
        tau = t - seg_start[idx]
        dur = self.segments["duration_s"].to_numpy(dtype=np.float64)[idx]
        v_peak = self.segments["v_peak"].to_numpy(dtype=np.float64)[idx]
        is_run = (self.segments["kind"].to_numpy() == "run")[idx]
        a, d = self.params.accel, self.params.decel
        v = np.minimum(np.minimum(a * tau, v_peak), d * (dur - tau))
        speed = np.where(is_run, np.clip(v, 0.0, None), 0.0)
        load = self.segments["load_frac"].to_numpy(dtype=np.float64)[idx]
        in_service = self.segments["in_service"].to_numpy(dtype=bool)[idx]
        t_amb = self.ambient_profile.temperature(t, offsets=self.day_offsets)
        return Timeline(
            t=t,
            speed=speed,
            load_frac=np.where(in_service, load, 0.0),
            T_amb=np.asarray(t_amb, dtype=np.float64),
            in_service=in_service,
            seg_id=self.segments["seg_id"].to_numpy()[idx],
            dt=float(dt),
            t0=self.t0,
        )

    def context_long(self, dt: float = 1.0, *, source: str = "sim", run_id: str = "run0") -> pd.DataFrame:
        """The train-context signals as schema-conformant long rows (subsystem ``"train"``)."""
        from nebulax.schema import coerce_long

        tl = self.timeline(dt)
        wide = tl.to_frame()
        frames = []
        for sig in ("speed", "load_frac", "T_amb", "in_service"):
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": wide["timestamp"],
                        "source": source,
                        "run_id": run_id,
                        "train_id": self.train_id,
                        "car": np.int8(0),
                        "subsystem": "train",
                        "component_id": "train",
                        "signal": sig,
                        "value": wide[sig].to_numpy(dtype=np.float32),
                    }
                )
            )
        return coerce_long(pd.concat(frames, ignore_index=True))


def generate_service(
    days: int,
    rng: np.random.Generator,
    ambient_profile: AmbientProfile | None = None,
    *,
    params: ServiceParams | None = None,
    t0: pd.Timestamp | str = "2026-09-01T00:00:00Z",
    train_id: str = "T01",
) -> Service:
    """Build one train's timetable: runs ``U(90, 150) s`` and dwells ``U(25, 45) s`` between
    05:30 and 00:30, the rest of the day stabled in the depot.

    Each segment carries its peak speed, distance, passenger load (double-peak profile) and
    mean ambient (Singapore diurnal). One call per train; every subsystem of that train shares
    the result, which is what keeps the operating-condition confounders honest.
    """
    if days <= 0:
        raise ValueError(f"generate_service: days must be >= 1, got {days}")
    p = params or ServiceParams()
    amb = ambient_profile or AmbientProfile.singapore_routine()
    start = pd.Timestamp(t0)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    start = start.tz_convert("UTC")
    offsets = amb.day_offsets(days, rng)

    total = float(days) * SEC_PER_DAY
    kinds: list[str] = []
    t_start: list[float] = []
    t_end: list[float] = []

    def _add(kind: str, a: float, b: float) -> None:
        if b - a > 1e-9:
            kinds.append(kind)
            t_start.append(float(a))
            t_end.append(float(b))

    cursor = 0.0
    for day in range(int(days)):
        day0 = day * SEC_PER_DAY
        svc_start = day0 + p.service_start_h * 3600.0
        svc_end = min(day0 + p.service_end_h * 3600.0, total)
        if svc_start >= total:
            break
        if svc_start > cursor:
            _add("depot", cursor, svc_start)
            cursor = svc_start
        span = svc_end - svc_start
        n_pairs = int(span / (p.run_s[0] + p.dwell_s[0])) + 2
        runs = rng.uniform(p.run_s[0], p.run_s[1], size=n_pairs)
        dwells = rng.uniform(p.dwell_s[0], p.dwell_s[1], size=n_pairs)
        durations = np.empty(2 * n_pairs, dtype=np.float64)
        durations[0::2] = runs
        durations[1::2] = dwells
        edges = svc_start + np.concatenate([[0.0], np.cumsum(durations)])
        # only whole segments: a run truncated by the end of service would break the
        # U(90, 150) s contract the feature code relies on.
        keep = edges[1:] <= svc_end
        n_keep = int(np.argmin(keep)) if not keep.all() else keep.size
        for i in range(n_keep):
            _add("run" if i % 2 == 0 else "dwell", edges[i], edges[i + 1])
        cursor = float(edges[n_keep]) if n_keep else cursor
    if cursor < total:
        _add("depot", cursor, total)

    ts = np.asarray(t_start, dtype=np.float64)
    te = np.asarray(t_end, dtype=np.float64)
    kind_arr = np.asarray(kinds, dtype=object)
    dur = te - ts
    is_run = kind_arr == "run"
    v_tri = dur / (1.0 / max(p.accel, 1e-9) + 1.0 / max(p.decel, 1e-9))
    v_peak = np.where(is_run, np.minimum(p.v_max, v_tri), 0.0)
    # trapezoid area = distance
    dist = np.where(is_run, v_peak * dur - 0.5 * v_peak**2 * (1.0 / p.accel + 1.0 / p.decel), 0.0)
    hour = (ts / 3600.0) % 24.0
    load = passenger_load(hour, rng)
    in_service = kind_arr != "depot"
    t_amb = amb.temperature(0.5 * (ts + te), offsets=offsets)

    segments = pd.DataFrame(
        {
            "seg_id": np.arange(ts.size, dtype=np.int64),
            "day": np.floor(ts / SEC_PER_DAY).astype(np.int32),
            "kind": pd.Categorical(kind_arr.astype(str), categories=list(SEGMENT_KINDS)),
            "t_start": ts,
            "t_end": te,
            "duration_s": dur,
            "v_peak": v_peak,
            "distance_m": np.clip(dist, 0.0, None),
            "load_frac": np.where(in_service, load, 0.0),
            "T_amb": np.asarray(t_amb, dtype=np.float64),
            "hour_of_day": hour,
            "in_service": in_service,
        }
    )
    return Service(
        segments=segments,
        days=int(days),
        t0=start,
        params=p,
        ambient_profile=amb,
        day_offsets=offsets,
        train_id=train_id,
    )


# --------------------------------------------------------------------------------------
# Sensor model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SensorSpec:
    """Measurement chain applied to a clean simulated signal.

    noise -> bias+drift -> quantisation -> clipping -> burst NaN dropouts, in that order.
    ``dropout_burst_prob`` is the per-sample probability of *starting* a dropout burst whose
    length is geometric with mean ``dropout_len_mean`` samples; ``NaN`` is the schema's dropout
    marker. ``drift_per_day`` is a slow linear sensor drift in signal units per day.
    """

    noise_sigma: float = 0.0
    bias: float = 0.0
    drift_per_day: float = 0.0
    quantum: float = 0.0
    dropout_burst_prob: float = 0.0
    dropout_len_mean: float = 5.0
    clip: tuple[float, float] | None = None

    def with_(self, **changes: Any) -> "SensorSpec":
        """Copy with fields replaced (sims keep a dict of per-signal specs)."""
        return replace(self, **changes)


def apply_sensor(
    x: np.ndarray,
    spec: SensorSpec,
    rng: np.random.Generator,
    dt: float = 1.0,
    t0: float = 0.0,
) -> np.ndarray:
    """Apply :class:`SensorSpec` to a clean signal, returning ``float32`` with ``NaN`` dropouts.

    ``dt``/``t0`` (seconds) only matter for the drift term. Never mutates ``x``.
    """
    y = np.asarray(x, dtype=np.float64).copy()
    n = y.size
    if n == 0:
        return y.astype(np.float32)
    if spec.noise_sigma > 0.0:
        y += rng.normal(0.0, spec.noise_sigma, size=y.shape)
    if spec.bias != 0.0 or spec.drift_per_day != 0.0:
        t = t0 + np.arange(n, dtype=np.float64) * dt
        y += spec.bias + spec.drift_per_day * (t / SEC_PER_DAY)
    if spec.quantum > 0.0:
        y = np.rint(y / spec.quantum) * spec.quantum
    if spec.clip is not None:
        y = np.clip(y, spec.clip[0], spec.clip[1])
    if spec.dropout_burst_prob > 0.0:
        starts = np.flatnonzero(rng.random(n) < spec.dropout_burst_prob)
        if starts.size:
            mean_len = max(spec.dropout_len_mean, 1.0)
            lengths = rng.geometric(1.0 / mean_len, size=starts.size)
            delta = np.zeros(n + 1, dtype=np.int32)
            np.add.at(delta, starts, 1)
            np.add.at(delta, np.minimum(starts + lengths, n), -1)
            mask = np.cumsum(delta[:-1]) > 0
            y[mask] = np.nan
    return y.astype(np.float32)


# --------------------------------------------------------------------------------------
# Scenario sampling
# --------------------------------------------------------------------------------------

#: Fault-type priors per subsystem. The door row is the ScotRail Class 380 observed
#: distribution [rail_phm 4.1, R73] - electrical/switch faults dominate, not mechanical wear.
#: Pneumatic and bearing rows are OUR assumptions (MetroPT-3 contains air leaks only).
FAULT_PRIORS: Final[dict[str, dict[str, float]]] = {
    "door": {
        "limit_switch": 0.25,
        "dcu_dropout": 0.20,
        "obstruction": 0.15,
        "friction": 0.15,
        "backlash": 0.10,
        "brush_wear": 0.10,
        "misalignment": 0.05,
    },
    "pneumatic": {
        "air_leak": 0.35,
        "oil_leak": 0.20,
        "compressor_wear": 0.15,
        "dryer_valve_stuck": 0.10,
        "clogged_filter": 0.10,
        "valve_leak_inlet": 0.05,
        "valve_leak_outlet": 0.05,
    },
    "bearing": {
        "bearing_degradation": 0.45,
        "outer_race": 0.15,
        "inner_race": 0.15,
        "hot_axle_box": 0.15,
        "sensor_offset": 0.05,
        "sensor_stuck": 0.05,
    },
}

#: Faults whose severity is a step, not a ramp (stuck valve, failed switch, stuck sensor).
STEP_FAULTS: Final[frozenset[str]] = frozenset({"limit_switch", "dryer_valve_stuck", "sensor_stuck", "sensor_offset"})
#: Faults driven by a discrete shock process rather than a continuous law [rail_phm 4.1].
SHOCK_FAULTS: Final[frozenset[str]] = frozenset({"shock_wear"})
#: Acute faults: short onset-to-failure span (hours, not days).
ACUTE_FAULTS: Final[dict[str, tuple[float, float]]] = {"hot_axle_box": (6.0 / 24.0, 48.0 / 24.0)}


@dataclass(slots=True)
class Scenario:
    """One simulator run: which component, how long, and the faults injected into it.

    ``healthy=True`` means ``faults == ()`` - roughly ``p_healthy`` of all runs, so the
    benchmark has a real negative class.
    """

    run_id: str
    subsystem: str
    train_id: str
    car: int
    component_id: str
    days: int
    healthy: bool
    faults: tuple[DegradationTrajectory, ...]
    seed: int

    def fault_log_rows(self, t0: pd.Timestamp) -> list[dict[str, Any]]:
        """Fault-log rows for every injected fault of this scenario."""
        return [
            f.to_fault_log_row(run_id=self.run_id, train_id=self.train_id, car=self.car, t0=t0)
            for f in self.faults
        ]


def sample_trajectory(
    subsystem: str,
    component_id: str,
    rng: np.random.Generator,
    *,
    days: int = 30,
    fault_type: str | None = None,
    onset_days: tuple[float, float] = (5.0, 18.0),
    span_days: tuple[float, float] = (3.0, 12.0),
    gamma_range: tuple[float, float] = (1.0, 3.0),
) -> DegradationTrajectory:
    """Draw one fault trajectory: type from :data:`FAULT_PRIORS`, onset ``U(day 5, 18)``,
    onset-to-failure ``U(3, 12) d``, ``gamma ~ U(1, 3)``; step-shaped for the faults in
    :data:`STEP_FAULTS` and acute spans for :data:`ACUTE_FAULTS`."""
    if subsystem not in FAULT_PRIORS:
        raise ValueError(f"sample_trajectory: unknown subsystem {subsystem!r}; expected one of {list(FAULT_PRIORS)}")
    priors = FAULT_PRIORS[subsystem]
    if fault_type is None:
        names = list(priors)
        probs = np.asarray([priors[n] for n in names], dtype=np.float64)
        fault_type = str(rng.choice(names, p=probs / probs.sum()))
    onset_hi = min(onset_days[1], max(onset_days[0], days - 1.0))
    t_onset = float(rng.uniform(onset_days[0], onset_hi) * SEC_PER_DAY)
    lo, hi = ACUTE_FAULTS.get(fault_type, span_days)
    span = float(rng.uniform(lo, hi) * SEC_PER_DAY)
    shape: Shape = "step" if fault_type in STEP_FAULTS else ("shock" if fault_type in SHOCK_FAULTS else "power")
    shocks = None
    if shape == "shock":
        shocks = ShockProcess().sample(days * SEC_PER_DAY, rng, t_start=t_onset)
    return DegradationTrajectory(
        fault_type=fault_type,
        subsystem=subsystem,
        component_id=component_id,
        t_onset=t_onset,
        t_failure=t_onset + (0.0 if shape == "step" else span),
        gamma=float(rng.uniform(*gamma_range)) if shape == "power" else 1.0,
        shape=shape,
        shocks=shocks,
    )


def sample_scenarios(
    n_runs: int,
    subsystem: str,
    rng: np.random.Generator,
    p_healthy: float = 0.4,
    *,
    days: int = 30,
    train_ids: Sequence[str] | None = None,
    cars: Sequence[int] | None = None,
    components: Sequence[str] | None = None,
    run_id_prefix: str | None = None,
    max_faults: int = 1,
) -> list[Scenario]:
    """Sample ``n_runs`` scenarios for one subsystem (default 40 % healthy, 30-day runs).

    Faulty runs get ``1..max_faults`` trajectories on the same component, drawn from
    :data:`FAULT_PRIORS`. Train ids, cars and component ids cycle deterministically so a
    fleet of scenarios covers every component instead of piling onto one.
    """
    if subsystem not in FAULT_PRIORS:
        raise ValueError(f"sample_scenarios: unknown subsystem {subsystem!r}; expected one of {list(FAULT_PRIORS)}")
    if not 0.0 <= p_healthy <= 1.0:
        raise ValueError(f"sample_scenarios: p_healthy must lie in [0, 1], got {p_healthy}")
    if n_runs <= 0:
        raise ValueError(f"sample_scenarios: n_runs must be >= 1, got {n_runs}")
    comps = list(components) if components is not None else list(COMPONENT_IDS[subsystem])
    tids = list(train_ids) if train_ids is not None else ["T01"]
    car_list = list(cars) if cars is not None else [1, 2, 3, 4, 5, 6]
    prefix = run_id_prefix or f"{subsystem}"
    out: list[Scenario] = []
    for i in range(int(n_runs)):
        healthy = bool(rng.random() < p_healthy)
        cid = comps[i % len(comps)]
        faults: tuple[DegradationTrajectory, ...] = ()
        if not healthy:
            k = 1 if max_faults <= 1 else int(rng.integers(1, max_faults + 1))
            faults = tuple(
                sample_trajectory(subsystem, cid, rng, days=days) for _ in range(k)
            )
        out.append(
            Scenario(
                run_id=f"{prefix}_{i:04d}",
                subsystem=subsystem,
                train_id=tids[i % len(tids)],
                car=int(car_list[i % len(car_list)]),
                component_id=cid,
                days=int(days),
                healthy=healthy,
                faults=faults,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
        )
    return out


# --------------------------------------------------------------------------------------
# The simulate() contract
# --------------------------------------------------------------------------------------

#: What every ``simulate()`` returns: ``(long, features, events)``.
SimResult = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


@runtime_checkable
class SimulateFn(Protocol):
    """The contract of ``nebulax.sim.{door,pneumatic,bearing}.simulate``.

    ::

        simulate(params, faults, service, rng, store_every=10) -> (long, features, events)

    Parameters
    ----------
    params
        The subsystem's frozen ``Params`` dataclass (``DoorParams``, ``PneumaticParams``,
        ``BearingParams``), carrying every physical constant with its provenance in the
        docstring. Calibration variants are classmethods (``DoorParams.cranfield()``).
    faults
        The subsystem's ``Faults`` dataclass, or a sequence of
        :class:`DegradationTrajectory`, all for components of this simulator. A healthy run
        passes an empty container. The simulator maps ``s(t)`` onto physical parameters
        through a small documented table and writes ``t_functional_failure`` back onto each
        trajectory when its functional criterion trips.
    service
        A :class:`Service` from :func:`generate_service`. All subsystems of one train share
        the same instance, so speed, load and ambient confounders are consistent.
    rng
        ``numpy.random.Generator``. All stochasticity goes through it; the same seed must
        reproduce the run bit-for-bit.
    store_every
        Waveform thinning: store full-rate telemetry for every ``store_every``-th cycle
        (plus every cycle in the last 2 days before a failure). Per-cycle/window features
        are emitted for **all** cycles regardless.

    Returns
    -------
    long
        Schema-conformant long telemetry (:func:`nebulax.schema.validate_long` passes),
        including the train-context rows via :meth:`Service.context_long`.
    features
        One row per door cycle / compressor cycle / 5-min bearing window, with
        :data:`nebulax.schema.FEATURE_KEY_COLUMNS` + feature columns +
        :data:`nebulax.schema.LABEL_COLUMNS` joined from the fault log.
    events
        Discrete events (:data:`nebulax.schema.EVENT_TYPES`).

    The fault log itself is built by the caller from the scenario
    (:meth:`Scenario.fault_log_rows`) after ``simulate`` has written back
    ``t_functional_failure``.

    Budget: one simulated day must run in well under 10 s.
    """

    def __call__(
        self,
        params: Any,
        faults: Any,
        service: Service,
        rng: np.random.Generator,
        store_every: int = 10,
    ) -> SimResult:
        ...
