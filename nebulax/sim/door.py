"""Bi-parting electric passenger door simulator (Level 2 behavioural twin).

One leaf of a two-leaf bi-parting door: brushed DC motor -> gearbox -> toothed belt/pulley
-> leaf, ``x in [0, stroke]`` with ``x = 0`` closed.  Discrete-time semi-implicit Euler at
``dt = 1 ms`` (no ODE solver), telemetry decimated to **50 Hz** for stored cycles, per-cycle
features for **every** cycle.

Provenance and the four places we depart from the plan
------------------------------------------------------
The starting point is the working first cut in ``sim/demo_runs.py::door_cycle`` - the same
plant, the same cascaded PI + position term with gains ``kp = 120, ki = 200, kx = 2500``, the
same ``dt = 1 ms`` inner loop, the same stiction branch, the same reverse-100 mm-and-retry
obstruction handler.  Where this module deviates, it deviates on purpose:

1. **Friction baseline and fault map are the plan's, not the demo's.**  ``demo_runs`` used
   ``F_c0 = 150 N, b0 = 250 N.s/m``; the approved plan says ``F_c0 = 40 N, b0 = 60 N.s/m``
   and ``F_c = F_c0 (1 + 2 s)``, ``b = b0 (1 + 1.5 s)``.  The plan wins because
   ``docs/research/rail_phm.md`` section 1.4 pins the obstruction operating point at
   **100-200 N** [R67][R85], and a door whose *healthy* Coulomb friction is already 150 N
   cannot represent that band at all.  Component and per-cycle scatter multiply the healthy
   baseline only and the wear increment is added on top, so a x1.25 unit does not also
   scale the fault by x1.25 and push severity 1 over the force ceiling on some seeds only.
2. **A contact-force ceiling exists, with inertia and seal feed-forward.**  ``demo_runs``
   had no current limit at all, so the motor could deliver ~4.8 kN and no mechanical fault
   could ever lengthen the closing time or bound the force on an obstacle.
   :attr:`DoorParams.f_limit_n` = 140 N is the force the leaf may exert on something that is
   *not* moving with it; the DCU adds back the inertia it is knowingly accelerating
   (``m_eff a_ref``, the profile's own feed-forward) and the seal it is knowingly
   compressing, but never friction.  A worn door therefore slows down instead of pushing
   harder: the cruise speed settles at ``(F_limit - F_c) / b``, closing time creeps up,
   and the plan's "closing time > 5 s" functional-failure criterion becomes reachable.
3. **The obstruction detector is relative, not a fixed force.**  A fixed trip level makes
   "friction raises the current" and "no phantom reversal" mutually exclusive: any wear
   ramp that reaches the level fires on 100 % of cycles.  Detection instead answers an
   *abrupt* rise of current, or drop of speed, against this cycle's own asymmetric baseline
   (rail_phm 1.2 [R65] condition-conditional threshold, [R67] transient on the closing
   current), with the 100 N band bottom only as an absolute floor.
4. **A profile governor and a stall watchdog.**  When the drive is on its ceiling the
   reference clock slows in proportion to the lag it has built up, so the profile never runs
   away from a force-limited leaf into a spurious tracking-error alarm; a leaf that is on
   the ceiling and not moving at all is aborted after
   :attr:`DoorParams.stall_timeout_s`.

Everything else that rail_phm.md corrects is implemented:

* **>= 2 s closing warning before movement** (PRM TSI 4.2.2.3.2 [R165]): every cycle has a
  warning phase of :attr:`DoorParams.warning_s` between the door-release drop and the first
  closing motion, exposed in telemetry as the falling edge of ``door_key`` and as the
  ``warning_time`` feature.  An injected obstruction therefore always sits *after* a
  legitimate warning.
* **Obstruction operating point 100-200 N, detected within 0.3 s** [R67][R85] - a *design
  assumption*, never quoted as an EN 14752 requirement.  ``obs_detect_s`` is emitted per
  cycle so detection latency can be reported against 0.3 s as a first-class metric, and
  ``obs_class`` carries soft (compliant bag) vs hard (rigid limb/case) instead of a binary.
* **First-class discretes** ``interlock``, ``door_key``, ``emergency_relay`` [R74].
* **Shock-driven closing-position degradation** [R69][R71]: ``shock_wear`` walks a hard stop
  inwards, ``pos_close_max`` is emitted per cycle, functional failure below 10 % of stroke.
  The shock process itself is latent: its per-cycle count is emitted as **``meta_shock_count``**
  (metadata, kept for analysis, never a model input - see
  :func:`nebulax.schema.feature_columns`) beside the observable estimator
  ``shock_jump_count_est`` from :func:`estimate_shock_jumps`, which a real DCU could compute from
  ``i_peak``/``pos_err_max`` alone.
* **``nff`` labelling**: a cycle that raises an alarm with no injected fault is labelled
  ``nff``, the 42.4 % no-fault-found class [R73].

Functional failure (plan "Door", OR of three criteria, first trip wins):
closing time > 5 s, **or** 3 phantom obstruction retries in one cycle, **or** ``door_fault``
twice within 10 cycles.  The trip time is written back onto the driving
:class:`~nebulax.sim.common.DegradationTrajectory` as ``t_functional_failure``.

Performance: every cycle of the run is integrated **simultaneously** as a batch of numpy
state vectors, so the Python-level loop runs once per 1 ms tick, not once per sample per
cycle.  One door-day (~440 cycles from the shared service generator) takes well under 10 s.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Final, Iterator, Sequence

import numpy as np
import pandas as pd

from nebulax import schema as S
from nebulax.sim.common import (
    SEC_PER_DAY,
    DegradationTrajectory,
    Scenario,
    SensorSpec,
    Service,
    apply_sensor,
    to_timestamp,
)

__all__ = [
    "DOOR_FAULT_FIELDS",
    "DoorFaults",
    "DoorParams",
    "SHOCK_EST_K",
    "SHOCK_EST_MIN_CYCLES",
    "SHOCK_EST_SMOOTH",
    "SHOCK_EST_WINDOW",
    "cycle_features",
    "estimate_shock_jumps",
    "simulate",
    "simulate_scenario",
]

#: Trailing median filter (cycles) of :func:`estimate_shock_jumps`: a passenger obstruction
#: perturbs a single cycle, a shock every cycle after it.
SHOCK_EST_SMOOTH: Final[int] = 9
#: Trailing window (cycles) of the observable shock estimator :func:`estimate_shock_jumps`.
SHOCK_EST_WINDOW: Final[int] = 50
#: Robust-sigma threshold of :func:`estimate_shock_jumps`.
SHOCK_EST_K: Final[float] = 6.0
#: Minimum trailing cycles before the estimator will fire at all (no baseline, no call).
SHOCK_EST_MIN_CYCLES: Final[int] = 10
#: Floor on the robust sigma, as a fraction of the trailing median: with a noiseless baseline
#: the IQR collapses to 0 and every rounding wobble would otherwise read as a shock.
_SHOCK_EST_REL_FLOOR: Final[float] = 1.0e-3
_IQR_TO_SIGMA: Final[float] = 1.349  # IQR of a normal = 1.349 sigma

#: ``DoorFaults`` field name per ``schema.FAULT_TYPES['door']`` entry.
DOOR_FAULT_FIELDS: Final[dict[str, str]] = {
    "friction": "friction",
    "brush_wear": "brush_wear",
    "backlash": "backlash",
    "misalignment": "misalignment",
    "obstruction": "obstruction_rate",
    "limit_switch": "limit_switch",
    "dcu_dropout": "dcu_dropout",
    "shock_wear": "shock_wear",
}

#: Ball-screw lead of the Cranfield linear-actuator rig (m).  Sets both the rotational-to-
#: linear gain of :meth:`DoorParams.cranfield` (``pulley_r_m = lead / 2 pi`` with
#: ``gear_ratio = 1``) and the spatial period of its misalignment/spalling ripple, so the two
#: cannot drift apart.  ``UNVERIFIED``: a 5 mm lead is the commonest size for a rig of this
#: class, not a measured value - ``scripts/calibrate_door.py`` overwrites it from the
#: measured ripple wavelength once the real CORD files exist.
CRANFIELD_SCREW_LEAD_M: Final[float] = 5.0e-3

#: Position-periodic (spalling/misalignment) amplitude gain for the Cranfield rig variant:
#: ``mis_amp = k_mis * f_c0 * s``.  **Calibrated** by ``scripts/calibrate_door.py`` against the
#: 8 real spalling stages of the CORD release: the measured current-ripple fraction grows
#: 1.04 -> 1.45 from stage 1 to stage 8, and reading that off the simulator's ``k_mis``
#: inversion curve gives 0.15 - a **twentieth** of the plan's 3.0, which would have put the
#: ripple ratio near 11x.  The ripple fraction is the one current observable that survives the
#: rig's stepper drive, because it is measured after detrending the current against position,
#: which removes the drive's large standing current from the ripple itself.  Applied to
#: :meth:`DoorParams.cranfield` only; the shipped 110 V door keeps the plan's ``k_mis``,
#: since it was never fitted on a ball-screw rig.  See ``docs/parameters.md`` (door section).
CRANFIELD_K_MIS: Final[float] = 0.15

#: Tracking error that reads as an obstacle, as a fraction of the stroke.  The plan's 20 mm
#: on the 0.725 m door leaf is 2.76 % of stroke; :meth:`DoorParams.cranfield` scales the same
#: fraction onto the rig's 0.10 m stroke rather than inheriting an absolute 20 mm, which on a
#: 0.10 m stroke is 20 % of full travel and can never be reached.
OBS_POS_ERR_STROKE_FRAC: Final[float] = 0.020 / 0.725

# Integrator modes (per cycle, vectorised).
_MOVE: Final[int] = 0
_REVERSE: Final[int] = 1
_PAUSE: Final[int] = 2
_DONE: Final[int] = 3


# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DoorParams:
    """Physical constants of one door leaf and its DCU.

    Defaults are the LTA-fleet electric door of the approved plan, corrected as described in
    the module docstring.  Every field is a plain SI quantity; the derived quantities
    (:attr:`gear_gain`, :attr:`force_per_amp`, :attr:`m_eff`, :attr:`i_limit_a`,
    :attr:`i_obs_a`) are properties so that a calibration agent can fit the raw constants
    and the thresholds move with them.
    """

    # --- geometry and drivetrain -------------------------------------------------------
    stroke_m: float = 0.725  # one leaf of a bi-parting door
    gear_ratio: float = 20.0
    pulley_r_m: float = 0.03
    eta: float = 0.75  # gearbox/belt efficiency
    m_leaf_kg: float = 40.0
    n_leaves: int = 2
    j_motor: float = 3.0e-5  # rotor inertia, kg.m^2

    # --- electrical --------------------------------------------------------------------
    r_ohm: float = 4.0
    k_t: float = 0.35  # N.m/A
    k_e: float = 0.35  # V.s/rad
    v_bus: float = 110.0  # LV bus
    r_temp_coeff: float = 0.0039  # copper, per K above 20 C (driven by ambient, see notes)

    # --- mechanical loads ---------------------------------------------------------------
    f_c0: float = 40.0  # Coulomb friction, plan value
    b0: float = 60.0  # viscous friction, plan value
    f_seal_n: float = 120.0  # door-seal compression at the closed end
    f_seal_allow_n: float = 200.0  # extra force the DCU may command inside the latching
    #                                zone, where obstruction detection is off anyway
    seal_zone_m: float = 0.015
    b_temp_coeff: float = -0.004  # grease thins as the motor/track warms, per K

    # --- motion reference ---------------------------------------------------------------
    v_ref: float = 0.40  # m/s cruise
    a_ref: float = 0.8  # m/s^2
    kp: float = 120.0  # velocity P  (demo_runs gains, kept)
    ki: float = 200.0  # velocity I
    kx: float = 2500.0  # position term

    # --- force ceiling and obstruction detection ---------------------------------------
    f_limit_n: float = 140.0  # contact-force ceiling, inside the 100-200 N band [R67][R85]
    f_obs_trip_n: float = 100.0  # absolute floor: nothing below the band bottom is an obstacle
    f_obs_delta_n: float = 60.0  # abrupt rise above THIS cycle's own current baseline
    obs_base_tau_s: float = 0.40  # baseline time constant *towards* an alarming change
    obs_base_fast_s: float = 0.05  # ... and away from one (asymmetric EWMA, see _integrate)
    obs_base_seed: bool = True  # seed the in-cycle baselines with the operating point at the
    #                             instant the current path arms, instead of starting them at
    #                             zero.  A zero-seeded asymmetric EWMA needs
    #                             ``obs_base_tau_s . ln(i_cruise / i_obs_delta_a)`` to climb
    #                             onto the cruise current, and until it gets there the door's
    #                             *own* steady current reads as an abrupt rise: every cycle
    #                             whose cruise current clears ``f_obs_trip_n`` then phantom-
    #                             trips.  The 110 V door escapes that only because its accel
    #                             ramp (0.50 s) happens to outlast the climb (0.34 s); the
    #                             Cranfield rig (ramp 0.25 s, climb 0.42 s) does not, and
    #                             reversed on 100 % of cycles at friction s >= 0.5.  Set
    #                             False to recover the pre-calibration behaviour.
    obs_vel_drop: float = 0.60  # a speed drop below 60 % of the leaf's own baseline = contact
    ref_lag_band_m: float = 0.012  # profile-governor band: the reference never runs further
    #                                than this ahead of a force-limited leaf
    stall_timeout_s: float = 1.50  # saturated and not moving for this long = DCU abort
    pos_err_obs_m: float = 0.020  # plan: error > 20 mm
    obs_debounce_s: float = 0.150  # plan: sustained 150 ms
    obs_detect_deadline_s: float = 0.30  # acceptance metric, NOT a regulatory limit
    obs_zone_m: float = 0.025  # detection disabled inside the final 25 mm (seal zone)
    obs_reverse_m: float = 0.100  # reverse 100 mm
    obs_reverse_v: float = 0.20
    obs_pause_s: float = 1.0
    obs_max_retries: int = 3
    obs_k_soft: float = 4.0e3  # N/m, compliant obstacle (bag, coat)
    obs_k_hard: float = 2.0e4  # N/m, rigid obstacle (limb, case)
    p_obs_base: float = 0.01  # plan: p = 0.01 + 0.03 * load
    p_obs_load: float = 0.03
    p_obs_fault: float = 0.60  # extra, driven by the injected obstruction trajectory

    # --- limit switches and timing ------------------------------------------------------
    ls_tol_m: float = 0.003  # switches within 3 mm of the ends
    warning_s: float = 2.0  # >= 2 s closing warning [R165 PRM 4.2.2.3.2]
    pre_open_s: float = 0.5
    post_s: float = 0.5
    timeout_margin_s: float = 4.0
    max_cycle_s: float = 12.0  # hard cap; anything this slow has already failed every test

    # --- motor thermal node --------------------------------------------------------------
    c_th: float = 150.0  # J/K
    r_th: float = 3.0  # K/W

    # --- fault maps (severity -> physics) -------------------------------------------------
    k_friction_c: float = 2.0  # F_c = F_c0 (1 + 2 s)  -> x3 at s = 1 (plan "Door")
    k_friction_b: float = 1.5  # b   = b0   (1 + 1.5 s) -> x2.5 at s = 1 (plan "Door")
    k_brush_r: float = 0.8  # R = R0 (1 + 0.8 s), plan
    k_brush_kt: float = 0.20  # commutation loss: k_t, k_e x (1 - 0.2 s)
    p_brush_dropout: float = 0.02  # current-channel dropouts at s = 1
    backlash_m: float = 0.008  # delta = 8 mm . s, plan
    mis_lambda_m: float = 0.30  # position-periodic friction, lambda = 0.3 m, plan
    k_mis: float = 3.0  # amplitude F_c0 * 3 * s
    p_dcu_base: float = 0.002  # plan: p = 0.002 + 0.05 s
    p_dcu_fault: float = 0.05
    dropout_len_mean: float = 5.0
    scatter_unit: float = 0.08  # per-door component scatter, drawn once per run
    scatter_cycle: float = 0.04  # per-cycle scatter (lubrication, seals, passengers)
    k_load_friction: float = 0.15  # crowded cars load the leaf guides
    scatter_v_ref: float = 0.02  # per-cycle scatter of the commanded cruise speed

    # --- functional failure ---------------------------------------------------------------
    dropout_fault_frac: float = 0.30  # telemetry loss above this reads as a DCU door fault
    closing_time_limit_s: float = 5.0
    phantom_retry_limit: int = 3
    door_fault_window: int = 10
    door_fault_limit: int = 2

    # --- sampling -------------------------------------------------------------------------
    dt: float = 0.001  # 1 ms inner loop (demo_runs)
    dt_out: float = 0.02  # 50 Hz telemetry
    acc_stride: int = 4  # feature accumulators run every 4th integration step (250 Hz)
    profile_n: int = 50  # current_profile_50

    # -----------------------------------------------------------------------------------
    @property
    def gear_gain(self) -> float:
        """Rotational-to-linear gain ``G/r`` (rad/m)."""
        return self.gear_ratio / self.pulley_r_m

    @property
    def force_per_amp(self) -> float:
        """Leaf force per motor amp, ``eta k_t G / r`` (N/A)."""
        return self.eta * self.k_t * self.gear_gain

    @property
    def m_eff(self) -> float:
        """Reflected mass: ``n_leaves m_leaf + J (G/r)^2`` (kg)."""
        return self.n_leaves * self.m_leaf_kg + self.j_motor * self.gear_gain**2

    @property
    def i_limit_a(self) -> float:
        """DCU current limit, the 200 N force ceiling expressed in amps."""
        return self.f_limit_n / self.force_per_amp

    @property
    def i_limit_peak_a(self) -> float:
        """Largest current the DCU ever commands: the contact ceiling plus the inertia and
        seal feed-forward it is allowed to add (see :func:`_integrate`)."""
        return (
            self.f_limit_n + self.m_eff * self.a_ref + self.f_seal_allow_n
        ) / self.force_per_amp

    @property
    def i_obs_a(self) -> float:
        """Absolute floor of the obstruction detector (band bottom) expressed in amps."""
        return self.f_obs_trip_n / self.force_per_amp

    @property
    def i_obs_delta_a(self) -> float:
        """Rise above the in-cycle baseline that reads as contact, expressed in amps."""
        return self.f_obs_delta_n / self.force_per_amp

    def with_(self, **changes: Any) -> "DoorParams":
        """Copy with fields replaced."""
        return replace(self, **changes)

    @classmethod
    def cranfield(cls) -> "DoorParams":
        """Variant matching the Cranfield linear-actuator rig (door proxy, CORD
        DOI 10.17862/cranfield.rd.5097649 [R86]).

        **24 V** bus and the rig's short ball-screw stroke instead of a 110 V door.

        Which of these are measured, after ``scripts/calibrate_door.py`` was run against the
        real release (see ``docs/parameters.md``):

        * :data:`CRANFIELD_SCREW_LEAD_M` is **measured** - the release's "Data description.pdf"
          section 2 names the screw as an RM1605-C7 with a 5 mm lead. It drives both
          ``pulley_r_m`` and ``mis_lambda_m``.
        * :data:`CRANFIELD_K_MIS` is **calibrated** on the 8 real spalling stages.
        * ``r_ohm``, ``k_t``, ``k_e``, ``f_c0``, ``b0``, ``m_leaf_kg`` remain **UNVERIFIED
          starting points**, and the real files say they cannot be improved from this dataset:
          it has no voltage channel (so ``R`` and ``k_t`` are not identifiable at all) and its
          motor is a chopper-regulated *stepper*, so the current is not the torque-proportional
          quantity the mechanical fit assumes - that regression scores R^2 = 0.03 on the real
          healthy runs and is reported as rejected rather than fitted.
        * ``k_friction_c``, ``k_friction_b`` and ``backlash_m`` are the plan's, deliberately:
          see the "Constants changed" table in ``docs/parameters.md`` for the measured ratios
          and the two gates that refused them.
        """
        return cls(
            stroke_m=0.10,
            gear_ratio=1.0,
            pulley_r_m=CRANFIELD_SCREW_LEAD_M / (2.0 * np.pi),
            m_leaf_kg=5.0,
            n_leaves=1,
            j_motor=2.0e-6,
            r_ohm=2.0,
            k_t=0.05,
            k_e=0.05,
            v_bus=24.0,
            f_c0=15.0,
            b0=30.0,
            f_seal_n=0.0,
            seal_zone_m=0.005,
            v_ref=0.05,
            a_ref=0.2,
            kp=40.0,
            ki=60.0,
            kx=800.0,
            f_limit_n=80.0,
            f_obs_trip_n=30.0,
            f_obs_delta_n=12.0,
            warning_s=0.0,
            pre_open_s=0.2,
            post_s=0.2,
            ls_tol_m=0.0005,
            obs_zone_m=0.004,
            pos_err_obs_m=OBS_POS_ERR_STROKE_FRAC * 0.10,  # 2.76 mm, not the door's 20 mm
            mis_lambda_m=CRANFIELD_SCREW_LEAD_M,  # spalling ripple repeats at the screw lead
            k_mis=CRANFIELD_K_MIS,  # calibrated on the 8 real spalling stages, see the constant
            obs_reverse_m=0.02,
            obs_reverse_v=0.03,
            max_cycle_s=10.0,
        )


# --------------------------------------------------------------------------------------
# Faults
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class DoorFaults:
    """The injected fault trajectories for one door, one slot per ``FAULT_TYPES['door']``.

    ``None`` means "this mode is healthy".  An all-``None`` instance is a healthy run.
    Severity maps onto physics as (plan "Door" fault map):

    ============= ==============================================================
    friction      ``F_c = F_c0 (1 + 2 s)``, ``b = b0 (1 + 1.5 s)`` (plan)
    brush_wear    ``R = R0 (1 + 0.8 s)``, ``k_t, k_e x (1 - 0.2 s)``, current dropouts
    backlash      dead band ``delta = 8 mm . s`` between motor side and leaf
    misalignment  position-periodic friction, ``+ 3 F_c0 s`` at ``lambda = 0.3 m``
    obstruction   ``p = 0.01 + 0.03 load + 0.6 s`` per cycle
    limit_switch  step: the closed limit switch stops reporting -> ``ls_timeout``
    dcu_dropout   ``p = 0.002 + 0.05 s`` per 50 Hz sample, burst NaN on every channel
    shock_wear    shock damage walks a hard stop inwards (closing-position ceiling) [R69]
    ============= ==============================================================
    """

    friction: DegradationTrajectory | None = None
    brush_wear: DegradationTrajectory | None = None
    backlash: DegradationTrajectory | None = None
    misalignment: DegradationTrajectory | None = None
    obstruction_rate: DegradationTrajectory | None = None
    limit_switch: DegradationTrajectory | None = None
    dcu_dropout: DegradationTrajectory | None = None
    shock_wear: DegradationTrajectory | None = None

    def __iter__(self) -> Iterator[DegradationTrajectory]:
        return iter(self.active())

    def __len__(self) -> int:
        return len(self.active())

    def active(self) -> tuple[DegradationTrajectory, ...]:
        """Every non-``None`` trajectory, in field order."""
        return tuple(
            t
            for t in (
                self.friction,
                self.brush_wear,
                self.backlash,
                self.misalignment,
                self.obstruction_rate,
                self.limit_switch,
                self.dcu_dropout,
                self.shock_wear,
            )
            if t is not None
        )

    @property
    def healthy(self) -> bool:
        """True when no fault is injected."""
        return not self.active()

    def component_id(self, default: str = "door_L1") -> str:
        """The component every injected trajectory refers to (all must agree)."""
        cids = {t.component_id for t in self.active()}
        if not cids:
            return default
        if len(cids) > 1:
            raise ValueError(
                f"DoorFaults: all trajectories must target one door, got {sorted(cids)}"
            )
        return cids.pop()

    @classmethod
    def from_trajectories(
        cls, trajectories: "DoorFaults | Sequence[DegradationTrajectory] | None"
    ) -> "DoorFaults":
        """Build from a sequence of :class:`DegradationTrajectory` (or pass one through)."""
        if trajectories is None:
            return cls()
        if isinstance(trajectories, DoorFaults):
            return trajectories
        kwargs: dict[str, DegradationTrajectory] = {}
        for traj in trajectories:
            if traj.subsystem != "door":
                raise ValueError(
                    f"DoorFaults.from_trajectories: trajectory {traj.fault_type!r} has "
                    f"subsystem {traj.subsystem!r}, expected 'door'"
                )
            fld = DOOR_FAULT_FIELDS.get(traj.fault_type)
            if fld is None:
                raise ValueError(
                    f"DoorFaults.from_trajectories: fault_type {traj.fault_type!r} has no "
                    f"door slot; known = {sorted(DOOR_FAULT_FIELDS)}"
                )
            if fld in kwargs:
                raise ValueError(
                    f"DoorFaults.from_trajectories: two trajectories for {traj.fault_type!r}"
                )
            kwargs[fld] = traj
        return cls(**kwargs)


# --------------------------------------------------------------------------------------
# Vectorised trapezoidal reference
# --------------------------------------------------------------------------------------


def _ref_params(
    x_from: np.ndarray, x_to: np.ndarray, v_lim: np.ndarray, a: float
) -> tuple[np.ndarray, ...]:
    """Trapezoid shape constants for a batch of moves ``x_from -> x_to``.

    Returns ``(d, sgn, t_acc, v_peak, t_cruise, t_total)``; degenerates to a triangle when
    the move is too short to reach ``v_lim`` and to zeros for a zero-length hold.
    """
    d = np.abs(x_to - x_from)
    sgn = np.sign(x_to - x_from)
    t_acc_full = v_lim / a
    triangular = d < v_lim * t_acc_full
    t_acc = np.where(triangular, np.sqrt(np.maximum(d, 0.0) / a), t_acc_full)
    v_peak = a * t_acc
    t_cruise = np.where(triangular, 0.0, (d - v_peak * t_acc) / np.maximum(v_peak, 1e-12))
    return d, sgn, t_acc, v_peak, t_cruise, 2.0 * t_acc + t_cruise


def _ref_eval(
    tau: np.ndarray,
    x_from: np.ndarray,
    d: np.ndarray,
    sgn: np.ndarray,
    t_acc: np.ndarray,
    v_peak: np.ndarray,
    t_cruise: np.ndarray,
    t_total: np.ndarray,
    a: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position/velocity/acceleration reference at ``tau`` for a ``_ref_params`` batch.

    The third return is the profile acceleration *along the direction of travel* (``+a``
    while ramping up, ``0`` at cruise, ``-a`` while ramping down).  The DCU uses it as
    inertia feed-forward when it works out how much current it may command, which is what
    makes :attr:`DoorParams.f_limit_n` a *contact*-force ceiling rather than a motor-torque
    ceiling: the force that reaches a stationary obstacle is the motor force minus what the
    93 kg of reflected mass is absorbing.
    """
    t_flat = t_acc + t_cruise
    t_rem = np.maximum(t_total - tau, 0.0)
    s = np.where(
        tau < t_acc,
        0.5 * a * tau * tau,
        np.where(
            tau < t_flat,
            0.5 * a * t_acc * t_acc + v_peak * (tau - t_acc),
            np.where(tau < t_total, d - 0.5 * a * t_rem * t_rem, d),
        ),
    )
    v = np.where(
        tau < t_acc,
        a * tau,
        np.where(tau < t_flat, v_peak, np.where(tau < t_total, a * t_rem, 0.0)),
    )
    a_prof = np.where(tau < t_acc, a, np.where(tau < t_flat, 0.0, np.where(tau < t_total, -a, 0.0)))
    return x_from + sgn * s, sgn * v, np.where(t_total > 0.0, a_prof, 0.0)


# --------------------------------------------------------------------------------------
# The batch integrator
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Motion:
    """Everything one motion pass (all cycles at once) produces."""

    t_done: np.ndarray  # s, per cycle, from the start of the pass
    ls_timeout: np.ndarray  # bool
    reversals: np.ndarray  # int
    obstruction: np.ndarray  # bool
    obs_detect_s: np.ndarray  # s from first contact to detection, NaN when none
    gave_up: np.ndarray  # bool: retries exhausted
    x_min: np.ndarray
    x_max: np.ndarray
    x_end: np.ndarray
    i_peak: np.ndarray
    i_start_peak: np.ndarray
    i_sum: np.ndarray
    i2_sum: np.ndarray
    n_samp: np.ndarray
    i_cruise_sum: np.ndarray
    i2_cruise_sum: np.ndarray
    n_cruise: np.ndarray
    i_end_sum: np.ndarray
    n_end: np.ndarray
    energy_j: np.ndarray
    pwm_sum: np.ndarray
    err_max: np.ndarray
    err2_sum: np.ndarray
    heat_j: np.ndarray
    prof_sum: np.ndarray  # (n_cycles * profile_n,)
    prof_cnt: np.ndarray
    cap_x: np.ndarray  # (n_out, n_cap)
    cap_xr: np.ndarray
    cap_v: np.ndarray
    cap_i: np.ndarray
    cap_volt: np.ndarray
    cap_obs: np.ndarray
    cap_n: np.ndarray  # (n_cap,) valid sample count per captured cycle


def _integrate(
    p: DoorParams,
    *,
    closing: bool,
    f_c: np.ndarray,
    b_visc: np.ndarray,
    r_ohm: np.ndarray,
    k_mag: np.ndarray,
    backlash: np.ndarray,
    mis_amp: np.ndarray,
    x_stop: np.ndarray,
    obs_pos: np.ndarray,
    obs_k: np.ndarray,
    obs_clear_after: np.ndarray,
    v_cmd: np.ndarray,
    ls_ok: np.ndarray,
    capture: np.ndarray,
    x_start: np.ndarray | None = None,
) -> _Motion:
    """Integrate one motion (open or close) for every cycle simultaneously.

    All per-cycle inputs are float arrays of length ``n``; ``capture`` is the integer index
    of the cycles whose 50 Hz waveform is kept.  ``ls_ok`` is the health of the limit switch
    at the *target* end - a failed switch turns a normal move into an ``ls_timeout``.
    """
    n = f_c.size
    dt = p.dt
    ke_g = p.k_e * p.gear_gain * k_mag
    fpa = np.maximum(p.eta * p.k_t * p.gear_gain * k_mag, 1e-9)  # N/A, brush wear included
    i_obs_floor = p.f_obs_trip_n / fpa  # band bottom, absolute
    i_obs_delta = p.f_obs_delta_n / fpa  # abrupt rise over the in-cycle baseline
    m_eff = p.m_eff

    x = np.full(n, 0.0) if x_start is None else np.asarray(x_start, dtype=np.float64).copy()
    xm = x.copy()
    v = np.zeros(n)
    integ = np.zeros(n)
    tau = np.zeros(n)
    elapsed = np.zeros(n)
    mode = np.zeros(n, dtype=np.int8)

    x_from = x.copy()
    x_to = np.full(n, 0.0 if closing else p.stroke_m)
    v_lim = v_cmd.copy()
    d, sgn, t_acc, v_peak, t_cruise, t_total = _ref_params(x_from, x_to, v_lim, p.a_ref)

    obs_pos = obs_pos.copy()
    obs_timer = np.zeros(n)
    obs_contact_t = np.full(n, np.nan)
    _NEVER = np.zeros(n, dtype=bool)
    reversals = np.zeros(n, dtype=np.int32)
    obstruction = np.zeros(n, dtype=bool)
    obs_flag = np.zeros(n, dtype=bool)  # telemetry-visible obstruction state
    obs_detect_s = np.full(n, np.nan)
    ls_timeout = np.zeros(n, dtype=bool)
    gave_up = np.zeros(n, dtype=bool)
    t_done = np.zeros(n)

    i_peak = np.zeros(n)
    i_start_peak = np.zeros(n)
    i_sum = np.zeros(n)
    i2_sum = np.zeros(n)
    n_samp = np.zeros(n)
    i_cruise_sum = np.zeros(n)
    i2_cruise_sum = np.zeros(n)
    n_cruise = np.zeros(n)
    i_end_sum = np.zeros(n)
    n_end = np.zeros(n)
    energy_j = np.zeros(n)
    pwm_sum = np.zeros(n)
    err_max = np.zeros(n)
    err2_sum = np.zeros(n)
    heat_j = np.zeros(n)
    x_min = x.copy()
    x_max = x.copy()

    # Profile governor + relative obstruction detection state.  ``gate`` slows the
    # reference clock whenever the drive is on its force ceiling, so a worn door tracks a
    # *slower* trapezoid instead of running away from the reference into a timeout; the two
    # EWMA baselines make the obstruction detector fire on an abrupt rise relative to this
    # cycle's own current/speed rather than on an absolute force level [R65][R67].
    gate = np.ones(n)
    i_base = np.zeros(n)
    v_base = np.zeros(n)
    obs_armed = np.zeros(n, dtype=bool)  # baselines seeded for this motion (obs_base_seed)
    stall_t = np.zeros(n)
    # Asymmetric baselines: each one follows the *benign* direction within ~50 ms and the
    # *alarming* one with a 0.4 s lag.  The current baseline therefore collapses onto the
    # cruise current the moment the acceleration ramp ends (no blind window behind a slowly
    # decaying baseline) while an obstacle's current rise outruns it, and the speed baseline
    # locks onto cruise speed immediately while a contact-induced drop outruns it.
    a_slow = min(dt / max(p.obs_base_tau_s, dt), 1.0)
    a_fast = min(dt / max(p.obs_base_fast_s, dt), 1.0)
    seal_allow = p.f_seal_allow_n if closing else 0.0

    prof_sum = np.zeros(n * p.profile_n)
    prof_cnt = np.zeros(n * p.profile_n)
    cyc_base = np.arange(n) * p.profile_n

    out_stride = max(int(round(p.dt_out / dt)), 1)
    cap_x: list[np.ndarray] = []
    cap_xr: list[np.ndarray] = []
    cap_v: list[np.ndarray] = []
    cap_i: list[np.ndarray] = []
    cap_volt: list[np.ndarray] = []
    cap_obs: list[np.ndarray] = []
    cap_n = np.zeros(capture.size, dtype=np.int64)

    has_mis = bool(np.any(mis_amp > 0.0))
    has_obs = bool(np.any(np.isfinite(obs_pos)))
    has_stop = bool(np.any(x_stop > 0.0))
    max_steps = int(round(p.max_cycle_s / dt))
    n_special = 0
    start_steps = int(round(0.3 / dt))
    acc_stride = max(int(p.acc_stride), 1)
    loss_gain = (1.0 - p.eta) * p.k_t * p.gear_gain * k_mag

    for step in range(max_steps):
        act = mode != _DONE
        if not act.any():
            break
        tau = tau + dt * gate
        # every live cycle advances in lockstep, so wall time is just the step count
        elapsed = (step + 1) * dt
        xr, vr, ar = _ref_eval(tau, x_from, d, sgn, t_acc, v_peak, t_cruise, t_total, p.a_ref)

        err_v = vr - v
        err_x = xr - x
        integ_new = integ + err_v * dt
        volt = p.kp * err_v + p.ki * integ_new + ke_g * vr + p.kx * err_x
        volt_c = np.clip(volt, -p.v_bus, p.v_bus)
        i_raw = (volt_c - ke_g * v) / r_ohm
        # Contact-force ceiling with inertia and seal feed-forward: what the DCU limits is
        # the force that could reach an obstacle, so the mass it is knowingly accelerating
        # and the seal it is knowingly compressing are added back.  Friction is *not*
        # compensated - that is what makes a worn door slow down instead of pushing harder.
        allow = p.f_limit_n + m_eff * np.maximum(ar, 0.0)
        if seal_allow:
            allow = allow + seal_allow * np.clip(1.0 - x / p.seal_zone_m, 0.0, 1.0)
        i_lim = allow / fpa
        i = np.clip(i_raw, -i_lim, i_lim)
        force_sat = i != i_raw
        sat = (volt_c != volt) | force_sat
        integ = np.where(sat & (err_v * integ_new > 0.0), integ, integ_new)
        # profile governor: the reference clock stalls in proportion to the lag it has
        # already built up, but only while the drive is actually on its ceiling
        gate = np.where(
            force_sat,
            np.clip(1.0 - np.abs(err_x) / p.ref_lag_band_m, 0.0, 1.0),
            1.0,
        )
        volt_eff = ke_g * v + i * r_ohm

        f_motor = fpa * i

        # external loads that resist closing (positive = pushes the leaf open)
        f_ext = np.zeros(n)
        if has_obs:
            contact = np.isfinite(obs_pos) & (x < obs_pos)
            f_ext = f_ext + np.where(contact, obs_k * (np.nan_to_num(obs_pos) - x), 0.0)
        if has_stop:
            f_ext = f_ext + np.where(x < x_stop, p.obs_k_hard * (x_stop - x), 0.0)
        if closing and p.f_seal_n > 0.0:
            f_ext = f_ext + p.f_seal_n * np.clip(1.0 - x / p.seal_zone_m, 0.0, 1.0)

        f_c_eff = f_c
        if has_mis:
            f_c_eff = f_c + mis_amp * (0.5 + 0.5 * np.sin(2.0 * np.pi * x / p.mis_lambda_m))

        moving = np.abs(v) > 1e-3
        f_drive = f_motor + f_ext
        f_load = np.where(
            moving,
            f_c_eff * np.sign(v) + b_visc * v,
            np.clip(f_drive, -f_c_eff, f_c_eff),
        )
        acc = (f_drive - f_load) / m_eff
        v = np.where(act, v + acc * dt, v)
        xm = np.where(act, xm + v * dt, xm)
        # backlash: the leaf trails the motor side inside a dead band
        half = 0.5 * backlash
        x = np.where(xm - x > half, xm - half, np.where(x - xm > half, xm + half, x))
        x = np.clip(x, -0.005, p.stroke_m + 0.005)

        abs_i = np.abs(i)
        abs_err = np.abs(err_x)
        # Scalar accumulators run at `acc_stride` x dt (250 Hz by default) rather than at
        # the 1 kHz integration rate: means, RMS and integrals are unchanged to 4 digits
        # and the per-step numpy overhead drops by a third.
        if step % acc_stride == 0:
            dt_acc = dt * acc_stride
            i_peak = np.maximum(i_peak, np.where(act, abs_i, 0.0))
            if step < start_steps:
                i_start_peak = np.maximum(i_start_peak, np.where(act, abs_i, 0.0))
            i2 = i * i
            i_sum += np.where(act, abs_i, 0.0)
            i2_sum += np.where(act, i2, 0.0)
            n_samp += act
            cruise = act & (mode == _MOVE) & (np.abs(vr) >= 0.95 * v_lim)
            i_cruise_sum += np.where(cruise, abs_i, 0.0)
            i2_cruise_sum += np.where(cruise, i2, 0.0)
            n_cruise += cruise
            if closing:
                end_zone = act & (mode == _MOVE) & (x < p.obs_zone_m)
                i_end_sum += np.where(end_zone, abs_i, 0.0)
                n_end += end_zone
            energy_j += np.where(act, np.abs(volt_eff * i), 0.0) * dt_acc
            pwm_sum += np.where(act, volt_eff, 0.0)
            err_max = np.maximum(err_max, np.where(act, abs_err, 0.0))
            err2_sum += np.where(act, abs_err * abs_err, 0.0)
            heat_j += np.where(act, i2 * r_ohm + loss_gain * np.abs(i * v), 0.0) * dt_acc
            x_min = np.minimum(x_min, np.where(act, x, x_min))
            x_max = np.maximum(x_max, np.where(act, x, x_max))

        # ---- obstruction detection (closing only, outside the seal zone) --------------
        if closing:
            # Three detection paths, all gated outside the seal zone: excess current once
            # the acceleration ramp is over (the 157 N trip), a speed drop at cruise (what
            # catches a compliant obstacle before its force has built up), and a
            # position-tracking error.  Gating the current path on "past the accel ramp"
            # is [R65]'s condition-conditional threshold: one global threshold would either
            # miss obstructions or fire on every start.
            at_cruise = np.abs(vr) >= 0.95 * v_lim
            past_accel = tau >= t_acc  # the accel ramp is the only high-current phase
            abs_v = np.abs(v)
            if p.obs_base_seed:
                # Seed each baseline with the operating point at the instant the current
                # path arms: the detector then answers "is this an abrupt rise *from here*"
                # from its very first sample instead of spending one EWMA rise time
                # mistaking the door's own cruise current for one.
                seed = act & (mode == _MOVE) & (x > p.obs_zone_m) & past_accel & ~obs_armed
                if seed.any():
                    i_base = np.where(seed, abs_i, i_base)
                    v_base = np.where(seed, abs_v, v_base)
                    obs_armed |= seed
            detect = act & (mode == _MOVE) & (x > p.obs_zone_m) & (
                (past_accel & (abs_i > i_base + i_obs_delta) & (abs_i > i_obs_floor))
                | (at_cruise & (v_base > 0.05) & (abs_v < p.obs_vel_drop * v_base))
                | (abs_err > p.pos_err_obs_m)
            )
            # The baselines follow the cycle's own operating point while nothing is
            # suspected and freeze the moment something is: gradual wear moves the baseline
            # with it (no phantom reversal), an obstacle outruns it within ~40 ms.
            upd = ~detect
            d_i = abs_i - i_base
            d_v = abs_v - v_base
            i_base = np.where(upd, i_base + d_i * np.where(d_i < 0.0, a_fast, a_slow), i_base)
            v_base = np.where(upd, v_base + d_v * np.where(d_v > 0.0, a_fast, a_slow), v_base)
            obs_timer = np.where(detect, obs_timer + dt, 0.0)
            if has_obs:
                first_contact = np.isfinite(obs_pos) & (x < obs_pos) & np.isnan(obs_contact_t)
                obs_contact_t = np.where(first_contact, elapsed, obs_contact_t)
            trip = obs_timer >= p.obs_debounce_s
            if trip.any():
                obstruction |= trip
                obs_flag |= trip
                obs_detect_s = np.where(
                    trip & np.isnan(obs_detect_s),
                    np.where(np.isnan(obs_contact_t), p.obs_debounce_s, elapsed - obs_contact_t),
                    obs_detect_s,
                )
                retry = trip & (reversals < p.obs_max_retries)
                stop = trip & ~retry
                reversals = reversals + retry
                gave_up |= stop
                if retry.any():
                    x_from = np.where(retry, x, x_from)
                    x_to = np.where(retry, np.minimum(x + p.obs_reverse_m, p.stroke_m), x_to)
                    v_lim = np.where(retry, p.obs_reverse_v, v_lim)
                    mode = np.where(retry, _REVERSE, mode)
                    tau = np.where(retry, 0.0, tau)
                    integ = np.where(retry, 0.0, integ)
                    gate = np.where(retry, 1.0, gate)
                    i_base = np.where(retry, 0.0, i_base)
                    v_base = np.where(retry, 0.0, v_base)
                    obs_armed = obs_armed & ~retry
                    cleared = retry & (reversals >= obs_clear_after)
                    if cleared.any():
                        obs_pos = np.where(cleared, np.nan, obs_pos)
                        has_obs = bool(np.any(np.isfinite(obs_pos)))
                    n_special += 1
                if stop.any():
                    t_done = np.where(stop, elapsed, t_done)
                    mode = np.where(stop, _DONE, mode)
                obs_timer = np.where(trip, 0.0, obs_timer)
                d, sgn, t_acc, v_peak, t_cruise, t_total = _ref_params(x_from, x_to, v_lim, p.a_ref)

        # ---- mode transitions ----------------------------------------------------------
        settled = np.abs(v) < 0.02
        at_target = (
            ((x < p.ls_tol_m) & ls_ok) if closing else ((x > p.stroke_m - p.ls_tol_m) & ls_ok)
        )
        # Stall watchdog: the governor holds the reference next to a blocked leaf, so a
        # hard block would otherwise never reach the profile timeout.  A DCU that is on its
        # force ceiling and not moving aborts the motion.
        stall_t = np.where((mode != _DONE) & force_sat & (np.abs(v) < 0.01), stall_t + dt, 0.0)
        stalled = (mode != _DONE) & (stall_t > p.stall_timeout_s)
        finish = (mode == _MOVE) & at_target & settled & (tau > t_total)
        timeout = ((mode != _DONE) & (tau > t_total + p.timeout_margin_s)) | stalled
        capped = (mode != _DONE) & (elapsed > p.max_cycle_s - dt)
        stopped = finish | timeout | capped
        if stopped.any():
            ls_timeout |= (timeout | capped) & (mode == _MOVE)
            t_done = np.where(stopped, elapsed, t_done)
            mode = np.where(stopped, _DONE, mode)

        if n_special == 0:
            continue_special = False
        else:
            continue_special = True
        rev_done = ((mode == _REVERSE) & settled & (tau > t_total)) if continue_special else _NEVER
        if continue_special and rev_done.any():
            x_from = np.where(rev_done, x, x_from)
            x_to = np.where(rev_done, x, x_to)
            mode = np.where(rev_done, _PAUSE, mode)
            tau = np.where(rev_done, 0.0, tau)
            integ = np.where(rev_done, 0.0, integ)
            gate = np.where(rev_done, 1.0, gate)
            obs_flag = np.where(rev_done, False, obs_flag)
            d, sgn, t_acc, v_peak, t_cruise, t_total = _ref_params(x_from, x_to, v_lim, p.a_ref)

        pause_done = ((mode == _PAUSE) & (tau > p.obs_pause_s)) if continue_special else _NEVER
        if continue_special and pause_done.any():
            x_from = np.where(pause_done, x, x_from)
            x_to = np.where(pause_done, 0.0 if closing else p.stroke_m, x_to)
            v_lim = np.where(pause_done, v_cmd, v_lim)
            mode = np.where(pause_done, _MOVE, mode)
            tau = np.where(pause_done, 0.0, tau)
            integ = np.where(pause_done, 0.0, integ)
            gate = np.where(pause_done, 1.0, gate)
            d, sgn, t_acc, v_peak, t_cruise, t_total = _ref_params(x_from, x_to, v_lim, p.a_ref)

        # ---- decimated capture + position-binned current profile ----------------------
        if step % out_stride == 0:
            act_now = mode != _DONE
            if closing:
                trav = np.clip((p.stroke_m - x) / p.stroke_m, 0.0, 1.0 - 1e-9)
                idx = cyc_base + (trav * p.profile_n).astype(np.int64)
                w = np.where(act_now, abs_i, 0.0)
                prof_sum += np.bincount(idx, weights=w, minlength=prof_sum.size)
                prof_cnt += np.bincount(idx, weights=act_now.astype(np.float64), minlength=prof_cnt.size)
            if capture.size:
                cap_x.append(x[capture].copy())
                cap_xr.append(xr[capture].copy())
                cap_v.append(v[capture].copy())
                cap_i.append(i[capture].copy())
                cap_volt.append(volt_eff[capture].copy())
                cap_obs.append(obs_flag[capture].copy())
                cap_n += act_now[capture]

    # any cycle still running when max_steps ran out
    still = mode != _DONE
    if still.any():
        ls_timeout |= still
        t_done = np.where(still, elapsed, t_done)

    empty = np.zeros((0, capture.size))
    return _Motion(
        t_done=t_done,
        ls_timeout=ls_timeout,
        reversals=reversals,
        obstruction=obstruction,
        obs_detect_s=obs_detect_s,
        gave_up=gave_up,
        x_min=x_min,
        x_max=x_max,
        x_end=x,
        i_peak=i_peak,
        i_start_peak=i_start_peak,
        i_sum=i_sum,
        i2_sum=i2_sum,
        n_samp=n_samp,
        i_cruise_sum=i_cruise_sum,
        i2_cruise_sum=i2_cruise_sum,
        n_cruise=n_cruise,
        i_end_sum=i_end_sum,
        n_end=n_end,
        energy_j=energy_j,
        pwm_sum=pwm_sum,
        err_max=err_max,
        err2_sum=err2_sum,
        heat_j=heat_j,
        prof_sum=prof_sum,
        prof_cnt=prof_cnt,
        cap_x=np.stack(cap_x) if cap_x else empty,
        cap_xr=np.stack(cap_xr) if cap_xr else empty,
        cap_v=np.stack(cap_v) if cap_v else empty,
        cap_i=np.stack(cap_i) if cap_i else empty,
        cap_volt=np.stack(cap_volt) if cap_volt else empty,
        cap_obs=np.stack(cap_obs) if cap_obs else np.zeros((0, capture.size), dtype=bool),
        cap_n=np.maximum(cap_n, 1),
    )


# --------------------------------------------------------------------------------------
# Sensor chain
# --------------------------------------------------------------------------------------

#: Measurement chain per door signal.  Digitals are read straight off the DCU and carry no
#: noise or quantisation; the DCU dropout fault NaNs *every* channel at once.
DOOR_SENSORS: Final[dict[str, SensorSpec]] = {
    "pos_ref": SensorSpec(),
    "pos": SensorSpec(noise_sigma=2.0e-4, quantum=1.0e-4),
    "vel": SensorSpec(noise_sigma=5.0e-3, quantum=1.0e-3),
    "current": SensorSpec(noise_sigma=0.010, quantum=0.005),
    "voltage": SensorSpec(noise_sigma=0.20, quantum=0.10),
    "pwm": SensorSpec(noise_sigma=0.002, quantum=0.001),
    "T_motor": SensorSpec(noise_sigma=0.10, quantum=0.05),
}


def _burst_mask(n: int, p: np.ndarray, mean_len: float, rng: np.random.Generator) -> np.ndarray:
    """Burst dropout mask: per-sample start probability ``p``, geometric burst length."""
    if n == 0:
        return np.zeros(0, dtype=bool)
    starts = np.flatnonzero(rng.random(n) < p)
    if starts.size == 0:
        return np.zeros(n, dtype=bool)
    lengths = rng.geometric(1.0 / max(mean_len, 1.0), size=starts.size)
    delta = np.zeros(n + 1, dtype=np.int32)
    np.add.at(delta, starts, 1)
    np.add.at(delta, np.minimum(starts + lengths, n), -1)
    return np.cumsum(delta[:-1]) > 0


def estimate_shock_jumps(
    series: Sequence[np.ndarray],
    *,
    smooth: int = SHOCK_EST_SMOOTH,
    window: int = SHOCK_EST_WINDOW,
    k: float = SHOCK_EST_K,
    min_cycles: int = SHOCK_EST_MIN_CYCLES,
) -> np.ndarray:
    """Observable estimator of the per-cycle shock count, from telemetry-derived features only.

    A hard shock walks the closing hard stop inwards **for good** [R69][R71], so what the DCU
    can actually see is a permanent *step* in the cycle statistics: chiefly the travel the leaf
    still achieves (``pos_close_max``, which steps down), and with it the current pushed
    against the stop (``i_peak``) and the tracking error that never closes (``pos_err_max``).
    Per series, causally:

    1. median-filter over the trailing ``smooth`` cycles - a passenger obstruction perturbs one
       cycle and must not read as damage, while a shock changes every cycle after it;
    2. compare that to the median of the trailing ``window`` cycles of the same filtered series,
       in units of its robust sigma ``IQR / 1.349`` over that window, floored at
       ``_SHOCK_EST_REL_FLOOR`` of the median so a quiet baseline cannot divide by zero;
    3. flag ``|z| > k`` - a step shows as a *drop* in travel and a *rise* in current, so the
       test is two-sided.

    The per-series flags are OR-ed and only the **rising edge** counts: a step holds its new
    level until the trailing median catches up, and that plateau is the same damage, not a new
    shock.  Nothing here reads the future: every window is trailing, so the column carries no
    look-ahead into the benchmark.

    This is an estimator, not the truth.  The truth is ``meta_shock_count``, which
    :func:`simulate` writes straight from the latent :class:`~nebulax.sim.common.ShockSeries`
    and which is metadata (``meta_``) precisely because no fleet DCU has it.  On a shock run the
    two track each other (cumulative correlation ~0.9 on the day-long fixtures); they are never
    equal - the estimator resolves roughly half of the shocks, because once the door is jamming
    on the walked-in stop every cycle, the reversal scatter is the same size as one shock's
    step.  That gap is the honest cost of not having the latent process, and it is the point.

    ``series`` are per-cycle feature arrays of equal length (:func:`simulate` passes
    ``pos_close_max``, ``i_peak`` and ``pos_err_max``).  Returns a ``float64`` array of
    0.0/1.0, one per cycle; :func:`simulate` publishes its cumulative sum as
    ``shock_jump_count_est``.
    """
    arrays = [np.asarray(x, dtype=np.float64).ravel() for x in series]
    if not arrays:
        raise ValueError("estimate_shock_jumps: no series given")
    n = arrays[0].size
    if any(a.size != n for a in arrays):
        raise ValueError(
            f"estimate_shock_jumps: series must share one length; got {[a.size for a in arrays]}"
        )
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    w = max(int(window), 2)
    m = max(int(min_cycles), 2)
    sm = max(int(smooth), 1)
    over = np.zeros(n, dtype=bool)
    for x in arrays:
        filt = pd.Series(x).rolling(sm, min_periods=1).median()
        trailing = filt.shift(1).rolling(w, min_periods=m)
        med = trailing.median().to_numpy(dtype=np.float64)
        sigma = (
            trailing.quantile(0.75).to_numpy(dtype=np.float64)
            - trailing.quantile(0.25).to_numpy(dtype=np.float64)
        ) / _IQR_TO_SIGMA
        scale = np.maximum(
            np.nan_to_num(sigma, nan=0.0), _SHOCK_EST_REL_FLOOR * np.abs(np.nan_to_num(med))
        )
        safe = np.where(scale > 0.0, scale, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(scale > 0.0, np.abs(filt.to_numpy(dtype=np.float64) - med) / safe, 0.0)
        over |= np.nan_to_num(z, nan=0.0, posinf=np.inf) > float(k)
    rising = over & ~np.concatenate([[False], over[:-1]])
    return rising.astype(np.float64)


# --------------------------------------------------------------------------------------
# simulate()
# --------------------------------------------------------------------------------------


def simulate(
    params: DoorParams,
    faults: DoorFaults | Sequence[DegradationTrajectory] | None,
    service: Service,
    rng: np.random.Generator,
    store_every: int = 10,
    *,
    run_id: str = "run0",
    source: str = "sim",
    car: int = 1,
    component_id: str | None = None,
    context_dt: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate every door cycle of ``service`` for one leaf.

    One cycle per dwell segment of the shared :class:`~nebulax.sim.common.Service` (the
    plan's "600 cycles/door/day" is the order of magnitude; the exact count is whatever the
    timetable produces, ~440/day with the frozen ``run_s``/``dwell_s`` bands).

    Returns ``(long, features, events)`` per the
    :class:`~nebulax.sim.common.SimulateFn` contract.  ``long`` carries 50 Hz waveforms for
    the stored cycles only - every ``store_every``-th cycle plus every cycle in the last two
    days before an injected fault's ``t_failure`` - and the 1 Hz train-context rows.  Only
    the two *motion activities* (opening, closing, each with its pad and the >= 2 s closing
    warning) are stored, not the static hold; that is exactly how a real DCU logs, one file
    per activity [R69].  ``features`` has one row per cycle, always.

    ``t_functional_failure`` is written back onto the injected trajectories before returning,
    so the caller can build the fault log from ``Scenario.fault_log_rows`` afterwards.
    """
    p = params
    df = DoorFaults.from_trajectories(faults)
    cid = component_id or df.component_id()
    if not S.is_valid_component_id("door", cid):
        raise ValueError(
            f"door.simulate: component_id {cid!r} is not a door; expected one of "
            f"{list(S.DOOR_COMPONENT_IDS)}"
        )
    if not 0 <= int(car) <= S.MAX_CAR:
        raise ValueError(f"door.simulate: car must lie in [0, {S.MAX_CAR}], got {car}")
    if store_every < 1:
        raise ValueError(f"door.simulate: store_every must be >= 1, got {store_every}")

    dwells = service.dwells()
    if dwells.empty:
        raise ValueError("door.simulate: the service has no dwell segments - no door cycles")
    t_cycle = dwells["t_start"].to_numpy(dtype=np.float64)
    dur = dwells["duration_s"].to_numpy(dtype=np.float64)
    load = dwells["load_frac"].to_numpy(dtype=np.float64)
    t_amb = dwells["T_amb"].to_numpy(dtype=np.float64)
    hour = dwells["hour_of_day"].to_numpy(dtype=np.float64)
    n = t_cycle.size
    t_mid = t_cycle + 0.5 * dur

    # ---- severity per cycle: jittered for the physics, clean for the label -------------
    def sev(traj: DegradationTrajectory | None, jitter: bool) -> np.ndarray:
        if traj is None:
            return np.zeros(n)
        return np.asarray(traj.severity(t_mid, rng if jitter else None), dtype=np.float64)

    s_fric = sev(df.friction, True)
    s_brush = sev(df.brush_wear, True)
    s_back = sev(df.backlash, True)
    s_mis = sev(df.misalignment, True)
    s_obs = sev(df.obstruction_rate, True)
    s_ls = sev(df.limit_switch, True)
    s_dcu = sev(df.dcu_dropout, True)
    s_shock = sev(df.shock_wear, True)
    sev_label = np.max(
        np.stack([sev(t, False) for t in df.active()]) if df.active() else np.zeros((1, n)),
        axis=0,
    )

    # ---- severity -> physics ----------------------------------------------------------
    # Per-component scatter (this leaf is not the fleet mean) and per-cycle scatter
    # (lubrication, seal state, how many people lean on the door).  Without these the
    # ML problem is dishonestly easy [plan "Risks"; rail_phm 4.1].
    unit_fc = float(1.0 + p.scatter_unit * rng.standard_normal())
    unit_b = float(1.0 + p.scatter_unit * rng.standard_normal())
    unit_r = float(1.0 + 0.5 * p.scatter_unit * rng.standard_normal())
    jit_fc = 1.0 + p.scatter_cycle * rng.standard_normal(n)
    jit_b = 1.0 + p.scatter_cycle * rng.standard_normal(n)
    # The plan's map is ``F_c = F_c0 (1 + 2 s)`` / ``b = b0 (1 + 1.5 s)``.  Component and
    # per-cycle scatter describe the *healthy* machine, so they multiply the baseline only
    # and the wear increment is added on top: a x1.25 unit would otherwise also scale the
    # fault by x1.25 and push a severity-1 door past the drive's force ceiling on some seeds
    # and not others, which is exactly the seizure cliff we are removing.
    f_c = p.f_c0 * (
        unit_fc * jit_fc * (1.0 + p.k_load_friction * load) + p.k_friction_c * s_fric
    )
    b_visc = p.b0 * (unit_b * jit_b + p.k_friction_b * s_fric)
    r_ohm = np.full(n, p.r_ohm * unit_r) * (1.0 + p.k_brush_r * s_brush)
    k_mag = 1.0 - p.k_brush_kt * s_brush
    backlash = p.backlash_m * s_back
    mis_amp = p.k_mis * p.f_c0 * s_mis
    x_stop = p.stroke_m * np.clip(s_shock, 0.0, 1.0)
    ls_closed_ok = s_ls < 0.5
    ls_open_ok = np.ones(n, dtype=bool)
    p_dcu = np.clip(p.p_dcu_base + p.p_dcu_fault * s_dcu, 0.0, 0.5)
    p_brush_drop = p.p_brush_dropout * s_brush

    # temperature-dependent R and b (motor node starts at ambient)
    r_ohm = r_ohm * (1.0 + p.r_temp_coeff * (t_amb - 20.0))
    b_visc = b_visc * np.maximum(1.0 + p.b_temp_coeff * (t_amb - 20.0), 0.2)

    # ---- obstacles ---------------------------------------------------------------------
    p_obstruct = np.clip(p.p_obs_base + p.p_obs_load * load + p.p_obs_fault * s_obs, 0.0, 1.0)
    has_obstacle = rng.random(n) < p_obstruct
    obs_hard = rng.random(n) < 0.5
    obs_pos = np.where(has_obstacle, rng.uniform(0.15, 0.55, size=n), np.nan)
    obs_k = np.where(obs_hard, p.obs_k_hard, p.obs_k_soft)
    # how many reversals the obstacle survives: most passengers withdraw on the first
    # re-open, a few percent are genuinely trapped and the door gives up (door_fault).
    obs_clear_after = rng.choice(
        np.array([1, 2, p.obs_max_retries + 1]), size=n, p=[0.80, 0.15, 0.05]
    ).astype(np.float64)
    # DCU profile-generator / belt-tension scatter: the commanded cruise speed is not
    # bit-identical cycle to cycle, so opening and closing times have a real spread.
    v_cmd = p.v_ref * (1.0 + p.scatter_v_ref * rng.standard_normal(n))

    # ---- which cycles keep their waveform ------------------------------------------------
    store = (np.arange(n) % max(store_every, 1)) == 0
    for traj in df.active():
        store |= (t_mid >= traj.t_failure - 2.0 * SEC_PER_DAY) & (t_mid <= traj.t_failure)
    store_idx = np.flatnonzero(store)

    # ---- integrate ------------------------------------------------------------------------
    no_obs = np.full(n, np.nan)
    opening = _integrate(
        p,
        closing=False,
        f_c=f_c,
        b_visc=b_visc,
        r_ohm=r_ohm,
        k_mag=k_mag,
        backlash=backlash,
        mis_amp=mis_amp,
        x_stop=np.zeros(n),
        obs_pos=no_obs,
        obs_k=obs_k,
        obs_clear_after=obs_clear_after,
        v_cmd=v_cmd,
        ls_ok=ls_open_ok,
        capture=store_idx,
    )
    closing = _integrate(
        p,
        closing=True,
        f_c=f_c,
        b_visc=b_visc,
        r_ohm=r_ohm,
        k_mag=k_mag,
        backlash=backlash,
        mis_amp=mis_amp,
        x_stop=x_stop,
        obs_pos=obs_pos,
        obs_k=obs_k,
        obs_clear_after=obs_clear_after,
        v_cmd=v_cmd,
        ls_ok=ls_closed_ok,
        capture=store_idx,
        x_start=opening.x_end,
    )

    # ---- timeline of each cycle -----------------------------------------------------------
    t_open_start = t_cycle + p.pre_open_s
    _, _, _, _, _, t_close_nom = _ref_params(
        np.array([p.stroke_m]), np.array([0.0]), np.array([p.v_ref]), p.a_ref
    )
    close_nom = float(t_close_nom[0]) + 0.3
    earliest = t_open_start + opening.t_done + 0.5 + p.warning_s
    t_close_start = np.maximum(t_cycle + dur - close_nom, earliest)
    t_closed = t_close_start + closing.t_done
    t_end = t_closed + p.post_s
    overrun = t_closed > (t_cycle + dur)

    # ---- motor thermal node (sequential across cycles, cheap) -------------------------------
    heat = opening.heat_j + closing.heat_j
    t_active = np.maximum(opening.t_done + closing.t_done, 1e-3)
    tau_th = p.c_th * p.r_th
    t_motor_start = np.empty(n)
    t_motor_end = np.empty(n)
    t_prev = float(t_amb[0])
    t_prev_end = t_cycle[0]
    for k in range(n):
        gap = max(t_cycle[k] - t_prev_end, 0.0)
        t_start_k = t_amb[k] + (t_prev - t_amb[k]) * float(np.exp(-gap / tau_th))
        t_end_k = t_start_k + (heat[k] - (t_start_k - t_amb[k]) / p.r_th * t_active[k]) / p.c_th
        t_motor_start[k] = t_start_k
        t_motor_end[k] = t_end_k
        t_prev = t_end_k
        t_prev_end = t_end[k]

    # ---- DCU dropouts, at the 50 Hz output rate, for every cycle ----------------------------
    n_pre = int(round(p.pre_open_s / p.dt_out))
    n_post = int(round(p.post_s / p.dt_out))
    n_warn = int(round(p.warning_s / p.dt_out))
    n_open_mot = np.maximum(np.ceil(opening.t_done / p.dt_out).astype(np.int64), 1)
    n_close_mot = np.maximum(np.ceil(closing.t_done / p.dt_out).astype(np.int64), 1)
    n_open_seg = n_pre + n_open_mot + n_post
    n_close_seg = n_warn + n_close_mot + n_post
    n_seg = n_open_seg + n_close_seg
    offsets = np.concatenate([[0], np.cumsum(n_seg)])
    drop_flat = _burst_mask(int(offsets[-1]), np.repeat(p_dcu, n_seg), p.dropout_len_mean, rng)
    dropout_frac = np.add.reduceat(drop_flat.astype(np.float64), offsets[:-1]) / n_seg

    # ---- per-cycle features ------------------------------------------------------------------
    closing_time = closing.t_done
    opening_time = opening.t_done
    i_peak = np.maximum(opening.i_peak, closing.i_peak)
    # A jammed door never reaches cruise speed; fall back to the whole-motion mean so the
    # feature stays finite (``pos_close_max`` is the feature that says it never moved).
    no_cruise = closing.n_cruise == 0
    i_mean_cruise = np.where(
        no_cruise,
        closing.i_sum / np.maximum(closing.n_samp, 1.0),
        closing.i_cruise_sum / np.maximum(closing.n_cruise, 1.0),
    )
    i_rms_cruise = np.sqrt(
        np.where(
            no_cruise,
            closing.i2_sum / np.maximum(closing.n_samp, 1.0),
            closing.i2_cruise_sum / np.maximum(closing.n_cruise, 1.0),
        )
    )
    i_end = np.where(
        closing.n_end == 0,
        closing.i_sum / np.maximum(closing.n_samp, 1.0),
        closing.i_end_sum / np.maximum(closing.n_end, 1.0),
    )
    energy_j = opening.energy_j + closing.energy_j
    pwm_mean = closing.pwm_sum / np.maximum(closing.n_samp, 1.0) / p.v_bus
    pos_err_max = np.maximum(opening.err_max, closing.err_max)
    pos_err_rms = np.sqrt(
        (opening.err2_sum + closing.err2_sum) / np.maximum(opening.n_samp + closing.n_samp, 1.0)
    )
    reversal_count = closing.reversals.astype(np.float64)
    obstruction = closing.obstruction
    ls_timeout = closing.ls_timeout | opening.ls_timeout
    # travel actually achieved on the closing stroke (0 for a door that never opened)
    pos_close_max = np.clip(opening.x_end - closing.x_min, 0.0, p.stroke_m)
    profile = (closing.prof_sum / np.maximum(closing.prof_cnt, 1.0)).reshape(n, p.profile_n)
    # carry the last observed bin forward where the leaf never reached it (stall, shock stop)
    seen = closing.prof_cnt.reshape(n, p.profile_n) > 0
    profile = np.where(seen, profile, np.nan)
    profile = pd.DataFrame(profile).ffill(axis=1).bfill(axis=1).fillna(0.0).to_numpy()

    phantom = obstruction & ~has_obstacle
    door_fault = ls_timeout | closing.gave_up | (dropout_frac > p.dropout_fault_frac)
    # LATENT ground truth: the shock process itself, not anything the DCU can see. It is
    # written as ``meta_shock_count`` (schema.METADATA_PREFIX) so it stays available for
    # analysis and plotting while schema.feature_columns() keeps it out of the model's X.
    # ``shock_jump_count_est`` beside it is the observable estimator a fleet DCU could actually run.
    meta_shock_count = np.zeros(n)
    if df.shock_wear is not None and df.shock_wear.shocks is not None:
        sh = df.shock_wear.shocks
        edges = np.concatenate([t_cycle, [t_cycle[-1] + dur[-1]]])
        meta_shock_count = np.histogram(sh.times, bins=edges)[0].astype(np.float64)
    # ... and the observable estimator of the same thing, from the cycle statistics only: a
    # running count of shock-like steps, comparable with ``meta_shock_count.cumsum()``.
    shock_jump_count_est = np.cumsum(estimate_shock_jumps((pos_close_max, i_peak, pos_err_max)))

    # ---- functional failure: first of the three plan criteria -----------------------------
    # A genuine passenger obstruction legitimately lengthens the cycle, so the plan's
    # "closing > 5 s" criterion is applied to cycles that ran without reversals.
    fail_close = (closing_time > p.closing_time_limit_s) & (closing.reversals == 0)
    fail_retry = phantom & (closing.reversals >= p.phantom_retry_limit)
    cum_faults = np.concatenate([[0.0], np.cumsum(door_fault.astype(np.float64))])
    lo = np.maximum(np.arange(n) + 1 - p.door_fault_window, 0)
    fail_repeat = (cum_faults[np.arange(n) + 1] - cum_faults[lo]) >= p.door_fault_limit
    tripped = fail_close | fail_retry | fail_repeat
    k_fail = int(np.argmax(tripped)) if tripped.any() else -1
    t_ff: float | None = float(t_end[k_fail]) if k_fail >= 0 else None
    fail_reason = ""
    if k_fail >= 0:
        fail_reason = (
            "closing_time"
            if fail_close[k_fail]
            else ("phantom_retries" if fail_retry[k_fail] else "door_fault_x2_in_10")
        )
        for traj in df.active():
            if traj.t_onset <= t_ff and traj.t_functional_failure is None:
                traj.t_functional_failure = t_ff

    # ---- labels ---------------------------------------------------------------------------
    active = df.active()
    if active:
        sev_each = np.stack([np.asarray(t.severity(t_mid), dtype=np.float64) for t in active])
        which = np.argmax(sev_each, axis=0)
        fault_type = np.array([active[j].fault_type for j in which], dtype=object)
        fault_type = np.where(sev_each.max(axis=0) > 0.0, fault_type, "healthy")
        t_ref = min(
            (t.t_functional_failure if t.t_functional_failure is not None else t.t_failure)
            for t in active
        )
        rul_s = np.maximum(t_ref - t_mid, 0.0).astype(np.float32)
        alarm_3d = (t_mid >= t_ref - 3.0 * SEC_PER_DAY) & (t_mid <= t_ref)
    else:
        fault_type = np.full(n, "healthy", dtype=object)
        rul_s = np.full(n, np.nan, dtype=np.float32)
        alarm_3d = np.zeros(n, dtype=bool)
    # no-fault-found: an alarm raised with nothing injected [rail_phm 1.3, R73]
    alarmed = obstruction | ls_timeout | door_fault
    fault_type = np.where(alarmed & (sev_label <= 0.0), "nff", fault_type)
    is_faulty = sev_label > 0.0

    features = pd.DataFrame(
        {
            "run_id": run_id,
            "source": source,
            "train_id": service.train_id,
            "car": np.int8(car),
            "subsystem": "door",
            "component_id": cid,
            "cycle_id": np.arange(n, dtype=np.int64),
            "t_start": to_timestamp(t_cycle, service.t0),
            "t_end": to_timestamp(t_end, service.t0),
            "closing_time": closing_time,
            "opening_time": opening_time,
            "i_peak": i_peak,
            "i_mean_cruise": i_mean_cruise,
            "i_rms_cruise": i_rms_cruise,
            "i_end": i_end,
            "i_start_peak": closing.i_start_peak,
            "energy_J": energy_j,
            "pwm_mean": pwm_mean,
            "pos_err_max": pos_err_max,
            "pos_err_rms": pos_err_rms,
            "pos_open_max": opening.x_max,
            "pos_close_max": pos_close_max,
            "reversal_count": reversal_count,
            "obstruction": obstruction.astype(np.float32),
            "obs_detect_s": closing.obs_detect_s,
            "ls_timeout": ls_timeout.astype(np.float32),
            "dropout_frac": dropout_frac,
            "shock_jump_count_est": shock_jump_count_est,
            "warning_time": np.full(n, p.warning_s),
            "T_motor": t_motor_end,
            "T_amb": t_amb,
            "load_frac": load,
            "hour_of_day": hour,
            "current_profile_50": [row.astype(np.float32).tolist() for row in profile],
            "meta_shock_count": meta_shock_count.astype(np.float32),
            "fault_type": fault_type,
            "severity": sev_label.astype(np.float32),
            "rul_s": rul_s,
            "is_faulty": is_faulty,
            "alarm_window_3d": alarm_3d,
        }
    )
    features = S.coerce_features(features)

    # ---- events ---------------------------------------------------------------------------
    ev_t: list[float] = []
    ev_kind: list[str] = []
    ev_detail: list[str] = []

    def _ev(t_s: float, kind: str, detail: dict[str, Any]) -> None:
        ev_t.append(float(t_s))
        ev_kind.append(kind)
        ev_detail.append(json.dumps(detail, default=str))

    for k in np.flatnonzero(obstruction):
        lat = closing.obs_detect_s[k]
        _ev(
            t_close_start[k] + (0.0 if not np.isfinite(lat) else lat),
            "obstruction",
            {
                "class": ("hard" if bool(obs_hard[k]) else "soft") if bool(has_obstacle[k]) else "phantom",
                "detect_s": None if not np.isfinite(lat) else round(float(lat), 4),
                "within_deadline": bool(np.isfinite(lat) and lat <= p.obs_detect_deadline_s),
                "retries": int(closing.reversals[k]),
            },
        )
    for k in np.flatnonzero(closing.reversals > 0):
        _ev(t_close_start[k], "reversal", {"count": int(closing.reversals[k])})
    for k in np.flatnonzero(ls_timeout):
        _ev(t_end[k], "ls_timeout", {"closing_time": round(float(closing_time[k]), 3)})
    for k in np.flatnonzero(door_fault):
        _ev(t_end[k], "door_fault", {"ls_timeout": bool(ls_timeout[k]), "gave_up": bool(closing.gave_up[k])})
    for k in np.flatnonzero(dropout_frac > 0.2):
        _ev(t_end[k], "dropout", {"dropout_frac": round(float(dropout_frac[k]), 4)})
    for k in np.flatnonzero(overrun):
        _ev(t_cycle[k] + dur[k], "station_fault", {"overrun_s": round(float(t_end[k] - t_cycle[k] - dur[k]), 3)})
    if df.shock_wear is not None and df.shock_wear.shocks is not None:
        for ts, mag in zip(df.shock_wear.shocks.times, df.shock_wear.shocks.magnitudes):
            if ts <= service.duration_s:
                _ev(ts, "shock", {"magnitude": round(float(mag), 4)})
    if t_ff is not None:
        _ev(t_ff, "functional_failure", {"criterion": fail_reason, "cycle_id": k_fail})

    if ev_t:
        events = pd.DataFrame(
            {
                "run_id": run_id,
                "timestamp": to_timestamp(np.asarray(ev_t), service.t0),
                "train_id": service.train_id,
                "car": np.int8(car),
                "subsystem": "door",
                "component_id": cid,
                "event": ev_kind,
                "detail_json": ev_detail,
            }
        ).sort_values("timestamp", kind="stable")
        events = S.coerce_events(events)
    else:
        events = S.empty_events()

    # ---- 50 Hz telemetry for the stored cycles ---------------------------------------------
    def _fit(a: np.ndarray, m: int, fill: float | None = None) -> np.ndarray:
        """Trim or pad ``a`` to exactly ``m`` samples (padding holds the last value)."""
        if a.size >= m:
            return a[:m]
        tail = float(a[-1]) if (fill is None and a.size) else (fill or 0.0)
        return np.concatenate([a, np.full(m - a.size, tail)])

    x_open_end = opening.x_end
    seg_t: list[np.ndarray] = []
    chans: dict[str, list[np.ndarray]] = {s: [] for s in S.SIGNALS["door"]}
    dt_out = p.dt_out
    for j, k in enumerate(store_idx):
        k = int(k)
        mo, mc = int(n_open_mot[k]), int(n_close_mot[k])
        no_, nc_ = int(n_open_seg[k]), int(n_close_seg[k])
        jo = min(int(opening.cap_n[j]), opening.cap_x.shape[0])
        jc = min(int(closing.cap_n[j]), closing.cap_x.shape[0])

        pos = np.concatenate(
            [
                np.zeros(n_pre),
                _fit(opening.cap_x[:jo, j], mo),
                np.full(n_post, x_open_end[k]),
                np.full(n_warn, x_open_end[k]),
                _fit(closing.cap_x[:jc, j], mc),
                np.full(n_post, 0.0),
            ]
        )
        if n_post:
            pos[n_pre + mo : n_pre + mo + n_post] = pos[n_pre + mo - 1]
            pos[-n_post:] = pos[-n_post - 1]
        pos_ref = np.concatenate(
            [
                np.zeros(n_pre),
                _fit(opening.cap_xr[:jo, j], mo),
                np.full(n_post, x_open_end[k]),
                np.full(n_warn, x_open_end[k]),
                _fit(closing.cap_xr[:jc, j], mc),
                np.zeros(n_post),
            ]
        )
        vel = np.concatenate(
            [np.zeros(n_pre), _fit(opening.cap_v[:jo, j], mo, 0.0), np.zeros(n_post),
             np.zeros(n_warn), _fit(closing.cap_v[:jc, j], mc, 0.0), np.zeros(n_post)]
        )
        cur = np.concatenate(
            [np.zeros(n_pre), _fit(opening.cap_i[:jo, j], mo, 0.0), np.zeros(n_post),
             np.zeros(n_warn), _fit(closing.cap_i[:jc, j], mc, 0.0), np.zeros(n_post)]
        )
        volt = np.concatenate(
            [np.zeros(n_pre), _fit(opening.cap_volt[:jo, j], mo, 0.0), np.zeros(n_post),
             np.zeros(n_warn), _fit(closing.cap_volt[:jc, j], mc, 0.0), np.zeros(n_post)]
        )
        obs = np.concatenate(
            [np.zeros(n_pre + mo + n_post + n_warn),
             _fit(closing.cap_obs[:jc, j].astype(np.float64), mc, 0.0), np.zeros(n_post)]
        )
        m_tot = no_ + nc_
        t_open_seg = t_cycle[k] + np.arange(no_) * dt_out
        t_close_seg = (t_close_start[k] - p.warning_s) + np.arange(nc_) * dt_out
        seg_t.append(np.concatenate([t_open_seg, t_close_seg]))

        ls_o = ((pos > p.stroke_m - p.ls_tol_m) & bool(ls_open_ok[k])).astype(np.float64)
        ls_c = ((pos < p.ls_tol_m) & bool(ls_closed_ok[k])).astype(np.float64)
        key = np.concatenate([np.ones(no_), np.zeros(nc_)])
        emerg = np.concatenate([np.zeros(no_), np.full(nc_, 1.0 if bool(door_fault[k]) else 0.0)])
        t_mot = np.linspace(t_motor_start[k], t_motor_end[k], m_tot)

        chans["pos_ref"].append(pos_ref)
        chans["pos"].append(pos)
        chans["vel"].append(vel)
        chans["current"].append(cur)
        chans["voltage"].append(volt)
        chans["pwm"].append(volt / p.v_bus)
        chans["ls_open"].append(ls_o)
        chans["ls_closed"].append(ls_c)
        chans["interlock"].append(ls_c)
        chans["door_key"].append(key)
        chans["emergency_relay"].append(emerg)
        chans["obstruction"].append(obs)
        chans["T_motor"].append(t_mot)

    if seg_t:
        t_all = np.concatenate(seg_t)
        wide = {"timestamp": to_timestamp(t_all, service.t0)}
        drop_all = np.concatenate(
            [drop_flat[offsets[int(k)] : offsets[int(k) + 1]] for k in store_idx]
        )
        cur_drop = _burst_mask(
            t_all.size,
            np.concatenate([np.full(int(n_seg[int(k)]), p_brush_drop[int(k)]) for k in store_idx]),
            p.dropout_len_mean,
            rng,
        )
        for name in S.SIGNALS["door"]:
            arr = np.concatenate(chans[name])
            spec = DOOR_SENSORS.get(name)
            vals = (
                apply_sensor(arr, spec, rng, dt=dt_out)
                if spec is not None
                else arr.astype(np.float32)
            )
            vals = np.asarray(vals, dtype=np.float32)
            vals[drop_all] = np.nan
            if name == "current":
                vals[cur_drop] = np.nan
            wide[name] = vals
        long_door = S.to_long(
            pd.DataFrame(wide),
            "door",
            source=source,
            run_id=run_id,
            train_id=service.train_id,
            car=int(car),
            component_id=cid,
        )
    else:
        long_door = S.empty_long()

    ctx = service.context_long(context_dt, source=source, run_id=run_id)
    return _concat_long(long_door, ctx), features, events


def _concat_long(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatenate already-coerced long frames without re-factorising 20 M categoricals.

    ``pd.concat`` degrades a categorical column to ``object`` when the parts carry different
    categories, and re-coercing a 30-day frame costs ~20 s.  Aligning the categories first
    keeps the concat zero-copy in the dtype sense.
    """
    parts = [f for f in frames if len(f)]
    if not parts:
        return S.empty_long()
    if len(parts) == 1:
        return parts[0].reset_index(drop=True)
    cat_cols = [c for c, d in S.LONG_COLUMNS.items() if d == "category"]
    aligned = []
    for f in parts:
        f = f.copy()
        aligned.append(f)
    for col in cat_cols:
        cats = pd.Index(sorted({str(v) for f in aligned for v in f[col].cat.categories}))
        for f in aligned:
            f[col] = f[col].cat.set_categories(cats)
    out = pd.concat(aligned, ignore_index=True)
    return out[list(S.LONG_COLUMNS)]


def simulate_scenario(
    scenario: Scenario,
    service: Service,
    rng: np.random.Generator,
    params: DoorParams | None = None,
    store_every: int = 10,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run :func:`simulate` for a :class:`~nebulax.sim.common.Scenario` from
    :func:`~nebulax.sim.common.sample_scenarios`, taking ``run_id``, ``car`` and
    ``component_id`` from it.  Build the fault log with ``scenario.fault_log_rows(service.t0)``
    **after** this returns, so it carries ``t_functional_failure``."""
    if scenario.subsystem != "door":
        raise ValueError(
            f"door.simulate_scenario: scenario subsystem is {scenario.subsystem!r}, expected 'door'"
        )
    return simulate(
        params or DoorParams(),
        DoorFaults.from_trajectories(scenario.faults),
        service,
        rng,
        store_every,
        run_id=scenario.run_id,
        car=scenario.car,
        component_id=scenario.component_id,
        **kwargs,
    )


def cycle_features(long: pd.DataFrame, component_id: str | None = None) -> pd.DataFrame:
    """Placeholder re-extraction hook: the simulator already emits the per-cycle feature
    table, so this only slices the stored cycles out of a long frame for inspection.

    ``nebulax.features.cycles`` is the module that re-derives features from *any* door
    telemetry (simulated or Cranfield); this helper exists so ``sim.door`` satisfies the
    plan's ``cycle_features`` interface name without duplicating that logic.
    """
    door = long[long["subsystem"].astype("string") == "door"]
    if component_id is not None:
        door = door[door["component_id"].astype("string") == component_id]
    return S.to_wide(door, "door")
