"""Bogie axle-box simulator: 8 boxes per car, 1 Hz, lumped thermal node + direct vibration
features.

Physics, provenance and the W1 corrections
------------------------------------------
Every constant below is either derived from a published correlation, anchored to a measured
table, or explicitly labelled as our assumption. Where the approved plan and
``docs/research/rail_phm.md`` disagree, **rail_phm.md wins** - the three places that happens
are marked ``W1 CORRECTION``.

*Thermal.* Palmgren friction torque [rail_phm 4.3.1, R150 Eqs. 7-11] feeds a single lumped
node per box::

    M   = 0.5 * mu_eff * F_box * d_bore  +  k_visc * omega**(2/3)
    P   = M * omega                                        (friction power, ~ v)
    hA  = hA_0 + hA_1 * v**0.57                            (forced convection, R149 Eq. 25)
    C_box dT/dt = P + P_soak - hA (T - T_amb)

so the steady rise scales as ``v / v**0.57 = v**0.43`` [rail_phm 4.3.1]. Scaling the measured
30-38 K rise at 200 km/h down to 80 km/h gives **20-26 K**, which is what
:meth:`BearingParams.steady_rise` is calibrated to at ``v = 22.2 m/s``.

Two terms bend the *implemented* log-log exponent above the ideal 0.43: the viscous
``k_visc omega**(2/3)`` torque, which makes ``P`` grow as ``v**(5/3)`` rather than ``v``, and
the natural-convection floor ``hA_0``, which flattens the denominator at low speed. Both are
*our* assumptions and both must stay non-zero (``hA_0`` is the only cooling path of a stabled
box, and ``T_inf = T_amb + P/hA`` is 0/0 without it), so they are kept as small as the rest of
the model allows - ``k_visc = 0.004``, ``hA_0 = 0.8 W/K``. What is left is an implemented
exponent of **0.466** over the 8-22 m/s metro band against the derived 0.43; the residual is
stated rather than hidden, and :meth:`BearingParams.speed_exponent` reports it.

*Severity.* **W1 CORRECTION, the important one.** [R149] swept wheel-flat length and found the
axle box gains **+1.02 K** where the rollers gain +15.6 K - roughly **15x less sensitive**. So
``T_box`` moves ~1 K per severity step (``dT_per_severity_k = 2.0`` K over the full
``s: 0 -> 1``, i.e. ~1 K per 30->60 mm-equivalent step) against a healthy inter-box spread of
**8.4 K** [R149 Table 1] and an ~8 K diurnal ambient swing [rail_phm 0]. The plan's law, which
drove ``T_box`` hard, is *falsified*: it would have made the severe class trivially separable.
Only ``hot_axle_box`` drives temperature hard, and it is acute (gamma=3, 6-48 h).

*Vibration.* Direct feature stream, no waveform [rail_phm 4.3.3]::

    vib_rms   = A_h*(v/22)**2.0 + A_d*s*(v/22)**1.2   # healthy floor ^2 (W1: plan said ^1.2)
    vib_kurt  = 3.0 + K * bump(s)                     # W1: RISES THEN COLLAPSES, not 3 + 6 s^2
    vib_crest = 4.5 + C * bump(s)                     # W1: healthy intercept 4.5, not 3
    vib_bpfo  = (B_h + B_d*s) * (v/22)**2             # absolute scale OURS; ratio from Ottawa

``bump`` is an asymmetric log-Gaussian peaking at ``s = 0.35``, fitted to the measured
inner-race-by-defect-size sequence **5.56 -> 21.69 -> 8.06 -> 3.29** [R157]. Healthy kurtosis
2.76-2.96 and healthy crest 4.22-5.59 are measured [R157]; the healthy RMS magnitude is
anchored on 8.8 m/s^2 at 300 km/h [R150] scaled by ``v**2`` to 0.62 m/s^2 at 22 m/s.

``vib_rms`` and ``vib_crest`` are **zero-mean (AC-coupled) broadband statistics**: they
synthesise the vibration about its own mean and carry no sensor DC bias, so they are directly
comparable with the AC-coupled ``vib_rms``/``vib_crest`` that ``nebulax.adapters.ottawa``
writes under the same signal names (that adapter mean-removes each window because 21 of the 60
Ottawa records are DC-biased; see ``docs/parameters.md`` bearing section 4). ``vib_kurt`` is
central by definition, so the coupling question never arises for it.

*Vibration calibration against real data.* ``scripts/calibrate_bearing.py`` cuts the 60
University of Ottawa UORED-VAFCLS recordings into 1 s windows at 42 kHz and re-derives the
endpoints of those laws from measured healthy / inner-race / outer-race contrasts
(``docs/parameters.md`` has the full table and the caveats; ``results/sim_checks/
cal_bearing_*.png`` are the overlays). Ottawa is a lab bench rig, so **only dimensionless
faulty/healthy ratios cross over, never absolute magnitudes**:

* healthy ``kurtosis`` 3.15 and ``crest`` 4.62 (per-record medians over 20 bearings)
  **CONFIRM** the [R157] anchors 3.0 / 4.5 - this is the first of our constants verified on
  data we hold rather than on a citation;
* faulty broadband RMS is **5.74x** healthy (bootstrap CI 2.9-7.9x), which sets
  ``vib_rms_defect = A_h*(5.74-1) = 2.94`` (was 2.00, our guess);
* the most impulsive measured class (outer race) reaches ``crest`` **11.82**, above the old
  bump peak of 8.5, so ``vib_crest_gain`` rises 4.0 -> 7.32; its ``kurtosis`` 16.75 stays
  *below* the [R157]-fitted peak of 22, so ``vib_kurt_gain`` is left at 19;
* the healthy BPFO envelope band is **not empty** - outer-race records carry only 10.05x more
  of it - so ``vib_bpfo`` gains a measured healthy floor ``B_h`` instead of being exactly 0
  when healthy, which was the same trivially-separable error rail_phm 4.3.2 caught on
  temperature. The ``s = 1`` endpoint ``B_h + B_d`` is held at its previous 1.50 m/s2 because
  the *absolute* scale of this channel has no source either way.
* the **speed exponents could not be fitted**: UORED-VAFCLS is the constant-speed member of
  the Ottawa family, its records span only 1700-2190 rpm (1.14x within the healthy class),
  and clustered on the record the healthy fit is ``alpha = -6.3 +- 5.0`` (95 % CI
  [-16.8, +4.2], R2 = 0.08) - too wide to refute anything, so the [R158]-derived 2.0 / 1.2
  stand. (Fitting per *window* instead would have given CI [-9.2, -3.3] and "refuted" them;
  that is pseudo-replication, and the script prints both so the trap is visible.) The
  dataset that can do this - ``ottawa_variable_speed``, DOI 10.17632/v43hmbwxpm.2 [R118],
  13.7-28.9 Hz *within* one record - is not downloaded in this repo yet.

*Positions.* On a **motor car** the leading bearing of a wheelset carries 13.5 % more
roller-raceway contact force than the trailing one and runs measurably hotter, while on a
trailer car the two are thermally identical [R150]. Peers are therefore compared **same side
of the train** [R151], never pooled across the car.

*Duty cycle.* "The bearing cools significantly by the wind while driving and warms up while
standing still" [R151]. A lumped node alone cannot do this: stop the train and the friction
power vanishes together with the forced convection, so the box would merely decay towards
ambient. The mechanism we model is the **brake/hub heat reservoir** - every stop dumps
``brake_soak_frac`` of the braking energy into the disc-hub-axle path, which reaches the box
through a slow first-order lag (``brake_soak_tau_s`` ~ 1.7 h, *our* assumption). While the
train runs that trickle is swamped by ``hA(v)``; at a standstill ``hA`` collapses to ``hA_0``
and the same trickle is enough to push the box back **up**. A 30-minute terminus stand
therefore ends ~1 K *warmer* than staying in service, rising monotonically from the moment the
train stops, and the effect is exposed as the ``dwell_fraction`` /
``dwell_fraction_last_hour`` features - the Singapore terminus stand is the confounder that
generates false alarms. The sign and the magnitude are pinned by
``tests/test_sim_bearing.py::test_a_terminus_stand_warms_the_box_relative_to_service``.

*Thresholds.* **W1 CORRECTION**: the plan's ``T_box > 90 C`` / ``dT > 30 K`` functional-failure
pair is replaced by the published Netherlands Railways operator rule set [R151], because no
harmonised European threshold exists (ERA marks on-board axle-bearing monitoring an open
point; EN 15437-1 excludes alarm criteria from its scope) [rail_phm 3.3]::

    level 1  dT_same_side > 30 K                      (dT = T_box - median of same-side peers)
    level 2  level-1 condition on >= 10 windows       (NS: 10 measurements within 30 days)
    level 3  dT_same_side > 50 K                      -> immediate
    level 4  T_box        > 80 C                      -> immediate
    wayside  T_box        > 115 C                     -> stop immediately

Functional failure = first entry to level >= 3. Note this is also a deviation from
[rail_phm 4.3.4] itself, which still lists ``T_box > 90 C`` as "our definition": we drop the
90 C line entirely and take the NS levels, which fire earlier (i.e. conservatively) and
follow [rail_phm 3.3]'s own reasoning that no harmonised absolute threshold exists.
Wayside snapshots carry a +-5 K scan-location
bias (**our assumption**, motivated by [R112] and bounded by the 8.4 K healthy spread) and an
occasional station-fault mode that biases *every* box of a passing train up, which the NS
data-quality guard (">= 4 boxes hot in one pass => blame the station") exists to catch.
**W1 CORRECTION to the guard itself**: NS publishes it as an absolute 50 C, but a *healthy*
Singapore box already sits at 50-57 C [rail_phm 0], so the absolute form fires on every pass and
would suppress every real alarm. We keep the NS structure and express the level as a rise above
ambient - 35 K, which equals the published 50 C at the ~15 C ambient the rule was written for.

Contract
--------
``simulate(params, faults, service, rng, store_every=10) -> (long, features, events)`` per
:class:`nebulax.sim.common.SimulateFn`. ``run_id``/``car``/``source`` are keyword-only extras,
because the bearing unit of analysis is one whole car's 8 boxes and the protocol's positional
signature carries no scenario. The simulator writes ``t_functional_failure`` back onto each
trajectory; the caller builds the fault log from ``Scenario.fault_log_rows`` afterwards.

Selftest::

    python -m nebulax.sim.bearing --days 2 --plot /tmp/bearing.png
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from nebulax import schema as S
from nebulax.sim.common import (
    SEC_PER_DAY,
    V_MAX_MS,
    AmbientProfile,
    DegradationTrajectory,
    SensorSpec,
    Service,
    SimResult,
    apply_sensor,
    generate_service,
    to_timestamp,
)

__all__ = [
    "BEARING_SIGNALS",
    "BOX_IDS",
    "BearingFaults",
    "BearingParams",
    "axle_of",
    "opposite_box",
    "same_side_boxes",
    "severity_bump",
    "simulate",
]

#: The five per-box signals this simulator emits continuously (``T_box_wayside`` is sparse).
BEARING_SIGNALS: tuple[str, ...] = ("T_box", "vib_rms", "vib_kurt", "vib_crest", "vib_bpfo")

#: ``axlebox_1L .. axlebox_4R`` in registry order; index == column in every ``(n, 8)`` array.
BOX_IDS: tuple[str, ...] = S.AXLEBOX_COMPONENT_IDS

_G = 9.81
_DEG_FAULTS = frozenset({"bearing_degradation", "outer_race", "inner_race", "ball", "cage"})
_SCAN_BLOCK = 512  # samples per cumprod block in the thermal scan (keeps products >~0.4)


# --------------------------------------------------------------------------------------
# Box geometry helpers
# --------------------------------------------------------------------------------------


def axle_of(box: str) -> int:
    """Axle index ``1..4`` of ``axlebox_<axle><side>``."""
    return int(box.split("_")[1][0])


def side_of(box: str) -> str:
    """``"L"`` or ``"R"``."""
    return box.split("_")[1][1]


def opposite_box(box: str) -> str:
    """The box on the other end of the same axle."""
    return f"axlebox_{axle_of(box)}{'R' if side_of(box) == 'L' else 'L'}"


def same_side_boxes(box: str) -> tuple[str, ...]:
    """The three peer boxes on the same side of the train [R151] - the comparison group."""
    s = side_of(box)
    return tuple(b for b in BOX_IDS if side_of(b) == s and b != box)


_BOX_INDEX: dict[str, int] = {b: i for i, b in enumerate(BOX_IDS)}
_SIDE_IDX: dict[str, np.ndarray] = {
    s: np.array([i for i, b in enumerate(BOX_IDS) if side_of(b) == s], dtype=np.intp)
    for s in ("L", "R")
}
_OPPOSITE_IDX: np.ndarray = np.array([_BOX_INDEX[opposite_box(b)] for b in BOX_IDS], dtype=np.intp)
_AXLE: np.ndarray = np.array([axle_of(b) for b in BOX_IDS], dtype=np.int64)


# --------------------------------------------------------------------------------------
# Severity -> vibration shape
# --------------------------------------------------------------------------------------


def severity_bump(
    s: np.ndarray | float,
    *,
    s_peak: float = 0.35,
    w_lo: float = 0.45,
    w_hi: float = 0.35,
) -> np.ndarray:
    """Asymmetric log-Gaussian in ``[0, 1]``, peaking at ``s_peak`` - the rise-then-collapse
    shape kurtosis and crest follow as a spall widens [rail_phm 4.3.3, R157].

    Zero at ``s <= 0``; ``w_lo``/``w_hi`` are the log-widths below/above the peak. Fitted so
    ``3 + 19 * bump(s)`` reproduces the measured inner-race sequence 5.56 -> 21.69 -> 8.06 ->
    3.29 at ``s = 0.15, 0.35, 0.6, 1.0``.
    """
    x = np.asarray(s, dtype=np.float64)
    out = np.zeros(np.shape(x), dtype=np.float64)
    pos = x > 1e-9
    if not np.any(pos):
        return out
    u = np.log(np.where(pos, x, 1.0) / s_peak)
    w = np.where(u < 0.0, w_lo, w_hi)
    out = np.where(pos, np.exp(-0.5 * (u / w) ** 2), 0.0)
    return out


# --------------------------------------------------------------------------------------
# Params
# --------------------------------------------------------------------------------------


def _default_sensors() -> dict[str, SensorSpec]:
    return {
        # 1 K-class industrial RTD/thermistor on a noisy vehicle bus.
        "T_box": SensorSpec(noise_sigma=0.25, quantum=0.1, dropout_burst_prob=2e-6,
                            dropout_len_mean=8.0, clip=(-20.0, 250.0)),
        "vib_rms": SensorSpec(noise_sigma=0.010, quantum=0.001, clip=(0.0, 200.0)),
        "vib_kurt": SensorSpec(noise_sigma=0.12, quantum=0.01, clip=(1.0, 60.0)),
        "vib_crest": SensorSpec(noise_sigma=0.10, quantum=0.01, clip=(1.0, 40.0)),
        "vib_bpfo": SensorSpec(noise_sigma=0.008, quantum=0.001, clip=(0.0, 100.0)),
    }


@dataclass(frozen=True, slots=True)
class BearingParams:
    """Physical constants of one car's eight axle boxes. Every field carries its provenance.

    ``derived``    = our arithmetic on a published correlation/table.
    ``measured``   = read off a published table.
    ``CALIBRATED`` = fitted by us on real data in ``data/raw/`` - see
                     ``scripts/calibrate_bearing.py`` and ``docs/parameters.md``.
    ``ours``       = a modelling assumption with no source; labelled as such on the slide.
    """

    # --- geometry and load -----------------------------------------------------------
    r_wheel_m: float = 0.425                 # plan; R151 wheel 850 mm
    d_bore_m: float = 0.13                   # plan, axle-box bearing bore
    tare_axle_t: float = 10.0                # ours: ~40 t tare over 4 axles
    payload_axle_t: float = 6.0              # ours: crush load adds ~24 t per car
    motor_lead_load_gain: float = 0.135      # measured [R150]: +13.5 % on the leading bearing
    motor_cars: tuple[int, ...] = (2, 3, 4, 5)   # DT-M-M-M-M-DT

    # --- friction --------------------------------------------------------------------
    mu_0: float = 0.0018                     # plan, Palmgren-equivalent effective mu
    k_visc: float = 0.004                    # ours; Palmgren M0 term, kept small so P stays ~ v

    # --- thermal ---------------------------------------------------------------------
    C_box_j_per_k: float = 19_000.0          # plan; tau = C/hA ~ 17 min in service
    hA_0_w_per_k: float = 0.8                # ours: natural-convection floor (tau ~6.6 h stabled)
    hA_1_w_per_k: float = 3.057              # derived: calibrated to steady_rise(22.2, 0.5) ~ 21 K
    h_speed_exp: float = 0.57                # measured [R149 Eq. 25]: h_a ~ v^0.57
    brake_soak_frac: float = 0.006           # ours: brake energy reaching the box via the hub
    brake_soak_tau_s: float = 6000.0         # ours: disc/hub reservoir -> box lag (~1.7 h)
    T_init_offset_k: float = 2.0             # ours: boxes start slightly above ambient

    # --- per-box scatter (this is what the 8.4 K healthy spread is made of) ------------
    mu_scatter_sigma: float = 0.05           # plan: mu +-5 %
    hA_scatter_sigma: float = 0.10           # plan: hA +-10 %
    sensor_bias_sigma_k: float = 0.5         # plan: sensor bias +-0.5 K

    # --- severity -> temperature (the W1 correction) ----------------------------------
    dT_per_severity_k: float = 2.0           # measured-scaled [R149]: +1.02 K per 30->60 mm step
    hot_box_dT_k: float = 55.0               # ours: the one fault that drives T_box hard

    # --- vibration -------------------------------------------------------------------
    vib_rms_healthy: float = 0.62            # derived [R150]: 8.8 m/s^2 @ 300 km/h scaled by v^2
    vib_rms_defect: float = 2.94             # CALIBRATED on Ottawa: faulty AC-RMS 5.74x healthy
    vib_speed_exp_healthy: float = 2.0       # measured [R158] ~2.3; Ottawa cannot refute it (CI below)
    vib_speed_exp_defect: float = 1.2        # measured [R158] 1.0-1.3; Ottawa cannot refute it
    vib_kurt_healthy: float = 3.0            # measured [R157] 2.76-2.96; Ottawa healthy 3.15 CONFIRMS
    vib_kurt_gain: float = 19.0              # fitted [R157]: peak ~22 at s ~ 0.35; > Ottawa's 16.75
    vib_crest_healthy: float = 4.5           # measured [R157] 4.22-5.59; Ottawa healthy 4.62 CONFIRMS
    vib_crest_gain: float = 7.32             # CALIBRATED on Ottawa: outer-race crest 11.82
    kurt_s_peak: float = 0.35
    kurt_w_lo: float = 0.45
    kurt_w_hi: float = 0.35
    vib_bpfo_healthy: float = 0.149          # CALIBRATED on Ottawa: healthy BPFO floor is NOT 0
    vib_bpfo_gain: float = 1.351             # OURS (absolute scale); Ottawa sets only the 10.05x ratio
    v_meas_min_ms: float = 2.0               # below this the vibration channels see only noise

    # --- wayside detector -------------------------------------------------------------
    wayside_passes_per_day: int = 3          # ours
    wayside_bias_k: float = 5.0              # ours [R112]: IR scan location dominates the reading
    wayside_station_fault_prob: float = 0.02 # ours: per pass
    wayside_station_bias_k: float = 16.0     # ours: the mode the NS guard exists to catch
    wayside_guard_boxes: int = 4             # [R151]: >= 4 boxes "hot" => blame the station
    wayside_guard_rise_k: float = 35.0       # W1: the NS 50 C guard, re-expressed above ambient

    # --- NS operator thresholds [R151] -------------------------------------------------
    level1_dT_k: float = 30.0
    level3_dT_k: float = 50.0
    level4_abs_c: float = 80.0
    wayside_stop_abs_c: float = 115.0
    level2_min_windows: int = 10             # [R151]: 10 measurements within 30 days

    # --- sensor faults ----------------------------------------------------------------
    sensor_offset_k: float = 6.0             # ours: a plausible miscalibration

    # --- sampling / windowing ----------------------------------------------------------
    dt_s: float = 1.0                        # registry fs_hz = 1.0 for every bearing signal
    window_s: float = 300.0                  # plan: 5-minute windows
    context_dt_s: float = 10.0               # context rate; T_box tau is ~17 min, 10 s is ample
    chunk_days: float = 1.0                  # memory bound: one day of (n, 8) arrays at a time
    dwell_v_ms: float = 0.5                  # below this the train counts as standing
    dwell_history_windows: int = 12          # 12 x 5 min = dwell_fraction_last_hour
    residual_fit_days: float = 3.0           # commissioning baseline for the healthy model
    residual_min_windows: int = 24
    store_tail_days: float = 2.0             # store every window in the last 2 d before failure

    sensors: Mapping[str, SensorSpec] = field(default_factory=_default_sensors)

    # ---------------------------------------------------------------- derived quantities

    def axle_force_n(self, load_frac: np.ndarray | float) -> np.ndarray:
        """Radial force per box (half the axle load) at passenger load ``load_frac``."""
        axle_t = self.tare_axle_t + self.payload_axle_t * np.asarray(load_frac, dtype=np.float64)
        return axle_t * 1000.0 * _G / 2.0

    def position_gain(self, car: int) -> np.ndarray:
        """Per-box radial-load multiplier. Motor cars load the leading box of each bogie
        13.5 % harder [R150]; trailer-car boxes of a wheelset are identical."""
        g = np.ones(len(BOX_IDS), dtype=np.float64)
        if car in self.motor_cars:
            g[np.isin(_AXLE, (1, 3))] = 1.0 + self.motor_lead_load_gain
        return g

    def hA(self, v: np.ndarray | float) -> np.ndarray:
        """Convective conductance ``hA_0 + hA_1 v^0.57`` [W/K] [R149 Eq. 25]."""
        vv = np.clip(np.asarray(v, dtype=np.float64), 0.0, None)
        return self.hA_0_w_per_k + self.hA_1_w_per_k * vv**self.h_speed_exp

    def friction_power_w(
        self, v: np.ndarray | float, load_frac: np.ndarray | float, mu_eff: np.ndarray | float
    ) -> np.ndarray:
        """``P = (0.5 mu F d + k_visc omega^(2/3)) omega`` [W]."""
        vv = np.clip(np.asarray(v, dtype=np.float64), 0.0, None)
        omega = vv / self.r_wheel_m
        M = 0.5 * np.asarray(mu_eff, dtype=np.float64) * self.axle_force_n(load_frac) * self.d_bore_m
        M = M + self.k_visc * omega ** (2.0 / 3.0)
        return M * omega

    def steady_rise(self, v: float = V_MAX_MS * 1.009, load_frac: float = 0.5) -> float:
        """Steady-state rise above ambient [K] of a healthy nominal box at constant ``v``.

        The default ``v = 22.2 m/s`` is 80 km/h, where [rail_phm 4.3.1] puts a healthy metro
        box at **20-26 K**.
        """
        return float(self.friction_power_w(v, load_frac, self.mu_0) / self.hA(v))

    def speed_exponent(self, v_lo: float = 8.0, v_hi: float = 22.2, load_frac: float = 0.5) -> float:
        """The *implemented* log-log exponent of ``steady_rise(v)``.

        The derivation in [rail_phm 4.3.1] gives 0.43 for a pure ``P ~ v`` / ``h ~ v^0.57``
        pair. The viscous torque term (``P`` grows a little faster than ``v``) and the
        natural-convection floor ``hA_0`` (the denominator grows a little slower than
        ``v^0.57``) both lift it; with both kept as small as the standstill physics allows the
        implemented value is ~0.466. We report the number rather than pretending it is 0.43.
        """
        return float(
            np.log(self.steady_rise(v_hi, load_frac) / self.steady_rise(v_lo, load_frac))
            / np.log(v_hi / v_lo)
        )

    def _mu_gain_per_k(self, v: float = V_MAX_MS * 1.009, load_frac: float = 0.5) -> float:
        """Fractional increase in ``mu`` that raises the steady rise by 1 K at the reference
        condition. Used to convert ``dT_per_severity_k`` / ``hot_box_dT_k`` into a friction law."""
        omega = v / self.r_wheel_m
        p_load = 0.5 * self.mu_0 * self.axle_force_n(load_frac) * self.d_bore_m * omega
        return float(self.hA(v) / p_load)

    @property
    def mu_gain_degradation(self) -> float:
        """``mu_eff = mu_0 (1 + g s)`` for ordinary defect growth - deliberately tiny."""
        return self.dT_per_severity_k * self._mu_gain_per_k()

    @property
    def mu_gain_hot_box(self) -> float:
        """``mu_eff = mu_0 (1 + g s)`` for ``hot_axle_box`` - the only fault that drives heat."""
        return self.hot_box_dT_k * self._mu_gain_per_k()


# --------------------------------------------------------------------------------------
# Faults
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class BearingFaults:
    """Trajectories injected into this car's boxes, routed by ``fault_type``.

    ``degradation`` covers ``bearing_degradation``/``outer_race``/``inner_race``/``ball``/
    ``cage``: they move vibration a lot and ``T_box`` by ~1 K per severity step.
    ``hot_box`` is ``hot_axle_box``, the acute thermal fault. ``sensor_stuck``/
    ``sensor_offset`` corrupt the measurement chain only, never the physics.
    """

    degradation: tuple[DegradationTrajectory, ...] = ()
    hot_box: tuple[DegradationTrajectory, ...] = ()
    sensor_stuck: tuple[DegradationTrajectory, ...] = ()
    sensor_offset: tuple[DegradationTrajectory, ...] = ()

    @classmethod
    def from_trajectories(cls, faults: Iterable[DegradationTrajectory]) -> "BearingFaults":
        """Route a flat sequence of trajectories into the four buckets.

        Raises ``ValueError`` on a non-bearing trajectory or an unroutable fault type.
        """
        deg: list[DegradationTrajectory] = []
        hot: list[DegradationTrajectory] = []
        stuck: list[DegradationTrajectory] = []
        off: list[DegradationTrajectory] = []
        for f in faults:
            if f.subsystem != "bearing":
                raise ValueError(
                    f"BearingFaults.from_trajectories: trajectory on subsystem {f.subsystem!r}; "
                    "the bearing simulator only accepts subsystem='bearing'"
                )
            if f.fault_type in _DEG_FAULTS:
                deg.append(f)
            elif f.fault_type == "hot_axle_box":
                hot.append(f)
            elif f.fault_type == "sensor_stuck":
                stuck.append(f)
            elif f.fault_type == "sensor_offset":
                off.append(f)
            elif f.fault_type in ("healthy", "nff"):
                continue
            else:  # pragma: no cover - schema already restricts the vocabulary
                raise ValueError(
                    f"BearingFaults.from_trajectories: no route for fault_type {f.fault_type!r}; "
                    f"expected one of {sorted(_DEG_FAULTS)} + hot_axle_box/sensor_stuck/sensor_offset"
                )
        return cls(tuple(deg), tuple(hot), tuple(stuck), tuple(off))

    def all(self) -> tuple[DegradationTrajectory, ...]:
        """Every injected trajectory, in routing order."""
        return (*self.degradation, *self.hot_box, *self.sensor_stuck, *self.sensor_offset)

    def __bool__(self) -> bool:
        return bool(self.all())


def _as_faults(faults: BearingFaults | Sequence[DegradationTrajectory] | None) -> BearingFaults:
    if faults is None:
        return BearingFaults()
    if isinstance(faults, BearingFaults):
        return faults
    return BearingFaults.from_trajectories(faults)


# --------------------------------------------------------------------------------------
# Numerics
# --------------------------------------------------------------------------------------


def _first_order_scan(alpha: np.ndarray, beta: np.ndarray, T0: np.ndarray) -> np.ndarray:
    """Vectorised ``T[i] = alpha[i] * T[i-1] + beta[i]`` over axis 0, boxes on axis 1.

    Solved in blocks with a cumulative product so the cost is ``n / 512`` numpy ops rather
    than ``n`` Python iterations; the block size keeps ``prod(alpha)`` above ~0.4, far from
    underflow. Exact for the exponential discretisation of a first-order node.
    """
    n = alpha.shape[0]
    out = np.empty_like(beta)
    T = np.asarray(T0, dtype=np.float64)
    for lo in range(0, n, _SCAN_BLOCK):
        a = alpha[lo : lo + _SCAN_BLOCK]
        b = beta[lo : lo + _SCAN_BLOCK]
        A = np.cumprod(a, axis=0)
        out[lo : lo + a.shape[0]] = A * (T + np.cumsum(b / A, axis=0))
        T = out[lo + a.shape[0] - 1]
    return out


def _one_pole(x: np.ndarray, tau_s: float, dt: float, carry: float) -> tuple[np.ndarray, float]:
    """First-order lag ``y[i] = r y[i-1] + (1-r) x[i]``, returning ``(y, y[-1])``."""
    r = float(np.exp(-dt / max(tau_s, 1e-9)))
    n = x.size
    if n == 0:
        return x.copy(), carry
    # y[i] = r^(i+1) carry + (1-r) sum_j r^(i-j) x[j]; the block scan handles it in O(n/512).
    alpha = np.full((n, 1), r)
    beta = ((1.0 - r) * x).reshape(n, 1)
    y = _first_order_scan(alpha, beta, np.array([carry]))[:, 0]
    return y, float(y[-1])


def _rolling_mean_prev(x: np.ndarray, k: int) -> np.ndarray:
    """Causal mean of the last ``k`` samples (shorter at the start)."""
    c = np.concatenate(([0.0], np.cumsum(np.asarray(x, dtype=np.float64))))
    i = np.arange(x.size)
    lo = np.maximum(i - k + 1, 0)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)


def _severity_matrix(
    trajs: Sequence[DegradationTrajectory], t: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """``(len(t), 8)`` severity, max over trajectories sharing a box."""
    s = np.zeros((t.size, len(BOX_IDS)), dtype=np.float64)
    for f in trajs:
        k = _BOX_INDEX[f.component_id]
        s[:, k] = np.maximum(s[:, k], f.severity(t, rng))
    return s


# --------------------------------------------------------------------------------------
# simulate
# --------------------------------------------------------------------------------------


def simulate(
    params: BearingParams,
    faults: BearingFaults | Sequence[DegradationTrajectory] | None,
    service: Service,
    rng: np.random.Generator,
    store_every: int = 10,
    *,
    run_id: str = "run0",
    car: int = 3,
    source: str = "sim",
) -> SimResult:
    """Simulate one car's eight axle boxes over ``service``.

    Returns ``(long, features, events)``: 1 Hz telemetry for every ``store_every``-th 5-minute
    window (plus every window in the last ``store_tail_days`` before a failure) together with
    the train-context rows, a feature row per box per window for **all** windows, and the event
    log. ``t_functional_failure`` is written back onto every trajectory that was active on the
    failing box.

    Cost is ~1 s per simulated day (all 8 boxes) and memory is bounded by ``chunk_days``.
    """
    if not isinstance(params, BearingParams):
        raise TypeError(f"simulate: params must be a BearingParams, got {type(params).__name__}")
    if not 1 <= int(car) <= S.MAX_CAR:
        raise ValueError(f"simulate: car must lie in 1..{S.MAX_CAR} (0 is unit level), got {car}")
    if store_every < 1:
        raise ValueError(f"simulate: store_every must be >= 1, got {store_every}")
    fl = _as_faults(faults)
    dt = params.dt_s
    win = int(round(params.window_s / dt))
    if win < 2:
        raise ValueError(f"simulate: window_s/dt_s must be >= 2 samples, got {win}")

    n_box = len(BOX_IDS)
    t0 = service.t0
    total_s = service.duration_s
    # --- per-box scatter: this is what makes the healthy inter-box spread ~8 K -----------
    mu_scatter = 1.0 + rng.normal(0.0, params.mu_scatter_sigma, n_box)
    hA_scatter = 1.0 + rng.normal(0.0, params.hA_scatter_sigma, n_box)
    bias_k = rng.normal(0.0, params.sensor_bias_sigma_k, n_box)
    pos_gain = params.position_gain(int(car))
    g_deg = params.mu_gain_degradation
    g_hot = params.mu_gain_hot_box

    # --- windows to store full telemetry for -------------------------------------------
    n_win_total = int(np.floor(total_s / params.window_s))
    if n_win_total < 1:
        raise ValueError(
            f"simulate: service is shorter than one {params.window_s:g} s window "
            f"({total_s:g} s); raise days or lower window_s"
        )
    win_end_s = (np.arange(n_win_total, dtype=np.float64) + 1.0) * params.window_s
    store_mask = (np.arange(n_win_total) % store_every) == 0
    tail = params.store_tail_days * SEC_PER_DAY
    for f in fl.all():
        store_mask |= (win_end_s > f.t_failure - tail) & (win_end_s <= f.t_failure + params.window_s)

    # --- wayside pass times --------------------------------------------------------------
    pass_times = _wayside_pass_times(service, params, rng)

    # --- chunked integration --------------------------------------------------------------
    chunk_win = max(int(round(params.chunk_days * SEC_PER_DAY / params.window_s)), 1)
    T_state = None
    stuck_val = np.full(n_box, np.nan)
    soak_carry = 0.0
    v_carry = 0.0
    agg: dict[str, list[np.ndarray]] = {}
    long_parts: list[pd.DataFrame] = []
    wayside_rows: list[pd.DataFrame] = []
    wayside_events: list[dict[str, Any]] = []

    for w0 in range(0, n_win_total, chunk_win):
        w1 = min(w0 + chunk_win, n_win_total)
        t_start = w0 * params.window_s
        t_end = w1 * params.window_s
        tl = service.timeline(dt, t_start, t_end)
        n = len(tl)
        if n == 0:
            continue
        v = tl.speed
        T_amb = tl.T_amb
        load = tl.load_frac
        t_abs = tl.t

        # severities
        s_deg = _severity_matrix(fl.degradation, t_abs)
        s_hot = _severity_matrix(fl.hot_box, t_abs)

        # friction power per box
        mu_eff = mu_scatter[None, :] * params.mu_0 * (1.0 + g_deg * s_deg + g_hot * s_hot)
        F_box = params.axle_force_n(load)[:, None] * pos_gain[None, :]
        omega = (v / params.r_wheel_m)[:, None]
        M = 0.5 * mu_eff * F_box * params.d_bore_m + params.k_visc * omega ** (2.0 / 3.0)
        P = M * omega

        # Brake/hub heat reservoir: a slow lag on the braking power, which is what makes a
        # standing box warm up again once hA has collapsed to hA_0 [R151]. See the module
        # docstring, "Duty cycle".
        dv = np.diff(np.concatenate(([v_carry], v))) / dt
        v_carry = float(v[-1])
        decel = np.clip(-dv, 0.0, None)
        m_box = (params.tare_axle_t + params.payload_axle_t * load) * 1000.0 / 2.0
        p_brake = params.brake_soak_frac * m_box * decel * v
        p_soak, soak_carry = _one_pole(p_brake, params.brake_soak_tau_s, dt, soak_carry)
        P = P + p_soak[:, None]

        # exponential discretisation of C dT/dt = P - hA (T - T_amb)
        hA = params.hA(v)[:, None] * hA_scatter[None, :]
        a = hA / params.C_box_j_per_k
        T_inf = T_amb[:, None] + P / hA
        alpha = np.exp(-a * dt)
        beta = (1.0 - alpha) * T_inf
        if T_state is None:
            T_state = T_amb[0] + params.T_init_offset_k + np.zeros(n_box)
        T = _first_order_scan(alpha, beta, T_state)
        T_state = T[-1].copy()

        # --- measurement chain -----------------------------------------------------------
        meas = np.empty_like(T)
        spec_T = params.sensors["T_box"]
        for k in range(n_box):
            meas[:, k] = apply_sensor(T[:, k], spec_T, rng, dt=dt, t0=t_start) + bias_k[k]
        for f in fl.sensor_offset:
            k = _BOX_INDEX[f.component_id]
            meas[t_abs >= f.t_onset, k] += params.sensor_offset_k
        for f in fl.sensor_stuck:
            k = _BOX_INDEX[f.component_id]
            on = t_abs >= f.t_onset
            if on.any():
                if not np.isfinite(stuck_val[k]):
                    stuck_val[k] = meas[np.argmax(on), k]
                meas[on, k] = stuck_val[k]

        # --- vibration feature stream ----------------------------------------------------
        gate = np.clip(v / params.v_meas_min_ms, 0.0, 1.0)[:, None]
        vr = (v / V_MAX_MS)[:, None]
        bump = severity_bump(
            s_deg, s_peak=params.kurt_s_peak, w_lo=params.kurt_w_lo, w_hi=params.kurt_w_hi
        )
        vib_rms = (
            params.vib_rms_healthy * vr**params.vib_speed_exp_healthy
            + params.vib_rms_defect * s_deg * vr**params.vib_speed_exp_defect
        )
        vib_kurt = params.vib_kurt_healthy + params.vib_kurt_gain * bump * gate
        vib_crest = params.vib_crest_healthy + params.vib_crest_gain * bump * gate
        vib_bpfo = (params.vib_bpfo_healthy + params.vib_bpfo_gain * s_deg) * vr**2.0
        chans = {"vib_rms": vib_rms, "vib_kurt": vib_kurt, "vib_crest": vib_crest, "vib_bpfo": vib_bpfo}
        for name, arr in chans.items():
            spec = params.sensors[name]
            for k in range(n_box):
                arr[:, k] = apply_sensor(arr[:, k], spec, rng, dt=dt, t0=t_start)

        # --- window aggregates ------------------------------------------------------------
        nw = w1 - w0
        r3 = lambda x: x[: nw * win].reshape(nw, win, n_box)  # noqa: E731
        r1 = lambda x: x[: nw * win].reshape(nw, win)  # noqa: E731
        with np.errstate(invalid="ignore"):
            _push(agg, "T_box_mean", np.nanmean(r3(meas), axis=1))
            _push(agg, "T_box_max", np.nanmax(r3(meas), axis=1))
            _push(agg, "T_box_min", np.nanmin(r3(meas), axis=1))
            _push(agg, "T_box_std", np.nanstd(r3(meas), axis=1))
            for name, arr in chans.items():
                _push(agg, f"{name}_mean", np.nanmean(r3(arr), axis=1))
            _push(agg, "vib_rms_max", np.nanmax(r3(chans["vib_rms"]), axis=1))
        _push(agg, "T_amb", r1(T_amb).mean(axis=1))
        _push(agg, "v_mean", r1(v).mean(axis=1))
        _push(agg, "v_max", r1(v).max(axis=1))
        _push(agg, "load_frac", r1(load).mean(axis=1))
        _push(agg, "dwell_fraction", (r1(v) < params.dwell_v_ms).mean(axis=1))
        _push(agg, "in_service_frac", r1(tl.in_service.astype(np.float64)).mean(axis=1))

        # --- stored telemetry ---------------------------------------------------------------
        sel = np.flatnonzero(store_mask[w0:w1])
        if sel.size:
            idx = (sel[:, None] * win + np.arange(win)[None, :]).ravel()
            long_parts.append(
                _long_block(
                    tl.t[idx], t0, source, run_id, service.train_id, car,
                    {"T_box": meas[idx], **{k: chans[k][idx] for k in chans}},
                )
            )

        # --- wayside snapshots ---------------------------------------------------------------
        in_chunk = pass_times[(pass_times >= t_start) & (pass_times < t_end)]
        if in_chunk.size:
            pidx = np.clip(((in_chunk - t_start) / dt).astype(np.intp), 0, n - 1)
            snap = meas[pidx] + rng.uniform(
                -params.wayside_bias_k, params.wayside_bias_k, size=(pidx.size, n_box)
            )
            station = rng.random(pidx.size) < params.wayside_station_fault_prob
            snap[station] += params.wayside_station_bias_k
            wayside_rows.append(
                _long_block(tl.t[pidx], t0, source, run_id, service.train_id, car,
                            {"T_box_wayside": snap})
            )
            # NS publishes the guard as an absolute 50 C. At Singapore's 24-33 C ambient a
            # HEALTHY box already sits at 50-57 C [rail_phm 0], so the absolute form fires on
            # every pass and would suppress every real alarm. We keep the NS structure and
            # express the level as a rise above ambient: 35 K equals the published 50 C at the
            # ~15 C ambient the rule was written for.
            guard_c = T_amb[pidx] + params.wayside_guard_rise_k
            n_hot = (snap > guard_c[:, None]).sum(axis=1)
            for j, tp in enumerate(in_chunk):
                suppressed = bool(n_hot[j] >= params.wayside_guard_boxes)
                wayside_events.append(
                    _event(run_id, tp, t0, service.train_id, car, "train", "wayside_pass",
                           {"n_boxes_hot": int(n_hot[j]), "suppressed": suppressed,
                            "guard_C": round(float(guard_c[j]), 2),
                            "T_max_C": round(float(np.nanmax(snap[j])), 2)})
                )
                if suppressed:
                    wayside_events.append(
                        _event(run_id, tp, t0, service.train_id, car, "train", "station_fault",
                               {"rule": ">=4 boxes hot in one pass => blame the station [R151], "
                                        "guard re-expressed as T_amb + 35 K for Singapore",
                                "n_boxes_hot": int(n_hot[j]),
                                "guard_C": round(float(guard_c[j]), 2)})
                    )
                if float(np.nanmax(snap[j])) > params.wayside_stop_abs_c and not suppressed:
                    kmax = int(np.nanargmax(snap[j]))
                    wayside_events.append(
                        _event(run_id, tp, t0, service.train_id, car, BOX_IDS[kmax],
                               "hot_box_alarm",
                               {"level": 4, "rule": "wayside immediate stop >115C [R151]",
                                "T_wayside_C": round(float(snap[j, kmax]), 2)})
                    )

    assert T_state is not None
    win_feats = {k: np.concatenate(v, axis=0) for k, v in agg.items()}
    n_win = win_feats["T_box_mean"].shape[0]

    # ---------------------------------------------------------------- peer + residual features
    Tm = win_feats["T_box_mean"]
    peer_med = np.empty_like(Tm)
    for k in range(n_box):
        peers = [_BOX_INDEX[b] for b in same_side_boxes(BOX_IDS[k])]
        peer_med[:, k] = np.nanmedian(Tm[:, peers], axis=1)
    dT_peer = Tm - peer_med
    dT_opp = Tm - Tm[:, _OPPOSITE_IDX]
    dwell_1h = _rolling_mean_prev(win_feats["dwell_fraction"], params.dwell_history_windows)
    resid = _thermal_residual(params, win_feats, peer_med, dwell_1h)

    # ---------------------------------------------------------------- NS alarm levels [R151]
    level, ff_idx, ff_box = _alarm_levels(params, dT_peer, win_feats["T_box_max"])
    t_win_end = (np.arange(n_win, dtype=np.float64) + 1.0) * params.window_s
    events = list(wayside_events)
    events.extend(
        _level_events(run_id, t0, service.train_id, car, level, dT_peer, win_feats["T_box_max"], t_win_end)
    )

    t_ff = float(t_win_end[ff_idx]) if ff_idx is not None else None
    if ff_idx is not None and ff_box is not None:
        events.append(
            _event(run_id, t_ff, t0, service.train_id, car, BOX_IDS[ff_box], "functional_failure",
                   {"rule": "NS level >= 3 [R151]", "level": int(level[ff_idx, ff_box]),
                    "dT_same_side_K": round(float(dT_peer[ff_idx, ff_box]), 2),
                    "T_box_C": round(float(win_feats["T_box_max"][ff_idx, ff_box]), 2)})
        )
        for f in fl.all():
            if _BOX_INDEX[f.component_id] == ff_box and f.t_functional_failure is None and f.t_onset <= t_ff:
                f.t_functional_failure = t_ff

    # ---------------------------------------------------------------- assemble frames
    features = _feature_frame(
        params, fl, win_feats, peer_med, dT_peer, dT_opp, dwell_1h, resid, level,
        run_id=run_id, source=source, train_id=service.train_id, car=car, t0=t0, n_win=n_win
    )
    long = _assemble_long(long_parts, wayside_rows, service, params, source, run_id)
    ev = S.coerce_events(pd.DataFrame(events, columns=list(S.EVENT_LOG_COLUMNS))) if events else S.empty_events()
    return long, features, ev


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _push(agg: dict[str, list[np.ndarray]], key: str, arr: np.ndarray) -> None:
    agg.setdefault(key, []).append(np.asarray(arr, dtype=np.float64))


def _wayside_pass_times(service: Service, params: BearingParams, rng: np.random.Generator) -> np.ndarray:
    """Detector-pass instants: ``wayside_passes_per_day`` per day, jittered, in service only."""
    runs = service.runs()
    if runs.empty or params.wayside_passes_per_day <= 0:
        return np.zeros(0, dtype=np.float64)
    n = int(params.wayside_passes_per_day * service.days)
    mid = (runs["t_start"].to_numpy(dtype=np.float64) + runs["t_end"].to_numpy(dtype=np.float64)) / 2.0
    pick = rng.choice(mid.size, size=min(n, mid.size), replace=False)
    return np.sort(mid[pick])


def _long_block(
    t_s: np.ndarray, t0: pd.Timestamp, source: str, run_id: str, train_id: str, car: int,
    channels: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Stack ``{signal: (m, 8)}`` (or ``(m, n_box)``) into long rows without per-row Python."""
    m = t_s.size
    n_box = len(BOX_IDS)
    sigs = list(channels)
    ts = to_timestamp(t_s, t0)
    return pd.DataFrame(
        {
            "timestamp": np.tile(np.asarray(ts), len(sigs) * n_box),
            "source": source,
            "run_id": run_id,
            "train_id": train_id,
            "car": np.int8(car),
            "subsystem": "bearing",
            "component_id": np.tile(np.repeat(np.asarray(BOX_IDS, dtype=object), m), len(sigs)),
            "signal": np.repeat(np.asarray(sigs, dtype=object), m * n_box),
            "value": np.concatenate([channels[s].T.ravel() for s in sigs]).astype(np.float32),
        }
    )


def _event(
    run_id: str, t_s: float, t0: pd.Timestamp, train_id: str, car: int, component_id: str,
    event: str, detail: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp": to_timestamp(float(t_s), t0),
        "train_id": train_id,
        "car": np.int8(car),
        "subsystem": "bearing",
        "component_id": component_id,
        "event": event,
        "detail_json": json.dumps(dict(detail), default=str),
    }


def _thermal_residual(
    params: BearingParams, win: Mapping[str, np.ndarray], peer_med: np.ndarray, dwell_1h: np.ndarray
) -> np.ndarray:
    """``T_box_mean - T_box_pred`` from a healthy model fitted on the commissioning baseline.

    Inputs follow [R110]/[R151]: speed, ambient, load, the same-side peer median and the duty
    cycle (``dwell_fraction`` now and over the last hour). The model is a per-box least-squares
    fit on the first ``residual_fit_days``, which is what a fleet operator actually has: a
    commissioning baseline, not oracle knowledge of which windows are healthy. If the run is
    shorter than that the whole run is used and the residual is correspondingly optimistic.
    """
    n_win, n_box = peer_med.shape
    X = np.column_stack(
        [
            np.ones(n_win),
            win["v_mean"],
            win["v_max"],
            win["T_amb"],
            win["load_frac"],
            win["dwell_fraction"],
            dwell_1h,
            win["in_service_frac"],
            np.zeros(n_win),  # per-box peer median, filled in below
        ]
    )
    n_fit = int(round(params.residual_fit_days * SEC_PER_DAY / params.window_s))
    n_fit = min(max(n_fit, params.residual_min_windows), n_win)
    out = np.empty((n_win, n_box), dtype=np.float64)
    for k in range(n_box):
        Xk = X.copy()
        Xk[:, -1] = peer_med[:, k]
        y = win["T_box_mean"][:, k]
        ok = np.isfinite(y[:n_fit]) & np.isfinite(Xk[:n_fit]).all(axis=1)
        if ok.sum() < Xk.shape[1] + 2:
            out[:, k] = np.nan
            continue
        coef, *_ = np.linalg.lstsq(Xk[:n_fit][ok], y[:n_fit][ok], rcond=None)
        out[:, k] = y - Xk @ coef
    return out


def _alarm_levels(
    params: BearingParams, dT_peer: np.ndarray, T_max: np.ndarray
) -> tuple[np.ndarray, int | None, int | None]:
    """NS [R151] decision rules -> a per-window per-box level 0-4 and the functional failure."""
    lvl = np.zeros(dT_peer.shape, dtype=np.int8)
    l1 = np.nan_to_num(dT_peer, nan=-np.inf) > params.level1_dT_k
    lvl[l1] = 1
    count = np.cumsum(l1, axis=0)
    lvl[l1 & (count >= params.level2_min_windows)] = 2
    lvl[np.nan_to_num(dT_peer, nan=-np.inf) > params.level3_dT_k] = 3
    lvl[np.nan_to_num(T_max, nan=-np.inf) > params.level4_abs_c] = 4
    hit = np.argwhere(lvl >= 3)
    if hit.size == 0:
        return lvl, None, None
    i = int(hit[0, 0])
    k = int(hit[hit[:, 0] == i][0, 1])
    return lvl, i, k


def _level_events(
    run_id: str, t0: pd.Timestamp, train_id: str, car: int, level: np.ndarray,
    dT_peer: np.ndarray, T_max: np.ndarray, t_win_end: np.ndarray,
) -> list[dict[str, Any]]:
    """One event per box per *new maximum* level - never per window, so the log stays readable."""
    out: list[dict[str, Any]] = []
    running = np.zeros(level.shape[1], dtype=np.int8)
    for i in range(level.shape[0]):
        rose = np.flatnonzero(level[i] > running)
        for k in rose:
            lv = int(level[i, k])
            running[k] = lv
            out.append(
                _event(
                    run_id, float(t_win_end[i]), t0, train_id, car, BOX_IDS[k],
                    "hot_box_alarm" if lv >= 3 else "peer_delta_alarm",
                    {"level": lv, "dT_same_side_K": round(float(dT_peer[i, k]), 2),
                     "T_box_C": round(float(T_max[i, k]), 2),
                     "rule": ["", ">30K", ">30K x10 windows", ">50K", ">80C absolute"][lv]},
                )
            )
    return out


def _feature_frame(
    params: BearingParams, fl: BearingFaults, win: Mapping[str, np.ndarray], peer_med: np.ndarray,
    dT_peer: np.ndarray, dT_opp: np.ndarray, dwell_1h: np.ndarray, resid: np.ndarray,
    level: np.ndarray, *, run_id: str, source: str, train_id: str, car: int,
    t0: pd.Timestamp, n_win: int,
) -> pd.DataFrame:
    n_box = len(BOX_IDS)
    t_start_s = np.arange(n_win, dtype=np.float64) * params.window_s
    t_end_s = t_start_s + params.window_s
    rep = lambda x: np.repeat(np.asarray(x, dtype=np.float64), n_box)  # noqa: E731
    flat = lambda x: np.asarray(x, dtype=np.float64).ravel()  # noqa: E731

    cols: dict[str, Any] = {
        "run_id": run_id,
        "source": source,
        "train_id": train_id,
        "car": np.int8(car),
        "subsystem": "bearing",
        "component_id": np.tile(np.asarray(BOX_IDS, dtype=object), n_win),
        "cycle_id": np.repeat(np.arange(n_win, dtype=np.int64), n_box),
        "t_start": np.repeat(np.asarray(to_timestamp(t_start_s, t0)), n_box),
        "t_end": np.repeat(np.asarray(to_timestamp(t_end_s, t0)), n_box),
        # thermal
        "T_box_mean": flat(win["T_box_mean"]),
        "T_box_max": flat(win["T_box_max"]),
        "T_box_min": flat(win["T_box_min"]),
        "T_box_std": flat(win["T_box_std"]),
        "dT_peer_same_side": flat(dT_peer),
        "dT_opposite": flat(dT_opp),
        "peer_median_same_side": flat(peer_med),
        "thermal_residual": flat(resid),
        # context / confounders
        "T_amb": rep(win["T_amb"]),
        "v_mean": rep(win["v_mean"]),
        "v_max": rep(win["v_max"]),
        "load_frac": rep(win["load_frac"]),
        "dwell_fraction": rep(win["dwell_fraction"]),
        "dwell_fraction_last_hour": rep(dwell_1h),
        "in_service_frac": rep(win["in_service_frac"]),
        # vibration
        "vib_rms_mean": flat(win["vib_rms_mean"]),
        "vib_rms_max": flat(win["vib_rms_max"]),
        "vib_kurt_mean": flat(win["vib_kurt_mean"]),
        "vib_crest_mean": flat(win["vib_crest_mean"]),
        "vib_bpfo_mean": flat(win["vib_bpfo_mean"]),
        "alarm_rule_code": flat(level),
        "axle": np.tile(_AXLE.astype(np.float64), n_win),
    }
    df = pd.DataFrame(cols)
    for c, v in _labels(fl, win, t_end_s, n_win, n_box).items():
        df[c] = v
    return S.coerce_features(df)


def _labels(
    fl: BearingFaults, win: Mapping[str, np.ndarray], t_end_s: np.ndarray, n_win: int, n_box: int
) -> dict[str, np.ndarray]:
    """Ground-truth labels, evaluated at each window end. RUL counts down to the functional
    failure when the simulator found one, otherwise to the trajectory's ``t_failure``."""
    ftype = np.full((n_win, n_box), "healthy", dtype=object)
    sev = np.zeros((n_win, n_box), dtype=np.float64)
    rul = np.full((n_win, n_box), np.nan, dtype=np.float64)
    faulty = np.zeros((n_win, n_box), dtype=bool)
    alarm3 = np.zeros((n_win, n_box), dtype=bool)
    for f in fl.all():
        k = _BOX_INDEX[f.component_id]
        s = np.asarray(f.severity(t_end_s), dtype=np.float64)
        on = t_end_s >= f.t_onset
        take = on & (s >= sev[:, k])
        ftype[take, k] = f.fault_type
        sev[:, k] = np.maximum(sev[:, k], s)
        faulty[:, k] |= on
        t_fail = f.t_functional_failure if f.t_functional_failure is not None else f.t_failure
        r = np.clip(t_fail - t_end_s, 0.0, None)
        rul[:, k] = np.where(np.isnan(rul[:, k]), r, np.minimum(rul[:, k], r))
        alarm3[:, k] |= on & (t_end_s >= t_fail - 3.0 * SEC_PER_DAY) & (t_end_s <= t_fail)
    return {
        "fault_type": ftype.ravel(),
        "severity": np.clip(sev, 0.0, 1.0).ravel(),
        "rul_s": rul.ravel(),
        "is_faulty": faulty.ravel(),
        "alarm_window_3d": alarm3.ravel(),
    }


def _assemble_long(
    parts: list[pd.DataFrame], wayside: list[pd.DataFrame], service: Service,
    params: BearingParams, source: str, run_id: str,
) -> pd.DataFrame:
    ctx = service.context_long(params.context_dt_s, source=source, run_id=run_id)
    frames = [*parts, *wayside]
    if not frames:
        return ctx
    body = S.coerce_long(pd.concat(frames, ignore_index=True))
    out = pd.concat([body, ctx], ignore_index=True)
    return S.coerce_long(out)


# --------------------------------------------------------------------------------------
# Selftest
# --------------------------------------------------------------------------------------


def _demo(days: int, seed: int, store_every: int = 6) -> dict[str, Any]:
    """Healthy vs bearing_degradation vs hot_axle_box on the same service, for the selftest."""
    # The 2-day "keep everything near the failure" tail is sized for a 30-day run; on a short
    # demo it would engulf the whole record, so scale it down here (demo-only, not a param change).
    p = dataclasses.replace(BearingParams(), store_tail_days=min(2.0, days / 4.0))
    svc = generate_service(days, np.random.default_rng(seed), AmbientProfile.singapore_routine())
    out: dict[str, Any] = {"params": p, "service": svc, "runs": {}}
    onset = 0.25 * days * SEC_PER_DAY
    specs = {
        "healthy": (),
        "bearing_degradation": (
            DegradationTrajectory(
                fault_type="bearing_degradation", subsystem="bearing", component_id="axlebox_3R",
                t_onset=onset, t_failure=0.95 * days * SEC_PER_DAY, gamma=1.5,
            ),
        ),
        "hot_axle_box": (
            DegradationTrajectory(
                fault_type="hot_axle_box", subsystem="bearing", component_id="axlebox_2L",
                t_onset=0.55 * days * SEC_PER_DAY, t_failure=0.55 * days * SEC_PER_DAY + 18 * 3600.0,
                gamma=3.0,
            ),
        ),
    }
    for name, fs in specs.items():
        long, feat, ev = simulate(
            p, BearingFaults.from_trajectories(fs), svc, np.random.default_rng(seed + 1),
            store_every=store_every, run_id=f"demo_{name}", car=3
        )
        out["runs"][name] = {"long": long, "features": feat, "events": ev, "faults": fs}
    return out


def _plot(demo: dict[str, Any], path: str) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p: BearingParams = demo["params"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for name, panel in (("healthy", ax[0, 0]), ("hot_axle_box", ax[0, 1])):
        f = demo["runs"][name]["features"]
        for box in BOX_IDS:
            g = f[f["component_id"] == box]
            panel.plot(g["t_end"], g["T_box_mean"], lw=0.8, label=box)
        g = f[f["component_id"] == BOX_IDS[0]]
        panel.plot(g["t_end"], g["T_amb"], "k--", lw=1.0, label="T_amb")
        panel.axhline(p.level4_abs_c, color="r", ls=":", lw=1.0, label="NS level 4 (80 C)")
        panel.set_title(f"T_box_mean - {name}")
        panel.set_ylabel("deg C")
        panel.tick_params(axis="x", rotation=20)
    panel = ax[1, 0]
    for name in ("healthy", "hot_axle_box"):
        f = demo["runs"][name]["features"]
        g = f[f["component_id"] == ("axlebox_2L" if name == "hot_axle_box" else "axlebox_1L")]
        panel.plot(g["t_end"], g["dT_peer_same_side"], lw=0.9, label=name)
    panel.axhline(p.level1_dT_k, color="orange", ls=":", label="level 1 (30 K)")
    panel.axhline(p.level3_dT_k, color="r", ls=":", label="level 3 (50 K)")
    panel.set_title("dT vs same-side peers [R151]")
    panel.legend(fontsize=7)
    panel.tick_params(axis="x", rotation=20)
    s = np.linspace(0, 1, 400)
    b = severity_bump(s, s_peak=p.kurt_s_peak, w_lo=p.kurt_w_lo, w_hi=p.kurt_w_hi)
    ax[1, 1].plot(s, p.vib_kurt_healthy + p.vib_kurt_gain * b, label="vib_kurt")
    ax[1, 1].plot(s, p.vib_crest_healthy + p.vib_crest_gain * b, label="vib_crest")
    ax[1, 1].scatter([0.15, 0.35, 0.6, 1.0], [5.56, 21.69, 8.06, 3.29], c="r", zorder=5,
                     label="[R157] measured inner race")
    ax[1, 1].set_title("rise-then-collapse severity law (W1 correction)")
    ax[1, 1].set_xlabel("severity s")
    ax[1, 1].legend(fontsize=7)
    ax[0, 0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m nebulax.sim.bearing`` - smoke-run the simulator and print a summary."""
    ap = argparse.ArgumentParser(description="Bearing simulator selftest")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store-every", type=int, default=6, help="waveform thinning")
    ap.add_argument("--plot", default=None, help="write a 4-panel PNG here")
    args = ap.parse_args(argv)

    p = BearingParams()
    print(f"steady_rise(22.2 m/s, load 0.5) = {p.steady_rise():.1f} K   "
          f"(rail_phm 4.3.1 target 20-26 K)")
    print(f"implemented speed exponent      = {p.speed_exponent():.3f}  (derived 0.43)")
    print(f"mu gain: degradation {p.mu_gain_degradation:.4f}/s  hot_axle_box {p.mu_gain_hot_box:.3f}/s")
    demo = _demo(args.days, args.seed, args.store_every)
    for name, r in demo["runs"].items():
        f, ev = r["features"], r["events"]
        S.validate_long(r["long"])
        S.validate_features(f, require_labels=True)
        S.validate_events(ev)
        print(
            f"{name:22s} long={len(r['long']):>9,}  features={len(f):>6,}  events={len(ev):>4}  "
            f"T_box max={f['T_box_max'].max():6.1f} C  dT_peer max={f['dT_peer_same_side'].max():6.2f} K  "
            f"kurt max={f['vib_kurt_mean'].max():5.2f}  "
            f"t_ff={[t.t_functional_failure for t in r['faults']]}"
        )
    h = demo["runs"]["healthy"]["features"]
    spread = h.groupby("component_id", observed=True)["T_box_mean"].mean()
    print(f"healthy inter-box spread = {spread.max() - spread.min():.2f} K "
          f"(R149 measured 8.4 K); rise over ambient = {(h['T_box_mean'] - h['T_amb']).mean():.1f} K")
    if args.plot:
        _plot(demo, args.plot)
        print("wrote", args.plot)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
