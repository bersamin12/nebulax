"""Brake air supply / air production unit (APU) simulator, 1 Hz, MetroPT-3 names verbatim.

The unit modelled is the one the MetroPT-3 descriptor describes [R87]: a screw compressor with
an oil sump, a cyclonic separator, an air/oil filter, a twin-tower desiccant dryer and a main
reservoir feeding the brake, the air springs and the auxiliary consumers of one train.
Singapore fleets are electro-pneumatic braked with air-spring bogies, so this module is framed
as *brake air supply + air suspension*, not as a door actuator (plan, "Context").

Constants and where they come from
----------------------------------
**Every number below is now calibrated on MetroPT-3 itself** by ``scripts/calibrate_pneumatic.py``,
which measures the failure-free window 1 Feb - 31 Mar 2020 (445,298 rows, 3,141 compressor
cycles, with a +/-3 day guard around every UCI failure episode - which removes nothing, because
all four episodes are later than 31 March) and prints a constant-by-constant audit. Re-running
it should report ``CONFIRMED`` on all 31 rows. Where the measurement, the approved plan and
``docs/research/rail_phm.md`` disagree, the order of precedence is **measurement > rail_phm >
plan**, and the disagreement is stated:

========================  ===============  ===============  ====================================
quantity                  plan / rail_phm  value used here  evidence
========================  ===============  ===============  ====================================
compressor start          "below 8.2 bar"  **8.06 bar**     median cycle minimum 8.066, corrected
                                                            for the file's 10 s sampling lag
compressor stop           "above 10.2"     10.2 bar         median cycle maximum 10.124 + half a
                                                            sample of charge slope = 10.21
``LPS`` trip              7.0 bar          7.0 bar          [R87]; 40 of 3,141 *healthy* cycles
                                                            touch it
``Motor_current`` states  0 / 4 / 7 A      **0 / 3.79 /     three clean clusters in the file's own
                                             5.92 A**       histogram, gaps at 0.25-3.5 and
                                                            4.25-4.75 A. 7 A is a rounding.
load/unload switching     40 s / 3 s       40 s / 3 s       [R155]; unresolvable at 10 s, kept
unloaded hold             45 s (invented)  **376 s**        unloaded run is 416 s per cycle,
                                                            p25-p75 416-426: a fixed timer
reservoir volume          600 L (guess)    **290 L**        charge + fall endpoint rates sum to
                                                            0.02149 bar/s, see ``V_res_l``
consumption               aux 0.6 NL/s,    **aux 0.076,     air balance: cycle duty 0.089 gives a
                          brake 120 NL       brake 22 NL**  mean draw of 0.547 NL/s, of which the
                                                            healthy leak is already 0.28
dryer tower period        90 s (invented)  **59 s**         median loaded-seconds between
                                                            ``TOWERS`` flips, 9,050 flips
leak law                  Q = 0.2 + 3.0 s  sharp orifice    [R154] rail_phm 4.2
                          (constant NL/s)  Q ~ d^2 * P_abs
leak orifice              0.55 -> 2.2 mm   0.55 ->          the 29 May-7 Jun episode settles at
                                           **2.88 mm**      8.38 bar; see below
oil node ``hA``           18 W/K (plan)    **32.0 W/K**     regression: ``hA/C`` = 0.001282 /s,
                                                            time constant 780 s
oil sink                  ``T_amb``        **T_amb+33.5 K** the fast node relaxes to the warm
                                                            compressor body, not to outdoor air
sensor noise / quantum    assumed          **measured**     second-difference method, see
                                                            ``_default_sensors``
========================  ===============  ===============  ====================================

Three constants are **pinned, not fitted**, and every derived number moves with them:

* ``Q_comp_nls = 7.0`` NL/s. MetroPT-3 has **no flow channel** (``Flowmeter`` is a MetroPT-1/2
  signal [rail_phm 2.1]), so pressure data fixes only the *ratio* ``Q_comp/V_res``. Pinning
  ``V_res = 600 L`` instead would give ``Q_comp = 14.5 NL/s`` and leave every pressure rate,
  duty and feature in this module bit-identical. Both are physical for a metro APU.
* ``C_oil_j_per_k = 25 000``. The thermal regression identifies ``hA/C`` and ``P_state/hA``,
  never the three separately; 25 kJ/K is ~13 L of oil plus the separator shell.
* ``leak_ref_nls = 0.2`` NL/s at 6.2 barg (the plan's healthy leak). MetroPT-3 confirms it as
  an upper-bounded fit: attributing the *whole* quiescent OFF draw to the leak gives an
  equivalent 0.62 mm orifice, so 0.55 mm fits with 0.076 NL/s left over for auxiliaries.

The reservoir is the DOE receiver relation ``V = T*C*Pa/(P1-P2)`` [R155] rearranged, i.e.

    dP_res/dt = (S_supply - C_demand - Q_leak - Q_purge - Q_vent) * P_atm / V_res  [rail_phm 4.2]

Everything that can be vectorised is (the consumption schedule, the severity series, the sensor
chain, the whole per-cycle feature extraction via ``reduceat``). Only the pressure / state
machine / oil-temperature recursion runs as a scalar loop, because it is a feedback loop whose
state at step *k* decides the supply at step *k*: ~2 us/step, so a 30-day run is ~25 s and one
simulated day is far inside the 10 s budget.

What the calibration changed about the *physics*, not only the numbers
----------------------------------------------------------------------
**The healthy APU is leak-dominated.** MetroPT-3's quiescent OFF draw is 0.342 NL/s, of which
the healthy leak alone is 0.28 - genuine auxiliaries are 0.076 NL/s and the whole brake + air
spring budget averages 0.16 NL/s. The first cut's consumption (aux 0.6, brake 120 NL/stop) was
~7x too large and forced a duty of 0.33 against the file's 0.089. Consequences:

* the healthy ``idle_run_ratio`` rises from ~1.8 to ~5.4 in service and stays above DOE's
  "well maintained" 9.0 while stabled [R155];
* a *reservoir* leak now dominates the demand budget very early. ``t_off`` in service reaches
  zero at **s ~ 0.13**, and the compressor stops cycling altogether (so no feature rows at all)
  at **s ~ 0.73**. That is not a defect - the same arithmetic applied to MetroPT-3 says the
  real unit loses its OFF phase at a total draw only ~1.6x healthy, and the 5-7 Jun 2020
  episode indeed shows ``t_off = 0`` with ``t_unloaded`` collapsed from 416 s to 10 s. It does
  mean **the detection-relevant band is ``s <~ 0.15``**, and that the warning it leaves is
  short. Measured on a 30-day ramp (onset day 8, nominal s = 1 on day 26, four seeds per
  gamma) with a rule that never fires on the matched healthy runs - a 12 h rolling median of
  ``idle_run_ratio`` below the p5 of that same statistic over the unit's own first 8 days,
  held 24 h, the alarm dated at the END of the hold - the lead time is **2.0 d at gamma 1,
  2.3 d at gamma 2, 2.5 d at gamma 3** (1.1-3.0 d over seeds), i.e. **short of [R101]'s
  1-4 week band**, and ``gamma`` barely moves
  it: it shifts alarm and failure together. The earlier figures here (6.3 / 9.2 / 10.8 d) were
  **withdrawn on 15 Sep 2026** together with the "use gamma >= 2 for air leaks" advice: they
  came from [R101]'s literal rule (6 h median below the healthy 5th percentile of the *raw*
  feature), which is not implementable on either record - ``idle_run_ratio`` is zero-inflated
  (17 % of MetroPT-3's healthy cycles never reach the OFF phase, and 13-19 % of ours), so its
  healthy p5 is exactly 0. Regenerate the table with
  ``python scripts/calibrate_pneumatic.py --leak-ramp``
  (:func:`calibrate_pneumatic.leak_ramp_lead_time`); docs/parameters.md pneumatic section 3
  carries it with the false-alarm audit.

**The loaded motor current is 5.92 A, not 7 A**, and it *falls* with oil temperature
(``I_loaded_kT`` is negative: thinner oil, less viscous drag). Both were wrong before.

**The oil node is two-time-constant and we model one.** The cycle-scale regression gives
``C/hA = 780 s`` and an OFF asymptote of 52 C, but the record's own cold starts (oil at 18.5 C
on 2020-03-07 and 15.4 C on 2020-08-17, both while charging an empty reservoir from < 2 bar)
put the outdoor ambient at 15-19 C. A single node cannot relax to 52 C on a 780 s constant
*and* sit 34 K above outdoor air: there is a slow, massive warm body - the compressor block and
its bay - that the fast oil node relaxes toward. We keep the fast node and give its sink the
measured offset :attr:`PneumaticParams.T_sump_offset_c`. Fitting the slow node from the
logger's own multi-hour gaps is **not identifiable** (the compressor state during a gap is
unknown; the fit lands at rms 9.5 K with ``T_amb`` anywhere from -4 C to +24 C depending on
which gaps are admitted), so it is recorded as a model limitation, not a number. Running the
*same* thermal constants at Singapore ambient rather than Porto's puts the sump ~10 K hotter,
which is a genuine finding: the 95 C failure line has far less margin here than at Porto.

Heterogeneous demand: why one number per stop was wrong (fixed 15 Sep 2026)
--------------------------------------------------------------------------
Until 15 Sep 2026 every ``run`` segment drew exactly ``brake_nl_per_stop`` and every ``dwell``
exactly its air-spring share. The medians came out right and **every distribution came out
wrong**: against MetroPT-3's 3,141 healthy cycles the two-sample KS distances sat at 0.46-0.75
even where the medians agreed to 3 %. MetroPT-3 says the demand is not one number:

* ``t_off`` spreads **0-1,814 s** (sd 600 s, median 903, 17 % of cycles never reach the OFF
  phase at all) while ours spread 565-1,519 s;
* ``duty_ratio`` spreads **0.015-0.99** with 13 % of cycles above 0.5, against our 0.053-0.104;
* the per-cycle *quiescent* draw read off the OFF decay spreads **0.205-0.351-0.97 NL/s**
  (p95 - median - p5), so even "nothing is happening" is not one number;
* and the three phases are **not equal**: the measured per-phase demand is
  **0.733 (loaded) / 0.473 (unloaded) / 0.342 (off) NL/s**. A burst large enough to matter
  pulls the pressure down fast enough to start the compressor, so it lands inside a loaded
  window *by construction*; ours drew air regardless of state and so polluted the OFF decay,
  which is why the simulated ``dP_dt_off`` was 1.55x too steep.

The fix is structural, and it changes **no calibrated mean**. Per service event (one stop per
run, one levelling opportunity per dwell) the air is now ``mean x M`` with ``M`` a unit-mean
three-component mixture - nothing at all / an ordinary application / a rare large event (door
+ brake test, coupling, crush-load levelling, brake-pipe recharge) - each spread by a unit-mean
gamma; see :func:`_burst_multipliers`. Because ``E[M] = 1`` exactly, ``brake_nl_per_stop``,
``spring_nl_per_dwell`` and ``spring_nl_per_unit_load`` keep their calibrated values, the air
balance and the duty are untouched, and ``fit_consumption``'s hour-band basis is unchanged (it
asks for the schedule with ``rng=None``, i.e. the *expected* schedule, precisely so that the
fit does not chase one noisy realisation). The auxiliaries get the same treatment on a 30 min
block, because the measured quiescent spread above cannot come from the bursts.

Two mechanisms then do the work, and both are emergent rather than imposed:

* a large event is **not** an impulse: the brake and the air springs hang off one pipe, so the
  draw is choked at :attr:`PneumaticParams.burst_max_nls` and simultaneous demands queue
  (:func:`_choke`). A large event therefore becomes a multi-minute episode of sustained demand,
  which is what cuts ``t_off`` to zero;
* the event that matters **starts the compressor**, so its air is spent inside a loaded window
  and the OFF windows that survive are the quiet ones. Nothing in ``_run_core`` has to know
  about the schedule for that: it falls out of the pressure feedback, and it is what finally
  splits the three phases apart (measured 0.733 / 0.473 / 0.342 NL/s, simulated **0.589 /
  0.504 / 0.362** on a 21-day healthy run; ``scripts/calibrate_pneumatic.py`` reports the row).

Result, on the same 21-day seed-3 healthy run the calibration script uses: ``t_off`` KS
0.480 -> **0.091**, ``dP_dt_off`` 0.620 -> **0.226**, ``t_loaded`` 0.472 -> **0.324**,
``T_oil_max`` 0.458 -> **0.454**, ``I_loaded_mean`` 0.748 -> **0.747**. The simulated ``t_off``
now spreads 0-1,610 s with sd 546 s (real: sd 600 s) and 16 % of cycles at zero (real: 17 %).

Sizing the compressor against the leak (why ``LPS`` can fire at all)
-------------------------------------------------------------------
``LPS`` fires on the real air-leak failures in MetroPT-2/3 [R88][R89], and rail_phm 4.2's own
acceptance target is "air leak detected >= 150 min **before LPS fires**" [R93], which is
unmeasurable if it never fires. MetroPT-3's episodes bound the ``s = 1`` orifice from *below*:
during 29 May - 7 Jun 2020 the unit ran continuously for 7,781 s and the reservoir settled at
**8.38 bar**, so the leak there exactly balanced the net delivery - an equivalent **2.66 mm**
sharp orifice; the 18 Apr episode settled at 8.86 bar (**2.57 mm**). Both were *repaired before
functional failure*, so ``s = 1`` must be larger. We define it as the leak that just beats the
compressor at the 7.0 bar ``LPS`` trip: **2.88 mm**, worst case 8.5 NL/s at 10.2 bar, 3.3x
below the [R156] freight "catastrophic" bound of 60 cfm = 28.3 NL/s (the plan's smaller
endpoint claimed 10x; the margin narrows because the measurement is larger than the guess).

Three regimes follow, and they are the physics, not a tuning artefact:

=================  ==========================================================================
severity           behaviour
=================  ==========================================================================
``s < 0.13``       the compressor still cycles with a measurable OFF phase in service;
                   ``idle_run_ratio`` and ``dP_dt_off`` both move monotonically. **This is
                   the band the detectors work in** and where the lead time of
                   docs/parameters.md pneumatic section 3 is measured.
``0.13 - 0.73``    no OFF phase in service (the unloaded hold now covers the whole coast-down)
                   but the unit still cycles, so ``duty_ratio``, ``t_unloaded`` and
                   ``dP_dt_loaded`` keep moving and the depot cycles stay measurable.
``s > 0.73``       one continuous run: no cycle boundaries, hence **no feature rows**.
                   ``LPS`` asserts, and > 60 s of it in service is the functional failure.
=================  ==========================================================================

A leak that beats the compressor *necessarily* has no duty cycle left to report, which is why
the feature table thins out at the top of the scale. ``PneumaticParams(leak_d1_mm=2.66)``
reproduces the observed 29 May - 7 Jun endpoint exactly if a run wants the measured value
rather than the failure definition; it will not reach ``LPS`` on a lone leak.

Functional failure, the thermal ceiling, and what happens after
--------------------------------------------------------------
Functional failure is ``LPS`` asserted for more than 60 consecutive in-service seconds, or oil
temperature above 95 C (plan, "Pneumatic air supply"). Reaching it does **not** stop the
simulation - a real train keeps running to the next depot - so ``simulate`` keeps emitting.
Two consequences are handled explicitly:

* **The oil node is capped at** :attr:`PneumaticParams.T_oil_max_c` **= 105 C**, a screw
  compressor's high-oil-temperature protection setpoint. The cap is modelled as a *saturation
  of the protection envelope*, not as a simulated trip/restart cycle, deliberately: latching
  the motor off and on at 105 C injects thermal off-time into ``t_off``/``idle_run_ratio``,
  which is exactly the pressure-side feature the leak label depends on, and it makes
  ``idle_run_ratio(severity)`` non-monotone. Saturating the node leaves every pressure-side
  feature untouched and still bounds the temperature at a physical number. With the calibrated
  thermal constants the cap binds on a 30-day leak ramp, so it is load-bearing, not decorative.
* **Post-failure rows are labelled, not silently truncated.** Every feature row after
  ``t_functional_failure`` carries ``rul_s = 0`` and ``is_faulty = True``. Whatever builds the
  training set (``scripts/generate.py``) must drop or down-weight ``rul_s == 0`` rows for the
  RUL task - the simulator cannot decide that for the caller, because the anomaly-detection
  and change-point tasks legitimately want the post-failure regime. ``simulate`` does not
  truncate.

Where ``TP2`` and ``H1`` actually sit (fixed 15 Sep 2026)
---------------------------------------------------------
Until 15 Sep 2026 this module emitted ``H1`` as a tap *downstream of the air/oil filter*, high
while loaded and ~0 otherwise - the mirror image of the real channel. (The earlier note claimed
*both* channels were inverted; the measurement below says only ``H1`` was. ``TP2``'s state shape
- zero off and unloaded, discharge pressure while loaded - was already right, and only its
0.02 bar idle floor moved, to the 0.0 bar the clip enforces.) Measured on MetroPT-3's
failure-free window (1 Feb - 31 Mar 2020, 445,298 rows, states from ``Motor_current`` exactly as
``scripts/calibrate_pneumatic.py`` cuts them - off < 2 A, loaded > 5 A):

=========  ==============  ==============  ==============  =================================
state      median ``TP2``  median ``H1``   median ``TP3``  n samples
=========  ==============  ==============  ==============  =================================
off        **-0.012 bar**  **8.706 bar**   8.718 bar       279,701
unloaded   **-0.012 bar**  **9.662 bar**   9.678 bar       114,672
loaded     **9.294 bar**   **-0.012 bar**  9.162 bar       50,925
=========  ==============  ==============  ==============  =================================

and while loaded ``TP2 - TP3 = +0.304`` bar (p25-p75 0.156-0.396) while ``H1 - TP3 = -9.168``
bar. The same split falls out of the ``COMP`` digital on its own (``COMP = 0`` is "compressor
working": ``TP2`` 9.234 / ``H1`` -0.012; ``COMP = 1``: ``TP2`` -0.012 / ``H1`` 8.958), and the
two segmentations agree on 99.4 % of samples, so the shape is not an artefact of the current
thresholds. The -0.012 bar floor is the transducers' common zero offset - every channel reads
it - and is below both the 0.0113 bar noise sigma and this module's ``(0, 16)`` clip, so the
clean floor here is 0.0.

So the two channels are **not two points of one chain**:

* ``TP2`` is the **compressor discharge** tap, upstream of the separator, the air/oil filter
  and the dryer. It is live only while the unit delivers and is vented with the unloader, which
  is why it reads zero through the whole 40 s blowdown, the unloaded hold and the OFF coast.
* ``H1`` is the **cyclonic-separator discharge** tap - the descriptor's "pressure generated by
  the pressure drop when the cyclonic separator discharges" [R87]. It reads panel pressure
  whenever the separator is not discharging (off and unloaded; measured ``H1 - TP3 = -0.012``
  bar in *both* states) and is dumped to atmosphere while the unit is loaded.

**Consequence for ``clogged_filter``.** The fault used to separate on ``TP2 - H1``, which is
meaningless now: ``H1`` is ~0 exactly when the filter is carrying flow. The filter drop is
instead observable on ``TP2 - TP3`` - discharge minus panel spans the separator, the filter and
the dryer, so a filter clogging to ``dp_filter*(1 + 6 s)`` opens it from ~0.55 to ~1.4 bar -
and ``H1`` becomes the fault's **negative control**: it is measured on the reservoir side of a
closed check valve, so it must not move at all. The pair (``TP2 - TP3`` up, ``H1`` flat) is a
stronger discriminator than the old single difference, because a *reservoir* leak moves ``H1``
(it follows the panel) without opening ``TP2 - TP3``.

Known gaps, stated rather than hidden
-------------------------------------
* **The simulated cycles are still more regular than the real ones, but no longer by much**
  (see "Heterogeneous demand" above): ``t_off`` KS 0.480 -> 0.091 and ``dP_dt_off`` 0.620 ->
  0.226 after the burst schedule was made heavy-tailed. What is left is the *loaded* phase.
  MetroPT-3 puts 13 % of its healthy cycles at ``duty_ratio > 0.5`` and 10 % at
  ``t_loaded <= 30`` s; we produce neither. The first is refused on purpose - the supply line
  is choked at :attr:`PneumaticParams.burst_max_nls` so that a healthy unit's duty saturates
  near 0.38, because at Singapore ambient a sustained duty above ~0.75 walks the sump through
  the 95 C functional-failure line and would label healthy runs as failed (the real unit is at
  Porto, 13 K cooler, and can afford them). The second is arithmetically impossible here: with
  a 2.15 bar band and 6.16 NL/s of net delivery a charge cannot take less than ~107 s, so
  MetroPT-3's 10-30 s loaded runs are either short-cycling the unloader or a 10 s sampling
  artefact, not something a 1 Hz state machine can reproduce. Together they hold
  ``t_loaded``'s KS at 0.324.
* ``TP2_minus_TP3_mean`` is 0.55 bar here against **0.304 bar** measured (see "Where ``TP2``
  and ``H1`` actually sit" above): our discharge drop is the sum of three handbook [R155]
  drops, not a fit. Fitting it is a *constants-table* change, not a signal-shape one, because
  ``I_loaded_a`` is defined as the current regression's intercept referred back through
  exactly that sum (``scripts/calibrate_pneumatic.py`` computes the row that way), so the two
  must move together. Left for the next calibration pass.
* ``Flowmeter`` does not exist in MetroPT-3, so neither its scaling nor its sensor chain is
  calibrated, and the [R93] ``Flowmeter_max > 16.05`` rule cannot be transferred numerically.

Memory note for batch generation: one 30-day APU run at ``store_every=10`` peaks around 2 GB.
Generate trains and subsystems **sequentially**, or stream each run to parquet before starting
the next; fanning several out in parallel will exhaust memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Final, Iterable, Sequence

import numpy as np
import pandas as pd

from nebulax import schema as S
from nebulax.sim.common import (
    SEC_PER_DAY,
    DegradationTrajectory,
    SensorSpec,
    Service,
    SimResult,
    apply_sensor,
    to_timestamp,
)

__all__ = [
    "PNEUMATIC_FAULTS",
    "STATE_NAMES",
    "OFF",
    "LOADING",
    "LOADED",
    "UNLOADING",
    "UNLOADED",
    "PneumaticParams",
    "PneumaticFaults",
    "CYCLE_FEATURE_COLUMNS",
    "leak_nls",
    "orifice_diameter_mm",
    "simulate",
    "selftest",
    "plot_run",
    "main",
]

# ---------------------------------------------------------------------------- constants

#: Compressor state machine. ``UNLOADING`` is the ~40 s sump/separator blowdown and
#: ``LOADING`` the ~3 s repressurisation [R155]; neither is instantaneous.
OFF: Final[int] = 0
LOADING: Final[int] = 1
LOADED: Final[int] = 2
UNLOADING: Final[int] = 3
UNLOADED: Final[int] = 4

STATE_NAMES: Final[tuple[str, ...]] = ("OFF", "LOADING", "LOADED", "UNLOADING", "UNLOADED")

#: Fault types this simulator injects. ``valve_leak_inlet`` / ``valve_leak_outlet`` are in
#: :data:`nebulax.sim.common.FAULT_PRIORS` so ``sample_scenarios("pneumatic", ...)`` emits
#: them; they are separable from a reservoir leak because they shorten ``dP_dt_loaded``
#: without touching the OFF-state decay rate [rail_phm 4.2].
PNEUMATIC_FAULTS: Final[tuple[str, ...]] = (
    "air_leak",
    "oil_leak",
    "compressor_wear",
    "dryer_valve_stuck",
    "clogged_filter",
    "valve_leak_inlet",
    "valve_leak_outlet",
)

#: DOE / Compressed Air Challenge sharp-edged-orifice leak table at 90 psig (6.2 barg),
#: converted to NL/s (1 cfm = 0.4719 NL/s) [R154, rail_phm 2.3b]. Used by the tests to check
#: that :func:`leak_nls` really scales as ``d**2``.
DOE_LEAK_TABLE_NLS: Final[dict[float, float]] = {
    0.40: 0.10,
    0.79: 0.42,
    1.59: 1.65,
    3.18: 6.65,
}

#: Per-compressor-cycle feature columns, in emission order.
CYCLE_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "t_loaded",
    "t_unloaded",
    "t_off",
    "dP_dt_off",
    "dP_dt_loaded",
    "I_loaded_mean",
    "I_start_peak",
    "T_oil_max",
    "TP2_minus_TP3_mean",
    "H1_loaded_mean",
    "tower_switches",
    "purge_count",
    "LPS_any",
    "hour_of_day",
    "load_frac_mean",
    "idle_run_ratio",
    "duty_ratio",
    "Flowmeter_max",
    "P_res_min",
    "in_service_frac",
    "dropout_frac",
)


def _default_sensors() -> dict[str, SensorSpec]:
    """Per-signal measurement chain, CALIBRATED on MetroPT-3.

    Every sigma and quantum below is measured, not assumed. For a signal this slow the second
    difference is almost pure noise and ``var(d2 x) = 6 sigma^2`` for white noise, so the
    high-frequency content is recoverable without knowing the trajectory; sigma is measured
    separately in each compressor state and pooled by state duration, because a
    :class:`~nebulax.sim.common.SensorSpec` carries exactly one additive sigma.

    ======================  =========================  ================  ==============
    channel                 measured sigma (off/unl/ld) pooled            quantum
    ======================  =========================  ================  ==============
    pressures               0.0061 / 0.0129 / 0.0254 b  **0.0113 bar**   **0.002 bar**
    ``Oil_temperature``     0.055 / 0.077 / 0.089 K     **0.065 K**      **0.025 K**
    ``Motor_current``       0.0011 / 0.039 / 0.147 A    **0.049 A**      **0.0025 A**
    ======================  =========================  ================  ==============

    The ``Motor_current`` spread is *not* instrument noise: the floor with the motor off is
    1.1 mA and the 0.147 A while loaded is real motor ripple. One additive sigma cannot be both,
    so the duration-pooled 0.049 A is used - which happens to confirm the first cut's 0.05 A.
    Digitals are clean but can still drop out.
    """
    press = SensorSpec(noise_sigma=0.0113, quantum=0.002, dropout_burst_prob=5e-6, clip=(0.0, 16.0))
    digital = SensorSpec(dropout_burst_prob=1e-6, clip=(0.0, 1.0))
    specs: dict[str, SensorSpec] = {
        "TP2": press,
        "TP3": press,
        "H1": press,
        "DV_pressure": press,
        "Reservoirs": press,
        "Oil_temperature": SensorSpec(noise_sigma=0.065, quantum=0.025, dropout_burst_prob=5e-6, clip=(-20.0, 200.0)),
        "Motor_current": SensorSpec(noise_sigma=0.049, quantum=0.0025, dropout_burst_prob=5e-6, clip=(0.0, 30.0)),
        # Flowmeter does not exist in MetroPT-3 (it is a MetroPT-1/2 channel [rail_phm 2.1]);
        # its chain is left at the first cut's assumption and is flagged as uncalibrated.
        "Flowmeter": SensorSpec(noise_sigma=0.02, quantum=0.01, dropout_burst_prob=5e-6, clip=(0.0, 60.0)),
    }
    for name in ("COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"):
        specs[name] = digital
    return specs


# ---------------------------------------------------------------------------- params


@dataclass(frozen=True, slots=True)
class PneumaticParams:
    """Physical constants of the air production unit. See the module docstring for sources."""

    # --- reservoir and control band [R87, rail_phm 2.3b] -------------------------------
    #: Main-reservoir volume. CALIBRATED on MetroPT-3: inside one compressor cycle the same
    #: demand is subtracted from the charge and added to the coast-down, so the per-cycle sum of
    #: the two endpoint rates cancels it and isolates the compressor -
    #: ``band/t_loaded + band/(t_unloaded + t_off) = Q_comp*(1 - purge_frac)*P_atm/V``
    #: (DOE's receiver relation [R155] read backwards). Measured 0.02149 bar/s over 3,141
    #: healthy cycles. Only ``Q_comp/V`` is identifiable - MetroPT-3 has no flow channel - so
    #: this number moves with :attr:`Q_comp_nls`; see ``scripts/calibrate_pneumatic.py``.
    V_res_l: float = 290.0
    P_atm_bar: float = 1.013
    #: CALIBRATED: the unit actually loads at 8.06 bar (median cycle minimum 8.066, corrected
    #: for half a 10 s sample of OFF decay). [R87] only bounds it - "starts below 8.2".
    P_start_bar: float = 8.06
    #: CONFIRMED against MetroPT-3 to 0.1 %: median cycle maximum 10.124 bar plus half a
    #: sample of charge slope = 10.21 (rail_phm corrects the plan's 10.0).
    P_stop_bar: float = 10.2
    P_lps_bar: float = 7.0  # LPS asserts below this
    P_init_bar: float = 9.2
    #: CALIBRATED: the unloaded run is 416 s per cycle and remarkably tight (p25-p75
    #: 416-426 s), i.e. a fixed timer rather than a pressure threshold; minus the 40 s [R155]
    #: blowdown that leaves 376 s of hold. The first cut's 45 s was invented.
    t_hold_s: float = 376.0
    t_unload_s: float = 40.0  # sump/separator relief on unload [R155]
    t_reload_s: float = 3.0  # repressurise on reload [R155]

    # --- compressor -------------------------------------------------------------------
    #: Free air delivery while loaded. Still a calibration placeholder (calibrate on the
    #: MetroPT-3 charge slope), but no longer free: see "Sizing the compressor against the leak"
    #: in the module docstring. 9.0 NL/s - the ``demo_runs`` first cut - out-supplies the
    #: ``s = 1`` leak at every pressure, so ``LPS`` could never assert on a single air leak.
    Q_comp_nls: float = 7.0
    I_unloaded_a: float = 3.785  # CALIBRATED: median of MetroPT-3's unloaded cluster ([R87]: ~4 A)
    #: CALIBRATED by OLS of ``Motor_current`` on ``Reservoirs`` and ``Oil_temperature`` over
    #: the loaded samples of 3,141 healthy cycles (R2 0.57, residual sd 0.16 A):
    #: ``I = 5.736 + 0.2231*(P - 8.06) - 0.01139*(T - 58)``. The intercept is referred back
    #: through the 0.53 bar discharge drop this module adds. MetroPT-3's loaded current is
    #: **5.92 A median**, not [R87]'s nominal 7 A - the descriptor rounds, the file measures.
    I_loaded_a: float = 5.618
    I_loaded_kp: float = 0.2231  # A per bar of discharge pressure
    #: NEGATIVE: the current falls with oil temperature (thinner oil, less viscous drag). The
    #: first cut's +0.02 A/K had the sign wrong.
    I_loaded_kT: float = -0.0114
    #: CONFIRMED: the largest ``Motor_current`` sample anywhere in the normal window is 9.29 A.
    #: A direct-on-line peak is badly under-sampled at 10 s, so 9.0 A is a floor, not a fit.
    I_start_a: float = 9.0  # direct-on-line starting peak
    t_start_peak_s: float = 2.0
    T_oil_ref_c: float = 58.0  # median MetroPT-3 Oil_temperature; the current regression's centre

    # --- twin-tower dryer [R155][R160, rail_phm 2.3b] ---------------------------------
    #: MEASURED from the ``TOWERS`` channel itself, as rail_phm 2.3b insists: the median
    #: loaded-time between flips is 59 s (p25-p75 50-60 s) over 9,050 flips, and 80 % of flips
    #: happen while the unit is loaded, which is the gating this module already implements.
    tower_period_s: float = 59.0
    purge_s: float = 10.0  # observable blowdown pulse at each tower switch
    purge_frac: float = 0.12  # regeneration purge, measured 12 % of flow [R160]
    purge_blowdown_nl: float = 8.0  # extra volume vented when a tower depressurises
    dp_dryer_bar: float = 0.28  # 3-5 psi across the towers [R155]
    dp_filter_bar: float = 0.15  # air/oil filter, healthy
    dp_sep_bar: float = 0.10  # cyclonic separator
    dryer_stuck_open: bool = True  # dryer_valve_stuck latches the purge valve OPEN

    # --- consumers, from the shared service generator ----------------------------------
    #: Continuous auxiliaries. CALIBRATED from the OFF decay, which is the one estimator that
    #: only ever sees quiescent periods (a burst large enough to matter starts the compressor
    #: and therefore lands in a loaded window by construction): 0.342 NL/s of quiescent draw
    #: over 290 L, of which 0.266 NL/s is the healthy leak. The demo_runs 0.6 NL/s was ~8x too
    #: large and forced a duty of 0.33 against MetroPT-3's 0.11.
    aux_nls: float = 0.076
    depot_aux_frac: float = 0.25  # auxiliaries while stabled
    #: Brake and air-spring draw, CALIBRATED as **one** scale factor - 0.183 against the
    #: demo_runs first cut - least-squares fitted on the 24 measured hour bands with the
    #: auxiliaries already pinned by the OFF decay, so the demo_runs *relative* tuning between
    #: brake and air spring is untouched. A joint fit of both knobs is ill-posed: on our service
    #: the two bases are both essentially "1 while in service". See
    #: ``scripts/calibrate_pneumatic.py::fit_consumption``.
    brake_nl_per_stop: float = 22.0  # MEAN air per brake application, one per run segment
    brake_s: float = 8.0  # ... delivered over the last 8 s of the run at the nominal size
    spring_nl_per_dwell: float = 5.5  # air-spring levelling floor per dwell (mean)
    spring_nl_per_unit_load: float = 66.0  # ... plus this per unit change of load_frac (mean)
    spring_s: float = 10.0  # delivered over the first 10 s of the dwell at the nominal size
    nl_per_impulse: float = 5.0  # Caudal_impulses pulses one sample per this much air

    # --- burst heterogeneity: the per-event size is a random draw, not the mean -----------
    # The five constants below spread the SAME air over the events differently; the mean draw
    # of every event is still exactly the calibrated ``brake_nl_per_stop`` /
    # ``spring_nl_per_dwell`` + ``spring_nl_per_unit_load*d_load`` above, so nothing in the
    # air balance, in ``scripts/calibrate_pneumatic.py::fit_consumption`` or in section 2 of
    # docs/parameters.md moves. See ``_burst_multipliers`` for the mixture and the module
    # docstring ("Heterogeneous demand") for why a fixed per-stop draw was falsified.
    #: Fraction of stops / dwells that actually draw reservoir air. A metro's service brake is
    #: blended: while the ED (regenerative) brake is receptive it does the whole stop and the
    #: friction brake never charges, and an air spring only levels when the load step exceeds
    #: its valve deadband. So most events draw ~nothing and a minority draw a full application.
    #: OURS (a structural assumption); its value is set by the shape of MetroPT-3's ``t_off`` /
    #: ``dP_dt_off`` distributions, not by a source - see docs/parameters.md pneumatic section 4.
    burst_draw_prob: float = 0.06
    #: Gamma shape of a drawing event, i.e. the within-class dispersion (1.0 = exponential,
    #: < 1 heavier, -> inf deterministic). ``sd/mean = 1/sqrt(k)``.
    burst_shape_k: float = 0.8
    #: Rate of a LARGE event per service event: a door + brake test at a terminus, a coupling,
    #: a full levelling cycle on crush load, a brake-pipe recharge. These are the events that
    #: cut the OFF phase short in the real record (13 % of MetroPT-3's healthy cycles never
    #: reach it at all, and 17 % have ``t_off = 0``).
    big_event_prob: float = 0.006
    #: Mean size of a large event, in units of the nominal per-event mean. It is drawn from the
    #: SAME budget as the ordinary events - the mixture is unit-mean by construction - so the
    #: calibrated means are untouched; see ``_burst_multipliers``.
    big_event_mult: float = 120.0
    #: Capacity of the **one** pipe that feeds the brake and the air springs, NL/s. An event
    #: delivers its air over ``brake_s`` / ``spring_s`` unless that would exceed this, and
    #: simultaneous events queue behind it (:func:`_choke`); no air is lost, it is deferred.
    #: That is what turns a large event into a multi-minute episode of sustained demand - the
    #: ``duty_ratio -> 0.5-1`` population MetroPT-3 shows in 13 % of its healthy cycles -
    #: instead of an impulse. It is also a safety bound: with the line capped, the total draw
    #: is at most ``burst_max_nls + aux + leak``, the duty saturates near 0.38 and the sump's
    #: own steady state stays below the 95 C failure line even at Singapore ambient. Without
    #: the cap two overlapping large events held the demand above the compressor's net
    #: delivery long enough to cook a *healthy* 30-day run on 3 of 4 seeds.
    burst_max_nls: float = 2.0
    #: The auxiliaries are intermittent too, and on the same evidence: MetroPT-3's *quiescent*
    #: OFF draw is not one number but spreads 0.205 (p95 of the rate) - 0.351 (median) - 0.97
    #: (p5) NL/s across healthy cycles. Automatic drain and blowdown valves, brake-cylinder
    #: hold on a standing train and the panel's distributed seat leakage come and go, so
    #: ``aux_nls`` is the MEAN of a draw that is zero on this fraction of blocks and
    #: correspondingly larger on the rest. OURS.
    aux_active_frac: float = 0.70
    #: How long one auxiliary regime lasts, seconds - a couple of compressor cycles, which is
    #: what makes the draw constant *within* an OFF window and different *between* them.
    aux_block_s: float = 1800.0
    #: Extra flow leaving the reservoir while the unit is unloaded, beyond leak + auxiliaries.
    #: CALIBRATED: MetroPT-3's unloaded phase falls at -0.00165 bar/s, measurably faster than
    #: leak-plus-auxiliaries predicts at that (higher) pressure - the separator and unloader
    #: keep bleeding after the motor offloads. Without it the simulated reservoir enters the
    #: OFF phase ~0.3 bar too high and ``t_off`` runs long.
    unloaded_vent_nls: float = 0.098

    # --- leak: sharp-edged orifice, Q ~ d^2 and rising with supply pressure [R154] -----
    leak_ref_nls: float = 0.20  # healthy leak at leak_ref_pressure_barg
    leak_ref_pressure_barg: float = 6.2  # the DOE table's 90 psig row
    leak_d0_mm: float = 0.55  # healthy equivalent orifice, s = 0
    #: ``s = 1`` orifice, CALIBRATED on MetroPT-3's own air-leak episodes. During the
    #: 29 May - 7 Jun 2020 episode the unit ran continuously and the reservoir settled at
    #: **8.38 bar**, so the leak there exactly balanced the net delivery at that pressure:
    #: an equivalent **2.66 mm** orifice (18 Apr, settling at 8.86 bar, gives 2.57 mm). Those
    #: episodes were repaired *before* functional failure, so they bound the endpoint from
    #: below. We define ``s = 1`` as the leak that just beats the compressor at the 7.0 bar
    #: ``LPS`` trip - 2.88 mm - which is the smallest endpoint that both collapses ``t_off``
    #: (any value above ~2.6 mm does) and makes ``LPS`` reachable, as it is in the real data.
    #: rail_phm 4.2's "roughly 2.2 mm" was a back-calculation from the plan's own discarded
    #: constant-draw law, not a measurement; this replaces it.
    leak_d1_mm: float = 2.88

    # --- oil thermal node --------------------------------------------------------------
    #: PINNED, not fitted: the MetroPT-3 regression identifies ``hA/C`` and ``P_state/hA``,
    #: never the three separately. 25 kJ/K is ~13 L of compressor oil (1.9 kJ/kg/K, 870 kg/m3)
    #: plus the separator shell.
    C_oil_j_per_k: float = 25_000.0
    #: CALIBRATED. Regressing ``dT/dt = a_state - (hA/C)*T`` on the 434,936 clean,
    #: state-constant 10 s intervals of the normal window gives ``hA/C = 0.001282 /s``
    #: (time constant **780 s**, R2 0.29) and per-state steady temperatures of 52.0 / 62.0 /
    #: 93.3 C for off / unloaded / loaded. With C pinned above that is 32.0 W/K - the first
    #: cut's 12 W/K made the sump 2.7x too sluggish.
    hA_oil_w_per_k: float = 32.0
    #: CALIBRATED: the loaded steady state sits 41.3 K above the OFF asymptote, the unloaded
    #: one 10.0 K, both times ``hA``.
    P_heat_loaded_w: float = 1323.0
    P_heat_unloaded_w: float = 320.0
    #: CALIBRATED: the sink the *fast* oil node relaxes toward is the warm compressor body, not
    #: outdoor air. MetroPT-3's OFF asymptote is 52.0 C while the record's own cold starts (oil
    #: at 18.5 C on 2020-03-07 and 15.4 C on 2020-08-17, both while charging an empty reservoir
    #: from < 2 bar) put the outdoor ambient at 15-19 C. The sump therefore sits ~33.5 K above
    #: whatever :class:`~nebulax.sim.common.Service` reports as ``T_amb``. See the module
    #: docstring for why this is a sink offset rather than a second thermal node.
    T_sump_offset_c: float = 33.5
    #: Starting sump temperature. Set to the Singapore operating point implied by the sink
    #: offset and the calibrated duty (~28 C ambient + 33.5 K offset + ~5 K duty-weighted rise)
    #: so that ``t = 0`` does not begin with a multi-hour warm-up transient.
    T_oil_init_c: float = 66.0

    # --- functional-failure criteria ----------------------------------------------------
    T_oil_fail_c: float = 95.0
    lps_fail_s: float = 60.0
    #: Thermal-protection ceiling: the sump temperature saturates here. A real screw compressor
    #: has a high-oil-temperature cut-out around 105 C, so past functional failure it sits in
    #: protective cycling rather than heating without bound. Without this the oil node
    #: integrates to 118 C on a 30-day leak ramp and to ~190 C on combined faults - a regime no
    #: compressor produces and no RUL model should be trained on. See the module docstring for
    #: why the ceiling is a saturation rather than a simulated trip/restart cycle.
    T_oil_max_c: float = 105.0

    # --- fault gains (plan "Fault map", corrected by rail_phm 4.2) ----------------------
    oil_leak_hA_gain: float = 0.40  # hA *= (1 - gain*s)
    oil_leak_C_gain: float = 0.30  # C_oil *= (1 - gain*s)
    oil_level_trip_s: float = 0.60  # Oil_level flips at this severity
    oil_level_reset_s: float = 0.55  # ... and resets below this (hysteresis, no switch chatter)
    comp_wear_flow_gain: float = 0.40  # Q_comp *= (1 - gain*s)
    comp_wear_current_gain: float = 0.15
    comp_wear_heat_gain: float = 0.20
    filter_dp_gain: float = 6.0  # dp_filter = dp_filter_bar*(1 + gain*s)
    filter_flow_gain: float = 0.04  # delivered flow lost per bar of extra filter drop
    valve_inlet_flow_gain: float = 0.35  # induction loss: flow down, current slightly down
    valve_outlet_flow_gain: float = 0.30  # discharge blow-by: flow down, current/heat up

    sensors: dict[str, SensorSpec] = field(default_factory=_default_sensors)

    def __post_init__(self) -> None:
        if self.V_res_l <= 0.0:
            raise ValueError(f"PneumaticParams: V_res_l must be > 0, got {self.V_res_l}")
        if not self.P_lps_bar < self.P_start_bar < self.P_stop_bar:
            raise ValueError(
                "PneumaticParams: need P_lps_bar < P_start_bar < P_stop_bar, got "
                f"{self.P_lps_bar} / {self.P_start_bar} / {self.P_stop_bar}"
            )
        if self.Q_comp_nls <= 0.0:
            raise ValueError(f"PneumaticParams: Q_comp_nls must be > 0, got {self.Q_comp_nls}")
        if min(self.t_unload_s, self.t_reload_s, self.t_hold_s, self.tower_period_s) <= 0.0:
            raise ValueError("PneumaticParams: t_unload_s/t_reload_s/t_hold_s/tower_period_s must be > 0")
        if not self.T_oil_fail_c < self.T_oil_max_c:
            raise ValueError(
                "PneumaticParams: need T_oil_fail_c < T_oil_max_c, got "
                f"{self.T_oil_fail_c} / {self.T_oil_max_c}"
            )
        if min(self.unloaded_vent_nls, self.T_sump_offset_c) < 0.0:
            raise ValueError(
                "PneumaticParams: unloaded_vent_nls/T_sump_offset_c must be >= 0, got "
                f"{self.unloaded_vent_nls} / {self.T_sump_offset_c}"
            )
        if not 0.0 < self.burst_draw_prob <= 1.0:
            raise ValueError(
                f"PneumaticParams: burst_draw_prob must be in (0, 1], got {self.burst_draw_prob}"
            )
        if not 0.0 <= self.big_event_prob < 1.0:
            raise ValueError(
                f"PneumaticParams: big_event_prob must be in [0, 1), got {self.big_event_prob}"
            )
        if self.burst_draw_prob + self.big_event_prob > 1.0:
            raise ValueError(
                "PneumaticParams: burst_draw_prob + big_event_prob must be <= 1, got "
                f"{self.burst_draw_prob} + {self.big_event_prob}"
            )
        if not 0.0 < self.aux_active_frac <= 1.0:
            raise ValueError(
                f"PneumaticParams: aux_active_frac must be in (0, 1], got {self.aux_active_frac}"
            )
        if self.aux_block_s <= 0.0:
            raise ValueError(f"PneumaticParams: aux_block_s must be > 0, got {self.aux_block_s}")
        if min(self.burst_shape_k, self.burst_max_nls) <= 0.0:
            raise ValueError(
                "PneumaticParams: burst_shape_k/burst_max_nls must be > 0, got "
                f"{self.burst_shape_k} / {self.burst_max_nls}"
            )
        if self.big_event_mult < 0.0 or self.big_event_prob * self.big_event_mult >= 1.0:
            # the ordinary-event mean is (1 - p_big*mult)/p_draw: the large events are drawn
            # from the SAME per-event budget, so they cannot exhaust it.
            raise ValueError(
                "PneumaticParams: need 0 <= big_event_prob*big_event_mult < 1 so the unit-mean "
                f"burst mixture stays positive, got {self.big_event_prob} * {self.big_event_mult}"
            )
        if self.leak_d1_mm < self.leak_d0_mm:
            raise ValueError(
                f"PneumaticParams: leak_d1_mm ({self.leak_d1_mm}) < leak_d0_mm ({self.leak_d0_mm})"
            )
        missing = [s for s in S.SIGNALS["pneumatic"] if s not in self.sensors]
        if missing:
            raise ValueError(f"PneumaticParams: no SensorSpec for signal(s) {missing}")

    def with_(self, **changes: Any) -> "PneumaticParams":
        """Copy with fields replaced (calibration variants, ablations)."""
        return replace(self, **changes)

    @property
    def leak_area_ratio_max(self) -> float:
        """``A(s=1)/A(s=0)`` - 27.4 for the calibrated 0.55 -> 2.88 mm orifice pair."""
        return float((self.leak_d1_mm / self.leak_d0_mm) ** 2)


# ---------------------------------------------------------------------------- faults


@dataclass(frozen=True, slots=True)
class PneumaticFaults:
    """The trajectories injected into one APU. All fields optional; none = healthy run."""

    air_leak: DegradationTrajectory | None = None
    oil_leak: DegradationTrajectory | None = None
    compressor_wear: DegradationTrajectory | None = None
    dryer_valve_stuck: DegradationTrajectory | None = None
    clogged_filter: DegradationTrajectory | None = None
    valve_leak_inlet: DegradationTrajectory | None = None
    valve_leak_outlet: DegradationTrajectory | None = None

    def __post_init__(self) -> None:
        for name in PNEUMATIC_FAULTS:
            traj = getattr(self, name)
            if traj is None:
                continue
            if traj.subsystem != "pneumatic":
                raise ValueError(
                    f"PneumaticFaults.{name}: trajectory subsystem is {traj.subsystem!r}, expected 'pneumatic'"
                )
            if traj.fault_type != name:
                raise ValueError(
                    f"PneumaticFaults.{name}: trajectory fault_type is {traj.fault_type!r}, expected {name!r}"
                )
            if name == "dryer_valve_stuck" and traj.shape != "step":
                raise ValueError(
                    "PneumaticFaults.dryer_valve_stuck: a stuck valve is a step fault, got "
                    f"shape={traj.shape!r}"
                )

    @classmethod
    def from_trajectories(
        cls, trajectories: Iterable[DegradationTrajectory] | "PneumaticFaults" | None
    ) -> "PneumaticFaults":
        """Build from a sequence of trajectories (what ``Scenario.faults`` holds)."""
        if trajectories is None:
            return cls()
        if isinstance(trajectories, PneumaticFaults):
            return trajectories
        kwargs: dict[str, DegradationTrajectory] = {}
        for traj in trajectories:
            if traj.fault_type not in PNEUMATIC_FAULTS:
                raise ValueError(
                    f"PneumaticFaults.from_trajectories: fault_type {traj.fault_type!r} is not simulated; "
                    f"expected one of {list(PNEUMATIC_FAULTS)}"
                )
            if traj.fault_type in kwargs:
                raise ValueError(
                    f"PneumaticFaults.from_trajectories: two {traj.fault_type!r} trajectories on one APU"
                )
            kwargs[traj.fault_type] = traj
        return cls(**kwargs)

    def active(self) -> tuple[DegradationTrajectory, ...]:
        """Every injected trajectory, in :data:`PNEUMATIC_FAULTS` order."""
        return tuple(t for t in (getattr(self, n) for n in PNEUMATIC_FAULTS) if t is not None)

    def __bool__(self) -> bool:
        return bool(self.active())


# ---------------------------------------------------------------------------- leak law


def orifice_diameter_mm(severity: np.ndarray | float, params: PneumaticParams | None = None) -> np.ndarray:
    """Equivalent sharp-edged orifice diameter at ``s``: ``d0*sqrt(1 + (A1/A0 - 1) s)``.

    0.55 mm healthy to :attr:`PneumaticParams.leak_d1_mm` (default 2.40 mm) at ``s = 1``, i.e.
    the leak area grows x19 [R154]. See the module docstring for why the top end is 2.40 rather
    than rail_phm 4.2's "roughly 2.2" mm.
    """
    p = params or PneumaticParams()
    s = np.clip(np.asarray(severity, dtype=np.float64), 0.0, 1.0)
    return p.leak_d0_mm * np.sqrt(1.0 + (p.leak_area_ratio_max - 1.0) * s)


def leak_nls(
    pressure_bar_abs: np.ndarray | float,
    diameter_mm: np.ndarray | float,
    params: PneumaticParams | None = None,
) -> np.ndarray:
    """Sharp-edged-orifice leak flow in NL/s.

    ``Q = Q_ref * (d/d_ref)^2 * (P_abs / P_ref_abs)``. The DOE table gives flow proportional to
    orifice **area** and rising with supply pressure [R154]; the linear pressure term is the
    choked-orifice law, and the flow is choked whenever ``P_abs/P_atm > ~1.9``, which holds over
    the whole 8.2-10.2 bar control band, so no sub-critical branch is needed [rail_phm 4.2].
    The discharge coefficient ``Cd = 0.61`` is folded into ``Q_ref`` because the reference point
    is itself a sharp-edged table entry.
    """
    p = params or PneumaticParams()
    p_ref_abs = p.leak_ref_pressure_barg + p.P_atm_bar
    d = np.asarray(diameter_mm, dtype=np.float64)
    pa = np.asarray(pressure_bar_abs, dtype=np.float64)
    return p.leak_ref_nls * (d / p.leak_d0_mm) ** 2 * np.clip(pa, 0.0, None) / p_ref_abs


# ---------------------------------------------------------------------------- helpers


def _severity_series(
    traj: DegradationTrajectory,
    t: np.ndarray,
    rng: np.random.Generator,
    block_s: float = 900.0,
) -> np.ndarray:
    """Severity sampled at ``t`` with slow (15-minute block) multiplicative jitter.

    The *labels* use the clean monotone severity; only the physics sees the jitter, so the
    ``severity`` column of the feature table stays monotone while the telemetry does not look
    unnaturally smooth.
    """
    s = np.asarray(traj.severity(t), dtype=np.float64)
    if traj.shape == "step" or traj.jitter_sigma <= 0.0:
        return s
    n_blocks = int(t.size // max(block_s, 1.0)) + 2
    jitter = rng.normal(0.0, traj.jitter_sigma, size=n_blocks)
    idx = np.minimum((t / block_s).astype(np.int64), n_blocks - 1)
    return np.clip(s * (1.0 + jitter[idx]), 0.0, 1.0)


def _burst_multipliers(rng: np.random.Generator, n_events: int, p: PneumaticParams) -> np.ndarray:
    """Per-event size multipliers, **unit mean by construction**, heavy-tailed.

    Three components, one draw per service event (one stop per ``run``, one levelling
    opportunity per ``dwell``):

    ==================================  =========================  ========================
    component                           probability                mean multiplier
    ==================================  =========================  ========================
    no draw (ED brake took the stop;    ``1 - p_draw - p_big``     0
    load step inside the valve deadband)
    ordinary application                ``p_draw``                 ``(1 - p_big*b)/p_draw``
    large event (door + brake test,     ``p_big``                  ``b = big_event_mult``
    coupling, crush-load levelling)
    ==================================  =========================  ========================

    so ``E[M] = (1 - p_big*b) + p_big*b = 1`` exactly, for every setting of the four knobs.
    That is the whole point: the *mean* air per stop stays the calibrated
    :attr:`PneumaticParams.brake_nl_per_stop`, the hour-band fit in
    ``scripts/calibrate_pneumatic.py::fit_consumption`` sees an unchanged basis, and only the
    *distribution* changes. Each component is spread by a unit-mean
    ``Gamma(k, 1/k)`` with ``k = burst_shape_k``, so the size within a class has
    ``sd/mean = 1/sqrt(k)``.
    """
    if n_events == 0:
        return np.zeros(0, dtype=np.float64)
    p_draw, p_big = p.burst_draw_prob, p.big_event_prob
    mean_ordinary = (1.0 - p_big * p.big_event_mult) / p_draw
    u = rng.random(n_events)
    scale = np.where(u < p_draw, mean_ordinary, np.where(u < p_draw + p_big, p.big_event_mult, 0.0))
    spread = rng.gamma(p.burst_shape_k, 1.0 / p.burst_shape_k, size=n_events)
    return scale * spread


def _spread_events(
    n: int, dt: float, t0: float, start_s: np.ndarray, air_nl: np.ndarray, nominal_s: np.ndarray, p: PneumaticParams
) -> np.ndarray:
    """Lay ``air_nl`` of demand on the sample grid as rectangular events, vectorised.

    An event delivers its air over ``nominal_s`` unless that would exceed the supply-line
    choke :attr:`PneumaticParams.burst_max_nls`, in which case it delivers at the choke and
    simply takes longer - which is how a large event becomes a multi-minute episode rather
    than an impulse. Events may overlap and may run past the end of the segment that started
    them (a brake test continues while the train stands); both are handled because the rates
    are accumulated on a difference array and integrated once.
    """
    out = np.zeros(n + 1, dtype=np.float64)
    live = air_nl > 0.0
    if not live.any():
        return out[:n]
    air = air_nl[live]
    rate = np.minimum(air / nominal_s[live], p.burst_max_nls)
    dur = air / rate
    lo = np.clip(np.ceil((start_s[live] - t0) / dt).astype(np.int64), 0, n)
    hi = np.clip(np.ceil((start_s[live] + dur - t0) / dt).astype(np.int64), 0, n)
    np.add.at(out, lo, rate)
    np.add.at(out, hi, -rate)
    return np.cumsum(out[:n])


def _choke(burst: np.ndarray, dt: float, p: PneumaticParams) -> np.ndarray:
    """Queue the burst demand behind one supply line of capacity ``burst_max_nls``.

    The brake and the air springs are fed from **one** pipe, so simultaneous demands do not
    add without limit: the excess waits. With ``R`` the requested cumulative volume and ``c``
    the choke, the served cumulative volume is the greedy (work-conserving) solution
    ``S(t) = min_{u <= t} [R(u) + c*(t - u)]``, which is one ``np.minimum.accumulate`` on
    ``R(t) - c*t``. No air is lost - it is only deferred - except for whatever backlog is
    still queued when the run ends, at most one event's worth.

    This is not cosmetic. Without it two overlapping large events can hold the demand above
    the compressor's net delivery for tens of minutes, which drives the sump past the 95 C
    functional-failure line on a *healthy* run (observed on 3 of 4 seeds of a 30-day healthy
    run before the choke was made a line property rather than an event property). With it the
    demand is bounded by ``burst_max_nls + aux + leak``, so the duty saturates near 0.38 and
    the sump's own steady state stays below the failure line even at Singapore ambient (worst
    sump temperature over 30-day healthy runs at seeds 3/11/29/101: 89.9-92.4 C).
    """
    cap = p.burst_max_nls
    if burst.size == 0 or float(burst.max()) <= cap:
        return burst
    lin = cap * dt * np.arange(1, burst.size + 1, dtype=np.float64)
    served_cum = lin + np.minimum.accumulate(np.cumsum(burst) * dt - lin)
    return np.diff(served_cum, prepend=0.0) / dt


def _consumption(
    service: Service,
    t: np.ndarray,
    seg_id: np.ndarray,
    p: PneumaticParams,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Vectorised consumer schedule (NL/s) from the shared service generator.

    Brake: one application per ``run`` segment, of **mean** ``brake_nl_per_stop``, delivered
    over the last ``brake_s`` of the run. Air springs: a levelling floor plus
    ``spring_nl_per_unit_load`` times the change of ``load_frac`` between consecutive segments,
    of that **mean**, over the first ``spring_s`` of every dwell. Auxiliaries: continuous,
    throttled to ``depot_aux_frac`` while stabled.

    With ``rng`` the per-event size is a :func:`_burst_multipliers` draw - zero-inflated,
    gamma-spread, with a rare large-event component - instead of the mean itself; without it
    (``rng=None``) every event gets exactly its mean, which is the deterministic schedule the
    calibration fits its hour bands against (``fit_consumption`` must see the *expected*
    basis, not one noisy realisation of it, or the fitted ``burst_scale`` would wander by
    ~10 % from run to run and the section-2 constants would stop reading CONFIRMED).

    The means are identical in both modes, so the air balance, the duty cycle and every
    calibrated constant are untouched; only the *distribution* over cycles changes. See the
    module docstring, "Heterogeneous demand", for the evidence that a fixed per-event draw is
    falsified by MetroPT-3.
    """
    seg = service.segments
    kind = seg["kind"].to_numpy().astype(str)
    seg_start = seg["t_start"].to_numpy(dtype=np.float64)
    seg_end = seg["t_end"].to_numpy(dtype=np.float64)
    load = seg["load_frac"].to_numpy(dtype=np.float64)
    in_svc = seg["in_service"].to_numpy(dtype=bool)

    d_load = np.abs(np.diff(load, prepend=load[0]))
    spring_nl = p.spring_nl_per_dwell + p.spring_nl_per_unit_load * d_load

    i = seg_id.astype(np.int64)
    aux = np.where(in_svc[i], p.aux_nls, p.aux_nls * p.depot_aux_frac)

    n = t.size
    dt = float(t[1] - t[0]) if n > 1 else 1.0
    t0 = float(t[0])
    is_run_seg = kind == "run"
    is_dwell_seg = kind == "dwell"
    mult = (
        np.ones(kind.size, dtype=np.float64)
        if rng is None
        else _burst_multipliers(rng, int(kind.size), p)
    )

    brake_air = np.where(is_run_seg, p.brake_nl_per_stop * mult, 0.0)
    spring_air = np.where(is_dwell_seg, spring_nl * mult, 0.0)
    nominal_brake = np.full(kind.size, p.brake_s, dtype=np.float64)
    nominal_spring = np.full(kind.size, p.spring_s, dtype=np.float64)
    brake = _spread_events(n, dt, t0, seg_end - p.brake_s, brake_air, nominal_brake, p)
    spring = _spread_events(n, dt, t0, seg_start, spring_air, nominal_spring, p)
    requested = brake + spring
    served = _choke(requested, dt, p)
    if served is not requested:
        # split the served flow back over the two consumers pro rata; while a backlog drains
        # with nothing newly requested the whole of it is booked to the brake, so that
        # ``brake + spring`` keeps summing to the served total exactly.
        share = np.divide(served, requested, out=np.zeros_like(served), where=requested > 0.0)
        spring = spring * share
        brake = served - spring
    if rng is not None:
        # intermittent auxiliaries, constant within a block and unit-mean over blocks
        n_blocks = int(n * dt // max(p.aux_block_s, 1.0)) + 2
        active = rng.random(n_blocks) < p.aux_active_frac
        spread_aux = rng.gamma(p.burst_shape_k, 1.0 / p.burst_shape_k, size=n_blocks)
        j = np.minimum(((t - t0) / p.aux_block_s).astype(np.int64), n_blocks - 1)
        aux = aux * (active * spread_aux / p.aux_active_frac)[j]
    return {"brake": brake, "spring": spring, "aux": aux, "demand": served + aux}


def _run_core(
    n: int,
    demand: np.ndarray,
    T_amb: np.ndarray,
    in_service: np.ndarray,
    sev: dict[str, np.ndarray],
    p: PneumaticParams,
) -> tuple[dict[str, np.ndarray], list[tuple[int, str, dict[str, Any]]], float | None]:
    """The scalar recursion: reservoir pressure, compressor state machine, dryer, oil node.

    Returns the clean (pre-sensor) signal arrays, the event list as ``(index, event, detail)``
    and the functional-failure time in seconds (or ``None``).
    """
    # clean signal buffers
    P_res = np.empty(n, dtype=np.float64)
    TP2 = np.empty(n, dtype=np.float64)
    TP3 = np.empty(n, dtype=np.float64)
    H1 = np.empty(n, dtype=np.float64)
    DVp = np.empty(n, dtype=np.float64)
    T_oil = np.empty(n, dtype=np.float64)
    I_mot = np.empty(n, dtype=np.float64)
    flow = np.empty(n, dtype=np.float64)
    state = np.empty(n, dtype=np.int8)
    comp = np.empty(n, dtype=np.int8)
    dv_el = np.empty(n, dtype=np.int8)
    towers = np.empty(n, dtype=np.int8)
    mpg = np.empty(n, dtype=np.int8)
    lps = np.empty(n, dtype=np.int8)
    p_switch = np.empty(n, dtype=np.int8)
    oil_lvl = np.empty(n, dtype=np.int8)
    caudal = np.empty(n, dtype=np.int8)
    tower_flip = np.zeros(n, dtype=np.int8)
    purge_start = np.zeros(n, dtype=np.int8)

    s_leak = sev["air_leak"]
    s_oil = sev["oil_leak"]
    s_wear = sev["compressor_wear"]
    s_dryer = sev["dryer_valve_stuck"]
    s_filter = sev["clogged_filter"]
    s_vin = sev["valve_leak_inlet"]
    s_vout = sev["valve_leak_outlet"]

    # locals for speed
    P_atm = p.P_atm_bar
    V_res = p.V_res_l
    P_start, P_stop, P_lps = p.P_start_bar, p.P_stop_bar, p.P_lps_bar
    t_unload, t_reload, t_hold = p.t_unload_s, p.t_reload_s, p.t_hold_s
    Q0 = p.Q_comp_nls
    leak_ref = p.leak_ref_nls / (p.leak_ref_pressure_barg + P_atm)
    area_gain = p.leak_area_ratio_max - 1.0
    dp_sep, dp_dry0 = p.dp_sep_bar, p.dp_dryer_bar
    tower_period, purge_s, purge_frac = p.tower_period_s, p.purge_s, p.purge_frac
    C_oil0, hA0 = p.C_oil_j_per_k, p.hA_oil_w_per_k
    vent_unloaded = p.unloaded_vent_nls
    T_sump_offset = p.T_sump_offset_c
    heat_loaded, heat_unloaded = p.P_heat_loaded_w, p.P_heat_unloaded_w
    T_oil_max = p.T_oil_max_c  # thermal-protection ceiling, see PneumaticParams

    P = p.P_init_bar
    st = OFF
    t_state = 0.0
    T = p.T_oil_init_c
    tower = 0
    t_tower = 0.0
    purge_timer = 0.0
    dv_decay = 0.0
    impulse_acc = 0.0
    oil_flag = False
    lps_run_s = 0.0
    t_func: float | None = None
    events: list[tuple[int, str, dict[str, Any]]] = []

    for k in range(n):
        sl, so, sw, sd, sf, svi, svo = (
            s_leak[k], s_oil[k], s_wear[k], s_dryer[k], s_filter[k], s_vin[k], s_vout[k]
        )
        stuck = sd >= 0.5

        # TP2 (compressor discharge) is upstream of the separator, the air/oil filter and the
        # dryer; TP3 (panel) is downstream of all three. H1 is NOT in that chain - it is the
        # separator-discharge tap, alive only while the unit is not delivering (see the
        # observables block below and the module docstring). So a clogged filter opens
        # TP2 - TP3 while loaded and leaves H1 untouched: that pair is the separation channel.
        dp_filter = p.dp_filter_bar * (1.0 + p.filter_dp_gain * sf)
        dp_dryer = dp_dry0
        Q_comp = (
            Q0
            * (1.0 - p.filter_flow_gain * (dp_filter - p.dp_filter_bar))
            * (1.0 - p.comp_wear_flow_gain * sw)
            * (1.0 - p.valve_inlet_flow_gain * svi)
            * (1.0 - p.valve_outlet_flow_gain * svo)
        )

        loaded = st == LOADED or st == LOADING
        if st == LOADING:
            ramp = min(1.0, (t_state + 1.0) / t_reload)
        elif st == LOADED:
            ramp = 1.0
        else:
            ramp = 0.0
        supply = Q_comp * ramp

        # dryer regeneration purge: 12 % of flow while loaded; a stuck-open purge valve
        # keeps venting with the compressor off, which is what makes the fault detectable.
        if stuck and p.dryer_stuck_open:
            purge = purge_frac * Q0
        elif loaded:
            purge = purge_frac * Q_comp
        else:
            purge = 0.0
        if purge_timer > 0.0:
            purge += p.purge_blowdown_nl / purge_s

        P_abs = P + P_atm
        d2 = 1.0 + area_gain * sl
        Q_leak = leak_ref * d2 * P_abs
        # the separator/unloader keeps bleeding for the whole unloaded phase, not only for the
        # 40 s blowdown - measured on MetroPT-3, see PneumaticParams.unloaded_vent_nls
        vent = vent_unloaded if (st == UNLOADING or st == UNLOADED) else 0.0
        Q_out = demand[k] + Q_leak + purge + vent

        P += (supply - Q_out) * P_atm / V_res
        if P < 0.0:
            P = 0.0
        P_abs = P + P_atm

        # ---- state machine -----------------------------------------------------------
        t_state += 1.0
        if st == OFF or st == UNLOADED:
            if P < P_start:
                st, t_state = LOADING, 0.0
                events.append((k, "comp_load", {"P_res": round(P, 3), "mpg": True}))
            elif st == UNLOADED and t_state >= t_hold:
                st, t_state = OFF, 0.0
                events.append((k, "comp_off", {"P_res": round(P, 3)}))
        elif st == LOADING:
            if t_state >= t_reload:
                st, t_state = LOADED, 0.0
        elif st == LOADED:
            if P > P_stop:
                st, t_state = UNLOADING, 0.0
                dv_decay = P
                events.append((k, "comp_unload", {"P_res": round(P, 3)}))
        elif st == UNLOADING:
            if t_state >= t_unload:
                st, t_state = UNLOADED, 0.0

        loaded = st == LOADED or st == LOADING

        # ---- twin-tower dryer --------------------------------------------------------
        if loaded and not stuck:
            t_tower += 1.0
            if t_tower >= tower_period:
                t_tower = 0.0
                tower ^= 1
                purge_timer = purge_s
                tower_flip[k] = 1
                purge_start[k] = 1
                events.append((k, "tower_switch", {"tower": tower}))
                events.append((k, "purge", {"tower": tower, "P_res": round(P, 3)}))
        if purge_timer > 0.0:
            purge_timer -= 1.0

        # ---- currents ----------------------------------------------------------------
        P_disch = P + dp_sep + dp_filter + dp_dryer
        if st == OFF:
            I = 0.0
        elif st == UNLOADING or st == UNLOADED:
            I = p.I_unloaded_a
        else:
            I = p.I_loaded_a + p.I_loaded_kp * (P_disch - P_start) + p.I_loaded_kT * (T - p.T_oil_ref_c)
            if st == LOADING and t_state < p.t_start_peak_s:
                I = max(I, p.I_start_a)
            I *= (1.0 + p.comp_wear_current_gain * sw) * (1.0 + 0.10 * svo) * (1.0 - 0.05 * svi)

        # ---- oil thermal node --------------------------------------------------------
        if st == OFF:
            heat = 0.0
        elif loaded:
            heat = heat_loaded
        else:
            heat = heat_unloaded
        heat *= 1.0 + p.comp_wear_heat_gain * sw + 0.25 * svo + 0.15 * sf
        hA = hA0 * (1.0 - p.oil_leak_hA_gain * so)
        C = C_oil0 * (1.0 - p.oil_leak_C_gain * so)
        T += (heat - hA * (T - T_amb[k] - T_sump_offset)) / C
        if T > T_oil_max:  # high-oil-temperature protection saturates the sump node
            T = T_oil_max

        # ---- observables -------------------------------------------------------------
        P_res[k] = P
        TP3[k] = P - 0.02
        if loaded:
            # MEASURED on MetroPT-3 (1 Feb - 31 Mar 2020, motor-current states): TP2 is the
            # compressor discharge tap and is live ONLY while the unit delivers - median
            # 9.294 bar loaded, -0.012 bar off and unloaded - sitting above the panel by the
            # separator + filter + dryer drop. H1 is the cyclonic-separator discharge tap and
            # does the opposite: median -0.012 bar loaded (99.3 % of loaded samples below
            # 0.1 bar), panel pressure otherwise. The two are NOT two points of one chain.
            TP2[k] = P_disch
            H1[k] = 0.0
        else:
            TP2[k] = 0.0
            # measured H1 - TP3 is -0.012 bar off AND unloaded (p25-p75 -0.014..-0.008), i.e.
            # the transducers' common zero offset, well inside the 0.0113 bar noise sigma:
            # while the separator is not discharging, H1 simply reads the panel.
            H1[k] = P - 0.02
        if st == UNLOADING:
            dv_decay *= 0.90
            DVp[k] = dv_decay
            p_switch[k] = 1
        elif purge_timer > 0.0:
            DVp[k] = 0.85 * P
            p_switch[k] = 1
        else:
            DVp[k] = 0.0
            p_switch[k] = 0
        T_oil[k] = T
        I_mot[k] = I
        flow[k] = demand[k] + Q_leak
        state[k] = st
        comp[k] = 0 if loaded else 1
        dv_el[k] = 1 if loaded else 0
        towers[k] = tower
        mpg[k] = 1 if P < P_start else 0
        lps[k] = 1 if P < P_lps else 0
        if oil_flag:
            oil_flag = so >= p.oil_level_reset_s
        else:
            oil_flag = so >= p.oil_level_trip_s
        oil_lvl[k] = 1 if oil_flag else 0

        impulse_acc += supply
        if impulse_acc >= p.nl_per_impulse:
            impulse_acc -= p.nl_per_impulse
            caudal[k] = 0
        else:
            caudal[k] = 1

        # ---- functional failure ------------------------------------------------------
        if lps[k] and in_service[k]:
            lps_run_s += 1.0
        else:
            lps_run_s = 0.0
        if t_func is None and (lps_run_s > p.lps_fail_s or T > p.T_oil_fail_c):
            t_func = float(k)
            reason = "lps_over_60s_in_service" if lps_run_s > p.lps_fail_s else "oil_temperature_over_95C"
            events.append(
                (k, "functional_failure", {"reason": reason, "P_res": round(P, 3), "T_oil": round(T, 2)})
            )

    clean = {
        "TP2": TP2,
        "TP3": TP3,
        "H1": H1,
        "DV_pressure": DVp,
        "Reservoirs": P_res,
        "Oil_temperature": T_oil,
        "Motor_current": I_mot,
        "COMP": comp.astype(np.float64),
        "DV_eletric": dv_el.astype(np.float64),
        "Towers": towers.astype(np.float64),
        "MPG": mpg.astype(np.float64),
        "LPS": lps.astype(np.float64),
        "Pressure_switch": p_switch.astype(np.float64),
        "Oil_level": oil_lvl.astype(np.float64),
        "Caudal_impulses": caudal.astype(np.float64),
        "Flowmeter": flow,
    }
    extras = {"state": state, "tower_flip": tower_flip, "purge_start": purge_start}
    return {**clean, **extras}, events, t_func


# ---------------------------------------------------------------------------- features


def _nan_mean_by(x: np.ndarray, mask: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """NaN-aware masked mean of ``x`` over the segments starting at ``bounds``."""
    ok = mask & np.isfinite(x)
    num = np.add.reduceat(np.where(ok, x, 0.0), bounds)
    den = np.add.reduceat(ok.astype(np.float64), bounds)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.0, num / np.maximum(den, 1.0), np.nan)


def _cycle_features(
    t: np.ndarray,
    sig: dict[str, np.ndarray],
    state: np.ndarray,
    tower_flip: np.ndarray,
    purge_start: np.ndarray,
    load_frac: np.ndarray,
    in_service: np.ndarray,
    bounds: np.ndarray,
    dt: float,
) -> dict[str, np.ndarray]:
    """Per-compressor-cycle features, computed with ``reduceat`` over the cycle boundaries.

    A cycle runs from one load command to the next, so it contains the loaded charge, the
    40 s unload blowdown, the unloaded hold and the OFF coast. ``bounds`` are the load-start
    indices; the trailing partial cycle is dropped by the caller.

    Two columns depend on the ``TP2`` / ``H1`` placement corrected on 15 Sep 2026, and their
    **semantics are unchanged** - both are still "mean of this channel over the loaded samples
    of the cycle" - but their healthy values moved:

    * ``TP2_minus_TP3_mean`` - compressor discharge minus panel while loaded, i.e. the whole
      separator + filter + dryer drop. Healthy value **~0.55 bar** (unchanged by the fix: both
      channels keep their loaded definition), against **+0.304 bar** measured on MetroPT-3
      (p25-p75 0.156-0.396). Ours is 0.25 bar high because the drop is the sum of the three
      handbook [R155] drops rather than a fit; narrowing it would have to move ``I_loaded_a``
      in the same breath, since that constant is referred back through exactly this sum. It is
      the ``clogged_filter`` separation channel: a clogged air/oil filter multiplies
      ``dp_filter`` by ``1 + 6 s`` and opens this difference to ~1.4 bar.
    * ``H1_loaded_mean`` - the separator-discharge tap while loaded. Healthy value **~0.0 bar**
      (was ~9.6 bar before the fix, which had ``H1`` inverted). MetroPT-3's own median is
      -0.012 bar, so this column is near-degenerate **in the real file too**, and the fix is
      what makes it agree with the identically named column
      :func:`nebulax.adapters.metropt3` extracts. It is kept as a vent-integrity check - a
      loaded ``H1`` that is not ~0 means the separator discharge line failed to vent - and as
      a negative control for ``clogged_filter``, which must not move it.
    """
    loaded = (state == LOADING) | (state == LOADED)
    unloaded = (state == UNLOADING) | (state == UNLOADED)
    off = state == OFF

    n_loaded = np.add.reduceat(loaded.astype(np.float64), bounds)
    n_unloaded = np.add.reduceat(unloaded.astype(np.float64), bounds)
    n_off = np.add.reduceat(off.astype(np.float64), bounds)

    P = sig["Reservoirs"]
    dP = np.diff(P, prepend=P[0])
    I = sig["Motor_current"]
    steady = loaded.copy()
    # drop the starting-current transient from I_loaded_mean: the first 4 samples of a cycle
    steady[np.clip(bounds[:, None] + np.arange(4)[None, :], 0, P.size - 1).ravel()] = False

    feats = {
        "t_loaded": n_loaded * dt,
        "t_unloaded": n_unloaded * dt,
        "t_off": n_off * dt,
        "dP_dt_off": _nan_mean_by(dP, off, bounds) / dt,
        "dP_dt_loaded": _nan_mean_by(dP, loaded, bounds) / dt,
        "I_loaded_mean": _nan_mean_by(I, steady, bounds),
        "I_start_peak": np.fmax.reduceat(I, bounds),
        "T_oil_max": np.fmax.reduceat(sig["Oil_temperature"], bounds),
        "TP2_minus_TP3_mean": _nan_mean_by(sig["TP2"] - sig["TP3"], loaded, bounds),
        "H1_loaded_mean": _nan_mean_by(sig["H1"], loaded, bounds),
        "tower_switches": np.add.reduceat(tower_flip.astype(np.float64), bounds),
        "purge_count": np.add.reduceat(purge_start.astype(np.float64), bounds),
        "LPS_any": np.fmax.reduceat(sig["LPS"], bounds),
        "hour_of_day": (t[bounds] / 3600.0) % 24.0,
        "load_frac_mean": _nan_mean_by(load_frac, np.ones_like(loaded), bounds),
        "Flowmeter_max": np.fmax.reduceat(sig["Flowmeter"], bounds),
        "P_res_min": -np.fmax.reduceat(-P, bounds),
        "in_service_frac": _nan_mean_by(in_service.astype(np.float64), np.ones_like(loaded), bounds),
    }
    total = n_loaded + n_unloaded + n_off
    with np.errstate(invalid="ignore", divide="ignore"):
        feats["idle_run_ratio"] = np.where(n_loaded > 0.0, n_off / np.maximum(n_loaded, 1.0), np.nan)
        feats["duty_ratio"] = np.where(total > 0.0, n_loaded / np.maximum(total, 1.0), np.nan)
    analog = np.vstack([sig[s] for s in ("Reservoirs", "TP2", "Motor_current", "Oil_temperature")])
    nan_frac = np.isnan(analog).mean(axis=0)
    feats["dropout_frac"] = _nan_mean_by(nan_frac, np.ones_like(loaded), bounds)
    return feats


def _labels(faults: PneumaticFaults, t_end_s: np.ndarray) -> dict[str, Any]:
    """Ground-truth labels per cycle, from the clean (un-jittered) severity laws."""
    m = t_end_s.size
    trajs = faults.active()
    if not trajs:
        return {
            "fault_type": pd.Categorical(["healthy"] * m, categories=list(S.FAULT_TYPES["pneumatic"])),
            "severity": np.zeros(m, dtype=np.float32),
            "rul_s": np.full(m, np.nan, dtype=np.float32),
            "is_faulty": np.zeros(m, dtype=bool),
            "alarm_window_3d": np.zeros(m, dtype=bool),
        }
    sev = np.vstack([np.asarray(tr.severity(t_end_s), dtype=np.float64) for tr in trajs])
    which = np.argmax(sev, axis=0)
    best = sev[which, np.arange(m)]
    names = np.asarray([tr.fault_type for tr in trajs], dtype=object)
    ftype = np.where(best > 0.0, names[which], "healthy")
    t_fail = np.asarray(
        [tr.t_functional_failure if tr.t_functional_failure is not None else tr.t_failure for tr in trajs],
        dtype=np.float64,
    )
    t_ref = float(np.min(t_fail))
    rul = np.clip(t_ref - t_end_s, 0.0, None)
    return {
        "fault_type": pd.Categorical(ftype.astype(str), categories=list(S.FAULT_TYPES["pneumatic"])),
        "severity": best.astype(np.float32),
        "rul_s": rul.astype(np.float32),
        "is_faulty": best > 0.0,
        "alarm_window_3d": (t_end_s >= t_ref - 3.0 * SEC_PER_DAY) & (t_end_s <= t_ref),
    }


# ---------------------------------------------------------------------------- simulate


def simulate(
    params: PneumaticParams,
    faults: PneumaticFaults | Sequence[DegradationTrajectory] | None,
    service: Service,
    rng: np.random.Generator,
    store_every: int = 10,
    *,
    run_id: str = "run0",
    source: str = "sim",
    car: int = 0,
    component_id: str = "apu_1",
    dt: float = 1.0,
) -> SimResult:
    """Simulate one APU over ``service`` and return ``(long, features, events)``.

    Implements :class:`nebulax.sim.common.SimulateFn`. The APU is a unit-level component, so
    ``car`` defaults to 0 and ``component_id`` to ``"apu_1"``.

    ``store_every`` thins the 1 Hz telemetry: the full stream is kept for every
    ``store_every``-th compressor cycle plus every cycle in the last 2 days before the failure
    time, while per-cycle features are emitted for **all** cycles. Train-context rows
    (:meth:`Service.context_long`) are thinned with the same mask so the two stay aligned.

    The functional-failure time (``LPS`` > 60 s in service, or oil > 95 C) is written back onto
    every injected trajectory as ``t_functional_failure``; the caller builds the fault log from
    ``Scenario.fault_log_rows`` afterwards.
    """
    if not isinstance(params, PneumaticParams):
        raise TypeError(f"simulate: params must be a PneumaticParams, got {type(params).__name__}")
    if store_every < 1:
        raise ValueError(f"simulate: store_every must be >= 1, got {store_every}")
    if not S.is_valid_component_id("pneumatic", component_id):
        raise ValueError(
            f"simulate: component_id {component_id!r} is not legal for 'pneumatic'; "
            f"expected one of {list(S.COMPONENT_IDS['pneumatic'])}"
        )
    fl = PneumaticFaults.from_trajectories(faults)
    for traj in fl.active():
        if traj.component_id != component_id:
            raise ValueError(
                f"simulate: trajectory {traj.fault_type!r} targets {traj.component_id!r} but the "
                f"simulated component is {component_id!r}"
            )

    tl = service.timeline(dt=dt)
    n = len(tl)
    t = tl.t
    cons = _consumption(service, t, tl.seg_id, params, rng)
    sev = {name: np.zeros(n, dtype=np.float64) for name in PNEUMATIC_FAULTS}
    for traj in fl.active():
        sev[traj.fault_type] = _severity_series(traj, t, rng)

    sig, raw_events, t_func_idx = _run_core(n, cons["demand"], tl.T_amb, tl.in_service, sev, params)
    state = sig.pop("state")
    tower_flip = sig.pop("tower_flip")
    purge_start = sig.pop("purge_start")

    t_func_s = None if t_func_idx is None else float(t_func_idx) * dt
    if t_func_s is not None:
        for traj in fl.active():
            if traj.t_functional_failure is None:
                traj.t_functional_failure = t_func_s

    # measurement chain, per signal
    sensed = {
        name: apply_sensor(sig[name], params.sensors[name], rng, dt=dt).astype(np.float64)
        for name in S.SIGNALS["pneumatic"]
    }

    # ---- cycle boundaries: each load command starts a compressor cycle -----------------
    load_start = np.flatnonzero((state == LOADING) & (np.diff(state, prepend=np.int8(OFF)) != 0))
    if load_start.size >= 2:
        bounds = load_start
        feats = _cycle_features(
            t, sensed, state, tower_flip, purge_start, tl.load_frac, tl.in_service, bounds, dt
        )
        m = bounds.size - 1  # drop the trailing (incomplete) cycle
        feats = {k: np.asarray(v)[:m] for k, v in feats.items()}
        t_start_s = t[bounds[:m]]
        t_end_s = t[bounds[1 : m + 1]]
    else:
        bounds = np.asarray([0], dtype=np.int64)
        m = 0
        feats = {k: np.zeros(0, dtype=np.float64) for k in CYCLE_FEATURE_COLUMNS}
        t_start_s = np.zeros(0)
        t_end_s = np.zeros(0)

    # ---- store_every mask over the sample axis -----------------------------------------
    seg_edges = np.concatenate([[0], load_start, [n]]) if load_start.size else np.asarray([0, n])
    seg_edges = np.unique(seg_edges)
    keep_seg = (np.arange(seg_edges.size - 1) % store_every) == 0
    if t_func_s is not None or fl.active():
        t_ref = t_func_s
        if t_ref is None:
            t_ref = float(min(tr.t_failure for tr in fl.active()))
        seg_t0 = t[np.clip(seg_edges[:-1], 0, n - 1)]
        keep_seg |= (seg_t0 >= t_ref - 2.0 * SEC_PER_DAY) & (seg_t0 <= t_ref)
    mask = np.repeat(keep_seg, np.diff(seg_edges))
    if mask.size != n:  # pragma: no cover - defensive
        raise AssertionError(f"simulate: store mask length {mask.size} != {n}")

    # ---- event rows, extracted before the long build so the clean float64 buffers can go ---
    # Holding `sig` (19 x n float64) and `sensed` (16 x n float64) alive while pandas melts and
    # concatenates the long table is what dominates peak RSS on a 30-day run; both are dropped
    # as soon as the last consumer has read them.
    lps_edge = np.flatnonzero(np.diff(sig["LPS"], prepend=0.0) > 0)
    oil_edge = np.flatnonzero(np.diff(sig["Oil_level"], prepend=0.0) > 0)
    p_at_lps = sig["Reservoirs"][lps_edge].astype(np.float64)
    sev_at_oil = sev["oil_leak"][oil_edge].astype(np.float64)
    del sig, sev, cons, state, tower_flip, purge_start

    # ---- long telemetry ------------------------------------------------------------------
    n_keep = int(mask.sum())
    # pre-typed key columns: melting them as categoricals/int8 instead of python strings keeps
    # the peak memory of a 30-day run in the hundreds of MB rather than the gigabytes.
    ts_keep = tl.timestamp[mask]
    wide = pd.DataFrame(
        {
            "timestamp": ts_keep,
            "source": pd.Categorical.from_codes(np.zeros(n_keep, dtype=np.int8), [source]),
            "run_id": pd.Categorical.from_codes(np.zeros(n_keep, dtype=np.int8), [run_id]),
            "train_id": pd.Categorical.from_codes(np.zeros(n_keep, dtype=np.int8), [service.train_id]),
            "car": np.full(n_keep, car, dtype=np.int8),
            "component_id": pd.Categorical.from_codes(np.zeros(n_keep, dtype=np.int8), [component_id]),
        }
    )
    for name in S.SIGNALS["pneumatic"]:
        wide[name] = sensed[name][mask].astype(np.float32)
    del sensed
    long = S.to_long(wide, "pneumatic")
    del wide["timestamp"]  # ctx_wide keeps its own reference; drop the duplicate below
    # The same rows Service.context_long() emits, but built from the already-masked timeline:
    # materialising the full 30-day context first and throwing 90 % of it away costs GBs.
    ctx_wide = pd.DataFrame(
        {
            "timestamp": ts_keep,
            "source": pd.Categorical.from_codes(np.zeros(n_keep, dtype=np.int8), [source]),
            "run_id": pd.Categorical.from_codes(np.zeros(n_keep, dtype=np.int8), [run_id]),
            "train_id": pd.Categorical.from_codes(
                np.zeros(n_keep, dtype=np.int8), [service.train_id]
            ),
            "car": np.zeros(n_keep, dtype=np.int8),
            "component_id": pd.Categorical.from_codes(
                np.zeros(n_keep, dtype=np.int8), [S.TRAIN_COMPONENT_ID]
            ),
            "speed": tl.speed[mask].astype(np.float32),
            "load_frac": tl.load_frac[mask].astype(np.float32),
            "T_amb": tl.T_amb[mask].astype(np.float32),
            "in_service": tl.in_service[mask].astype(np.float32),
        }
    )
    del wide, ts_keep
    ctx = S.to_long(ctx_wide, "train")
    del ctx_wide
    long = S.coerce_long(pd.concat([long, ctx], ignore_index=True, copy=False))
    del ctx

    # ---- feature table -------------------------------------------------------------------
    frame: dict[str, Any] = {
        "run_id": run_id,
        "source": source,
        "train_id": service.train_id,
        "car": np.full(m, car, dtype=np.int8),
        "subsystem": "pneumatic",
        "component_id": component_id,
        "cycle_id": np.arange(m, dtype=np.int64),
        "t_start": to_timestamp(t_start_s, service.t0),
        "t_end": to_timestamp(t_end_s, service.t0),
    }
    for col in CYCLE_FEATURE_COLUMNS:
        frame[col] = np.asarray(feats[col], dtype=np.float32)
    frame.update(_labels(fl, t_end_s))
    features = S.coerce_features(pd.DataFrame(frame))

    # ---- event log -----------------------------------------------------------------------
    rows = list(raw_events)
    rows += [(int(k), "lps", {"P_res": round(float(v), 3)}) for k, v in zip(lps_edge, p_at_lps)]
    rows += [
        (int(k), "oil_low", {"severity": round(float(v), 3)}) for k, v in zip(oil_edge, sev_at_oil)
    ]
    rows.sort(key=lambda r: r[0])
    events = S.coerce_events(
        pd.DataFrame(
            {
                "run_id": run_id,
                "timestamp": to_timestamp(np.asarray([r[0] for r in rows], dtype=np.float64) * dt, service.t0),
                "train_id": service.train_id,
                "car": np.full(len(rows), car, dtype=np.int8),
                "subsystem": "pneumatic",
                "component_id": component_id,
                "event": pd.Categorical([r[1] for r in rows], categories=list(S.EVENT_TYPES)),
                "detail_json": [json.dumps(r[2]) for r in rows],
            }
        )
    )
    return long, features, events


# ---------------------------------------------------------------------------- selftest


def selftest(
    days: int = 2,
    seed: int = 0,
    *,
    store_every: int = 1,
    fault: str | None = None,
    severity_at_end: float = 1.0,
) -> dict[str, Any]:
    """Run one short simulation and return the sanity numbers the plan asks for.

    ``fault`` injects one trajectory of that type ramping from day 0.4 to ``days`` so a short
    run still spans healthy -> failed.
    """
    from nebulax.sim.common import generate_service

    rng = np.random.default_rng(seed)
    service = generate_service(days, rng, train_id="T01")
    params = PneumaticParams()
    faults = PneumaticFaults()
    if fault is not None:
        shape = "step" if fault == "dryer_valve_stuck" else "power"
        traj = DegradationTrajectory(
            fault_type=fault,
            subsystem="pneumatic",
            component_id="apu_1",
            t_onset=0.4 * days * SEC_PER_DAY,
            t_failure=days * SEC_PER_DAY / max(severity_at_end, 1e-9),
            gamma=2.0,
            shape=shape,
        )
        faults = PneumaticFaults.from_trajectories([traj])
    long, features, events = simulate(
        params, faults, service, rng, store_every=store_every, run_id=f"selftest_{fault or 'healthy'}"
    )
    svc = features["in_service_frac"].to_numpy() > 0.5
    duty = features["duty_ratio"].to_numpy()
    idle = features["idle_run_ratio"].to_numpy()
    return {
        "days": days,
        "fault": fault or "healthy",
        "n_long": int(len(long)),
        "n_features": int(len(features)),
        "n_events": int(len(events)),
        "duty_in_service": float(np.nanmean(duty[svc])) if svc.any() else float("nan"),
        "duty_all": float(np.nanmean(duty)),
        "idle_run_ratio_in_service": float(np.nanmedian(idle[svc])) if svc.any() else float("nan"),
        "t_loaded_mean_s": float(np.nanmean(features["t_loaded"].to_numpy()[svc])) if svc.any() else float("nan"),
        "t_off_mean_s": float(np.nanmean(features["t_off"].to_numpy()[svc])) if svc.any() else float("nan"),
        "I_loaded_mean_A": float(np.nanmean(features["I_loaded_mean"].to_numpy())),
        "T_oil_mean_C": float(np.nanmean(features["T_oil_max"].to_numpy())),
        "T_oil_max_C": float(np.nanmax(features["T_oil_max"].to_numpy())),
        "Flowmeter_max": float(np.nanmax(features["Flowmeter_max"].to_numpy())),
        "P_res_min_bar": float(np.nanmin(features["P_res_min"].to_numpy())),
        "tower_switches_mean": float(np.nanmean(features["tower_switches"].to_numpy())),
        "t_functional_failure_s": (
            faults.active()[0].t_functional_failure if faults.active() else None
        ),
        "events": {str(k): int(v) for k, v in events["event"].value_counts().items() if v},
    }


def plot_run(out_dir: str = "results/sim_checks", days: int = 2, seed: int = 0) -> list[str]:
    """Write the pitch overlays: healthy control band + oil node, and healthy vs air leak."""
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nebulax.sim.common import generate_service

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    p = PneumaticParams()

    runs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for label, fault in (("healthy", None), ("air_leak", "air_leak")):
        rng = np.random.default_rng(seed)
        service = generate_service(days, rng, train_id="T01")
        faults = PneumaticFaults()
        if fault:
            faults = PneumaticFaults.from_trajectories(
                [
                    DegradationTrajectory(
                        fault_type=fault,
                        subsystem="pneumatic",
                        component_id="apu_1",
                        t_onset=0.3 * days * SEC_PER_DAY,
                        t_failure=days * SEC_PER_DAY,
                        gamma=2.0,
                    )
                ]
            )
        long, feats, _ = simulate(p, faults, service, rng, store_every=1, run_id=f"plot_{label}")
        wide = S.to_wide(long, "pneumatic")
        runs[label] = (wide, feats)

    wide, feats = runs["healthy"]
    seg = wide.iloc[9 * 3600 : 9 * 3600 + 3600]
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    ts = seg["timestamp"]
    ax[0].plot(ts, seg["Reservoirs"], lw=1.0, label="Reservoirs")
    # TP2 is the discharge tap: it is vented to ~0 bar whenever the unit is not delivering, so
    # plotting it raw would drag the y-axis off the 8-10.2 bar control band this panel is about.
    ax[0].plot(ts, seg["TP2"].where(seg["TP2"] > 1.0), lw=0.7, alpha=0.7, label="TP2 (delivering)")
    for y, lab in ((p.P_start_bar, "start 8.2"), (p.P_stop_bar, "stop 10.2"), (p.P_lps_bar, "LPS 7.0")):
        ax[0].axhline(y, ls="--", lw=0.7, color="0.5")
        ax[0].annotate(lab, (ts.iloc[5], y), fontsize=7, color="0.4")
    ax[0].set_ylabel("bar")
    ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_title("APU, one healthy hour in service: control band, currents, oil node")
    ax[1].plot(ts, seg["Motor_current"], lw=0.8, color="tab:orange")
    ax[1].set_ylabel("Motor_current [A]")
    ax[2].plot(ts, seg["Oil_temperature"], lw=0.9, color="tab:red")
    ax[2].set_ylabel("Oil_temperature [C]")
    ax[2].set_xlabel("time")
    fig.tight_layout()
    f1 = out / "pneumatic_healthy_hour.png"
    fig.savefig(f1, dpi=120)
    plt.close(fig)
    written.append(str(f1))

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for label, color in (("healthy", "tab:green"), ("air_leak", "tab:red")):
        _, f = runs[label]
        svc = f["in_service_frac"].to_numpy() > 0.5
        ax[0].plot(f["t_end"][svc], f["idle_run_ratio"][svc], ".", ms=3, color=color, label=label)
        ax[1].plot(f["t_end"][svc], f["t_off"][svc], ".", ms=3, color=color, label=label)
    for y, lab in ((9.0, "DOE <10 % leakage"), (4.0, "DOE 20 %"), (2.33, "DOE 30 %")):
        ax[0].axhline(y, ls="--", lw=0.7, color="0.5")
        ax[0].annotate(lab, (ax[0].get_xlim()[0], y), fontsize=7, color="0.4")
    ax[0].set_ylabel("idle_run_ratio = t_off / t_loaded")
    ax[0].set_title("Air leak shrinks the idle time: the [R101] duty-cycle signature")
    ax[0].legend(fontsize=8)
    ax[1].set_ylabel("t_off [s]")
    ax[1].set_xlabel("time")
    fig.tight_layout()
    f2 = out / "pneumatic_leak_idle_run_ratio.png"
    fig.savefig(f2, dpi=120)
    plt.close(fig)
    written.append(str(f2))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m nebulax.sim.pneumatic [--days 2] [--fault air_leak] [--plots]``."""
    import argparse

    ap = argparse.ArgumentParser(description="Pneumatic APU simulator selftest.")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store-every", type=int, default=1)
    ap.add_argument("--fault", choices=("healthy", *PNEUMATIC_FAULTS), default=None)
    ap.add_argument("--plots", action="store_true", help="write results/sim_checks/*.png")
    ap.add_argument("--out", default="results/sim_checks")
    args = ap.parse_args(argv)

    fault = None if args.fault in (None, "healthy") else args.fault
    report = selftest(days=args.days, seed=args.seed, store_every=args.store_every, fault=fault)
    print(json.dumps(report, indent=2, default=str))
    if args.plots:
        for path in plot_run(out_dir=args.out, days=args.days, seed=args.seed):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
