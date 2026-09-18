"""Domain glossary handed to the advisory model as grounding.

Six to nine lines per subsystem. Provenance for every number:

* ``readingmaterials/Rolling Stock & Depot Equipment .pdf`` (LTA "E&M Guide - Module 9",
  Guide for New Rail Electrical & Mechanical Engineers): section 3.1.10 *Doors* gives the
  2 cm obstruction rule and the 5 cm-per-side push-back; figure 27 (TCMS) names the door
  controller ``EDCU``; section 3.1.6 *Pneumatic System* describes the compressor /
  reservoir / intake-filter / oil-separator / air-dryer chain and what reservoir air is
  used for.
* ``docs/research/rail_phm.md`` (this repo's cited research pass) for the numeric
  constants: section 0 for the axle-box thermal baseline in Singapore ambient, section
  2.3b for the APU control band and the twin-tower dryer constants, section 3.1 and 4.3
  for the bearing sensing modalities and the measured healthy inter-box spread, section 5
  for the door fault mix and the obstruction operating point.

The text is deliberately frozen: it is the cached prefix of every advisory request, so any
edit here invalidates the prompt cache for the whole fleet.
"""

from __future__ import annotations

from typing import Final

__all__ = ["GLOSSARY", "glossary_for", "full_glossary"]

_DOOR: Final[str] = """\
- A metro sliding door leaf is driven by an electric motor under an EDCU (Electric Door Control
  Unit); the EDCU reports door state to the train control and management system (TCMS).
- Obstruction detection: an object thicker than about 2 cm stops the doors closing. Thinner
  objects are handled by a push-back system that lets a passenger push the leaves back about
  5 cm per side to retrieve the object. Detection latency is judged against 0.3 s.
- closing_time / opening_time are the leaf travel times of one cycle. Friction (dry guides,
  worn belt, a dragging seal) lengthens them and raises i_mean_cruise, i_rms_cruise, pwm_mean
  and energy_J together, because the drive has to work harder for the same travel.
- i_start_peak is the inrush at the start of travel. Backlash or a loose transmission shows as
  a high start peak with a larger position-following error (pos_err_max, pos_err_rms) while the
  cruise current stays near normal.
- reversal_count and obs_detect_s count re-openings and the time taken to detect an obstruction;
  repeated reversals at one leaf mean debris in the track or a misaligned leaf, not motor wear.
- dropout_frac is the fraction of the cycle with missing controller telemetry. A rising dropout
  fraction is a DCU / wiring / connector fault, not a mechanical one.
- ls_timeout flags a limit switch that did not make inside its time window - the classic door
  fault that holds a train at a platform. Limit-switch, DCU, connector and push-button faults
  together are the single largest reported door failure class (~45%); track debris is ~15%.
- T_motor is the motor-body temperature; a slow rise with elevated i_peak and i_end points at
  brush or commutator wear rather than at the mechanism."""

_PNEUMATIC: Final[str] = """\
- The Air Production Unit (APU) is the compressor plus main reservoirs plus an air control
  package: air passes an intake filter before the compressor and an oil separator on the way
  out, and the main unit of the control package is the twin-tower air dryer, which removes oil
  and moisture from the air.
- Reservoir air feeds brake operation, air-suspension (air-bag) levelling, door operation on
  older fleets, wheel-slide correction, horn and coupling controls, so genuine consumption is
  never zero on a train in service.
- Control band: the compressor starts below 8.2 bar and stops above 10.2 bar; the low-pressure
  switch LPS trips below 7 bar. Motor current is three-state: about 0 A off, 4 A offloaded,
  7 A under load.
- t_off is how long the compressor stays off between cycles. An air leak shortens t_off and
  raises duty_ratio and idle_run_ratio days to weeks before LPS fires; the operating target is
  at least 150 minutes of warning before the LPS trip.
- TP2 is compressor-outlet pressure and TP3 the downstream pressure. A rising TP2_minus_TP3 is
  pressure drop across the dryer and filter; a healthy twin-tower dryer costs only 3-5 psi.
- Twin-tower regeneration runs on a timed cycle, visible in the Towers / tower_switches and
  purge_count channels. Purge air is 10-18% of the dryer's rating (12% measured on heater-purge
  twin towers); a stuck dryer valve shows as an abnormal switching or purge pattern.
- The failure chain couples: low supply pressure raises volumetric flow, which cuts purge flow,
  which leaves the bed incompletely regenerated and the dew point degraded. A reservoir leak
  therefore drags the dryer down with it as a secondary effect.
- Oil_temperature / T_oil_max rising, especially with Oil_level dropping, is an oil leak or a
  cooling problem; the operating target is at least 2 days of warning.
- A single anomalous channel is weak evidence: the field rule is to raise a fault only when at
  least 3 of {TP2, TP3, Motor_current, Oil_temperature, idle_run_ratio} move together."""

_BEARING: Final[str] = """\
- The axle box carries the wheelset bearing in the bogie. Onboard axle-box temperature is the
  cheap continuous signal, onboard vibration sees damage far earlier than temperature, and the
  wayside HABD (hot axle box detector, infrared) is trackside and a late-stage indicator only.
- In Singapore's ambient a healthy box running 25 K above ambient sits near 53 C at the 28 C
  annual mean and near 62 C on a 37 C day. Against an 80 C absolute alarm line that leaves only
  18-27 K of headroom, so an absolute temperature threshold is a weak detector here.
- Because of that, the peer-relative feature is the primary detector: dT_peer_same_side is this
  box minus the median of its same-side peers on the same train, and dT_opposite is the
  left/right difference on the same axle.
- thermal_residual is the measured box temperature minus a lumped thermal model driven by speed,
  load and ambient; it removes duty cycle and weather, so a residual that keeps climbing while
  the fleet is flat is real degradation.
- Nominally identical healthy boxes are not identical: measured axle-box temperatures across 8
  cars of one train at one matched condition spanned 8.4 K (44.6-53.1 C). A few K on one box is
  not evidence on its own.
- vib_bpfo_mean is the outer-race ball-pass envelope band; vib_kurt_mean and vib_crest_mean are
  impulsiveness measures that rise with localised spalling before the RMS level does.
- hot_axle_box is end-of-life thermal runaway and a safety stop; bearing_degradation is the slow
  rise that should be caught weeks earlier. The distinction is the rate, not the level.
- HABD readings are noisy and biased - the infrared scan location on the bearing cup dominates
  the reading, and many flagged bearings turn out to have no discernible defect - so a wayside
  alarm is corroboration, never ground truth."""

#: Per-subsystem glossary text. ``train`` is the context pseudo-subsystem and has no advisory.
GLOSSARY: Final[dict[str, str]] = {
    "door": _DOOR,
    "pneumatic": _PNEUMATIC,
    "bearing": _BEARING,
}


def glossary_for(subsystem: str) -> str:
    """Glossary lines for one subsystem; empty string for an unknown subsystem."""
    return GLOSSARY.get(str(subsystem), "")


def full_glossary() -> str:
    """Every subsystem's glossary, in a fixed order, for the cached system prompt."""
    return "\n\n".join(
        f"{name.upper()}\n{GLOSSARY[name]}" for name in ("door", "pneumatic", "bearing")
    )
