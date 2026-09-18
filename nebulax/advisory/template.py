"""Deterministic fallback advisory ("template-v1").

This is what the demo shows when there is no API key, when the network is down and when
Claude times out, so it has to be good enough to stand on its own: it names a fault from
the top signals, sets urgency from the score/threshold ratio and the episode duration, and
fills all four CBM steps.

Resolving a signal name
-----------------------
``scores.top_signals_json`` does **not** carry bare physical channel names. What it carries
is whatever column the winner was fitted on:

* ``window_stats`` models (pneumatic, MetroPT-3) emit ``<channel>_<stat>`` for every stat in
  :data:`nebulax.features.stats.STAT_NAMES` - ``TP2_mean``, ``DV_pressure_std``,
  ``Motor_current_slope``, ...;
* ``cycle_features`` models with ``peer_norm=True`` (door, bearing) additionally emit
  ``<feature>_peer_delta`` and ``<feature>_peer_z`` from
  :func:`nebulax.features.cycles.peer_normalise` - on the sim bearing runs 62 % of the
  strongest signals carry one of those two suffixes;
* several cycle features *end* in something that looks like a stat (``pos_err_rms``,
  ``T_box_std``, ``vib_rms_mean``) but are single columns in their own right.

So a rule cannot be keyed on one spelling. :func:`signal_aliases` turns a column name into
the chain ``raw -> peer-stripped -> stat-stripped`` (most specific first) and a rule matches
on the *first* alias it knows; that keeps ``T_box_std`` (sensor_stuck) distinct from
``T_box_max`` (a hot box) while still letting a ``TP2`` rule catch ``TP2_rms_peer_z``.
A peer delta / peer z keeps the sign semantics of the underlying feature (this box is
hotter than its same-side peers = "up"), so no sign flipping is needed anywhere.

Every rule key below is a name that actually appears in ``data/scores/*.parquet`` -
``tests/test_advisory.py`` pins the observed vocabularies and asserts it.

Scoring
-------
Rules score by **evidence combination**, not by "strongest single signal wins": each rule is
a weighted bag of signals, a matched signal contributes ``rank_weight * rule_weight`` where
``rank_weight = 0.8 ** rank`` in |z| order, a rule can only collect **one** contribution per
physical channel (so ``TP2_rms``/``TP2_std``/``TP2_mean`` do not count three times), and the
highest total wins. That is what separates friction (cruise current / energy / travel time
up, position error may follow) from backlash, which now *requires* the start-current peak.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Iterable, Mapping, Sequence

from nebulax.advisory.schema import UNKNOWN_FAULT, AlertContext, Advisory, TopSignal, coerce_fault
from nebulax.features.stats import AC_STAT_NAMES, STAT_NAMES

__all__ = [
    "TEMPLATE_MODEL",
    "RULES",
    "Rule",
    "STAT_SUFFIXES",
    "PEER_SUFFIXES",
    "signal_aliases",
    "rule_keys",
    "classify",
    "template_advisory",
]

TEMPLATE_MODEL: Final[str] = "template-v1"

#: Suffixes appended by :func:`nebulax.features.stats.window_stats` (its ``ac_couple``
#: variants included). Longest first so ``rms_ac`` is stripped before ``rms``.
STAT_SUFFIXES: Final[tuple[str, ...]] = tuple(
    sorted(set(STAT_NAMES) | set(AC_STAT_NAMES), key=len, reverse=True)
)
#: Suffixes appended by :func:`nebulax.features.cycles.peer_normalise` (its defaults).
PEER_SUFFIXES: Final[tuple[str, ...]] = ("_peer_delta", "_peer_z")


def signal_aliases(name: str) -> tuple[str, ...]:
    """``name`` and its progressively stripped forms, most specific first.

    ``"T_box_max_peer_z" -> ("T_box_max_peer_z", "T_box_max", "T_box")``,
    ``"TP2_rms" -> ("TP2_rms", "TP2")``, ``"opening_time" -> ("opening_time",)``.
    Duplicates are dropped, so the chain is 1-3 entries long.
    """
    out = [name]
    stem = name
    for suffix in PEER_SUFFIXES:
        if stem.endswith(suffix) and len(stem) > len(suffix):
            stem = stem[: -len(suffix)]
            out.append(stem)
            break
    for stat in STAT_SUFFIXES:
        tail = "_" + stat
        if stem.endswith(tail) and len(stem) > len(tail):
            out.append(stem[: -len(tail)])
            break
    seen: dict[str, None] = {}
    for alias in out:
        seen.setdefault(alias, None)
    return tuple(seen)


def _base(name: str) -> str:
    """The most-stripped alias: the physical channel a column belongs to."""
    return signal_aliases(name)[-1]


@dataclass(frozen=True)
class Rule:
    """One fault and the evidence that speaks for it.

    ``signals`` maps a signal key (any alias, see :func:`signal_aliases`) to
    ``(direction, weight)``. ``direction`` is ``"up"``, ``"down"`` or ``"any"`` and refers to
    the sign of the robust z-score against the fitted healthy distribution (for a peer delta
    / peer z, against the component's peers). ``requires`` names keys that must be present
    *and* point the right way before the rule may score at all - that is how backlash is kept
    from swallowing every door alert with a large position error.
    """

    fault: str
    signals: Mapping[str, tuple[str, float]]
    requires: frozenset[str] = field(default_factory=frozenset)


# --------------------------------------------------------------------------------------
# Door. Cycle features (+ peer deltas against the sibling leaf at the same dwell).
# The cycle table has no obstruction / limit-switch column - those two faults are reached
# from the event log instead, see _EVENT_RULES.
# --------------------------------------------------------------------------------------
_DOOR_RULES: Final[tuple[Rule, ...]] = (
    Rule("dcu_dropout", {"dropout_frac": ("up", 1.0)}),
    Rule("shock_wear", {"shock_jump_count_est": ("up", 1.0)}),
    # Backlash is a *transmission* fault: the leaf breaks away late, so the start-current
    # peak grows while the cruise current stays where it was. Without that peak the
    # evidence is friction, not backlash.
    Rule(
        "backlash",
        {"i_start_peak": ("up", 1.0), "pos_err": ("up", 0.4)},
        requires=frozenset({"i_start_peak"}),
    ),
    # Friction: more work for the same travel - cruise current, energy and travel time up,
    # and the controller falls behind its reference, so pos_err follows.
    Rule(
        "friction",
        {
            "closing_time": ("up", 1.0),
            "opening_time": ("up", 1.0),
            "i_mean_cruise": ("up", 1.0),
            "i_rms_cruise": ("up", 1.0),
            "energy_J": ("up", 1.0),
            "pwm": ("up", 0.8),
            "pos_err": ("up", 0.6),
            "T_motor": ("up", 0.4),
        },
    ),
    Rule("brush_wear", {"i_peak": ("up", 1.0), "i_end": ("up", 0.8), "T_motor": ("up", 0.6)}),
    Rule("misalignment", {"pos_open": ("any", 0.8), "pos_close": ("any", 0.8)}),
)

# --------------------------------------------------------------------------------------
# Pneumatic. Same rule set covers the sim APU (window stats over the simulated channels)
# and MetroPT-3 (window stats over the seven analog channels the winner reads).
# --------------------------------------------------------------------------------------
_PNEUMATIC_RULES: Final[tuple[Rule, ...]] = (
    # Oil circuit: a leak drops both the oil thermal mass and the cooling coefficient
    # (nebulax.sim.pneumatic: hA *= 1 - 0.40 s, C *= 1 - 0.30 s), so the sump runs hot.
    # Oil temperature is the only channel that sees the oil at all.
    Rule("oil_leak", {"Oil_temperature": ("up", 1.2), "Motor_current": ("up", 0.3)}),
    # A clogged air/oil filter opens TP2 - TP3 while loaded and leaves H1 alone. TP2 alone
    # is not enough (it moves whenever the duty cycle moves), so TP3 is required.
    Rule(
        "clogged_filter",
        {"TP2": ("up", 1.0), "TP3": ("down", 0.9)},
        requires=frozenset({"TP3"}),
    ),
    # A stuck changeover valve is identified by the tower switching / purge pattern.
    # DV_pressure on its own only says the unit is cycling.
    Rule(
        "dryer_valve_stuck",
        {"Towers": ("any", 1.2), "DV_pressure": ("any", 0.8)},
        requires=frozenset({"Towers"}),
    ),
    # Air leak: the reservoir bleeds down, so the unit loads more often and for longer.
    # TP2 is live only while delivering and DV_pressure pulses on every load/unload, so any
    # shift in their window statistics means the duty cycle moved - the primary symptom.
    # H1 reads zero while loaded, so more loaded time pushes its mean down.
    Rule(
        "air_leak",
        {
            "Motor_current": ("up", 1.0),
            "Reservoirs": ("down", 1.0),
            "TP3": ("down", 1.0),
            "DV_pressure": ("any", 0.9),
            "TP2": ("any", 0.8),
            "H1": ("down", 0.7),
            "Oil_temperature": ("up", 0.4),
            "Towers": ("up", 0.4),
            "Flowmeter": ("up", 0.3),
        },
    ),
    # Wear costs current and free-air delivery at the same time.
    Rule(
        "compressor_wear",
        {
            "Motor_current": ("up", 1.0),
            "TP2": ("down", 0.8),
            "Flowmeter": ("down", 0.6),
            "Oil_temperature": ("up", 0.5),
        },
    ),
    Rule("valve_leak_outlet", {"Flowmeter": ("up", 1.0), "Motor_current": ("up", 0.3)}),
    Rule(
        "valve_leak_inlet",
        {"Flowmeter": ("down", 0.9), "TP2": ("down", 0.5), "Reservoirs": ("down", 0.4)},
    ),
)

# --------------------------------------------------------------------------------------
# Bearing. Cycle features (+ peer deltas / peer z against the same-side boxes).
# The thermal pair is split into hot_axle_box (fast) and bearing_degradation (slow) by
# :func:`_thermal_fault`, never by level alone.
# --------------------------------------------------------------------------------------
_BEARING_RULES: Final[tuple[Rule, ...]] = (
    Rule("sensor_stuck", {"T_box_std": ("down", 1.2)}),
    Rule("outer_race", {"vib_bpfo": ("up", 1.0), "vib_rms": ("up", 0.4)}),
    Rule("inner_race", {"vib_kurt": ("up", 1.0), "vib_crest": ("up", 0.9), "vib_rms": ("up", 0.3)}),
    Rule(
        "bearing_degradation",
        {
            "thermal_residual": ("up", 1.0),
            "dT_peer_same_side": ("up", 1.0),
            "dT_opposite": ("up", 0.8),
            "vib_rms": ("up", 0.8),
            "T_box_max": ("up", 0.7),
            "T_box_mean": ("up", 0.7),
            "vib_crest": ("up", 0.3),
        },
    ),
    # A box reading colder than its own history *and* than every same-side peer is a
    # calibration problem, not a bearing problem.
    Rule(
        "sensor_offset",
        {
            "T_box_min": ("up", 0.8),
            "T_box": ("down", 0.7),
            "peer_median_same_side": ("down", 0.6),
            "dT_peer_same_side": ("down", 0.5),
        },
    ),
)

RULES: Final[dict[str, tuple[Rule, ...]]] = {
    "door": _DOOR_RULES,
    "pneumatic": _PNEUMATIC_RULES,
    "bearing": _BEARING_RULES,
}

#: Event-log evidence. The door cycle table carries no obstruction / limit-switch column,
#: but ``events.parquet`` records both directly; ``AlertContext.recent_events`` is rendered
#: as ``"<ts> <event> <detail>"`` by the API, so a substring test is enough. Only these two
#: unambiguous observations are used - a bare ``reversal`` has too many innocent causes.
_EVENT_RULES: Final[dict[str, tuple[tuple[str, str, float], ...]]] = {
    "door": (
        ("obstruction", "obstruction", 1.0),
        ("limit_switch", "ls_timeout", 1.0),
    ),
}

#: Channels whose rate decides hot_axle_box vs bearing_degradation.
_THERMAL_BASES: Final[frozenset[str]] = frozenset(
    {"dT_peer_same_side", "thermal_residual", "dT_opposite", "T_box"}
)

_ACTIONS: Final[dict[str, str]] = {
    "friction": "Book the leaf for a guide/belt inspection: clean and re-lubricate the runners, "
    "check the belt tension and the seal drag, then re-run the door self-test.",
    "brush_wear": "Inspect the door motor brushes and commutator at the next depot visit; "
    "replace the brush set if the wear limit is reached.",
    "backlash": "Check the transmission for backlash: belt tension, pulley grub screws and the "
    "rack/pinion or spindle nut play, and re-datum the leaf.",
    "misalignment": "Re-align the leaf and check the hanger/roller height; verify the closed "
    "position against the datum before returning to service.",
    "obstruction": "Inspect the door track for debris and check the obstruction-detection "
    "calibration (2 cm object, 5 cm per side push-back).",
    "limit_switch": "Check the limit switch, its actuator and the loom connector; confirm the "
    "switch makes inside its time window over ten cycles.",
    "dcu_dropout": "Treat as a controller/communications fault: check the EDCU connector, the "
    "loom and the TCMS link for this leaf before touching the mechanism.",
    "shock_wear": "Inspect the leaf and hanger for impact damage; review the shock events in the "
    "event log against station and time of day.",
    "nff": "No fault found: log the episode, keep the unit under watch and re-assess at the next "
    "scored window.",
    "air_leak": "Leak-test the reservoir, pipework and flexible hoses on this unit. The operating "
    "target is to fix it before LPS trips (7 bar); the compressor is already cycling harder.",
    "oil_leak": "Check the compressor oil level and the oil separator for leakage, and check the "
    "cooling path; target is at least two days of warning before removal.",
    "compressor_wear": "Schedule a compressor health check: motor current under load, free-air "
    "delivery and the intake filter.",
    "dryer_valve_stuck": "Inspect the twin-tower dryer changeover valve and the purge path; verify "
    "the tower switching period and purge count against a healthy unit.",
    "clogged_filter": "Replace the intake filter and check the dryer pressure drop; a healthy "
    "twin-tower dryer costs only 3-5 psi.",
    "valve_leak_inlet": "Check the inlet/unloader valve and the check valve on the compressor "
    "outlet: the unit charges more slowly than the fleet for the same load.",
    "valve_leak_outlet": "Check the outlet-side valves and the downstream consumers; measured flow "
    "is high for the duty being served.",
    "bearing_degradation": "Raise a planned bogie inspection for this axle box: grease condition, "
    "endplay and an envelope-spectrum check at the next depot slot.",
    "hot_axle_box": "Safety-critical: verify with a second reading, restrict the unit and plan "
    "immediate removal of the axle box. Do not wait for a wayside HABD alarm.",
    "outer_race": "Envelope-spectrum check on this axle box for outer-race ball-pass tones; plan "
    "a bearing change before the temperature follows.",
    "inner_race": "Envelope-spectrum check on this axle box for inner-race tones; the impulsive "
    "content is rising before the RMS level.",
    "ball": "Envelope-spectrum check on this axle box for rolling-element tones and inspect the "
    "cage clearance.",
    "cage": "Inspect the bearing cage and grease condition on this axle box.",
    "sensor_stuck": "Check the axle-box temperature sensor and its wiring: the channel has stopped "
    "varying, which is an instrumentation fault, not a bearing fault.",
    "sensor_offset": "Check the axle-box temperature sensor calibration against its same-side "
    "peers before acting on the temperature.",
    UNKNOWN_FAULT: "Keep the component under watch and review the next scored window; the evidence "
    "does not yet name a single fault.",
}

_COMPONENT_LABEL: Final[dict[str, str]] = {
    "door": "door leaf {cid} drive and mechanism",
    "pneumatic": "air production unit {cid} (compressor, reservoir and twin-tower dryer)",
    "bearing": "axle box {cid} bearing",
}

#: Geometric decay of a signal's weight with its rank in |z| order.
_RANK_DECAY: Final[float] = 0.8


def rule_keys(subsystem: str) -> frozenset[str]:
    """Every signal key the rules for ``subsystem`` can match (test helper)."""
    return frozenset(key for rule in RULES.get(subsystem, ()) for key in rule.signals)


def _direction_ok(z: float, direction: str) -> bool:
    if direction == "any":
        return z != 0.0
    return z > 0 if direction == "up" else z < 0


def _thermal_fault(z: float, duration_h: float) -> str:
    """hot_axle_box when the thermal excursion is fast, bearing_degradation when it is slow.

    ``rate`` is |z| per hour of open episode. A box that is already several robust sigmas
    hot within a few hours is running away; the same excursion accumulated over a day or
    more is the slow degradation the demo is trying to catch early.
    """
    rate = abs(z) / max(float(duration_h), 1.0)
    if abs(z) >= 8.0 or rate >= 0.75:
        return "hot_axle_box"
    return "bearing_degradation"


@dataclass(frozen=True)
class _Ranked:
    """One top signal with its aliases, its physical channel and its rank weight."""

    signal: str
    aliases: tuple[str, ...]
    base: str
    z: float
    weight: float


def _rank(signals: Sequence[TopSignal]) -> list[_Ranked]:
    ordered = sorted(signals, key=lambda s: abs(s.z), reverse=True)
    return [
        _Ranked(
            signal=s.signal,
            aliases=signal_aliases(s.signal),
            base=_base(s.signal),
            z=float(s.z),
            weight=_RANK_DECAY**i,
        )
        for i, s in enumerate(ordered)
    ]


def _score_rule(rule: Rule, ranked: Iterable[_Ranked]) -> tuple[float, _Ranked | None]:
    """``(total evidence, the matched signal with the largest |z|)`` for one rule.

    At most one contribution per physical channel, so three window statistics of the same
    channel cannot out-vote three different channels.
    """
    best_per_base: dict[str, tuple[float, _Ranked]] = {}
    satisfied: set[str] = set()
    for item in ranked:
        key = next((a for a in item.aliases if a in rule.signals), None)
        if key is None:
            continue
        direction, weight = rule.signals[key]
        if not _direction_ok(item.z, direction):
            continue
        if key in rule.requires:
            satisfied.add(key)
        contribution = item.weight * weight
        current = best_per_base.get(item.base)
        if current is None or contribution > current[0]:
            best_per_base[item.base] = (contribution, item)
    if not best_per_base or not rule.requires <= satisfied:
        return 0.0, None
    total = sum(contribution for contribution, _ in best_per_base.values())
    lead = max((item for _, item in best_per_base.values()), key=lambda i: abs(i.z))
    return total, lead


def _event_scores(alert: AlertContext) -> dict[str, float]:
    text = " ".join(alert.recent_events).lower()
    out: dict[str, float] = {}
    for fault, token, weight in _EVENT_RULES.get(alert.subsystem, ()):
        if token in text:
            out[fault] = max(out.get(fault, 0.0), weight)
    return out


def classify(alert: AlertContext) -> tuple[str, str]:
    """Return ``(fault_type, reason)`` for one alert from its top signals.

    Faults are scored by evidence combination; the highest total wins and ties go to the
    earlier (more specific) rule. The reason quotes the strongest signal that actually
    supported the winning fault, so the sentence the engineer reads matches the bar the UI
    draws.
    """
    ranked = _rank(alert.top_signals)
    events = _event_scores(alert)

    best_fault = UNKNOWN_FAULT
    best_total = 0.0
    best_lead: _Ranked | None = None
    for rule in RULES.get(alert.subsystem, ()):
        total, lead = _score_rule(rule, ranked)
        total += events.get(rule.fault, 0.0)
        if total > best_total:
            best_fault, best_total, best_lead = rule.fault, total, lead

    for fault, weight in events.items():
        if weight > best_total:
            best_fault, best_total, best_lead = fault, weight, None

    if best_total <= 0.0:
        return UNKNOWN_FAULT, "no top signal matched a known failure pattern"

    if best_lead is None:
        reason = "the event log records it directly"
        return coerce_fault(best_fault, alert.subsystem), reason

    # Only the thermal rule splits: a cold-reading box wins on "sensor_offset" with a
    # temperature lead too, and that is an instrumentation fault, not a running-away box.
    if best_fault == "bearing_degradation" and best_lead.base in _THERMAL_BASES:
        best_fault = _thermal_fault(best_lead.z, alert.duration_h)

    reason = (
        f"{best_lead.signal} is {abs(best_lead.z):.1f} robust sigma "
        f"{'above' if best_lead.z > 0 else 'below'} the fitted healthy median"
    )
    return coerce_fault(best_fault, alert.subsystem), reason


def _urgency(alert: AlertContext, fault: str) -> str:
    if fault == "hot_axle_box":
        return "high"
    ratio, hours = alert.ratio, float(alert.duration_h)
    if ratio >= 2.0 or hours >= 24.0:
        return "high"
    if ratio >= 1.3 or hours >= 6.0:
        return "medium"
    return "low"


def _confidence(alert: AlertContext, fault: str) -> float:
    if fault == UNKNOWN_FAULT:
        return 0.2
    top_z = max((abs(s.z) for s in alert.top_signals), default=0.0)
    corroborating = sum(1 for s in alert.top_signals if abs(s.z) >= 2.0)
    conf = 0.25 + 0.05 * min(top_z, 8.0) + 0.05 * min(corroborating, 3)
    conf += 0.10 * min(max(alert.ratio - 1.0, 0.0), 2.0)
    return min(0.92, max(0.05, conf))


def _evidence(alert: AlertContext, reason: str) -> list[str]:
    bullets = [
        f"{alert.model_name} score {alert.score:.3g} against threshold {alert.threshold:.3g} "
        f"({alert.ratio:.2f}x), episode open {alert.duration_h:.1f} h"
    ]
    for sig in sorted(alert.top_signals, key=lambda s: abs(s.z), reverse=True)[:3]:
        val = "n/a" if sig.value is None else f"{sig.value:.4g}"
        bullets.append(f"{sig.signal} z={sig.z:+.2f} (value {val})")
    if alert.recent_events:
        bullets.append("recent events: " + "; ".join(alert.recent_events[:3]))
    elif alert.fleet_peer_summary:
        bullets.append(alert.fleet_peer_summary)
    if len(bullets) < 2:
        bullets.append(reason)
    return bullets[:5]


def template_advisory(alert: AlertContext) -> Advisory:
    """Deterministic advisory for ``alert``. Never raises, never touches the network."""
    fault, reason = classify(alert)
    urgency = _urgency(alert, fault)
    confidence = _confidence(alert, fault)
    label = _COMPONENT_LABEL.get(alert.subsystem, "{cid}").format(cid=alert.component_id)
    fault_words = fault.replace("_", " ")
    where = f"{alert.train_id} car {alert.car} {alert.component_id}"
    open_or_closed = "still open" if alert.t_end is None else "closed"

    if fault == UNKNOWN_FAULT:
        summary = (
            f"{where} has been in alarm for {alert.duration_h:.1f} h at "
            f"{alert.ratio:.2f}x its false-alarm threshold ({open_or_closed}). "
            "The top signals do not match a single known failure pattern, so treat it as "
            "an unclassified anomaly."
        )
    else:
        summary = (
            f"{where} looks like {fault_words}: {reason}. "
            f"The episode has been running {alert.duration_h:.1f} h at {alert.ratio:.2f}x "
            f"the false-alarm threshold ({open_or_closed})."
        )

    return Advisory(
        summary=summary,
        evidence=_evidence(alert, reason),
        likely_component=label,
        likely_fault=fault,
        recommended_action=_ACTIONS.get(fault, _ACTIONS[UNKNOWN_FAULT]),
        urgency=urgency,
        confidence=confidence,
        cbm_steps={
            "state_detection": (
                f"{alert.model_name} scored {alert.score:.3g} against a false-alarm-budget "
                f"threshold of {alert.threshold:.3g} on {alert.component_id}; the rows stayed "
                f"above threshold long enough to form episode {alert.episode_id}."
            ),
            "health_assessment": (
                f"Top-signal pattern points at {fault_words} ({reason})."
                if fault != UNKNOWN_FAULT
                else f"No known failure pattern matched; {reason}."
            ),
            "prognostic_assessment": (
                f"Episode open {alert.duration_h:.1f} h and {open_or_closed}; severity proxy "
                f"score/threshold = {alert.ratio:.2f}. No calibrated remaining-life estimate is "
                "available from the template path."
            ),
            "advisory": _ACTIONS.get(fault, _ACTIONS[UNKNOWN_FAULT]),
        },
        source="template",
        model=TEMPLATE_MODEL,
    )
