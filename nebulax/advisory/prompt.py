"""System prompt and user-message rendering for the Claude advisory call.

The system prompt is **frozen**: it carries the role, the four CBM steps and the full
glossary for all three subsystems, and it is sent with ``cache_control`` so that every
advisory in a demo run reuses the same cached prefix. Anything that varies per alert
(numbers, top signals, fault candidates) lives in the user message, after the breakpoint.
"""

from __future__ import annotations

from typing import Final

from nebulax.advisory.glossary import full_glossary
from nebulax.advisory.schema import AlertContext

__all__ = ["SYSTEM", "build_system", "build_user_message"]


def build_system() -> str:
    return f"""\
You are the condition-based-maintenance (CBM) advisor for an LTA rolling-stock fleet in
Singapore. A depot engineer reads what you write on a wall display during a shift handover,
so be concrete, quantitative and short. You never invent numbers: every figure you use must
come from the alert you are given.

Work through the four CBM steps in order and report one sentence for each in `cbm_steps`:

1. state_detection - what the detector actually saw: which model, what score, against what
   false-alarm-budget threshold, over how long, and on which component.
2. health_assessment - what the top signals imply physically. Name the mechanism, not just
   the signal. Say when the signals disagree or when one signal is carrying the episode
   alone.
3. prognostic_assessment - how fast it is moving and what that means for time-to-action.
   If you have no basis for a remaining-life number, say so instead of guessing one.
4. advisory - the single next action for the depot, phrased as an instruction.

Rules:
- `likely_fault` must be exactly one of the fault candidates listed in the alert, or the
  literal string "unknown". Never return a fault name that is not on that list.
- `evidence` is two to five bullets and every bullet quotes a number you were given.
- `summary` is at most two sentences of plain language.
- `confidence` is between 0 and 1 and reflects how well the signals separate one fault from
  the others, not how alarming the score is.
- `urgency` is operational: "high" means act this shift, "medium" means book it into the
  next planned slot, "low" means keep it under watch.
- A single anomalous channel is weak evidence. Corroborating signals raise confidence; one
  lone signal lowers it.
- Distinguish instrumentation faults (dropouts, stuck or offset sensors) from physical
  faults - the recommended action is completely different.
- Do not include internal or system XML tags in your response.

DOMAIN GLOSSARY (authoritative - prefer it to general knowledge)

{full_glossary()}
"""


#: Frozen system prompt. Built once at import so the cached prefix is byte-stable.
SYSTEM: Final[str] = build_system()


def _render_signals(alert: AlertContext) -> str:
    if not alert.top_signals:
        return "  (none recorded)"
    rows = sorted(alert.top_signals, key=lambda s: abs(s.z), reverse=True)
    header = f"  {'signal':<26s}{'z':>8s}  value"
    lines = [header, "  " + "-" * 46]
    for sig in rows:
        val = "n/a" if sig.value is None else f"{sig.value:.4g}"
        lines.append(f"  {sig.signal:<26s}{sig.z:>+8.2f}  {val}")
    return "\n".join(lines)


def build_user_message(alert: AlertContext) -> str:
    """Compact, unit-carrying rendering of ``alert`` for the user turn."""
    t_end = alert.t_end.isoformat().replace("+00:00", "Z") if alert.t_end else "open"
    t_start = alert.t_start.isoformat().replace("+00:00", "Z")
    events = (
        "\n".join(f"  - {e}" for e in alert.recent_events[:10])
        if alert.recent_events
        else "  (none in the window)"
    )
    peers = alert.fleet_peer_summary or "  (no peer comparison supplied)"
    glossary = alert.glossary or "(see the glossary in your instructions)"
    candidates = ", ".join(alert.fault_candidates) or "unknown"

    return f"""\
ALARM EPISODE {alert.episode_id}

  train            {alert.train_id}
  car              {alert.car}
  subsystem        {alert.subsystem}
  component        {alert.component_id}
  detector         {alert.model_name}
  score            {alert.score:.4g}
  threshold        {alert.threshold:.4g}   (false-alarm-budget threshold)
  score/threshold  {alert.ratio:.2f}x
  episode start    {t_start}
  episode end      {t_end}
  duration         {alert.duration_h:.2f} h

TOP SIGNALS (robust z-score against the fitted healthy distribution, |z| descending)

{_render_signals(alert)}

RECENT EVENTS
{events}

FLEET / PEER CONTEXT
  {peers}

SUBSYSTEM GLOSSARY
{glossary}

FAULT CANDIDATES (choose exactly one of these for likely_fault, or "unknown")
  {candidates}

Produce the advisory now.
"""
