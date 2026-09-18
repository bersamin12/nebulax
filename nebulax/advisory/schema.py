"""Pydantic models for the advisory layer (``docs/app_contract.md`` section 6).

:class:`AlertContext` is what the API hands in, :class:`Advisory` is what comes back.
:class:`AdvisoryDraft` is an internal detail: it is the model actually used as the
structured-output schema for the Claude call, so the model cannot invent ``source`` /
``model`` and so ``cbm_steps`` is a closed object rather than an open map (open maps are
awkward for constrained decoding). :meth:`AdvisoryDraft.to_advisory` converts.

Nothing here raises on bad model output. Out-of-range confidences are clipped, unknown
fault names collapse to ``"unknown"``, and missing CBM steps are filled in.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nebulax.schema import ALL_FAULT_TYPES, FAULT_TYPES

__all__ = [
    "CBM_STEPS",
    "TopSignal",
    "AlertContext",
    "CbmSteps",
    "Advisory",
    "AdvisoryDraft",
    "coerce_fault",
    "UNKNOWN_FAULT",
    "UNKNOWN_CONFIDENCE_CAP",
]

UNKNOWN_FAULT: Final[str] = "unknown"

#: Ceiling on ``Advisory.confidence`` once ``likely_fault`` is ``"unknown"`` - see
#: :meth:`Advisory.narrow_to`. Same number the template path reports for an unmatched alert.
UNKNOWN_CONFIDENCE_CAP: Final[float] = 0.3

#: The four J177 / OSA-CBM steps, in order. Frozen keys of ``Advisory.cbm_steps``.
CBM_STEPS: Final[tuple[str, ...]] = (
    "state_detection",
    "health_assessment",
    "prognostic_assessment",
    "advisory",
)

_ALLOWED_FAULTS: Final[frozenset[str]] = frozenset(ALL_FAULT_TYPES) | {UNKNOWN_FAULT}


def coerce_fault(value: Any, subsystem: str | None = None) -> str:
    """Map ``value`` onto a legal fault type, never raising.

    With ``subsystem`` given the candidate set is ``FAULT_TYPES[subsystem] + {"unknown"}``;
    without it, any fault type known to :mod:`nebulax.schema` is accepted. Anything else
    (including ``None``, an empty string or a hallucinated label) becomes ``"unknown"``.
    """
    if value is None:
        return UNKNOWN_FAULT
    name = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if not name:
        return UNKNOWN_FAULT
    if subsystem is not None:
        allowed = set(FAULT_TYPES.get(str(subsystem), ())) | {UNKNOWN_FAULT}
    else:
        allowed = set(_ALLOWED_FAULTS)
    return name if name in allowed else UNKNOWN_FAULT


def _as_utc(value: Any) -> Any:
    """Coerce a datetime to timezone-aware UTC; leave anything else to pydantic."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return value


class TopSignal(BaseModel):
    """One entry of ``scores.top_signals_json``."""

    model_config = ConfigDict(extra="ignore")

    signal: str
    z: float = 0.0
    value: float | None = None

    @field_validator("z", mode="before")
    @classmethod
    def _finite_z(cls, v: Any) -> Any:
        return 0.0 if v is None else v

    def render(self) -> str:
        val = "n/a" if self.value is None else f"{self.value:.4g}"
        return f"{self.signal:<24s} z={self.z:+.2f}  value={val}"


class AlertContext(BaseModel):
    """Everything the advisory is allowed to know about one alarm episode."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    train_id: str
    car: int = 0
    subsystem: str
    component_id: str
    model_name: str
    score: float
    threshold: float
    t_start: datetime
    t_end: datetime | None = None
    duration_h: float = 0.0
    top_signals: list[TopSignal] = Field(default_factory=list)
    recent_events: list[str] = Field(default_factory=list)
    fleet_peer_summary: str = ""
    glossary: str = ""
    fault_candidates: list[str] = Field(default_factory=list)

    @field_validator("t_start", "t_end", mode="before")
    @classmethod
    def _utc_before(cls, v: Any) -> Any:
        return _as_utc(v)

    @field_validator("t_start", "t_end", mode="after")
    @classmethod
    def _utc_after(cls, v: Any) -> Any:
        # catches naive ISO strings, which only become datetimes after parsing
        return _as_utc(v)

    @model_validator(mode="after")
    def _fill_defaults(self) -> "AlertContext":
        if not self.fault_candidates:
            self.fault_candidates = [
                ft for ft in FAULT_TYPES.get(self.subsystem, ()) if ft != "healthy"
            ]
        if not self.glossary:
            from nebulax.advisory.glossary import glossary_for

            self.glossary = glossary_for(self.subsystem)
        return self

    @property
    def ratio(self) -> float:
        """score / threshold, guarded against a zero or negative threshold."""
        if self.threshold is None or self.threshold <= 0:
            return 1.0
        return float(self.score) / float(self.threshold)

    def signal(self, name: str) -> TopSignal | None:
        for sig in self.top_signals:
            if sig.signal == name:
                return sig
        return None


class CbmSteps(BaseModel):
    """The four CBM steps as a closed object.

    All four fields are required so that the generated JSON schema lists them in
    ``required`` - constrained decoding wants a closed, fully-required object. A
    ``before`` validator fills anything the caller omitted, so constructing one by hand
    still never raises.
    """

    model_config = ConfigDict(extra="ignore")

    state_detection: str = Field(description="What the detector saw, with the numbers.")
    health_assessment: str = Field(description="What the top signals imply physically.")
    prognostic_assessment: str = Field(description="How fast it is moving, time to action.")
    advisory: str = Field(description="The single next action for the depot.")

    @model_validator(mode="before")
    @classmethod
    def _fill(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {key: data.get(key, "") for key in CBM_STEPS}
        return data

    def as_dict(self) -> dict[str, str]:
        return {key: str(getattr(self, key) or "not assessed") for key in CBM_STEPS}


class Advisory(BaseModel):
    """The advisory returned to the API. Frozen field names - see the app contract."""

    model_config = ConfigDict(extra="ignore")

    summary: str
    evidence: list[str] = Field(default_factory=list)
    likely_component: str
    likely_fault: str = UNKNOWN_FAULT
    recommended_action: str
    urgency: Literal["low", "medium", "high"] = "medium"
    confidence: float = 0.5
    cbm_steps: dict[str, str] = Field(default_factory=dict)
    source: Literal["claude", "template"] = "template"
    model: str = "template-v1"

    @field_validator("evidence", mode="before")
    @classmethod
    def _clean_evidence(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        items = [str(x).strip() for x in v if str(x).strip()]
        return items[:5]

    @field_validator("likely_fault", mode="before")
    @classmethod
    def _clean_fault(cls, v: Any) -> str:
        return coerce_fault(v)

    @field_validator("urgency", mode="before")
    @classmethod
    def _clean_urgency(cls, v: Any) -> Any:
        name = str(v or "").strip().lower()
        return name if name in ("low", "medium", "high") else "medium"

    @field_validator("confidence", mode="before")
    @classmethod
    def _clip_confidence(cls, v: Any) -> float:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return 0.5
        if x != x:  # NaN
            return 0.5
        return min(1.0, max(0.0, x))

    @field_validator("cbm_steps", mode="before")
    @classmethod
    def _fill_steps(cls, v: Any) -> dict[str, str]:
        raw = dict(v) if isinstance(v, dict) else {}
        return {key: str(raw.get(key) or "not assessed") for key in CBM_STEPS}

    def narrow_to(self, subsystem: str) -> "Advisory":
        """Return a copy whose ``likely_fault`` is legal for ``subsystem``.

        An advisory that ends up on ``"unknown"`` is not allowed to keep a high confidence:
        the number would be read as confidence in a diagnosis that no longer exists (the
        model named a fault from another subsystem, or one nobody models). It is clamped to
        :data:`UNKNOWN_CONFIDENCE_CAP`, matching what the template path reports when nothing
        matches.
        """
        narrowed = coerce_fault(self.likely_fault, subsystem)
        update: dict[str, Any] = {}
        if narrowed != self.likely_fault:
            update["likely_fault"] = narrowed
        if narrowed == UNKNOWN_FAULT and self.confidence > UNKNOWN_CONFIDENCE_CAP:
            update["confidence"] = UNKNOWN_CONFIDENCE_CAP
        return self.model_copy(update=update) if update else self


class AdvisoryDraft(BaseModel):
    """Structured-output schema for the Claude call (no ``source`` / ``model`` fields)."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(
        description="At most two sentences of plain language for a depot engineer."
    )
    evidence: list[str] = Field(
        description="Two to five short bullets, each quoting a number you were given."
    )
    likely_component: str = Field(description="The physical part you suspect.")
    likely_fault: str = Field(
        description="Exactly one of the listed fault candidates, or 'unknown'."
    )
    recommended_action: str = Field(description="What the depot should do next.")
    urgency: Literal["low", "medium", "high"]
    confidence: float = Field(description="0..1, your confidence in likely_fault.")
    cbm_steps: CbmSteps = Field(
        description="One sentence per CBM step: state detection, health assessment, "
        "prognostic assessment, advisory."
    )

    def to_advisory(self, *, model: str, subsystem: str | None = None) -> Advisory:
        adv = Advisory(
            summary=self.summary,
            evidence=self.evidence,
            likely_component=self.likely_component,
            likely_fault=self.likely_fault,
            recommended_action=self.recommended_action,
            urgency=self.urgency,
            confidence=self.confidence,
            cbm_steps=self.cbm_steps.as_dict(),
            source="claude",
            model=model,
        )
        return adv.narrow_to(subsystem) if subsystem else adv
