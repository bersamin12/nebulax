"""Anthropic client plumbing for the advisory.

Three jobs, all of them small:

* find ``ANTHROPIC_API_KEY`` (process environment first, then a repo-root ``.env``, parsed
  by :func:`parse_env_file` because python-dotenv is not a dependency of this project);
* build an :class:`anthropic.Anthropic` client lazily, so importing ``nebulax.advisory``
  never needs the SDK or a key;
* make the one structured-output call and hand back an :class:`Advisory`.

Nothing here is allowed to block the demo: every failure path raises an ordinary exception
that :func:`nebulax.advisory.advise` turns into the template advisory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final, Mapping

from nebulax.advisory.prompt import SYSTEM, build_user_message
from nebulax.advisory.schema import AdvisoryDraft, AlertContext, Advisory

__all__ = [
    "CLAUDE_MODEL",
    "MAX_TOKENS",
    "EFFORT",
    "API_KEY_ENV",
    "TIMEOUT_FRACTION",
    "MAX_RETRIES",
    "request_timeout",
    "parse_env_file",
    "repo_root",
    "find_api_key",
    "build_client",
    "call_claude",
]

#: Frozen by ``docs/app_contract.md`` section 6.
CLAUDE_MODEL: Final[str] = "claude-opus-5"
MAX_TOKENS: Final[int] = 1024
EFFORT: Final[str] = "low"
API_KEY_ENV: Final[str] = "ANTHROPIC_API_KEY"
#: The SDK gets this fraction of the caller's budget, so the HTTP call gives up *before*
#: ``nebulax.advisory.advise`` fires its hard join timeout; the join stays the guarantee.
TIMEOUT_FRACTION: Final[float] = 0.8
#: No SDK-level retries: a retry would spend the rest of the budget and the demo would get
#: the template anyway. One attempt, then fall back.
MAX_RETRIES: Final[int] = 0


def request_timeout(timeout_s: float) -> float:
    """The per-request HTTP timeout for a caller budget of ``timeout_s`` seconds."""
    return max(1.0, float(timeout_s) * TIMEOUT_FRACTION)


def repo_root() -> Path:
    """The repository root (``nebulax/advisory/client.py`` -> two parents up)."""
    return Path(__file__).resolve().parents[2]


def parse_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    """Parse a dotenv-style file. Returns ``{}`` for a missing or unreadable file.

    Supports ``KEY=value``, ``export KEY=value``, ``#`` comments, blank lines and single or
    double quoted values. Deliberately tiny: python-dotenv is not installed and this project
    never installs packages.
    """
    out: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            # strip a trailing inline comment on an unquoted value
            hash_at = value.find(" #")
            if hash_at != -1:
                value = value[:hash_at].rstrip()
        out[key] = value
    return out


def find_api_key(
    *,
    env: Mapping[str, str] | None = None,
    dotenv_path: str | os.PathLike[str] | None = None,
) -> str | None:
    """``ANTHROPIC_API_KEY`` from the environment, else from the repo-root ``.env``.

    The process environment wins. The ``.env`` file is never written and never exported into
    ``os.environ`` - it is read on demand so that tests can point at a temporary file.
    """
    env = os.environ if env is None else env
    key = (env.get(API_KEY_ENV) or "").strip()
    if key:
        return key
    path = Path(dotenv_path) if dotenv_path is not None else repo_root() / ".env"
    key = (parse_env_file(path).get(API_KEY_ENV) or "").strip()
    return key or None


def build_client(
    *,
    env: Mapping[str, str] | None = None,
    dotenv_path: str | os.PathLike[str] | None = None,
    timeout_s: float = 20.0,
) -> Any:
    """Construct an ``anthropic.Anthropic``. Raises if there is no key or no SDK."""
    key = find_api_key(env=env, dotenv_path=dotenv_path)
    if not key:
        raise RuntimeError(f"{API_KEY_ENV} is not set (checked the environment and .env)")
    import anthropic  # imported lazily: the SDK is optional at import time

    return anthropic.Anthropic(
        api_key=key, timeout=request_timeout(timeout_s), max_retries=MAX_RETRIES
    )


def call_claude(alert: AlertContext, client: Any, *, timeout_s: float = 20.0) -> Advisory:
    """One structured-output call. Raises on any failure; the caller falls back.

    ``client.messages.parse`` is the SDK's structured-output helper (``anthropic`` 1.5.0):
    ``output_format`` takes the pydantic model and ``output_config={"effort": ...}`` sets the
    thinking/effort level. Thinking is disabled here because the contract freezes
    ``max_tokens`` at 1024 and Claude Opus 5 runs adaptive thinking by default, which would
    eat the budget the advisory itself needs; ``effort="low"`` keeps it cheap either way
    (and ``{"type": "disabled"}`` is only accepted at effort ``high`` or below).

    The HTTP call gets :func:`request_timeout` - a fraction of the caller's budget - and
    ``max_retries=0``, so the daemon thread in :func:`nebulax.advisory.advise` returns on its
    own before the hard join timeout fires instead of being abandoned mid-request.
    """
    budget = request_timeout(timeout_s)
    target = client
    with_options = getattr(client, "with_options", None)
    if callable(with_options):  # the real SDK; a test double may not have it
        try:
            target = with_options(max_retries=MAX_RETRIES, timeout=budget)
        except TypeError:  # pragma: no cover - defensive, an odd stand-in
            target = client
    response = target.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"effort": EFFORT},
        thinking={"type": "disabled"},
        system=[
            {
                "type": "text",
                "text": SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": build_user_message(alert)}],
        output_format=AdvisoryDraft,
        timeout=budget,
    )
    draft = getattr(response, "parsed_output", None)
    if draft is None:
        raise ValueError(
            "advisory: no parsed output in the response "
            f"(stop_reason={getattr(response, 'stop_reason', None)!r})"
        )
    if not isinstance(draft, AdvisoryDraft):
        draft = AdvisoryDraft.model_validate(
            draft if isinstance(draft, dict) else draft.model_dump()
        )
    return draft.to_advisory(model=CLAUDE_MODEL, subsystem=alert.subsystem)
