"""Advisory layer: turn one alarm episode into a CBM advisory a depot engineer can act on.

``docs/app_contract.md`` section 6 is the interface. The only entry point the API uses is
:func:`advise`, and it is total: with no API key, with a broken client, with a slow network
or with malformed model output it returns the deterministic template advisory instead of
raising. The Claude call runs on a daemon thread with a hard join timeout, so a hung HTTP
request can never hold the demo (or the interpreter at exit).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Mapping, TypeVar

from nebulax.advisory.client import (
    API_KEY_ENV,
    CLAUDE_MODEL,
    MAX_RETRIES,
    TIMEOUT_FRACTION,
    build_client,
    call_claude,
    find_api_key,
    parse_env_file,
    request_timeout,
)
from nebulax.advisory.glossary import GLOSSARY, full_glossary, glossary_for
from nebulax.advisory.prompt import SYSTEM, build_system, build_user_message
from nebulax.advisory.schema import (
    CBM_STEPS,
    UNKNOWN_CONFIDENCE_CAP,
    UNKNOWN_FAULT,
    Advisory,
    AdvisoryDraft,
    AlertContext,
    CbmSteps,
    TopSignal,
    coerce_fault,
)
from nebulax.advisory.template import (
    RULES,
    TEMPLATE_MODEL,
    Rule,
    classify,
    rule_keys,
    signal_aliases,
    template_advisory,
)

__all__ = [
    "Advisory",
    "AdvisoryDraft",
    "AlertContext",
    "CbmSteps",
    "TopSignal",
    "CBM_STEPS",
    "UNKNOWN_FAULT",
    "UNKNOWN_CONFIDENCE_CAP",
    "CLAUDE_MODEL",
    "TEMPLATE_MODEL",
    "API_KEY_ENV",
    "TIMEOUT_FRACTION",
    "MAX_RETRIES",
    "RULES",
    "Rule",
    "request_timeout",
    "rule_keys",
    "signal_aliases",
    "SYSTEM",
    "GLOSSARY",
    "advise",
    "build_system",
    "build_user_message",
    "build_client",
    "call_claude",
    "classify",
    "coerce_fault",
    "find_api_key",
    "full_glossary",
    "glossary_for",
    "parse_env_file",
    "template_advisory",
]

log = logging.getLogger(__name__)

_T = TypeVar("_T")


def _run_with_timeout(fn: Callable[[], _T], timeout_s: float) -> _T:
    """Run ``fn`` on a daemon thread and give up after ``timeout_s`` seconds.

    A daemon thread (rather than ``ThreadPoolExecutor``) is deliberate: a hung HTTP call
    must not be joined at interpreter exit.
    """
    box: dict[str, Any] = {}

    def runner() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread
            box["error"] = exc

    thread = threading.Thread(target=runner, name="nebulax-advisory", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"advisory call exceeded {timeout_s:g}s")
    if "error" in box:
        raise box["error"]
    if "value" not in box:  # pragma: no cover - thread died without setting either
        raise RuntimeError("advisory call produced no result")
    return box["value"]  # type: ignore[return-value]


def advise(
    alert: AlertContext,
    *,
    client: Any = None,
    timeout_s: float = 20.0,
    env: Mapping[str, str] | None = None,
    dotenv_path: str | os.PathLike[str] | None = None,
) -> Advisory:
    """Advise on one alarm episode.

    Returns a Claude-authored :class:`Advisory` (``source="claude"``) when a client can be
    built or was injected and the call succeeds inside ``timeout_s``; otherwise the
    deterministic template advisory (``source="template"``). Never raises.

    ``env`` and ``dotenv_path`` are test seams; the API passes neither.
    """
    try:
        active = client
        if active is None:
            active = build_client(env=env, dotenv_path=dotenv_path, timeout_s=timeout_s)
        return _run_with_timeout(
            lambda: call_claude(alert, active, timeout_s=timeout_s), timeout_s
        )
    except Exception as exc:  # noqa: BLE001 - the demo must never fail on this
        log.info("advisory: falling back to the template (%s: %s)", type(exc).__name__, exc)
    return template_advisory(alert)
