"""NEBULA X demo backend - FastAPI over ``data/scores/*`` and the sim partitions.

The app is the server half of ``docs/app_contract.md`` sections 4 and 5: a read-mostly JSON
API over the scored fleet, a WebSocket replay that pushes one *frame* per tick, and a
fault-injection route that re-simulates one component and re-scores it into an in-memory
overlay every other route reads first.

``uvicorn nebulax.api.main:app --port 8000`` from the repo root.
"""

from __future__ import annotations

from nebulax.api.state import FleetState, Settings

__all__ = ["FleetState", "Settings", "create_app", "app"]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    if name in ("create_app", "app"):
        from nebulax.api import main

        return getattr(main, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
