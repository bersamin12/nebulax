"""Importing this package registers every model module with
``nebulax.bench.registry`` (each module's ``@register`` decorators run at import time).

``nebulax.bench.runner._import_models`` does ``import nebulax.models`` for exactly this
reason. Add new model modules here by name in ``FAMILIES``; never remove another module's entry.

A family whose third-party dependency is not installed is skipped, not fatal: the shipped app
(``submission/nebulax/app/requirements.lock.txt``) has no xgboost, stumpy or torch, yet the PS3
predictors import ``nebulax.models.physics`` (door features) through this package. Skipped
families are listed in ``UNAVAILABLE`` (module name -> missing dependency) and announced with one
warning, so a benchmark run in a thin environment says which ladders it cannot build.
"""

from __future__ import annotations

import importlib
import warnings

FAMILIES: tuple[str, ...] = (
    "boosting",
    "classical",
    "foundation",
    "statistical",
    "deep",
    "tsc",
    "physics",
    "drift",
)

#: Families this environment could not import, as ``{module: missing dependency}``.
UNAVAILABLE: dict[str, str] = {}

for _name in FAMILIES:
    try:
        globals()[_name] = importlib.import_module(f"nebulax.models.{_name}")
    except ModuleNotFoundError as _exc:
        if _exc.name and _exc.name.startswith("nebulax."):
            raise  # our own module is missing: that is a bug, not an optional dependency
        UNAVAILABLE[_name] = _exc.name or str(_exc)

if UNAVAILABLE:
    warnings.warn(
        "nebulax.models: skipped "
        + ", ".join(f"{m} (no module named {d!r})" for m, d in UNAVAILABLE.items())
        + "; install environment.yml for the full model ladder",
        RuntimeWarning,
        stacklevel=2,
    )

del _name
