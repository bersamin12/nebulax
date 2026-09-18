"""NEBULA X Track 3 - predictive fault detection on rolling stock.

Sub-packages
------------
``nebulax.schema``    the frozen common data schema (long telemetry, feature table,
                      fault log, event log, scores) plus readers/writers. Everything
                      in the repo speaks this.
``nebulax.sim``       behavioural twin: shared degradation/service/sensor machinery in
                      ``sim.common`` and one module per subsystem (door, pneumatic, bearing).
``nebulax.bench``     model benchmark: abstract model interfaces and the model registry.
``nebulax.adapters``  real-dataset loaders (MetroPT-3, Cranfield, Ottawa) emitting the
                      common schema, plus the ``python -m nebulax.adapters.validate`` CLI.

No editable install: the repo root is on ``sys.path`` (``pythonpath = ["."]`` for pytest,
cwd for ``python -c``/``python -m``).
"""

from __future__ import annotations

__version__ = "0.1.0"
SCHEMA_VERSION = 1

__all__ = ["__version__", "SCHEMA_VERSION"]
