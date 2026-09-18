"""Model benchmark: frozen interfaces (``base``) and the model registry (``registry``).

The rest of the benchmark (``splits``, ``thresholds``, ``metrics``, ``runner``, ``report``)
is built on top of these two modules and may not change them.

Protocol rules that these interfaces exist to enforce (plan "Evaluation protocol",
``configs/model_ladder.yaml:evaluation``): no point adjustment, no best-of-N reporting, and
no test data in threshold calibration.
"""

from __future__ import annotations

from nebulax.bench.base import (
    INPUT_KINDS,
    TASKS,
    AnomalyDetector,
    BaseModel,
    Classifier,
    InputKind,
    Regressor,
    Task,
)
from nebulax.bench.registry import (
    ModelSpec,
    build,
    get_spec,
    is_registered,
    list_models,
    register,
    unregister,
)

__all__ = [
    "INPUT_KINDS",
    "TASKS",
    "AnomalyDetector",
    "BaseModel",
    "Classifier",
    "InputKind",
    "Regressor",
    "Task",
    "ModelSpec",
    "build",
    "get_spec",
    "is_registered",
    "list_models",
    "register",
    "unregister",
]
