"""Problem Statement 3: the four subsystem tasks behind one shared contract.

`nebulax.ps3.common` holds the frozen interface (the ``Task`` protocol, the ``Explanation``
payload, the door timestamp codec, the paths and the model-artefact helpers); `scoring` holds the
four organiser metrics; `submission` validates and packs `predictions.zip`. The per-subsystem
modules (`door`, `acv`, `rail`, `shm`) register themselves on import - use :func:`get_task`, which
imports them lazily, rather than importing them by hand.

See `docs/ps3_contract.md` for the prose version of all of this.
"""

from __future__ import annotations

from nebulax.ps3.common import (
    ACCEPTED_SUFFIXES,
    CACHE_DIR,
    DATA_ROOT,
    MODEL_DIR,
    OUTPUT_FILENAMES,
    RESULTS_DIR,
    TASK_LABELS,
    TASK_NAMES,
    TASKS,
    BaseTask,
    Explanation,
    PredictionResult,
    Task,
    Trace,
    Viewport,
    available_tasks,
    data_root,
    get_task,
    load_model,
    register_task,
    save_model,
)

__all__ = [
    "ACCEPTED_SUFFIXES",
    "CACHE_DIR",
    "DATA_ROOT",
    "MODEL_DIR",
    "OUTPUT_FILENAMES",
    "RESULTS_DIR",
    "TASKS",
    "TASK_LABELS",
    "TASK_NAMES",
    "BaseTask",
    "Explanation",
    "PredictionResult",
    "Task",
    "Trace",
    "Viewport",
    "available_tasks",
    "data_root",
    "get_task",
    "load_model",
    "register_task",
    "save_model",
]
