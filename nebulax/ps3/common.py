"""Shared contracts for the four Problem Statement 3 subsystems (door, ACV, rail, SHM).

This module is the **frozen interface** every other PS3 agent codes against; `docs/ps3_contract.md`
is its prose companion. Nothing here fits a model, reads a dataset end to end or holds state -
it defines types, paths, the door timestamp codec and the task registry, and nothing else.

Binding rules (plan section "W4", `docs/ps3_contract.md`):

* **Fold-local rule.** Every scaler, threshold, template, rule weight, feature selection, bias
  correction and augmentation is fitted *inside* the training fold. Model selection is nested or
  on a frozen outer split, never on the CV that produces the headline number. Held-out files are
  never augmented. Nothing in this module lets you sidestep that; the task modules must obey it.
* **Organiser schema is law.** `OUTPUT_FILENAMES`, `CSV_HEADERS` (in `nebulax.ps3.submission`)
  and the label vocabularies below are copied from `04_Example_Submission/` and the Info Kits.
* **Read-only data.** `readingmaterials/problem_statement/` is the organisers' clone; never write
  into it. Derived caches go to `CACHE_DIR` (`data/ps3_cache`, gitignored).

Layout of a task module (`nebulax/ps3/<name>.py`, written by the four model agents)::

    from nebulax.ps3.common import BaseTask, PredictionResult, register_task

    class DoorTask(BaseTask):
        name = "door"
        def load(self, path): ...        # -> Raw   (a DataFrame / ndarray / dict, task's choice)
        def featurise(self, raw): ...    # -> Feats (a DataFrame of one row per predicted item)
        def predict(self, feats, model=None): ...  # -> PredictionResult
    register_task(DoorTask())

`to_rows()` and `explain()` have working defaults on `BaseTask`; override them when the task needs
more than the straight pass-through.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

__all__ = [
    # vocabulary
    "TASK_NAMES",
    "TASK_LABELS",
    "OUTPUT_FILENAMES",
    "ACCEPTED_SUFFIXES",
    "DOOR_LABELS",
    "RAIL_LABELS",
    "HEALTH_STATES",
    "VIEWPORT_SIDES",
    "CAR_COL_RE",
    # paths
    "REPO_ROOT",
    "DATA_ROOT",
    "MODEL_DIR",
    "CACHE_DIR",
    "RESULTS_DIR",
    "MAX_MODEL_BYTES",
    "data_root",
    "dataset_dir",
    "train_dir",
    "test_dir",
    "labels_path",
    # timestamps
    "DOOR_TS_PATTERN",
    "parse_door_timestamp",
    "parse_door_timestamps",
    "format_door_timestamp",
    "format_door_timestamps",
    "to_ms",
    "read_door_segments",
    # payloads
    "Trace",
    "Viewport",
    "Explanation",
    "PredictionResult",
    # registry
    "Task",
    "BaseTask",
    "TASKS",
    "register_task",
    "get_task",
    "available_tasks",
    # artefacts
    "save_model",
    "load_model",
    "model_meta",
    "model_path",
    # misc
    "acv_car_ids",
    "natural_key",
    "git_rev",
]

# --------------------------------------------------------------------------------------
# Vocabulary (copied from the Info Kits and 04_Example_Submission - do not "improve")
# --------------------------------------------------------------------------------------

#: The four subsystems, in the order the app shows them.
TASK_NAMES: tuple[str, ...] = ("door", "acv", "rail", "shm")

#: Human-readable names for the UI.
TASK_LABELS: dict[str, str] = {
    "door": "Door",
    "acv": "ACV",
    "rail": "Rail Corrugation",
    "shm": "SHM",
}

#: Organiser-mandated output file names (Problem Statement 3 section 4.1, item 2).
OUTPUT_FILENAMES: dict[str, str] = {
    "door": "door_predictions.csv",
    "acv": "acv_predictions.csv",
    "rail": "rail_predictions.csv",
    "shm": "shm_predictions.csv",
}

#: Input extensions each task accepts (lowercase, leading dot). The upload page uses these.
ACCEPTED_SUFFIXES: dict[str, tuple[str, ...]] = {
    "door": (".csv",),
    "acv": (".xlsx", ".xls"),
    "rail": (".csv",),
    "shm": (".csv",),
}

#: Door status vocabulary (`Train_Segments_Answer.csv:status`, Door Info Kit section 3).
DOOR_LABELS: tuple[str, ...] = ("Normal", "Abnormal resistance")

#: Rail class vocabulary (`Train_Labels.csv:label`, Rail Info Kit section 4).
RAIL_LABELS: tuple[str, ...] = ("Normal", "Side I", "Side II")

#: Health tints the 3D viewport understands.
HEALTH_STATES: tuple[str, ...] = ("ok", "warn", "crit")

#: Side identifiers the viewport understands (rail sides I/II, door leaf L/R).
VIEWPORT_SIDES: tuple[str, ...] = ("I", "II", "L", "R")

#: ACV per-car column header, e.g. ``"Car 03 - ACV Running Mode"`` (ACV Info Kit section 2.1).
#: Accept the one-digit spelling too and preserve whichever spelling the uploaded workbook uses.
CAR_COL_RE: re.Pattern[str] = re.compile(r"^Car\s+(\d{1,2})\s*-\s*(.+)$")


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Where the organisers' datasets live; override with ``NEBULAX_PS3_DATA``.
DEFAULT_DATA_ROOT: Path = REPO_ROOT / "readingmaterials" / "problem_statement" / "PS3" / "02_Datasets"


def data_root() -> Path:
    """Dataset root, honouring ``NEBULAX_PS3_DATA`` **at call time** (tests monkeypatch it)."""
    env = os.environ.get("NEBULAX_PS3_DATA", "").strip()
    return Path(env).expanduser() if env else DEFAULT_DATA_ROOT


#: Convenience snapshot of :func:`data_root` taken at import. Prefer the function in library code.
DATA_ROOT: Path = data_root()

#: Committed model artefacts (small: ``MAX_MODEL_BYTES`` each).
MODEL_DIR: Path = REPO_ROOT / "models" / "ps3"

#: Derived caches (per-file features, resampled spectra). ``data/`` is gitignored.
CACHE_DIR: Path = REPO_ROOT / "data" / "ps3_cache"

#: CV jsons, leaderboards, ablation reports.
RESULTS_DIR: Path = REPO_ROOT / "results" / "ps3"

#: Hard cap on a committed artefact (plan: "artefacts must stay small, < 5 MB each").
MAX_MODEL_BYTES: int = 5 * 1024 * 1024

_DATASET_DIRS: dict[str, str] = {
    "door": "Door",
    "acv": "ACV",
    "rail": "Rail_Corrugation",
    "shm": "SHM",
}


def _check_task(name: str) -> str:
    key = str(name).strip().lower()
    if key not in TASK_NAMES:
        raise KeyError(f"unknown PS3 task {name!r}; expected one of {list(TASK_NAMES)}")
    return key


def dataset_dir(task: str, *, root: Path | str | None = None) -> Path:
    """``<data_root>/<subsystem folder>`` for one task."""
    base = Path(root) if root is not None else data_root()
    return base / _DATASET_DIRS[_check_task(task)]


def train_dir(task: str, *, root: Path | str | None = None) -> Path:
    """Training inputs. Door has no per-file structure: this is the folder holding ``Train.csv``."""
    d = dataset_dir(task, root=root)
    return d if _check_task(task) == "door" else d / "Train"


def test_dir(task: str, *, root: Path | str | None = None) -> Path:
    """Distributed held-out inputs. Door: the folder holding ``Test.csv`` (one continuous stream)."""
    d = dataset_dir(task, root=root)
    return d if _check_task(task) == "door" else d / "Test"


def labels_path(task: str, *, root: Path | str | None = None) -> Path:
    """Training-label file for a task (door's is the segment answer file)."""
    key = _check_task(task)
    d = dataset_dir(key, root=root)
    return d / ("Train_Segments_Answer.csv" if key == "door" else "Train_Labels.csv")


# --------------------------------------------------------------------------------------
# Door timestamps: "2023-7-5-0-11-17-664" = Y-M-D-H-M-S-ms, no zero padding, ms kept exactly
# --------------------------------------------------------------------------------------

#: The native door timestamp, e.g. ``2023-7-5-0-0-0-20``. Fields are **not** zero padded and the
#: last field is whole milliseconds (``...-59-77`` is 77 ms, not 770 ms).
DOOR_TS_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,3})\s*$"
)

_MS = "datetime64[ms]"


def parse_door_timestamp(value: Any) -> pd.Timestamp:
    """Parse one door timestamp; native ``Y-M-D-H-M-S-ms`` first, then any ISO-parseable string.

    Milliseconds are preserved exactly (the result is a millisecond-resolution ``Timestamp``).
    Datetime-like inputs pass straight through, so this is safe to call on already-parsed values.
    """
    if isinstance(value, pd.Timestamp):
        return value.as_unit("ms")
    if isinstance(value, (datetime, np.datetime64)):
        return pd.Timestamp(value).as_unit("ms")
    text = str(value).strip()
    m = DOOR_TS_PATTERN.match(text)
    if m is not None:
        y, mo, d, h, mi, s, ms = (int(g) for g in m.groups())
        return pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi, second=s).as_unit("ms") + pd.Timedelta(
            milliseconds=ms
        )
    try:
        ts = pd.Timestamp(text)
    except (ValueError, TypeError) as exc:  # pragma: no cover - message matters more than the path
        raise ValueError(f"not a door timestamp: {value!r} (want Y-M-D-H-M-S-ms or ISO)") from exc
    if pd.isna(ts):
        raise ValueError(f"not a door timestamp: {value!r} (want Y-M-D-H-M-S-ms or ISO)")
    return ts.as_unit("ms")


def parse_door_timestamps(values: Iterable[Any]) -> pd.Series:
    """Vectorised :func:`parse_door_timestamp` (18k-row streams parse in milliseconds).

    Returns a ``datetime64[ms]`` Series with the input's index when given a Series.
    """
    s = values if isinstance(values, pd.Series) else pd.Series(list(values), dtype="object")
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.astype(_MS)
    text = s.astype("string").str.strip()
    parts = text.str.extract(DOOR_TS_PATTERN)
    native = parts.notna().all(axis=1) & text.notna()
    out = pd.Series(np.full(len(s), np.datetime64("NaT"), dtype=_MS), index=s.index)
    if native.any():
        p = parts[native].astype("int64")
        p.columns = ["year", "month", "day", "hour", "minute", "second", "ms"]
        base = pd.to_datetime(p[["year", "month", "day", "hour", "minute", "second"]]).astype(_MS)
        out.loc[native] = (base + pd.to_timedelta(p["ms"], unit="ms")).astype(_MS)
    rest = ~native
    if rest.any():
        try:
            iso = pd.to_datetime(text[rest], format="ISO8601", errors="raise")
        except (ValueError, TypeError) as exc:
            bad = text[rest].iloc[0]
            raise ValueError(f"not a door timestamp: {bad!r} (want Y-M-D-H-M-S-ms or ISO)") from exc
        if getattr(iso.dt, "tz", None) is not None:
            iso = iso.dt.tz_convert("UTC").dt.tz_localize(None)
        out.loc[rest] = iso.astype(_MS)
    return out


def format_door_timestamp(value: Any) -> str:
    """Format a timestamp back into the native ``Y-M-D-H-M-S-ms`` form (no zero padding).

    Exact inverse of :func:`parse_door_timestamp` for native inputs. Timezone-aware values are
    converted to UTC and written naive.
    """
    ts = value if isinstance(value, pd.Timestamp) else parse_door_timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    ms = int(ts.microsecond // 1000)
    if ts.nanosecond or ts.microsecond % 1000:
        raise ValueError(f"sub-millisecond precision cannot be written in the door format: {value!r}")
    return f"{ts.year}-{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}-{ms}"


def format_door_timestamps(values: Iterable[Any]) -> list[str]:
    """:func:`format_door_timestamp` over an iterable."""
    return [format_door_timestamp(v) for v in values]


def to_ms(value: Any) -> int:
    """Coerce a segment boundary to integer epoch-milliseconds (what the IoU metric works in).

    Accepts datetimes, native/ISO strings and plain ints/floats (already milliseconds).
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"not a timestamp: {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"not a timestamp: {value!r}")
        return int(round(float(value)))
    ts = parse_door_timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return int(ts.as_unit("ms").value // 1_000_000)


def read_door_segments(path: Path | str, *, label_col: str | None = None) -> pd.DataFrame:
    """Read a door segment table (`Train_Segments_Answer.csv` **or** a `door_predictions.csv`).

    Returns the source columns plus parsed ``t_start`` / ``t_end`` (``datetime64[ms]``) and a
    normalised ``label`` column. ``label_col`` defaults to ``status`` then ``prediction``.
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    for col in ("start_time", "end_time"):
        if col not in df.columns:
            raise ValueError(f"{path}: missing required column {col!r}")
    col = label_col or next((c for c in ("status", "prediction", "label") if c in df.columns), None)
    if col is None:
        raise ValueError(f"{path}: no label column (looked for status / prediction / label)")
    out = df.copy()
    out["t_start"] = parse_door_timestamps(df["start_time"])
    out["t_end"] = parse_door_timestamps(df["end_time"])
    out["label"] = df[col].astype(str).str.strip()
    return out


# --------------------------------------------------------------------------------------
# Explanation payload (the web agent renders exactly this shape)
# --------------------------------------------------------------------------------------


@dataclass
class Trace:
    """One signal trace for the explanation panel.

    ``x`` is whatever the task's x axis is (door: native timestamp strings; rail: wavelength in
    cm; SHM: stress range; ACV: ISO timestamps). ``marks`` are free-form annotation dicts; use
    ``{"x": <point>, "label": str, "kind": str}`` for a point and additionally ``"x1"`` for a span.
    """

    x: list[Any] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    marks: list[dict[str, Any]] = field(default_factory=list)
    label: str = ""

    def __post_init__(self) -> None:
        self.x = [x.item() if isinstance(x, np.generic) else x for x in list(self.x)]
        self.y = [float(v) for v in np.asarray(list(self.y), dtype=float).tolist()]
        if len(self.x) != len(self.y):
            raise ValueError(f"Trace: x has {len(self.x)} points but y has {len(self.y)}")
        self.marks = [dict(m) for m in list(self.marks)]
        self.label = str(self.label)

    def as_dict(self) -> dict[str, Any]:
        return {"x": list(self.x), "y": list(self.y), "marks": [dict(m) for m in self.marks], "label": self.label}


@dataclass
class Viewport:
    """Where the 3D twin should point, and how to tint it."""

    car: int | None = None
    side: str | None = None
    health: str = "ok"
    component: str = ""

    def __post_init__(self) -> None:
        self.car = None if self.car is None else int(self.car)
        if self.car is not None and not 1 <= self.car <= 8:
            raise ValueError(f"Viewport.car must be 1..8 or None, got {self.car}")
        self.side = None if self.side in (None, "") else str(self.side)
        if self.side is not None and self.side not in VIEWPORT_SIDES:
            raise ValueError(f"Viewport.side must be one of {list(VIEWPORT_SIDES)} or None, got {self.side!r}")
        self.health = str(self.health)
        if self.health not in HEALTH_STATES:
            raise ValueError(f"Viewport.health must be one of {list(HEALTH_STATES)}, got {self.health!r}")
        self.component = str(self.component)

    def as_dict(self) -> dict[str, Any]:
        return {"car": self.car, "side": self.side, "health": self.health, "component": self.component}


@dataclass
class Explanation:
    """What the app shows next to one prediction: numbers, a trace, and a viewport target."""

    file_id: str
    numbers: dict[str, float] = field(default_factory=dict)
    trace: Trace = field(default_factory=Trace)
    viewport: Viewport = field(default_factory=Viewport)

    def __post_init__(self) -> None:
        self.file_id = str(self.file_id)
        self.numbers = {str(k): float(v) for k, v in dict(self.numbers).items()}
        if isinstance(self.trace, Mapping):
            self.trace = Trace(**dict(self.trace))
        if isinstance(self.viewport, Mapping):
            self.viewport = Viewport(**dict(self.viewport))
        if not isinstance(self.trace, Trace) or not isinstance(self.viewport, Viewport):
            raise TypeError("Explanation.trace must be a Trace and .viewport a Viewport")

    def as_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "numbers": dict(self.numbers),
            "trace": self.trace.as_dict(),
            "viewport": self.viewport.as_dict(),
        }


@dataclass
class PredictionResult:
    """What ``Task.predict`` returns for **one input file** (door: for the whole stream).

    ``rows`` are already in the organiser schema (see ``nebulax.ps3.submission.CSV_HEADERS``);
    ``numbers``/``trace``/``viewport`` feed the default :meth:`BaseTask.explain`; ``extras`` is
    the task's own scratch space (per-segment frames, feature importances, warnings) and is never
    written to a CSV.
    """

    task: str
    file_id: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)
    numbers: dict[str, float] = field(default_factory=dict)
    trace: Trace | None = None
    viewport: Viewport | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.task = _check_task(self.task)
        self.file_id = str(self.file_id)
        self.rows = [dict(r) for r in list(self.rows)]
        self.numbers = {str(k): float(v) for k, v in dict(self.numbers).items()}
        if isinstance(self.trace, Mapping):
            self.trace = Trace(**dict(self.trace))
        if isinstance(self.viewport, Mapping):
            self.viewport = Viewport(**dict(self.viewport))


# --------------------------------------------------------------------------------------
# Task protocol and registry
# --------------------------------------------------------------------------------------


@runtime_checkable
class Task(Protocol):
    """The five-call contract the app, the CLI and the submission packer rely on."""

    name: str
    label: str
    accepts: tuple[str, ...]
    output_filename: str

    def load(self, path: Path | str) -> Any:
        """Read one input file (door: the whole stream) into the task's ``Raw`` type."""

    def featurise(self, raw: Any) -> Any:
        """Turn ``Raw`` into the task's ``Feats`` (usually a DataFrame, one row per item)."""

    def predict(self, feats: Any, model: Any = None) -> PredictionResult:
        """Score ``Feats`` with a fitted model (``None`` = load the committed artefact)."""

    def to_rows(self, result: PredictionResult) -> list[dict[str, Any]]:
        """Organiser-schema rows for the CSV."""

    def explain(self, result: PredictionResult) -> Explanation:
        """The payload the UI renders beside the prediction."""


class BaseTask:
    """Convenience base: fills ``label``/``accepts``/``output_filename`` and gives defaults.

    Subclasses set ``name`` and implement :meth:`load`, :meth:`featurise` and :meth:`predict`.
    """

    name: str = ""

    def __init__(self) -> None:
        self.name = _check_task(self.name)
        self.label = getattr(self, "label", None) or TASK_LABELS[self.name]
        self.accepts = tuple(getattr(self, "accepts", None) or ACCEPTED_SUFFIXES[self.name])
        self.output_filename = getattr(self, "output_filename", None) or OUTPUT_FILENAMES[self.name]

    # -- to implement -------------------------------------------------------------------
    def load(self, path: Path | str) -> Any:
        raise NotImplementedError(f"{type(self).__name__}.load")

    def featurise(self, raw: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__}.featurise")

    def predict(self, feats: Any, model: Any = None) -> PredictionResult:
        raise NotImplementedError(f"{type(self).__name__}.predict")

    # -- defaults -----------------------------------------------------------------------
    def to_rows(self, result: PredictionResult) -> list[dict[str, Any]]:
        return [dict(r) for r in result.rows]

    def explain(self, result: PredictionResult) -> Explanation:
        return Explanation(
            file_id=result.file_id,
            numbers=dict(result.numbers),
            trace=result.trace or Trace(),
            viewport=result.viewport or Viewport(),
        )

    # -- helpers ------------------------------------------------------------------------
    def run(self, path: Path | str, model: Any = None) -> PredictionResult:
        """``load`` -> ``featurise`` -> ``predict``, with ``file_id`` defaulted to the basename."""
        result = self.predict(self.featurise(self.load(path)), model)
        if not result.file_id and self.name != "door":
            result.file_id = Path(path).name
        return result

    def __repr__(self) -> str:  # pragma: no cover - debugging sugar
        return f"<{type(self).__name__} name={self.name!r}>"


#: Registered tasks, keyed by name. Populated by the task modules' import-time ``register_task``.
TASKS: dict[str, Any] = {}

_REQUIRED_ATTRS = ("name", "label", "accepts", "output_filename")
_REQUIRED_METHODS = ("load", "featurise", "predict", "to_rows", "explain")


def register_task(task: Any) -> Any:
    """Validate and register a task; idempotent (re-registering replaces the entry)."""
    for attr in _REQUIRED_ATTRS:
        if not hasattr(task, attr):
            raise TypeError(f"{task!r} is not a PS3 Task: missing attribute {attr!r}")
    for meth in _REQUIRED_METHODS:
        if not callable(getattr(task, meth, None)):
            raise TypeError(f"{task!r} is not a PS3 Task: missing method {meth!r}")
    name = _check_task(task.name)
    accepts = tuple(task.accepts)
    if not accepts or any(not str(s).startswith(".") for s in accepts):
        raise ValueError(f"task {name!r}: accepts must be non-empty suffixes with a leading dot, got {accepts!r}")
    if task.output_filename != OUTPUT_FILENAMES[name]:
        raise ValueError(
            f"task {name!r}: output_filename must be {OUTPUT_FILENAMES[name]!r}, got {task.output_filename!r}"
        )
    TASKS[name] = task
    return task


def get_task(name: str) -> Any:
    """Return a registered task, importing ``nebulax.ps3.<name>`` on first use."""
    key = _check_task(name)
    if key not in TASKS:
        try:
            importlib.import_module(f"nebulax.ps3.{key}")
        except ImportError as exc:
            raise KeyError(
                f"PS3 task {key!r} is not registered and nebulax.ps3.{key} could not be imported ({exc})"
            ) from exc
    if key not in TASKS:
        raise KeyError(f"nebulax.ps3.{key} imported but never called register_task()")
    return TASKS[key]


def available_tasks() -> list[str]:
    """Names of the tasks whose modules import and register cleanly, in ``TASK_NAMES`` order."""
    out = []
    for name in TASK_NAMES:
        try:
            get_task(name)
        except KeyError:
            continue
        out.append(name)
    return out


# --------------------------------------------------------------------------------------
# Model artefacts
# --------------------------------------------------------------------------------------


def git_rev() -> str:
    """Short git revision of this checkout (``"unknown"`` outside one)."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - environment dependent
        return "unknown"


def model_path(task: str, *, model_dir: Path | str | None = None) -> Path:
    """``models/ps3/<task>.pkl``."""
    base = Path(model_dir) if model_dir is not None else MODEL_DIR
    return base / f"{_check_task(task)}.pkl"


def _meta_path(pkl: Path) -> Path:
    return pkl.with_suffix(".json")


def save_model(task: str, obj: Any, meta: Mapping[str, Any] | None = None, *, model_dir: Path | str | None = None) -> Path:
    """Pickle a fitted model to ``models/ps3/<task>.pkl`` with a json sidecar.

    The sidecar records the git rev, the save time and whatever ``meta`` the trainer passes (the
    CV summary belongs here). Raises if the artefact exceeds :data:`MAX_MODEL_BYTES`, so an
    accidentally huge model is caught before it is committed.
    """
    import joblib

    key = _check_task(task)
    pkl = model_path(key, model_dir=model_dir)
    pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, pkl, compress=3)
    size = pkl.stat().st_size
    if size > MAX_MODEL_BYTES:
        pkl.unlink()
        raise ValueError(
            f"{key} artefact is {size / 1e6:.1f} MB, over the {MAX_MODEL_BYTES / 1e6:.0f} MB cap; "
            "shrink the model (fewer trees / float32 / drop the training data you stashed on it)"
        )
    payload = {
        "task": key,
        "git_rev": git_rev(),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes": int(size),
        **{str(k): v for k, v in dict(meta or {}).items()},
    }
    _meta_path(pkl).write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return pkl


def load_model(task: str, *, model_dir: Path | str | None = None) -> Any:
    """Load the committed artefact for a task."""
    import joblib

    pkl = model_path(task, model_dir=model_dir)
    if not pkl.exists():
        raise FileNotFoundError(f"no model at {pkl}; run `python scripts/ps3_train.py --task {_check_task(task)}`")
    return joblib.load(pkl)


def model_meta(task: str, *, model_dir: Path | str | None = None) -> dict[str, Any]:
    """The json sidecar next to the artefact (``{}`` when there is none)."""
    meta = _meta_path(model_path(task, model_dir=model_dir))
    if not meta.exists():
        return {}
    return json.loads(meta.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------------------


def acv_car_ids(source: Any) -> list[str]:
    """Car ids present in an ACV file's own headers, sorted and spelled as supplied.

    ``source`` may be a path to the xlsx, a DataFrame, or an iterable of column names. This is the
    single definition of "the car identifier exactly as it appears in that file's own headers",
    which is what ``acv_predictions.csv:ranked_cars`` must use.
    """
    if isinstance(source, pd.DataFrame):
        cols: Sequence[Any] = list(source.columns)
    elif isinstance(source, (str, Path)):
        cols = list(pd.read_excel(source, nrows=0).columns)
    else:
        cols = list(source)
    ids = {m.group(1) for m in (CAR_COL_RE.match(str(c)) for c in cols) if m}
    return sorted(ids)


def natural_key(name: str) -> tuple[Any, ...]:
    """Sort key that puts ``Test2.csv`` before ``Test10.csv``."""
    return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(name)))
