"""The frozen common data schema. Every simulator, adapter, feature module and model speaks it.

Five tables
-----------
1. **Telemetry (long)** - one row per (timestamp, component, signal). Columns/dtypes in
   :data:`LONG_COLUMNS`. Written as parquet+zstd partitioned by ``source``/``run_id``.
2. **Feature table** - one row per door cycle / compressor cycle / bearing window.
   Key columns in :data:`FEATURE_KEY_COLUMNS`, labels in :data:`LABEL_COLUMNS`, columns
   prefixed :data:`METADATA_PREFIX` (``meta_``) describe the sample but are **not** model
   inputs, and everything else is a feature column (vector features such as
   ``current_profile_50`` are list columns). :func:`feature_columns` is the single source of
   truth for that split and is what the benchmark builds ``X`` from.
3. **Fault log** - ground truth, one row per injected/known fault. :data:`FAULT_LOG_COLUMNS`.
4. **Event log** - discrete events (obstruction, LPS, purge, hot box...). :data:`EVENT_LOG_COLUMNS`.
5. **Scores** - benchmark output consumed by the API/demo. :data:`SCORES_COLUMNS`.

Conventions that are NOT negotiable
-----------------------------------
* ``timestamp`` is timezone-aware UTC with millisecond resolution (``datetime64[ms, UTC]``).
* ``value`` is ``float32`` and ``NaN`` means *dropout*, never "zero" and never "unknown category".
  Digital channels are carried as 0.0/1.0 floats in the same column.
* ``car`` is ``int8``; ``0`` means unit-level (a signal that belongs to the whole set).
* Component ids: ``door_L1..door_L4`` / ``door_R1..door_R4``, ``apu_1``, ``axlebox_1L..axlebox_4R``,
  and ``train`` for train-context signals. The ``car`` column disambiguates per-car components.
* ``subsystem`` takes a fourth value ``"train"`` beyond door/pneumatic/bearing: it is the
  *context pseudo-subsystem* carrying ``speed, load_frac, T_amb, in_service`` on
  ``component_id == "train"`` once per train (not once per subsystem). :data:`FAULT_SUBSYSTEMS`
  is the three real ones.
* Key columns are pandas ``category`` dtype. Call :func:`coerce_long` (and friends) rather
  than hand-rolling ``astype``; the validators refuse object dtype on purpose so that a
  30-day fleet stays small in memory and in parquet.
* **``meta_`` is a reserved prefix on the feature table.** Anything that describes the sample
  but is not a legitimate model input wears it: dataset class labels (``meta_class``),
  condition ids and severity stages (``meta_level``, ``meta_state``), latent ground-truth
  counters (``meta_shock_count``) and the ids used only for grouping splits
  (``meta_bearing_id``, ``meta_test_id``). Any dtype is allowed - a ``meta_`` column is never
  fed to a model, so it is free to be a string, an id or a counter. Group ids for
  leave-one-out splits are read from ``meta_*`` or key columns, never from ``X``.

Physical-constant provenance: where the approved plan and ``docs/research/rail_phm.md``
disagree, **rail_phm.md wins**. Signal-registry consequences of that rule are marked
``[rail_phm]`` below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from nebulax import SCHEMA_VERSION, __version__

__all__ = [
    "SCHEMA_VERSION",
    "SOURCES",
    "SUBSYSTEMS",
    "FAULT_SUBSYSTEMS",
    "LONG_COLUMNS",
    "FEATURE_KEY_COLUMNS",
    "FEATURE_KEY_DTYPES",
    "LABEL_COLUMNS",
    "METADATA_PREFIX",
    "feature_columns",
    "FAULT_LOG_COLUMNS",
    "EVENT_LOG_COLUMNS",
    "SCORES_COLUMNS",
    "SignalSpec",
    "SIGNAL_SPECS",
    "SIGNALS",
    "METROPT3_SIGNALS",
    "CONTEXT_SIGNALS",
    "FAULT_TYPES",
    "EVENT_TYPES",
    "DEGRADATION_SHAPES",
    "DOOR_COMPONENT_IDS",
    "APU_COMPONENT_IDS",
    "AXLEBOX_COMPONENT_IDS",
    "TRAIN_COMPONENT_ID",
    "COMPONENT_IDS",
    "component_ids",
    "is_valid_component_id",
    "empty_long",
    "empty_features",
    "empty_fault_log",
    "empty_events",
    "empty_scores",
    "coerce_long",
    "coerce_features",
    "coerce_fault_log",
    "coerce_events",
    "coerce_scores",
    "validate_long",
    "validate_features",
    "validate_fault_log",
    "validate_events",
    "validate_scores",
    "to_wide",
    "to_long",
    "Dataset",
    "write_dataset",
    "read_dataset",
]

Source = Literal["sim", "metropt3", "metropt2", "cranfield", "ottawa"]
Subsystem = Literal["door", "pneumatic", "bearing", "train"]

SOURCES: Final[tuple[str, ...]] = ("sim", "metropt3", "metropt2", "cranfield", "ottawa")
SUBSYSTEMS: Final[tuple[str, ...]] = ("door", "pneumatic", "bearing", "train")
FAULT_SUBSYSTEMS: Final[tuple[str, ...]] = ("door", "pneumatic", "bearing")

TIMESTAMP_DTYPE: Final[str] = "datetime64[ms, UTC]"

# --------------------------------------------------------------------------------------
# Table column specs
# --------------------------------------------------------------------------------------

#: Long telemetry table: ordered column -> pandas dtype string.
LONG_COLUMNS: Final[dict[str, str]] = {
    "timestamp": TIMESTAMP_DTYPE,
    "source": "category",
    "run_id": "category",
    "train_id": "category",
    "car": "int8",
    "subsystem": "category",
    "component_id": "category",
    "signal": "category",
    "value": "float32",
}

#: Feature table key columns (one row per cycle/window), ordered.
FEATURE_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "run_id",
    "source",
    "train_id",
    "car",
    "subsystem",
    "component_id",
    "cycle_id",
    "t_start",
    "t_end",
)

FEATURE_KEY_DTYPES: Final[dict[str, str]] = {
    "run_id": "category",
    "source": "category",
    "train_id": "category",
    "car": "int8",
    "subsystem": "category",
    "component_id": "category",
    "cycle_id": "int64",
    "t_start": TIMESTAMP_DTYPE,
    "t_end": TIMESTAMP_DTYPE,
}

#: Label columns joined onto the feature table from the fault log. All optional at write
#: time (real datasets without ground truth simply omit them), but if present they must
#: carry these dtypes. ``severity`` is sim-only.
LABEL_COLUMNS: Final[dict[str, str]] = {
    "fault_type": "category",
    "severity": "float32",
    "rul_s": "float32",
    "is_faulty": "bool",
    "alarm_window_3d": "bool",
}

#: Reserved prefix for feature-table columns that describe the sample but are NOT legitimate
#: model inputs: dataset class labels, condition ids, severity stages, latent ground-truth
#: counters and the bearing/test ids used only for grouping. See :func:`feature_columns`.
METADATA_PREFIX: Final[str] = "meta_"

FAULT_LOG_COLUMNS: Final[dict[str, str]] = {
    "run_id": "category",
    "train_id": "category",
    "car": "int8",
    "subsystem": "category",
    "component_id": "category",
    "fault_type": "category",
    "t_onset": TIMESTAMP_DTYPE,
    "t_failure": TIMESTAMP_DTYPE,
    "t_functional_failure": TIMESTAMP_DTYPE,
    "gamma": "float32",
    "shape": "category",
    "params_json": "object",
}

EVENT_LOG_COLUMNS: Final[dict[str, str]] = {
    "run_id": "category",
    "timestamp": TIMESTAMP_DTYPE,
    "train_id": "category",
    "car": "int8",
    "subsystem": "category",
    "component_id": "category",
    "event": "category",
    "detail_json": "object",
}

SCORES_COLUMNS: Final[dict[str, str]] = {
    "timestamp": TIMESTAMP_DTYPE,
    "train_id": "category",
    "car": "int8",
    "subsystem": "category",
    "component_id": "category",
    "model": "category",
    "score": "float32",
    "threshold": "float32",
    "alert": "bool",
    "top_signals_json": "object",
}

# --------------------------------------------------------------------------------------
# Signal registry
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """One telemetry channel.

    ``kind`` is ``"analog"`` or ``"digital"`` (digital channels still live in the float32
    ``value`` column as 0.0/1.0). ``fs_hz`` is the nominal native rate of the simulator,
    which adapters may legitimately differ from - it is documentation, not a constraint.
    """

    name: str
    subsystem: str
    unit: str
    kind: Literal["analog", "digital"]
    fs_hz: float
    note: str = ""


def _specs(rows: Sequence[tuple[str, str, str, str, float, str]]) -> tuple[SignalSpec, ...]:
    return tuple(SignalSpec(n, sub, u, k, f, note) for n, sub, u, k, f, note in rows)  # type: ignore[arg-type]


_DOOR = _specs(
    [
        ("pos_ref", "door", "m", "analog", 100.0, "commanded leaf position, one leaf 0..0.725 m"),
        ("pos", "door", "m", "analog", 100.0, "measured leaf position"),
        ("vel", "door", "m/s", "analog", 100.0, "leaf velocity"),
        ("current", "door", "A", "analog", 100.0, "motor current, the one channel the field actually uses [rail_phm 1.1]"),
        ("voltage", "door", "V", "analog", 100.0, "motor terminal voltage, 110 Vdc LV bus"),
        ("pwm", "door", "frac", "analog", 100.0, "PWM duty, -1..1"),
        ("ls_open", "door", "bool", "digital", 100.0, "open limit switch"),
        ("ls_closed", "door", "bool", "digital", 100.0, "closed limit switch"),
        ("interlock", "door", "bool", "digital", 100.0, "door interlock switch [rail_phm 4.1, R74]"),
        ("door_key", "door", "bool", "digital", 100.0, "door key switch [rail_phm 4.1, R74]"),
        ("emergency_relay", "door", "bool", "digital", 100.0, "emergency passenger relay [rail_phm 4.1, R74]"),
        ("obstruction", "door", "bool", "digital", 100.0, "obstruction detected this cycle"),
        ("T_motor", "door", "degC", "analog", 100.0, "motor thermal node"),
    ]
)

# MetroPT-3 (UCI 791) column names VERBATIM - do not rename, do not fix the 'eletric' typo.
_METROPT3_NAMES: Final[tuple[str, ...]] = (
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
)

_PNEUMATIC = _specs(
    [
        ("TP2", "pneumatic", "bar", "analog", 1.0, "compressor discharge pressure, upstream of separator/filter/dryer: ~0 bar off and unloaded, panel + line drop while loaded"),
        ("TP3", "pneumatic", "bar", "analog", 1.0, "pneumatic panel pressure"),
        ("H1", "pneumatic", "bar", "analog", 1.0, "cyclonic-separator discharge tap: reads panel pressure while the compressor is off or unloaded, ~0 bar while loaded"),
        ("DV_pressure", "pneumatic", "bar", "analog", 1.0, "pressure drop across the air-dryer towers; 0 while loaded"),
        ("Reservoirs", "pneumatic", "bar", "analog", 1.0, "main reservoir pressure, control band 8.2-10.2 bar"),
        ("Oil_temperature", "pneumatic", "degC", "analog", 1.0, "compressor oil temperature"),
        ("Motor_current", "pneumatic", "A", "analog", 1.0, "0 off / ~4 offloaded / ~7 loaded [rail_phm 2.3b]"),
        ("COMP", "pneumatic", "bool", "digital", 1.0, "air intake valve; 0 while the compressor is working"),
        ("DV_eletric", "pneumatic", "bool", "digital", 1.0, "outlet valve of the compressor; 1 while working"),
        ("Towers", "pneumatic", "bool", "digital", 1.0, "which drying tower is active"),
        ("MPG", "pneumatic", "bool", "digital", 1.0, "starts the compressor below 8.2 bar"),
        ("LPS", "pneumatic", "bool", "digital", 1.0, "low pressure signal, active below 7 bar"),
        ("Pressure_switch", "pneumatic", "bool", "digital", 1.0, "discharge in the air-drying towers"),
        ("Oil_level", "pneumatic", "bool", "digital", 1.0, "1 when oil is below the expected level"),
        ("Caudal_impulses", "pneumatic", "bool", "digital", 1.0, "air-flow pulse counter"),
        ("Flowmeter", "pneumatic", "NL/s", "analog", 1.0, "MetroPT-1/2 only; sim emits it too so the R93 flow rule is reproducible [rail_phm 4.2]"),
    ]
)

_BEARING = _specs(
    [
        ("T_box", "bearing", "degC", "analog", 1.0, "axle-box temperature, onboard continuous"),
        ("T_box_wayside", "bearing", "degC", "analog", 1.0, "wayside detector snapshot, NaN except at passes; +-5 K scan bias [rail_phm 4.3.4]"),
        ("vib_rms", "bearing", "m/s2", "analog", 1.0, "broadband RMS; healthy floor ~ v^2.0, defect term ~ s*v^1.2 [rail_phm 4.3.3]"),
        ("vib_kurt", "bearing", "-", "analog", 1.0, "kurtosis; RISES THEN COLLAPSES with severity, healthy 2.76-2.96 [rail_phm 4.3.3]"),
        ("vib_crest", "bearing", "-", "analog", 1.0, "crest factor; healthy intercept ~4.5, NOT 3 [rail_phm 4.3.3]"),
        ("vib_bpfo", "bearing", "m/s2", "analog", 1.0, "BPFO envelope band energy; our assumption B*s*(v/22)^2"),
    ]
)

_TRAIN = _specs(
    [
        ("speed", "train", "m/s", "analog", 1.0, "train speed, trapezoidal profile to 22 m/s"),
        ("load_frac", "train", "frac", "analog", 1.0, "passenger load 0..1 (0 = tare ~40 t, 1 = crush ~64 t)"),
        ("T_amb", "train", "degC", "analog", 1.0, "ambient; Singapore routine 24-33 C with a diurnal shape [rail_phm 0]"),
        ("in_service", "train", "bool", "digital", 1.0, "1 during revenue service 05:30-00:30, 0 stabled"),
    ]
)

SIGNAL_SPECS: Final[dict[tuple[str, str], SignalSpec]] = {
    (s.subsystem, s.name): s for s in (*_DOOR, *_PNEUMATIC, *_BEARING, *_TRAIN)
}

#: subsystem -> ordered tuple of signal names.
SIGNALS: Final[dict[str, tuple[str, ...]]] = {
    "door": tuple(s.name for s in _DOOR),
    "pneumatic": tuple(s.name for s in _PNEUMATIC),
    "bearing": tuple(s.name for s in _BEARING),
    "train": tuple(s.name for s in _TRAIN),
}

#: The MetroPT-3 channel names, verbatim and in file order.
METROPT3_SIGNALS: Final[tuple[str, ...]] = _METROPT3_NAMES

#: Train-context signals (live on component_id == "train", subsystem == "train").
CONTEXT_SIGNALS: Final[tuple[str, ...]] = SIGNALS["train"]

DIGITAL_SIGNALS: Final[frozenset[tuple[str, str]]] = frozenset(
    k for k, v in SIGNAL_SPECS.items() if v.kind == "digital"
)

# --------------------------------------------------------------------------------------
# Fault / event vocabularies
# --------------------------------------------------------------------------------------

#: Allowed ``fault_type`` per subsystem. "healthy" and "nff" (no fault found, 42.4 % of real
#: door defects [rail_phm 1.3]) are legal values everywhere ground truth is emitted.
FAULT_TYPES: Final[dict[str, tuple[str, ...]]] = {
    "door": (
        "healthy",
        "nff",
        "friction",
        "brush_wear",
        "backlash",
        "misalignment",
        "obstruction",
        "limit_switch",
        "dcu_dropout",
        "shock_wear",
    ),
    "pneumatic": (
        "healthy",
        "nff",
        "air_leak",
        "oil_leak",
        "compressor_wear",
        "dryer_valve_stuck",
        "clogged_filter",
        "valve_leak_inlet",
        "valve_leak_outlet",
    ),
    "bearing": (
        "healthy",
        "nff",
        "bearing_degradation",
        "hot_axle_box",
        "outer_race",
        "inner_race",
        "ball",
        "cage",
        "sensor_stuck",
        "sensor_offset",
    ),
    "train": ("healthy", "nff"),
}

ALL_FAULT_TYPES: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(ft for types in FAULT_TYPES.values() for ft in types)
)

#: Allowed ``event`` values in the event log.
EVENT_TYPES: Final[tuple[str, ...]] = (
    "obstruction",
    "reversal",
    "ls_timeout",
    "door_fault",
    "shock",
    "comp_load",
    "comp_unload",
    "comp_off",
    "purge",
    "tower_switch",
    "lps",
    "oil_low",
    "hot_box_alarm",
    "peer_delta_alarm",
    "wayside_pass",
    "station_fault",
    "functional_failure",
    "maintenance",
    "dropout",
)

#: Allowed ``shape`` values in the fault log / :class:`nebulax.sim.common.DegradationTrajectory`.
DEGRADATION_SHAPES: Final[tuple[str, ...]] = ("power", "step", "shock")

# --------------------------------------------------------------------------------------
# Component ids
# --------------------------------------------------------------------------------------

DOOR_COMPONENT_IDS: Final[tuple[str, ...]] = tuple(
    f"door_{side}{i}" for side in ("L", "R") for i in range(1, 5)
)
APU_COMPONENT_IDS: Final[tuple[str, ...]] = ("apu_1",)
AXLEBOX_COMPONENT_IDS: Final[tuple[str, ...]] = tuple(
    f"axlebox_{i}{side}" for i in range(1, 5) for side in ("L", "R")
)
TRAIN_COMPONENT_ID: Final[str] = "train"

COMPONENT_IDS: Final[dict[str, tuple[str, ...]]] = {
    "door": DOOR_COMPONENT_IDS,
    "pneumatic": APU_COMPONENT_IDS,
    "bearing": AXLEBOX_COMPONENT_IDS,
    "train": (TRAIN_COMPONENT_ID,),
}

MAX_CAR: Final[int] = 6  # DT-M-M-M-M-DT; car 0 == unit level


def component_ids(subsystem: str) -> tuple[str, ...]:
    """Component ids of ``subsystem``. Raises ValueError on an unknown subsystem."""
    if subsystem not in COMPONENT_IDS:
        raise ValueError(
            f"component_ids: unknown subsystem {subsystem!r}; expected one of {SUBSYSTEMS}"
        )
    return COMPONENT_IDS[subsystem]


def is_valid_component_id(subsystem: str, component_id: str) -> bool:
    """True if ``component_id`` is legal for ``subsystem`` (``"train"`` is legal everywhere)."""
    if component_id == TRAIN_COMPONENT_ID:
        return True
    return component_id in COMPONENT_IDS.get(subsystem, ())


_ALLOWED_COMPONENT_PAIRS: Final[frozenset[tuple[str, str]]] = frozenset(
    [(sub, cid) for sub, cids in COMPONENT_IDS.items() for cid in cids]
    + [(sub, TRAIN_COMPONENT_ID) for sub in SUBSYSTEMS]
)
_ALLOWED_SIGNAL_PAIRS: Final[frozenset[tuple[str, str]]] = frozenset(SIGNAL_SPECS.keys())

# --------------------------------------------------------------------------------------
# Empty frames + coercion
# --------------------------------------------------------------------------------------


def _empty(columns: Mapping[str, str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=("object" if d == "category" else d)) for c, d in columns.items()}).astype(
        {c: d for c, d in columns.items()}
    )


def empty_long() -> pd.DataFrame:
    """Empty, correctly typed long telemetry frame."""
    return _empty(LONG_COLUMNS)


def empty_features() -> pd.DataFrame:
    """Empty, correctly typed feature frame (key columns only)."""
    return _empty(FEATURE_KEY_DTYPES)


def empty_fault_log() -> pd.DataFrame:
    """Empty, correctly typed fault log."""
    return _empty(FAULT_LOG_COLUMNS)


def empty_events() -> pd.DataFrame:
    """Empty, correctly typed event log."""
    return _empty(EVENT_LOG_COLUMNS)


def empty_scores() -> pd.DataFrame:
    """Empty, correctly typed scores frame."""
    return _empty(SCORES_COLUMNS)


def _coerce(df: pd.DataFrame, columns: Mapping[str, str], *, what: str, reorder: bool = True) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{what}: expected a pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{what}: missing required column(s) {missing}; required = {list(columns)}")
    out = df.copy()
    for col, dtype in columns.items():
        if dtype == "object":
            continue
        if dtype == "category":
            out[col] = out[col].astype("string").astype("category")
        elif dtype.startswith("datetime64"):
            s = pd.to_datetime(out[col], utc=True, errors="raise")
            out[col] = s.astype(dtype)
        else:
            out[col] = out[col].astype(dtype)
    if reorder:
        extra = [c for c in out.columns if c not in columns]
        out = out[[*columns, *extra]]
    return out.reset_index(drop=True)


def coerce_long(df: pd.DataFrame) -> pd.DataFrame:
    """Cast a long telemetry frame to :data:`LONG_COLUMNS` dtypes and column order."""
    return _coerce(df, LONG_COLUMNS, what="coerce_long")


def coerce_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cast feature-table key columns (and any present label columns) to their dtypes.

    Feature columns themselves are left alone apart from ``float64 -> float32`` for plain
    numeric columns, which keeps a 30-day fleet small; list columns are untouched. ``meta_*``
    columns (:data:`METADATA_PREFIX`) are never touched at all - metadata may be any dtype, and
    an id or a counter must survive the round trip exactly as written.
    """
    out = _coerce(df, FEATURE_KEY_DTYPES, what="coerce_features")
    for col, dtype in LABEL_COLUMNS.items():
        if col in out.columns:
            if dtype == "category":
                out[col] = out[col].astype("string").astype("category")
            elif dtype == "bool":
                out[col] = out[col].fillna(False).astype("bool")
            else:
                out[col] = out[col].astype(dtype)
    for col in out.columns:
        if col in FEATURE_KEY_DTYPES or col in LABEL_COLUMNS or str(col).startswith(METADATA_PREFIX):
            continue
        if pd.api.types.is_float_dtype(out[col]) and out[col].dtype == np.float64:
            out[col] = out[col].astype("float32")
    return out


def coerce_fault_log(df: pd.DataFrame) -> pd.DataFrame:
    """Cast a fault log to :data:`FAULT_LOG_COLUMNS` dtypes."""
    return _coerce(df, FAULT_LOG_COLUMNS, what="coerce_fault_log")


def coerce_events(df: pd.DataFrame) -> pd.DataFrame:
    """Cast an event log to :data:`EVENT_LOG_COLUMNS` dtypes."""
    return _coerce(df, EVENT_LOG_COLUMNS, what="coerce_events")


def coerce_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Cast a scores frame to :data:`SCORES_COLUMNS` dtypes."""
    return _coerce(df, SCORES_COLUMNS, what="coerce_scores")


# --------------------------------------------------------------------------------------
# Feature / metadata split
# --------------------------------------------------------------------------------------


def feature_columns(df: pd.DataFrame | Iterable[str]) -> list[str]:
    """The columns of a feature table that are legitimate **model inputs**, in table order.

    A column is a feature unless it is a key column (:data:`FEATURE_KEY_COLUMNS`), a label
    (:data:`LABEL_COLUMNS`) or metadata - anything prefixed :data:`METADATA_PREFIX`
    (``meta_``). Metadata is the escape hatch for everything that describes the sample but
    would leak the target or the split: dataset class labels (``meta_class``), condition ids
    and severity stages (``meta_level``, ``meta_state``), latent ground-truth counters
    (``meta_shock_count``) and grouping ids (``meta_bearing_id``, ``meta_test_id``).

    The benchmark builds ``X`` from exactly this list; group ids for leave-one-out splits are
    read from ``meta_*`` or key columns instead. Accepts a DataFrame or any iterable of column
    names, so callers holding a bare column index do not have to build a frame.

    >>> feature_columns(["run_id", "i_peak", "meta_shock_count", "severity"])
    ['i_peak']
    """
    cols: Iterable[str] = df.columns if isinstance(df, pd.DataFrame) else df
    return [
        str(c)
        for c in cols
        if str(c) not in FEATURE_KEY_COLUMNS
        and str(c) not in LABEL_COLUMNS
        and not str(c).startswith(METADATA_PREFIX)
    ]


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


def _check_dtype(df: pd.DataFrame, col: str, dtype: str, what: str) -> None:
    actual = df[col].dtype
    if dtype == "category":
        if not isinstance(actual, pd.CategoricalDtype):
            raise ValueError(
                f"{what}: column {col!r} must be pandas 'category' dtype, got {actual}; "
                f"call nebulax.schema.coerce_* before validating"
            )
        return
    if dtype == "object":
        return
    if dtype.startswith("datetime64"):
        if not isinstance(actual, pd.DatetimeTZDtype):
            raise ValueError(
                f"{what}: column {col!r} must be timezone-aware {dtype}, got {actual}"
            )
        if str(actual.tz) not in ("UTC", "utc"):
            raise ValueError(f"{what}: column {col!r} must be UTC, got tz={actual.tz}")
        # Resolution is normalised to ms by the coerce_* helpers and on write; validation
        # accepts any pandas datetime unit because ordinary Timedelta arithmetic promotes
        # ms -> ns and failing that would be pure friction.
        return
    if str(actual) != dtype:
        raise ValueError(
            f"{what}: column {col!r} must be dtype {dtype}, got {actual}; "
            f"call nebulax.schema.coerce_* before validating"
        )


def _check_frame(
    df: pd.DataFrame,
    columns: Mapping[str, str],
    *,
    what: str,
    strict: bool,
    non_null: Iterable[str] = (),
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{what}: expected a pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{what}: missing required column(s) {missing}; required = {list(columns)}")
    if strict:
        extra = [c for c in df.columns if c not in columns]
        if extra:
            raise ValueError(f"{what}: unexpected column(s) {extra}; pass strict=False to allow extras")
    for col, dtype in columns.items():
        _check_dtype(df, col, dtype, what)
    for col in non_null:
        if df[col].isna().any():
            n = int(df[col].isna().sum())
            raise ValueError(f"{what}: column {col!r} has {n} null value(s); key columns must be complete")


def _cats(df: pd.DataFrame, col: str) -> list[str]:
    """Observed (used) category values of a categorical column."""
    s = df[col]
    if isinstance(s.dtype, pd.CategoricalDtype):
        used = s.cat.remove_unused_categories() if len(s) else s
        return [str(v) for v in used.cat.categories]
    return [str(v) for v in pd.unique(s.dropna())]


def validate_long(df: pd.DataFrame, *, strict: bool = False) -> None:
    """Validate a long telemetry frame. Raises ``ValueError`` naming the exact offence.

    Checks: columns and dtypes, non-null keys, known ``source``/``subsystem``, signal names
    legal for their subsystem, component ids legal for their subsystem, ``car`` in
    ``0..MAX_CAR``, and finite (or NaN) values - ``+-inf`` is never a legal reading.
    """
    what = "validate_long"
    _check_frame(
        df,
        LONG_COLUMNS,
        what=what,
        strict=strict,
        non_null=("timestamp", "source", "run_id", "train_id", "car", "subsystem", "component_id", "signal"),
    )
    if df.empty:
        return
    bad_source = sorted(set(_cats(df, "source")) - set(SOURCES))
    if bad_source:
        raise ValueError(f"{what}: unknown source value(s) {bad_source}; expected one of {list(SOURCES)}")
    bad_sub = sorted(set(_cats(df, "subsystem")) - set(SUBSYSTEMS))
    if bad_sub:
        raise ValueError(f"{what}: unknown subsystem value(s) {bad_sub}; expected one of {list(SUBSYSTEMS)}")

    pairs = pd.MultiIndex.from_arrays(
        [df["subsystem"].astype("string"), df["signal"].astype("string")]
    )
    bad_sig = sorted({tuple(p) for p in pairs.unique() if tuple(p) not in _ALLOWED_SIGNAL_PAIRS})
    if bad_sig:
        sub0 = bad_sig[0][0]
        raise ValueError(
            f"{what}: signal(s) not in the registry for their subsystem: {bad_sig[:8]}; "
            f"SIGNALS[{sub0!r}] = {list(SIGNALS.get(sub0, ()))}"
        )

    cpairs = pd.MultiIndex.from_arrays(
        [df["subsystem"].astype("string"), df["component_id"].astype("string")]
    )
    bad_cid = sorted({tuple(p) for p in cpairs.unique() if tuple(p) not in _ALLOWED_COMPONENT_PAIRS})
    if bad_cid:
        sub0 = bad_cid[0][0]
        raise ValueError(
            f"{what}: component_id(s) not legal for their subsystem: {bad_cid[:8]}; "
            f"COMPONENT_IDS[{sub0!r}] = {list(COMPONENT_IDS.get(sub0, ()))} (plus 'train')"
        )

    car = df["car"].to_numpy()
    if car.min() < 0 or car.max() > MAX_CAR:
        raise ValueError(
            f"{what}: car out of range [0, {MAX_CAR}] (0 = unit level): "
            f"observed min={int(car.min())} max={int(car.max())}"
        )
    vals = df["value"].to_numpy(dtype=np.float32, copy=False)
    if np.isinf(vals).any():
        n = int(np.isinf(vals).sum())
        raise ValueError(f"{what}: column 'value' contains {n} infinite value(s); use NaN for dropouts")


def validate_features(df: pd.DataFrame, *, require_labels: bool = False) -> None:
    """Validate a feature table: key columns, dtypes, ``t_start <= t_end``, label dtypes.

    Set ``require_labels=True`` for simulator output, where the fault log must already be
    joined on. Real-data adapters may legitimately have no labels.

    ``meta_*`` columns (:data:`METADATA_PREFIX`) are accepted with **any** dtype and never
    count as features: the table must still carry at least one real feature column, as
    reported by :func:`feature_columns`.
    """
    what = "validate_features"
    _check_frame(df, FEATURE_KEY_DTYPES, what=what, strict=False, non_null=FEATURE_KEY_COLUMNS)
    feature_cols = feature_columns(df)
    # An empty, correctly-typed frame is legal (adapters that leave feature extraction to
    # nebulax.features return empty_features()); a populated one must carry features. A
    # ``meta_*`` column does not count: it is provenance/grouping, never a model input, and it
    # may carry ANY dtype (string class names, ids, counters) - no dtype check applies to it.
    if not feature_cols and len(df):
        meta_cols = [c for c in df.columns if str(c).startswith(METADATA_PREFIX)]
        raise ValueError(
            f"{what}: no feature columns found; expected at least one column beyond "
            f"{list(FEATURE_KEY_COLUMNS)} + {list(LABEL_COLUMNS)} that is not "
            f"{METADATA_PREFIX}-prefixed (metadata present: {meta_cols})"
        )
    if require_labels:
        missing = [c for c in LABEL_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{what}: require_labels=True but label column(s) {missing} are absent")
    for col, dtype in LABEL_COLUMNS.items():
        if col in df.columns:
            _check_dtype(df, col, dtype, what)
    if df.empty:
        return
    bad_sub = sorted(set(_cats(df, "subsystem")) - set(SUBSYSTEMS))
    if bad_sub:
        raise ValueError(f"{what}: unknown subsystem value(s) {bad_sub}; expected one of {list(SUBSYSTEMS)}")
    for sub, cid in {(str(a), str(b)) for a, b in zip(df["subsystem"], df["component_id"])}:
        if (sub, cid) not in _ALLOWED_COMPONENT_PAIRS:
            raise ValueError(
                f"{what}: component_id {cid!r} is not legal for subsystem {sub!r}; "
                f"expected one of {list(COMPONENT_IDS.get(sub, ()))} (plus 'train')"
            )
    bad = df["t_end"] < df["t_start"]
    if bool(bad.any()):
        i = int(np.flatnonzero(bad.to_numpy())[0])
        raise ValueError(
            f"{what}: t_end < t_start on {int(bad.sum())} row(s), first at index {i} "
            f"({df['t_start'].iloc[i]} -> {df['t_end'].iloc[i]})"
        )
    if "severity" in df.columns:
        sev = df["severity"].to_numpy(dtype=np.float32, copy=False)
        finite = sev[np.isfinite(sev)]
        if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
            raise ValueError(
                f"{what}: severity must lie in [0, 1] (NaN allowed): observed "
                f"min={float(finite.min()):.4g} max={float(finite.max()):.4g}"
            )
    if "rul_s" in df.columns:
        rul = df["rul_s"].to_numpy(dtype=np.float32, copy=False)
        finite = rul[np.isfinite(rul)]
        if finite.size and finite.min() < 0.0:
            raise ValueError(f"{what}: rul_s must be >= 0 (NaN allowed): observed min={float(finite.min()):.4g}")


def validate_fault_log(df: pd.DataFrame, *, strict: bool = False) -> None:
    """Validate a fault log: dtypes, known fault types per subsystem, ordering of the
    onset/failure timestamps, ``gamma > 0``, known ``shape``, parseable ``params_json``."""
    what = "validate_fault_log"
    _check_frame(
        df,
        FAULT_LOG_COLUMNS,
        what=what,
        strict=strict,
        non_null=("run_id", "train_id", "car", "subsystem", "component_id", "fault_type", "t_onset"),
    )
    if df.empty:
        return
    for sub, ft in {(str(a), str(b)) for a, b in zip(df["subsystem"], df["fault_type"])}:
        if sub not in FAULT_TYPES:
            raise ValueError(f"{what}: unknown subsystem {sub!r}; expected one of {list(SUBSYSTEMS)}")
        if ft not in FAULT_TYPES[sub]:
            raise ValueError(
                f"{what}: fault_type {ft!r} is not legal for subsystem {sub!r}; "
                f"FAULT_TYPES[{sub!r}] = {list(FAULT_TYPES[sub])}"
            )
    for sub, cid in {(str(a), str(b)) for a, b in zip(df["subsystem"], df["component_id"])}:
        if (sub, cid) not in _ALLOWED_COMPONENT_PAIRS:
            raise ValueError(
                f"{what}: component_id {cid!r} is not legal for subsystem {sub!r}; "
                f"expected one of {list(COMPONENT_IDS.get(sub, ()))}"
            )
    bad_shape = sorted(set(_cats(df, "shape")) - set(DEGRADATION_SHAPES))
    if bad_shape:
        raise ValueError(f"{what}: unknown shape value(s) {bad_shape}; expected one of {list(DEGRADATION_SHAPES)}")
    order_bad = df["t_failure"].notna() & (df["t_failure"] < df["t_onset"])
    if bool(order_bad.any()):
        i = int(np.flatnonzero(order_bad.to_numpy())[0])
        raise ValueError(
            f"{what}: t_failure < t_onset on {int(order_bad.sum())} row(s), first at index {i} "
            f"({df['t_onset'].iloc[i]} -> {df['t_failure'].iloc[i]})"
        )
    ff_bad = df["t_functional_failure"].notna() & (df["t_functional_failure"] < df["t_onset"])
    if bool(ff_bad.any()):
        i = int(np.flatnonzero(ff_bad.to_numpy())[0])
        raise ValueError(f"{what}: t_functional_failure < t_onset on row {i}")
    gamma = df["gamma"].to_numpy(dtype=np.float32, copy=False)
    finite = gamma[np.isfinite(gamma)]
    if finite.size and finite.min() <= 0.0:
        raise ValueError(f"{what}: gamma must be > 0: observed min={float(finite.min()):.4g}")
    for i, raw in enumerate(df["params_json"].tolist()):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        if not isinstance(raw, str):
            raise ValueError(f"{what}: params_json on row {i} must be a JSON string, got {type(raw).__name__}")
        try:
            json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{what}: params_json on row {i} is not valid JSON: {exc}") from exc


def validate_events(df: pd.DataFrame, *, strict: bool = False) -> None:
    """Validate an event log: dtypes, known ``event`` values, parseable ``detail_json``."""
    what = "validate_events"
    _check_frame(
        df,
        EVENT_LOG_COLUMNS,
        what=what,
        strict=strict,
        non_null=("run_id", "timestamp", "train_id", "car", "subsystem", "component_id", "event"),
    )
    if df.empty:
        return
    bad = sorted(set(_cats(df, "event")) - set(EVENT_TYPES))
    if bad:
        raise ValueError(f"{what}: unknown event value(s) {bad}; EVENT_TYPES = {list(EVENT_TYPES)}")
    for i, raw in enumerate(df["detail_json"].tolist()):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        if not isinstance(raw, str):
            raise ValueError(f"{what}: detail_json on row {i} must be a JSON string, got {type(raw).__name__}")
        try:
            json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{what}: detail_json on row {i} is not valid JSON: {exc}") from exc


def validate_scores(df: pd.DataFrame, *, strict: bool = False) -> None:
    """Validate a scores frame: dtypes, finite scores, known subsystem/component pairs."""
    what = "validate_scores"
    _check_frame(
        df,
        SCORES_COLUMNS,
        what=what,
        strict=strict,
        non_null=("timestamp", "train_id", "car", "subsystem", "component_id", "model"),
    )
    if df.empty:
        return
    score = df["score"].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(score).all():
        raise ValueError(f"{what}: column 'score' must be finite; found {int((~np.isfinite(score)).sum())} NaN/inf")


# --------------------------------------------------------------------------------------
# long <-> wide
# --------------------------------------------------------------------------------------

_WIDE_INDEX: Final[tuple[str, ...]] = (
    "timestamp",
    "source",
    "run_id",
    "train_id",
    "car",
    "component_id",
)


def to_wide(
    df: pd.DataFrame,
    subsystem: str,
    *,
    signals: Sequence[str] | None = None,
    include_context: bool = False,
    validate: bool = False,
) -> pd.DataFrame:
    """Pivot long telemetry of one subsystem to wide (one column per signal).

    Parameters
    ----------
    df : long telemetry frame (may contain several subsystems).
    subsystem : one of :data:`SUBSYSTEMS`.
    signals : subset/order of signals to emit; default = all registered signals of the
        subsystem that are actually present.
    include_context : also attach the train-context signals (``speed, load_frac, T_amb,
        in_service``) with a backward ``merge_asof`` on ``(run_id, train_id)``, so a 100 Hz
        door stream can carry the 1 Hz context without resampling.
    validate : run :func:`validate_long` on the input first.

    Returns a frame with columns ``[timestamp, source, run_id, train_id, car, component_id,
    <signals...>]``, sorted by component then time.
    """
    if subsystem not in SUBSYSTEMS:
        raise ValueError(f"to_wide: unknown subsystem {subsystem!r}; expected one of {list(SUBSYSTEMS)}")
    if validate:
        validate_long(df)
    sub = df[df["subsystem"].astype("string") == subsystem]
    if sub.empty:
        cols = list(signals) if signals is not None else list(SIGNALS[subsystem])
        out = pd.DataFrame({c: pd.Series(dtype="float32") for c in cols})
        for c in reversed(_WIDE_INDEX):
            out.insert(0, c, pd.Series(dtype=LONG_COLUMNS[c]))
        return out
    idx = list(_WIDE_INDEX)
    dup = sub.duplicated(subset=[*idx, "signal"])
    if bool(dup.any()):
        i = int(np.flatnonzero(dup.to_numpy())[0])
        row = sub.iloc[i]
        raise ValueError(
            "to_wide: duplicate (timestamp, component_id, signal) rows cannot be pivoted; "
            f"first duplicate: timestamp={row['timestamp']}, component_id={row['component_id']}, "
            f"signal={row['signal']}"
        )
    wide = (
        sub.set_index([*idx, "signal"])["value"]
        .unstack("signal")
        .reset_index()
        .rename_axis(columns=None)
    )
    present = [s for s in SIGNALS[subsystem] if s in wide.columns]
    if signals is not None:
        unknown = [s for s in signals if s not in SIGNALS[subsystem]]
        if unknown:
            raise ValueError(
                f"to_wide: signal(s) {unknown} are not registered for subsystem {subsystem!r}; "
                f"SIGNALS[{subsystem!r}] = {list(SIGNALS[subsystem])}"
            )
        for s in signals:
            if s not in wide.columns:
                wide[s] = np.float32("nan")
        present = list(signals)
    wide = wide[[*idx, *present]]
    for s in present:
        wide[s] = wide[s].astype("float32")
    wide = wide.sort_values(["component_id", "timestamp"], kind="stable").reset_index(drop=True)

    if include_context:
        ctx = to_wide(df, "train")
        if not ctx.empty:
            ctx = ctx.drop(columns=["source", "car", "component_id"]).sort_values("timestamp", kind="stable")
            wide = pd.merge_asof(
                wide.sort_values("timestamp", kind="stable"),
                ctx,
                on="timestamp",
                by=["run_id", "train_id"],
                direction="backward",
            ).sort_values(["component_id", "timestamp"], kind="stable").reset_index(drop=True)
        else:
            for c in CONTEXT_SIGNALS:
                wide[c] = np.float32("nan")
    return wide


def to_long(
    wide: pd.DataFrame,
    subsystem: str,
    *,
    signals: Sequence[str] | None = None,
    source: str | None = None,
    run_id: str | None = None,
    train_id: str | None = None,
    car: int | None = None,
    component_id: str | None = None,
    dropna: bool = False,
) -> pd.DataFrame:
    """Inverse of :func:`to_wide`: melt a wide frame back to the long schema.

    Key columns missing from ``wide`` may be supplied as scalars (``source``, ``run_id``,
    ``train_id``, ``car``, ``component_id``). ``dropna=True`` drops NaN readings instead of
    storing them (default keeps them: NaN *is* the dropout signal).
    """
    if subsystem not in SUBSYSTEMS:
        raise ValueError(f"to_long: unknown subsystem {subsystem!r}; expected one of {list(SUBSYSTEMS)}")
    if "timestamp" not in wide.columns:
        raise ValueError("to_long: wide frame must carry a 'timestamp' column")
    scalars: dict[str, Any] = {
        "source": source,
        "run_id": run_id,
        "train_id": train_id,
        "car": car,
        "component_id": component_id,
    }
    out = wide.copy()
    for col, val in scalars.items():
        if col not in out.columns:
            if val is None:
                raise ValueError(
                    f"to_long: wide frame has no {col!r} column and no {col}= scalar was given"
                )
            out[col] = val
    known = [c for c in SIGNALS[subsystem] if c in out.columns]
    value_cols = list(signals) if signals is not None else known
    unknown = [c for c in value_cols if c not in SIGNALS[subsystem]]
    if unknown:
        raise ValueError(
            f"to_long: signal(s) {unknown} are not registered for subsystem {subsystem!r}; "
            f"SIGNALS[{subsystem!r}] = {list(SIGNALS[subsystem])}"
        )
    if not value_cols:
        raise ValueError(
            f"to_long: no signal columns found in the wide frame for subsystem {subsystem!r}; "
            f"expected some of {list(SIGNALS[subsystem])}"
        )
    id_cols = ["timestamp", "source", "run_id", "train_id", "car", "component_id"]
    long = out.melt(id_vars=id_cols, value_vars=value_cols, var_name="signal", value_name="value")
    long["subsystem"] = subsystem
    if dropna:
        long = long[long["value"].notna()]
    long = coerce_long(long)
    return long.sort_values(["component_id", "signal", "timestamp"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Dataset container + parquet IO
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class Dataset:
    """The five tables of one source/run, as returned by adapters and simulators.

    ``meta`` carries free-form provenance (source paths, sampling rates, citations, the
    simulator params). It is written next to the parquet files as ``meta.json``.
    """

    long: pd.DataFrame
    features: pd.DataFrame
    fault_log: pd.DataFrame
    events: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, require_labels: bool = False) -> None:
        """Validate all four tables; raises ``ValueError`` on the first offence."""
        validate_long(self.long)
        validate_features(self.features, require_labels=require_labels)
        validate_fault_log(self.fault_log)
        validate_events(self.events)

    def summary(self) -> dict[str, Any]:
        """Row counts, date range, signals present and fault-log rows (used by the CLI)."""
        ts = self.long["timestamp"]
        return {
            "n_long": int(len(self.long)),
            "n_features": int(len(self.features)),
            "n_fault_log": int(len(self.fault_log)),
            "n_events": int(len(self.events)),
            "t_min": (ts.min() if len(ts) else None),
            "t_max": (ts.max() if len(ts) else None),
            "sources": _cats(self.long, "source") if len(self.long) else [],
            "subsystems": _cats(self.long, "subsystem") if len(self.long) else [],
            "signals": _cats(self.long, "signal") if len(self.long) else [],
            "components": _cats(self.long, "component_id") if len(self.long) else [],
        }


_TABLE_FILES: Final[dict[str, str]] = {
    "long": "telemetry.parquet",
    "features": "features.parquet",
    "fault_log": "fault_log.parquet",
    "events": "events.parquet",
}


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)


def write_dataset(
    long: pd.DataFrame,
    features: pd.DataFrame | None,
    fault_log: pd.DataFrame | None,
    events: pd.DataFrame | None,
    out_dir: str | Path,
    source: str,
    run_id: str,
    *,
    meta: Mapping[str, Any] | None = None,
    validate: bool = True,
) -> Path:
    """Write one run to ``out_dir/source=<source>/run_id=<run_id>/`` as parquet+zstd.

    Files: ``telemetry.parquet``, ``features.parquet``, ``fault_log.parquet``,
    ``events.parquet``, ``meta.json``. ``None`` tables are written as correctly-typed empty
    frames so readers never special-case them. Returns the partition directory.
    """
    if source not in SOURCES:
        raise ValueError(f"write_dataset: unknown source {source!r}; expected one of {list(SOURCES)}")
    if not run_id or "/" in run_id or "=" in run_id:
        raise ValueError(f"write_dataset: run_id {run_id!r} must be a non-empty path-safe token")
    tables = {
        "long": coerce_long(long),
        "features": coerce_features(features) if features is not None else empty_features(),
        "fault_log": coerce_fault_log(fault_log) if fault_log is not None else empty_fault_log(),
        "events": coerce_events(events) if events is not None else empty_events(),
    }
    if validate:
        validate_long(tables["long"])
        if len(tables["features"]):
            validate_features(tables["features"])
        validate_fault_log(tables["fault_log"])
        validate_events(tables["events"])

    part = Path(out_dir) / f"source={source}" / f"run_id={run_id}"
    part.mkdir(parents=True, exist_ok=True)
    for key, fname in _TABLE_FILES.items():
        _write_parquet(tables[key], part / fname)
    info: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "nebulax_version": __version__,
        "source": source,
        "run_id": run_id,
        "written_at": pd.Timestamp.utcnow().isoformat(),
        "rows": {k: int(len(v)) for k, v in tables.items()},
        **(dict(meta) if meta else {}),
    }
    (part / "meta.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return part


def read_dataset(
    out_dir: str | Path,
    *,
    source: str | None = None,
    run_id: str | None = None,
) -> Dataset:
    """Read every partition under ``out_dir`` (optionally filtered) back into a :class:`Dataset`.

    ``out_dir`` may be the root written by :func:`write_dataset` or a single partition
    directory. ``meta`` gains a ``"partitions"`` list with one entry per ``meta.json`` read.
    """
    root = Path(out_dir)
    if not root.exists():
        raise ValueError(f"read_dataset: {root} does not exist")
    if (root / _TABLE_FILES["long"]).exists():
        parts = [root]
    else:
        pattern = f"source={source}" if source else "source=*"
        rpattern = f"run_id={run_id}" if run_id else "run_id=*"
        parts = sorted(p for p in root.glob(f"{pattern}/{rpattern}") if (p / _TABLE_FILES["long"]).exists())
    if not parts:
        raise ValueError(
            f"read_dataset: no partitions with {_TABLE_FILES['long']} found under {root} "
            f"(source={source!r}, run_id={run_id!r})"
        )
    frames: dict[str, list[pd.DataFrame]] = {k: [] for k in _TABLE_FILES}
    metas: list[dict[str, Any]] = []
    for part in parts:
        for key, fname in _TABLE_FILES.items():
            path = part / fname
            if path.exists():
                frames[key].append(pd.read_parquet(path, engine="pyarrow"))
        mpath = part / "meta.json"
        if mpath.exists():
            metas.append(json.loads(mpath.read_text(encoding="utf-8")))

    def _concat(key: str, empty: pd.DataFrame) -> pd.DataFrame:
        chunks = [f for f in frames[key] if len(f)]
        if not chunks:
            return empty
        return pd.concat(chunks, ignore_index=True)

    return Dataset(
        long=coerce_long(_concat("long", empty_long())),
        features=coerce_features(_concat("features", empty_features())),
        fault_log=coerce_fault_log(_concat("fault_log", empty_fault_log())),
        events=coerce_events(_concat("events", empty_events())),
        meta={"partitions": metas},
    )
