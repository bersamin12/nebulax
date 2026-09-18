"""Shared fixtures. Also guarantees the repo root is importable without any install.

``pyproject.toml`` sets ``pythonpath = ["."]`` for pytest; this belt-and-braces insert means
the suite also runs when pytest is invoked from elsewhere or with ``-p no:cacheprovider``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebulax import schema as S  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic generator - every test that draws randomness uses this."""
    return np.random.default_rng(20260918)


@pytest.fixture
def t0() -> pd.Timestamp:
    """Run start: the first evening of the hackathon, UTC."""
    return pd.Timestamp("2026-09-18T09:00:00Z")


@pytest.fixture
def door_long(t0: pd.Timestamp) -> pd.DataFrame:
    """A small schema-conformant door telemetry frame plus train-context rows."""
    n = 240
    ts = pd.date_range(t0, periods=n, freq="10ms", tz="UTC").as_unit("ms")
    frames = []
    for car, cid in ((3, "door_L1"), (3, "door_L2")):
        for sig, val in (
            ("pos", np.linspace(0.0, 0.725, n)),
            ("current", 2.0 + 0.1 * np.sin(np.arange(n))),
            ("ls_closed", (np.arange(n) < 5).astype(float)),
        ):
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": ts,
                        "source": "sim",
                        "run_id": "run0",
                        "train_id": "T01",
                        "car": np.int8(car),
                        "subsystem": "door",
                        "component_id": cid,
                        "signal": sig,
                        "value": val,
                    }
                )
            )
    ctx_ts = ts[::100]
    for sig, val in (("speed", np.array([0.0, 11.0, 22.0])), ("T_amb", np.array([29.0, 29.5, 30.0]))):
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": ctx_ts,
                    "source": "sim",
                    "run_id": "run0",
                    "train_id": "T01",
                    "car": np.int8(0),
                    "subsystem": "train",
                    "component_id": "train",
                    "signal": sig,
                    "value": val[: len(ctx_ts)],
                }
            )
        )
    return S.coerce_long(pd.concat(frames, ignore_index=True))


@pytest.fixture
def door_features(t0: pd.Timestamp) -> pd.DataFrame:
    """A small feature table with labels joined on."""
    n = 12
    starts = pd.date_range(t0, periods=n, freq="60s", tz="UTC").as_unit("ms")
    return S.coerce_features(
        pd.DataFrame(
            {
                "run_id": "run0",
                "source": "sim",
                "train_id": "T01",
                "car": np.int8(3),
                "subsystem": "door",
                "component_id": "door_L1",
                "cycle_id": np.arange(n, dtype=np.int64),
                "t_start": starts,
                "t_end": starts + pd.Timedelta(seconds=3),
                "closing_time": np.linspace(2.6, 3.4, n),
                "i_peak": np.linspace(5.0, 7.5, n),
                "fault_type": ["healthy"] * 6 + ["friction"] * 6,
                "severity": np.concatenate([np.zeros(6), np.linspace(0.1, 0.9, 6)]),
                "rul_s": np.linspace(8000, 0, n),
                "is_faulty": [False] * 6 + [True] * 6,
                "alarm_window_3d": [False] * 4 + [True] * 8,
            }
        )
    )


@pytest.fixture
def door_fault_log(t0: pd.Timestamp) -> pd.DataFrame:
    """A one-row fault log matching :func:`door_features`."""
    return S.coerce_fault_log(
        pd.DataFrame(
            {
                "run_id": ["run0"],
                "train_id": ["T01"],
                "car": np.array([3], dtype=np.int8),
                "subsystem": ["door"],
                "component_id": ["door_L1"],
                "fault_type": ["friction"],
                "t_onset": [t0 + pd.Timedelta(minutes=6)],
                "t_failure": [t0 + pd.Timedelta(minutes=12)],
                "t_functional_failure": [pd.NaT],
                "gamma": np.array([2.0], dtype=np.float32),
                "shape": ["power"],
                "params_json": ['{"F_c0": 40.0}'],
            }
        )
    )


@pytest.fixture
def door_events(t0: pd.Timestamp) -> pd.DataFrame:
    """A two-row event log."""
    return S.coerce_events(
        pd.DataFrame(
            {
                "run_id": ["run0", "run0"],
                "timestamp": [t0 + pd.Timedelta(minutes=7), t0 + pd.Timedelta(minutes=9)],
                "train_id": ["T01", "T01"],
                "car": np.array([3, 3], dtype=np.int8),
                "subsystem": ["door", "door"],
                "component_id": ["door_L1", "door_L1"],
                "event": ["obstruction", "reversal"],
                "detail_json": ['{"force_N": 150}', "{}"],
            }
        )
    )
