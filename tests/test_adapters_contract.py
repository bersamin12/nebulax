"""Contract tests for nebulax.adapters: lazy resolution and the validate CLI."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from nebulax import adapters as A
from nebulax.adapters import validate as V
from nebulax import schema as S


def _fake_dataset(t0: pd.Timestamp) -> S.Dataset:
    n = 120
    ts = pd.date_range(t0, periods=n, freq="1s", tz="UTC").as_unit("ms")
    frames = []
    for sig, val in (
        ("Reservoirs", np.linspace(8.2, 10.2, n)),
        ("Motor_current", np.where(np.arange(n) % 2, 7.0, 4.0)),
        ("DV_eletric", (np.arange(n) % 2).astype(float)),
    ):
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": ts, "source": "metropt3", "run_id": "metropt3",
                    "train_id": "porto_apu", "car": np.int8(0), "subsystem": "pneumatic",
                    "component_id": "apu_1", "signal": sig, "value": val,
                }
            )
        )
    long = S.coerce_long(pd.concat(frames, ignore_index=True))
    fault_log = S.coerce_fault_log(
        pd.DataFrame(
            {
                "run_id": ["metropt3"], "train_id": ["porto_apu"], "car": np.array([0], dtype=np.int8),
                "subsystem": ["pneumatic"], "component_id": ["apu_1"], "fault_type": ["air_leak"],
                "t_onset": [t0], "t_failure": [t0 + pd.Timedelta(minutes=1)],
                "t_functional_failure": [pd.NaT], "gamma": np.array([2.0], dtype=np.float32),
                "shape": ["power"], "params_json": ["{}"],
            }
        )
    )
    return S.Dataset(
        long=long,
        features=S.empty_features(),
        fault_log=fault_log,
        events=S.empty_events(),
        meta={"doi": "10.24432/C5VW3R", "licence": "CC BY 4.0"},
    )


@pytest.fixture
def fake_adapter(t0, monkeypatch):
    """Install a throwaway nebulax.adapters.metropt3 module for the duration of a test."""
    module = types.ModuleType("nebulax.adapters.metropt3")
    module.load = lambda raw_dir: _fake_dataset(t0)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nebulax.adapters.metropt3", module)
    return module


def test_contract_tables_are_documented():
    assert set(A.ADAPTER_MODULES) == set(A.DEFAULT_RAW_DIRS) == set(A.ADAPTER_SUBSYSTEM)
    assert A.ADAPTER_SUBSYSTEM["metropt3"] == "pneumatic"
    assert A.ADAPTER_SUBSYSTEM["cranfield"] == "door"
    assert A.ADAPTER_SUBSYSTEM["ottawa"] == "bearing"
    assert A.DEFAULT_RAW_DIRS["metropt3"] == "data/raw/metropt3"
    assert "load" in A.__doc__ and "Dataset" in A.__doc__


def test_resolve_adapter_unknown_source():
    with pytest.raises(ValueError, match="unknown source"):
        A.resolve_adapter("kaggle")


def test_resolve_adapter_missing_module_names_what_it_tried(monkeypatch):
    # nebulax.adapters.cranfield now exists (this repo implements it), so force a genuinely
    # missing module name to keep exercising resolve_adapter's not-found error message.
    monkeypatch.setitem(A.ADAPTER_MODULES, "cranfield", ("nebulax.adapters._missing_for_test",))
    with pytest.raises(ModuleNotFoundError, match=r"tried \['nebulax.adapters._missing_for_test'\]"):
        A.resolve_adapter("cranfield")


def test_resolve_adapter_rejects_module_without_load(monkeypatch):
    module = types.ModuleType("nebulax.adapters.ottawa")
    monkeypatch.setitem(sys.modules, "nebulax.adapters.ottawa", module)
    with pytest.raises(AttributeError, match="exposes no load"):
        A.resolve_adapter("ottawa")


def test_load_source_uses_the_adapter(fake_adapter, tmp_path):
    ds = A.load_source("metropt3", tmp_path)
    assert isinstance(ds, S.Dataset)
    ds.validate()
    assert ds.meta["licence"] == "CC BY 4.0"


def test_validate_cli_reports_and_passes(fake_adapter, tmp_path, capsys):
    rc = V.main(["--source", "metropt3", "--raw", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK: metropt3 passes the nebulax schema." in out
    assert "telemetry rows    : 360" in out
    assert "fault-log rows    : 1" in out
    assert "date range (UTC)" in out
    assert "DV_eletric" in out  # signals present, verbatim MetroPT-3 name
    assert "air_leak" in out


def test_validate_cli_missing_raw_dir(fake_adapter, tmp_path, capsys):
    rc = V.main(["--source", "metropt3", "--raw", str(tmp_path / "nope")])
    assert rc == 1
    assert "does not exist" in capsys.readouterr().err


def test_validate_cli_reports_schema_violations(t0, tmp_path, monkeypatch, capsys):
    bad = _fake_dataset(t0)
    bad.long = bad.long.copy()
    bad.long["value"] = bad.long["value"].astype("float64")
    module = types.ModuleType("nebulax.adapters.metropt3")
    module.load = lambda raw_dir: bad  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nebulax.adapters.metropt3", module)
    rc = V.main(["--source", "metropt3", "--raw", str(tmp_path), "--quiet"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "schema violation" in err and "[long]" in err


def test_validate_cli_rejects_non_dataset(tmp_path, monkeypatch, capsys):
    module = types.ModuleType("nebulax.adapters.ottawa")
    module.load = lambda raw_dir: pd.DataFrame()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nebulax.adapters.ottawa", module)
    rc = V.main(["--source", "ottawa", "--raw", str(tmp_path)])
    assert rc == 1
    assert "expected nebulax.schema.Dataset" in capsys.readouterr().err


def test_validate_cli_missing_adapter(tmp_path, capsys, monkeypatch):
    # Same reasoning as test_resolve_adapter_missing_module_names_what_it_tried above: force a
    # genuinely missing module now that nebulax.adapters.cranfield is implemented.
    monkeypatch.setitem(A.ADAPTER_MODULES, "cranfield", ("nebulax.adapters._missing_for_test",))
    rc = V.main(["--source", "cranfield", "--raw", str(tmp_path)])
    assert rc == 1
    assert "no adapter module" in capsys.readouterr().err
