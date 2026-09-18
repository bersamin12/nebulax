"""Tests for nebulax.adapters.synthetic - the thin Dataset wrapper over scripts/generate.py's
``data/sim`` output.

Generates a tiny (2-train, 1-day) fleet with ``scripts/generate.py`` into a module-scoped tmp
dir so the whole file stays well under the pytest budget; the real 10-train/30-day dataset
under ``data/sim`` is generated and reported on separately, not exercised by this suite.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

from nebulax import schema as S
from nebulax.adapters import synthetic as SY

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate.py"


def _load_generate_module():
    spec = importlib.util.spec_from_file_location("generate_for_synthetic_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generate_for_synthetic_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load_generate_module()


@pytest.fixture(scope="module")
def fleet_dir(tmp_path_factory, gen) -> Path:
    """A tiny 2-train/1-day fleet: door/bearing each get RUNS_PER_TRAIN * 2 = 4 partitions,
    pneumatic gets 2 - enough to distinguish "sampled" (1/subsystem) from "full" selection."""
    out = tmp_path_factory.mktemp("synthetic_fleet") / "sim"
    rc = gen.main(
        [
            "--subsystem", "all",
            "--n-trains", "2",
            "--days", "1",
            "--seed", "0",
            "--out", str(out),
            "--p-healthy", "0.5",
            "--workers", "2",
            "--store-every", "20",
            "--no-calibrate",
        ]
    )
    assert rc == 0
    return out


# --------------------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------------------


def test_load_returns_a_valid_dataset(fleet_dir: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = SY.load(fleet_dir)
    assert isinstance(ds, S.Dataset)
    ds.validate(require_labels=True)


def test_pure_and_offline_does_not_modify_raw_dir(fleet_dir: Path):
    parts = sorted((fleet_dir / "source=sim").glob("run_id=*"))
    before = {p: {f.name: f.stat().st_mtime for f in p.iterdir()} for p in parts}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SY.load(fleet_dir, subsystem="door")
        SY.load(fleet_dir, include_long=False)
    after = {p: {f.name: f.stat().st_mtime for f in p.iterdir()} for p in parts}
    assert before == after


def test_missing_raw_dir_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        SY.load(tmp_path / "does_not_exist")


def test_empty_dir_raises_file_not_found(tmp_path: Path):
    empty = tmp_path / "empty_sim"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        SY.load(empty)


# --------------------------------------------------------------------------------------
# default (unrestricted) call: sampled, cheap, warns
# --------------------------------------------------------------------------------------


def test_unrestricted_call_samples_long_and_warns(fleet_dir: Path):
    with pytest.warns(UserWarning, match="sample"):
        ds = SY.load(fleet_dir)
    # all partitions' features/fault_log/events are present...
    assert ds.meta["n_partitions_selected"] == ds.meta["n_partitions_total"]
    assert len(ds.features) > 0
    # ...but long telemetry is sampled to <= 1 partition per subsystem, not every partition.
    assert ds.meta["long_sampled"] is True
    assert len(ds.meta["partitions_with_long"]) < ds.meta["n_partitions_selected"]
    # one partition per subsystem actually present
    present_subs = {r["subsystem"] for r in ds.meta["index_json"]["runs"]}
    assert len(ds.meta["partitions_with_long"]) == len(present_subs)
    assert set(ds.long["subsystem"].astype(str)) <= present_subs | {"train"}


def test_unrestricted_call_is_deterministic(fleet_dir: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = SY.load(fleet_dir)
        b = SY.load(fleet_dir)
    assert a.meta["partitions_with_long"] == b.meta["partitions_with_long"]


def test_include_long_false_is_cheap_and_empty(fleet_dir: Path):
    ds = SY.load(fleet_dir, include_long=False)
    assert len(ds.long) == 0
    assert "warning" not in ds.meta  # include_long=False never triggers the sampling warning
    assert ds.meta["long_sampled"] is False
    assert len(ds.features) > 0  # features/fault_log/events unaffected


# --------------------------------------------------------------------------------------
# explicit subsystem / run_ids: the full-fleet path, no sampling
# --------------------------------------------------------------------------------------


def test_subsystem_filter_reads_every_matching_partition_in_full(fleet_dir: Path, gen):
    ds = SY.load(fleet_dir, subsystem="door")
    assert ds.meta["long_sampled"] is False
    assert len(ds.meta["partitions_with_long"]) == gen.RUNS_PER_TRAIN["door"] * 2  # 2 trains
    assert set(ds.long["subsystem"].astype(str)) <= {"door", "train"}
    assert set(ds.features["subsystem"].astype(str)) == {"door"}


def test_unknown_subsystem_raises_value_error(fleet_dir: Path):
    with pytest.raises(ValueError, match="subsystem"):
        SY.load(fleet_dir, subsystem="nope")


def test_run_ids_filter_reads_exactly_those_partitions(fleet_dir: Path):
    index = json.loads((fleet_dir / "index.json").read_text())
    some_ids = sorted(r["run_id"] for r in index["runs"] if r["subsystem"] == "bearing")[:2]
    ds = SY.load(fleet_dir, run_ids=some_ids)
    assert ds.meta["long_sampled"] is False
    assert ds.meta["partitions"] == sorted(some_ids)
    assert ds.meta["partitions_with_long"] == sorted(some_ids)
    assert set(ds.long["run_id"].astype(str)) <= set(some_ids)


def test_unknown_run_id_raises_value_error(fleet_dir: Path):
    with pytest.raises(ValueError, match="run_id"):
        SY.load(fleet_dir, run_ids=["door_9999_does_not_exist"])


# --------------------------------------------------------------------------------------
# schema conformance across concatenation
# --------------------------------------------------------------------------------------


def test_categorical_dtypes_preserved_across_partitions(fleet_dir: Path):
    ds = SY.load(fleet_dir, subsystem="bearing")
    for col in ("source", "run_id", "train_id", "subsystem", "component_id", "signal"):
        assert isinstance(ds.long[col].dtype, pd.CategoricalDtype), col
    assert set(ds.long["run_id"].astype(str)) == set(ds.meta["partitions_with_long"])


def test_meta_carries_index_json_and_partition_list(fleet_dir: Path):
    ds = SY.load(fleet_dir, include_long=False)
    on_disk_index = json.loads((fleet_dir / "index.json").read_text())
    assert ds.meta["index_json"] == on_disk_index
    expected_ids = sorted(r["run_id"] for r in on_disk_index["runs"])
    assert ds.meta["partitions"] == expected_ids
