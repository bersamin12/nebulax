"""Tests for scripts/generate.py: the multiprocessing fleet-dataset generator.

No huge runs here - everything uses 1-2 simulated days and 1-3 trains so the whole file
stays well under the 60 s budget; the real 10-train/30-day generation is run separately
and reported on, not exercised by this suite.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nebulax import schema as S

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generate"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load_module()


# --------------------------------------------------------------------------------------
# deterministic seeding
# --------------------------------------------------------------------------------------


def test_stable_seed_is_deterministic_and_distinct(gen):
    a = gen._stable_seed(0, "door")
    b = gen._stable_seed(0, "door")
    c = gen._stable_seed(0, "pneumatic")
    d = gen._stable_seed(1, "door")
    assert a == b
    assert a != c
    assert a != d
    assert isinstance(a, int) and 0 <= a < 2**32


def test_train_seed_is_stable_across_python_processes(gen):
    """Regression guard for the builtin-hash trap: train_seed must not depend on
    PYTHONHASHSEED, since two worker processes reconstruct the same train's Service from it."""
    import os
    import subprocess

    env = dict(os.environ, PYTHONHASHSEED="42")
    out = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(SCRIPT_PATH.parent)!r}); "
         "import importlib.util as u; spec = u.spec_from_file_location('generate', "
         f"{str(SCRIPT_PATH)!r}); m = u.module_from_spec(spec); spec.loader.exec_module(m); "
         "print(m.train_seed(0, 'T01'))"],
        capture_output=True, text=True, env=env,
    )
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) == gen.train_seed(0, "T01")


def test_train_seed_gives_bit_identical_service_across_subsystems(gen):
    """The core 'shared Service' contract: two independent reconstructions for the same
    (seed, train_id) must be bit-identical regardless of which subsystem asked."""
    from nebulax.sim.common import generate_service

    seed = gen.train_seed(7, "T03")
    svc_a = generate_service(2, np.random.default_rng(seed), train_id="T03")
    svc_b = generate_service(2, np.random.default_rng(seed), train_id="T03")
    pd.testing.assert_frame_equal(svc_a.segments, svc_b.segments)
    assert svc_a.t0 == svc_b.t0


# --------------------------------------------------------------------------------------
# scenario building
# --------------------------------------------------------------------------------------


def test_build_scenarios_counts_and_car_constraints(gen):
    for sub, expected_per_train in gen.RUNS_PER_TRAIN.items():
        scenarios = gen.build_scenarios(sub, n_trains=4, days=1, seed=0, p_healthy=0.4, max_faults=1)
        assert len(scenarios) == 4 * expected_per_train
        assert {sc.subsystem for sc in scenarios} == {sub}
        assert {sc.train_id for sc in scenarios} <= {"T01", "T02", "T03", "T04"}
        for sc in scenarios:
            assert sc.car in gen.CARS_PER_SUBSYSTEM[sub]
            if sub == "pneumatic":
                assert sc.car == 0
            if not sc.healthy:
                assert len(sc.faults) >= 1


def test_build_scenarios_is_reproducible(gen):
    a = gen.build_scenarios("door", n_trains=2, days=1, seed=3, p_healthy=0.4, max_faults=1)
    b = gen.build_scenarios("door", n_trains=2, days=1, seed=3, p_healthy=0.4, max_faults=1)
    assert [sc.run_id for sc in a] == [sc.run_id for sc in b]
    assert [sc.healthy for sc in a] == [sc.healthy for sc in b]
    assert [sc.seed for sc in a] == [sc.seed for sc in b]


# --------------------------------------------------------------------------------------
# one worker job, end to end
# --------------------------------------------------------------------------------------


def test_run_one_writes_a_valid_partition(gen, tmp_path):
    scenarios = gen.build_scenarios("bearing", n_trains=1, days=1, seed=5, p_healthy=0.0, max_faults=1)
    job = {
        "scenario": scenarios[0],
        "subsystem": "bearing",
        "store_every": 50,
        "out_dir": str(tmp_path),
        "base_seed": 5,
    }
    result = gen._run_one(job)
    assert result["subsystem"] == "bearing"
    assert result["healthy"] is False
    assert result["n_long"] > 0 and result["n_features"] > 0
    assert result["n_fault_log"] == len(result["fault_log_rows"]) == len(scenarios[0].faults)
    part = tmp_path / result["path"]
    assert (part / "telemetry.parquet").exists()
    assert (part / "meta.json").exists()
    ds = S.read_dataset(part)
    ds.validate(require_labels=True)


def test_run_one_healthy_scenario_has_empty_fault_log(gen, tmp_path):
    scenarios = gen.build_scenarios("pneumatic", n_trains=1, days=1, seed=9, p_healthy=1.0, max_faults=1)
    job = {
        "scenario": scenarios[0],
        "subsystem": "pneumatic",
        "store_every": 50,
        "out_dir": str(tmp_path),
        "base_seed": 9,
    }
    result = gen._run_one(job)
    assert result["healthy"] is True
    assert result["fault_log_rows"] == []
    assert result["n_fault_log"] == 0


# --------------------------------------------------------------------------------------
# wall-clock projection / fallback (calibration mocked out - no real sims here)
# --------------------------------------------------------------------------------------


def test_projection_falls_back_when_over_budget(gen, monkeypatch):
    monkeypatch.setattr(gen, "_calibrate", lambda sub, seed: 1000.0)  # absurdly slow
    used, report = gen.project_and_maybe_fallback(
        list(gen.SUBSYSTEMS), n_trains=10, days=30, seed=0, p_healthy=0.4, max_faults=1, workers=4, budget_min=30.0
    )
    assert used == gen.FALLBACK_N_TRAINS
    assert report["fell_back"] is True
    assert report["used_n_trains"] == gen.FALLBACK_N_TRAINS
    assert report["projected_wall_seconds_used"] < report["projected_wall_seconds"]


def test_projection_keeps_requested_trains_when_within_budget(gen, monkeypatch):
    monkeypatch.setattr(gen, "_calibrate", lambda sub, seed: 0.01)  # comfortably fast
    used, report = gen.project_and_maybe_fallback(
        list(gen.SUBSYSTEMS), n_trains=10, days=30, seed=0, p_healthy=0.4, max_faults=1, workers=8, budget_min=30.0
    )
    assert used == 10
    assert report["fell_back"] is False


def test_projection_never_falls_back_below_the_requested_size(gen, monkeypatch):
    """--n-trains below the fallback floor must never be raised or otherwise altered."""
    monkeypatch.setattr(gen, "_calibrate", lambda sub, seed: 1000.0)
    used, report = gen.project_and_maybe_fallback(
        list(gen.SUBSYSTEMS), n_trains=3, days=30, seed=0, p_healthy=0.4, max_faults=1, workers=1, budget_min=0.001
    )
    assert used == 3
    assert report["fell_back"] is False


# --------------------------------------------------------------------------------------
# full CLI, small
# --------------------------------------------------------------------------------------


def test_main_end_to_end_small_fleet(gen, tmp_path):
    out = tmp_path / "sim"
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
    assert (out / "index.json").exists()
    assert (out / "fault_log.parquet").exists()

    index = json.loads((out / "index.json").read_text())
    assert index["args"]["n_trains_used"] == 2
    expected_runs = sum(2 * n for n in gen.RUNS_PER_TRAIN.values())
    assert len(index["runs"]) == expected_runs
    assert index["totals"]["n_runs"] == expected_runs
    assert all("fault_log_rows" not in r for r in index["runs"])  # popped before writing

    fault_log = pd.read_parquet(out / "fault_log.parquet")
    S.validate_fault_log(fault_log)

    ds = S.read_dataset(out)
    ds.validate(require_labels=True)
    assert set(ds.long["train_id"].astype(str)) <= {"T01", "T02"}


def test_main_single_subsystem_only_writes_that_subsystem(gen, tmp_path):
    out = tmp_path / "sim_door_only"
    rc = gen.main(
        [
            "--subsystem", "door",
            "--n-trains", "1",
            "--days", "1",
            "--out", str(out),
            "--workers", "1",
            "--no-calibrate",
        ]
    )
    assert rc == 0
    run_dirs = sorted(p.name for p in (out / "source=sim").iterdir())
    assert all(name.startswith("run_id=door_") for name in run_dirs)
    assert len(run_dirs) == gen.RUNS_PER_TRAIN["door"]


# --------------------------------------------------------------------------------------
# partial --subsystem regeneration must not clobber the other subsystems' rollups
# --------------------------------------------------------------------------------------


def test_partial_subsystem_rerun_preserves_other_subsystems(gen, tmp_path):
    """Regenerating just one subsystem in an existing --subsystem all dataset must leave
    the other subsystems' partitions, their index.json entries and their fault_log.parquet
    rows untouched, while still updating the regenerated subsystem's own entries."""
    out = tmp_path / "sim_partial"
    common_args = [
        "--n-trains", "2",
        "--days", "1",
        "--out", str(out),
        "--p-healthy", "0.5",
        "--workers", "2",
        "--store-every", "20",
        "--no-calibrate",
    ]
    assert gen.main(["--subsystem", "all", "--seed", "0", *common_args]) == 0

    index_before = json.loads((out / "index.json").read_text())
    fault_before = pd.read_parquet(out / "fault_log.parquet")
    door_paths_before = {r["path"] for r in index_before["runs"] if r["subsystem"] == "door"}
    bearing_paths_before = {r["path"] for r in index_before["runs"] if r["subsystem"] == "bearing"}
    door_partition_mtimes_before = {
        p: (out / p / "telemetry.parquet").stat().st_mtime_ns for p in door_paths_before
    }

    # a different seed so the regenerated pneumatic partitions provably change content
    assert gen.main(["--subsystem", "pneumatic", "--seed", "1", *common_args]) == 0

    index_after = json.loads((out / "index.json").read_text())
    fault_after = pd.read_parquet(out / "fault_log.parquet")

    expected_total = sum(2 * n for n in gen.RUNS_PER_TRAIN.values())
    assert len(index_after["runs"]) == expected_total
    assert index_after["totals"]["n_runs"] == expected_total
    assert "merged_from" in index_after
    assert index_after["merged_from"]["regenerated_subsystems"] == ["pneumatic"]

    # door/bearing entries and partitions are byte-for-byte untouched
    assert {r["path"] for r in index_after["runs"] if r["subsystem"] == "door"} == door_paths_before
    assert {r["path"] for r in index_after["runs"] if r["subsystem"] == "bearing"} == bearing_paths_before
    for p, mtime in door_partition_mtimes_before.items():
        assert (out / p / "telemetry.parquet").stat().st_mtime_ns == mtime

    S.validate_fault_log(fault_after)
    kept_before = fault_before[fault_before["subsystem"] != "pneumatic"]
    kept_after = fault_after[fault_after["subsystem"] != "pneumatic"]
    pd.testing.assert_frame_equal(
        kept_before.reset_index(drop=True), kept_after.reset_index(drop=True)
    )

    ds = S.read_dataset(out)
    ds.validate(require_labels=True)
