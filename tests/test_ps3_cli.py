"""Tests for the PS3 command line: ``scripts/ps3_predict.py``, root ``predict.py`` and
``scripts/ps3_submission.py``.

Like ``tests/test_api_ps3.py``, these run against a **dummy task** registered in
``nebulax.ps3.common.TASKS`` (the four real modules are written in parallel and may not exist
yet) and against the tiny fixtures in ``tests/fixtures/ps3/``. What is under test is the CLI
contract: task inference, the input ordering, the CSV bytes, the validation gate, the exit codes
and the packed archive - never a model.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from nebulax.api import ps3 as api_ps3
from nebulax.ps3 import common
from nebulax.ps3.submission import validate_zip
from test_api_ps3 import ACV_XLSX, DOOR_CSV, RAIL_CSV, SHM_CSV, DummyTask  # one definition, two files

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod  # dataclasses need the module in sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cli():
    return _load(REPO_ROOT / "scripts" / "ps3_predict.py", "ps3_predict_cli")


@pytest.fixture
def root_cli():
    return _load(REPO_ROOT / "predict.py", "ps3_predict_root")


@pytest.fixture
def submission_cli():
    return _load(REPO_ROOT / "scripts" / "ps3_submission.py", "ps3_submission_cli")


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEBULAX_PS3_MODELS", str(tmp_path / "models_ps3"))
    monkeypatch.setenv("NEBULAX_PS3_RESULTS", str(tmp_path / "results_ps3"))


@pytest.fixture
def task_factory(monkeypatch: pytest.MonkeyPatch):
    def _register(name: str) -> DummyTask:
        task = DummyTask(name)
        monkeypatch.setitem(common.TASKS, name, task)
        return task

    return _register


@pytest.fixture
def test_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A miniature ``02_Datasets`` tree: the organisers' folder names, the fixtures as inputs."""
    root = tmp_path / "datasets"
    (root / "Door").mkdir(parents=True)
    shutil.copyfile(DOOR_CSV, root / "Door" / "Test.csv")
    (root / "Rail_Corrugation" / "Test").mkdir(parents=True)
    for n in (2, 10):
        shutil.copyfile(RAIL_CSV, root / "Rail_Corrugation" / "Test" / f"Test{n}.csv")
    (root / "SHM" / "Test").mkdir(parents=True)
    shutil.copyfile(SHM_CSV, root / "SHM" / "Test" / "test01.csv")
    (root / "ACV" / "Test").mkdir(parents=True)
    shutil.copyfile(ACV_XLSX, root / "ACV" / "Test" / "acv_test_case.xlsx")
    monkeypatch.setenv("NEBULAX_PS3_DATA", str(root))
    return root


# --------------------------------------------------------------------------------------
# Task inference (the root predict.py interface)
# --------------------------------------------------------------------------------------


def test_infer_task_from_each_fixture(tmp_path: Path) -> None:
    assert api_ps3.infer_task(RAIL_CSV) == "rail"  # 129 columns
    assert api_ps3.infer_task(SHM_CSV) == "shm"  # headerless, single column
    assert api_ps3.infer_task(DOOR_CSV) == "door"  # Datetime,Motor current(mA),...
    assert api_ps3.infer_task(ACV_XLSX) == "acv"  # .xlsx


def test_infer_task_from_a_directory(test_root: Path) -> None:
    assert api_ps3.infer_task(test_root / "Rail_Corrugation" / "Test") == "rail"
    assert api_ps3.infer_task(test_root / "SHM" / "Test") == "shm"
    assert api_ps3.infer_task(test_root / "ACV" / "Test") == "acv"
    assert api_ps3.infer_task(test_root / "Door") == "door"


def test_infer_task_refuses_an_unknown_input(tmp_path: Path) -> None:
    junk = tmp_path / "notes.csv"
    junk.write_text("alpha,beta\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pass --task"):
        api_ps3.infer_task(junk)
    png = tmp_path / "photo.png"
    png.write_bytes(b"\x89PNG")
    with pytest.raises(ValueError, match="PS3 inputs are"):
        api_ps3.infer_task(png)
    with pytest.raises(FileNotFoundError):
        api_ps3.infer_task(tmp_path / "nowhere")


# --------------------------------------------------------------------------------------
# scripts/ps3_predict.py
# --------------------------------------------------------------------------------------


def test_predict_cli_writes_the_organiser_csv(cli, task_factory, test_root: Path, tmp_path: Path) -> None:
    task = task_factory("rail")
    out = tmp_path / "rail_predictions.csv"
    assert cli.run("rail", test_root / "Rail_Corrugation" / "Test", out, expect_ids=True, quiet=True) == 0
    assert out.read_text(encoding="utf-8") == "file_id,prediction\nTest2.csv,Normal\nTest10.csv,Side I\n"
    assert task.calls == ["Test2.csv", "Test10.csv"]  # natural order, one file at a time


def test_predict_cli_on_every_subsystem(cli, task_factory, test_root: Path, tmp_path: Path) -> None:
    for name, target, head in (
        ("door", test_root / "Door" / "Test.csv", "start_time,end_time,prediction"),
        ("acv", test_root / "ACV" / "Test", "file_id,ranked_cars"),
        ("shm", test_root / "SHM" / "Test", "file_id,prediction"),
    ):
        task_factory(name)
        out = tmp_path / f"{name}.csv"
        assert cli.run(name, target, out, quiet=True) == 0, name
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0] == head
    assert (tmp_path / "acv.csv").read_text(encoding="utf-8").splitlines()[1] == (
        "acv_test_case.xlsx,01|02|03|04|05|06|07|08"
    )
    assert (tmp_path / "shm.csv").read_text(encoding="utf-8").splitlines()[1] == "test01.csv,0.1234"


def test_predict_cli_infers_the_task_when_not_given(cli, task_factory, test_root: Path, tmp_path: Path) -> None:
    task_factory("shm")
    out = tmp_path / "shm_predictions.csv"
    assert cli.run(None, test_root / "SHM" / "Test", out, quiet=True) == 0
    assert out.read_text(encoding="utf-8").startswith("file_id,prediction\n")


def test_predict_cli_exit_1_on_a_missing_input(cli, tmp_path: Path, capsys) -> None:
    assert cli.run("rail", tmp_path / "nope", tmp_path / "out.csv") == 1
    assert "no such input" in capsys.readouterr().err


def test_predict_cli_exit_1_when_the_csv_does_not_validate(
    cli, task_factory, tmp_path: Path, capsys
) -> None:
    task = task_factory("rail")
    task.predict = lambda feats, model=None: common.PredictionResult(  # type: ignore[method-assign]
        task="rail", file_id=feats.name, rows=[{"file_id": feats.name, "prediction": "Bogus"}]
    )
    out = tmp_path / "rail_predictions.csv"
    assert cli.run("rail", RAIL_CSV, out, quiet=True) == 1
    assert "not in ['Normal', 'Side I', 'Side II']" in capsys.readouterr().err
    assert out.exists()  # written, then rejected - the operator can look at it


def test_predict_cli_exit_2_when_the_task_is_not_available(cli, tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(api_ps3, "get_task", lambda name: (_ for _ in ()).throw(KeyError("not written yet")))
    assert cli.run("rail", RAIL_CSV, tmp_path / "out.csv") == 2
    assert "ps3_train.py --task rail" in capsys.readouterr().err


def test_predict_cli_help_runs_standalone() -> None:
    for script in ("scripts/ps3_predict.py", "predict.py"):
        proc = subprocess.run(
            [PYTHON, str(REPO_ROOT / script), "--help"], capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120
        )
        assert proc.returncode == 0, proc.stderr
        assert "--input" in proc.stdout and "--output" in proc.stdout


# --------------------------------------------------------------------------------------
# predict.py (the info kits' two-flag interface)
# --------------------------------------------------------------------------------------


def test_root_predict_infers_the_subsystem(root_cli, task_factory, test_root: Path, tmp_path: Path) -> None:
    task_factory("door")
    out = tmp_path / "door_predictions.csv"
    assert root_cli.main(["--input", str(test_root / "Door" / "Test.csv"), "--output", str(out), "--quiet"]) == 0
    assert out.read_text(encoding="utf-8").splitlines()[0] == "start_time,end_time,prediction"


def test_root_predict_task_flag_overrides_inference(root_cli, task_factory, tmp_path: Path) -> None:
    task_factory("rail")
    out = tmp_path / "rail_predictions.csv"
    assert root_cli.main(["-i", str(RAIL_CSV), "-o", str(out), "--task", "rail", "-q"]) == 0
    assert out.read_text(encoding="utf-8").splitlines()[1].startswith("train1_slice.csv,")


# --------------------------------------------------------------------------------------
# scripts/ps3_submission.py
# --------------------------------------------------------------------------------------


def test_submission_packs_every_ready_subsystem(submission_cli, task_factory, test_root: Path, tmp_path: Path) -> None:
    for name in ("door", "acv", "rail", "shm"):
        task_factory(name)
    out_dir = tmp_path / "submission" / "nebulax"
    outcomes, zip_path = submission_cli.build(["door", "acv", "rail", "shm"], out_dir, quiet=True)
    assert [o.status for o in outcomes] == ["ok"] * 4, [(o.task, o.status, o.reason, o.errors) for o in outcomes]
    assert zip_path == out_dir / "predictions.zip"
    with zipfile.ZipFile(zip_path) as zf:
        assert sorted(zf.namelist()) == [
            "acv_predictions.csv",
            "door_predictions.csv",
            "rail_predictions.csv",
            "shm_predictions.csv",
        ]
    assert validate_zip(zip_path).ok
    # deterministic: the same predictions pack to the same bytes
    first = zip_path.read_bytes()
    submission_cli.build(["door", "acv", "rail", "shm"], out_dir, quiet=True)
    assert zip_path.read_bytes() == first
    # and the CSVs sit next to the zip, byte-identical to their zip members
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.read("rail_predictions.csv") == (out_dir / "rail_predictions.csv").read_bytes()


def test_submission_tag_names_the_zip(submission_cli, task_factory, test_root: Path, tmp_path: Path) -> None:
    task_factory("shm")
    out_dir = tmp_path / "sub"
    outcomes, zip_path = submission_cli.build(["shm"], out_dir, tag="baseline", quiet=True)
    assert zip_path.name == "predictions_baseline.zip" and outcomes[0].status == "ok"
    assert validate_zip(zip_path, {"shm": ["test01.csv"]}).ok


def test_submission_skips_a_task_that_is_not_ready(
    submission_cli, task_factory, test_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_factory("rail")  # rail is ready; shm has no module yet
    real_get_task = api_ps3.get_task

    def get_task(name: str):
        if name == "shm":
            raise KeyError(f"PS3 task {name!r} is not registered and nebulax.ps3.{name} could not be imported")
        return real_get_task(name)

    monkeypatch.setattr(api_ps3, "get_task", get_task)
    out_dir = tmp_path / "sub"
    outcomes, zip_path = submission_cli.build(["rail", "shm"], out_dir, quiet=True)
    by_task = {o.task: o for o in outcomes}
    assert by_task["rail"].status == "ok"
    assert by_task["shm"].status == "skipped" and "not available yet" in by_task["shm"].reason
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist() == ["rail_predictions.csv"]
    assert submission_cli.main(["--tasks", "rail,shm", "--out-dir", str(out_dir), "--quiet"]) == 0


def test_submission_exit_2_when_nothing_can_be_attempted(submission_cli, test_root: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(api_ps3, "get_task", lambda name: (_ for _ in ()).throw(KeyError("not written yet")))
    assert submission_cli.main(["--tasks", "rail", "--out-dir", str(tmp_path / "sub"), "--quiet"]) == 2


def test_submission_exit_1_when_a_csv_is_invalid(submission_cli, task_factory, test_root: Path, tmp_path: Path) -> None:
    task = task_factory("rail")
    task.predict = lambda feats, model=None: common.PredictionResult(  # type: ignore[method-assign]
        task="rail", file_id=feats.name, rows=[{"file_id": feats.name, "prediction": "Bogus"}]
    )
    out_dir = tmp_path / "sub"
    outcomes, zip_path = submission_cli.build(["rail"], out_dir, quiet=True)
    assert outcomes[0].status == "invalid" and zip_path is None
    assert submission_cli.main(["--tasks", "rail", "--out-dir", str(out_dir), "--quiet"]) == 1
