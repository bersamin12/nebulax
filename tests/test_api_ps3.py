"""Tests for the PS3 predict routes (``nebulax/api/ps3.py``) - the app's upload page contract.

The four task modules (``nebulax/ps3/{door,acv,rail,shm}.py``) are written in parallel to the
protocol in ``docs/ps3_contract.md`` and may not exist yet, so every test here registers its own
**dummy task** in ``nebulax.ps3.common.TASKS`` (which is exactly where ``get_task`` looks first).
That also keeps the suite fast and deterministic: the routes, the session accumulation, the CSV
bytes and the 4xx/503 paths are what is under test, not anybody's model.

The uploads root, the results dir and the model dir are all pointed at ``tmp_path`` - no test
touches the real ``data/ps3_cache``, ``results/ps3`` or ``models/ps3``.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nebulax.api import ps3 as api_ps3
from nebulax.api.main import create_app
from nebulax.api.state import Settings
from nebulax.ps3 import common
from nebulax.ps3.common import BaseTask, PredictionResult, Trace, Viewport, acv_car_ids

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ps3"
RAIL_CSV = FIXTURES / "rail" / "train1_slice.csv"
SHM_CSV = FIXTURES / "shm" / "train01_slice.csv"
DOOR_CSV = FIXTURES / "door" / "train_slice.csv"
ACV_XLSX = FIXTURES / "acv" / "case01_slice.xlsx"


# --------------------------------------------------------------------------------------
# Dummy tasks: the protocol, nothing else
# --------------------------------------------------------------------------------------


class DummyTask(BaseTask):
    """A ``Task`` stand-in that predicts from the file name (and, for ACV, from the headers)."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__()
        self.calls: list[str] = []

    def load(self, path: Path | str) -> Path:
        p = Path(path)
        self.calls.append(p.name)
        return p

    def featurise(self, raw: Path) -> Path:
        return raw

    def predict(self, feats: Path, model: Any = None) -> PredictionResult:
        p = feats
        trace = Trace(x=[2.0, 2.5, 3.0], y=[0.1, 0.9, 0.2], marks=[{"x": 2.5, "label": "6.3 cm peak", "kind": "peak"}], label="wavelength PSD")
        viewport = Viewport(car=3, side="I", health="crit", component="axlebox_c3_p1")
        if self.name == "rail":
            rows = [{"file_id": p.name, "prediction": "Side I" if "1" in p.stem else "Normal"}]
        elif self.name == "shm":
            rows = [{"file_id": p.name, "prediction": 0.1234}]
        elif self.name == "acv":
            rows = [{"file_id": p.name, "ranked_cars": "|".join(acv_car_ids(p))}]
        else:  # door: one row per predicted segment of the single stream, no file_id
            rows = [
                {"start_time": "2023-7-5-0-0-0-0", "end_time": "2023-7-5-0-0-4-500", "prediction": "Normal"},
                {"start_time": "2023-7-5-0-0-30-0", "end_time": "2023-7-5-0-0-33-800", "prediction": "Abnormal resistance"},
            ]
        return PredictionResult(
            task=self.name,
            file_id="" if self.name == "door" else p.name,
            rows=rows,
            numbers={"side_i_score": 0.81, "speed_kmh": 52.0},
            trace=trace,
            viewport=viewport,
            extras={"warnings": []},
        )


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every path the router writes to or reads from lives under ``tmp_path``."""
    monkeypatch.setenv("NEBULAX_PS3_UPLOADS", str(tmp_path / "uploads"))
    monkeypatch.setenv("NEBULAX_PS3_RESULTS", str(tmp_path / "results_ps3"))
    monkeypatch.setenv("NEBULAX_PS3_MODELS", str(tmp_path / "models_ps3"))
    api_ps3.reset_sessions()
    yield
    api_ps3.reset_sessions()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(api_ps3.router)
    return TestClient(app)


@pytest.fixture
def task_factory(monkeypatch: pytest.MonkeyPatch):
    def _register(name: str) -> DummyTask:
        task = DummyTask(name)
        monkeypatch.setitem(common.TASKS, name, task)
        return task

    return _register


def _upload(path: Path, field: str = "files") -> tuple[str, tuple[str, bytes, str]]:
    return (field, (path.name, path.read_bytes(), "application/octet-stream"))


def _cli():
    """``scripts/ps3_predict.py`` imported by path (``scripts/`` is not a package)."""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("ps3_predict_cli", root / "scripts" / "ps3_predict.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod  # dataclasses need the module in sys.modules
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------
# GET /api/ps3/tasks
# --------------------------------------------------------------------------------------


def test_tasks_lists_all_four_even_with_no_modules(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_ps3, "get_task", lambda name: (_ for _ in ()).throw(KeyError(f"{name} missing")))
    body = client.get("/api/ps3/tasks").json()
    assert [t["name"] for t in body] == ["door", "acv", "rail", "shm"]
    rail = next(t for t in body if t["name"] == "rail")
    assert rail["label"] == "Rail Corrugation"
    assert rail["accepts"] == [".csv"]
    assert rail["output_filename"] == "rail_predictions.csv"
    assert rail["available"] is False and rail["model_loaded"] is False and rail["cv"] is None
    assert "rail" in rail["detail"]
    acv = next(t for t in body if t["name"] == "acv")
    assert acv["accepts"] == [".xlsx", ".xls"]
    assert rail["max_file_bytes"] == api_ps3.MAX_UPLOAD_BYTES == 64 * 1024 * 1024
    assert rail["max_files_per_request"] == api_ps3.MAX_BATCH_FILES


def test_tasks_reports_cv_summary_and_availability(client: TestClient, task_factory, tmp_path: Path) -> None:
    results = tmp_path / "results_ps3"
    results.mkdir(parents=True, exist_ok=True)
    (results / "rail_cv.json").write_text(
        json.dumps(
            {
                "scheme": "stratified 5-fold by file x 3 seeds",
                "metric": "macro_f1",
                "score": 0.72,
                "seeds": [0, 1, 2],
                "folds": [{"fold": i, "macro_f1": 0.7} for i in range(5)],
                "headline": {"macro_f1": 0.72, "folds": [{"fold": i} for i in range(5)]},
                "summary": {"macro_f1": 0.72, "n_files": 204},
            }
        ),
        encoding="utf-8",
    )
    task_factory("rail")
    rail = next(t for t in client.get("/api/ps3/tasks").json() if t["name"] == "rail")
    assert rail["available"] is True and rail["detail"] is None
    assert rail["cv"] == {
        "scheme": "stratified 5-fold by file x 3 seeds",
        "metric": "macro_f1",
        "score": 0.72,
        "seeds": [0, 1, 2],
        "summary": {"macro_f1": 0.72, "n_files": 204},
        "headline": {"macro_f1": 0.72},  # scalars only: the per-fold array is dropped
    }
    assert "folds" not in rail["cv"]


def test_shm_task_summary_uses_honest_nested_ladder_headline(
    client: TestClient, task_factory, tmp_path: Path
) -> None:
    task_factory("shm")
    results = tmp_path / "results"
    results.mkdir()
    (results / "shm_cv.json").write_text(
        json.dumps({"task": "shm", "metric": "max(0, 1 - MAPE)", "loo": {"score": 0.88}}),
        encoding="utf-8",
    )
    (results / "shm_ladder.json").write_text(
        json.dumps({"nested": {"score": 0.9797, "mape": 0.0203, "n_folds": 64}}),
        encoding="utf-8",
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("NEBULAX_PS3_RESULTS", str(results))
        shm = next(t for t in client.get("/api/ps3/tasks").json() if t["name"] == "shm")
    assert shm["cv"]["metric"] == "mape_score"
    assert shm["cv"]["headline"] == {"score": 0.9797, "mape": 0.0203, "n_folds": 64}


# --------------------------------------------------------------------------------------
# POST /api/ps3/{task}/predict
# --------------------------------------------------------------------------------------


def test_predict_rail_two_batches_accumulate_one_session(client: TestClient, task_factory, tmp_path: Path) -> None:
    task = task_factory("rail")
    second = tmp_path / "Test2.csv"
    shutil.copyfile(RAIL_CSV, second)

    r1 = client.post("/api/ps3/rail/predict", files=[_upload(RAIL_CSV)])
    assert r1.status_code == 200, r1.text
    b1 = r1.json()
    assert set(b1) == {
        "storage",
        "session",
        "task",
        "output_filename",
        "rows",
        "explanations",
        "csv_url",
        "n_done",
        "files",
        "errors",
        "expires_at",
    }
    assert b1["task"] == "rail" and b1["n_done"] == 1 and b1["errors"] == []
    assert b1["rows"] == [{"file_id": "train1_slice.csv", "prediction": "Side I"}]
    assert b1["csv_url"] == f"/api/ps3/results/{b1['session']}.csv"
    exp = b1["explanations"][0]
    assert set(exp) == {"file_id", "numbers", "trace", "viewport"}
    assert exp["file_id"] == "train1_slice.csv"
    assert exp["numbers"] == {"side_i_score": 0.81, "speed_kmh": 52.0}
    assert set(exp["trace"]) == {"x", "y", "marks", "label"}
    assert len(exp["trace"]["x"]) == len(exp["trace"]["y"]) == 3
    assert exp["viewport"] == {"car": 3, "side": "I", "health": "crit", "component": "axlebox_c3_p1"}

    r2 = client.post("/api/ps3/rail/predict", files=[_upload(second)], data={"session": b1["session"]})
    assert r2.status_code == 200, r2.text
    b2 = r2.json()
    assert b2["session"] == b1["session"]
    assert b2["n_done"] == 2 and b2["files"] == ["train1_slice.csv", "Test2.csv"]
    assert [r["file_id"] for r in b2["rows"]] == ["train1_slice.csv", "Test2.csv"]
    assert len(b2["explanations"]) == 1  # this batch only
    assert task.calls == ["train1_slice.csv", "Test2.csv"]  # sequential, in upload order


def test_predict_accepts_the_html_files_array_spelling(client: TestClient, task_factory) -> None:
    task_factory("shm")
    r = client.post("/api/ps3/shm/predict", files=[_upload(SHM_CSV, field="files[]")])
    assert r.status_code == 200, r.text
    assert r.json()["rows"] == [{"file_id": "train01_slice.csv", "prediction": 0.1234}]


def test_predict_acv_xlsx_ranks_the_cars_in_the_file(client: TestClient, task_factory) -> None:
    task_factory("acv")
    body = client.post("/api/ps3/acv/predict", files=[_upload(ACV_XLSX)]).json()
    assert body["rows"] == [{"file_id": "case01_slice.xlsx", "ranked_cars": "01|02|03|04|05|06|07|08"}]
    csv_text = client.get(body["csv_url"]).text
    assert csv_text == "file_id,ranked_cars\ncase01_slice.xlsx,01|02|03|04|05|06|07|08\n"


def test_predict_door_stream_has_no_file_id_column(client: TestClient, task_factory) -> None:
    task_factory("door")
    body = client.post("/api/ps3/door/predict", files=[_upload(DOOR_CSV)]).json()
    assert body["rows"][0] == {
        "start_time": "2023-7-5-0-0-0-0",
        "end_time": "2023-7-5-0-0-4-500",
        "prediction": "Normal",
    }
    r = client.get(body["csv_url"])
    assert r.status_code == 200, r.text
    assert r.text.splitlines()[0] == "start_time,end_time,prediction"
    assert r.headers["content-disposition"] == 'attachment; filename="door_predictions.csv"'


def test_predict_mixed_batch_reports_the_bad_file_and_keeps_the_good_one(
    client: TestClient, task_factory, tmp_path: Path
) -> None:
    task_factory("rail")
    junk = tmp_path / "notes.txt"
    junk.write_text("not a rail file\n", encoding="utf-8")
    body = client.post("/api/ps3/rail/predict", files=[_upload(RAIL_CSV), _upload(junk)]).json()
    assert body["n_done"] == 1 and len(body["rows"]) == 1
    assert body["errors"] == [{"file": "notes.txt", "message": "Rail Corrugation accepts .csv, not .txt"}]


# --------------------------------------------------------------------------------------
# The CSV download is the CLI's bytes
# --------------------------------------------------------------------------------------


def test_download_is_byte_identical_to_the_cli_output(
    client: TestClient, task_factory, tmp_path: Path
) -> None:
    task_factory("rail")
    inputs = tmp_path / "in"
    inputs.mkdir()
    shutil.copyfile(RAIL_CSV, inputs / "Test1.csv")
    shutil.copyfile(RAIL_CSV, inputs / "Test2.csv")

    body = client.post(
        "/api/ps3/rail/predict",
        files=[_upload(inputs / "Test1.csv"), _upload(inputs / "Test2.csv")],
    ).json()
    served = client.get(body["csv_url"])
    assert served.status_code == 200, served.text

    out = tmp_path / "rail_predictions.csv"
    assert _cli().run("rail", inputs, out, expect_ids=True, quiet=True) == 0
    assert served.content == out.read_bytes()
    assert out.read_text(encoding="utf-8") == "file_id,prediction\nTest1.csv,Side I\nTest2.csv,Normal\n"


# --------------------------------------------------------------------------------------
# 4xx / 503
# --------------------------------------------------------------------------------------


def test_unknown_task_is_404(client: TestClient) -> None:
    r = client.post("/api/ps3/bogie/predict", files=[_upload(RAIL_CSV)])
    assert r.status_code == 404
    assert "unknown PS3 task" in r.json()["detail"]
    assert client.get("/api/ps3/bogie/cv").status_code == 404


def test_wrong_extension_only_is_400(client: TestClient, task_factory, tmp_path: Path) -> None:
    task_factory("rail")
    junk = tmp_path / "photo.png"
    junk.write_bytes(b"\x89PNG\r\n")
    r = client.post("/api/ps3/rail/predict", files=[_upload(junk)])
    assert r.status_code == 400
    assert "accepts .csv" in r.json()["detail"]


def test_unreadable_file_is_400(client: TestClient, task_factory, tmp_path: Path) -> None:
    task_factory("acv")
    broken = tmp_path / "acv_test_case.xlsx"
    broken.write_text("this is not a workbook", encoding="utf-8")
    r = client.post("/api/ps3/acv/predict", files=[_upload(broken)])
    assert r.status_code == 400
    assert "acv_test_case.xlsx" in r.json()["detail"]


def test_too_many_files_in_one_request_is_413(client: TestClient, task_factory, tmp_path: Path) -> None:
    task_factory("shm")
    small = tmp_path / "t.csv"
    small.write_text("1.0\n2.0\n", encoding="utf-8")
    uploads = [("files", (f"test{i:02d}.csv", small.read_bytes(), "text/csv")) for i in range(api_ps3.MAX_BATCH_FILES + 1)]
    r = client.post("/api/ps3/shm/predict", files=uploads)
    assert r.status_code == 413
    assert "per request" in r.json()["detail"]


def test_no_files_is_400(client: TestClient, task_factory) -> None:
    task_factory("rail")
    r = client.post("/api/ps3/rail/predict", data={"session": ""})
    assert r.status_code == 400
    assert "no files uploaded" in r.json()["detail"]


def test_missing_task_module_is_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(name: str):
        raise KeyError(f"PS3 task {name!r} is not registered and nebulax.ps3.{name} could not be imported")

    monkeypatch.setattr(api_ps3, "get_task", boom)
    r = client.post("/api/ps3/rail/predict", files=[_upload(RAIL_CSV)])
    assert r.status_code == 503
    assert "ps3_train.py --task rail" in r.json()["detail"]


def test_missing_artefact_is_503(client: TestClient, task_factory) -> None:
    task = task_factory("shm")

    def no_model(feats, model=None):
        raise FileNotFoundError("no model at models/ps3/shm.pkl; run `python scripts/ps3_train.py --task shm`")

    task.predict = no_model  # type: ignore[method-assign]
    r = client.post("/api/ps3/shm/predict", files=[_upload(SHM_CSV)])
    assert r.status_code == 503
    assert "no model at models/ps3/shm.pkl" in r.json()["detail"]


def test_unknown_or_expired_session_is_404(client: TestClient, task_factory) -> None:
    task_factory("rail")
    assert client.get("/api/ps3/results/deadbeefdeadbeef.csv").status_code == 404
    assert client.delete("/api/ps3/results/deadbeefdeadbeef").status_code == 404
    assert client.get("/api/ps3/results/..%2F..%2Fetc%2Fpasswd.csv").status_code == 404
    r = client.post("/api/ps3/rail/predict", files=[_upload(RAIL_CSV)], data={"session": "nope-nope-nope"})
    assert r.status_code == 404


def test_session_belongs_to_one_task(client: TestClient, task_factory) -> None:
    task_factory("rail")
    task_factory("shm")
    token = client.post("/api/ps3/rail/predict", files=[_upload(RAIL_CSV)]).json()["session"]
    r = client.post("/api/ps3/shm/predict", files=[_upload(SHM_CSV)], data={"session": token})
    assert r.status_code == 400
    assert "belongs to task 'rail'" in r.json()["detail"]


def test_cv_route_reads_the_results_json(client: TestClient, tmp_path: Path) -> None:
    assert client.get("/api/ps3/shm/cv").status_code == 404
    results = tmp_path / "results_ps3"
    results.mkdir(parents=True, exist_ok=True)
    payload = {"scheme": "LOO + repeated 5x10-fold", "folds": [1, 2, 3], "summary": {"mape": 0.19}}
    (results / "shm_cv.json").write_text(json.dumps(payload), encoding="utf-8")
    assert client.get("/api/ps3/shm/cv").json() == payload


# --------------------------------------------------------------------------------------
# Session lifecycle
# --------------------------------------------------------------------------------------


def test_delete_removes_the_session_and_its_temp_files(client: TestClient, task_factory, tmp_path: Path) -> None:
    task_factory("rail")
    token = client.post("/api/ps3/rail/predict", files=[_upload(RAIL_CSV)]).json()["session"]
    session_dir = tmp_path / "uploads" / token
    assert (session_dir / "batch001" / "train1_slice.csv").exists()
    body = client.delete(f"/api/ps3/results/{token}").json()
    assert body == {"session": token, "deleted": True, "n_files": 1, "n_rows": 1}
    assert not session_dir.exists()
    assert client.get(f"/api/ps3/results/{token}.csv").status_code == 404


def test_idle_sessions_expire_after_the_ttl(client: TestClient, task_factory, tmp_path: Path) -> None:
    task_factory("rail")
    token = client.post("/api/ps3/rail/predict", files=[_upload(RAIL_CSV)]).json()["session"]
    api_ps3._SESSIONS[token].touched -= api_ps3.SESSION_TTL_S + 1.0
    assert client.get("/api/ps3/tasks").status_code == 200  # any call sweeps
    assert token not in api_ps3._SESSIONS
    assert not (tmp_path / "uploads" / token).exists()
    assert client.get(f"/api/ps3/results/{token}.csv").status_code == 404


# --------------------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------------------


def test_create_app_mounts_the_ps3_router_before_the_static_files(tmp_path: Path) -> None:
    """The router is reachable through the real factory, and the static mount never shadows it."""
    base = dict(data_dir=tmp_path / "data", results_dir=tmp_path / "r")
    with TestClient(create_app(Settings(web_dist=tmp_path / "nodist", **base))) as c:
        assert c.get("/api/ps3/tasks").status_code == 200
        assert c.get("/api/health").status_code == 200

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>x</title>", encoding="utf-8")
    with TestClient(create_app(Settings(web_dist=dist, **base))) as c:
        assert c.get("/").status_code == 200  # the static mount is live
        assert c.get("/api/ps3/tasks").status_code == 200  # and does not shadow /api/ps3
        assert c.get("/api/ps3/results/deadbeefdeadbeef.csv").status_code == 404
