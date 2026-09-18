"""Desktop path fallback: access guard and prediction parity with ordinary upload."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nebulax.api import ps3


def _client(host="127.0.0.1"):
    app = FastAPI()
    app.include_router(ps3.router)
    return TestClient(app, base_url=f"http://{host}", client=(host, 50000))


def test_local_paths_require_opt_in_loopback_and_same_origin(tmp_path, monkeypatch):
    path = tmp_path / "Test.csv"
    path.write_text("datetime,motor current\n2023-7-5-0-0-0-0,1\n", encoding="utf-8")
    url = "/api/ps3/door/local/inspect"
    assert _client().post(url, json={"path": str(path)}).status_code == 403
    monkeypatch.setenv("NEBULAX_LOCAL_PATHS", "1")
    assert _client("192.0.2.10").post(url, json={"path": str(path)}).status_code == 403
    assert _client().post(url, json={"path": str(path)}, headers={"Origin": "https://other.example"}).status_code == 403
    assert _client().post(url, json={"path": "relative/Test.csv"}).status_code == 400
    response = _client().post(url, json={"path": str(path)}, headers={"Origin": "http://127.0.0.1"})
    assert response.status_code == 200
    assert response.json()["files"] == [{"name": "Test.csv", "path": str(path), "size": path.stat().st_size}]
    wrong_tab = _client().post("/api/ps3/rail/local/inspect", json={"path": str(path)})
    assert wrong_tab.status_code == 400
    assert "Select the Door tab" in wrong_tab.json()["detail"]
    wrong_run = _client().post("/api/ps3/rail/local/stream", json={"path": str(path)})
    assert wrong_run.status_code == 400
    assert "Select the Door tab" in wrong_run.json()["detail"]


def test_local_door_stream_matches_uploaded_door_stream(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "readingmaterials/problem_statement/PS3/02_Datasets/Door/Test.csv"
    if not source.exists():
        return
    monkeypatch.setenv("NEBULAX_LOCAL_PATHS", "1")
    monkeypatch.setenv("NEBULAX_PS3_UPLOADS", str(tmp_path / "uploads"))
    client = _client()
    with source.open("rb") as fh:
        upload = client.post("/api/ps3/door/stream", files={"files": (source.name, fh, "text/csv")})
    local = client.post("/api/ps3/door/local/stream", json={"path": str(source)})
    assert upload.status_code == local.status_code == 200
    assert local.json()["rows"] == upload.json()["rows"]
    assert local.json()["errors"] == upload.json()["errors"] == []
    ps3.reset_sessions()
