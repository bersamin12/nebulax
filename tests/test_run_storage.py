"""Archive integration with an in-memory GCS double; no credentials or cloud mutations."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nebulax.api import ps3, run_store
from nebulax.ps3 import common
from test_api_ps3 import DummyTask


class Missing(Exception):
    code = 404


class Blob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    def exists(self):
        return self.name in self.bucket.data

    def upload_from_string(self, data, **kwargs):
        if self.bucket.fail:
            raise RuntimeError("storage unavailable")
        if kwargs.get("if_generation_match") == 0:
            assert not self.exists()
        self.bucket.data[self.name] = data.encode() if isinstance(data, str) else data

    def download_as_bytes(self):
        if not self.exists():
            raise Missing()
        return self.bucket.data[self.name]

    def reload(self):
        self.size = len(self.download_as_bytes())
        self.generation = 1
        self.crc32c = "test-checksum"

    def create_resumable_upload_session(self, **kwargs):
        self.bucket.sessions.append((self.name, kwargs))
        return "https://storage.googleapis.com/test-upload"

    def download_to_file(self, fh, **kwargs):
        fh.write(self.download_as_bytes())


class Bucket:
    name = "test-bucket"
    def __init__(self):
        self.client = self
        self.data = {}
        self.fail = False
        self.sessions = []

    def blob(self, name):
        return Blob(self, name)

    def list_blobs(self, name, prefix):
        return [self.blob(k) for k in self.data if k.startswith(prefix)]


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.setenv("NEBULAX_PS3_UPLOADS", str(tmp_path / "uploads"))
    monkeypatch.setenv("NEBULAX_RUN_BUCKET", "test-bucket")
    bucket = Bucket()
    monkeypatch.setattr(run_store, "bucket", lambda: bucket)
    monkeypatch.setattr(ps3, "task_model", lambda _: None)
    task = DummyTask("rail")
    monkeypatch.setitem(common.TASKS, "rail", task)
    ps3.reset_sessions()
    app = FastAPI()
    app.include_router(ps3.router)
    yield TestClient(app), bucket, task
    ps3.reset_sessions()


def upload(client, name="Train1.csv", **data):
    return client.post("/api/ps3/rail/predict", files=[("files", (name, b"raw input\n1\n", "text/csv"))], data=data)


def test_archive_survives_session_loss_and_open_does_not_predict(archive):
    client, bucket, task = archive
    first = upload(client, run_name="Morning / inspection").json()
    assert first["storage"]["status"] == "saved"
    run = first["storage"]["run"]
    assert run["prefix"].endswith("/Morning-inspection")
    second = upload(client, "Train2.csv", session=first["session"]).json()
    assert second["storage"]["run"]["id"] == run["id"]
    assert second["n_done"] == 2
    ps3.reset_sessions()
    calls = list(task.calls)
    loaded = client.get(f"/api/ps3/runs/{run['id']}").json()
    assert len(loaded["rows"]) == len(loaded["explanations"]) == 2
    assert loaded["session"] is None
    assert task.calls == calls
    csv = client.get(loaded["csv_url"])
    assert csv.status_code == 200 and "Train2.csv,Normal" in csv.text
    assert bucket.data[loaded["run"]["files"][0]["object"]] == b"raw input\n1\n"
    rerun = client.post(f"/api/ps3/runs/{run['id']}/predict", json={"file": "Train1.csv", "run_name": "Rerun"})
    assert rerun.status_code == 200, rerun.text
    assert len(task.calls) == len(calls) + 1
    assert rerun.json()["storage"]["run"]["id"] != run["id"]


def test_failed_save_is_visible_and_retry_does_not_predict(archive):
    client, bucket, task = archive
    bucket.fail = True
    res = upload(client).json()
    assert res["storage"]["status"] == "error"
    assert len(res["rows"]) == 1
    bucket.fail = False
    saved = client.post(f"/api/ps3/results/{res['session']}/save").json()
    assert saved["status"] == "saved"
    assert len(task.calls) == 1


def test_filters_latest_and_invalid_references(archive):
    client, _, _ = archive
    a = upload(client, run_name="first").json()["storage"]["run"]
    b = upload(client, run_name="second").json()["storage"]["run"]
    date = a["created_at"][:10]
    listing = client.get(f"/api/ps3/runs?task=rail&start={date}&end={date}").json()
    assert [r["id"] for r in listing["runs"]] == [b["id"], a["id"]]
    assert client.get("/api/ps3/runs?task=acv").json()["runs"] == []
    assert client.get("/api/ps3/runs?start=2026-99-99").status_code == 400
    assert client.get("/api/ps3/runs?start=2026-09-22&end=2026-09-20").status_code == 400
    assert client.get("/api/ps3/runs/bad-id").status_code == 400
    assert client.post(f"/api/ps3/runs/{a['id']}/predict", json={"file": "../secret"}).status_code == 404


def test_failed_revision_does_not_replace_published_result(archive, monkeypatch):
    client, bucket, _ = archive
    first = upload(client).json()
    run_id = first["storage"]["run"]["id"]
    original = Blob.upload_from_string
    def fail_csv(self, data, **kwargs):
        if self.name.endswith(".csv"):
            raise RuntimeError("output write failed")
        original(self, data, **kwargs)
    monkeypatch.setattr(Blob, "upload_from_string", fail_csv)
    res = upload(client, "Train2.csv", session=first["session"]).json()
    assert res["storage"]["status"] == "error"
    assert client.get(f"/api/ps3/runs/{run_id}").json()["n_done"] == 1


def test_direct_batch_resume_restart_and_no_input_reupload(archive):
    client, bucket, task = archive
    files = [{"name": "Train1.csv", "size": 12}, {"name": "Train2.csv", "size": 12}]
    batch = client.post("/api/ps3/rail/uploads", json={"files": files, "run_name": "Direct batch"}).json()
    route = f"/api/ps3/uploads/{batch['id']}"
    session = client.post(route + "/session", json={"file": "Train1.csv"}, headers={"origin": "https://app.example"})
    assert session.status_code == 200 and session.headers["cache-control"] == "no-store"
    assert bucket.sessions[0][1] == {"content_type": "application/octet-stream", "size": 12,
                                    "origin": "https://app.example", "if_generation_match": 0}
    assert client.post(route + "/predict", json={"file": "Train1.csv"}).status_code == 409
    for i, item in enumerate(batch["files"]):
        # Simulate bytes arriving directly at storage, bypassing the FastAPI multipart route.
        bucket.data[item["object"]] = b"raw input\n1\n"
        assert client.post(route + "/session", json={"file": item["name"]}).json() == {"complete": True}
        res = client.post(route + "/predict", json={"file": item["name"]})
        assert res.status_code == 200, res.text
        assert res.json()["n_done"] == i + 1
        assert res.json()["storage"]["run"]["id"] == batch["id"]
        ps3.reset_sessions()
    again = client.post(route + "/predict", json={"file": "Train1.csv"})
    assert again.status_code == 200 and again.json()["n_done"] == 2
    assert len(task.calls) == 2  # retry after restart does not recompute
    keys = [key for key in bucket.data if "/inputs/" in key]
    assert set(keys) == {f["object"] for f in batch["files"]}
    saved = client.get(f"/api/ps3/runs/{batch['id']}").json()
    assert len(saved["rows"]) == len(saved["run"]["files"]) == 2


def test_direct_manifest_validation_and_size_verification(archive):
    client, bucket, _ = archive
    for files in ([], [{"name": "../Train.csv", "size": 3}], [{"name": "Train.csv", "size": "3"}],
                  [{"name": "Train.csv", "size": 0}], [{"name": "Train.csv", "size": ps3.MAX_UPLOAD_BYTES + 1}],
                  [{"name": "a.csv", "size": 3}] * 2):
        assert client.post("/api/ps3/rail/uploads", json={"files": files}).status_code in (400, 413)
    batch = client.post("/api/ps3/rail/uploads", json={"files": [{"name": "Train.csv", "size": 3}]}).json()
    route = f"/api/ps3/uploads/{batch['id']}"
    assert client.post(route + "/session", json={"file": "secret.csv"}).status_code == 404
    bucket.data[batch["files"][0]["object"]] = b"oversized"
    assert client.post(route + "/predict", json={"file": "Train.csv"}).status_code == 413


def test_direct_next_upload_keeps_combined_run(archive):
    client, bucket, _ = archive
    run_id = None
    for i in range(2):
        name = f"Train{i}.csv"
        batch = client.post("/api/ps3/rail/uploads", json={"files": [{"name": name, "size": 12}]}).json()
        bucket.data[batch["files"][0]["object"]] = b"raw input\n1\n"
        result = client.post(f"/api/ps3/uploads/{batch['id']}/predict", json={"file": name, "run_id": run_id})
        assert result.status_code == 200, result.text
        run_id = run_id or batch["id"]
        assert result.json()["storage"]["run"]["id"] == run_id
        assert result.json()["n_done"] == i + 1
        ps3.reset_sessions()
