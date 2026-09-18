"""The ACV picker checks workbook structure before adding a file to the queue."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("openpyxl")
from openpyxl import Workbook  # noqa: E402

from nebulax.api.ps3 import router  # noqa: E402


def workbook(*, time: bool = True, cars: int = 8) -> bytes:
    book = Workbook()
    sheet = book.active
    sheet.append((["Time"] if time else ["Other"]) + [f"Car {i:02d} - Indoor Average Temperature" for i in range(1, cars + 1)])
    sheet.append(([datetime(2021, 6, 24)] if time else ["x"]) + [24.0] * cars)
    buffer = BytesIO()
    book.save(buffer)
    return buffer.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("NEBULAX_PS3_UPLOADS", str(tmp_path))
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_acv_preflight_accepts_eight_car_workbook_without_retaining_upload(client, tmp_path):
    response = client.post("/api/ps3/acv/validate", files={"file": ("case.xlsx", workbook())})
    assert response.status_code == 200
    assert response.json()["valid"] is True
    assert len(response.json()["summary"]["cars"]) == 8
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("payload, issue", [(workbook(time=False), "Time"), (workbook(cars=2), "8 cars"), (b"bad zip", "readable")])
def test_acv_preflight_explains_wrong_workbooks(client, payload, issue):
    response = client.post("/api/ps3/acv/validate", files={"file": ("case.xlsx", payload)})
    assert response.status_code == 200
    report = response.json()
    assert report["valid"] is False
    assert issue in " ".join(report["errors"])
