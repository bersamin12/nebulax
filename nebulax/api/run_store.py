"""Shared showcase run archive. Immutable revisions, published by one index object."""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException

ID_RE = re.compile(r"^\d{8}T\d{12}Z_[a-f0-9]{12}$")


def enabled():
    return bool(os.getenv("NEBULAX_RUN_BUCKET"))


def bucket():
    if not enabled():
        raise HTTPException(503, "Cloud run storage is not configured")
    provider = os.getenv("NEBULAX_STORAGE_PROVIDER", "gcs")
    if provider == "r2":
        from .r2_store import Bucket
        return Bucket(os.environ["NEBULAX_RUN_BUCKET"])
    if provider != "gcs":
        raise HTTPException(503, "Unknown storage provider")
    from google.cloud import storage
    return storage.Client().bucket(os.environ["NEBULAX_RUN_BUCKET"])


def new_run(task, name):
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:12]
    label = str(name or "").strip()[:80] or f"{task}-run"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", label).strip("-")[:80] or f"{task}-run"
    return {"id": run_id, "name": label, "task": task, "created_at": now.isoformat(),
            "prefix": f"runs/{now:%Y-%m-%d}/{run_id.split('T')[1]}/{slug}"}


def index_key(run_id):
    if not ID_RE.fullmatch(run_id):
        raise HTTPException(400, "Invalid run ID")
    return f"runs/index/{run_id}.json"


def read_index(run_id):
    blob = bucket().blob(index_key(run_id))
    try:
        return json.loads(blob.download_as_bytes())
    except Exception as exc:
        if getattr(exc, "code", None) == 404:
            raise HTTPException(404, "Saved run not found") from exc
        raise


def save(run, inputs, payload, csv_text, output_filename, input_cache=None, cloud_inputs=None):
    target = bucket()
    files = list((cloud_inputs or {}).values())
    input_cache = input_cache if input_cache is not None else {}
    for name, path in inputs.items():
        if name in (cloud_inputs or {}):
            continue
        stat = path.stat()
        signature = (str(path), stat.st_size, stat.st_mtime_ns)
        cached = input_cache.get(name)
        if cached and cached[0] == signature:
            files.append(cached[1])
            continue
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        key = f"{run['prefix']}/inputs/{digest[:16]}/{name}"
        blob = target.blob(key)
        # The checksum in the key makes retries safe and avoids overwriting inputs.
        if not blob.exists():
            blob.upload_from_string(content, if_generation_match=0, timeout=120)
        record = {"name": name, "size": len(content), "sha256": digest, "object": key}
        files.append(record)
        input_cache[name] = (signature, record)
    revision = f"{run['prefix']}/outputs/{uuid.uuid4().hex}"
    result_key = f"{revision}/results.json"
    csv_key = f"{revision}/{output_filename}"
    target.blob(result_key).upload_from_string(json.dumps(payload, allow_nan=False), content_type="application/json", if_generation_match=0)
    target.blob(csv_key).upload_from_string(csv_text, content_type="text/csv", if_generation_match=0)
    manifest = {**run, "updated_at": datetime.now(timezone.utc).isoformat(), "files": files,
                "n_done": payload["n_done"], "n_rows": len(payload["rows"]),
                "result_object": result_key, "csv_object": csv_key,
                "output_filename": output_filename, "errors": payload.get("errors", [])}
    # Publish only after all inputs and both outputs have been durably written.
    target.blob(index_key(run["id"])).upload_from_string(json.dumps(manifest), content_type="application/json")
    return manifest


def list_runs(task=None, start=None, end=None):
    for value in (start, end):
        if value:
            try:
                datetime.strptime(value, "%Y-%m-%d")
            except ValueError as exc:
                raise HTTPException(400, "Dates must use YYYY-MM-DD") from exc
    if start and end and start > end:
        raise HTTPException(400, "Start date must not follow end date")
    target = bucket()
    runs = []
    for blob in target.client.list_blobs(target.name, prefix="runs/index/"):
        stamp = blob.name.rsplit("/", 1)[-1][:8]
        date = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
        if (start and date < start) or (end and date > end):
            continue
        entry = json.loads(blob.download_as_bytes())
        if not task or entry["task"] == task:
            runs.append(entry)
    return sorted(runs, key=lambda r: r["id"], reverse=True)


def result(run_id):
    entry = read_index(run_id)
    payload = json.loads(bucket().blob(entry["result_object"]).download_as_bytes())
    return {**payload, "session": None, "run": entry,
            "csv_url": f"/api/ps3/runs/{run_id}/csv", "storage": {"status": "saved", "run": entry}}


def input_blob(run_id, name):
    entry = read_index(run_id)
    selected = next((f for f in entry["files"] if f["name"] == name), None)
    if not selected:
        raise HTTPException(404, "Input file not found in this run")
    return entry, selected, bucket().blob(selected["object"])


def create_upload_batch(task, name, files, max_bytes, suffixes):
    """Persist an immutable allowlist before issuing any upload capabilities."""
    if not isinstance(files, list) or not 1 <= len(files) <= 1024:
        raise HTTPException(400, "Choose between 1 and 1024 files per batch")
    run = new_run(task, name)
    records, names = [], set()
    for item in files:
        filename = item.get("name") if isinstance(item, dict) else None
        size = item.get("size") if isinstance(item, dict) else None
        if (not isinstance(filename, str) or not filename or len(filename) > 200
                or any(c in filename for c in '/\\\r\n') or filename in names
                or Path(filename).suffix.lower() not in suffixes):
            raise HTTPException(400, "Invalid, duplicate, or unsupported input filename")
        if type(size) is not int or not 0 < size <= max_bytes:
            raise HTTPException(413, f"Each input must contain 1 to {max_bytes} bytes")
        names.add(filename)
        records.append({"name": filename, "size": size,
                        "object": f"{run['prefix']}/inputs/{uuid.uuid4().hex}/{filename}"})
    manifest = {**run, "files": records}
    bucket().blob(upload_key(run["id"])).upload_from_string(
        json.dumps(manifest), content_type="application/json", if_generation_match=0)
    return manifest


def upload_key(batch_id):
    index_key(batch_id)  # validate before forming any storage key
    return f"runs/uploads/{batch_id}.json"


def upload_input(batch_id, name):
    target = bucket()
    try:
        entry = json.loads(target.blob(upload_key(batch_id)).download_as_bytes())
    except Exception as exc:
        if getattr(exc, "code", None) == 404:
            raise HTTPException(404, "Upload batch not found") from exc
        raise
    selected = next((f for f in entry["files"] if f["name"] == name), None)
    if not selected:
        raise HTTPException(404, "File is not in this upload batch")
    return entry, selected, target.blob(selected["object"])


def upload_session(batch_id, name, origin):
    _, selected, blob = upload_input(batch_id, name)
    if os.getenv("NEBULAX_STORAGE_PROVIDER", "gcs") == "r2":
        from .r2_store import session
        return session(blob, selected["size"])
    if blob.exists():
        return {"complete": True}
    url = blob.create_resumable_upload_session(
        content_type="application/octet-stream", size=selected["size"],
        origin=origin, if_generation_match=0)
    return {"complete": False, "url": url}


def complete_upload(batch_id, name):
    if os.getenv("NEBULAX_STORAGE_PROVIDER", "gcs") != "r2":
        raise HTTPException(400, "This endpoint is only used for R2 multipart uploads")
    from .r2_store import complete
    _, selected, blob = upload_input(batch_id, name)
    return complete(blob, selected["size"])
