"""R2 implementation of the small object-store interface used by run_store.

ETags pin reads; create-only writes use If-None-Match. No Google credentials
are used in this provider. Multipart sessions are persisted for restart recovery.
"""
import json
import math
import os
import threading
from functools import lru_cache

from fastapi import HTTPException

CHUNK = 8 * 1024 * 1024
_locks = {}
_locks_guard = threading.Lock()


class StorageError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(f"R2 operation failed ({code})")


def invoke(client, operation, **kwargs):
    from botocore.exceptions import ClientError
    try:
        return getattr(client, operation)(**kwargs)
    except ClientError as exc:
        code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 500)
        raise StorageError(code) from exc


@lru_cache(maxsize=1)
def client():
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto",
        config=Config(signature_version="s3v4", request_checksum_calculation="when_required",
                      response_checksum_validation="when_required",
                      retries={"max_attempts": 3, "mode": "standard"}))


class Bucket:
    def __init__(self, name, s3=None):
        self.name, self.s3 = name, s3 if s3 is not None else client()
        self.client = self

    def blob(self, name):
        return Blob(self, name)

    def list_blobs(self, name, prefix):
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=name, Prefix=prefix):
            for item in page.get("Contents", []):
                yield self.blob(item["Key"])


class Blob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    def call(self, operation, **kwargs):
        return invoke(self.bucket.s3, operation, Bucket=self.bucket.name, Key=self.name, **kwargs)

    def reload(self):
        info = self.call("head_object")
        self.size, self.generation, self.crc32c = info["ContentLength"], info["ETag"], None

    def exists(self):
        try:
            self.reload()
            return True
        except StorageError as exc:
            if exc.code == 404:
                return False
            raise

    def upload_from_string(self, data, content_type="application/octet-stream", if_generation_match=None, **unused):
        args = {"Body": data.encode() if isinstance(data, str) else data, "ContentType": content_type}
        if if_generation_match == 0:
            args["IfNoneMatch"] = "*"
        elif if_generation_match is not None:
            args["IfMatch"] = if_generation_match
        self.call("put_object", **args)

    def download_as_bytes(self):
        stream = self.call("get_object")["Body"]
        try:
            return stream.read()
        finally:
            stream.close()

    def download_to_file(self, fh, if_generation_match=None):
        args = {"IfMatch": if_generation_match} if if_generation_match is not None else {}
        stream = self.call("get_object", **args)["Body"]
        try:
            for chunk in stream.iter_chunks(chunk_size=1024 * 1024):
                fh.write(chunk)
        finally:
            stream.close()


def lock_for(key):
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def state_blob(blob):
    # The allowed object key was generated server-side, never accepted from the browser.
    import hashlib
    digest = hashlib.sha256(blob.name.encode()).hexdigest()
    return blob.bucket.blob(f"runs/multipart/{digest}.json")


def parts(blob, upload_id):
    found = []
    args = {"Bucket": blob.bucket.name, "Key": blob.name, "UploadId": upload_id}
    for page in blob.bucket.s3.get_paginator("list_parts").paginate(**args):
        found.extend(page.get("Parts", []))
    return sorted(found, key=lambda item: item["PartNumber"])


def session(blob, size):
    with lock_for(blob.name):
        if blob.exists():
            if blob.size != size:
                raise HTTPException(409, "Stored input has an unexpected size")
            return {"complete": True, "provider": "r2"}
        state = state_blob(blob)
        upload_id = json.loads(state.download_as_bytes())["upload_id"] if state.exists() else None
        uploaded = []
        if upload_id:
            from botocore.exceptions import ClientError
            try:
                uploaded = parts(blob, upload_id)
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") != "NoSuchUpload":
                    raise
                upload_id = None
        if not upload_id:
            upload_id = blob.call("create_multipart_upload", ContentType="application/octet-stream")["UploadId"]
            state.upload_from_string(json.dumps({"upload_id": upload_id}), content_type="application/json")
        accepted = {p["PartNumber"]: p["Size"] for p in uploaded}
        plan = []
        for number in range(1, math.ceil(size / CHUNK) + 1):
            length = min(CHUNK, size - (number - 1) * CHUNK)
            args = {"Bucket": blob.bucket.name, "Key": blob.name, "UploadId": upload_id, "PartNumber": number}
            plan.append({"number": number, "size": length, "done": accepted.get(number) == length,
                         "url": blob.bucket.s3.generate_presigned_url("upload_part", Params=args, ExpiresIn=900)})
        return {"complete": False, "provider": "r2", "chunk_size": CHUNK, "parts": plan}


def complete(blob, size):
    with lock_for(blob.name):
        if blob.exists():
            if blob.size != size:
                raise HTTPException(409, "Stored input has an unexpected size")
            return {"complete": True}
        state = state_blob(blob)
        if not state.exists():
            raise HTTPException(409, "Start the upload first")
        upload_id = json.loads(state.download_as_bytes())["upload_id"]
        uploaded = parts(blob, upload_id)
        if len(uploaded) != math.ceil(size / CHUNK) or any(
            p["PartNumber"] != i or p["Size"] != min(CHUNK, size - (i-1)*CHUNK)
            for i, p in enumerate(uploaded, 1)
        ):
            raise HTTPException(409, "Upload is incomplete or has unexpected part sizes")
        blob.call("complete_multipart_upload", UploadId=upload_id,
                  MultipartUpload={"Parts": [{"PartNumber": p["PartNumber"], "ETag": p["ETag"]} for p in uploaded]})
        blob.reload()
        if blob.size != size:
            raise HTTPException(409, "Stored input has an unexpected size")
        return {"complete": True}
