"""Copy Google objects to R2 with read-back SHA256 verification. Never delete sources.

Dry-run by default. Refuses more than 8 GiB combined source/destination projected
storage. Indexes publish after data; matching existing files are verified, conflicting
objects stop the migration rather than being overwritten.
"""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from free_env import load_env, require_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nebulax.api.r2_store import client


def hash_stream(stream):
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def main():
    load_env()
    require_env("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "NEBULAX_RUN_BUCKET")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="nebulax-ps3-qwiklabs-gcp-02-ebc381898f1f")
    p.add_argument("--project", default="qwiklabs-gcp-02-ebc381898f1f")
    p.add_argument("--copy", action="store_true")
    p.add_argument("--gcloud-auth", action="store_true")
    args = p.parse_args()
    from google.cloud import storage
    credentials = None
    if args.gcloud_auth:
        import subprocess
        from google.oauth2.credentials import Credentials
        from gcp_deploy import find_gcloud
        token = subprocess.check_output([find_gcloud(), "auth", "print-access-token"], text=True).strip()
        credentials = Credentials(token)
    gcs = storage.Client(project=args.project, credentials=credentials)
    source = [b for b in gcs.list_blobs(args.source)
              if b.name.startswith(("runs/", "inputs/", "outputs/"))
              and not b.name.startswith(("runs/uploads/", "runs/multipart/"))]
    source.sort(key=lambda b: (b.name.startswith("runs/index/"), b.name))
    s3 = client()
    bucket = os.environ["NEBULAX_RUN_BUCKET"]
    existing = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        existing.update({item["Key"]: item["Size"] for item in page.get("Contents", [])})
    projected = sum(existing.values()) + sum(b.size for b in source if b.name not in existing)
    print(f"Source: {len(source)} objects, {sum(b.size for b in source)/1024**3:.3f} GiB")
    print(f"Projected destination: {projected/1024**3:.3f} GiB")
    if projected > 8 * 1024**3:
        p.error("Migration exceeds the 8 GiB budget guard; select less data before proceeding")
    if not args.copy:
        print("Preview only. No objects copied.")
        return
    verified = 0
    for blob in source:
        with tempfile.TemporaryFile() as fh:
            blob.download_to_file(fh, if_generation_match=blob.generation)
            fh.seek(0)
            digest = hash_stream(fh)
            fh.seek(0)
            if blob.name not in existing:
                # Simple PUT is valid for our bounded migration files and supports create-only writes.
                if blob.size > 1024**3:
                    raise RuntimeError("Object exceeds this tool's 1 GiB single-object migration limit")
                s3.put_object(Bucket=bucket, Key=blob.name, Body=fh,
                    ContentType=blob.content_type or "application/octet-stream", IfNoneMatch="*",
                    Metadata={"sha256": digest, "source-generation": str(blob.generation)})
            response = s3.get_object(Bucket=bucket, Key=blob.name)
            try:
                actual = hash_stream(response["Body"])
            finally:
                response["Body"].close()
            if actual != digest:
                raise RuntimeError(f"Destination conflict or checksum mismatch: {blob.name}")
            verified += 1
    print(f"Copied/verified {verified} objects. Google originals retained. No continuous synchronization configured.")


if __name__ == "__main__":
    main()
