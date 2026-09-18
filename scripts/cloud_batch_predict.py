#!/usr/bin/env python
"""Cloud Run Job: score a GCS folder and write one validated PS3 CSV to GCS.

Set PS3_TASK, PS3_INPUT_PREFIX (gs://bucket/folder/), and PS3_OUTPUT_URI
(gs://bucket/path/predictions.csv). Uses Application Default Credentials.
The same task implementation and CSV writer serve the interactive app.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nebulax.api.ps3 import resolve_task, run_file, task_model, write_csv  # noqa: E402
from nebulax.ps3.common import ACCEPTED_SUFFIXES, OUTPUT_FILENAMES, TASK_NAMES, acv_car_ids, natural_key  # noqa: E402
from nebulax.ps3.submission import validate_csv  # noqa: E402


def gcs_location(uri: str, *, prefix: bool = False) -> tuple[str, str]:
    """Accept an object URI, or a non-root folder URI ending in '/' for input."""
    parsed = urlsplit(uri)
    if parsed.scheme != "gs" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"expected gs://bucket/path, got {uri!r}")
    name = parsed.path.lstrip("/")
    if not name or (prefix and not name.endswith("/")) or (not prefix and name.endswith("/")):
        raise ValueError(f"expected {'folder ending in /' if prefix else 'object path'}: {uri!r}")
    if any(part in (".", "..") for part in name.split("/")):
        raise ValueError(f"unsafe GCS path: {uri!r}")
    return parsed.netloc, name


def run(task: str, input_prefix: str, output_uri: str, *, client=None) -> dict[str, object]:
    """Process inputs in filename order and publish only after full CSV validation."""
    if task not in TASK_NAMES:
        raise ValueError(f"unknown task {task!r}; expected {', '.join(TASK_NAMES)}")
    source_bucket, source_prefix = gcs_location(input_prefix, prefix=True)
    target_bucket, target_name = gcs_location(output_uri)
    if source_bucket == target_bucket and target_name.startswith(source_prefix):
        raise ValueError("the output object must be outside the input folder")
    if not target_name.lower().endswith(".csv"):
        raise ValueError("the output object must be a CSV")

    if client is None:
        from google.cloud import storage

        client = storage.Client()

    suffixes = ACCEPTED_SUFFIXES[task]
    blobs = [
        blob for blob in client.list_blobs(source_bucket, prefix=source_prefix)
        if not blob.name.endswith("/")
        and not Path(blob.name).name.startswith((".", "~$"))
        and Path(blob.name).suffix.lower() in suffixes
    ]
    blobs.sort(key=lambda blob: natural_key(Path(blob.name).name))
    if not blobs:
        raise ValueError(f"no {task} input files found under {input_prefix}")
    names = [Path(blob.name).name for blob in blobs]
    if len(set(names)) != len(names):
        raise ValueError("input folder contains duplicate basenames; use unique file names")

    task_obj = resolve_task(task)
    model = task_model(task)
    rows: list[dict] = []
    expected_cars: dict[str, list[str]] = {}
    with TemporaryDirectory(prefix="nebulax-batch-") as tmp:
        folder = Path(tmp)
        for blob in blobs:
            path = folder / Path(blob.name).name
            blob.download_to_filename(str(path))
            try:
                result = run_file(task_obj, path, model)
                rows.extend(task_obj.to_rows(result))
                if task == "acv":
                    expected_cars[path.name] = acv_car_ids(path)
            finally:
                path.unlink(missing_ok=True)
        if not rows:
            raise ValueError("no predictions were produced")
        csv_path = write_csv(task, rows, folder / OUTPUT_FILENAMES[task])
        report = validate_csv(
            task, csv_path, names if task in ("rail", "shm") else None,
            expected_cars=expected_cars or None,
        )
        if not report.ok:
            raise ValueError("invalid prediction CSV: " + "; ".join(report.errors))
        # A retry cannot silently replace a previous result. Use a new output URI per run.
        client.bucket(target_bucket).blob(target_name).upload_from_filename(
            str(csv_path), content_type="text/csv", if_generation_match=0,
        )
    return {"task": task, "input_files": len(blobs), "rows": len(rows), "output_uri": output_uri}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=os.getenv("PS3_TASK"), choices=list(TASK_NAMES))
    parser.add_argument("--input-prefix", default=os.getenv("PS3_INPUT_PREFIX"))
    parser.add_argument("--output-uri", default=os.getenv("PS3_OUTPUT_URI"))
    args = parser.parse_args(argv)
    if not all((args.task, args.input_prefix, args.output_uri)):
        parser.error("set PS3_TASK, PS3_INPUT_PREFIX, PS3_OUTPUT_URI or pass all three flags")
    print(json.dumps(run(args.task, args.input_prefix, args.output_uri)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
