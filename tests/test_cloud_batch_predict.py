"""Storage boundary and publish-on-valid-output checks for the Cloud Run batch worker."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cloud_batch_predict.py"
SPEC = importlib.util.spec_from_file_location("cloud_batch_predict", SCRIPT)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class Blob:
    def __init__(self, name: str, data: bytes = b"1\n2\n") -> None:
        self.name = name
        self.data = data
        self.uploaded: bytes | None = None

    def download_to_filename(self, filename: str) -> None:
        Path(filename).write_bytes(self.data)

    def upload_from_filename(self, filename: str, **kwargs) -> None:
        assert kwargs == {"content_type": "text/csv", "if_generation_match": 0}
        self.uploaded = Path(filename).read_bytes()


class Storage:
    def __init__(self, names: list[str]) -> None:
        self.inputs = [Blob(name) for name in names]
        self.output = Blob("output")

    def list_blobs(self, bucket: str, *, prefix: str):
        assert (bucket, prefix) == ("source", "run-1/")
        return self.inputs

    def bucket(self, bucket: str):
        assert bucket == "output"
        return self

    def blob(self, name: str) -> Blob:
        assert name == "run-1/shm_predictions.csv"
        return self.output


class BatchPredictTests(unittest.TestCase):
    def test_rejects_output_inside_input_folder(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the input folder"):
            worker.run("shm", "gs://source/run-1/", "gs://source/run-1/out.csv", client=Storage([]))

    def test_rejects_duplicate_basenames_before_predicting(self) -> None:
        storage = Storage(["run-1/a/x.csv", "run-1/b/x.csv"])
        with self.assertRaisesRegex(ValueError, "duplicate basenames"):
            worker.run("shm", "gs://source/run-1/", "gs://output/run-1/shm_predictions.csv", client=storage)

    def test_uploads_only_validated_complete_csv(self) -> None:
        storage = Storage(["run-1/file2.csv", "run-1/file1.csv", "run-1/readme.txt"])
        class Task:
            @staticmethod
            def to_rows(path: Path):
                return [{"file_id": path.name, "prediction": 0.2}]

        with patch.object(worker, "resolve_task", return_value=Task()), \
             patch.object(worker, "task_model", return_value=object()), \
             patch.object(worker, "run_file", side_effect=lambda _task, path, _model: path):
            result = worker.run("shm", "gs://source/run-1/", "gs://output/run-1/shm_predictions.csv", client=storage)
        self.assertEqual(result["input_files"], 2)
        self.assertEqual(result["rows"], 2)
        self.assertIsNotNone(storage.output.uploaded)
        self.assertIn(b"file1.csv,0.2\nfile2.csv,0.2", storage.output.uploaded)

    def test_failure_does_not_publish_partial_output(self) -> None:
        storage = Storage(["run-1/file1.csv", "run-1/file2.csv"])
        with patch.object(worker, "resolve_task", return_value=object()), \
             patch.object(worker, "task_model", return_value=object()), \
             patch.object(worker, "run_file", side_effect=RuntimeError("bad input")):
            with self.assertRaisesRegex(RuntimeError, "bad input"):
                worker.run("shm", "gs://source/run-1/", "gs://output/run-1/shm_predictions.csv", client=storage)
        self.assertIsNone(storage.output.uploaded)


if __name__ == "__main__":
    unittest.main()
