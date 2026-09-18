"""Offline unit tests for scripts/download_data.py.

No network access: these only exercise the pure helpers (hashing, size
formatting, manifest round-tripping, idempotent-skip logic) so the suite
stays well under the 60s budget with the repo's real datasets untouched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "download_data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("download_data", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["download_data"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dd():
    return _load_module()


def test_fmt_size_units(dd):
    assert dd.fmt_size(0) == "0.0B"
    assert dd.fmt_size(512) == "512.0B"
    assert dd.fmt_size(1536) == "1.5KB"
    assert dd.fmt_size(3 * 1024**3) == "3.0GB"


def test_sha256_of_matches_hashlib(dd, tmp_path):
    import hashlib

    p = tmp_path / "sample.bin"
    payload = b"nebulax" * 10000
    p.write_bytes(payload)
    assert dd.sha256_of(p) == hashlib.sha256(payload).hexdigest()


def test_download_skips_when_size_matches(dd, tmp_path, monkeypatch):
    dest = tmp_path / "existing.bin"
    dest.write_bytes(b"x" * 42)

    def fail_get(*args, **kwargs):
        raise AssertionError("network should not be touched when size already matches")

    monkeypatch.setattr(dd.requests, "get", fail_get)
    transferred = dd.download("http://example.invalid/file", dest, expected_size=42)
    assert transferred is False
    assert dest.read_bytes() == b"x" * 42


def test_load_prior_manifest_roundtrip(dd, tmp_path):
    ds_dir = tmp_path / "somewhere"
    ds_dir.mkdir()
    assert dd.load_prior_manifest(ds_dir) is None

    manifest = {"dataset": "x", "status": "ok", "files": [{"path": "a.csv", "size": 3, "sha256": "abc"}]}
    (ds_dir / "MANIFEST.json").write_text(json.dumps(manifest))
    loaded = dd.load_prior_manifest(ds_dir)
    assert loaded == manifest


def test_files_already_present(dd, tmp_path):
    ds_dir = tmp_path / "ds"
    ds_dir.mkdir()
    (ds_dir / "a.csv").write_bytes(b"12345")
    manifest = {"files": [{"path": "a.csv", "size": 5}]}
    assert dd.files_already_present(manifest, ds_dir) is True

    manifest_wrong_size = {"files": [{"path": "a.csv", "size": 999}]}
    assert dd.files_already_present(manifest_wrong_size, ds_dir) is False

    manifest_missing = {"files": [{"path": "missing.csv", "size": 1}]}
    assert dd.files_already_present(manifest_missing, ds_dir) is False

    assert dd.files_already_present({"files": []}, ds_dir) is False


def test_looks_like_bot_challenge(dd):
    assert dd._looks_like_bot_challenge("<title>Just a moment...</title>") is True
    assert dd._looks_like_bot_challenge("<html><body>hello dataset</body></html>") is False


def test_manifest_to_json_shape(dd):
    m = dd.DatasetManifest(dataset="ottawa", status="ok", source_url="http://x", licence="CC BY 4.0")
    m.files.append(dd.FileEntry(path="a.csv", size=10, sha256="deadbeef", source_url="http://x/a.csv"))
    data = m.to_json()
    assert data["dataset"] == "ottawa"
    assert data["files"] == [{"path": "a.csv", "size": 10, "sha256": "deadbeef", "source_url": "http://x/a.csv"}]
    assert "generated_at" in data


def test_dataset_order_and_funcs_agree(dd):
    assert set(dd.DATASET_ORDER) == set(dd.DATASET_FUNCS.keys())
    parser_choices = dd.build_parser()._option_string_actions["--dataset"].choices
    assert set(parser_choices) == {"all", *dd.DATASET_ORDER}
