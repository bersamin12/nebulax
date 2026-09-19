from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / "scripts" / "gcp_deploy.py"
SPEC = importlib.util.spec_from_file_location("gcp_deploy", PATH)
assert SPEC and SPEC.loader
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


class GcpDeployTests(unittest.TestCase):
    def test_digest_ref_replaces_tag(self) -> None:
        with patch.object(deploy, "call", return_value="sha256:abc"):
            self.assertEqual(
                deploy.digest_ref("gcloud", "us-central1-docker.pkg.dev/project/repo/app:dev"),
                "us-central1-docker.pkg.dev/project/repo/app@sha256:abc",
            )

    def test_digest_ref_rejects_bad_output(self) -> None:
        with patch.object(deploy, "call", return_value="latest"):
            with self.assertRaisesRegex(RuntimeError, "invalid digest"):
                deploy.digest_ref("gcloud", "registry/app:dev")

    def test_call_uses_repo_root(self) -> None:
        completed = subprocess.CompletedProcess(["gcloud"], 0, stdout="ok\n")
        with patch.object(deploy.subprocess, "run", return_value=completed) as run:
            self.assertEqual(deploy.call("gcloud", "version", capture=True), "ok")
        self.assertEqual(run.call_args.kwargs["cwd"], deploy.ROOT)
        self.assertTrue(run.call_args.kwargs["check"])
