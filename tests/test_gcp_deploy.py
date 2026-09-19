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
    def test_defaults_are_all_public_and_autoscale(self) -> None:
        args = deploy.parser().parse_args([])
        self.assertIsNone(args.target)
        self.assertIsNone(args.target_flag)
        self.assertTrue(args.public)
        self.assertTrue(args.autoscale)

    def test_flag_forms(self) -> None:
        args = deploy.parser().parse_args(["--batch", "--private", "--pause"])
        self.assertEqual(args.target_flag, "batch")
        self.assertFalse(args.public)
        self.assertFalse(args.autoscale)

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

    def test_web_deploy_sets_access_scaling_and_returns_url(self) -> None:
        calls: list[tuple[str, ...]] = []

        def fake_call(_gcloud: str, *args: str, capture: bool = False) -> str:
            calls.append(args)
            if args[:4] == ("run", "services", "describe", "nebulax-app"):
                return "https://example.run.app"
            return ""

        with patch.object(deploy, "call", side_effect=fake_call), patch.object(
            deploy, "digest_ref", return_value="registry/app@sha256:abc"
        ):
            image, url = deploy.deploy_web("gcloud", "project", "region", public=True, autoscale=True)
        self.assertEqual(image, "registry/app@sha256:abc")
        self.assertEqual(url, "https://example.run.app")
        command = next(args for args in calls if args[:3] == ("run", "deploy", "nebulax-app"))
        self.assertIn("--no-invoker-iam-check", command)
        self.assertEqual(command[command.index("--scaling") + 1], "auto")
