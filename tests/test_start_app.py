"""The launcher accepts only the Anaconda channels named by Conda's ToS error."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / "scripts" / "start_app.py"
SPEC = importlib.util.spec_from_file_location("start_app", PATH)
assert SPEC and SPEC.loader
start_app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(start_app)


class StartAppTests(unittest.TestCase):
    def test_conda_python_location(self) -> None:
        expected = start_app.CONDA_ENV / ("python.exe" if os.name == "nt" else "bin/python")
        self.assertEqual(start_app.env_python(start_app.CONDA_ENV), expected)
        self.assertEqual(start_app.env_root(expected), start_app.CONDA_ENV)

    def test_conda_tos_channels_are_accepted_then_create_retried(self) -> None:
        error = """CondaToSNonInteractiveError: Terms of Service have not been accepted:
    - https://repo.anaconda.com/pkgs/main
    - https://repo.anaconda.com/pkgs/r
    - https://repo.anaconda.com/pkgs/msys2
"""
        first = subprocess.CompletedProcess(["conda", "create"], 1, "", error)
        with patch.object(start_app.subprocess, "run", return_value=first), patch.object(start_app, "run") as retry:
            start_app.create_conda_environment("conda")
        calls = [call.args[0] for call in retry.call_args_list]
        self.assertEqual(calls[:3], [
            ["conda", "tos", "accept", "--override-channels", "--channel", channel]
            for channel in (
                "https://repo.anaconda.com/pkgs/main",
                "https://repo.anaconda.com/pkgs/r",
                "https://repo.anaconda.com/pkgs/msys2",
            )
        ])
        self.assertEqual(calls[3][:2], ["conda", "create"])

    def test_other_conda_failure_does_not_accept_terms(self) -> None:
        first = subprocess.CompletedProcess(["conda", "create"], 1, "", "PackagesNotFoundError")
        with patch.object(start_app.subprocess, "run", return_value=first), patch.object(start_app, "run") as retry:
            with self.assertRaises(subprocess.CalledProcessError):
                start_app.create_conda_environment("conda")
        retry.assert_not_called()
