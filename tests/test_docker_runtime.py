from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DockerRuntimeTests(unittest.TestCase):
    def test_lightgbm_openmp_runtime_is_in_both_images(self) -> None:
        for name in ("Dockerfile", "Dockerfile.batch"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("libgomp1", text, f"{name} must install LightGBM's OpenMP runtime")
