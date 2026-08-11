"""Regression tests for dependencies imported by released-model inference."""

from pathlib import Path
import unittest


class RuntimeRequirementsTests(unittest.TestCase):
    def test_released_model_declares_kornia_dependency(self) -> None:
        requirements = (
            Path(__file__).resolve().parents[1] / "requirements.txt"
        ).read_text(encoding="utf-8")
        declared_packages = {
            line.split("==", 1)[0].strip().lower()
            for line in requirements.splitlines()
            if line.strip() and not line.lstrip().startswith(("#", "-"))
        }

        self.assertIn("kornia", declared_packages)


if __name__ == "__main__":
    unittest.main()
