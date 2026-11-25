"""Keep operator additions on the published layout and exercise rejection paths."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("layout_check", ROOT / "tools/check_repository_layout.py")
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


class RepositoryLayoutTests(unittest.TestCase):
    def test_current_operators_and_includes_follow_layout(self):
        self.assertEqual(layout.check(), [])

    def test_exporting_an_operator_without_its_files_is_rejected(self):
        original = Path.read_text

        def changed(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            if path == ROOT / "src/xpu_kernels/__init__.py":
                text += "\nfrom .missing_operator import missing_operator\n"
            return text

        with patch.object(Path, "read_text", changed):
            errors = layout.check()
        self.assertIn("missing src/xpu_kernels/missing_operator_golden.py", errors)
        self.assertIn("no backend for missing_operator", errors)

    def test_broken_local_include_is_rejected(self):
        original = Path.read_text

        def changed(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            if path == ROOT / "kernels/include/xpu_kernels/rms_norm/kernel.hpp":
                text += "\n#include <xpu_kernels/rms_norm/missing.hpp>\n"
            return text

        with patch.object(Path, "read_text", changed):
            self.assertTrue(any("missing.hpp" in e for e in layout.check()))

    def test_missing_operator_design_document_is_rejected(self):
        original = Path.is_file

        def missing(path):
            if path == ROOT / "docs/operators/rms_norm/optimization.md":
                return False
            return original(path)

        with patch.object(Path, "is_file", missing):
            self.assertIn("missing docs/operators/rms_norm/optimization.md", layout.check())

    def test_public_document_cannot_delegate_usage_to_a_skill_link(self):
        original = Path.read_text

        def changed(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            if path == ROOT / "README.md":
                text += "\n[Usage](skills/xpu-kernels/SKILL.md)\n"
            return text

        with patch.object(Path, "read_text", changed):
            self.assertTrue(any("public doc links to internal instructions: README.md" in e for e in layout.check()))


if __name__ == "__main__":
    unittest.main()
