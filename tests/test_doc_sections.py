"""Check generated performance updates preserve explanations and moved links."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("doc_sections", ROOT / "tools/doc_sections.py")
sections = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sections)


class DocumentSectionTests(unittest.TestCase):
    def test_measurement_update_preserves_design_and_trailing_text(self):
        parent = ROOT / "tmp"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            target = Path(folder) / "optimization.md"
            target.write_text("# Design\nKeep code and figure.\n" + sections.BEGIN +
                              "\nold data\n" + sections.END + "\nKeep notes.\n", encoding="utf-8")
            result = sections.update_measured_section(target, "new measured data")
        self.assertIn("Keep code and figure.", result)
        self.assertIn("Keep notes.", result)
        self.assertIn("new measured data", result)
        self.assertNotIn("old data", result)

    def test_moved_links_resolve_to_new_operator_folder_and_original_data(self):
        body = sections.measured_body("# Results\n## Data\n[raw](../../results/x.jsonl) "
            "[api](../operators/rms_norm.md)", "docs/performance/rms_norm.md",
            "docs/operators/rms_norm/optimization.md")
        self.assertIn("### Data", body)
        self.assertIn("[raw](../../../results/x.jsonl)", body)
        self.assertIn("[api](README.md)", body)

    def test_missing_marker_is_rejected(self):
        parent = ROOT / "tmp"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            target = Path(folder) / "optimization.md"
            target.write_text("# Explanation", encoding="utf-8")
            with self.assertRaises(ValueError):
                sections.update_measured_section(target, "data")


if __name__ == "__main__":
    unittest.main()
