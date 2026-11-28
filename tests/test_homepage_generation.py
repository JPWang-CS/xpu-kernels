"""Regress the generated README losing a newly published performance section."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from render_native_comparisons import BEGIN, END, homepage


class HomepageGenerationTests(unittest.TestCase):
    def test_generated_performance_preserves_all_three_groups(self):
        generated = homepage()
        for title in ("### RMSNorm", "### mHC Post Backward", "### MoE · 固定容量路径"):
            self.assertIn(title, generated)
        self.assertIn("docs/assets/benchmark-moe.png", generated)
        self.assertIn("整网设备执行区间", generated)
        self.assertIn("TP2/DP1/EP2、MTP3", generated)

    def test_readme_uses_the_generated_performance_block(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        block = text.split(BEGIN, 1)[1].split(END, 1)[0].strip()
        self.assertEqual(block, homepage())
