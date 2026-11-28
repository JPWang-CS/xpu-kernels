"""Checks for the historical fixed-capacity MoE homepage chart contract."""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from tools.render_homepage_charts import ROOT, chart_data, moe_measurements


class MoEHomepageChartTests(unittest.TestCase):
    def test_seconds_are_converted_from_microseconds(self):
        cases = {case["case"]: case for case in moe_measurements()}
        self.assertEqual(cases["S3"]["case_label"], "输入 188 · 输出 1024")
        self.assertEqual(cases["S3"]["seconds"], [72.037833596, 22.108768173])
        for case in cases.values():
            self.assertTrue(all(math.isfinite(value) and value > 0 for value in case["seconds"]))

    def test_chart_scope_and_sources(self):
        data = chart_data()
        chart = next(item for item in data["charts"] if item["id"] == "moe")
        self.assertEqual(chart["value_key"], "seconds")
        self.assertEqual(chart["unit"], "s")
        self.assertEqual(chart["note"], "固定容量通信 + 三段融合内核 + 独立 EP + 图适配")
        self.assertEqual([case["case"] for case in chart["cases"]], ["S3", "S6", "S8", "S9"])
        self.assertIn("results/benchmarks/moe-fixed-capacity-historical.jsonl", data["sources"])
        self.assertNotIn("D:\\", json.dumps(data, ensure_ascii=False))
        self.assertNotIn("/home/", json.dumps(data, ensure_ascii=False))

    def test_source_has_exact_historical_scope(self):
        path = ROOT / "results/benchmarks/moe-fixed-capacity-historical.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row["hardware"] == "2 x Ascend 910B3" for row in rows))
        self.assertTrue(all(row["model"] == "Pangu 92B" for row in rows))
        self.assertTrue(all(row["expert_weight_dtype"] == "int8" for row in rows))
        self.assertTrue(all(row["hidden_shape"] == [16, 2560] and row["top_k"] == 8 for row in rows))
        self.assertTrue(all(row["precision_21"] == "NOT_RUN" for row in rows))
        self.assertTrue(all(row["suite_precision"] == "UNRESOLVED_S7_TOKEN_MISMATCH" for row in rows))


if __name__ == "__main__":
    unittest.main()
