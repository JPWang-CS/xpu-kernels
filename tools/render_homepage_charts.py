"""Prepare verified homepage chart values and an ImageGen brief, or check saved assets."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from render_native_comparisons import load, mhc_records
from render_rms_norm import measurements

ASSETS = ROOT / "docs/assets"
MANIFEST = ASSETS / "performance-chart-data.json"
BRIEF = ASSETS / "benchmark-rmsnorm.prompt.txt"
COLORS = {"native": "#94A3B8", "optimized": "#0D9488", "gpu": "#2563EB"}


def chart_data():
    mhc, _ = mhc_records()
    gpu = load("mhc-post-backward-gpu-ncu.jsonl", "Nsight Compute gpu__time_duration.sum", "mhc_post_backward")
    series = [(mhc, "native", "NPU 950DT（原生）", COLORS["native"]),
              (mhc, "optimized", "NPU 950DT（优化）", COLORS["optimized"]),
              (gpu, "tilelang", "GPU L20X（TileLang）", COLORS["gpu"])]
    cases = []
    for shape in ((2048, 4, 2560), (2048, 4, 6144), (8192, 4, 6144), (32768, 4, 6144)):
        rows = [index[(shape, "bfloat16", impl)] for index, impl, _, _ in series]
        if any(r["correctness"]["status"] != "PASS" for r in rows):
            raise ValueError("unverified mHC chart data")
        values = [r.get("latency_us_median", r.get("latency_ms_median", 0) * 1000) for r in rows]
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("invalid latency")
        cases.append({"shape": list(shape), "latency_us": values})
    shared = {"dtype": "bfloat16", "metric": "absolute_latency_us", "lower_is_better": True}
    charts = [{**shared, "id": "mhc", "title": "mHC Post Backward", "method": "BF16 · Kernel time · μs",
               "series": [{"implementation": impl, "label": label, "color": color} for _, impl, label, color in series],
               "cases": cases},
              {**shared, "id": "rmsnorm", "title": "RMSNorm", "method": "BF16 · Event call time · μs",
               "series": [{"implementation": "native", "label": "NPU 950DT（原生）", "color": COLORS["native"]},
                          {"implementation": "optimized", "label": "NPU 950DT（优化）", "color": COLORS["optimized"]}],
               "cases": measurements()}]
    return {"schema_version": 3, "sources": ["results/benchmarks/mhc-post-backward-msprof.jsonl",
            "results/benchmarks/mhc-post-backward-gpu-ncu.jsonl", "results/benchmarks/rmsnorm-npu-cann.jsonl"],
            "charts": charts}


def image_brief(data):
    chart = next(c for c in data["charts"] if c["id"] == "rmsnorm")
    lines = ["ImageGen chart brief: RMSNorm", "Wide, restrained professional GitHub README chart; white background.",
             "Title: RMSNorm. Subtitle: Ascend 950DT · BF16 · Event time.",
             "Two side-by-side panels, horizontal bars; shared linear x-axis 0..240 μs, ticks 0,60,120,180,240.",
             "Gray #94A3B8 原生; teal #0D9488 优化; labels at bar ends. No GPU series."]
    for case in chart["cases"]:
        values = case["latency_us"]
        lines.append(f"Panel {' × '.join(map(str, case['shape']))}: 原生 {values[0]:.2f} μs; 优化 {values[1]:.2f} μs.")
    lines.append("Footer: 单位：μs/次 · 越低越快. Inspect text, units, and bar geometry before publishing.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = chart_data()
    if args.check:
        if json.loads(MANIFEST.read_text(encoding="utf-8")) != data or BRIEF.read_text(encoding="utf-8") != image_brief(data):
            raise SystemExit("chart data/brief no longer matches measurements")
        for chart in data["charts"]:
            raw = (ASSETS / f"benchmark-{chart['id']}.png").read_bytes()
            if raw[:8] != b"\x89PNG\r\n\x1a\n":
                raise SystemExit("invalid chart PNG")
    else:
        MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        BRIEF.write_text(image_brief(data), encoding="utf-8", newline="\n")
        print("Use the generated brief with ImageGen and inspect the saved chart before publishing.")
    print("Two charts: six shape panels and sixteen matched measurements; data check PASS")


if __name__ == "__main__":
    main()
