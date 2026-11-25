"""Render each operator with the same Matplotlib style and verified benchmark data."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from render_native_comparisons import load, mhc_records
from render_rms_norm import measurements

ASSETS = ROOT / "docs/assets"
MANIFEST = ASSETS / "performance-chart-data.json"
COLORS = {"native": "#94A3B8", "optimized": "#0D9488", "gpu": "#2563EB"}
STYLE = {"columns": 2, "width_inches": 14, "panel_height_inches": 2.6,
         "panel_gap_inches": 1.0, "header_inches": 1.95, "footer_inches": .5,
         "dpi": 160, "background": "#FAFCFE",
         "text": "#0F172A", "muted": "#64748B", "grid": "#E2E8F0",
         "title_size": 24, "panel_title_size": 14, "label_size": 12.5,
         "value_decimals": 2, "bar_width": 0.60, "colors": COLORS}


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
    return {"schema_version": 4, "style": STYLE, "sources": ["results/benchmarks/mhc-post-backward-msprof.jsonl",
            "results/benchmarks/mhc-post-backward-gpu-ncu.jsonl", "results/benchmarks/rmsnorm-npu-cann.jsonl"],
            "charts": charts}


def plot(data, font_path=None):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    if font_path is None and Path("C:/Windows/Fonts/msyh.ttc").is_file():
        font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path:
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        family = next((f.name for f in font_manager.fontManager.ttflist if f.name in
                       ("Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei")), None)
        if family is None:
            raise RuntimeError("provide an installed Chinese font using --font")
    style = data["style"]
    plt.rcParams.update({"font.family": [family, "DejaVu Sans"], "font.size": style["label_size"],
                         "axes.unicode_minus": False})
    for chart in data["charts"]:
        rows = math.ceil(len(chart["cases"]) / style["columns"])
        height = (style["header_inches"] + style["footer_inches"]
                  + rows * style["panel_height_inches"] + (rows-1)*style["panel_gap_inches"])
        fig, axes = plt.subplots(rows, style["columns"], squeeze=False,
                                 figsize=(style["width_inches"], height), dpi=style["dpi"],
                                 facecolor=style["background"])
        fig.subplots_adjust(left=.075, right=.975, top=1-style["header_inches"]/height,
                            bottom=style["footer_inches"]/height, wspace=.22,
                            hspace=style["panel_gap_inches"]/style["panel_height_inches"])
        fig.text(.075, 1-.43/height, chart["title"], fontsize=style["title_size"],
                 fontweight="bold", color=style["text"])
        fig.text(.075, 1-.81/height, chart["method"], color=style["muted"])
        handles = [Patch(facecolor=s["color"], label=s["label"]) for s in chart["series"]]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.066, 1-.99/height),
                   ncol=3, frameon=False, handlelength=1.0, columnspacing=1.8)
        for ax, case in zip(axes.flat, chart["cases"]):
            ax.set_facecolor(style["background"])
            ax.set_axisbelow(True)
            ax.grid(axis="y", color=style["grid"], linewidth=.7)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#CBD5E1")
            ax.tick_params(axis="both", length=0, labelcolor=style["muted"])
            bars = ax.bar(range(len(chart["series"])), case["latency_us"], width=style["bar_width"],
                          color=[s["color"] for s in chart["series"]], zorder=3)
            ax.set_ylim(0, max(case["latency_us"])*1.26)
            ax.set_xticks([])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            ax.set_ylabel("μs", color=style["muted"])
            ax.set_title(" × ".join(map(str, case["shape"])), loc="left", pad=12,
                         fontsize=style["panel_title_size"], color=style["text"])
            for bar, value in zip(bars, case["latency_us"]):
                if bar.get_height() != value:
                    raise ValueError("bar geometry differs from benchmark data")
                ax.annotate(f"{value:.{style['value_decimals']}f}",
                            (bar.get_x()+bar.get_width()/2, value), xytext=(0,5),
                            textcoords="offset points", ha="center", va="bottom", color=style["text"])
        for ax in list(axes.flat)[len(chart["cases"]):]:
            ax.set_visible(False)
        fig.text(.075, .13/height, "各面板独立刻度 · μs/次 · 越低越快", fontsize=11, color=style["muted"])
        fig.savefig(ASSETS / f"benchmark-{chart['id']}.png", dpi=style["dpi"],
                    facecolor=fig.get_facecolor(), metadata={"Software": "xpu-kernels shared Matplotlib renderer",
                    "BenchmarkData": json.dumps({"style": style, "chart": chart}, ensure_ascii=False, sort_keys=True)})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    data = chart_data()
    if args.check:
        if json.loads(MANIFEST.read_text(encoding="utf-8")) != data:
            raise SystemExit("chart data no longer matches measurements")
        from PIL import Image
        for chart in data["charts"]:
            with Image.open(ASSETS / f"benchmark-{chart['id']}.png") as saved:
                if saved.format != "PNG" or json.loads(saved.info.get("BenchmarkData", "null")) != {"style": STYLE, "chart": chart}:
                    raise SystemExit(f"saved {chart['id']} chart no longer matches data/style")
    else:
        plot(data, args.font)
        MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Two charts: six shape panels and sixteen matched measurements; data check PASS")


if __name__ == "__main__":
    main()
