"""Plot homepage bars directly from verified benchmark records."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from render_native_comparisons import load, mhc_records, rms_records

ASSETS = ROOT / "docs/assets"
MANIFEST = ASSETS / "performance-chart-data.json"
COLORS = {"baseline": "#94A3B8", "optimized": "#0D9488", "triton": "#7C3AED"}


def chart_data():
    mhc, _ = mhc_records()
    rms, _ = rms_records()
    gpu = load("rmsnorm-gpu-v2.jsonl", "torch.cuda.Event.elapsed_time", "rms_norm")
    rms_shapes = ((128,4096), (129,4097), (1024,4096), (4096,8192))
    specs = (
        ("mhc", "mHC Post Backward", "NPU 950DT · BF16", "原生", "msprof · Task Duration",
         mhc, ((2048,4,2560), (2048,4,6144), (8192,4,6144), (32768,4,6144)),
         (("native","CANN 原生","baseline"),("optimized","xpu-kernels","optimized")), 4.0),
        ("rms-gpu", "RMSNorm · GPU", "GPU L20X · BF16", "单算子拼接", "Event · Eager",
         gpu, rms_shapes, (("torch","单算子拼接","baseline"),("cuda","xpu-kernels","optimized"),
                          ("triton","Triton","triton")), 16.0),
        ("rms-npu", "RMSNorm · NPU", "NPU 950DT · BF16", "原生", "Event · Eager",
         rms, rms_shapes, (("native","CANN 原生","baseline"),("ascendc","xpu-kernels","optimized")), 1.5),
    )
    charts = []
    for key, title, hardware, baseline, method, index, shapes, implementations, ymax in specs:
        chart = {"id": key, "title": title, "hardware": hardware, "baseline": baseline,
            "method": method, "metric": "baseline_latency / implementation_latency", "higher_is_better": True,
            "ymax": ymax, "colors": COLORS, "cases": [], "series": []}
        for impl, label, color in implementations:
            chart["series"].append({"implementation": impl, "label": label, "color": COLORS[color]})
        for shape in shapes:
            rows = [index[(shape,"bfloat16",impl)] for impl, _, _ in implementations]
            if any(any(row[k] != rows[0][k] for k in ("hardware","torch","runtime","measurement_method")) for row in rows):
                raise ValueError("incompatible chart measurements")
            times = [r.get("latency_us_median", r.get("latency_ms_median",0)*1000) for r in rows]
            speeds = [times[0]/t for t in times]
            if any(not math.isfinite(v) or v <= 0 or v > ymax for v in speeds):
                raise ValueError("invalid or clipped chart bar")
            chart["cases"].append({"shape": list(shape), "latency_us": times, "speedup": speeds})
        charts.append(chart)
    return {"schema_version": 1, "sources": ["results/benchmarks/mhc-post-backward-msprof.jsonl",
        "results/benchmarks/rmsnorm-gpu-v2.jsonl", "results/benchmarks/rmsnorm-npu-mainline.jsonl"],
        "charts": charts}


def plot(data, font_path=None):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FuncFormatter

    if font_path is None:
        candidate = Path("C:/Windows/Fonts/msyh.ttc")
        if candidate.is_file():
            font_path = candidate
    if font_path:
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        family = next((f.name for f in font_manager.fontManager.ttflist
                       if f.name in ("Noto Sans CJK SC","Source Han Sans SC","Microsoft YaHei")), None)
        if family is None:
            raise RuntimeError("provide a Chinese font with --font /path/to/font.otf")
    plt.rcParams.update({"font.family": [family,"DejaVu Sans"], "font.size": 13,
        "axes.unicode_minus": False, "svg.fonttype": "none"})
    ASSETS.mkdir(parents=True, exist_ok=True)
    for chart in data["charts"]:
        fig = plt.figure(figsize=(14,4.8), dpi=160, facecolor="#FAFCFE")
        ax = fig.add_axes([0.075,0.20,0.90,0.54], facecolor="#FAFCFE")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.7)
        for side in ("top","right","left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.tick_params(axis="both", length=0, colors="#475569", pad=10)
        ax.set_ylim(0, chart["ymax"])
        ax.set_xlim(-0.5, len(chart["cases"])-0.5)
        ax.yaxis.set_major_locator(MultipleLocator(0.5 if chart["id"]=="rms-npu" else 2 if chart["id"]=="rms-gpu" else 1))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
        ax.axhline(1, color="#64748B", linewidth=1.0, linestyle=(0,(3,4)), zorder=1)
        count = len(chart["series"])
        width = 0.20 if count == 3 else 0.25
        for j, series in enumerate(chart["series"]):
            positions = [i+(j-(count-1)/2)*width for i in range(len(chart["cases"]))]
            values = [case["speedup"][j] for case in chart["cases"]]
            bars = ax.bar(positions, values, width*0.88, color=series["color"],
                label=series["label"], zorder=3)
            for rect, value in zip(bars, values):
                ax.annotate(f"{value:.2f}×", (rect.get_x()+rect.get_width()/2, value),
                    xytext=(0,6), textcoords="offset points", ha="center", va="bottom",
                    fontsize=12.5, color="#0F172A", fontweight="medium")
        labels = [" × ".join(map(str,c["shape"])) for c in chart["cases"]]
        ax.set_xticks(range(len(labels)), labels, fontsize=13)
        fig.text(0.075,0.92,chart["title"], fontsize=23, color="#0F172A", fontweight="bold")
        fig.text(0.075,0.835,chart["hardware"], fontsize=13, color="#64748B")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0,1.11), ncol=count,
            frameon=False, handlelength=1.0, handleheight=0.85, columnspacing=1.6, fontsize=13)
        fig.text(0.075,0.045,f"相对{chart['baseline']}的加速比 · 越高越快", fontsize=12, color="#475569")
        fig.text(0.975,0.045,chart["method"], ha="right", fontsize=12, color="#64748B")
        fig.savefig(ASSETS/f"benchmark-{chart['id']}.png", dpi=160, facecolor=fig.get_facecolor(),
            metadata={"Software":"xpu-kernels benchmark renderer"})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    data = chart_data()
    if args.check:
        if json.loads(MANIFEST.read_text(encoding="utf-8")) != data:
            raise SystemExit("chart data no longer matches benchmarks; regenerate the plots")
        for c in data["charts"]:
            raw = (ASSETS/f"benchmark-{c['id']}.png").read_bytes()
            if raw[:8] != b"\x89PNG\r\n\x1a\n":
                raise SystemExit("invalid chart PNG")
    else:
        plot(data, args.font)
        MANIFEST.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n", encoding="utf-8")
    print("Three charts match 12 shape groups and 28 implementation measurements")


if __name__ == "__main__":
    main()
