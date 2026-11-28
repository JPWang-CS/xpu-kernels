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


def _positive_finite(value, field):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be positive and finite")
    return value


def moe_measurements():
    path = ROOT / "results/benchmarks/moe-fixed-capacity-historical.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 5:
        raise ValueError("MoE historical chart requires five cases")
    required_scope = {
        "record_kind": "historical_model_device_interval",
        "operator_group": "fixed_capacity_moe",
        "hardware": "2 x Ascend 910B3",
        "timing_scope": "whole_model_device_business_span_slowest_rank",
        "comparison_scope": "historical combined deployment; not independent kernel timing or migrated JIT measurement",
        "precision_21": "NOT_RUN",
        "suite_precision": "UNRESOLVED_S7_TOKEN_MISMATCH",
        "formal_windows_per_arm": 1,
    }
    cases = []
    seen = set()
    for row in rows:
        for key, expected in required_scope.items():
            if row.get(key) != expected:
                raise ValueError(f"invalid MoE {key}: {row.get(key)!r}")
        case = row.get("case")
        if not isinstance(case, str) or not case or case in seen:
            raise ValueError("MoE cases must be unique non-empty strings")
        seen.add(case)
        if row.get("dtype") != "bfloat16" or row.get("execution_mode") != "FULL_DECODE_ONLY":
            raise ValueError("unexpected MoE dtype or execution mode")
        if row.get("model") != "Pangu 92B" or row.get("expert_weight_dtype") != "int8":
            raise ValueError("unexpected MoE model or expert weight dtype")
        if row.get("hidden_shape") != [16, 2560] or row.get("top_k") != 8:
            raise ValueError("unexpected MoE hidden shape or top_k")
        if row.get("concurrency") != 1 or row.get("parallelism") != {"tp": 2, "dp": 1, "ep": 2, "mtp": 3}:
            raise ValueError("unexpected MoE workload scope")
        if not isinstance(row.get("input_tokens"), int) or not isinstance(row.get("output_tokens"), int):
            raise ValueError("MoE token counts must be integers")
        baseline = _positive_finite(row.get("baseline_us"), "baseline_us") / 1_000_000
        candidate = _positive_finite(row.get("candidate_us"), "candidate_us") / 1_000_000
        cases.append({"case": case, "case_label": f"输入 {row['input_tokens']} · 输出 {row['output_tokens']}",
                      "seconds": [baseline, candidate]})
    selected = [case for case in ("S3", "S6", "S8", "S9") if case in seen]
    if len(selected) != 4:
        raise ValueError("MoE chart requires S3, S6, S8 and S9")
    return [next(item for item in cases if item["case"] == case) for case in selected]


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
               "cases": measurements()},
              {"dtype": "bfloat16", "metric": "absolute_latency_s", "lower_is_better": True,
               "id": "moe", "title": "MoE · 固定容量路径",
               "method": "Pangu 92B · 2×910B3 · 历史整网设备区间 · s · n=1",
               "note": "固定容量通信 + 三段融合内核 + 独立 EP + 图适配",
               "footer": "各面板独立刻度 · s/次 · 越低越快", "value_key": "seconds", "unit": "s",
               "series": [{"implementation": "native", "label": "原生路径", "color": COLORS["native"]},
                          {"implementation": "optimized", "label": "组合路径", "color": COLORS["optimized"]}],
               "cases": moe_measurements()}]
    return {"schema_version": 5, "style": STYLE, "sources": ["results/benchmarks/mhc-post-backward-msprof.jsonl",
            "results/benchmarks/mhc-post-backward-gpu-ncu.jsonl", "results/benchmarks/rmsnorm-npu-cann.jsonl",
            "results/benchmarks/moe-fixed-capacity-historical.jsonl"],
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
        header_inches = style["header_inches"] + (0.35 if chart.get("note") else 0)
        rows = math.ceil(len(chart["cases"]) / style["columns"])
        height = (header_inches + style["footer_inches"]
                  + rows * style["panel_height_inches"] + (rows-1)*style["panel_gap_inches"])
        fig, axes = plt.subplots(rows, style["columns"], squeeze=False,
                                 figsize=(style["width_inches"], height), dpi=style["dpi"],
                                 facecolor=style["background"])
        fig.subplots_adjust(left=.075, right=.975, top=1-header_inches/height,
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
            value_key = chart.get("value_key", "latency_us")
            values = case[value_key]
            bars = ax.bar(range(len(chart["series"])), values, width=style["bar_width"],
                          color=[s["color"] for s in chart["series"]], zorder=3)
            ax.set_ylim(0, max(values)*1.26)
            ax.set_xticks([])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            ax.set_ylabel(chart.get("unit", "μs"), color=style["muted"])
            case_label = case.get("case_label")
            if case_label is None:
                case_label = " × ".join(map(str, case["shape"]))
            ax.set_title(case_label, loc="left", pad=12,
                         fontsize=style["panel_title_size"], color=style["text"])
            for bar, value in zip(bars, values):
                if bar.get_height() != value:
                    raise ValueError("bar geometry differs from benchmark data")
                ax.annotate(f"{value:.{style['value_decimals']}f}",
                            (bar.get_x()+bar.get_width()/2, value), xytext=(0,5),
                            textcoords="offset points", ha="center", va="bottom", color=style["text"])
        for ax in list(axes.flat)[len(chart["cases"]):]:
            ax.set_visible(False)
        fig.text(.075, .13/height, chart.get("footer", "各面板独立刻度 · μs/次 · 越低越快"),
                 fontsize=11, color=style["muted"])
        if chart.get("note"):
            fig.text(.075, 1-1.58/height, chart["note"], fontsize=10.5, color=style["muted"])
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
    print(f"Rendered {len(data['charts'])} charts ({', '.join(chart['id'] for chart in data['charts'])}); "
          f"{sum(len(chart['cases']) for chart in data['charts'])} panels; data check PASS")


if __name__ == "__main__":
    main()
