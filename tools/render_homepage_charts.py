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
COLORS = {"npu_native":"#94A3B8","npu":"#0D9488","gpu":"#2563EB","triton":"#7C3AED","composed":"#D97706"}


def chart_data():
    mhc, _ = mhc_records()
    rms, _ = rms_records()
    gpu_mhc = load("mhc-post-backward-gpu-ncu.jsonl", "Nsight Compute gpu__time_duration.sum", "mhc_post_backward")
    gpu_rms = load("rmsnorm-gpu-mainline.jsonl", "torch.cuda.Event.elapsed_time", "rms_norm")
    definitions = [
        ("mhc", "mHC Post Backward", "BF16 · Kernel time · μs", [
            (mhc,"native","NPU 950DT（原生）",COLORS["npu_native"]),
            (mhc,"optimized","NPU 950DT（优化）",COLORS["npu"]),
            (gpu_mhc,"tilelang","GPU L20X（TileLang）",COLORS["gpu"])],
         ((2048,4,2560),(2048,4,6144),(8192,4,6144),(32768,4,6144))),
        ("rmsnorm", "RMSNorm", "BF16 · Eager call time · μs", [
            (rms,"native","NPU 950DT（原生）",COLORS["npu_native"]),
            (rms,"ascendc","NPU 950DT（优化）",COLORS["npu"]),
            (gpu_rms,"cuda","GPU L20X",COLORS["gpu"]),
            (gpu_rms,"triton","GPU L20X（Triton）",COLORS["triton"]),
            (gpu_rms,"torch","GPU L20X（单算子拼接）",COLORS["composed"])],
         ((128,4096),(129,4097),(1024,4096),(4096,8192))),
    ]
    charts=[]
    for key,title,method,series,shapes in definitions:
        cases=[]
        for shape in shapes:
            rows=[index[(shape,"bfloat16",impl)] for index,impl,_,_ in series]
            if any(r["correctness"]["status"]!="PASS" for r in rows):raise ValueError("unverified plot data")
            values=[r.get("latency_us_median",r.get("latency_ms_median",0)*1000) for r in rows]
            if any(not math.isfinite(v) or v<=0 for v in values):raise ValueError("invalid latency")
            if key=="rmsnorm":
                for r in rows:
                    if r["schema_version"]!=4 or r["output_allocation"]["strategy"]=="preallocated_out":
                        raise ValueError("RMSNorm device comparison requires per-call output allocation")
            cases.append({"shape":list(shape),"latency_us":values})
        charts.append({"id":key,"title":title,"method":method,"dtype":"bfloat16",
            "metric":"absolute_latency_us","lower_is_better":True,
            "series":[{"implementation":impl,"label":label,"color":color} for _,impl,label,color in series],"cases":cases})
    return {"schema_version":2,"sources":["results/benchmarks/mhc-post-backward-msprof.jsonl",
        "results/benchmarks/mhc-post-backward-gpu-ncu.jsonl","results/benchmarks/rmsnorm-npu-mainline.jsonl",
        "results/benchmarks/rmsnorm-gpu-mainline.jsonl"],"charts":charts}


def plot(data,font_path=None):
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    if font_path is None and Path("C:/Windows/Fonts/msyh.ttc").is_file():font_path=Path("C:/Windows/Fonts/msyh.ttc")
    if font_path:
        font_manager.fontManager.addfont(str(font_path))
        family=font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        family=next((f.name for f in font_manager.fontManager.ttflist if f.name in
            ("Noto Sans CJK SC","Source Han Sans SC","Microsoft YaHei")),None)
        if family is None:raise RuntimeError("provide a Chinese font using --font")
    plt.rcParams.update({"font.family":[family,"DejaVu Sans"],"font.size":13,"axes.unicode_minus":False})
    for chart in data["charts"]:
        fig,axes=plt.subplots(2,2,figsize=(14,8),dpi=160,facecolor="#FAFCFE")
        fig.subplots_adjust(left=.075,right=.975,top=.73,bottom=.08,wspace=.20,hspace=.44)
        fig.text(.075,.945,chart["title"],fontsize=24,fontweight="bold",color="#0F172A")
        fig.text(.075,.897,chart["method"],fontsize=13,color="#64748B")
        handles=[Patch(facecolor=s["color"],label=s["label"]) for s in chart["series"]]
        fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.067,.865),ncol=3,
            frameon=False,fontsize=12.5,handlelength=1.0,columnspacing=1.8)
        for ax,case in zip(axes.flat,chart["cases"]):
            ax.set_facecolor("#FAFCFE")
            ax.set_axisbelow(True);ax.grid(axis="y",color="#E2E8F0",linewidth=.7)
            for side in ("top","right","left"):ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#CBD5E1")
            ax.tick_params(axis="both",length=0,labelcolor="#64748B")
            bars=ax.bar(range(len(chart["series"])),case["latency_us"],width=.60,color=[s["color"] for s in chart["series"]],zorder=3)
            ax.set_ylim(0,max(case["latency_us"])*1.26);ax.set_xticks([])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
            ax.set_ylabel("μs",color="#64748B",fontsize=12)
            ax.set_title(" × ".join(map(str,case["shape"])),loc="left",pad=12,fontsize=14,color="#0F172A")
            for rect,value in zip(bars,case["latency_us"]):
                ax.annotate(f"{value:.2f}",(rect.get_x()+rect.get_width()/2,value),
                    xytext=(0,5),textcoords="offset points",ha="center",va="bottom",fontsize=11.5,color="#0F172A")
        fig.text(.075,.02,"各面板使用独立刻度 · μs/次 · 越低越快",fontsize=11,color="#64748B")
        fig.savefig(ASSETS/f"benchmark-{chart['id']}.png",dpi=160,facecolor=fig.get_facecolor(),metadata={"Software":"xpu-kernels benchmark renderer"})
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
    print("Two absolute-time charts match eight shape groups and32 device measurements")


if __name__ == "__main__":
    main()
