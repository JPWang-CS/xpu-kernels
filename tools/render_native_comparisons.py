"""Render native/final comparisons from device-validated public measurements."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- BEGIN homepage-performance -->"
END = "<!-- END homepage-performance -->"


def load(name, method, operator):
    index = {}
    for line in (ROOT / "results/benchmarks" / name).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["operator"] != operator or row["correctness"]["status"] != "PASS" or row["measurement_method"] != method:
            raise ValueError(f"invalid evidence in {name}")
        key = (tuple(row["shape"]), row["dtype"], row["implementation"])
        if key in index:
            raise ValueError(f"duplicate evidence in {name}: {key}")
        value = row.get("latency_us_median", row.get("latency_ms_median", 0)*1000)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid latency")
        index[key] = row
    return index


def table(index, shapes, dtype, optimized="optimized"):
    lines = ["| Shape | 原生 | 优化 |", "|---|---:|---:|"]
    for shape in shapes:
        rows = [index[(tuple(shape), dtype, impl)] for impl in ("native", optimized)]
        if any(rows[0][key] != rows[1][key] for key in ("hardware", "torch", "runtime", "measurement_method")):
            raise ValueError("native and optimized measurement conditions differ")
        values = [row.get("latency_us_median", row.get("latency_ms_median", 0)*1000) for row in rows]
        cells = [f"**{t:.3f}**" if t == min(values) else f"{t:.3f}" for t in values]
        lines.append("| " + " × ".join(map(str, shape)) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mhc_records():
    records = load("mhc-post-backward-msprof.jsonl", "msprof Task Duration(us)", "mhc_post_backward")
    from xpu_kernels.mhc_workload import SHAPES
    if set(records) != {(s, "bfloat16", impl) for s in SHAPES for impl in ("native", "optimized")}:
        raise ValueError("incomplete mHC matrix")
    for shape, _, _ in records:
        t = shape[0] if len(shape) == 3 else shape[0]*shape[1]
        if not (2048 <= t <= 49152 and shape[-2] == 4 and shape[-1] in (2560,5120,6144,7168)):
            raise ValueError("public mHC shape is outside the optimized range")
    return records, SHAPES


def rms_records():
    from xpu_kernels.bench import validate_benchmark_record
    from xpu_kernels.suites import RMSNORM_SHAPES
    records = load("rmsnorm-npu-mainline.jsonl", "torch.npu.Event.elapsed_time", "rms_norm")
    if set(records) != {(s, dtype, impl) for s in RMSNORM_SHAPES
                           for dtype in ("float32","float16","bfloat16") for impl in ("native","ascendc")}:
        raise ValueError("incomplete native RMSNorm matrix")
    for row in records.values():
        validate_benchmark_record(row)
        if (row["warmup_iterations"], row["batches"], row["iterations_per_batch"]) != (20,10,50):
            raise ValueError("inconsistent RMSNorm settings")
    return records, RMSNORM_SHAPES


def homepage():
    # Preserve table generation in detailed documents; the README is visual.
    mhc_records()
    rms_records()
    return """柱高为相对同 shape 基线的加速比，**1× 为基线，越高越快**。灰色为基线，青色为本库实现，紫色为 Triton。

### mHC Post Backward

![mHC Post Backward: CANN native versus xpu-kernels](docs/assets/benchmark-mhc.png)

[16 个 shape 与采样](docs/performance/mhc-post-backward.md) · [原始数据](results/benchmarks/mhc-post-backward-msprof.jsonl)

### RMSNorm

![RMSNorm on GPU L20X: composition, xpu-kernels and Triton](docs/assets/benchmark-rms-gpu.png)

![RMSNorm on NPU 950DT: CANN native versus xpu-kernels](docs/assets/benchmark-rms-npu.png)

<sub>NPU 两种实现的中间舍入顺序不同，分别按对应公式检查。</sub>

[GPU 完整结果](docs/performance/README.md) · [NPU 原生对照](docs/performance/rmsnorm-native.md)

<details>
<summary>测量条件与比较范围</summary>

- **mHC / NPU 950DT**：BF16，基线为 CANN 9.2.0 B096 的主线算子。使用 msprof `Task Duration(us)`，预热 20 次，取 5 次正式采样中位数。展示的 shape 均在优化范围内。
- **RMSNorm / GPU L20X**：BF16，基线为 GPU 单算子拼接。CUDA C++、Triton 和拼接使用相同数值顺序。
- **RMSNorm / NPU 950DT**：BF16，基线为 CANN 原生实现。本库按 Pangu 的中间舍入顺序计算，两者分别对照对应公式；原生更快的 shape 同样保留。
- RMSNorm 使用设备 Event / Eager，预热 20 次，10 批 × 50 次，取批均值的中位数。Event 区间包含 CPU 提交间隙。每张图在同一设备内比较。

[复现方法](docs/benchmarking.md) · [图表数据](docs/assets/performance-chart-data.json)

</details>"""


def documents():
    mhc, shapes = mhc_records()
    text = """# mHC Post Backward 性能

**NPU Ascend950DT_9582 · BF16 · μs/次**。PyTorch 2.9.0+cpu，torch_npu 2.9.0.post7.dev20260817，CANN 9.2.0 B096。

原生调用 `torch_npu.npu_mhc_post_backward`，进入 CANN 主线 `aclnnMhcPostBackward`；优化调用本库 DeepJIT 内核。采集时私有 OPP 覆盖路径为空，原生加载 CANN 包的 `libopapi_transformer.so`。

## Kernel 时间

使用官方 msprof 的 `MIX_AIC` / `Task Duration(us)`。同一张设备卡、相同输入、相同当前 stream，每个 shape 预热 20 次后采集 5 次，表中取中位数。缓存使用同一缓冲区的稳态，时钟保持原设置。加粗为该行耗时较低的实现。

""" + table(mhc,shapes,"bfloat16") + """

这些 shape 均位于优化范围：4 条残差流，h_res 存在，2048–49152 个 token，D 为 2560、5120、6144 或 7168。四维输入的 token 数为 B×S。

每项在首尾及内部共六个 token 上用 CPU FP64 公式核对四个输出；单元测试另覆盖代表 shape 的全量输出、FP16/BF16、非默认 stream 和输出复用。性能采集在正确性检查后独立进行。

## 复现

先激活项目私有 Python 环境并手动 source CANN，在健康空闲设备上运行：

```bash
python -m xpu_kernels.cli test --device npu --operator mhc_post_backward
python -m xpu_kernels.cli bench --device npu --operator mhc_post_backward \\
  --warmup 20 --batches 10 --iterations-per-batch 50 \\
  --output results/benchmarks/mhc-post-backward-npu-mainline.jsonl
python tools/profile_mhc_matrix.py --output results/profiles/mhc-matrix
```

每次使用新的 profiler 输出目录。脚本按实际调用顺序提取目标 kernel，核对原生 CSV 中的输入 shape，要求每个 shape 恰有 25 次目标 launch，剔除前 20 次；保留 800 条原始 kernel 记录和 32 项统计。两种实现都在每次调用中分配四个输出。

## 数据

- [Kernel 样本与统计](../../results/benchmarks/mhc-post-backward-msprof.jsonl)
- [原生 CSV](data/mhc-mainline-native.csv) · [优化 CSV](data/mhc-mainline-optimized.csv)
- [重复 Event 调用记录](../../results/benchmarks/mhc-post-backward-npu-mainline.jsonl)：10 批 × 50 次，记录设备流调用区间，包含 CPU 提交间隙。

旧项目中内部实现之间的对照保留在原始历史文件，当前表使用上述 CANN 主线与本库实际调用重新测量。
"""
    rms, shapes = rms_records()
    rtext = """# RMSNorm：NPU 原生与本库对照

**NPU Ascend950DT_9582 · μs/次**。PyTorch 2.9.0+cpu，torch_npu 2.9.0.post7.dev20260817，CANN 9.2.0 B096。原生为 `torch_npu.npu_rms_norm`，优化为本库 Ascend C 实现。

设备 Event，Eager 模式，预热 20 次，10 批 × 50 次，取各批单次平均时间的中位数。两边每次调用均分配输出；原生同时返回归一化结果和 rstd，本库返回归一化结果。Event 区间包含 CPU 提交间隙。

本库遵循 Pangu 的数值顺序：FP32 归一化值先转回输入精度，再乘权重；CANN 原生在融合计算后转回输出精度。两者分别与对应的独立公式比较。小 shape 的调用开销和大 shape 的设备计算占比不同，表中保留全部结果。加粗为该行较低耗时。
"""
    for dtype, title in (("bfloat16","BF16"),("float16","FP16"),("float32","FP32")):
        rtext += f"\n## {title}\n\n" + table(rms,shapes,dtype,"ascendc") + "\n"
    rtext += """
## 复现

```bash
python -m xpu_kernels.cli bench --device npu --native-reference \\
  --warmup 20 --batches 10 --iterations-per-batch 50 \\
  --output results/benchmarks/rmsnorm-npu-mainline.jsonl
```

[288 项原始记录](../../results/benchmarks/rmsnorm-npu-mainline.jsonl)保存误差、每批样本、软件版本及输出分配方式。GPU 各实现的相同数值顺序对照见[完整性能表](README.md)。
"""
    return {ROOT / "docs/performance/mhc-post-backward.md": text,
            ROOT / "docs/performance/rmsnorm-native.md": rtext}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    for path, content in documents().items():
        if args.check:
            if path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"outdated table: {path.name}")
        else:
            path.write_text(content, encoding="utf-8")
    print("Native comparison tables match their recorded measurements")


if __name__ == "__main__":
    main()
