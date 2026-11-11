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
    mhc_records()
    rms_records()
    return """每组柱形对应同一个 shape，纵轴为 **μs/次，越低越快**。各面板独立使用从零开始的刻度。

### RMSNorm

![RMSNorm — NPU 950DT 与 GPU L20X 的绝对耗时](docs/assets/benchmark-rmsnorm.png)

BF16，设备 Event / Eager 调用区间。各实现每次调用分配输出，预热 20 次，10 批 × 50 次，取批均值的中位数。

[48 个 shape 的完整结果](docs/performance/README.md) · [公式与接口](docs/operators/rmsnorm.md)

### mHC Post Backward

![mHC Post Backward — NPU 原生、NPU 优化与 GPU TileLang 的内核耗时](docs/assets/benchmark-mhc.png)

BF16，NPU 采用 msprof `Task Duration(us)`，GPU 采用 Nsight Compute 内核时间；各 shape 预热 20 次，取 5 次正式采样的中位数。NPU 原生为 CANN 包主线算子，GPU 使用 DeepSeek TileKernels 实现。

[16 个 shape 的完整结果](docs/performance/mhc-post-backward.md) · [公式与接口](docs/operators/mhc-post-backward.md)

[测试与采集方法](docs/benchmarking.md) · [图表数据](docs/assets/performance-chart-data.json)"""


def legacy_documents():
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
    return {ROOT / "docs/performance/rmsnorm-native.md": rtext}


def latency(row):
    return row.get("latency_us_median",row.get("latency_ms_median",0)*1000)


def combined_table(shapes, columns):
    lines=["| Shape | "+" | ".join(label for label,_ in columns)+" |",
           "|---|"+"---:|"*len(columns)]
    for shape in shapes:
        values=[latency(index[(tuple(shape),"bfloat16",impl)]) for _,(index,impl) in columns]
        lines.append("| "+" × ".join(map(str,shape))+" | "+" | ".join(f"{v:.3f}" for v in values)+" |")
    return "\n".join(lines)


def documents():
    files=legacy_documents()
    mhc,shapes=mhc_records()
    gpu=load("mhc-post-backward-gpu-ncu.jsonl","Nsight Compute gpu__time_duration.sum","mhc_post_backward")
    if set(gpu)!={(s,"bfloat16","tilelang") for s in shapes}:
        raise ValueError("incomplete GPU mHC matrix")
    files[ROOT/"docs/performance/mhc-post-backward.md"]=r"""# mHC Post Backward 性能

BF16，单位 **μs/次**。各行采用相同 shape、输入精度和四个输出的检查规则，分别在 NPU 950DT 和 GPU L20X 上运行。NPU 原生调用 CANN 包主线 `aclnnMhcPostBackward`，优化调用本库内核；GPU 调用 DeepSeek TileKernels 的 TileLang 实现。

## 内核时间

NPU 使用 msprof `MIX_AIC / Task Duration(us)`；GPU 使用 Nsight Compute `gpu__time_duration.sum`，从 ns 换算为 μs。每项预热 20 次，保留随后 5 次内核时间，表中为中位数。使用同一组缓冲区连续执行，保持设备原有时钟；NCU 使用 kernel replay，cache-control 和 clock-control 均为 none。

"""+combined_table(shapes,[("NPU 950DT（原生）",(mhc,"native")),("NPU 950DT（优化）",(mhc,"optimized")),("GPU L20X（TileLang）",(gpu,"tilelang"))])+r"""

三维 shape 为 T×N×D，四维为 B×S×N×D，N=4，四维输入的 token 数为 B×S。各设备使用相同输入分布；设备随机数实现不同，各设备内的原生与优化共享输入。

## 正确性与运行环境

16 个 shape 均在首尾及内部共六个 token 上用独立 CPU FP64 公式检查四个输出。设备单元测试另检查代表 shape 的全量输出、三维/四维输入、非默认 stream 和输出复用。GPU 四个输出依次为 grad_x、grad_h_res、grad_h_out、grad_h_post，已与上游返回顺序逐项对应。

| 项目 | NPU | GPU |
|---|---|---|
| 设备 | Ascend950DT_9582 | NVIDIA L20X，CC 9.0 |
| PyTorch | 2.9.0+cpu | 2.13.0+cu126 |
| 运行环境 | torch_npu 2.9.0.post7.dev20260817，CANN 9.2.0 B096 | CUDA runtime 12.6，NVCC 12.8，TileLang 0.1.15 |
| 原始内核样本 | 原生 400 条、优化 400 条 | TileLang 400 条 |

## 复现

在对应设备环境中先执行正确性测试，再采集重复 Event 调用时间及官方 profiler 内核时间：

```bash
python -m xpu_kernels.cli test --device cuda --operator mhc_post_backward
python -m xpu_kernels.cli bench --device cuda --operator mhc_post_backward \
  --warmup 20 --batches 10 --iterations-per-batch 50 \
  --output results/benchmarks/mhc-post-backward-gpu-tilelang.jsonl
python tools/profile_mhc_gpu.py --output results/profiles/mhc-gpu --records results/benchmarks/mhc-post-backward-gpu-ncu.jsonl
```

NPU 手动 source 已选择的 CANN 后运行：

```bash
python -m xpu_kernels.cli test --device npu --operator mhc_post_backward
python -m xpu_kernels.cli bench --device npu --operator mhc_post_backward \
  --warmup 20 --batches 10 --iterations-per-batch 50 \
  --output results/benchmarks/mhc-post-backward-npu-mainline.jsonl
python tools/profile_mhc_matrix.py --output results/profiles/mhc-matrix
```

Event 测试在每次调用中分配四个输出，记录设备 stream 的调用区间；CPU 提交间隙可能包含在内。上表采用独立 profiler 的内核时间。每次采集使用新的 profiler 输出目录。

## 数据

- [NPU 内核统计](../../results/benchmarks/mhc-post-backward-msprof.jsonl) · [原生样本](data/mhc-mainline-native.csv) · [优化样本](data/mhc-mainline-optimized.csv)
- [GPU 内核统计](../../results/benchmarks/mhc-post-backward-gpu-ncu.jsonl) · [NCU 样本](data/mhc-gpu-tilelang.csv)
- Event 调用记录：[NPU](../../results/benchmarks/mhc-post-backward-npu-mainline.jsonl) · [GPU](../../results/benchmarks/mhc-post-backward-gpu-tilelang.jsonl)
- [算子公式、适用范围与调用示例](../operators/mhc-post-backward.md)
"""
    rms,rshapes=rms_records()
    rgpu=load("rmsnorm-gpu-mainline.jsonl","torch.cuda.Event.elapsed_time","rms_norm")
    from xpu_kernels.bench import validate_benchmark_record
    expected={(s,"bfloat16",impl) for s in rshapes for impl in ("cuda","triton","torch","native")}
    if set(rgpu)!=expected:raise ValueError("incomplete matched GPU RMSNorm matrix")
    for row in rgpu.values():
        validate_benchmark_record(row)
        if row['schema_version']!=4 or 'per_call' not in row['output_allocation']['strategy']:
            raise ValueError("RMSNorm comparison requires per-call allocation")
    files[ROOT/"docs/performance/README.md"]=r"""# RMSNorm 性能

**BF16 · μs/次 · 48 个 shape**。NPU 950DT 和 GPU L20X 的同 shape 结果放在同一行，分别展示原生实现、本库实现及 GPU 单算子拼接。

## 测量方法

设备 Event / Eager 调用区间，预热 20 次，10 批 × 50 次，取批均值的中位数。各实现每次调用均分配输出；Event 区间可能包含 CPU 提交间隙。原生 NPU 额外返回 rstd，本库接口返回归一化结果。

本库及 GPU 单算子拼接按 FP32 计算归一化值，转回输入精度后乘权重。CANN / PyTorch 原生 RMSNorm 按各自融合公式计算，分别使用对应独立公式检查数值。表中保留全部 shape，包括原生或 Triton 更快的结果。

"""+combined_table(rshapes,[("NPU 950DT（原生）",(rms,"native")),("NPU 950DT（优化）",(rms,"ascendc")),("GPU L20X",(rgpu,"cuda")),("GPU L20X（Triton）",(rgpu,"triton")),("GPU L20X（单算子拼接）",(rgpu,"torch")),("GPU L20X（原生）",(rgpu,"native"))])+r"""

## 环境与复现

NPU：Ascend950DT_9582，PyTorch 2.9.0+cpu，torch_npu 2.9.0.post7.dev20260817，CANN 9.2.0 B096。
GPU：NVIDIA L20X，CC 9.0，PyTorch 2.13.0+cu126，CUDA runtime 12.6，NVCC 12.8，Triton 3.7.1。

```bash
python -m xpu_kernels.cli bench --device cuda --native-reference --dtypes bfloat16 \
  --warmup 20 --batches 10 --iterations-per-batch 50 \
  --output results/benchmarks/rmsnorm-gpu-mainline.jsonl
python -m xpu_kernels.cli bench --device npu --native-reference \
  --warmup 20 --batches 10 --iterations-per-batch 50 \
  --output results/benchmarks/rmsnorm-npu-mainline.jsonl
```

## 完整记录

- [GPU 原始数据](../../results/benchmarks/rmsnorm-gpu-mainline.jsonl)：48 shape × 4 实现，BF16。
- [NPU 原始数据](../../results/benchmarks/rmsnorm-npu-mainline.jsonl)：48 shape × 2 实现 × 3 精度。
- [NPU 三种精度对照](rmsnorm-native.md) · [预分配输出测试](rmsnorm-preallocated.md)：保留 FP32 / FP16 / BF16 的历史测量及 profiler 数据。
- [公式与接口](../operators/rmsnorm.md) · [采集方法](../benchmarking.md)
"""
    return files


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
