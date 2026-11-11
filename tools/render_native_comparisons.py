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
    mhc, _ = mhc_records()
    rms, _ = rms_records()
    text = """### mHC Post Backward · NPU 950DT

![mHC 原生与优化对照](docs/assets/mhc-native-performance.png)

**BF16 · μs/次**。原生为 CANN 9.2.0 B096 的 `aclnnMhcPostBackward`，优化为本库实现。每个 shape 预热 20 次，取 5 次 `msprof Task Duration(us)` 的中位数。

"""
    text += table(mhc, ((2048,4,2560), (2048,4,6144), (8192,4,6144), (32768,4,6144), (2,1024,4,6144)), "bfloat16")
    text += """

优化范围：BF16、4 条残差流、传入 `h_res`，token 数为 2048–49152，特征宽度为 2560、5120、6144 或 7168。其它受支持输入自动调用 CANN 原生实现。

[16 个 shape 与原始采样](docs/performance/mhc-post-backward.md) · [公式与接口](docs/operators/mhc-post-backward.md)

### RMSNorm · GPU L20X

![RMSNorm GPU 实现对比](docs/assets/rmsnorm-performance.png)

**BF16 · μs/次**。设备 Event，Eager 模式，预热 20 次，测量 10 批、每批 50 次。加粗为同一 GPU 上最低耗时。

| Shape | GPU L20X | GPU L20X（Triton） | GPU L20X（单算子拼接） |
|---|---:|---:|---:|
"""
    gpu = load("rmsnorm-gpu-v2.jsonl", "torch.cuda.Event.elapsed_time", "rms_norm")
    shapes = ((128,4096),(129,4097),(1024,4096),(4096,8192))
    for shape in shapes:
        values = [gpu[(shape,"bfloat16",impl)]["latency_ms_median"]*1000 for impl in ("cuda","triton","torch")]
        cells = [f"**{t:.3f}**" if t == min(values) else f"{t:.3f}" for t in values]
        text += "| " + " × ".join(map(str,shape)) + " | " + " | ".join(cells) + " |\n"
    text += """
单算子拼接在 GPU 上依次执行平方、均值、归一化和权重乘法，各实现保持相同的中间舍入顺序。

<details>
<summary><strong>RMSNorm · NPU 950DT 原生对照</strong></summary>

**BF16 · μs/次**。设备 Event，预热 20 次，10 批 × 50 次，两边每次调用均分配输出。

""" + table(rms,shapes,"bfloat16","ascendc") + """

本库遵循 Pangu 的数值顺序：归一化值先转回输入精度，再乘权重；CANN 原生在融合计算后转回输出精度。两者按各自公式检查。表中保留原生更快的 shape。

</details>

[48 个 shape · GPU 各实现](docs/performance/README.md) · [48 个 shape · NPU 原生对照](docs/performance/rmsnorm-native.md)

GPU 与 NPU 分别测量；Event 区间包含 CPU 提交间隙，kernel 时间采用独立 profiler 数据。
"""
    return text.strip()


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
