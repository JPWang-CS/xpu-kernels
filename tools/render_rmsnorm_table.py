"""Render RMSNorm performance tables from committed measurements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = (("npu", "ascendc", "NPU 950DT"), ("cuda", "cuda", "GPU L20X"),
           ("cuda", "triton", "GPU L20X（Triton）"), ("cuda", "torch", "GPU L20X（单算子拼接）"))
DTYPES = ("bfloat16", "float16", "float32")


def render() -> str:
    from xpu_kernels.bench import validate_benchmark_record
    from xpu_kernels.suites import RMSNORM_SHAPES

    index = {}
    for backend in ("gpu", "npu"):
        path = ROOT / "results/benchmarks" / f"rmsnorm-{backend}-v2.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            validate_benchmark_record(row)
            if row["correctness"]["status"] != "PASS":
                raise ValueError(f"unverified result: {path}")
            if (row["warmup_iterations"], row["batches"], row["iterations_per_batch"]) != (20, 10, 50):
                raise ValueError("inconsistent benchmark settings")
            key = (row["device"], row["implementation"], row["dtype"], tuple(row["shape"]))
            if key in index:
                raise ValueError(f"duplicate result: {key}")
            index[key] = row["latency_ms_median"] * 1000
    expected = {(device, impl, dtype, shape) for device, impl, _ in COLUMNS
                for dtype in DTYPES for shape in RMSNORM_SHAPES}
    if set(index) != expected:
        raise ValueError(f"incomplete matrix: missing={expected-set(index)}, extra={set(index)-expected}")
    profiler = json.loads((ROOT / "docs/performance/rmsnorm-v2-profiler.json").read_text(encoding="utf-8"))
    text = """# RMSNorm 性能

48 个 shape，FP32、FP16、BF16 三种精度，共 576 项正确性检查与性能测量，全部通过。表格单位为 **μs/次**，shape 为行数 × 归一化宽度。

`NPU 950DT` 和 `GPU L20X` 是本库实现；GPU 另列 Triton 实现与“单算子拼接”实现。“单算子拼接”在 GPU 上依次执行平方、均值、加 epsilon、倒数平方根、归一化和权重乘法，低精度转换也在 GPU 上完成。各实现使用相同输入、权重和舍入顺序；输出和临时缓冲区均预分配。

## 测量方法

输入连续，沿最后一维归一化，epsilon=1e-5。FP32 计算均方与归一化值，归一化结果转回输入 dtype 后再乘 weight。每项先与独立 golden 比较，再预热 20 次，测量 10 批、每批 50 次，取各批单次平均耗时的中位数。

CUDA/NPU Event 记录 eager 模式下当前设备流的调用区间，包含等待 CPU 提交后续操作的时间。表中数值用于比较这种调用方式；单个 kernel 的执行时间另见下方 profiler 结果。设备时钟未锁定，两种设备分别测量。

## BF16

"""
    for dtype in DTYPES:
        if dtype != "bfloat16":
            text += f"\n<details>\n<summary>{'FP16' if dtype == 'float16' else 'FP32'}</summary>\n\n"
        text += "| Shape | " + " | ".join(c[2] for c in COLUMNS) + " |\n|---|---:|---:|---:|---:|\n"
        for shape in sorted(RMSNORM_SHAPES):
            values = [f"{index[device, impl, dtype, shape]:.3f}" for device, impl, _ in COLUMNS]
            text += f"| {shape[0]} × {shape[1]} | " + " | ".join(values) + " |\n"
        if dtype != "bfloat16":
            text += "\n</details>\n"
    text += f"""
BF16、128 × 4096 下，本库 GPU 实现为 7.011 μs，Triton 为 16.201 μs；4096 × 8192 下，Triton 为 37.066 μs，本库 GPU 实现为 58.522 μs。小形状与大形状的表现不同，两套实现均保留，分别记录迭代结果。

## 官方 profiler

独立采集 BF16、128 × 4096。Nsight Compute 使用 kernel replay，保留设备时钟与缓存状态；msprof 排除前 5 次预热，统计随后 100 次任务。

| 工具 | 设备 | 指标 | μs |
|---|---|---|---:|
| Nsight Compute | GPU L20X | 本库实现的一个 launch，Duration | {profiler['gpu']['duration_us']:.3f} |
| msprof | NPU 950DT | Task Duration 中位数 | {profiler['npu']['median_us']:.3f} |

NCU 的单次 replay 用于诊断 kernel；msprof 表中值来自重复任务。采集方式不同，分别用于各设备后续分析。[完整摘要](rmsnorm-v2-profiler.json)保留指标、采集参数及 NPU 样本。

## 环境

| 项目 | GPU | NPU |
|---|---|---|
| 完整型号 | NVIDIA L20X，CC 9.0 | Ascend950DT_9582 |
| Python | 3.12.3 | 3.11.18 |
| PyTorch | 2.13.0+cu126 | 2.9.0+cpu |
| 后端 | CUDA runtime 12.6，NVCC 12.8.61，Triton 3.7.1 | torch_npu 2.9.0.post7.dev20260817，CANN 9.2.0 B096 |
| C++ 编译器 | GCC 13.3 | GCC 13.4 |

## 复现与历史记录

先激活项目私有环境；NPU 额外手动 source 已选择的 CANN。从仓库根目录执行：

```bash
python -m xpu_kernels.cli test --device cuda
python -m xpu_kernels.cli bench --device cuda --warmup 20 --batches 10 --iterations-per-batch 50
python -m xpu_kernels.cli profile --device cuda --tool ncu --implementation cuda --shape 128x4096 --dtype bfloat16 --output results/profiles/gpu-rmsnorm
```

NPU 使用 `--device npu`，profiler 使用 `--tool msprof --implementation ascendc`。单独测某种 GPU 实现可加 `--implementations cuda`、`triton` 或 `torch`。

表格直接从 [GPU 记录](../../results/benchmarks/rmsnorm-gpu-v2.jsonl)与 [NPU 记录](../../results/benchmarks/rmsnorm-npu-v2.jsonl)生成，每条保留误差、10 批样本和软件版本。重新生成：

```bash
python tools/render_rmsnorm_table.py
python tools/render_rmsnorm_table.py --check
```

首版 [GPU](../../results/benchmarks/rmsnorm-gpu-v1.jsonl)、[NPU](../../results/benchmarks/rmsnorm-npu-v1.jsonl) 和 [profiler](rmsnorm-v1-profiler.json) 保留。首版 GPU 使用 Triton；历史最佳按实现分别比较。后续迭代复用同一组 shape、精度及测量设置，保留每次原始记录。
"""
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "docs/performance/README.md"
    content = render()
    if args.check:
        if path.read_text(encoding="utf-8") != content:
            raise SystemExit("performance table differs from measurements")
        print("PASS: 576 measurements match the performance table")
    else:
        path.write_text(content, encoding="utf-8")
        print("Rendered 48 shapes x 3 dtypes x 4 implementations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
