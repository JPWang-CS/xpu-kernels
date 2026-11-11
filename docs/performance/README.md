# RMSNorm 首版性能记录

RMSNorm 已在 NVIDIA L20X 和 Ascend950DT_9582 上完成编译、正确性验证与实际执行。GPU 使用 Triton，NPU 使用 AscendC 与 DeepJIT。当前保留一套正确的基础实现；性能记录用于后续逐项比较和替换实现。

## 测量条件

以下为 BF16，单位为微秒/次。输入连续，归一化最后一维，epsilon 为 1e-5。每个 case 预热 20 次，再测 10 批、每批 50 次，取批次均值的中位数。输入、权重和自写算子的输出缓冲区重复使用；PyTorch/torch_npu 原生 API 自行分配输出。未锁定设备时钟。

Event 记录的是同一设备流内一批调用的经过时间，包含该区间内等待 CPU 提交后续操作的空闲时间。它适合比较当前 eager 调用方式。单个 kernel 的设备执行时间另由 NCU/msprof 采集，二者分别列出。

Golden 检查采用 FP32 归约、归一化结果转回输入 dtype、再乘 weight 的顺序。原生算子的累加和舍入顺序可能不同，表中原生 API 为性能参照。

| 设备型号 | 形状（行 × 宽度） | xpu-kernels（μs/次） | 原生 RMSNorm（μs/次） |
|---|---:|---:|---:|
| NVIDIA L20X | 1 × 128 | 16.160 | 8.113 |
| NVIDIA L20X | 1 × 4096 | 16.400 | 8.065 |
| NVIDIA L20X | 16 × 4096 | 16.729 | 8.223 |
| NVIDIA L20X | 128 × 4096 | 16.273 | 8.169 |
| NVIDIA L20X | 1024 × 4096 | 16.565 | 8.542 |
| NVIDIA L20X | 128 × 8193 | 16.454 | 18.746 |
| Ascend950DT_9582 | 1 × 128 | 9.239 | 12.584 |
| Ascend950DT_9582 | 1 × 4096 | 9.486 | 12.724 |
| Ascend950DT_9582 | 16 × 4096 | 9.304 | 12.165 |
| Ascend950DT_9582 | 128 × 4096 | 9.240 | 12.253 |
| Ascend950DT_9582 | 1024 × 4096 | 26.663 | 12.349 |
| Ascend950DT_9582 | 128 × 8193 | 9.324 | 12.149 |

GPU 的常规宽度下，原生 RMSNorm 调用更快；宽度 8193 时，本实现的调用区间略短。NPU 的小、中等形状优于当前原生调用；1024 × 4096 下仍明显落后。后续优化要同时观察这些形状，保留退化项及其原因。

FP32、FP16、BF16 的全部 36 条记录和每批样本保存在 [GPU JSONL](../../results/benchmarks/rmsnorm-gpu-v1.jsonl) 与 [NPU JSONL](../../results/benchmarks/rmsnorm-npu-v1.jsonl)。这些是首版实测基线，尚无历史自写实现可比较，因此 `baseline_best_ms` 和 `speedup_vs_baseline` 为空。后续同配置记录会参与历史最佳比较。

## 官方 profiler 结果

独立采集 BF16、128 × 4096，只统计自写 RMSNorm kernel。

| 工具 | 设备型号 | 指标 | 结果 |
|---|---|---|---:|
| Nsight Compute | NVIDIA L20X | 一个捕获 launch 的 Duration | 3.010 μs |
| msprof | Ascend950DT_9582 | 排除前 5 次预热后的 100 次 Task Duration，中位数 | 5.319 μs |

NCU 使用 kernel replay、SpeedOfLight 和 LaunchStats，保留原时钟与缓存状态。这个单次 replay 样本用于定位 kernel 行为。msprof 的 100 次样本及统计保存在 [profiler 摘要](rmsnorm-v1-profiler.json)。两次采集分别运行，数值用于各自设备上的后续分析。

NCU 显示当前 grid 有 128 个 program，设备有 132 个 SM；小批量时并行工作量不足。Event 调用区间大于 kernel Duration，说明评估 Python 调用开销和评估设备 kernel 时应分别计量。NPU 的大行数形状需要单独采集报告，再根据瓶颈选择优化方法。

## 已验证环境

| 项目 | GPU | NPU |
|---|---|---|
| 设备 | NVIDIA L20X，CC 9.0 | Ascend950DT_9582 |
| Python | 3.12.3 | 3.11.18 |
| PyTorch | 2.13.0+cu126 | 2.9.0+cpu |
| 后端软件 | Triton 3.7.1，CUDA runtime 12.6 | torch_npu 2.9.0.post7.dev20260817 |
| CANN / host compiler | — | 9.2.0（V100R001C12B096），GCC 13.4 |

两端分别验证 3 个常规形状 × 3 种精度，并检查零输入、宽度 1、三维输入、空行、指定输出缓冲区及非默认 stream。相同 wheel 已在 GPU 和 NPU 上安装并运行；NPU 从 wheel 内的 host/device 源码完成编译。

## 复现

先激活项目私有环境；NPU 额外手动 source 已选择的 CANN。以下命令从仓库根目录执行：

```bash
python -m xpu_kernels.cli test --device cuda
python -m xpu_kernels.cli bench --device cuda \
  --shapes 1x128,1x4096,16x4096,128x4096,1024x4096,128x8193 \
  --warmup 20 --batches 10 --iterations-per-batch 50
python -m xpu_kernels.cli profile --device cuda --tool ncu \
  --shape 128x4096 --dtype bfloat16 --output results/profiles/gpu-rmsnorm
```

NPU 将 `--device cuda` 改为 `--device npu`，profiler 使用 `--tool msprof`。原始报告分别位于验证项目的 `runs/gpu-release/ncu/` 和 `runs/npu-release/msprof/`；编译、stdout、stderr 和退出码与各自 run 一起保留。大型 trace 不进入 Git。
