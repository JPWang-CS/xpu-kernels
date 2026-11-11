# mHC Post Backward 性能

**NPU Ascend950DT_9582 · BF16 · μs/次**。PyTorch 2.9.0+cpu，torch_npu 2.9.0.post7.dev20260817，CANN 9.2.0 B096。

原生调用 `torch_npu.npu_mhc_post_backward`，进入 CANN 主线 `aclnnMhcPostBackward`；优化调用本库 DeepJIT 内核。采集时私有 OPP 覆盖路径为空，原生加载 CANN 包的 `libopapi_transformer.so`。

## Kernel 时间

使用官方 msprof 的 `MIX_AIC` / `Task Duration(us)`。同一张设备卡、相同输入、相同当前 stream，每个 shape 预热 20 次后采集 5 次，表中取中位数。缓存使用同一缓冲区的稳态，时钟保持原设置。加粗为该行耗时较低的实现。

| Shape | 原生 | 优化 |
|---|---:|---:|
| 2048 × 4 × 2560 | 125.430 | **43.167** |
| 2048 × 4 × 5120 | 205.597 | **75.942** |
| 2048 × 4 × 6144 | 295.689 | **89.442** |
| 2048 × 4 × 7168 | 329.050 | **120.865** |
| 8192 × 4 × 2560 | 502.189 | **187.783** |
| 8192 × 4 × 5120 | 815.806 | **367.347** |
| 8192 × 4 × 6144 | 1174.600 | **439.071** |
| 8192 × 4 × 7168 | 1304.922 | **522.717** |
| 32768 × 4 × 2560 | 2000.422 | **738.546** |
| 32768 × 4 × 5120 | 3242.825 | **1455.092** |
| 32768 × 4 × 6144 | 4678.831 | **1736.369** |
| 32768 × 4 × 7168 | 5199.662 | **2076.415** |
| 2 × 1024 × 4 × 6144 | 295.395 | **89.943** |
| 3 × 1024 × 4 × 6144 | 445.595 | **172.502** |
| 2 × 4096 × 4 × 6144 | 1174.741 | **442.260** |
| 3 × 8192 × 4 × 6144 | 3513.410 | **1337.927** |

这些 shape 均位于优化范围：4 条残差流，h_res 存在，2048–49152 个 token，D 为 2560、5120、6144 或 7168。四维输入的 token 数为 B×S。

每项在首尾及内部共六个 token 上用 CPU FP64 公式核对四个输出；单元测试另覆盖代表 shape 的全量输出、FP16/BF16、非默认 stream 和输出复用。性能采集在正确性检查后独立进行。

## 复现

先激活项目私有 Python 环境并手动 source CANN，在健康空闲设备上运行：

```bash
python -m xpu_kernels.cli test --device npu --operator mhc_post_backward
python -m xpu_kernels.cli bench --device npu --operator mhc_post_backward \
  --warmup 20 --batches 10 --iterations-per-batch 50 \
  --output results/benchmarks/mhc-post-backward-npu-mainline.jsonl
python tools/profile_mhc_matrix.py --output results/profiles/mhc-matrix
```

每次使用新的 profiler 输出目录。脚本按实际调用顺序提取目标 kernel，核对原生 CSV 中的输入 shape，要求每个 shape 恰有 25 次目标 launch，剔除前 20 次；保留 800 条原始 kernel 记录和 32 项统计。两种实现都在每次调用中分配四个输出。

## 数据

- [Kernel 样本与统计](../../results/benchmarks/mhc-post-backward-msprof.jsonl)
- [原生 CSV](data/mhc-mainline-native.csv) · [优化 CSV](data/mhc-mainline-optimized.csv)
- [重复 Event 调用记录](../../results/benchmarks/mhc-post-backward-npu-mainline.jsonl)：10 批 × 50 次，记录设备流调用区间，包含 CPU 提交间隙。

旧项目中内部实现之间的对照保留在原始历史文件，当前表使用上述 CANN 主线与本库实际调用重新测量。
