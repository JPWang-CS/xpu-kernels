# mHC Post Backward 性能

BF16，单位 **μs/次**。各行采用相同 shape、输入精度和四个输出的检查规则，分别在 NPU 950DT 和 GPU L20X 上运行。NPU 原生调用 CANN 包主线 `aclnnMhcPostBackward`，优化调用本库内核；GPU 调用 DeepSeek TileKernels 的 TileLang 实现。

## 内核时间

NPU 使用 msprof `MIX_AIC / Task Duration(us)`；GPU 使用 Nsight Compute `gpu__time_duration.sum`，从 ns 换算为 μs。每项预热 20 次，保留随后 5 次内核时间，表中为中位数。使用同一组缓冲区连续执行，保持设备原有时钟；NCU 使用 kernel replay，cache-control 和 clock-control 均为 none。

| Shape | NPU 950DT（原生） | NPU 950DT（优化） | GPU L20X（TileLang） |
|---|---:|---:|---:|
| 2048 × 4 × 2560 | 125.430 | 43.167 | 41.120 |
| 2048 × 4 × 5120 | 205.597 | 75.942 | 78.816 |
| 2048 × 4 × 6144 | 295.689 | 89.442 | 93.056 |
| 2048 × 4 × 7168 | 329.050 | 120.865 | 104.864 |
| 8192 × 4 × 2560 | 502.189 | 187.783 | 143.136 |
| 8192 × 4 × 5120 | 815.806 | 367.347 | 281.248 |
| 8192 × 4 × 6144 | 1174.600 | 439.071 | 337.216 |
| 8192 × 4 × 7168 | 1304.922 | 522.717 | 394.592 |
| 32768 × 4 × 2560 | 2000.422 | 738.546 | 550.816 |
| 32768 × 4 × 5120 | 3242.825 | 1455.092 | 1106.240 |
| 32768 × 4 × 6144 | 4678.831 | 1736.369 | 1330.080 |
| 32768 × 4 × 7168 | 5199.662 | 2076.415 | 1558.304 |
| 2 × 1024 × 4 × 6144 | 295.395 | 89.943 | 89.888 |
| 3 × 1024 × 4 × 6144 | 445.595 | 172.502 | 132.256 |
| 2 × 4096 × 4 × 6144 | 1174.741 | 442.260 | 336.544 |
| 3 × 8192 × 4 × 6144 | 3513.410 | 1337.927 | 998.720 |

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
