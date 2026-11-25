# RMSNorm：CANN 主线与优化

**Ascend950DT_9582 · BF16 · μs/次**。原生实现调用测试时 CANN 包中的 `torch_npu.npu_rms_norm`，优化实现通过本库公开的 `rms_norm` 入口调用。

| Shape | 原生 | 优化 | 加速比 |
|---|---:|---:|---:|
| 2048 × 20480 | 83.16 | 48.53 | 1.71× |
| 4096 × 20480 | 203.16 | 106.04 | 1.92× |

## 测量条件

同一物理卡、连续输入、`eps=1e-5`。每个 shape、每种实现测量两轮；每轮预热 20 次，随后采集 7 批，每批调用 100 次。表中数值为两轮共 14 个批均值的中位数。两种实现均在每次调用中分配 BF16 `Y` 与 FP32 `rstd`，使用同一框架 stream；公开接口返回 `Y`。

Event 测量设备 stream 上的调用区间，其中可能包含 CPU 提交间隙。官方 profiler 的内核时间单独用于分析。

环境为 PyTorch 2.9.0+cpu、torch_npu 2.9.0.post7.dev20260817、CANN 9.2.0 B096。原生采集命中 `RmsNorm_normal_all_bf16_high_performance_2001`；该批记录的优化内核名为 `xpu_rms_norm_cann`，当前统一入口为 `rms_norm`。

## 数值与适用范围

公式为 `rstd = 1/sqrt(mean(X²)+eps)`、`Y = cast(X*rstd*gamma)`，最后转换到输入精度。42 组输入共 84 项输出检查通过精度标准 2.1；其中 12 组执行目标优化，30 组执行主线。FP64 参考值由精确 BF16 输入重新计算，GPU 标杆来自已保存的同输入 NVIDIA L20X 实跑结果。

`M=2048` 使用两个 X 行槽与整行 Y；`M=4096` 使用三个 X 行槽与两段 Y。其他输入由 CANN 主线处理。完整接口见 [RMSNorm](../operators/rms_norm.md)。

## 复现

```bash
python -m xpu_kernels.cli bench --device npu --native-reference \
  --implementations optimized --shapes 2048x20480,4096x20480 \
  --dtypes bfloat16 --warmup 20 --batches 7 --iterations-per-batch 100
```

[逐批性能记录](../../results/benchmarks/rmsnorm-npu-cann.jsonl) · [精度检查结果](data/rmsnorm-cann-precision21.json) · [测量方法](../benchmarking.md)
