# 安装与设备验证

GPU 的 0.3.0 版本已完成源码测试、wheel 构建、独立安装及设备运行。NPU 沿用已验证的内核与 CANN 主线路径，保留 0.2.0 的安装和设备结果。两端使用项目私有环境，构建缓存和输出留在各自项目目录。

| 验证项 | GPU L20X | NPU 950DT |
|---|---|---|
| 版本 | 0.3.0 | 0.2.0 设备记录 |
| 源码测试 | 46 项通过，5 项跳过 | 40 项通过，9 项跳过 |
| 跳过内容 | NPU 专用测试等当前后端不适用项 | GPU、Triton 专用测试等当前后端不适用项 |
| wheel 构建与独立安装 | 通过 | 通过 |
| wheel 内设备源码运行 | RMSNorm，FP32 / FP16 / BF16；mHC，BF16 | RMSNorm，FP32 / FP16 / BF16；mHC，FP16 / BF16 |
| mHC 优化路径 | TileKernels，2×1024×4×6144，四个输出通过 | 本库优化实现，2×1024×4×6144，四个输出通过 |
| 非默认 stream 与输出复用 | RMSNorm、mHC 通过 | RMSNorm、mHC 通过 |

GPU wheel 从独立安装目录导入，并使用项目内 XPU_KERNELS_CACHE_DIR 编译、运行。检查了 TileKernels 源码及 MIT 许可、CUDA/Ascend 源码、DeepJIT 头文件及许可证。mHC 的 wheel 测试执行六项，其中四项通过，两项 NPU 专用测试跳过。GPU 全量输出检查包括 512×4×2560、513×4×2688 和 2×256×4×6144；独立 FP64 golden 另与 PyTorch autograd 对照。

## 性能记录

- RMSNorm：GPU BF16 的 48 个 shape × 四种实现，共 192 项；NPU 48 个 shape × 三种精度 × 两种实现，共 288 项。各实现逐调用分配输出，按对应数值公式检查。预分配输出的历史测量另行保留。
- mHC：GPU 与 NPU 使用相同 16 个 BF16 shape。GPU TileKernels 共 16 项，NPU 原生与优化共 32 项 Event 记录；每项用 CPU FP64 公式检查边界和内部 token。
- mHC 官方 profiler：NPU msprof 原生与优化各 400 次目标 launch；GPU Nsight Compute 共 400 次。每个 shape、每个实现剔除 20 次预热，保留五次正式采样，公开 CSV 与统计表逐项对应。

GPU 环境为 L20X（CC 9.0）、PyTorch 2.13.0+cu126、CUDA runtime 12.6、NVCC 12.8、Triton 3.7.1、TileLang 0.1.15。NPU 环境为 Ascend950DT_9582、PyTorch 2.9.0+cpu、torch_npu 2.9.0.post7.dev20260817、CANN 9.2.0 B096。

GPU 日志与退出码保存在项目的 runs/mhc-gpu-accept/、runs/mhc-gpu-profile/、runs/mhc-gpu-release/ 和 runs/mhc-gpu-030-final/。NPU 安装及性能日志保留在 runs/mhc-npu-final/、runs/mhc-mainline-accept/、runs/mhc-mainline-profiler/ 和 runs/rms-native-comparison/。公开数据见[记录索引](../results/benchmarks/README.md)。

## 复核表格与图形

```bash
PYTHONPATH=src python tools/render_native_comparisons.py --check
PYTHONPATH=src python tools/render_homepage_performance.py --check
PYTHONPATH=src python tools/render_homepage_charts.py --check
PYTHONPATH=src python tools/render_rmsnorm_table.py --check
```
