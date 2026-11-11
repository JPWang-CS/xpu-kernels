# 安装与开始使用

本页说明如何安装 xpu-kernels、构建设备后端并运行一个算子示例。算子的输入约束和数值行为见 [RMSNorm](operators/rmsnorm.md) 与 [mHC Post Backward](operators/mhc-post-backward.md)；完整测试和性能复现见 [Benchmark 与 Profiler](benchmarking.md)。

## 环境要求

需要 Python 3.10+、PyTorch、Ninja，以及与目标设备相匹配的工具链。GPU mHC 使用 TileLang；项目的 `mhc-gpu` extra 安装 TileLang 0.1.15 或兼容版本。建议在项目私有 Python 环境中安装，已有满足要求的 PyTorch 环境可保留其版本。

| 目标设备 | 运行与构建依赖 |
|---|---|
| NVIDIA GPU | CUDA PyTorch、完整 CUDA Toolkit（含 NVCC）、C++20 编译器；Triton 实现需安装与 PyTorch 匹配的 Triton |
| 昇腾 NPU | `torch_npu`、CANN、C++20 工具链及 Python 开发头文件；先手动 source 所选 CANN |

项目依赖以 [pyproject.toml](../pyproject.toml) 为准。`requirements.txt` 提供基础安装转发入口。源码构建使用 Ninja、setuptools 和 wheel；兼容依赖已安装时，可执行 `python -m pip install --no-deps --no-build-isolation -e .`。

## 获取并安装

```bash
git clone --recurse-submodules https://github.com/JPWang-CS/xpu-kernels.git
cd xpu-kernels
```

### NVIDIA GPU

```bash
python -m pip install -e '.[gpu,mhc-gpu,dev]'
python -m xpu_kernels.cli check-env --backend gpu
python -m xpu_kernels.cli build --backend gpu
python -m xpu_kernels.cli example --device cuda --operator rms_norm --shape 128x4096
python -m xpu_kernels.cli example --device cuda --operator mhc_post_backward --shape 2x1024x4x6144
```

GPU RMSNorm 默认选择 CUDA C++；可在 Python API 中使用 `implementation="triton"` 选择 Triton。GPU mHC 使用 TileLang 实现。RMSNorm 的 CUDA Toolkit 路径应通过 `CUDA_HOME` 指向含 `include/cuda_runtime.h` 的完整 Toolkit 根目录。

### 昇腾 NPU

先手动 source 已选定的 CANN，再从项目根目录执行：

```bash
python -m pip install -e '.[dev]'
python -m xpu_kernels.cli check-env --backend npu
python -m xpu_kernels.cli build --backend npu
python -m xpu_kernels.cli example --device npu --operator rms_norm --shape 128x4096
python -m xpu_kernels.cli example --device npu --operator mhc_post_backward --shape 2x1024x4x6144
```

RMSNorm 使用 Ascend C。NPU 构建需要 Bisheng、`ld.lld`、`adv_api`、支持 C++ `<format>` 的工具链、`Python.h` 和 elfutils 开发头文件；可用 `CXX` 指定项目选用的编译器，`check-env` 会显示实际检测结果。mHC 当前设备实现面向 Ascend 950；输入条件和原生/优化路径见 [mHC 算子说明](operators/mhc-post-backward.md)。首次调用通过 DeepJIT 编译 NPU kernel，后续调用复用缓存，无需将算子安装到公共 OPP。

## Python API

RMSNorm 的公开接口是 `rms_norm(x, weight, eps=1e-5, out=None, *, implementation=None)`。张量沿最后一维归一化；`weight` 为匹配最后一维的连续一维张量。传入同形状、同 dtype、同设备的 `out` 可复用输出缓冲区。

```python
import torch
from xpu_kernels import rms_norm

x = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(4096, device=x.device, dtype=x.dtype)
y = rms_norm(x, weight)
y_triton = rms_norm(x, weight, implementation="triton")
```

mHC 函数 `mhc_post_backward(grad_y, x, h_res, h_out, h_post, out=None, *, implementation="auto")` 返回 `(grad_x, grad_h_res, grad_h_out, grad_h_post)`。GPU 可用 `auto`、`optimized` 或 `tilelang` 选择同一 TileKernels 路径；NPU 使用 `auto`、`native` 或 `optimized` 选择对应路径。完整张量形状、dtype 和设备支持范围见 [算子文档](operators/mhc-post-backward.md)。

## 构建缓存与 wheel

源码安装的构建结果和运行缓存保存在项目的 `build/`、`cache/`、`tmp/`。wheel 携带设备源码、DeepJIT 头文件及第三方许可证；使用 wheel 时，将项目根目录与可写缓存目录明确传给运行时：

```bash
export XPU_KERNELS_PROJECT_ROOT="$(pwd)"
export XPU_KERNELS_CACHE_DIR="$XPU_KERNELS_PROJECT_ROOT/cache/xpu-kernels"
mkdir -p "$XPU_KERNELS_CACHE_DIR"
```

## 设备验证记录

0.3.0 GPU wheel 已完成独立安装与设备运行，RMSNorm 的三种精度及 mHC BF16 示例通过。源码测试 46 项通过、5 项跳过，mHC 测试覆盖四个完整输出、非默认 stream 和输出复用。[验证报告](validation.md)保留测试范围和环境信息；[Benchmark 与 Profiler](benchmarking.md)提供复现命令。
