# 安装与使用

需要 Python 3.10+ 和与设备匹配的 PyTorch。使用项目私有 Python 环境；已有完整框架环境时保留原版本。

| 后端 | 依赖 |
|---|---|
| NVIDIA GPU | 完整 CUDA Toolkit、NVCC、C++20 编译器；Triton 与 PyTorch 匹配 |
| 昇腾 NPU | torch_npu、CANN、C++20 工具链、Python 开发头文件；先手动 source CANN |

## 获取源码

```bash
git clone --recurse-submodules https://github.com/JPWang-CS/xpu-kernels.git
cd xpu-kernels
```

依赖以 [pyproject.toml](../pyproject.toml) 为准；`pip install -r requirements.txt` 提供基础安装入口。已有兼容依赖时使用 `python -m pip install --no-deps --no-build-isolation -e .`；构建环境须已具备 Ninja、setuptools 与 wheel。

## NVIDIA GPU

```bash
python -m pip install -e '.[gpu,dev]'
python -m xpu_kernels.cli check-env --backend gpu
python -m xpu_kernels.cli build --backend gpu
python -m xpu_kernels.cli example --device cuda
```

GPU 默认使用 CUDA C++；`implementation="triton"` 选择 Triton。

## 昇腾 NPU

先手动 source 已选择的 CANN，再执行：

```bash
python -m pip install -e '.[dev]'
python -m xpu_kernels.cli check-env --backend npu
python -m xpu_kernels.cli build --backend npu
python -m xpu_kernels.cli example --device npu
python -m xpu_kernels.cli example --device npu --operator mhc_post_backward --shape 2x1024x4x6144
```

首次调用通过 DeepJIT 编译 kernel，后续复用项目内缓存，无需向公共 OPP 安装算子。

## Python API

```python
import torch
from xpu_kernels import rms_norm

x = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(4096, device=x.device, dtype=x.dtype)
y = rms_norm(x, weight)
y_triton = rms_norm(x, weight, implementation="triton")
```

`rms_norm(x, weight, eps=1e-5, out=None, *, implementation=None)` 沿连续输入的最后一维归一化。`weight` 是同设备、同精度的一维向量；可用同形状的 `out` 复用输出。NPU 张量使用相同接口。

[mHC Post Backward 的输入、公式与完整代码](operators/mhc-post-backward.md)说明五个输入、四个梯度以及原生/优化路径的选择。其优化路径支持 BF16、4 条残差流、h_res 存在，token 数 2048–49152，D 为 2560、5120、6144 或 7168；其它受当前接口支持的输入调用 CANN 原生实现。

## 实测环境

| 项目 | GPU | NPU |
|---|---|---|
| 设备 | NVIDIA L20X，CC 9.0 | Ascend950DT_9582 |
| Python / PyTorch | 3.12.3 / 2.13.0+cu126 | 3.11.18 / 2.9.0+cpu |
| 后端 | CUDA runtime 12.6，NVCC 12.8.61，Triton 3.7.1 | torch_npu 2.9.0.post7.dev20260817，CANN 9.2.0 B096 |
| C++ 编译器 | GCC 13.3 | GCC 13.4 |

0.2.0 wheel 已在两端独立安装，并从包内源码编译、运行通过。[设备验证](validation.md)

## 编译依赖与缓存

GPU 的 `CUDA_HOME` 指向包含 `include/cuda_runtime.h` 的完整 Toolkit 根目录。NPU 需要 Bisheng、ld.lld、adv_api、C++ `<format>`、Python.h 和 elfutils 开发头文件。通过 `CXX` 指定私有编译器，`check-env` 显示实际选择。

源码安装的编译结果保存在项目 `cache/`、`tmp/`。wheel 携带 host/device 源码和 DeepJIT 头文件；安装后将 `XPU_KERNELS_CACHE_DIR` 设为项目内可写目录。CLI 从仓库根运行，或设置 `XPU_KERNELS_PROJECT_ROOT`。

正确性、重复采样和性能分析见 [Benchmark 与 Profiler](benchmarking.md)。
