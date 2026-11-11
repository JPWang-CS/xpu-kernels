<div align="center">

<h1>xpu-kernels</h1>
<p><strong>面向大模型的 GPU / NPU 算子库</strong></p>
<p>CUDA C++ &nbsp;·&nbsp; Triton &nbsp;·&nbsp; Ascend C</p>

<p>
  <a href="#operators">算子</a> &nbsp;·&nbsp;
  <a href="#performance">性能</a> &nbsp;·&nbsp;
  <a href="#quick-start">快速开始</a> &nbsp;·&nbsp;
  <a href="#benchmark">测试与分析</a> &nbsp;·&nbsp;
  <a href="skills/xpu-kernels/SKILL.md">项目 Skill</a>
</p>

</div>

---

以同一 Python 接口调用 NVIDIA GPU 与昇腾 NPU 算子。每个算子维护独立 golden、多 shape 正确性检查、实测性能与历史记录，优化结果随代码一起保存。

| 已收录算子 | 实测设备 | RMSNorm shapes | 正确性与性能记录 |
|:---:|:---:|:---:|:---:|
| **1** | **2** | **48** | **576** |

<a id="operators"></a>

## 算子

| 算子 | GPU | NPU | 精度 | 性能记录 |
|---|---|---|---|---|
| **RMSNorm** | CUDA C++ / Triton | Ascend C | FP32 / FP16 / BF16 | [48 个 shape · 完整对比](docs/performance/README.md) |

GPU 默认使用 CUDA C++，可显式选择 Triton；NPU 使用 Ascend C，由 DeepJIT 编译、加载并缓存。两端支持多维连续输入，沿最后一维归一化，并使用当前 PyTorch stream。

<a id="performance"></a>

## 性能精选

**RMSNorm · BF16 · μs/次**。加粗值为同一 shape 下三种 GPU 实现中的最低耗时。

<!-- BEGIN homepage-performance -->
| Shape | NPU 950DT | GPU L20X | GPU L20X（Triton） | GPU L20X（单算子拼接） |
| --- | ---: | ---: | ---: | ---: |
| 128 × 4096 | 9.569 | **7.011** | 16.201 | 36.486 |
| 129 × 4097 | 9.424 | **7.268** | 16.594 | 37.841 |
| 1024 × 4096 | 25.472 | **9.518** | 16.288 | 82.181 |
| 4096 × 8192 | 128.972 | 58.522 | **37.066** | 493.193 |

相对 GPU 单算子拼接，128×4096 的 CUDA 加速比为 5.20×，4096×8192 的 Triton 加速比为 13.31×。
<!-- END homepage-performance -->

“单算子拼接”是在 GPU 上依次执行平方、均值、加 epsilon、倒数平方根、归一化及权重乘法，输出与中间缓冲区预分配。表中时间来自 eager 模式的设备 Event，包含 CPU 提交间隙；每项预热 20 次，测量 10 批、每批 50 次，取批均值的中位数。NPU 与 GPU 分别测量。

[全部 shape 与三种精度](docs/performance/README.md) &nbsp;·&nbsp; [GPU 原始样本](results/benchmarks/rmsnorm-gpu-v2.jsonl) &nbsp;·&nbsp; [NPU 原始样本](results/benchmarks/rmsnorm-npu-v2.jsonl)

<a id="quick-start"></a>

## 快速开始

需要 Python 3.10+ 和与设备匹配的 PyTorch。GPU 需要完整 CUDA Toolkit、NVCC 与 C++20 编译器；Triton 需匹配当前 PyTorch/CUDA。NPU 需要 torch_npu、CANN、C++20 工具链及 Python 开发头文件。

```bash
git clone --recurse-submodules https://github.com/JPWang-CS/xpu-kernels.git
cd xpu-kernels
```

<details open>
<summary><strong>NVIDIA GPU</strong></summary>

在项目私有 Python 环境中执行：

```bash
python -m pip install -e '.[gpu,dev]'
python -m xpu_kernels.cli check-env --backend gpu
python -m xpu_kernels.cli build --backend gpu
python -m xpu_kernels.cli example --device cuda
```

</details>

<details>
<summary><strong>昇腾 NPU</strong></summary>

先手动 source 已选择的 CANN，再在项目私有 Python 环境中执行：

```bash
python -m pip install -e '.[dev]'
python -m xpu_kernels.cli check-env --backend npu
python -m xpu_kernels.cli build --backend npu
python -m xpu_kernels.cli example --device npu
```

自定义 kernel 编译到项目缓存，无需安装到公共 OPP。

</details>

已有兼容的框架环境时，可用 `python -m pip install --no-deps --no-build-isolation -e .` 保留框架版本；需要当前环境已具备 Ninja、setuptools 和 wheel。

### Python 接口

```python
import torch
from xpu_kernels import rms_norm

x = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(4096, device=x.device, dtype=x.dtype)

y = rms_norm(x, weight)  # GPU 默认 CUDA C++
y_triton = rms_norm(x, weight, implementation="triton")

# 已准备 NPU 张量时，使用同一接口：rms_norm(x_npu, weight_npu)
```

`rms_norm(x, weight, eps=1e-5, out=None, *, implementation=None)` 接受连续的 FP32、FP16、BF16 张量。`weight` 是同设备、同 dtype 的一维向量，长度等于输入最后一维；可传入同形状的 `out` 复用输出缓冲区。

Golden 遵循 Pangu RMSNorm 的数值顺序：在 FP32 中计算均方和归一化值，将归一化结果转回输入 dtype，再乘 weight。输出保持输入 dtype；golden 仅用于正确性检查。

<a id="benchmark"></a>

## 测试与性能分析

先检查正确性，再独立测量性能。默认 benchmark 覆盖 48 个 shape 与三种精度；GPU 测量 CUDA C++、Triton、单算子拼接，NPU 测量 Ascend C。

```bash
python -m xpu_kernels.cli test --device cuda
python -m xpu_kernels.cli bench --device cuda \
  --warmup 20 --batches 10 --iterations-per-batch 50

# 只测选定的形状与实现
python -m xpu_kernels.cli bench --device cuda \
  --shapes 128x4096,129x4097 --dtypes bfloat16 --implementations cuda,triton,torch
```

NPU 将 `--device cuda` 换成 `--device npu`；指定实现时使用 `--implementations ascendc`。没有可用设备或目标测试被跳过时，设备验收会报错。

<details>
<summary><strong>独立采集 NCU / Nsight Systems / msprof</strong></summary>

```bash
python -m xpu_kernels.cli profile --device cuda --tool ncu \
  --implementation cuda --shape 128x4096 --dtype bfloat16 \
  --output results/profiles/gpu-ncu

python -m xpu_kernels.cli profile --device cuda --tool nsys \
  --implementation cuda --shape 128x4096 --dtype bfloat16 \
  --output results/profiles/gpu-nsys

python -m xpu_kernels.cli profile --device npu --tool msprof \
  --implementation ascendc --shape 128x4096 --dtype bfloat16 \
  --output results/profiles/npu-msprof
```

NCU 收集一个匹配 launch，使用 kernel replay、SpeedOfLight / LaunchStats，并保留设备时钟与缓存状态；msprof 根据本机 `--help` 检查 `--application` 与 `--output`。GPU 可切换为 `--implementation triton`。

Event benchmark 与 profiler 独立运行。NCU、msprof 已实际采集；[指标及样本摘要](docs/performance/rmsnorm-v2-profiler.json)保留在仓库，原始 trace 留在项目运行目录。

</details>

Benchmark 保存设备型号、shape、dtype、软件版本、每批样本及匹配的历史最佳。历史比较按实现、设备与软件环境分别进行；框架原生 RMSNorm 可用 `--native-reference` 额外测量。[记录格式](results/benchmarks/README.md)

### 已验证环境

| 项目 | GPU | NPU |
|---|---|---|
| 设备 | NVIDIA L20X，CC 9.0 | Ascend950DT_9582 |
| Python / PyTorch | 3.12.3 / 2.13.0+cu126 | 3.11.18 / 2.9.0+cpu |
| 后端软件 | CUDA runtime 12.6，NVCC 12.8.61，Triton 3.7.1 | torch_npu 2.9.0.post7.dev20260817，CANN 9.2.0 B096 |
| C++ 编译器 | GCC 13.3 | GCC 13.4 |

同一个 wheel 已分别在两端安装，并从包内源码编译、执行通过；验证包含多 shape、三种精度、边界输入、输出复用和非默认 stream。

<details>
<summary><strong>编译依赖与缓存设置</strong></summary>

- **GPU**：NVCC、C++20 编译器和完整 CUDA Toolkit。若 `nvcc` 是软链接，`CUDA_HOME` 应指向包含 `include/cuda_runtime.h` 的 Toolkit 根目录。CUDA C++ 通过 PyTorch extension JIT 编译，Triton 使用自身编译缓存。
- **NPU**：CANN 的 Bisheng、ld.lld、adv_api；C++ 标准库 `<format>`；Python 开发头文件 `Python.h`；elfutils 开发头文件 `elfutils/libdwfl.h`。通过 `CXX` 指定私有编译器，`check-env` 会显示实际选择。host extension 链接 `-ldl`，DeepJIT 在运行时查找 libdw，用于增强 stack trace。
- **打包与缓存**：wheel、sdist 携带 CUDA/NPU host/device 源码及 DeepJIT 头文件。源码安装使用仓库 `cache/`、`tmp/`；wheel 安装后设置 `XPU_KERNELS_CACHE_DIR` 为项目内可写缓存目录。CLI 在仓库根目录运行，或设置 `XPU_KERNELS_PROJECT_ROOT`。
- **NPU 宽行**：启动前检查 UB 容量，宽行复用归约工作区保存权重。当前实测宽度覆盖 1 至 16385；超过片上容量时会明确报错。

</details>

## 通过 AI 使用

让 AI 读取根 [AGENTS.md](AGENTS.md)，再按[项目 Skill](skills/xpu-kernels/SKILL.md)检查环境、构建、验证和测量。Skill 覆盖整个仓库，包含数值规则、设备选择、性能比较和 profiler 操作。

```text
读取 AGENTS.md 和项目 skill，在我的项目私有环境中检查依赖，
构建并验证 RMSNorm；正确性通过后测量性能，保存设备型号与原始结果。
```

依赖以 [pyproject.toml](pyproject.toml) 为准，`pip install -r requirements.txt` 提供基础安装入口。

## 致谢与许可证

NPU 运行时复用 [DeepJIT](https://github.com/deepseek-ai/DeepJIT) 的编译、加载与 stream 支持。DeepJIT 作为子模块保留上游许可证；本库采用 [Apache-2.0](LICENSE)。
