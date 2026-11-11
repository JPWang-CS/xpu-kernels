# xpu-kernels

GPU/NPU 算子库。首个算子是 RMSNorm：GPU 默认使用 CUDA C++，也可显式选择 Triton，昇腾 NPU 后端使用 Ascend C，并由 DeepJIT 管理运行时编译、加载和缓存。Golden 独立实现 Pangu RMSNorm 的数值顺序，供正确性验收使用。

## 接口

```python
import torch
from xpu_kernels import rms_norm

x = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(4096, device=x.device, dtype=x.dtype)
y = rms_norm(x, weight, eps=1e-5)  # CUDA C++
y_triton = rms_norm(x, weight, eps=1e-5, implementation="triton")
```

`rms_norm(x, weight, eps=1e-5, out=None, *, implementation=None)` 接受连续的 float32、bfloat16、float16 输入。`x` 可有任意前导维度，最后一维为归一化宽度；`weight` 是同设备、同 dtype 的一维向量。指定 `out` 时，形状、设备和 dtype 必须与输入一致。设备决定后端；缺少该设备的实现或依赖时会明确报错，不会静默运行 golden。

NPU 按行分配片上缓冲区，启动前查询 UB 容量；宽行复用归约工作区存放权重。实测宽度覆盖 1 至 16385，包含对齐与非对齐形状；超出可用片上容量会明确报错。

Golden 先在 float32 中计算 `mean(x*x)` 和归一化值，再将归一化结果转回输入 dtype，最后乘以同 dtype 的 `weight`。输出保持输入 dtype。此舍入顺序作为首版的数值合同。

## 安装与运行

Python 基础依赖为 PyTorch；GPU 的 CUDA C++ 实现需要 NVCC、C++20 编译器和完整 CUDA Toolkit；Triton 实现需要与 PyTorch/CUDA 匹配的 Triton。NVCC 入口为软链接时，`CUDA_HOME` 应指向包含 `include/cuda_runtime.h` 的完整 Toolkit 根目录。NPU 需要 Linux、PyTorch、torch_npu、CANN 的 Bisheng/ld.lld/adv_api、C++20 工具链，以及仓库内的 DeepJIT 头文件。已验证的软件组合见下表。

克隆时初始化 DeepJIT 子模块：

```bash
git clone --recurse-submodules https://github.com/JPWang-CS/xpu-kernels.git
cd xpu-kernels
git submodule update --init --recursive
```

Python 依赖包含编译扩展所需的 Ninja；NVCC、CANN 和 C++ 工具链由当前设备环境提供。

NPU host extension 编译还需要 Python 开发头文件（`Python.h`）、C++ 标准库 `<format>`，以及 elfutils 开发头文件（`elfutils/libdwfl.h`）。DeepJIT 运行时通过 `dlopen` 查找 libdw，缺少运行库时只影响增强 stack trace；host extension 只链接 `-ldl`，不强制链接 `-ldw/-lelf`。`CXX` 指向当前私有 C++20 编译器时，`check-env` 会报告该选择。

wheel 与 sdist 会携带 CUDA/NPU host/device 源码和 DeepJIT 头文件。wheel 安装到只读环境时，首次 GPU/NPU 调用前将 `XPU_KERNELS_CACHE_DIR` 指到当前项目可写目录；源码 checkout/editable install 默认把 Triton、PyTorch extension 和 DeepJIT 缓存放在仓库 `cache/`。

```bash
python -m pip install -e '.[gpu,dev]'
python -m xpu_kernels.cli check-env --backend gpu
python -m xpu_kernels.cli example --device cuda
python -m xpu_kernels.cli test --device cuda
python -m xpu_kernels.cli bench --device cuda --warmup 20 --batches 10 --iterations-per-batch 50
```

NPU 容器中先从项目根目录构建 host extension，再运行同一套 correctness/benchmark 命令：

```bash
python -m pip install -e '.[dev]'
python -m xpu_kernels.cli check-env --backend npu
python -m xpu_kernels.cli build --backend npu
python -m xpu_kernels.cli example --device npu
python -m xpu_kernels.cli test --device npu
python -m xpu_kernels.cli bench --device npu --warmup 20 --batches 10 --iterations-per-batch 50
```

`build`、JIT 编译产物、Python/Triton 扩展缓存均指向项目内 `build/`、`cache/` 和 `tmp/`。`test` 先执行多 shape、三种 dtype 的 correctness，再检查目标设备测试确实运行；没有可运行设备、零个测试或目标用例被跳过都会失败。`bench` 用 `torch.cuda.Event` 或 `torch.npu.Event` 记录当前设备流区间，按多个 batch 测量并复用预分配输出。默认测量 GPU CUDA C++、Triton、基础算子组合和 NPU AscendC，使用同一组 48 个 shape 与三种精度。GPU 表中的“单算子”指基础操作在 GPU 上组合执行，所有临时张量预分配。Golden 只检查正确性；框架 RMSNorm API 可用 `--native-reference` 额外测量。历史 best 只从同设备、硬件、shape、dtype、软件版本、Event 方法和 eager 模式的 xpu-kernels 测量中比较。可提交 JSONL 记录软件版本，不记录本机 CANN、编译器或缓存路径。

## 独立 profiler

性能数据与 profiler trace 分开采集。使用项目 skill 中的命令前先完成正确性与同 case 重复 benchmark。GPU 可用 `nsys` 或 `ncu`；NPU 使用 `msprof`。每种工具单独运行，不混用 graph capture、编译阶段和正式测量区间。

```bash
python -m xpu_kernels.cli profile --device npu --tool msprof --output results/profiles/npu --dtype bfloat16
python -m xpu_kernels.cli profile --device cuda --tool nsys --output results/profiles/gpu
python -m xpu_kernels.cli profile --device cuda --tool ncu --output results/profiles/gpu-ncu --dtype bfloat16
```

Benchmark 默认把 JSONL 写到 `results/benchmarks/`；Profiler 默认把原始工具输出写到 `results/profiles/`。Profiler 先单独核对一个 shape/dtype，再启动包内 workload；msprof 通过 `msprof --help` 检查 `--application` 和 `--output` 后传入应用。NCU 仅收集选定 RMSNorm 实现的首个匹配 launch、Replay mode kernel、SpeedOfLight/LaunchStats，且关闭 cache/clock control。结果表和 JSONL 白名单位于 `results/benchmarks/`，原始 profiler 产物不纳入 Git。

## DeepJIT

`third-party/deep-jit` 是 DeepSeek 官方 header-only C++20 runtime 子模块。其上游 README 说明 CUDA 需要 CUDA headers 12.4+ / NVCC 12.9+，Ascend 需要 Bisheng、ld.lld、ACL/torch_npu headers 和 adv_api。NPU 复用其 runtime；CUDA C++ 使用 PyTorch extension JIT，Triton 使用自身编译缓存。


## 验证结果与后续迭代

| 后端 | 已实算设备 | 软件组合 |
|---|---|---|
| GPU | NVIDIA L20X，CC 9.0 | PyTorch 2.13.0+cu126，Triton 3.7.1 |
| NPU | Ascend950DT_9582 | PyTorch 2.9.0、torch_npu 2.9.0.post7.dev20260817、CANN 9.2.0 B096、GCC 13.4 |

两端均已完成多形状、三种精度、边界和 stream 验证；同一个 wheel 已分别安装并运行。NCU 与 msprof 入口已实际采集报告。速度、测量条件及原始样本见[性能记录](docs/performance/README.md)。目前只有 RMSNorm，后续算子逐个加入同一套安装、验证和性能记录流程。

## 通过 AI 使用项目

让 AI 先读取根 `AGENTS.md`，随后按唯一的[项目 skill](skills/xpu-kernels/SKILL.md)检查环境、安装、验证和测量。Python 依赖以 `pyproject.toml` 为准；`requirements.txt` 提供普通 `pip install -r requirements.txt` 入口。

已有兼容的 PyTorch/后端环境时，可用 `python -m pip install --no-deps --no-build-isolation -e .` 安装本项目，保留现有框架版本。NPU 自定义 kernel 由 DeepJIT 编译到项目缓存，无需安装到公共 OPP。
