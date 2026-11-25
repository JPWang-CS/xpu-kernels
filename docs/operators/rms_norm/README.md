# RMSNorm

RMSNorm 沿输入的最后一维计算均方根归一化，再逐元素乘以权重 `gamma`。输入的前导维度保持不变；例如 `[B,S,D]` 可以看作 `B×S` 行，每行有 `D` 个数，各行独立计算。

[优化设计与实测结果](optimization.md) · [安装指南](../../getting-started.md)

## 计算什么

对第 `m` 行，先算一个归一化系数，再用它缩放这一行：

$$
a_m=\frac{1}{D}\sum_{d=0}^{D-1}x_{m,d}^{2}+\varepsilon,
\qquad r_m=\frac{1}{\sqrt{a_m}},
\qquad y_{m,d}=\mathrm{cast}_{\mathrm{dtype}(x)}(x_{m,d}r_m\gamma_d).
$$

`r_m` 就是代码中的 `rstd`。一行只有一个 `rstd`，它广播到这一行的所有元素；`gamma` 则有 `D` 个数，所有行共用。`eps` 是开根号前加入的正数。

实现按 CANN 主线的数值顺序，在 FP32 中计算归一化和权重乘法，最后转换为输入 dtype。独立 golden 使用同一份已经量化到输入 dtype 的数据重新以 FP64 计算，作为误差参考。

## 调用方法

```python
import torch
from xpu_kernels import rms_norm

x = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
gamma = torch.ones(4096, device=x.device, dtype=x.dtype)
y = rms_norm(x, gamma)
```

```text
rms_norm(x, weight, eps=1e-5, out=None, *, implementation=None)
```

| 参数或结果 | 形状与含义 |
|---|---|
| `x` | `[..., D]`，最后一维 `D>0` |
| `weight` | `[D]`，即公式中的 `gamma`，与 `x` 同 dtype、同设备 |
| `eps` | 有限正数，默认 `1e-5` |
| `out` | 可选的输出缓冲，与 `x` 同形状、dtype 和设备，连续布局 |
| 返回值 | `Y`，形状与 dtype 均与 `x` 相同 |

NPU 内部同时计算 FP32 `rstd`，形状为 `[...,1]`，用于验证和后续计算；公开函数返回 `Y`。

## 实现与适用范围

| 设备与选项 | 实际执行 |
|---|---|
| CUDA：默认或 `"cuda"` | CUDA C++ 实现 |
| CUDA：`"triton"` | Triton 实现 |
| NPU：默认、`"auto"` 或 `"ascendc"` | 满足下表时选择优化实现，其余输入调用 CANN 主线 |
| NPU：`"native"` | `torch_npu.npu_rms_norm` |
| NPU：`"optimized"` | 要求满足下表，参数超出范围时报告错误 |

公开接口支持 FP32、FP16、BF16。CUDA 实现要求输入、权重和输出连续；NPU 主线路径将合法的非连续输入交给所安装的 CANN 实现处理，其支持情况以该版本为准。空的前导维度产生同形状空输出。

当前 NPU 优化针对以下两种输入：

| 条件 | 范围 |
|---|---|
| 硬件 | Ascend 950，设备编译目标 `dav-3510` |
| 输入与输出 | 连续 BF16 `X`、`gamma`、`Y`；FP32 `rstd` |
| 行宽 | `D=20480` |
| 行数 | 前导维度展平后，`M` 为 **2048 或 4096** |
| epsilon | `1e-5` |
| 启动资源 | 64 个 Vector 核；每核需要 209664 字节 UB |

这里的行数是两个明确的值。例如 `M=2049` 会选择主线，而不会按相近大小套用优化配置。多维输入只要展平后满足条件，按相同方式分配行。

## 构建与验证

在仓库根目录准备所需依赖。GPU：

```bash
python -m pip install -e '.[gpu,dev]'
python -m xpu_kernels.cli check-env --backend gpu
python -m xpu_kernels.cli build --backend gpu --operator rms_norm
python -m xpu_kernels.cli test --device cuda --operator rms_norm
```

NPU 先手动 source 选定的 CANN，再执行：

```bash
python -m pip install -e '.[dev]'
python -m xpu_kernels.cli check-env --backend npu
python -m xpu_kernels.cli build --backend npu --operator rms_norm
python -m xpu_kernels.cli test --device npu --operator rms_norm
```

设备执行使用当前框架 stream。DeepJIT 在首次调用时编译设备入口，后续复用项目缓存；wheel 安装的缓存配置见安装指南。

优化精度按标准 2.1 验收：`Y` 与 `rstd` 分别比较目标实现、同输入真实 GPU 结果相对 FP64 golden 的误差。已有集成记录覆盖 42 组输入、84 项输出判定，包含 12 组优化执行与 30 组主线路径。记录和计时条件放在[实测结果](optimization.md#实测结果)中。
