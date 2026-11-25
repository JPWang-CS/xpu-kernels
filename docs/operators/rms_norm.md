# RMSNorm

`rms_norm` 对输入最后一维计算均方根归一化，并逐元素乘以权重。公共 API 保持原签名；实现根据输入设备和显式 `implementation` 选择 CUDA、Triton 或 CANN 主线路径。

## Python 接口

```python
import torch
from xpu_kernels import rms_norm

x = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(4096, device=x.device, dtype=x.dtype)
y = rms_norm(x, weight)
```

公开签名保持为：

```python
rms_norm(x, weight, eps=1e-5, out=None, *, implementation=None)
```

`x` 至少有一维，最后一维宽度大于零；前导维度会展平为行。`weight` 是长度等于最后一维宽度的一维向量。输入和权重必须位于同一设备并使用相同 dtype；支持 FP32、FP16、BF16。NPU 的优化路径要求输入和权重连续。`eps` 必须为正数，默认值为 `1e-5`。传入 `out` 时，它必须与 `x` 形状、设备和 dtype 相同并保持连续。

## CANN 主线数值约定

数值合同以 CANN `ops-nn/norm/rms_norm` 的主线定义为准，参考入口为 [CANN ops-nn RMSNorm](https://gitcode.com/cann/ops-nn/tree/master/norm/rms_norm)。对一行输入 `x`、权重 `w` 和宽度 `D`，先计算：

$$
r = \left(\frac{1}{D}\sum_{i=0}^{D-1}x_i^2 + \varepsilon\right)^{-1/2},
\qquad
y_i = \mathrm{cast}_{\mathrm{dtype}(x)}(x_i r w_i).
$$

CANN 主线的顺序是 `x * rstd` 后乘 `weight`，最后再将结果转换到输入 dtype。归一化、权重乘法和输出 cast 的中间精度由对应设备实现负责。

`xpu_kernels.rms_norm_golden.rms_norm_golden` 是这个合同的独立参考实现。GPU、NPU CANN native 和 NPU 目标范围优化都使用对应设备实现；缺少设备实现、CANN、编译器或运行时依赖时直接报错。

## 实现选择

### CUDA

| `implementation` | 行为 |
|---|---|
| `None` 或 `"cuda"` | CUDA C++ |
| `"triton"` | Triton |

CUDA 输入不接受 NPU 专用选项。

### 昇腾 NPU

| `implementation` | 行为 |
|---|---|
| `None` | 等价于 `"auto"` |
| `"auto"` | 目标范围走新的 CANN 优化 kernel，范围外走 `torch_npu.npu_rms_norm` |
| `"ascendc"` | `"auto"` 的别名 |
| `"native"` | 始终调用 CANN 主线 `torch_npu.npu_rms_norm` |
| `"optimized"` | 强制使用目标范围优化 kernel；输入不满足范围时抛错 |

`auto` 选择新的优化 kernel 必须同时满足：

- Ascend950；
- BF16 `x`、`weight` 和输出 `y`；
- 输入与权重连续；
- 最后一维 `D=20480`；
- 展平后的行数 `M` 为 `2048` 或 `4096`；
- `eps=1e-5`。

其中 `M=2048` 使用两行保留槽的候选，`M=4096` 使用三行预取槽、两段 Y 写出的候选。两个候选通过同一个 DeepJIT 入口 `rms_norm` 导出，编译单元用 `XPU_RMS_ROW_SLOTS=2/3` 选择实现。两种配置共用数值公式与归约顺序，按行数选择预取和写出方式。

范围外的 NPU 输入，包括相邻的 `M`、其他 `D`、非 BF16、非连续布局、不同 `eps` 或其他硬件，`auto` 均调用 CANN 主线。

## 构建与运行

先按[安装指南](../getting-started.md)准备项目自己的构建缓存，并手动 source 选定的 CANN 环境：

```bash
python -m xpu_kernels.cli check-env --backend npu
python -m xpu_kernels.cli build --backend npu --operator rms_norm
python -m xpu_kernels.cli test --device npu --operator rms_norm
```

GPU 路径：

```bash
python -m xpu_kernels.cli check-env --backend gpu
python -m xpu_kernels.cli build --backend gpu --operator rms_norm
python -m xpu_kernels.cli test --device cuda --operator rms_norm
```

首次调用时，DeepJIT 会按 `XPU_RMS_ROW_SLOTS` 编译并缓存对应编译单元；缓存必须位于项目可写目录。设备测试与性能测试命令如下。

## 正确性与性能验收

正确性先于 benchmark。设备输出必须和 CANN 主线合同一致；优化验收采用精度标准 2.1 的双标杆：

1. 将同一份输入量化到实际输入 dtype，再用 FP64 计算 `Y` 和 `rstd` golden；
2. 在真实 GPU 上用同一输入执行 benchmark 实现；
3. 对 actual 与 golden、GPU benchmark 与 golden 分别逐输出检查 `Y` 和 `rstd`，并记录输出 dtype、误差和判定。

没有真实 GPU 双标杆时，记录为“待 GPU 双标杆”。设备性能在 correctness gate 通过后，使用匹配的硬件、shape、dtype、分配策略、stream 和采样方法测量。
