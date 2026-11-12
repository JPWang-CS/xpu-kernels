# RMSNorm

`rms_norm` 对输入最后一维计算均方根归一化，并逐元素乘以可学习权重。统一 Python API 根据输入设备选择后端；CUDA 输入默认走 CUDA C++，`implementation="triton"` 选择 Triton，NPU 输入走 Ascend C。

## Python 接口

```python
import torch
from xpu_kernels import rms_norm

x = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(4096, device=x.device, dtype=x.dtype)
y = rms_norm(x, weight)
```

公开签名：

```python
rms_norm(x, weight, eps=1e-5, out=None, *, implementation=None)
```

`x` 至少有一维，最后一维宽度大于零；实现把前导维度展平为行。`weight` 是长度等于该宽度的一维向量。`x` 和 `weight` 必须连续、同设备、同 dtype；支持 FP32、FP16、BF16。`eps` 为正数，默认值为 `1e-5`。传入 `out` 时，它必须与 `x` 具有相同形状、设备、dtype，且保持连续。

## 数值约定

令一行输入为 `x`，宽度为 `D`，权重为 `w`。参考实现先用 FP32 累加平方和并计算归一化值，再将归一化结果转换到输入 dtype，最后乘以同 dtype 的权重：

$$
 r = \left(\frac{1}{D}\sum_{i=0}^{D-1}x_i^2 + \varepsilon\right)^{-1/2}, \qquad
 y_i = \mathrm{cast}_{\mathrm{dtype}(x)}(x_i r) \cdot w_i.
$$

独立参考为 `xpu_kernels.golden.rms_norm_golden`。NPU 的 CANN 原生算子可能采用不同的中间舍入顺序；做原生比较时按对应公式分别检查数值结果。

## 实现选择与运行

| 输入设备 | 默认实现 | 可选实现 |
|---|---|---|
| NVIDIA CUDA | CUDA C++ | Triton：`implementation="triton"` |
| 昇腾 NPU | Ascend C | `implementation="ascendc"` |

调用按输入张量所在设备选择表中实现；数值参考由 `xpu_kernels.golden.rms_norm_golden` 独立提供。空张量在输入和输出约束通过后返回对应空输出。

先按[安装指南](../getting-started.md)检查工具链和构建后端，再运行正确性用例：

```bash
python -m xpu_kernels.cli build --backend gpu --operator rms_norm
python -m xpu_kernels.cli test --device cuda --operator rms_norm

# NPU 前先手动 source 已选 CANN
python -m xpu_kernels.cli build --backend npu --operator rms_norm
python -m xpu_kernels.cli test --device npu --operator rms_norm
```

GPU BF16 的 48 个 shape 已对 CUDA C++、Triton、PyTorch composition 和 PyTorch 原生 RMSNorm 完成 192 条测量记录，使用 `--native-reference` 让所有实现每次调用分配输出。composition 每次调用也会创建中间张量。原始记录见 [rmsnorm-gpu-mainline.jsonl](../../results/benchmarks/rmsnorm-gpu-mainline.jsonl)；跨设备性能表见[性能索引](../performance/README.md)。重复 benchmark、精度与 shape 选择以及 profiler 命令见[性能复现指南](../benchmarking.md)。
