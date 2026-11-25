# mHC Post Backward

`mhc_post_backward` 是 mHC 残差混合的反向算子。它接收上游梯度和前向保存的混合量，一次返回四个梯度：`grad_x`、`grad_h_res`、`grad_h_out`、`grad_h_post`。

## 目录

- [接口与公式](#接口与公式)：输入、输出、四个反向式和 dtype。
- [当前范围与选择规则](#当前范围与选择规则)：GPU、NPU 原生和 NPU 优化路径的边界。
- [最小运行示例](#最小运行示例)：构造输入并调用公共 API。
- [优化设计](optimization.md)：数据流、Cube/Vector 分工、流水和精度取舍。

## 接口与公式

公共接口是：

```text
mhc_post_backward(
    grad_y, x, h_res, h_out, h_post, out=None, *, implementation="auto"
)
```

输入布局为连续的 `[T, 4, D]` 或 `[B, S, 4, D]`。四维输入的 token 数按 `T = B × S` 计算；`N=4` 是当前公共 API 和优化内核固定支持的残差流数。各输入的形状和 dtype 为：

| 输入 | 形状 | dtype | 含义 |
|---|---|---|---|
| `grad_y` | `[..., 4, D]` | 与 `x` 相同，FP16/BF16 | 混合输出 `Y` 的上游梯度 `grad_y` |
| `x` | `[..., 4, D]` | FP16/BF16 | 四路残差输入 `x` |
| `h_res` | `[..., 4, 4]` | FP32 | 残差混合矩阵 `h_res` |
| `h_out` | `[..., D]` | 与 `x` 相同 | 分支输出 `h_out` |
| `h_post` | `[..., 4]` | FP32 | 后处理混合权重 `h_post` |

`x` 保存旧的四条残差流，`grad_y` 是上游梯度。其余三个输入的作用如下：

- `h_res`：`4×4` 混合系数，决定旧流怎样组合成新流。
- `h_out`：长度为 `D` 的子层输出向量，例如 Attention 或 MoE 的输出。
- `h_post`：4 个系数，决定把子层输出按多大比例加回每条流。

令前向混合为：

$$
Y_{j,d}=\sum_i \mathrm{h\_res}_{i,j}x_{i,d}+\mathrm{h\_post}_{j} \mathrm{h\_out}_{d}.
$$

`i` 是旧残差流编号，`j` 是新残差流编号，两者取 0 到 3；`d` 是向量中的特征位置，取 0 到 `D-1`。因此，`h_res[i,j]` 决定旧流 `i` 对新流 `j` 的贡献，`h_post[j]` 决定子层输出 `h_out` 对新流 `j` 的贡献。

当前独立 golden 在 CPU 上先把输入转为 FP64，再逐项计算以下四个输出，最后转换到公共 API 规定的输出 dtype：

$$
\begin{aligned}
\mathrm{grad\_x}_{i,d} &= \sum_j \mathrm{h\_res}_{i,j}\mathrm{grad\_y}_{j,d},\\
\mathrm{grad\_h\_res}_{i,j} &= \sum_d x_{i,d}\mathrm{grad\_y}_{j,d},\\
\mathrm{grad\_h\_out}_{d} &= \sum_j \mathrm{h\_post}_{j}\mathrm{grad\_y}_{j,d},\\
\mathrm{grad\_h\_post}_{j} &= \sum_d \mathrm{h\_out}_{d}\mathrm{grad\_y}_{j,d}.
\end{aligned}
$$

返回值严格按下面顺序排列：

```python
grad_x, grad_h_res, grad_h_out, grad_h_post
```

`grad_x` 和 `grad_h_out` 使用输入数据 dtype；`grad_h_res` 和 `grad_h_post` 使用 FP32。`out` 可以传入四个连续、形状和 dtype 匹配的输出张量组成的 tuple；输入、输出必须在同一设备上，输出不能与输入或其他输出重叠。独立参考的当前实现见 [mhc_post_backward_golden.py](../../../src/xpu_kernels/mhc_post_backward_golden.py)，公共参数校验和 dispatch 见 [mhc_post_backward.py](../../../src/xpu_kernels/mhc_post_backward.py)。

## 当前范围与选择规则

公共 API 的共同边界是：

- 输入为连续的 TND 或 BSND，且最后两维为 `[4, D]`；
- `512 ≤ T ≤ 49152`，`1 ≤ D ≤ 8192`；
- 当前设备必须是 CUDA 或 Ascend NPU；缺少设备实现时不会静默退回 golden；
- NPU 支持 FP16/BF16，GPU 当前 TileLang 路径支持 BF16，且 `D` 必须是 128 的倍数。

实现选择按设备分别处理：

| 设备 | `implementation` | 实际路径 | 选择条件 |
|---|---|---|---|
| CUDA | `auto` / `optimized` / `tilelang` | DeepSeek TileKernels TileLang | 当前 GPU mHC 实现；BF16、`D` 为 128 倍数 |
| Ascend 950 | `native` | 已安装 CANN 主线 `torch_npu.npu_mhc_post_backward` | 直接调用原生算子 |
| Ascend 950 | `auto` | 优化内核或 CANN 主线 | 只有满足下表优化范围时选优化，否则走原生 |
| Ascend 950 | `optimized` | 本库 Ascend C 混合 Cube/Vector 内核 | 不满足优化范围时直接报错 |

当前 NPU 优化范围由 [mhc_post_backward.py](../../../src/xpu_kernels/mhc_post_backward.py) 的 `optimized_scope` 和 [mhc_post_backward.cpp](../../../csrc/ascend/mhc_post_backward.cpp) 的 dispatch 共同限定：

| 条件 | 当前优化范围 |
|---|---|
| SoC | Ascend 950；运行时还要求 dav-3510 tiling 架构 |
| dtype | BF16 |
| 残差流 | `N=4`，`h_res` 有效 |
| token 数 | `2048 ≤ T ≤ 49152` |
| 特征宽度 | `D ∈ {2560, 5120, 6144, 7168}` |

范围外的 NPU 输入仍可以使用 `implementation="native"`，或者由 `implementation="auto"` 交给 CANN 主线；这不等于本库优化内核支持范围扩大。当前适配器还会检查设备、连续性、输入输出 shape、FP32 混合权重、输出 dtype、无重叠和 token/D 上限。

## 最小运行示例

GPU 适配层沿用 DeepSeek TileKernels 的计算实现。上游返回 `(da, db, dc, dd)`，本库按公开参数顺序整理为四个梯度：

| 上游结果 | 本库返回项 |
|---|---|
| `db` | `grad_x` |
| `da` | `grad_h_res` |
| `dd` | `grad_h_out` |
| `dc` | `grad_h_post` |

适配层同时将前导维度展平为 token 轴，返回时恢复原形状。

在仓库根目录安装依赖。GPU 需要 mHC extra；NPU 运行前先手动 source 选定的 CANN。环境说明见[安装指南](../../getting-started.md)。

```bash
python -m pip install -e '.[gpu,mhc-gpu,dev]'
python -m xpu_kernels.cli build --backend gpu --operator mhc_post_backward
python -m xpu_kernels.cli test --device cuda --operator mhc_post_backward
```

```python
import torch
from xpu_kernels import mhc_post_backward

device = "cuda"
dtype = torch.bfloat16
B, S, N, D = 2, 1024, 4, 6144

x = torch.randn(B, S, N, D, device=device, dtype=dtype)
grad_y = torch.randn_like(x)
h_res = torch.randn(B, S, N, N, device=device, dtype=torch.float32)
h_out = torch.randn(B, S, D, device=device, dtype=dtype)
h_post = torch.randn(B, S, N, device=device, dtype=torch.float32)

grad_x, grad_h_res, grad_h_out, grad_h_post = mhc_post_backward(
    grad_y, x, h_res, h_out, h_post, implementation="auto"
)
```

如果要明确比较 NPU 原生与优化路径，可以分别使用：

```bash
python -m xpu_kernels.cli bench --device npu --operator mhc_post_backward \
  --implementations native,optimized --warmup 20 --batches 10 --iterations-per-batch 50
```

正确性应先对照独立 FP64 golden，再解释性能。

## 资料导航

- [优化设计](optimization.md)：当前 NPU 优化内核的公式映射、代码路由、数据搬运和流水。
- [公共安装指南](../../getting-started.md)：环境、构建和依赖。
- [Python API 与 dispatch](../../../src/xpu_kernels/mhc_post_backward.py)：公共范围和实现选择。
- [独立 FP64 golden](../../../src/xpu_kernels/mhc_post_backward_golden.py)：四个输出的参考定义。
- [Ascend host adapter](../../../csrc/ascend/mhc_post_backward.cpp)：设备校验、优化范围和 JIT 参数。
- [Ascend kernel entry](../../../kernels/include/xpu_kernels/mhc_post_backward/kernel.hpp)：混合 Cube/Vector kernel 入口。
