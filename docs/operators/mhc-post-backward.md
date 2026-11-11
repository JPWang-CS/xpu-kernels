# mHC Post Backward

`mhc_post_backward` 一次计算残差混合过程的四个梯度。NPU 实现将三个梯度的 Vector 运算融合，并让 Cube 计算残差混合矩阵的梯度；双缓冲连接数据搬运和计算，长维度归约保留 FP32 补偿累加。

## 输入与输出

用 `G` 表示上游梯度 `grad_y`，`X` 表示残差输入，`H` 表示 `h_res`，`u` 表示 `h_out`，`p` 表示 `h_post`。`N=4` 是残差流数量，`D` 是每条流的特征宽度。前导维度可以是 `T`，也可以是 `B,S`，后者的 token 数为 `B×S`。

| 输入 | 形状 | dtype |
|---|---|---|
| `grad_y`、`x` | `[T,4,D]` 或 `[B,S,4,D]` | FP16 / BF16 |
| `h_res` | `[T,4,4]` 或 `[B,S,4,4]` | FP32 |
| `h_out` | `[T,D]` 或 `[B,S,D]` | 与 `x` 相同 |
| `h_post` | `[T,4]` 或 `[B,S,4]` | FP32 |

函数返回 `(grad_x, grad_h_res, grad_h_out, grad_h_post)`。各输出的形状、dtype 分别对应 `x`、`h_res`、`h_out`、`h_post`。

对每个 token，四项计算是：

$$
\begin{aligned}
\mathrm{grad\_x}_{i,d}&=\sum_j H_{i,j}G_{j,d},\\
\mathrm{grad\_h\_res}_{i,j}&=\sum_d X_{i,d}G_{j,d},\\
\mathrm{grad\_h\_out}_{d}&=\sum_j p_jG_{j,d},\\
\mathrm{grad\_h\_post}_{j}&=\sum_d u_dG_{j,d}.
\end{aligned}
$$

独立 golden 在 CPU 上按这些公式使用 FP64 计算，再转回规定的输出 dtype。它用于正确性对照。

## 接口支持与优化生效范围

当前安装包提供 Ascend 950 后端。输入须连续，`h_res` 必须传入，`N=4`，`512≤T≤49152`，`1≤D≤8192`，数据为 FP16/BF16。`T` 在四维输入中取 `B×S`。

默认调用按输入选择实现，以下范围使用本库的最终优化内核：

| 条件 | 范围 |
|---|---|
| 设备 | Ascend 950 |
| 数据精度 | BF16 |
| 残差流数量 | `N=4` |
| 残差混合矩阵 | 传入 `h_res` |
| token 数 | `2048≤T≤49152` |
| 特征宽度 | `D∈{2560,5120,6144,7168}` |

优化内核用成本模型选择特征维度的分块，预测改善不足 2% 时保留原分块。其它已支持输入调用 CANN 主线 `torch_npu.npu_mhc_post_backward`。`implementation="optimized"` 可强制选择优化路径，超出上表范围会报错；`implementation="native"` 可直接调用原生算子。完整实测 shape 见性能表。

## 安装与运行

按项目首页激活私有 Python 环境，并手动 source 已选择的 CANN。所有命令从仓库根目录执行：

```bash
python -m pip install -e '.[dev]'
python -m xpu_kernels.cli build --backend npu --operator mhc_post_backward
python -m xpu_kernels.cli example --device npu --operator mhc_post_backward --shape 2x1024x4x6144
python -m xpu_kernels.cli test --device npu --operator mhc_post_backward
```

首次调用由 DeepJIT 编译内核，后续复用项目内缓存。wheel 携带 host/device 源码、成本模型和许可证；无需向公共 OPP 安装算子。

```python
import torch
import torch_npu
from xpu_kernels import mhc_post_backward

B, S, N, D = 2, 1024, 4, 6144
x = torch.randn(B, S, N, D, device="npu", dtype=torch.bfloat16)
grad_y = torch.randn_like(x)
h_res = torch.randn(B, S, N, N, device=x.device, dtype=torch.float32)
h_out = torch.randn(B, S, D, device=x.device, dtype=x.dtype)
h_post = torch.randn(B, S, N, device=x.device, dtype=torch.float32)

grad_x, grad_h_res, grad_h_out, grad_h_post = mhc_post_backward(
    grad_y, x, h_res, h_out, h_post,
)
```

可用 `out=(grad_x, grad_h_res, grad_h_out, grad_h_post)` 复用输出。输入与输出、不同输出之间保持独立存储。实现使用当前 NPU stream；切换 stream 时，调用方应建立输入数据的依赖。

## 测量

```bash
python -m xpu_kernels.cli bench --device npu --operator mhc_post_backward \
  --warmup 20 --batches 10 --iterations-per-batch 50

python -m xpu_kernels.cli profile --device npu --operator mhc_post_backward \
  --shape 2x1024x4x6144 --dtype bfloat16 --tool msprof \
  --output results/profiles/mhc-post-backward
```

默认 benchmark 在同一组输入上对比 `native,optimized`，两边每次调用均分配四个输出；可通过 `--implementations native` 或 `--implementations optimized` 单独测量。example/profile 使用单数参数 `--implementation`。

正确性检查取首尾及内部 token，与 CPU FP64 公式逐输出比较；单元测试另外对代表 shape 的全部输出进行比较，并覆盖非默认 stream 与输出复用。Event 测量记录设备流上多次调用的平均时间，包含 CPU 提交间隔。`msprof` 的 `Task Duration(us)` 用于分析设备 kernel，独立列在[性能记录](../performance/mhc-post-backward.md)中。

原生路径由当前 CANN 包提供 `aclnnMhcPostBackward`，测量时保持私有算子覆盖路径为空。优化路径通过本库 DeepJIT 加载。每份公开结果都保存设备、软件版本、shape、精度、采样和正确性记录。

## 许可证

NPU 内核与成本模型保留 CANN Open Software License Agreement Version 2.0，见[第三方说明](../../THIRD_PARTY_NOTICES.md)。公共 Python API、独立 golden、测试及 CLI 采用本项目许可证。
