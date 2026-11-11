# mHC Post Backward

`mhc_post_backward` 计算 mHC 残差混合过程的四个梯度，并在 NVIDIA GPU 与 Ascend 950 NPU 上提供设备实现。GPU 后端使用 DeepSeek TileKernels 的 TileLang 实现；NPU 后端提供 Ascend C 优化实现，并在其优化范围之外调用 CANN 主线算子。

## Python 接口

```python
mhc_post_backward(
    grad_y, x, h_res, h_out, h_post, out=None, *, implementation="auto"
)
```

函数返回 `(grad_x, grad_h_res, grad_h_out, grad_h_post)`。输入可采用 `[T,4,D]` 或 `[B,S,4,D]`；四维输入的 token 数为 `B×S`。`grad_y` 与 `x` 形状相同，`h_res` 为 `[...,4,4]` FP32，`h_out` 为 `[...,D]` 且 dtype 与 `x` 相同，`h_post` 为 `[...,4]` FP32。所有输入连续并位于同一设备；`N=4`。

输出分别对应 `x`、`h_res`、`h_out`、`h_post` 的形状；`grad_x` 和 `grad_h_out` 使用输入数据 dtype，`grad_h_res` 与 `grad_h_post` 为 FP32。`out` 可传入四个连续、匹配形状和 dtype 的输出张量组成的 tuple。

## 计算公式

令 `G=grad_y`、`X=x`、`H=h_res`、`u=h_out`、`p=h_post`。对每个 token，四个输出为：

$$
\begin{aligned}
\mathrm{grad\_x}_{i,d}&=\sum_j H_{i,j}G_{j,d},\\
\mathrm{grad\_h\_res}_{i,j}&=\sum_d X_{i,d}G_{j,d},\\
\mathrm{grad\_h\_out}_{d}&=\sum_j p_jG_{j,d},\\
\mathrm{grad\_h\_post}_{j}&=\sum_d u_dG_{j,d}.
\end{aligned}
$$

`xpu_kernels.mhc_post_backward.mhc_post_backward_golden` 将这些公式在 CPU 上以 FP64 计算，再转换到各输出 dtype。设备测试和性能矩阵以此独立 golden 检查数值正确性。

## GPU：DeepSeek TileKernels

GPU 实现保留 DeepSeek 官方 TileKernels 的 TileLang kernel 代码。项目内的 [post_cuda.py](../../src/xpu_kernels/backends/_tilekernels/post_cuda.py) 是上游 `post_cuda.py` 的原样 MIT 许可副本；对应许可证见同目录 [LICENSE](../../src/xpu_kernels/backends/_tilekernels/LICENSE)，上游来源为 [DeepSeek TileKernels](https://github.com/deepseek-ai/TileKernels/blob/main/tile_kernels/mhc/post_cuda.py)。[适配层](../../src/xpu_kernels/backends/mhc_cuda.py)完成输入、token 维布局和返回值映射，TileLang 实现负责生成 GPU kernel。

上游函数接收 `grad_y, h_res, x, h_post, h_out`，返回 `(da, db, dc, dd)`。适配层的映射如下：

| 上游参数或结果 | 公共 API 对应项 |
|---|---|
| 输入 `h_res` | `h_res` |
| 输入 `x` | `x` |
| 输入 `h_post` | `h_post` |
| 输入 `h_out` | `h_out` |
| 返回 `db` | `grad_x` |
| 返回 `da` | `grad_h_res` |
| 返回 `dd` | `grad_h_out` |
| 返回 `dc` | `grad_h_post` |

GPU 支持 BF16、连续的三维或四维输入、`N=4`、512 至 49152 个 token、`D≤8192` 且 `D` 为 128 的倍数。GPU 实测环境为 NVIDIA L20X（Compute Capability 9.0）、PyTorch 2.13.0+cu126、TileLang 0.1.15。`implementation="auto"`、`"optimized"` 和 `"tilelang"` 均选择此 TileKernels 路径。

GPU 代表 shape 的四个完整输出、非默认 stream 和输出复用测试均通过。GPU 与 NPU 使用相同的 16 个 shape 完成采样正确性检查；每个 shape 对边界和内部 token 抽样，并与 CPU FP64 golden 对照。GPU Event 记录见 [mhc-post-backward-gpu-tilelang.jsonl](../../results/benchmarks/mhc-post-backward-gpu-tilelang.jsonl)。

## NPU：Ascend C 与 CANN 原生

NPU API 支持 Ascend 950、FP16/BF16、连续三维或四维输入、512 至 49152 个 token，以及 `1≤D≤8192`。`T` 对三维输入取首维，对四维输入取 `B×S`。最终优化 kernel 的适用条件如下：

| 条件 | 优化范围 |
|---|---|
| 数据类型 | BF16 |
| 残差流数 | `N=4`，传入 `h_res` |
| token 数 | `2048≤T≤49152` |
| 特征宽度 | `D∈{2560,5120,6144,7168}` |

`implementation="auto"` 在表内使用本库优化 kernel，在其它 API 支持范围调用 CANN 主线 `torch_npu.npu_mhc_post_backward`；`"optimized"` 选择优化 kernel，`"native"` 选择 CANN 原生算子。优化 kernel 用成本模型选择特征维度分块，并在预测收益不足 2% 时保留原分块。NPU 的 Cube 与 Vector 运算融合，双缓冲连接数据搬运和计算，长维度归约使用 FP32 补偿累加。

## 安装、构建与运行

GPU 安装包含 mHC 的 extras，并在仓库根目录执行 CLI：

```bash
python -m pip install -e '.[gpu,mhc-gpu,dev]'
python -m xpu_kernels.cli build --backend gpu --operator mhc_post_backward
python -m xpu_kernels.cli example --device cuda --operator mhc_post_backward --shape 2x1024x4x6144
python -m xpu_kernels.cli test --device cuda --operator mhc_post_backward
```

GPU mHC 依赖 TileLang 0.1.15 或兼容版本。NPU 用户先手动 source 已选 CANN：

```bash
python -m pip install -e '.[dev]'
python -m xpu_kernels.cli build --backend npu --operator mhc_post_backward
python -m xpu_kernels.cli example --device npu --operator mhc_post_backward --shape 2x1024x4x6144
python -m xpu_kernels.cli test --device npu --operator mhc_post_backward
```

GPU 和 NPU 均支持 `build`、`example`、`test`、`bench`、`profile` 子命令配合 `--operator mhc_post_backward`。完整依赖和环境检查见[安装指南](../getting-started.md)。

## Benchmark 与 Profiler

两种设备使用同一组 16 个 shape。GPU 测量 TileKernels；NPU 测量 CANN 原生与优化实现。Event 计时采用 eager 模式、预热 20 次、10 批、每批 50 次。两端各实现均在每次调用中分配四个输出。

```bash
python -m xpu_kernels.cli bench --device cuda --operator mhc_post_backward \
  --implementations tilelang --warmup 20 --batches 10 --iterations-per-batch 50

python -m xpu_kernels.cli bench --device npu --operator mhc_post_backward \
  --implementations native,optimized --warmup 20 --batches 10 --iterations-per-batch 50

python -m xpu_kernels.cli profile --device cuda --operator mhc_post_backward \
  --implementation tilelang --shape 2x1024x4x6144 --dtype bfloat16 --tool ncu \
  --output results/profiles/mhc-ncu

python -m xpu_kernels.cli profile --device npu --operator mhc_post_backward \
  --implementation optimized --shape 2x1024x4x6144 --dtype bfloat16 --tool msprof \
  --output results/profiles/mhc-msprof
```

CUDA Event 记录设备流区间，包含设备任务时间和可能的 CPU 提交间隙；NPU 的 kernel-only 数据采用 msprof `Task Duration(us)` 单独记录。GPU NCU 对 16 个 shape 各执行 20 次 warmup 和 5 次正式采样，共 400 次 launch；采集的 kernel 为 `mhc_post_bwd_cuda_kernel`，使用 kernel replay 和 `gpu__time_duration.sum`。NCU 中位数见 [NCU 记录](../../results/benchmarks/mhc-post-backward-gpu-ncu.jsonl)，逐次 launch 数据见 [TileKernels profiler CSV](../performance/data/mhc-gpu-tilelang.csv)。NPU 性能矩阵和采样细节见[mHC 性能记录](../performance/mhc-post-backward.md)。

## 许可证

GPU 上游 TileKernels 文件按其随附的 MIT License 发布；NPU 内核与成本模型保留 [CANN Open Software License Agreement Version 2.0](../../kernels/include/xpu_kernels/mhc_post_backward/LICENSE)。Python API、参数适配、独立 golden、测试和 CLI 属于本项目代码。
