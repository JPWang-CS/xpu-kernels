# 正确性、Benchmark 与 Profiler

验证顺序是环境检查、后端构建、正确性测试、重复 benchmark，最后按需采集 profiler。性能记录保留设备、软件版本、输入 shape、dtype、实现、输出分配策略和采样条件，便于复现同一工作负载。

## 1. 环境与构建

从仓库根目录检查目标设备及工具链，并构建对应后端：

```bash
python -m xpu_kernels.cli check-env --backend gpu
python -m xpu_kernels.cli build --backend gpu

# NPU 运行前先手动 source 项目选定的 CANN
python -m xpu_kernels.cli check-env --backend npu
python -m xpu_kernels.cli build --backend npu
```

`build` 可使用 `--operator rms_norm` 或 `--operator mhc_post_backward` 限定算子。完整安装和依赖说明见[安装指南](getting-started.md)。

## 2. 正确性

CLI 的设备测试对照对应 golden/reference。按算子和设备运行：

```bash
python -m xpu_kernels.cli test --device cuda --operator rms_norm
python -m xpu_kernels.cli test --device npu --operator rms_norm
python -m xpu_kernels.cli test --device cuda --operator mhc_post_backward
python -m xpu_kernels.cli test --device npu --operator mhc_post_backward
```

RMSNorm 测试覆盖对齐、尾部宽度及多种 dtype；mHC 测试覆盖 CPU FP64 golden、autograd 参考以及 CUDA/NPU 设备路径。GPU mHC 完整设备测试通过，检查 TND/BSND 全部四个输出、非默认 CUDA stream、`out` 写回复用，以及不支持 dtype、宽度和输出别名的输入边界。具体数值合同见[RMSNorm](operators/rmsnorm.md)和[mHC](operators/mhc-post-backward.md)。

也可运行单个 correctness example 快速检查安装：

```bash
python -m xpu_kernels.cli example --device cuda --operator rms_norm --shape 128x4096
python -m xpu_kernels.cli example --device npu --operator mhc_post_backward --shape 2x1024x4x6144
```

## 3. 重复 Benchmark

RMSNorm 默认矩阵包含 48 个 shape 和 3 种 dtype。GPU 使用 `--native-reference` 时同时测量 CUDA C++、Triton、PyTorch 单算子拼接和 PyTorch 原生 RMSNorm，并为每次调用分配输出。单算子拼接的中间张量也在每次调用中创建。该策略让各实现的输出分配口径一致。下面采用预热 20 次、10 批、每批 50 次：

```bash
python -m xpu_kernels.cli bench --device cuda --native-reference \
  --warmup 20 --batches 10 --iterations-per-batch 50

python -m xpu_kernels.cli bench --device npu --native-reference \
  --warmup 20 --batches 10 --iterations-per-batch 50
```

可用 `--shapes 128x4096,129x4097` 指定 RMSNorm shape，用 `--dtypes bfloat16` 限定精度。未指定 `--native-reference` 时，库实现复用预分配输出；指定后，CUDA C++、Triton、单算子拼接和原生实现均按每次调用分配输出。结果默认写入 `results/benchmarks/`，也可用 `--output` 指定路径。benchmark 使用当前框架 stream 上的设备 Event，记录每批样本及中位数。Event 区间包含 CPU 提交间隙。

GPU BF16 实测包含 48 个 shape 与 CUDA C++、Triton、PyTorch 单算子拼接、PyTorch 原生 RMSNorm 四种实现，共 192 条记录。四种实现均通过对应正确性检查，并在每次调用中分配输出；单算子拼接同时创建中间张量。原始记录见 [rmsnorm-gpu-mainline.jsonl](../results/benchmarks/rmsnorm-gpu-mainline.jsonl)。

mHC GPU 与 NPU benchmark 覆盖相同的 16 个 shape、dtype 和输入分布；两台设备分别生成张量，使用相同测试种子，不共享同一份设备输入。GPU 测量 TileKernels；NPU 对照 CANN 原生与优化实现。GPU `auto`、`optimized`、`tilelang` 均选择 TileKernels 路径，输出均在每次调用中分配；NPU 两种实现同样每次调用分配四个输出。两端使用预热 20 次、10 批、每批 50 次：

```bash
python -m xpu_kernels.cli bench --device cuda --operator mhc_post_backward \
  --implementations tilelang --warmup 20 --batches 10 --iterations-per-batch 50

python -m xpu_kernels.cli bench --device npu --operator mhc_post_backward \
  --implementations native,optimized \
  --warmup 20 --batches 10 --iterations-per-batch 50
```

两端均通过 FP64 抽样正确性检查。记录包含硬件、运行时、实现、分配策略、Event 采样及每个输出的误差。NPU kernel-only 对比使用 msprof 的 `Task Duration(us)`，与 Event 区间分表记录；跨设备性能表见[mHC 性能记录](performance/mhc-post-backward.md)。

## 4. Profiler

Profiler 在正确性和匹配条件的重复 benchmark 之后独立运行。每次选择一个工具，输出保存在项目内 `results/profiles/`：

```bash
python -m xpu_kernels.cli profile --device cuda --operator rms_norm \
  --implementation cuda --shape 128x4096 --dtype bfloat16 \
  --tool ncu --output results/profiles/gpu-ncu

python -m xpu_kernels.cli profile --device cuda --operator rms_norm \
  --implementation triton --shape 128x4096 --dtype bfloat16 \
  --tool nsys --output results/profiles/gpu-nsys

python -m xpu_kernels.cli profile --device cuda --operator mhc_post_backward \
  --implementation tilelang --shape 2x1024x4x6144 --dtype bfloat16 \
  --tool ncu --output results/profiles/mhc-ncu

python -m xpu_kernels.cli profile --device npu --operator mhc_post_backward \
  --implementation optimized --shape 2x1024x4x6144 --dtype bfloat16 \
  --tool msprof --output results/profiles/mhc-msprof
```

NCU 对所选 CUDA C++、Triton 或 TileLang kernel 采集匹配 launch，使用 kernel replay 和 `SpeedOfLight`、`LaunchStats` section，不控制时钟或缓存。mHC GPU 的 NCU 矩阵包含 16 个 shape、每个 shape 20 次 warmup 和 5 次测量，共 400 次 launch；采集 kernel 为 `mhc_post_bwd_cuda_kernel`，16 个 shape 的 FP64 抽样正确性与 Event 测量均通过。中位数及逐次 launch 数据见 [NCU JSONL](../results/benchmarks/mhc-post-backward-gpu-ncu.jsonl) 和 [profiler CSV](performance/data/mhc-gpu-tilelang.csv)。msprof 根据已安装版本支持的参数启动。Profiler 时间用于分析 kernel，不代替重复 benchmark 的性能结论。

## 5. 记录与图表

benchmark JSONL 保存原始采样和环境信息，目录及字段说明见[记录格式](../results/benchmarks/README.md)。跨版本或不同测量方式的数据分开比较；历史最佳只与匹配的设备、实现、dtype、shape 和测量设置对照。PyTorch golden 用于正确性，不作为性能基线。

首页图表从已验证的 JSONL 记录生成。更新图表前先核对数据来源与比较条件；渲染和检查命令见[项目 AI Skill](../skills/xpu-kernels/SKILL.md#homepage-performance-highlights)。完整延迟表见[性能索引](performance/README.md)。
