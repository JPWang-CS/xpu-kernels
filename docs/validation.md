# 安装与设备验证

0.2.0 的源码安装、wheel 构建和设备运行分别检查。GPU 与 NPU 使用独立私有环境，依赖现有驱动及框架；NPU 内核通过项目缓存加载，未安装到公共 OPP。

| 验证项 | GPU L20X | NPU 950DT |
|---|---|---|
| 测试集 | 44 项通过，5 项跳过 | 40 项通过，9 项跳过 |
| 跳过内容 | NPU 专用测试等当前后端不适用项 | GPU、Triton 专用测试等当前后端不适用项 |
| wheel 构建与独立安装 | 通过 | 通过 |
| 从 wheel 包内源码编译并运行 | RMSNorm，FP32 / FP16 / BF16 | RMSNorm，FP32 / FP16 / BF16；mHC，FP16 / BF16 |
| mHC 优化路径 | 未提供 | BF16、2×1024×4×6144，四个输出检查通过 |
| 非默认 stream 与输出复用 | RMSNorm 通过 | RMSNorm、mHC 通过 |

测试集包含独立 golden、输入合同、后端选择、打包资源、benchmark 记录与设备数值测试。mHC golden 另外与 PyTorch 自动微分比较；设备测试覆盖三维、四维输入及尾块。wheel 检查确认 NPU 内核、成本模型、DeepJIT 头文件和各自许可证随包携带。

性能检查独立于上述安装验证：

- RMSNorm：保留 48 个 shape、三种精度的 GPU/NPU 实现记录；新增 NPU 原生对照 288 项，各实现按对应数值公式检查。
- mHC：16 个优化范围内的 BF16 shape，原生和优化共 32 项重复 Event 记录；用 CPU FP64 公式检查首尾及内部 token。
- mHC msprof：原生与优化共 800 次目标 kernel launch；每个 shape、每个实现剔除 20 次预热，保留 5 次正式采样。公开 CSV 保存 Task ID、stream、shape 和时间，表格从样本生成。

运行命令、完整日志和退出码保存在各设备项目的 `runs/mhc-npu-final/`、`runs/mhc-gpu-final/`、`runs/mhc-mainline-accept/`、`runs/mhc-mainline-profiler/` 及 `runs/rms-native-comparison/`。公开性能数据见[索引](../results/benchmarks/README.md)。

复核表格：

```bash
PYTHONPATH=src python tools/render_native_comparisons.py --check
PYTHONPATH=src python tools/render_homepage_performance.py --check
PYTHONPATH=src python tools/render_rmsnorm_table.py --check
```
