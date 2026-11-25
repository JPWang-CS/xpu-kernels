<div align="center">

<img src="docs/assets/xpu-kernels-masthead.png" alt="xpu-kernels — GPU / NPU Kernel Library" width="640" />

[安装与使用](#quick-start) &nbsp;·&nbsp; [算子](#operators) &nbsp;·&nbsp; [性能](#benchmarks) &nbsp;·&nbsp; [复现](docs/benchmarking.md)

</div>

NVIDIA GPU 与昇腾 NPU 算子库，面向训练和推理中的高性能张量算子。提供统一 Python API、设备实现、独立数值参考与可复现的正确性和性能测量。

<a id="quick-start"></a>

## 安装与最小示例

```bash
git clone --recurse-submodules https://github.com/JPWang-CS/xpu-kernels.git
cd xpu-kernels
python -m pip install -e '.[gpu,mhc-gpu,dev]'  # NVIDIA GPU，含 mHC
```

昇腾 NPU 环境先手动 source 所选 CANN，再执行 `python -m pip install -e '.[dev]'`。依赖、工具链与构建说明见[安装指南](docs/getting-started.md)。

```python
import torch
from xpu_kernels import rms_norm

x = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(4096, device=x.device, dtype=x.dtype)
y = rms_norm(x, weight)
```

<a id="operators"></a>

## 算子

**RMSNorm** 沿最后一维归一化并应用权重，计算顺序与 CANN 主线一致。NPU 按输入选择主线原生或优化实现，GPU 提供 CUDA 与 Triton 实现。见 [RMSNorm 完整文档](docs/operators/rms_norm/README.md)。

**mHC Post Backward** 计算残差混合过程的四个梯度，提供 GPU 与昇腾 NPU 实现。

贡献与新增算子遵循统一的[目录和接入规范](docs/repository-layout.md)，覆盖 API、设备源码、golden、测试及文档。

<a id="benchmarks"></a>

## 性能

优化实现后续可能合入 CANN 上游。NPU 性能对比以测试时 CANN 包内的主线算子为基线。

- **性能**：在相同 shape 和 dtype 下，预热后重复测量并报告耗时中位数；内核耗时与 Event 计时分别记录。
- **精度**：常规测试核对独立参考结果；优化验收采用精度标准 2.1，逐输出比较 FP64 参考值与同输入的真实 GPU 结果，通过后再测性能。

<!-- BEGIN homepage-performance -->
### RMSNorm

![RMSNorm — NPU 原生与优化](docs/assets/benchmark-rmsnorm.png)

[RMSNorm 完整文档](docs/operators/rms_norm/README.md)

### mHC Post Backward

![mHC Post Backward — NPU 与 GPU](docs/assets/benchmark-mhc.png)

[mHC Post Backward 完整文档](docs/operators/mhc_post_backward/README.md)
<!-- END homepage-performance -->

## License

公共代码采用 [Apache-2.0](LICENSE)；DeepSeek TileKernels GPU 源码保留随附的 [MIT License](src/xpu_kernels/backends/_tilekernels/LICENSE)；CANN 衍生的 NPU 内核保留 [CANN Open Software License 2.0](kernels/include/xpu_kernels/rms_norm/LICENSE)。NPU JIT 运行时使用 [DeepJIT](https://github.com/deepseek-ai/DeepJIT)。[第三方说明](THIRD_PARTY_NOTICES.md)
