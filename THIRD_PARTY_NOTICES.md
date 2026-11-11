# Third-party notices

## DeepJIT

`third-party/deep-jit` is an upstream Git submodule. Its license and notices remain in that directory. The project uses DeepJIT for Ascend runtime compilation, loading and launch.

## Ascend mHC post backward

The NPU implementation under `kernels/include/xpu_kernels/mhc_post_backward/` derives from the `AiInfraMhcPostBackward` implementation in omni-ops, at `training/ascendc/src/ops-transformer/mhc/ai_infra_mhc_post_backward/op_kernel/arch35/`.

Copyright (c) 2026 Huawei Technologies Co., Ltd. These files retain the **CANN Open Software License Agreement Version 2.0**, included at [kernels/include/xpu_kernels/mhc_post_backward/LICENSE](kernels/include/xpu_kernels/mhc_post_backward/LICENSE). They are NPU-specific components, distributed under their own license rather than the repository's Apache-2.0 license.

The original Cube computation and fused Vector implementation are retained. Integration changes use canonical includes for DeepJIT, retain the optimized tiling record, and add `mhc_post_backward_ascend.hpp` as a JIT entry. The latter is distributed under the same CANN license. The host adapter calculates shape/device-dependent launch metadata and uses the current NPU stream. No CANN OPP package installation is required for this JIT path.

The public Python API, independent FP64 equations, CLI and tests are project code. Supported shapes and validation results are documented in [the operator guide](docs/operators/mhc-post-backward.md).

## DeepSeek TileKernels GPU mHC post backward

`src/xpu_kernels/backends/_tilekernels/post_cuda.py` is the upstream GPU implementation from [deepseek-ai/TileKernels](https://github.com/deepseek-ai/TileKernels/blob/main/tile_kernels/mhc/post_cuda.py). It is retained under the MIT License, copyright 2026 DeepSeek, included alongside the file. The xpu-kernels adapter flattens token axes, reorders inputs/outputs, and selects the upstream thread/block parameters. The kernel computation is unchanged and uses TileLang to generate CUDA code.
