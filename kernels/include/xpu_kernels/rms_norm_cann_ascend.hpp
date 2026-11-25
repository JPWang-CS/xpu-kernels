// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
//
// DeepJIT entry for the two fixed Pangu RMSNorm row-retain candidates.
// Each translation unit selects one implementation with XPU_RMS_ROW_SLOTS;
// both variants intentionally export the same symbol and argument contract.
#pragma once

#include <kernel_operator.h>

#ifndef XPU_RMS_ROW_SLOTS
#error "XPU_RMS_ROW_SLOTS must be 2 (Y-full) or 3 (prefetch)"
#endif

#if XPU_RMS_ROW_SLOTS == 2
#include <xpu_kernels/rms_norm_cann/yfull/rms_norm_row_retain.h>
namespace xpu_rms_cann_selected {
using TilingData = xpu_rms_cann_yfull::RowRetainTilingData;
using Kernel = xpu_rms_cann_yfull::KernelRmsNormCannYfull<bfloat16_t, bfloat16_t, false>;
}  // namespace xpu_rms_cann_selected
#elif XPU_RMS_ROW_SLOTS == 3
#include <xpu_kernels/rms_norm_cann/prefetch/rms_norm_row_retain.h>
namespace xpu_rms_cann_selected {
using TilingData = xpu_rms_cann_prefetch::RowRetainTilingData;
using Kernel = xpu_rms_cann_prefetch::KernelRmsNormCannPrefetch<bfloat16_t, bfloat16_t, false>;
}  // namespace xpu_rms_cann_selected
#else
#error "XPU_RMS_ROW_SLOTS must be 2 (Y-full) or 3 (prefetch)"
#endif

// Fixed contract: D=20480, BF16 X/gamma/Y, FP32 rstd, epsilon=1e-5.
// The host launches 64 AIV blocks. For the supported M values this gives
// block_factor=ceil(M/64), so every block receives at most 64 rows.
extern "C" __global__ __vector__ void xpu_rms_norm_cann(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, uint32_t rows) {
    xpu_rms_cann_selected::TilingData tiling{};
    tiling.num_row = rows;
    tiling.num_col = 20480;
    tiling.num_col_align = 20480;
    tiling.block_factor = (static_cast<uint64_t>(rows) + 63U) / 64U;
    tiling.row_factor = 64;
    tiling.ub_factor = 8192;
    tiling.ub_loop = 2;
    tiling.col_buffer_length = 20480;
    tiling.epsilon = 1.0e-5F;
    tiling.avg_factor = 1.0F / 20480.0F;
    tiling.is_gemma = 0;

    xpu_rms_cann_selected::Kernel op;
    op.Init(x, gamma, y, rstd, &tiling);
    op.Process();
}
