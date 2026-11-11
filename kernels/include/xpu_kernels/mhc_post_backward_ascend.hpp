// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
// Project JIT entry for the retained V1 mixed Cube/Vector implementation.
#pragma once
#include <kernel_operator.h>
#include <xpu_kernels/mhc_post_backward/ai_infra_mhc_post_backward_kernel_merge_batch.h>
using namespace AiInfraMhcPostBackward;

extern "C" __global__ __aicore__ void XPU_MHC_KERNEL(
    GM_ADDR grad_y, GM_ADDR x, GM_ADDR h_res, GM_ADDR h_out, GM_ADDR h_post,
    GM_ADDR grad_x, GM_ADDR grad_h_res, GM_ADDR grad_h_out, GM_ADDR grad_h_post, uint32_t tokens) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    constexpr uint64_t n = 4;
    constexpr uint64_t group = 4;
    constexpr uint64_t batch_per_aiv = group / 2;
    constexpr uint64_t ub = XPU_MHC_UB;
    constexpr uint64_t width = XPU_MHC_D;
    constexpr uint64_t weights = 16 + 8;
    constexpr uint64_t previous_fixed = (2 * weights + 2 * n * 128) * 4;
    constexpr uint64_t previous_capacity = (ub - previous_fixed) / ((4 * (n + 1) + n) * 2) / 128 * 128;
    constexpr uint64_t aligned_d = (width + 127) / 128 * 128;
    constexpr uint64_t previous_ub_tile = aligned_d < previous_capacity ? aligned_d : previous_capacity;
    constexpr uint64_t l1_capacity = TOTAL_L1_SIZE / (4 * 16 * 2) / 64 * 128;
    constexpr uint64_t previous_tile = previous_ub_tile < l1_capacity ? previous_ub_tile : l1_capacity;
    constexpr uint64_t fixed = (2 * batch_per_aiv * weights + batch_per_aiv * n * 128 + 2 * 8) * 4;
    constexpr uint64_t per_k = (4 * batch_per_aiv * (n + 1) + 2 * n) * 2;
    constexpr uint64_t capacity = (ub - fixed) / per_k / 64 * 64;
    constexpr uint64_t v1_tile = previous_tile / 2 < capacity ? previous_tile / 2 : capacity;
#ifdef XPU_MHC_TILE
    constexpr uint64_t tile = XPU_MHC_TILE;
#else
    constexpr uint64_t tile = v1_tile;
#endif
    static_assert(tile >= 64, "insufficient UB for mixed mHC backward");
    constexpr uint64_t l0_capacity = (TOTAL_L0A_SIZE < TOTAL_L0B_SIZE ? TOTAL_L0A_SIZE : TOTAL_L0B_SIZE) / (2 * 16 * 2);
    constexpr uint64_t base_k = (l0_capacity < 1024 ? l0_capacity : 1024) / 16 * 16;
    AiInfraMhcPostBackwardOptimizedTilingDataArch35 t;
    t.totalItems = tokens;
    t.usedAic = XPU_MHC_CORES;
    t.n = n;
    t.D = width;
    t.tileD = tile;
    t.cubeBatchGroup = group;
    t.cubeL1K = tile;
    t.cubeBaseK = base_k;
    t.cubeKPartitionCount = 1;
    t.cubeKPartitionSize = (width + tile - 1) / tile * tile;
    AiInfraMhcPostBackward::AiInfraMhcPostBackwardKernelOptimized<XPU_MHC_TYPE, true, false, false> op;
    op.Init(grad_y, x, h_res, h_out, h_post, grad_x, grad_h_res, grad_h_out, grad_h_post,
            nullptr, &t, &pipe);
    op.Process();
}
