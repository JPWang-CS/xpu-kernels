#pragma once
/* K2: fixed-capacity send-slot builder, one token (8 pairs) per core.
 *
 * Bit-for-bit equivalent of the helper:
 *   local_slots[t,k] = (expert_start <= route_ids[t,k] < expert_end) & gathered_active[t]
 *   slot_values[p]   = local_slots[p] ? down_proj[inverse_index[p]] : 0
 *   send_slots       = slot_values.reshape(2,16,8,2560)
 * clamp(0, downRows-1) is a no-op for local pairs (inverse_index is a valid row
 * exactly when the pair is local), so the raw row index is used.
 */
#include "kernel_operator.h"
#include <xpu_kernels/fused_send_slots/tiling.hpp>

extern "C" __global__ __aicore__ void fused_send_slots(GM_ADDR downProj, GM_ADDR inverseIndex, GM_ADDR routeIds, GM_ADDR gatheredActive, GM_ADDR out, uint32_t pairs, uint32_t downRows, uint32_t hiddenSize, uint32_t expertStart, uint32_t expertEnd, uint32_t pairsPerBlock) {
    FusedSendSlotsTilingData tilingData{pairs, downRows, hiddenSize, expertStart, expertEnd, pairsPerBlock};
    const uint32_t blockIdx = AscendC::GetBlockIdx();
    if (blockIdx * tilingData.pairsPerBlock >= tilingData.pairs) {
        return;
    }
    const uint32_t h = tilingData.hiddenSize;
    const uint32_t pBegin = blockIdx * tilingData.pairsPerBlock;
    uint32_t pEnd = pBegin + tilingData.pairsPerBlock;
    if (pEnd > tilingData.pairs) {
        pEnd = tilingData.pairs;
    }
    const uint32_t nPairs = pEnd - pBegin;
    const uint32_t tBegin = pBegin / 8;              /* 8 pairs per token */
    const uint32_t tEndIncl = (pEnd - 1) / 8;
    const uint32_t nTokens = tEndIncl - tBegin + 1;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> metaQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> actQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> rowInQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> rowOutQue;
    pipe.InitBuffer(metaQue, 1, 64); /* two 8-int32 slices */
    pipe.InitBuffer(actQue, 1, 32);
    pipe.InitBuffer(rowInQue, 1, h * 2);
    pipe.InitBuffer(rowOutQue, 1, h * 2);

    AscendC::GlobalTensor<int32_t> invGm;
    invGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(inverseIndex));
    AscendC::GlobalTensor<int32_t> ridGm;
    ridGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(routeIds));
    AscendC::GlobalTensor<uint8_t> actGm;
    actGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(gatheredActive));
    AscendC::GlobalTensor<bfloat16_t> downGm;
    downGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(downProj));
    AscendC::GlobalTensor<bfloat16_t> outGm;
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out));

    /* inverse_index + route_ids slices for this block's pairs
     * (route_ids is [32,8] int32; flattened pair p addresses element p) */
    AscendC::LocalTensor<int32_t> metaUb = metaQue.AllocTensor<int32_t>();
    AscendC::DataCopy(metaUb, invGm[pBegin], nPairs);
    AscendC::DataCopy(metaUb[8], ridGm[pBegin], nPairs);
    metaQue.EnQue(metaUb);
    AscendC::LocalTensor<uint8_t> actUb = actQue.AllocTensor<uint8_t>();
    AscendC::DataCopyExtParams actExt(1, nTokens, 0, 0, 0);
    AscendC::DataCopyPadExtParams<uint8_t> actPad(false, 0, 0, 0);
    AscendC::DataCopyPad(actUb, actGm[tBegin], actExt, actPad);
    actQue.EnQue(actUb);

    AscendC::LocalTensor<int32_t> metaDeq = metaQue.DeQue<int32_t>();
    AscendC::LocalTensor<uint8_t> actDeq = actQue.DeQue<uint8_t>();
    int32_t inv[8];
    int32_t rid[8];
    for (uint32_t i = 0; i < nPairs; i++) {
        inv[i] = metaDeq.GetValue(i);
        rid[i] = metaDeq.GetValue(8 + i);
    }
    metaQue.FreeTensor(metaDeq);

    for (uint32_t i = 0; i < nPairs; i++) {
        const uint32_t p = pBegin + i;
        const uint32_t t = p / 8;
        const bool act = actDeq.GetValue(t - tBegin) != 0;
        const int32_t ridV = rid[i];
        const bool isLocal = act && (ridV >= static_cast<int32_t>(tilingData.expertStart)) &&
                             (ridV < static_cast<int32_t>(tilingData.expertEnd));
        AscendC::LocalTensor<bfloat16_t> rowUb = rowInQue.AllocTensor<bfloat16_t>();
        if (isLocal) {
            const uint32_t srcRow = static_cast<uint32_t>(inv[i]);
            AscendC::DataCopy(rowUb, downGm[srcRow * h], h);
        } else {
            AscendC::Duplicate(rowUb, static_cast<bfloat16_t>(0.0f), h);
        }
        rowInQue.EnQue(rowUb);
        AscendC::LocalTensor<bfloat16_t> rowDeq = rowInQue.DeQue<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> outUb = rowOutQue.AllocTensor<bfloat16_t>();
        AscendC::DataCopy(outUb, rowDeq, h);
        rowInQue.FreeTensor(rowDeq);
        rowOutQue.EnQue(outUb);
        AscendC::LocalTensor<bfloat16_t> outDeq = rowOutQue.DeQue<bfloat16_t>();
        AscendC::DataCopy(outGm[p * h], outDeq, h);
        rowOutQue.FreeTensor(outDeq);
    }
    actQue.FreeTensor(actDeq);
    AscendC::PipeBarrier<PIPE_ALL>();
}
