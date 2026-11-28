#pragma once
/* K3: owner gather + sequential combine, one token row per core.
 *
 * Bit-for-bit equivalent of the helper's post-all_to_all combine:
 *   source_ids[t,k] = active[t] ? topk_ids[t,k] : 0
 *   owner           = source_ids[t,k] // local_expert_num        (int floor div)
 *   selected[t,k,:] = received_slots[owner, t, k, :]
 *   output          = zeros(fp32)
 *   for k in 0..7:  output = output + selected[:,k].float() * weights[:,k]
 * Every multiply and add rounds to fp32 separately (no FMA, no tree reduction).
 *   final row       = bf16(rint(where(active, output, 0)))
 */
#include "kernel_operator.h"
#include <xpu_kernels/fused_owner_combine/tiling.hpp>

extern "C" __global__ __aicore__ void fused_owner_combine(GM_ADDR receivedSlots, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR activeMask, GM_ADDR out, uint32_t batchSize, uint32_t hiddenSize, uint32_t topKValue, uint32_t localExpertNum, uint32_t epSize) {
    FusedOwnerCombineTilingData tilingData{batchSize, hiddenSize, topKValue, localExpertNum, epSize};
    const uint32_t blockIdx = AscendC::GetBlockIdx();
    if (blockIdx >= tilingData.batchSize) {
        return;
    }
    const uint32_t t = blockIdx;
    const uint32_t h = tilingData.hiddenSize;
    const uint32_t topK = tilingData.topK;
    const uint32_t bs = tilingData.batchSize;
    const uint32_t ownerDiv = tilingData.localExpertNum; /* 128 */

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> slotInQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> idsQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> wQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> actQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    pipe.InitBuffer(slotInQue, 1, h * 2);
    pipe.InitBuffer(idsQue, 1, 32);
    pipe.InitBuffer(wQue, 1, 32);
    pipe.InitBuffer(actQue, 1, 32);
    pipe.InitBuffer(outQue, 1, h * 2);
    pipe.InitBuffer(calcBuf, 3 * h * 4);

    AscendC::GlobalTensor<bfloat16_t> recvGm;
    recvGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(receivedSlots));
    AscendC::GlobalTensor<int32_t> idsGm;
    idsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(topkIds));
    AscendC::GlobalTensor<float> wGm;
    wGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(topkWeights));
    AscendC::GlobalTensor<uint8_t> actGm;
    actGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(activeMask));
    AscendC::GlobalTensor<bfloat16_t> outGm;
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out));

    /* ---- per-token metadata: ids row, weights row, active byte ---- */
    AscendC::LocalTensor<int32_t> idsUb = idsQue.AllocTensor<int32_t>();
    AscendC::DataCopy(idsUb, idsGm[t * topK], topK);
    idsQue.EnQue(idsUb);
    AscendC::LocalTensor<float> wUb = wQue.AllocTensor<float>();
    AscendC::DataCopy(wUb, wGm[t * topK], topK);
    wQue.EnQue(wUb);
    AscendC::LocalTensor<uint8_t> actUb = actQue.AllocTensor<uint8_t>();
    AscendC::DataCopyExtParams actExt(1, 1, 0, 0, 0);
    AscendC::DataCopyPadExtParams<uint8_t> actPad(false, 0, 0, 0);
    AscendC::DataCopyPad(actUb, actGm[t], actExt, actPad);
    actQue.EnQue(actUb);

    AscendC::LocalTensor<int32_t> idsDeq = idsQue.DeQue<int32_t>();
    AscendC::LocalTensor<float> wDeq = wQue.DeQue<float>();
    AscendC::LocalTensor<uint8_t> actDeq = actQue.DeQue<uint8_t>();
    const bool act = actDeq.GetValue(0) != 0;
    int32_t ids[8];
    float w[8];
    for (uint32_t k = 0; k < topK; k++) {
        ids[k] = idsDeq.GetValue(k);
        w[k] = wDeq.GetValue(k);
    }
    idsQue.FreeTensor(idsDeq);
    wQue.FreeTensor(wDeq);
    actQue.FreeTensor(actDeq);

    AscendC::LocalTensor<float> calc = calcBuf.Get<float>();
    AscendC::LocalTensor<float> accUb = calc[0];
    AscendC::LocalTensor<float> vUb = calc[h];
    AscendC::LocalTensor<float> prodUb = calc[2 * h];

    /* output = zeros(fp32); for k: output = output + selected[:,k].float() * weights[:,k] */
    AscendC::Duplicate(accUb, 0.0f, h);
    for (uint32_t k = 0; k < topK; k++) {
        const int32_t id = act ? ids[k] : 0;
        const uint32_t owner = static_cast<uint32_t>(id / static_cast<int32_t>(ownerDiv));
        const uint32_t rowIdx = ((owner * bs + t) * topK + k) * h;
        AscendC::LocalTensor<bfloat16_t> slotUb = slotInQue.AllocTensor<bfloat16_t>();
        AscendC::DataCopy(slotUb, recvGm[rowIdx], h);
        slotInQue.EnQue(slotUb);
        AscendC::LocalTensor<bfloat16_t> slotDeq = slotInQue.DeQue<bfloat16_t>();
        AscendC::Cast(vUb, slotDeq, AscendC::RoundMode::CAST_NONE, h);
        slotInQue.FreeTensor(slotDeq);
        AscendC::Muls(prodUb, vUb, w[k], h);
        AscendC::Add(accUb, accUb, prodUb, h);
    }

    /* bf16(rint(where(active, output, 0))) */
    AscendC::LocalTensor<bfloat16_t> outUb = outQue.AllocTensor<bfloat16_t>();
    if (act) {
        AscendC::Cast(outUb, accUb, AscendC::RoundMode::CAST_RINT, h);
    } else {
        AscendC::Duplicate(outUb, static_cast<bfloat16_t>(0.0f), h);
    }
    outQue.EnQue(outUb);
    AscendC::LocalTensor<bfloat16_t> outDeq = outQue.DeQue<bfloat16_t>();
    AscendC::DataCopy(outGm[t * h], outDeq, h);
    outQue.FreeTensor(outDeq);
    AscendC::PipeBarrier<PIPE_ALL>();
}
