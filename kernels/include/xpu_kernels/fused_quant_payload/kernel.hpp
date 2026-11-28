#pragma once
/* K1: mask-aware MC2-compatible quant + int8 payload, one token row per core.
 *
 * Reproduces the accepted fixed-capacity helper bit-for-bit:
 *   quant_input = where(active, hidden, 0)          (BF16)
 *   xf = fp32(quant_input); xm = amax(|xf|, -1)
 *   zero_row: q=0, scale=+0.0 (bits 0x00000000)
 *   d  = correctly-rounded fp32 127/xm_safe (scalar Newton/Dekker refinement,
 *        matching the helper's _cr_div tensor sequence item-for-item)
 *   q  = rint(xf*d) clamped to [-128,127] as int8
 *   s  = correctly-rounded fp32 (0|1)/d
 *   payload row = q(2560B) | scale bits(4B) | source_ids bytes(32B) | active(1B) | 0(27B)
 * Vector ops are used exactly where the helper used tensor ops; branch decisions
 * are device-side scalars (no host sync).
 */
#include "kernel_operator.h"
#include <xpu_kernels/fused_quant_payload/tiling.hpp>

namespace {
/* Python double constant rounding semantics preserved: the helper computes
 * 3.4e38/4097.0 - 1.0 in double, then torch casts to fp32 for the compare. */
constexpr double kSplitSafeMaxD = 3.4e38 / 4097.0 - 1.0;
constexpr float kSplitSafeMax = static_cast<float>(kSplitSafeMaxD);
constexpr float kScaleDown = 1.0f / 18446744073709551616.0f; /* 2^-64 */
constexpr float kScaleUp = 18446744073709551616.0f;          /* 2^64  */

/* Scalar correctly-rounded fp32 division: identical arithmetic sequence to the
 * helper's _cr_div (Dekker two-product + two Newton steps, every intermediate
 * rounded to fp32; no FMA contraction). */
__aicore__ inline float ScalarAbsF(float v)
{
    return v < 0.0f ? -v : v;
}

__aicore__ inline float ScalarTwoProdErr(float a, float b, float& p)
{
    bool aBig = ScalarAbsF(a) > kSplitSafeMax;
    bool bBig = ScalarAbsF(b) > kSplitSafeMax;
    bool both = aBig && bBig;
    float aS = (aBig && !bBig) ? a * kScaleDown : a;
    float bS = (bBig && !aBig) ? b * kScaleDown : b;
    float aTw = aS * 4097.0f;
    float aHi = aTw - (aTw - aS);
    float bTw = bS * 4097.0f;
    float bHi = bTw - (bTw - bS);
    float pS = aS * bS;
    float t1 = (aHi * bHi - pS);
    float t2 = (aHi * (bS - bHi) + (aS - aHi) * bHi);
    float t3 = (aS - aHi) * (bS - bHi);
    float errS = (t1 + t2) + t3;
    float upA = (aBig && !bBig) ? kScaleUp : 1.0f;
    float upB = (bBig && !aBig) ? kScaleUp : 1.0f;
    if (both) {
        p = a * b;
        return 0.0f;
    }
    p = pS * upA * upB;
    return errS * upA * upB;
}

__aicore__ inline float ScalarCrDiv(float n, float den)
{
    float x0 = n / den;
    float p, e;
    e = ScalarTwoProdErr(den, x0, p);
    float r = (n - p) - e;
    float x1 = x0 + r / den;
    float p1, e1;
    e1 = ScalarTwoProdErr(den, x1, p1);
    float r1 = (n - p1) - e1;
    return x1 + r1 / den;
}
}  // namespace

extern "C" __global__ __aicore__ void fused_quant_payload(GM_ADDR x, GM_ADDR topkIds, GM_ADDR xActiveMask, GM_ADDR payload, GM_ADDR scaleOut, uint32_t batchSize, uint32_t hiddenSize, uint32_t topK, uint32_t rowBytes) {
    FusedQuantPayloadTilingData tilingData{batchSize, hiddenSize, topK, rowBytes};
    const uint32_t blockIdx = AscendC::GetBlockIdx();
    if (blockIdx >= tilingData.batchSize) {
        return;
    }
    const uint32_t row = blockIdx;
    const uint32_t h = tilingData.hiddenSize;      /* 2560 */

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> xQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> idsQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> actQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> qOutQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> tailOutQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> scaleOutQue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    pipe.InitBuffer(xQue, 1, h * 2);
    pipe.InitBuffer(idsQue, 1, 32);
    pipe.InitBuffer(actQue, 1, 32);
    pipe.InitBuffer(qOutQue, 1, h);
    pipe.InitBuffer(tailOutQue, 1, 64);
    pipe.InitBuffer(scaleOutQue, 1, 32);
    pipe.InitBuffer(calcBuf, 18 * h + 2112);

    AscendC::GlobalTensor<bfloat16_t> xGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
    AscendC::GlobalTensor<int32_t> idsGm;
    idsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(topkIds));
    AscendC::GlobalTensor<uint8_t> actGm;
    actGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(xActiveMask));
    AscendC::GlobalTensor<int8_t> payloadGm;
    payloadGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(payload));
    AscendC::GlobalTensor<float> scaleGm;
    scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scaleOut));

    /* ---- CopyIn (x row bf16, ids row int32, active byte) ---- */
    AscendC::LocalTensor<bfloat16_t> xUb = xQue.AllocTensor<bfloat16_t>();
    AscendC::DataCopy(xUb, xGm[row * h], h);
    xQue.EnQue(xUb);
    AscendC::LocalTensor<int32_t> idsUb = idsQue.AllocTensor<int32_t>();
    AscendC::DataCopy(idsUb, idsGm[row * 8], 8);
    idsQue.EnQue(idsUb);
    AscendC::LocalTensor<uint8_t> actUb = actQue.AllocTensor<uint8_t>();
    AscendC::DataCopyExtParams actExt(1, 1, 0, 0, 0);
    AscendC::DataCopyPadExtParams<uint8_t> actPad(false, 0, 0, 0);
    AscendC::DataCopyPad(actUb, actGm[row], actExt, actPad);
    actQue.EnQue(actUb);

    AscendC::LocalTensor<bfloat16_t> xDeq = xQue.DeQue<bfloat16_t>();
    AscendC::LocalTensor<int32_t> idsDeq = idsQue.DeQue<int32_t>();
    AscendC::LocalTensor<uint8_t> actDeq = actQue.DeQue<uint8_t>();

    AscendC::LocalTensor<float> calc = calcBuf.Get<float>();
    AscendC::LocalTensor<float> xfUb = calc[0];
    AscendC::LocalTensor<float> absUb = calc[h];
    AscendC::LocalTensor<float> workUb = calc[2 * h];
    AscendC::LocalTensor<float> redTmpUb = calc[3 * h];
    AscendC::LocalTensor<float> echoUb = calc[3 * h + 512];  /* 8 lanes */

    /* active + ids as device scalars */
    const uint8_t actByte = actDeq.GetValue(0);
    const bool act = actByte != 0;
    int32_t ids[8];
    for (uint32_t i = 0; i < 8; i++) {
        ids[i] = idsDeq.GetValue(i);
    }

    /* ---- absmax over the row (quant_input = where(active, hidden, 0)) ---- */
    if (act) {
        AscendC::Cast(xfUb, xDeq, AscendC::RoundMode::CAST_NONE, h);
    } else {
        AscendC::Duplicate(xfUb, 0.0f, h);
    }
    AscendC::Abs(absUb, xfUb, h);
    AscendC::ReduceMax(echoUb, absUb, redTmpUb, h);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    float xm = echoUb.GetValue(0);
    bool zeroRow = (xm == 0.0f);
    float xmSafe = zeroRow ? 1.0f : xm;
    float sNum = zeroRow ? 0.0f : 1.0f;

    /* ---- d = correctly-rounded 127/xmSafe (scalar) ---- */
    float d = ScalarCrDiv(127.0f, xmSafe);

    /* ---- q = clamp(rint(xf*d), -128, 127) as int8 ---- */
    AscendC::Muls(workUb, xfUb, d, h);
    /* no same-dtype in-place fp32 Cast here (illegal construct; removing it is
     * conformance-correct even though it proved non-destructive when removed). */
    AscendC::LocalTensor<int8_t> qUb = qOutQue.AllocTensor<int8_t>();
    /* Round to nearest-even in FP32 before the INT8 conversion. */
    AscendC::Adds(workUb, workUb, 12582912.0f, h);
    AscendC::Adds(workUb, workUb, -12582912.0f, h);
    AscendC::Maxs(workUb, workUb, -128.0f, h);
    AscendC::Mins(workUb, workUb, 127.0f, h);
    AscendC::LocalTensor<half> tmp16Ub = calcBuf.Get<half>()[(3 * h + 520) * 2 + 2 * h];
    AscendC::Cast(tmp16Ub, workUb, AscendC::RoundMode::CAST_RINT, h);
    AscendC::Cast(qUb, tmp16Ub, AscendC::RoundMode::CAST_RINT, h);
    xQue.FreeTensor(xDeq);
    qOutQue.EnQue(qUb);
    AscendC::LocalTensor<int8_t> qDeq = qOutQue.DeQue<int8_t>();
    AscendC::DataCopy(payloadGm[row * tilingData.rowBytes], qDeq, h);
    qOutQue.FreeTensor(qDeq);

    /* ---- s = correctly-rounded sNum/d (scalar); write scale bits ---- */
    float s = ScalarCrDiv(sNum, d);
    AscendC::Duplicate(echoUb, s, 8);  /* hold fp32 bits in UB */
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
    const int32_t sBits = echoUb.ReinterpretCast<int32_t>().GetValue(0);

    /* ---- tail staging: scale(4B) | ids(32B) | active(1B) | pad(27B) ---- */
    AscendC::LocalTensor<int8_t> tailUb = tailOutQue.AllocTensor<int8_t>();
    tailUb.SetValue(0, static_cast<int8_t>(sBits & 0xff));
    tailUb.SetValue(1, static_cast<int8_t>((sBits >> 8) & 0xff));
    tailUb.SetValue(2, static_cast<int8_t>((sBits >> 16) & 0xff));
    tailUb.SetValue(3, static_cast<int8_t>((sBits >> 24) & 0xff));
    for (uint32_t i = 0; i < 8; i++) {
        const int32_t v = act ? ids[i] : 0;
        tailUb.SetValue(4 + i * 4 + 0, static_cast<int8_t>(v & 0xff));
        tailUb.SetValue(4 + i * 4 + 1, static_cast<int8_t>((v >> 8) & 0xff));
        tailUb.SetValue(4 + i * 4 + 2, static_cast<int8_t>((v >> 16) & 0xff));
        tailUb.SetValue(4 + i * 4 + 3, static_cast<int8_t>((v >> 24) & 0xff));
    }
    tailUb.SetValue(36, act ? static_cast<int8_t>(1) : static_cast<int8_t>(0));
    for (uint32_t i = 37; i < 64; i++) {
        tailUb.SetValue(i, static_cast<int8_t>(0));
    }
    idsQue.FreeTensor(idsDeq);
    actQue.FreeTensor(actDeq);
    tailOutQue.EnQue(tailUb);
    AscendC::LocalTensor<int8_t> tailDeq = tailOutQue.DeQue<int8_t>();
    AscendC::DataCopy(payloadGm[row * tilingData.rowBytes + h], tailDeq, 64);
    tailOutQue.FreeTensor(tailDeq);

    /* ---- scale_out fp32 (4 bytes) ---- */
    AscendC::LocalTensor<int8_t> sUb = scaleOutQue.AllocTensor<int8_t>();
    sUb.SetValue(0, static_cast<int8_t>(sBits & 0xff));
    sUb.SetValue(1, static_cast<int8_t>((sBits >> 8) & 0xff));
    sUb.SetValue(2, static_cast<int8_t>((sBits >> 16) & 0xff));
    sUb.SetValue(3, static_cast<int8_t>((sBits >> 24) & 0xff));
    scaleOutQue.EnQue(sUb);
    AscendC::LocalTensor<int8_t> sDeq = scaleOutQue.DeQue<int8_t>();
    AscendC::DataCopyExtParams sExt(1, 4, 0, 0, 0);
    AscendC::DataCopyPad(scaleGm[row], sDeq.ReinterpretCast<float>(), sExt);
    scaleOutQue.FreeTensor(sDeq);

    AscendC::PipeBarrier<PIPE_ALL>();
}
