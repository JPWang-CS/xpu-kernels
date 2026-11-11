#pragma once

#include <kernel_operator.h>

#if XPU_RMS_DTYPE == 0
using XpuRmsType = float;
#define XPU_RMS_ENTRY rms_norm_f32
#elif XPU_RMS_DTYPE == 1
using XpuRmsType = half;
#define XPU_RMS_ENTRY rms_norm_f16
#elif XPU_RMS_DTYPE == 2
using XpuRmsType = bfloat16_t;
#define XPU_RMS_ENTRY rms_norm_bf16
#else
#error "XPU_RMS_DTYPE must be 0 (float32), 1 (float16), or 2 (bfloat16)"
#endif

template <typename T>
__aicore__ inline void xpu_rms_norm_row(
    __gm__ T* output, __gm__ const T* input, __gm__ const T* weight,
    uint32_t width, float epsilon) {
    const uint32_t row = AscendC::GetBlockIdx();
    constexpr uint32_t elements_per_32b = 32U / sizeof(T);
    const uint32_t padded = (width + elements_per_32b - 1U) / elements_per_32b * elements_per_32b;
    const uint64_t offset = static_cast<uint64_t>(row) * width;

    AscendC::GlobalTensor<T> x_global;
    AscendC::GlobalTensor<T> w_global;
    AscendC::GlobalTensor<T> y_global;
    x_global.SetGlobalBuffer(const_cast<__gm__ T*>(input + offset), width);
    w_global.SetGlobalBuffer(const_cast<__gm__ T*>(weight), width);
    y_global.SetGlobalBuffer(output + offset, width);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> x_buf, w_buf, xf_buf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduce_buf, reduce_tmp_buf;
    pipe.InitBuffer(x_buf, padded * sizeof(T));
    pipe.InitBuffer(w_buf, padded * sizeof(T));
    pipe.InitBuffer(xf_buf, padded * sizeof(float));
    pipe.InitBuffer(reduce_buf, 32);
    pipe.InitBuffer(reduce_tmp_buf, padded * sizeof(float));

    auto x_local = x_buf.Get<T>();
    auto w_local = w_buf.Get<T>();
    auto xf_local = xf_buf.Get<float>();
    auto reduce_local = reduce_buf.Get<float>();
    auto reduce_tmp = reduce_tmp_buf.Get<float>();

    AscendC::DataCopyExtParams copy_params{1, width * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad_params{true, 0, static_cast<uint8_t>(padded - width), static_cast<T>(0)};
    AscendC::DataCopyPad(x_local, x_global, copy_params, pad_params);
    AscendC::DataCopyPad(w_local, w_global, copy_params, pad_params);
    AscendC::PipeBarrier<PIPE_ALL>();  // MTE2 -> V: input and weight are ready.
#if XPU_RMS_DTYPE == 0
    AscendC::DataCopy(xf_local, x_local, padded);
#else
    AscendC::Cast<float, T>(xf_local, x_local, AscendC::RoundMode::CAST_NONE, padded);
#endif
    AscendC::PipeBarrier<PIPE_ALL>();  // Cast -> V arithmetic.
    AscendC::Mul<float>(xf_local, xf_local, xf_local, padded);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::ReduceSum<float>(reduce_local, xf_local, reduce_tmp, padded);
    AscendC::PipeBarrier<PIPE_ALL>();  // V -> scalar GetValue.

    const float mean_square = reduce_local.GetValue(0) / static_cast<float>(width);
    const float scale = 1.0f / sqrt(mean_square + epsilon);
    AscendC::PipeBarrier<PIPE_ALL>();  // Scalar scale -> V arithmetic.
#if XPU_RMS_DTYPE == 0
    AscendC::DataCopy(xf_local, x_local, padded);
#else
    AscendC::Cast<float, T>(xf_local, x_local, AscendC::RoundMode::CAST_NONE, padded);
#endif
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Muls<float>(xf_local, xf_local, scale, padded);
    AscendC::PipeBarrier<PIPE_ALL>();  // Normalization -> dtype cast.
    // The explicit cast here is part of the Pangu golden's dtype contract.
#if XPU_RMS_DTYPE == 0
    AscendC::DataCopy(x_local, xf_local, padded);
#else
    AscendC::Cast<T, float>(x_local, xf_local, AscendC::RoundMode::CAST_RINT, padded);
#endif
    AscendC::PipeBarrier<PIPE_ALL>();  // Cast -> gamma multiplication.
    AscendC::Mul<T>(x_local, x_local, w_local, padded);
    AscendC::PipeBarrier<PIPE_ALL>();  // V -> MTE3 output copy.
    AscendC::DataCopyPad(y_global, x_local, copy_params);
}

extern "C" __global__ __vector__ void XPU_RMS_ENTRY(
    __gm__ XpuRmsType* output, __gm__ const XpuRmsType* input,
    __gm__ const XpuRmsType* weight, uint32_t width, float epsilon) {
    xpu_rms_norm_row<XpuRmsType>(output, input, weight, width, epsilon);
}

#undef XPU_RMS_ENTRY
