#ifndef XPU_RMS_NORM_CANN_PREFETCH_ROW_RETAIN_TILING_H_
#define XPU_RMS_NORM_CANN_PREFETCH_ROW_RETAIN_TILING_H_

#include <cstdint>

namespace xpu_rms_cann_prefetch {

// This layout is owned by the candidate launcher. It is intentionally kept
// at the original 64-byte ABI and constructed locally by the JIT entry.
struct alignas(8) RowRetainTilingData {
    uint64_t num_row;
    uint64_t num_col;
    uint64_t num_col_align;
    uint64_t block_factor;
    uint32_t row_factor;
    uint32_t ub_factor;
    uint32_t ub_loop;
    uint32_t col_buffer_length;
    float epsilon;
    float avg_factor;
    uint8_t is_gemma;
};

static_assert(sizeof(RowRetainTilingData) == 64, "RowRetainTilingData ABI must remain 64 bytes");

}  // namespace xpu_rms_cann_prefetch

#endif  // XPU_RMS_NORM_CANN_PREFETCH_ROW_RETAIN_TILING_H
