// NPU-only host adapter for the CANN-licensed V1 mHC backward kernel.
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <ATen/MemoryOverlap.h>
#include <torch/extension.h>
#include <deep_jit/backend/ascend/backend.hpp>
#include <deep_jit/runtime/runtime.hpp>
#include <xpu_kernels/mhc_post_backward/ai_infra_mhc_post_backward_tiling_cost_model.h>

namespace {
using Runtime = deep_jit::Runtime<deep_jit::Ascend>;
Runtime& mhc_runtime() {
    static auto runtime = [] {
        const char* value = std::getenv("XPU_KERNELS_ROOT");
        TORCH_CHECK(value, "XPU_KERNELS_ROOT must identify native resources");
        const auto root = std::filesystem::absolute(value).lexically_normal();
        return deep_jit::create_lazy_jit<deep_jit::Ascend>(deep_jit::Config(
            root, "XPU_KERNELS", "mhc-post-backward-v1", {root / "kernels" / "include"}, {"xpu_kernels/"}));
    }();
    return *runtime.get();
}
}

std::vector<torch::Tensor> mhc_post_backward(
    torch::Tensor grad_y, torch::Tensor x, torch::Tensor h_res, torch::Tensor h_out, torch::Tensor h_post,
    torch::Tensor grad_x, torch::Tensor grad_h_res, torch::Tensor grad_h_out, torch::Tensor grad_h_post, bool adaptive) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "mHC backward requires NPU tensors");
    TORCH_CHECK(x.dim() == 3 || x.dim() == 4, "input must be TND or BSND");
    TORCH_CHECK(x.size(-2) == 4, "this optimized kernel supports four residual streams");
    const int64_t d = x.size(-1), tokens = x.numel() / (4 * std::max<int64_t>(1, d));
    TORCH_CHECK(tokens >= 512 && tokens <= 49152 && d >= 1 && d <= 8192,
                "mHC JIT supports 512..49152 tokens and width 1..8192");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16, "input must be FP16 or BF16");
    TORCH_CHECK(grad_y.sizes() == x.sizes() && grad_x.sizes() == x.sizes(), "input gradient shape mismatch");
    TORCH_CHECK(grad_y.scalar_type() == x.scalar_type() && grad_x.scalar_type() == x.scalar_type()
                && h_out.scalar_type() == x.scalar_type() && grad_h_out.scalar_type() == x.scalar_type(), "data dtype mismatch");
    TORCH_CHECK(h_res.numel() == tokens * 16 && h_post.numel() == tokens * 4 && h_out.numel() == tokens * d,
                "coefficient or branch-output size mismatch");
    TORCH_CHECK(grad_h_res.sizes() == h_res.sizes() && grad_h_post.sizes() == h_post.sizes()
                && grad_h_out.sizes() == h_out.sizes(), "gradient output shape mismatch");
    std::vector<torch::Tensor> inputs{grad_y, x, h_res, h_out, h_post};
    std::vector<torch::Tensor> outputs{grad_x, grad_h_res, grad_h_out, grad_h_post};
    for (auto& tensor : inputs) {
        TORCH_CHECK(tensor.device() == x.device() && tensor.is_contiguous(), "inputs must be contiguous on the same NPU");
    }
    for (auto& tensor : {h_res, h_post, grad_h_res, grad_h_post}) {
        TORCH_CHECK(tensor.scalar_type() == at::kFloat, "mixing weights and their gradients must be FP32");
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        TORCH_CHECK(outputs[i].device() == x.device() && outputs[i].is_contiguous(), "outputs must be contiguous on the same NPU");
        for (const auto& input : inputs) at::assert_no_overlap(outputs[i], input);
        for (size_t j = 0; j < i; ++j) at::assert_no_overlap(outputs[i], outputs[j]);
    }
    c10::DeviceGuard guard(x.device());
    auto& runtime = mhc_runtime();
    TORCH_CHECK(runtime.device.get_prop().soc_name.find("Ascend950") == 0, "mHC mixed kernel requires Ascend 950");
    const int cores = std::min<int64_t>(runtime.device.get_num_cube_cores(), (tokens + 3) / 4);
    const auto ub = runtime.device.get_num_ubuf_bytes_per_vec_core();
    const bool bf16 = x.scalar_type() == at::kBFloat16;
    uint32_t selected_tile = 0;
    // This is the exact A1b outer dispatch domain of the measured source.
    const bool eligible = bf16 && tokens >= 2048 && tokens <= 49152
        && (d == 2560 || d == 5120 || d == 6144 || d == 7168);
    if (adaptive && eligible) {
        TORCH_CHECK(runtime.device.get_npu_arch() == 3510, "A1b tiling requires dav-3510");
        // These capacities are defined by CANN's dav-3510 kernel constants.
        constexpr uint64_t l1 = 512 * 1024;
        constexpr uint32_t base_k = 1024;
        const uint64_t previous_fixed = (2 * 24 + 2 * 4 * 128) * 4;
        const uint64_t previous_capacity = (ub - previous_fixed) / 48 / 128 * 128;
        const uint64_t previous_tile = std::min<uint64_t>((d + 127) / 128 * 128,
            std::min<uint64_t>(previous_capacity, l1 / 128 / 64 * 128));
        constexpr uint64_t fixed = (2 * 2 * 24 + 2 * 4 * 128 + 2 * 8) * 4;
        constexpr uint64_t per_k = 96;
        const uint64_t v1_tile = std::min<uint64_t>(previous_tile / 2, (ub - fixed) / per_k / 64 * 64);
        optiling::mhc_adaptive::AdaptiveShapeInput in{};
        in.totalItems=tokens; in.n=4; in.d=d; in.aicCores=runtime.device.get_num_cube_cores();
        in.group=4; in.baseK=base_k; in.ubSize=ub; in.l1Size=l1;
        in.fixedUb=fixed; in.bytesPerK=per_k; in.v1Tile=v1_tile;
        selected_tile=optiling::mhc_adaptive::SelectTileDAdaptive(in).tileD;
    }
    const std::string name = bf16 ? "xpu_mhc_post_backward_bf16" : "xpu_mhc_post_backward_fp16";
    const std::string source = "#define XPU_MHC_KERNEL " + name
        + "\n#define XPU_MHC_TYPE " + (bf16 ? "bfloat16_t" : "half")
        + "\n#define XPU_MHC_D " + std::to_string(d)
        + "\n#define XPU_MHC_CORES " + std::to_string(cores)
        + "\n#define XPU_MHC_UB " + std::to_string(ub)
        + (selected_tile ? "\n#define XPU_MHC_TILE " + std::to_string(selected_tile) : "")
        + "\n#include <xpu_kernels/mhc_post_backward_ascend.hpp>\n";
    const auto kernel = runtime.compile(name, source);
    runtime.launch(kernel, {.num_blocks = cores, .num_launch_timeout_secs = 30},
        grad_y.data_ptr(), x.data_ptr(), h_res.data_ptr(), h_out.data_ptr(), h_post.data_ptr(),
        grad_x.data_ptr(), grad_h_res.data_ptr(), grad_h_out.data_ptr(), grad_h_post.data_ptr(), static_cast<uint32_t>(tokens));
    return outputs;
}
