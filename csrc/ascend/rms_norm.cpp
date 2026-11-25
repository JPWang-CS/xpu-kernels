#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <ATen/MemoryOverlap.h>
#include <torch/extension.h>
#include <deep_jit/backend/ascend/backend.hpp>
#include <deep_jit/runtime/runtime.hpp>

namespace {
using Runtime = deep_jit::Runtime<deep_jit::Ascend>;

Runtime& rms_norm_runtime() {
    static auto runtime = [] {
        const char* value = std::getenv("XPU_KERNELS_ROOT");
        TORCH_CHECK(value, "XPU_KERNELS_ROOT must identify the native resources");
        const auto root = std::filesystem::absolute(value).lexically_normal();
        return deep_jit::create_lazy_jit<deep_jit::Ascend>(deep_jit::Config(
            root, "XPU_KERNELS", "rms-norm", {root / "kernels" / "include"}, {"xpu_kernels/"}));
    }();
    return *runtime.get();
}
}  // namespace

std::vector<torch::Tensor> rms_norm(
    torch::Tensor x, torch::Tensor weight, torch::Tensor y, torch::Tensor rstd,
    const double eps) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "RMSNorm requires NPU tensors");
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) == 20480, "optimized RMSNorm requires D=20480");
    const int64_t rows = x.numel() / 20480;
    TORCH_CHECK(rows == 2048 || rows == 4096, "optimized RMSNorm requires exactly 2048 or 4096 rows");
    TORCH_CHECK(std::isfinite(eps) && eps == 1e-5, "optimized RMSNorm requires eps=1e-5");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16
                && y.scalar_type() == at::kBFloat16 && rstd.scalar_type() == at::kFloat,
                "optimized RMSNorm requires BF16 x/weight/y and FP32 rstd");
    TORCH_CHECK(weight.dim() == 1 && weight.numel() == 20480, "weight must have shape (20480,)");
    TORCH_CHECK(y.sizes() == x.sizes() && rstd.numel() == rows, "RMSNorm output shape mismatch");
    for (const auto& t : {x, weight, y, rstd}) {
        TORCH_CHECK(t.device() == x.device() && t.is_contiguous(), "tensors must be contiguous on the same NPU");
    }
    for (const auto& output : {y, rstd}) {
        at::assert_no_overlap(output, x);
        at::assert_no_overlap(output, weight);
    }
    at::assert_no_overlap(y, rstd);
    c10::DeviceGuard guard(x.device());
    auto& runtime = rms_norm_runtime();
    TORCH_CHECK(runtime.device.get_npu_arch() == 3510, "optimized RMSNorm requires Ascend dav-3510");
    TORCH_CHECK(runtime.device.get_num_vec_cores() >= 64, "optimized RMSNorm requires at least 64 Vector cores");
    TORCH_CHECK(runtime.device.get_num_ubuf_bytes_per_vec_core() >= 209664,
                "optimized RMSNorm requires at least 209664 bytes of UB per Vector core");
    const int slots = rows == 2048 ? 2 : 3;
    const std::string source = "#define XPU_RMS_ROW_SLOTS " + std::to_string(slots)
        + "\n#include <xpu_kernels/rms_norm/kernel.hpp>\n";
    const auto kernel = runtime.compile("rms_norm", source,
        {.extra_bisheng_flags = {"-std=c++17"}});
    runtime.launch(kernel, {.num_blocks = 64}, x.data_ptr(), weight.data_ptr(),
                   y.data_ptr(), rstd.data_ptr(), static_cast<uint32_t>(rows));
    return {y, rstd};
}
