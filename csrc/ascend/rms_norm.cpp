#include <filesystem>
#include <memory>
#include <cstdlib>
#include <limits>
#include <string>

#include <torch/extension.h>

#include <deep_jit/backend/ascend/backend.hpp>
#include <deep_jit/runtime/runtime.hpp>

std::vector<torch::Tensor> mhc_post_backward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, bool);

namespace {

using Runtime = deep_jit::Runtime<deep_jit::Ascend>;

Runtime& get_runtime() {
    static auto jit = [] {
        const char* root_value = std::getenv("XPU_KERNELS_ROOT");
        TORCH_CHECK(root_value != nullptr, "XPU_KERNELS_ROOT must point to the project root");
        const std::filesystem::path root = std::filesystem::absolute(root_value).lexically_normal();
        return deep_jit::create_lazy_jit<deep_jit::Ascend>(deep_jit::Config(
            root, "XPU_KERNELS", "rms-norm-v1",
            {root / "kernels" / "include"}, {"xpu_kernels/"}));
    }();
    return *jit.get();
}

std::string source_for_dtype(const int dtype_code, const torch::Tensor& x) {
    const uint64_t width = x.size(-1);
    const uint64_t align = 32 / x.element_size();
    const uint64_t padded = (width + align - 1) / align * align;
    const uint64_t separate_ub = padded * (2 * x.element_size() + 2 * sizeof(float)) + 32;
    const bool reuse = separate_ub > static_cast<uint64_t>(get_runtime().device.get_num_ubuf_bytes_per_vec_core());
    return "#define XPU_RMS_DTYPE " + std::to_string(dtype_code) +
           "\n#define XPU_RMS_REUSE_WORK " + std::to_string(reuse) +
           "\n#include <xpu_kernels/rms_norm_ascend.hpp>\n";
}

void launch_float(torch::Tensor& x, torch::Tensor& weight, torch::Tensor& out, const double eps) {
    auto& runtime = get_runtime();
    const auto kernel = runtime.compile("rms_norm_f32", source_for_dtype(0, x));
    runtime.launch(kernel, {.num_blocks = static_cast<int>(x.numel() / x.size(-1))},
                   out.data_ptr<float>(), x.data_ptr<float>(), weight.data_ptr<float>(),
                   static_cast<uint32_t>(x.size(-1)), static_cast<float>(eps));
}

void launch_half(torch::Tensor& x, torch::Tensor& weight, torch::Tensor& out, const double eps) {
    auto& runtime = get_runtime();
    const auto kernel = runtime.compile("rms_norm_f16", source_for_dtype(1, x));
    runtime.launch(kernel, {.num_blocks = static_cast<int>(x.numel() / x.size(-1))},
                   reinterpret_cast<uint16_t*>(out.data_ptr()),
                   reinterpret_cast<const uint16_t*>(x.data_ptr()),
                   reinterpret_cast<const uint16_t*>(weight.data_ptr()),
                   static_cast<uint32_t>(x.size(-1)), static_cast<float>(eps));
}

void launch_bfloat16(torch::Tensor& x, torch::Tensor& weight, torch::Tensor& out, const double eps) {
    auto& runtime = get_runtime();
    const auto kernel = runtime.compile("rms_norm_bf16", source_for_dtype(2, x));
    runtime.launch(kernel, {.num_blocks = static_cast<int>(x.numel() / x.size(-1))},
                   reinterpret_cast<uint16_t*>(out.data_ptr()),
                   reinterpret_cast<const uint16_t*>(x.data_ptr()),
                   reinterpret_cast<const uint16_t*>(weight.data_ptr()),
                   static_cast<uint32_t>(x.size(-1)), static_cast<float>(eps));
}

torch::Tensor rms_norm(torch::Tensor x, torch::Tensor weight, torch::Tensor out, const double eps) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "input must be an NPU tensor");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) > 0, "input last dimension must be non-empty");
    TORCH_CHECK(weight.dim() == 1 && weight.numel() == x.size(-1), "weight must match the last dimension");
    TORCH_CHECK(weight.scalar_type() == x.scalar_type() && out.scalar_type() == x.scalar_type(), "dtype mismatch");
    TORCH_CHECK(weight.device() == x.device() && out.device() == x.device(), "device mismatch");
    TORCH_CHECK(out.sizes() == x.sizes(), "output shape must match input");
    TORCH_CHECK(eps > 0.0, "eps must be positive");
    c10::DeviceGuard device_guard(x.device());
    auto& runtime = get_runtime();
    const auto width = static_cast<uint64_t>(x.size(-1));
    const auto alignment = 32 / x.element_size();
    const auto padded = (width + alignment - 1) / alignment * alignment;
    const auto required_ub = padded * (x.element_size() + 2 * sizeof(float)) + 32;
    TORCH_CHECK(width <= std::numeric_limits<uint32_t>::max(), "width exceeds kernel indexing range");
    TORCH_CHECK(required_ub <= static_cast<uint64_t>(runtime.device.get_num_ubuf_bytes_per_vec_core()),
                "RMSNorm row requires ", required_ub, " bytes of UB; split-row implementation required for this width");
    TORCH_CHECK(x.numel() / x.size(-1) <= std::numeric_limits<int>::max(), "row count exceeds launch range");
    switch (x.scalar_type()) {
        case at::kFloat: launch_float(x, weight, out, eps); break;
        case at::kHalf: launch_half(x, weight, out, eps); break;
        case at::kBFloat16: launch_bfloat16(x, weight, out, eps); break;
        default: TORCH_CHECK(false, "unsupported NPU dtype: ", x.scalar_type());
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("mhc_post_backward", &mhc_post_backward, "NPU mixed Cube/Vector mHC post backward");
    module.def("rms_norm", &rms_norm, "xpu-kernels AscendC RMSNorm");
}
