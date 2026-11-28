#include <cstdlib>
#include <filesystem>
#include <vector>
#include <ATen/MemoryOverlap.h>
#include <torch/extension.h>
#include <deep_jit/backend/ascend/backend.hpp>
#include <deep_jit/runtime/runtime.hpp>
namespace {
using Runtime = deep_jit::Runtime<deep_jit::Ascend>;
Runtime& runtime() {
    static auto instance = [] {
        const char* value = std::getenv("XPU_KERNELS_ROOT");
        TORCH_CHECK(value, "XPU_KERNELS_ROOT must identify native resources");
        auto root = std::filesystem::absolute(value).lexically_normal();
        return deep_jit::create_lazy_jit<deep_jit::Ascend>(deep_jit::Config(
            root, "XPU_KERNELS", "fused_quant_payload", {root / "kernels" / "include"}, {"xpu_kernels/"}));
    }();
    return *instance.get();
}
void check(const torch::Tensor& value, const torch::Tensor& first,
           const std::vector<int64_t>& shape, at::ScalarType dtype, const char* name) {
    TORCH_CHECK(value.device().type()==c10::DeviceType::PrivateUse1, "NPU required");
    TORCH_CHECK(value.device()==first.device() && value.is_contiguous(), "same NPU and contiguous tensors required");
    TORCH_CHECK(value.sizes().vec()==shape && value.scalar_type()==dtype, "invalid shape/dtype for ", name);
}
}
std::vector<torch::Tensor> fused_quant_payload(torch::Tensor x, torch::Tensor ids, torch::Tensor active, torch::Tensor payload, torch::Tensor scale) {
    check(x, x, std::vector<int64_t>{16,2560}, at::kBFloat16, "x");
    check(ids, x, std::vector<int64_t>{16,8}, at::kInt, "ids");
    check(active, x, std::vector<int64_t>{16}, at::kBool, "active");
    check(payload, x, std::vector<int64_t>{16,2624}, at::kChar, "payload");
    check(scale, x, std::vector<int64_t>{16}, at::kFloat, "scale");
    at::assert_no_overlap(payload, x);
    at::assert_no_overlap(payload, ids);
    at::assert_no_overlap(payload, active);
    at::assert_no_overlap(scale, x);
    at::assert_no_overlap(scale, ids);
    at::assert_no_overlap(scale, active);
    at::assert_no_overlap(payload, scale);
    c10::DeviceGuard guard(x.device());
    auto& jit = runtime();
    const auto kernel = jit.compile("fused_quant_payload", "#include <xpu_kernels/fused_quant_payload/kernel.hpp>\n",
        {.extra_bisheng_flags = {"-std=c++17", "-ffp-contract=off"}});
    jit.launch(kernel, {.num_blocks=16}, x.data_ptr(), ids.data_ptr(), active.data_ptr(), payload.data_ptr(), scale.data_ptr(), uint32_t(16), uint32_t(2560), uint32_t(8), uint32_t(2624));
    return {payload, scale};
}
