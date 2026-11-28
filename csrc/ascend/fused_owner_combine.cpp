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
            root, "XPU_KERNELS", "fused_owner_combine", {root / "kernels" / "include"}, {"xpu_kernels/"}));
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
std::vector<torch::Tensor> fused_owner_combine(torch::Tensor received, torch::Tensor ids, torch::Tensor weights, torch::Tensor active, torch::Tensor output) {
    check(received, received, std::vector<int64_t>{2,16,8,2560}, at::kBFloat16, "received");
    check(ids, received, std::vector<int64_t>{16,8}, at::kInt, "ids");
    check(weights, received, std::vector<int64_t>{16,8}, at::kFloat, "weights");
    check(active, received, std::vector<int64_t>{16}, at::kBool, "active");
    check(output, received, std::vector<int64_t>{16,2560}, at::kBFloat16, "output");
    at::assert_no_overlap(output, received);
    at::assert_no_overlap(output, ids);
    at::assert_no_overlap(output, weights);
    at::assert_no_overlap(output, active);
    c10::DeviceGuard guard(received.device());
    auto& jit = runtime();
    const auto kernel = jit.compile("fused_owner_combine", "#include <xpu_kernels/fused_owner_combine/kernel.hpp>\n",
        {.extra_bisheng_flags = {"-std=c++17", "-ffp-contract=off"}});
    jit.launch(kernel, {.num_blocks=16}, received.data_ptr(), ids.data_ptr(), weights.data_ptr(), active.data_ptr(), output.data_ptr(), uint32_t(16), uint32_t(2560), uint32_t(8), uint32_t(128), uint32_t(2));
    return {output};
}
