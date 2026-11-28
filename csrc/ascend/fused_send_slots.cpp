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
            root, "XPU_KERNELS", "fused_send_slots", {root / "kernels" / "include"}, {"xpu_kernels/"}));
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
std::vector<torch::Tensor> fused_send_slots(torch::Tensor down, torch::Tensor inverse, torch::Tensor routes, torch::Tensor active, torch::Tensor slots, int64_t expert_start, int64_t expert_end) {
    TORCH_CHECK(down.dim()==2 && down.size(0)>=1 && down.size(0)<=256, "down must have 1..256 rows");
    TORCH_CHECK((expert_start==0 || expert_start==128) && expert_end==expert_start+128, "invalid expert range");
    check(down, down, std::vector<int64_t>{down.size(0),2560}, at::kBFloat16, "down");
    check(inverse, down, std::vector<int64_t>{256}, at::kInt, "inverse");
    check(routes, down, std::vector<int64_t>{32,8}, at::kInt, "routes");
    check(active, down, std::vector<int64_t>{32}, at::kBool, "active");
    check(slots, down, std::vector<int64_t>{2,16,8,2560}, at::kBFloat16, "slots");
    at::assert_no_overlap(slots, down);
    at::assert_no_overlap(slots, inverse);
    at::assert_no_overlap(slots, routes);
    at::assert_no_overlap(slots, active);
    c10::DeviceGuard guard(down.device());
    auto& jit = runtime();
    const auto kernel = jit.compile("fused_send_slots", "#include <xpu_kernels/fused_send_slots/kernel.hpp>\n",
        {.extra_bisheng_flags = {"-std=c++17", "-ffp-contract=off"}});
    jit.launch(kernel, {.num_blocks=32}, down.data_ptr(), inverse.data_ptr(), routes.data_ptr(), active.data_ptr(), slots.data_ptr(), uint32_t(256), uint32_t(down.size(0)), uint32_t(2560), uint32_t(expert_start), uint32_t(expert_end), uint32_t(8));
    return {slots};
}
