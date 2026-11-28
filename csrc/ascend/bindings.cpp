#include <torch/extension.h>

std::vector<torch::Tensor> rms_norm(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);
std::vector<torch::Tensor> mhc_post_backward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, bool);

std::vector<torch::Tensor> fused_quant_payload(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> fused_send_slots(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);
std::vector<torch::Tensor> fused_owner_combine(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("fused_quant_payload", &fused_quant_payload, "Fixed-capacity MoE fused_quant_payload");
    module.def("fused_send_slots", &fused_send_slots, "Fixed-capacity MoE fused_send_slots");
    module.def("fused_owner_combine", &fused_owner_combine, "Fixed-capacity MoE fused_owner_combine");
    module.def("rms_norm", &rms_norm, "RMSNorm: FP32 normalization and gamma, final output cast");
    module.def("mhc_post_backward", &mhc_post_backward, "NPU mixed Cube/Vector mHC post backward");
}
