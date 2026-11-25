#include <torch/extension.h>

std::vector<torch::Tensor> rms_norm(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);
std::vector<torch::Tensor> mhc_post_backward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, bool);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("rms_norm", &rms_norm, "RMSNorm: FP32 normalization and gamma, final output cast");
    module.def("mhc_post_backward", &mhc_post_backward, "NPU mixed Cube/Vector mHC post backward");
}
