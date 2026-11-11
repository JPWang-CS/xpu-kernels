#include <cmath>

#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

void launch_rms_norm_cuda(
    const at::Tensor& input, const at::Tensor& weight, at::Tensor& output, float epsilon);

namespace {

at::Tensor rms_norm(at::Tensor input, at::Tensor weight, at::Tensor output, double epsilon) {
    TORCH_CHECK(input.is_cuda(), "CUDA RMSNorm requires a CUDA input tensor");
    TORCH_CHECK(input.dim() >= 1 && input.size(-1) > 0,
                "input must have a positive last dimension");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && output.is_contiguous(),
                "input, weight, and output must be contiguous");
    TORCH_CHECK(weight.dim() == 1 && weight.numel() == input.size(-1),
                "weight must be one-dimensional and match the input width");
    TORCH_CHECK(weight.device() == input.device() && output.device() == input.device(),
                "input, weight, and output must be on the same CUDA device");
    TORCH_CHECK(weight.scalar_type() == input.scalar_type() &&
                    output.scalar_type() == input.scalar_type(),
                "input, weight, and output dtypes must match");
    TORCH_CHECK(output.sizes() == input.sizes(), "output shape must match input shape");
    TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf ||
                    input.scalar_type() == at::kBFloat16,
                "CUDA RMSNorm supports only float32, float16, and bfloat16");
    TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0, "epsilon must be finite and positive");
    // Exact input/output aliasing is supported; shifted aliases and gamma writes are unsafe.
    at::assert_no_partial_overlap(output, input);
    at::assert_no_overlap(output, weight);
    const c10::cuda::CUDAGuard device_guard(input.device());
    if (input.numel() != 0) {
        launch_rms_norm_cuda(input, weight, output, static_cast<float>(epsilon));
    }
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("rms_norm", &rms_norm, "CUDA RMSNorm (cast normalized values before gamma)");
}
