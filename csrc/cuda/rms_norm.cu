#include <cstdint>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = __fadd_rn(value, __shfl_down_sync(0xffffffffU, value, offset));
    }
    return value;
}

template <typename scalar_t, int BlockThreads>
__global__ void rms_norm_cuda_kernel(
    const scalar_t* input, const scalar_t* weight, scalar_t* output,
    int64_t width, float epsilon) {
    constexpr int WarpCount = BlockThreads / 32;
    __shared__ float warp_sums[WarpCount];
    __shared__ float inverse_rms;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * width;

    float sum = 0.0f;
    for (int64_t column = threadIdx.x; column < width; column += BlockThreads) {
        const float value = static_cast<float>(input[row_offset + column]);
        sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        float block_sum = lane < WarpCount ? warp_sums[lane] : 0.0f;
        block_sum = warp_sum(block_sum);
        if (lane == 0) {
            inverse_rms = rsqrtf(block_sum / static_cast<float>(width) + epsilon);
        }
    }
    __syncthreads();

    for (int64_t column = threadIdx.x; column < width; column += BlockThreads) {
        const float value = static_cast<float>(input[row_offset + column]);
        // Materializing scalar_t enforces the golden's rounding before multiplication.
        const scalar_t normalized = static_cast<scalar_t>(__fmul_rn(value, inverse_rms));
        const float result = __fmul_rn(
            static_cast<float>(normalized), static_cast<float>(weight[column]));
        output[row_offset + column] = static_cast<scalar_t>(result);
    }
}

template <typename scalar_t>
void launch_typed(
    const at::Tensor& input, const at::Tensor& weight, at::Tensor& output,
    int64_t rows, int64_t width, float epsilon, cudaStream_t stream) {
    const auto* x = input.data_ptr<scalar_t>();
    const auto* gamma = weight.data_ptr<scalar_t>();
    auto* y = output.data_ptr<scalar_t>();
    const auto grid = static_cast<unsigned int>(rows);
    if (width <= 128) {
        rms_norm_cuda_kernel<scalar_t, 32><<<grid, 32, 0, stream>>>(x, gamma, y, width, epsilon);
    } else if (width <= 1024) {
        rms_norm_cuda_kernel<scalar_t, 128><<<grid, 128, 0, stream>>>(x, gamma, y, width, epsilon);
    } else {
        rms_norm_cuda_kernel<scalar_t, 256><<<grid, 256, 0, stream>>>(x, gamma, y, width, epsilon);
    }
}

}  // namespace

void launch_rms_norm_cuda(
    const at::Tensor& input, const at::Tensor& weight, at::Tensor& output, float epsilon) {
    const int64_t width = input.size(-1);
    const int64_t rows = input.numel() / width;
    const auto* properties = at::cuda::getDeviceProperties(input.get_device());
    TORCH_CHECK(rows <= properties->maxGridSize[0],
                "CUDA RMSNorm row count exceeds the device's one-dimensional grid limit");
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(input.get_device());
    switch (input.scalar_type()) {
        case at::kFloat:
            launch_typed<float>(input, weight, output, rows, width, epsilon, stream);
            break;
        case at::kHalf:
            launch_typed<at::Half>(input, weight, output, rows, width, epsilon, stream);
            break;
        case at::kBFloat16:
            launch_typed<at::BFloat16>(input, weight, output, rows, width, epsilon, stream);
            break;
        default:
            TORCH_CHECK(false, "unsupported CUDA RMSNorm dtype");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
