#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

std::vector<torch::Tensor> hybrid_model_forward_cuda(
    torch::Tensor input, // [N, 2]
    torch::Tensor gaussian_means, // [M, 2]
    torch::Tensor gaussian_mats, // [M, 2, 2]
    torch::Tensor gaussian_colors, // [M, 3]
    torch::Tensor wave_means, // [W, 2]
    torch::Tensor wave_mats, // [W, 2, 2]
    torch::Tensor wave_frequencies, // [W, 2]
    torch::Tensor wave_coefficients, // [W, 5]
    torch::Tensor wave_colors) {

    // Get sizes for the output
    const int batch_size = input.size(0);
    const int num_gaussians = gaussian_means.size(0);
    const int num_waves = wave_means.size(0);
    const int output_size = 0;
    if(num_gaussians > 0){ output_size = gaussian_colors.size(1); }
    else{ output_size = wave_colors.size(1); }

    // Create output tensor
    auto output = torch::zeros({batch_size, output_size});

    // Define block size and threads per block
    const int blockSize = 256; // number of threads (per block)
    const int numBlocks = (batch_size + blockSize - 1) / blockSize; // number of blocks

    // Dispatch jobs
    AT_DISPATCH_FLOATING_TYPES(input.type(), "hybrid_model_forward_cuda", ([&] {
    hybrid_model_forward_cuda_kernel<scalar_t><<<numBlocks, blockSize>>>(
        batch_size, num_gaussians, num_waves,
        input.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        gaussian_means.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        gaussian_mats.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        gaussian_colors.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        wave_means.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        wave_mats.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        wave_frequencies.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        wave_coefficients.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        wave_colors.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()
        );
    }));

    return {output};
}


template <typename scalar_t>
__global__ void hybrid_model_forward_cuda_kernel(
    const int batch_size, 
    const int num_gaussians,
    const int num_waves,
    const int num_channels,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> gaussian_means,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gaussian_mats,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> gaussian_colors,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> wave_means,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> wave_mats,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> wave_frequencies,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> wave_coefficients,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> wave_colors,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output) {

    // Get block/thread related numbers   
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < batch_size; i += stride){        
        auto x = input[i];
        for(int j = 0; j < num_gaussians; j++){
            auto g_x = x - gaussian_means[j];
            auto g_x_x = g_x[0] * gaussian_mats_a[j][0][0] + g_x[1] * gaussian_mats[j][1][0];
            auto g_x_y = g_x[0] * gaussian_mats[j][0][1] + g_x[1] * gaussian_mats[j][1][1];
            g_x = exp(-(g_x_x * g_x_x + g_x_y * g_x) / 2.0f);
            for (int k = 0; k < num_channels; k++){ output[i][k] += g_x*gaussian_colors[j][k]; }
        }
    }
}