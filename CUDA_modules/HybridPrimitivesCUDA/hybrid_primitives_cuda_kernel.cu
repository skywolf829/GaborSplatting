#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace{

    template <typename scalar_t>
    __global__ void hybrid_model_forward_cuda_kernel(        
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> gaussian_colors,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> gaussian_means,
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gaussian_mats,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_colors,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_means,
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> wave_mats,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_frequencies,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_coefficients,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {

        // Get block/thread related numbers   
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        
        for (int i = index; i < output.size(0); i+=stride){  
            const auto x = input[i];
            for(int j = 0; j < gaussian_colors.size(0); j++){
                const auto g_x = (x[0] - gaussian_means[j][0]) * gaussian_mats[j][0][0] +
                        (x[1] - gaussian_means[j][1]) * gaussian_mats[j][1][0];
                const auto g_y = (x[0] - gaussian_means[j][0]) * gaussian_mats[j][0][1] + 
                        (x[1] - gaussian_means[j][1]) * gaussian_mats[j][1][1];
                const auto g = exp(-(g_x * g_x + g_y * g_y) / 2.0f);
                for (int k = 0; k < output.size(1); k++){ output[i][k] += g*gaussian_colors[j][k]; }
            }            
        }
    }

    
    template <typename scalar_t>
    __global__ void hybrid_model_backward_cuda_kernel(
    const int batch_size, 
    const int num_gaussians,
    const int num_waves,
    const int num_channels,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> gaussian_means,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gaussian_mats,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> gaussian_colors,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_means,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> wave_mats,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_frequencies,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_coefficients,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_colors,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_gaussian_means,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_gaussian_mats,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_gaussian_colors,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_wave_means,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_wave_mats,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_wave_frequencies,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_wave_coefficients,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_wave_colors,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
        return;
    }
    



}

std::vector<torch::Tensor> hybrid_model_forward_cuda(
    torch::Tensor input,                // [N, n_dims]
    torch::Tensor gaussian_colors,      // [M, n_chans]
    torch::Tensor gaussian_means,       // [M, n_dims]
    torch::Tensor gaussian_mats,        // [M, n_dims, n_dims]
    torch::Tensor wave_colors,          // [W, n_chans]
    torch::Tensor wave_means,           // [W, n_dims]
    torch::Tensor wave_mats,            // [W, n_dims, n_dims]
    torch::Tensor wave_frequencies,     // [W, n_dims]
    torch::Tensor wave_coefficients     // [W, 5]
    ) {

    // Get sizes for the output
    const int batch_size = input.size(0);
    const int num_gaussians = gaussian_means.size(0);
    const int num_waves = wave_means.size(0);
    const int output_size = gaussian_colors.size(1);

    // Create output tensor
    auto output = torch::zeros({batch_size, output_size}, input.device());

    // Define block size and threads per block
    const int threads = 1024;
    const int blocks = (batch_size + threads - 1) / threads;

    // Dispatch jobs
    AT_DISPATCH_FLOATING_TYPES(input.type(), "hybrid_model_cuda_forward", ([&] {
    hybrid_model_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        gaussian_colors.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        gaussian_means.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        gaussian_mats.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        wave_colors.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        wave_means.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        wave_mats.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        wave_frequencies.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        wave_coefficients.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>()
        );
    }));
    
    return {output};
}


std::vector<torch::Tensor> hybrid_model_backward_cuda(
    torch::Tensor input, // [N, 2]
    torch::Tensor gaussian_means, // [M, 2]
    torch::Tensor gaussian_mats, // [M, 2, 2]
    torch::Tensor gaussian_colors, // [M, 3]
    torch::Tensor wave_means, // [W, 2]
    torch::Tensor wave_mats, // [W, 2, 2]
    torch::Tensor wave_frequencies, // [W, 2]
    torch::Tensor wave_coefficients, // [W, 5]
    torch::Tensor wave_colors,    // [W, 3]
    torch::Tensor grad_gaussian_means, // [M, 2]
    torch::Tensor grad_gaussian_mats, // [M, 2, 2]
    torch::Tensor grad_gaussian_colors, // [M, 3]
    torch::Tensor grad_wave_means, // [W, 2]
    torch::Tensor grad_wave_mats, // [W, 2, 2]
    torch::Tensor grad_wave_frequencies, // [W, 2]
    torch::Tensor grad_wave_coefficients, // [W, 5]
    torch::Tensor grad_wave_colors) {
        auto output = torch::zeros({1});
        return {output};
    }
