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

        const float TWO_PI = 2.0f*3.1415926f;

        // Get block/thread related numbers   
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        const int num_primitives = gaussian_colors.size(0) + wave_colors.size(0);

        for (int i = index; i < output.size(0)*num_primitives; i += stride){
            const int output_idx = i / num_primitives;

            // Check if we're out of bounds and return if so
            if(output_idx>output.size(0)) return;

            int primitive_index = i % num_primitives;
            const auto x = input[output_idx];

            if(primitive_index < gaussian_colors.size(0)){
                const float g_x = (x[0] - gaussian_means[primitive_index][0]) * gaussian_mats[primitive_index][0][0] +
                        (x[1] - gaussian_means[primitive_index][1]) * gaussian_mats[primitive_index][1][0];
                const float g_y = (x[0] - gaussian_means[primitive_index][0]) * gaussian_mats[primitive_index][0][1] + 
                        (x[1] - gaussian_means[primitive_index][1]) * gaussian_mats[primitive_index][1][1];
                const float g = expf(-(g_x * g_x + g_y * g_y) / 2.0f);
                if(g>0.000001f) for (int k = 0; k < output.size(1); k++){ atomicAdd(&output[output_idx][k], g*gaussian_colors[primitive_index][k]); }
                //if(g>0.000001f) for (int k = 0; k < output.size(1); k++){ output[output_idx][k] += g*gaussian_colors[primitive_index][k]; }

            }
            else{
                primitive_index -= gaussian_colors.size(0);
                const float g_x = (x[0] - wave_means[primitive_index][0]) * wave_mats[primitive_index][0][0] +
                            (x[1] - wave_means[primitive_index][1]) * wave_mats[primitive_index][1][0];
                const float g_y = (x[0] - wave_means[primitive_index][0]) * wave_mats[primitive_index][0][1] + 
                        (x[1] - wave_means[primitive_index][1]) * wave_mats[primitive_index][1][1];
                const float g = expf(-(g_x * g_x + g_y * g_y) / 2.0f);
                const float sx = sinf(TWO_PI*x[0]*wave_frequencies[primitive_index][0]);
                const float sy = sinf(TWO_PI*x[1]*wave_frequencies[primitive_index][1]);
                const float cx = cosf(TWO_PI*x[0]*wave_frequencies[primitive_index][0]);
                const float cy = cosf(TWO_PI*x[1]*wave_frequencies[primitive_index][1]);
                const float w = wave_coefficients[primitive_index][0]*cx*cy +
                    wave_coefficients[primitive_index][1]*cx*sy +
                    wave_coefficients[primitive_index][2]*sx*cy +
                    wave_coefficients[primitive_index][3]*sx*sy +
                    wave_coefficients[primitive_index][4];                               
                //if(abs(g*w)>0.000001f) for (int k = 0; k < output.size(1); k++){ output[output_idx][k] += g*w*wave_colors[primitive_index][k]; }
                if(abs(g*w)>0.000001f) for (int k = 0; k < output.size(1); k++){ atomicAdd(&output[output_idx][k], g*w*wave_colors[primitive_index][k]); }
            }
        }
    }

    
    template <typename scalar_t>
    __global__ void hybrid_model_backward_cuda_kernel(
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_output,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> gaussian_colors,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> gaussian_means,
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gaussian_mats,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_colors,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_means,
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> wave_mats,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_frequencies,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_coefficients,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_gaussian_colors,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_gaussian_means,
        torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_gaussian_mats,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_wave_colors,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_wave_means,
        torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_wave_mats,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_wave_frequencies,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_wave_coefficients
        ) {
            
        const float TWO_PI = 2.0f*3.1415926f;

        // Get block/thread related numbers   
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        const int num_primitives = gaussian_colors.size(0) + wave_colors.size(0);

        for (int i = index; i < input.size(0)*num_primitives; i += stride){
            const int output_idx = i / num_primitives;

            // Check if we're out of bounds and return if so
            if(output_idx>input.size(0)) return;

            int primitive_index = i % num_primitives;
            const auto x = input[output_idx];
            const auto y = grad_output[output_idx];

            if(primitive_index < gaussian_colors.size(0)){
                const float g_x = (x[0] - gaussian_means[primitive_index][0]) * gaussian_mats[primitive_index][0][0] +
                        (x[1] - gaussian_means[primitive_index][1]) * gaussian_mats[primitive_index][1][0];
                const float g_y = (x[0] - gaussian_means[primitive_index][0]) * gaussian_mats[primitive_index][0][1] + 
                        (x[1] - gaussian_means[primitive_index][1]) * gaussian_mats[primitive_index][1][1];
                const float g = expf(-(g_x * g_x + g_y * g_y) / 2.0f);
                if(g>0.000000001f) {
                    for (int k = 0; k < grad_output.size(1); k++){ 
                        // Gaussian color gradient update
                        atomicAdd(&grad_gaussian_colors[primitive_index][k], y[k]*g); 

                        // Gaussian position gradient update
                        atomicAdd(&grad_gaussian_means[primitive_index][0], 
                            y[k]*g*gaussian_colors[primitive_index][k]*
                            (g_x*gaussian_mats[primitive_index][0][0]+g_y*gaussian_mats[primitive_index][0][1]));
                        atomicAdd(&grad_gaussian_means[primitive_index][1], 
                            y[k]*g*gaussian_colors[primitive_index][k]*
                            (g_x*gaussian_mats[primitive_index][1][0]+g_y*gaussian_mats[primitive_index][1][1]));
                            
                        // Gaussian covariance matrix update
                        atomicAdd(&grad_gaussian_mats[primitive_index][0][0], 
                            y[k]*g*gaussian_colors[primitive_index][k]*g_x*-(x[0] - gaussian_means[primitive_index][0]));
                        atomicAdd(&grad_gaussian_mats[primitive_index][0][1], 
                            y[k]*g*gaussian_colors[primitive_index][k]*g_y*-(x[0] - gaussian_means[primitive_index][0]));
                        atomicAdd(&grad_gaussian_mats[primitive_index][1][0], 
                            y[k]*g*gaussian_colors[primitive_index][k]*g_x*-(x[1] - gaussian_means[primitive_index][1]));
                        atomicAdd(&grad_gaussian_mats[primitive_index][1][1], 
                            y[k]*g*gaussian_colors[primitive_index][k]*g_y*-(x[1] - gaussian_means[primitive_index][1]));
                    }
                }
            }
            else{
                primitive_index -= gaussian_colors.size(0);
                const float g_x = (x[0] - wave_means[primitive_index][0]) * wave_mats[primitive_index][0][0] +
                            (x[1] - wave_means[primitive_index][1]) * wave_mats[primitive_index][1][0];
                const float g_y = (x[0] - wave_means[primitive_index][0]) * wave_mats[primitive_index][0][1] + 
                        (x[1] - wave_means[primitive_index][1]) * wave_mats[primitive_index][1][1];
                const float g = expf(-(g_x * g_x + g_y * g_y) / 2.0f);
                                              
                if(g>0.0000000001f){
                    const float sx = sinf(TWO_PI*x[0]*wave_frequencies[primitive_index][0]);
                    const float sy = sinf(TWO_PI*x[1]*wave_frequencies[primitive_index][1]);
                    const float cx = cosf(TWO_PI*x[0]*wave_frequencies[primitive_index][0]);
                    const float cy = cosf(TWO_PI*x[1]*wave_frequencies[primitive_index][1]);
                    const float w = wave_coefficients[primitive_index][0]*cx*cy +
                        wave_coefficients[primitive_index][1]*cx*sy +
                        wave_coefficients[primitive_index][2]*sx*cy +
                        wave_coefficients[primitive_index][3]*sx*sy +
                        wave_coefficients[primitive_index][4]; 
                    for (int k = 0; k < grad_output.size(1); k++){ 
                        // Wave color gradient update
                        atomicAdd(&grad_wave_colors[primitive_index][k], y[k]*g*w); 
                        
                        // Wave position gradient update
                        atomicAdd(&grad_wave_means[primitive_index][0], 
                            y[k]*w*g*wave_colors[primitive_index][k]*
                            (g_x*wave_mats[primitive_index][0][0]+g_y*wave_mats[primitive_index][0][1]));
                        atomicAdd(&grad_wave_means[primitive_index][1], 
                            y[k]*w*g*wave_colors[primitive_index][k]*
                            (g_x*wave_mats[primitive_index][1][0]+g_y*wave_mats[primitive_index][1][1]));
                            
                        // Wave covariance matrix gradient update
                        atomicAdd(&grad_wave_mats[primitive_index][0][0], 
                            y[k]*w*g*wave_colors[primitive_index][k]*g_x*-(x[0] - wave_means[primitive_index][0]));
                        atomicAdd(&grad_wave_mats[primitive_index][0][1], 
                            y[k]*w*g*wave_colors[primitive_index][k]*g_y*-(x[0] - wave_means[primitive_index][0]));
                        atomicAdd(&grad_wave_mats[primitive_index][1][0], 
                            y[k]*w*g*wave_colors[primitive_index][k]*g_x*-(x[1] - wave_means[primitive_index][1]));
                        atomicAdd(&grad_wave_mats[primitive_index][1][1], 
                            y[k]*w*g*wave_colors[primitive_index][k]*g_y*-(x[1] - wave_means[primitive_index][1]));

                        // Wave coefficients gradient update
                        atomicAdd(&grad_wave_coefficients[primitive_index][0], y[k]*g*cx*cy*wave_colors[primitive_index][k]);
                        atomicAdd(&grad_wave_coefficients[primitive_index][1], y[k]*g*cx*sy*wave_colors[primitive_index][k]);
                        atomicAdd(&grad_wave_coefficients[primitive_index][2], y[k]*g*sx*cy*wave_colors[primitive_index][k]);
                        atomicAdd(&grad_wave_coefficients[primitive_index][3], y[k]*g*sx*sy*wave_colors[primitive_index][k]);
                        atomicAdd(&grad_wave_coefficients[primitive_index][4], y[k]*g*wave_colors[primitive_index][k]);

                        // Wave frequency gradient update
                        atomicAdd(&grad_wave_frequencies[primitive_index][0], 
                            y[k]*g*TWO_PI*x[0]*wave_colors[primitive_index][k]*(
                                wave_coefficients[primitive_index][0]*cy*-sx +
                                wave_coefficients[primitive_index][1]*sy*-sx +
                                wave_coefficients[primitive_index][2]*cy*cx +
                                wave_coefficients[primitive_index][3]*sy*cx
                            ));
                        atomicAdd(&grad_wave_frequencies[primitive_index][1], 
                            y[k]*g*TWO_PI*x[1]*wave_colors[primitive_index][k]*(
                                wave_coefficients[primitive_index][0]*cx*-sy +
                                wave_coefficients[primitive_index][1]*cx*cy +
                                wave_coefficients[primitive_index][2]*sx*-sy +
                                wave_coefficients[primitive_index][3]*sx*cy
                            ));
                    }
                }
            }
        }
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
    const int threads = 256;
    const int blocks = (batch_size*(num_gaussians+num_waves) + threads - 1) / threads;

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
    torch::Tensor grad_output,          // [N, n_chans]
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
        // Info for launching CUDA kernel
        const int batch_size = input.size(0);
        const int num_gaussians = gaussian_means.size(0);
        const int num_waves = wave_means.size(0);
        const int output_size = gaussian_colors.size(1);

        // Set up gradient tensors
        // auto grad_input = torch::zeros_like(input);
        auto grad_gaussian_colors = torch::zeros_like(gaussian_colors);
        auto grad_gaussian_means = torch::zeros_like(gaussian_means); 
        auto grad_gaussian_mats = torch::zeros_like(gaussian_mats); 
        auto grad_wave_colors = torch::zeros_like(wave_colors);
        auto grad_wave_means = torch::zeros_like(wave_means);
        auto grad_wave_mats = torch::zeros_like(wave_mats);
        auto grad_wave_frequencies = torch::zeros_like(wave_frequencies); 
        auto grad_wave_coefficients = torch::zeros_like(wave_coefficients);

        // Define block size and threads per block
        const int threads = 256;
        const int blocks = (batch_size*(num_gaussians+num_waves) + threads - 1) / threads;

        // Dispatch jobs
        AT_DISPATCH_FLOATING_TYPES(input.type(), "hybrid_model_cuda_backward", ([&] {
        hybrid_model_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            gaussian_colors.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            gaussian_means.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            gaussian_mats.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            wave_colors.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            wave_means.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            wave_mats.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            wave_frequencies.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            wave_coefficients.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            grad_gaussian_colors.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            grad_gaussian_means.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            grad_gaussian_mats.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            grad_wave_colors.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            grad_wave_means.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            grad_wave_mats.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            grad_wave_frequencies.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            grad_wave_coefficients.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>()
            );
        }));
        
        return {grad_gaussian_colors, grad_gaussian_means, grad_gaussian_mats,
                grad_wave_colors, grad_wave_means, grad_wave_mats, 
                grad_wave_frequencies, grad_wave_coefficients};
    }
