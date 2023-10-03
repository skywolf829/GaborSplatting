#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace{

    template <typename scalar_t>
    __global__ void gaussian_forward_cuda_kernel(        
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> gaussian_colors,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> gaussian_means,
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gaussian_mats,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {

        // Get block/thread related numbers   
        const long index = blockIdx.x * blockDim.x + threadIdx.x;
        const long stride = blockDim.x * gridDim.x;
        const long num_primitives = gaussian_colors.size(0);

        long max_iters = input.size(0)*num_primitives;
        for(long i = index; i < max_iters; i+= stride){
            const long output_idx = i / num_primitives;
            const long primitive_index = i % num_primitives;

            // Check if we're out of bounds and return if so
            if(output_idx>output.size(0)) continue;

            const auto x = input[output_idx];
            const float g_x = (x[0] - gaussian_means[primitive_index][0]) * gaussian_mats[primitive_index][0][0] +
                    (x[1] - gaussian_means[primitive_index][1]) * gaussian_mats[primitive_index][1][0];
            const float g_y = (x[0] - gaussian_means[primitive_index][0]) * gaussian_mats[primitive_index][0][1] + 
                    (x[1] - gaussian_means[primitive_index][1]) * gaussian_mats[primitive_index][1][1];
            const float g = expf(-(g_x * g_x + g_y * g_y) / 2.0f);
            if(g<0.0000001f) continue; 
            for (int k = 0; k < output.size(1); k++){ atomicAdd(&output[output_idx][k], g*gaussian_colors[primitive_index][k]); }
        }
    }

    template <typename scalar_t>
    __global__ void wave_forward_cuda_kernel(        
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_colors,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_means,
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> wave_mats,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_frequencies,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_coefficients,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {

        const float TWO_PI = 2.0f*3.1415926f;

        // Get block/thread related numbers   
        const long index = blockIdx.x * blockDim.x + threadIdx.x;
        const long stride = blockDim.x * gridDim.x;
        const long num_primitives = wave_colors.size(0);

        long max_iters = input.size(0)*num_primitives;
        for(long i = index; i < max_iters; i+= stride){
            const long output_idx = i / num_primitives;
            const long primitive_index = i % num_primitives;

            // Check if we're out of bounds and return if so
            if(output_idx > output.size(0)) continue;

            const auto x = input[output_idx];
            
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
            if(g<0.0000001f) continue; 
            for (int k = 0; k < output.size(1); k++){ atomicAdd(&output[output_idx][k], g*w*wave_colors[primitive_index][k]); }
        }
    }

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
        const long index = blockIdx.x * blockDim.x + threadIdx.x;
        const long stride = blockDim.x * gridDim.x;
        const long num_primitives = gaussian_colors.size(0) + wave_colors.size(0);

        long max_iters = input.size(0)*num_primitives;
        for(long i = index; i < max_iters; i+= stride){
            const long output_idx = i / num_primitives;
            long primitive_index = i % num_primitives;

            // Check if we're out of bounds and return if so
            if(output_idx>output.size(0)) continue;

            const auto x = input[output_idx];

            if(primitive_index < gaussian_colors.size(0)){
                const float g_x = (x[0] - gaussian_means[primitive_index][0]) * gaussian_mats[primitive_index][0][0] +
                        (x[1] - gaussian_means[primitive_index][1]) * gaussian_mats[primitive_index][1][0];
                const float g_y = (x[0] - gaussian_means[primitive_index][0]) * gaussian_mats[primitive_index][0][1] + 
                        (x[1] - gaussian_means[primitive_index][1]) * gaussian_mats[primitive_index][1][1];
                const float g = expf(-(g_x * g_x + g_y * g_y) / 2.0f);
                if(g<0.0000001f) continue; 
                for (int k = 0; k < output.size(1); k++){ atomicAdd(&output[output_idx][k], g*gaussian_colors[primitive_index][k]); }
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
                if(g<0.0000001f) continue; 
                for (int k = 0; k < output.size(1); k++){ atomicAdd(&output[output_idx][k], g*w*wave_colors[primitive_index][k]); }
            }
        }
    }

    template <typename scalar_t>
    __global__ void pointwise_forward_cuda_kernel(        
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
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;

        __syncthreads();
        int max_iters = input.size(0);
        for(int i = index; i < max_iters; i+= stride){
            const auto x = input[i];
            //float3 result = {0.0f, 0.0f, 0.0f};
            // Loop over gaussians
            for(int j = 0; j < gaussian_colors.size(0); j++){
                const float g_x = (x[0] - gaussian_means[j][0]) * gaussian_mats[j][0][0] +
                        (x[1] - gaussian_means[j][1]) * gaussian_mats[j][1][0];
                const float g_y = (x[0] - gaussian_means[j][0]) * gaussian_mats[j][0][1] + 
                        (x[1] - gaussian_means[j][1]) * gaussian_mats[j][1][1];
                const float g = expf(-(g_x * g_x + g_y * g_y) / 2.0f);
                if(g<0.0000001f) continue; 
                for (int k = 0; k < output.size(1); k++){ output[i][k] += g*gaussian_colors[j][k]; }
            }
            // Loop over waves
            for(int j = 0; j < wave_colors.size(0); j++){
                const float g_x = (x[0] - wave_means[j][0]) * wave_mats[j][0][0] +
                            (x[1] - wave_means[j][1]) * wave_mats[j][1][0];
                const float g_y = (x[0] - wave_means[j][0]) * wave_mats[j][0][1] + 
                        (x[1] - wave_means[j][1]) * wave_mats[j][1][1];
                const float g = expf(-(g_x * g_x + g_y * g_y) / 2.0f);                     
                if(g<0.0000001f) continue; 
                const float sx = sinpif(2.0f*x[0]*wave_frequencies[j][0]);
                const float sy = sinpif(2.0f*x[1]*wave_frequencies[j][1]);
                const float cx = cospif(2.0f*x[0]*wave_frequencies[j][0]);
                const float cy = cospif(2.0f*x[1]*wave_frequencies[j][1]);
                const float w = wave_coefficients[j][0]*cx*cy +
                    wave_coefficients[j][1]*cx*sy +
                    wave_coefficients[j][2]*sx*cy +
                    wave_coefficients[j][3]*sx*sy +
                    wave_coefficients[j][4];         
                if(abs(w)<0.0000001f) continue;  
                for (int k = 0; k < output.size(1); k++){ output[i][k] += g*w*wave_colors[j][k]; }
            }
        }
    }

    
    template <typename scalar_t>
    __global__ void gaussian_backward_cuda_kernel(
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_output,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> gaussian_colors,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> gaussian_means,
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gaussian_mats,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_gaussian_colors,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_gaussian_means,
        torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_gaussian_mats
        ) {
            
        // Get block/thread related numbers   
        const long index = blockIdx.x * blockDim.x + threadIdx.x;
        const long stride = blockDim.x * gridDim.x;
        const long num_primitives = gaussian_colors.size(0);

        long max_iters = input.size(0)*num_primitives;
        for(long i = index; i < max_iters; i+= stride){
            const long output_idx = i / num_primitives;
            const long primitive_index = i % num_primitives;

            // Check if we're out of bounds and return if so
            if(output_idx>input.size(0)) continue;

            const auto x = input[output_idx];
            const auto y = grad_output[output_idx];

            const float g_x = (x[0] - gaussian_means[primitive_index][0]) * gaussian_mats[primitive_index][0][0] +
                    (x[1] - gaussian_means[primitive_index][1]) * gaussian_mats[primitive_index][1][0];
            const float g_y = (x[0] - gaussian_means[primitive_index][0]) * gaussian_mats[primitive_index][0][1] + 
                    (x[1] - gaussian_means[primitive_index][1]) * gaussian_mats[primitive_index][1][1];
            const float g = expf(-(g_x * g_x + g_y * g_y) / 2.0f);
            if(g<0.0000001f) continue; 
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
    

    template <typename scalar_t>
    __global__ void wave_backward_cuda_kernel(
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_output,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_colors,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_means,
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> wave_mats,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_frequencies,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> wave_coefficients,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_wave_colors,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_wave_means,
        torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_wave_mats,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_wave_frequencies,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_wave_coefficients
        ) {
            
        const float TWO_PI = 2.0f*3.1415926f;

        // Get block/thread related numbers   
        const long index = blockIdx.x * blockDim.x + threadIdx.x;
        const long stride = blockDim.x * gridDim.x;
        const long num_primitives = wave_colors.size(0);

        long max_iters = input.size(0)*num_primitives;
        for(long i = index; i < max_iters; i+= stride){
            const long output_idx = i / num_primitives;
            const long primitive_index = i % num_primitives;

            // Check if we're out of bounds and return if so
            if(output_idx>input.size(0)) continue;

            const auto x = input[output_idx];
            const auto y = grad_output[output_idx];

            const float g_x = (x[0] - wave_means[primitive_index][0]) * wave_mats[primitive_index][0][0] +
                        (x[1] - wave_means[primitive_index][1]) * wave_mats[primitive_index][1][0];
            const float g_y = (x[0] - wave_means[primitive_index][0]) * wave_mats[primitive_index][0][1] + 
                    (x[1] - wave_means[primitive_index][1]) * wave_mats[primitive_index][1][1];
            const float g = expf(-(g_x * g_x + g_y * g_y) / 2.0f);
            // Save time by not computing if unnecessary
            if(g<0.0000001f) continue;                
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
        const long index = blockIdx.x * blockDim.x + threadIdx.x;
        const long stride = blockDim.x * gridDim.x;
        const long num_primitives = gaussian_colors.size(0) + wave_colors.size(0);

        long max_iters = input.size(0)*num_primitives;
        for(long i = index; i < max_iters; i+= stride){
            const long output_idx = i / num_primitives;
            long primitive_index = i % num_primitives;

            // Check if we're out of bounds and return if so
            if(output_idx>input.size(0)) continue;

            const auto x = input[output_idx];
            const auto y = grad_output[output_idx];

            if(primitive_index < gaussian_colors.size(0)){
                const float g_x = (x[0] - gaussian_means[primitive_index][0]) * gaussian_mats[primitive_index][0][0] +
                        (x[1] - gaussian_means[primitive_index][1]) * gaussian_mats[primitive_index][1][0];
                const float g_y = (x[0] - gaussian_means[primitive_index][0]) * gaussian_mats[primitive_index][0][1] + 
                        (x[1] - gaussian_means[primitive_index][1]) * gaussian_mats[primitive_index][1][1];
                const float g = expf(-(g_x * g_x + g_y * g_y) / 2.0f);
                if(g<0.0000001f) continue; 
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
            else{
                primitive_index -= gaussian_colors.size(0);
                const float g_x = (x[0] - wave_means[primitive_index][0]) * wave_mats[primitive_index][0][0] +
                            (x[1] - wave_means[primitive_index][1]) * wave_mats[primitive_index][1][0];
                const float g_y = (x[0] - wave_means[primitive_index][0]) * wave_mats[primitive_index][0][1] + 
                        (x[1] - wave_means[primitive_index][1]) * wave_mats[primitive_index][1][1];
                const float g = expf(-(g_x * g_x + g_y * g_y) / 2.0f);
                                              
                if(g<0.0000001f) continue; 
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
     template <typename scalar_t>
    __global__ void pointwise_backward_cuda_kernel(
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
            
        // Get block/thread related numbers   
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;

        int max_iters = input.size(0);
        for(int i = index; i < max_iters; i+= stride){
            const auto x = input[i];
            const auto y = grad_output[i];

            for(int j = 0; j < gaussian_colors.size(0); j++){
                const float g_x = (x[0] - gaussian_means[j][0]) * gaussian_mats[j][0][0] +
                        (x[1] - gaussian_means[j][1]) * gaussian_mats[j][1][0];
                const float g_y = (x[0] - gaussian_means[j][0]) * gaussian_mats[j][0][1] + 
                        (x[1] - gaussian_means[j][1]) * gaussian_mats[j][1][1];
                const float g = expf(-(g_x * g_x + g_y * g_y) / 2.0f);
                if(g<0.0000001f) continue; 
                for (int k = 0; k < grad_output.size(1); k++){ 
                    // Gaussian color gradient update
                    grad_gaussian_colors[j][k] += y[k]*g; 

                    // Gaussian position gradient update
                    grad_gaussian_means[j][0] += 
                        y[k]*g*gaussian_colors[j][k]*
                        (g_x*gaussian_mats[j][0][0]+g_y*gaussian_mats[j][0][1]);
                    grad_gaussian_means[j][1] += 
                        y[k]*g*gaussian_colors[j][k]*
                        (g_x*gaussian_mats[j][1][0]+g_y*gaussian_mats[j][1][1]);
                        
                    // Gaussian covariance matrix update
                    grad_gaussian_mats[j][0][0] +=
                        y[k]*g*gaussian_colors[j][k]*g_x*-(x[0] - gaussian_means[j][0]);
                    grad_gaussian_mats[j][0][1] +=
                        y[k]*g*gaussian_colors[j][k]*g_y*-(x[0] - gaussian_means[j][0]);
                    grad_gaussian_mats[j][1][0] +=
                        y[k]*g*gaussian_colors[j][k]*g_x*-(x[1] - gaussian_means[j][1]);
                    grad_gaussian_mats[j][1][1] +=
                        y[k]*g*gaussian_colors[j][k]*g_y*-(x[1] - gaussian_means[j][1]);
                }
            }
            for(int j = 0; j < wave_colors.size(0); j++){
                const float g_x = (x[0] - wave_means[j][0]) * wave_mats[j][0][0] +
                            (x[1] - wave_means[j][1]) * wave_mats[j][1][0];
                const float g_y = (x[0] - wave_means[j][0]) * wave_mats[j][0][1] + 
                        (x[1] - wave_means[j][1]) * wave_mats[j][1][1];
                const float g = expf(-(g_x * g_x + g_y * g_y) / 2.0f);  
                if(g<0.0000001f) continue; 
                const float sx = sinpif(2.0f*x[0]*wave_frequencies[j][0]);
                const float sy = sinpif(2.0f*x[1]*wave_frequencies[j][1]);
                const float cx = cospif(2.0f*x[0]*wave_frequencies[j][0]);
                const float cy = cospif(2.0f*x[1]*wave_frequencies[j][1]);
                const float w = wave_coefficients[j][0]*cx*cy +
                    wave_coefficients[j][1]*cx*sy +
                    wave_coefficients[j][2]*sx*cy +
                    wave_coefficients[j][3]*sx*sy +
                    wave_coefficients[j][4]; 
                for (int k = 0; k < grad_output.size(1); k++){ 
                    // Wave color gradient update
                    grad_wave_colors[j][k] += y[k]*g*w; 
                    
                    // Wave position gradient update
                    grad_wave_means[j][0] +=
                        y[k]*w*g*wave_colors[j][k]*
                        (g_x*wave_mats[j][0][0]+g_y*wave_mats[j][0][1]);
                    grad_wave_means[j][1] +=
                        y[k]*w*g*wave_colors[j][k]*
                        (g_x*wave_mats[j][1][0]+g_y*wave_mats[j][1][1]);
                        
                    // Wave covariance matrix gradient update
                    grad_wave_mats[j][0][0] += 
                        y[k]*w*g*wave_colors[j][k]*g_x*-(x[0] - wave_means[j][0]);
                    grad_wave_mats[j][0][1] += 
                        y[k]*w*g*wave_colors[j][k]*g_y*-(x[0] - wave_means[j][0]);
                    grad_wave_mats[j][1][0] += 
                        y[k]*w*g*wave_colors[j][k]*g_x*-(x[1] - wave_means[j][1]);
                    grad_wave_mats[j][1][1] += 
                        y[k]*w*g*wave_colors[j][k]*g_y*-(x[1] - wave_means[j][1]);

                    // Wave coefficients gradient update
                    grad_wave_coefficients[j][0] += y[k]*g*cx*cy*wave_colors[j][k];
                    grad_wave_coefficients[j][1] += y[k]*g*cx*sy*wave_colors[j][k];
                    grad_wave_coefficients[j][2] += y[k]*g*sx*cy*wave_colors[j][k];
                    grad_wave_coefficients[j][3] += y[k]*g*sx*sy*wave_colors[j][k];
                    grad_wave_coefficients[j][4] += y[k]*g*wave_colors[j][k];

                    // Wave frequency gradient update
                    grad_wave_frequencies[j][0] +=
                        y[k]*g*TWO_PI*x[0]*wave_colors[j][k]*(
                            wave_coefficients[j][0]*cy*-sx +
                            wave_coefficients[j][1]*sy*-sx +
                            wave_coefficients[j][2]*cy*cx +
                            wave_coefficients[j][3]*sy*cx
                        );
                    grad_wave_frequencies[j][1] +=
                        y[k]*g*TWO_PI*x[1]*wave_colors[j][k]*(
                            wave_coefficients[j][0]*cx*-sy +
                            wave_coefficients[j][1]*cx*cy +
                            wave_coefficients[j][2]*sx*-sy +
                            wave_coefficients[j][3]*sx*cy
                        );
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
    const int threads = 512;
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    const int blocks = numSMs*32;

    // Dispatch jobs
    AT_DISPATCH_FLOATING_TYPES(input.type(), "gaussian_cuda_forward", ([&] {
    pointwise_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
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
        const int threads = 512;
        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
        const int blocks = numSMs*32;

        // Dispatch jobs
        AT_DISPATCH_FLOATING_TYPES(input.type(), "gaussian_cuda_backward", ([&] {
        pointwise_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
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
