#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace{

    
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

        // Params
        const int num_channels = output.size(1);
        const int num_gaussians = gaussian_colors.size(0);
        const int num_waves = wave_colors.size(0);
        const int batch_size = input.size(0);
        
        // Temp values
        float temp_result[3];
        int i, j, idx;
            
        // Declare shared floats for later
        __shared__ float R[512], G[512], B[512];
        __shared__ float mx[512], my[512];
        __shared__ float cov00[512], cov01[512], cov10[512], cov11[512];
        __shared__ float w_coeff1[512], w_coeff2[512], w_coeff3[512], w_coeff4[512], w_coeff5[512];
        __shared__ float w_freqx[512], w_freqy[512];

        // (x,y) input coordinate
        float x, y;

        // Iterate over the query points this thread is responsible for
        for(i = index; i < 512*(1+batch_size/512); i += stride){

            // Only get data if we are in bounds
            if(i < batch_size){
                temp_result[0] = 0.0f;
                temp_result[1] = 0.0f;
                temp_result[2] = 0.0f;
                x = input[i][0];
                y = input[i][1];
            }

            // Loop over gaussians
            for(j = 0; j < num_gaussians; j++){  
                // Load shared memory every block_size (512 threads)
                if(j % 512 == 0){
                    __syncthreads();  
                    // Need to do this IF check inside because the 
                    // threads need to be synced even if they are out of bounds.
                    if(j + threadIdx.x < num_gaussians){
                        mx[threadIdx.x] = gaussian_means[j + threadIdx.x][0];
                        my[threadIdx.x] = gaussian_means[j + threadIdx.x][1];
                        cov00[threadIdx.x] = gaussian_mats[j + threadIdx.x][0][0];
                        cov01[threadIdx.x] = gaussian_mats[j + threadIdx.x][0][1];
                        cov10[threadIdx.x] = gaussian_mats[j + threadIdx.x][1][0];
                        cov11[threadIdx.x] = gaussian_mats[j + threadIdx.x][1][1];
                        R[threadIdx.x] = gaussian_colors[j + threadIdx.x][0];
                        G[threadIdx.x] = gaussian_colors[j + threadIdx.x][1];
                        B[threadIdx.x] = gaussian_colors[j + threadIdx.x][2];
                    }
                    __syncthreads();
                }
                if(i < batch_size){
                    idx = j % 512;

                    const float g_x = (x - mx[idx]) * cov00[idx] +
                            (y - my[idx]) * cov10[idx];
                    const float g_y = (x - mx[idx]) * cov01[idx] + 
                            (y - my[idx]) * cov11[idx];
                    const float g = expf(-(g_x * g_x + g_y * g_y) / 2.0f);
                    if(g<0.0000001f) continue; 
                    temp_result[0] += g*R[idx];
                    temp_result[1] += g*G[idx];
                    temp_result[2] += g*B[idx];
                }
            }

            // Loop over waves
            for(j = 0; j < num_waves; j++){
                if(j % 512 == 0){
                    __syncthreads();  
                    // Need to do this IF check inside because the 
                    // threads need to be synced even if they are out of bounds.
                    if(j + threadIdx.x < num_waves){
                        mx[threadIdx.x] = wave_means[j + threadIdx.x][0];
                        my[threadIdx.x] = wave_means[j + threadIdx.x][1];
                        cov00[threadIdx.x] = wave_mats[j + threadIdx.x][0][0];
                        cov01[threadIdx.x] = wave_mats[j + threadIdx.x][0][1];
                        cov10[threadIdx.x] = wave_mats[j + threadIdx.x][1][0];
                        cov11[threadIdx.x] = wave_mats[j + threadIdx.x][1][1];
                        R[threadIdx.x] = wave_colors[j + threadIdx.x][0];
                        G[threadIdx.x] = wave_colors[j + threadIdx.x][1];
                        B[threadIdx.x] = wave_colors[j + threadIdx.x][2];      
                        w_coeff1[threadIdx.x] = wave_coefficients[j + threadIdx.x][0];
                        w_coeff2[threadIdx.x] = wave_coefficients[j + threadIdx.x][1];
                        w_coeff3[threadIdx.x] = wave_coefficients[j + threadIdx.x][2];
                        w_coeff4[threadIdx.x] = wave_coefficients[j + threadIdx.x][3];
                        w_coeff5[threadIdx.x] = wave_coefficients[j + threadIdx.x][4];
                        w_freqx[threadIdx.x] = wave_frequencies[j + threadIdx.x][0];
                        w_freqy[threadIdx.x] = wave_frequencies[j + threadIdx.x][1];
                    }
                    __syncthreads();
                }
                if(i < batch_size){
                    idx = j % 512;
                    const float g_x = (x - mx[idx]) * cov00[idx] +
                                        (y - my[idx]) * cov10[idx];
                    const float g_y = (x - mx[idx]) * cov01[idx] + 
                                        (y - my[idx]) * cov11[idx];
                    const float g = expf(-(g_x * g_x + g_y * g_y) / 2.0f);                     
                    if(g<0.0000001f) continue; 
                    const float sx = sinpif(2.0f*x*w_freqx[idx]);
                    const float sy = sinpif(2.0f*y*w_freqy[idx]);
                    const float cx = cospif(2.0f*x*w_freqx[idx]);
                    const float cy = cospif(2.0f*y*w_freqy[idx]);
                    const float w = w_coeff1[idx]*cx*cy +
                                    w_coeff2[idx]*cx*sy +
                                    w_coeff3[idx]*sx*cy +
                                    w_coeff4[idx]*sx*sy +
                                    w_coeff5[idx];         
                    if(abs(w)<0.0000001f) continue;  
                    temp_result[0] += g*w*R[idx]; 
                    temp_result[1] += g*w*G[idx]; 
                    temp_result[2] += g*w*B[idx]; 
                }
            }
            
            if(i < batch_size){
                output[i][0] = temp_result[0]; 
                output[i][1] = temp_result[1]; 
                output[i][2] = temp_result[2];
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

        // Params
        const int num_channels = grad_output.size(1);
        const int num_gaussians = gaussian_colors.size(0);
        const int num_waves = wave_colors.size(0);
        const int batch_size = input.size(0);
        
        // Temp values
        int i, j, k, idx;

        // Declare shared floats for later
        __shared__ float RGB[512*3];
        __shared__ float mx[512], my[512];
        __shared__ float cov00[512], cov01[512], cov10[512], cov11[512];
        __shared__ float w_coeff1[512], w_coeff2[512], w_coeff3[512], w_coeff4[512], w_coeff5[512];
        __shared__ float w_freqx[512], w_freqy[512];

        // (x,y) input coordinate
        float x, y;
        float dRGB[3];

        for(i = index; i < 512*(1+batch_size/512); i+= stride){
            if(i < batch_size){
                x = input[i][0];
                y = input[i][1];
                dRGB[0] = grad_output[i][0];
                dRGB[1] = grad_output[i][1];
                dRGB[2] = grad_output[i][2];
            }

            for(j = 0; j < num_gaussians; j++){
                // Load shared memory every block_size (512 threads)
                if(j % 512 == 0){
                    __syncthreads();  
                    // Need to do this IF check inside because the 
                    // threads need to be synced even if they are out of bounds.
                    if(j + threadIdx.x < num_gaussians){
                        mx[threadIdx.x] = gaussian_means[j + threadIdx.x][0];
                        my[threadIdx.x] = gaussian_means[j + threadIdx.x][1];
                        cov00[threadIdx.x] = gaussian_mats[j + threadIdx.x][0][0];
                        cov01[threadIdx.x] = gaussian_mats[j + threadIdx.x][0][1];
                        cov10[threadIdx.x] = gaussian_mats[j + threadIdx.x][1][0];
                        cov11[threadIdx.x] = gaussian_mats[j + threadIdx.x][1][1];
                        RGB[3*threadIdx.x] = gaussian_colors[j + threadIdx.x][0];
                        RGB[3*threadIdx.x+1] = gaussian_colors[j + threadIdx.x][1];
                        RGB[3*threadIdx.x+2] = gaussian_colors[j + threadIdx.x][2];
                    }
                    __syncthreads();
                }
                if(i < batch_size){
                    idx = j % 512;
                    const float g_x = (x - mx[idx]) * cov00[idx] +
                            (y - my[idx]) * cov10[idx];
                    const float g_y = (x - mx[idx]) * cov01[idx] + 
                            (y - my[idx]) * cov11[idx];
                    const float g = expf(-(g_x * g_x + g_y * g_y) / 2.0f);
                    if(g<0.0000001f) continue; 
                    for (k = 0; k < num_channels; k++){ 
                        if(abs(dRGB[k]) < 0.00000001f) continue;

                        // Gaussian color gradient update
                        grad_gaussian_colors[j][k] += dRGB[k]*g; 

                        // Gaussian position gradient update
                        grad_gaussian_means[j][0] += 
                            dRGB[k]*g*RGB[idx*3+k]*
                            (g_x*cov00[idx]+g_y*cov01[idx]);
                        grad_gaussian_means[j][1] += 
                            dRGB[k]*g*RGB[idx*3+k]*
                            (g_x*cov10[idx]+g_y*cov11[idx]);
                            
                        // Gaussian covariance matrix update
                        grad_gaussian_mats[j][0][0] +=
                            dRGB[k]*g*RGB[idx*3+k]*g_x*-(x - mx[idx]);
                        grad_gaussian_mats[j][0][1] +=
                            dRGB[k]*g*RGB[idx*3+k]*g_y*-(x - mx[idx]);
                        grad_gaussian_mats[j][1][0] +=
                            dRGB[k]*g*RGB[idx*3+k]*g_x*-(y - my[idx]);
                        grad_gaussian_mats[j][1][1] +=
                            dRGB[k]*g*RGB[idx*3+k]*g_y*-(y - my[idx]);
                    }
                }
            }
            for(j = 0; j < num_waves; j++){
                if(j % 512 == 0){
                    __syncthreads();  
                    // Need to do this IF check inside because the 
                    // threads need to be synced even if they are out of bounds.
                    if(j + threadIdx.x < num_waves){
                        mx[threadIdx.x] = wave_means[j + threadIdx.x][0];
                        my[threadIdx.x] = wave_means[j + threadIdx.x][1];
                        cov00[threadIdx.x] = wave_mats[j + threadIdx.x][0][0];
                        cov01[threadIdx.x] = wave_mats[j + threadIdx.x][0][1];
                        cov10[threadIdx.x] = wave_mats[j + threadIdx.x][1][0];
                        cov11[threadIdx.x] = wave_mats[j + threadIdx.x][1][1];
                        R[threadIdx.x] = wave_colors[j + threadIdx.x][0];
                        G[threadIdx.x] = wave_colors[j + threadIdx.x][1];
                        B[threadIdx.x] = wave_colors[j + threadIdx.x][2];      
                        w_coeff1[threadIdx.x] = wave_coefficients[j + threadIdx.x][0];
                        w_coeff2[threadIdx.x] = wave_coefficients[j + threadIdx.x][1];
                        w_coeff3[threadIdx.x] = wave_coefficients[j + threadIdx.x][2];
                        w_coeff4[threadIdx.x] = wave_coefficients[j + threadIdx.x][3];
                        w_coeff5[threadIdx.x] = wave_coefficients[j + threadIdx.x][4];
                        w_freqx[threadIdx.x] = wave_frequencies[j + threadIdx.x][0];
                        w_freqy[threadIdx.x] = wave_frequencies[j + threadIdx.x][1];
                    }
                    __syncthreads();
                }
                if(i < batch_size){
                    idx = j % 512;
                    const float g_x = (x - mx[idx]) * cov00[idx] +
                                (y - my[idx]) * cov10[idx];
                    const float g_y = (x - mx[idx]) * cov01[idx] + 
                                (y - my[idx]) * cov11[idx];
                    const float g = expf(-(g_x * g_x + g_y * g_y) / 2.0f);  
                    if(g<0.0000001f) continue; 
                    const float sx = sinpif(2.0f*x*w_freqx[idx]);
                    const float sy = sinpif(2.0f*y*w_freqy[idx]);
                    const float cx = cospif(2.0f*x*w_freqx[idx]);
                    const float cy = cospif(2.0f*y*w_freqy[idx]);
                    const float w = w_coeff1[idx]*cx*cy +
                        w_coeff2[idx]*cx*sy +
                        w_coeff3[idx]*sx*cy +
                        w_coeff4[idx]*sx*sy +
                        w_coeff5[idx]; 
                    for (k = 0; k < num_channels; k++){ 
                        if(abs(dRGB[k]) < 0.00000001f) continue;
                        // Wave color gradient update
                        grad_wave_colors[j][k] += dRGB[k]*g*w; 
                        
                        // Wave position gradient update
                        grad_wave_means[j][0] +=
                            dRGB[k]*w*g*RGB[3*idx+k]*
                            (g_x*cov00[idx]+g_y*cov01[idx]);
                        grad_wave_means[j][1] +=
                            dRGB[k]*w*g*RGB[3*idx+k]*
                            (g_x*cov10[idx]+g_y*cov11[idx]);
                            
                        // Wave covariance matrix gradient update
                        grad_wave_mats[j][0][0] += 
                            dRGB[k]*w*g*RGB[3*idx+k]*g_x*-(x - mx[idx]);
                        grad_wave_mats[j][0][1] += 
                            dRGB[k]*w*g*RGB[3*idx+k]*g_y*-(x - mx[idx]);
                        grad_wave_mats[j][1][0] += 
                            dRGB[k]*w*g*RGB[3*idx+k]*g_x*-(y - my[idx]);
                        grad_wave_mats[j][1][1] += 
                            dRGB[k]*w*g*RGB[3*idx+k]*g_y*-(y - my[idx]);

                        // Wave coefficients gradient update
                        grad_wave_coefficients[j][0] += dRGB[k]*g*cx*cy*RGB[3*idx+k];
                        grad_wave_coefficients[j][1] += dRGB[k]*g*cx*sy*RGB[3*idx+k];
                        grad_wave_coefficients[j][2] += dRGB[k]*g*sx*cy*RGB[3*idx+k];
                        grad_wave_coefficients[j][3] += dRGB[k]*g*sx*sy*RGB[3*idx+k];
                        grad_wave_coefficients[j][4] += dRGB[k]*g*RGB[3*idx+k];

                        // Wave frequency gradient update
                        grad_wave_frequencies[j][0] +=
                            dRGB[k]*g*2.0f*3.1415926f*x*RGB[3*idx+k]*(
                                w_coeff1[idx]*cy*-sx +
                                w_coeff2[idx]*sy*-sx +
                                w_coeff3[idx]*cy*cx +
                                w_coeff4[idx]*sy*cx
                            );
                        grad_wave_frequencies[j][1] +=
                            dRGB[k]*g*2.0f*3.1415926f*y*RGB[3*idx+k]*(
                                w_coeff1[idx]*cx*-sy +
                                w_coeff2[idx]*cx*cy +
                                w_coeff3[idx]*sx*-sy +
                                w_coeff4[idx]*sx*cy
                            );
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
    const int threads = 512;
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    const int blocks = (batch_size+threads-1)/threads;

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
        const int blocks = (batch_size+threads-1)/threads;

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
