#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define NUM_THREADS 32
#define SELECTED_NUM_FREQUENCIES 16
#define NUM_CHANNELS 3
#define NUM_DIMENSIONS 2

namespace{

    __global__ void periodic_primitives_forward_cuda_kernel(  
        int NUM_PRIMITIVES,
        int BATCH_SIZE,   
        int NUM_FREQUENCIES,   
        float MAX_FREQUENCY,
        const float* input,
        const float* colors,
        const float* positions,
        const float* scales,
        const float* rotations,
        const float* wave_coefficients,
        const short* wave_coefficient_indices,
        float* output
        ) {

        // Get block/thread related numbers   
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        
        // Temp values
        float temp_result[NUM_CHANNELS];
        int i, j, k, m, idx;
            
        // Declare shared floats for later
        __shared__ float RGB[NUM_THREADS][NUM_CHANNELS];
        __shared__ float p[NUM_THREADS][NUM_DIMENSIONS];
        __shared__ float s[NUM_THREADS][NUM_DIMENSIONS];
        __shared__ float r[NUM_THREADS];
        __shared__ float c[NUM_THREADS][NUM_DIMENSIONS][NUM_FREQUENCIES];

        // (x,y) input coordinate
        float x[NUM_DIMENSIONS];
        float gx[NUM_DIMENSIONS], wx[NUM_DIMENSIONS];
        float g, gexp, w;
        
        /*
        // Iterate over the query points this thread is responsible for
        for(i = index; i < NUM_THREADS*(1+BATCH_SIZE/NUM_THREADS); i += stride){

            // Only get data if we are in bounds
            if(i < BATCH_SIZE){
                temp_result = { 0.0f };
                x = { 0.0f };
            }

            // Loop over primitives
            for(j = 0; j < NUM_PRIMITIVES; j++){  

                // Load shared memory every block_size (512 threads)
                // All threads do this even if theyre beyond the batch size
                if(j % NUM_THREADS == 0){
                    __syncthreads();  
                    // Need to do this IF check inside because the 
                    // threads need to be synced even if they are out of bounds.
                    if(j + threadIdx.x < NUM_PRIMITIVES){
                        cudaMemcpy(p[threadIdx.x], positions[j+threadIdx.x], 
                            NUM_DIMENSIONS*sizeof(float), cudaMemcpyHostToDevice);
                        cudaMemcpy(s[threadIdx.x], scales[j+threadIdx.x], 
                            NUM_DIMENSIONS*sizeof(float), cudaMemcpyHostToDevice);
                        cudaMemcpy(r[threadIdx.x], rotations[j+threadIdx.x], 
                            sizeof(float), cudaMemcpyHostToDevice);
                        cudaMemcpy(RGB[threadIdx.x], colors[j+threadIdx.x], 
                            NUM_CHANNELS*sizeof(float), cudaMemcpyHostToDevice);
                        cudaMemcpy(c[threadIdx.x], wave_coefficients[j+threadIdx.x], 
                            NUM_DIMENSIONS*NUM_FREQUENCIES*sizeof(float), cudaMemcpyHostToDevice);
                    }
                    __syncthreads();
                }

                // Threads within the batch size process the output using the current shared memory
                if(i < BATCH_SIZE){
                    idx = j % NUM_THREADS;
                    cudaMemcpy(x, input[i], 
                        NUM_DIMENSIONS*sizeof(float), cudaMemcpyHostToDevice);

                    // Get the gaussian weight for this primitive
                    // (x-p)cov^(-1/2) * cov(-1/2)(x-p)
                    g = 0.0f;
                    for(k = 0; k < NUM_DIMENSIONS; k++){
                        gx[k] = 0.0f;
                        for(m = 0; m < NUM_DIMENSIONS; m++){
                            gx[k] += (x[m]-p[idx][m])*covs[idx][m][k];
                        }
                        g += gx[k]*gx[k];
                    }
                    gexp = expf(-g / 2.0f);
                    if(gexp<0.00000001f) continue; 

                    // Calculate the waveform effect for this primitive
                    w = 1.0f;
                    for(k = 0; k < NUM_DIMENSIONS; k++) {
                        wx[k] = 0.0f;
                        for(m = 0; m < SELECTED_NUM_FREQUENCIES; m++){
                            wx[k] += coeffs[idx][k][m]*cosf(x[k]*(m*MAX_FREQUENCY)/SELECTED_NUM_FREQUENCIES);
                        }
                        w *= wx[k];
                    }
                    if(w<0.00000001f) continue;

                    // Update local result
                    for(k = 0; k < NUM_CHANNELS; k++) temp_result[k] += gexp*w*RGB[idx][k];
                }
            }

            // Update global memory with the final result
            if(i < BATCH_SIZE) for(k = 0; k < NUM_CHANNELS; k++) output[i][k] += g*temp_result[k];
        }
        */
    }

    
    template <typename scalar_t>
    __global__ void periodic_primitives_backward_cuda_kernel(
        const int NUM_PRIMITIVES,
        const int BATCH_SIZE, 
        const float MAX_FREQUENCY, 
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_output,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> colors,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> position,
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cov,
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> coefficients,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dColors,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dPositions,
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dCovs,
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dCoefficients
        ) 
    {
            return;
    }
    
}

std::vector<torch::Tensor> periodic_primitives_forward_cuda(
    torch::Tensor input,                        // [N, n_dim]
    torch::Tensor colors,                       // [M, n_chan]
    torch::Tensor positions,                    // [M, n_dim]
    torch::Tensor scales,                       // [M, n_dim]
    torch::Tensor rotations,                    // [M, 1]
    torch::Tensor wave_coefficients,            // [M, n_dim, n_freqs]
    torch::Tensor wave_coefficient_indices,     // [M, n_dim, n_freqs]
    const float MAX_FREQUENCY
    ) {

    // Get sizes for the output
    const int batch_size = input.size(0);
    const int num_primitives = colors.size(0);
    const int num_frequencies = coefficients.size(2);

    // Create output tensor
    auto output = torch::zeros({batch_size, NUM_CHANNELS}, input.device());

    // If we want to adjust the number of blocks, this is a good way
    // Scales with the number of SMs on the GPU
    //int numSMs;
    //cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    const int blocks = (batch_size+NUM_THREADS-1)/NUM_THREADS;

    // Dispatch jobs
    AT_DISPATCH_FLOATING_TYPES(input.type(), "periodic_primitives_cuda_forward", ([&] {
    periodic_primitives_forward_cuda_kernel<<<blocks, NUM_THREADS>>>(
        num_primitives,
        batch_size,
        MAX_FREQUENCY,
        input.contiguous().data<float>(),
        colors.contiguous().data<float>(),
        positions.contiguous().data<float>(),
        scales.contiguous().data<float>(),
        rotations.contiguous().data<float>(),
        wave_coefficients.contiguous().data<float>(),
        wave_coefficient_indices.contiguous().data<short>(),
        output.contiguous().data<float>()
        );
    }));

    return {output};
}


std::vector<torch::Tensor> periodic_primitives_backward_cuda(
    torch::Tensor grad_output,                  // [N, n_chan]
    torch::Tensor input,                        // [N, n_dim]
    torch::Tensor colors,                       // [M, n_chan]
    torch::Tensor positions,                    // [M, n_dim]
    torch::Tensor scales,                       // [M, n_dim, n_dim]
    torch::Tensor rotations,                    // [M, n_dim, n_freqs]
    torch::Tensor wave_coefficients,            // [M, n_dim, n_freqs]
    torch::Tensor wave_coefficient_indices,     // [M, n_dim, n_freqs]
    const float MAX_FREQUENCY
    ) {
        // Get sizes for the output
        const int batch_size = input.size(0);
        const int num_primitives = colors.size(0);
        const int num_frequencies = wave_coefficients.size(2);

        // Set up gradient tensors
        auto dColors = torch::zeros_like(colors);
        auto dPositions = torch::zeros_like(positions); 
        auto dScales = torch::zeros_like(scales); 
        auto dRotations = torch::zeros_like(rotations); 
        auto dCoefficients = torch::zeros_like(coefficients);
        
        //int numSMs;
        //cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
        const int blocks = (batch_size+NUM_THREADS-1)/NUM_THREADS;

        // Dispatch jobs
        AT_DISPATCH_FLOATING_TYPES(input.type(), "periodic_primitives_cuda_backward", ([&] {
        periodic_primitives_backward_cuda_kernel<scalar_t><<<blocks, NUM_THREADS>>>(
            num_primitives,
            batch_size,
            num_frequencies,
            MAX_FREQUENCY,
            grad_output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            colors.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            positions.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            scales.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            rotations.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            wave_coefficients.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            wave_coefficient_indices.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            dColors.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            dPositions.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            dScales.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            dRotations.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            dCoefficients.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>()
            );
        }));
    
        return {dColors, dPositions, dScales, dRotations, dCoefficients };
    }
