#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define NUM_THREADS 32
#define SELECTED_NUM_FREQUENCIES 16
#define NUM_CHANNELS 3
#define NUM_DIMENSIONS 2

__device__ float3 operator*(const float a, const float3 &b) {
    return make_float3(a*b.x, a*b.y, a*b.z);
}

__device__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ void operator+=(float3 &a, const float3 &b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__device__ float2 operator-(const float2 &a, const float2 &b) {
    return make_float2(a.x-b.x, a.y-b.y);
}

__device__ float2 operator-=(float2 &a, const float2 &b) {
    a.x -= b.x;
    a.y -= b.y;
}


namespace{

    __global__ void top_k_frequencies_cuda(  
        int NUM_PRIMITIVES,
        int NUM_FREQUENCIES,
        const float* __restrict__ wave_coefficients,
        float* __restrict__ top_wave_coefficient_values,
        int* __restrict__ top_wave_coefficient_indices
        ) {

        // Get block/thread related numbers   
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
                    
        // Declare shared floats for later
        __shared__ float coeffs[NUM_THREADS];
        //__shared__ float c[NUM_THREADS][NUM_DIMENSIONS][SELECTED_NUM_FREQUENCIES];
        
        // Iterate over the primitives this is responsible for
        for(int i = index; i < NUM_PRIMITIVES; i += stride){
            for(int j = 0; j < NUM_DIMENSIONS; j++){

                float topk[SELECTED_NUM_FREQUENCIES];
                int topk_indices[SELECTED_NUM_FREQUENCIES];

                int start_ind = i*NUM_DIMENSIONS*NUM_FREQUENCIES + j*NUM_FREQUENCIES;
                float minVal = 0;
                float maxVal = wave_coefficients[start_ind];
                topk[0] = wave_coefficients[start_ind];
                topk_indices[0] = 0;

                for(int k = 0; k < NUM_FREQUENCIES; k++){
                    int idx = start_ind + k;
                    float v = wave_coefficients[idx];
                    if(fabsf(v) > minVal){
                        int minIdx = 0;
                        
                        for (int m = 1; m < SELECTED_NUM_FREQUENCIES; m++){

                        }
                    }
                }
            }
        }
    
    }

    __global__ void periodic_primitives_forward_cuda_kernel(  
        int NUM_PRIMITIVES,
        int BATCH_SIZE,   
        int NUM_FREQUENCIES,   
        float MAX_FREQUENCY,
        const float* __restrict__ input,
        const float* __restrict__ colors,
        const float* __restrict__ positions,
        const float* __restrict__ scales,
        const float* __restrict__ rotations,
        const float* __restrict__ wave_coefficients,
        const int* __restrict__ wave_coefficient_indices,
        float* __restrict__ output
        ) {

        // Get block/thread related numbers   
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
                    
        // Declare shared floats for later
        __shared__ float3 RGB[NUM_THREADS];
        __shared__ float2 p[NUM_THREADS];
        __shared__ float2 s[NUM_THREADS];
        __shared__ float r[NUM_THREADS];
        //__shared__ float c[NUM_THREADS][NUM_DIMENSIONS][SELECTED_NUM_FREQUENCIES];
        


        // Iterate over the query points this thread is responsible for
        for(int i = index; i < NUM_THREADS*(1+BATCH_SIZE/NUM_THREADS); i += stride){
            
            float3 temp_result = { 0.0f, 0.0f, 0.0f };
            
            // Loop over primitives
            for(int j = 0; j < NUM_PRIMITIVES; j++){  

                // Load shared memory every block_size (NUM_THREADS threads)
                // All threads do this even if theyre beyond the batch size
                if(j % NUM_THREADS == 0){
                    __syncthreads();  
                    // Need to do this IF check inside because the 
                    // threads need to be synced even if they are out of bounds.
                    if(j + threadIdx.x < NUM_PRIMITIVES){
                        p[threadIdx.x] = { positions[2*(j+threadIdx.x)], positions[2*(j+threadIdx.x) + 1] };
                        s[threadIdx.x] = { scales[2*(j+threadIdx.x)], scales[2*(j+threadIdx.x) + 1] };
                        r[threadIdx.x] = rotations[j+threadIdx.x];
                        RGB[threadIdx.x] = { colors[3*(j+threadIdx.x)], colors[3*(j+threadIdx.x) + 1], colors[3*(j+threadIdx.x) + 2] };
                        //memcpy(&c[threadIdx.x], &wave_coefficients[j+threadIdx.x], 
                        //    NUM_DIMENSIONS*NUM_FREQUENCIES*sizeof(float));
                    }
                    __syncthreads();
                }
                
                // Threads within the batch size process the output using the current shared memory
                if(i < BATCH_SIZE){
                    int idx = j % NUM_THREADS;
                    float2 x = { input[2*i], input[2*i+1] };

                    // Get the gaussian weight for this primitive
                    x -= p[idx];
                    float cosr = cosf(r[idx]);
                    float sinr = sinf(r[idx]);
                    float2 tx = make_float2(s[idx].x*(x.x*cosr  + x.y*sinr),
                                            s[idx].y*(x.x*-sinr + x.y*cosr));
                    float g = expf(-(tx.x*tx.x + tx.y*tx.y) / 2.0f);

                    // Update local result
                    temp_result += g*RGB[idx];
                }
            }

            // Update global memory with the final result
            if(i < BATCH_SIZE){
                output[i*3] = temp_result.x;
                output[i*3+1] = temp_result.y;
                output[i*3+2] = temp_result.z;
            }
        }
    }

    __global__ void periodic_primitives_backward_cuda_kernel(
        const int NUM_PRIMITIVES,
        const int BATCH_SIZE, 
        const int num_freq_backward, 
        const int total_num_freqs, 
        const float MAX_FREQUENCY, 
        const float* __restrict__ grad_output,
        const float* __restrict__ input,
        const float* __restrict__ colors,
        const float* __restrict__ positions,
        const float* __restrict__ scales,
        const float* __restrict__ rotations,
        const float* __restrict__ wave_coefficients,
        const int* __restrict__ wave_coefficient_indices,
        float* __restrict__ dColors,
        float* __restrict__ dPositions,
        float* __restrict__ dScales,
        float* __restrict__ dRotations,
        float* __restrict__ dCoefficients
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
    const float MAX_FREQUENCY
    ) {

    // Get sizes for the output
    const int batch_size = input.size(0);
    const int num_primitives = colors.size(0);
    const int num_frequencies = wave_coefficients.size(2);

    // Create output tensor
    auto output = torch::zeros({batch_size, NUM_CHANNELS}, input.device());
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(input.device());
    auto wave_coefficient_indices = torch::zeros(
        {num_primitives, NUM_DIMENSIONS, SELECTED_NUM_FREQUENCIES},
        options);

    // If we want to adjust the number of blocks, this is a good way
    // Scales with the number of SMs on the GPU
    //int numSMs;
    //cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    // First get the top K frequencies from each waveform
    // Faster to launch a different kernel to parallelize over gaussians

    
    periodic_primitives_forward_cuda_kernel<<<(batch_size+NUM_THREADS-1)/NUM_THREADS, NUM_THREADS>>>(
        num_primitives,
        batch_size,
        num_frequencies,
        MAX_FREQUENCY,
        input.contiguous().data<float>(),
        colors.contiguous().data<float>(),
        positions.contiguous().data<float>(),
        scales.contiguous().data<float>(),
        rotations.contiguous().data<float>(),
        wave_coefficients.contiguous().data<float>(),
        wave_coefficient_indices.contiguous().data<int>(),
        output.contiguous().data<float>()
        );

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
        const int total_num_frequencies = wave_coefficients.size(2);
        const int num_frequencies_backward = wave_coefficient_indices.size(2);

        // Set up gradient tensors
        auto dColors = torch::zeros_like(colors);
        auto dPositions = torch::zeros_like(positions); 
        auto dScales = torch::zeros_like(scales); 
        auto dRotations = torch::zeros_like(rotations); 
        auto dCoefficients = torch::zeros_like(wave_coefficients);
        
        //int numSMs;
        //cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
        const int blocks = (batch_size+NUM_THREADS-1)/NUM_THREADS;

        // Dispatch jobs
        periodic_primitives_backward_cuda_kernel<<<blocks, NUM_THREADS>>>(
            num_primitives,
            batch_size,
            num_frequencies_backward,
            total_num_frequencies,
            MAX_FREQUENCY,
            grad_output.contiguous().data<float>(),
            input.contiguous().data<float>(),
            colors.contiguous().data<float>(),
            positions.contiguous().data<float>(),
            scales.contiguous().data<float>(),
            rotations.contiguous().data<float>(),
            wave_coefficients.contiguous().data<float>(),
            wave_coefficient_indices.contiguous().data<int>(),
            dColors.contiguous().data<float>(),
            dPositions.contiguous().data<float>(),
            dScales.contiguous().data<float>(),
            dRotations.contiguous().data<float>(),
            dCoefficients.contiguous().data<float>()
            );
    
        return {dColors, dPositions, dScales, dRotations, dCoefficients };
    }
