#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define NUM_THREADS 1024
#define TOTAL_NUM_FREQUENCIES 1024
#define SELECTED_NUM_FREQUENCIES 16
#define NUM_CHANNELS 3
#define NUM_DIMENSIONS 2
#define BLOCKS_X 16
#define BLOCKS_Y 16


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


__device__ __forceinline__ int get256bitOffset(const uint32_t bits[8], const int offset){
    int idx = offset / 32;
    int shift = offset % 32;
    return ((bits[idx]>>shift)&1) == 1 ? 1 : 0;    
}

__device__ __forceinline__ void set256bitOffset(uint32_t bits[8], const int offset){
    int idx = offset / 32;
    int shift = offset % 32;
    //bits[idx] = bits[idx] | ((uint32_t)1 << offset)
}


namespace{

    __global__ void periodic_primitives_forward_cuda_kernel_old(  
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
        
        float2 x;
        // Iterate over the query points this thread is responsible for
        for(int i = index; i < NUM_THREADS*(1+BATCH_SIZE/NUM_THREADS); i += stride){
            
            float3 temp_result = { 0.0f, 0.0f, 0.0f };
            if(i<BATCH_SIZE) x = {input[2*i], input[2*i+1]};

            // Loop over primitives
            for(int j = 0; j < NUM_PRIMITIVES; j++){  

                // Load shared memory every block_size (NUM_THREADS threads)
                // All threads do this even if theyre beyond the batch size
                if(j % NUM_THREADS == 0){
                    // Need to do this IF check inside because the 
                    // threads need to be synced even if they are out of bounds.
                    __syncthreads();
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
                if( i >= BATCH_SIZE) continue;

                int idx = j % NUM_THREADS;

                // Get the gaussian weight for this primitive
                float2 px = x - p[idx];
                float cosr = __cosf(r[idx]);
                float sinr = __sinf(r[idx]);
                float2 tx = { s[idx].x*(px.x*cosr  + px.y*sinr),
                            s[idx].y*(px.x*-sinr + px.y*cosr) };
                float g = __expf(-(tx.x*tx.x + tx.y*tx.y)/2);
                if(g < 0.00000001f) continue;

                // Update local result
                temp_result += g*RGB[idx];
                
            }

            // Update global memory with the final result
            if(i < BATCH_SIZE){
                atomicAdd(&output[i*3], temp_result.x);
                atomicAdd(&output[i*3+1], temp_result.y);
                atomicAdd(&output[i*3+2], temp_result.z);
            }
        }
    }

    
    __global__ void preprocess_gaussians(  
        int NUM_PRIMITIVES,
        float min_x, float min_y, 
        float range_x, float range_y,
        const float* __restrict__ positions,
        const float* __restrict__ scales,
        uint32_t* __restrict__ gaussian_blocks
        ) {
            // Get block/thread related numbers   
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = blockDim.x * gridDim.x;
            for(int i = index; i < NUM_PRIMITIVES; i += stride){
                uint32_t block_values[8] = {(uint32_t)0,(uint32_t)0,(uint32_t)0,(uint32_t)0,
                                            (uint32_t)0,(uint32_t)0,(uint32_t)0,(uint32_t)0};
                float sx = scales[2*i];
                float sy = scales[2*i+1];
                float px = positions[2*i];
                float py = positions[2*i+1];
                float r = 3.0f/min(sx, sy);

                float x_min = px - r;
                float x_max = px + r;
                float y_min = py - r;
                float y_max = py + r;
                x_min -= min_x;
                x_max -= min_x;
                x_min /= range_x;
                x_max /= range_x;
                y_min -= min_y;
                y_max -= min_y;
                y_min /= range_y;
                y_max /= range_y;
                int x_block_min = x_min*BLOCKS_X;
                int x_block_max = x_max*BLOCKS_X;
                int y_block_min = y_min*BLOCKS_Y;
                int y_block_max = y_max*BLOCKS_Y;
                if(x_block_min < 0) x_block_min = 0;
                if(y_block_min < 0) y_block_min = 0;
                for (int x = x_block_min; x <= x_block_max; x++){
                    for (int y = y_block_min; y <= y_block_max; y++){
                        set256bitOffset(block_values, y*BLOCKS_X+x);
                    }
                }
                gaussian_blocks[8*i] = block_values[0];
                gaussian_blocks[8*i+1] = block_values[1];
                gaussian_blocks[8*i+2] = block_values[2];
                gaussian_blocks[8*i+3] = block_values[3];
                gaussian_blocks[8*i+4] = block_values[4];
                gaussian_blocks[8*i+5] = block_values[5];
                gaussian_blocks[8*i+6] = block_values[6];
                gaussian_blocks[8*i+7] = block_values[7];
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
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_to_process = min(NUM_THREADS, NUM_PRIMITIVES - blockIdx.y * blockDim.y);
        const int j = blockIdx.y * blockDim.x + threadIdx.x ;


        // Declare shared floats
        __shared__ float RGB[NUM_THREADS*3];
        __shared__ float p[NUM_THREADS*2];
        __shared__ float s[NUM_THREADS*2];
        __shared__ float r[NUM_THREADS];
        __shared__ float c[NUM_THREADS*NUM_DIMENSIONS*SELECTED_NUM_FREQUENCIES];
        
        if(j < NUM_PRIMITIVES){
            p[2*threadIdx.x] = positions[2*j];
            p[2*threadIdx.x+1] = positions[2*j+1];
            s[2*threadIdx.x] = scales[2*j];
            s[2*threadIdx.x+1] = scales[2*j+1];
            r[threadIdx.x] = rotations[j];
            RGB[3*threadIdx.x] = colors[3*j];
            RGB[3*threadIdx.x+1] = colors[3*j+1];
            RGB[3*threadIdx.x+2] = colors[3*j+2];
            //memcpy(&c[threadIdx.x], &wave_coefficients[j+threadIdx.x], 
            //    NUM_DIMENSIONS*NUM_FREQUENCIES*sizeof(float));
        }
    
        __syncthreads();
        if(i >= BATCH_SIZE) return;

        float2 x = {input[2*i], input[2*i+1]};
        float3 temp_result = { 0.0f, 0.0f, 0.0f };
        

        for(int idx = 0; idx < num_to_process; idx++){
            float2 pos = {p[2*idx], p[2*idx+1] };
            float2 scale = { s[2*idx], s[2*idx+1]};
            float rotation = r[idx];
            float3 color = {RGB[3*idx], RGB[3*idx+1], RGB[3*idx+2]};
            //temp_result.x += color.x + rotation + scale.x + pos.x;
            //temp_result.y += color.y + rotation + scale.y + pos.y;
            //temp_result.z += color.z;
            //continue;
            float2 dx = x - pos;
            
            float cosr = __cosf(rotation);
            float sinr = __sinf(rotation);
            float2 tx = { scale.x*(dx.x*cosr  + dx.y*sinr),
                        scale.y*(dx.x*-sinr + dx.y*cosr) };
            float g = __expf(-(tx.x*tx.x + tx.y*tx.y)/2);

            temp_result += g*color;
        }
        atomicAdd(&output[i*3], temp_result.x);
        atomicAdd(&output[i*3+1], temp_result.y);
        atomicAdd(&output[i*3+2], temp_result.z);
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
    torch::Tensor wave_coefficient_indices,    // [M, n_dim, n_freqs]
    const float min_x, const float max_x, 
    const float min_y, const float max_y,
    const float MAX_FREQUENCY
    ) {

    // Get sizes for the output
    const int batch_size = input.size(0);
    const int num_primitives = colors.size(0);
    const int num_frequencies = wave_coefficients.size(2);

    // Create output tensor and other tensors
    auto output = torch::zeros({batch_size, NUM_CHANNELS}, input.device());
    uint32_t *gaussian_blocks;
    unsigned long long bytes_needed = (8ull)*num_primitives*sizeof(uint32_t);
    cudaError_t status = cudaMalloc((void**)&gaussian_blocks, bytes_needed);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
        return {output};
    }
    // If we want to adjust the number of blocks, this is a good way
    // Scales with the number of SMs on the GPU
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    
    
    preprocess_gaussians<<<(num_primitives+NUM_THREADS-1)/NUM_THREADS,NUM_THREADS>>>(  
        num_primitives,
        min_x, min_y,
        max_x - min_x + 0.0000001f, max_y - min_y + 0.0000001f,
        positions.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        gaussian_blocks
        );

    // Now parallelize over query points
    dim3 numBlocks ((batch_size+NUM_THREADS-1)/NUM_THREADS, (num_primitives+NUM_THREADS-1)/NUM_THREADS);
    periodic_primitives_forward_cuda_kernel<<<numBlocks, NUM_THREADS>>>(
        num_primitives,
        batch_size,
        num_frequencies,
        MAX_FREQUENCY,
        input.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        positions.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        wave_coefficients.contiguous().data_ptr<float>(),
        wave_coefficient_indices.contiguous().data_ptr<int>(),
        //radii.contiguous().data_ptr<float>(),
        output.contiguous().data_ptr<float>()
        );
        
    cudaFree(gaussian_blocks);
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
            grad_output.contiguous().data_ptr<float>(),
            input.contiguous().data_ptr<float>(),
            colors.contiguous().data_ptr<float>(),
            positions.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(),
            rotations.contiguous().data_ptr<float>(),
            wave_coefficients.contiguous().data_ptr<float>(),
            wave_coefficient_indices.contiguous().data_ptr<int>(),
            dColors.contiguous().data_ptr<float>(),
            dPositions.contiguous().data_ptr<float>(),
            dScales.contiguous().data_ptr<float>(),
            dRotations.contiguous().data_ptr<float>(),
            dCoefficients.contiguous().data_ptr<float>()
            );
    
        return {dColors, dPositions, dScales, dRotations, dCoefficients };
    }
