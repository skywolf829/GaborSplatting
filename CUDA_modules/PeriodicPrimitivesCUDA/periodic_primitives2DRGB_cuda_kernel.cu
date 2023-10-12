#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#define NUM_THREADS 128
#define TOTAL_NUM_FREQUENCIES 1024
#define SELECTED_NUM_FREQUENCIES 4
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

__device__ __forceinline__ bool get256bitOffset(const uint32_t bits[8], const int bitNo){
    int i = bitNo / 32;
    int shift = bitNo % 32;
    return ((bits[i]>>shift)&(uint32_t)1) == 1;    
}

__device__ __forceinline__ void set256bitOffset(uint32_t bits[8], const int bitNo){
    int i = bitNo / 32;
    int shift = bitNo % 32;
    bits[i] |= ((uint32_t)1 << shift);
}


namespace{

    __global__ void periodic_primitives_forward_cuda_kernel_pointwise(  
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
        __shared__ float c[NUM_THREADS][NUM_DIMENSIONS*SELECTED_NUM_FREQUENCIES];
        
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

    __global__ void periodic_primitives_forward_cuda_kernel_efficient(  
        int NUM_PRIMITIVES,
        int BATCH_SIZE,   
        int NUM_FREQUENCIES,   
        float MAX_FREQUENCY,
        const bool gaussian_only,
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
        const int num_to_process = min(NUM_THREADS, NUM_PRIMITIVES - blockIdx.y * blockDim.x);
        const int j = blockIdx.y * blockDim.x + threadIdx.x ;


        // Declare shared floats
        __shared__ float RGB[NUM_THREADS*3];
        __shared__ float p[NUM_THREADS*2];
        __shared__ float s[NUM_THREADS*2];
        __shared__ float r[NUM_THREADS];
        __shared__ float c[NUM_THREADS*NUM_DIMENSIONS*SELECTED_NUM_FREQUENCIES];
        __shared__ int c_idx[NUM_THREADS*NUM_DIMENSIONS*SELECTED_NUM_FREQUENCIES];
        
        if(j < NUM_PRIMITIVES){
            p[2*threadIdx.x] = positions[2*j];
            p[2*threadIdx.x+1] = positions[2*j+1];
            s[2*threadIdx.x] = scales[2*j];
            s[2*threadIdx.x+1] = scales[2*j+1];
            r[threadIdx.x] = rotations[j];
            RGB[3*threadIdx.x] = colors[3*j];
            RGB[3*threadIdx.x+1] = colors[3*j+1];
            RGB[3*threadIdx.x+2] = colors[3*j+2];
            for (int idx = 0; idx < NUM_DIMENSIONS*SELECTED_NUM_FREQUENCIES && !gaussian_only; idx++){
                c[threadIdx.x*NUM_DIMENSIONS*SELECTED_NUM_FREQUENCIES+idx] =
                    wave_coefficients[j*NUM_DIMENSIONS*SELECTED_NUM_FREQUENCIES + idx];
                c_idx[threadIdx.x*NUM_DIMENSIONS*SELECTED_NUM_FREQUENCIES+idx] =
                    wave_coefficient_indices[j*NUM_DIMENSIONS*SELECTED_NUM_FREQUENCIES + idx];
            }
        }
    
        __syncthreads();
        if(i >= BATCH_SIZE) return;

        float2 x = {input[2*i], input[2*i+1]};
        float3 temp_result = { 0.0f, 0.0f, 0.0f };
        

        for(int idx = 0; idx < num_to_process; idx++){
            // Get some necessary shared data first
            float2 pos = {p[2*idx], p[2*idx+1] };
            float2 scale = { s[2*idx], s[2*idx+1]};
            float2 dx = x - pos;

            float rotation = r[idx];
            float3 color = {RGB[3*idx], RGB[3*idx+1], RGB[3*idx+2]};            
            float cosr = __cosf(rotation);
            float sinr = __sinf(rotation);
            float2 tx = { scale.x*(dx.x*cosr  + dx.y*sinr),
                        scale.y*(dx.x*-sinr + dx.y*cosr) };
            float g = __expf(-0.5*(tx.x*tx.x + tx.y*tx.y));
            
            float wx = 0.0f, wy = 0.0f;
            for(int w_idx = 0; w_idx < SELECTED_NUM_FREQUENCIES && !gaussian_only; w_idx++){
                int spot = idx*SELECTED_NUM_FREQUENCIES*NUM_DIMENSIONS + w_idx*NUM_DIMENSIONS;
                float fx = MAX_FREQUENCY*c_idx[spot]/(float)TOTAL_NUM_FREQUENCIES;
                float fy = MAX_FREQUENCY*c_idx[spot+1]/(float)TOTAL_NUM_FREQUENCIES;
                wx += c[spot]*__cosf(fx*x.x);
                wy += c[spot+1]*__cosf(fy*x.y);
            }            
            if(!gaussian_only) g *= wx*wy;
            temp_result += g*color;
        }
        
        atomicAdd(&output[i*3], temp_result.x);
        atomicAdd(&output[i*3+1], temp_result.y);
        atomicAdd(&output[i*3+2], temp_result.z);
    }

    __global__ void periodic_primitives_heatmap_cuda_kernel(  
        int NUM_PRIMITIVES,
        int BATCH_SIZE,   
        int NUM_FREQUENCIES,   
        float MAX_FREQUENCY,
        const bool gaussian_only,
        const float* __restrict__ input,
        const float* __restrict__ positions,
        const float* __restrict__ scales,
        const float* __restrict__ rotations,
        const float* __restrict__ wave_coefficients,
        const int* __restrict__ wave_coefficient_indices,
        float* __restrict__ output
        ) {

        // Get block/thread related numbers   
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_to_process = min(NUM_THREADS, NUM_PRIMITIVES - blockIdx.y * blockDim.x);
        const int j = blockIdx.y * blockDim.x + threadIdx.x ;


        // Declare shared floats
        __shared__ float p[NUM_THREADS*2];
        __shared__ float s[NUM_THREADS*2];
        __shared__ float r[NUM_THREADS];
        __shared__ float c[NUM_THREADS*NUM_DIMENSIONS*SELECTED_NUM_FREQUENCIES];
        __shared__ int c_idx[NUM_THREADS*NUM_DIMENSIONS*SELECTED_NUM_FREQUENCIES];
        
        if(j < NUM_PRIMITIVES){
            p[2*threadIdx.x] = positions[2*j];
            p[2*threadIdx.x+1] = positions[2*j+1];
            s[2*threadIdx.x] = scales[2*j];
            s[2*threadIdx.x+1] = scales[2*j+1];
            r[threadIdx.x] = rotations[j];
            for (int idx = 0; idx < NUM_DIMENSIONS*SELECTED_NUM_FREQUENCIES && !gaussian_only; idx++){
                c[threadIdx.x*NUM_DIMENSIONS*SELECTED_NUM_FREQUENCIES+idx] =
                    wave_coefficients[j*NUM_DIMENSIONS*SELECTED_NUM_FREQUENCIES + idx];
                c_idx[threadIdx.x*NUM_DIMENSIONS*SELECTED_NUM_FREQUENCIES+idx] =
                    wave_coefficient_indices[j*NUM_DIMENSIONS*SELECTED_NUM_FREQUENCIES + idx];
            }
        }
    
        __syncthreads();
        if(i >= BATCH_SIZE) return;

        float2 x = {input[2*i], input[2*i+1]};
        float temp_result = 0.0f;
        

        for(int idx = 0; idx < num_to_process; idx++){
            // Get some necessary shared data first
            float2 pos = {p[2*idx], p[2*idx+1] };
            float2 scale = { s[2*idx], s[2*idx+1]};
            float2 dx = x - pos;
            float rotation = r[idx];
            float cosr = __cosf(rotation);
            float sinr = __sinf(rotation);
            float2 tx = { scale.x*(dx.x*cosr  + dx.y*sinr),
                        scale.y*(dx.x*-sinr + dx.y*cosr) };
            float g = __expf(-0.5*(tx.x*tx.x + tx.y*tx.y));
            
            float wx = 0.0f, wy = 0.0f;
            for(int w_idx = 0; w_idx < SELECTED_NUM_FREQUENCIES && !gaussian_only; w_idx++){
                int spot = idx*SELECTED_NUM_FREQUENCIES*NUM_DIMENSIONS + w_idx*NUM_DIMENSIONS;
                float fx = MAX_FREQUENCY*c_idx[spot]/(float)TOTAL_NUM_FREQUENCIES;
                float fy = MAX_FREQUENCY*c_idx[spot+1]/(float)TOTAL_NUM_FREQUENCIES;
                
                wx += c[spot]*__cosf(fx*x.x);
                wy += c[spot+1]*__cosf(fy*x.y);
            }            
            if(!gaussian_only) g *= fabsf(wx*wy);            
            temp_result += g;
        }
        
        atomicAdd(&output[i], temp_result);
    }

    __global__ void preprocess_primitives(  
        int NUM_PRIMITIVES,
        float min_x, float min_y, 
        float range_x, float range_y,
        const float* __restrict__ positions,
        const float* __restrict__ scales,
        uint8_t* blocks_per_gaussian
        ) {
            // Get block/thread related numbers   
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = blockDim.x * gridDim.x;

            uint8_t total_blocks = 0;
            for(int i = index; i < NUM_PRIMITIVES; i += stride){
                float sx = scales[2*i];
                float sy = scales[2*i+1];
                float px = positions[2*i];
                float py = positions[2*i+1];
                float r = 3.0f/min(sx, sy);
                int x_block_min = BLOCKS_X*(px - r - min_x)/range_x;
                int x_block_max = BLOCKS_X*(px + r - min_x)/range_x;
                int y_block_min = BLOCKS_Y*(py - r - min_y)/range_y;
                int y_block_max = BLOCKS_X*(py + r - min_y)/range_y;
                
                if(x_block_min < 0) x_block_min = 0;
                if(y_block_min < 0) y_block_min = 0;
                if(x_block_max >= BLOCKS_X) x_block_max = BLOCKS_X-1;
                if(y_block_max >= BLOCKS_Y) y_block_max = BLOCKS_Y-1;
                for (int x = x_block_min; x <= x_block_max; x++){
                    for (int y = y_block_min; y <= y_block_max; y++){
                        total_blocks++;
                    }
                }
                blocks_per_gaussian[i] = total_blocks;
            }
        }

    __global__ void create_gaussian_instances(  
        int NUM_PRIMITIVES,
        float min_x, float min_y, 
        float range_x, float range_y,
        const float* __restrict__ positions,
        const float* __restrict__ scales,
        uint32_t* __restrict__ offsets,
        uint64_t* __restrict__ unsorted_gaussians
        ) {
            // Get block/thread related numbers   
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = blockDim.x * gridDim.x;

            uint8_t total_blocks = 0;
            for(int i = index; i < NUM_PRIMITIVES; i += stride){
                //uint32_t off = (idx == 0) ? 0 : offsets[i - 1];
                uint32_t off = offsets[i];
                float sx = scales[2*i];
                float sy = scales[2*i+1];
                float px = positions[2*i];
                float py = positions[2*i+1];
                float r = 3.0f/min(sx, sy);
                int x_block_min = BLOCKS_X*(px - r - min_x)/range_x;
                int x_block_max = BLOCKS_X*(px + r - min_x)/range_x;
                int y_block_min = BLOCKS_Y*(py - r - min_y)/range_y;
                int y_block_max = BLOCKS_X*(py + r - min_y)/range_y;
                
                if(x_block_min < 0) x_block_min = 0;
                if(y_block_min < 0) y_block_min = 0;
                if(x_block_max >= BLOCKS_X) x_block_max = BLOCKS_X-1;
                if(y_block_max >= BLOCKS_Y) y_block_max = BLOCKS_Y-1;
                for (int x = x_block_min; x <= x_block_max; x++){
                    for (int y = y_block_min; y <= y_block_max; y++){
                        uint64_t key = y*BLOCKS_X+x;
                        key <<= 32;
                        key |= (uint32_t)i;
                        unsorted_gaussians[off] = key;
                        off++;
                    }
                }
            }
        }

    __global__ void periodic_primitives_forward_cuda_kernel(  
        int NUM_PRIMITIVES,
        int BATCH_SIZE,   
        int NUM_FREQUENCIES,   
        float MAX_FREQUENCY,
        float min_x, float min_y, 
        float range_x, float range_y,
        const float* __restrict__ input,
        const float* __restrict__ colors,
        const float* __restrict__ positions,
        const float* __restrict__ scales,
        const float* __restrict__ rotations,
        const float* __restrict__ wave_coefficients,
        const int* __restrict__ wave_coefficient_indices,
        const uint32_t* __restrict__ gaussian_blocks,
        float* __restrict__ output
        ) {

        // Get block/thread related numbers   
        const int threadID = threadIdx.x;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
       
        for(int i = threadID; i < BATCH_SIZE; i+=NUM_THREADS){
            float2 x = {input[2*i], input[2*i+1]};
            int bx = BLOCKS_X*(x.x - min_x)/range_x;
            int by = BLOCKS_Y*(x.y - min_y)/range_y;
            if(bx < 0) bx = 0;
            if(bx >= BLOCKS_X) bx = BLOCKS_X-1;
            if(by < 0) by = 0;
            if(by >= BLOCKS_Y) by = BLOCKS_Y-1;
            if(bx != block_x || by != block_y) continue;
            float3 temp_result = {0.0f, 0.0f, 0.0f};

            for(int idx = 0; idx < NUM_PRIMITIVES; idx++){
                uint32_t block_data[8] = {gaussian_blocks[8*idx], gaussian_blocks[8*idx+1],
                                        gaussian_blocks[8*idx+2], gaussian_blocks[8*idx+3],
                                        gaussian_blocks[8*idx+4], gaussian_blocks[8*idx+5],
                                        gaussian_blocks[8*idx+6], gaussian_blocks[8*idx+7]};
                if(!get256bitOffset(block_data, block_y*BLOCKS_X+block_x)) continue;

                float3 color = {colors[3*idx], colors[3*idx+1], colors[3*idx+2]};
                float2 position = { positions[2*idx], positions[2*idx+1]};
                float2 scale = { scales[2*idx], scales[2*idx+1]};
                float rotation = rotations[idx];
                float2 dx = x - position;
                
                float cosr = __cosf(rotation);
                float sinr = __sinf(rotation);
                float2 tx = { scale.x*(dx.x*cosr  + dx.y*sinr),
                            scale.y*(dx.x*-sinr + dx.y*cosr) };
                float g = __expf(-(tx.x*tx.x + tx.y*tx.y)/2);

                temp_result += g*color;
                //temp_result.x += 1.0;
            }
            
            output[i*3] = temp_result.x;
            output[i*3+1] = temp_result.y;
            output[i*3+2] = temp_result.z;          
        }
    }


    __global__ void periodic_primitives_backward_cuda_kernel(
        const int NUM_PRIMITIVES,
        const int BATCH_SIZE, 
        const float MAX_FREQUENCY,
        const bool gaussian_only,
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
        // Get block/thread related numbers   
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;

        for(int j = index; j < NUM_PRIMITIVES; j+= stride){
            // Load all data for the primitive from global mem.
            float rotation = rotations[j];
            float3 color = {colors[3*j], colors[3*j+1], colors[3*j+2]};  
            float2 pos = {positions[2*j], positions[2*j+1] };
            float2 scale = { scales[2*j], scales[2*j+1]};
            float coeffs[SELECTED_NUM_FREQUENCIES][2];
            int coeff_idx[SELECTED_NUM_FREQUENCIES][2];
            if(!gaussian_only){
                for(int w_idx = 0; w_idx < SELECTED_NUM_FREQUENCIES; w_idx++){
                    int spot = j*SELECTED_NUM_FREQUENCIES*NUM_DIMENSIONS + w_idx*NUM_DIMENSIONS;
                    coeffs[w_idx][0] = wave_coefficients[spot];
                    coeffs[w_idx][1] = wave_coefficients[spot+1];
                    coeff_idx[w_idx][0] = wave_coefficient_indices[spot];
                    coeff_idx[w_idx][1] = wave_coefficient_indices[spot+1];
                } 
            }   

            float cosr = __cosf(rotation);
            float sinr = __sinf(rotation);
            
            // Use local memory for temp updates instead of global
            float3 dColor_temp = {0.0f, 0.0f, 0.0f};
            float2 dPosition_temp = {0.0f, 0.0f};
            float2 dScale_temp = {0.0f, 0.0f};
            float dRotation_temp = 0.0f;
            float dCoefficients_temp[SELECTED_NUM_FREQUENCIES][2] = { 0.0f };
            
            // Iterate over all input points
            for(int i = 0; i < BATCH_SIZE; i++){
                float2 x = {input[2*i], input[2*i+1]};
                float3 dRGB = {grad_output[3*i], grad_output[3*i+1], grad_output[3*i+2]};
                float2 dx = x - pos;

                float2 tx = { scale.x*(dx.x*cosr  + dx.y*sinr),
                            scale.y*(dx.x*-sinr + dx.y*cosr) };
                float g = __expf(-(tx.x*tx.x + tx.y*tx.y)/2);

                float wx = 0.0f, wy = 0.0f;
                if(!gaussian_only){
                    for(int w_idx = 0; w_idx < SELECTED_NUM_FREQUENCIES; w_idx++){
                        float fx = MAX_FREQUENCY*coeff_idx[w_idx][0]/(float)TOTAL_NUM_FREQUENCIES;
                        float fy = MAX_FREQUENCY*coeff_idx[w_idx][1]/(float)TOTAL_NUM_FREQUENCIES;
                        wx += coeffs[w_idx][0]*__cosf(fx*x.x);
                        wy += coeffs[w_idx][1]*__cosf(fy*x.y);
                    }            
                }
                else{
                    wx = 1.0;
                    wy = 1.0;
                }
                float color_contribution = dRGB.x*color.x+dRGB.y*color.y+dRGB.z*color.z;
                float shared_coeff = color_contribution*g*wx*wy;
                
                dColor_temp += g*wx*wy*dRGB;
                dPosition_temp.x += shared_coeff*-(-tx.x*scale.x*cosr - tx.y*scale.y*-sinr);
                dPosition_temp.y += shared_coeff*-(-tx.x*scale.x*sinr - tx.y*scale.y*cosr);
                dScale_temp.x += shared_coeff*tx.x*-(dx.x*cosr  + dx.y*sinr);
                dScale_temp.y += shared_coeff*tx.y*-(dx.x*-sinr+dx.y*cosr);
                dRotation_temp += shared_coeff*-(tx.x*(scale.x*dx.x*-sinr+scale.x*dx.y*cosr) 
                                                + tx.y*(scale.y*dx.x*-cosr + scale.y*dx.y*-sinr)); 
                                                
                if(!gaussian_only){
                    for(int w_idx = 0; w_idx < SELECTED_NUM_FREQUENCIES; w_idx++){
                        float fx = MAX_FREQUENCY*coeff_idx[w_idx][0]/(float)TOTAL_NUM_FREQUENCIES;
                        float fy = MAX_FREQUENCY*coeff_idx[w_idx][1]/(float)TOTAL_NUM_FREQUENCIES;
                        dCoefficients_temp[w_idx][0] += color_contribution*g*wy*__cosf(fx*x.x);
                        dCoefficients_temp[w_idx][1] += color_contribution*g*wx*__cosf(fy*x.y);
                        //dPosition_temp.x += color_contribution*g*wy*coeffs[w_idx][0]*fx*sinf(fx*dx.x);
                        //dPosition_temp.y += color_contribution*g*wx*coeffs[w_idx][1]*fy*sinf(fy*dx.y);
                    }  
                }
            }
            dColors[3*j] = dColor_temp.x;
            dColors[3*j+1] = dColor_temp.y;
            dColors[3*j+2] = dColor_temp.z;
            dPositions[2*j] = dPosition_temp.x;
            dPositions[2*j+1] = dPosition_temp.y;
            dScales[2*j] = dScale_temp.x;
            dScales[2*j+1] = dScale_temp.y;
            dRotations[j] = dRotation_temp;
            
            if(!gaussian_only){
                for(int w_idx = 0; w_idx < SELECTED_NUM_FREQUENCIES; w_idx++){
                    dCoefficients[j*SELECTED_NUM_FREQUENCIES*2 + 2*w_idx] = dCoefficients_temp[w_idx][0];
                    dCoefficients[j*SELECTED_NUM_FREQUENCIES*2 + 2*w_idx + 1] = dCoefficients_temp[w_idx][1];
                }
            }
        }
    }
}

std::vector<torch::Tensor> periodic_primitives_forward_cuda(
    torch::Tensor input,                        // [N, n_dim]
    torch::Tensor colors,                       // [M, n_chan]
    torch::Tensor positions,                    // [M, n_dim]
    torch::Tensor scales,                       // [M, n_dim]
    torch::Tensor rotations,                    // [M, 1]
    torch::Tensor wave_coefficients,            // [M, n_freqs, n_dim]
    torch::Tensor wave_coefficient_indices,    // [M, n_freqs, n_dim]
    const float MAX_FREQUENCY,    
    const bool gaussian_only
    ) {

    // Get sizes for the output
    const int batch_size = input.size(0);
    const int num_primitives = colors.size(0);
    const int num_frequencies = wave_coefficients.size(2);

    // Create output tensor and other tensors
    auto output = torch::zeros({batch_size, NUM_CHANNELS}, input.device());
    /*
    //unsigned long long bytes_needed = (8ull)*num_primitives*sizeof(uint32_t);
    //cudaError_t status = cudaMalloc((void**)&gaussian_blocks, bytes_needed);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
        return {output};
    }
    */
    // If we want to adjust the number of blocks, this is a good way
    // Scales with the number of SMs on the GPU
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    
    /*
    float range_x = max_x - min_x + 0.0000001f;
    float range_y = max_y - min_y + 0.0000001f;
    uint32_t blocks_per_gaussian[num_primitives];
    uint32_t cumulative_sum[num_primitives];
    uint32_t offsets[num_primitives];
    preprocess_primitives<<<(num_primitives+512-1)/512,512>>>(  
        num_primitives,
        min_x, min_y,
        range_x, range_y,
        positions.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        blocks_per_gaussian        
        );
    
    std::cout << "Hello World!" << std::endl;
    
    void* d_temp_storage = NULL;
    *size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
		blocks_per_gaussian, offsets, batch_size);    
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
		blocks_per_gaussian, offsets, batch_size);    

    uint32_t total_gaussian_instances = cumulative_sum[batch_size-1];
    uint64_t unsorted_gaussians[total_gaussian_instances];

    create_gaussian_instances<<<(num_primitives+512-1)/512,512>>>(
        num_primitives,
        min_x, min_y,
        range_x, range_y,
        positions.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        offsets,
        unsorted_gaussians
    );
    d_temp_storage = NULL;
    temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortPairs(
		d_temp_storage,
		temp_storage_bytes,
		unsorted_gaussians, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		total_gaussian_instances);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortPairs(
		d_temp_storage,
		temp_storage_bytes,
		unsorted_gaussians, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		total_gaussian_instances);
    */

    // Now parallelize over query points
    dim3 numBlocks ((batch_size+NUM_THREADS-1)/NUM_THREADS, (num_primitives+NUM_THREADS-1)/NUM_THREADS);
    //dim3 numBlocks(BLOCKS_X, BLOCKS_Y);
    periodic_primitives_forward_cuda_kernel_efficient<<<numBlocks, NUM_THREADS>>>(
        num_primitives,
        batch_size,
        num_frequencies,
        MAX_FREQUENCY,
        gaussian_only,
        input.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        positions.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        wave_coefficients.contiguous().data_ptr<float>(),
        wave_coefficient_indices.contiguous().data_ptr<int>(),
        output.contiguous().data_ptr<float>()
        );
        
    return {output};
}

std::vector<torch::Tensor> periodic_primitives_heatmap_cuda(
    torch::Tensor input,                        // [N, n_dim]
    torch::Tensor positions,                    // [M, n_dim]
    torch::Tensor scales,                       // [M, n_dim]
    torch::Tensor rotations,                    // [M, 1]
    torch::Tensor wave_coefficients,            // [M, n_freqs, n_dim]
    torch::Tensor wave_coefficient_indices,    // [M, n_freqs, n_dim]
    const float MAX_FREQUENCY,
    const bool gaussian_only
    ) {

    // Get sizes for the output
    const int batch_size = input.size(0);
    const int num_primitives = scales.size(0);
    const int num_frequencies = wave_coefficients.size(2);

    // Create output tensor and other tensors
    auto output = torch::zeros({batch_size}, input.device());
        
    // If we want to adjust the number of blocks, this is a good way
    // Scales with the number of SMs on the GPU
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    

    // Now parallelize over query points
    dim3 numBlocks ((batch_size+NUM_THREADS-1)/NUM_THREADS, (num_primitives+NUM_THREADS-1)/NUM_THREADS);
    //dim3 numBlocks(BLOCKS_X, BLOCKS_Y);
    periodic_primitives_heatmap_cuda_kernel<<<numBlocks, NUM_THREADS>>>(
        num_primitives,
        batch_size,
        num_frequencies,
        MAX_FREQUENCY,
        gaussian_only,
        input.contiguous().data_ptr<float>(),
        positions.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        wave_coefficients.contiguous().data_ptr<float>(),
        wave_coefficient_indices.contiguous().data_ptr<int>(),
        output.contiguous().data_ptr<float>()
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
    const float MAX_FREQUENCY,
    const bool gaussian_only
    ) {
        // Get sizes for the output
        const int batch_size = input.size(0);
        const int num_primitives = colors.size(0);

        // Set up gradient tensors
        auto dColors = torch::empty_like(colors);
        auto dPositions = torch::empty_like(positions); 
        auto dScales = torch::empty_like(scales); 
        auto dRotations = torch::empty_like(rotations); 
        auto dCoefficients = torch::empty_like(wave_coefficients);
        
        //int numSMs;
        //cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
        const int blocks = (num_primitives+128-1)/128;
        //std::cout << dCoefficients.sizes() << std::endl;
        // Dispatch jobs
        periodic_primitives_backward_cuda_kernel<<<blocks, 128>>>(
            num_primitives,
            batch_size,
            MAX_FREQUENCY,
            gaussian_only,
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
