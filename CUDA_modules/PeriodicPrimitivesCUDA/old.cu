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

__global__ void periodic_primitives_forward_cuda_kernel_oldversion(
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




__global__ void periodic_primitives_heatmap_cuda_kernel_oldversion(  
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


    /* Old version. Works fine, gets slow
    // Now parallelize over query points
    dim3 numBlocks ((batch_size+NUM_THREADS-1)/NUM_THREADS, (num_primitives+NUM_THREADS-1)/NUM_THREADS);
    //dim3 numBlocks(BLOCKS_X, BLOCKS_Y);
    periodic_primitives_forward_cuda_kernel_oldversion<<<numBlocks, NUM_THREADS>>>(
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
    */