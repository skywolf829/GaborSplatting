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

#define FORWARD_NUM_THREADS 512
#define TOTAL_NUM_FREQUENCIES 128
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


__global__ void point_to_block_and_index(  
    int num_points,
    float2 min_position, float2 range,
    const float* __restrict__ positions,
    int* block,
    int* point_index
) {
    // Get block/thread related numbers   
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for(int i = index; i < num_points; i += stride){
        float px = positions[2*i];
        float py = positions[2*i+1];
        int x_block = BLOCKS_X*((px - min_position.x)/range.x);
        int y_block = BLOCKS_Y*((py - min_position.y)/range.y);
        x_block = min(max(x_block, 0), BLOCKS_X-1);
        y_block = min(max(y_block, 0), BLOCKS_Y-1);
        block[i] = y_block*BLOCKS_X+x_block;
        point_index[i] = i;
    }
}

__global__ void find_blocks_per_gaussian(  
    int num_points,
    float2 min_position, 
    float2 range,
    const float* __restrict__ positions,
    const float* __restrict__ scales,
    int* blocks_per_gaussian
) {
    // Get block/thread related numbers   
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for(int i = index; i < num_points; i += stride){
        float sx = scales[2*i];
        float sy = scales[2*i+1];
        float px = positions[2*i];
        float py = positions[2*i+1];
        float r = 3.0f/min(sx, sy);
        int x_block_min = BLOCKS_X*(px - r - min_position.x)/range.x;
        int x_block_max = BLOCKS_X*(px + r - min_position.x)/range.x;
        int y_block_min = BLOCKS_Y*(py - r - min_position.y)/range.y;
        int y_block_max = BLOCKS_Y*(py + r - min_position.y)/range.y;
        
        x_block_min = min(max(0, x_block_min), BLOCKS_X);
        y_block_min = min(max(0, y_block_min), BLOCKS_Y);
        x_block_max = max(min(BLOCKS_X-1, x_block_max), -1);
        y_block_max = max(min(BLOCKS_Y-1, y_block_max), -1);

        blocks_per_gaussian[i] = (x_block_max-x_block_min+1)*(y_block_max-y_block_min+1);
    }
}

__global__ void create_gaussian_instances(  
    int num_gaussians,
    float2 min_pos, float2 range,
    const float* __restrict__ positions,
    const float* __restrict__ scales,
    int* __restrict__ cumulative_sums,
    int* __restrict__ unsorted_gaussian_keys,
    int* __restrict__ unsorted_gaussian_indices
    ) {
        // Get block/thread related numbers   
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;

        int offset = 0;
        for(auto i = index; i < num_gaussians; i += stride){
            offset = (i == 0) ? 0 : cumulative_sums[i-1];
            float sx = scales[2*i];
            float sy = scales[2*i+1];
            float px = positions[2*i];
            float py = positions[2*i+1];
            float r = 3.0f/min(sx, sy);
            int x_block_min = BLOCKS_X*(px - r - min_pos.x)/range.x;
            int x_block_max = BLOCKS_X*(px + r - min_pos.x)/range.x;
            int y_block_min = BLOCKS_Y*(py - r - min_pos.y)/range.y;
            int y_block_max = BLOCKS_Y*(py + r - min_pos.y)/range.y;
            
            x_block_min = min(max(0, x_block_min), BLOCKS_X);
            y_block_min = min(max(0, y_block_min), BLOCKS_Y);
            x_block_max = max(min(BLOCKS_X-1, x_block_max), -1);
            y_block_max = max(min(BLOCKS_Y-1, y_block_max), -1);
            for (int x = x_block_min; x <= x_block_max && x < BLOCKS_X; x++){
                for (int y = y_block_min; y <= y_block_max && x < BLOCKS_Y; y++){
                    int key = (y*BLOCKS_X+x);
                    //key <<= 32;
                    //key |= (uint32_t)i;
                    unsorted_gaussian_keys[offset] = key;
                    unsorted_gaussian_indices[offset] = i;
                    offset++;
                }
            }
        }
    }

__global__ void key_start_end_indices_cuda(int num_instances, int* keys, int* tile_start_end)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for(auto i = index; i < num_instances; i += stride){
        int this_key = keys[i];

        if(i > 0){       
            int last_key = keys[i-1];
            if(this_key != last_key){
                tile_start_end[2*last_key+1] = i;
                tile_start_end[2*this_key] = i;
            }
        }
        if(i < num_instances-1){            
            int next_key = keys[i+1];
            if(this_key != next_key){
                tile_start_end[2*this_key+1] = i+1;
                tile_start_end[2*next_key] = i+1;
            }
        }
        else{
            tile_start_end[2*this_key+1] = num_instances;
        }
    }
}

__global__ void periodic_primitives_forward_cuda_kernel(  
    int num_query_points, int num_gaussians, int num_gaussian_instances,   
    float max_frequency, bool gaussian_only, bool heatmap,
    const float* __restrict__ input,
    const float* __restrict__ colors,
    const float* __restrict__ positions,
    const float* __restrict__ scales,
    const float* __restrict__ rotations,
    const float* __restrict__ wave_coefficients,
    const int* __restrict__ wave_coefficient_indices,
    const int* __restrict__ gaussian_instance_indices,
    const int* __restrict__ block_start_end_index_gaussians,
    const int* __restrict__ query_indices,
    const int* __restrict__ block_start_end_index_query_points,
    float* __restrict__ output
    ) {

    // Get block/thread related numbers   
    const int threadID = threadIdx.x;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int this_block_idx = BLOCKS_X*block_y + block_x;

    int query_point_start_idx = block_start_end_index_query_points[2*this_block_idx];
    int query_point_end_idx = block_start_end_index_query_points[2*this_block_idx+1];
    int gaussian_start_idx = block_start_end_index_gaussians[2*this_block_idx];
    int gaussian_end_idx = block_start_end_index_gaussians[2*this_block_idx+1];

    // return if no query points or gaussians in this block
    if(query_point_start_idx == query_point_end_idx || gaussian_start_idx == gaussian_end_idx) return;

    __shared__ float gaussian_positions[FORWARD_NUM_THREADS][2];
    __shared__ float gaussian_scales[FORWARD_NUM_THREADS][2];
    __shared__ float gaussian_colors[FORWARD_NUM_THREADS][3];
    __shared__ float gaussian_rotations[FORWARD_NUM_THREADS];
    __shared__ float coefficients[FORWARD_NUM_THREADS][SELECTED_NUM_FREQUENCIES];
    __shared__ int coefficient_indices[FORWARD_NUM_THREADS][SELECTED_NUM_FREQUENCIES];
    
    int num_point_batchs = 1 + (gaussian_end_idx - gaussian_start_idx) / FORWARD_NUM_THREADS;

    for(int batch = 0; batch < num_point_batchs; batch++){

        int end_idx_this_batch = min(FORWARD_NUM_THREADS, gaussian_end_idx-gaussian_start_idx-batch*FORWARD_NUM_THREADS);

        // Each thread loads a part of global memory to shared (random reads)
        int collect_idx = gaussian_start_idx + batch*FORWARD_NUM_THREADS + threadID;
        __syncthreads();
        if(collect_idx < num_gaussian_instances){
            int idx = gaussian_instance_indices[collect_idx];
            gaussian_positions[threadID][0] = positions[2*idx];
            gaussian_positions[threadID][1] = positions[2*idx+1];
            gaussian_scales[threadID][0] = scales[2*idx];
            gaussian_scales[threadID][1] = scales[2*idx+1];
            gaussian_colors[threadID][0] = colors[3*idx];
            gaussian_colors[threadID][1] = colors[3*idx+1];
            gaussian_colors[threadID][2] = colors[3*idx+2];
            gaussian_rotations[threadID] = rotations[idx];
            for(int i = 0; i < SELECTED_NUM_FREQUENCIES && !gaussian_only; i++){
                coefficients[threadID][i] = wave_coefficients[idx*SELECTED_NUM_FREQUENCIES+i];
                coefficient_indices[threadID][i] = wave_coefficient_indices[idx*SELECTED_NUM_FREQUENCIES+i];
            }
        }
        __syncthreads();
        // Iterate over all query points this thread is responsible for
        // Update its value according to the currently cached gaussians
        for(int i = query_point_start_idx+threadID; i < query_point_end_idx; i += FORWARD_NUM_THREADS){
            
            int real_query_index = query_indices[i];
            float2 x = {input[2*real_query_index], input[2*real_query_index+1]};
            float3 temp_result = {0.0f, 0.0f, 0.0f};
            for(int j = 0; j < end_idx_this_batch; j++){

                float2 dx = x - make_float2(gaussian_positions[j][0], gaussian_positions[j][1]);
                float cosr = __cosf(gaussian_rotations[j]);
                float sinr = __sinf(gaussian_rotations[j]);
                float2 tx = { gaussian_scales[j][0]*(dx.x*cosr  + dx.y*sinr),
                            gaussian_scales[j][1]*(dx.x*-sinr + dx.y*cosr) };
                float g = __expf(-(tx.x*tx.x + tx.y*tx.y)/2);
                float w = 0.0f;
                for(int w_idx = 0; w_idx < SELECTED_NUM_FREQUENCIES && !gaussian_only; w_idx++){
                    float f = max_frequency*(coefficient_indices[j][w_idx])/(float)TOTAL_NUM_FREQUENCIES;
                    // old (radial)
                    //w += coefficients[j][w_idx]*__cosf(f*(1-g));
                    w += coefficients[j][w_idx]*__cosf(f*tx.x);
                }            
                if(!gaussian_only) g *= w;
                if(heatmap){
                    temp_result.x += fabsf(g);
                    temp_result.y += fabsf(g);
                    temp_result.z += fabsf(g);
                }
                else{
                    temp_result.x += g*gaussian_colors[j][0];
                    temp_result.y += g*gaussian_colors[j][1];
                    temp_result.z += g*gaussian_colors[j][2];
                }
            }
            output[3*real_query_index] += temp_result.x;
            output[3*real_query_index+1] += temp_result.y;
            output[3*real_query_index+2] += temp_result.z;
        }      
    }
}



__global__ void periodic_primitives_backward_cuda_kernel(
    const int num_primitives,
    const int batch_size, 
    const float max_frequency,
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
    float* __restrict__ dCoefficients,
    const int* __restrict__ gaussian_instance_indices,
    const int* __restrict__ block_start_end_index_gaussians,
    const int* __restrict__ query_indices,
    const int* __restrict__ block_start_end_index_query_points
    ) 
{
   // Get block/thread related numbers   
    const int threadID = threadIdx.x;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int this_block_idx = BLOCKS_X*block_y + block_x;

    int query_point_start_idx = block_start_end_index_query_points[2*this_block_idx];
    int query_point_end_idx = block_start_end_index_query_points[2*this_block_idx+1];
    int gaussian_start_idx = block_start_end_index_gaussians[2*this_block_idx];
    int gaussian_end_idx = block_start_end_index_gaussians[2*this_block_idx+1];

    // return if no query points or gaussians in this block
    if(query_point_start_idx == query_point_end_idx || gaussian_start_idx == gaussian_end_idx) return;

    __shared__ float query_point_positions[FORWARD_NUM_THREADS][2];
    __shared__ float dRGB[FORWARD_NUM_THREADS][3];
    
    int num_point_batchs = 1 + (query_point_end_idx - query_point_start_idx) / FORWARD_NUM_THREADS;

    for(int batch = 0; batch < num_point_batchs; batch++){

        int end_idx_this_batch = min(FORWARD_NUM_THREADS, query_point_end_idx-query_point_start_idx-batch*FORWARD_NUM_THREADS);

        // Each thread loads a part of global memory to shared (random reads)
        int collect_idx = query_point_start_idx + batch*FORWARD_NUM_THREADS + threadID;
        __syncthreads();
        if(collect_idx < batch_size){
            int idx = query_indices[collect_idx];
            query_point_positions[threadID][0] = input[2*idx];
            query_point_positions[threadID][1] = input[2*idx+1];
            dRGB[threadID][0] = grad_output[3*idx];
            dRGB[threadID][1] = grad_output[3*idx+1];
            dRGB[threadID][2] = grad_output[3*idx+2];
        }
        __syncthreads();
        // Iterate over all gaussians points this thread is responsible for
        // Update its value according to the currently cached gaussians
        for(int i = gaussian_start_idx+threadID; i < gaussian_end_idx; i += FORWARD_NUM_THREADS){
            
            int real_query_index = gaussian_instance_indices[i];
            float3 color = {colors[3*real_query_index], colors[3*real_query_index+1], colors[3*real_query_index+2]};
            float2 pos = {positions[2*real_query_index], positions[2*real_query_index+1]};
            float2 s = {scales[2*real_query_index], scales[2*real_query_index+1]};
            float r = rotations[real_query_index];
            float coeff[SELECTED_NUM_FREQUENCIES];
            int coeff_index[SELECTED_NUM_FREQUENCIES];
            for(int j = 0; j < SELECTED_NUM_FREQUENCIES && !gaussian_only; j++){
                coeff[j] = wave_coefficients[SELECTED_NUM_FREQUENCIES*real_query_index+j];
                coeff_index[j] = wave_coefficient_indices[SELECTED_NUM_FREQUENCIES*real_query_index+j];
            }

            float3 dColor_temp = {0.0f, 0.0f, 0.0f};
            float2 dPosition_temp = {0.0f, 0.0f};
            float2 dScale_temp = {0.0f, 0.0f};
            float dRotation_temp = 0.0f;
            float dCoefficients_temp[SELECTED_NUM_FREQUENCIES] = { 0.0f };

            for(int j = 0; j < end_idx_this_batch; j++){
                float2 x = {query_point_positions[j][0], query_point_positions[j][1]};
                float3 dRGB_f3 = {dRGB[j][0], dRGB[j][1], dRGB[j][2]};
                float color_contribution = dRGB_f3.x*color.x+dRGB_f3.y*color.y+dRGB_f3.z*color.z;

                float2 dx = x - pos;
                float cosr = __cosf(r);
                float sinr = __sinf(r);
                float2 tx = { s.x*(dx.x*cosr  + dx.y*sinr),
                            s.y*(dx.x*-sinr + dx.y*cosr) };
                float g = __expf(-0.5*(tx.x*tx.x + tx.y*tx.y));
                float w = 0.0f;
                for(int w_idx = 0; w_idx < SELECTED_NUM_FREQUENCIES && !gaussian_only; w_idx++){
                    float f = max_frequency*(coeff_index[w_idx])/(float)TOTAL_NUM_FREQUENCIES;
                    // old radial
                    // w += coeff[w_idx]*__cosf(f*(1-g));
                    // dCoefficients_temp[w_idx] += color_contribution*g*__cosf(f*(1-g));                    
                    /*
                    float deriv_coeff = color_contribution*g*coeff[w_idx]*__sinf(f*(1-g))*f;
                    dPosition_temp.x += deriv_coeff*g*-(-tx.x*s.x*cosr - tx.y*s.y*-sinr);
                    dPosition_temp.y += deriv_coeff*g*-(-tx.x*s.x*sinr - tx.y*s.y*cosr);
                    dScale_temp.x += deriv_coeff*tx.x*g*-(dx.x*cosr  + dx.y*sinr);
                    dScale_temp.y += deriv_coeff*tx.y*g*-(dx.x*-sinr+dx.y*cosr);
                    dRotation_temp += deriv_coeff*g*-(tx.x*(s.x*dx.x*-sinr+s.x*dx.y*cosr) 
                                                    + tx.y*(s.y*dx.x*-cosr + s.y*dx.y*-sinr)); */

                    w += coeff[w_idx]*__cosf(f*tx.x);
                    dCoefficients_temp[w_idx] += color_contribution*g*__cosf(f*tx.x);

                    
                    float deriv_coeff = color_contribution*g*coeff[w_idx]*-__sinf(f*tx.x)*f;
                    dPosition_temp.x += deriv_coeff*-(s.x*cosr);
                    dPosition_temp.y += deriv_coeff*-(s.x*sinr);                    
                    dScale_temp.x += deriv_coeff*(dx.x*cosr  + dx.y*sinr);
                    dRotation_temp += deriv_coeff*(s.x*(dx.x*-sinr  + dx.y*cosr)); 
                }       

                float shared_coeff = color_contribution*g;
                if(!gaussian_only){
                    shared_coeff *= w;
                    dColor_temp += g*w*dRGB_f3;
                }
                else{
                    dColor_temp += g*dRGB_f3;
                }

                dPosition_temp.x += shared_coeff*-(-tx.x*s.x*cosr - tx.y*s.y*-sinr);
                dPosition_temp.y += shared_coeff*-(-tx.x*s.x*sinr - tx.y*s.y*cosr);
                dScale_temp.x += shared_coeff*tx.x*-(dx.x*cosr  + dx.y*sinr);
                dScale_temp.y += shared_coeff*tx.y*-(dx.x*-sinr+dx.y*cosr);
                dRotation_temp += shared_coeff*-(tx.x*(s.x*dx.x*-sinr+s.x*dx.y*cosr) 
                                                + tx.y*(s.y*dx.x*-cosr + s.y*dx.y*-sinr)); 
                      
            }
            atomicAdd(&dColors[3*real_query_index], dColor_temp.x);
            atomicAdd(&dColors[3*real_query_index+1], dColor_temp.y);
            atomicAdd(&dColors[3*real_query_index+2], dColor_temp.z);
            atomicAdd(&dPositions[2*real_query_index], dPosition_temp.x);
            atomicAdd(&dPositions[2*real_query_index+1], dPosition_temp.y);
            atomicAdd(&dScales[2*real_query_index], dScale_temp.x);
            atomicAdd(&dScales[2*real_query_index+1], dScale_temp.y);
            atomicAdd(&dRotations[real_query_index], dRotation_temp);
            
            if(!gaussian_only){
                for(int w_idx = 0; w_idx < SELECTED_NUM_FREQUENCIES; w_idx++){
                    atomicAdd(&dCoefficients[real_query_index*SELECTED_NUM_FREQUENCIES + w_idx], 
                        dCoefficients_temp[w_idx]);
                }
            }
        }      
    }
}

void sort_query_points_to_blocks(torch::Tensor positions,
float2 min_pos, float2 max_pos, int *&sorted_point_indices, int *&query_point_block_start_end_indices){
    int num_points = positions.size(0);
    float2 range = make_float2(max_pos.x-min_pos.x,
                                max_pos.y-min_pos.y);
    
    
    int *unsorted_point_blocks, *unsorted_point_indices;
    cudaMalloc((void**)&unsorted_point_blocks, num_points*sizeof(int));  
    cudaMalloc((void**)&unsorted_point_indices, num_points*sizeof(int));
    point_to_block_and_index<<<num_points+512-1/512, 512>>>(num_points,
        min_pos, range,
        positions.contiguous().data_ptr<float>(),
        unsorted_point_blocks, unsorted_point_indices);

    int *sorted_point_blocks;
    cudaMalloc((void**)&sorted_point_blocks, num_points*sizeof(int));  
    cudaMalloc((void**)&sorted_point_indices, num_points*sizeof(int));

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
		d_temp_storage,
		temp_storage_bytes,
		unsorted_point_blocks, sorted_point_blocks,
		unsorted_point_indices, sorted_point_indices,
		num_points);
    
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortPairs(
		d_temp_storage,
		temp_storage_bytes,
		unsorted_point_blocks, sorted_point_blocks,
		unsorted_point_indices, sorted_point_indices,
		num_points);


    cudaMalloc((void**)&query_point_block_start_end_indices, 2*(BLOCKS_X*BLOCKS_Y)*sizeof(int));   
    cudaMemset(query_point_block_start_end_indices, 0, 2*(BLOCKS_X*BLOCKS_Y)*sizeof(int));
    key_start_end_indices_cuda<<<(num_points + 512 - 1) / 512, 512>>> (
        num_points,
        sorted_point_blocks,
        query_point_block_start_end_indices
        );    
    cudaFree(unsorted_point_blocks);
    cudaFree(unsorted_point_indices);
    cudaFree(sorted_point_blocks);
    cudaFree(d_temp_storage);
}

int sort_gaussians_to_blocks(torch::Tensor gaussian_positions, torch::Tensor gaussian_scales,
float2 min_pos, float2 max_pos, int *&sorted_gaussian_indices, int *&block_start_end_indices){

    // 1. Determine the number of gaussians per block
    int num_gaussians = gaussian_positions.size(0);
    float2 range = make_float2(max_pos.x-min_pos.x,max_pos.y-min_pos.y);
    int* blocks_per_gaussian;
    cudaMalloc((void**)&blocks_per_gaussian, num_gaussians*sizeof(int));   

    find_blocks_per_gaussian<<<(num_gaussians+512-1)/512,512>>>(
        num_gaussians, min_pos, range,
        gaussian_positions.contiguous().data_ptr<float>(),
        gaussian_scales.contiguous().data_ptr<float>(),
        blocks_per_gaussian  
        );

    // 2. Inclusive sum on gaussians per block to find total number
    // of gaussian instances needed
    // Allocate temp storage for the inclusive sum
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    int* cumulative_sums;
    cudaMalloc((void**)&cumulative_sums, num_gaussians*sizeof(int));    
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
		blocks_per_gaussian, cumulative_sums, num_gaussians);    
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
		blocks_per_gaussian, cumulative_sums, num_gaussians);  

    
    
    // Get the total number of gaussian instances we have on host (cpu)
    int total_gaussian_instances;
    cudaMemcpy(&total_gaussian_instances, &cumulative_sums[num_gaussians-1], sizeof(int), cudaMemcpyDeviceToHost);
    //std::cout << "Total gaussian instances: " << total_gaussian_instances << std::endl;

    // If 0 gaussians need to be rendered, return
    if(total_gaussian_instances == 0) return total_gaussian_instances;
    // 3. Create the gaussian instances
    int *unsorted_gaussian_keys, *sorted_gaussian_keys;
    int *unsorted_gaussian_indices;
    cudaMalloc((void**)&unsorted_gaussian_keys, total_gaussian_instances*sizeof(int));   
    cudaMalloc((void**)&sorted_gaussian_keys, total_gaussian_instances*sizeof(int));   
    cudaMalloc((void**)&unsorted_gaussian_indices, total_gaussian_instances*sizeof(int));  
    cudaMalloc((void**)&sorted_gaussian_indices, total_gaussian_instances*sizeof(int));   

    create_gaussian_instances<<<(num_gaussians+512-1)/512,512>>>(
        num_gaussians,
        min_pos, range, 
        gaussian_positions.contiguous().data_ptr<float>(),
        gaussian_scales.contiguous().data_ptr<float>(),
        cumulative_sums,
        unsorted_gaussian_keys,
        unsorted_gaussian_indices
    );

    // 4. Sort the gaussian instances by keys (tileID)
    cudaFree(d_temp_storage);
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
		d_temp_storage,
		temp_storage_bytes,
		unsorted_gaussian_keys, sorted_gaussian_keys,
		unsorted_gaussian_indices, sorted_gaussian_indices,
		total_gaussian_instances);

    // Then actually sort
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortPairs(
		d_temp_storage,
		temp_storage_bytes,
		unsorted_gaussian_keys, sorted_gaussian_keys,
		unsorted_gaussian_indices, sorted_gaussian_indices,
		total_gaussian_instances);

    // 5. Identify index start/end index for gaussians in each tile 
    cudaMalloc((void**)&block_start_end_indices, 2*(BLOCKS_X*BLOCKS_Y)*sizeof(int));   
    cudaMemset(block_start_end_indices, 0, 2*(BLOCKS_X*BLOCKS_Y)*sizeof(int));
    key_start_end_indices_cuda<<<(total_gaussian_instances + 512 - 1) / 512, 512>>> (
        total_gaussian_instances,
        sorted_gaussian_keys,
        block_start_end_indices
        );
    // Only relevant memory is block_start_end_indices and sorted_gaussian_indices.
    // Free the rest.
    cudaFree(blocks_per_gaussian);
    cudaFree(d_temp_storage);
    cudaFree(unsorted_gaussian_indices);
    cudaFree(unsorted_gaussian_keys);
    cudaFree(sorted_gaussian_keys);
    cudaFree(cumulative_sums);
    return total_gaussian_instances;
}

std::vector<torch::Tensor> periodic_primitives_forward_cuda(
    torch::Tensor input,                        // [N, n_dim]
    torch::Tensor colors,                       // [M, n_chan]
    torch::Tensor positions,                    // [M, n_dim]
    torch::Tensor scales,                       // [M, n_dim]
    torch::Tensor rotations,                    // [M, 1]
    torch::Tensor wave_coefficients,            // [M, n_freqs, n_dim]
    torch::Tensor wave_coefficient_indices,    // [M, n_freqs, n_dim]
    const float max_frequency,    
    const bool gaussian_only,
    const bool heatmap = false
    ) {
        
    // Create output tensor and other tensors
    auto output = torch::zeros({input.size(0), NUM_CHANNELS}, input.device());
    
    // Sort query points and gaussians into 16x16 blocks
    int* sorted_gaussian_indices;
    int* blocks_gaussian_start_end_indices;
    int num_gaussian_instances = sort_gaussians_to_blocks(
        positions, scales,
        make_float2(0.0f, 0.0f), make_float2(1.0f, 1.0f),
        sorted_gaussian_indices, 
        blocks_gaussian_start_end_indices);
        
    int* sorted_query_point_indices;
    int* blocks_query_points_start_end_indices;
    sort_query_points_to_blocks(
        input, make_float2(0.0f, 0.0f), make_float2(1.0f, 1.0f),
        sorted_query_point_indices, 
        blocks_query_points_start_end_indices);

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor sorted_gaussian_indices_tensor = torch::from_blob(sorted_gaussian_indices, {num_gaussian_instances}, options).clone();
    torch::Tensor blocks_gaussian_start_end_indices_tensor = torch::from_blob(blocks_gaussian_start_end_indices, {BLOCKS_X*BLOCKS_Y, 2}, options).clone();
    torch::Tensor sorted_query_point_indices_tensor = torch::from_blob(sorted_query_point_indices, {input.size(0)}, options).clone();
    torch::Tensor blocks_query_points_start_end_indices_tensor = torch::from_blob(blocks_query_points_start_end_indices, {BLOCKS_X*BLOCKS_Y, 2}, options).clone();
    
    if(num_gaussian_instances == 0) return {output, sorted_gaussian_indices_tensor, 
        blocks_gaussian_start_end_indices_tensor, sorted_query_point_indices_tensor, 
        blocks_query_points_start_end_indices_tensor};
    

    // Now sorted_gaussian_indices orders the indices of the original gaussian
    // tensors in block order, so items are in block [0, 0, ..., 0, 1, 1, ..., 1, 2, ...]
    // Similar with sorted_query_point_indices.
    // cumulative_gaussians_per_block and cumulative_query_points_per_block are the
    // indices for which block 0->1 (so each thread block knows where to stop)

    // Finally evaluate results such that query points only evaulate with gaussians
    // within the block.
    dim3 numBlocks (16, 16);
    periodic_primitives_forward_cuda_kernel<<<numBlocks, FORWARD_NUM_THREADS>>>(
        input.size(0), positions.size(0), num_gaussian_instances,
        max_frequency, gaussian_only, heatmap,
        input.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        positions.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        wave_coefficients.contiguous().data_ptr<float>(),
        wave_coefficient_indices.contiguous().data_ptr<int>(),
        sorted_gaussian_indices,
        blocks_gaussian_start_end_indices,
        sorted_query_point_indices,
        blocks_query_points_start_end_indices,
        output.contiguous().data_ptr<float>()
        );
    cudaFree(sorted_gaussian_indices);
    cudaFree(blocks_gaussian_start_end_indices);
    cudaFree(sorted_query_point_indices);
    cudaFree(blocks_query_points_start_end_indices);
    return {output, sorted_gaussian_indices_tensor, 
        blocks_gaussian_start_end_indices_tensor, 
        sorted_query_point_indices_tensor,
        blocks_query_points_start_end_indices_tensor};
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
    torch::Tensor gaussian_instance_indices,
    torch::Tensor block_start_end_index_gaussians,
    torch::Tensor query_indices,
    torch::Tensor block_start_end_index_query_points,
    const float max_frequency,    
    const bool gaussian_only
    ) {
        // Get sizes for the output
        const int batch_size = input.size(0);
        const int num_primitives = colors.size(0);

        // Set up gradient tensors
        auto dColors = torch::zeros_like(colors);
        auto dPositions = torch::zeros_like(positions); 
        auto dScales = torch::zeros_like(scales); 
        auto dRotations = torch::zeros_like(rotations); 
        auto dCoefficients = torch::zeros_like(wave_coefficients);
        
        //int numSMs;
        //cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
        //const int blocks = (num_primitives+FORWARD_NUM_THREADS-1)/FORWARD_NUM_THREADS;
        
        dim3 blocks (16, 16);
        periodic_primitives_backward_cuda_kernel<<<blocks, FORWARD_NUM_THREADS>>>(
            num_primitives,
            batch_size,
            max_frequency,
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
            dCoefficients.contiguous().data_ptr<float>(),
            gaussian_instance_indices.contiguous().data_ptr<int>(),
            block_start_end_index_gaussians.contiguous().data_ptr<int>(),
            query_indices.contiguous().data_ptr<int>(),
            block_start_end_index_query_points.contiguous().data_ptr<int>()
            );
    
        return {dColors, dPositions, dScales, dRotations, dCoefficients };
    }
