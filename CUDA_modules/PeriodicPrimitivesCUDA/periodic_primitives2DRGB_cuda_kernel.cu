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

#define FORWARD_NUM_THREADS 128
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
        int x_block = max(min(BLOCKS_X*(px - min.x)/range.x, 0), BLOCKS_X-1);
        int y_block = max(min(BLOCKS_Y*(py - min.y)/range.y, 0), BLOCKS_Y-1);

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
        int x_block_min = BLOCKS_X*(px - r - min.x)/range.x;
        int x_block_max = BLOCKS_X*(px + r - min.x)/range.x;
        int y_block_min = BLOCKS_Y*(py - r - min.y)/range.y;
        int y_block_max = BLOCKS_Y*(py + r - min.y)/range.y;
        
        x_block_min = min(max(0, x_block_min), BLOCKS_X);
        y_block_min = min(max(0, y_block_min), BLOCKS_Y);
        x_block_max = max(min(BLOCKS_X-1, x_block_max), -1);
        y_block_max = max(min(BLOCKS_Y-1, y_block_max), -1);

        blocks_per_gaussian[i] = (x_block_max-x_block_min+1)*(y_block_max-y_block_min+1);
    }
}

__global__ void create_gaussian_instances(  
    int NUM_PRIMITIVES,
    float min_x, float min_y, 
    float range_x, float range_y,
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
        for(auto i = index; i < NUM_PRIMITIVES; i += stride){
            offset = (i == 0) ? 0 : cumulative_sums[i-1];
            float sx = scales[2*i];
            float sy = scales[2*i+1];
            float px = positions[2*i];
            float py = positions[2*i+1];
            float r = 3.0f/min(sx, sy);
            int x_block_min = BLOCKS_X*(px - r - min_x)/range_x;
            int x_block_max = BLOCKS_X*(px + r - min_x)/range_x;
            int y_block_min = BLOCKS_Y*(py - r - min_y)/range_y;
            int y_block_max = BLOCKS_Y*(py + r - min_y)/range_y;
            
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

__global__ void cumulative_items_per_block_cuda(int num_instances, int* keys, int* tile_start_end)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    if(index == 0) return;

    for(auto i = index; i < num_instances; i += stride){
        int this_key = keys[i];
        int last_key = keys[i-1];
        if(this_key != last_key){
            tile_start_end[this_key-1] = i;
        }
    }
}

__global__ void periodic_primitives_forward_cuda_kernel(  
    int num_query_points, int num_gaussians, int num_gaussian_instances,,   
    float max_frequency, bool gaussian_only, bool heatmap,
    const float* __restrict__ input,
    const float* __restrict__ colors,
    const float* __restrict__ positions,
    const float* __restrict__ scales,
    const float* __restrict__ rotations,
    const float* __restrict__ wave_coefficients,
    const int* __restrict__ wave_coefficient_indices,
    const int* __restrict__ gaussian_instance_indices,
    const int* __restrict__ cumulative_gaussians_per_block,
    const int* __restrict__ query_indices,
    const int* __restrict__ cumulative_query_points_per_block,
    float* __restrict__ output
    ) {

    // Get block/thread related numbers   
    const int threadID = threadIdx.x;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int this_block_idx = BLOCKS_X*block_y + block_x;

    int query_point_start_idx = (this_block_idx == 0) ? 0 : cumulative_query_points_per_block[this_block_idx-1];
    int query_point_end_idx = (this_block_idx == BLOCKS_X*BLOCKS_Y-1) ? num_query_points : cumulative_query_points_per_block[this_block_idx];
    int gaussian_start_idx = (this_block_idx == 0) ? 0 : cumulative_gaussians_per_block[this_block_idx-1];
    int gaussian_end_idx = (this_block_idx == BLOCKS_X*BLOCKS_Y-1) ? num_gaussian_instances : cumulative_gaussians_per_block[this_block_idx];
    
    
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
        int collect_idx = gaussian_start_idx + threadID + batch*FORWARD_NUM_THREADS;
        int idx = gaussian_instance_indices[collect_idx];
        __syncthreads();
        gaussian_positions[threadID][0] = positions[2*idx];
        gaussian_positions[threadID][1] = positions[2*idx+1];
        gaussian_scales[threadID][0] = scales[2*idx];
        gaussian_scales[threadID][1] = scales[2*idx+1];
        gaussian_colors[threadID][0] = colors[2*idx];
        gaussian_colors[threadID][1] = colors[2*idx+1];
        gaussian_colors[threadID][2] = colors[2*idx+2];
        gaussian_rotations[threadID] = rotations[idx];
        for(int i = 0; i < SELECTED_NUM_FREQUENCIES; i++){
            coefficients[threadID][i] = wave_coefficients[idx*SELECTED_NUM_FREQUENCIES+i];
            coefficient_indices[threadID][i] = wave_coefficient_indices[idx*SELECTED_NUM_FREQUENCIES+i];
        }
        __syncthreads();

        // Iterate over all query points this thread is responsible for
        // Update its value according to the currently cached gaussians
        for(int i = query_point_start_idx+threadID; i < query_point_end_idx; i += FORWARD_NUM_THREADS){
            int real_query_index = query_indices[i];
            float2 x = {input[2*real_query_index], input[2*real_query_index+1]};
            float3 temp_result = {0.0f, 0.0f, 0.0f};

            for(int j = 0; j < end_idx_this_batch; j++){

                float2 dx = x - {gaussian_positions[j][0], gaussian_positions[j][1]};
                
                float cosr = __cosf(gaussian_rotations[threadID]);
                float sinr = __sinf(gaussian_rotations[threadID]);
                float2 tx = { gaussian_scales[j][0]*(dx.x*cosr  + dx.y*sinr),
                            gaussian_scales[j][1]*(dx.x*-sinr + dx.y*cosr) };
                float g = __expf(-(tx.x*tx.x + tx.y*tx.y)/2);
                float w = 0.0f;
                for(int w_idx = 0; w_idx < SELECTED_NUM_FREQUENCIES && !gaussian_only; w_idx++){
                    float f = max_frequency*coefficient_indices[j][w_idx]/(float)TOTAL_NUM_FREQUENCIES;
                    w += coefficients[j][w_idx]*__cosf(f*g);
                }            
                if(!gaussian_only) g *= w;
                if(heatmap){
                    temp_result.x += g;
                    temp_result.y += g;
                    temp_result.z += g;
                }
                else{
                    temp_result.x += g*gaussian_colors[j][0];
                    temp_result.y += g*gaussian_colors[j][1];
                    temp_result.z += g*gaussian_colors[j][2];
                }
            }

            output[real_query_index*3] += temp_result.x;
            output[real_query_index*3+1] += temp_result.y;
            output[real_query_index*3+2] += temp_result.z;
        }      
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

int sort_query_points_to_blocks(torch::Tensor positions,
float2 min_pos, float2 max_pos, int* sorted_point_indices, int* cumulative_query_points_per_block){
    int num_points = positions.size(0);
    float2 range = max_pos-min_pos;
    
    int *unsorted_point_blocks, *unsorted_point_indices;
    point_to_block_and_index(num_points,
        min_pos, range,
        positions.contiguous().data_ptr<float>(),
        unsorted_point_blocks, unsorted_point_indices);

    int *sorted_point_blocks;

    cudaMalloc((void**)&sorted_point_blocks, num_points*sizeof(int));  
    cudaMalloc((void**)&sorted_point_indices, num_points*sizeof(int));  
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
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
    
    cudaMalloc((void**)&cumulative_query_points_per_block, (BLOCKS_X*BLOCKS_Y-1)*sizeof(int));   
    cumulative_items_per_block_cuda<<<(num_points + 512 - 1) / 512, 512>>> (
        num_points,
        sorted_point_blocks,
        cumulative_query_points_per_block
        );    
}

int sort_gaussians_to_blocks(torch::Tensor gaussian_positions, torch::Tensor gaussian_scales,
float2 min_pos, float2 max_pos, int* sorted_gaussian_indices, int* cumulative_gaussians_per_block){

    // 1. Determine the number of gaussians per block
    int num_gaussians = gaussian_positions.size(0);
    float2 range = max_pos-min_pos;
    int* blocks_per_gaussian;
    cudaMalloc((void**)&blocks_per_gaussian, num_gaussians*sizeof(int));   

    find_blocks_per_gaussian<<<(blocks_per_gaussian+512-1)/512,512>>>(  
        num_gaussians, min_pos, range
        gaussian_positions.contiguous().data_ptr<float>(),
        gaussian_scales.contiguous().data_ptr<float>(),
        blocks_per_gaussian  
        );

    /*
    // To debug
    int* local_blocks = (int*)malloc(num_primitives*sizeof(int));
    cudaMemcpy(local_blocks, blocks_per_gaussian, sizeof(int)*num_primitives, cudaMemcpyDeviceToHost);
    for(int i = 0; i < num_primitives; i++){
        std::cout << local_blocks[i] << std::endl;
    }
    free(local_blocks);
    */

    // 2. Inclusive sum on gaussians per block to find total number
    // of gaussian instances needed
    // Allocate temp storage for the inclusive sum
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    int* cumulative_sums;
    cudaMalloc((void**)&cumulative_sums, num_primitives*sizeof(int));    
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
		blocks_per_gaussian, cumulative_sums, num_primitives);    
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    std::cout << "" << std::endl;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
		blocks_per_gaussian, cumulative_sums, num_primitives);  

    /*  
    // To debug
    int* local_offsets = (int*)malloc(num_primitives*sizeof(int));
    cudaMemcpy(local_offsets, cumulative_sums, sizeof(int)*num_primitives, cudaMemcpyDeviceToHost);
    for(int i = 0; i < num_primitives; i++){
        std::cout << local_offsets[i] << std::endl;
    }
    free(local_offsets);
    */
    
    // Get the total number of gaussian instances we have on host (cpu)
    int total_gaussian_instances;
    cudaMemcpy(&total_gaussian_instances, &cumulative_sums[num_primitives-1], sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Total gaussian instances: " << total_gaussian_instances << std::endl;

    // If 0 gaussians need to be rendered, return
    if(total_gaussian_instances == 0) return total_gaussian_instances;

    // 3. Create the gaussian instances
    int *unsorted_gaussian_keys, *sorted_gaussian_keys;
    int *unsorted_gaussian_indices;
    cudaMalloc((void**)&unsorted_gaussian_keys, total_gaussian_instances*sizeof(int));   
    cudaMalloc((void**)&sorted_gaussian_keys, total_gaussian_instances*sizeof(int));   
    cudaMalloc((void**)&unsorted_gaussian_indices, total_gaussian_instances*sizeof(int));  
    cudaMalloc((void**)&sorted_gaussian_indices, total_gaussian_instances*sizeof(int));   

    create_gaussian_instances<<<(num_primitives+512-1)/512,512>>>(
        num_primitives,
        min_position, range, 
        positions.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
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
    cudaMalloc((void**)&cumulative_gaussians_per_block, (BLOCKS_X*BLOCKS_Y-1)*sizeof(int));   
    cumulative_items_per_block_cuda<<<(total_gaussian_instances + 512 - 1) / 512, 512>>> (
        total_gaussian_instances,
        sorted_gaussian_keys,
        cumulative_gaussians_per_block
        );
    
    // Only relevant memory is cumulative_gaussians_per_block and sorted_gaussian_indices.
    // Free the rest.
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
    const bool heatmap
    ) {

    // Create output tensor and other tensors
    auto output = torch::zeros({input.size(0), NUM_CHANNELS}, input.device());
    
    // Sort query points and gaussians into 16x16 blocks
    int* sorted_gaussian_indices;
    int* cumulative_gaussians_per_block;
    int num_gaussian_instances = sort_gaussians_to_blocks(
        positions, scales,
        {0.0f, 0.0f}, {1.0f, 1.0f},
        sorted_gaussian_indices, 
        cumulative_gaussians_per_block);
    if(num_gaussian_instances == 0) return output;
    
    int* sorted_query_point_indices;
    int* cumulative_query_points_per_block;
    sort_query_points_to_blocks(
        input, {0.0f, 0.0f}, {1.0f, 1.0f},
        sorted_query_point_indices, 
        cumulative_query_points_per_block);

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
        num_frequencies, max_frequency, gaussian_only, heatmap,
        input.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        positions.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        wave_coefficients.contiguous().data_ptr<float>(),
        wave_coefficient_indices.contiguous().data_ptr<int>(),
        sorted_gaussian_indices,
        cumulative_gaussians_per_block,
        sorted_query_point_indices,
        cumulative_query_points_per_block,
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
        const int blocks = (num_primitives+FORWARD_NUM_THREADS-1)/FORWARD_NUM_THREADS;
        //std::cout << dCoefficients.sizes() << std::endl;
        // Dispatch jobs
        periodic_primitives_backward_cuda_kernel<<<blocks, FORWARD_NUM_THREADS>>>(
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
