#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#define FORWARD_NUM_THREADS 512
#define TOTAL_NUM_FREQUENCIES 128
#define SELECTED_NUM_FREQUENCIES 4
#define NUM_CHANNELS 3
#define NUM_DIMENSIONS 3
#define BLOCKS_X 16
#define BLOCKS_Y 16

__forceinline__ __device__ float fov2focal(float fov, int pixels){
    return pixels / (2.0f * __tanf(fov / 2.0f));
}

__forceinline__ __device__ float focal2fov(float focal, int pixels){
    return 2.0f*atanf(pixels/(2.0f*focal));
}

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

__forceinline__ __device__ float ndc2Pix(float pos, int pix_total)
{
	return ((pos + 1.0) * pix_total - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 pos, int max_radius, 
    const int width, const int height,
    int2& rect_min, int2& rect_max)
{
	rect_min = {
		min(BLOCKS_X, max((int)-1, (int)(((pos.x - max_radius)/width) * (BLOCKS_X-1)))),
		min(BLOCKS_Y, max((int)-1, (int)(((pos.y - max_radius)/height)* (BLOCKS_Y-1))))
	};
	rect_max = {
		min(BLOCKS_X, max((int)-1, (int)(((pos.x + max_radius)/width) * (BLOCKS_X-1)))),
		min(BLOCKS_Y, max((int)-1, (int)(((pos.y + max_radius)/height)* (BLOCKS_Y-1))))
	};
}
__forceinline__ __device__ int rectToNumBlocks(int2 rect_min, int2 rect_max)
{
	if(rect_min.x == BLOCKS_X || rect_min.y == BLOCKS_Y || rect_max.x == -1 || rect_max.y == -1) return 0;
    return (rect_max.x - rect_min.x + 1) * (rect_max.y - rect_min.y + 1);
}
__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__device__ float3 operator*(const float a, const float3 &b) {
    return make_float3(a*b.x, a*b.y, a*b.z);
}


__device__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ void operator+=(float3 &a, const float b) {
    a.x += b;
    a.y += b;
    a.z += b;
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


/*
    Taken from Gaussian Splatting
*/
__forceinline__ __device__ bool in_frustum(int idx,
	const float* positions,
	const float* view_matrix,
	const float* proj_matrix,
	float3& p_view)
{
	float3 p_orig = { positions[3 * idx], positions[3 * idx + 1], positions[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, proj_matrix);
	//float p_w = 1.0f / (p_hom.w + 0.0000001f);
	//float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, view_matrix);

	return p_view.z > 0.2f; // || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
}


/*
    Taken from Gaussian Splatting
*/
__global__ void check_visibility_cuda(int num_primitives,
    const float* positions,
    const float* view_matrix,
    const float* proj_matrix,
    bool* visible)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for(int i = index; i < num_primitives; i += stride){
        float3 p_view;
        visible[i] = in_frustum(i, positions, view_matrix, proj_matrix, p_view);
    }
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void check_visibility(
	int num_primitives,
    float* positions,
    float* view_matrix,
    float* proj_matrix,
    bool* visible)
{
	check_visibility_cuda <<<(num_primitives + 511) / 512, 512>>> (
		num_primitives,
        positions,
        view_matrix,
        proj_matrix,
        visible);
}


// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
// Taken from Gaussian Splatting code.
__device__ void computeCov3D(const float3 scale, float mod, 
    const float4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	float r = rot.x;
	float x = rot.y;
	float y = rot.z;
	float z = rot.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Forward version of 2D covariance matrix computation
// Taken from Gaussian Splatting paper
__device__ float3 computeCov2D(const float3& position, 
    float focal_x, float focal_y, 
    float tan_fovx, float tan_fovy, 
    const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(position, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__global__ void preprocess_primitives_cuda(  
    const int num_primitives,
    const int width,
    const int height,
    const float* __restrict__ positions, 
    const float* __restrict__ scales, 
    const float scale_modifier,
    const float* __restrict__ rotations,
    const float* __restrict__ opacities,
    const float* __restrict__ cam_position,
    const float* __restrict__ view_matrix,
    const float* __restrict__ proj_matrix,
    const float fov_x,
    const float fov_y,
    float2* __restrict__ positions_2d,
    float* __restrict__ depths,
    float* __restrict__ radii,
    float4* __restrict__ covariance_opacity_2d,    
    int* __restrict__ blocks_per_gaussian
    ) {
        // Get block/thread related numbers   
        auto index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for(auto i = index; i < num_primitives; i += stride){
            float3 p_orig = { positions[3*i], positions[3*i+1], positions[3*i+2] };
            float4 p_hom = transformPoint4x4(p_orig, proj_matrix);
            float p_w = 1.0f / (p_hom.w + 0.0000001f);
            float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
            float3 p_view = transformPoint4x3(p_orig, view_matrix);
            if(p_view.z <= 0.2) continue; // near clipping plane

            
            float3 scale = {scales[3*i], scales[3*i+1], scales[3*i+2]};
            float4 r = {rotations[4*i], rotations[4*i+1], 
                rotations[4*i+2], rotations[4*i+3]};
            float cov3D[6] = {};
            computeCov3D(scale, scale_modifier, r, cov3D);
            // Compute 2D screen-space covariance matrix
	        float3 cov = computeCov2D(p_orig, 
                fov2focal(fov_x, width), fov2focal(fov_y, height), 
                __tanf(fov_x * 0.5f), __tanf(fov_y * 0.5f), 
                cov3D, view_matrix);

            // Invert covariance (EWA algorithm)
            float det = (cov.x * cov.z - cov.y * cov.y);
            if (det == 0.0f){
                blocks_per_gaussian[i] = 0;
                continue;
            }
            float det_inv = 1.f / det;
            float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

            // Compute extent in screen space (by finding eigenvalues of
            // 2D covariance matrix). Use extent to compute a bounding rectangle
            // of screen-space tiles that this Gaussian overlaps with. Quit if
            // rectangle covers 0 tiles. 
            float mid = 0.5f * (cov.x + cov.z);
            float lambda1 = mid + sqrtf(max(0.1f, mid * mid - det));
            float lambda2 = mid - sqrtf(max(0.1f, mid * mid - det));
            float my_radius = ceil(3.f * sqrtf(max(lambda1, lambda2)));
            float2 point_image = { ndc2Pix(p_proj.x, width), ndc2Pix(p_proj.y, height) };
            int2 rect_min, rect_max;
            getRect(point_image, my_radius, width, height, rect_min, rect_max);
            int num_blocks = rectToNumBlocks(rect_min, rect_max);
            blocks_per_gaussian[i] = num_blocks;
            if (num_blocks == 0)
                continue;

            // Store some useful helper data for the next steps.
            depths[i] = p_view.z;
            radii[i] = my_radius;
            positions_2d[i] = point_image;
            // Inverse 2D covariance and opacity neatly pack into one float4
            covariance_opacity_2d[i] = {conic.x, conic.y, conic.z, opacities[i]};
        }
    }


__global__ void create_primitive_instances(  
    const int num_primitives, 
    const int width, 
    const int height,
    const float2* __restrict__ positions_2d,
    const float4* __restrict__ covariance_opacity_2d,
    const float* __restrict__ depths,
    const float* __restrict__ radii,
    uint32_t* __restrict__ cumulative_sums,
    uint64_t* __restrict__ unsorted_gaussian_keys,
    uint32_t* __restrict__ unsorted_gaussian_indices
    ) {
        // Get block/thread related numbers   
        auto index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        uint32_t offset = 0;
        for(auto i = index; i < num_primitives; i += stride){
            offset = (i == 0) ? 0 : cumulative_sums[i-1];
            float r = radii[i];
            float2 px = positions_2d[i];
            float d = depths[i];
            int2 rect_min, rect_max;
            getRect(px, r, width, height, rect_min, rect_max);
            
            for (int x = rect_min.x; x <= rect_max.x && x < BLOCKS_X; x++){
                for (int y = rect_min.y; y <= rect_max.y && y < BLOCKS_Y; y++){
                    uint64_t key = (y*BLOCKS_X+x);
                    key <<= 32;
                    key |= (uint32_t)d;
                    unsorted_gaussian_keys[offset] = key;
                    unsorted_gaussian_indices[offset] = i;
                    offset++;
                }
            }
        }
    }

__global__ void key_start_end_indices_cuda(uint32_t num_instances, 
    uint64_t* keys, uint32_t* tile_start_end)
{
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    auto stride = blockDim.x * gridDim.x;

    for(auto i = index; i < num_instances; i += stride){
        uint32_t this_key = (uint32_t)(keys[i] >> 32);

        if(i > 0){       
            uint32_t last_key = (uint32_t)(keys[i-1] >> 32);
            if(this_key != last_key){
                tile_start_end[2*last_key+1] = i;
                tile_start_end[2*this_key] = i;
            }
        }
        if(i < num_instances-1){            
            uint32_t next_key = (uint32_t)(keys[i+1] >> 32);
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

__global__ void __launch_bounds__(BLOCKS_X * BLOCKS_Y)
periodic_primitives_forward_cuda_kernel(  
    const int num_primitives,
    const uint32_t num_primitive_instances,   
    const float max_frequency, 
    const bool gaussian_only, 
    const bool heatmap,
    const int width, const int height,
    const float* __restrict__ colors,
    const float2* __restrict__ positions_2d,
    const float4* __restrict__ covariance_opacity_2d,
    const float* __restrict__ wave_coefficients,
    const int* __restrict__ wave_coefficient_indices,
    const uint32_t* __restrict__ primitive_instance_indices,
    const uint32_t* __restrict__ block_start_end_index,
    float* __restrict__ final_T,
	int* __restrict__ n_contributors,
    float* __restrict__ output
    ) {

    // Get block/thread related numbers   
    const int threadID = threadIdx.x;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int this_block_idx = BLOCKS_X*block_y + block_x;

    uint32_t primitive_start_idx = block_start_end_index[2*this_block_idx];
    uint32_t primitive_end_idx = block_start_end_index[2*this_block_idx+1];

    // return if no query points or gaussians in this block
    if(primitive_start_idx == primitive_end_idx) return;

    __shared__ float2 primitive_positions[FORWARD_NUM_THREADS];
    __shared__ float4 primitive_covariance_opacity[FORWARD_NUM_THREADS];
    __shared__ float3 primitive_colors[FORWARD_NUM_THREADS];
    __shared__ float coefficients[FORWARD_NUM_THREADS][SELECTED_NUM_FREQUENCIES];
    __shared__ int coefficient_indices[FORWARD_NUM_THREADS][SELECTED_NUM_FREQUENCIES];
    
    int num_point_batchs = 1 + (primitive_end_idx - primitive_start_idx) 
        / FORWARD_NUM_THREADS;
    int x_pix_per_block = ceil((float)width / BLOCKS_X);
    int y_pix_per_block = ceil((float)height / BLOCKS_Y);

    int2 top_left = make_int2(x_pix_per_block*block_x, 
        y_pix_per_block*block_y);
    int2 bot_right = make_int2(min(width, x_pix_per_block*(block_x+1)),
        min(height, y_pix_per_block*(block_y+1)));
    int2 block_size = make_int2(bot_right.x - top_left.x,
        bot_right.y - top_left.y);
    int pix_in_block = block_size.x*block_size.y;

    
    for(int batch = 0; batch < num_point_batchs; batch++){

        uint32_t end_idx_this_batch = min(FORWARD_NUM_THREADS, 
            primitive_end_idx-primitive_start_idx-batch*FORWARD_NUM_THREADS);

        // Each thread loads a part of global memory to shared (random reads)
        uint32_t collect_idx = primitive_start_idx + batch*FORWARD_NUM_THREADS 
            + threadID;
        __syncthreads();
        if(collect_idx < num_primitive_instances){
            uint32_t idx = primitive_instance_indices[collect_idx];
            primitive_positions[threadID] = positions_2d[idx];
            primitive_covariance_opacity[threadID] = covariance_opacity_2d[idx];
            primitive_colors[threadID] = {colors[3*idx], colors[3*idx+1], colors[3*idx+2]};
            for(int i = 0; i < SELECTED_NUM_FREQUENCIES && !gaussian_only; i++){
                coefficients[threadID][i] = wave_coefficients[idx*SELECTED_NUM_FREQUENCIES+i];
                coefficient_indices[threadID][i] = wave_coefficient_indices[idx*SELECTED_NUM_FREQUENCIES+i];
            }
        }
        __syncthreads();
        // Iterate over all query points this thread is responsible for
        // Update its value according to the currently cached gaussians
        for(int i = threadID; i < pix_in_block; i += FORWARD_NUM_THREADS){
            int x = i % block_size.x;
            int y = i / block_size.x;
            uint32_t pix_id = width * y + x;
            
            float T = 1.0f;
            int last_contributor = -1;
            if(batch > 0) T = final_T[pix_id]; 
            
            float3 temp_result = {0.0f, 0.0f, 0.0f};
            for(int j = 0; j < end_idx_this_batch && T > 0.0001f; j++){

                float2 dx = make_float2(x,y) - primitive_positions[j];
                float4 cov_opacity = primitive_covariance_opacity[j];
                float3 c = primitive_colors[j];
                
                float w = 0.0f;
                for(int w_idx = 0; w_idx < SELECTED_NUM_FREQUENCIES && !gaussian_only; w_idx++){
                    float f = max_frequency*(coefficient_indices[j][w_idx])/(float)TOTAL_NUM_FREQUENCIES;
                    // TODO : Add direction to modulate wave in!
                    w += coefficients[j][w_idx]*__cosf(f);
                }            
                float power = -0.5f * (cov_opacity.x * dx.x * dx.x + 
                    cov_opacity.z * dx.y * dx.y) - cov_opacity.y * dx.x * dx.y;
                if (power > 0.0f) continue;

                float alpha = min(0.99f, cov_opacity.w * __expf(power));
                if (alpha < 1.0f / 500.0f) continue;

                float test_T = T * (1 - alpha);

                if(!gaussian_only) alpha *= w;
                if(heatmap) temp_result += alpha;                
                else temp_result += (alpha * T) * c;
                
                T = test_T;
			    last_contributor = batch*FORWARD_NUM_THREADS+j;
            }
            // will be greater than -1 if at least one gaussian affected this pixel
            // in this batch
            if(last_contributor >= 0){
                n_contributors[pix_id] = last_contributor;
                final_T[pix_id] = T;
                output[0*height*width+pix_id] += temp_result.x;
                output[1*height*width+pix_id] += temp_result.y;
                output[2*height*width+pix_id] += temp_result.z;
            }
        }      
    }
}

uint32_t preprocess_primitives(
    const int width, const int height,
    const torch::Tensor& positions, 
    const torch::Tensor& scales, 
    const float scale_modifier,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& cam_position,
    const torch::Tensor& view_matrix,
    const torch::Tensor& proj_matrix,
    const float fov_x,
    const float fov_y,
    float2* positions_2d,
    float* depths,
    float* radii,
    float4* covariance_opacity_2d,
    uint32_t* sorted_primitive_indices, 
    uint32_t* blocks_start_end_indices){

    // 1. Project data to 2D screen space and find num gaussians per block
    int num_primitives = positions.size(0);
    int* blocks_per_primitive;
    cudaMalloc((void**)&blocks_per_primitive, num_primitives*sizeof(int));   

    preprocess_primitives_cuda<<<(num_primitives+512-1)/512,512>>>(
        num_primitives, width, height,
        positions.contiguous().data_ptr<float>(), 
        scales.contiguous().data_ptr<float>(), 
        scale_modifier,
        rotations.contiguous().data_ptr<float>(),
        cam_position.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        view_matrix.contiguous().data_ptr<float>(),
        proj_matrix.contiguous().data_ptr<float>(),
        fov_x,
        fov_y,
        positions_2d,
        depths,
        radii,
        covariance_opacity_2d,    
        blocks_per_primitive
        );

    // 2. Inclusive sum on primitives per block to find total number
    // of primivite instances needed
    // Allocate temp storage for the inclusive sum
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    uint32_t* cumulative_sums;
    cudaMalloc((void**)&cumulative_sums, num_primitives*sizeof(uint32_t));    
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
		blocks_per_primitive, cumulative_sums, num_primitives);    
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
		blocks_per_primitive, cumulative_sums, num_primitives);  

    
    // Get the total number of primitive instances we have on host (cpu)
    uint32_t total_primitive_instances;
    cudaMemcpy(&total_primitive_instances, &cumulative_sums[num_primitives-1], 
        sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //std::cout << "Total primitive instances: " << total_primitive_instances << std::endl;

    // If 0 primitives need to be rendered, return
    if(total_primitive_instances == 0) return total_primitive_instances;

    // 3. Create the primitive instances
    uint64_t *unsorted_primitive_keys, *sorted_primitive_keys;
    uint32_t *unsorted_primitive_indices;
    cudaMalloc((void**)&unsorted_primitive_keys, total_primitive_instances*sizeof(uint64_t));   
    cudaMalloc((void**)&sorted_primitive_keys, total_primitive_instances*sizeof(uint64_t));   
    cudaMalloc((void**)&unsorted_primitive_indices, total_primitive_instances*sizeof(uint32_t));  
    cudaMalloc((void**)&sorted_primitive_indices, total_primitive_instances*sizeof(uint32_t));   

    create_primitive_instances<<<(num_primitives+512-1)/512,512>>>(
        num_primitives, width, height,
        positions_2d,
        covariance_opacity_2d, 
        depths,
        radii,
        cumulative_sums,
        unsorted_primitive_keys,
        unsorted_primitive_indices
    );

    // 4. Sort the primitive instances by keys (tileID - depth)
    // get MSB for the # tiles (bits up to 16x16) to only sort 
    // 32+msb bits instead of full 64 bits
    uint32_t bit = getHigherMsb((uint32_t)(BLOCKS_X * BLOCKS_Y));
    cudaFree(d_temp_storage);
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
		d_temp_storage,
		temp_storage_bytes,
		unsorted_primitive_keys, sorted_primitive_keys,
		unsorted_primitive_indices, sorted_primitive_indices,
		total_primitive_instances, 0, 32+bit);

    // Then actually sort
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortPairs(
		d_temp_storage,
		temp_storage_bytes,
		unsorted_primitive_keys, sorted_primitive_keys,
		unsorted_primitive_indices, sorted_primitive_indices,
		total_primitive_instances, 0, 32+bit);

    // 5. Identify index start/end index for primitives in each tile 
    cudaMalloc((void**)&blocks_start_end_indices, 2*(BLOCKS_X*BLOCKS_Y)*sizeof(uint32_t));   
    cudaMemset(blocks_start_end_indices, 0, 2*(BLOCKS_X*BLOCKS_Y)*sizeof(uint32_t));
    key_start_end_indices_cuda<<<(total_primitive_instances + 512 - 1) / 512, 512>>> (
        total_primitive_instances,
        sorted_primitive_keys,
        blocks_start_end_indices
        );

    // Only relevant memory is blocks_start_end_indices and sorted_gaussian_indices.
    // Free the rest.
    cudaFree(blocks_per_primitive);
    cudaFree(d_temp_storage);
    cudaFree(unsorted_primitive_indices);
    cudaFree(unsorted_primitive_keys);
    cudaFree(sorted_primitive_keys);
    cudaFree(cumulative_sums);
    return total_primitive_instances;
}

std::vector<torch::Tensor> periodic_primitives_forward_cuda(
    const torch::Tensor& rgb_colors,                   // [M, 3]
    const torch::Tensor& opacities,                    // [M, 1]
    const torch::Tensor& background_color,              // [3]
    const torch::Tensor& positions,                    // [M, n_dim]
    const torch::Tensor& scales,                       // [M, n_dim]
    const float scale_modifier,                        // 
    const torch::Tensor& rotations,                    // [M, 4] quaternion
    const torch::Tensor& wave_coefficients,            // [M, n_dim, n_freqs]
    const torch::Tensor& wave_coefficient_indices,     // [M, n_dim, n_freqs]
    const float max_frequency,
    const torch::Tensor& cam_position,                 // [3]
	const torch::Tensor& view_matrix,                   // [4, 4]
	const torch::Tensor& proj_matrix,                   // [4, 4]
    const float fov_x,
    const float fov_y,
    const int image_width,
    const int image_height,
    const bool gaussian_only,
    const bool heatmap = false
    ) {
        
    // Create output tensor and other tensors
    auto output = torch::zeros({NUM_CHANNELS, image_height, image_width}, positions.device());
    
    float2 *positions_2d;
    float *depths;
    float *radii;
    float4 *covariance_opacity_2d;
    cudaMalloc((void **)&positions_2d,sizeof(float2)*positions.size(0));
    cudaMalloc((void **)&depths,sizeof(float)*positions.size(0));
    cudaMalloc((void **)&radii,sizeof(float)*positions.size(0));
    cudaMalloc((void **)&covariance_opacity_2d,sizeof(float4)*positions.size(0));

    // Sort query points and primitives into 16x16 blocks
    uint32_t* sorted_primitive_indices;
    uint32_t* blocks_start_end_indices;
    uint32_t num_primitive_instances = preprocess_primitives(
        image_width, image_height,
        positions, 
        scales, 
        scale_modifier,
        rotations,
        opacities,
        cam_position,
        view_matrix,
        proj_matrix,
        fov_x,
        fov_y,
        positions_2d,
        depths, 
        radii,
        covariance_opacity_2d,
        sorted_primitive_indices, 
        blocks_start_end_indices);
    
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor sorted_primitive_indices_tensor = torch::from_blob(sorted_primitive_indices, 
        {num_primitive_instances}, options_int).clone();
    torch::Tensor blocks_start_end_indices_tensor = torch::from_blob(blocks_start_end_indices, 
        {BLOCKS_X*BLOCKS_Y, 2}, options_int).clone();
   
    if(num_primitive_instances == 0) return {output, 
        sorted_primitive_indices_tensor, 
        blocks_start_end_indices_tensor};
    

    // Now sorted_gaussian_indices orders the indices of the original gaussian
    // tensors in block order, so items are in block [0, 0, ..., 0, 1, 1, ..., 1, 2, ...]
    // Similar with sorted_query_point_indices.
    // cumulative_gaussians_per_block and cumulative_query_points_per_block are the
    // indices for which block 0->1 (so each thread block knows where to stop)

    // Finally evaluate results such that query points only evaulate with gaussians
    // within the block.
    auto final_T = torch::zeros({image_height, image_width}, options_float);
    auto num_contributors = torch::zeros({image_height, image_width}, options_int);
    dim3 numBlocks (16, 16);
    periodic_primitives_forward_cuda_kernel<<<numBlocks, FORWARD_NUM_THREADS>>>(
        positions.size(0), num_primitive_instances,
        max_frequency, gaussian_only, heatmap,
        image_width, image_height,
        rgb_colors.contiguous().data_ptr<float>(),
        positions_2d,
        covariance_opacity_2d,
        wave_coefficients.contiguous().data_ptr<float>(),
        wave_coefficient_indices.contiguous().data_ptr<int>(),
        sorted_primitive_indices,
        blocks_start_end_indices,        
        final_T.contiguous().data_ptr<float>(),
        num_contributors.contiguous().data_ptr<int>(),
        output.contiguous().data_ptr<float>()
        );
    cudaFree(sorted_primitive_indices);
    cudaFree(blocks_start_end_indices);
    return {output,
        sorted_primitive_indices_tensor,
        blocks_start_end_indices_tensor};
}


