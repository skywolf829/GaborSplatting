#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> periodic_primitives_forward_cuda(
    torch::Tensor input,                        // [N, n_dim]
    torch::Tensor colors,                       // [M, n_chan]
    torch::Tensor positions,                    // [M, n_dim]
    torch::Tensor scales,                       // [M, n_dim]
    torch::Tensor rotations,                    // [M, 1]
    torch::Tensor wave_coefficients,            // [M, n_dim, n_freqs]
    torch::Tensor wave_coefficient_indices,    
    const float max_frequency,
    const bool gaussian_only,
    const bool heatmap = false
);

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
    );        


// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> periodic_primitives_forward(
    torch::Tensor input,                        // [N, n_dim]
    torch::Tensor colors,                       // [M, n_chan]
    torch::Tensor positions,                    // [M, n_dim]
    torch::Tensor scales,                       // [M, n_dim]
    torch::Tensor rotations,                    // [M, 1]
    torch::Tensor wave_coefficients,            // [M, n_dim, n_freqs]
    torch::Tensor wave_coefficient_indices,     // [M, n_dim, n_freqs]
    const float max_frequency,
    const bool gaussian_only,
    const bool heatmap = false
    ) {
    CHECK_INPUT(input);
    CHECK_INPUT(colors);
    CHECK_INPUT(positions);
    CHECK_INPUT(scales);
    CHECK_INPUT(rotations);
    CHECK_INPUT(wave_coefficients);
    CHECK_INPUT(wave_coefficient_indices);
    
    return periodic_primitives_forward_cuda(
        input, 
        colors,
        positions, 
        scales,  
        rotations,
        wave_coefficients,
        wave_coefficient_indices,
        max_frequency,
        gaussian_only,
        heatmap
        );
}

std::vector<torch::Tensor> periodic_primitives_backward(
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
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(colors);
    CHECK_INPUT(positions);
    CHECK_INPUT(scales);
    CHECK_INPUT(rotations);
    CHECK_INPUT(wave_coefficients);
    CHECK_INPUT(wave_coefficient_indices);
    CHECK_INPUT(gaussian_instance_indices);
    CHECK_INPUT(block_start_end_index_gaussians);
    CHECK_INPUT(query_indices);
    CHECK_INPUT(block_start_end_index_query_points);
    
    return periodic_primitives_backward_cuda(
        grad_output, 
        input,
        colors,
        positions, 
        scales,  
        rotations,
        wave_coefficients,
        wave_coefficient_indices,
        gaussian_instance_indices,
        block_start_end_index_gaussians,
        query_indices,
        block_start_end_index_query_points,
        max_frequency,
        gaussian_only
        );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &periodic_primitives_forward, "Periodic primitives forward (CUDA)");
  m.def("backward", &periodic_primitives_backward, "Periodic primitives backward (CUDA)");
}