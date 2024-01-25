#include <torch/extension.h>
#include <vector>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::vector<torch::Tensor> periodic_primitives_forward_cuda(
    const torch::Tensor& input,                        // [N, n_dim]
    const torch::Tensor& colors,                       // [M, n_chan]
    const torch::Tensor& positions,                    // [M, n_dim]
    const torch::Tensor& scales,                       // [M, n_dim]
    const torch::Tensor& rotations,                    // [M, 1]
    const torch::Tensor& wave_coefficients,            // [M, n_dim, n_freqs]
    const torch::Tensor& wave_coefficient_indices,    
    const float max_frequency,
    const bool gaussian_only,
    const bool heatmap = false
);

std::vector<torch::Tensor> periodic_primitives_backward_cuda(
    const torch::Tensor& grad_output,                  // [N, n_chan]
    const torch::Tensor& input,                        // [N, n_dim]
    const torch::Tensor& colors,                       // [M, n_chan]
    const torch::Tensor& positions,                    // [M, n_dim]
    const torch::Tensor& scales,                       // [M, n_dim, n_dim]
    const torch::Tensor& rotations,                    // [M, n_dim, n_freqs]
    const torch::Tensor& wave_coefficients,            // [M, n_dim, n_freqs]
    const torch::Tensor& wave_coefficient_indices,     // [M, n_dim, n_freqs]
    const torch::Tensor& gaussian_instance_indices,
    const torch::Tensor& block_start_end_index_gaussians,
    const torch::Tensor& query_indices,
    const torch::Tensor& block_start_end_index_query_points,
    const float max_frequency,    
    const bool gaussian_only
    );        


// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> periodic_primitives_forward(
    const torch::Tensor& input,                        // [N, n_dim]
    const torch::Tensor& colors,                       // [M, n_chan]
    const torch::Tensor& positions,                    // [M, n_dim]
    const torch::Tensor& scales,                       // [M, n_dim]
    const torch::Tensor& rotations,                    // [M, 1]
    const torch::Tensor& wave_coefficients,            // [M, n_dim, n_freqs]
    const torch::Tensor& wave_coefficient_indices,     // [M, n_dim, n_freqs]
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
    /*
    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);

    torch::Tensor queryPointBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> queryPointBufferFunct = resizeFunctional(queryPointBuffer);

    torch::Tensor gaussiansBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> gaussiansBufferFunct = resizeFunctional(gaussiansBuffer);
    */
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
    const torch::Tensor& grad_output,                  // [N, n_chan]
    const torch::Tensor& input,                        // [N, n_dim]
    const torch::Tensor& colors,                       // [M, n_chan]
    const torch::Tensor& positions,                    // [M, n_dim]
    const torch::Tensor& scales,                       // [M, n_dim, n_dim]
    const torch::Tensor& rotations,                    // [M, n_dim, n_freqs]
    const torch::Tensor& wave_coefficients,            // [M, n_dim, n_freqs]
    const torch::Tensor& wave_coefficient_indices,     // [M, n_dim, n_freqs]
    const torch::Tensor& gaussian_instance_indices,
    const torch::Tensor& block_start_end_index_gaussians,
    const torch::Tensor& query_indices,
    const torch::Tensor& block_start_end_index_query_points,
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