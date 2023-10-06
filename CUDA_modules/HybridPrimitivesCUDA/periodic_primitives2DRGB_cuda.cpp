#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> periodic_primitives_forward_cuda(
    torch::Tensor input,                        // [N, n_dim]
    torch::Tensor colors,                       // [M, n_chan]
    torch::Tensor positions,                    // [M, n_dim]
    torch::Tensor scales,                       // [M, n_dim]
    torch::Tensor rotations,                    // [M, 1]
    torch::Tensor wave_coefficients,            // [M, n_dim, n_freqs]
    torch::Tensor wave_coefficient_indices,     // [M, n_dim, n_freqs]
    const float MAX_FREQUENCY
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
    const float MAX_FREQUENCY
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
    const float MAX_FREQUENCY
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
        MAX_FREQUENCY
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
    const float MAX_FREQUENCY
    ) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(colors);
    CHECK_INPUT(positions);
    CHECK_INPUT(scales);
    CHECK_INPUT(rotations);
    CHECK_INPUT(wave_coefficients);
    CHECK_INPUT(wave_coefficient_indices);

    return periodic_primitives_backward_cuda(
        grad_output, 
        input,
        colors,
        positions, 
        scales,  
        rotations,
        wave_coefficients,
        wave_coefficient_indices,
        MAX_FREQUENCY
        );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &periodic_primitives_forward, "Periodic primitives forward (CUDA)");
  m.def("backward", &periodic_primitives_backward, "Periodic primitives backward (CUDA)");
}