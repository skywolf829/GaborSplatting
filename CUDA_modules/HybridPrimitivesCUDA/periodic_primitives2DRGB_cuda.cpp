#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> periodic_primitives_forward_cuda(
    torch::Tensor input,                // [N, n_dim]
    torch::Tensor colors,               // [M, n_chan]
    torch::Tensor position,             // [M, n_dim]
    torch::Tensor cov,                  // [M, n_dim, n_dim]
    torch::Tensor wave_coefficients,    // [M, n_dim, n_freqs]
    const float MAX_FREQUENCY
);

std::vector<torch::Tensor> periodic_primitives_backward_cuda(
    torch::Tensor grad_output,          // [N, n_chan]
    torch::Tensor input,                // [N, n_dim]
    torch::Tensor colors,               // [M, n_chan]
    torch::Tensor position,             // [M, n_dim]
    torch::Tensor cov,                  // [M, n_dim, n_dim]
    torch::Tensor coefficients,        // [M, n_dim, n_freqs]
    const float MAX_FREQUENCY
    );        

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> periodic_primitives_forward(
    torch::Tensor input,                // [N, n_dims]
    torch::Tensor colors,               // [M, n_chan]
    torch::Tensor position,             // [M, n_dim]
    torch::Tensor cov,                  // [M, n_dim, n_dim]
    torch::Tensor coefficients,          // [M, n_dim, n_freqs]
    const float MAX_FREQUENCY
    ) {
    CHECK_INPUT(input);
    CHECK_INPUT(colors);
    CHECK_INPUT(position);
    CHECK_INPUT(cov);
    CHECK_INPUT(coefficients);

    return periodic_primitives_forward_cuda(
        input, 
        colors,
        position, 
        cov,  
        coefficients,
        MAX_FREQUENCY
        );
}


std::vector<torch::Tensor> periodic_primitives_backward(
    torch::Tensor grad_output,          // [N, n_chans]
    torch::Tensor input,                // [N, n_dims]
    torch::Tensor colors,               // [M, n_chan]
    torch::Tensor position,             // [M, n_dim]
    torch::Tensor cov,                  // [M, n_dim, n_dim]
    torch::Tensor coefficients,         // [M, n_dim, n_freqs]
    const float MAX_FREQUENCY
    ) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(colors);
    CHECK_INPUT(position);
    CHECK_INPUT(cov);
    CHECK_INPUT(coefficients);

    return periodic_primitives_backward_cuda(
        grad_output, 
        input,
        colors,
        position, 
        cov,  
        coefficients,
        MAX_FREQUENCY
        );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &periodic_primitives_forward, "Periodic primitives forward (CUDA)");
  m.def("backward", &periodic_primitives_backward, "Periodic primitives backward (CUDA)");
}