#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> hybrid_model_forward_cuda(
    torch::Tensor input,                // [N, n_dim]
    torch::Tensor gaussian_colors,      // [M, n_chan]
    torch::Tensor gaussian_means,       // [M, n_dim]
    torch::Tensor gaussian_mats,        // [M, n_dim, n_dim]
    torch::Tensor wave_colors,          // [W, n_chan]
    torch::Tensor wave_means,           // [W, n_dim]
    torch::Tensor wave_mats,            // [W, n_dim, n_dim]
    torch::Tensor wave_frequencies,     // [W, n_dim]
    torch::Tensor wave_coefficients     // [W, 5]
);

std::vector<torch::Tensor> hybrid_model_backward_cuda(
    torch::Tensor grad_output,          // [N, n_chan]
    torch::Tensor input,                // [N, n_dim]
    torch::Tensor gaussian_colors,      // [M, n_chan]
    torch::Tensor gaussian_means,       // [M, n_dim]
    torch::Tensor gaussian_mats,        // [M, n_dim, n_dim]
    torch::Tensor wave_colors,          // [W, n_chan]
    torch::Tensor wave_means,           // [W, n_dim]
    torch::Tensor wave_mats,            // [W, n_dim, n_dim]
    torch::Tensor wave_frequencies,     // [W, n_dim]
    torch::Tensor wave_coefficients     // [W, 5]
    );        

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> hybrid_model_forward(
    torch::Tensor input,                // [N, n_dims]
    torch::Tensor gaussian_colors,      // [M, n_chans]
    torch::Tensor gaussian_means,       // [M, n_dims]
    torch::Tensor gaussian_mats,        // [M, n_dims, n_dims]
    torch::Tensor wave_colors,          // [W, n_chans]
    torch::Tensor wave_means,           // [W, n_dims]
    torch::Tensor wave_mats,            // [W, n_dims, n_dims]
    torch::Tensor wave_frequencies,     // [W, n_dims]
    torch::Tensor wave_coefficients     // [W, 5]
    ) {
    CHECK_INPUT(input);
    CHECK_INPUT(gaussian_means);
    CHECK_INPUT(gaussian_mats);
    CHECK_INPUT(gaussian_colors);
    CHECK_INPUT(wave_means);
    CHECK_INPUT(wave_mats);
    CHECK_INPUT(wave_frequencies);
    CHECK_INPUT(wave_coefficients);
    CHECK_INPUT(wave_colors);

    return hybrid_model_forward_cuda(input, 
        gaussian_colors,
        gaussian_means, 
        gaussian_mats,  
        wave_colors,
        wave_means, 
        wave_mats, 
        wave_frequencies, 
        wave_coefficients
        );
}


std::vector<torch::Tensor> hybrid_model_backward(
    torch::Tensor grad_output,          // [N, n_chans]
    torch::Tensor input,                // [N, n_dims]
    torch::Tensor gaussian_colors,      // [M, n_chans]
    torch::Tensor gaussian_means,       // [M, n_dims]
    torch::Tensor gaussian_mats,        // [M, n_dims, n_dims]
    torch::Tensor wave_colors,          // [W, n_chans]
    torch::Tensor wave_means,           // [W, n_dims]
    torch::Tensor wave_mats,            // [W, n_dims, n_dims]
    torch::Tensor wave_frequencies,     // [W, n_dims]
    torch::Tensor wave_coefficients     // [W, 5]
    ) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(gaussian_means);
    CHECK_INPUT(gaussian_mats);
    CHECK_INPUT(gaussian_colors);
    CHECK_INPUT(wave_means);
    CHECK_INPUT(wave_mats);
    CHECK_INPUT(wave_frequencies);
    CHECK_INPUT(wave_coefficients);
    CHECK_INPUT(wave_colors);

    return hybrid_model_backward_cuda(
        grad_output, 
        input,
        gaussian_colors,
        gaussian_means, 
        gaussian_mats,  
        wave_colors,
        wave_means, 
        wave_mats, 
        wave_frequencies, 
        wave_coefficients
        );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &hybrid_model_forward, "Hybrid model forward (CUDA)");
  m.def("backward", &hybrid_model_backward, "Hybrid model backward (CUDA)");
}