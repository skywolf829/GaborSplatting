#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> hybrid_model_forward_cuda(
    torch::Tensor input, // [N, 2]
    torch::Tensor gaussian_means, // [M, 2]
    torch::Tensor gaussian_mats, // [M, 2, 2]
    torch::Tensor gaussian_colors, // [M, 3]
    torch::Tensor wave_means, // [W, 2]
    torch::Tensor wave_mats, // [W, 2, 2]
    torch::Tensor wave_frequencies, // [W, 2]
    torch::Tensor wave_coefficients, // [W, 5]
    torch::Tensor wave_colors); // [W, 3]

std::vector<torch::Tensor> hybrid_model_backward_cuda(
    torch::Tensor input, // [N, 2]
    torch::Tensor gaussian_means, // [M, 2]
    torch::Tensor gaussian_mats, // [M, 2, 2]
    torch::Tensor gaussian_colors, // [M, 3]
    torch::Tensor wave_means, // [W, 2]
    torch::Tensor wave_mats, // [W, 2, 2]
    torch::Tensor wave_frequencies, // [W, 2]
    torch::Tensor wave_coefficients, // [W, 5]
    torch::Tensor wave_colors,    // [W, 3]
    torch::Tensor grad_gaussian_means, // [M, 2]
    torch::Tensor grad_gaussian_mats, // [M, 2, 2]
    torch::Tensor grad_gaussian_colors, // [M, 3]
    torch::Tensor grad_wave_means, // [W, 2]
    torch::Tensor grad_wave_mats, // [W, 2, 2]
    torch::Tensor grad_wave_frequencies, // [W, 2]
    torch::Tensor grad_wave_coefficients, // [W, 5]
    torch::Tensor grad_wave_colors); // [W, 3]

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> hybrid_model_forward(
    torch::Tensor input, // [N, 2]
    torch::Tensor gaussian_means, // [M, 2]
    torch::Tensor gaussian_mats, // [M, 2, 2]
    torch::Tensor gaussian_colors, // [M, 3]
    torch::Tensor wave_means, // [W, 2]
    torch::Tensor wave_mats, // [W, 2, 2]
    torch::Tensor wave_frequencies, // [W, 2]
    torch::Tensor wave_coefficients, // [W, 5]
    torch::Tensor wave_colors) {
  CHECK_INPUT(input);
  CHECK_INPUT(gaussian_means);
  CHECK_INPUT(gaussian_mats);
  CHECK_INPUT(gaussian_colors);
  CHECK_INPUT(wave_means);
  CHECK_INPUT(wave_mats);
  CHECK_INPUT(wave_frequencies);
  CHECK_INPUT(wave_coefficients);
  CHECK_INPUT(wave_colors);

  return cuda_forward(input, 
    gaussian_mean, 
    gaussian_mats, 
    gaussian_colors,
    wave_means, 
    wave_mats, 
    wave_frequencies, 
    wave_coefficients, 
    wave_colors);
}

std::vector<torch::Tensor> hybrid_model_backward(
    torch::Tensor input, // [N, 2]
    torch::Tensor gaussian_means, // [M, 2]
    torch::Tensor gaussian_mats, // [M, 2, 2]
    torch::Tensor gaussian_colors, // [M, 3]
    torch::Tensor wave_means, // [W, 2]
    torch::Tensor wave_mats, // [W, 2, 2]
    torch::Tensor wave_frequencies, // [W, 2]
    torch::Tensor wave_coefficients, // [W, 5]
    torch::Tensor wave_colors,    // [W, 3]
    torch::Tensor grad_gaussian_means, // [M, 2]
    torch::Tensor grad_gaussian_mats, // [M, 2, 2]
    torch::Tensor grad_gaussian_colors, // [M, 3]
    torch::Tensor grad_wave_means, // [W, 2]
    torch::Tensor grad_wave_mats, // [W, 2, 2]
    torch::Tensor grad_wave_frequencies, // [W, 2]
    torch::Tensor grad_wave_coefficients, // [W, 5]
    torch::Tensor grad_wave_colors) {
  CHECK_INPUT(input);
  CHECK_INPUT(gaussian_means);
  CHECK_INPUT(gaussian_mats);
  CHECK_INPUT(gaussian_colors);
  CHECK_INPUT(wave_means);
  CHECK_INPUT(wave_mats);
  CHECK_INPUT(wave_frequencies);
  CHECK_INPUT(wave_coefficients);
  CHECK_INPUT(wave_colors);
  CHECK_INPUT(grad_gaussian_means);
  CHECK_INPUT(grad_gaussian_mats);
  CHECK_INPUT(grad_gaussian_colors);
  CHECK_INPUT(grad_wave_means);
  CHECK_INPUT(grad_wave_mats);
  CHECK_INPUT(grad_wave_frequencies);
  CHECK_INPUT(grad_wave_coefficients);
  CHECK_INPUT(grad_wave_colors);

  return hybrid_model_backward_cuda(
        input,
        gaussian_means,
        gaussian_mats,
        gaussian_colors, 
        wave_means, 
        wave_mats, 
        wave_frequencies, 
        wave_coefficients,
        wave_colors,    
        grad_gaussian_means,
        grad_gaussian_mats,
        grad_gaussian_colors, 
        grad_wave_means, 
        grad_wave_mats,
        grad_wave_frequencies,
        grad_wave_coefficients,
        grad_wave_colors);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &hybrid_model_forward, "Hybrid model forward (CUDA)");
  m.def("backward", &hybrid_model_backward, "Hybrid model backward (CUDA)");
}