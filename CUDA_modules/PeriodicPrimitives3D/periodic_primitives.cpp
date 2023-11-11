#include <torch/extension.h>
#include <vector>

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
);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> periodic_primitives_forward(
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
    CHECK_INPUT(rgb_colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(background_color);
    CHECK_INPUT(positions);
    CHECK_INPUT(scales);
    CHECK_INPUT(rotations);
    CHECK_INPUT(wave_coefficients);
    CHECK_INPUT(wave_coefficient_indices);
    CHECK_INPUT(cam_position);
    CHECK_INPUT(view_matrix);
    CHECK_INPUT(proj_matrix);
    
    return periodic_primitives_forward_cuda(
        rgb_colors,                   // [M, 3]
        opacities,                    // [M, 1]
        background_color,              // [3]
        positions,                    // [M, n_dim]
        scales,                       // [M, n_dim]
        scale_modifier,                        // 
        rotations,                    // [M, 4] quaternion
        wave_coefficients,            // [M, n_dim, n_freqs]
        wave_coefficient_indices,     // [M, n_dim, n_freqs]
        max_frequency,
        cam_position,                 // [3]
        view_matrix,                   // [4, 4]
        proj_matrix,                   // [4, 4]
        fov_x,
        fov_y,
        image_width,
        image_height,
        gaussian_only,
        heatmap
        );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &periodic_primitives_forward, "Periodic primitives forward (CUDA)");
}