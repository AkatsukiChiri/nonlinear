#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

void launch_quantize_activation_int8(
    const half* x,
    signed char* quantized_x,
    float* activation_scale,
    float* square_sum,
    int batch,
    int hidden_dim,
    cudaStream_t stream);

void launch_dp4a_projection_int8(
    const signed char* quantized_x,
    const float* activation_scale,
    const signed char* quantized_weight,
    const float* weight_scale,
    const float* square_sum,
    const half* bias,
    half* output,
    int batch,
    int out_dim,
    int hidden_dim,
    float eps,
    cudaStream_t stream);

namespace {

constexpr int kGroupSize = 64;

void check_common_inputs(
    const torch::Tensor& x,
    const torch::Tensor& quantized_weight,
    const torch::Tensor& weight_scale,
    double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(quantized_weight.is_cuda(), "quantized_weight must be CUDA");
  TORCH_CHECK(weight_scale.is_cuda(), "weight_scale must be CUDA");
  TORCH_CHECK(
      x.device() == quantized_weight.device() &&
          x.device() == weight_scale.device(),
      "all inputs must be on the same device");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
  TORCH_CHECK(
      quantized_weight.scalar_type() == torch::kInt8,
      "quantized_weight must be int8");
  TORCH_CHECK(
      weight_scale.scalar_type() == torch::kFloat32,
      "weight_scale must be float32");
  TORCH_CHECK(
      x.dim() == 3 && x.size(1) == 1,
      "x must have shape [Batch, 1, HiddenDim]");
  TORCH_CHECK(
      quantized_weight.dim() == 2,
      "quantized_weight must have shape [OutDim, HiddenDim]");
  TORCH_CHECK(
      x.size(2) == quantized_weight.size(1),
      "hidden dimensions must match");
  TORCH_CHECK(
      x.size(2) % kGroupSize == 0,
      "HiddenDim must be divisible by 64");
  TORCH_CHECK(
      weight_scale.dim() == 2 &&
          weight_scale.size(0) == quantized_weight.size(0) &&
          weight_scale.size(1) == x.size(2) / kGroupSize,
      "weight_scale must have shape [OutDim, HiddenDim / 64]");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(quantized_weight.is_contiguous(), "quantized_weight must be contiguous");
  TORCH_CHECK(weight_scale.is_contiguous(), "weight_scale must be contiguous");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8, "Batch must be in [1, 8]");
  TORCH_CHECK(quantized_weight.size(0) > 0, "OutDim must be positive");
  TORCH_CHECK(eps >= 0.0, "eps must be non-negative");
}

torch::Tensor run_int8(
    torch::Tensor x,
    torch::Tensor quantized_weight,
    torch::Tensor weight_scale,
    const torch::Tensor* bias,
    double eps) {
  check_common_inputs(x, quantized_weight, weight_scale, eps);
  if (bias != nullptr) {
    TORCH_CHECK(bias->is_cuda(), "bias must be CUDA");
    TORCH_CHECK(bias->device() == x.device(), "bias must be on the same device");
    TORCH_CHECK(bias->scalar_type() == torch::kFloat16, "bias must be float16");
    TORCH_CHECK(
        bias->dim() == 1 && bias->size(0) == quantized_weight.size(0),
        "bias must have shape [OutDim]");
    TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
  }

  const c10::cuda::CUDAGuard device_guard(x.device());
  const int batch = static_cast<int>(x.size(0));
  const int hidden_dim = static_cast<int>(x.size(2));
  const int out_dim = static_cast<int>(quantized_weight.size(0));
  const int group_count = hidden_dim / kGroupSize;
  auto quantized_x = torch::empty({batch, hidden_dim}, x.options().dtype(torch::kInt8));
  auto activation_scale = torch::empty(
      {batch, group_count}, x.options().dtype(torch::kFloat32));
  auto square_sum = torch::zeros({batch}, x.options().dtype(torch::kFloat32));
  auto output = torch::empty({batch, 1, out_dim}, x.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  launch_quantize_activation_int8(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<signed char*>(quantized_x.data_ptr<int8_t>()),
      activation_scale.data_ptr<float>(),
      square_sum.data_ptr<float>(),
      batch,
      hidden_dim,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  launch_dp4a_projection_int8(
      reinterpret_cast<const signed char*>(quantized_x.data_ptr<int8_t>()),
      activation_scale.data_ptr<float>(),
      reinterpret_cast<const signed char*>(quantized_weight.data_ptr<int8_t>()),
      weight_scale.data_ptr<float>(),
      square_sum.data_ptr<float>(),
      bias == nullptr
          ? nullptr
          : reinterpret_cast<const half*>(bias->data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      batch,
      out_dim,
      hidden_dim,
      static_cast<float>(eps),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

torch::Tensor fused_rmsnorm_linear_int8(
    torch::Tensor x,
    torch::Tensor quantized_weight,
    torch::Tensor weight_scale,
    double eps) {
  return run_int8(x, quantized_weight, weight_scale, nullptr, eps);
}

torch::Tensor fused_rmsnorm_linear_bias_int8(
    torch::Tensor x,
    torch::Tensor quantized_weight,
    torch::Tensor weight_scale,
    torch::Tensor bias,
    double eps) {
  return run_int8(x, quantized_weight, weight_scale, &bias, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("fused_rmsnorm_linear_int8", &fused_rmsnorm_linear_int8);
  module.def("fused_rmsnorm_linear_bias_int8", &fused_rmsnorm_linear_bias_int8);
}
