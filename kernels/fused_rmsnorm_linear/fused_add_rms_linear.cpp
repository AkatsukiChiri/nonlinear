#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <vector>

void launch_fused_add_rms_linear_fp16(
    const half* x,
    const half* x_res,
    const half* weight,
    const half* bias,
    half* output,
    half* residual_sum,
    int batch,
    int out_dim,
    int hidden_dim,
    float eps,
    cudaStream_t stream);

namespace {

void validate_inputs(
    const torch::Tensor& x,
    const torch::Tensor& x_res,
    const torch::Tensor& fused_weight,
    const torch::Tensor* bias,
    double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x_res.is_cuda(), "x_res must be a CUDA tensor");
  TORCH_CHECK(fused_weight.is_cuda(), "fused_weight must be a CUDA tensor");
  TORCH_CHECK(
      x.device() == x_res.device() && x.device() == fused_weight.device(),
      "x, x_res, and fused_weight must be on one device");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
  TORCH_CHECK(x_res.scalar_type() == torch::kFloat16, "x_res must be float16");
  TORCH_CHECK(
      fused_weight.scalar_type() == torch::kFloat16,
      "fused_weight must be float16");
  TORCH_CHECK(
      x.dim() == 3 && x.size(1) == 1,
      "x must have shape [Batch, 1, HiddenDim]");
  TORCH_CHECK(
      x_res.sizes() == x.sizes(),
      "x_res must have the same shape as x");
  TORCH_CHECK(
      fused_weight.dim() == 2,
      "fused_weight must have shape [OutDim, HiddenDim]");
  TORCH_CHECK(
      x.size(2) == fused_weight.size(1),
      "hidden dimensions must match");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x_res.is_contiguous(), "x_res must be contiguous");
  TORCH_CHECK(fused_weight.is_contiguous(), "fused_weight must be contiguous");
  TORCH_CHECK(
      x.size(0) >= 1 && x.size(0) <= 8,
      "the optimized kernel supports Batch in [1, 8]");
  TORCH_CHECK(
      x.size(2) > 0 && fused_weight.size(0) > 0,
      "all dimensions must be positive");
  TORCH_CHECK(eps >= 0.0, "eps must be non-negative");

  if (bias != nullptr) {
    TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias->device() == x.device(), "bias must be on the input device");
    TORCH_CHECK(
        bias->scalar_type() == torch::kFloat16,
        "bias must be float16");
    TORCH_CHECK(
        bias->dim() == 1 && bias->size(0) == fused_weight.size(0),
        "bias must have shape [OutDim]");
    TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
  }
}

std::vector<torch::Tensor> run_fused_add_rms_linear(
    torch::Tensor x,
    torch::Tensor x_res,
    torch::Tensor fused_weight,
    const torch::Tensor* bias,
    double eps) {
  validate_inputs(x, x_res, fused_weight, bias, eps);
  const c10::cuda::CUDAGuard device_guard(x.device());
  const int batch = static_cast<int>(x.size(0));
  const int hidden_dim = static_cast<int>(x.size(2));
  const int out_dim = static_cast<int>(fused_weight.size(0));
  auto output = torch::empty({batch, 1, out_dim}, x.options());
  auto residual_sum = torch::empty_like(x);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  launch_fused_add_rms_linear_fp16(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(x_res.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(fused_weight.data_ptr<at::Half>()),
      bias == nullptr
          ? nullptr
          : reinterpret_cast<const half*>(bias->data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      reinterpret_cast<half*>(residual_sum.data_ptr<at::Half>()),
      batch,
      out_dim,
      hidden_dim,
      static_cast<float>(eps),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, residual_sum};
}

}  // namespace

std::vector<torch::Tensor> fused_add_rms_linear_fp16(
    torch::Tensor x,
    torch::Tensor x_res,
    torch::Tensor fused_weight,
    double eps) {
  return run_fused_add_rms_linear(
      std::move(x),
      std::move(x_res),
      std::move(fused_weight),
      nullptr,
      eps);
}

std::vector<torch::Tensor> fused_add_rms_linear_bias_fp16(
    torch::Tensor x,
    torch::Tensor x_res,
    torch::Tensor fused_weight,
    torch::Tensor bias,
    double eps) {
  return run_fused_add_rms_linear(
      std::move(x),
      std::move(x_res),
      std::move(fused_weight),
      &bias,
      eps);
}
