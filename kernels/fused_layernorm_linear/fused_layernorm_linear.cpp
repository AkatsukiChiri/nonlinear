#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

void launch_layernorm_projection_stats_fp16(
    const half* x,
    const half* weight,
    float* projection,
    float* input_sum,
    float* input_square_sum,
    int batch,
    int out_dim,
    int hidden_dim,
    cudaStream_t stream);

void launch_layernorm_epilogue_fp16(
    const float* projection,
    const float* input_sum,
    const float* input_square_sum,
    const half* weight_sum,
    const half* fused_bias,
    half* output,
    int batch,
    int out_dim,
    int hidden_dim,
    float eps,
    cudaStream_t stream);

void launch_layernorm_stats_expand_fp16(
    const half* x,
    half* expanded_x,
    float* input_sum,
    float* input_square_sum,
    int batch,
    int hidden_dim,
    cudaStream_t stream);

void launch_layernorm_wmma_output_fp16(
    const half* expanded_x,
    const half* weight,
    const float* input_sum,
    const float* input_square_sum,
    const float* weight_sum,
    const float* fused_bias,
    half* output,
    int batch,
    int out_dim,
    int hidden_dim,
    float eps,
    cudaStream_t stream);

cudaError_t launch_cooperative_layernorm_linear_fp16(
    const half* x,
    half* packed_x,
    const half* weight,
    const half* weight_sum,
    const half* fused_bias,
    half* output,
    float* statistics,
    int batch,
    int out_dim,
    int hidden_dim,
    int multiprocessor_count,
    float eps,
    cudaStream_t stream);

torch::Tensor fused_layernorm_linear_fp16(
    torch::Tensor x,
    torch::Tensor fused_weight,
    torch::Tensor weight_sum,
    torch::Tensor fused_bias,
    double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(fused_weight.is_cuda(), "fused_weight must be a CUDA tensor");
  TORCH_CHECK(weight_sum.is_cuda(), "weight_sum must be a CUDA tensor");
  TORCH_CHECK(fused_bias.is_cuda(), "fused_bias must be a CUDA tensor");
  TORCH_CHECK(
      x.device() == fused_weight.device() &&
          x.device() == weight_sum.device() &&
          x.device() == fused_bias.device(),
      "all tensors must be on one device");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
  TORCH_CHECK(
      fused_weight.scalar_type() == torch::kFloat16,
      "fused_weight must be float16");
  TORCH_CHECK(
      weight_sum.scalar_type() == torch::kFloat16,
      "weight_sum must be float16");
  TORCH_CHECK(
      fused_bias.scalar_type() == torch::kFloat16,
      "fused_bias must be float16");
  TORCH_CHECK(
      x.dim() == 3 && x.size(1) == 1,
      "x must have shape [Batch, 1, HiddenDim]");
  TORCH_CHECK(
      fused_weight.dim() == 2,
      "fused_weight must have shape [OutDim, HiddenDim]");
  TORCH_CHECK(
      weight_sum.dim() == 1 &&
          weight_sum.size(0) == fused_weight.size(0),
      "weight_sum must have shape [OutDim]");
  TORCH_CHECK(
      fused_bias.dim() == 1 &&
          fused_bias.size(0) == fused_weight.size(0),
      "fused_bias must have shape [OutDim]");
  TORCH_CHECK(
      x.size(2) == fused_weight.size(1),
      "hidden dimensions must match");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(
      fused_weight.is_contiguous(), "fused_weight must be contiguous");
  TORCH_CHECK(weight_sum.is_contiguous(), "weight_sum must be contiguous");
  TORCH_CHECK(fused_bias.is_contiguous(), "fused_bias must be contiguous");
  TORCH_CHECK(
      x.size(0) >= 1 && x.size(0) <= 8,
      "the optimized kernel supports Batch in [1, 8]");
  TORCH_CHECK(
      x.size(2) > 0 && fused_weight.size(0) > 0,
      "all dimensions must be positive");
  TORCH_CHECK(eps >= 0.0, "eps must be non-negative");

  const c10::cuda::CUDAGuard device_guard(x.device());
  const int batch = static_cast<int>(x.size(0));
  const int hidden_dim = static_cast<int>(x.size(2));
  const int out_dim = static_cast<int>(fused_weight.size(0));
  auto projection = torch::empty(
      {batch, 1, out_dim}, x.options().dtype(torch::kFloat32));
  auto input_sum = torch::empty(
      {batch}, x.options().dtype(torch::kFloat32));
  auto input_square_sum = torch::empty(
      {batch}, x.options().dtype(torch::kFloat32));
  auto output = torch::empty({batch, 1, out_dim}, x.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  launch_layernorm_projection_stats_fp16(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(fused_weight.data_ptr<at::Half>()),
      projection.data_ptr<float>(),
      input_sum.data_ptr<float>(),
      input_square_sum.data_ptr<float>(),
      batch,
      out_dim,
      hidden_dim,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  launch_layernorm_epilogue_fp16(
      projection.data_ptr<float>(),
      input_sum.data_ptr<float>(),
      input_square_sum.data_ptr<float>(),
      reinterpret_cast<const half*>(weight_sum.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(fused_bias.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      batch,
      out_dim,
      hidden_dim,
      static_cast<float>(eps),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "fused_layernorm_linear_fp16",
      &fused_layernorm_linear_fp16,
      "Small-batch fused LayerNorm and Linear FP16 kernel");
}
