#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

void launch_fused_rmsnorm_projection_fp16(
    const half* x,
    const half* weight,
    half* output,
    float* square_sum,
    int batch,
    int out_dim,
    int hidden_dim,
    cudaStream_t stream);

void launch_rms_scale_fp16(
    half* output,
    const float* square_sum,
    int batch,
    int out_dim,
    int hidden_dim,
    float eps,
    cudaStream_t stream);

void launch_rms_scale_bias_fp16(
    half* output,
    const float* square_sum,
    const half* bias,
    int batch,
    int out_dim,
    int hidden_dim,
    float eps,
    cudaStream_t stream);

void launch_fused_rmsnorm_direct_fp16(
    const half* x,
    const half* weight,
    const half* bias,
    half* output,
    int out_dim,
    int hidden_dim,
    float eps,
    cudaStream_t stream);

void launch_fused_rmsnorm_direct_batch2_fp16(
    const half* x,
    const half* weight,
    const half* bias,
    half* output,
    int out_dim,
    int hidden_dim,
    float eps,
    cudaStream_t stream);

torch::Tensor fused_rmsnorm_linear_fp16(
    torch::Tensor x,
    torch::Tensor fused_weight,
    double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(fused_weight.is_cuda(), "fused_weight must be a CUDA tensor");
  TORCH_CHECK(x.device() == fused_weight.device(),
              "x and fused_weight must be on one device");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
  TORCH_CHECK(fused_weight.scalar_type() == torch::kFloat16,
              "fused_weight must be float16");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1,
              "x must have shape [Batch, 1, HiddenDim]");
  TORCH_CHECK(fused_weight.dim() == 2,
              "fused_weight must have shape [OutDim, HiddenDim]");
  TORCH_CHECK(x.size(2) == fused_weight.size(1),
              "hidden dimensions must match");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(fused_weight.is_contiguous(), "fused_weight must be contiguous");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
              "the optimized kernel supports Batch in [1, 8]");
  TORCH_CHECK(x.size(2) > 0 && fused_weight.size(0) > 0,
              "all dimensions must be positive");
  TORCH_CHECK(eps >= 0.0, "eps must be non-negative");

  const c10::cuda::CUDAGuard device_guard(x.device());
  const int batch = static_cast<int>(x.size(0));
  const int hidden_dim = static_cast<int>(x.size(2));
  const int out_dim = static_cast<int>(fused_weight.size(0));
  auto output = torch::empty({batch, 1, out_dim}, x.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (batch == 1) {
    launch_fused_rmsnorm_direct_fp16(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(fused_weight.data_ptr<at::Half>()),
        nullptr,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        out_dim,
        hidden_dim,
        static_cast<float>(eps),
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }
  if (batch == 2) {
    launch_fused_rmsnorm_direct_batch2_fp16(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(fused_weight.data_ptr<at::Half>()),
        nullptr,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        out_dim,
        hidden_dim,
        static_cast<float>(eps),
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }
  auto square_sum = torch::empty(
      {batch}, x.options().dtype(torch::kFloat32));

  launch_fused_rmsnorm_projection_fp16(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(fused_weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      square_sum.data_ptr<float>(),
      batch,
      out_dim,
      hidden_dim,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  launch_rms_scale_fp16(
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      square_sum.data_ptr<float>(),
      batch,
      out_dim,
      hidden_dim,
      static_cast<float>(eps),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fused_rmsnorm_linear_bias_fp16(
    torch::Tensor x,
    torch::Tensor fused_weight,
    torch::Tensor bias,
    double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(fused_weight.is_cuda(), "fused_weight must be a CUDA tensor");
  TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
  TORCH_CHECK(x.device() == fused_weight.device() && x.device() == bias.device(),
              "x, fused_weight, and bias must be on one device");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
  TORCH_CHECK(fused_weight.scalar_type() == torch::kFloat16,
              "fused_weight must be float16");
  TORCH_CHECK(bias.scalar_type() == torch::kFloat16, "bias must be float16");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1,
              "x must have shape [Batch, 1, HiddenDim]");
  TORCH_CHECK(fused_weight.dim() == 2,
              "fused_weight must have shape [OutDim, HiddenDim]");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == fused_weight.size(0),
              "bias must have shape [OutDim]");
  TORCH_CHECK(x.size(2) == fused_weight.size(1),
              "hidden dimensions must match");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(fused_weight.is_contiguous(), "fused_weight must be contiguous");
  TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
              "the optimized kernel supports Batch in [1, 8]");
  TORCH_CHECK(x.size(2) > 0 && fused_weight.size(0) > 0,
              "all dimensions must be positive");
  TORCH_CHECK(eps >= 0.0, "eps must be non-negative");

  const c10::cuda::CUDAGuard device_guard(x.device());
  const int batch = static_cast<int>(x.size(0));
  const int hidden_dim = static_cast<int>(x.size(2));
  const int out_dim = static_cast<int>(fused_weight.size(0));
  auto output = torch::empty({batch, 1, out_dim}, x.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (batch == 1) {
    launch_fused_rmsnorm_direct_fp16(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(fused_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        out_dim,
        hidden_dim,
        static_cast<float>(eps),
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }
  if (batch == 2) {
    launch_fused_rmsnorm_direct_batch2_fp16(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(fused_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        out_dim,
        hidden_dim,
        static_cast<float>(eps),
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }
  auto square_sum = torch::empty(
      {batch}, x.options().dtype(torch::kFloat32));

  launch_fused_rmsnorm_projection_fp16(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(fused_weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      square_sum.data_ptr<float>(),
      batch,
      out_dim,
      hidden_dim,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  launch_rms_scale_bias_fp16(
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      square_sum.data_ptr<float>(),
      reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
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
      "fused_rmsnorm_linear_fp16",
      &fused_rmsnorm_linear_fp16,
      "Small-batch fused RMSNorm and Linear FP16 kernel");
  module.def(
      "fused_rmsnorm_linear_bias_fp16",
      &fused_rmsnorm_linear_bias_fp16,
      "Small-batch fused RMSNorm and biased Linear FP16 kernel");
}
