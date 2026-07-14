#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

void launch_fused_square_linear_fp16(
    const half* x,
    const half* weight,
    half* output,
    int batch,
    int out_dim,
    int hidden_dim,
    cudaStream_t stream);

torch::Tensor fused_square_linear_fp16(
    torch::Tensor x,
    torch::Tensor weight) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(x.device() == weight.device(), "x and weight must be on one device");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "weight must be float16");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1,
              "x must have shape [Batch, 1, HiddenDim]");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [OutDim, HiddenDim]");
  TORCH_CHECK(x.size(2) == weight.size(1), "hidden dimensions must match");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(x.size(0) > 0 && x.size(2) > 0 && weight.size(0) > 0,
              "all dimensions must be positive");
  TORCH_CHECK(x.size(0) <= INT_MAX && x.size(2) <= INT_MAX &&
                  weight.size(0) <= INT_MAX,
              "tensor dimensions exceed the CUDA kernel integer range");

  const c10::cuda::CUDAGuard device_guard(x.device());
  const int batch = static_cast<int>(x.size(0));
  const int hidden_dim = static_cast<int>(x.size(2));
  const int out_dim = static_cast<int>(weight.size(0));
  auto output = torch::empty({batch, 2, out_dim}, x.options());

  launch_fused_square_linear_fp16(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      batch,
      out_dim,
      hidden_dim,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "fused_square_linear_fp16",
      &fused_square_linear_fp16,
      "Fused square-and-linear FP16 Tensor Core kernel");
}
