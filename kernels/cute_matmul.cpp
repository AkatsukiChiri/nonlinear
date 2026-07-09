#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

void launch_cute_matmul_fp16(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

torch::Tensor cute_matmul_fp16(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(a.dtype() == torch::kFloat16, "a must be float16");
  TORCH_CHECK(b.dtype() == torch::kFloat16, "b must be float16");
  TORCH_CHECK(a.dim() == 2, "a must have shape [M, K]");
  TORCH_CHECK(b.dim() == 2, "b must have shape [K, N]");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
  TORCH_CHECK(a.size(1) == b.size(0), "a.size(1) must equal b.size(0)");

  auto m = static_cast<int>(a.size(0));
  auto k = static_cast<int>(a.size(1));
  auto n = static_cast<int>(b.size(1));
  auto c = torch::empty({m, n}, a.options());

  launch_cute_matmul_fp16(
      reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
      reinterpret_cast<half*>(c.data_ptr<at::Half>()),
      m,
      n,
      k,
      at::cuda::getCurrentCUDAStream());

  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cute_matmul_fp16", &cute_matmul_fp16, "CuTe FP16 matmul");
}
