#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

void launch_int4_projection(const half*, const uint8_t*, const float*,
                            const half*, half*, float*, int, int, int, float,
                            cudaStream_t);

torch::Tensor forward(torch::Tensor x, torch::Tensor weight,
                      torch::Tensor scale, std::optional<torch::Tensor> bias,
                      double eps) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && scale.is_cuda(),
              "inputs must be CUDA");
  TORCH_CHECK(x.device() == weight.device() && x.device() == scale.device(),
              "device mismatch");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
  TORCH_CHECK(weight.scalar_type() == torch::kUInt8,
              "packed weight must be uint8");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32,
              "scale must be float32");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1,
              "x must be [Batch, 1, HiddenDim]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(1) * 2 == x.size(2),
              "packed weight must be [OutDim, HiddenDim / 2]");
  TORCH_CHECK(x.size(2) % 256 == 0,
              "HiddenDim must be divisible by 256");
  TORCH_CHECK(scale.sizes() == torch::IntArrayRef({weight.size(0), 1}),
              "invalid scale shape");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
              "Batch must be in [1, 8]");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() &&
                  scale.is_contiguous(),
              "inputs must be contiguous");
  const half* bias_pointer = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->device() == x.device(),
                "invalid bias device");
    TORCH_CHECK(bias->scalar_type() == torch::kFloat16 && bias->dim() == 1 &&
                    bias->size(0) == weight.size(0) && bias->is_contiguous(),
                "invalid bias");
    bias_pointer = reinterpret_cast<const half*>(bias->data_ptr<at::Half>());
  }
  const c10::cuda::CUDAGuard guard(x.device());
  auto output = torch::empty({x.size(0), 1, weight.size(0)}, x.options());
  auto square_sum = torch::empty({x.size(0)},
                                 x.options().dtype(torch::kFloat32));
  launch_int4_projection(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      weight.data_ptr<uint8_t>(), scale.data_ptr<float>(), bias_pointer,
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      square_sum.data_ptr<float>(), x.size(0), weight.size(0), x.size(2),
      static_cast<float>(eps), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &forward);
}
