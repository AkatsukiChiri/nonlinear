#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

void launch_quantize_rms_int8_prefill(const half*, int8_t*, float*, int, int,
                                      float, cudaStream_t);
void launch_gemm_int8_prefill(const int8_t*, const int8_t*, const float*,
                              const float*, const half*, half*, int, int, int,
                              cudaStream_t);

torch::Tensor forward(torch::Tensor x, torch::Tensor weight,
                      torch::Tensor weight_scale,
                      std::optional<torch::Tensor> bias, double eps) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && weight_scale.is_cuda(),
              "inputs must be CUDA tensors");
  TORCH_CHECK(x.device() == weight.device() && x.device() == weight_scale.device(),
              "device mismatch");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 && x.dim() == 3,
              "x must be [Batch, SeqLen, HiddenDim] float16");
  TORCH_CHECK(weight.scalar_type() == torch::kInt8 && weight.dim() == 2,
              "weight must be [OutDim, HiddenDim] int8");
  TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat32 &&
              weight_scale.dim() == 1 && weight_scale.size(0) == weight.size(0),
              "weight_scale must be [OutDim] float32");
  TORCH_CHECK(x.size(2) == weight.size(1), "HiddenDim mismatch");
  TORCH_CHECK(x.size(2) % 32 == 0 && weight.size(0) % 8 == 0,
              "Tensor Core dimensions are not aligned");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() &&
              weight_scale.is_contiguous(), "inputs must be contiguous");
  const half* bias_pointer = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->device() == x.device() &&
                bias->scalar_type() == torch::kFloat16 && bias->dim() == 1 &&
                bias->size(0) == weight.size(0) && bias->is_contiguous(),
                "bias must be contiguous [OutDim] float16 CUDA");
    bias_pointer = reinterpret_cast<const half*>(bias->data_ptr<at::Half>());
  }
  const c10::cuda::CUDAGuard guard(x.device());
  const int rows = static_cast<int>(x.size(0) * x.size(1));
  const int hidden_dim = static_cast<int>(x.size(2));
  const int out_dim = static_cast<int>(weight.size(0));
  auto quantized_x = torch::empty({rows, hidden_dim}, x.options().dtype(torch::kInt8));
  auto row_scale = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  auto output = torch::empty({x.size(0), x.size(1), out_dim}, x.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_quantize_rms_int8_prefill(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      quantized_x.data_ptr<int8_t>(), row_scale.data_ptr<float>(), rows,
      hidden_dim, static_cast<float>(eps), stream);
  launch_gemm_int8_prefill(
      quantized_x.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
      row_scale.data_ptr<float>(), weight_scale.data_ptr<float>(), bias_pointer,
      reinterpret_cast<half*>(output.data_ptr<at::Half>()), rows, out_dim,
      hidden_dim, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("forward", &forward); }
