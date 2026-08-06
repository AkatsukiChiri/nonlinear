#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <vector>

void launch_quantize_rms_int8_prefill(const half*, int8_t*, float*, int, int,
                                      float, cudaStream_t);
void launch_quantize_rms_int8_prefill_bf16(const __nv_bfloat16*, int8_t*,
                                           float*, int, int, float,
                                           cudaStream_t);
void launch_quantize_add_rms_int8_prefill(
    const half*, const half*, half*, int8_t*, float*, int, int, float,
    cudaStream_t);
void launch_quantize_add_rms_int8_prefill_bf16(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int8_t*,
    float*, int, int, float, cudaStream_t);
void launch_gemm_int8_prefill(const int8_t*, const int8_t*, const float*,
                              const float*, const half*, half*, int, int, int,
                              cudaStream_t);
void launch_gemm_int8_prefill_bf16(
    const int8_t*, const int8_t*, const float*, const float*,
    const __nv_bfloat16*, __nv_bfloat16*, int, int, int, cudaStream_t);

torch::Tensor forward(torch::Tensor x, torch::Tensor weight,
                      torch::Tensor weight_scale,
                      std::optional<torch::Tensor> bias, double eps) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && weight_scale.is_cuda(),
              "inputs must be CUDA tensors");
  TORCH_CHECK(x.device() == weight.device() && x.device() == weight_scale.device(),
              "device mismatch");
  TORCH_CHECK((x.scalar_type() == torch::kFloat16 ||
               x.scalar_type() == torch::kBFloat16) && x.dim() == 3,
              "x must be [Batch, SeqLen, HiddenDim] float16 or bfloat16");
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
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->device() == x.device() &&
                bias->scalar_type() == x.scalar_type() && bias->dim() == 1 &&
                bias->size(0) == weight.size(0) && bias->is_contiguous(),
                "bias must be contiguous [OutDim] with the input dtype");
  }
  const c10::cuda::CUDAGuard guard(x.device());
  const int rows = static_cast<int>(x.size(0) * x.size(1));
  const int hidden_dim = static_cast<int>(x.size(2));
  const int out_dim = static_cast<int>(weight.size(0));
  auto quantized_x = torch::empty({rows, hidden_dim}, x.options().dtype(torch::kInt8));
  auto row_scale = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  auto output = torch::empty({x.size(0), x.size(1), out_dim}, x.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == torch::kFloat16) {
    const half* bias_pointer = bias.has_value()
        ? reinterpret_cast<const half*>(bias->data_ptr<at::Half>()) : nullptr;
    launch_quantize_rms_int8_prefill(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        quantized_x.data_ptr<int8_t>(), row_scale.data_ptr<float>(), rows,
        hidden_dim, static_cast<float>(eps), stream);
    launch_gemm_int8_prefill(
        quantized_x.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
        row_scale.data_ptr<float>(), weight_scale.data_ptr<float>(),
        bias_pointer, reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        rows, out_dim, hidden_dim, stream);
  } else {
    const __nv_bfloat16* bias_pointer = bias.has_value()
        ? reinterpret_cast<const __nv_bfloat16*>(
              bias->data_ptr<at::BFloat16>()) : nullptr;
    launch_quantize_rms_int8_prefill_bf16(
        reinterpret_cast<const __nv_bfloat16*>(
            x.data_ptr<at::BFloat16>()),
        quantized_x.data_ptr<int8_t>(), row_scale.data_ptr<float>(), rows,
        hidden_dim, static_cast<float>(eps), stream);
    launch_gemm_int8_prefill_bf16(
        quantized_x.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
        row_scale.data_ptr<float>(), weight_scale.data_ptr<float>(),
        bias_pointer,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        rows, out_dim, hidden_dim, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> run_forward_add(
    torch::Tensor x, torch::Tensor x_res, torch::Tensor weight,
    torch::Tensor weight_scale, std::optional<torch::Tensor> bias,
    double eps, torch::Tensor quantized_x, torch::Tensor row_scale,
    torch::Tensor output, torch::Tensor residual_sum) {
  TORCH_CHECK(x_res.is_cuda() && x_res.device() == x.device() &&
              x_res.scalar_type() == x.scalar_type() &&
              x_res.sizes() == x.sizes() && x_res.is_contiguous(),
              "x_res must match x shape, dtype, device, and layout");
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && weight_scale.is_cuda(),
              "inputs must be CUDA tensors");
  TORCH_CHECK(x.device() == weight.device() && x.device() == weight_scale.device(),
              "device mismatch");
  TORCH_CHECK((x.scalar_type() == torch::kFloat16 ||
               x.scalar_type() == torch::kBFloat16) && x.dim() == 3,
              "x must be 3D float16 or bfloat16");
  TORCH_CHECK(weight.scalar_type() == torch::kInt8 && weight.dim() == 2,
              "weight must be [OutDim, HiddenDim] int8");
  TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat32 &&
              weight_scale.dim() == 1 && weight_scale.size(0) == weight.size(0),
              "weight_scale must be [OutDim] float32");
  TORCH_CHECK(x.size(2) == weight.size(1) && x.size(2) % 32 == 0 &&
              weight.size(0) % 8 == 0,
              "Tensor Core dimensions are not aligned");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() &&
              weight_scale.is_contiguous(), "inputs must be contiguous");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->device() == x.device() &&
                bias->scalar_type() == x.scalar_type() && bias->dim() == 1 &&
                bias->size(0) == weight.size(0) && bias->is_contiguous(),
                "invalid bias");
  }
  const c10::cuda::CUDAGuard guard(x.device());
  const int rows = static_cast<int>(x.size(0) * x.size(1));
  const int hidden_dim = static_cast<int>(x.size(2));
  const int out_dim = static_cast<int>(weight.size(0));
  TORCH_CHECK(quantized_x.is_cuda() && quantized_x.device() == x.device() &&
              quantized_x.scalar_type() == torch::kInt8 &&
              quantized_x.sizes() == torch::IntArrayRef({rows, hidden_dim}) &&
              quantized_x.is_contiguous(), "invalid quantized_x workspace");
  TORCH_CHECK(row_scale.is_cuda() && row_scale.device() == x.device() &&
              row_scale.scalar_type() == torch::kFloat32 &&
              row_scale.sizes() == torch::IntArrayRef({rows}) &&
              row_scale.is_contiguous(), "invalid row_scale workspace");
  TORCH_CHECK(output.is_cuda() && output.device() == x.device() &&
              output.scalar_type() == x.scalar_type() &&
              output.sizes() == torch::IntArrayRef(
                  {x.size(0), x.size(1), out_dim}) && output.is_contiguous(),
              "invalid output buffer");
  TORCH_CHECK(residual_sum.is_cuda() && residual_sum.device() == x.device() &&
              residual_sum.scalar_type() == x.scalar_type() &&
              residual_sum.sizes() == x.sizes() && residual_sum.is_contiguous(),
              "invalid residual buffer");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == torch::kFloat16) {
    const half* bias_pointer = bias.has_value()
        ? reinterpret_cast<const half*>(bias->data_ptr<at::Half>()) : nullptr;
    launch_quantize_add_rms_int8_prefill(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x_res.data_ptr<at::Half>()),
        reinterpret_cast<half*>(residual_sum.data_ptr<at::Half>()),
        quantized_x.data_ptr<int8_t>(), row_scale.data_ptr<float>(), rows,
        hidden_dim, static_cast<float>(eps), stream);
    launch_gemm_int8_prefill(
        quantized_x.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
        row_scale.data_ptr<float>(), weight_scale.data_ptr<float>(),
        bias_pointer, reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        rows, out_dim, hidden_dim, stream);
  } else {
    const __nv_bfloat16* bias_pointer = bias.has_value()
        ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr<at::BFloat16>())
        : nullptr;
    launch_quantize_add_rms_int8_prefill_bf16(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            x_res.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            residual_sum.data_ptr<at::BFloat16>()),
        quantized_x.data_ptr<int8_t>(), row_scale.data_ptr<float>(), rows,
        hidden_dim, static_cast<float>(eps), stream);
    launch_gemm_int8_prefill_bf16(
        quantized_x.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
        row_scale.data_ptr<float>(), weight_scale.data_ptr<float>(),
        bias_pointer,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        rows, out_dim, hidden_dim, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, residual_sum};
}

std::vector<torch::Tensor> forward_add(
    torch::Tensor x, torch::Tensor x_res, torch::Tensor weight,
    torch::Tensor weight_scale, std::optional<torch::Tensor> bias,
    double eps) {
  const int rows = static_cast<int>(x.size(0) * x.size(1));
  const int hidden_dim = static_cast<int>(x.size(2));
  const int out_dim = static_cast<int>(weight.size(0));
  auto quantized_x = torch::empty(
      {rows, hidden_dim}, x.options().dtype(torch::kInt8));
  auto row_scale = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  auto output = torch::empty({x.size(0), x.size(1), out_dim}, x.options());
  auto residual_sum = torch::empty_like(x);
  return run_forward_add(
      std::move(x), std::move(x_res), std::move(weight),
      std::move(weight_scale), std::move(bias), eps, std::move(quantized_x),
      std::move(row_scale), std::move(output), std::move(residual_sum));
}

std::vector<torch::Tensor> forward_add_out(
    torch::Tensor x, torch::Tensor x_res, torch::Tensor weight,
    torch::Tensor weight_scale, std::optional<torch::Tensor> bias, double eps,
    torch::Tensor quantized_x, torch::Tensor row_scale, torch::Tensor output,
    torch::Tensor residual_sum) {
  return run_forward_add(
      std::move(x), std::move(x_res), std::move(weight),
      std::move(weight_scale), std::move(bias), eps, std::move(quantized_x),
      std::move(row_scale), std::move(output), std::move(residual_sum));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &forward);
  module.def("forward_add", &forward_add);
  module.def("forward_add_out", &forward_add_out);
}
