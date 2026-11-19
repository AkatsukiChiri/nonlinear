#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <optional>
#include <cstdint>
#include <vector>

void launch_tensorcore_rms_projection(
    const void* x, const void* residual, const half* weight,
    const void* bias, void* output, void* residual_sum, int out_dim,
    int batch, int hidden_dim, int n_tiles, float eps, bool is_bfloat16,
    cudaStream_t stream);

void launch_tensorcore_rms_projection_int8_weight(
    const void* x, const void* residual, const int8_t* quantized_weight,
    const half* weight_scale, const void* bias, void* output,
    void* residual_sum, int out_dim, int batch, int hidden_dim, int n_tiles,
    float eps, bool is_bfloat16, cudaStream_t stream);

std::vector<torch::Tensor> run_forward(
    torch::Tensor x, std::optional<torch::Tensor> residual,
    torch::Tensor weight, int64_t out_dim, std::optional<torch::Tensor> bias, double eps,
    torch::Tensor output, std::optional<torch::Tensor> residual_sum) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(x.device() == weight.device(), "device mismatch");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 ||
                  x.scalar_type() == torch::kBFloat16,
              "x must be float16 or bfloat16");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1 &&
                  (x.size(0) == 1 || x.size(0) == 4 || x.size(0) == 8 ||
                   x.size(0) == 16),
              "x must have shape [Batch, 1, HiddenDim], Batch in {1,4,8,16}");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat16 && weight.dim() == 4 &&
                  (weight.size(1) == 1 || weight.size(1) == 2 ||
                   weight.size(1) == 4 ||
                   weight.size(1) == 8 || weight.size(1) == 16 ||
                   weight.size(1) == 32) &&
                  weight.size(2) == x.size(2) && weight.size(3) == 16,
              "weight must be packed float16 [CTAs, NTiles, HiddenDim, 16]");
  const int64_t projection_columns = weight.size(1) * 16 - 1;
  TORCH_CHECK(weight.size(0) ==
                  (out_dim + projection_columns - 1) / projection_columns,
              "packed weight tile count does not match OutDim");
  TORCH_CHECK(x.size(2) % 128 == 0,
              "HiddenDim must be divisible by 128");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(),
              "inputs must be contiguous");
  TORCH_CHECK(output.is_cuda() && output.device() == x.device() &&
                  output.scalar_type() == x.scalar_type() &&
                  output.sizes() ==
                      torch::IntArrayRef({x.size(0), 1, out_dim}) &&
                  output.is_contiguous(),
              "invalid output buffer");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->device() == x.device() &&
                    bias->scalar_type() == x.scalar_type() &&
                    bias->dim() == 1 && bias->size(0) == out_dim &&
                    bias->is_contiguous(),
                "invalid bias");
  }
  if (residual.has_value()) {
    TORCH_CHECK(residual->is_cuda() && residual->device() == x.device() &&
                    residual->scalar_type() == x.scalar_type() &&
                    residual->sizes() == x.sizes() &&
                    residual->is_contiguous(),
                "residual must match x");
    TORCH_CHECK(residual_sum.has_value(),
                "residual output is required when residual is present");
    TORCH_CHECK(residual_sum->is_cuda() &&
                    residual_sum->device() == x.device() &&
                    residual_sum->scalar_type() == x.scalar_type() &&
                    residual_sum->sizes() == x.sizes() &&
                    residual_sum->is_contiguous(),
                "invalid residual output buffer");
  }

  const c10::cuda::CUDAGuard guard(x.device());
  launch_tensorcore_rms_projection(
      x.data_ptr(), residual.has_value() ? residual->data_ptr() : nullptr,
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      bias.has_value() ? bias->data_ptr() : nullptr, output.data_ptr(),
      residual_sum.has_value() ? residual_sum->data_ptr() : nullptr,
      out_dim, x.size(0), x.size(2), weight.size(1), static_cast<float>(eps),
      x.scalar_type() == torch::kBFloat16,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (residual.has_value()) return {output, *residual_sum};
  return {output};
}

std::vector<torch::Tensor> forward(
    torch::Tensor x, std::optional<torch::Tensor> residual,
    torch::Tensor weight, int64_t out_dim, std::optional<torch::Tensor> bias,
    double eps) {
  auto output = torch::empty({x.size(0), 1, out_dim}, x.options());
  std::optional<torch::Tensor> residual_sum = std::nullopt;
  if (residual.has_value()) residual_sum = torch::empty_like(x);
  return run_forward(std::move(x), std::move(residual), std::move(weight), out_dim,
                     std::move(bias), eps, std::move(output),
                     std::move(residual_sum));
}

std::vector<torch::Tensor> forward_out(
    torch::Tensor x, std::optional<torch::Tensor> residual,
    torch::Tensor weight, int64_t out_dim, std::optional<torch::Tensor> bias, double eps,
    torch::Tensor output, std::optional<torch::Tensor> residual_sum) {
  return run_forward(std::move(x), std::move(residual), std::move(weight), out_dim,
                     std::move(bias), eps, std::move(output),
                     std::move(residual_sum));
}

std::vector<torch::Tensor> forward_int8_weight_out(
    torch::Tensor x, std::optional<torch::Tensor> residual,
    torch::Tensor quantized_weight, torch::Tensor weight_scale,
    int64_t out_dim, std::optional<torch::Tensor> bias, double eps,
    torch::Tensor output, std::optional<torch::Tensor> residual_sum) {
  TORCH_CHECK(x.is_cuda() && quantized_weight.is_cuda() &&
                  weight_scale.is_cuda(),
              "inputs must be CUDA");
  TORCH_CHECK(x.device() == quantized_weight.device() &&
                  x.device() == weight_scale.device(),
              "device mismatch");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 ||
                  x.scalar_type() == torch::kBFloat16,
              "x must be float16 or bfloat16");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1 &&
                  (x.size(0) == 1 || x.size(0) == 4 || x.size(0) == 8 ||
                   x.size(0) == 16),
              "x must have shape [Batch, 1, HiddenDim], Batch in {1,4,8,16}");
  TORCH_CHECK(
      quantized_weight.scalar_type() == torch::kInt8 &&
          quantized_weight.dim() == 4 &&
          (quantized_weight.size(1) == 1 ||
           quantized_weight.size(1) == 2 ||
           quantized_weight.size(1) == 4 ||
           quantized_weight.size(1) == 8 ||
           quantized_weight.size(1) == 16 ||
           quantized_weight.size(1) == 32) &&
          quantized_weight.size(2) == x.size(2) &&
          quantized_weight.size(3) == 16,
      "quantized_weight must be int8 [CTAs, NTiles, HiddenDim, 16]");
  TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat16 &&
                  weight_scale.dim() == 3 &&
                  weight_scale.size(0) == quantized_weight.size(0) &&
                  weight_scale.size(1) == quantized_weight.size(1) &&
                  weight_scale.size(2) == 16,
              "weight_scale must be float16 [CTAs, NTiles, 16]");
  const int64_t projection_columns = quantized_weight.size(1) * 16 - 1;
  TORCH_CHECK(quantized_weight.size(0) ==
                  (out_dim + projection_columns - 1) / projection_columns,
              "packed weight tile count does not match OutDim");
  TORCH_CHECK(x.size(2) % 128 == 0,
              "HiddenDim must be divisible by 128");
  TORCH_CHECK(x.is_contiguous() && quantized_weight.is_contiguous() &&
                  weight_scale.is_contiguous(),
              "inputs must be contiguous");
  TORCH_CHECK(output.is_cuda() && output.device() == x.device() &&
                  output.scalar_type() == x.scalar_type() &&
                  output.sizes() ==
                      torch::IntArrayRef({x.size(0), 1, out_dim}) &&
                  output.is_contiguous(),
              "invalid output buffer");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->device() == x.device() &&
                    bias->scalar_type() == x.scalar_type() &&
                    bias->dim() == 1 && bias->size(0) == out_dim &&
                    bias->is_contiguous(),
                "invalid bias");
  }
  if (residual.has_value()) {
    TORCH_CHECK(residual->is_cuda() && residual->device() == x.device() &&
                    residual->scalar_type() == x.scalar_type() &&
                    residual->sizes() == x.sizes() &&
                    residual->is_contiguous(),
                "residual must match x");
    TORCH_CHECK(residual_sum.has_value(),
                "residual output is required when residual is present");
    TORCH_CHECK(residual_sum->is_cuda() &&
                    residual_sum->device() == x.device() &&
                    residual_sum->scalar_type() == x.scalar_type() &&
                    residual_sum->sizes() == x.sizes() &&
                    residual_sum->is_contiguous(),
                "invalid residual output buffer");
  }

  const c10::cuda::CUDAGuard guard(x.device());
  launch_tensorcore_rms_projection_int8_weight(
      x.data_ptr(), residual.has_value() ? residual->data_ptr() : nullptr,
      quantized_weight.data_ptr<int8_t>(),
      reinterpret_cast<const half*>(weight_scale.data_ptr<at::Half>()),
      bias.has_value() ? bias->data_ptr() : nullptr, output.data_ptr(),
      residual_sum.has_value() ? residual_sum->data_ptr() : nullptr,
      out_dim, x.size(0), x.size(2), quantized_weight.size(1),
      static_cast<float>(eps), x.scalar_type() == torch::kBFloat16,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (residual.has_value()) return {output, *residual_sum};
  return {output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &forward);
  module.def("forward_out", &forward_out);
  module.def("forward_int8_weight_out", &forward_int8_weight_out);
}
