#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <optional>

void launch_w4a16_sentinel_preprocess(
    const at::Half* x, const at::Half* residual, const at::Half* gamma,
    at::Half* augmented, at::Half* residual_sum, int batch, int hidden_dim,
    cudaStream_t stream);

void launch_w4a16_sentinel_finalize(
    const at::Half* gemm_output, const at::Half* bias, at::Half* output,
    int batch, int out_dim, int padded_out_dim, int hidden_dim, float eps,
    cudaStream_t stream);

void launch_w4a16_awq_fused_gemv_bs1(
    const at::Half*, const at::Half*, const at::Half*, const int32_t*,
    const int32_t*, const at::Half*, const at::Half*, at::Half*, at::Half*,
    int, int, int, float, cudaStream_t);

void launch_w4a16_awq_repacked_gemv_bs1(
    const at::Half*, const at::Half*, const at::Half*, const uint8_t*,
    const uint8_t*, const at::Half*, const at::Half*, at::Half*, at::Half*,
    int, int, int, float, cudaStream_t);

void launch_w4a16_awq_tensorcore(
    const at::Half*, const at::Half*, const at::Half*, const uint8_t*,
    const uint8_t*, const at::Half*, const at::Half*, at::Half*, at::Half*,
    int, int, int, float, cudaStream_t);

void awq_tensorcore_out(
    torch::Tensor x, std::optional<torch::Tensor> residual,
    torch::Tensor gamma, torch::Tensor weight, torch::Tensor zeros,
    torch::Tensor scales, std::optional<torch::Tensor> bias,
    torch::Tensor output, std::optional<torch::Tensor> residual_sum,
    double eps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
                  x.dim() == 3 && x.size(1) == 1 &&
                  (x.size(0) == 1 || x.size(0) == 4 || x.size(0) == 8),
              "x must be CUDA float16 [Batch, 1, HiddenDim], Batch 1/4/8");
  const int64_t batch = x.size(0);
  const int64_t hidden_dim = x.size(2);
  TORCH_CHECK(hidden_dim % 128 == 0 && weight.is_cuda() &&
                  weight.scalar_type() == torch::kUInt8 &&
                  weight.dim() == 4 && weight.size(1) == 4 &&
                  weight.size(2) == hidden_dim && weight.size(3) == 8,
              "weight must be uint8 [CTAs, 4, HiddenDim, 8]");
  const int64_t out_dim = output.size(2);
  const int64_t groups = hidden_dim / 128;
  TORCH_CHECK(weight.size(0) == (out_dim + 63) / 64,
              "weight CTA count does not match output");
  TORCH_CHECK(zeros.is_cuda() && zeros.scalar_type() == torch::kUInt8 &&
                  zeros.sizes() == torch::IntArrayRef(
                      {weight.size(0), 4, groups, 16}),
              "invalid tensorcore zeros");
  TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == torch::kFloat16 &&
                  scales.sizes() == zeros.sizes(),
              "invalid tensorcore scales");
  TORCH_CHECK(gamma.is_cuda() && gamma.scalar_type() == torch::kFloat16 &&
                  gamma.numel() == hidden_dim,
              "invalid gamma");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::kFloat16 &&
                  output.sizes() == torch::IntArrayRef({batch, 1, out_dim}),
              "invalid output");
  const at::Half* residual_pointer = nullptr;
  at::Half* residual_sum_pointer = nullptr;
  if (residual.has_value()) {
    TORCH_CHECK(residual->sizes() == x.sizes() && residual->is_cuda() &&
                    residual->scalar_type() == torch::kFloat16 &&
                    residual_sum.has_value() &&
                    residual_sum->sizes() == x.sizes(),
                "residual and residual_sum must match x");
    residual_pointer = residual->data_ptr<at::Half>();
    residual_sum_pointer = residual_sum->data_ptr<at::Half>();
  }
  const at::Half* bias_pointer = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == torch::kFloat16 &&
                    bias->numel() == out_dim,
                "invalid bias");
    bias_pointer = bias->data_ptr<at::Half>();
  }
  const c10::cuda::CUDAGuard guard(x.device());
  launch_w4a16_awq_tensorcore(
      x.data_ptr<at::Half>(), residual_pointer, gamma.data_ptr<at::Half>(),
      weight.data_ptr<uint8_t>(), zeros.data_ptr<uint8_t>(),
      scales.data_ptr<at::Half>(), bias_pointer, output.data_ptr<at::Half>(),
      residual_sum_pointer, static_cast<int>(batch),
      static_cast<int>(out_dim), static_cast<int>(hidden_dim),
      static_cast<float>(eps), at::cuda::getCurrentCUDAStream());
}

void awq_repacked_gemv_bs1_out(
    torch::Tensor x, std::optional<torch::Tensor> residual,
    torch::Tensor gamma, torch::Tensor weight, torch::Tensor zeros,
    torch::Tensor scales, std::optional<torch::Tensor> bias,
    torch::Tensor output, std::optional<torch::Tensor> residual_sum,
    int64_t group_size, double eps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
                  x.dim() == 3 && x.size(0) == 1 && x.size(1) == 1,
              "x must be CUDA float16 [1, 1, HiddenDim]");
  const int64_t hidden_dim = x.size(2);
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == torch::kUInt8 &&
                  weight.dim() == 2 && weight.size(1) * 2 == hidden_dim,
              "weight must be uint8 [OutDim, HiddenDim / 2]");
  const int64_t out_dim = weight.size(0);
  TORCH_CHECK(hidden_dim == 4096 && group_size == 128,
              "repacked GEMV requires HiddenDim 4096 and group size 128");
  TORCH_CHECK(zeros.scalar_type() == torch::kUInt8 && zeros.is_cuda() &&
                  zeros.sizes() == torch::IntArrayRef(
                      {hidden_dim / group_size, out_dim}),
              "invalid unpacked zeros");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16 && scales.is_cuda() &&
                  scales.sizes() == zeros.sizes(),
              "invalid scales");
  TORCH_CHECK(gamma.scalar_type() == torch::kFloat16 && gamma.is_cuda() &&
                  gamma.numel() == hidden_dim && output.is_cuda() &&
                  output.scalar_type() == torch::kFloat16 &&
                  output.sizes() == torch::IntArrayRef({1, 1, out_dim}),
              "invalid gamma or output");
  const at::Half* residual_pointer = nullptr;
  at::Half* residual_sum_pointer = nullptr;
  if (residual.has_value()) {
    TORCH_CHECK(residual->sizes() == x.sizes() && residual->is_cuda() &&
                    residual->scalar_type() == torch::kFloat16 &&
                    residual_sum.has_value() &&
                    residual_sum->sizes() == x.sizes(),
                "residual and residual_sum must match x");
    residual_pointer = residual->data_ptr<at::Half>();
    residual_sum_pointer = residual_sum->data_ptr<at::Half>();
  }
  const at::Half* bias_pointer = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == torch::kFloat16 &&
                    bias->numel() == out_dim,
                "invalid bias");
    bias_pointer = bias->data_ptr<at::Half>();
  }
  const c10::cuda::CUDAGuard guard(x.device());
  launch_w4a16_awq_repacked_gemv_bs1(
      x.data_ptr<at::Half>(), residual_pointer, gamma.data_ptr<at::Half>(),
      weight.data_ptr<uint8_t>(), zeros.data_ptr<uint8_t>(),
      scales.data_ptr<at::Half>(), bias_pointer, output.data_ptr<at::Half>(),
      residual_sum_pointer, static_cast<int>(out_dim),
      static_cast<int>(hidden_dim), static_cast<int>(group_size),
      static_cast<float>(eps), at::cuda::getCurrentCUDAStream());
}

void awq_fused_gemv_bs1_out(
    torch::Tensor x, std::optional<torch::Tensor> residual,
    torch::Tensor gamma, torch::Tensor qweight, torch::Tensor qzeros,
    torch::Tensor scales, std::optional<torch::Tensor> bias,
    torch::Tensor output, std::optional<torch::Tensor> residual_sum,
    int64_t group_size, double eps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
                  x.dim() == 3 && x.size(0) == 1 && x.size(1) == 1,
              "x must be CUDA float16 [1, 1, HiddenDim]");
  const int64_t hidden_dim = x.size(2);
  TORCH_CHECK(qweight.is_cuda() && qweight.scalar_type() == torch::kInt32 &&
                  qweight.dim() == 2 && qweight.size(0) == hidden_dim,
              "qweight must be CUDA int32 [HiddenDim, OutDim / 8]");
  const int64_t out_dim = qweight.size(1) * 8;
  TORCH_CHECK(out_dim % 256 == 0 && hidden_dim % 8 == 0,
              "OutDim must be divisible by 256 and HiddenDim by 8");
  TORCH_CHECK(group_size > 0 && hidden_dim % group_size == 0,
              "invalid group size");
  TORCH_CHECK(qzeros.scalar_type() == torch::kInt32 && qzeros.is_cuda() &&
                  qzeros.sizes() == torch::IntArrayRef(
                      {hidden_dim / group_size, out_dim / 8}),
              "invalid qzeros");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16 && scales.is_cuda() &&
                  scales.sizes() == torch::IntArrayRef(
                      {hidden_dim / group_size, out_dim}),
              "invalid scales");
  TORCH_CHECK(gamma.scalar_type() == torch::kFloat16 && gamma.is_cuda() &&
                  gamma.numel() == hidden_dim,
              "invalid gamma");
  TORCH_CHECK(output.scalar_type() == torch::kFloat16 && output.is_cuda() &&
                  output.sizes() == torch::IntArrayRef({1, 1, out_dim}),
              "invalid output");
  TORCH_CHECK(x.is_contiguous() && gamma.is_contiguous() &&
                  qweight.is_contiguous() && qzeros.is_contiguous() &&
                  scales.is_contiguous() && output.is_contiguous(),
              "all tensors must be contiguous");
  const at::Half* residual_pointer = nullptr;
  at::Half* residual_sum_pointer = nullptr;
  if (residual.has_value()) {
    TORCH_CHECK(residual->scalar_type() == torch::kFloat16 &&
                    residual->is_cuda() && residual->sizes() == x.sizes() &&
                    residual->is_contiguous() && residual_sum.has_value() &&
                    residual_sum->sizes() == x.sizes(),
                "residual and residual_sum must match x");
    residual_pointer = residual->data_ptr<at::Half>();
    residual_sum_pointer = residual_sum->data_ptr<at::Half>();
  }
  const at::Half* bias_pointer = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->scalar_type() == torch::kFloat16 && bias->is_cuda() &&
                    bias->numel() == out_dim && bias->is_contiguous(),
                "invalid bias");
    bias_pointer = bias->data_ptr<at::Half>();
  }
  const c10::cuda::CUDAGuard guard(x.device());
  launch_w4a16_awq_fused_gemv_bs1(
      x.data_ptr<at::Half>(), residual_pointer, gamma.data_ptr<at::Half>(),
      qweight.data_ptr<int32_t>(), qzeros.data_ptr<int32_t>(),
      scales.data_ptr<at::Half>(), bias_pointer, output.data_ptr<at::Half>(),
      residual_sum_pointer, static_cast<int>(out_dim),
      static_cast<int>(hidden_dim), static_cast<int>(group_size),
      static_cast<float>(eps), at::cuda::getCurrentCUDAStream());
}

void preprocess_out(
    torch::Tensor x, std::optional<torch::Tensor> residual,
    torch::Tensor gamma, torch::Tensor augmented,
    std::optional<torch::Tensor> residual_sum) {
  TORCH_CHECK(x.is_cuda() && gamma.is_cuda() && augmented.is_cuda(),
              "inputs must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 && x.dim() == 3 &&
                  x.size(1) == 1,
              "x must be float16 [Batch, 1, HiddenDim]");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
              "Batch must be in [1, 8]");
  TORCH_CHECK(gamma.scalar_type() == torch::kFloat16 && gamma.dim() == 1 &&
                  gamma.size(0) == x.size(2),
              "gamma must be float16 [HiddenDim]");
  TORCH_CHECK(augmented.scalar_type() == torch::kFloat16 &&
                  augmented.sizes() ==
                      torch::IntArrayRef({x.size(0) * 2, x.size(2)}),
              "augmented must be float16 [2 * Batch, HiddenDim]");
  TORCH_CHECK(x.device() == gamma.device() && x.device() == augmented.device(),
              "device mismatch");
  TORCH_CHECK(x.is_contiguous() && gamma.is_contiguous() &&
                  augmented.is_contiguous(),
              "inputs must be contiguous");
  if (residual.has_value()) {
    TORCH_CHECK(residual->is_cuda() && residual->device() == x.device() &&
                    residual->scalar_type() == torch::kFloat16 &&
                    residual->sizes() == x.sizes() &&
                    residual->is_contiguous(),
                "residual must match x");
    TORCH_CHECK(residual_sum.has_value() && residual_sum->is_cuda() &&
                    residual_sum->device() == x.device() &&
                    residual_sum->scalar_type() == torch::kFloat16 &&
                    residual_sum->sizes() == x.sizes() &&
                    residual_sum->is_contiguous(),
                "residual_sum must match x");
  }

  const c10::cuda::CUDAGuard guard(x.device());
  launch_w4a16_sentinel_preprocess(
      x.data_ptr<at::Half>(),
      residual.has_value() ? residual->data_ptr<at::Half>() : nullptr,
      gamma.data_ptr<at::Half>(), augmented.data_ptr<at::Half>(),
      residual_sum.has_value() ? residual_sum->data_ptr<at::Half>() : nullptr,
      static_cast<int>(x.size(0)), static_cast<int>(x.size(2)),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void finalize_out(
    torch::Tensor gemm_output, std::optional<torch::Tensor> bias,
    torch::Tensor output, int64_t out_dim, int64_t hidden_dim, double eps) {
  TORCH_CHECK(gemm_output.is_cuda() && output.is_cuda(),
              "tensors must be CUDA");
  TORCH_CHECK(gemm_output.scalar_type() == torch::kFloat16 &&
                  gemm_output.dim() == 2,
              "gemm_output must be float16 [2 * Batch, PaddedOutDim]");
  TORCH_CHECK(gemm_output.size(0) % 2 == 0,
              "gemm_output row count must be even");
  const int64_t batch = gemm_output.size(0) / 2;
  TORCH_CHECK(batch >= 1 && batch <= 8, "Batch must be in [1, 8]");
  TORCH_CHECK(out_dim > 0 && out_dim < gemm_output.size(1),
              "sentinel column must follow the logical output");
  TORCH_CHECK(out_dim % 2 == 0,
              "OutDim must be even for vectorized finalization");
  TORCH_CHECK(output.scalar_type() == torch::kFloat16 &&
                  output.sizes() ==
                      torch::IntArrayRef({batch, 1, out_dim}),
              "output must be float16 [Batch, 1, OutDim]");
  TORCH_CHECK(gemm_output.device() == output.device() &&
                  gemm_output.is_contiguous() && output.is_contiguous(),
              "invalid output layout or device");
  const at::Half* bias_pointer = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->device() == output.device() &&
                    bias->scalar_type() == torch::kFloat16 &&
                    bias->sizes() == torch::IntArrayRef({out_dim}) &&
                    bias->is_contiguous(),
                "bias must be float16 [OutDim]");
    bias_pointer = bias->data_ptr<at::Half>();
  }

  const c10::cuda::CUDAGuard guard(output.device());
  launch_w4a16_sentinel_finalize(
      gemm_output.data_ptr<at::Half>(), bias_pointer,
      output.data_ptr<at::Half>(), static_cast<int>(batch),
      static_cast<int>(out_dim), static_cast<int>(gemm_output.size(1)),
      static_cast<int>(hidden_dim), static_cast<float>(eps),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("preprocess_out", &preprocess_out);
  module.def("finalize_out", &finalize_out);
  module.def("awq_fused_gemv_bs1_out", &awq_fused_gemv_bs1_out);
  module.def("awq_repacked_gemv_bs1_out", &awq_repacked_gemv_bs1_out);
  module.def("awq_tensorcore_out", &awq_tensorcore_out);
}
