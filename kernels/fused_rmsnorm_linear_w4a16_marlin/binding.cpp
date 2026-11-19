#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <optional>

void launch_w4a16_marlin_sentinel_wmma(
    const void* input, const void* gamma, const uint8_t* qweight,
    const uint8_t* zeros, const void* scales, const void* bias,
    void* output, void* workspace, int* locks, int batch, int hidden_dim,
    int logical_out_dim, int physical_out_dim, int stripe_n, float eps,
    cudaStream_t stream);

torch::Tensor w4a16_marlin_sentinel_wmma(
    torch::Tensor input, torch::Tensor gamma, torch::Tensor qweight,
    torch::Tensor zeros, torch::Tensor scales, int64_t logical_out_dim,
    int64_t stripe_n, double eps, std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> workspace_opt,
    std::optional<torch::Tensor> locks_opt) {
  TORCH_CHECK(input.is_cuda() && gamma.is_cuda() && qweight.is_cuda() &&
                  zeros.is_cuda() && scales.is_cuda(),
              "all inputs must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 && input.dim() == 2 &&
                  input.size(0) >= 1 && input.size(0) <= 8,
              "input must be float16 [Batch, HiddenDim], Batch in [1, 8]");
  const int64_t batch = input.size(0);
  const int64_t hidden_dim = input.size(1);
  TORCH_CHECK(hidden_dim % 128 == 0, "HiddenDim must be divisible by 128");
  TORCH_CHECK(gamma.scalar_type() == torch::kFloat16 && gamma.dim() == 1 &&
                  gamma.numel() == hidden_dim,
              "gamma must be float16 [HiddenDim]");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8 && qweight.dim() == 2 &&
                  qweight.size(0) == hidden_dim,
              "qweight must be uint8 [HiddenDim, PhysicalOutDim / 2]");
  const int64_t physical_out_dim = qweight.size(1) * 2;
  TORCH_CHECK((stripe_n == 32 || stripe_n == 64 || stripe_n == 128) &&
                  physical_out_dim % stripe_n == 0,
              "stripe_n must be 32, 64, or 128 and divide PhysicalOutDim");
  TORCH_CHECK(zeros.scalar_type() == torch::kUInt8 && zeros.dim() == 2 &&
                  zeros.size(1) == physical_out_dim &&
                  zeros.size(0) == hidden_dim / 128,
              "zeros must be uint8 [Groups, PhysicalOutDim]");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16 && scales.sizes() == zeros.sizes(),
              "scales must be float16 [Groups, PhysicalOutDim]");
  TORCH_CHECK(logical_out_dim > 0 && logical_out_dim <= physical_out_dim,
              "invalid logical output width");
  TORCH_CHECK(input.is_contiguous() && gamma.is_contiguous() &&
                  qweight.is_contiguous() && zeros.is_contiguous() &&
                  scales.is_contiguous(),
              "all inputs must be contiguous");
  const void* bias_ptr = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == torch::kFloat16 &&
                    bias->dim() == 1 && bias->numel() == logical_out_dim &&
                    bias->is_contiguous(),
                "bias must be contiguous float16 [LogicalOutDim]");
    bias_ptr = bias->data_ptr<at::Half>();
  }
  TORCH_CHECK(input.device() == gamma.device() && input.device() == qweight.device() &&
                  input.device() == zeros.device() && input.device() == scales.device(),
              "all tensors must share a CUDA device");
  const c10::cuda::CUDAGuard guard(input.device());
  auto output = torch::empty({batch, logical_out_dim}, input.options());
  constexpr int kSplitK = 8;
  constexpr int kMaxAugmentedRows = 16;
  auto workspace = workspace_opt.has_value() ? *workspace_opt : torch::empty(
      {kSplitK, kMaxAugmentedRows, physical_out_dim},
      input.options().dtype(torch::kFloat));
  auto locks = locks_opt.has_value() ? *locks_opt : torch::zeros(
      {physical_out_dim / stripe_n}, input.options().dtype(torch::kInt));
  TORCH_CHECK(workspace.is_cuda() && workspace.scalar_type() == torch::kFloat &&
                  workspace.is_contiguous() &&
                  workspace.numel() >= kSplitK * kMaxAugmentedRows * physical_out_dim,
              "workspace must be contiguous CUDA float with K-split capacity");
  TORCH_CHECK(locks.is_cuda() && locks.scalar_type() == torch::kInt &&
                  locks.is_contiguous() && locks.numel() >= physical_out_dim / stripe_n,
              "locks must be contiguous CUDA int [PhysicalOutDim / stripe_n]");
  launch_w4a16_marlin_sentinel_wmma(
      input.data_ptr<at::Half>(), gamma.data_ptr<at::Half>(),
      qweight.data_ptr<uint8_t>(), zeros.data_ptr<uint8_t>(),
      scales.data_ptr<at::Half>(), bias_ptr, output.data_ptr<at::Half>(),
      workspace.data_ptr<float>(), locks.data_ptr<int>(),
      static_cast<int>(batch), static_cast<int>(hidden_dim),
      static_cast<int>(logical_out_dim), static_cast<int>(physical_out_dim),
      static_cast<int>(stripe_n), static_cast<float>(eps),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("w4a16_marlin_sentinel_wmma", &w4a16_marlin_sentinel_wmma,
             pybind11::arg("input"), pybind11::arg("gamma"),
             pybind11::arg("qweight"), pybind11::arg("zeros"),
             pybind11::arg("scales"), pybind11::arg("logical_out_dim"),
             pybind11::arg("stripe_n") = 32, pybind11::arg("eps") = 1e-5,
             pybind11::arg("bias") = std::nullopt,
             pybind11::arg("workspace") = std::nullopt,
             pybind11::arg("locks") = std::nullopt);
}
