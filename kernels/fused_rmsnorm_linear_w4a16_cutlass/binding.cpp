#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <optional>

void launch_cutlass_w4a16_probe(const at::Half* a, const uint8_t* b,
                                at::Half* output, int rows, int out_dim,
                                int hidden_dim, cudaStream_t stream);

void launch_cutlass_w4a16_rms_sentinel(
    const at::Half* input, const at::Half* gamma,
    const uint8_t* packed_storage,
    const at::Half* bias, at::Half* output, float* workspace, int* counters,
    int batch, int logical_out_dim, int physical_out_dim, int hidden_dim,
    int split_k_slices, float eps, cudaStream_t stream);

void launch_w4a16_async_decode(
    const at::Half* input, const at::Half* gamma, const uint8_t* storage,
    const at::Half* bias, at::Half* output, int batch, int logical_n,
    int physical_n, int hidden_dim, int cta_n, float eps,
    cudaStream_t stream);

torch::Tensor cutlass_w4a16_probe(torch::Tensor a, torch::Tensor b,
                                  int64_t out_dim) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == torch::kFloat16 && a.dim() == 2,
              "a must be float16 [Rows, HiddenDim]");
  TORCH_CHECK(b.scalar_type() == torch::kUInt8 && b.dim() == 1,
              "b must be a packed W4/zero/scale storage blob");
  const int64_t hidden_dim = a.size(1);
  TORCH_CHECK(hidden_dim % 128 == 0,
              "HiddenDim must be divisible by group size 128");
  const int64_t expected_bytes = hidden_dim * out_dim / 2 +
      hidden_dim / 128 * out_dim * 3;
  TORCH_CHECK(b.numel() == expected_bytes, "invalid packed storage size");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
              "inputs must be contiguous");
  TORCH_CHECK(a.device() == b.device(), "device mismatch");
  TORCH_CHECK(a.size(1) % 64 == 0 && out_dim % 64 == 0,
              "probe requires K and N divisible by 64");
  const c10::cuda::CUDAGuard guard(a.device());
  // Keep C finite while the probe uses the universal C/D interface. The
  // production epilogue will not consume C.
  auto output = torch::zeros({a.size(0), out_dim}, a.options());
  launch_cutlass_w4a16_probe(
      a.data_ptr<at::Half>(), b.data_ptr<uint8_t>(),
      output.data_ptr<at::Half>(), static_cast<int>(a.size(0)),
      static_cast<int>(out_dim), static_cast<int>(hidden_dim),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor cutlass_w4a16_rms_sentinel(
    torch::Tensor input, torch::Tensor gamma, torch::Tensor packed_storage,
    int64_t logical_out_dim, int64_t physical_out_dim, int64_t hidden_dim,
    double eps, std::optional<torch::Tensor> bias, torch::Tensor workspace,
    torch::Tensor counters, int64_t split_k_slices) {
  TORCH_CHECK(input.is_cuda() && gamma.is_cuda() && packed_storage.is_cuda(),
              "inputs must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 && input.dim() == 2 &&
                  input.size(1) == hidden_dim && input.size(0) <= 8,
              "input must be float16 [Batch, HiddenDim]");
  TORCH_CHECK(gamma.scalar_type() == torch::kFloat16 && gamma.dim() == 1 &&
                  gamma.numel() == hidden_dim,
              "gamma must be float16 [HiddenDim]");
  TORCH_CHECK(packed_storage.scalar_type() == torch::kUInt8 &&
                  packed_storage.dim() == 1,
              "packed_storage must be a uint8 vector");
  TORCH_CHECK(hidden_dim % 128 == 0 && physical_out_dim % 64 == 0,
              "HiddenDim must divide 128 and physical N must divide 64");
  TORCH_CHECK(split_k_slices >= 1 && split_k_slices <= hidden_dim / 64,
              "split_k_slices must be in [1, HiddenDim / 64]");
  const int64_t groups = hidden_dim / 128;
  const int64_t expected_bytes = hidden_dim * physical_out_dim / 2 +
      groups * physical_out_dim * 3;
  TORCH_CHECK(packed_storage.numel() == expected_bytes,
              "invalid packed storage size");
  TORCH_CHECK(input.is_contiguous() && gamma.is_contiguous() &&
                  packed_storage.is_contiguous(),
              "inputs must be contiguous");
  const at::Half* bias_pointer = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == torch::kFloat16 &&
                    bias->numel() == logical_out_dim && bias->is_contiguous(),
                "bias must be contiguous CUDA float16 [OutDim]");
    bias_pointer = bias->data_ptr<at::Half>();
  }
  const int64_t batch = input.size(0);
  const int64_t tile_count = physical_out_dim / 64;
  const int64_t workspace_elements =
      split_k_slices * tile_count * 8 * 64;
  TORCH_CHECK(workspace.is_cuda() && workspace.scalar_type() == torch::kFloat32 &&
                  workspace.is_contiguous() &&
                  workspace.numel() >= workspace_elements,
              "workspace must be contiguous CUDA float32 with sufficient size");
  TORCH_CHECK(counters.is_cuda() && counters.scalar_type() == torch::kInt32 &&
                  counters.is_contiguous() && counters.numel() >= tile_count,
              "counters must be contiguous CUDA int32 with sufficient size");
  TORCH_CHECK(input.device() == gamma.device() &&
                  input.device() == packed_storage.device() &&
                  input.device() == workspace.device() &&
                  input.device() == counters.device(),
              "all tensors must be on the same CUDA device");
  auto output = torch::empty({batch, logical_out_dim}, input.options());
  const c10::cuda::CUDAGuard guard(input.device());
  launch_cutlass_w4a16_rms_sentinel(
      input.data_ptr<at::Half>(), gamma.data_ptr<at::Half>(),
      packed_storage.data_ptr<uint8_t>(),
      bias_pointer, output.data_ptr<at::Half>(), workspace.data_ptr<float>(),
      counters.data_ptr<int>(), static_cast<int>(batch),
      static_cast<int>(logical_out_dim), static_cast<int>(physical_out_dim),
      static_cast<int>(hidden_dim), static_cast<int>(split_k_slices),
      static_cast<float>(eps),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor w4a16_async_decode(
    torch::Tensor input, torch::Tensor gamma, torch::Tensor packed_storage,
    int64_t logical_out_dim, int64_t physical_out_dim, int64_t hidden_dim,
    int64_t cta_n, double eps, std::optional<torch::Tensor> bias) {
  TORCH_CHECK(input.is_cuda() && gamma.is_cuda() && packed_storage.is_cuda(),
              "inputs must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 && input.dim() == 2 &&
                  input.size(1) == hidden_dim && input.size(0) <= 8,
              "input must be float16 [Batch, HiddenDim]");
  TORCH_CHECK(gamma.scalar_type() == torch::kFloat16 && gamma.dim() == 1 &&
                  gamma.numel() == hidden_dim,
              "gamma must be float16 [HiddenDim]");
  TORCH_CHECK(packed_storage.scalar_type() == torch::kUInt8 &&
                  packed_storage.dim() == 1,
              "packed_storage must be a uint8 vector");
  TORCH_CHECK((cta_n == 64 || cta_n == 128) &&
                  physical_out_dim % cta_n == 0,
              "physical N must divide the selected async CTA width");
  TORCH_CHECK(hidden_dim % 128 == 0 && hidden_dim % 64 == 0,
              "HiddenDim must be divisible by 128");
  const int64_t groups = hidden_dim / 128;
  const int64_t expected_bytes = hidden_dim * physical_out_dim / 2 +
      groups * physical_out_dim * 3;
  TORCH_CHECK(packed_storage.numel() == expected_bytes,
              "invalid packed storage size");
  TORCH_CHECK(input.is_contiguous() && gamma.is_contiguous() &&
                  packed_storage.is_contiguous(),
              "inputs must be contiguous");
  const at::Half* bias_pointer = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == torch::kFloat16 &&
                    bias->numel() == logical_out_dim && bias->is_contiguous(),
                "bias must be contiguous CUDA float16 [OutDim]");
    bias_pointer = bias->data_ptr<at::Half>();
  }
  TORCH_CHECK(input.device() == gamma.device() &&
                  input.device() == packed_storage.device(),
              "all tensors must be on the same CUDA device");
  auto output = torch::empty(
      {input.size(0), logical_out_dim}, input.options());
  const c10::cuda::CUDAGuard guard(input.device());
  launch_w4a16_async_decode(
      input.data_ptr<at::Half>(), gamma.data_ptr<at::Half>(),
      packed_storage.data_ptr<uint8_t>(), bias_pointer,
      output.data_ptr<at::Half>(), static_cast<int>(input.size(0)),
      static_cast<int>(logical_out_dim), static_cast<int>(physical_out_dim),
      static_cast<int>(hidden_dim), static_cast<int>(cta_n),
      static_cast<float>(eps), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("cutlass_w4a16_probe", &cutlass_w4a16_probe);
  module.def("cutlass_w4a16_rms_sentinel", &cutlass_w4a16_rms_sentinel);
  module.def("w4a16_async_decode", &w4a16_async_decode);
}
