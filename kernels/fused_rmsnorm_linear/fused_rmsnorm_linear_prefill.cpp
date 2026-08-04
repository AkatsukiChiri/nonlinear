#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

void launch_prefill_tensor_core_projection_fp16(
    const half* x,
    const half* weight,
    half* projection,
    int token_count,
    int out_dim,
    int hidden_dim,
    cudaStream_t stream);

void launch_prefill_rms_square_sum_fp16(
    const half* x,
    float* square_sum,
    int token_count,
    int hidden_dim,
    cudaStream_t stream);

void launch_prefill_rms_scale_fp16(
    const half* projection,
    const float* square_sum,
    const half* bias,
    half* output,
    int token_count,
    int out_dim,
    int hidden_dim,
    float eps,
    cudaStream_t stream);

void launch_prefill_inverse_rms_fp16(
    const half* x,
    float* inverse_rms,
    int token_count,
    int hidden_dim,
    float eps,
    cudaStream_t stream);

void launch_prefill_tensor_core_scaled_projection_fp16(
    const half* x,
    const half* weight,
    const float* inverse_rms,
    half* output,
    int token_count,
    int out_dim,
    int hidden_dim,
    cudaStream_t stream);

namespace {

struct PrefillEvents {
  cudaEvent_t input_ready = nullptr;
  cudaEvent_t rms_ready = nullptr;
  int device_index = -1;

  ~PrefillEvents() {
    if (input_ready != nullptr) {
      cudaEventDestroy(input_ready);
    }
    if (rms_ready != nullptr) {
      cudaEventDestroy(rms_ready);
    }
  }

  void initialize(int requested_device) {
    if (device_index == requested_device) {
      return;
    }
    if (input_ready != nullptr) {
      C10_CUDA_CHECK(cudaEventDestroy(input_ready));
      input_ready = nullptr;
    }
    if (rms_ready != nullptr) {
      C10_CUDA_CHECK(cudaEventDestroy(rms_ready));
      rms_ready = nullptr;
    }
    device_index = requested_device;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(
        &input_ready, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(
        &rms_ready, cudaEventDisableTiming));
  }
};

thread_local PrefillEvents prefill_events;

void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& fused_weight,
    const c10::optional<torch::Tensor>& bias,
    double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(fused_weight.is_cuda(), "fused_weight must be a CUDA tensor");
  TORCH_CHECK(
      x.device() == fused_weight.device(),
      "x and fused_weight must be on one device");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
  TORCH_CHECK(
      fused_weight.scalar_type() == torch::kFloat16,
      "fused_weight must be float16");
  TORCH_CHECK(
      x.dim() == 3,
      "x must have shape [Batch, SeqLen, HiddenDim]");
  TORCH_CHECK(
      fused_weight.dim() == 2,
      "fused_weight must have shape [OutDim, HiddenDim]");
  TORCH_CHECK(
      x.size(2) == fused_weight.size(1),
      "hidden dimensions must match");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(
      fused_weight.is_contiguous(), "fused_weight must be contiguous");
  TORCH_CHECK(
      x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0 &&
          fused_weight.size(0) > 0,
      "all dimensions must be positive");
  TORCH_CHECK(
      x.size(2) % 16 == 0,
      "HiddenDim must be divisible by 16 for the Tensor Core kernel");
  TORCH_CHECK(
      fused_weight.size(0) % 16 == 0,
      "OutDim must be divisible by 16 for the Tensor Core kernel");
  TORCH_CHECK(eps >= 0.0, "eps must be non-negative");

  if (bias.has_value()) {
    const torch::Tensor& bias_tensor = bias.value();
    TORCH_CHECK(bias_tensor.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(
        bias_tensor.device() == x.device(),
        "x, fused_weight, and bias must be on one device");
    TORCH_CHECK(
        bias_tensor.scalar_type() == torch::kFloat16,
        "bias must be float16");
    TORCH_CHECK(
        bias_tensor.dim() == 1 &&
            bias_tensor.size(0) == fused_weight.size(0),
        "bias must have shape [OutDim]");
    TORCH_CHECK(bias_tensor.is_contiguous(), "bias must be contiguous");
  }
}

torch::Tensor run_prefill(
    torch::Tensor x,
    torch::Tensor fused_weight,
    const c10::optional<torch::Tensor>& bias,
    double eps) {
  check_inputs(x, fused_weight, bias, eps);
  const c10::cuda::CUDAGuard device_guard(x.device());

  const int batch = static_cast<int>(x.size(0));
  const int seq_len = static_cast<int>(x.size(1));
  const int token_count = batch * seq_len;
  const int hidden_dim = static_cast<int>(x.size(2));
  const int out_dim = static_cast<int>(fused_weight.size(0));

  // Reserve the small FP32 RMS workspace at the tail of the output storage.
  // token_count*out_dim is a multiple of 16, so the tail remains naturally
  // aligned for float. The returned projection view owns the full storage.
  const int64_t output_elements =
      static_cast<int64_t>(token_count) * out_dim;
  auto output_storage = torch::empty(
      {output_elements + static_cast<int64_t>(token_count) * 2},
      x.options());
  auto projection = output_storage.narrow(0, 0, output_elements).view(
      {batch, seq_len, out_dim});
  float* rms_workspace = reinterpret_cast<float*>(
      output_storage.data_ptr<at::Half>() + output_elements);

  const cudaStream_t current_stream = at::cuda::getCurrentCUDAStream();

  // The no-bias path uses CUTLASS's broadcast epilogue. It scales FP32
  // accumulators before their only global-memory store, eliminating both the
  // projection readback and the standalone elementwise launch.
  if (!bias.has_value()) {
    launch_prefill_inverse_rms_fp16(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        rms_workspace,
        token_count,
        hidden_dim,
        static_cast<float>(eps),
        current_stream);
    launch_prefill_tensor_core_scaled_projection_fp16(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(fused_weight.data_ptr<at::Half>()),
        rms_workspace,
        reinterpret_cast<half*>(projection.data_ptr<at::Half>()),
        token_count,
        out_dim,
        hidden_dim,
        current_stream);
    return projection;
  }

  const c10::cuda::CUDAStream rms_stream =
      c10::cuda::getStreamFromPool(false, x.get_device());
  prefill_events.initialize(x.get_device());

  // The caller may have produced x on the current stream immediately before
  // this operation. Make that dependency explicit before the auxiliary RMS
  // stream reads x.
  C10_CUDA_CHECK(cudaEventRecord(
      prefill_events.input_ready, current_stream));
  C10_CUDA_CHECK(
      cudaStreamWaitEvent(
          rms_stream.stream(), prefill_events.input_ready, 0));

  launch_prefill_tensor_core_projection_fp16(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(fused_weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(projection.data_ptr<at::Half>()),
      token_count,
      out_dim,
      hidden_dim,
      current_stream);

  launch_prefill_rms_square_sum_fp16(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      rms_workspace,
      token_count,
      hidden_dim,
      rms_stream.stream());

  C10_CUDA_CHECK(cudaEventRecord(
      prefill_events.rms_ready, rms_stream.stream()));
  C10_CUDA_CHECK(cudaStreamWaitEvent(
      current_stream, prefill_events.rms_ready, 0));

  const half* bias_pointer = nullptr;
  if (bias.has_value()) {
    bias_pointer = reinterpret_cast<const half*>(
        bias.value().data_ptr<at::Half>());
  }
  launch_prefill_rms_scale_fp16(
      reinterpret_cast<const half*>(projection.data_ptr<at::Half>()),
      rms_workspace,
      bias_pointer,
      reinterpret_cast<half*>(projection.data_ptr<at::Half>()),
      token_count,
      out_dim,
      hidden_dim,
      static_cast<float>(eps),
      current_stream);
  return projection;
}

}  // namespace

torch::Tensor fused_rmsnorm_linear_prefill_fp16(
    torch::Tensor x,
    torch::Tensor fused_weight,
    double eps) {
  return run_prefill(x, fused_weight, c10::nullopt, eps);
}

torch::Tensor fused_rmsnorm_linear_prefill_bias_fp16(
    torch::Tensor x,
    torch::Tensor fused_weight,
    torch::Tensor bias,
    double eps) {
  return run_prefill(x, fused_weight, bias, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "fused_rmsnorm_linear_prefill_fp16",
      &fused_rmsnorm_linear_prefill_fp16,
      "Tensor Core prefill RMSNorm and Linear FP16 kernel");
  module.def(
      "fused_rmsnorm_linear_prefill_bias_fp16",
      &fused_rmsnorm_linear_prefill_bias_fp16,
      "Tensor Core prefill RMSNorm and biased Linear FP16 kernel");
}
