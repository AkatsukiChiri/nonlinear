#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <vector>

void launch_projection(const half*, const signed char*, const float*, const half*,
                       half*, float*, float*, int, int, int, float, cudaStream_t);
void launch_projection_bf16(const __nv_bfloat16*, const signed char*,
                            const float*, const __nv_bfloat16*,
                            __nv_bfloat16*, float*, int, int, int, float,
                            cudaStream_t);
void launch_projection_add(const half*, const half*, const signed char*,
                           const float*, const half*, half*, half*, float*,
                           int, int, int, float, cudaStream_t);
void launch_projection_add_bf16(
    const __nv_bfloat16*, const __nv_bfloat16*, const signed char*,
    const float*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    float*, int, int, int, float, cudaStream_t);

torch::Tensor run_forward(
    torch::Tensor x, torch::Tensor weight, torch::Tensor scale,
    std::optional<torch::Tensor> bias, double eps, torch::Tensor output) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && scale.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(x.device() == weight.device() && x.device() == scale.device(), "device mismatch");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 ||
              x.scalar_type() == torch::kBFloat16,
              "x must be float16 or bfloat16");
  TORCH_CHECK(weight.scalar_type() == torch::kInt8, "weight must be int8");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be float32");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1, "x must be [Batch, 1, HiddenDim]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(1) == x.size(2), "invalid weight shape");
  TORCH_CHECK(x.size(2) % 64 == 0, "HiddenDim must be divisible by 64");
  TORCH_CHECK(scale.sizes() == torch::IntArrayRef({weight.size(0), 1}), "invalid scale shape");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8, "Batch must be in [1, 8]");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && scale.is_contiguous(), "inputs must be contiguous");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->device() == x.device(), "invalid bias device");
    TORCH_CHECK(bias->scalar_type() == x.scalar_type() && bias->dim() == 1 &&
                bias->size(0) == weight.size(0) && bias->is_contiguous(), "invalid bias");
  }
  TORCH_CHECK(output.is_cuda() && output.device() == x.device() &&
              output.scalar_type() == x.scalar_type() &&
              output.sizes() == torch::IntArrayRef(
                  {x.size(0), 1, weight.size(0)}) && output.is_contiguous(),
              "output buffer has an invalid shape, dtype, device, or layout");
  const c10::cuda::CUDAGuard guard(x.device());
  torch::Tensor square_sum;
  float* square_sum_pointer = nullptr;
  if (x.size(0) > 2) {
    square_sum = torch::empty(
        {x.size(0)}, x.options().dtype(torch::kFloat32));
    square_sum_pointer = square_sum.data_ptr<float>();
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == torch::kFloat16) {
    const half* bias_pointer = bias.has_value()
        ? reinterpret_cast<const half*>(bias->data_ptr<at::Half>()) : nullptr;
    launch_projection(reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const signed char*>(weight.data_ptr<int8_t>()),
        scale.data_ptr<float>(), bias_pointer,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        nullptr, square_sum_pointer, x.size(0),
        weight.size(0), x.size(2), static_cast<float>(eps), stream);
  } else {
    const __nv_bfloat16* bias_pointer = bias.has_value()
        ? reinterpret_cast<const __nv_bfloat16*>(
              bias->data_ptr<at::BFloat16>()) : nullptr;
    launch_projection_bf16(
        reinterpret_cast<const __nv_bfloat16*>(
            x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const signed char*>(weight.data_ptr<int8_t>()),
        scale.data_ptr<float>(), bias_pointer,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        square_sum_pointer, x.size(0), weight.size(0), x.size(2),
        static_cast<float>(eps), stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor forward(torch::Tensor x, torch::Tensor weight, torch::Tensor scale,
                      std::optional<torch::Tensor> bias, double eps) {
  auto output = torch::empty({x.size(0), 1, weight.size(0)}, x.options());
  return run_forward(std::move(x), std::move(weight), std::move(scale),
                     std::move(bias), eps, std::move(output));
}

torch::Tensor forward_out(
    torch::Tensor x, torch::Tensor weight, torch::Tensor scale,
    std::optional<torch::Tensor> bias, double eps, torch::Tensor output) {
  return run_forward(std::move(x), std::move(weight), std::move(scale),
                     std::move(bias), eps, std::move(output));
}

std::vector<torch::Tensor> run_forward_add(
    torch::Tensor x, torch::Tensor x_res, torch::Tensor weight,
    torch::Tensor scale, std::optional<torch::Tensor> bias, double eps,
    torch::Tensor output, torch::Tensor residual_sum) {
  TORCH_CHECK(x_res.is_cuda() && x_res.device() == x.device(),
              "x_res must be on the same CUDA device as x");
  TORCH_CHECK(x_res.scalar_type() == x.scalar_type() &&
              x_res.sizes() == x.sizes() && x_res.is_contiguous(),
              "x_res must match x shape, dtype, and layout");
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && scale.is_cuda(),
              "inputs must be CUDA");
  TORCH_CHECK(x.device() == weight.device() && x.device() == scale.device(),
              "device mismatch");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 ||
              x.scalar_type() == torch::kBFloat16,
              "x must be float16 or bfloat16");
  TORCH_CHECK(weight.scalar_type() == torch::kInt8 && weight.dim() == 2,
              "weight must be int8 [OutDim, HiddenDim]");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32 &&
              scale.sizes() == torch::IntArrayRef({weight.size(0), 1}),
              "scale must be float32 [OutDim, 1]");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1 &&
              weight.size(1) == x.size(2) && x.size(2) % 64 == 0,
              "invalid input shape");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
              "Batch must be in [1, 8]");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && scale.is_contiguous(),
              "inputs must be contiguous");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->device() == x.device() &&
                bias->scalar_type() == x.scalar_type() && bias->dim() == 1 &&
                bias->size(0) == weight.size(0) && bias->is_contiguous(),
                "invalid bias");
  }
  TORCH_CHECK(output.is_cuda() && output.device() == x.device() &&
              output.scalar_type() == x.scalar_type() &&
              output.sizes() == torch::IntArrayRef(
                  {x.size(0), 1, weight.size(0)}) && output.is_contiguous(),
              "output buffer has an invalid shape, dtype, device, or layout");
  TORCH_CHECK(residual_sum.is_cuda() && residual_sum.device() == x.device() &&
              residual_sum.scalar_type() == x.scalar_type() &&
              residual_sum.sizes() == x.sizes() && residual_sum.is_contiguous(),
              "residual buffer must match x shape, dtype, device, and layout");
  const c10::cuda::CUDAGuard guard(x.device());
  torch::Tensor square_sum;
  float* square_sum_pointer = nullptr;
  if (x.size(0) > 2) {
    square_sum = torch::empty(
        {x.size(0)}, x.options().dtype(torch::kFloat32));
    square_sum_pointer = square_sum.data_ptr<float>();
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == torch::kFloat16) {
    const half* bias_pointer = bias.has_value()
        ? reinterpret_cast<const half*>(bias->data_ptr<at::Half>()) : nullptr;
    launch_projection_add(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x_res.data_ptr<at::Half>()),
        reinterpret_cast<const signed char*>(weight.data_ptr<int8_t>()),
        scale.data_ptr<float>(), bias_pointer,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<half*>(residual_sum.data_ptr<at::Half>()),
        square_sum_pointer, x.size(0), weight.size(0), x.size(2),
        static_cast<float>(eps), stream);
  } else {
    const __nv_bfloat16* bias_pointer = bias.has_value()
        ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr<at::BFloat16>())
        : nullptr;
    launch_projection_add_bf16(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            x_res.data_ptr<at::BFloat16>()),
        reinterpret_cast<const signed char*>(weight.data_ptr<int8_t>()),
        scale.data_ptr<float>(), bias_pointer,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            residual_sum.data_ptr<at::BFloat16>()),
        square_sum_pointer, x.size(0), weight.size(0), x.size(2),
        static_cast<float>(eps), stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, residual_sum};
}

std::vector<torch::Tensor> forward_add(
    torch::Tensor x, torch::Tensor x_res, torch::Tensor weight,
    torch::Tensor scale, std::optional<torch::Tensor> bias, double eps) {
  auto output = torch::empty({x.size(0), 1, weight.size(0)}, x.options());
  auto residual_sum = torch::empty_like(x);
  return run_forward_add(
      std::move(x), std::move(x_res), std::move(weight), std::move(scale),
      std::move(bias), eps, std::move(output), std::move(residual_sum));
}

std::vector<torch::Tensor> forward_add_out(
    torch::Tensor x, torch::Tensor x_res, torch::Tensor weight,
    torch::Tensor scale, std::optional<torch::Tensor> bias, double eps,
    torch::Tensor output, torch::Tensor residual_sum) {
  return run_forward_add(
      std::move(x), std::move(x_res), std::move(weight), std::move(scale),
      std::move(bias), eps, std::move(output), std::move(residual_sum));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &forward);
  module.def("forward_out", &forward_out);
  module.def("forward_add", &forward_add);
  module.def("forward_add_out", &forward_add_out);
}
