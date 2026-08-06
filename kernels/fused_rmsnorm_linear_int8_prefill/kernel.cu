#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/threadblock/fusion/visitor_compute.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitor_load.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitor_store.hpp>
#include <cutlass/layout/matrix.h>
#include <type_traits>

namespace {
constexpr int kQuantThreads = 256;

template <typename Input>
__device__ __forceinline__ float input_to_float(Input value);

template <typename Input>
__device__ __forceinline__ Input input_from_float(float value);

template <>
__device__ __forceinline__ float input_to_float(half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ half input_from_float(float value) {
  return __float2half(value);
}

template <>
__device__ __forceinline__ float input_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 input_from_float(float value) {
  return __float2bfloat16_rn(value);
}

template <typename Input>
__global__ __launch_bounds__(kQuantThreads) void quantize_rms_kernel(
    const Input* __restrict__ x, int8_t* __restrict__ quantized_x,
    float* __restrict__ row_scale, int hidden_dim, float eps) {
  extern __shared__ unsigned char shared_storage[];
  Input* shared_x = reinterpret_cast<Input*>(shared_storage);
  __shared__ float warp_max[kQuantThreads / 32];
  __shared__ float warp_sum[kQuantThreads / 32];
  __shared__ float scale_shared;
  float local_max = 0.0f;
  float local_sum = 0.0f;
  const int row = blockIdx.x;
  const Input* input = x + row * hidden_dim;
  for (int index = threadIdx.x; index < hidden_dim; index += blockDim.x) {
    const Input value_raw = input[index];
    shared_x[index] = value_raw;
    const float value = input_to_float(value_raw);
    local_max = fmaxf(local_max, fabsf(value));
    local_sum = fmaf(value, value, local_sum);
  }
  for (int offset = 16; offset; offset >>= 1) {
    local_max = fmaxf(local_max,
        __shfl_down_sync(0xffffffff, local_max, offset));
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_max[warp] = local_max;
    warp_sum[warp] = local_sum;
  }
  __syncthreads();
  if (warp == 0) {
    float maximum = lane < kQuantThreads / 32 ? warp_max[lane] : 0.0f;
    float sum = lane < kQuantThreads / 32 ? warp_sum[lane] : 0.0f;
    for (int offset = 16; offset; offset >>= 1) {
      maximum = fmaxf(maximum,
          __shfl_down_sync(0xffffffff, maximum, offset));
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
      const float activation_scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
      scale_shared = activation_scale;
      row_scale[row] = activation_scale *
          rsqrtf(sum / static_cast<float>(hidden_dim) + eps);
    }
  }
  __syncthreads();
  const float inverse_scale = 1.0f / scale_shared;
  for (int index = threadIdx.x; index < hidden_dim; index += blockDim.x) {
    int value = __float2int_rn(input_to_float(shared_x[index]) * inverse_scale);
    value = max(-127, min(127, value));
    quantized_x[row * hidden_dim + index] = static_cast<int8_t>(value);
  }
}

template <typename Input>
__global__ __launch_bounds__(kQuantThreads) void quantize_add_rms_kernel(
    const Input* __restrict__ x, const Input* __restrict__ x_res,
    Input* __restrict__ residual_sum, int8_t* __restrict__ quantized_x,
    float* __restrict__ row_scale, int hidden_dim, float eps) {
  extern __shared__ unsigned char shared_storage[];
  Input* shared_x = reinterpret_cast<Input*>(shared_storage);
  __shared__ float warp_max[kQuantThreads / 32];
  __shared__ float warp_sum[kQuantThreads / 32];
  __shared__ float scale_shared;
  float local_max = 0.0f;
  float local_sum = 0.0f;
  const int row = blockIdx.x;
  const int row_offset = row * hidden_dim;
  for (int index = threadIdx.x * 2; index < hidden_dim;
       index += blockDim.x * 2) {
    const int offset = row_offset + index;
    float2 values;
    if constexpr (std::is_same_v<Input, half>) {
      const half2 sum = __hadd2(
          *reinterpret_cast<const half2*>(x + offset),
          *reinterpret_cast<const half2*>(x_res + offset));
      *reinterpret_cast<half2*>(residual_sum + offset) = sum;
      *reinterpret_cast<half2*>(shared_x + index) = sum;
      values = __half22float2(sum);
    } else {
      const __nv_bfloat162 sum = __hadd2(
          *reinterpret_cast<const __nv_bfloat162*>(x + offset),
          *reinterpret_cast<const __nv_bfloat162*>(x_res + offset));
      *reinterpret_cast<__nv_bfloat162*>(residual_sum + offset) = sum;
      *reinterpret_cast<__nv_bfloat162*>(shared_x + index) = sum;
      values = __bfloat1622float2(sum);
    }
    local_max = fmaxf(local_max, fmaxf(fabsf(values.x), fabsf(values.y)));
    local_sum = fmaf(values.x, values.x, local_sum);
    local_sum = fmaf(values.y, values.y, local_sum);
  }
  for (int offset = 16; offset; offset >>= 1) {
    local_max = fmaxf(local_max,
        __shfl_down_sync(0xffffffff, local_max, offset));
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_max[warp] = local_max;
    warp_sum[warp] = local_sum;
  }
  __syncthreads();
  if (warp == 0) {
    float maximum = lane < kQuantThreads / 32 ? warp_max[lane] : 0.0f;
    float sum = lane < kQuantThreads / 32 ? warp_sum[lane] : 0.0f;
    for (int offset = 16; offset; offset >>= 1) {
      maximum = fmaxf(maximum,
          __shfl_down_sync(0xffffffff, maximum, offset));
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
      const float activation_scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
      scale_shared = activation_scale;
      row_scale[row] = activation_scale *
          rsqrtf(sum / static_cast<float>(hidden_dim) + eps);
    }
  }
  __syncthreads();
  const float inverse_scale = 1.0f / scale_shared;
  for (int index = threadIdx.x * 2; index < hidden_dim;
       index += blockDim.x * 2) {
    float2 values;
    if constexpr (std::is_same_v<Input, half>) {
      values = __half22float2(
          *reinterpret_cast<const half2*>(shared_x + index));
    } else {
      values = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(shared_x + index));
    }
    int value_0 = __float2int_rn(values.x * inverse_scale);
    int value_1 = __float2int_rn(values.y * inverse_scale);
    value_0 = max(-127, min(127, value_0));
    value_1 = max(-127, min(127, value_1));
    const unsigned short packed =
        static_cast<unsigned char>(static_cast<int8_t>(value_0)) |
        (static_cast<unsigned short>(
             static_cast<unsigned char>(static_cast<int8_t>(value_1))) << 8);
    *reinterpret_cast<unsigned short*>(quantized_x + row_offset + index) = packed;
  }
}

template <typename Output>
__global__ void add_bias_kernel(Output* output, const Output* bias, int total,
                                int out_dim);

template <>
__global__ void add_bias_kernel(half* output, const half* bias, int total,
                                int out_dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total)
    output[index] = __hadd(output[index], bias[index % out_dim]);
}

template <>
__global__ void add_bias_kernel(__nv_bfloat16* output,
                                const __nv_bfloat16* bias, int total,
                                int out_dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total) {
    const float value = __bfloat162float(output[index]) +
                        __bfloat162float(bias[index % out_dim]);
    output[index] = __float2bfloat16_rn(value);
  }
}
}  // namespace

void launch_quantize_rms_int8_prefill(const half* x, int8_t* quantized_x,
                                      float* row_scale, int rows,
                                      int hidden_dim, float eps,
                                      cudaStream_t stream) {
  quantize_rms_kernel<half><<<
      rows, kQuantThreads, hidden_dim * sizeof(half), stream>>>(
      x, quantized_x, row_scale, hidden_dim, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_quantize_rms_int8_prefill_bf16(
    const __nv_bfloat16* x, int8_t* quantized_x, float* row_scale, int rows,
    int hidden_dim, float eps, cudaStream_t stream) {
  quantize_rms_kernel<__nv_bfloat16><<<
      rows, kQuantThreads, hidden_dim * sizeof(__nv_bfloat16), stream>>>(
      x, quantized_x, row_scale, hidden_dim, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_quantize_add_rms_int8_prefill(
    const half* x, const half* x_res, half* residual_sum,
    int8_t* quantized_x, float* row_scale, int rows, int hidden_dim,
    float eps, cudaStream_t stream) {
  quantize_add_rms_kernel<half><<<
      rows, kQuantThreads, hidden_dim * sizeof(half), stream>>>(
      x, x_res, residual_sum, quantized_x, row_scale, hidden_dim, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_quantize_add_rms_int8_prefill_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* x_res,
    __nv_bfloat16* residual_sum, int8_t* quantized_x, float* row_scale,
    int rows, int hidden_dim, float eps, cudaStream_t stream) {
  quantize_add_rms_kernel<__nv_bfloat16><<<
      rows, kQuantThreads, hidden_dim * sizeof(__nv_bfloat16), stream>>>(
      x, x_res, residual_sum, quantized_x, row_scale, hidden_dim, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Output, typename CudaOutput>
void launch_gemm_int8_prefill_typed(
    const int8_t* x, const int8_t* weight, const float* row_scale,
    const float* weight_scale, const CudaOutput* bias, CudaOutput* output,
    int rows, int out_dim, int hidden_dim, cudaStream_t stream) {
  using Element = cutlass::int8_t;
  using Accumulator = int32_t;
  using Compute = float;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape, WarpShape, Output, 8, 1>;
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  using RowScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
      OutputTileThreadMap, Compute,
      cute::Stride<cute::_1, cute::_0, int64_t>>;
  using WeightScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, Compute,
      cute::Stride<cute::_0, cute::_1, int64_t>>;
  using Multiply = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, Compute, Compute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using ScaleRows = cutlass::epilogue::threadblock::Sm80EVT<
      Multiply, Accum, RowScale>;
  using ScaleColumns = cutlass::epilogue::threadblock::Sm80EVT<
      Multiply, ScaleRows, WeightScale>;
  using Store = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, Output, cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, int64_t>>;
  using Epilogue = cutlass::epilogue::threadblock::Sm80EVT<Store, ScaleColumns>;
  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
      Element, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
      Output, cutlass::layout::RowMajor, 8,
      Accumulator, Compute,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      ThreadblockShape, WarpShape, InstructionShape, Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3,
      cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  typename Epilogue::Arguments epilogue_arguments{
      {
          {
              {},
              {row_scale, 0.0f,
               {cute::_1{}, cute::_0{}, static_cast<int64_t>(rows)}},
              {},
          },
          {weight_scale, 0.0f,
           {cute::_0{}, cute::_1{}, static_cast<int64_t>(out_dim)}},
          {},
      },
      {reinterpret_cast<Output*>(output),
       {static_cast<int64_t>(out_dim), cute::_1{},
        static_cast<int64_t>(rows) * out_dim}}};
  typename Gemm::Arguments arguments(
      cutlass::gemm::GemmUniversalMode::kGemm,
      {rows, out_dim, hidden_dim}, 1, epilogue_arguments,
      reinterpret_cast<const Element*>(x),
      reinterpret_cast<const Element*>(weight), nullptr, nullptr,
      static_cast<int64_t>(rows) * hidden_dim,
      static_cast<int64_t>(out_dim) * hidden_dim, 0, 0,
      hidden_dim, hidden_dim, 0, 0, nullptr, nullptr, nullptr);
  Gemm gemm;
  const cutlass::Status status = gemm(arguments, nullptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS INT8 prefill GEMM failed: ",
              cutlassGetStatusString(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (bias != nullptr) {
    const int total = rows * out_dim;
    add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        output, bias, total, out_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void launch_gemm_int8_prefill(const int8_t* x, const int8_t* weight,
                              const float* row_scale,
                              const float* weight_scale, const half* bias,
                              half* output, int rows, int out_dim,
                              int hidden_dim, cudaStream_t stream) {
  launch_gemm_int8_prefill_typed<cutlass::half_t>(
      x, weight, row_scale, weight_scale, bias, output, rows, out_dim,
      hidden_dim, stream);
}

void launch_gemm_int8_prefill_bf16(
    const int8_t* x, const int8_t* weight, const float* row_scale,
    const float* weight_scale, const __nv_bfloat16* bias,
    __nv_bfloat16* output, int rows, int out_dim, int hidden_dim,
    cudaStream_t stream) {
  launch_gemm_int8_prefill_typed<cutlass::bfloat16_t>(
      x, weight, row_scale, weight_scale, bias, output, rows, out_dim,
      hidden_dim, stream);
}
