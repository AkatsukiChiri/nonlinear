#include <cuda_fp16.h>
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

namespace {
constexpr int kQuantThreads = 256;

__global__ __launch_bounds__(kQuantThreads) void quantize_rms_kernel(
    const half* __restrict__ x, uint8_t* __restrict__ quantized_x,
    float* __restrict__ row_scale, int hidden_dim, float eps) {
  extern __shared__ half shared_x[];
  __shared__ float warp_max[kQuantThreads / 32];
  __shared__ float warp_sum[kQuantThreads / 32];
  __shared__ float scale_shared;
  float local_max = 0.0f;
  float local_sum = 0.0f;
  const int row = blockIdx.x;
  const half* input = x + row * hidden_dim;
  for (int pair = threadIdx.x; pair < hidden_dim / 2; pair += blockDim.x) {
    const half2 values = reinterpret_cast<const half2*>(input)[pair];
    reinterpret_cast<half2*>(shared_x)[pair] = values;
    const float2 converted = __half22float2(values);
    local_max = fmaxf(local_max,
        fmaxf(fabsf(converted.x), fabsf(converted.y)));
    local_sum = fmaf(converted.x, converted.x, local_sum);
    local_sum = fmaf(converted.y, converted.y, local_sum);
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
      const float activation_scale = maximum > 0.0f ? maximum / 7.0f : 1.0f;
      scale_shared = activation_scale;
      row_scale[row] = activation_scale *
          rsqrtf(sum / static_cast<float>(hidden_dim) + eps);
    }
  }
  __syncthreads();
  const float inverse_scale = 1.0f / scale_shared;
  for (int pair = threadIdx.x; pair < hidden_dim / 2; pair += blockDim.x) {
    int low = __float2int_rn(__half2float(shared_x[pair * 2]) * inverse_scale);
    int high = __float2int_rn(
        __half2float(shared_x[pair * 2 + 1]) * inverse_scale);
    low = max(-7, min(7, low));
    high = max(-7, min(7, high));
    quantized_x[row * (hidden_dim / 2) + pair] =
        static_cast<uint8_t>((low & 0xf) | ((high & 0xf) << 4));
  }
}

__global__ void add_bias_kernel(half* output, const half* bias, int total,
                                int out_dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total)
    output[index] = __hadd(output[index], bias[index % out_dim]);
}
}  // namespace

void launch_quantize_rms_int4_prefill(const half* x, uint8_t* quantized_x,
                                      float* row_scale, int rows,
                                      int hidden_dim, float eps,
                                      cudaStream_t stream) {
  quantize_rms_kernel<<<rows, kQuantThreads, hidden_dim * sizeof(half), stream>>>(
      x, quantized_x, row_scale, hidden_dim, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_gemm_int4_prefill(const uint8_t* x, const uint8_t* weight,
                              const float* row_scale,
                              const float* weight_scale, const half* bias,
                              half* output, int rows, int out_dim,
                              int hidden_dim, cudaStream_t stream) {
  using Element = cutlass::int4b_t;
  using Output = cutlass::half_t;
  using Accumulator = int32_t;
  using Compute = float;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 128>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
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
      Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 32,
      Element, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 32,
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
              "CUTLASS INT4 prefill GEMM failed: ",
              cutlassGetStatusString(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (bias != nullptr) {
    const int total = rows * out_dim;
    add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        output, bias, total, out_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
