#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/threadblock/fusion/visitor_compute.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitor_load.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitor_store.hpp>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>

namespace nonlinear::prefill {

namespace wmma = nvcuda::wmma;

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpSize * kWarpsPerBlock;
constexpr int kTile = 16;
constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kStageK = 32;
constexpr int kStageStride = kStageK + 8;
constexpr int kWarpTilesM = 2;
constexpr int kWarpTilesN = 4;
constexpr int kRmsThreads = 256;
constexpr int kEpilogueThreads = 256;

__device__ __forceinline__ void copy_async_16(
    void* shared_destination,
    const void* global_source) {
  const unsigned shared_address =
      static_cast<unsigned>(__cvta_generic_to_shared(shared_destination));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n" ::
          "r"(shared_address),
          "l"(global_source));
}

__global__ __launch_bounds__(kThreads) void tensor_core_projection_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ projection,
    int token_count,
    int out_dim,
    int hidden_dim) {
  const int warp_index = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int warp_m = warp_index / 2;
  const int warp_n = warp_index % 2;
  const int block_row = blockIdx.y * kBlockM;
  const int block_column = blockIdx.x * kBlockN;

  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>
      accumulator[kWarpTilesM][kWarpTilesN];
#pragma unroll
  for (int tile_m = 0; tile_m < kWarpTilesM; ++tile_m) {
#pragma unroll
    for (int tile_n = 0; tile_n < kWarpTilesN; ++tile_n) {
      wmma::fill_fragment(accumulator[tile_m][tile_n], 0.0f);
    }
  }

  union SharedStorage {
    struct {
      half x[2][kBlockM * kStageStride];
      // Physical weight rows form a column-major matrix-B tile.
      half weight[2][kBlockN * kStageStride];
    } operand;
    float output[kWarpsPerBlock * kTile * kTile];
  };
  __shared__ SharedStorage shared;

  const bool full_tile =
      block_row + kBlockM <= token_count &&
      block_column + kBlockN <= out_dim;
  if (full_tile) {
    // Prime an Ampere cp.async pipeline. While WMMA consumes one stage, the
    // next 32-wide K tile is copied into the other stage.
    int current_stage = 0;
    {
      for (int load = 0; load < 2; ++load) {
        const int vector_index = threadIdx.x + load * blockDim.x;
        const int x_row = vector_index / 4;
        const int x_vector = vector_index % 4;
        copy_async_16(
            reinterpret_cast<uint4*>(
                shared.operand.x[0] + x_row * kStageStride) +
                x_vector,
            reinterpret_cast<const uint4*>(
                x + (block_row + x_row) * hidden_dim) +
                x_vector);
      }
#pragma unroll
      for (int load = 0; load < 2; ++load) {
        const int vector_index = threadIdx.x + load * blockDim.x;
        const int weight_column = vector_index / 4;
        const int weight_vector = vector_index % 4;
        copy_async_16(
            reinterpret_cast<uint4*>(
                shared.operand.weight[0] +
                weight_column * kStageStride) +
                weight_vector,
            reinterpret_cast<const uint4*>(
                weight +
                (block_column + weight_column) * hidden_dim) +
                weight_vector);
      }
      asm volatile("cp.async.commit_group;\n" ::);
      asm volatile("cp.async.wait_group 0;\n" ::);
      __syncthreads();
    }

    for (int k_block = 0; k_block < hidden_dim; k_block += kStageK) {
      const int next_k = k_block + kStageK;
      const int next_stage = current_stage ^ 1;
      if (next_k < hidden_dim) {
        for (int load = 0; load < 2; ++load) {
          const int vector_index = threadIdx.x + load * blockDim.x;
          const int x_row = vector_index / 4;
          const int x_vector = vector_index % 4;
          copy_async_16(
              reinterpret_cast<uint4*>(
                  shared.operand.x[next_stage] +
                  x_row * kStageStride) +
                  x_vector,
              reinterpret_cast<const uint4*>(
                  x + (block_row + x_row) * hidden_dim +
                  next_k) +
                  x_vector);
        }
#pragma unroll
        for (int load = 0; load < 2; ++load) {
          const int vector_index = threadIdx.x + load * blockDim.x;
          const int weight_column = vector_index / 4;
          const int weight_vector = vector_index % 4;
          copy_async_16(
              reinterpret_cast<uint4*>(
                  shared.operand.weight[next_stage] +
                  weight_column * kStageStride) +
                  weight_vector,
              reinterpret_cast<const uint4*>(
                  weight +
                  (block_column + weight_column) * hidden_dim +
                  next_k) +
                  weight_vector);
        }
        asm volatile("cp.async.commit_group;\n" ::);
      }
#pragma unroll
      for (int k_offset = 0; k_offset < kStageK; k_offset += kTile) {
        wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, half,
                       wmma::row_major>
            x_fragment[kWarpTilesM];
        wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, half,
                       wmma::col_major>
            weight_fragment[kWarpTilesN];

#pragma unroll
        for (int tile_m = 0; tile_m < kWarpTilesM; ++tile_m) {
          const int local_row =
              warp_m * kWarpTilesM * kTile + tile_m * kTile;
          wmma::load_matrix_sync(
              x_fragment[tile_m],
              shared.operand.x[current_stage] +
                  local_row * kStageStride + k_offset,
              kStageStride);
        }
#pragma unroll
        for (int tile_n = 0; tile_n < kWarpTilesN; ++tile_n) {
          const int local_column =
              warp_n * kWarpTilesN * kTile + tile_n * kTile;
          wmma::load_matrix_sync(
              weight_fragment[tile_n],
              shared.operand.weight[current_stage] +
                  local_column * kStageStride + k_offset,
              kStageStride);
        }
#pragma unroll
        for (int tile_m = 0; tile_m < kWarpTilesM; ++tile_m) {
#pragma unroll
          for (int tile_n = 0; tile_n < kWarpTilesN; ++tile_n) {
            wmma::mma_sync(
                accumulator[tile_m][tile_n],
                x_fragment[tile_m],
                weight_fragment[tile_n],
                accumulator[tile_m][tile_n]);
          }
        }
      }
      if (next_k < hidden_dim) {
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        current_stage = next_stage;
      }
    }
  } else {
    // Boundary CTAs use zero-padded shared tiles to keep WMMA loads in bounds.
    for (int k_block = 0; k_block < hidden_dim; k_block += kStageK) {
      const int staged_k = min(kStageK, hidden_dim - k_block);

      for (int element = threadIdx.x;
           element < kBlockM * kStageK;
           element += blockDim.x) {
        const int local_row = element / kStageK;
        const int local_k = element % kStageK;
        const int global_row = block_row + local_row;
        shared.operand.x[0][local_row * kStageStride + local_k] =
            global_row < token_count && local_k < staged_k
            ? x[global_row * hidden_dim + k_block + local_k]
            : __float2half(0.0f);
      }
      for (int element = threadIdx.x;
           element < kBlockN * kStageK;
           element += blockDim.x) {
        const int local_column = element / kStageK;
        const int local_k = element % kStageK;
        const int global_column = block_column + local_column;
        shared.operand.weight[0][
            local_column * kStageStride + local_k] =
            global_column < out_dim && local_k < staged_k
            ? weight[
                  global_column * hidden_dim + k_block + local_k]
            : __float2half(0.0f);
      }
      __syncthreads();

#pragma unroll
      for (int k_offset = 0; k_offset < kStageK; k_offset += kTile) {
        wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, half,
                       wmma::row_major>
            x_fragment[kWarpTilesM];
        wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, half,
                       wmma::col_major>
            weight_fragment[kWarpTilesN];

#pragma unroll
        for (int tile_m = 0; tile_m < kWarpTilesM; ++tile_m) {
          const int local_row = warp_m * kWarpTilesM * kTile +
                                tile_m * kTile;
          wmma::load_matrix_sync(
              x_fragment[tile_m],
              shared.operand.x[0] +
                  local_row * kStageStride + k_offset,
              kStageStride);
        }
#pragma unroll
        for (int tile_n = 0; tile_n < kWarpTilesN; ++tile_n) {
          const int local_column =
              warp_n * kWarpTilesN * kTile + tile_n * kTile;
          wmma::load_matrix_sync(
              weight_fragment[tile_n],
              shared.operand.weight[0] +
                  local_column * kStageStride + k_offset,
              kStageStride);
        }
#pragma unroll
        for (int tile_m = 0; tile_m < kWarpTilesM; ++tile_m) {
#pragma unroll
          for (int tile_n = 0; tile_n < kWarpTilesN; ++tile_n) {
            wmma::mma_sync(
                accumulator[tile_m][tile_n],
                x_fragment[tile_m],
                weight_fragment[tile_n],
                accumulator[tile_m][tile_n]);
          }
        }
      }
      __syncthreads();
    }
  }

  // Operand double buffers and the output tile intentionally alias.
  __syncthreads();
#pragma unroll
  for (int tile_m = 0; tile_m < kWarpTilesM; ++tile_m) {
#pragma unroll
    for (int tile_n = 0; tile_n < kWarpTilesN; ++tile_n) {
      const int local_row =
          warp_m * kWarpTilesM * kTile + tile_m * kTile;
      const int local_column =
          warp_n * kWarpTilesN * kTile + tile_n * kTile;
      wmma::store_matrix_sync(
          shared.output + warp_index * kTile * kTile,
          accumulator[tile_m][tile_n],
          kTile,
          wmma::mem_row_major);
      __syncwarp();

      for (int element = lane; element < kTile * kTile;
           element += kWarpSize) {
        const int tile_row = element / kTile;
        const int tile_column = element % kTile;
        const int global_row = block_row + local_row + tile_row;
        const int global_column =
            block_column + local_column + tile_column;
        if (global_row < token_count && global_column < out_dim) {
          projection[global_row * out_dim + global_column] =
              __float2half(
                  shared.output[
                      warp_index * kTile * kTile + element]);
        }
      }
    }
  }
}

__global__ __launch_bounds__(kRmsThreads) void rms_square_sum_kernel(
    const half* __restrict__ x,
    float* __restrict__ square_sum,
    int hidden_dim) {
  const int token_index = blockIdx.x;
  const half* token = x + token_index * hidden_dim;
  float local_sum = 0.0f;

  if ((hidden_dim & 1) == 0) {
    const half2* token_pairs = reinterpret_cast<const half2*>(token);
    const int pair_count = hidden_dim / 2;
    for (int index = threadIdx.x; index < pair_count; index += blockDim.x) {
      const float2 value = __half22float2(token_pairs[index]);
      local_sum = fmaf(value.x, value.x, local_sum);
      local_sum = fmaf(value.y, value.y, local_sum);
    }
  } else {
    for (int index = threadIdx.x; index < hidden_dim; index += blockDim.x) {
      const float value = __half2float(token[index]);
      local_sum = fmaf(value, value, local_sum);
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  }

  __shared__ float warp_sums[kRmsThreads / kWarpSize];
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  if (lane == 0) {
    warp_sums[warp] = local_sum;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum =
        lane < kRmsThreads / kWarpSize ? warp_sums[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
    }
    if (lane == 0) {
      square_sum[token_index] = block_sum;
    }
  }
}

__global__ __launch_bounds__(kRmsThreads) void inverse_rms_kernel(
    const half* __restrict__ x,
    float* __restrict__ inverse_rms,
    int hidden_dim,
    float eps) {
  const int token_index = blockIdx.x;
  const half2* token = reinterpret_cast<const half2*>(
      x + token_index * hidden_dim);
  float local_sum = 0.0f;
  const int pair_count = hidden_dim / 2;
  for (int index = threadIdx.x; index < pair_count; index += blockDim.x) {
    const float2 value = __half22float2(token[index]);
    local_sum = fmaf(value.x, value.x, local_sum);
    local_sum = fmaf(value.y, value.y, local_sum);
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  }
  __shared__ float warp_sums[kRmsThreads / kWarpSize];
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  if (lane == 0) {
    warp_sums[warp] = local_sum;
  }
  __syncthreads();
  if (warp == 0) {
    float block_sum =
        lane < kRmsThreads / kWarpSize ? warp_sums[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
    }
    if (lane == 0) {
      inverse_rms[token_index] =
          rsqrtf(block_sum / static_cast<float>(hidden_dim) + eps);
    }
  }
}

__global__ __launch_bounds__(kEpilogueThreads) void rms_scale_kernel(
    const half* __restrict__ projection,
    const float* __restrict__ square_sum,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int total_pairs,
    int out_dim,
    int hidden_dim,
    float eps) {
  const int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair_index >= total_pairs) {
    return;
  }

  const int pairs_per_token = out_dim / 2;
  const int token_index = pair_index / pairs_per_token;
  const int output_pair_index =
      pair_index - token_index * pairs_per_token;
  const float inverse_rms = rsqrtf(
      square_sum[token_index] / static_cast<float>(hidden_dim) + eps);
  const half2 projected = reinterpret_cast<const half2*>(
      projection)[pair_index];
  float2 value = __half22float2(projected);
  value.x *= inverse_rms;
  value.y *= inverse_rms;
  if (bias != nullptr) {
    const float2 bias_value = __half22float2(
        reinterpret_cast<const half2*>(bias)[output_pair_index]);
    value.x += bias_value.x;
    value.y += bias_value.y;
  }
  reinterpret_cast<half2*>(output)[pair_index] =
      __floats2half2_rn(value.x, value.y);
}

}  // namespace nonlinear::prefill

void launch_prefill_tensor_core_projection_fp16(
    const half* x,
    const half* weight,
    half* projection,
    int token_count,
    int out_dim,
    int hidden_dim,
    cudaStream_t stream) {
  using Element = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  // Row-major [OutDim, HiddenDim] storage is the column-major representation
  // of the logical [HiddenDim, OutDim] matrix B.
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using Accumulator = float;
  using Epilogue = cutlass::epilogue::thread::LinearCombination<
      Element,
      128 / cutlass::sizeof_bits<Element>::value,
      Accumulator,
      Accumulator>;
  using CutlassGemm = cutlass::gemm::device::Gemm<
      Element,
      LayoutA,
      Element,
      LayoutB,
      Element,
      LayoutC,
      Accumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      3>;

  typename CutlassGemm::Arguments arguments(
      {token_count, out_dim, hidden_dim},
      {reinterpret_cast<const Element*>(x), hidden_dim},
      {reinterpret_cast<const Element*>(weight), hidden_dim},
      {reinterpret_cast<const Element*>(projection), out_dim},
      {reinterpret_cast<Element*>(projection), out_dim},
      {1.0f, 0.0f});
  CutlassGemm gemm;
  const cutlass::Status status = gemm(arguments, nullptr, stream);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS prefill GEMM failed: ",
      cutlassGetStatusString(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_prefill_inverse_rms_fp16(
    const half* x,
    float* inverse_rms,
    int token_count,
    int hidden_dim,
    float eps,
    cudaStream_t stream) {
  nonlinear::prefill::inverse_rms_kernel<<<
      token_count, nonlinear::prefill::kRmsThreads, 0, stream>>>(
      x, inverse_rms, hidden_dim, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_prefill_tensor_core_scaled_projection_fp16(
    const half* x,
    const half* weight,
    const float* inverse_rms,
    half* output,
    int token_count,
    int out_dim,
    int hidden_dim,
    cudaStream_t stream) {
  using Element = cutlass::half_t;
  using Accumulator = float;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape, WarpShape, Element, 8, 1>;
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  using InvRms = cutlass::epilogue::threadblock::VisitorColBroadcast<
      OutputTileThreadMap,
      Accumulator,
      cute::Stride<cute::_1, cute::_0, int64_t>>;
  using Multiply = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies,
      Element,
      Accumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using Scale = cutlass::epilogue::threadblock::Sm80EVT<
      Multiply, Accum, InvRms>;
  using Store = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap,
      Element,
      cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, int64_t>>;
  using Epilogue = cutlass::epilogue::threadblock::Sm80EVT<Store, Scale>;
  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
      Element, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,
      Element, cutlass::layout::RowMajor, 8,
      Accumulator, Accumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      ThreadblockShape, WarpShape, InstructionShape,
      Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      3,
      cutlass::arch::OpMultiplyAdd,
      1>::GemmKernel;
  using CutlassGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  typename Epilogue::Arguments epilogue_arguments{
      {{},
       {inverse_rms,
        0.0f,
        {cute::_1{}, cute::_0{}, static_cast<int64_t>(token_count)}},
       {}},
      {reinterpret_cast<Element*>(output),
       {static_cast<int64_t>(out_dim),
        cute::_1{},
        static_cast<int64_t>(token_count) * out_dim}}};
  typename CutlassGemm::Arguments arguments(
      cutlass::gemm::GemmUniversalMode::kGemm,
      {token_count, out_dim, hidden_dim},
      1,
      epilogue_arguments,
      reinterpret_cast<const Element*>(x),
      reinterpret_cast<const Element*>(weight),
      nullptr,
      nullptr,
      static_cast<int64_t>(token_count) * hidden_dim,
      static_cast<int64_t>(out_dim) * hidden_dim,
      0,
      0,
      hidden_dim,
      hidden_dim,
      0,
      0,
      nullptr,
      nullptr,
      nullptr);
  CutlassGemm gemm;
  const cutlass::Status status = gemm(arguments, nullptr, stream);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS prefill scaled GEMM failed: ",
      cutlassGetStatusString(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_prefill_rms_square_sum_fp16(
    const half* x,
    float* square_sum,
    int token_count,
    int hidden_dim,
    cudaStream_t stream) {
  const dim3 block(nonlinear::prefill::kRmsThreads);
  const dim3 grid(token_count);
  nonlinear::prefill::rms_square_sum_kernel<<<grid, block, 0, stream>>>(
      x, square_sum, hidden_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_prefill_rms_scale_fp16(
    const half* projection,
    const float* square_sum,
    const half* bias,
    half* output,
    int token_count,
    int out_dim,
    int hidden_dim,
    float eps,
    cudaStream_t stream) {
  const int total_pairs = token_count * out_dim / 2;
  const dim3 block(nonlinear::prefill::kEpilogueThreads);
  const dim3 grid(
      (total_pairs + nonlinear::prefill::kEpilogueThreads - 1) /
      nonlinear::prefill::kEpilogueThreads);
  nonlinear::prefill::rms_scale_kernel<<<grid, block, 0, stream>>>(
      projection,
      square_sum,
      bias,
      output,
      total_pairs,
      out_dim,
      hidden_dim,
      eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
