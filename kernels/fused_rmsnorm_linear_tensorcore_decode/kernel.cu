#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <mma.h>
#include <cstdint>

namespace {

constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 16;
// The warp-specialized prototype is retained for further profiling, but it is
// not selected by default: on A100 it regresses Q and gate latency and its
// partial-CTA named-barrier protocol is not yet stable for every batch size.
constexpr bool kEnableInt8WarpSpecializedPipeline = false;

template <typename Element>
struct Convert;

template <>
struct Convert<half> {
  __device__ static float to_float(half value) { return __half2float(value); }
  __device__ static half from_float(float value) { return __float2half(value); }
};

template <>
struct Convert<__nv_bfloat16> {
  __device__ static float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
  }
  __device__ static __nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
  }
};

template <int NTiles, int ChunkK>
__device__ __forceinline__ void stage_int8_weight_tile(
    const int8_t* __restrict__ quantized_weight,
    const half* __restrict__ weight_scale, half* __restrict__ tile_b,
    int cta, int n_tile, int warp, int lane, int hidden_dim, int k_begin) {
  const int scale_column = (lane & 3) * 4;
  const half2 scale_01 = *reinterpret_cast<const half2*>(
      weight_scale + (cta * NTiles + n_tile) * kTileN + scale_column);
  const half2 scale_23 = *reinterpret_cast<const half2*>(
      weight_scale + (cta * NTiles + n_tile) * kTileN + scale_column + 2);
  const int8_t* global_tile =
      quantized_weight +
      ((cta * NTiles + n_tile) * hidden_dim + k_begin) * kTileN;
  half* shared_tile = tile_b + warp * ChunkK * kTileN;
  constexpr int kPackedValues = ChunkK * kTileN / 4;
  for (int packed_index = lane; packed_index < kPackedValues;
       packed_index += 32) {
    const int packed =
        reinterpret_cast<const int*>(global_tile)[packed_index];
    const half2 quantized_01 = __halves2half2(
        __int2half_rn(int(static_cast<int8_t>(packed & 0xff))),
        __int2half_rn(int(static_cast<int8_t>((packed >> 8) & 0xff))));
    const half2 quantized_23 = __halves2half2(
        __int2half_rn(int(static_cast<int8_t>((packed >> 16) & 0xff))),
        __int2half_rn(int(static_cast<int8_t>((packed >> 24) & 0xff))));
    const half2 value_01 = __hmul2(quantized_01, scale_01);
    const half2 value_23 = __hmul2(quantized_23, scale_23);
    const int scalar_index = packed_index * 4;
    *reinterpret_cast<half2*>(shared_tile + scalar_index) = value_01;
    const bool contains_sentinel = n_tile == NTiles - 1 &&
                                   scale_column == kTileN - 4;
    if (contains_sentinel) {
      shared_tile[scalar_index + 2] = __low2half(value_23);
      shared_tile[scalar_index + 3] = __float2half(1.0f);
    } else {
      *reinterpret_cast<half2*>(shared_tile + scalar_index + 2) = value_23;
    }
  }
}

template <int BarrierId, int ThreadCount>
__device__ __forceinline__ void named_barrier() {
  asm volatile("bar.sync %0, %1;"
               :
               : "n"(BarrierId), "n"(ThreadCount)
               : "memory");
}

template <int BarrierId, int ThreadCount>
__device__ __forceinline__ void named_barrier_arrive() {
  asm volatile("bar.arrive %0, %1;"
               :
               : "n"(BarrierId), "n"(ThreadCount)
               : "memory");
}

template <int BarrierBase>
__device__ __forceinline__ void pair_barrier(int pair) {
  switch (pair) {
    case 0:
      named_barrier<BarrierBase, 64>();
      break;
    case 1:
      named_barrier<BarrierBase + 1, 64>();
      break;
    case 2:
      named_barrier<BarrierBase + 2, 64>();
      break;
    case 3:
      named_barrier<BarrierBase + 3, 64>();
      break;
  }
}


template <int BarrierBase>
__device__ __forceinline__ void pair_barrier_arrive(int pair) {
  switch (pair) {
    case 0:
      named_barrier_arrive<BarrierBase, 64>();
      break;
    case 1:
      named_barrier_arrive<BarrierBase + 1, 64>();
      break;
    case 2:
      named_barrier_arrive<BarrierBase + 2, 64>();
      break;
    case 3:
      named_barrier_arrive<BarrierBase + 3, 64>();
      break;
  }
}

template <typename Element, int Batch>
__global__ __launch_bounds__(256) void
tensorcore_rms_projection_int8_pipeline_kernel(
    const Element* __restrict__ x, const Element* __restrict__ residual,
    const int8_t* __restrict__ quantized_weight,
    const half* __restrict__ weight_scale,
    const Element* __restrict__ bias, Element* __restrict__ output,
    Element* __restrict__ residual_sum, int out_dim, int hidden_dim,
    float eps) {
  using namespace nvcuda;
  constexpr int kNTiles = 4;
  constexpr int kProducerWarps = 4;
  constexpr int kChunkK = 64;
  constexpr int kRows = Batch * 2;
  constexpr int kMTiles = (kRows + kTileM - 1) / kTileM;
  constexpr int kProjectionColumns = kNTiles * kTileN - 1;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const bool producer = warp < kProducerWarps;
  const int consumer_warp = warp - kProducerWarps;
  const int output_begin = blockIdx.x * kProjectionColumns;
  const int stages = hidden_dim / kChunkK;

  // Each producer/consumer warp pair owns its A buffers. This duplicates the
  // small decode activation tile, but makes buffer lifetime independent for
  // every output tile and avoids a block-wide barrier in the K loop.
  __shared__ half tile_a[2][kNTiles][kMTiles * kTileM * kChunkK];
  __shared__ half tile_b[2][kNTiles * kChunkK * kTileN];
  __shared__ float sentinel_c[kMTiles * kTileM * kTileN];
  __shared__ float tile_c[kNTiles][kTileM * kTileN];
  __shared__ float inverse_rms[Batch];

  wmma::fragment<wmma::accumulator, kTileM, kTileN, kTileK, float>
      accum[kMTiles];
#pragma unroll
  for (int m_tile = 0; m_tile < kMTiles; ++m_tile)
    wmma::fill_fragment(accum[m_tile], 0.0f);

  if (producer) {
    for (int stage = 0; stage < stages; ++stage) {
      const int buffer = stage & 1;
      if (stage >= 2)
        pair_barrier<5>(warp);
      const int k_begin = stage * kChunkK;
      for (int index = lane;
           index < kMTiles * kTileM * kChunkK;
           index += 32) {
        const int row = index / kChunkK;
        const int local_k = index % kChunkK;
        half staged = __float2half(0.0f);
        if (row < kRows) {
          const int batch = row < Batch ? row : row - Batch;
          const int k = k_begin + local_k;
          float value = Convert<Element>::to_float(
              x[batch * hidden_dim + k]);
          if (residual != nullptr) {
            value += Convert<Element>::to_float(
                residual[batch * hidden_dim + k]);
            const Element rounded = Convert<Element>::from_float(value);
            value = Convert<Element>::to_float(rounded);
            if (row < Batch && blockIdx.x == 0)
              residual_sum[batch * hidden_dim + k] = rounded;
          }
          const half value_fp16 = __float2half(value);
          staged = row < Batch ? value_fp16
                               : __hmul(value_fp16, value_fp16);
        }
        tile_a[buffer][warp][index] = staged;
      }
      stage_int8_weight_tile<kNTiles, kChunkK>(
          quantized_weight, weight_scale, tile_b[buffer], blockIdx.x, warp,
          warp, lane, hidden_dim, k_begin);
      pair_barrier<1>(warp);
    }
  } else {
    for (int stage = 0; stage < stages; ++stage) {
      const int buffer = stage & 1;
      pair_barrier<1>(consumer_warp);
      for (int k_offset = 0; k_offset < kChunkK; k_offset += kTileK) {
        wmma::fragment<wmma::matrix_b, kTileM, kTileN, kTileK, half,
                       wmma::row_major>
            b;
        wmma::load_matrix_sync(
            b,
            tile_b[buffer] +
                consumer_warp * kChunkK * kTileN + k_offset * kTileN,
            kTileN);
#pragma unroll
        for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
          wmma::fragment<wmma::matrix_a, kTileM, kTileN, kTileK, half,
                         wmma::row_major>
              a;
          wmma::load_matrix_sync(
              a,
              tile_a[buffer][consumer_warp] +
                  m_tile * kTileM * kChunkK + k_offset,
              kChunkK);
          wmma::mma_sync(accum[m_tile], a, b, accum[m_tile]);
        }
      }
      if (stage + 2 < stages)
        pair_barrier_arrive<5>(consumer_warp);
    }
  }

  __syncthreads();
  if (!producer && consumer_warp == kNTiles - 1) {
#pragma unroll
    for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
      wmma::store_matrix_sync(
          sentinel_c + m_tile * kTileM * kTileN, accum[m_tile], kTileN,
          wmma::mem_row_major);
    }
  }
  __syncthreads();
  if (threadIdx.x < Batch) {
    const int square_row = Batch + threadIdx.x;
    const int square_m_tile = square_row / kTileM;
    const int square_local_row = square_row % kTileM;
    const float square_sum =
        sentinel_c[square_m_tile * kTileM * kTileN +
                   square_local_row * kTileN + kTileN - 1];
    inverse_rms[threadIdx.x] =
        rsqrtf(square_sum / float(hidden_dim) + eps);
  }
  __syncthreads();

  if (!producer) {
#pragma unroll
    for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
      wmma::store_matrix_sync(
          tile_c[consumer_warp], accum[m_tile], kTileN,
          wmma::mem_row_major);
      __syncwarp();
      const int row_begin = m_tile * kTileM;
      const int rows = min(kTileM, Batch - row_begin);
      for (int index = lane; index < rows * kTileN; index += 32) {
        const int local_row = index / kTileN;
        const int column = index % kTileN;
        const int batch = row_begin + local_row;
        const int physical_column = consumer_warp * kTileN + column;
        const int output_column = output_begin + physical_column;
        if (physical_column < kProjectionColumns &&
            output_column < out_dim) {
          float value = tile_c[consumer_warp][local_row * kTileN + column] *
                        inverse_rms[batch];
          if (bias != nullptr)
            value += Convert<Element>::to_float(bias[output_column]);
          output[batch * out_dim + output_column] =
              Convert<Element>::from_float(value);
        }
      }
      __syncwarp();
    }
  }
}

template <typename Element, int Batch, int NTiles, bool Int8Weight>
__global__ void tensorcore_rms_projection_kernel(
    const Element* __restrict__ x, const Element* __restrict__ residual,
    const half* __restrict__ weight,
    const int8_t* __restrict__ quantized_weight,
    const half* __restrict__ weight_scale,
    const Element* __restrict__ bias,
    Element* __restrict__ output, Element* __restrict__ residual_sum,
    int out_dim, int hidden_dim, float eps) {
  using namespace nvcuda;
  constexpr int kRows = Batch * 2;
  constexpr int kMTiles = (kRows + kTileM - 1) / kTileM;
  constexpr int kActiveWarps = NTiles < 8 ? NTiles : 8;
  constexpr int kBlockThreads = kActiveWarps * 32;
  constexpr int kRounds = NTiles / kActiveWarps;
  constexpr int kChunkK = 128;
  constexpr int kPhysicalColumns = NTiles * kTileN;
  constexpr int kProjectionColumns = kPhysicalColumns - 1;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int output_begin = blockIdx.x * kProjectionColumns;

  // A is reused by every output warp. FP16 B loads directly from global;
  // INT8 B is vector-dequantized once into a warp-private FP16 shared tile.
  __shared__ half tile_a[kMTiles * kTileM * kChunkK];
  __shared__ half tile_b[Int8Weight
                             ? kActiveWarps * kChunkK * kTileN
                             : 1];
  __shared__ float sentinel_c[kMTiles * kTileM * kTileN];
  __shared__ float tile_c[kActiveWarps][kTileM * kTileN];
  __shared__ float inverse_rms[Batch];

  wmma::fragment<wmma::accumulator, kTileM, kTileN, kTileK, float>
      accum[kRounds][kMTiles];
#pragma unroll
  for (int round = 0; round < kRounds; ++round) {
#pragma unroll
    for (int m_tile = 0; m_tile < kMTiles; ++m_tile)
      wmma::fill_fragment(accum[round][m_tile], 0.0f);
  }

  for (int k_begin = 0; k_begin < hidden_dim; k_begin += kChunkK) {
    for (int index = threadIdx.x;
         index < kMTiles * kTileM * kChunkK; index += kBlockThreads) {
      const int row = index / kChunkK;
      const int local_k = index % kChunkK;
      half staged = __float2half(0.0f);
      if (row < kRows) {
        const int batch = row < Batch ? row : row - Batch;
        const int k = k_begin + local_k;
        float value = Convert<Element>::to_float(x[batch * hidden_dim + k]);
        if (residual != nullptr) {
          value += Convert<Element>::to_float(
              residual[batch * hidden_dim + k]);
          const Element rounded = Convert<Element>::from_float(value);
          value = Convert<Element>::to_float(rounded);
          if (row < Batch && blockIdx.x == 0)
            residual_sum[batch * hidden_dim + k] = rounded;
        }
        const half value_fp16 = __float2half(value);
        staged = row < Batch ? value_fp16
                             : __hmul(value_fp16, value_fp16);
      }
      tile_a[index] = staged;
    }

    if constexpr (Int8Weight) {
      const int n_tile = warp;
      stage_int8_weight_tile<NTiles, kChunkK>(
          quantized_weight, weight_scale, tile_b, blockIdx.x, n_tile, warp,
          lane, hidden_dim, k_begin);
    }

    __syncthreads();

#pragma unroll
    for (int round = 0; round < kRounds; ++round) {
      const int n_tile = round * kActiveWarps + warp;
      if constexpr (Int8Weight) {
        if (round != 0) {
          stage_int8_weight_tile<NTiles, kChunkK>(
              quantized_weight, weight_scale, tile_b, blockIdx.x, n_tile,
              warp, lane, hidden_dim, k_begin);
          __syncwarp();
        }
      }
      for (int k_offset = 0; k_offset < kChunkK; k_offset += kTileK) {
        wmma::fragment<wmma::matrix_b, kTileM, kTileN, kTileK, half,
                       wmma::row_major>
            b;
        if constexpr (Int8Weight) {
          wmma::load_matrix_sync(
              b,
              tile_b + warp * kChunkK * kTileN + k_offset * kTileN,
              kTileN);
        } else {
          wmma::load_matrix_sync(
              b,
              weight + (((blockIdx.x * NTiles + n_tile) * hidden_dim +
                         k_begin + k_offset) *
                        kTileN),
              kTileN);
        }
#pragma unroll
        for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
          wmma::fragment<wmma::matrix_a, kTileM, kTileN, kTileK, half,
                         wmma::row_major>
              a;
          wmma::load_matrix_sync(
              a, tile_a + m_tile * kTileM * kChunkK + k_offset, kChunkK);
          wmma::mma_sync(
              accum[round][m_tile], a, b, accum[round][m_tile]);
        }
      }
    }
    // No warp may overwrite A until every output tile has consumed it.
    __syncthreads();
  }

  // The final column of the final tile is the CTA-local all-ones sentinel.
  if (warp == kActiveWarps - 1) {
#pragma unroll
    for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
      wmma::store_matrix_sync(
          sentinel_c + m_tile * kTileM * kTileN,
          accum[kRounds - 1][m_tile], kTileN, wmma::mem_row_major);
    }
  }
  __syncthreads();
  if (threadIdx.x < Batch) {
    const int square_row = Batch + threadIdx.x;
    const int square_m_tile = square_row / kTileM;
    const int square_local_row = square_row % kTileM;
    const float square_sum =
        sentinel_c[square_m_tile * kTileM * kTileN +
                   square_local_row * kTileN + kTileN - 1];
    inverse_rms[threadIdx.x] =
        rsqrtf(square_sum / float(hidden_dim) + eps);
  }
  __syncthreads();

#pragma unroll
  for (int round = 0; round < kRounds; ++round) {
#pragma unroll
    for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
      wmma::store_matrix_sync(
          tile_c[warp], accum[round][m_tile], kTileN,
          wmma::mem_row_major);
      __syncwarp();
      const int row_begin = m_tile * kTileM;
      const int rows = min(kTileM, Batch - row_begin);
      for (int index = lane; index < rows * kTileN; index += 32) {
        const int local_row = index / kTileN;
        const int column = index % kTileN;
        const int batch = row_begin + local_row;
        const int physical_column =
            (round * kActiveWarps + warp) * kTileN + column;
        const int output_column = output_begin + physical_column;
        if (physical_column < kProjectionColumns &&
            output_column < out_dim) {
          float value = tile_c[warp][local_row * kTileN + column] *
                        inverse_rms[batch];
          if (bias != nullptr)
            value += Convert<Element>::to_float(bias[output_column]);
          output[batch * out_dim + output_column] =
              Convert<Element>::from_float(value);
        }
      }
      __syncwarp();
    }
  }
}

}  // namespace

template <bool Int8Weight>
void launch_projection(
    const void* x, const void* residual, const half* weight,
    const int8_t* quantized_weight, const half* weight_scale,
    const void* bias, void* output, void* residual_sum, int out_dim, int batch,
    int hidden_dim, int n_tiles, float eps, bool is_bfloat16,
    cudaStream_t stream) {
  const int projection_columns = n_tiles * kTileN - 1;
  const dim3 grid((out_dim + projection_columns - 1) / projection_columns);
  const int threads = min(n_tiles, 8) * 32;
  if constexpr (Int8Weight) {
    // Batch 16 needs two M tiles. Four pair-private double-buffered A tiles
    // would exceed the default static shared-memory limit, so retain the
    // lower-footprint non-pipelined kernel for that case.
    if constexpr (kEnableInt8WarpSpecializedPipeline) {
      if (n_tiles == 4 && batch != 16) {
#define LAUNCH_PIPELINE(Element, X, R, B, O, RS)                         \
      switch (batch) {                                                   \
        case 1:                                                          \
          tensorcore_rms_projection_int8_pipeline_kernel<Element, 1>     \
              <<<grid, 256, 0, stream>>>(                               \
                  X, R, quantized_weight, weight_scale, B, O, RS,        \
                  out_dim, hidden_dim, eps);                             \
          break;                                                         \
        case 4:                                                          \
          tensorcore_rms_projection_int8_pipeline_kernel<Element, 4>     \
              <<<grid, 256, 0, stream>>>(                               \
                  X, R, quantized_weight, weight_scale, B, O, RS,        \
                  out_dim, hidden_dim, eps);                             \
          break;                                                         \
        case 8:                                                          \
          tensorcore_rms_projection_int8_pipeline_kernel<Element, 8>     \
              <<<grid, 256, 0, stream>>>(                               \
                  X, R, quantized_weight, weight_scale, B, O, RS,        \
                  out_dim, hidden_dim, eps);                             \
          break;                                                         \
        case 16:                                                         \
          tensorcore_rms_projection_int8_pipeline_kernel<Element, 16>    \
              <<<grid, 256, 0, stream>>>(                               \
                  X, R, quantized_weight, weight_scale, B, O, RS,        \
                  out_dim, hidden_dim, eps);                             \
          break;                                                         \
      }
      if (is_bfloat16) {
        LAUNCH_PIPELINE(
            __nv_bfloat16, reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(residual),
            reinterpret_cast<const __nv_bfloat16*>(bias),
            reinterpret_cast<__nv_bfloat16*>(output),
            reinterpret_cast<__nv_bfloat16*>(residual_sum));
      } else {
        LAUNCH_PIPELINE(
            half, reinterpret_cast<const half*>(x),
            reinterpret_cast<const half*>(residual),
            reinterpret_cast<const half*>(bias),
            reinterpret_cast<half*>(output),
            reinterpret_cast<half*>(residual_sum));
      }
#undef LAUNCH_PIPELINE
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
      }
    }
  }
#define LAUNCH_BATCH(Element, NT, X, R, B, O, RS)                         \
  switch (batch) {                                                       \
    case 1:                                                              \
      tensorcore_rms_projection_kernel<Element, 1, NT, Int8Weight>       \
          <<<grid, threads, 0, stream>>>(                                \
              X, R, weight, quantized_weight, weight_scale, B, O, RS,    \
              out_dim, hidden_dim, eps);                                 \
      break;                                                             \
    case 4:                                                              \
      tensorcore_rms_projection_kernel<Element, 4, NT, Int8Weight>       \
          <<<grid, threads, 0, stream>>>(                                \
              X, R, weight, quantized_weight, weight_scale, B, O, RS,    \
              out_dim, hidden_dim, eps);                                 \
      break;                                                             \
    case 8:                                                              \
      tensorcore_rms_projection_kernel<Element, 8, NT, Int8Weight>       \
          <<<grid, threads, 0, stream>>>(                                \
              X, R, weight, quantized_weight, weight_scale, B, O, RS,    \
              out_dim, hidden_dim, eps);                                 \
      break;                                                             \
    case 16:                                                             \
      tensorcore_rms_projection_kernel<Element, 16, NT, Int8Weight>      \
          <<<grid, threads, 0, stream>>>(                                \
              X, R, weight, quantized_weight, weight_scale, B, O, RS,    \
              out_dim, hidden_dim, eps);                                 \
      break;                                                             \
  }
#define LAUNCH_TILES(Element, X, R, B, O, RS)                            \
  switch (n_tiles) {                                                     \
    case 1:                                                              \
      LAUNCH_BATCH(Element, 1, X, R, B, O, RS);                          \
      break;                                                             \
    case 2:                                                              \
      LAUNCH_BATCH(Element, 2, X, R, B, O, RS);                          \
      break;                                                             \
    case 4:                                                              \
      LAUNCH_BATCH(Element, 4, X, R, B, O, RS);                          \
      break;                                                             \
    case 8:                                                              \
      LAUNCH_BATCH(Element, 8, X, R, B, O, RS);                          \
      break;                                                             \
    case 16:                                                             \
      LAUNCH_BATCH(Element, 16, X, R, B, O, RS);                         \
      break;                                                             \
    case 32:                                                             \
      LAUNCH_BATCH(Element, 32, X, R, B, O, RS);                         \
      break;                                                             \
  }
  if (is_bfloat16) {
    LAUNCH_TILES(__nv_bfloat16, reinterpret_cast<const __nv_bfloat16*>(x),
                 reinterpret_cast<const __nv_bfloat16*>(residual),
                 reinterpret_cast<const __nv_bfloat16*>(bias),
                 reinterpret_cast<__nv_bfloat16*>(output),
                 reinterpret_cast<__nv_bfloat16*>(residual_sum));
  } else {
    LAUNCH_TILES(half, reinterpret_cast<const half*>(x),
                 reinterpret_cast<const half*>(residual),
                 reinterpret_cast<const half*>(bias),
                 reinterpret_cast<half*>(output),
                 reinterpret_cast<half*>(residual_sum));
  }
#undef LAUNCH_TILES
#undef LAUNCH_BATCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_tensorcore_rms_projection(
    const void* x, const void* residual, const half* weight,
    const void* bias, void* output, void* residual_sum, int out_dim,
    int batch, int hidden_dim, int n_tiles, float eps, bool is_bfloat16,
    cudaStream_t stream) {
  launch_projection<false>(
      x, residual, weight, nullptr, nullptr, bias, output, residual_sum,
      out_dim, batch, hidden_dim, n_tiles, eps, is_bfloat16, stream);
}

void launch_tensorcore_rms_projection_int8_weight(
    const void* x, const void* residual, const int8_t* quantized_weight,
    const half* weight_scale, const void* bias, void* output,
    void* residual_sum, int out_dim, int batch, int hidden_dim, int n_tiles,
    float eps, bool is_bfloat16, cudaStream_t stream) {
  launch_projection<true>(
      x, residual, nullptr, quantized_weight, weight_scale, bias, output,
      residual_sum, out_dim, batch, hidden_dim, n_tiles, eps, is_bfloat16,
      stream);
}
