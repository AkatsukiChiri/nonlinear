#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mma.h>

namespace nonlinear {

constexpr int kLayerNormWarps = 8;
constexpr int kLayerNormThreads = kLayerNormWarps * 32;
constexpr int kRowsPerWarp = 1;
constexpr int kWmmaTile = 16;
constexpr int kWmmaOutputsPerCta = kLayerNormWarps * kWmmaTile;

union Half8Pack {
  uint4 vector;
  half2 pairs[4];
};

template <int Batch>
__device__ __forceinline__ void compute_layernorm_stats(
    const half* __restrict__ x,
    float* __restrict__ input_sum,
    float* __restrict__ input_square_sum,
    int hidden_dim,
    float (&warp_sum)[Batch][kLayerNormWarps],
    float (&warp_square_sum)[Batch][kLayerNormWarps]) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float local_sum[Batch] = {};
  float local_square_sum[Batch] = {};

  for (int k_index = threadIdx.x; k_index < hidden_dim;
       k_index += kLayerNormThreads) {
#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      const float value =
          __half2float(x[batch_index * hidden_dim + k_index]);
      local_sum[batch_index] += value;
      local_square_sum[batch_index] =
          fmaf(value, value, local_square_sum[batch_index]);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      local_sum[batch_index] += __shfl_down_sync(
          0xffffffff, local_sum[batch_index], offset);
      local_square_sum[batch_index] += __shfl_down_sync(
          0xffffffff, local_square_sum[batch_index], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      warp_sum[batch_index][warp] = local_sum[batch_index];
      warp_square_sum[batch_index][warp] =
          local_square_sum[batch_index];
    }
  }
  __syncthreads();
  if (threadIdx.x < Batch) {
    float total = 0.0f;
    float total_square = 0.0f;
#pragma unroll
    for (int warp_index = 0; warp_index < kLayerNormWarps; ++warp_index) {
      total += warp_sum[threadIdx.x][warp_index];
      total_square += warp_square_sum[threadIdx.x][warp_index];
    }
    input_sum[threadIdx.x] = total;
    input_square_sum[threadIdx.x] = total_square;
  }
}

template <int Batch>
__global__ __launch_bounds__(kLayerNormThreads)
void layernorm_projection_stats_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    float* __restrict__ projection,
    float* __restrict__ input_sum,
    float* __restrict__ input_square_sum,
    int out_dim,
    int hidden_dim) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  // LayerNorm statistics are independent of the projection. Computing them
  // only in the first CTA avoids both another launch and duplicated work.
  if (blockIdx.x == 0) {
    __shared__ float warp_sum[Batch][kLayerNormWarps];
    __shared__ float warp_square_sum[Batch][kLayerNormWarps];
    compute_layernorm_stats<Batch>(
        x,
        input_sum,
        input_square_sum,
        hidden_dim,
        warp_sum,
        warp_square_sum);
  }

  const int output_index = blockIdx.x * kLayerNormWarps + warp;
  if (output_index >= out_dim) {
    return;
  }

  float accumulator[Batch] = {};
  const half* weight_row = weight + output_index * hidden_dim;
  if ((hidden_dim & 1) == 0) {
    const int pair_count = hidden_dim / 2;
    for (int pair_index = lane; pair_index < pair_count; pair_index += 32) {
      const float2 weight_value = __half22float2(
          reinterpret_cast<const half2*>(weight_row)[pair_index]);
#pragma unroll
      for (int batch_index = 0; batch_index < Batch; ++batch_index) {
        const float2 x_value = __half22float2(
            reinterpret_cast<const half2*>(
                x + batch_index * hidden_dim)[pair_index]);
        accumulator[batch_index] = fmaf(
            x_value.x, weight_value.x, accumulator[batch_index]);
        accumulator[batch_index] = fmaf(
            x_value.y, weight_value.y, accumulator[batch_index]);
      }
    }
  } else {
    for (int k_index = lane; k_index < hidden_dim; k_index += 32) {
      const float weight_value = __half2float(weight_row[k_index]);
#pragma unroll
      for (int batch_index = 0; batch_index < Batch; ++batch_index) {
        const float x_value =
            __half2float(x[batch_index * hidden_dim + k_index]);
        accumulator[batch_index] = fmaf(
            x_value, weight_value, accumulator[batch_index]);
      }
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      accumulator[batch_index] += __shfl_down_sync(
          0xffffffff, accumulator[batch_index], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      projection[batch_index * out_dim + output_index] =
          accumulator[batch_index];
    }
  }
}

template <int Batch>
__global__ __launch_bounds__(kLayerNormThreads)
void layernorm_projection_stats_wmma_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    float* __restrict__ projection,
    float* __restrict__ input_sum,
    float* __restrict__ input_square_sum,
    int out_dim,
    int hidden_dim) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  if (blockIdx.x == 0) {
    __shared__ float warp_sum[Batch][kLayerNormWarps];
    __shared__ float warp_square_sum[Batch][kLayerNormWarps];
    compute_layernorm_stats<Batch>(
        x,
        input_sum,
        input_square_sum,
        hidden_dim,
        warp_sum,
        warp_square_sum);
  }

  const int output_base =
      blockIdx.x * kWmmaOutputsPerCta + warp * kWmmaTile;
  if (output_base >= out_dim) {
    return;
  }

  // Each WMMA M row holds one decode batch. Unused rows are zero-filled.
  // Per-warp shared tiles avoid a CTA-wide barrier for every K tile.
  __shared__ __align__(16)
      half a_tiles[kLayerNormWarps][kWmmaTile * kWmmaTile];
  __shared__ __align__(16)
      float c_tiles[kLayerNormWarps][kWmmaTile * kWmmaTile];

  using namespace nvcuda;
  wmma::fragment<
      wmma::matrix_a,
      kWmmaTile,
      kWmmaTile,
      kWmmaTile,
      half,
      wmma::row_major>
      a_fragment;
  wmma::fragment<
      wmma::matrix_b,
      kWmmaTile,
      kWmmaTile,
      kWmmaTile,
      half,
      wmma::col_major>
      b_fragment;
  wmma::fragment<
      wmma::accumulator,
      kWmmaTile,
      kWmmaTile,
      kWmmaTile,
      float>
      accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  for (int k_base = 0; k_base < hidden_dim; k_base += kWmmaTile) {
    for (int tile_index = lane;
         tile_index < kWmmaTile * kWmmaTile;
         tile_index += 32) {
      const int batch_index = tile_index / kWmmaTile;
      const int k_offset = tile_index - batch_index * kWmmaTile;
      a_tiles[warp][tile_index] =
          batch_index < Batch
          ? x[batch_index * hidden_dim + k_base + k_offset]
          : __float2half(0.0f);
    }
    __syncwarp();

    wmma::load_matrix_sync(
        a_fragment,
        a_tiles[warp],
        kWmmaTile);
    // A col-major B tile maps B[k, n] to weight[n, k].
    wmma::load_matrix_sync(
        b_fragment,
        weight + output_base * hidden_dim + k_base,
        hidden_dim);
    wmma::mma_sync(
        accumulator,
        a_fragment,
        b_fragment,
        accumulator);
  }

  wmma::store_matrix_sync(
      c_tiles[warp],
      accumulator,
      kWmmaTile,
      wmma::mem_row_major);
  __syncwarp();
  for (int output_index = lane;
       output_index < Batch * kWmmaTile;
       output_index += 32) {
    const int batch_index = output_index / kWmmaTile;
    const int n_offset = output_index - batch_index * kWmmaTile;
    projection[
        batch_index * out_dim + output_base + n_offset] =
        c_tiles[warp][output_index];
  }
}

template <int Batch>
__global__ __launch_bounds__(kLayerNormThreads)
void layernorm_stats_expand_kernel(
    const half* __restrict__ x,
    half* __restrict__ expanded_x,
    float* __restrict__ input_sum,
    float* __restrict__ input_square_sum,
    int hidden_dim) {
  __shared__ float warp_sum[Batch][kLayerNormWarps];
  __shared__ float warp_square_sum[Batch][kLayerNormWarps];
  compute_layernorm_stats<Batch>(
      x,
      input_sum,
      input_square_sum,
      hidden_dim,
      warp_sum,
      warp_square_sum);
  __syncthreads();

  const int expanded_elements = kWmmaTile * hidden_dim;
  for (int index = threadIdx.x; index < expanded_elements;
       index += kLayerNormThreads) {
    const int batch_index = index / hidden_dim;
    const int k_index = index - batch_index * hidden_dim;
    expanded_x[index] =
        batch_index < Batch
        ? x[batch_index * hidden_dim + k_index]
        : __float2half(0.0f);
  }
}

template <int Batch>
__global__ __launch_bounds__(kLayerNormThreads)
void layernorm_wmma_output_kernel(
    const half* __restrict__ expanded_x,
    const half* __restrict__ weight,
    const float* __restrict__ input_sum,
    const float* __restrict__ input_square_sum,
    const float* __restrict__ weight_sum,
    const float* __restrict__ fused_bias,
    half* __restrict__ output,
    int out_dim,
    int hidden_dim,
    float eps) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int output_base =
      blockIdx.x * kWmmaOutputsPerCta + warp * kWmmaTile;
  if (output_base >= out_dim) {
    return;
  }

  __shared__ __align__(16)
      float c_tiles[kLayerNormWarps][kWmmaTile * kWmmaTile];
  using namespace nvcuda;
  wmma::fragment<
      wmma::matrix_a,
      kWmmaTile,
      kWmmaTile,
      kWmmaTile,
      half,
      wmma::row_major>
      a_fragment;
  wmma::fragment<
      wmma::matrix_b,
      kWmmaTile,
      kWmmaTile,
      kWmmaTile,
      half,
      wmma::col_major>
      b_fragment;
  wmma::fragment<
      wmma::accumulator,
      kWmmaTile,
      kWmmaTile,
      kWmmaTile,
      float>
      accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  for (int k_base = 0; k_base < hidden_dim; k_base += kWmmaTile) {
    wmma::load_matrix_sync(
        a_fragment,
        expanded_x + k_base,
        hidden_dim);
    wmma::load_matrix_sync(
        b_fragment,
        weight + output_base * hidden_dim + k_base,
        hidden_dim);
    wmma::mma_sync(
        accumulator,
        a_fragment,
        b_fragment,
        accumulator);
  }
  wmma::store_matrix_sync(
      c_tiles[warp],
      accumulator,
      kWmmaTile,
      wmma::mem_row_major);
  __syncwarp();

  for (int output_index = lane;
       output_index < Batch * kWmmaTile;
       output_index += 32) {
    const int batch_index = output_index / kWmmaTile;
    const int n_offset = output_index - batch_index * kWmmaTile;
    const int n_index = output_base + n_offset;
    const float inverse_hidden_dim =
        1.0f / static_cast<float>(hidden_dim);
    const float mean = input_sum[batch_index] * inverse_hidden_dim;
    const float second_moment =
        input_square_sum[batch_index] * inverse_hidden_dim;
    const float variance = fmaxf(
        second_moment - mean * mean,
        0.0f);
    const float inverse_std = rsqrtf(variance + eps);
    const float centered_projection =
        c_tiles[warp][output_index] - mean * weight_sum[n_index];
    output[batch_index * out_dim + n_index] = __float2half(
        centered_projection * inverse_std + fused_bias[n_index]);
  }
}

template <int Batch>
__global__ __launch_bounds__(kLayerNormThreads)
void cooperative_layernorm_linear_kernel(
    const half* __restrict__ x,
    half* __restrict__ packed_x,
    const half* __restrict__ weight,
    const half* __restrict__ weight_sum,
    const half* __restrict__ fused_bias,
    half* __restrict__ output,
    float* __restrict__ statistics,
    int out_dim,
    int hidden_dim,
    float eps) {
  namespace cg = cooperative_groups;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float* input_sum = statistics;
  float* input_square_sum = statistics + Batch;

  if (blockIdx.x == 0) {
    __shared__ float warp_sum[Batch][kLayerNormWarps];
    __shared__ float warp_square_sum[Batch][kLayerNormWarps];
    compute_layernorm_stats<Batch>(
        x,
        input_sum,
        input_square_sum,
        hidden_dim,
        warp_sum,
        warp_square_sum);
    if constexpr (false && ((Batch & 1) == 0)) {
      __syncthreads();
      for (int index = threadIdx.x; index < hidden_dim * Batch;
           index += kLayerNormThreads) {
        const int k_index = index / Batch;
        const int batch_index = index - k_index * Batch;
        packed_x[index] =
            x[batch_index * hidden_dim + k_index];
      }
    }
  }
  cg::this_grid().sync();

  for (int output_base =
           (blockIdx.x * kLayerNormWarps + warp) * kRowsPerWarp;
       output_base < out_dim;
       output_base +=
           gridDim.x * kLayerNormWarps * kRowsPerWarp) {
    float accumulator[kRowsPerWarp][Batch] = {};
    if constexpr (false && ((Batch & 1) == 0)) {
      constexpr int kBatchPairs = Batch / 2;
      constexpr int kHalf2Chunk = 2;
      float2 packed_accumulator[kBatchPairs] = {};
      const half* weight_row = weight + output_base * hidden_dim;
      for (int k_base = lane; k_base < hidden_dim;
           k_base += 32 * kHalf2Chunk) {
        half2 half_accumulator[kBatchPairs];
#pragma unroll
        for (int batch_pair = 0; batch_pair < kBatchPairs;
             ++batch_pair) {
          half_accumulator[batch_pair] = __float2half2_rn(0.0f);
        }
#pragma unroll
        for (int chunk_index = 0; chunk_index < kHalf2Chunk;
             ++chunk_index) {
          const int k_index = k_base + chunk_index * 32;
          if (k_index < hidden_dim) {
            const half weight_value = weight_row[k_index];
            const half2 broadcast_weight =
                __halves2half2(weight_value, weight_value);
#pragma unroll
            for (int batch_pair = 0; batch_pair < kBatchPairs;
                 ++batch_pair) {
              const half2 x_value =
                  reinterpret_cast<const half2*>(
                      packed_x + k_index * Batch)[batch_pair];
              half_accumulator[batch_pair] = __hfma2(
                  x_value,
                  broadcast_weight,
                  half_accumulator[batch_pair]);
            }
          }
        }
#pragma unroll
        for (int batch_pair = 0; batch_pair < kBatchPairs;
             ++batch_pair) {
          const float2 chunk_sum =
              __half22float2(half_accumulator[batch_pair]);
          packed_accumulator[batch_pair].x += chunk_sum.x;
          packed_accumulator[batch_pair].y += chunk_sum.y;
        }
      }
#pragma unroll
      for (int batch_pair = 0; batch_pair < kBatchPairs;
           ++batch_pair) {
        accumulator[0][batch_pair * 2] =
            packed_accumulator[batch_pair].x;
        accumulator[0][batch_pair * 2 + 1] =
            packed_accumulator[batch_pair].y;
      }
    } else if (hidden_dim % 8 == 0) {
      float vector_accumulator[4][Batch] = {};
      const half* weight_row = weight + output_base * hidden_dim;
      const int vector_count = hidden_dim / 8;
      for (int vector_index = lane; vector_index < vector_count;
           vector_index += 32) {
        Half8Pack weight_pack;
        weight_pack.vector =
            reinterpret_cast<const uint4*>(weight_row)[vector_index];
        Half8Pack x_pack[Batch];
#pragma unroll
        for (int batch_index = 0; batch_index < Batch; ++batch_index) {
          x_pack[batch_index].vector =
              reinterpret_cast<const uint4*>(
                  x + batch_index * hidden_dim)[vector_index];
        }
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          const float2 weight_value =
              __half22float2(weight_pack.pairs[pair]);
#pragma unroll
          for (int batch_index = 0; batch_index < Batch;
               ++batch_index) {
            const float2 x_value =
                __half22float2(x_pack[batch_index].pairs[pair]);
            vector_accumulator[pair][batch_index] = fmaf(
                x_value.x,
                weight_value.x,
                vector_accumulator[pair][batch_index]);
            vector_accumulator[pair][batch_index] = fmaf(
                x_value.y,
                weight_value.y,
                vector_accumulator[pair][batch_index]);
          }
        }
      }
#pragma unroll
      for (int batch_index = 0; batch_index < Batch; ++batch_index) {
        accumulator[0][batch_index] =
            vector_accumulator[0][batch_index] +
            vector_accumulator[1][batch_index] +
            vector_accumulator[2][batch_index] +
            vector_accumulator[3][batch_index];
      }
    } else if ((hidden_dim & 1) == 0) {
      const int pair_count = hidden_dim / 2;
      for (int pair_index = lane; pair_index < pair_count;
           pair_index += 32) {
        float2 x_value[Batch];
#pragma unroll
        for (int batch_index = 0; batch_index < Batch; ++batch_index) {
          x_value[batch_index] = __half22float2(
              reinterpret_cast<const half2*>(
                  x + batch_index * hidden_dim)[pair_index]);
        }
#pragma unroll
        for (int row = 0; row < kRowsPerWarp; ++row) {
          const int output_index = output_base + row;
          if (output_index < out_dim) {
            const half* weight_row =
                weight + output_index * hidden_dim;
            const float2 weight_value = __half22float2(
                reinterpret_cast<const half2*>(
                    weight_row)[pair_index]);
#pragma unroll
            for (int batch_index = 0; batch_index < Batch;
                 ++batch_index) {
              accumulator[row][batch_index] = fmaf(
                  x_value[batch_index].x,
                  weight_value.x,
                  accumulator[row][batch_index]);
              accumulator[row][batch_index] = fmaf(
                  x_value[batch_index].y,
                  weight_value.y,
                  accumulator[row][batch_index]);
            }
          }
        }
      }
    } else {
      for (int k_index = lane; k_index < hidden_dim; k_index += 32) {
        float x_value[Batch];
#pragma unroll
        for (int batch_index = 0; batch_index < Batch; ++batch_index) {
          x_value[batch_index] =
              __half2float(x[batch_index * hidden_dim + k_index]);
        }
#pragma unroll
        for (int row = 0; row < kRowsPerWarp; ++row) {
          const int output_index = output_base + row;
          if (output_index < out_dim) {
            const float weight_value = __half2float(
                weight[output_index * hidden_dim + k_index]);
#pragma unroll
            for (int batch_index = 0; batch_index < Batch;
                 ++batch_index) {
              accumulator[row][batch_index] = fmaf(
                  x_value[batch_index],
                  weight_value,
                  accumulator[row][batch_index]);
            }
          }
        }
      }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
        for (int batch_index = 0; batch_index < Batch; ++batch_index) {
          accumulator[row][batch_index] += __shfl_down_sync(
              0xffffffff,
              accumulator[row][batch_index],
              offset);
        }
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int output_index = output_base + row;
        if (output_index >= out_dim) {
          continue;
        }
#pragma unroll
        for (int batch_index = 0; batch_index < Batch; ++batch_index) {
          const float inverse_hidden_dim =
              1.0f / static_cast<float>(hidden_dim);
          const float mean =
              input_sum[batch_index] * inverse_hidden_dim;
          const float second_moment =
              input_square_sum[batch_index] * inverse_hidden_dim;
          const float variance = fmaxf(
              second_moment - mean * mean,
              0.0f);
          const float inverse_std = rsqrtf(variance + eps);
          const float centered_projection =
              accumulator[row][batch_index] -
              mean * __half2float(weight_sum[output_index]);
          output[batch_index * out_dim + output_index] = __float2half(
              centered_projection * inverse_std +
              __half2float(fused_bias[output_index]));
        }
      }
    }
  }
}

__global__ void layernorm_epilogue_kernel(
    const float* __restrict__ projection,
    const float* __restrict__ input_sum,
    const float* __restrict__ input_square_sum,
    const half* __restrict__ weight_sum,
    const half* __restrict__ fused_bias,
    half* __restrict__ output,
    int total_elements,
    int out_dim,
    int hidden_dim,
    float eps) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total_elements) {
    return;
  }

  const int batch_index = index / out_dim;
  const int output_index = index - batch_index * out_dim;
  const float inverse_hidden_dim =
      1.0f / static_cast<float>(hidden_dim);
  const float mean = input_sum[batch_index] * inverse_hidden_dim;
  const float second_moment =
      input_square_sum[batch_index] * inverse_hidden_dim;
  const float variance = fmaxf(second_moment - mean * mean, 0.0f);
  const float inverse_std = rsqrtf(variance + eps);
  const float centered_projection =
      projection[index] - mean * __half2float(weight_sum[output_index]);
  output[index] = __float2half(
      centered_projection * inverse_std +
      __half2float(fused_bias[output_index]));
}

}  // namespace nonlinear

void launch_layernorm_projection_stats_fp16(
    const half* x,
    const half* weight,
    float* projection,
    float* input_sum,
    float* input_square_sum,
    int batch,
    int out_dim,
    int hidden_dim,
    cudaStream_t stream) {
  const dim3 block(nonlinear::kLayerNormThreads);
  // The short-M decode workload is currently faster on the chunked half2
  // SIMT path. Keep the WMMA implementation available for further tuning.
  const bool use_wmma = false;
  const dim3 grid(
      use_wmma
          ? (out_dim + nonlinear::kWmmaOutputsPerCta - 1) /
              nonlinear::kWmmaOutputsPerCta
          : (out_dim + nonlinear::kLayerNormWarps - 1) /
              nonlinear::kLayerNormWarps);
#define LAUNCH_BATCH(BATCH)                                                  \
  do {                                                                      \
    if (use_wmma) {                                                         \
      nonlinear::layernorm_projection_stats_wmma_kernel<BATCH>              \
          <<<grid, block, 0, stream>>>(                                     \
              x,                                                           \
              weight,                                                      \
              projection,                                                  \
              input_sum,                                                   \
              input_square_sum,                                            \
              out_dim,                                                     \
              hidden_dim);                                                 \
    } else {                                                                \
      nonlinear::layernorm_projection_stats_kernel<BATCH>                   \
          <<<grid, block, 0, stream>>>(                                     \
              x,                                                           \
              weight,                                                      \
              projection,                                                  \
              input_sum,                                                   \
              input_square_sum,                                            \
              out_dim,                                                     \
              hidden_dim);                                                 \
    }                                                                       \
  } while (0)
  switch (batch) {
    case 1: LAUNCH_BATCH(1); break;
    case 2: LAUNCH_BATCH(2); break;
    case 3: LAUNCH_BATCH(3); break;
    case 4: LAUNCH_BATCH(4); break;
    case 5: LAUNCH_BATCH(5); break;
    case 6: LAUNCH_BATCH(6); break;
    case 7: LAUNCH_BATCH(7); break;
    case 8: LAUNCH_BATCH(8); break;
  }
#undef LAUNCH_BATCH
}

void launch_layernorm_epilogue_fp16(
    const float* projection,
    const float* input_sum,
    const float* input_square_sum,
    const half* weight_sum,
    const half* fused_bias,
    half* output,
    int batch,
    int out_dim,
    int hidden_dim,
    float eps,
    cudaStream_t stream) {
  constexpr int kEpilogueThreads = 256;
  const int total_elements = batch * out_dim;
  const dim3 block(kEpilogueThreads);
  const dim3 grid(
      (total_elements + kEpilogueThreads - 1) / kEpilogueThreads);
  nonlinear::layernorm_epilogue_kernel<<<grid, block, 0, stream>>>(
      projection,
      input_sum,
      input_square_sum,
      weight_sum,
      fused_bias,
      output,
      total_elements,
      out_dim,
      hidden_dim,
      eps);
}

void launch_layernorm_stats_expand_fp16(
    const half* x,
    half* expanded_x,
    float* input_sum,
    float* input_square_sum,
    int batch,
    int hidden_dim,
    cudaStream_t stream) {
  const dim3 block(nonlinear::kLayerNormThreads);
#define LAUNCH_BATCH(BATCH)                                                  \
  nonlinear::layernorm_stats_expand_kernel<BATCH><<<1, block, 0, stream>>>( \
      x,                                                                    \
      expanded_x,                                                           \
      input_sum,                                                            \
      input_square_sum,                                                     \
      hidden_dim)
  switch (batch) {
    case 1: LAUNCH_BATCH(1); break;
    case 2: LAUNCH_BATCH(2); break;
    case 3: LAUNCH_BATCH(3); break;
    case 4: LAUNCH_BATCH(4); break;
    case 5: LAUNCH_BATCH(5); break;
    case 6: LAUNCH_BATCH(6); break;
    case 7: LAUNCH_BATCH(7); break;
    case 8: LAUNCH_BATCH(8); break;
  }
#undef LAUNCH_BATCH
}

void launch_layernorm_wmma_output_fp16(
    const half* expanded_x,
    const half* weight,
    const float* input_sum,
    const float* input_square_sum,
    const float* weight_sum,
    const float* fused_bias,
    half* output,
    int batch,
    int out_dim,
    int hidden_dim,
    float eps,
    cudaStream_t stream) {
  const dim3 block(nonlinear::kLayerNormThreads);
  const dim3 grid(
      (out_dim + nonlinear::kWmmaOutputsPerCta - 1) /
      nonlinear::kWmmaOutputsPerCta);
#define LAUNCH_BATCH(BATCH)                                                  \
  nonlinear::layernorm_wmma_output_kernel<BATCH>                            \
      <<<grid, block, 0, stream>>>(                                         \
          expanded_x,                                                       \
          weight,                                                           \
          input_sum,                                                        \
          input_square_sum,                                                 \
          weight_sum,                                                       \
          fused_bias,                                                       \
          output,                                                           \
          out_dim,                                                          \
          hidden_dim,                                                       \
          eps)
  switch (batch) {
    case 1: LAUNCH_BATCH(1); break;
    case 2: LAUNCH_BATCH(2); break;
    case 3: LAUNCH_BATCH(3); break;
    case 4: LAUNCH_BATCH(4); break;
    case 5: LAUNCH_BATCH(5); break;
    case 6: LAUNCH_BATCH(6); break;
    case 7: LAUNCH_BATCH(7); break;
    case 8: LAUNCH_BATCH(8); break;
  }
#undef LAUNCH_BATCH
}

template <int Batch>
cudaError_t launch_cooperative_layernorm_linear_typed(
    const half* x,
    half* packed_x,
    const half* weight,
    const half* weight_sum,
    const half* fused_bias,
    half* output,
    float* statistics,
    int out_dim,
    int hidden_dim,
    int multiprocessor_count,
    float eps,
    cudaStream_t stream) {
  const int output_blocks =
      (out_dim +
       nonlinear::kLayerNormWarps * nonlinear::kRowsPerWarp - 1) /
      (nonlinear::kLayerNormWarps * nonlinear::kRowsPerWarp);
  const int grid_blocks =
      min(output_blocks, multiprocessor_count * 2);
  const dim3 grid(grid_blocks);
  const dim3 block(nonlinear::kLayerNormThreads);
  void* arguments[] = {
      &x,
      &packed_x,
      &weight,
      &weight_sum,
      &fused_bias,
      &output,
      &statistics,
      &out_dim,
      &hidden_dim,
      &eps,
  };
  return cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(
          nonlinear::cooperative_layernorm_linear_kernel<Batch>),
      grid,
      block,
      arguments,
      0,
      stream);
}

cudaError_t launch_cooperative_layernorm_linear_fp16(
    const half* x,
    half* packed_x,
    const half* weight,
    const half* weight_sum,
    const half* fused_bias,
    half* output,
    float* statistics,
    int batch,
    int out_dim,
    int hidden_dim,
    int multiprocessor_count,
    float eps,
    cudaStream_t stream) {
  switch (batch) {
    case 1:
      return launch_cooperative_layernorm_linear_typed<1>(
          x, packed_x, weight, weight_sum, fused_bias, output, statistics,
          out_dim, hidden_dim, multiprocessor_count, eps, stream);
    case 2:
      return launch_cooperative_layernorm_linear_typed<2>(
          x, packed_x, weight, weight_sum, fused_bias, output, statistics,
          out_dim, hidden_dim, multiprocessor_count, eps, stream);
    case 3:
      return launch_cooperative_layernorm_linear_typed<3>(
          x, packed_x, weight, weight_sum, fused_bias, output, statistics,
          out_dim, hidden_dim, multiprocessor_count, eps, stream);
    case 4:
      return launch_cooperative_layernorm_linear_typed<4>(
          x, packed_x, weight, weight_sum, fused_bias, output, statistics,
          out_dim, hidden_dim, multiprocessor_count, eps, stream);
    case 5:
      return launch_cooperative_layernorm_linear_typed<5>(
          x, packed_x, weight, weight_sum, fused_bias, output, statistics,
          out_dim, hidden_dim, multiprocessor_count, eps, stream);
    case 6:
      return launch_cooperative_layernorm_linear_typed<6>(
          x, packed_x, weight, weight_sum, fused_bias, output, statistics,
          out_dim, hidden_dim, multiprocessor_count, eps, stream);
    case 7:
      return launch_cooperative_layernorm_linear_typed<7>(
          x, packed_x, weight, weight_sum, fused_bias, output, statistics,
          out_dim, hidden_dim, multiprocessor_count, eps, stream);
    case 8:
      return launch_cooperative_layernorm_linear_typed<8>(
          x, packed_x, weight, weight_sum, fused_bias, output, statistics,
          out_dim, hidden_dim, multiprocessor_count, eps, stream);
  }
  return cudaErrorInvalidValue;
}
