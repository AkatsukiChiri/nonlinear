#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <mma.h>

namespace {
constexpr int kWarps = 2;
constexpr int kThreads = 64;
constexpr int kGroupSize = 64;
constexpr int kSplitK = 4;

template <int Batch>
__global__ __launch_bounds__(kThreads) void projection_kernel(
    const half* __restrict__ x, const signed char* __restrict__ weight,
    const float* __restrict__ scale, half* __restrict__ output,
    float* __restrict__ square_sum, const half* __restrict__ bias,
    int out_dim, int hidden_dim, float eps) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if constexpr (Batch > 2) if (blockIdx.x == 0) {
    __shared__ float warp_sum[Batch][kWarps];
    float local[Batch] = {};
    for (int k = threadIdx.x; k < hidden_dim; k += kThreads) {
#pragma unroll
      for (int b = 0; b < Batch; ++b) {
        const float value = __half2float(x[b * hidden_dim + k]);
        local[b] = fmaf(value, value, local[b]);
      }
    }
    for (int offset = 16; offset; offset >>= 1) {
#pragma unroll
      for (int b = 0; b < Batch; ++b)
        local[b] += __shfl_down_sync(0xffffffff, local[b], offset);
    }
    if (lane == 0) {
#pragma unroll
      for (int b = 0; b < Batch; ++b) warp_sum[b][warp] = local[b];
    }
    __syncthreads();
    if (threadIdx.x < Batch) {
      float total = 0.0f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) total += warp_sum[threadIdx.x][w];
      square_sum[threadIdx.x] = total;
    }
  }

  const int n = blockIdx.x * kWarps + warp;
  if (n >= out_dim) return;
  constexpr int kAccumulatorStreams = Batch <= 3 ? 4 : 1;
  float accumulator[Batch][kAccumulatorStreams] = {};
  float direct_square[Batch][4] = {};
  const int groups = hidden_dim / kGroupSize;
  const half2 scale_pair = __float2half2_rn(scale[n]);
  for (int group_pair = 0; group_pair < groups / 2; ++group_pair) {
    const int group = group_pair * 2 + (lane >> 4);
    const int k = group * kGroupSize + (lane & 15) * 4;
    const int packed = *reinterpret_cast<const int*>(weight + n * hidden_dim + k);
    const half2 quantized_01 = __halves2half2(
        __int2half_rn(static_cast<signed char>(packed & 0xff)),
        __int2half_rn(static_cast<signed char>((packed >> 8) & 0xff)));
    const half2 quantized_23 = __halves2half2(
        __int2half_rn(static_cast<signed char>((packed >> 16) & 0xff)),
        __int2half_rn(static_cast<signed char>((packed >> 24) & 0xff)));
    const half2 weight_01 = __hmul2(quantized_01, scale_pair);
    const half2 weight_23 = __hmul2(quantized_23, scale_pair);
    const float2 weights_01 = __half22float2(weight_01);
    const float2 weights_23 = __half22float2(weight_23);
#pragma unroll
    for (int b = 0; b < Batch; ++b) {
      const float2 values_01 = __half22float2(
          *reinterpret_cast<const half2*>(x + b * hidden_dim + k));
      const float2 values_23 = __half22float2(
          *reinterpret_cast<const half2*>(x + b * hidden_dim + k + 2));
      if constexpr (Batch <= 2) {
        direct_square[b][0] = fmaf(values_01.x, values_01.x, direct_square[b][0]);
        direct_square[b][1] = fmaf(values_01.y, values_01.y, direct_square[b][1]);
        direct_square[b][2] = fmaf(values_23.x, values_23.x, direct_square[b][2]);
        direct_square[b][3] = fmaf(values_23.y, values_23.y, direct_square[b][3]);
      }
      if constexpr (Batch <= 3) {
        accumulator[b][0] = fmaf(values_01.x, weights_01.x, accumulator[b][0]);
        accumulator[b][1] = fmaf(values_01.y, weights_01.y, accumulator[b][1]);
        accumulator[b][2] = fmaf(values_23.x, weights_23.x, accumulator[b][2]);
        accumulator[b][3] = fmaf(values_23.y, weights_23.y, accumulator[b][3]);
      } else {
        accumulator[b][0] = fmaf(values_01.x, weights_01.x, accumulator[b][0]);
        accumulator[b][0] = fmaf(values_01.y, weights_01.y, accumulator[b][0]);
        accumulator[b][0] = fmaf(values_23.x, weights_23.x, accumulator[b][0]);
        accumulator[b][0] = fmaf(values_23.y, weights_23.y, accumulator[b][0]);
      }
    }
  }
#pragma unroll
  for (int b = 0; b < Batch; ++b) {
    if constexpr (Batch <= 3)
      accumulator[b][0] += accumulator[b][1] + accumulator[b][2] + accumulator[b][3];
    for (int offset = 16; offset; offset >>= 1)
      accumulator[b][0] += __shfl_down_sync(0xffffffff, accumulator[b][0], offset);
  }
  if constexpr (Batch <= 2) {
#pragma unroll
    for (int b = 0; b < Batch; ++b) {
      direct_square[b][0] += direct_square[b][1] + direct_square[b][2] +
                             direct_square[b][3];
      for (int offset = 16; offset; offset >>= 1)
        direct_square[b][0] += __shfl_down_sync(
            0xffffffff, direct_square[b][0], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int b = 0; b < Batch; ++b) {
      float value = accumulator[b][0];
      if constexpr (Batch <= 2) {
        value *= rsqrtf(direct_square[b][0] / float(hidden_dim) + eps);
        if (bias != nullptr) value += __half2float(bias[n]);
      }
      output[b * out_dim + n] = __float2half(value);
    }
  }
}

__global__ void scale_kernel(half* output, const float* square_sum,
                             const half* bias, int total, int out_dim,
                             int hidden_dim, float eps) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total) {
    const int batch = index / out_dim;
    float value = __half2float(output[index]) *
        rsqrtf(square_sum[batch] / static_cast<float>(hidden_dim) + eps);
    if (bias != nullptr) value += __half2float(bias[index % out_dim]);
    output[index] = __float2half(value);
  }
}

template <int Batch>
__global__ __launch_bounds__(kThreads, 2) void tensor_group_kernel(
    const half* x, const signed char* weight, const float* scale,
    float* partial, float* square_sum, int out_dim, int hidden_dim) {
  using namespace nvcuda;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (blockIdx.x == 0 && blockIdx.y == 0) {
    __shared__ float sums[Batch][kWarps];
    float local[Batch] = {};
    for (int k = threadIdx.x; k < hidden_dim; k += kThreads) {
#pragma unroll
      for (int row = 0; row < Batch; ++row) {
        const float value = __half2float(x[row * hidden_dim + k]);
        local[row] = fmaf(value, value, local[row]);
      }
    }
    for (int offset = 16; offset; offset >>= 1) {
#pragma unroll
      for (int row = 0; row < Batch; ++row)
        local[row] += __shfl_down_sync(0xffffffff, local[row], offset);
    }
    if (lane == 0) {
#pragma unroll
      for (int row = 0; row < Batch; ++row) sums[row][warp] = local[row];
    }
    __syncthreads();
    if (threadIdx.x < Batch) {
      float total = 0.0f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) total += sums[threadIdx.x][w];
      square_sum[threadIdx.x] = total;
    }
    __syncthreads();
  }
  __shared__ half tile_a[16 * kGroupSize];
  __shared__ half tile_b[kWarps][16 * kGroupSize];
  __shared__ float tile_c[kWarps][16 * 16];
  const int n_start = (blockIdx.x * kWarps + warp) * 16;
  const int groups = hidden_dim / kGroupSize;
  const int groups_per_split = groups / kSplitK;
  const int group_begin = blockIdx.y * groups_per_split;
  const int group_end = group_begin + groups_per_split;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  for (int group = group_begin; group < group_end; ++group) {
    for (int index = threadIdx.x; index < 16 * kGroupSize; index += kThreads) {
      const int row = index / kGroupSize;
      const int k_offset = index % kGroupSize;
      tile_a[index] = row < Batch
          ? x[row * hidden_dim + group * kGroupSize + k_offset]
          : __float2half(0.0f);
    }
    for (int index = lane; index < 16 * kGroupSize; index += 32) {
      const int column = index / kGroupSize;
      const int k_offset = index % kGroupSize;
      float value = 0.0f;
      if (n_start + column < out_dim)
        value = float(weight[(n_start + column) * hidden_dim +
                      group * kGroupSize + k_offset]) *
                scale[(n_start + column) * groups + group];
      tile_b[warp][index] = __float2half(value);
    }
    __syncthreads();
#pragma unroll
    for (int k_tile = 0; k_tile < kGroupSize; k_tile += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
      wmma::load_matrix_sync(a, tile_a + k_tile, kGroupSize);
      wmma::load_matrix_sync(b, tile_b[warp] + k_tile, kGroupSize);
      wmma::mma_sync(accumulator, a, b, accumulator);
    }
    __syncthreads();
  }
  if (n_start < out_dim) {
    wmma::store_matrix_sync(tile_c[warp], accumulator, 16, wmma::mem_row_major);
    __syncwarp();
    for (int index = lane; index < Batch * 16; index += 32) {
      const int row = index >> 4;
      const int column = index & 15;
      if (n_start + column < out_dim)
        partial[(blockIdx.y * Batch + row) * out_dim + n_start + column] =
            tile_c[warp][index];
    }
  }
}

__global__ void tensor_finalize_kernel(const float* partial,
    const float* square_sum, const half* bias, half* output, int total,
    int out_dim, int hidden_dim, float eps) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) return;
  float value = 0.0f;
#pragma unroll
  for (int split = 0; split < kSplitK; ++split)
    value += partial[split * total + index];
  value *= rsqrtf(square_sum[index / out_dim] / float(hidden_dim) + eps);
  if (bias != nullptr) value += __half2float(bias[index % out_dim]);
  output[index] = __float2half(value);
}

}  // namespace

void launch_projection(const half* x, const signed char* weight,
                       const float* scale, const half* bias, half* output,
                       float* partial, float* square_sum, int batch, int out_dim,
                       int hidden_dim, float eps, cudaStream_t stream) {
  const dim3 block(kThreads);
  // Retained as an experimental path for profiling; padding decode M to 16 is
  // slower than the warp-GEMV path on SM86 for every supported batch.
  if (batch > 8) {
    const dim3 tensor_grid(
        (out_dim + kWarps * 16 - 1) / (kWarps * 16), kSplitK);
#define TENSOR_LAUNCH(B) tensor_group_kernel<B><<<tensor_grid, block, 0, stream>>>( \
      x, weight, scale, partial, square_sum, out_dim, hidden_dim)
    switch (batch) {
      case 4: TENSOR_LAUNCH(4); break; case 5: TENSOR_LAUNCH(5); break;
      case 6: TENSOR_LAUNCH(6); break; case 7: TENSOR_LAUNCH(7); break;
      case 8: TENSOR_LAUNCH(8); break;
    }
#undef TENSOR_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int total = batch * out_dim;
    tensor_finalize_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        partial, square_sum, bias, output, total, out_dim, hidden_dim, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  const dim3 grid((out_dim + kWarps - 1) / kWarps);
#define LAUNCH(B) projection_kernel<B><<<grid, block, 0, stream>>>( \
      x, weight, scale, output, square_sum, bias, out_dim, hidden_dim, eps)
  switch (batch) {
    case 1: LAUNCH(1); break; case 2: LAUNCH(2); break;
    case 3: LAUNCH(3); break; case 4: LAUNCH(4); break;
    case 5: LAUNCH(5); break; case 6: LAUNCH(6); break;
    case 7: LAUNCH(7); break; case 8: LAUNCH(8); break;
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (batch <= 2) return;
  const int total = batch * out_dim;
  scale_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      output, square_sum, bias, total, out_dim, hidden_dim, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
