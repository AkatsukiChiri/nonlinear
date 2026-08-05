#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

namespace {
constexpr int kWarps = 2;
constexpr int kThreads = 64;

__device__ __forceinline__ int signed_nibble(uint32_t value, int shift) {
  return static_cast<int>(value << (28 - shift)) >> 28;
}

template <int Batch>
__global__ __launch_bounds__(kThreads) void projection_kernel(
    const half* __restrict__ x, const uint8_t* __restrict__ packed_weight,
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
  constexpr int Streams = Batch <= 2 ? 8 : (Batch <= 3 ? 4 : 1);
  float accumulator[Batch][Streams] = {};
  float direct_square[Batch][8] = {};
  const half2 scale_pair = __float2half2_rn(scale[n]);
  const int packed_stride = hidden_dim >> 1;
  for (int k = lane * 8; k < hidden_dim; k += 256) {
    const uint32_t packed = *reinterpret_cast<const uint32_t*>(
        packed_weight + n * packed_stride + (k >> 1));
    const half2 weight_01 = __hmul2(__halves2half2(
        __int2half_rn(signed_nibble(packed, 0)),
        __int2half_rn(signed_nibble(packed, 4))), scale_pair);
    const half2 weight_23 = __hmul2(__halves2half2(
        __int2half_rn(signed_nibble(packed, 8)),
        __int2half_rn(signed_nibble(packed, 12))), scale_pair);
    const half2 weight_45 = __hmul2(__halves2half2(
        __int2half_rn(signed_nibble(packed, 16)),
        __int2half_rn(signed_nibble(packed, 20))), scale_pair);
    const half2 weight_67 = __hmul2(__halves2half2(
        __int2half_rn(signed_nibble(packed, 24)),
        __int2half_rn(signed_nibble(packed, 28))), scale_pair);
    const float2 w01 = __half22float2(weight_01);
    const float2 w23 = __half22float2(weight_23);
    const float2 w45 = __half22float2(weight_45);
    const float2 w67 = __half22float2(weight_67);
#pragma unroll
    for (int b = 0; b < Batch; ++b) {
      const half2* values = reinterpret_cast<const half2*>(
          x + b * hidden_dim + k);
      const float2 v01 = __half22float2(values[0]);
      const float2 v23 = __half22float2(values[1]);
      const float2 v45 = __half22float2(values[2]);
      const float2 v67 = __half22float2(values[3]);
      if constexpr (Batch <= 2) {
        direct_square[b][0] = fmaf(v01.x, v01.x, direct_square[b][0]);
        direct_square[b][1] = fmaf(v01.y, v01.y, direct_square[b][1]);
        direct_square[b][2] = fmaf(v23.x, v23.x, direct_square[b][2]);
        direct_square[b][3] = fmaf(v23.y, v23.y, direct_square[b][3]);
        direct_square[b][4] = fmaf(v45.x, v45.x, direct_square[b][4]);
        direct_square[b][5] = fmaf(v45.y, v45.y, direct_square[b][5]);
        direct_square[b][6] = fmaf(v67.x, v67.x, direct_square[b][6]);
        direct_square[b][7] = fmaf(v67.y, v67.y, direct_square[b][7]);
      }
      if constexpr (Streams == 8) {
        accumulator[b][0] = fmaf(v01.x, w01.x, accumulator[b][0]);
        accumulator[b][1] = fmaf(v01.y, w01.y, accumulator[b][1]);
        accumulator[b][2] = fmaf(v23.x, w23.x, accumulator[b][2]);
        accumulator[b][3] = fmaf(v23.y, w23.y, accumulator[b][3]);
        accumulator[b][4] = fmaf(v45.x, w45.x, accumulator[b][4]);
        accumulator[b][5] = fmaf(v45.y, w45.y, accumulator[b][5]);
        accumulator[b][6] = fmaf(v67.x, w67.x, accumulator[b][6]);
        accumulator[b][7] = fmaf(v67.y, w67.y, accumulator[b][7]);
      } else if constexpr (Streams == 4) {
        accumulator[b][0] = fmaf(v01.x, w01.x, accumulator[b][0]);
        accumulator[b][1] = fmaf(v01.y, w01.y, accumulator[b][1]);
        accumulator[b][2] = fmaf(v23.x, w23.x, accumulator[b][2]);
        accumulator[b][3] = fmaf(v23.y, w23.y, accumulator[b][3]);
        accumulator[b][0] = fmaf(v45.x, w45.x, accumulator[b][0]);
        accumulator[b][1] = fmaf(v45.y, w45.y, accumulator[b][1]);
        accumulator[b][2] = fmaf(v67.x, w67.x, accumulator[b][2]);
        accumulator[b][3] = fmaf(v67.y, w67.y, accumulator[b][3]);
      } else {
        accumulator[b][0] = fmaf(v01.x, w01.x, accumulator[b][0]);
        accumulator[b][0] = fmaf(v01.y, w01.y, accumulator[b][0]);
        accumulator[b][0] = fmaf(v23.x, w23.x, accumulator[b][0]);
        accumulator[b][0] = fmaf(v23.y, w23.y, accumulator[b][0]);
        accumulator[b][0] = fmaf(v45.x, w45.x, accumulator[b][0]);
        accumulator[b][0] = fmaf(v45.y, w45.y, accumulator[b][0]);
        accumulator[b][0] = fmaf(v67.x, w67.x, accumulator[b][0]);
        accumulator[b][0] = fmaf(v67.y, w67.y, accumulator[b][0]);
      }
    }
  }
#pragma unroll
  for (int b = 0; b < Batch; ++b) {
#pragma unroll
    for (int stream = 1; stream < Streams; ++stream)
      accumulator[b][0] += accumulator[b][stream];
    for (int offset = 16; offset; offset >>= 1)
      accumulator[b][0] += __shfl_down_sync(0xffffffff, accumulator[b][0], offset);
    if constexpr (Batch <= 2) {
#pragma unroll
      for (int stream = 1; stream < 8; ++stream)
        direct_square[b][0] += direct_square[b][stream];
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
        rsqrtf(square_sum[batch] / float(hidden_dim) + eps);
    if (bias != nullptr) value += __half2float(bias[index % out_dim]);
    output[index] = __float2half(value);
  }
}
}  // namespace

void launch_int4_projection(const half* x, const uint8_t* weight,
                            const float* scale, const half* bias, half* output,
                            float* square_sum, int batch, int out_dim,
                            int hidden_dim, float eps, cudaStream_t stream) {
  const dim3 block(kThreads);
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
