#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

namespace {
constexpr int kWarps = 8;
constexpr int kThreads = 256;
constexpr int kGroupSize = 64;

template <int Batch>
__global__ __launch_bounds__(kThreads) void projection_kernel(
    const half* __restrict__ x, const signed char* __restrict__ weight,
    const float* __restrict__ scale, half* __restrict__ output,
    float* __restrict__ square_sum, int out_dim, int hidden_dim) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (blockIdx.x == 0) {
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
  float accumulator[Batch] = {};
  const int groups = hidden_dim / kGroupSize;
  for (int group = 0; group < groups; ++group) {
    const int k = group * kGroupSize + lane * 2;
    const short packed = *reinterpret_cast<const short*>(weight + n * hidden_dim + k);
    const float lane_scale = lane == 0 ? scale[n * groups + group] : 0.0f;
    const float group_scale = __shfl_sync(0xffffffff, lane_scale, 0);
    const float weight_0 = static_cast<float>(static_cast<signed char>(packed & 0xff)) * group_scale;
    const float weight_1 = static_cast<float>(static_cast<signed char>((packed >> 8) & 0xff)) * group_scale;
#pragma unroll
    for (int b = 0; b < Batch; ++b) {
      const float2 values = __half22float2(*reinterpret_cast<const half2*>(x + b * hidden_dim + k));
      accumulator[b] = fmaf(values.x, weight_0, accumulator[b]);
      accumulator[b] = fmaf(values.y, weight_1, accumulator[b]);
    }
  }
  for (int offset = 16; offset; offset >>= 1) {
#pragma unroll
    for (int b = 0; b < Batch; ++b)
      accumulator[b] += __shfl_down_sync(0xffffffff, accumulator[b], offset);
  }
  if (lane == 0) {
#pragma unroll
    for (int b = 0; b < Batch; ++b) output[b * out_dim + n] = __float2half(accumulator[b]);
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
}  // namespace

void launch_projection(const half* x, const signed char* weight, const float* scale,
                       const half* bias, half* output, float* square_sum,
                       int batch, int out_dim, int hidden_dim, float eps,
                       cudaStream_t stream) {
  const dim3 block(kThreads), grid((out_dim + kWarps - 1) / kWarps);
#define LAUNCH(B) projection_kernel<B><<<grid, block, 0, stream>>>(x, weight, scale, output, square_sum, out_dim, hidden_dim)
  switch (batch) {
    case 1: LAUNCH(1); break; case 2: LAUNCH(2); break;
    case 3: LAUNCH(3); break; case 4: LAUNCH(4); break;
    case 5: LAUNCH(5); break; case 6: LAUNCH(6); break;
    case 7: LAUNCH(7); break; case 8: LAUNCH(8); break;
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int total = batch * out_dim;
  scale_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      output, square_sum, bias, total, out_dim, hidden_dim, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
