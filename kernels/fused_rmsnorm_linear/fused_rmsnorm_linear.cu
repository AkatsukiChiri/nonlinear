#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace nonlinear {

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;

template <int Batch>
__global__ __launch_bounds__(kThreads) void fused_projection_square_sum_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ output,
    float* __restrict__ square_sum,
    int out_dim,
    int hidden_dim) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  // Only the first CTA computes the tiny FP32 RMS statistic. This work is
  // independent of the weight projection and avoids an additional launch.
  if (blockIdx.x == 0) {
    __shared__ float warp_square_sum[Batch][kWarps];
    float local_square_sum[Batch] = {};
    for (int k_index = threadIdx.x; k_index < hidden_dim;
         k_index += kThreads) {
#pragma unroll
      for (int batch_index = 0; batch_index < Batch; ++batch_index) {
        const float value = __half2float(x[batch_index * hidden_dim + k_index]);
        local_square_sum[batch_index] =
            fmaf(value, value, local_square_sum[batch_index]);
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (int batch_index = 0; batch_index < Batch; ++batch_index) {
        local_square_sum[batch_index] += __shfl_down_sync(
            0xffffffff, local_square_sum[batch_index], offset);
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int batch_index = 0; batch_index < Batch; ++batch_index) {
        warp_square_sum[batch_index][warp] = local_square_sum[batch_index];
      }
    }
    __syncthreads();
    if (threadIdx.x < Batch) {
      float total = 0.0f;
#pragma unroll
      for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
        total += warp_square_sum[threadIdx.x][warp_index];
      }
      square_sum[threadIdx.x] = total;
    }
  }

  const int output_index = blockIdx.x * kWarps + warp;
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
            reinterpret_cast<const half2*>(x + batch_index * hidden_dim)[
                pair_index]);
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
      output[batch_index * out_dim + output_index] =
          __float2half(accumulator[batch_index]);
    }
  }
}

__global__ void rms_scale_kernel(
    half* __restrict__ output,
    const float* __restrict__ square_sum,
    int total_elements,
    int out_dim,
    int hidden_dim,
    float eps) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total_elements) {
    const int batch_index = index / out_dim;
    const float inverse_rms = rsqrtf(
        square_sum[batch_index] / static_cast<float>(hidden_dim) + eps);
    output[index] = __float2half(__half2float(output[index]) * inverse_rms);
  }
}

__global__ void rms_scale_bias_kernel(
    half* __restrict__ output,
    const float* __restrict__ square_sum,
    const half* __restrict__ bias,
    int total_elements,
    int out_dim,
    int hidden_dim,
    float eps) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total_elements) {
    const int batch_index = index / out_dim;
    const int output_index = index - batch_index * out_dim;
    const float inverse_rms = rsqrtf(
        square_sum[batch_index] / static_cast<float>(hidden_dim) + eps);
    const float scaled = __half2float(output[index]) * inverse_rms;
    output[index] = __float2half(scaled + __half2float(bias[output_index]));
  }
}

}  // namespace nonlinear

void launch_fused_rmsnorm_projection_fp16(
    const half* x,
    const half* weight,
    half* output,
    float* square_sum,
    int batch,
    int out_dim,
    int hidden_dim,
    cudaStream_t stream) {
  const dim3 block(nonlinear::kThreads);
  const dim3 grid((out_dim + nonlinear::kWarps - 1) / nonlinear::kWarps);
#define LAUNCH_BATCH(BATCH)                                                   \
  nonlinear::fused_projection_square_sum_kernel<BATCH>                       \
      <<<grid, block, 0, stream>>>(                                          \
          x, weight, output, square_sum, out_dim, hidden_dim)
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

void launch_rms_scale_fp16(
    half* output,
    const float* square_sum,
    int batch,
    int out_dim,
    int hidden_dim,
    float eps,
    cudaStream_t stream) {
  constexpr int kScaleThreads = 256;
  const int total_elements = batch * out_dim;
  const dim3 block(kScaleThreads);
  const dim3 grid((total_elements + kScaleThreads - 1) / kScaleThreads);
  nonlinear::rms_scale_kernel<<<grid, block, 0, stream>>>(
      output, square_sum, total_elements, out_dim, hidden_dim, eps);
}

void launch_rms_scale_bias_fp16(
    half* output,
    const float* square_sum,
    const half* bias,
    int batch,
    int out_dim,
    int hidden_dim,
    float eps,
    cudaStream_t stream) {
  constexpr int kScaleThreads = 256;
  const int total_elements = batch * out_dim;
  const dim3 block(kScaleThreads);
  const dim3 grid((total_elements + kScaleThreads - 1) / kScaleThreads);
  nonlinear::rms_scale_bias_kernel<<<grid, block, 0, stream>>>(
      output, square_sum, bias, total_elements, out_dim, hidden_dim, eps);
}
