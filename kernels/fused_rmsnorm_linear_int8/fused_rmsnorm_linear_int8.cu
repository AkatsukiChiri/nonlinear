#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace nonlinear_int8 {

constexpr int kGroupSize = 64;
constexpr int kQuantThreads = 64;
constexpr int kWarps = 8;
constexpr int kGemmThreads = kWarps * 32;
constexpr int kOutputsPerWarp = 2;
constexpr int kOutputsPerBlock = kWarps * kOutputsPerWarp;

__global__ __launch_bounds__(kQuantThreads) void quantize_activation_kernel(
    const half* __restrict__ x,
    signed char* __restrict__ quantized_x,
    float* __restrict__ activation_scale,
    float* __restrict__ square_sum,
    int hidden_dim,
    int group_count) {
  const int batch_index = blockIdx.y;
  const int group_index = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int k_index = group_index * kGroupSize + threadIdx.x;
  const float value = __half2float(x[batch_index * hidden_dim + k_index]);
  float maximum = fabsf(value);
  float sum = value * value;

  for (int offset = 16; offset > 0; offset >>= 1) {
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }

  __shared__ float warp_maximum[2];
  __shared__ float warp_sum[2];
  __shared__ float group_scale;
  if (lane == 0) {
    warp_maximum[warp] = maximum;
    warp_sum[warp] = sum;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const float absolute_maximum = fmaxf(warp_maximum[0], warp_maximum[1]);
    group_scale = absolute_maximum > 0.0f ? absolute_maximum / 127.0f : 1.0f;
    activation_scale[batch_index * group_count + group_index] = group_scale;
    atomicAdd(square_sum + batch_index, warp_sum[0] + warp_sum[1]);
  }
  __syncthreads();

  int quantized = __float2int_rn(value / group_scale);
  quantized = max(-127, min(127, quantized));
  quantized_x[batch_index * hidden_dim + k_index] =
      static_cast<signed char>(quantized);
}

template <bool HasBias>
__global__ __launch_bounds__(kGemmThreads) void dp4a_projection_kernel(
    const signed char* __restrict__ quantized_x,
    const float* __restrict__ activation_scale,
    const signed char* __restrict__ quantized_weight,
    const float* __restrict__ weight_scale,
    const float* __restrict__ square_sum,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int batch,
    int out_dim,
    int hidden_dim,
    int group_count,
    float eps) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int half_warp = lane >> 4;
  const int lane_in_half_warp = lane & 15;
  const int output_index =
      blockIdx.x * kOutputsPerBlock + warp * kOutputsPerWarp + half_warp;
  const bool valid_output = output_index < out_dim;

  for (int batch_index = 0; batch_index < batch; ++batch_index) {
    float projected = 0.0f;
    for (int group_index = 0; group_index < group_count; ++group_index) {
      const int k_offset = group_index * kGroupSize + lane_in_half_warp * 4;
      const int packed_x = *reinterpret_cast<const int*>(
          quantized_x + batch_index * hidden_dim + k_offset);
      int packed_weight = 0;
      if (valid_output) {
        packed_weight = *reinterpret_cast<const int*>(
            quantized_weight + output_index * hidden_dim + k_offset);
      }
      int integer_dot = __dp4a(packed_x, packed_weight, 0);
      for (int offset = 8; offset > 0; offset >>= 1) {
        integer_dot += __shfl_down_sync(
            0xffffffff, integer_dot, offset, 16);
      }
      if (lane_in_half_warp == 0 && valid_output) {
        const float x_scale =
            activation_scale[batch_index * group_count + group_index];
        const float w_scale =
            weight_scale[output_index * group_count + group_index];
        projected = fmaf(
            static_cast<float>(integer_dot), x_scale * w_scale, projected);
      }
    }

    if (lane_in_half_warp == 0 && valid_output) {
      const float inverse_rms = rsqrtf(
          square_sum[batch_index] / static_cast<float>(hidden_dim) + eps);
      float result = projected * inverse_rms;
      if constexpr (HasBias) {
        result += __half2float(bias[output_index]);
      }
      output[batch_index * out_dim + output_index] = __float2half(result);
    }
  }
}

}  // namespace nonlinear_int8

void launch_quantize_activation_int8(
    const half* x,
    signed char* quantized_x,
    float* activation_scale,
    float* square_sum,
    int batch,
    int hidden_dim,
    cudaStream_t stream) {
  const int group_count = hidden_dim / nonlinear_int8::kGroupSize;
  const dim3 block(nonlinear_int8::kQuantThreads);
  const dim3 grid(group_count, batch);
  nonlinear_int8::quantize_activation_kernel<<<grid, block, 0, stream>>>(
      x,
      quantized_x,
      activation_scale,
      square_sum,
      hidden_dim,
      group_count);
}

void launch_dp4a_projection_int8(
    const signed char* quantized_x,
    const float* activation_scale,
    const signed char* quantized_weight,
    const float* weight_scale,
    const float* square_sum,
    const half* bias,
    half* output,
    int batch,
    int out_dim,
    int hidden_dim,
    float eps,
    cudaStream_t stream) {
  const int group_count = hidden_dim / nonlinear_int8::kGroupSize;
  const dim3 block(nonlinear_int8::kGemmThreads);
  const dim3 grid(
      (out_dim + nonlinear_int8::kOutputsPerBlock - 1) /
      nonlinear_int8::kOutputsPerBlock);
  if (bias == nullptr) {
    nonlinear_int8::dp4a_projection_kernel<false><<<grid, block, 0, stream>>>(
        quantized_x, activation_scale, quantized_weight, weight_scale,
        square_sum, bias, output, batch, out_dim, hidden_dim, group_count, eps);
  } else {
    nonlinear_int8::dp4a_projection_kernel<true><<<grid, block, 0, stream>>>(
        quantized_x, activation_scale, quantized_weight, weight_scale,
        square_sum, bias, output, batch, out_dim, hidden_dim, group_count, eps);
  }
}
