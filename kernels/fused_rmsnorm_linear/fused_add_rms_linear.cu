#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace nonlinear {

constexpr int kWarpSize = 32;

union Half8Pack {
  uint4 vector;
  half2 pairs[4];
};

template <int Batch, int Warps>
__global__ __launch_bounds__(Warps * kWarpSize)
void fused_add_rms_linear_kernel(
    const half* __restrict__ x,
    const half* __restrict__ x_res,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    half* __restrict__ residual_sum,
    int out_dim,
    int hidden_dim,
    float eps) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int output_index = blockIdx.x * Warps + warp;
  if (output_index >= out_dim) {
    return;
  }

  const bool writes_residual = blockIdx.x == 0 && warp == 0;
  const half* weight_row = weight + output_index * hidden_dim;
  float projection[Batch] = {};
  float square_sum[Batch] = {};

  if ((hidden_dim & 7) == 0) {
    float projection_streams[4][Batch] = {};
    float square_streams[4][Batch] = {};
    const int vector_count = hidden_dim / 8;

    for (int vector_index = lane; vector_index < vector_count;
         vector_index += kWarpSize) {
      Half8Pack weight_pack;
      weight_pack.vector =
          reinterpret_cast<const uint4*>(weight_row)[vector_index];

#pragma unroll
      for (int batch_index = 0; batch_index < Batch; ++batch_index) {
        Half8Pack x_pack;
        Half8Pack x_res_pack;
        Half8Pack sum_pack;
        x_pack.vector =
            reinterpret_cast<const uint4*>(
                x + batch_index * hidden_dim)[vector_index];
        x_res_pack.vector =
            reinterpret_cast<const uint4*>(
                x_res + batch_index * hidden_dim)[vector_index];

#pragma unroll
        for (int pair_index = 0; pair_index < 4; ++pair_index) {
          sum_pack.pairs[pair_index] =
              __hadd2(x_pack.pairs[pair_index], x_res_pack.pairs[pair_index]);
          const float2 value =
              __half22float2(sum_pack.pairs[pair_index]);
          const float2 weight_value =
              __half22float2(weight_pack.pairs[pair_index]);
          projection_streams[pair_index][batch_index] = fmaf(
              value.x,
              weight_value.x,
              projection_streams[pair_index][batch_index]);
          projection_streams[pair_index][batch_index] = fmaf(
              value.y,
              weight_value.y,
              projection_streams[pair_index][batch_index]);
          square_streams[pair_index][batch_index] = fmaf(
              value.x,
              value.x,
              square_streams[pair_index][batch_index]);
          square_streams[pair_index][batch_index] = fmaf(
              value.y,
              value.y,
              square_streams[pair_index][batch_index]);
        }

        if (writes_residual) {
          reinterpret_cast<uint4*>(
              residual_sum + batch_index * hidden_dim)[vector_index] =
              sum_pack.vector;
        }
      }
    }

#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      projection[batch_index] =
          projection_streams[0][batch_index] +
          projection_streams[1][batch_index] +
          projection_streams[2][batch_index] +
          projection_streams[3][batch_index];
      square_sum[batch_index] =
          square_streams[0][batch_index] +
          square_streams[1][batch_index] +
          square_streams[2][batch_index] +
          square_streams[3][batch_index];
    }
  } else {
    for (int k_index = lane; k_index < hidden_dim; k_index += kWarpSize) {
      const float weight_value = __half2float(weight_row[k_index]);
#pragma unroll
      for (int batch_index = 0; batch_index < Batch; ++batch_index) {
        const int input_index = batch_index * hidden_dim + k_index;
        const half sum = __hadd(x[input_index], x_res[input_index]);
        const float value = __half2float(sum);
        projection[batch_index] = fmaf(
            value,
            weight_value,
            projection[batch_index]);
        square_sum[batch_index] = fmaf(
            value,
            value,
            square_sum[batch_index]);
        if (writes_residual) {
          residual_sum[input_index] = sum;
        }
      }
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      projection[batch_index] += __shfl_down_sync(
          0xffffffff,
          projection[batch_index],
          offset);
      square_sum[batch_index] += __shfl_down_sync(
          0xffffffff,
          square_sum[batch_index],
          offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      float value = projection[batch_index] * rsqrtf(
          square_sum[batch_index] / static_cast<float>(hidden_dim) + eps);
      if (bias != nullptr) {
        value += __half2float(bias[output_index]);
      }
      output[batch_index * out_dim + output_index] = __float2half(value);
    }
  }
}

}  // namespace nonlinear

void launch_fused_add_rms_linear_fp16(
    const half* x,
    const half* x_res,
    const half* weight,
    const half* bias,
    half* output,
    half* residual_sum,
    int batch,
    int out_dim,
    int hidden_dim,
    float eps,
    cudaStream_t stream) {
#define LAUNCH_ADD_RMS(BATCH, WARPS)                                         \
  do {                                                                       \
    const dim3 block((WARPS) * nonlinear::kWarpSize);                        \
    const dim3 grid((out_dim + (WARPS) - 1) / (WARPS));                      \
    nonlinear::fused_add_rms_linear_kernel<BATCH, WARPS>                     \
        <<<grid, block, 0, stream>>>(                                        \
            x,                                                               \
            x_res,                                                           \
            weight,                                                          \
            bias,                                                            \
            output,                                                          \
            residual_sum,                                                    \
            out_dim,                                                         \
            hidden_dim,                                                      \
            eps);                                                            \
  } while (false)

  switch (batch) {
    case 1: LAUNCH_ADD_RMS(1, 32); break;
    case 2: LAUNCH_ADD_RMS(2, 32); break;
    case 3: LAUNCH_ADD_RMS(3, 8); break;
    case 4: LAUNCH_ADD_RMS(4, 8); break;
    case 5: LAUNCH_ADD_RMS(5, 8); break;
    case 6: LAUNCH_ADD_RMS(6, 8); break;
    case 7: LAUNCH_ADD_RMS(7, 8); break;
    case 8: LAUNCH_ADD_RMS(8, 8); break;
  }

#undef LAUNCH_ADD_RMS
}
