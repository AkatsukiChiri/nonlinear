#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace nonlinear {

constexpr int kWarpSize = 32;

union Half8Pack {
  uint4 vector;
  half2 pairs[4];
};

__global__ void residual_add_fp16_kernel(
    const half* __restrict__ x,
    const half* __restrict__ x_res,
    half* __restrict__ residual_sum,
    int total_elements) {
  const int vector_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int vector_count = total_elements / 8;
  if (vector_index < vector_count) {
    Half8Pack x_pack;
    Half8Pack x_res_pack;
    Half8Pack sum_pack;
    x_pack.vector =
        reinterpret_cast<const uint4*>(x)[vector_index];
    x_res_pack.vector =
        reinterpret_cast<const uint4*>(x_res)[vector_index];
#pragma unroll
    for (int pair_index = 0; pair_index < 4; ++pair_index) {
      sum_pack.pairs[pair_index] = __hadd2(
          x_pack.pairs[pair_index],
          x_res_pack.pairs[pair_index]);
    }
    reinterpret_cast<uint4*>(residual_sum)[vector_index] =
        sum_pack.vector;
  }
  const int tail_start = vector_count * 8;
  for (int index = tail_start + vector_index;
       index < total_elements;
       index += gridDim.x * blockDim.x) {
    residual_sum[index] = __hadd(x[index], x_res[index]);
  }
}

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

template <int Batch, int StreamCount>
__global__ __launch_bounds__(8 * kWarpSize)
void fused_add_projection_square_sum_kernel(
    const half* __restrict__ x,
    const half* __restrict__ x_res,
    const half* __restrict__ weight,
    half* __restrict__ output,
    half* __restrict__ residual_sum,
    float* __restrict__ square_sum,
    int out_dim,
    int hidden_dim) {
  constexpr int kWarps = 8;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;

  if (blockIdx.x == 0) {
    __shared__ float warp_square_sum[Batch][kWarps];
    float local_square_sum[Batch] = {};
#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      for (int k_index = threadIdx.x; k_index < hidden_dim;
           k_index += blockDim.x) {
        const int input_index = batch_index * hidden_dim + k_index;
        const half sum = __hadd(x[input_index], x_res[input_index]);
        residual_sum[input_index] = sum;
        const float value = __half2float(sum);
        local_square_sum[batch_index] = fmaf(
            value,
            value,
            local_square_sum[batch_index]);
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (int batch_index = 0; batch_index < Batch; ++batch_index) {
        local_square_sum[batch_index] += __shfl_down_sync(
            0xffffffff,
            local_square_sum[batch_index],
            offset);
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int batch_index = 0; batch_index < Batch; ++batch_index) {
        warp_square_sum[batch_index][warp] =
            local_square_sum[batch_index];
      }
    }
    __syncthreads();
    if (threadIdx.x < Batch) {
      float block_square_sum = 0.0f;
#pragma unroll
      for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
        block_square_sum +=
            warp_square_sum[threadIdx.x][warp_index];
      }
      square_sum[threadIdx.x] = block_square_sum;
    }
  }

  const int output_index = blockIdx.x * kWarps + warp;
  if (output_index >= out_dim) {
    return;
  }
  const half* weight_row = weight + output_index * hidden_dim;
  float projection[Batch] = {};
  if ((hidden_dim & 7) == 0) {
    float streams[StreamCount][Batch] = {};
    const int vector_count = hidden_dim / 8;
    for (int vector_index = lane; vector_index < vector_count;
         vector_index += kWarpSize) {
      Half8Pack weight_pack;
      weight_pack.vector =
          reinterpret_cast<const uint4*>(weight_row)[vector_index];
      if constexpr (Batch == 8) {
        Half8Pack x_pack[Batch];
#pragma unroll
        for (int batch_index = 0; batch_index < Batch; ++batch_index) {
          x_pack[batch_index].vector = reinterpret_cast<const uint4*>(
              x + batch_index * hidden_dim)[vector_index];
        }
#pragma unroll
        for (int pair_index = 0; pair_index < 4; ++pair_index) {
          const int stream_index = pair_index % StreamCount;
          const float2 weight_value =
              __half22float2(weight_pack.pairs[pair_index]);
          const int input_pair_index = vector_index * 4 + pair_index;
#pragma unroll
          for (int batch_index = 0; batch_index < Batch; ++batch_index) {
            const half2 x_res_pair = reinterpret_cast<const half2*>(
                x_res + batch_index * hidden_dim)[input_pair_index];
            const float2 value = __half22float2(__hadd2(
                x_pack[batch_index].pairs[pair_index],
                x_res_pair));
            streams[stream_index][batch_index] = fmaf(
                value.x,
                weight_value.x,
                streams[stream_index][batch_index]);
            streams[stream_index][batch_index] = fmaf(
                value.y,
                weight_value.y,
                streams[stream_index][batch_index]);
          }
        }
      } else {
        Half8Pack sum_pack[Batch];
#pragma unroll
        for (int batch_index = 0; batch_index < Batch; ++batch_index) {
          Half8Pack x_pack;
          Half8Pack x_res_pack;
          x_pack.vector = reinterpret_cast<const uint4*>(
              x + batch_index * hidden_dim)[vector_index];
          x_res_pack.vector = reinterpret_cast<const uint4*>(
              x_res + batch_index * hidden_dim)[vector_index];
#pragma unroll
          for (int pair_index = 0; pair_index < 4; ++pair_index) {
            sum_pack[batch_index].pairs[pair_index] = __hadd2(
                x_pack.pairs[pair_index],
                x_res_pack.pairs[pair_index]);
          }
        }
#pragma unroll
        for (int pair_index = 0; pair_index < 4; ++pair_index) {
          const int stream_index = pair_index % StreamCount;
          const float2 weight_value =
              __half22float2(weight_pack.pairs[pair_index]);
#pragma unroll
          for (int batch_index = 0; batch_index < Batch; ++batch_index) {
            const float2 value =
                __half22float2(sum_pack[batch_index].pairs[pair_index]);
            streams[stream_index][batch_index] = fmaf(
                value.x,
                weight_value.x,
                streams[stream_index][batch_index]);
            streams[stream_index][batch_index] = fmaf(
                value.y,
                weight_value.y,
                streams[stream_index][batch_index]);
          }
        }
      }
    }
#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      float batch_projection = 0.0f;
#pragma unroll
      for (int stream_index = 0; stream_index < StreamCount;
           ++stream_index) {
        batch_projection += streams[stream_index][batch_index];
      }
      projection[batch_index] = batch_projection;
    }
  } else {
    for (int k_index = lane; k_index < hidden_dim; k_index += kWarpSize) {
      const float weight_value = __half2float(weight_row[k_index]);
#pragma unroll
      for (int batch_index = 0; batch_index < Batch; ++batch_index) {
        const int input_index = batch_index * hidden_dim + k_index;
        const float value = __half2float(
            __hadd(x[input_index], x_res[input_index]));
        projection[batch_index] = fmaf(
            value,
            weight_value,
            projection[batch_index]);
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
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      output[batch_index * out_dim + output_index] =
          __float2half(projection[batch_index]);
    }
  }
}

template <int Batch>
__global__ void residual_add_inverse_rms_kernel(
    const half* __restrict__ x,
    const half* __restrict__ x_res,
    half* __restrict__ residual_sum,
    float* __restrict__ inverse_rms,
    int hidden_dim,
    float eps) {
  constexpr int kWarps = 8;
  __shared__ float warp_square_sum[Batch][kWarps];
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  float local_square_sum[Batch] = {};
  for (int k_index = threadIdx.x; k_index < hidden_dim;
       k_index += blockDim.x) {
#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      const int input_index = batch_index * hidden_dim + k_index;
      const half sum = __hadd(x[input_index], x_res[input_index]);
      residual_sum[input_index] = sum;
      const float value = __half2float(sum);
      local_square_sum[batch_index] = fmaf(
          value,
          value,
          local_square_sum[batch_index]);
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      local_square_sum[batch_index] += __shfl_down_sync(
          0xffffffff,
          local_square_sum[batch_index],
          offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      warp_square_sum[batch_index][warp] =
          local_square_sum[batch_index];
    }
  }
  __syncthreads();
  if (threadIdx.x < Batch) {
    float square_sum = 0.0f;
#pragma unroll
    for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
      square_sum += warp_square_sum[threadIdx.x][warp_index];
    }
    inverse_rms[threadIdx.x] = rsqrtf(
        square_sum / static_cast<float>(hidden_dim) + eps);
  }
}

template <int Batch>
__global__ __launch_bounds__(8 * kWarpSize)
void rms_linear_projection_direct_kernel(
    const half* __restrict__ residual_sum,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const float* __restrict__ inverse_rms,
    half* __restrict__ output,
    int out_dim,
    int hidden_dim) {
  constexpr int kWarps = 8;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int output_index = blockIdx.x * kWarps + warp;
  if (output_index >= out_dim) {
    return;
  }

  float accumulator[Batch] = {};
  const half* weight_row = weight + output_index * hidden_dim;
  if ((hidden_dim & 7) == 0) {
    float streams[4][Batch] = {};
    const int vector_count = hidden_dim / 8;
    for (int vector_index = lane; vector_index < vector_count;
         vector_index += kWarpSize) {
      Half8Pack weight_pack;
      Half8Pack residual_pack[Batch];
      weight_pack.vector =
          reinterpret_cast<const uint4*>(weight_row)[vector_index];
#pragma unroll
      for (int batch_index = 0; batch_index < Batch; ++batch_index) {
        residual_pack[batch_index].vector =
            reinterpret_cast<const uint4*>(
                residual_sum + batch_index * hidden_dim)[vector_index];
      }
#pragma unroll
      for (int pair_index = 0; pair_index < 4; ++pair_index) {
        const float2 weight_value =
            __half22float2(weight_pack.pairs[pair_index]);
#pragma unroll
        for (int batch_index = 0; batch_index < Batch; ++batch_index) {
          const float2 value = __half22float2(
              residual_pack[batch_index].pairs[pair_index]);
          streams[pair_index][batch_index] = fmaf(
              value.x,
              weight_value.x,
              streams[pair_index][batch_index]);
          streams[pair_index][batch_index] = fmaf(
              value.y,
              weight_value.y,
              streams[pair_index][batch_index]);
        }
      }
    }
#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      accumulator[batch_index] =
          streams[0][batch_index] +
          streams[1][batch_index] +
          streams[2][batch_index] +
          streams[3][batch_index];
    }
  } else {
    for (int k_index = lane; k_index < hidden_dim; k_index += kWarpSize) {
      const float weight_value = __half2float(weight_row[k_index]);
#pragma unroll
      for (int batch_index = 0; batch_index < Batch; ++batch_index) {
        accumulator[batch_index] = fmaf(
            __half2float(
                residual_sum[batch_index * hidden_dim + k_index]),
            weight_value,
            accumulator[batch_index]);
      }
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      accumulator[batch_index] += __shfl_down_sync(
          0xffffffff,
          accumulator[batch_index],
          offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      float value =
          accumulator[batch_index] * inverse_rms[batch_index];
      if (bias != nullptr) {
        value += __half2float(bias[output_index]);
      }
      output[batch_index * out_dim + output_index] =
          __float2half(value);
    }
  }
}

}  // namespace nonlinear

void launch_fused_add_rms_linear_batch8_fp16(
    const half* x,
    const half* x_res,
    const half* weight,
    const half* bias,
    half* output,
    half* residual_sum,
    float* inverse_rms,
    int out_dim,
    int hidden_dim,
    float eps,
    cudaStream_t stream) {
  constexpr int kWarps = 8;
  constexpr int kThreads = kWarps * nonlinear::kWarpSize;
  nonlinear::residual_add_inverse_rms_kernel<8>
      <<<1, kThreads, 0, stream>>>(
          x,
          x_res,
          residual_sum,
          inverse_rms,
          hidden_dim,
          eps);
  const dim3 block(kThreads);
  const dim3 grid((out_dim + kWarps - 1) / kWarps);
  nonlinear::rms_linear_projection_direct_kernel<8>
      <<<grid, block, 0, stream>>>(
          residual_sum,
          weight,
          bias,
          inverse_rms,
          output,
          out_dim,
          hidden_dim);
}

void launch_residual_add_fp16(
    const half* x,
    const half* x_res,
    half* residual_sum,
    int total_elements,
    cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int vector_count = (total_elements + 7) / 8;
  const dim3 block(kThreads);
  const dim3 grid((vector_count + kThreads - 1) / kThreads);
  nonlinear::residual_add_fp16_kernel<<<grid, block, 0, stream>>>(
      x,
      x_res,
      residual_sum,
      total_elements);
}

void launch_fused_add_rms_projection_fp16(
    const half* x,
    const half* x_res,
    const half* weight,
    half* output,
    half* residual_sum,
    float* square_sum,
    int batch,
    int out_dim,
    int hidden_dim,
    cudaStream_t stream) {
  constexpr int kWarps = 8;
  const dim3 block(kWarps * nonlinear::kWarpSize);
  const dim3 grid((out_dim + kWarps - 1) / kWarps);
  nonlinear::fused_add_projection_square_sum_kernel<4, 4>
      <<<grid, block, 0, stream>>>(
          x,
          x_res,
          weight,
          output,
          residual_sum,
          square_sum,
          out_dim,
          hidden_dim);
}

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
