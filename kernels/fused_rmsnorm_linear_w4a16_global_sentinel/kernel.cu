#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int kAwqWarps = 8;
constexpr int kAwqThreads = kAwqWarps * 32;
constexpr int kAwqOutputsPerBlock = 64;
constexpr int kAwqPackedColumnsPerBlock = kAwqOutputsPerBlock / 8;

template <int Batch>
__global__ __launch_bounds__(128) void awq_tensorcore_kernel(
    const half* __restrict__ x, const half* __restrict__ residual,
    const half* __restrict__ gamma, const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ zeros, const half* __restrict__ scales,
    const half* __restrict__ bias, half* __restrict__ output,
    half* __restrict__ residual_sum, int out_dim, int hidden_dim, float eps) {
  using namespace nvcuda;
  constexpr int kWarps = 4;
  constexpr int kChunkK = 128;
  constexpr int kColumns = 64;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int groups = hidden_dim / kChunkK;
  const int cta = blockIdx.x;
  const int output_begin = cta * kColumns;

  __shared__ __align__(32) half tile_a[16 * kChunkK];
  __shared__ __align__(32) half tile_b[kWarps][kChunkK * 16];
  __shared__ float inverse_rms[Batch];
  __shared__ __align__(32) float tile_c[kWarps][16 * 16];

  for (int batch = warp; batch < Batch; batch += kWarps) {
    float square_sum = 0.0f;
    for (int k = lane; k < hidden_dim; k += 32) {
      half value = x[batch * hidden_dim + k];
      if (residual != nullptr) {
        value = __hadd(value, residual[batch * hidden_dim + k]);
      }
      const float converted = __half2float(value);
      square_sum = fmaf(converted, converted, square_sum);
    }
    for (int offset = 16; offset; offset >>= 1) {
      square_sum += __shfl_down_sync(0xffffffff, square_sum, offset);
    }
    if (lane == 0) {
      inverse_rms[batch] =
          rsqrtf(square_sum / static_cast<float>(hidden_dim) + eps);
    }
  }
  __syncthreads();

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  for (int group = 0; group < groups; ++group) {
    for (int index = threadIdx.x; index < 16 * kChunkK;
         index += blockDim.x) {
      const int row = index / kChunkK;
      const int local_k = index % kChunkK;
      half staged = __float2half(0.0f);
      if (row < Batch) {
        const int k = group * kChunkK + local_k;
        half value = x[row * hidden_dim + k];
        if (residual != nullptr) {
          value = __hadd(value, residual[row * hidden_dim + k]);
          if (cta == 0) residual_sum[row * hidden_dim + k] = value;
        }
        const half normalized = __float2half_rn(
            __half2float(value) * inverse_rms[row]);
        staged = __hmul(normalized, gamma[k]);
      }
      tile_a[index] = staged;
    }

    const int packed_tile_offset =
        ((cta * kWarps + warp) * groups + group) * (kChunkK * 8);
    const int parameter_offset =
        ((cta * kWarps + warp) * groups + group) * 16;
    for (int packed_index = lane; packed_index < kChunkK * 8;
         packed_index += 32) {
      const uint8_t packed = weight[packed_tile_offset + packed_index];
      const int local_k = packed_index / 8;
      const int pair = packed_index % 8;
      const int column0 = pair * 2;
      const int quantized0 = packed & 0xF;
      const int quantized1 = packed >> 4;
      const int zero0 = zeros[parameter_offset + column0];
      const int zero1 = zeros[parameter_offset + column0 + 1];
      tile_b[warp][local_k * 16 + column0] = __float2half_rn(
          static_cast<float>(quantized0 - zero0) *
          __half2float(scales[parameter_offset + column0]));
      tile_b[warp][local_k * 16 + column0 + 1] = __float2half_rn(
          static_cast<float>(quantized1 - zero1) *
          __half2float(scales[parameter_offset + column0 + 1]));
    }
    __syncthreads();
    for (int k_offset = 0; k_offset < kChunkK; k_offset += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
          a_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
          b_fragment;
      wmma::load_matrix_sync(a_fragment, tile_a + k_offset, kChunkK);
      wmma::load_matrix_sync(
          b_fragment, tile_b[warp] + k_offset * 16, 16);
      wmma::mma_sync(
          accumulator, a_fragment, b_fragment, accumulator);
    }
    __syncthreads();
  }
  wmma::store_matrix_sync(
      tile_c[warp], accumulator, 16, wmma::mem_row_major);
  __syncwarp();
  for (int index = lane; index < Batch * 16; index += 32) {
    const int batch = index / 16;
    const int local_column = index % 16;
    const int column = output_begin + warp * 16 + local_column;
    if (column < out_dim) {
      float value = tile_c[warp][batch * 16 + local_column];
      if (bias != nullptr) value += __half2float(bias[column]);
      output[batch * out_dim + column] = __float2half_rn(value);
    }
  }
}

__global__ __launch_bounds__(256) void awq_repacked_gemv_bs1_kernel(
    const half* __restrict__ x, const half* __restrict__ residual,
    const half* __restrict__ gamma, const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ zeros, const half* __restrict__ scales,
    const half* __restrict__ bias, half* __restrict__ output,
    half* __restrict__ residual_sum, int out_dim, int hidden_dim,
    int group_size, float eps) {
  constexpr int kWarps = 8;
  constexpr int kColumnsPerWarp = 2;
  constexpr int kOutputsPerBlock = kWarps * kColumnsPerWarp;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int column0 = blockIdx.x * kOutputsPerBlock + warp;
  const int column1 = column0 + kWarps;
  const bool valid0 = column0 < out_dim;
  const bool valid1 = column1 < out_dim;
  float accumulator0[4] = {};
  float accumulator1[4] = {};
  float local_square = 0.0f;
  const int packed_stride = hidden_dim / 2;
  const int groups = hidden_dim / group_size;
  __shared__ half activation[4096];
  // One partial per warp is sufficient.  The former 256-way shared-memory
  // tree inserted nine CTA barriers in every output tile; in decode this
  // launch/synchronization cost is material relative to the W4 dot product.
  __shared__ float square_partial[kWarps];
  __shared__ float inverse_rms;
  __shared__ float raw_output[kWarps][kColumnsPerWarp];
  for (int index = threadIdx.x; index < hidden_dim; index += blockDim.x) {
    half value = x[index];
    if (residual != nullptr) value = __hadd(value, residual[index]);
    const float value_float = __half2float(value);
    local_square = fmaf(value_float, value_float, local_square);
    activation[index] = __hmul(value, gamma[index]);
    if (blockIdx.x == 0 && residual_sum != nullptr) {
      residual_sum[index] = value;
    }
  }
  __syncthreads();
  for (int group = 0; group < groups; ++group) {
    int zero0 = 0;
    int zero1 = 0;
    float scale0 = 0.0f;
    float scale1 = 0.0f;
    if (lane == 0) {
      if (valid0) {
        zero0 = zeros[group * out_dim + column0];
        scale0 = __half2float(scales[group * out_dim + column0]);
      }
      if (valid1) {
        zero1 = zeros[group * out_dim + column1];
        scale1 = __half2float(scales[group * out_dim + column1]);
      }
    }
    zero0 = __shfl_sync(0xffffffff, zero0, 0);
    zero1 = __shfl_sync(0xffffffff, zero1, 0);
    scale0 = __shfl_sync(0xffffffff, scale0, 0);
    scale1 = __shfl_sync(0xffffffff, scale1, 0);
    for (int group_offset = lane * 4; group_offset < group_size;
         group_offset += 128) {
      const int k = group * group_size + group_offset;
      const uint16_t packed0 = valid0
          ? *reinterpret_cast<const uint16_t*>(
                weight + column0 * packed_stride + k / 2)
          : 0;
      const uint16_t packed1 = valid1
          ? *reinterpret_cast<const uint16_t*>(
                weight + column1 * packed_stride + k / 2)
          : 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float activation_value = __half2float(
            activation[group * group_size + group_offset + i]);
        accumulator0[i] = fmaf(
            activation_value,
            static_cast<float>(((packed0 >> (i * 4)) & 0xF) - zero0) * scale0,
            accumulator0[i]);
        accumulator1[i] = fmaf(
            activation_value,
            static_cast<float>(((packed1 >> (i * 4)) & 0xF) - zero1) * scale1,
            accumulator1[i]);
      }
    }
  }
#pragma unroll
  for (int i = 1; i < 4; ++i) {
    accumulator0[0] += accumulator0[i];
    accumulator1[0] += accumulator1[i];
  }
  for (int offset = 16; offset; offset >>= 1) {
    accumulator0[0] +=
        __shfl_down_sync(0xffffffff, accumulator0[0], offset);
    accumulator1[0] +=
        __shfl_down_sync(0xffffffff, accumulator1[0], offset);
  }
  if (lane == 0) {
    raw_output[warp][0] = accumulator0[0];
    raw_output[warp][1] = accumulator1[0];
  }
  for (int offset = 16; offset; offset >>= 1) {
    local_square += __shfl_down_sync(0xffffffff, local_square, offset);
  }
  if (lane == 0) square_partial[warp] = local_square;
  __syncthreads();
  if (warp == 0) {
    float total = lane < kWarps ? square_partial[lane] : 0.0f;
    for (int offset = 16; offset; offset >>= 1) {
      total += __shfl_down_sync(0xffffffff, total, offset);
    }
    if (lane == 0) {
      inverse_rms = rsqrtf(total / static_cast<float>(hidden_dim) + eps);
    }
  }
  __syncthreads();
  if (lane == 0) {
    if (valid0) {
      float value = raw_output[warp][0] * inverse_rms;
      if (bias != nullptr) value += __half2float(bias[column0]);
      output[column0] = __float2half_rn(value);
    }
    if (valid1) {
      float value = raw_output[warp][1] * inverse_rms;
      if (bias != nullptr) value += __half2float(bias[column1]);
      output[column1] = __float2half_rn(value);
    }
  }
}

__device__ __forceinline__ int awq_shift(int index) {
  // AutoAWQ packs logical columns in [0, 4, 1, 5, 2, 6, 3, 7]
  // nibble order inside each int32.
  return ((index & 1) * 4 + (index >> 1)) * 4;
}

__global__ __launch_bounds__(kAwqThreads) void awq_fused_gemv_bs1_kernel(
    const half* __restrict__ x, const half* __restrict__ residual,
    const half* __restrict__ gamma, const int32_t* __restrict__ qweight,
    const int32_t* __restrict__ qzeros, const half* __restrict__ scales,
    const half* __restrict__ bias, half* __restrict__ output,
    half* __restrict__ residual_sum, int out_dim, int hidden_dim,
    int group_size, float eps) {
  __shared__ float reduction[kAwqThreads];
  __shared__ float partial[kAwqWarps][kAwqOutputsPerBlock];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float local_square = 0.0f;
  for (int k = threadIdx.x; k < hidden_dim; k += kAwqThreads) {
    half value = x[k];
    if (residual != nullptr) value = __hadd(value, residual[k]);
    const float value_float = __half2float(value);
    local_square = fmaf(value_float, value_float, local_square);
    if (blockIdx.x == 0 && residual_sum != nullptr) residual_sum[k] = value;
  }
  reduction[threadIdx.x] = local_square;
  __syncthreads();
  for (int offset = kAwqThreads / 2; offset >= 1; offset >>= 1) {
    if (threadIdx.x < offset) {
      reduction[threadIdx.x] += reduction[threadIdx.x + offset];
    }
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf(reduction[0] / static_cast<float>(hidden_dim) + eps);

  float accumulator[8] = {};
  const int packed_out_dim = out_dim / 8;
  const bool active_column = lane < kAwqPackedColumnsPerBlock;
  const int packed_column =
      blockIdx.x * kAwqPackedColumnsPerBlock + lane;
  const int column_base = blockIdx.x * kAwqOutputsPerBlock + lane * 8;
  const int chunk = hidden_dim / kAwqWarps;
  const int k_begin = warp * chunk;
  const int k_end = k_begin + chunk;
  int current_group = -1;
  uint32_t packed_zero = 0;
  half scale_values[8] = {};

  for (int k = k_begin; k < k_end; ++k) {
    const int group = k / group_size;
    if (active_column && group != current_group) {
      current_group = group;
      packed_zero = static_cast<uint32_t>(
          qzeros[group * packed_out_dim + packed_column]);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        scale_values[i] = scales[group * out_dim + column_base + i];
      }
    }
    half value = x[k];
    if (residual != nullptr) value = __hadd(value, residual[k]);
    value = __float2half_rn(__half2float(value) * inverse_rms);
    value = __hmul(value, gamma[k]);
    const float activation = __half2float(value);
    const uint32_t packed_weight = active_column
        ? static_cast<uint32_t>(qweight[k * packed_out_dim + packed_column])
        : 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int shift = awq_shift(i);
      const int quantized = (packed_weight >> shift) & 0xF;
      const int zero = (packed_zero >> shift) & 0xF;
      const float weight =
          static_cast<float>(quantized - zero) * __half2float(scale_values[i]);
      if (active_column) {
        accumulator[i] = fmaf(activation, weight, accumulator[i]);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (active_column) partial[warp][lane * 8 + i] = accumulator[i];
  }
  __syncthreads();
  if (warp == 0 && active_column) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float value = partial[0][lane * 8 + i];
#pragma unroll
      for (int source_warp = 1; source_warp < kAwqWarps; ++source_warp) {
        value += partial[source_warp][lane * 8 + i];
      }
      const int column = column_base + i;
      if (bias != nullptr) value += __half2float(bias[column]);
      output[column] = __float2half_rn(value);
    }
  }
}

__global__ void preprocess_kernel(
    const half* __restrict__ x, const half* __restrict__ residual,
    const half* __restrict__ gamma, half* __restrict__ augmented,
    half* __restrict__ residual_sum, int batch, int hidden_dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * hidden_dim;
  if (index >= total) return;
  const int k = index % hidden_dim;
  half value = x[index];
  if (residual != nullptr) {
    value = __hadd(value, residual[index]);
    residual_sum[index] = value;
  }
  augmented[index] = __hmul(value, gamma[k]);
  augmented[total + index] = __hmul(value, value);
}

__global__ void finalize_kernel(
    const half* __restrict__ gemm_output, const half* __restrict__ bias,
    half* __restrict__ output, int batch, int out_dim, int padded_out_dim,
    int hidden_dim, float eps) {
  __shared__ float inverse_rms;
  const int row = blockIdx.y;
  if (threadIdx.x == 0) {
    const float square_sum = __half2float(
        gemm_output[(batch + row) * padded_out_dim + out_dim]);
    inverse_rms = rsqrtf(square_sum / static_cast<float>(hidden_dim) + eps);
  }
  __syncthreads();
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  const int pair_count = out_dim / 2;
  if (pair < pair_count) {
    const half2 value = reinterpret_cast<const half2*>(
        gemm_output + row * padded_out_dim)[pair];
    float2 converted = __half22float2(value);
    converted.x *= inverse_rms;
    converted.y *= inverse_rms;
    if (bias != nullptr) {
      const float2 bias_pair =
          __half22float2(reinterpret_cast<const half2*>(bias)[pair]);
      converted.x += bias_pair.x;
      converted.y += bias_pair.y;
    }
    reinterpret_cast<half2*>(output + row * out_dim)[pair] =
        __floats2half2_rn(converted.x, converted.y);
  }
}

}  // namespace

void launch_w4a16_awq_fused_gemv_bs1(
    const at::Half* x, const at::Half* residual, const at::Half* gamma,
    const int32_t* qweight, const int32_t* qzeros, const at::Half* scales,
    const at::Half* bias, at::Half* output, at::Half* residual_sum,
    int out_dim, int hidden_dim, int group_size, float eps,
    cudaStream_t stream) {
  awq_fused_gemv_bs1_kernel<<<out_dim / kAwqOutputsPerBlock,
                              kAwqThreads, 0, stream>>>(
      reinterpret_cast<const half*>(x),
      reinterpret_cast<const half*>(residual),
      reinterpret_cast<const half*>(gamma), qweight, qzeros,
      reinterpret_cast<const half*>(scales),
      reinterpret_cast<const half*>(bias), reinterpret_cast<half*>(output),
      reinterpret_cast<half*>(residual_sum), out_dim, hidden_dim, group_size,
      eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_w4a16_awq_repacked_gemv_bs1(
    const at::Half* x, const at::Half* residual, const at::Half* gamma,
    const uint8_t* weight, const uint8_t* zeros, const at::Half* scales,
    const at::Half* bias, at::Half* output, at::Half* residual_sum,
    int out_dim, int hidden_dim, int group_size, float eps,
    cudaStream_t stream) {
  awq_repacked_gemv_bs1_kernel<<<(out_dim + 15) / 16, 256, 0, stream>>>(
      reinterpret_cast<const half*>(x),
      reinterpret_cast<const half*>(residual),
      reinterpret_cast<const half*>(gamma), weight, zeros,
      reinterpret_cast<const half*>(scales),
      reinterpret_cast<const half*>(bias), reinterpret_cast<half*>(output),
      reinterpret_cast<half*>(residual_sum), out_dim, hidden_dim, group_size,
      eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_w4a16_awq_tensorcore(
    const at::Half* x, const at::Half* residual, const at::Half* gamma,
    const uint8_t* weight, const uint8_t* zeros, const at::Half* scales,
    const at::Half* bias, at::Half* output, at::Half* residual_sum,
    int batch, int out_dim, int hidden_dim, float eps, cudaStream_t stream) {
  const dim3 grid((out_dim + 63) / 64);
#define LAUNCH_AWQ_TC(BATCH)                                             \
  awq_tensorcore_kernel<BATCH><<<grid, 128, 0, stream>>>(                \
      reinterpret_cast<const half*>(x),                                 \
      reinterpret_cast<const half*>(residual),                          \
      reinterpret_cast<const half*>(gamma), weight, zeros,              \
      reinterpret_cast<const half*>(scales),                            \
      reinterpret_cast<const half*>(bias), reinterpret_cast<half*>(output), \
      reinterpret_cast<half*>(residual_sum), out_dim, hidden_dim, eps)
  switch (batch) {
    case 1: LAUNCH_AWQ_TC(1); break;
    case 4: LAUNCH_AWQ_TC(4); break;
    case 8: LAUNCH_AWQ_TC(8); break;
  }
#undef LAUNCH_AWQ_TC
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_w4a16_sentinel_preprocess(
    const at::Half* x, const at::Half* residual, const at::Half* gamma,
    at::Half* augmented, at::Half* residual_sum, int batch, int hidden_dim,
    cudaStream_t stream) {
  const int total = batch * hidden_dim;
  preprocess_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const half*>(x),
      reinterpret_cast<const half*>(residual),
      reinterpret_cast<const half*>(gamma),
      reinterpret_cast<half*>(augmented),
      reinterpret_cast<half*>(residual_sum), batch, hidden_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_w4a16_sentinel_finalize(
    const at::Half* gemm_output, const at::Half* bias, at::Half* output,
    int batch, int out_dim, int padded_out_dim, int hidden_dim, float eps,
    cudaStream_t stream) {
  const dim3 grid((out_dim / 2 + 255) / 256, batch);
  finalize_kernel<<<grid, 256, 0, stream>>>(
      reinterpret_cast<const half*>(gemm_output),
      reinterpret_cast<const half*>(bias), reinterpret_cast<half*>(output),
      batch, out_dim, padded_out_dim, hidden_dim, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
