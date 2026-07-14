#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>

namespace nonlinear {

using namespace cute;

constexpr int kBlockM = 16;
constexpr int kBlockN = 64;
constexpr int kBlockK = 64;
constexpr int kThreads = 64;

struct alignas(16) SharedStorage {
  half a[kBlockM * kBlockK];
  half b[kBlockN * kBlockK];
};

constexpr int kSimtWarps = 8;
constexpr int kSimtThreads = kSimtWarps * 32;

template <int Batch>
__global__ __launch_bounds__(kSimtThreads) void fused_square_linear_simt_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int out_dim,
    int hidden_dim) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int output_index = blockIdx.x * kSimtWarps + warp;
  if (output_index >= out_dim) {
    return;
  }

  float accumulator[Batch][2] = {};
  const half* weight_row = weight + output_index * hidden_dim;
  if ((hidden_dim & 1) == 0) {
    const int pair_count = hidden_dim / 2;
    for (int pair_index = lane; pair_index < pair_count; pair_index += 32) {
      const half2 weight_value =
          reinterpret_cast<const half2*>(weight_row)[pair_index];
      const float2 weight_float = __half22float2(weight_value);
#pragma unroll
      for (int batch_index = 0; batch_index < Batch; ++batch_index) {
        const half2 x_value = reinterpret_cast<const half2*>(
            x + batch_index * hidden_dim)[pair_index];
        const half2 squared_value = __hmul2(x_value, x_value);
        const float2 x_float = __half22float2(x_value);
        const float2 squared_float = __half22float2(squared_value);
        accumulator[batch_index][0] = fmaf(
            x_float.x, weight_float.x, accumulator[batch_index][0]);
        accumulator[batch_index][0] = fmaf(
            x_float.y, weight_float.y, accumulator[batch_index][0]);
        accumulator[batch_index][1] = fmaf(
            squared_float.x, weight_float.x, accumulator[batch_index][1]);
        accumulator[batch_index][1] = fmaf(
            squared_float.y, weight_float.y, accumulator[batch_index][1]);
      }
    }
  } else {
    for (int k_index = lane; k_index < hidden_dim; k_index += 32) {
      const float weight_value = __half2float(weight_row[k_index]);
#pragma unroll
      for (int batch_index = 0; batch_index < Batch; ++batch_index) {
        const half x_value = x[batch_index * hidden_dim + k_index];
        const float x_float = __half2float(x_value);
        const float squared_float = __half2float(__hmul(x_value, x_value));
        accumulator[batch_index][0] =
            fmaf(x_float, weight_value, accumulator[batch_index][0]);
        accumulator[batch_index][1] =
            fmaf(squared_float, weight_value, accumulator[batch_index][1]);
      }
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      accumulator[batch_index][0] += __shfl_down_sync(
          0xffffffff, accumulator[batch_index][0], offset);
      accumulator[batch_index][1] += __shfl_down_sync(
          0xffffffff, accumulator[batch_index][1], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int batch_index = 0; batch_index < Batch; ++batch_index) {
      output[(batch_index * 2) * out_dim + output_index] =
          __float2half(accumulator[batch_index][0]);
      output[(batch_index * 2 + 1) * out_dim + output_index] =
          __float2half(accumulator[batch_index][1]);
    }
  }
}

__global__ __launch_bounds__(kThreads) void fused_square_linear_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int batch,
    int out_dim,
    int hidden_dim) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __shared__ SharedStorage storage;

  // CuTe tensors describe both shared-memory operands in the TN convention:
  // A is [logical M, K], B is [N, K], and C is [logical M, N].
  auto s_a = make_tensor(
      make_smem_ptr(reinterpret_cast<cute::half_t*>(storage.a)),
      make_layout(make_shape(Int<kBlockM>{}, Int<kBlockK>{}), LayoutRight{}));
  auto s_b = make_tensor(
      make_smem_ptr(reinterpret_cast<cute::half_t*>(storage.b)),
      make_layout(make_shape(Int<kBlockN>{}, Int<kBlockK>{}), LayoutRight{}));

  auto tiled_mma = make_tiled_mma(
      SM80_16x8x16_F32F16F16F32_TN{},
      Layout<Shape<_1, _2, _1>, Stride<_2, _1, _0>>{},
      Tile<_16, _64, _16>{});
  auto thread_mma = tiled_mma.get_slice(threadIdx.x);
  auto register_a = thread_mma.partition_fragment_A(s_a);
  auto register_b = thread_mma.partition_fragment_B(s_b);

  auto coordinate_c = make_identity_tensor(
      make_shape(Int<kBlockM>{}, Int<kBlockN>{}));
  auto thread_coordinate_c = thread_mma.partition_C(coordinate_c);
  auto accumulator = thread_mma.make_fragment_C(thread_coordinate_c);
  clear(accumulator);

  auto copy_a = make_tiled_copy_A(
      Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t>{}, tiled_mma);
  auto copy_b = make_tiled_copy_B(
      Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t>{}, tiled_mma);
  auto thread_copy_a = copy_a.get_slice(threadIdx.x);
  auto thread_copy_b = copy_b.get_slice(threadIdx.x);
  auto copy_source_a = thread_copy_a.partition_S(s_a);
  auto copy_dest_a = thread_copy_a.retile_D(register_a);
  auto copy_source_b = thread_copy_b.partition_S(s_b);
  auto copy_dest_b = thread_copy_b.retile_D(register_b);

  const int logical_m_base = blockIdx.y * kBlockM;
  const int output_n_base = blockIdx.x * kBlockN;
  const int valid_m_rows = min(kBlockM, 2 * batch - logical_m_base);

  // Invalid logical rows never change between K tiles. Initialize them once
  // instead of repeatedly generating 14 padded rows in the B=1 fast path.
  for (int linear = threadIdx.x; linear < kBlockM * kBlockK;
       linear += kThreads) {
    storage.a[linear] = __float2half(0.0f);
  }
  __syncthreads();

  for (int k_base = 0; k_base < hidden_dim; k_base += kBlockK) {
    const bool vector_weight_path =
        hidden_dim % 8 == 0 && k_base + kBlockK <= hidden_dim;
    if (vector_weight_path) {
      CUTE_UNROLL
      for (int segment = 0; segment < kBlockK / 8; ++segment) {
        const int local_n = threadIdx.x;
        const int global_n = output_n_base + local_n;
        const bool valid = global_n < out_dim;
        const half* source = valid
            ? weight + global_n * hidden_dim + k_base + segment * 8
            : weight;
        auto* destination = reinterpret_cast<uint128_t*>(
            storage.b + local_n * kBlockK + segment * 8);
        SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>::copy(
            *reinterpret_cast<const uint128_t*>(source), *destination, valid);
      }
      cp_async_fence();
    }

    for (int linear = threadIdx.x; linear < valid_m_rows * kBlockK;
         linear += kThreads) {
      const int local_m = linear / kBlockK;
      const int local_k = linear % kBlockK;
      const int logical_m = logical_m_base + local_m;
      const int batch_index = logical_m >> 1;
      const int branch = logical_m & 1;
      const int global_k = k_base + local_k;
      half value = __float2half(0.0f);
      if (batch_index < batch && global_k < hidden_dim) {
        value = x[batch_index * hidden_dim + global_k];
        if (branch != 0) {
          value = __hmul(value, value);
        }
      }
      storage.a[linear] = value;
    }

    if (!vector_weight_path) {
      for (int linear = threadIdx.x; linear < kBlockN * kBlockK;
           linear += kThreads) {
        const int local_n = linear / kBlockK;
        const int local_k = linear % kBlockK;
        const int global_n = output_n_base + local_n;
        const int global_k = k_base + local_k;
        storage.b[linear] =
            global_n < out_dim && global_k < hidden_dim
            ? weight[global_n * hidden_dim + global_k]
            : __float2half(0.0f);
      }
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    copy(copy_a, copy_source_a, copy_dest_a);
    copy(copy_b, copy_source_b, copy_dest_b);
    gemm(tiled_mma, register_a, register_b, accumulator);
    __syncthreads();
  }

  // CuTe's MMA partition supplies each lane's exact output coordinates.
  // Predication here supports arbitrary Batch and OutDim values.
  CUTE_UNROLL
  for (int index = 0; index < size(accumulator); ++index) {
    auto coordinate = thread_coordinate_c(index);
    const int local_m = get<0>(coordinate);
    const int local_n = get<1>(coordinate);
    const int logical_m = logical_m_base + local_m;
    const int global_n = output_n_base + local_n;
    if (logical_m < 2 * batch && global_n < out_dim) {
      const int batch_index = logical_m >> 1;
      const int branch = logical_m & 1;
      output[(batch_index * 2 + branch) * out_dim + global_n] =
          __float2half(accumulator(index));
    }
  }
#endif
}

}  // namespace nonlinear

void launch_fused_square_linear_fp16(
    const half* x,
    const half* weight,
    half* output,
    int batch,
    int out_dim,
    int hidden_dim,
    cudaStream_t stream) {
  if (batch <= 8 || batch == 16) {
    const dim3 simt_block(nonlinear::kSimtThreads);
    const dim3 simt_grid(
        (out_dim + nonlinear::kSimtWarps - 1) / nonlinear::kSimtWarps);
    switch (batch) {
      case 1:
        nonlinear::fused_square_linear_simt_kernel<1>
            <<<simt_grid, simt_block, 0, stream>>>(
                x, weight, output, out_dim, hidden_dim);
        return;
      case 2:
        nonlinear::fused_square_linear_simt_kernel<2>
            <<<simt_grid, simt_block, 0, stream>>>(
                x, weight, output, out_dim, hidden_dim);
        return;
      case 3:
        nonlinear::fused_square_linear_simt_kernel<3>
            <<<simt_grid, simt_block, 0, stream>>>(
                x, weight, output, out_dim, hidden_dim);
        return;
      case 4:
        nonlinear::fused_square_linear_simt_kernel<4>
            <<<simt_grid, simt_block, 0, stream>>>(
                x, weight, output, out_dim, hidden_dim);
        return;
      case 5:
        nonlinear::fused_square_linear_simt_kernel<5>
            <<<simt_grid, simt_block, 0, stream>>>(
                x, weight, output, out_dim, hidden_dim);
        return;
      case 6:
        nonlinear::fused_square_linear_simt_kernel<6>
            <<<simt_grid, simt_block, 0, stream>>>(
                x, weight, output, out_dim, hidden_dim);
        return;
      case 7:
        nonlinear::fused_square_linear_simt_kernel<7>
            <<<simt_grid, simt_block, 0, stream>>>(
                x, weight, output, out_dim, hidden_dim);
        return;
      case 8:
        nonlinear::fused_square_linear_simt_kernel<8>
            <<<simt_grid, simt_block, 0, stream>>>(
                x, weight, output, out_dim, hidden_dim);
        return;
      case 16:
        nonlinear::fused_square_linear_simt_kernel<16>
            <<<simt_grid, simt_block, 0, stream>>>(
                x, weight, output, out_dim, hidden_dim);
        return;
    }
  }

  const dim3 block(nonlinear::kThreads);
  const dim3 grid(
      (out_dim + nonlinear::kBlockN - 1) / nonlinear::kBlockN,
      (2 * batch + nonlinear::kBlockM - 1) / nonlinear::kBlockM);
  nonlinear::fused_square_linear_kernel<<<grid, block, 0, stream>>>(
      x, weight, output, batch, out_dim, hidden_dim);
}
