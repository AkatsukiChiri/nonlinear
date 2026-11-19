#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kKTile = 64;

__device__ __forceinline__ void copy_async_16(void* dst, const void* src) {
#if __CUDA_ARCH__ >= 800
  uint32_t const dst_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                   "r"(dst_address), "l"(src));
#else
  *reinterpret_cast<int4*>(dst) = *reinterpret_cast<int4 const*>(src);
#endif
}

__device__ __forceinline__ void copy_async_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void copy_async_wait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_all;\n" ::);
#endif
}

template <int TileN, int Threads>
__global__ void fused_rms_w4a16_async_decode(
    const __half* __restrict__ input, const __half* __restrict__ gamma,
    const uint8_t* __restrict__ storage, const __half* __restrict__ bias,
    __half* __restrict__ output, int batch, int logical_n, int physical_n,
    int hidden_dim, float eps) {
  static_assert(TileN == 64 || TileN == 128, "unsupported output tile");
  namespace wmma = nvcuda::wmma;

  // Two raw-W4 stages overlap the global fetch for K[t+1] with the Tensor
  // Core work for K[t]. FP16 B is materialized only in shared memory.
  __shared__ __align__(16) uint8_t packed_b[2][TileN * kKTile / 2];
  __shared__ __align__(16) __half matrix_a[16 * kKTile];
  __shared__ __align__(16) __half matrix_b[kKTile * TileN];
  __shared__ __align__(16) float accumulators[16 * TileN];
  __shared__ __half scales[TileN];
  __shared__ uint8_t zeros[TileN];

  const int tid = threadIdx.x;
  const int tile = blockIdx.x;
  const int physical_base = tile * TileN;
  const int real_base = tile * (TileN - 1);
  const int weight_bytes = hidden_dim * physical_n / 2;
  const int zero_bytes = (hidden_dim / 128) * physical_n;
  const uint8_t* const weight = storage;
  const uint8_t* const zero_ptr = storage + weight_bytes;
  const __half* const scale_ptr = reinterpret_cast<__half const*>(
      zero_ptr + zero_bytes);

  auto prefetch_weight = [&](int stage, int k_base) {
    const int copies = TileN * kKTile / (2 * 16);
    for (int copy = tid; copy < copies; copy += Threads) {
      const int column = copy / (kKTile / (2 * 16));
      const int copy_in_column = copy % (kKTile / (2 * 16));
      const int64_t global_offset =
          int64_t(physical_base + column) * (hidden_dim / 2) +
          k_base / 2 + copy_in_column * 16;
      copy_async_16(
          &packed_b[stage][column * (kKTile / 2) + copy_in_column * 16],
          weight + global_offset);
    }
    copy_async_commit();
  };

  nvcuda::wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
  const int warp = tid / 32;
  const bool computes = warp < TileN / 16;
  if (computes) {
    wmma::fill_fragment(accumulator, 0.0f);
  }

  prefetch_weight(0, 0);
  int stage = 0;
  for (int k_base = 0; k_base < hidden_dim; k_base += kKTile) {
    copy_async_wait();
    __syncthreads();

    if (tid < TileN) {
      const int64_t parameter_offset =
          int64_t(k_base / 128) * physical_n + physical_base + tid;
      zeros[tid] = zero_ptr[parameter_offset];
      scales[tid] = scale_ptr[parameter_offset];
    }
    for (int linear = tid; linear < 16 * kKTile; linear += Threads) {
      const int row = linear / kKTile;
      const int column = linear % kKTile;
      __half value = __float2half(0.0f);
      if (row < batch) {
        value = __hmul(input[row * hidden_dim + k_base + column],
                       gamma[k_base + column]);
      } else if (row < 2 * batch) {
        const __half z = input[(row - batch) * hidden_dim + k_base + column];
        value = __hmul(z, z);
      }
      matrix_a[linear] = value;
    }
    __syncthreads();

    // Dequantize from the asynchronously fetched packed bytes. Consecutive
    // threads decode consecutive half2 values, preserving coalesced reads.
    for (int pair = tid; pair < TileN * kKTile / 2; pair += Threads) {
      const int column = pair / (kKTile / 2);
      const int pair_in_column = pair % (kKTile / 2);
      const uint8_t packed = packed_b[stage][pair];
      const __half scale = scales[column];
      const __half zero = __ushort_as_half(
          static_cast<unsigned short>(0x6400 | zeros[column]));
      const __half2 zero_pair = __halves2half2(zero, zero);
      const uint32_t magic =
          (uint32_t(packed & 0x0f) | (uint32_t(packed >> 4) << 16)) |
          0x64006400u;
      union {
        uint32_t bits;
        __half2 value;
      } decoded;
      decoded.bits = magic;
      const __half2 scale_pair = __halves2half2(scale, scale);
      const __half2 value = __hmul2(__hsub2(decoded.value, zero_pair),
                                    scale_pair);
      const int k = pair_in_column * 2;
      matrix_b[column * kKTile + k] = __low2half(value);
      matrix_b[column * kKTile + k + 1] = __high2half(value);
    }
    __syncthreads();

    if (k_base + kKTile < hidden_dim) {
      prefetch_weight(stage ^ 1, k_base + kKTile);
    }

    if (computes) {
#pragma unroll
      for (int step = 0; step < kKTile / 16; ++step) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                       wmma::row_major>
            a_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                       wmma::col_major>
            b_fragment;
        wmma::load_matrix_sync(a_fragment, matrix_a + step * 16, kKTile);
        wmma::load_matrix_sync(b_fragment,
                               matrix_b + warp * 16 * kKTile + step * 16,
                               kKTile);
        wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
      }
    }
    stage ^= 1;
  }

  if (computes) {
    wmma::store_matrix_sync(accumulators + warp * 16, accumulator, TileN,
                            wmma::mem_row_major);
  }
  __syncthreads();

  const int values = batch * (TileN - 1);
  for (int linear = tid; linear < values; linear += Threads) {
    const int row = linear / (TileN - 1);
    const int local_column = linear % (TileN - 1);
    const int output_column = real_base + local_column;
    if (output_column < logical_n) {
      const float square_sum = accumulators[(row + batch) * TileN + TileN - 1];
      float value = accumulators[row * TileN + local_column] *
          rsqrtf(square_sum / float(hidden_dim) + eps);
      if (bias != nullptr) {
        value += __half2float(bias[output_column]);
      }
      output[int64_t(row) * logical_n + output_column] = __float2half(value);
    }
  }
}

}  // namespace

void launch_w4a16_async_decode(
    const at::Half* input, const at::Half* gamma, const uint8_t* storage,
    const at::Half* bias, at::Half* output, int batch, int logical_n,
    int physical_n, int hidden_dim, int cta_n, float eps,
    cudaStream_t stream) {
  dim3 grid(physical_n / cta_n, 1, 1);
  if (cta_n == 64) {
    fused_rms_w4a16_async_decode<64, 128><<<grid, 128, 0, stream>>>(
        reinterpret_cast<__half const*>(input),
        reinterpret_cast<__half const*>(gamma), storage,
        reinterpret_cast<__half const*>(bias), reinterpret_cast<__half*>(output),
        batch, logical_n, physical_n, hidden_dim, eps);
  } else if (cta_n == 128) {
    fused_rms_w4a16_async_decode<128, 256><<<grid, 256, 0, stream>>>(
        reinterpret_cast<__half const*>(input),
        reinterpret_cast<__half const*>(gamma), storage,
        reinterpret_cast<__half const*>(bias), reinterpret_cast<__half*>(output),
        batch, logical_n, physical_n, hidden_dim, eps);
  } else {
    TORCH_CHECK(false, "unsupported async decode cta_n: ", cta_n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
