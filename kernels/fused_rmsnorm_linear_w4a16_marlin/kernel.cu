#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "marlin_awq_primitives.cuh"

namespace {

constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 16;
constexpr int kBlockK = 64;
constexpr int kWideSplitK = 4;
constexpr int kNarrowSplitK = 8;

__host__ __device__ inline int choose_split_count(int hidden_dim,
                                                   int stripe_count) {
  // A narrow N grid leaves most SMs without a resident CTA.  Double the K
  // partitioning only in that case; wide projections already have enough
  // independent stripes and avoid the extra global partial reduction.
  const int preferred_splits = stripe_count <= 82 ? kNarrowSplitK : kWideSplitK;
  if (hidden_dim >= preferred_splits * kBlockK) return preferred_splits;
  return hidden_dim >= 2 * kBlockK ? 2 : 1;
}

struct FragmentA { uint32_t values[4]; };
struct FragmentB { half2 values[2]; };
struct FragmentC { float values[4]; };

__device__ inline void cp_async_16(void* shared_ptr, const void* global_ptr) {
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               : : "r"(shared_address), "l"(global_ptr));
}

__device__ inline void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ inline void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ inline void load_a_fragment(FragmentA& fragment,
                                       const half* shared_a, int lane) {
  const uint32_t address = static_cast<uint32_t>(
      __cvta_generic_to_shared(shared_a + (lane & 15) * kMmaK +
                               (lane >> 4) * 8));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(fragment.values[0]), "=r"(fragment.values[1]),
        "=r"(fragment.values[2]), "=r"(fragment.values[3])
      : "r"(address));
}

__device__ inline void mma(const FragmentA& a, const FragmentB& b,
                           FragmentC& c) {
  const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&b.values[0]);
  const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&b.values[1]);
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(c.values[0]), "=f"(c.values[1]), "=f"(c.values[2]),
        "=f"(c.values[3])
      : "r"(a.values[0]), "r"(a.values[1]), "r"(a.values[2]),
        "r"(a.values[3]), "r"(b0), "r"(b1), "f"(c.values[0]),
        "f"(c.values[1]), "f"(c.values[2]), "f"(c.values[3]));
}

// Directly transplanted from vLLM gptq_marlin's dequant<half, kU4> path.
// q is prearranged as q0 | q2<<4 | q1<<16 | q3<<20, which yields
// half2(q0,q1), half2(q2,q3) in the exact B-register order of mma.sync.
__device__ inline FragmentB marlin_dequant_u4(uint32_t q) {
  const auto fragment = nonlinear::marlin_awq::dequantize_u4(q);
  return {fragment[0], fragment[1]};
}

// m16n8k16 B-fragment mapping used by Marlin. ``lane >> 2`` selects one N
// column; ``lane & 3`` selects two K rows, plus their counterparts eight rows
// later. The W4 values become the two half2 registers consumed by mma.sync,
// never an FP16 shared tile.
__device__ inline FragmentB load_dequantized_b_fragment(
    const uint8_t* packed, const uint8_t* zeros, const half* scales,
    int mma_n_begin, int lane) {
  const int column = mma_n_begin + (lane >> 2);
  const int zero = zeros[column];
  // Model-load packing stores q0,q2,q1,q3 in one lossless uint16 record.
  // This replaces the old four byte/nibble shared loads per fragment.
  const int n_tile = mma_n_begin >> 4;
  const int half_tile = (mma_n_begin >> 3) & 1;
  const uint16_t compact = reinterpret_cast<const uint16_t*>(packed)[
      n_tile * 64 + half_tile * 32 + lane];
  const uint32_t quantized = (static_cast<uint32_t>(compact) & 0x00ffu) |
      ((static_cast<uint32_t>(compact) & 0xff00u) << 8);
  const half2 scale_pair = __half2half2(scales[column]);
  const half2 zero_pair = __half2half2(__int2half_rn(zero));
  FragmentB fragment = marlin_dequant_u4(quantized);
  fragment.values[0] = __hmul2(__hsub2(fragment.values[0], zero_pair), scale_pair);
  fragment.values[1] = __hmul2(__hsub2(fragment.values[1], zero_pair), scale_pair);
  return fragment;
}

template <int kStripeN>
__device__ inline void store_accumulator(const FragmentC& accumulator,
                                         float* results, int row0, int column0,
                                         int active_rows) {
  if (row0 < active_rows) {
    results[row0 * kStripeN + column0] = accumulator.values[0];
    results[row0 * kStripeN + column0 + 1] = accumulator.values[1];
  }
  if (row0 + 8 < active_rows) {
    results[(row0 + 8) * kStripeN + column0] = accumulator.values[2];
    results[(row0 + 8) * kStripeN + column0 + 1] = accumulator.values[3];
  }
}

template <int kThreads, int kStripeN>
__global__ __launch_bounds__(kThreads) void w4a16_marlin_sentinel_direct_kernel(
    const half* __restrict__ input, const half* __restrict__ gamma,
    const uint8_t* __restrict__ qweight, const uint8_t* __restrict__ zeros,
    const half* __restrict__ scales, const half* __restrict__ bias,
    half* __restrict__ output, float* __restrict__ workspace,
    int* __restrict__ locks, int batch, int hidden_dim, int logical_out_dim,
    int physical_out_dim, float eps) {
  __shared__ __align__(16) half a_tile[kMmaM * kBlockK];
  // Compact 4-bit staging per K=64,N=32 block. The old WMMA bring-up
  // expanded every K=16 slice to a 4 KiB FP16 B tile and read it back through
  // ldmatrix. Four direct MMA slices now share one CTA synchronization.
  __shared__ __align__(16) uint8_t packed_b[2][kBlockK * (kStripeN / 2)];
  __shared__ __align__(16) float results[kMmaM * kStripeN];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int stripe = blockIdx.x;
  const int split = blockIdx.y;
  // The pre-tuned decode configuration keeps four slices whenever K permits.
  // On this GPU even bs=8 is latency-bound on CTA availability, not workspace
  // traffic; reducing this to two slices was measured to be slower.
  const int split_count = choose_split_count(hidden_dim,
                                             physical_out_dim / kStripeN);
  const int stripe_begin = stripe * kStripeN;
  const int active_rows = batch * 2;
  FragmentC accumulator0{{0.0f, 0.0f, 0.0f, 0.0f}};
  FragmentC accumulator1{{0.0f, 0.0f, 0.0f, 0.0f}};

  // The K/N physical layout is 16-byte aligned: a 32-column stripe starts
  // at 16 packed bytes and every K row has a multiple-of-16 byte stride.
  // Preload the first compact W4 tile; later tiles are fetched into the other
  // stage while this tile's four direct-register MMA operations execute.
  // Fragment-native blocks are contiguous [K64, stripe] records.  A K64,N
  // stripe has 2*N aligned 16-byte vectors, exactly one per thread.
  const int first_copy = threadIdx.x * 16;
  const int split_k_begin = split * (hidden_dim / split_count);
  const int split_k_end = (split + 1) * (hidden_dim / split_count);
  const int stripe_bytes = kBlockK * (kStripeN / 2);
  const int stripe_count = physical_out_dim / kStripeN;
  cp_async_16(packed_b[0] + first_copy,
              qweight + ((split_k_begin / kBlockK) * stripe_count + stripe) *
                  stripe_bytes + first_copy);
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();

  for (int k_base = split_k_begin, stage = 0; k_base < split_k_end;
       k_base += kBlockK, stage ^= 1) {
    // Only rows [0, 2*batch) contribute to valid output.  MMA still pads M
    // to 16, but its remaining rows are discarded, so eagerly constructing
    // their A values only wastes ALU instructions and shared-memory stores.
    for (int index = threadIdx.x; index < active_rows * kBlockK;
         index += kThreads) {
      const int row = index / kBlockK;
      const int k = k_base + index % kBlockK;
      const float x = __half2float(input[(row >> 1) * hidden_dim + k]);
      const float value = (row & 1) ? x * x : x * __half2float(gamma[k]);
      const int k_local = index % kBlockK;
      a_tile[(k_local / kMmaK) * (kMmaM * kMmaK) +
             row * kMmaK + (k_local % kMmaK)] = __float2half_rn(value);
    }
    // Begin the next global-to-shared W4 transfer before consuming this stage.
  // 64 threads x 16 bytes cover the complete K=64,N=32 packed tile.
    const int next_k_base = k_base + kBlockK;
    if (next_k_base < split_k_end) {
      cp_async_16(packed_b[stage ^ 1] + first_copy,
                  qweight + ((next_k_base / kBlockK) * stripe_count + stripe) *
                      stripe_bytes + first_copy);
      cp_async_commit();
    }
    __syncthreads();

    const int warp_column = warp * 16;
#pragma unroll
    for (int sub_tile = 0; sub_tile < kBlockK / kMmaK; ++sub_tile) {
      FragmentA a_fragment;
      load_a_fragment(a_fragment,
                      a_tile + sub_tile * (kMmaM * kMmaK), lane);
      const uint8_t* b_tile = packed_b[stage] +
          sub_tile * kMmaK * (kStripeN / 2);
      const int group_offset = ((k_base + sub_tile * kMmaK) >> 7) *
          physical_out_dim + stripe_begin;
      const FragmentB b0 = load_dequantized_b_fragment(
          b_tile, zeros + group_offset, scales + group_offset, warp_column, lane);
      const FragmentB b1 = load_dequantized_b_fragment(
          b_tile, zeros + group_offset, scales + group_offset,
          warp_column + kMmaN, lane);
      mma(a_fragment, b0, accumulator0);
      mma(a_fragment, b1, accumulator1);
    }
    __syncthreads();
    if (next_k_base < split_k_end) {
      cp_async_wait_all();
      __syncthreads();
    }
  }

  const int row0 = lane >> 2;
  const int column0 = warp * 16 + (lane & 3) * 2;
  store_accumulator<kStripeN>(accumulator0, results, row0, column0, active_rows);
  store_accumulator<kStripeN>(accumulator1, results, row0, column0 + kMmaN,
                               active_rows);
  __syncthreads();

  // Each split writes a disjoint FP32 partial tile. A release fence before the
  // counter increment makes the last split's in-kernel reduction safe.
  float* partial = workspace + (split * kMmaM) * physical_out_dim + stripe_begin;
  for (int index = threadIdx.x; index < active_rows * kStripeN;
       index += kThreads) {
    partial[(index / kStripeN) * physical_out_dim + index % kStripeN] =
        results[index];
  }
  __syncthreads();
  __shared__ int is_last_split;
  if (threadIdx.x == 0) {
    __threadfence();
    is_last_split = atomicAdd(locks + stripe, 1) == split_count - 1;
    // Calls on one stream are ordered. Once the final split observes every
    // partial tile, the counter can be reset for the next decode invocation;
    // this removes the separate lock-memset launch from steady state.
    if (is_last_split) locks[stripe] = 0;
  }
  __syncthreads();
  if (!is_last_split) return;

  for (int index = threadIdx.x; index < active_rows * kStripeN;
       index += kThreads) {
    const int row = index / kStripeN;
    const int local_column = index % kStripeN;
    if ((row & 1) || local_column == kStripeN - 1) continue;
    const int logical_column = blockIdx.x * (kStripeN - 1) + local_column;
    if (logical_column >= logical_out_dim) continue;
    float sum_sq = 0.0f;
    float sum = 0.0f;
    for (int part = 0; part < split_count; ++part) {
      const float* tile = workspace + (part * kMmaM) * physical_out_dim +
          stripe_begin;
      sum_sq += tile[(row + 1) * physical_out_dim + kStripeN - 1];
      sum += tile[row * physical_out_dim + local_column];
    }
    float value = sum * rsqrtf(sum_sq / hidden_dim + eps);
    if (bias != nullptr) value += __half2float(bias[logical_column]);
    output[(row >> 1) * logical_out_dim + logical_column] =
        __float2half_rn(value);
  }
}

}  // namespace

namespace {

template <int kThreads, int kStripeN>
void launch_specialized_w4a16_marlin_sentinel(
    const void* input, const void* gamma, const uint8_t* qweight,
    const uint8_t* zeros, const void* scales, const void* bias, void* output,
    void* workspace, int* locks, int batch, int hidden_dim,
    int logical_out_dim, int physical_out_dim, float eps, int split_count,
    cudaStream_t stream) {
  const dim3 grid(physical_out_dim / kStripeN, split_count);
  w4a16_marlin_sentinel_direct_kernel<kThreads, kStripeN>
      <<<grid, kThreads, 0, stream>>>(
          static_cast<const half*>(input), static_cast<const half*>(gamma),
          qweight, zeros, static_cast<const half*>(scales),
          static_cast<const half*>(bias), static_cast<half*>(output),
          static_cast<float*>(workspace), locks, batch, hidden_dim,
          logical_out_dim, physical_out_dim, eps);
}

}  // namespace

void launch_w4a16_marlin_sentinel_wmma(
    const void* input, const void* gamma, const uint8_t* qweight,
    const uint8_t* zeros, const void* scales, const void* bias, void* output,
    void* workspace, int* locks, int batch, int hidden_dim,
    int logical_out_dim, int physical_out_dim, int stripe_n, float eps,
    cudaStream_t stream) {
  const int split_count = choose_split_count(hidden_dim,
                                             physical_out_dim / stripe_n);
  if (stripe_n == 32) {
    launch_specialized_w4a16_marlin_sentinel<64, 32>(
        input, gamma, qweight, zeros, scales, bias, output, workspace, locks,
        batch, hidden_dim, logical_out_dim, physical_out_dim, eps, split_count,
        stream);
  } else if (stripe_n == 64) {
    launch_specialized_w4a16_marlin_sentinel<128, 64>(
        input, gamma, qweight, zeros, scales, bias, output, workspace, locks,
        batch, hidden_dim, logical_out_dim, physical_out_dim, eps, split_count,
        stream);
  } else if (stripe_n == 128) {
    launch_specialized_w4a16_marlin_sentinel<256, 128>(
        input, gamma, qweight, zeros, scales, bias, output, workspace, locks,
        batch, hidden_dim, logical_out_dim, physical_out_dim, eps, split_count,
        stream);
  }
}
