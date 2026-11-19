// Minimal primitives adapted from vLLM 0.8.5's Apache-2.0 Marlin sources.
//
// This local, fixed subset is the foundation for migrating the AWQ U4/FP16
// decode data path without making the fused extension depend on vLLM's source
// tree at JIT-build time.  It deliberately contains no host launcher or
// registration code: the fused kernel owns its A-side generation, launch, and
// epilogue.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace nonlinear::marlin_awq {

constexpr int kThreads = 256;
constexpr int kPipelineStages = 4;
constexpr int kTile = 16;

template <typename T, int Count>
struct Vec {
  T elements[Count];

  __device__ T& operator[](int index) { return elements[index]; }
  __device__ const T& operator[](int index) const { return elements[index]; }
};

using Int4 = Vec<int, 4>;
using FragmentA = Vec<half2, 4>;
using FragmentB = Vec<half2, 2>;
using FragmentC = Vec<float, 4>;
using FragmentScale = Vec<half2, 1>;
using FragmentZeroPoint = Vec<half2, 4>;

__device__ inline void cp_async_16(void* shared_ptr, const void* global_ptr) {
  const auto shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               : : "r"(shared_address), "l"(global_ptr));
}

__device__ inline void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int PendingGroups>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(PendingGroups));
}

template <int Lut>
__device__ inline uint32_t lop3(uint32_t first, uint32_t second,
                                uint32_t third) {
  uint32_t result;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(result)
               : "r"(first), "r"(second), "r"(third), "n"(Lut));
  return result;
}

// Exact FP16 U4 conversion used by Marlin: input nibbles are arranged as
// q0 | q2<<4 | q1<<16 | q3<<20 and output registers are MMA-ready half2s.
__device__ inline FragmentB dequantize_u4(uint32_t packed) {
  constexpr uint32_t kLowMask = 0x000f000f;
  constexpr uint32_t kHighMask = 0x00f000f0;
  constexpr uint32_t kExponent = 0x64006400;
  constexpr uint32_t kSubtract = 0x64006400;
  constexpr uint32_t kMultiply = 0x2c002c00;
  constexpr uint32_t kAdd = 0xd400d400;
  const uint32_t low = lop3<(0xf0 & 0xcc) | 0xaa>(
      packed, kLowMask, kExponent);
  const uint32_t high = lop3<(0xf0 & 0xcc) | 0xaa>(
      packed, kHighMask, kExponent);
  const half2 first = __hsub2(*reinterpret_cast<const half2*>(&low),
                              *reinterpret_cast<const half2*>(&kSubtract));
  const half2 second = __hfma2(*reinterpret_cast<const half2*>(&high),
                                *reinterpret_cast<const half2*>(&kMultiply),
                                *reinterpret_cast<const half2*>(&kAdd));
  return {{first, second}};
}

}  // namespace nonlinear::marlin_awq
