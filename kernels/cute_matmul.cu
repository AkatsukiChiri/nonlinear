#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>

namespace {

constexpr int kTileM = 16;
constexpr int kTileN = 16;

__global__ void cute_matmul_fp16_kernel(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k) {
  int row = blockIdx.y * kTileM + threadIdx.y;
  int col = blockIdx.x * kTileN + threadIdx.x;

  auto tensor_a = cute::make_tensor(
      cute::make_gmem_ptr(a),
      cute::make_shape(m, k),
      cute::make_stride(k, 1));
  auto tensor_b = cute::make_tensor(
      cute::make_gmem_ptr(b),
      cute::make_shape(k, n),
      cute::make_stride(n, 1));
  auto tensor_c = cute::make_tensor(
      cute::make_gmem_ptr(c),
      cute::make_shape(m, n),
      cute::make_stride(n, 1));

  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int idx = 0; idx < k; ++idx) {
    acc += __half2float(tensor_a(row, idx)) * __half2float(tensor_b(idx, col));
  }
  tensor_c(row, col) = __float2half(acc);
}

}  // namespace

void launch_cute_matmul_fp16(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  dim3 block(kTileN, kTileM);
  dim3 grid((n + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM);
  cute_matmul_fp16_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
}
