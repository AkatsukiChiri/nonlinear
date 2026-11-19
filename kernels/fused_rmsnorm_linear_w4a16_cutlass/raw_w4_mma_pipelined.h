// SPDX-License-Identifier: BSD-3-Clause
//
// A deliberately narrow adaptation of CUTLASS MmaPipelined for the project's
// W4A16 decode path.  The CUTLASS tensor-op warp MMA, A pipeline, shared
// layouts, and epilogue contract remain unchanged.  Only operand B's global
// transport changes: packed W4 bytes are moved with cp.async and are expanded
// after the asynchronous copy completes.  This is an intermediate mainloop:
// it removes W4 decoding from the global-load dependency chain while retaining
// CUTLASS's FP16 B shared-memory layout for its warp iterators.

#pragma once

#include <cutlass/aligned_buffer.h>
#include <cutlass/gemm/threadblock/mma_base.h>

namespace nonlinear {
namespace threadblock {

template <typename Shape_, typename IteratorA_, typename SmemIteratorA_,
          typename IteratorB_, typename SmemIteratorB_, typename ElementC_,
          typename LayoutC_, typename Policy_, typename TransformA_>
class RawW4MmaPipelined
    : public cutlass::gemm::threadblock::MmaBase<Shape_, Policy_, 2> {
 public:
  using Base = cutlass::gemm::threadblock::MmaBase<Shape_, Policy_, 2>;
  using Shape = Shape_;
  using IteratorA = IteratorA_;
  using IteratorB = IteratorB_;
  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using Policy = Policy_;
  using TransformA = TransformA_;
  using FragmentA = typename IteratorA::Fragment;
  using FragmentB = typename IteratorB::Fragment;
  using FragmentC = typename Policy::Operator::FragmentC;
  using Operator = typename Policy::Operator;
  using WarpCount = typename Base::WarpCount;
  using ArchTag = typename Operator::ArchTag;
  static cutlass::ComplexTransform const kTransformA = Operator::kTransformA;
  static cutlass::ComplexTransform const kTransformB = Operator::kTransformB;
  static int const kStages = Base::kStages;

  static int constexpr kRawBytesPerStage = Shape::kK * Shape::kN / 2;
  static int constexpr kRawStages = 3;

  class SharedStorage : public Base::SharedStorage {
   public:
    // The FP16 CUTLASS operand remains double-buffered. Packed W4 has a
    // separate three-stage queue: it is small (2 KiB for 64x64) and lets the
    // next-next global load overlap both W4 expansion and tensor-core work.
    cutlass::AlignedBuffer<uint8_t, kRawStages * kRawBytesPerStage, 16>
        packed_b;
  };

 protected:
  Operator warp_mma_;
  SmemIteratorA smem_iterator_a_;
  SmemIteratorB smem_iterator_b_;
  TransformA transform_a_;
  SharedStorage& shared_;
  int smem_write_stage_idx_;

  CUTLASS_DEVICE void advance_smem_write_stage() {
    ++smem_iterator_a_;
    ++smem_iterator_b_;
    if (smem_write_stage_idx_ == 1) {
      smem_iterator_a_.add_tile_offset({0, -Base::kStages});
      smem_iterator_b_.add_tile_offset({-Base::kStages, 0});
    }
    smem_write_stage_idx_ ^= 1;
  }

  CUTLASS_DEVICE void advance_smem_stages() {
    ++smem_iterator_a_;
    ++smem_iterator_b_;
    if (smem_write_stage_idx_ == 1) {
      smem_iterator_a_.add_tile_offset({0, -Base::kStages});
      smem_iterator_b_.add_tile_offset({-Base::kStages, 0});
    } else {
      this->warp_tile_iterator_A_.add_tile_offset(
          {0, -Base::kStages * Policy::kPartitionsK *
                  Base::kWarpGemmIterations});
      this->warp_tile_iterator_B_.add_tile_offset(
          {-Base::kStages * Policy::kPartitionsK *
                   Base::kWarpGemmIterations,
           0});
    }
    smem_write_stage_idx_ ^= 1;
  }

  CUTLASS_DEVICE uint8_t* raw_stage(int stage) {
    return shared_.packed_b.data() + (stage % kRawStages) * kRawBytesPerStage;
  }

 public:
  CUTLASS_DEVICE RawW4MmaPipelined(
      SharedStorage& shared_storage, int thread_idx, int warp_idx,
      int lane_idx, TransformA transform_a = TransformA())
      : Base(static_cast<typename Base::SharedStorage&>(shared_storage),
             thread_idx, warp_idx, lane_idx),
        warp_mma_(),
        smem_iterator_a_(shared_storage.operand_A_ref(), thread_idx),
        smem_iterator_b_(shared_storage.operand_B_ref(), thread_idx),
        transform_a_(transform_a), shared_(shared_storage),
        smem_write_stage_idx_(0) {
    int const warp_idx_mn =
        warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int const warp_idx_k =
        warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);
    int const warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int const warp_idx_n = warp_idx_mn / Base::WarpCount::kM;
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

  CUTLASS_DEVICE void operator()(int gemm_k_iterations, FragmentC& accum,
                                 IteratorA iterator_a, IteratorB iterator_b,
                                 FragmentC const& src_accum) {
    using WarpFragmentA = typename Operator::FragmentA;
    using WarpFragmentB = typename Operator::FragmentB;

    // CUTLASS's ordinary prologue, except B stays packed through cp.async.
    // Preload B[0] and B[1] into a raw three-stage queue. The FP16 shared
    // layout remains exactly the stock two-stage CUTLASS layout.
    FragmentA first_a;
    first_a.clear();
    iterator_a.load(first_a);
    ++iterator_a;
    IteratorB staged_b0 = iterator_b;
    staged_b0.copy_raw_tile_async(raw_stage(0), threadIdx.x);
    ++iterator_b;
    IteratorB next_decode_b;
    bool const has_second_tile = gemm_k_iterations > 1;
    if (has_second_tile) {
      next_decode_b = iterator_b;
      next_decode_b.copy_raw_tile_async(raw_stage(1), threadIdx.x);
      ++iterator_b;
      IteratorB::template wait_for_raw_tile<1>();
    } else {
      IteratorB::template wait_for_raw_tile<0>();
    }
    __syncthreads();
    FragmentB first_b;
    first_b.clear();
    staged_b0.load_from_raw_tile(first_b, raw_stage(0));
    smem_iterator_a_.store(transform_a_(first_a));
    smem_iterator_b_.store(first_b);
    advance_smem_write_stage();
    __syncthreads();

    accum = src_accum;
    WarpFragmentA warp_a[2];
    WarpFragmentB warp_b[2];
    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(warp_a[0]);
    ++this->warp_tile_iterator_A_;
    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(warp_b[0]);
    ++this->warp_tile_iterator_B_;

    iterator_a.clear_mask(gemm_k_iterations <= 1);
    iterator_b.clear_mask(gemm_k_iterations <= 2);
    int raw_read_stage = 0;
    int raw_write_stage = 2;
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      IteratorB prefetched_b;
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
           ++warp_mma_k) {
        if (warp_mma_k == Base::kWarpGemmIterations - 1 &&
            gemm_k_iterations > 1) {
          smem_iterator_a_.store(transform_a_(pending_a_));
          // With B[i+2] in flight, wait only until B[i+1] is complete. On
          // the tail there is no newer group, so wait for all outstanding
          // copies to make B[i+1] visible before W4 expansion.
          if (gemm_k_iterations > 2) {
            IteratorB::template wait_for_raw_tile<1>();
          } else {
            IteratorB::template wait_for_raw_tile<0>();
          }
          __syncthreads();
          FragmentB pending_b_fragment;
          pending_b_fragment.clear();
          int const next_raw_stage = (raw_read_stage + 1) % kRawStages;
          next_decode_b.load_from_raw_tile(pending_b_fragment,
                                           raw_stage(next_raw_stage));
          smem_iterator_b_.store(pending_b_fragment);
          __syncthreads();
          advance_smem_stages();
          raw_read_stage = next_raw_stage;
          if (gemm_k_iterations > 2) {
            next_decode_b = prefetched_b;
          }
        }

        this->warp_tile_iterator_A_.set_kgroup_index(
            (warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index(
            (warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_A_.load(
            warp_a[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load(
            warp_b[(warp_mma_k + 1) % 2]);
        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        if (warp_mma_k == 0 && gemm_k_iterations > 1) {
          pending_a_.clear();
          iterator_a.load(pending_a_);
          ++iterator_a;
          // B[i+1] was bootstrapped in the prologue. Start B[i+2] while
          // tensor cores consume B[i], leaving it pending across the next
          // W4 expansion.
          if (gemm_k_iterations > 2) {
            prefetched_b = iterator_b;
            prefetched_b.copy_raw_tile_async(raw_stage(raw_write_stage),
                                              threadIdx.x);
            ++iterator_b;
            raw_write_stage = (raw_write_stage + 1) % kRawStages;
          }
          iterator_a.clear_mask(gemm_k_iterations <= 2);
          iterator_b.clear_mask(gemm_k_iterations <= 3);
        }
        warp_mma_(accum, warp_a[warp_mma_k % 2], warp_b[warp_mma_k % 2],
                  accum);
      }
    }
  }

 private:
  // Keeping this as a class member avoids allocating another large fragment
  // per warp-MMA iteration and matches the lifetime in CUTLASS MmaPipelined.
  FragmentA pending_a_;
};

}  // namespace threadblock
}  // namespace nonlinear
