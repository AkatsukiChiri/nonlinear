#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include <cutlass/device_kernel.h>
#include <cutlass/gemm/device/gemm_universal_base.h>
#include <cutlass/gemm/kernel/gemm_universal.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm80.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>

#include "gemm_with_epilogue_visitor.h"
#include "raw_w4_mma_pipelined.h"
#include "warp_local_raw_w4_mma.h"

namespace {

// CUTLASS's stock SM80 builder makes global and shared B use the same type.
// W4A16 instead retains uint4 in global memory and expands it to fp16 through
// MmaPipelined's TransformB hook immediately before the shared-memory store.
using ElementA = cutlass::half_t;
using ElementGlobalB = cutlass::uint4b_t;
using ElementMmaB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;
using ThreadblockShape = cutlass::gemm::GemmShape<16, 64, 64>;
using WarpShape = cutlass::gemm::GemmShape<16, 16, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
    ThreadblockShape, WarpShape, InstructionShape,
    ElementA, cutlass::layout::RowMajor,
    ElementMmaB, cutlass::layout::ColumnMajor,
    ElementAccumulator, cutlass::layout::RowMajor,
    cutlass::arch::OpClassTensorOp, 2, cutlass::arch::OpMultiplyAdd>;

class RmsInputIteratorA {
 public:
  using Element = ElementA;
  using Layout = cutlass::layout::RowMajor;
  using TensorCoord = cutlass::MatrixCoord;
  using Index = int;
  using LongIndex = int64_t;
  using ThreadMap = typename MmaCore::IteratorThreadMapA;
  using AccessType = cutlass::AlignedArray<ElementA, 8>;
  using Fragment = cutlass::Array<
      ElementA,
      ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
  struct Mask {};

  struct Params {
    int64_t stride;
    ElementA const* gamma;
    int batch;
    CUTLASS_HOST_DEVICE Params() : stride(0), gamma(nullptr), batch(0) {}
    CUTLASS_HOST_DEVICE explicit Params(int64_t stride_)
        : stride(stride_), gamma(nullptr), batch(0) {}
    CUTLASS_HOST_DEVICE explicit Params(Layout const& layout)
        : stride(layout.stride(0)), gamma(nullptr), batch(0) {}
  };

 private:
  ElementA const* pointer_;
  ElementA const* gamma_;
  int64_t stride_;
  int extent_k_;
  int extent_m_;
  int batch_;
  int base_k_;
  int base_m_;
  bool mask_;

 public:
  CUTLASS_HOST_DEVICE RmsInputIteratorA() = default;
  CUTLASS_HOST_DEVICE RmsInputIteratorA(
      Params const& params, ElementA* pointer, TensorCoord extent,
      int thread_id, TensorCoord const& threadblock_offset,
      int const* = nullptr)
      : pointer_(pointer), gamma_(params.gamma), stride_(params.stride),
        extent_k_(extent.column()), extent_m_(extent.row()),
        batch_(params.batch), mask_(true) {
    auto thread_offset = ThreadMap::initial_offset(thread_id);
    base_k_ = threadblock_offset.column() + thread_offset.contiguous();
    base_m_ = threadblock_offset.row() + thread_offset.strided();
  }
  CUTLASS_HOST_DEVICE RmsInputIteratorA(
      Params const& params, ElementA* pointer, TensorCoord extent,
      int thread_id)
      : RmsInputIteratorA(params, pointer, extent, thread_id,
                          TensorCoord(0, 0)) {}
  CUTLASS_HOST_DEVICE RmsInputIteratorA& operator++() {
    base_k_ += ThreadblockShape::kK;
    return *this;
  }
  CUTLASS_HOST_DEVICE void clear_mask(bool enable = true) {
    if (enable) mask_ = false;
  }
  CUTLASS_HOST_DEVICE void enable_mask() { mask_ = true; }
  CUTLASS_HOST_DEVICE void set_mask(Mask const&) { mask_ = true; }
  CUTLASS_HOST_DEVICE void get_mask(Mask&) const {}
  CUTLASS_HOST_DEVICE void add_pointer_offset(LongIndex offset) {
    pointer_ += offset;
  }

  CUTLASS_DEVICE void load(Fragment& fragment) const {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < ThreadMap::kElementsPerAccess; ++e) {
          int const fragment_index =
              e + ThreadMap::kElementsPerAccess *
                      (c + s * ThreadMap::Iterations::kContiguous);
          int const k = base_k_ + c * ThreadMap::Delta::kContiguous + e;
          int const row = base_m_ + s * ThreadMap::Delta::kStrided;
          ElementA value(0.0f);
          if (mask_ && k < extent_k_ && row < extent_m_) {
            if (gamma_) {
              int const batch_row = row < batch_ ? row : row - batch_;
              ElementA const z = pointer_[int64_t(batch_row) * stride_ + k];
              value = row < batch_ ? ElementA(float(z) * float(gamma_[k]))
                                   : ElementA(float(z) * float(z));
            } else {
              value = pointer_[int64_t(row) * stride_ + k];
            }
          }
          fragment[fragment_index] = value;
        }
      }
    }
  }

};

using IteratorA = RmsInputIteratorA;

// Reuse the fp16 core's logical thread map. Each access covers the same eight
// logical coordinates, while the uint4 access transfers four packed bytes.
class PackedW4IteratorB {
 public:
  using Element = ElementGlobalB;
  using Layout = cutlass::layout::ColumnMajor;
  using TensorCoord = cutlass::MatrixCoord;
  using Index = int;
  using LongIndex = int64_t;
  using ThreadMap = typename MmaCore::IteratorThreadMapB;
  using AccessType = cutlass::AlignedArray<ElementGlobalB, 8, 4>;
  using Fragment = cutlass::Array<
      ElementMmaB,
      ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  struct Mask {};

  struct Params {
    int64_t stride;
    CUTLASS_HOST_DEVICE Params() : stride(0) {}
    CUTLASS_HOST_DEVICE explicit Params(int64_t stride_) : stride(stride_) {}
    CUTLASS_HOST_DEVICE explicit Params(Layout const& layout)
        : stride(layout.stride(0)) {}
  };

 private:
  uint8_t const* pointer_;
  int64_t stride_;
  int extent_k_;
  int extent_n_;
  uint8_t const* zeros_;
  ElementMmaB const* scales_;
  int base_k_;
  int base_n_;
  int tile_base_k_;
  int tile_base_n_;
  bool mask_;

 public:
  CUTLASS_HOST_DEVICE PackedW4IteratorB() = default;

  CUTLASS_HOST_DEVICE PackedW4IteratorB(
      Params const& params, ElementGlobalB* pointer, TensorCoord extent,
      int thread_id, TensorCoord const& threadblock_offset,
      int const* = nullptr)
      : pointer_(reinterpret_cast<uint8_t const*>(pointer)),
        stride_(params.stride),
        extent_k_(extent.row()),
        extent_n_(extent.column()),
        mask_(true) {
    int64_t const weight_bytes = int64_t(extent_k_) * extent_n_ / 2;
    int64_t const zero_bytes = int64_t(extent_k_ / 128) * extent_n_;
    zeros_ = pointer_ + weight_bytes;
    scales_ = reinterpret_cast<ElementMmaB const*>(zeros_ + zero_bytes);
    auto thread_offset = ThreadMap::initial_offset(thread_id);
    tile_base_k_ = threadblock_offset.row();
    tile_base_n_ = threadblock_offset.column();
    base_k_ = threadblock_offset.row() + thread_offset.contiguous();
    base_n_ = threadblock_offset.column() + thread_offset.strided();
  }

  CUTLASS_HOST_DEVICE PackedW4IteratorB(
      Params const& params, ElementGlobalB* pointer, TensorCoord extent,
      int thread_id)
      : PackedW4IteratorB(params, pointer, extent, thread_id,
                          TensorCoord(0, 0)) {}

  CUTLASS_HOST_DEVICE PackedW4IteratorB& operator++() {
    base_k_ += ThreadblockShape::kK;
    tile_base_k_ += ThreadblockShape::kK;
    return *this;
  }

  CUTLASS_HOST_DEVICE void clear_mask(bool enable = true) {
    if (enable) mask_ = false;
  }
  CUTLASS_HOST_DEVICE void enable_mask() { mask_ = true; }
  CUTLASS_HOST_DEVICE void set_mask(Mask const&) { mask_ = true; }
  CUTLASS_HOST_DEVICE void get_mask(Mask&) const {}
  CUTLASS_HOST_DEVICE void add_pointer_offset(LongIndex logical_offset) {
    pointer_ += logical_offset / 2;
  }

  // The stock MmaPipelined path consumes `load()` below and consequently
  // expands W4 to FP16 before shared memory.  The raw staging interface keeps
  // the packed representation until the custom CUTLASS mainloop consumes it.
  // One 16-byte transfer is issued per participating thread for a 64x64 tile.
  CUTLASS_DEVICE void copy_raw_tile_async(uint8_t* destination,
                                          int thread_idx) const {
    int const copies_per_column = ThreadblockShape::kK / 32;
    int const copy_count = ThreadblockShape::kN * copies_per_column;
    for (int copy = thread_idx; copy < copy_count; copy += 128) {
      int const column = copy / copies_per_column;
      int const byte_offset = (copy % copies_per_column) * 16;
      int64_t const source_offset =
          int64_t(tile_base_n_ + column) * (stride_ / 2) + tile_base_k_ / 2 +
          byte_offset;
#if __CUDA_ARCH__ >= 800
      uint32_t const shared_address = static_cast<uint32_t>(
          __cvta_generic_to_shared(destination + copy * 16));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                       "r"(shared_address), "l"(pointer_ + source_offset));
#else
      *reinterpret_cast<int4*>(destination + copy * 16) =
          *reinterpret_cast<int4 const*>(pointer_ + source_offset);
#endif
    }
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
  }

  template <int kOutstandingGroups>
  CUTLASS_DEVICE static void wait_for_raw_tile() {
#if __CUDA_ARCH__ >= 800
    static_assert(kOutstandingGroups >= 0 && kOutstandingGroups <= 2,
                  "invalid cp.async wait group");
    asm volatile("cp.async.wait_group %0;\n" :: "n"(kOutstandingGroups));
#endif
  }

  // Materialize the same FP16 fragment as load(), but source its nibbles from
  // a staged logical [N, K/2] W4 tile. This is the bridge used by the custom
  // raw-W4 mainloop before handing B to CUTLASS's existing SmemIteratorB.
  CUTLASS_DEVICE void load_from_raw_tile(Fragment& fragment,
                                         uint8_t const* staged) const {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int const k_begin = base_k_ + c * ThreadMap::Delta::kContiguous;
        int const n = base_n_ + s * ThreadMap::Delta::kStrided;
        bool const valid = mask_ &&
            k_begin + ThreadMap::kElementsPerAccess <= extent_k_ &&
            n < extent_n_;
        uint32_t packed_values = 0;
        int zero = 0;
        ElementMmaB scale(0.0f);
        if (valid) {
          int const local_n = n - tile_base_n_;
          int const local_k = k_begin - tile_base_k_;
          packed_values = *reinterpret_cast<uint32_t const*>(
              staged + local_n * (ThreadblockShape::kK / 2) + local_k / 2);
          int const group = k_begin / 128;
          int64_t const parameter_offset = int64_t(group) * extent_n_ + n;
          zero = zeros_[parameter_offset];
          scale = scales_[parameter_offset];
        }
        int const fragment_base = ThreadMap::kElementsPerAccess *
            (c + s * ThreadMap::Iterations::kContiguous);
        __half const scale_half = __ushort_as_half(scale.raw());
        __half2 const scale_pair = __halves2half2(scale_half, scale_half);
        __half const zero_magic = __int2half_rn(1024 + zero);
        __half2 const zero_pair = __halves2half2(zero_magic, zero_magic);
        CUTLASS_PRAGMA_UNROLL
        for (int pair = 0; pair < 4; ++pair) {
          uint32_t const magic_bits =
              ((packed_values >> (4 * pair)) & 0x000f000fu) |
              0x64006400u;
          union {
            __half2 value;
            uint32_t bits;
          } decoded, scaled;
          decoded.bits = magic_bits;
          scaled.value = __hmul2(
              __hsub2(decoded.value, zero_pair), scale_pair);
          fragment[fragment_base + pair] =
              ElementMmaB::bitcast(uint16_t(scaled.bits));
          fragment[fragment_base + pair + 4] =
              ElementMmaB::bitcast(uint16_t(scaled.bits >> 16));
        }
      }
    }
  }

  CUTLASS_DEVICE void load(Fragment& fragment) const {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int const k_begin = base_k_ + c * ThreadMap::Delta::kContiguous;
        int const n = base_n_ + s * ThreadMap::Delta::kStrided;
        bool const valid_access =
            mask_ && k_begin + ThreadMap::kElementsPerAccess <= extent_k_ &&
            n < extent_n_;
        uint32_t packed_values = 0;
        int zero = 0;
        ElementMmaB scale(0.0f);
        if (valid_access) {
          int64_t const logical_offset = int64_t(n) * stride_ + k_begin;
          packed_values = *reinterpret_cast<uint32_t const*>(
              pointer_ + (logical_offset >> 1));
          int const group = k_begin / 128;
          int64_t const parameter_offset = int64_t(group) * extent_n_ + n;
          zero = zeros_[parameter_offset];
          scale = scales_[parameter_offset];
        }
        int const fragment_base = ThreadMap::kElementsPerAccess *
            (c + s * ThreadMap::Iterations::kContiguous);
        __half const scale_half = __ushort_as_half(scale.raw());
        __half2 const scale_pair = __halves2half2(scale_half, scale_half);
        __half const zero_magic =
            __int2half_rn(valid_access ? 1024 + zero : 1024);
        __half2 const zero_pair = __halves2half2(zero_magic, zero_magic);
        CUTLASS_PRAGMA_UNROLL
        for (int pair = 0; pair < 4; ++pair) {
          // OR each nibble into the mantissa of FP16 1024.  One integer
          // lop3-compatible operation materializes q0/q4 (then q1/q5, ...)
          // as a half2, avoiding eight scalar int-to-float conversions.
          uint32_t const magic_bits =
              ((packed_values >> (4 * pair)) & 0x000f000fu) |
              0x64006400u;
          union {
            __half2 value;
            uint32_t bits;
          } decoded, scaled;
          decoded.bits = magic_bits;
          scaled.value = __hmul2(__hsub2(decoded.value, zero_pair), scale_pair);
          fragment[fragment_base + pair] =
              ElementMmaB::bitcast(uint16_t(scaled.bits));
          fragment[fragment_base + pair + 4] =
              ElementMmaB::bitcast(uint16_t(scaled.bits >> 16));
        }
      }
    }
  }

  template <typename ScratchLayout>
  CUTLASS_DEVICE void decode_raw_warp_tile(
      ElementMmaB* destination, uint8_t const* staged, int warp_n,
      int k_group, int lane, ScratchLayout const& layout) const {
    for (int pair = lane; pair < 128; pair += 32) {
      int const local_n = pair / 8;
      int const local_k = (pair % 8) * 2;
      int const source_n = warp_n * 16 + local_n;
      int const source_k = k_group * 16 + local_k;
      uint8_t const packed = staged[source_n * 32 + source_k / 2];
      int64_t const parameter = int64_t(tile_base_k_ / 128) * extent_n_ + tile_base_n_ + source_n;
      __half const scale = __ushort_as_half(scales_[parameter].raw());
      __half const zero = __int2half_rn(1024 + zeros_[parameter]);
      union { uint32_t bits; __half2 value; } decoded;
      decoded.bits = (uint32_t(packed & 15) | (uint32_t(packed >> 4) << 16)) | 0x64006400u;
      __half2 const value = __hmul2(__hsub2(decoded.value, __halves2half2(zero, zero)), __halves2half2(scale, scale));
      destination[layout(cutlass::MatrixCoord(local_k, local_n))] = ElementMmaB(__low2half(value));
      destination[layout(cutlass::MatrixCoord(local_k + 1, local_n))] = ElementMmaB(__high2half(value));
    }
  }
};

using IteratorB = PackedW4IteratorB;

using Mma = nonlinear::threadblock::RawW4MmaPipelined<
    ThreadblockShape,
    IteratorA, typename MmaCore::SmemIteratorA,
    IteratorB, typename MmaCore::SmemIteratorB,
    ElementAccumulator, cutlass::layout::RowMajor,
    typename MmaCore::MmaPolicy,
    cutlass::NumericArrayConverter<
        ElementA, ElementA, IteratorA::Fragment::kElements>>;

static_assert(IteratorB::Fragment::kElements ==
                  MmaCore::SmemIteratorB::Fragment::kElements,
              "global W4 and shared FP16 fragments must cover one tile equally");

using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 4, ElementAccumulator, ElementAccumulator>;
using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape, typename Mma::Operator,
    ThreadblockShape::kK / WarpShape::kK,
    OutputOp, OutputOp::kCount>::Epilogue;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Mma, Epilogue, Swizzle>;
// This kernel is already composed in the native row-major problem convention.
// GemmUniversalAdapter would transpose legacy 2.x row-major kernels a second
// time, exchanging our custom B iterator with A. Use the base launcher directly.
using Gemm = cutlass::gemm::device::GemmUniversalBase<Kernel>;

class RmsSentinelVisitor {
 public:
  using OutputTileIterator = typename Epilogue::OutputTileIterator;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  static int const kIterations = OutputTileIterator::kIterations;
  static int const kElementsPerAccess =
      OutputTileIterator::kElementsPerAccess;
  using AccumulatorFragment =
      cutlass::Array<ElementAccumulator, kElementsPerAccess>;

  struct Arguments {
    ElementOutput* output;
    ElementOutput const* bias;
    float* workspace;
    int* counters;
    int logical_out_dim;
    int hidden_dim;
    int split_k_slices;
    float eps;

    Arguments()
        : output(nullptr), bias(nullptr), workspace(nullptr), counters(nullptr),
          logical_out_dim(0), hidden_dim(0), split_k_slices(1), eps(0.0f) {}
    Arguments(ElementOutput* output_, ElementOutput const* bias_,
              float* workspace_, int* counters_, int logical_out_dim_,
              int hidden_dim_, int split_k_slices_, float eps_)
        : output(output_), bias(bias_), workspace(workspace_),
          counters(counters_), logical_out_dim(logical_out_dim_),
          hidden_dim(hidden_dim_), split_k_slices(split_k_slices_), eps(eps_) {}
  };

  using Params = Arguments;

  struct SharedStorage {
    float accumulator[ThreadblockShape::kM][ThreadblockShape::kN];
    int is_last_split;
  };

 private:
  Params const& params_;
  SharedStorage& shared_;
  cutlass::MatrixCoord extent_;
  cutlass::MatrixCoord threadblock_offset_;
  OutputTileIterator coordinate_iterator_;
  int split_k_index_;
  int split_k_slices_;

 public:
  CUTLASS_DEVICE RmsSentinelVisitor(
      Params const& params, SharedStorage& shared,
      cutlass::MatrixCoord const& extent, int thread_idx, int, int,
      cutlass::MatrixCoord const& threadblock_offset)
      : params_(params), shared_(shared), extent_(extent),
        threadblock_offset_(threadblock_offset),
        coordinate_iterator_(
            typename OutputTileIterator::Params(
                cutlass::layout::RowMajor(extent.column())),
            static_cast<ElementOutput*>(nullptr), extent, thread_idx,
            threadblock_offset), split_k_index_(0), split_k_slices_(1) {}

  CUTLASS_DEVICE void set_k_partition(int index, int slices) {
    split_k_index_ = index;
    split_k_slices_ = slices;
  }
  CUTLASS_DEVICE void set_batch_index(int) {}
  CUTLASS_DEVICE void begin_epilogue() {}
  CUTLASS_DEVICE void begin_step(int) {}
  CUTLASS_DEVICE void begin_row(int) {}

  CUTLASS_DEVICE void visit(int, int, int, int fragment_index,
                            AccumulatorFragment const& accumulator) {
    cutlass::MatrixCoord coordinate =
        coordinate_iterator_.thread_start() +
        OutputTileIterator::ThreadMap::iteration_offset(fragment_index);
    CUTLASS_PRAGMA_UNROLL
    for (int element = 0; element < kElementsPerAccess; ++element) {
      int const row = coordinate.row() - threadblock_offset_.row();
      int const column = coordinate.column() + element -
                         threadblock_offset_.column();
      if (coordinate.row() < extent_.row() &&
          coordinate.column() + element < extent_.column() &&
          row >= 0 && row < ThreadblockShape::kM &&
          column >= 0 && column < ThreadblockShape::kN) {
        shared_.accumulator[row][column] = accumulator[element];
      }
    }
  }

  CUTLASS_DEVICE void end_row(int) {}
  CUTLASS_DEVICE void end_step(int) { ++coordinate_iterator_; }

  CUTLASS_DEVICE void end_epilogue() {
    __syncthreads();
    int const batch = extent_.row() / 2;
    int const tile = threadblock_offset_.column() / ThreadblockShape::kN;
    int const real_begin = tile * (ThreadblockShape::kN - 1);
    if (split_k_slices_ > 1) {
      int64_t const workspace_base =
          (int64_t(split_k_index_) * gridDim.y + tile) *
          (ThreadblockShape::kM / 2) * ThreadblockShape::kN;
      int const compact_values = batch * ThreadblockShape::kN;
      for (int linear = threadIdx.x; linear < compact_values;
           linear += blockDim.x) {
        int const row = linear / ThreadblockShape::kN;
        int const column = linear % ThreadblockShape::kN;
        params_.workspace[workspace_base + linear] =
            column == ThreadblockShape::kN - 1
                ? shared_.accumulator[row + batch][column]
                : shared_.accumulator[row][column];
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        __threadfence();
        int const previous = atomicAdd(params_.counters + tile, 1);
        shared_.is_last_split = previous == split_k_slices_ - 1;
      }
      __syncthreads();
      if (!shared_.is_last_split) {
        return;
      }
    }

    int const values = batch * (ThreadblockShape::kN - 1);
    for (int linear = threadIdx.x; linear < values; linear += blockDim.x) {
      int const row = linear / (ThreadblockShape::kN - 1);
      int const local_column = linear % (ThreadblockShape::kN - 1);
      int const output_column = real_begin + local_column;
      if (output_column < params_.logical_out_dim) {
        float raw = shared_.accumulator[row][local_column];
        float square_sum =
            shared_.accumulator[row + batch][ThreadblockShape::kN - 1];
        if (split_k_slices_ > 1) {
          raw = 0.0f;
          square_sum = 0.0f;
          CUTLASS_PRAGMA_UNROLL
          for (int split = 0; split < split_k_slices_; ++split) {
            int64_t const workspace_base =
                (int64_t(split) * gridDim.y + tile) *
                (ThreadblockShape::kM / 2) * ThreadblockShape::kN;
            raw += params_.workspace[
                workspace_base + row * ThreadblockShape::kN + local_column];
            square_sum += params_.workspace[
                workspace_base + row * ThreadblockShape::kN +
                ThreadblockShape::kN - 1];
          }
        }
        float value = raw *
            rsqrtf(square_sum / float(params_.hidden_dim) + params_.eps);
        if (params_.bias) {
          value += static_cast<float>(params_.bias[output_column]);
        }
        params_.output[int64_t(row) * params_.logical_out_dim +
                       output_column] = ElementOutput(value);
      }
    }
    if (split_k_slices_ > 1) {
      __syncthreads();
      if (threadIdx.x == 0) {
        __threadfence();
        atomicExch(params_.counters + tile, 0);
      }
    }
  }
};

using RmsEpilogue = typename cutlass::epilogue::threadblock::
    EpilogueWithVisitorFromExistingEpilogue<
        RmsSentinelVisitor, Epilogue>::Epilogue;

// CUTLASS's example visitor kernel fixes grid_tiled_shape.k() to one.  This
// specialization keeps its MMA and epilogue implementation, but lets several
// CTAs cooperate on K.  The final arriving CTA performs the RMS recovery in
// the visitor, so the complete operation remains one CUDA kernel launch.
struct RmsKernel {
  using Epilogue = RmsEpilogue;
  using EpilogueVisitor = typename Epilogue::Visitor;
  using ThreadblockSwizzle = Swizzle;
  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using TensorRefA = cutlass::TensorRef<ElementA, LayoutA>;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using TensorRefB = cutlass::TensorRef<ElementB, LayoutB>;
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  struct Arguments {
    cutlass::gemm::GemmCoord problem_size;
    TensorRefA ref_A;
    TensorRefB ref_B;
    typename EpilogueVisitor::Arguments epilogue_visitor;
    int split_k_slices;

    Arguments(cutlass::gemm::GemmCoord problem_size_, TensorRefA ref_A_,
              TensorRefB ref_B_,
              typename EpilogueVisitor::Arguments epilogue_visitor_,
              int split_k_slices_)
        : problem_size(problem_size_), ref_A(ref_A_), ref_B(ref_B_),
          epilogue_visitor(epilogue_visitor_),
          split_k_slices(split_k_slices_) {}
  };

  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorB::Params params_B;
    int gemm_k_size;
    void* ptr_A;
    void* ptr_B;
    typename EpilogueVisitor::Params epilogue_visitor;

    Params(Arguments const& args)
        : problem_size(args.problem_size), swizzle_log_tile(0),
          params_A(args.ref_A.layout()), params_B(args.ref_B.layout()),
          gemm_k_size(cutlass::round_up(
              cutlass::ceil_div(args.problem_size.k(), args.split_k_slices),
              Mma::Shape::kK)),
          ptr_A(args.ref_A.data()), ptr_B(args.ref_B.data()),
          epilogue_visitor(args.epilogue_visitor) {
      ThreadblockSwizzle swizzle;
      grid_tiled_shape = swizzle.get_tiled_shape(
          args.problem_size,
          {Mma::Shape::kM, Mma::Shape::kN, Mma::Shape::kK},
          args.split_k_slices);
      swizzle_log_tile = swizzle.get_log_tile(grid_tiled_shape);
    }
  };

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    struct {
      typename Epilogue::SharedStorage epilogue;
      typename EpilogueVisitor::SharedStorage visitor;
    } epilogue;
  };

  CUTLASS_DEVICE void operator()(Params const& params,
                                 SharedStorage& shared_storage) {
    ThreadblockSwizzle swizzle;
    cutlass::gemm::GemmCoord tile =
        swizzle.get_tile_offset(params.swizzle_log_tile);
    if (tile.m() >= params.grid_tiled_shape.m() ||
        tile.n() >= params.grid_tiled_shape.n() ||
        tile.k() >= params.grid_tiled_shape.k()) {
      return;
    }
    int const offset_k = tile.k() * params.gemm_k_size;
    int const problem_size_k =
        cutlass::min(params.gemm_k_size, params.problem_size.k() - offset_k);
    if (problem_size_k <= 0) {
      return;
    }

    int const thread_idx = threadIdx.x;
    cutlass::MatrixCoord tb_offset_A{
        tile.m() * Mma::Shape::kM, offset_k};
    cutlass::MatrixCoord tb_offset_B{
        offset_k, tile.n() * Mma::Shape::kN};
    typename Mma::IteratorA iterator_A(
        params.params_A, static_cast<ElementA*>(params.ptr_A),
        {params.problem_size.m(), params.problem_size.k()}, thread_idx,
        tb_offset_A);
    typename Mma::IteratorB iterator_B(
        params.params_B, static_cast<ElementB*>(params.ptr_B),
        {params.problem_size.k(), params.problem_size.n()}, thread_idx,
        tb_offset_B);
    int const warp_idx =
        __shfl_sync(0xffffffff, static_cast<int>(threadIdx.x) / 32, 0);
    int const lane_idx = threadIdx.x % 32;
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC accumulators;
    accumulators.clear();
    int const gemm_k_iterations =
        cutlass::ceil_div(problem_size_k, Mma::Shape::kK);
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B,
        accumulators);

    cutlass::MatrixCoord threadblock_offset(
        tile.m() * Mma::Shape::kM, tile.n() * Mma::Shape::kN);
    EpilogueVisitor visitor(
        params.epilogue_visitor, shared_storage.epilogue.visitor,
        params.problem_size.mn(), thread_idx, warp_idx, lane_idx,
        threadblock_offset);
    visitor.set_k_partition(tile.k(), params.grid_tiled_shape.k());
    Epilogue epilogue(shared_storage.epilogue.epilogue, thread_idx,
                      warp_idx, lane_idx);
    epilogue(visitor, accumulators);
  }
};

}  // namespace

void launch_cutlass_w4a16_probe(const at::Half* a, const uint8_t* b,
                                at::Half* output, int rows, int out_dim,
                                int hidden_dim, cudaStream_t stream) {
  typename Gemm::Arguments arguments(
      cutlass::gemm::GemmUniversalMode::kGemm,
      {rows, out_dim, hidden_dim},
      1,
      {1.0f, 0.0f},
      reinterpret_cast<const ElementA*>(a),
      reinterpret_cast<const ElementGlobalB*>(b),
      reinterpret_cast<const ElementOutput*>(output),
      reinterpret_cast<ElementOutput*>(output),
      static_cast<int64_t>(rows) * hidden_dim,
      static_cast<int64_t>(out_dim) * hidden_dim,
      static_cast<int64_t>(rows) * out_dim,
      static_cast<int64_t>(rows) * out_dim,
      hidden_dim,
      hidden_dim,
      out_dim,
      out_dim);
  Gemm gemm;
  const cutlass::Status status = gemm(arguments, nullptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS W4A16 probe failed: ",
              cutlassGetStatusString(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_cutlass_w4a16_rms_sentinel(
    const at::Half* input, const at::Half* gamma,
    const uint8_t* packed_storage,
    const at::Half* bias, at::Half* output, float* workspace, int* counters,
    int batch, int logical_out_dim, int physical_out_dim, int hidden_dim,
    int split_k_slices, float eps, cudaStream_t stream) {
  typename RmsKernel::Arguments arguments(
      {batch * 2, physical_out_dim, hidden_dim},
      {reinterpret_cast<ElementA*>(const_cast<at::Half*>(input)),
       hidden_dim},
      {reinterpret_cast<ElementGlobalB*>(const_cast<uint8_t*>(packed_storage)),
       hidden_dim},
      typename RmsSentinelVisitor::Arguments(
          reinterpret_cast<ElementOutput*>(output),
          reinterpret_cast<ElementOutput const*>(bias), workspace, counters,
          logical_out_dim, hidden_dim, split_k_slices, eps),
      split_k_slices);
  typename RmsKernel::Params params(arguments);
  params.params_A.gamma = reinterpret_cast<ElementA const*>(gamma);
  params.params_A.batch = batch;
  Swizzle swizzle;
  dim3 grid = swizzle.get_grid_shape(params.grid_tiled_shape);
  dim3 block(RmsKernel::kThreadCount, 1, 1);
  int const shared_bytes = sizeof(typename RmsKernel::SharedStorage);
  if (shared_bytes > 48 * 1024) {
    cudaFuncSetAttribute(cutlass::Kernel<RmsKernel>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_bytes);
  }
  cutlass::Kernel<RmsKernel><<<grid, block, shared_bytes, stream>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
