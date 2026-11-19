#pragma once
#include <cutlass/aligned_buffer.h>
#include <cutlass/gemm/gemm.h>

namespace nonlinear { namespace threadblock {
template <typename Shape_, typename WarpShape_, typename IteratorA_, typename SmemIteratorA_, typename IteratorB_, typename ElementC_, typename LayoutC_, typename Policy_, typename TransformA_>
class WarpLocalRawW4Mma {
 public:
  using Shape=Shape_; using WarpShape=WarpShape_; using IteratorA=IteratorA_; using IteratorB=IteratorB_; using SmemIteratorA=SmemIteratorA_; using ElementC=ElementC_; using LayoutC=LayoutC_; using Policy=Policy_; using TransformA=TransformA_;
  using Operator=typename Policy::Operator; using FragmentA=typename IteratorA::Fragment; using FragmentC=typename Operator::FragmentC;
  using WarpCount=cutlass::gemm::GemmShape<Shape::kM/WarpShape::kM,Shape::kN/WarpShape::kN,Shape::kK/WarpShape::kK>;
  using ArchTag=typename Operator::ArchTag;
  static cutlass::ComplexTransform const kTransformA=Operator::kTransformA;
  static cutlass::ComplexTransform const kTransformB=Operator::kTransformB;
  static int const kStages=1;
  static int const kWarpGemmIterations=WarpShape::kK/Operator::Policy::MmaShape::kK;
  using TensorRefA=cutlass::TensorRef<typename Operator::ElementA,typename Operator::LayoutA>;
  using TensorRefB=cutlass::TensorRef<typename Operator::ElementB,typename Operator::LayoutB>;
  static int const kAElements=Shape::kM*Shape::kK;
  static int const kScratchElements=WarpShape::kK*WarpShape::kN;
  static int const kRawBytes=Shape::kK*Shape::kN/2;
  class SharedStorage {
   public:
    cutlass::AlignedBuffer<typename Operator::ElementA,kAElements,16> operand_a;
    cutlass::AlignedBuffer<uint8_t,kRawBytes,16> packed_b;
    cutlass::AlignedBuffer<typename Operator::ElementB,WarpCount::kCount*kScratchElements,16> warp_b;
    CUTLASS_DEVICE TensorRefA operand_A_ref() { return {operand_a.data(),typename Operator::LayoutA::packed({Shape::kM,Shape::kK})}; }
    CUTLASS_DEVICE TensorRefB warp_B_ref(int warp) { return {warp_b.data()+warp*kScratchElements,typename Operator::LayoutB::packed({WarpShape::kK,WarpShape::kN})}; }
  };
 private:
  Operator mma_;
  SharedStorage& shared_;
  int thread_idx_,warp_idx_,lane_idx_;
 public:
  CUTLASS_DEVICE WarpLocalRawW4Mma(SharedStorage& s,int t,int w,int l,TransformA=TransformA()):mma_(),shared_(s),thread_idx_(t),warp_idx_(w),lane_idx_(l){}
  CUTLASS_DEVICE void operator()(int iterations,FragmentC& accum,IteratorA a,IteratorB b,FragmentC const& src){
    accum=src;
    for(int tile=0;tile<iterations;++tile){
      FragmentA fa; fa.clear(); a.load(fa); ++a;
      IteratorB staged=b; staged.copy_raw_tile_async(shared_.packed_b.data(),thread_idx_); ++b;
      IteratorB::wait_for_raw_tile(); __syncthreads();
      SmemIteratorA sa(shared_.operand_A_ref(),thread_idx_); sa.store(fa); __syncthreads();
      #pragma unroll
      for(int kg=0;kg<kWarpGemmIterations;++kg){
        typename Operator::FragmentA wa; typename Operator::FragmentB wb;
        typename Operator::IteratorA ia(shared_.operand_A_ref(),lane_idx_);
        #pragma unroll
        for(int i=0;i<kWarpGemmIterations;++i){ if(i<kg){ia.set_kgroup_index(i); ++ia;} }
        ia.set_kgroup_index(kg); ia.load(wa);
        staged.decode_raw_warp_tile(shared_.warp_B_ref(warp_idx_).data(),shared_.packed_b.data(),warp_idx_,kg,lane_idx_);
        __syncwarp();
        typename Operator::IteratorB ib(shared_.warp_B_ref(warp_idx_),lane_idx_);
        ib.set_kgroup_index(0); ib.load(wb);
        mma_(accum,wa,wb,accum);
      }
      __syncthreads();
    }
  }
};
} }
