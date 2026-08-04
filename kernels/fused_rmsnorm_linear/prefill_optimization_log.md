# Prefill RMSNorm + Linear Optimization Log

This log records every optimization experiment made to
`fused_rmsnorm_linear_prefill.cu`, including its motivation, correctness
result, and measured performance.

## Acceptance criteria

- Benchmark:
  `benchmark/fused_rmsnorm_linear/bench_fused_rmsnorm_linear_prefill.py`
- Default shape: `[Batch, SeqLen, HiddenDim] = [4, 128, 4096]`
- Default `OutDim`: `4096`
- Correctness: maximum absolute error at most `2e-3`; relative error is not
  used.
- Target: more than `3.0x` speedup over PyTorch RMSNorm + Linear.

## Baseline: initial WMMA implementation

### Implementation

- FP16 WMMA `16x16x16` matrix multiplication with FP32 accumulation.
- Eight independent output-tile warps per CTA.
- FP32 RMS square-sum reduction on a separate CUDA stream.
- CUDA events establish current-stream-to-RMS and RMS-to-epilogue
  dependencies.
- The projection is stored as FP16 before the RMS scaling epilogue.

### Static verification

- PyTorch 2.6 / CUDA 12.4 extension compilation: passed.
- Python syntax checks: passed.
- `git diff --check`: passed.
- GPU correctness tests cannot run in the Codex execution environment because
  the NVIDIA driver is unavailable.

### Performance

Measured by the user on an NVIDIA GeForce RTX 3090 with 20 warmup iterations
and 100 measured iterations:

| Implementation | Mean | p50 | p95 | Effective TFLOPS |
|---|---:|---:|---:|---:|
| Optimization 1 fused | 1.2342 ms | 1.2329 ms | 1.2472 ms | 13.920 |
| PyTorch RMSNorm + Linear | 0.4570 ms | 0.4557 ms | 0.4721 ms | 37.591 |

Speedup: `0.370x`.

Maximum absolute error: `0.00195312`, within the `2e-3` requirement.

### Bottleneck assessment

The initial WMMA kernel reloads the same `x` tile independently for every
output tile and does not stage/reuse `x` and weight tiles at CTA scope. It is
therefore expected to be substantially less efficient than cuBLAS for the
default GEMM. Performance work must start from an actual baseline and GPU
profile because the requested `3x` target is stronger than the speedup normally
available from eliminating only the RMSNorm materialization around a
GEMM-dominated workload.

## Optimization 1: CTA-wide weight-tile reuse

### Motivation

The grid maps one CTA to one 16-column output tile and eight 16-row token
tiles. Consequently, all eight warps consume exactly the same `16x16` weight
tile in every K iteration. The baseline issued one global-memory WMMA load per
warp, redundantly loading that tile eight times.

### Modification

- Added a CTA-wide shared-memory `16x64` FP16 weight tile.
- All 256 CTA threads cooperatively load four weight elements.
- The eight warps load their matrix-B fragments from that shared tile.
- Added CTA barriers before consumption and before overwriting the tile in the
  next 64-wide K block. Staging four WMMA K tiles at once amortizes the
  barriers over four MMA instructions.
- The GEMM arithmetic, RMS kernel, epilogue, and numerical behavior are
  unchanged.

### Expected effect

The global-memory traffic for matrix-B tile loads is reduced to one eighth of
the baseline amount. This trades the redundant loads for two CTA barriers per
K iteration. Whether that trade is favorable must be determined on the target
GPU.

### Verification

- RTX 3090 performance: `1.2342 ms`, or `0.370x` versus PyTorch.
- RTX 3090 maximum absolute error: `0.00195312`, passed.
- CUDA 12.4 extension compilation: passed.
- `git diff --check`: passed.

### Decision

The narrow `128x16` output tile leaves no cross-column reuse of `x` and
achieves only 13.920 effective TFLOPS. Replace this mapping in Optimization 2.

## Optimization 2: two-dimensional `64x128` CTA output tile

### Motivation

Optimization 1 reuses weights across token rows, but every `x` tile is still
reloaded for each 16-column output tile. The CTA covers only 16 output columns,
which produces 1024 CTAs and very low arithmetic intensity at CTA scope.

### Modification

- Changed the CTA output tile from `128x16` to `64x128`.
- Arranged the eight warps as a `2x4` grid.
- Each warp owns a `32x32` output region using four FP32 WMMA accumulator
  fragments.
- Cooperatively stage a `64x32` `x` tile and a `128x32` weight tile.
- Reuse every staged `x` value across 128 output columns and every staged
  weight value across 64 token rows.
- Retain guarded loads and stores for partial M/N tiles.
- Retain FP16 projection storage so the numerical contract is unchanged.

### Expected effect

The number of CTAs for the default shape falls from 1024 to 256. Both operands
now receive two-dimensional shared-memory reuse, and each synchronization
protects eight MMA operations per warp across a 32-wide K stage.

### Verification

- RTX 3090 correctness: all 7 tests passed.
- RTX 3090 maximum absolute error: `0.00195312`, passed.
- RTX 3090 performance:

  | Implementation | Mean | p50 | p95 | Effective TFLOPS |
  |---|---:|---:|---:|---:|
  | Optimization 2 fused | 2.1835 ms | 2.1545 ms | 2.3009 ms | 7.868 |
  | PyTorch RMSNorm + Linear | 0.4145 ms | 0.4106 ms | 0.4250 ms | 41.445 |

  Speedup: `0.190x`.

### Decision

The two-dimensional reuse is not enough to offset shared-memory WMMA loads and
two CTA barriers every 32 K values. This version is slower than Optimization 1.
Retain the four-fragment warp accumulator mapping but remove shared staging from
the complete-tile fast path in Optimization 3.

## Optimization 3: barrier-free global-load fast path

### Motivation

The default `[512, 4096] x [4096, 4096]` GEMM consists entirely of complete
`64x128` output tiles. Optimization 2 executes 256 CTA barriers and repeatedly
round-trips both operands through shared memory in every CTA. On RTX 3090 this
reduced throughput from 13.920 to 7.868 TFLOPS.

### Modification

- Keep four FP32 accumulator fragments per warp and the `64x128` CTA mapping.
- For complete M/N tiles, load WMMA matrix-A and matrix-B fragments directly
  from global memory.
- Remove all K-loop CTA barriers from the default fast path.
- Rely on the GPU cache hierarchy to satisfy duplicate operand reads.
- Preserve the zero-padded shared-memory path only for partial edge CTAs.

### Expected effect

This tests whether accumulator reuse without synchronization overhead is more
valuable than explicit operand reuse for this fixed prefill shape.

### Verification

- RTX 3090 correctness: all 7 tests passed.
- RTX 3090 maximum absolute error: `0.00195312`, passed.
- RTX 3090 performance:

  | Implementation | Mean | p50 | p95 | Effective TFLOPS |
  |---|---:|---:|---:|---:|
  | Optimization 3 fused | 0.7276 ms | 0.7259 ms | 0.7442 ms | 23.613 |
  | PyTorch RMSNorm + Linear | 0.4508 ms | 0.4547 ms | 0.4659 ms | 38.109 |

  Speedup: `0.620x`.

### Decision

Removing barriers recovers most of the loss from Optimization 2 and exceeds
Optimization 1, but duplicate operand traffic limits throughput. Reintroduce
staging only after fixing shared-memory bank conflicts and scalar copy
overhead.

## Optimization 4: skewed shared tiles with 128-bit copies

### Motivation

Optimization 2 used a 32-half shared leading dimension. This maps successive
WMMA rows onto the same bank pattern and creates severe `ldmatrix` conflicts.
It also copied staged operands through scalar half assignments.

### Modification

- Add an eight-half skew: shared leading dimension changes from 32 to 40.
- Copy the complete-tile `x` stage with one aligned 128-bit load/store per
  thread.
- Copy the weight stage with two aligned 128-bit load/stores per thread.
- Keep a 32-wide K stage and four MMA operations per warp per K tile.
- Retain the scalar zero-padded fallback for partial tiles.

### Expected effect

The skew targets shared-memory load bank conflicts, while vector copies reduce
the staging instruction count and produce fully coalesced 16-byte global
transactions.

### Verification

- RTX 3090 correctness: all 7 tests passed.
- RTX 3090 maximum absolute error: `0.00195312`, passed.
- RTX 3090 performance:

  | Implementation | Mean | p50 | p95 | Effective TFLOPS |
  |---|---:|---:|---:|---:|
  | Optimization 4 fused | 0.4702 ms | 0.4700 ms | 0.4782 ms | 36.541 |
  | PyTorch RMSNorm + Linear | 0.4579 ms | 0.4567 ms | 0.4710 ms | 37.517 |

  Speedup: `0.974x`.

### Decision

Skewed, vectorized staging is 1.55x faster than Optimization 3 and 4.64x
faster than Optimization 2. Global-to-shared copies remain serialized with
WMMA computation, so overlap them using Ampere asynchronous copies.

## Optimization 5: `cp.async` double-buffered K pipeline

### Motivation

Optimization 4 alternates synchronous 128-bit copies, a CTA barrier, WMMA,
and another CTA barrier. Ampere can overlap global-to-shared copies with
Tensor Core instructions through `cp.async`.

### Modification

- Allocate two skewed operand stages.
- Prime stage zero, then asynchronously load the next 32-wide K tile while
  computing the current tile.
- Use `cp.async.commit_group` and `cp.async.wait_group` for stage ownership.
- Alias the operand double buffers and FP32 output staging through a shared
  union: operands need about 30 KiB and output needs 32 KiB, so total static
  shared memory remains 32 KiB rather than their sum.
- Keep the scalar boundary fallback.

### Expected effect

Hide a substantial portion of global-memory staging latency and remove one
blocking synchronization phase from the steady-state K loop.

### Verification

- RTX 3090 correctness: all 7 tests passed.
- RTX 3090 maximum absolute error: `0.00195312`, passed.
- RTX 3090 performance:

  | Implementation | Mean | p50 | p95 | Effective TFLOPS |
  |---|---:|---:|---:|---:|
  | Optimization 5 fused | 0.4155 ms | 0.4147 ms | 0.4196 ms | 41.351 |
  | PyTorch RMSNorm + Linear | 0.4519 ms | 0.4547 ms | 0.4598 ms | 38.018 |

  Speedup: `1.088x`.

### Decision

Asynchronous double buffering improves the fused kernel by 11.7% over
Optimization 4 and makes it faster than PyTorch, but the requested 3x target
is not reached. Sweep CTA shapes to improve occupancy and Tensor Core issue
rate.

## Optimization 6: `64x64` CTA with four warps

### Motivation

The `64x128` CTA uses eight warps, four accumulator fragments per warp, and a
32 KiB shared union. A `64x64` CTA retains the same `32x32` per-warp work but
halves the warp count and output staging footprint. This may allow more
resident CTAs and better latency hiding.

### Modification

- Reduce CTA N tile from 128 to 64.
- Reduce warps from eight to four, arranged as a `2x2` grid.
- Keep the `32x32` per-warp accumulator tile.
- Each thread copies two `x` vectors and two weight vectors per stage.
- Preserve the skewed, double-buffered `cp.async` pipeline.

### Verification

- RTX 3090 correctness: all 7 tests passed.
- RTX 3090 maximum absolute error: `0.00195312`, passed.
- RTX 3090 performance:

  | Implementation | Mean | p50 | p95 | Effective TFLOPS |
  |---|---:|---:|---:|---:|
  | Optimization 6 fused | 0.4940 ms | 0.4936 ms | 0.5006 ms | 34.780 |
  | PyTorch RMSNorm + Linear | 0.4565 ms | 0.4555 ms | 0.4618 ms | 37.632 |

  Speedup: `0.924x`.

### Decision

Four warps do not issue enough independent Tensor Core work. The higher CTA
residency does not recover the lost throughput. Restore eight warps and keep
the output tile area at 8192 elements.

## Optimization 7: weight-reuse-oriented `128x64` CTA

### Motivation

The default weight matrix is 32 MiB while `x` is 4 MiB. Compared with the
`64x128` tile, transposing the CTA aspect ratio doubles weight reuse across M
while halving reuse of the much smaller `x` operand. The number of output
elements, accumulator fragments, CTAs, and output staging bytes stays fixed.

### Modification

- Use a `128x64` CTA output tile with eight warps arranged `4x2`.
- Each warp continues to compute `32x32`.
- Each thread asynchronously copies two `x` vectors and one weight vector per
  K stage.
- Preserve skewing, double buffering, and output/operand shared aliasing.

### Verification

- RTX 3090 correctness: all 7 tests passed.
- RTX 3090 maximum absolute error: `0.00195312`, passed.
- RTX 3090 performance:

  | Implementation | Mean | p50 | p95 | Effective TFLOPS |
  |---|---:|---:|---:|---:|
  | Optimization 7 fused | 0.4291 ms | 0.4270 ms | 0.4332 ms | 40.041 |
  | PyTorch RMSNorm + Linear | 0.4561 ms | 0.4557 ms | 0.4595 ms | 37.667 |

  Speedup: `1.063x`.

### Decision

The weight-reuse-oriented aspect ratio is slower than Optimization 5. Restore
the `64x128` CTA. The result suggests wider N reuse and/or its warp scheduling
is more valuable than doubling weight reuse across M.

## Optimization 8: per-warp output staging

### Motivation

Each warp stores and converts one accumulator fragment at a time, but previous
versions reserve a full `64x128` FP32 output tile (32 KiB). Since fragments
from different warps do not need to share output data, only one `16x16` tile
per warp is necessary.

### Modification

- Restore the best `64x128`, eight-warp CTA from Optimization 5.
- Replace the 32 KiB CTA-wide output array with eight private `16x16` FP32
  tiles totaling 8 KiB.
- Sequentially reuse each warp's private tile for its four accumulator
  fragments.
- Operand double buffers remain about 30 KiB, so the shared union footprint
  falls from 32 KiB to about 30 KiB.

### Verification

- RTX 3090 correctness: all 7 tests passed.
- RTX 3090 maximum absolute error: `0.00195312`, passed.
- RTX 3090 performance:

  | Implementation | Mean | p50 | p95 | Effective TFLOPS |
  |---|---:|---:|---:|---:|
  | Optimization 8 fused | 0.4113 ms | 0.4106 ms | 0.4137 ms | 41.765 |
  | PyTorch RMSNorm + Linear | 0.4574 ms | 0.4567 ms | 0.4608 ms | 37.563 |

  Speedup: `1.112x`.

### Profile

Nsight Compute measured the projection kernel at `439.58 us` under profiling:

- Compute throughput: `33.59%`
- L2 throughput: `66.01%`
- DRAM throughput: `31.14%`
- Achieved occupancy: `44.89%` (`21.55` active warps/SM)
- Theoretical occupancy: `50%`, limited by both registers and shared memory
- Average issue interval: `34.18` cycles/instruction, including an estimated
  `35.98%` issue-pipe stall opportunity

The kernel is more L2-bound than compute-bound and needs more computation per
operand tile.

## Optimization 9: `128x128` CTA and eight accumulators per warp

### Motivation

Doubling M coverage halves both the number of CTAs and the number of times the
32 MiB weight matrix is fetched from L2. A square `128x128` CTA is a standard
high-intensity dense-GEMM shape and directly addresses the profiler's L2-heavy
result.

### Modification

- Increase the CTA output tile from `64x128` to `128x128`.
- Arrange eight warps as `4x2`.
- Each warp computes `32x64` using eight FP32 accumulator fragments.
- Each thread copies two `x` vectors and two weight vectors per K stage.
- Keep skewed `cp.async` double buffering and per-warp output staging.
- Default grid size decreases from 256 to 128 CTAs.

### Expected effect

Halve cross-CTA operand traffic and double arithmetic intensity, at the cost
of twice as many accumulator registers per warp and lower potential occupancy.

### Verification

- RTX 3090 correctness: all 7 tests passed.
- RTX 3090 maximum absolute error: `0.00195312`, passed.
- RTX 3090 performance:

  | Implementation | Mean | p50 | p95 | Effective TFLOPS |
  |---|---:|---:|---:|---:|
  | Optimization 9 fused | 0.4015 ms | 0.4004 ms | 0.4045 ms | 42.788 |
  | PyTorch RMSNorm + Linear | 0.4571 ms | 0.4567 ms | 0.4598 ms | 37.587 |

  Speedup: `1.138x`.

### Profile

- Projection duration under profiling: `424.74 us`
- Compute throughput: `34.45%`
- L2 throughput: `45.72%`
- Achieved occupancy: `26.47%`
- Theoretical occupancy: `33.33%`, limited by registers and shared memory
- Grid: 128 CTAs, reported as only `0.8` full resource waves

### Decision

The larger tile slightly improves normal timing but underfills the GPU and
reduces occupancy sharply. Do not enlarge the accumulator tile further.

## Optimization 10: occupancy-oriented `32x128` CTA

### Motivation

The profiler shows insufficient active warps and an underfilled grid. Halving
M coverage relative to Optimization 5 reduces accumulator registers per warp
from four fragments to two and doubles the grid size.

### Modification

- Use a `32x128` CTA output tile.
- Arrange eight warps as `1x8`; each warp computes `32x16`.
- Keep two FP32 accumulator fragments per warp.
- Only 128 threads copy the `x` stage; all threads cooperatively copy weight.
- Preserve the skewed asynchronous double buffer.
- Increase the default grid from 256 to 512 CTAs.

### Expected effect

Increase residency and latency hiding at the cost of doubling weight traffic
across M tiles.

### Verification

- RTX 3090 correctness: all 7 tests passed.
- RTX 3090 maximum absolute error: `0.00195312`, passed.
- RTX 3090 performance:

  | Implementation | Mean | p50 | p95 | Effective TFLOPS |
  |---|---:|---:|---:|---:|
  | Optimization 10 fused | 0.5785 ms | 0.5784 ms | 0.5867 ms | 29.697 |
  | PyTorch RMSNorm + Linear | 0.4577 ms | 0.4566 ms | 0.4680 ms | 37.538 |

  Speedup: `0.791x`.

### Decision

The extra occupancy cannot compensate for doubled cross-CTA weight traffic.
Restore the `128x128` tile from Optimization 9.

## Optimization 11: experimental FP16 accumulation

### Motivation

Optimization 9 is limited to 33% theoretical occupancy by registers and
shared memory. FP16 accumulator fragments use fewer registers and can be
stored directly to the FP16 projection without an FP32 shared conversion
tile. This experiment tests whether the accuracy budget permits that trade.

### Modification

- Restore the `128x128`, eight-accumulator-per-warp layout.
- Change WMMA accumulator fragments from FP32 to FP16.
- Store complete tiles directly from WMMA fragments to global FP16
  projection.
- Keep a small FP16 per-warp staging tile only for boundary output.

### Risk

FP16 accumulation over K=4096 can introduce much larger numerical error. This
version is retained only if the existing `2e-3` absolute tolerance passes.

### Verification

- Correctness: failed.
- Default shape:
  - `406376 / 2097152` elements (`19.4%`) exceeded `2e-3`.
  - Maximum absolute error: `0.025390625`.
- Smaller K=256 cases also failed with maximum absolute errors between
  `0.00488` and `0.00586`.
- Performance was not accepted or used because correctness failed.

### Decision

Reject FP16 accumulation and restore FP32 accumulator fragments. The error
grows substantially with K and cannot satisfy the requested numerical
contract.

## Optimization 12: CUTLASS Ampere multistage GEMM

### Motivation

The hand-written WMMA kernel reaches only about 43 effective TFLOPS. Achieving
the requested threefold end-to-end speedup requires roughly 113 TFLOPS for the
GEMM-dominated workload. The repository environment already provides CUTLASS,
whose Ampere TensorOp kernels implement instruction-level `cp.async`
multistaging, conflict-free shared layouts, warp-specialized iterator
pipelines, and tuned epilogues.

### Modification

- Keep the experimental WMMA kernel source for comparison.
- Route the projection launcher through
  `cutlass::gemm::device::Gemm`.
- Use FP16 A/B/C, FP32 accumulation, row-major A/C, and column-major logical B
  over the existing row-major `[OutDim, HiddenDim]` weight storage.
- Instantiate threadblock `128x128x32`, warp `64x64x32`, instruction
  `16x8x16`, and three pipeline stages for SM80.
- Add the existing `$CUTLASS_PATH` or `~/third_party/cutlass` headers to the
  isolated prefill extension build.

### Expected effect

Replace the remaining hand-written WMMA scheduling bottlenecks with CUTLASS's
production Ampere GEMM mainloop while preserving the separate concurrent RMS
reduction and FP16 projection contract.

### Verification

- RTX 3090 correctness: all 7 tests passed.
- Maximum absolute error: `0.00195312`, passed.
- RTX 3090 performance:

  | Implementation | Mean | p50 | p95 | Effective TFLOPS |
  |---|---:|---:|---:|---:|
  | Optimization 12 fused | 0.3769 ms | 0.3758 ms | 0.3821 ms | 45.579 |
  | PyTorch RMSNorm + Linear | 0.4822 ms | 0.4792 ms | 0.4946 ms | 35.627 |

  Speedup: `1.279x`.

### Decision

CUTLASS is the best implementation so far. Continue tuning its Ampere
mainloop rather than the hand-written WMMA path.

## Optimization 13: four-stage CUTLASS mainloop

### Motivation

The three-stage mainloop may not fully hide global-memory latency for K=4096.
CUTLASS supports deeper `cp.async` staging on SM80.

### Modification

- Increase CUTLASS pipeline stages from three to four.
- Keep threadblock, warp, instruction, layouts, and epilogue unchanged.

### Verification

- Benchmark correctness passed; maximum absolute error `0.00195312`.
- Fused mean: `0.3739 ms`, p50 `0.3736 ms`, p95 `0.3748 ms`.
- PyTorch mean: `0.4555 ms`.
- Same-run speedup: `1.219x`.

### Decision

Four stages improve absolute fused latency by about 0.8% over three stages.
Retain provisionally and test one deeper stage.

## Optimization 14: five-stage CUTLASS mainloop

### Modification

- Increase CUTLASS pipeline stages from four to five.
- Keep all other GEMM configuration parameters unchanged.

### Verification

- Benchmark correctness passed; maximum absolute error `0.00195312`.
- Fused mean: `0.3748 ms`.
- PyTorch mean: `0.4567 ms`.
- Same-run speedup: `1.218x`.

### Decision

Five stages are slightly slower than four. Restore four stages.

## Optimization 15: CUTLASS `64x128` threadblock

### Motivation

Nsight showed that a 128-CTA grid can underfill the RTX 3090 when each CTA is
resource-heavy. Halving threadblock M doubles the grid and reduces the number
of warps and accumulator registers per CTA.

### Modification

- Restore four pipeline stages.
- Change threadblock shape from `128x128x32` to `64x128x32`.
- Change warp shape from `64x64x32` to `32x64x32`.
- Keep the instruction shape and data layouts unchanged.

### Verification

- Benchmark correctness passed; maximum absolute error `0.00195312`.
- Fused mean: `0.3723 ms`, p50 `0.3717 ms`, p95 `0.3758 ms`.
- PyTorch mean: `0.4573 ms`.
- Same-run speedup: `1.228x`.

### Decision

This is the best absolute fused latency so far, narrowly beating the
`128x128` four-stage configuration. Continue the threadblock sweep.

## Optimization 16: CUTLASS `64x64` threadblock

### Modification

- Change threadblock shape to `64x64x32`.
- Change warp shape to `32x32x32`.
- Retain four stages and all data layouts.

### Verification

- Benchmark correctness passed; maximum absolute error `0.00195312`.
- Fused mean: `0.3310 ms`, p50 `0.3297 ms`, p95 `0.3359 ms`.
- PyTorch mean: `0.4565 ms`.
- Same-run speedup: `1.379x`.

### Decision

The smaller tile is substantially faster and becomes the new best candidate.
Tune pipeline depth again for this resource footprint.

## Optimization 17: three-stage `64x64` CUTLASS kernel

### Motivation

Reducing stages from four to three decreases shared memory per CTA and may
increase residency for the small threadblock shape.

### Verification

- Benchmark correctness passed; maximum absolute error `0.00195312`.
- Fused mean: `0.3303 ms`.
- PyTorch mean: `0.4549 ms`.
- Same-run speedup: `1.377x`.

### Decision

Three stages are marginally better than four. Test whether two stages improve
residency further without exposing memory latency.

## Optimization 18: two-stage `64x64` CUTLASS kernel

### Verification

- Benchmark correctness passed; maximum absolute error `0.00195312`.
- Fused mean: `0.3444 ms`.
- PyTorch mean: `0.4594 ms`.
- Same-run speedup: `1.334x`.

### Decision

Two stages expose too much memory latency and are 4.3% slower than three.
Restore three stages. Use CUTLASS profiler enumeration before further shape
changes.

## Optimization 19: generator-selected ten-stage `64x64` kernel

### Motivation

CUTLASS's SM80 generator emits `64x64x32` FP16/FP32-accumulate kernels with
ten stages, indicating that this small operand tile can use a much deeper
pipeline within the SM80 shared-memory budget. The full profiler build was
stopped because unrelated reference-library compilation was excessively slow,
but the generated kernel manifest provides the configuration directly.

### Modification

- Keep `64x64x32` threadblock and `32x32x32` warp shapes.
- Increase pipeline stages from three to ten.

### Verification

- Benchmark correctness passed; maximum absolute error `0.00195312`.
- Fused mean: `0.3618 ms`.
- PyTorch mean: `0.4570 ms`.
- Same-run speedup: `1.263x`.

### Decision

Ten stages regress by 9.5% versus the three-stage best. The generator entry
targets a different FP32-output epilogue and is not directly transferable.
Restore three stages.

## Optimization 20: K=64 CUTLASS mainloop tile

### Motivation

A 64-wide mainloop tile performs four `16x8x16` K steps per stage and reduces
pipeline loop/control frequency. CUTLASS's generated SM80 configuration set
also includes K=64 variants.

### Modification

- Use threadblock shape `64x64x64`.
- Use warp shape `32x32x64`.
- Restore three pipeline stages.

### Verification

- Benchmark correctness passed; maximum absolute error `0.00195312`.
- Fused mean: `0.3345 ms`.
- PyTorch mean: `0.4593 ms`.
- Same-run speedup: `1.373x`.

### Decision

K=64 is 1.3% slower than the K=32 three-stage best. Restore K=32.

## Optimization 21: reuse CUDA synchronization events

### Motivation

The C++ binding creates and destroys two CUDA events on every invocation.
Because the benchmark records its start event before entering the binding,
CPU-side event management can leave the measured stream idle and inflate
end-to-end latency.

### Modification

- Add thread-local, device-aware cached input-ready and RMS-ready events.
- Re-record and reuse the events on each invocation.
- Preserve current-stream to RMS-stream and RMS-stream to current-stream
  dependencies exactly.

### Expected effect

Remove repeated CUDA event allocation/destruction overhead without changing
kernel execution or synchronization semantics.

### Verification

- RTX 3090 correctness: all 7 tests passed.
- Maximum absolute error: `0.00195312`, passed.
- Fused mean: `0.3296 ms`, p50 `0.3287 ms`, p95 `0.3328 ms`.
- PyTorch mean: `0.4568 ms`.
- Same-run speedup: `1.386x`.

### Decision

Event reuse provides a small improvement and is retained.

## Optimization 22: CUTLASS 8-way threadblock swizzle

### Motivation

CUTLASS's generated SM80 kernels use
`GemmIdentityThreadblockSwizzle<8>`, while the manually instantiated device
GEMM still uses the default swizzle. A larger swizzle can improve traversal and
L2 locality for the 4096-wide output grid.

### Modification

- Change threadblock swizzle from the default factor to factor 8.
- Keep the best `64x64x32`, `32x32x32`, three-stage configuration.

### Verification

- Benchmark correctness passed; maximum absolute error `0.00195312`.
- Fused mean: `0.3282 ms`.
- PyTorch mean: `0.4567 ms`.
- Same-run speedup: `1.392x`.

### Decision

Factor 8 is marginally better than the default. Sweep smaller non-default
factors before finalizing.

## Optimization 23: CUTLASS swizzle factor 4

### Verification

- Benchmark correctness passed; maximum absolute error `0.00195312`.
- Fused mean: `0.3295 ms`.
- PyTorch mean: `0.4579 ms`.
- Same-run speedup: `1.390x`.

### Decision

Factor 4 is slightly slower than factor 8. Restore factor 8.

## Optimization 24: in-place RMS scaling

### Motivation

The CUTLASS projection and scaled output have identical shape and dtype.
Allocating a second 4 MiB output tensor is unnecessary because each epilogue
thread reads and overwrites exactly one projection element.

### Modification

- Remove the separate output allocation.
- Pass the projection buffer as both epilogue input and output.
- Return the in-place scaled projection.
- Preserve bias and RMS arithmetic.

### Expected effect

Remove one caching-allocator operation and one live 4 MiB buffer without
changing the number of epilogue reads/writes.

### Verification

- RTX 3090 correctness: all 7 tests passed.
- Maximum absolute error: `0.00195312`, passed.
- Fused mean: `0.3257 ms`, p50 `0.3248 ms`, p95 `0.3277 ms`.
- PyTorch mean: `0.4558 ms`.
- Same-run speedup: `1.399x`.

### Nsight Systems breakdown

- CUTLASS GEMM: `280.4 us` average in the traced run.
- RMS square-sum: `13.9 us` average and overlapped with GEMM.
- In-place RMS scale: `16.7 us` average.
- The GEMM is the dominant remaining cost.

## Optimization 25: generator-shaped `128x64`, six-stage GEMM

### Motivation

The CUTLASS generator includes a `128x64x32` six-stage SM80 candidate. For the
default shape it launches 256 CTAs, avoiding the `128x128` grid-underfill
problem while doubling weight reuse across M relative to the `64x64` best.

### Modification

- Threadblock shape: `128x64x32`.
- Warp shape: `64x32x32`.
- Pipeline stages: six.
- Preserve factor-8 swizzle and FP16-output/FP32-accumulate epilogue.

### Verification

- Benchmark correctness passed; maximum absolute error `0.00195312`.
- Fused mean: `0.3479 ms`.
- PyTorch mean: `0.4591 ms`.
- Same-run speedup: `1.320x`.

### Decision

The six-stage `128x64` candidate is slower. Restore the best `64x64x32`,
three-stage, factor-8-swizzle GEMM.

## Optimization 26: vectorized `half2` scale epilogue

### Motivation

The scalar scale kernel spends about `16.7 us` and computes the same token RMS
for adjacent output elements. OutDim is guaranteed divisible by 16, so pairs
are always aligned and remain within one token row.

### Modification

- Process two projection elements per thread with `half2`.
- Compute token indexing and inverse RMS once per pair.
- Vectorize optional bias load and final FP16 store.
- Halve the epilogue element count and grid work.

### Verification

- RTX 3090 correctness: all 7 tests passed.
- Maximum absolute error: `0.00195312`.
- Fused mean: `0.3211 ms`.
- PyTorch mean: `0.4582 ms`.
- Same-run speedup: `1.427x`.

## Optimization 27: CUTLASS SM80 EVT row-scale epilogue

### Motivation

The no-bias fast path still wrote an unscaled 4 MiB projection, launched an
elementwise kernel, then read and rewrote the full projection. The RMS result
is one scalar per token and is naturally a column-broadcast operand in a
row-major GEMM epilogue.

### Modification

- Add a half2/FP32 RMS reduction that directly produces `inv_rms`.
- Replace the no-bias GEMM epilogue with a CUTLASS SM80 epilogue visitor tree:
  `FP32 accumulator * VisitorColBroadcast(inv_rms) -> FP16 output`.
- Retain the `64x64x32` threadblock, `32x32x32` warp tile,
  `16x8x16` MMA instruction, three pipeline stages, and factor-8 swizzle.
- Reduce the no-bias execution from three kernels to two and write the output
  matrix only once.
- Keep the existing overlapped three-kernel path for bias so its
  `scaled_projection + bias` semantics remain unchanged.

### Verification

- RTX 3090 correctness: all 7 tests passed.
- Maximum absolute error: `0.00195312`.
- Fused mean: `0.3228 ms`, p50 `0.3215 ms`, p95 `0.3308 ms`.
- PyTorch mean: `0.4818 ms`.
- Same-run speedup: `1.492x`.

### Decision

Keep the EVT path because it removes a full projection round trip and one
launch, even though its measured kernel latency is statistically flat versus
the `half2` three-kernel path (`0.3228 ms` versus `0.3211 ms`). The reason is
that RMS must finish before a conventional CUTLASS epilogue can consume
`inv_rms`; the saved `~16.7 us` scale kernel is offset by losing overlap with
the `~13.9 us` RMS reduction. A true one-launch implementation needs a custom
persistent/cooperative GEMM scheduler or redundant per-CTA RMS work; the latter
would reread X once per N tile and is not competitive for N=4096.

## Optimization 28: preserve CUTLASS GEMM and co-allocate RMS workspace

### Motivation

Nsight Systems measured the active CUTLASS EVT GEMM at `279.5 us`, while the
same run's cuBLAS/PyTorch GEMM kernel took `319.1 us`. Replacing or burdening
the CUTLASS mainloop would therefore violate the GEMM-first requirement. The
remaining safe target was host/allocation overhead around the GEMM.

### Modification

- Keep the `64x64x32`, three-stage Tensor Core mainloop and EVT epilogue
  unchanged.
- Allocate one FP16 storage object containing the output followed by an aligned
  tail large enough for the per-token FP32 RMS workspace.
- Return a contiguous `[Batch, SeqLen, OutDim]` view over the output prefix.
- Use the tail workspace for both the serialized no-bias `inv_rms` path and the
  concurrent biased square-sum path.

### Verification

- RTX 3090 correctness: all 7 tests passed.
- No bias: `0.3171 ms`, p50 `0.3164 ms`, p95 `0.3204 ms`, `54.179 TFLOPS`.
- No-bias speedup versus PyTorch: `1.442x`.
- Bias: `0.3252 ms`, p50 `0.3244 ms`, p95 `0.3295 ms`, `52.830 TFLOPS`.
- Biased speedup versus PyTorch: `1.406x`.
- No-bias latency improves by about `1.8%` versus Optimization 27's
  `0.3228 ms`, without changing Tensor Core utilization, register allocation,
  shared-memory allocation, or GEMM launch geometry.

## Optimization 29: Ampere tile and warp-shape completion sweep

### Constraint

RTX 3090 is SM86. WGMMA, TMA, and warp-group specialization require SM90 and
cannot execute on this GPU. The applicable mechanisms remain `mma.sync`,
`cp.async` multistaging, and warp-level scheduling, all already provided by the
CUTLASS SM80 mainloop.

### Profiling baseline

- Active CUTLASS EVT GEMM: `279.5 us` in Nsight Systems.
- Same-run cuBLAS/PyTorch GEMM: `319.1 us`.
- The fused GEMM therefore already exceeds the available cuBLAS kernel for
  `M=512, K=N=4096`; preserving it is a hard constraint.

### Candidates

- `128x64x32` CTA, `64x32x32` warp, three stages: `0.3567 ms`.
- `64x128x32` CTA, `32x64x32` warp, three stages: `0.3616 ms`.
- `64x64x32` CTA, `64x32x32` warp, two warps, three stages: `0.3655 ms`.

All candidates passed the benchmark's absolute-error check but regressed from
the `64x64x32` CTA, `32x32x32` warp, four-warp result of `0.3171 ms`. Restore
the four-warp configuration. The results show that larger accumulator tiles
and reduced warp-level parallelism cost more register/scheduling capacity than
they recover through operand reuse.

### Three-times target audit

The measured PyTorch reference is about `0.457 ms`; a `3x` result requires
`<=0.152 ms`. That target is only about 54% of the measured standalone cuBLAS
GEMM time and about 54% of the current CUTLASS GEMM kernel time, before adding
RMS or launch overhead. It is therefore not reachable by RMS fusion or
epilogue optimization on this RTX 3090 for the specified shape; it requires a
fundamentally faster GEMM than both measured vendor and CUTLASS kernels.

## Current implementation summary

### Mathematical transformation

The layer folds the RMSNorm affine weight into the linear weight once:

```text
W_fused = W * rms_weight[None, :]
```

The runtime uses the scalar property of RMS normalization:

```text
(x / rms(x)) @ W_fused.T = (x @ W_fused.T) * inv_rms(x)
```

It therefore never materializes a normalized activation tensor, removing a
full activation write/read and the kernel sequence between RMSNorm and GEMM.

### Active no-bias path

1. One 256-thread CTA per token reads X with aligned `half2` accesses, sums
   squares in FP32, performs warp-shuffle plus 32-byte shared-memory reduction,
   and writes one FP32 `inv_rms` value.
2. A CUTLASS SM80 TensorOp GEMM computes `X @ W_fused.T` with FP32
   accumulation.
3. A CUTLASS SM80 EVT epilogue uses `VisitorColBroadcast` to multiply each
   accumulator row by its token's `inv_rms`, converts once to FP16, and performs
   the only global output store.

This uses two launches but has no normalized-activation tensor, no unscaled
projection tensor, and no projection read/modify/write round trip.

### Tensor Core configuration

- SM80 TensorOp kernel on RTX 3090 SM86: FP16 inputs, FP32 accumulators and
  Ampere `mma.sync` instructions.
- CTA tile `64x64x32`; warp tile `32x32x32`; instruction `16x8x16`.
- Four warps / 128 threads per GEMM CTA.
- Three-stage CUTLASS multistage mainloop with Ampere asynchronous operand
  staging (`cp.async`).
- Eight-way threadblock swizzle for traversal and L2 locality.
- Eight-element vector alignment for A, B, and output accesses.

This is the best measured compromise between reuse and occupancy. Larger
CTA/warp tiles raise accumulator-register/shared-memory pressure or reduce
warp-level parallelism; two stages expose memory latency, while deeper
pipelines consume too much shared memory.

### Memory and allocation design

- The layer folds the RMSNorm affine vector into W once and stores the result
  contiguous as physical `[OutDim, HiddenDim]`, matching CUTLASS's logical
  column-major B view.
- X and fused W feed the Tensor Core mainloop directly.
- The output allocation has an aligned tail for the per-token FP32 RMS
  workspace, reducing two allocator operations to one; the returned prefix is
  still contiguous `[Batch, SeqLen, OutDim]`.
- The no-bias epilogue scales FP32 accumulators before FP16 rounding and writes
  each output element once.
- The bias path retains concurrent RMS/GEMM streams and a vectorized `half2`
  final epilogue so bias remains mathematically after normalization.

### Why it is fast

1. The dominant GEMM remains on Tensor Cores. Nsight Systems measured the
   active CUTLASS EVT GEMM at `279.5 us`, versus `319.1 us` for the same-run
   cuBLAS/PyTorch GEMM kernel at the default shape.
2. Folding the RMSNorm affine weight into W removes that multiply at runtime.
3. The normalized activation and unscaled projection intermediates are both
   eliminated on the no-bias path.
4. FP32 reduction/accumulation meets the `0.002` absolute-error bound, while
   `half2` and aligned vector accesses keep memory transactions coalesced.

### Prefill and decode behavior

The implementation flattens `[Batch, SeqLen, HiddenDim]` to GEMM M, so the
same path accelerates both workloads:

- **Prefill:** large M supplies enough CTAs to occupy the GPU and amortizes
  fixed overhead. At `B=4, S=128, M=512, K=N=4096`, the best verified result is
  about `0.317-0.318 ms`, approximately `1.44x` over PyTorch.
- **Decode-like input:** small M underutilizes the 64-row GEMM tile in absolute
  terms, but PyTorch's separate RMSNorm kernels, normalized-tensor traffic,
  dispatch, and allocation overhead become a much larger fraction of latency.
  Removing those fixed costs produces a larger relative speedup even though
  absolute Tensor Core utilization is lower.

An RTX 3090 reproduction with `B=4, S=1, M=4, K=N=4096` measured:

```text
Fused mean:   0.0889 ms
PyTorch mean: 0.1670 ms
Speedup:      1.880x
Max abs err:  0.00195312
```

Other runs have reached or exceeded approximately `1.9x`; this is a shape- and
system-dependent observation rather than a universal guarantee. The stronger
decode speedup comes from removing fixed overhead and intermediate traffic,
not from higher small-M Tensor Core efficiency. RTX 3090 cannot execute
WGMMA, TMA, or Hopper warp-group specialization.

### Current verified status

- Correctness suite: 7 tests passed on RTX 3090.
- Standard benchmark maximum absolute error: `0.00195312`.
- Default prefill effective throughput: approximately `54 TFLOPS`.
- This is the fastest configuration verified in this project, though not a
  proof of global optimality across every possible SM86 schedule.

## Unified layer-level dispatch

`FusedRMSNormLinear` is now the single inference entry point for decode and
prefill. It computes `M = Batch * SeqLen` in Python and dispatches without
changing either validated CUDA implementation:

```text
M < 8   -> existing small-M fused_rmsnorm_linear extension
M >= 8  -> CUTLASS Tensor Core prefill extension
```

For the decode path, contiguous `[B,S,K]` is viewed as `[M,1,K]` without a
copy, passed to the original kernel, and its output is viewed back as
`[B,S,N]`. Tensor-Core-compatible dimensions (K and N divisible by 16) follow
the threshold exactly. Legacy non-aligned shapes retain the decode path where
it is supported, because the prefill CUTLASS iterators require 16-element
alignment.

`FusedRMSNormLinearPrefill` remains as a backward-compatible subclass of the
unified layer, including its original constructor-time alignment validation.
This preserves existing imports and state-dict buffer names while avoiding a
second implementation of weight folding and input validation.

### Dispatch verification

- Unit dispatch: `M=7` selected decode and reshaped to `[7,1,K]`.
- Unit dispatch: `M=8` selected prefill and retained `[B,S,K]`.
- Combined decode/prefill correctness: 19 tests passed on RTX 3090.
- Decode `M=4, K=N=4096`: `0.0786 ms` versus PyTorch `0.1715 ms`, `2.182x`.
- Prefill `M=512, K=N=4096`: `0.3185 ms` versus PyTorch `0.4574 ms`, `1.436x`.
- Both benchmark paths retained maximum absolute error `0.00195312` where
  reported by the benchmark.
