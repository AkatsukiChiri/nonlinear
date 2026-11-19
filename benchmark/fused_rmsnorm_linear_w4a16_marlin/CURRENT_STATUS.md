# W4A16 fused decode: recorded baseline

This note records the state before replacing the custom W4 data path with a
Marlin-derived one. Measurements are single decode projections from the
Llama-3.1 8B AWQ checkpoint on an RTX 3090, using FP16 input and steady-state
CUDA-event p50 timing. They are not full-model end-to-end latency results.

## Current implementation

The active layer is `FusedRMSNormLinearW4A16Marlin`. It is a single CUDA
kernel for `[Batch, 1, HiddenDim]`, with batch in `[1, 8]`.

1. At model load, AWQ W4 weights are expanded with one all-one sentinel column
   per physical N stripe and packed by
   `pack_awq_sentinel_fragment`. The packing is lossless and stores the four
   W4 values required by one MMA B fragment as a 16-bit lane record.
2. In each CTA, the active rows of an augmented A tile are generated directly:
   even rows are `x * gamma`; odd rows are `x * x`.
3. Normal output columns compute the linear projection. A sentinel output
   column computes `sum(x^2)`, letting the epilogue recover RMSNorm scale
   without an intermediate normalized activation.
4. The kernel uses FP16 `mma.m16n8k16` with FP32 accumulators. It supports
   N-stripe specializations 32/64/128 with 64/128/256 threads respectively.
5. K splitting is adaptive: N grids with at most 82 stripes use eight K
   splits; wider grids use four. Per-split FP32 partial tiles are reduced by
   the final CTA for each N stripe.
6. Only `2 * batch` valid augmented rows are generated and written to the
   accumulator staging buffer. The padded M rows still execute in MMA, but
   no longer incur A-tile construction or accumulator-store work.

## Verified results

| Projection | Batch | Selected N stripe | Fused p50 | vLLM p50 | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q, 4096x4096 | 1 | 64 | 0.0655 ms | 0.0860 ms | 1.31x |
| gate-up, 4096x14336 | 1 | 64 | 0.0849 ms | about 0.1075 ms | 1.27x |
| Q, 4096x4096 | 2 | 128 | 0.0666 ms | 0.0892 ms | 1.34x |
| gate-up, 4096x14336 | 2 | 64 | 0.0927 ms | 0.1085 ms | 1.17x |
| Q, 4096x4096 | 4 | 64 | 0.0614 ms | 0.1075 ms | 1.75x |
| Q, 4096x4096 | 8 | 128 | 0.0686 ms | 0.0850 ms | 1.24x |
| gate-up, 4096x14336 | 8 | 128 | 0.1096 ms | 0.1096 ms | 1.00x |

The bs=4 gate-up measurement needs a clean re-run on the current code before
it is treated as a baseline.

Correctness coverage currently passes `21` W4A16-Marlin tests on GPU. The
fragment pack also has a CPU lossless lane-mapping test.

## Why local tuning is no longer sufficient

Nsight Compute on gate-up, batch 1, N=64 after valid-row elimination reports
about 66.98 us kernel duration, 57% DRAM throughput, 57% SM throughput, 52.5%
achieved occupancy, and ALU as the highest-utilized pipeline. Before removing
invalid M-row construction, duration was about 89.63 us and ALU utilization
was about 64%.

The remaining gap cannot be attributed to one avoidable launch or one obvious
global-memory intermediate. The custom kernel still differs from vLLM Marlin
in the critical W4 execution path:

- It uses a simplified K-split-per-N-stripe scheduler rather than Marlin's
  K/N stripe partitioner and reduction strategy.
- Its W4 fragment loader, scale/zero handling, and shared-memory pipeline are
  custom. The offline fragment pack removed four nibble loads per fragment,
  but did not reproduce Marlin's complete register/shared-memory schedule.
- Its 128-thread N=64 CTAs cannot use the same warp-level work partition and
  pipeline depth as Marlin's common 256-thread AWQ configuration.
- For small batch, the Tensor Core instruction still pads M to 16. The invalid
  row setup is removed, but the padded MMA work remains architectural overhead.

Therefore, reaching the project target (more than 2x vLLM across both Llama2
7B and Llama3 8B, batches 1--8, and all relevant projections) requires a
Marlin-derived fused kernel: preserve upstream AWQ U4 packed loading,
fragment dequantization, CTA scheduler, and reduction flow; replace only its
A-side fetch with CTA-generated `[x * gamma, x^2]`; and add sentinel-aware
RMSNorm work to its epilogue.

## Experiments intentionally not retained

- Global sentinel / two-kernel variants: retain intermediate global-memory or
  launch costs inconsistent with the single-kernel objective.
- N=128 for batch-1 gate-up: slower than N=64.
- 16-way K split for narrow Q: slower than eight-way due to extra partial
  traffic and final reduction.
- Two-way K split for wide gate-up: slower than four-way due to lower CTA
  parallelism.
- FP16 partial workspace: no measured win over FP32 partials.
- Precomputed `-zero * scale` FMA dequantization: correct but no measurable
  speedup versus the existing zero/subtract/scale path.
- Shared staging of scale/zero metadata: correct but slower because L1 already
  captures the reuse and staging adds synchronization.
