"""Steady-state decode benchmark on an AWQ Llama projection.

This compares the persistent single-launch fused layer with vLLM's RMSNorm
plus AWQ-Marlin using the identical checkpoint tensors.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.fused_rmsnorm_linear_w4a16_marlin.baseline import (
    DEFAULT_MODEL,
    load_tensors,
    make_native_marlin,
)
from layers.fused_rmsnorm_linear_w4a16_marlin.layer import (
    FusedRMSNormLinearW4A16Marlin,
)


def _measure(function: Callable[[], object], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iterations):
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--projection", choices=("q", "gate"), default="q")
    parser.add_argument("--batch", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--stripe-n", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    qweight, qzeros, scales, gamma = load_tensors(
        args.model, 0, args.projection
    )
    qweight, qzeros, scales, gamma = (
        qweight.cuda(), qzeros.cuda(), scales.cuda(), gamma.cuda()
    )
    hidden_dim = qweight.shape[0]
    out_dim = scales.shape[1]
    group_size = hidden_dim // scales.shape[0]
    x = torch.randn(args.batch, 1, hidden_dim, device="cuda", dtype=torch.float16)
    fused = FusedRMSNormLinearW4A16Marlin(
        qweight, qzeros, scales, gamma, 1e-5, stripe_n=args.stripe_n
    ).eval()

    native = make_native_marlin(qweight, qzeros, scales, group_size)
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_awq_marlin_linear,
    )
    from vllm.scalar_type import scalar_types

    norm = RMSNorm(hidden_dim, eps=1e-5, dtype=torch.float16).cuda().eval()
    with torch.no_grad():
        norm.weight.copy_(gamma)

    def fused_forward() -> torch.Tensor:
        return fused(x)

    def vllm_forward() -> torch.Tensor:
        return apply_awq_marlin_linear(
            input=norm(x[:, 0]), weight=native[0], weight_scale=native[1],
            weight_zp=native[2], g_idx=native[3],
            g_idx_sort_indices=native[4], workspace=native[5],
            quant_type=scalar_types.uint4, output_size_per_partition=out_dim,
            input_size_per_partition=hidden_dim,
        )

    with torch.inference_mode():
        fused_output = fused_forward()
        native_output = vllm_forward().unsqueeze(1)
        cosine = torch.nn.functional.cosine_similarity(
            fused_output.float().flatten(), native_output.float().flatten(), dim=0
        ).item()
        fused_ms = _measure(fused_forward, args.warmup, args.iterations)
        vllm_ms = _measure(vllm_forward, args.warmup, args.iterations)
    print(
        f"projection={args.projection} batch={args.batch} stripe_n={args.stripe_n} "
        f"shape={hidden_dim}x{out_dim} fused_p50_ms={fused_ms:.4f} "
        f"vllm_p50_ms={vllm_ms:.4f} speedup={vllm_ms / fused_ms:.3f}x "
        f"cosine={cosine:.7f}"
    )


if __name__ == "__main__":
    main()
