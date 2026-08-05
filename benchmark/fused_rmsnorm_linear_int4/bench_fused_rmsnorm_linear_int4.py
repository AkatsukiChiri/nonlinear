"""Benchmark unified decode/prefill INT4 fused RMSNorm + Linear."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear import FusedRMSNormLinear
from layers.fused_rmsnorm_linear_int4 import (
    PREFILL_MIN_TOKENS,
    FusedRMSNormLinearInt4,
)
from layers.fused_rmsnorm_linear_int4_prefill import FusedRMSNormLinearInt4Prefill
from layers.fused_rmsnorm_linear_weight_only_int4_dequant import (
    FusedRMSNormLinearWeightOnlyInt4Dequant,
)


def measure(
    function: Callable[[], torch.Tensor], warmup: int, iterations: int
) -> tuple[torch.Tensor, list[float]]:
    for _ in range(warmup):
        output = function()
    torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return output, timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--out-dim", type=int, default=4096)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 8:
        raise RuntimeError("Ampere GPU is required")
    if args.batch <= 0 or args.seq_len <= 0:
        raise ValueError("batch and seq-len must be positive")

    torch.manual_seed(20260806)
    rmsnorm = nn.RMSNorm(
        args.hidden_dim, eps=args.eps, device="cuda", dtype=torch.float16
    ).eval()
    linear = nn.Linear(
        args.hidden_dim,
        args.out_dim,
        bias=args.bias,
        device="cuda",
        dtype=torch.float16,
    ).eval()
    with torch.no_grad():
        rmsnorm.weight.normal_(1.0, 0.1)
    int4_layer = FusedRMSNormLinearInt4(rmsnorm, linear).eval()
    fp16_layer = FusedRMSNormLinear(rmsnorm, linear).eval()
    rows = args.batch * args.seq_len
    direct_layer: nn.Module
    if rows < PREFILL_MIN_TOKENS:
        direct_layer = FusedRMSNormLinearWeightOnlyInt4Dequant(
            rmsnorm, linear
        ).eval()
    else:
        direct_layer = FusedRMSNormLinearInt4Prefill(rmsnorm, linear).eval()
    x = torch.randn(
        args.batch,
        args.seq_len,
        args.hidden_dim,
        device="cuda",
        dtype=torch.float16,
    ) / 4
    functions = {
        "INT4 unified": lambda: int4_layer(x),
        "INT4 direct": lambda: (
            direct_layer(x)
            if rows >= PREFILL_MIN_TOKENS or args.seq_len == 1
            else direct_layer(x.reshape(rows, 1, args.hidden_dim)).reshape(
                args.batch, args.seq_len, args.out_dim
            )
        ),
        "FP16 fused": lambda: fp16_layer(x),
        "PyTorch FP16": lambda: linear(rmsnorm(x)),
    }
    results: dict[str, list[float]] = {}
    with torch.inference_mode():
        int4_output = functions["INT4 unified"]()
        pytorch_output = functions["PyTorch FP16"]()
        max_error = (int4_output.float() - pytorch_output.float()).abs().max().item()
        for name, function in functions.items():
            _, results[name] = measure(function, args.warmup, args.iterations)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"B={args.batch}, S={args.seq_len}, M={rows}, K={args.hidden_dim}, "
        f"N={args.out_dim}, dispatch={'prefill' if rows >= PREFILL_MIN_TOKENS else 'decode'}"
    )
    print(f"max absolute error vs FP16 PyTorch: {max_error:.6g}")
    for name, timings in results.items():
        print(
            f"{name:20s} mean={statistics.mean(timings):.4f} ms  "
            f"p50={statistics.median(timings):.4f} ms"
        )
    int4_mean = statistics.mean(results["INT4 unified"])
    direct_mean = statistics.mean(results["INT4 direct"])
    print(f"unified/direct latency ratio: {int4_mean / direct_mean:.3f}x")
    print(
        f"speedup vs FP16 fused: {statistics.mean(results['FP16 fused']) / int4_mean:.3f}x"
    )
    print(
        f"speedup vs PyTorch: {statistics.mean(results['PyTorch FP16']) / int4_mean:.3f}x"
    )


if __name__ == "__main__":
    main()
