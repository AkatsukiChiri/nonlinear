"""Benchmark prefill-only INT4 fused RMSNorm + Linear."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear.prefill import FusedRMSNormLinearPrefill
from layers.fused_rmsnorm_linear_int4_prefill import FusedRMSNormLinearInt4Prefill
from layers.fused_rmsnorm_linear_int8_prefill import FusedRMSNormLinearInt8Prefill


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


def report(name: str, timings: list[float], operations: int) -> None:
    mean = statistics.mean(timings)
    print(
        f"{name:24s} mean={mean:.4f} ms  p50={statistics.median(timings):.4f} ms  "
        f"effective_TOPS={operations / (mean / 1000) / 1e12:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--out-dim", type=int, default=4096)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 8:
        raise RuntimeError("Ampere GPU is required")
    torch.manual_seed(20260805)
    rmsnorm = nn.RMSNorm(
        args.hidden_dim, eps=args.eps, device="cuda", dtype=torch.float16
    ).eval()
    linear = nn.Linear(
        args.hidden_dim, args.out_dim, bias=False,
        device="cuda", dtype=torch.float16,
    ).eval()
    with torch.no_grad():
        rmsnorm.weight.normal_(1.0, 0.1)
    int4_layer = FusedRMSNormLinearInt4Prefill(rmsnorm, linear).eval()
    int8_layer = FusedRMSNormLinearInt8Prefill(rmsnorm, linear).eval()
    fp16_layer = FusedRMSNormLinearPrefill(rmsnorm, linear).eval()
    x = torch.randn(
        args.batch, args.seq_len, args.hidden_dim,
        device="cuda", dtype=torch.float16,
    ) / 4
    functions = {
        "INT4 prefill fused": lambda: int4_layer(x),
        "INT8 prefill fused": lambda: int8_layer(x),
        "FP16 prefill fused": lambda: fp16_layer(x),
        "PyTorch FP16": lambda: linear(rmsnorm(x)),
    }
    results: dict[str, list[float]] = {}
    with torch.inference_mode():
        int4_output = functions["INT4 prefill fused"]()
        pytorch_output = functions["PyTorch FP16"]()
        max_error = (int4_output.float() - pytorch_output.float()).abs().max().item()
        for name, function in functions.items():
            _, results[name] = measure(function, args.warmup, args.iterations)
    operations = 2 * args.batch * args.seq_len * args.hidden_dim * args.out_dim
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"B={args.batch}, S={args.seq_len}, M={args.batch * args.seq_len}, "
        f"K={args.hidden_dim}, N={args.out_dim}"
    )
    print(f"max absolute error vs FP16 PyTorch: {max_error:.6g}")
    for name, timings in results.items():
        report(name, timings, operations)
    int4_mean = statistics.mean(results["INT4 prefill fused"])
    print(
        f"speedup vs INT8: {statistics.mean(results['INT8 prefill fused']) / int4_mean:.3f}x"
    )
    print(
        f"speedup vs FP16 fused: {statistics.mean(results['FP16 prefill fused']) / int4_mean:.3f}x"
    )
    print(
        f"speedup vs PyTorch: {statistics.mean(results['PyTorch FP16']) / int4_mean:.3f}x"
    )


if __name__ == "__main__":
    main()
