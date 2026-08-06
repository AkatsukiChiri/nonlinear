"""Benchmark packed weight-only INT4 fused RMSNorm + Linear decode."""

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
from layers.fused_rmsnorm_linear_weight_only_int4_dequant import (
    FusedRMSNormLinearWeightOnlyInt4Dequant,
)
from layers.fused_rmsnorm_linear_weight_only_int8_dequant import (
    FusedRMSNormLinearInt8Decode,
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


def report(name: str, timings: list[float]) -> None:
    print(
        f"{name:28s} mean={statistics.mean(timings):.4f} ms  "
        f"p50={statistics.median(timings):.4f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--out-dim", type=int, default=4096)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 1 <= args.batch <= 8:
        raise ValueError("batch must be in [1, 8]")
    if args.hidden_dim % 256:
        raise ValueError("hidden-dim must be divisible by 256")

    torch.manual_seed(20260805)
    rmsnorm = nn.RMSNorm(
        args.hidden_dim, eps=args.eps, device="cuda", dtype=torch.float16
    ).eval()
    linear = nn.Linear(
        args.hidden_dim, args.out_dim, bias=args.bias,
        device="cuda", dtype=torch.float16,
    ).eval()
    with torch.no_grad():
        rmsnorm.weight.normal_(1.0, 0.1)
    int4_layer = FusedRMSNormLinearWeightOnlyInt4Dequant(rmsnorm, linear).eval()
    int8_layer = FusedRMSNormLinearInt8Decode(rmsnorm, linear).eval()
    fp16_layer = FusedRMSNormLinear(rmsnorm, linear).eval()
    x = torch.randn(
        args.batch, 1, args.hidden_dim, device="cuda", dtype=torch.float16
    ) / 4
    functions = {
        "INT4 weight-only dequant": lambda: int4_layer(x),
        "INT8 weight-only dequant": lambda: int8_layer(x),
        "FP16 fused": lambda: fp16_layer(x),
        "PyTorch FP16": lambda: linear(rmsnorm(x)),
    }
    results: dict[str, list[float]] = {}
    with torch.inference_mode():
        int4_output = functions["INT4 weight-only dequant"]()
        pytorch_output = functions["PyTorch FP16"]()
        max_error = (int4_output - pytorch_output).abs().max().item()
        for name, function in functions.items():
            _, results[name] = measure(function, args.warmup, args.iterations)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"B={args.batch}, K={args.hidden_dim}, N={args.out_dim}, "
        f"bias={args.bias}, max_abs_error_vs_FP16={max_error:.6g}"
    )
    print(
        f"packed weight: {int4_layer.packed_weight.numel() / 2**20:.2f} MiB"
    )
    for name, timings in results.items():
        report(name, timings)
    int4_mean = statistics.mean(results["INT4 weight-only dequant"])
    print(
        "speedup vs INT8 dequant: "
        f"{statistics.mean(results['INT8 weight-only dequant']) / int4_mean:.3f}x"
    )
    print(
        "speedup vs FP16 fused: "
        f"{statistics.mean(results['FP16 fused']) / int4_mean:.3f}x"
    )
    print(
        "speedup vs PyTorch: "
        f"{statistics.mean(results['PyTorch FP16']) / int4_mean:.3f}x"
    )


if __name__ == "__main__":
    main()
