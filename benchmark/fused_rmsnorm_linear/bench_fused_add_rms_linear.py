"""Benchmark fused residual add + RMSNorm + Linear decode inference."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear import (
    FusedAddRMSNormLinear,
    FusedRMSNormLinear,
)


OutputPair = tuple[torch.Tensor, torch.Tensor]


def _measure(
    function: Callable[[], OutputPair],
    warmup: int,
    iterations: int,
) -> tuple[OutputPair, list[float]]:
    output: OutputPair
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


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def _report(name: str, timings: list[float], operations: int) -> None:
    mean_ms = statistics.mean(timings)
    effective_tflops = operations / (mean_ms / 1000.0) / 1.0e12
    print(
        f"{name:34s} mean={mean_ms:.4f} ms  "
        f"p50={_percentile(timings, 0.50):.4f} ms  "
        f"p95={_percentile(timings, 0.95):.4f} ms  "
        f"effective_TFLOPS={effective_tflops:.3f}"
    )


def _check_pair(actual: OutputPair, expected: OutputPair) -> None:
    torch.testing.assert_close(
        actual[0],
        expected[0],
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual[1],
        expected[1],
        atol=0,
        rtol=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--out-dim", type=int, default=4096)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--bias",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 1 <= args.batch <= 8:
        raise ValueError("batch must be in [1, 8]")
    if args.hidden_dim <= 0 or args.out_dim <= 0:
        raise ValueError("hidden-dim and out-dim must be positive")

    torch.manual_seed(20260724)
    device = torch.device("cuda")
    dtype = torch.float16
    rmsnorm = nn.RMSNorm(
        args.hidden_dim,
        eps=args.eps,
        device=device,
        dtype=dtype,
    ).eval()
    linear = nn.Linear(
        args.hidden_dim,
        args.out_dim,
        bias=args.bias,
        device=device,
        dtype=dtype,
    ).eval()
    with torch.no_grad():
        rmsnorm.weight.normal_(mean=1.0, std=0.1)

    fused_add_layer = FusedAddRMSNormLinear(rmsnorm, linear).eval()
    fused_norm_layer = FusedRMSNormLinear(rmsnorm, linear).eval()
    x = torch.randn(
        args.batch,
        1,
        args.hidden_dim,
        device=device,
        dtype=dtype,
    ) / 4
    x_res = torch.randn_like(x) / 4

    def fused_add_forward() -> OutputPair:
        return fused_add_layer(x, x_res)

    def separate_add_fused_norm_forward() -> OutputPair:
        residual_sum = x + x_res
        return fused_norm_layer(residual_sum), residual_sum

    def pytorch_forward() -> OutputPair:
        residual_sum = x + x_res
        return linear(rmsnorm(residual_sum)), residual_sum

    with torch.inference_mode():
        fused_check = fused_add_forward()
        separate_check = separate_add_fused_norm_forward()
        pytorch_check = pytorch_forward()
        _check_pair(fused_check, pytorch_check)
        _check_pair(separate_check, pytorch_check)

        max_abs_error = (
            fused_check[0].float() - pytorch_check[0].float()
        ).abs().max().item()
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}")
        print(
            f"B={args.batch}, K={args.hidden_dim}, N={args.out_dim}, "
            f"eps={args.eps:g}, bias={args.bias}"
        )
        print(f"max absolute output error: {max_abs_error:.6g}")

        fused_output, fused_times = _measure(
            fused_add_forward,
            args.warmup,
            args.iterations,
        )
        separate_output, separate_times = _measure(
            separate_add_fused_norm_forward,
            args.warmup,
            args.iterations,
        )
        pytorch_output, pytorch_times = _measure(
            pytorch_forward,
            args.warmup,
            args.iterations,
        )
        _check_pair(fused_output, pytorch_output)
        _check_pair(separate_output, pytorch_output)

    operations = 2 * args.batch * args.hidden_dim * args.out_dim
    _report("FusedAddRMSNormLinear", fused_times, operations)
    _report("Add + FusedRMSNormLinear", separate_times, operations)
    _report("PyTorch Add + RMSNorm + Linear", pytorch_times, operations)
    fused_mean = statistics.mean(fused_times)
    print(
        "speedup vs Add + FusedRMSNormLinear: "
        f"{statistics.mean(separate_times) / fused_mean:.3f}x"
    )
    print(
        "speedup vs PyTorch: "
        f"{statistics.mean(pytorch_times) / fused_mean:.3f}x"
    )


if __name__ == "__main__":
    main()
