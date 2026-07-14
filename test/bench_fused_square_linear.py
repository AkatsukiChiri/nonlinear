"""Benchmark the fused square-and-linear kernel against PyTorch."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layers.fused_square_linear import fused_square_linear_fp16


def _measure(
    function: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
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


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)]


def _report(name: str, timings: list[float], operations: int) -> None:
    mean_ms = statistics.mean(timings)
    tflops = operations / (mean_ms / 1000.0) / 1.0e12
    print(
        f"{name:24s} mean={mean_ms:.4f} ms  "
        f"p50={_percentile(timings, 0.50):.4f} ms  "
        f"p95={_percentile(timings, 0.95):.4f} ms  "
        f"effective_TFLOPS={tflops:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--out-dim", type=int, default=4097)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260713)
    x = torch.randn(
        args.batch, 1, args.hidden_dim, device="cuda", dtype=torch.float16
    ) / 4
    weight = torch.randn(
        args.out_dim, args.hidden_dim, device="cuda", dtype=torch.float16
    ) / 4

    def pytorch_unfused() -> torch.Tensor:
        expanded = torch.stack((x[:, 0], x[:, 0].square()), dim=1)
        return torch.matmul(expanded, weight.t())

    def pytorch_two_matmuls() -> torch.Tensor:
        first = torch.matmul(x[:, 0], weight.t())
        second = torch.matmul(x[:, 0].square(), weight.t())
        return torch.stack((first, second), dim=1)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}")
    print(f"B={args.batch}, K={args.hidden_dim}, N={args.out_dim}")
    backend = (
        "CUDA Core SIMT"
        if args.batch <= 8 or args.batch == 16
        else "CuTe Tensor Core"
    )
    print(f"custom backend: {backend}")

    fused_output, fused_times = _measure(
        lambda: fused_square_linear_fp16(x, weight), args.warmup, args.iterations
    )
    unfused_output, unfused_times = _measure(
        pytorch_unfused, args.warmup, args.iterations
    )
    two_output, two_times = _measure(
        pytorch_two_matmuls, args.warmup, args.iterations
    )
    torch.testing.assert_close(fused_output, unfused_output, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(fused_output, two_output, atol=1e-2, rtol=1e-2)

    operations = 4 * args.batch * args.out_dim * args.hidden_dim
    _report("fused custom", fused_times, operations)
    _report("PyTorch stack + matmul", unfused_times, operations)
    _report("PyTorch two matmuls", two_times, operations)
    print(
        "speedup vs stack+matmul: "
        f"{statistics.mean(unfused_times) / statistics.mean(fused_times):.3f}x"
    )


if __name__ == "__main__":
    main()
