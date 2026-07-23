"""Benchmark FusedRMSNormLinear against eager PyTorch RMSNorm + Linear."""

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
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def _report(name: str, timings: list[float], operations: int) -> None:
    mean_ms = statistics.mean(timings)
    effective_tflops = operations / (mean_ms / 1000.0) / 1.0e12
    print(
        f"{name:28s} mean={mean_ms:.4f} ms  "
        f"p50={_percentile(timings, 0.50):.4f} ms  "
        f"p95={_percentile(timings, 0.95):.4f} ms  "
        f"effective_TFLOPS={effective_tflops:.3f}"
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
        help="benchmark a Linear with bias (default: --no-bias)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch <= 0 or args.hidden_dim <= 0 or args.out_dim <= 0:
        raise ValueError("batch, hidden-dim, and out-dim must be positive")

    torch.manual_seed(20260714)
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
    fused_layer = FusedRMSNormLinear(rmsnorm, linear).eval()
    x = torch.randn(
        args.batch,
        1,
        args.hidden_dim,
        device=device,
        dtype=dtype,
    ) / 4

    def fused_forward() -> torch.Tensor:
        return fused_layer(x)

    def pytorch_forward() -> torch.Tensor:
        return linear(rmsnorm(x))

    with torch.inference_mode():
        fused_check = fused_forward()
        pytorch_check = pytorch_forward()
        torch.testing.assert_close(
            fused_check,
            pytorch_check,
            atol=1e-2,
            rtol=1e-2,
        )

        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}")
        print(
            f"B={args.batch}, K={args.hidden_dim}, N={args.out_dim}, "
            f"eps={args.eps:g}, bias={args.bias}"
        )
        print(f"fused weight shape: {tuple(fused_layer.fused_weight.shape)}")

        fused_output, fused_times = _measure(
            fused_forward,
            args.warmup,
            args.iterations,
        )
        pytorch_output, pytorch_times = _measure(
            pytorch_forward,
            args.warmup,
            args.iterations,
        )
        torch.testing.assert_close(
            fused_output,
            pytorch_output,
            atol=1e-2,
            rtol=1e-2,
        )

    # Count the dominant Linear operations. RMSNorm and the fused auxiliary
    # reduction are intentionally excluded so both rows use the same metric.
    operations = 2 * args.batch * args.hidden_dim * args.out_dim
    _report("FusedRMSNormLinear", fused_times, operations)
    _report("PyTorch RMSNorm + Linear", pytorch_times, operations)
    print(
        "speedup vs PyTorch: "
        f"{statistics.mean(pytorch_times) / statistics.mean(fused_times):.3f}x"
    )


if __name__ == "__main__":
    main()
