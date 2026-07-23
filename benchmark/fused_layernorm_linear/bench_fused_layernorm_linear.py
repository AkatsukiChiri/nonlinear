"""Benchmark fused LayerNorm + Linear against eager PyTorch."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_layernorm_linear import FusedLayerNormLinear


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


def _report_error(actual: torch.Tensor, expected: torch.Tensor) -> None:
    absolute_error = (actual.float() - expected.float()).abs()
    print(
        "correctness: "
        f"max_abs_error={absolute_error.max().item():.8e}, "
        f"mean_abs_error={absolute_error.mean().item():.8e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="decode batch sizes to benchmark (default: 1 2 4 8)",
    )
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--out-dim", type=int, default=4096)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use a bias in the source Linear (default: --bias)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        raise RuntimeError("an Ampere GPU is required")
    if not args.batches or any(
        batch < 1 or batch > 8 for batch in args.batches
    ):
        raise ValueError("every batch must be in [1, 8]")
    if args.hidden_dim <= 0 or args.out_dim <= 0:
        raise ValueError("hidden-dim and out-dim must be positive")

    torch.manual_seed(20260723)
    device = torch.device("cuda")
    dtype = torch.float16
    layernorm = nn.LayerNorm(
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
        layernorm.weight.normal_(mean=1.0, std=0.1)
        layernorm.bias.normal_(mean=0.0, std=0.1)
    fused_layer = FusedLayerNormLinear(layernorm, linear).eval()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}")
    print(
        f"K={args.hidden_dim}, N={args.out_dim}, eps={args.eps:g}, "
        f"linear_bias={args.bias}, batches={args.batches}"
    )
    print(
        "buffers: "
        f"fused_weight={tuple(fused_layer.fused_weight.shape)} FP16, "
        f"weight_sum={tuple(fused_layer.weight_sum.shape)} FP16, "
        f"fused_bias={tuple(fused_layer.fused_bias.shape)} FP16"
    )

    speedups: list[tuple[int, float]] = []
    with torch.inference_mode():
        for batch in args.batches:
            x = (
                torch.randn(
                    batch,
                    1,
                    args.hidden_dim,
                    device=device,
                    dtype=dtype,
                )
                / 4
            )

            def fused_forward() -> torch.Tensor:
                return fused_layer(x)

            def pytorch_forward() -> torch.Tensor:
                return linear(layernorm(x))

            # The first invocation performs JIT compilation outside timing.
            fused_check = fused_forward()
            pytorch_check = pytorch_forward()
            print(f"\nB={batch}")
            _report_error(fused_check, pytorch_check)
            torch.testing.assert_close(
                fused_check,
                pytorch_check,
                atol=1e-3,
                rtol=1e-3,
            )

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
                atol=1e-3,
                rtol=1e-3,
            )

            operations = 2 * batch * args.hidden_dim * args.out_dim
            _report("FusedLayerNormLinear", fused_times, operations)
            _report("PyTorch LayerNorm + Linear", pytorch_times, operations)
            speedup = (
                statistics.mean(pytorch_times)
                / statistics.mean(fused_times)
            )
            speedups.append((batch, speedup))
            print(f"speedup vs PyTorch: {speedup:.3f}x")

    best_batch, best_speedup = max(speedups, key=lambda item: item[1])
    print(
        f"\nbest speedup: {best_speedup:.3f}x at Batch={best_batch}"
    )


if __name__ == "__main__":
    main()
