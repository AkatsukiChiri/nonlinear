"""Benchmark INT8 fused RMSNorm + Linear against both FP16 paths."""

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
from layers.fused_rmsnorm_linear_int8 import FusedRMSNormLinearInt8
from layers.fused_rmsnorm_linear_weight_only_int8_dequant import (
    FusedRMSNormLinearWeightOnlyInt8Dequant,
)


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
    parser.add_argument("--seq-len", type=int, default=1)
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
    if (
        args.batch <= 0
        or args.seq_len <= 0
        or args.hidden_dim <= 0
        or args.out_dim <= 0
    ):
        raise ValueError("batch, seq-len, hidden-dim, and out-dim must be positive")
    if args.hidden_dim % 64 != 0:
        raise ValueError("hidden-dim must be divisible by 64")

    torch.manual_seed(20260716)
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

    fp16_fused_layer = FusedRMSNormLinear(rmsnorm, linear).eval()
    int8_fused_layer = FusedRMSNormLinearInt8(rmsnorm, linear).eval()
    token_count = args.batch * args.seq_len
    benchmark_decode_variants = args.seq_len == 1 and args.batch <= 8
    dequant_layer = (
        FusedRMSNormLinearWeightOnlyInt8Dequant(rmsnorm, linear).eval()
        if benchmark_decode_variants
        else None
    )
    x = torch.randn(
        args.batch,
        args.seq_len,
        args.hidden_dim,
        device=device,
        dtype=dtype,
    ) / 4

    def int8_fused_forward() -> torch.Tensor:
        return int8_fused_layer(x)

    def fp16_fused_forward() -> torch.Tensor:
        return fp16_fused_layer(x)

    def dequant_forward() -> torch.Tensor:
        assert dequant_layer is not None
        return dequant_layer(x)

    def pytorch_fp16_forward() -> torch.Tensor:
        return linear(rmsnorm(x))

    with torch.inference_mode():
        int8_check = int8_fused_forward()
        dequant_check = dequant_forward() if benchmark_decode_variants else None
        fp16_fused_check = fp16_fused_forward()
        pytorch_check = pytorch_fp16_forward()
        torch.testing.assert_close(
            fp16_fused_check,
            pytorch_check,
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            int8_check,
            pytorch_check,
            atol=5e-2,
            rtol=5e-2,
        )
        if dequant_check is not None:
            torch.testing.assert_close(
                dequant_check, pytorch_check, atol=5e-2, rtol=5e-2
            )

        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}")
        print(
            f"B={args.batch}, S={args.seq_len}, M={token_count}, "
            f"K={args.hidden_dim}, N={args.out_dim}, eps={args.eps:g}, "
            f"bias={args.bias}, dispatch={'decode' if token_count <= 8 else 'prefill'}"
        )
        print(
            "INT8 weight shape: "
            f"{tuple(int8_fused_layer.quantized_weight.shape)}, "
            f"scale shape: {tuple(int8_fused_layer.weight_scale.shape)}"
        )

        int8_output, int8_times = _measure(
            int8_fused_forward,
            args.warmup,
            args.iterations,
        )
        dequant_output, dequant_times = (None, [])
        if benchmark_decode_variants:
            dequant_output, dequant_times = _measure(
                dequant_forward, args.warmup, args.iterations
            )
        fp16_fused_output, fp16_fused_times = _measure(
            fp16_fused_forward,
            args.warmup,
            args.iterations,
        )
        pytorch_output, pytorch_times = _measure(
            pytorch_fp16_forward,
            args.warmup,
            args.iterations,
        )
        torch.testing.assert_close(
            fp16_fused_output,
            pytorch_output,
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            int8_output,
            pytorch_output,
            atol=5e-2,
            rtol=5e-2,
        )
        if dequant_output is not None:
            torch.testing.assert_close(
                dequant_output, pytorch_output, atol=5e-2, rtol=5e-2
            )

    operations = 2 * token_count * args.hidden_dim * args.out_dim
    _report("INT8 FusedRMSNormLinear", int8_times, operations)
    if benchmark_decode_variants:
        _report("INT8 weight-only FP16 dequant", dequant_times, operations)
    _report("FP16 FusedRMSNormLinear", fp16_fused_times, operations)
    _report("PyTorch FP16 RMSNorm+Linear", pytorch_times, operations)
    int8_mean = statistics.mean(int8_times)
    print(
        "INT8 speedup vs FP16 fused: "
        f"{statistics.mean(fp16_fused_times) / int8_mean:.3f}x"
    )
    print(
        "INT8 speedup vs PyTorch FP16: "
        f"{statistics.mean(pytorch_times) / int8_mean:.3f}x"
    )
    for name, timings in (("weight-only FP16 dequant", dequant_times),):
        if not timings:
            continue
        mean = statistics.mean(timings)
        print(
            f"{name} speedup vs FP16 fused: "
            f"{statistics.mean(fp16_fused_times) / mean:.3f}x"
        )
        print(
            f"{name} speedup vs PyTorch FP16: "
            f"{statistics.mean(pytorch_times) / mean:.3f}x"
        )


if __name__ == "__main__":
    main()
