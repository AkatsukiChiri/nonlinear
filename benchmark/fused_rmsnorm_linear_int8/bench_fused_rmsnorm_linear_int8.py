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
from layers.fused_rmsnorm_linear_int8 import (
    FusedRMSNormLinearInt8Decode,
    FusedRMSNormLinearInt8Prefill,
)


Output = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def _measure(
    function: Callable[[], Output],
    warmup: int,
    iterations: int,
) -> tuple[Output, list[float]]:
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
        "--dtype",
        choices=("float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument(
        "--mode",
        choices=("decode", "prefill"),
        required=True,
        help="Select the fixed INT8 implementation; no token-count dispatch.",
    )
    parser.add_argument(
        "--bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="benchmark a Linear with bias (default: --no-bias)",
    )
    parser.add_argument(
        "--fused-add",
        action="store_true",
        help="Fuse x + x_res and return the updated residual.",
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
    dtype = getattr(torch, args.dtype)
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

    fp16_fused_layer = (
        FusedRMSNormLinear(rmsnorm, linear).eval()
        if dtype == torch.float16
        else None
    )
    token_count = args.batch * args.seq_len
    if args.mode == "decode":
        if args.seq_len != 1 or not 1 <= args.batch <= 8:
            raise ValueError("decode mode requires seq-len=1 and batch in [1, 8]")
        int8_fused_layer = FusedRMSNormLinearInt8Decode(rmsnorm, linear).eval()
    else:
        int8_fused_layer = FusedRMSNormLinearInt8Prefill(rmsnorm, linear).eval()
    x = torch.randn(
        args.batch,
        args.seq_len,
        args.hidden_dim,
        device=device,
        dtype=dtype,
    ) / 4
    x_res = torch.randn_like(x) / 4 if args.fused_add else None

    def int8_fused_forward() -> Output:
        return int8_fused_layer(x, x_res) if x_res is not None else int8_fused_layer(x)

    def fp16_fused_forward() -> Output:
        assert fp16_fused_layer is not None
        if x_res is not None:
            residual_sum = x + x_res
            return fp16_fused_layer(residual_sum), residual_sum
        return fp16_fused_layer(x)

    def separate_add_int8_forward() -> Output:
        assert x_res is not None
        residual_sum = x + x_res
        return int8_fused_layer(residual_sum), residual_sum

    def pytorch_forward() -> Output:
        if x_res is not None:
            residual_sum = x + x_res
            return linear(rmsnorm(residual_sum)), residual_sum
        return linear(rmsnorm(x))

    def check_output(actual: Output, expected: Output, tolerance: float) -> None:
        if isinstance(actual, tuple):
            assert isinstance(expected, tuple)
            torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
            torch.testing.assert_close(
                actual[0], expected[0], atol=tolerance, rtol=tolerance
            )
        else:
            assert isinstance(expected, torch.Tensor)
            torch.testing.assert_close(
                actual, expected, atol=tolerance, rtol=tolerance
            )

    with torch.inference_mode():
        int8_check = int8_fused_forward()
        fp16_fused_check = (
            fp16_fused_forward() if fp16_fused_layer is not None else None
        )
        pytorch_check = pytorch_forward()
        if fp16_fused_check is not None:
            check_output(fp16_fused_check, pytorch_check, 1e-2)
        check_output(int8_check, pytorch_check, 5e-2)
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(
            f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}, "
            f"dtype={args.dtype}"
        )
        print(
            f"B={args.batch}, S={args.seq_len}, M={token_count}, "
            f"K={args.hidden_dim}, N={args.out_dim}, eps={args.eps:g}, "
            f"bias={args.bias}, mode={args.mode}, fused_add={args.fused_add}"
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
        separate_int8_output, separate_int8_times = (None, [])
        if x_res is not None:
            separate_int8_output, separate_int8_times = _measure(
                separate_add_int8_forward,
                args.warmup,
                args.iterations,
            )
        fp16_fused_output, fp16_fused_times = (None, [])
        if fp16_fused_layer is not None:
            fp16_fused_output, fp16_fused_times = _measure(
                fp16_fused_forward,
                args.warmup,
                args.iterations,
            )
        pytorch_output, pytorch_times = _measure(
            pytorch_forward,
            args.warmup,
            args.iterations,
        )
        if fp16_fused_output is not None:
            check_output(fp16_fused_output, pytorch_output, 1e-2)
        if separate_int8_output is not None:
            check_output(separate_int8_output, pytorch_output, 5e-2)
        check_output(int8_output, pytorch_output, 5e-2)

    operations = 2 * token_count * args.hidden_dim * args.out_dim
    int8_name = (
        "INT8 FusedAddRMSNormLinear"
        if args.fused_add
        else "INT8 FusedRMSNormLinear"
    )
    _report(int8_name, int8_times, operations)
    if separate_int8_times:
        _report("Add + INT8 FusedRMSNormLinear", separate_int8_times, operations)
    if fp16_fused_times:
        _report("FP16 FusedRMSNormLinear", fp16_fused_times, operations)
    pytorch_name = f"PyTorch {args.dtype} " + (
        "Add+RMSNorm+Linear" if args.fused_add else "RMSNorm+Linear"
    )
    _report(pytorch_name, pytorch_times, operations)
    int8_mean = statistics.mean(int8_times)
    if separate_int8_times:
        print(
            "Fused-add speedup vs separate add + INT8: "
            f"{statistics.mean(separate_int8_times) / int8_mean:.3f}x"
        )
    if fp16_fused_times:
        print(
            "INT8 speedup vs FP16 fused: "
            f"{statistics.mean(fp16_fused_times) / int8_mean:.3f}x"
        )
    print(
        f"INT8 speedup vs PyTorch {args.dtype}: "
        f"{statistics.mean(pytorch_times) / int8_mean:.3f}x"
    )


if __name__ == "__main__":
    main()
