from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layers.cute_matmul import cute_matmul_fp16


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def test_cute_matmul_fp16_matches_torch() -> None:
    if not torch.cuda.is_available():
        return

    torch.manual_seed(20260709)
    a = torch.randn(32, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, 48, device="cuda", dtype=torch.float16)

    actual = cute_matmul_fp16(a, b)
    expected = torch.matmul(a, b)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def test_cute_matmul_fp16_print_tops() -> None:
    if not torch.cuda.is_available():
        return

    m = _env_int("CUTE_MATMUL_M", 4096)
    n = _env_int("CUTE_MATMUL_N", 4096)
    k = _env_int("CUTE_MATMUL_K", 4096)
    warmup = _env_int("CUTE_MATMUL_WARMUP", 10)
    iters = _env_int("CUTE_MATMUL_ITERS", 50)

    torch.manual_seed(20260709)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)

    for _ in range(warmup):
        cute_matmul_fp16(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = cute_matmul_fp16(a, b)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iters
    ops = 2 * m * n * k
    tops = ops / (elapsed_ms / 1000.0) / 1.0e12
    tflops = tops

    print(
        f"CuTe FP16 GEMM: M={m}, N={n}, K={k}, "
        f"avg_ms={elapsed_ms:.4f}, TOPS={tops:.4f}, TFLOPS={tflops:.4f}"
    )
    assert out.shape == (m, n)
