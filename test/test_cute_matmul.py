from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layers.cute_matmul import cute_matmul_fp16


def test_cute_matmul_fp16_matches_torch() -> None:
    if not torch.cuda.is_available():
        return

    torch.manual_seed(20260709)
    a = torch.randn(32, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, 48, device="cuda", dtype=torch.float16)

    actual = cute_matmul_fp16(a, b)
    expected = torch.matmul(a, b)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
