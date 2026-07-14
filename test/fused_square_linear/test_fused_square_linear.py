from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_square_linear import fused_square_linear_fp16
from layers.fused_square_linear.reference import fused_square_linear_ref


@pytest.mark.parametrize(
    ("batch", "out_dim", "hidden_dim"),
    [
        (1, 17, 32),
        (2, 65, 64),
        (3, 37, 47),
        (5, 33, 64),
        (8, 129, 128),
        (16, 33, 64),
    ],
)
def test_fused_square_linear_matches_reference(
    batch: int,
    out_dim: int,
    hidden_dim: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(20260713)
    # Keep squared values and the long reduction well inside FP16 range.
    x = torch.randn(batch, 1, hidden_dim, device="cuda", dtype=torch.float16) / 4
    weight = (
        torch.randn(out_dim, hidden_dim, device="cuda", dtype=torch.float16) / 4
    )

    actual = fused_square_linear_fp16(x, weight)
    expected = fused_square_linear_ref(x, weight)

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_fused_square_linear_zero_input() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    x = torch.zeros(2, 1, 64, device="cuda", dtype=torch.float16)
    weight = torch.randn(33, 64, device="cuda", dtype=torch.float16)
    actual = fused_square_linear_fp16(x, weight)
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)


def test_fused_square_linear_non_default_stream() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    x = torch.randn(2, 1, 64, device="cuda", dtype=torch.float16) / 4
    weight = torch.randn(32, 64, device="cuda", dtype=torch.float16) / 4
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = fused_square_linear_fp16(x, weight)
        expected = fused_square_linear_ref(x, weight)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
