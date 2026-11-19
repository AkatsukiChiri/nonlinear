"""Focused tests for the vLLM fused Llama integration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytest.importorskip("vllm")

from models.vllm.fused_llama_int8 import FusedInt8Projection


class FakeCompressedLinear(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        weight = torch.randint(
            -127, 128, (out_dim, hidden_dim), dtype=torch.int8
        )
        self.weight = nn.Parameter(weight.t(), requires_grad=False)
        self.weight_scale = nn.Parameter(
            torch.rand(out_dim, dtype=torch.float32) / 100,
            requires_grad=False,
        )
        self.input_size = hidden_dim
        self.logical_widths = [out_dim]
        self.register_parameter("bias", None)


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("with_residual", [False, True])
def test_fused_projection_explicit_mode(
    mode: str, with_residual: bool
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")
    hidden_dim, out_dim = 128, 96
    rmsnorm = nn.RMSNorm(
        hidden_dim, eps=1e-6, device="cuda", dtype=torch.float16
    )
    linear = FakeCompressedLinear(hidden_dim, out_dim).cuda()
    projection = FusedInt8Projection(rmsnorm, linear).cuda()
    projection.force_mode = mode  # type: ignore[assignment]
    tokens = 4 if mode == "decode" else 17
    x = torch.randn(tokens, hidden_dim, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x) if with_residual else None

    actual = projection(x, residual)
    executor = projection.decode if mode == "decode" else projection.prefill
    expected = executor(x, residual)

    if with_residual:
        assert isinstance(actual, tuple) and isinstance(expected, tuple)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
    else:
        assert isinstance(actual, torch.Tensor)
        assert isinstance(expected, torch.Tensor)
        torch.testing.assert_close(actual, expected)

    assert (
        projection.prefill.quantized_weight.data_ptr()
        == projection.decode.quantized_weight.data_ptr()
    )
