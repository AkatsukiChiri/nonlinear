from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear import FusedRMSNormLinear
from layers.fused_rmsnorm_linear.reference import rmsnorm_linear_ref


def test_init_precomputes_fused_weight() -> None:
    rmsnorm = nn.RMSNorm(8, eps=1e-6, dtype=torch.float16)
    linear = nn.Linear(8, 5, bias=False, dtype=torch.float16)
    layer = FusedRMSNormLinear(rmsnorm, linear)

    expected = linear.weight * rmsnorm.weight.unsqueeze(0)
    torch.testing.assert_close(layer.fused_weight, expected)
    assert layer.fused_weight.shape == (5, 8)
    assert layer.bias is None
    assert dict(layer.named_parameters()) == {}


def test_init_snapshots_linear_bias() -> None:
    rmsnorm = nn.RMSNorm(8, eps=1e-6, dtype=torch.float16)
    linear = nn.Linear(8, 5, bias=True, dtype=torch.float16)
    layer = FusedRMSNormLinear(rmsnorm, linear)

    assert layer.bias is not None
    torch.testing.assert_close(layer.bias, linear.bias)
    assert layer.bias.data_ptr() != linear.bias.data_ptr()
    assert dict(layer.named_parameters()) == {}


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_fused_rmsnorm_linear_matches_reference(
    batch: int,
    bias: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(20260714)
    hidden_dim = 64
    out_dim = 33
    rmsnorm = nn.RMSNorm(
        hidden_dim, eps=1e-6, device="cuda", dtype=torch.float16
    )
    linear = nn.Linear(
        hidden_dim, out_dim, bias=bias, device="cuda", dtype=torch.float16
    )
    x = torch.randn(
        batch, 1, hidden_dim, device="cuda", dtype=torch.float16
    ) / 4
    layer = FusedRMSNormLinear(rmsnorm, linear)

    actual = layer(x)
    expected = rmsnorm_linear_ref(x, rmsnorm.weight, linear.weight, eps=1e-6)
    if linear.bias is not None:
        expected = expected + linear.bias

    assert actual.shape == (batch, 1, out_dim)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
