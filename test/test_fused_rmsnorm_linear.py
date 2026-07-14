from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layers.fused_rmsnorm_linear import FusedRMSNormLinear
from layers.rmsnorm_linear_ref import rmsnorm_linear_ref


def test_init_precomputes_augmented_weight() -> None:
    rmsnorm = nn.RMSNorm(8, eps=1e-6, dtype=torch.float16)
    linear = nn.Linear(8, 5, bias=False, dtype=torch.float16)
    layer = FusedRMSNormLinear(rmsnorm, linear)

    expected = torch.cat(
        (
            linear.weight * rmsnorm.weight.unsqueeze(0),
            torch.ones(1, 8, dtype=torch.float16),
        ),
        dim=0,
    )
    torch.testing.assert_close(layer.fused_weight, expected)
    assert layer.fused_weight.shape == (6, 8)
    assert dict(layer.named_parameters()) == {}


def test_init_rejects_linear_bias() -> None:
    rmsnorm = nn.RMSNorm(8, eps=1e-6, dtype=torch.float16)
    linear = nn.Linear(8, 5, bias=True, dtype=torch.float16)
    with pytest.raises(ValueError, match="bias=False"):
        FusedRMSNormLinear(rmsnorm, linear)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_fused_rmsnorm_linear_matches_reference(batch: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(20260714)
    hidden_dim = 64
    out_dim = 33
    rmsnorm = nn.RMSNorm(
        hidden_dim, eps=1e-6, device="cuda", dtype=torch.float16
    )
    linear = nn.Linear(
        hidden_dim, out_dim, bias=False, device="cuda", dtype=torch.float16
    )
    x = torch.randn(
        batch, 1, hidden_dim, device="cuda", dtype=torch.float16
    ) / 4
    layer = FusedRMSNormLinear(rmsnorm, linear)

    actual = layer(x)
    expected = rmsnorm_linear_ref(x, rmsnorm.weight, linear.weight, eps=1e-6)

    assert actual.shape == (batch, 1, out_dim)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
