from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear import FusedAddRMSNormLinear
from layers.fused_rmsnorm_linear.reference import add_rmsnorm_linear_ref


def test_fused_add_init_precomputes_weight_and_bias() -> None:
    rmsnorm = nn.RMSNorm(8, eps=1e-6, dtype=torch.float16)
    linear = nn.Linear(8, 5, bias=True, dtype=torch.float16)

    layer = FusedAddRMSNormLinear(rmsnorm, linear)

    torch.testing.assert_close(
        layer.fused_weight,
        linear.weight * rmsnorm.weight.unsqueeze(0),
    )
    assert layer.bias is not None
    torch.testing.assert_close(layer.bias, linear.bias)
    assert dict(layer.named_parameters()) == {}


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_fused_add_rms_linear_matches_reference(
    batch: int,
    bias: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(20260724)
    hidden_dim = 64
    out_dim = 33
    rmsnorm = nn.RMSNorm(
        hidden_dim,
        eps=1e-6,
        device="cuda",
        dtype=torch.float16,
    )
    linear = nn.Linear(
        hidden_dim,
        out_dim,
        bias=bias,
        device="cuda",
        dtype=torch.float16,
    )
    x = torch.randn(
        batch,
        1,
        hidden_dim,
        device="cuda",
        dtype=torch.float16,
    ) / 4
    x_res = torch.randn_like(x) / 4
    layer = FusedAddRMSNormLinear(rmsnorm, linear)

    actual_output, actual_residual = layer(x, x_res)
    expected_output, expected_residual = add_rmsnorm_linear_ref(
        x,
        x_res,
        rmsnorm.weight,
        linear.weight,
        linear.bias,
        eps=1e-6,
    )

    assert actual_output.shape == (batch, 1, out_dim)
    assert actual_residual.shape == x.shape
    assert actual_output.dtype == torch.float16
    assert actual_residual.dtype == torch.float16
    torch.testing.assert_close(actual_residual, expected_residual, atol=0, rtol=0)
    torch.testing.assert_close(
        actual_output,
        expected_output,
        atol=2e-3,
        rtol=2e-3,
    )


def test_fused_add_rejects_mismatched_input_shape() -> None:
    rmsnorm = nn.RMSNorm(8, eps=1e-6, dtype=torch.float16)
    linear = nn.Linear(8, 5, bias=False, dtype=torch.float16)
    layer = FusedAddRMSNormLinear(rmsnorm, linear)

    with pytest.raises(ValueError, match="same shape"):
        layer(torch.empty(1, 1, 8), torch.empty(2, 1, 8))
