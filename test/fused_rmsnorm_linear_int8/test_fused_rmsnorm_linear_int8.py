from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear_int8 import FusedRMSNormLinearInt8
from layers.fused_rmsnorm_linear_int8.quantization import (
    quantize_weight_symmetric_int8,
)
from layers.fused_rmsnorm_linear_int8.reference import (
    fused_rmsnorm_linear_int8_ref,
)


def test_weight_quantization_uses_groups_of_64() -> None:
    weight = torch.linspace(-2, 2, 3 * 128).reshape(3, 128)
    quantized, scale = quantize_weight_symmetric_int8(weight)

    assert quantized.shape == weight.shape
    assert quantized.dtype == torch.int8
    assert scale.shape == (3, 2)
    assert scale.dtype == torch.float32
    assert quantized.min() >= -127
    assert quantized.max() <= 127


def test_layer_prequantizes_fused_weight() -> None:
    rmsnorm = nn.RMSNorm(128, eps=1e-6, dtype=torch.float16)
    linear = nn.Linear(128, 17, bias=True, dtype=torch.float16)
    layer = FusedRMSNormLinearInt8(rmsnorm, linear)
    expected_weight = linear.weight * rmsnorm.weight.unsqueeze(0)
    expected_quantized, expected_scale = quantize_weight_symmetric_int8(
        expected_weight
    )

    torch.testing.assert_close(layer.quantized_weight, expected_quantized)
    torch.testing.assert_close(layer.weight_scale, expected_scale)
    assert layer.quantized_weight.shape == (17, 128)
    assert layer.weight_scale.shape == (17, 2)
    assert layer.bias is not None
    assert dict(layer.named_parameters()) == {}


def test_layer_rejects_non_multiple_of_64_hidden_dim() -> None:
    rmsnorm = nn.RMSNorm(96, eps=1e-6, dtype=torch.float16)
    linear = nn.Linear(96, 17, bias=False, dtype=torch.float16)
    with pytest.raises(ValueError, match="divisible by 64"):
        FusedRMSNormLinearInt8(rmsnorm, linear)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_cuda_matches_quantized_reference(batch: int, bias: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("the CUDA implementation currently targets Ampere")

    torch.manual_seed(20260716)
    hidden_dim = 128
    out_dim = 35
    rmsnorm = nn.RMSNorm(
        hidden_dim, eps=1e-6, device="cuda", dtype=torch.float16
    )
    linear = nn.Linear(
        hidden_dim, out_dim, bias=bias, device="cuda", dtype=torch.float16
    )
    x = torch.randn(batch, 1, hidden_dim, device="cuda", dtype=torch.float16) / 4
    layer = FusedRMSNormLinearInt8(rmsnorm, linear)

    actual = layer(x)
    expected = fused_rmsnorm_linear_int8_ref(
        x,
        layer.quantized_weight,
        layer.weight_scale,
        layer.eps,
        layer.bias,
    )

    assert actual.shape == (batch, 1, out_dim)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
