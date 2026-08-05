from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear_weight_only_int4_dequant import (
    FusedRMSNormLinearWeightOnlyInt4Dequant,
)
from layers.fused_rmsnorm_linear_weight_only_int4_dequant.quantization import (
    quantize_weight_symmetric_int4,
    unpack_weight_int4,
)
from layers.fused_rmsnorm_linear_weight_only_int4_dequant.reference import (
    weight_only_int4_dequant_ref,
)


def test_int4_pack_round_trip() -> None:
    weight = torch.tensor([[-7.0, -3.0, 0.0, 2.0, 6.0, 7.0]])
    packed, scale = quantize_weight_symmetric_int4(weight)
    assert packed.shape == (1, 3)
    assert packed.dtype == torch.uint8
    torch.testing.assert_close(unpack_weight_int4(packed), weight.to(torch.int8))
    torch.testing.assert_close(scale, torch.ones_like(scale))


def test_layer_uses_half_byte_per_weight() -> None:
    rmsnorm = nn.RMSNorm(256, eps=1e-6, dtype=torch.float16)
    linear = nn.Linear(256, 17, bias=True, dtype=torch.float16)
    layer = FusedRMSNormLinearWeightOnlyInt4Dequant(rmsnorm, linear)
    assert layer.packed_weight.shape == (17, 128)
    assert layer.packed_weight.dtype == torch.uint8
    assert layer.weight_scale.shape == (17, 1)
    assert dict(layer.named_parameters()) == {}


@pytest.mark.parametrize("batch", range(1, 9))
@pytest.mark.parametrize("bias", [False, True])
def test_cuda_matches_quantized_reference(batch: int, bias: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")
    torch.manual_seed(20260805)
    hidden_dim = 256
    out_dim = 37
    rmsnorm = nn.RMSNorm(
        hidden_dim, eps=1e-6, device="cuda", dtype=torch.float16
    )
    linear = nn.Linear(
        hidden_dim, out_dim, bias=bias, device="cuda", dtype=torch.float16
    )
    x = torch.randn(batch, 1, hidden_dim, device="cuda", dtype=torch.float16) / 4
    layer = FusedRMSNormLinearWeightOnlyInt4Dequant(rmsnorm, linear)
    expected = weight_only_int4_dequant_ref(
        x, layer.packed_weight, layer.weight_scale, layer.eps, layer.bias
    )
    actual = layer(x)
    assert actual.shape == (batch, 1, out_dim)
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
