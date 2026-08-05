from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear_weight_only_int8_dequant import (
    FusedRMSNormLinearWeightOnlyInt8Dequant,
)
from layers.fused_rmsnorm_linear_weight_only_int8_dequant.reference import (
    weight_only_int8_dequant_ref,
)
from layers.fused_rmsnorm_linear_weight_only_int8_mixed import (
    FusedRMSNormLinearWeightOnlyInt8Mixed,
)
from layers.fused_rmsnorm_linear_weight_only_int8_mixed.reference import (
    weight_only_int8_mixed_ref,
)


@pytest.mark.parametrize(
    "layer_type",
    [FusedRMSNormLinearWeightOnlyInt8Mixed, FusedRMSNormLinearWeightOnlyInt8Dequant],
)
def test_layer_prequantizes_weight(layer_type: type[nn.Module]) -> None:
    rmsnorm = nn.RMSNorm(128, eps=1e-6, dtype=torch.float16)
    linear = nn.Linear(128, 17, bias=True, dtype=torch.float16)
    layer = layer_type(rmsnorm, linear)
    assert layer.quantized_weight.shape == (17, 128)
    assert layer.quantized_weight.dtype == torch.int8
    expected_groups = (
        1 if layer_type is FusedRMSNormLinearWeightOnlyInt8Dequant else 2
    )
    assert layer.weight_scale.shape == (17, expected_groups)
    assert layer.weight_scale.dtype == torch.float32
    assert dict(layer.named_parameters()) == {}


@pytest.mark.parametrize("batch", range(1, 9))
@pytest.mark.parametrize("bias", [False, True])
def test_cuda_variants_match_references(batch: int, bias: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")
    torch.manual_seed(20260717)
    rmsnorm = nn.RMSNorm(128, eps=1e-6, device="cuda", dtype=torch.float16)
    linear = nn.Linear(128, 35, bias=bias, device="cuda", dtype=torch.float16)
    x = torch.randn(batch, 1, 128, device="cuda", dtype=torch.float16) / 4
    mixed = FusedRMSNormLinearWeightOnlyInt8Mixed(rmsnorm, linear)
    dequant = FusedRMSNormLinearWeightOnlyInt8Dequant(rmsnorm, linear)

    mixed_expected = weight_only_int8_mixed_ref(
        x, mixed.quantized_weight, mixed.weight_scale, mixed.eps, mixed.bias
    )
    dequant_expected = weight_only_int8_dequant_ref(
        x, dequant.quantized_weight, dequant.weight_scale, dequant.eps, dequant.bias
    )
    mixed_actual = mixed(x)
    dequant_actual = dequant(x)
    assert mixed_actual.shape == (batch, 1, 35)
    assert dequant_actual.shape == (batch, 1, 35)
    torch.testing.assert_close(mixed_actual, mixed_expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(dequant_actual, dequant_expected, atol=2e-3, rtol=2e-3)
