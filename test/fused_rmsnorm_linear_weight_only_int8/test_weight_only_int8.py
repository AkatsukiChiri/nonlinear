from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear_weight_only_int8_dequant import (
    FusedRMSNormLinearInt8Decode,
)
from layers.fused_rmsnorm_linear_weight_only_int8_dequant.reference import (
    fused_add_weight_only_int8_dequant_ref,
    weight_only_int8_dequant_ref,
)
def test_layer_prequantizes_weight() -> None:
    rmsnorm = nn.RMSNorm(128, eps=1e-6, dtype=torch.float16)
    linear = nn.Linear(128, 17, bias=True, dtype=torch.float16)
    layer = FusedRMSNormLinearInt8Decode(rmsnorm, linear)
    assert layer.quantized_weight.shape == (17, 128)
    assert layer.quantized_weight.dtype == torch.int8
    assert layer.weight_scale.shape == (17, 1)
    assert layer.weight_scale.dtype == torch.float32
    assert dict(layer.named_parameters()) == {}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", range(1, 9))
@pytest.mark.parametrize("bias", [False, True])
def test_cuda_matches_reference(
    batch: int, bias: bool, dtype: torch.dtype
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")
    torch.manual_seed(20260717)
    rmsnorm = nn.RMSNorm(128, eps=1e-6, device="cuda", dtype=dtype)
    linear = nn.Linear(128, 35, bias=bias, device="cuda", dtype=dtype)
    x = torch.randn(batch, 1, 128, device="cuda", dtype=dtype) / 4
    dequant = FusedRMSNormLinearInt8Decode(rmsnorm, linear)

    dequant_expected = weight_only_int8_dequant_ref(
        x, dequant.quantized_weight, dequant.weight_scale, dequant.eps, dequant.bias
    )
    dequant_actual = dequant(x)
    assert dequant_actual.shape == (batch, 1, 35)
    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(
        dequant_actual, dequant_expected, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("bias", [False, True])
def test_fused_add_cuda_matches_reference(
    batch: int, bias: bool, dtype: torch.dtype
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")
    torch.manual_seed(20260806)
    rmsnorm = nn.RMSNorm(128, eps=1e-6, device="cuda", dtype=dtype)
    linear = nn.Linear(128, 35, bias=bias, device="cuda", dtype=dtype)
    x = torch.randn(batch, 1, 128, device="cuda", dtype=dtype) / 4
    x_res = torch.randn_like(x) / 4
    layer = FusedRMSNormLinearInt8Decode(rmsnorm, linear)

    expected_output, expected_residual = fused_add_weight_only_int8_dequant_ref(
        x,
        x_res,
        layer.quantized_weight,
        layer.weight_scale,
        layer.eps,
        layer.bias,
    )
    actual_output, actual_residual = layer(x, x_res)
    tolerance = 3e-3 if dtype == torch.float16 else 3e-2
    torch.testing.assert_close(actual_residual, expected_residual, atol=0, rtol=0)
    torch.testing.assert_close(
        actual_output, expected_output, atol=tolerance, rtol=tolerance
    )


def test_fused_add_reuses_decode_output_buffers() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")
    rmsnorm = nn.RMSNorm(128, eps=1e-6, device="cuda", dtype=torch.float16)
    linear = nn.Linear(128, 32, bias=False, device="cuda", dtype=torch.float16)
    layer = FusedRMSNormLinearInt8Decode(rmsnorm, linear)
    x = torch.randn(1, 1, 128, device="cuda", dtype=torch.float16)
    x_res = torch.randn_like(x)

    first_output, first_residual = layer(x, x_res)
    second_output, second_residual = layer(x, x_res)

    assert first_output.data_ptr() == second_output.data_ptr()
    assert first_residual.data_ptr() == second_residual.data_ptr()
