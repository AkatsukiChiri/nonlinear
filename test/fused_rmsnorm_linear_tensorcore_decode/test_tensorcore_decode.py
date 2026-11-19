from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear_tensorcore_decode import (
    FusedRMSNormLinearTensorCoreDecode,
    FusedRMSNormLinearTensorCoreInt8WeightDecode,
)
from layers.fused_rmsnorm_linear_tensorcore_decode.reference import (
    tensorcore_rms_linear_decode_ref,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 4, 8, 16])
@pytest.mark.parametrize("out_dim", [127, 257])
@pytest.mark.parametrize("with_residual", [False, True])
def test_tensorcore_decode_matches_reference(
    dtype: torch.dtype, batch: int, out_dim: int, with_residual: bool
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")
    torch.manual_seed(20261014)
    hidden_dim = 128
    quantized_weight = torch.randint(
        -32, 33, (out_dim, hidden_dim), device="cuda", dtype=torch.int8
    )
    weight_scale = torch.rand(out_dim, 1, device="cuda") / 64
    bias = torch.randn(out_dim, device="cuda", dtype=dtype) / 8
    layer = FusedRMSNormLinearTensorCoreDecode(
        quantized_weight, weight_scale, bias, 1e-6, dtype
    )
    x = torch.randn(batch, 1, hidden_dim, device="cuda", dtype=dtype) / 4
    residual = torch.randn_like(x) / 4 if with_residual else None

    expected = tensorcore_rms_linear_decode_ref(
        x, residual, layer.weight_fp16, layer.out_dim, layer.bias, layer.eps
    )
    actual = layer(x, residual)
    expected_output = expected[0] if isinstance(expected, tuple) else expected
    actual_output = actual[0] if isinstance(actual, tuple) else actual
    tolerance = 5e-3 if dtype == torch.float16 else 3e-2
    torch.testing.assert_close(
        actual_output, expected_output, atol=tolerance, rtol=tolerance
    )
    if with_residual:
        assert isinstance(expected, tuple) and isinstance(actual, tuple)
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


def test_tensorcore_decode_reuses_output_buffers() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    quantized_weight = torch.ones(127, 128, device="cuda", dtype=torch.int8)
    weight_scale = torch.ones(127, 1, device="cuda")
    layer = FusedRMSNormLinearTensorCoreDecode(
        quantized_weight, weight_scale, None, 1e-6, torch.float16
    )
    x = torch.randn(8, 128, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    first = layer(x, residual)
    second = layer(x, residual)
    assert isinstance(first, tuple) and isinstance(second, tuple)
    assert first[0].data_ptr() == second[0].data_ptr()
    assert first[1].data_ptr() == second[1].data_ptr()


@pytest.mark.parametrize("tiles_per_cta", [1, 2, 4, 8, 16, 32])
def test_tensorcore_decode_cta_widths(tiles_per_cta: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(20261014)
    quantized_weight = torch.randint(
        -32, 33, (65, 128), device="cuda", dtype=torch.int8
    )
    weight_scale = torch.rand(65, 1, device="cuda") / 64
    layer = FusedRMSNormLinearTensorCoreDecode(
        quantized_weight,
        weight_scale,
        None,
        1e-6,
        torch.float16,
        tiles_per_cta,
    )
    x = torch.randn(4, 1, 128, device="cuda", dtype=torch.float16) / 4
    expected = tensorcore_rms_linear_decode_ref(
        x, None, layer.weight_fp16, layer.out_dim, None, layer.eps
    )
    actual = layer(x)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [4, 8, 16])
def test_int8_weight_tensorcore_decode_matches_fp16_weight_reference(
    dtype: torch.dtype, batch: int
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(20261014)
    quantized_weight = torch.randint(
        -32, 33, (257, 128), device="cuda", dtype=torch.int8
    )
    weight_scale = torch.rand(257, 1, device="cuda") / 64
    bias = torch.randn(257, device="cuda", dtype=dtype) / 8
    layer = FusedRMSNormLinearTensorCoreInt8WeightDecode(
        quantized_weight, weight_scale, bias, 1e-6, dtype, 4
    )
    x = torch.randn(batch, 1, 128, device="cuda", dtype=dtype) / 4
    residual = torch.randn_like(x) / 4

    dequantized = (
        layer.quantized_weight.to(torch.float16)
        * layer.weight_scale.unsqueeze(2)
    )
    dequantized[:, -1, :, -1].fill_(1.0)
    expected = tensorcore_rms_linear_decode_ref(
        x, residual, dequantized, layer.out_dim, layer.bias, layer.eps
    )
    actual = layer(x, residual)
    assert isinstance(expected, tuple) and isinstance(actual, tuple)
    tolerance = 5e-3 if dtype == torch.float16 else 3e-2
    torch.testing.assert_close(
        actual[0], expected[0], atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


def test_int8_weight_decode_handles_more_than_two_k_stages() -> None:
    """Cover a multi-stage K loop instead of only the minimal K=128 case."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")
    torch.manual_seed(20261015)
    batch = 4
    hidden_dim = 256
    out_dim = 65
    quantized_weight = torch.randint(
        -32, 33, (out_dim, hidden_dim), device="cuda", dtype=torch.int8
    )
    weight_scale = torch.rand(out_dim, 1, device="cuda") / 64
    layer = FusedRMSNormLinearTensorCoreInt8WeightDecode(
        quantized_weight,
        weight_scale,
        None,
        1e-6,
        torch.float16,
        4,
    )
    x = torch.randn(
        batch, 1, hidden_dim, device="cuda", dtype=torch.float16
    ) / 4
    residual = torch.randn_like(x) / 4
    dequantized = (
        layer.quantized_weight.to(torch.float16)
        * layer.weight_scale.unsqueeze(2)
    )
    dequantized[:, -1, :, -1].fill_(1.0)

    expected = tensorcore_rms_linear_decode_ref(
        x, residual, dequantized, layer.out_dim, None, layer.eps
    )
    actual = layer(x, residual)

    assert isinstance(expected, tuple) and isinstance(actual, tuple)
    torch.testing.assert_close(actual[0], expected[0], atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
