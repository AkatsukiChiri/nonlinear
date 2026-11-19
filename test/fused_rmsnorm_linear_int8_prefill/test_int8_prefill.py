from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear_int8_prefill import (
    FusedRMSNormLinearInt8Prefill,
)
from layers.fused_rmsnorm_linear_int8_prefill.reference import (
    fused_add_rmsnorm_linear_int8_prefill_ref,
    fused_rmsnorm_linear_int8_prefill_ref,
)


def _require_ampere() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("batch", "seq_len", "hidden_dim", "out_dim"),
    [(2, 7, 128, 96), (4, 8, 256, 320)],
)
def test_int8_prefill_matches_quantized_reference(
    batch: int,
    seq_len: int,
    hidden_dim: int,
    out_dim: int,
    bias: bool,
    dtype: torch.dtype,
) -> None:
    _require_ampere()
    torch.manual_seed(20260804)
    rmsnorm = nn.RMSNorm(
        hidden_dim, eps=1e-6, device="cuda", dtype=dtype
    )
    linear = nn.Linear(
        hidden_dim, out_dim, bias=bias, device="cuda", dtype=dtype
    )
    x = torch.randn(
        batch, seq_len, hidden_dim, device="cuda", dtype=dtype
    ) / 4
    layer = FusedRMSNormLinearInt8Prefill(rmsnorm, linear)
    expected = fused_rmsnorm_linear_int8_prefill_ref(
        x, layer.quantized_weight, layer.weight_scale, layer.eps, layer.bias
    )
    actual = layer(x)
    assert actual.shape == (batch, seq_len, out_dim)
    tolerance = 2e-2 if dtype == torch.float16 else 4e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0.0)


def test_int8_prefill_weight_is_compact() -> None:
    rmsnorm = nn.RMSNorm(128, eps=1e-6, dtype=torch.float16)
    linear = nn.Linear(128, 96, bias=False, dtype=torch.float16)
    layer = FusedRMSNormLinearInt8Prefill(rmsnorm, linear)
    assert layer.quantized_weight.dtype == torch.int8
    assert layer.quantized_weight.shape == (96, 128)
    assert layer.weight_scale.dtype == torch.float32
    assert layer.weight_scale.shape == (96,)
    assert dict(layer.named_parameters()) == {}


def test_int8_prefill_default_shape_matches_quantized_reference() -> None:
    _require_ampere()
    torch.manual_seed(20260804)
    batch, seq_len, hidden_dim, out_dim = 4, 128, 4096, 4096
    rmsnorm = nn.RMSNorm(
        hidden_dim, eps=1e-6, device="cuda", dtype=torch.float16
    )
    linear = nn.Linear(
        hidden_dim, out_dim, bias=False, device="cuda", dtype=torch.float16
    )
    x = torch.randn(
        batch, seq_len, hidden_dim, device="cuda", dtype=torch.float16
    ) / 4
    layer = FusedRMSNormLinearInt8Prefill(rmsnorm, linear)
    actual = layer(x)
    expected = fused_rmsnorm_linear_int8_prefill_ref(
        x, layer.quantized_weight, layer.weight_scale, layer.eps
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=0.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_fused_add_int8_prefill_matches_reference(
    dtype: torch.dtype, bias: bool
) -> None:
    _require_ampere()
    torch.manual_seed(20260806)
    batch, seq_len, hidden_dim, out_dim = 2, 7, 128, 96
    rmsnorm = nn.RMSNorm(
        hidden_dim, eps=1e-6, device="cuda", dtype=dtype
    )
    linear = nn.Linear(
        hidden_dim, out_dim, bias=bias, device="cuda", dtype=dtype
    )
    x = torch.randn(
        batch, seq_len, hidden_dim, device="cuda", dtype=dtype
    ) / 4
    x_res = torch.randn_like(x) / 4
    layer = FusedRMSNormLinearInt8Prefill(rmsnorm, linear)

    expected_output, expected_residual = (
        fused_add_rmsnorm_linear_int8_prefill_ref(
            x,
            x_res,
            layer.quantized_weight,
            layer.weight_scale,
            layer.eps,
            layer.bias,
        )
    )
    actual_output, actual_residual = layer(x, x_res)
    tolerance = 2e-2 if dtype == torch.float16 else 4e-2
    torch.testing.assert_close(actual_residual, expected_residual, atol=0, rtol=0)
    torch.testing.assert_close(
        actual_output, expected_output, atol=tolerance, rtol=0.0
    )


def test_fused_add_prefill_reuses_output_and_workspace_buffers() -> None:
    _require_ampere()
    rmsnorm = nn.RMSNorm(128, eps=1e-6, device="cuda", dtype=torch.float16)
    linear = nn.Linear(128, 96, bias=False, device="cuda", dtype=torch.float16)
    layer = FusedRMSNormLinearInt8Prefill(rmsnorm, linear)
    x = torch.randn(2, 7, 128, device="cuda", dtype=torch.float16)
    x_res = torch.randn_like(x)

    first_output, first_residual = layer(x, x_res)
    second_output, second_residual = layer(x, x_res)

    assert first_output.data_ptr() == second_output.data_ptr()
    assert first_residual.data_ptr() == second_residual.data_ptr()
    assert len(layer._fused_add_buffers) == 1


@pytest.mark.parametrize("with_residual", [False, True])
def test_prefill_accepts_vllm_flat_token_layout(with_residual: bool) -> None:
    _require_ampere()
    torch.manual_seed(20260812)
    rmsnorm = nn.RMSNorm(128, eps=1e-6, device="cuda", dtype=torch.float16)
    linear = nn.Linear(128, 96, bias=False, device="cuda", dtype=torch.float16)
    layer = FusedRMSNormLinearInt8Prefill(rmsnorm, linear)
    x = torch.randn(17, 128, device="cuda", dtype=torch.float16) / 4
    x_res = torch.randn_like(x) / 4 if with_residual else None

    actual = layer(x, x_res)
    expected = layer(
        x.reshape(1, 17, 128),
        None if x_res is None else x_res.reshape(1, 17, 128),
    )

    if with_residual:
        assert isinstance(actual, tuple) and isinstance(expected, tuple)
        torch.testing.assert_close(actual[0], expected[0].reshape(17, 96))
        torch.testing.assert_close(actual[1], expected[1].reshape(17, 128))
    else:
        assert isinstance(actual, torch.Tensor)
        assert isinstance(expected, torch.Tensor)
        torch.testing.assert_close(actual, expected.reshape(17, 96))
