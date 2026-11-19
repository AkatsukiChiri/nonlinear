from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear_w4a16_global_sentinel import (
    FusedRMSNormLinearW4A16GlobalSentinel,
)


@pytest.mark.parametrize("batch", [1, 4, 8])
@pytest.mark.parametrize("with_residual", [False, True])
def test_global_sentinel_marlin_matches_reference(
    batch: int, with_residual: bool
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")
    torch.manual_seed(20261015)
    hidden_dim = 128
    out_dim = 64
    # One unsigned INT4 value per logical weight and zero-point zero. This
    # keeps the synthetic AWQ packing transparent while exercising the real
    # AWQ-to-Marlin repack and W4A16 GEMM.
    qweight = torch.full(
        (hidden_dim, out_dim // 8),
        0x11111111,
        device="cuda",
        dtype=torch.int32,
    )
    qzeros = torch.zeros(
        (hidden_dim // 128, out_dim // 8),
        device="cuda",
        dtype=torch.int32,
    )
    scales = (
        torch.rand(
            hidden_dim // 128,
            out_dim,
            device="cuda",
            dtype=torch.float16,
        )
        / 16
    )
    gamma = torch.rand(hidden_dim, device="cuda", dtype=torch.float16) + 0.5
    layer = FusedRMSNormLinearW4A16GlobalSentinel(
        qweight, qzeros, scales, gamma, None, 1e-5, 128,
        use_tensorcore=batch in (4, 8),
    )
    x = torch.randn(
        batch, 1, hidden_dim, device="cuda", dtype=torch.float16
    ) / 4
    residual = torch.randn_like(x) / 4 if with_residual else None

    actual = layer(x, residual)
    actual_output = actual[0] if isinstance(actual, tuple) else actual
    z = x if residual is None else (x + residual).to(torch.float16)
    top = (z * gamma).to(torch.float16).reshape(batch, hidden_dim)
    weight = scales[0].unsqueeze(0).expand(hidden_dim, out_dim)
    raw = (top.float() @ weight.float()).to(torch.float16).float()
    square_sum = (
        z.reshape(batch, hidden_dim).square().to(torch.float16).float().sum(1)
    ).to(torch.float16).float()
    inverse_rms = torch.rsqrt(square_sum / hidden_dim + layer.eps)
    expected = (raw * inverse_rms.unsqueeze(1)).to(torch.float16).reshape(
        batch, 1, out_dim
    )

    torch.testing.assert_close(actual_output, expected, atol=2e-2, rtol=2e-2)
    if residual is not None:
        assert isinstance(actual, tuple)
        torch.testing.assert_close(actual[1], z, atol=0, rtol=0)


@pytest.mark.parametrize("with_residual", [False, True])
def test_repacked_awq_gemv_bs1_matches_reference(
    with_residual: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")
    torch.manual_seed(20261016)
    hidden_dim = 4096
    out_dim = 256
    qweight = torch.full(
        (hidden_dim, out_dim // 8),
        0x11111111,
        device="cuda",
        dtype=torch.int32,
    )
    qzeros = torch.zeros(
        (hidden_dim // 128, out_dim // 8),
        device="cuda",
        dtype=torch.int32,
    )
    scales = torch.rand(
        hidden_dim // 128,
        out_dim,
        device="cuda",
        dtype=torch.float16,
    ) / 16
    gamma = torch.rand(hidden_dim, device="cuda", dtype=torch.float16) + 0.5
    layer = FusedRMSNormLinearW4A16GlobalSentinel(
        qweight, qzeros, scales, gamma, None, 1e-5, 128
    )
    x = torch.randn(
        1, 1, hidden_dim, device="cuda", dtype=torch.float16
    ) / 4
    residual = torch.randn_like(x) / 4 if with_residual else None

    actual = layer(x, residual)
    actual_output = actual[0] if isinstance(actual, tuple) else actual
    z = x if residual is None else (x + residual).to(torch.float16)
    inverse_rms = torch.rsqrt(z.float().square().mean(dim=-1, keepdim=True) + 1e-5)
    normalized = (z.float() * inverse_rms).to(torch.float16)
    weighted_activation = (normalized * gamma).to(torch.float16).float()
    weight = scales.repeat_interleave(128, dim=0).float()
    expected = (weighted_activation.reshape(1, hidden_dim) @ weight).to(
        torch.float16
    ).reshape(1, 1, out_dim)
    torch.testing.assert_close(actual_output, expected, atol=2e-2, rtol=2e-2)
    if residual is not None:
        assert isinstance(actual, tuple)
        torch.testing.assert_close(actual[1], z, atol=0, rtol=0)
