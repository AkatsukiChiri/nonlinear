from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear.prefill import FusedRMSNormLinearPrefill


def _require_ampere() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("the experimental prefill kernel requires an Ampere GPU")


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    ("batch", "seq_len", "hidden_dim", "out_dim"),
    [
        (4, 128, 256, 320),
        (3, 17, 64, 80),
    ],
)
def test_prefill_matches_pytorch(
    batch: int,
    seq_len: int,
    hidden_dim: int,
    out_dim: int,
    bias: bool,
) -> None:
    _require_ampere()
    torch.manual_seed(20260728)
    rmsnorm = nn.RMSNorm(
        hidden_dim, eps=1e-6, device="cuda", dtype=torch.float16
    )
    linear = nn.Linear(
        hidden_dim,
        out_dim,
        bias=bias,
        device="cuda",
        dtype=torch.float16,
    )
    x = (
        torch.randn(
            batch,
            seq_len,
            hidden_dim,
            device="cuda",
            dtype=torch.float16,
        )
        / 4
    )
    layer = FusedRMSNormLinearPrefill(rmsnorm, linear)

    actual = layer(x)
    expected = linear(rmsnorm(x))

    assert actual.shape == (batch, seq_len, out_dim)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=0.0)


def test_prefill_default_shape_matches_pytorch() -> None:
    _require_ampere()
    torch.manual_seed(20260728)
    batch, seq_len, hidden_dim = 4, 128, 4096
    rmsnorm = nn.RMSNorm(
        hidden_dim, eps=1e-6, device="cuda", dtype=torch.float16
    )
    linear = nn.Linear(
        hidden_dim,
        hidden_dim,
        bias=False,
        device="cuda",
        dtype=torch.float16,
    )
    x = (
        torch.randn(
            batch,
            seq_len,
            hidden_dim,
            device="cuda",
            dtype=torch.float16,
        )
        / 4
    )

    actual = FusedRMSNormLinearPrefill(rmsnorm, linear)(x)
    expected = linear(rmsnorm(x))
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=0.0)


def test_prefill_rejects_non_tensor_core_dimensions() -> None:
    rmsnorm = nn.RMSNorm(18, eps=1e-6, dtype=torch.float16)
    linear = nn.Linear(18, 17, bias=False, dtype=torch.float16)
    with pytest.raises(ValueError, match="divisible by 16"):
        FusedRMSNormLinearPrefill(rmsnorm, linear)


def test_prefill_rejects_noncontiguous_input() -> None:
    _require_ampere()
    rmsnorm = nn.RMSNorm(
        64, eps=1e-6, device="cuda", dtype=torch.float16
    )
    linear = nn.Linear(
        64, 64, bias=False, device="cuda", dtype=torch.float16
    )
    layer = FusedRMSNormLinearPrefill(rmsnorm, linear)
    x = torch.randn(4, 64, 128, device="cuda", dtype=torch.float16)
    noncontiguous = x[:, :, ::2]
    with pytest.raises(ValueError, match="contiguous"):
        layer(noncontiguous)
