from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear_int4 import (
    PREFILL_MIN_TOKENS,
    FusedRMSNormLinearInt4,
)
from layers.fused_rmsnorm_linear_int4_prefill.reference import (
    fused_rmsnorm_linear_int4_prefill_ref,
)
from layers.fused_rmsnorm_linear_weight_only_int4_dequant.reference import (
    weight_only_int4_dequant_ref,
)


def test_dispatch_threshold_is_six_tokens() -> None:
    assert PREFILL_MIN_TOKENS == 6


def test_layer_stores_one_packed_weight() -> None:
    rmsnorm = nn.RMSNorm(256, eps=1e-6, dtype=torch.float16)
    linear = nn.Linear(256, 24, bias=True, dtype=torch.float16)
    layer = FusedRMSNormLinearInt4(rmsnorm, linear)
    assert layer.packed_weight.shape == (24, 128)
    assert layer.weight_scale.shape == (24, 1)
    assert dict(layer.named_parameters()) == {}


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("batch,seq_len", [(1, 1), (1, 5), (2, 3), (4, 128)])
def test_cuda_matches_dispatched_reference(
    batch: int, seq_len: int, bias: bool
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")
    torch.manual_seed(20260806)
    hidden_dim = 256
    out_dim = 40
    rmsnorm = nn.RMSNorm(
        hidden_dim, eps=1e-6, device="cuda", dtype=torch.float16
    )
    linear = nn.Linear(
        hidden_dim, out_dim, bias=bias, device="cuda", dtype=torch.float16
    )
    x = torch.randn(
        batch, seq_len, hidden_dim, device="cuda", dtype=torch.float16
    ) / 4
    layer = FusedRMSNormLinearInt4(rmsnorm, linear)
    actual = layer(x)
    rows = batch * seq_len
    if rows < PREFILL_MIN_TOKENS:
        expected = weight_only_int4_dequant_ref(
            x.reshape(rows, 1, hidden_dim),
            layer.packed_weight,
            layer.weight_scale,
            layer.eps,
            layer.bias,
        ).reshape(batch, seq_len, out_dim)
        tolerance = 3e-3
    else:
        expected = fused_rmsnorm_linear_int4_prefill_ref(
            x,
            layer.packed_weight,
            layer.weight_scale.reshape(-1),
            layer.eps,
            layer.bias,
        )
        tolerance = 6e-2
    assert actual.shape == (batch, seq_len, out_dim)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0.0)
