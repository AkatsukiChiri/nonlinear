from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear_int4_prefill import (
    FusedRMSNormLinearInt4Prefill,
)
from layers.fused_rmsnorm_linear_int4_prefill.reference import (
    fused_rmsnorm_linear_int4_prefill_ref,
)


def test_layer_stores_packed_int4_weight() -> None:
    rmsnorm = nn.RMSNorm(64, eps=1e-6, dtype=torch.float16)
    linear = nn.Linear(64, 24, bias=True, dtype=torch.float16)
    layer = FusedRMSNormLinearInt4Prefill(rmsnorm, linear)
    assert layer.packed_weight.shape == (24, 32)
    assert layer.packed_weight.dtype == torch.uint8
    assert layer.weight_scale.shape == (24,)
    assert dict(layer.named_parameters()) == {}


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("batch,seq_len", [(1, 9), (2, 16), (4, 128)])
def test_cuda_matches_quantized_reference(
    batch: int, seq_len: int, bias: bool
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")
    torch.manual_seed(20260805)
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
    layer = FusedRMSNormLinearInt4Prefill(rmsnorm, linear)
    actual = layer(x)
    expected = fused_rmsnorm_linear_int4_prefill_ref(
        x, layer.packed_weight, layer.weight_scale, layer.eps, layer.bias
    )
    assert actual.shape == (batch, seq_len, out_dim)
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=0.0)
