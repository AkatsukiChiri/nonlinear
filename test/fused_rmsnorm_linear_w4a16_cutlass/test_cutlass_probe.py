from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear_w4a16_cutlass import (
    FusedRMSNormLinearW4A16AsyncDecode,
    FusedRMSNormLinearW4A16Cutlass,
    cutlass_w4a16_probe,
    cutlass_w4a16_rms_sentinel,
    pack_awq_with_sentinels,
)


def _pack_awq(values: torch.Tensor) -> torch.Tensor:
    shifts = torch.tensor(
        [0, 16, 4, 20, 8, 24, 12, 28],
        dtype=torch.int64,
        device=values.device,
    )
    chunks = values.reshape(values.shape[0], -1, 8).to(torch.int64)
    packed = torch.zeros(
        chunks.shape[:2], dtype=torch.int64, device=values.device
    )
    for index in range(8):
        packed |= chunks[..., index] << shifts[index]
    return packed.to(torch.int32)


def test_cutlass_w4a16_probe_matches_pytorch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")
    torch.manual_seed(20261017)
    # MmaPipelined is a two-stage mainloop; exercise at least two K tiles.
    rows, hidden_dim, out_dim = 2, 128, 64
    activation = torch.randn(
        rows, hidden_dim, device="cuda", dtype=torch.float16
    ) / 4
    logical_weight = torch.randint(
        0, 16, (out_dim, hidden_dim), device="cuda", dtype=torch.uint8
    )
    packed_values = (
        logical_weight[:, 0::2] | (logical_weight[:, 1::2] << 4)
    ).contiguous().view(-1)
    zeros = torch.randint(
        0, 8, (hidden_dim // 128, out_dim), device="cuda", dtype=torch.uint8
    )
    scales = torch.rand(
        hidden_dim // 128, out_dim, device="cuda", dtype=torch.float16
    ) / 8
    packed_storage = torch.cat(
        [
            packed_values,
            zeros.view(-1),
            scales.view(torch.uint8).view(-1),
        ]
    )
    actual = cutlass_w4a16_probe(activation, packed_storage, out_dim)
    # Surface asynchronous custom-kernel failures at their origin instead
    # of attributing them to the following cuBLAS reference operation.
    torch.cuda.synchronize()
    dequantized = (
        logical_weight.float() - zeros[0].float().unsqueeze(1)
    ) * scales[0].float().unsqueeze(1)
    expected = activation.float() @ dequantized.t()
    torch.testing.assert_close(
        actual.float(), expected, atol=2e-1, rtol=2e-2
    )


def test_n_minus_1_sentinel_gemm_contains_square_sum() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")
    torch.manual_seed(20261018)
    hidden_dim, out_dim = 128, 72
    qweight = torch.randint(
        0, 16, (hidden_dim, out_dim), device="cuda", dtype=torch.uint8
    )
    qzeros = torch.randint(
        0, 8, (1, out_dim), device="cuda", dtype=torch.uint8
    )
    scales = torch.rand(1, out_dim, device="cuda", dtype=torch.float16) / 8
    packed = pack_awq_with_sentinels(
        _pack_awq(qweight),
        _pack_awq(qzeros),
        scales,
        cta_n=64,
        layout="n_minus_1",
    )
    z = torch.randn(hidden_dim, device="cuda", dtype=torch.float16) / 4
    gamma = torch.rand(hidden_dim, device="cuda", dtype=torch.float16) + 0.5
    augmented = torch.stack([(z * gamma).half(), z.square().half()])
    gemm_output = cutlass_w4a16_probe(
        augmented, packed.storage, packed.physical_out_dim
    )

    dequantized = (
        qweight.float() - qzeros[0].float().unsqueeze(0)
    ) * scales[0].float().unsqueeze(0)
    expected_projection = augmented[0].float() @ dequantized
    torch.testing.assert_close(
        gemm_output[0, packed.real_to_physical.long()].float(),
        expected_projection,
        atol=2e-1,
        rtol=2e-2,
    )
    expected_square_sum = augmented[1].float().sum()
    torch.testing.assert_close(
        gemm_output[1, packed.sentinel_columns.long()].float(),
        expected_square_sum.expand_as(packed.sentinel_columns).float(),
        atol=2e-1,
        rtol=2e-2,
    )

    bias = torch.randn(out_dim, device="cuda", dtype=torch.float16) / 16
    fused = cutlass_w4a16_rms_sentinel(
        z.unsqueeze(0),
        gamma,
        packed.storage,
        out_dim,
        packed.physical_out_dim,
        hidden_dim,
        1e-5,
        bias,
    )
    inverse_rms = torch.rsqrt(z.float().square().mean() + 1e-5)
    expected_fused = (
        expected_projection * inverse_rms + bias.float()
    ).half()
    torch.testing.assert_close(
        fused[0], expected_fused, atol=2e-1, rtol=2e-2
    )


@pytest.mark.parametrize("batch", [1, 4, 8])
def test_cutlass_layer_matches_pytorch(batch: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")
    torch.manual_seed(20261019 + batch)
    hidden_dim, out_dim = 128, 72
    values = torch.randint(
        0, 16, (hidden_dim, out_dim), device="cuda", dtype=torch.uint8
    )
    zeros = torch.randint(
        0, 8, (1, out_dim), device="cuda", dtype=torch.uint8
    )
    scales = torch.rand(1, out_dim, device="cuda", dtype=torch.float16) / 8
    gamma = torch.rand(hidden_dim, device="cuda", dtype=torch.float16) + 0.5
    bias = torch.randn(out_dim, device="cuda", dtype=torch.float16) / 16
    layer = FusedRMSNormLinearW4A16Cutlass(
        _pack_awq(values), _pack_awq(zeros), scales, gamma, bias
    )
    input = torch.randn(
        batch, 1, hidden_dim, device="cuda", dtype=torch.float16
    ) / 4
    actual = layer(input)

    dequantized = (
        values.float() - zeros[0].float().unsqueeze(0)
    ) * scales[0].float().unsqueeze(0)
    inverse_rms = torch.rsqrt(
        input.float().square().mean(dim=-1, keepdim=True) + layer.eps
    )
    weighted_input = (input.float() * gamma.float()).half()
    raw = weighted_input.reshape(batch, hidden_dim).float() @ dequantized
    expected = (
        raw * inverse_rms.reshape(batch, 1) + bias.float()
    ).half().unsqueeze(1)
    torch.testing.assert_close(actual, expected, atol=2e-1, rtol=2e-2)


@pytest.mark.parametrize("batch", [1, 4, 8])
@pytest.mark.parametrize(("cta_n", "out_dim"), [(64, 72), (128, 136)])
def test_async_decode_layer_matches_pytorch(batch: int, cta_n: int,
                                             out_dim: int) -> None:
    """Both adaptive CTA widths preserve the fused RMSNorm projection."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")
    torch.manual_seed(20261030 + batch + cta_n)
    hidden_dim = 128
    values = torch.randint(
        0, 16, (hidden_dim, out_dim), device="cuda", dtype=torch.uint8
    )
    zeros = torch.randint(
        0, 8, (1, out_dim), device="cuda", dtype=torch.uint8
    )
    scales = torch.rand(1, out_dim, device="cuda", dtype=torch.float16) / 8
    gamma = torch.rand(hidden_dim, device="cuda", dtype=torch.float16) + 0.5
    bias = torch.randn(out_dim, device="cuda", dtype=torch.float16) / 16
    layer = FusedRMSNormLinearW4A16AsyncDecode(
        _pack_awq(values), _pack_awq(zeros), scales, gamma, bias,
        cta_n=cta_n,
    )
    input = torch.randn(
        batch, 1, hidden_dim, device="cuda", dtype=torch.float16
    ) / 4
    actual = layer(input)
    dequantized = (
        values.float() - zeros[0].float().unsqueeze(0)
    ) * scales[0].float().unsqueeze(0)
    inverse_rms = torch.rsqrt(
        input.float().square().mean(dim=-1, keepdim=True) + layer.eps
    )
    raw = (input.float() * gamma.float()).half().reshape(
        batch, hidden_dim
    ).float() @ dequantized
    expected = (
        raw * inverse_rms.reshape(batch, 1) + bias.float()
    ).half().unsqueeze(1)
    torch.testing.assert_close(actual, expected, atol=2e-1, rtol=2e-2)


def test_cutlass_layer_llama_hidden_size_groupwise() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")
    torch.manual_seed(20261023)
    batch, hidden_dim, out_dim = 4, 4096, 72
    values = torch.randint(
        0, 16, (hidden_dim, out_dim), device="cuda", dtype=torch.uint8
    )
    zeros = torch.randint(
        0,
        8,
        (hidden_dim // 128, out_dim),
        device="cuda",
        dtype=torch.uint8,
    )
    scales = torch.rand(
        hidden_dim // 128,
        out_dim,
        device="cuda",
        dtype=torch.float16,
    ) / 16
    gamma = torch.rand(hidden_dim, device="cuda", dtype=torch.float16) + 0.5
    layer = FusedRMSNormLinearW4A16Cutlass(
        _pack_awq(values), _pack_awq(zeros), scales, gamma
    )
    input = torch.randn(
        batch, 1, hidden_dim, device="cuda", dtype=torch.float16
    ) / 4
    actual = layer(input)

    group_index = torch.arange(hidden_dim, device="cuda") // 128
    dequantized = (
        values.float() - zeros[group_index].float()
    ) * scales[group_index].float()
    inverse_rms = torch.rsqrt(
        input.float().square().mean(dim=-1, keepdim=True) + layer.eps
    )
    raw = (
        (input.float() * gamma.float())
        .half()
        .reshape(batch, hidden_dim)
        .float()
        @ dequantized
    )
    expected = (raw * inverse_rms.reshape(batch, 1)).half().unsqueeze(1)
    torch.testing.assert_close(actual, expected, atol=1.0, rtol=3e-2)
