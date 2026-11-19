"""Tests for model-load sentinel insertion before Marlin repacking."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear_w4a16_marlin.config import choose_decode_config
from layers.fused_rmsnorm_linear_w4a16_marlin.cuda import (
    w4a16_marlin_sentinel_wmma,
)
from layers.fused_rmsnorm_linear_w4a16_marlin.layer import (
    FusedRMSNormLinearW4A16Marlin,
)
from layers.fused_rmsnorm_linear_w4a16_marlin.packing import (
    _insert_sentinel_columns,
    _pack_awq,
    _unpack_awq,
    pack_awq_marlin_sentinels,
    pack_awq_sentinel_fragment,
    pack_awq_sentinel_wmma,
    quantize_weight_to_awq_groupwise,
)


def test_sentinel_columns_preserve_logical_awq_values() -> None:
    """Logical columns survive unchanged and every stripe has an exact one."""
    torch.manual_seed(7)
    hidden_dim, out_dim, groups, stripe_n = 128, 200, 2, 64
    qweight_values = torch.randint(0, 16, (hidden_dim, out_dim), dtype=torch.uint8)
    qzero_values = torch.randint(0, 16, (groups, out_dim), dtype=torch.uint8)
    scales = torch.rand(groups, out_dim, dtype=torch.float16)

    physical_w, physical_z, physical_s = _insert_sentinel_columns(
        _pack_awq(qweight_values), _pack_awq(qzero_values), scales, stripe_n
    )

    assert physical_w.shape == (hidden_dim, 256)
    restored_w = []
    restored_z = []
    restored_s = []
    for begin in range(0, physical_w.shape[1], stripe_n):
        sentinel = begin + stripe_n - 1
        assert torch.equal(physical_w[:, sentinel], torch.ones(hidden_dim, dtype=torch.uint8))
        assert torch.equal(physical_z[:, sentinel], torch.zeros(groups, dtype=torch.uint8))
        assert torch.equal(physical_s[:, sentinel], torch.ones(groups, dtype=torch.float16))
        restored_w.append(physical_w[:, begin:sentinel])
        restored_z.append(physical_z[:, begin:sentinel])
        restored_s.append(physical_s[:, begin:sentinel])

    assert torch.equal(torch.cat(restored_w, dim=1)[:, :out_dim], qweight_values)
    assert torch.equal(torch.cat(restored_z, dim=1)[:, :out_dim], qzero_values)
    assert torch.equal(torch.cat(restored_s, dim=1)[:, :out_dim], scales)


def test_awq_pack_round_trip() -> None:
    values = torch.randint(0, 16, (4, 24), dtype=torch.uint8)
    assert torch.equal(_unpack_awq(_pack_awq(values), values.shape[1]), values)


def test_groupwise_fp16_weight_quantizes_to_awq_layout() -> None:
    weight = torch.randn(16, 128, dtype=torch.float16)
    qweight, qzeros, scales = quantize_weight_to_awq_groupwise(weight, 128)
    assert qweight.shape == (128, 2)
    assert qzeros.shape == (1, 2)
    assert scales.shape == (1, 16)
    restored = (_unpack_awq(qweight, 16).float() -
                _unpack_awq(qzeros, 16).float()) * scales.float()
    assert torch.isfinite(restored).all()


def test_decode_config_is_selected_per_projection() -> None:
    measured = {64: 8.1, 128: 5.0, 256: 6.3}
    config = choose_decode_config(14336, lambda stripe_n: measured[stripe_n])
    assert config.stripe_n == 128
    assert config.physical_out_dim == 14464


def test_marlin_repack_accepts_physical_sentinel_stripes() -> None:
    """The prefilled layout obeys Marlin's K/N alignment requirements."""
    if not torch.cuda.is_available():
        return
    torch.manual_seed(9)
    hidden_dim, out_dim, groups = 128, 200, 1
    qweight = _pack_awq(
        torch.randint(0, 16, (hidden_dim, out_dim), device="cuda", dtype=torch.uint8)
    )
    qzeros = _pack_awq(
        torch.randint(0, 16, (groups, out_dim), device="cuda", dtype=torch.uint8)
    )
    scales = torch.rand(groups, out_dim, device="cuda", dtype=torch.float16)
    packed = pack_awq_marlin_sentinels(qweight, qzeros, scales)
    assert packed.logical_out_dim == out_dim
    assert packed.physical_out_dim == 256
    assert packed.qweight.is_cuda and packed.scales.is_cuda and packed.zeros.is_cuda
    assert packed.qweight.is_contiguous()


def test_wmma_pack_keeps_w4_storage_and_sentinels() -> None:
    hidden_dim, out_dim, groups = 128, 200, 1
    qweight_values = torch.randint(0, 16, (hidden_dim, out_dim), dtype=torch.uint8)
    qzero_values = torch.randint(0, 16, (groups, out_dim), dtype=torch.uint8)
    scales = torch.rand(groups, out_dim, dtype=torch.float16)
    packed = pack_awq_sentinel_wmma(
        _pack_awq(qweight_values), _pack_awq(qzero_values), scales
    )
    assert packed.qweight.shape == (hidden_dim, 112)
    assert packed.zeros.shape == (groups, 224)
    assert packed.scales.shape == (groups, 224)
    for sentinel in range(31, 224, 32):
        values = packed.qweight[:, sentinel // 2]
        assert torch.equal(values >> 4, torch.ones(hidden_dim, dtype=torch.uint8))
        assert torch.equal(packed.zeros[:, sentinel], torch.zeros(groups, dtype=torch.uint8))
        assert torch.equal(packed.scales[:, sentinel], torch.ones(groups, dtype=torch.float16))


def test_fragment_pack_is_lossless_for_mma_lane_mapping() -> None:
    """The model-load fragment order must preserve every W4 value exactly."""
    hidden_dim, out_dim, stripe_n = 128, 200, 64
    values = torch.randint(0, 16, (hidden_dim, out_dim), dtype=torch.uint8)
    zeros = torch.randint(0, 16, (1, out_dim), dtype=torch.uint8)
    scales = torch.rand(1, out_dim, dtype=torch.float16)
    packed = pack_awq_sentinel_fragment(
        _pack_awq(values), _pack_awq(zeros), scales, stripe_n=stripe_n
    )
    expected, _, _ = _insert_sentinel_columns(
        _pack_awq(values), _pack_awq(zeros), scales, stripe_n
    )
    records = packed.qweight.reshape(
        hidden_dim // 64, packed.physical_out_dim // stripe_n,
        4, stripe_n // 16, 2, 32, 2
    ).contiguous().view(torch.int16).squeeze(-1)
    restored = torch.empty_like(expected)
    for lane in range(32):
        row0 = (lane % 4) * 2
        row1, row2, row3 = row0 + 1, row0 + 8, row0 + 9
        for half_tile, column_base in enumerate((0, 8)):
            source = records[:, :, :, :, half_tile, lane]
            source = source.permute(0, 2, 1, 3)
            column = column_base + lane // 4
            restored_blocks = restored.reshape(
                hidden_dim // 64, 4, 16,
                packed.physical_out_dim // stripe_n, stripe_n // 16, 16
            )
            restored_blocks[:, :, row0, :, :, column] = (source & 15).to(torch.uint8)
            restored_blocks[:, :, row2, :, :, column] = ((source >> 4) & 15).to(torch.uint8)
            restored_blocks[:, :, row1, :, :, column] = ((source >> 8) & 15).to(torch.uint8)
            restored_blocks[:, :, row3, :, :, column] = ((source >> 12) & 15).to(torch.uint8)
    assert torch.equal(restored, expected)


def test_native_marlin_preserves_each_prefilled_sentinel() -> None:
    """Use vLLM only as a numerical oracle for the offline Marlin layout."""
    if not torch.cuda.is_available():
        return
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_awq_marlin_linear,
        marlin_make_empty_g_idx,
        marlin_make_workspace,
    )
    from vllm.scalar_type import scalar_types

    torch.manual_seed(11)
    hidden_dim, out_dim, groups = 128, 200, 1
    qweight_values = torch.randint(
        0, 16, (hidden_dim, out_dim), device="cuda", dtype=torch.uint8
    )
    qzero_values = torch.randint(
        0, 16, (groups, out_dim), device="cuda", dtype=torch.uint8
    )
    scales = torch.rand(groups, out_dim, device="cuda", dtype=torch.float16) / 8
    qweight = _pack_awq(qweight_values)
    qzeros = _pack_awq(qzero_values)
    packed = pack_awq_marlin_sentinels(qweight, qzeros, scales)
    x = torch.randn(2, hidden_dim, device="cuda", dtype=torch.float16) / 4
    actual = apply_awq_marlin_linear(
        input=x,
        weight=packed.qweight,
        weight_scale=packed.scales,
        weight_zp=packed.zeros,
        g_idx=marlin_make_empty_g_idx(x.device),
        g_idx_sort_indices=marlin_make_empty_g_idx(x.device),
        workspace=marlin_make_workspace(packed.physical_out_dim, x.device),
        quant_type=scalar_types.uint4,
        output_size_per_partition=packed.physical_out_dim,
        input_size_per_partition=hidden_dim,
    )
    torch.cuda.synchronize()
    for sentinel in range(
        packed.real_columns_per_stripe,
        packed.physical_out_dim,
        packed.stripe_n,
    ):
        torch.testing.assert_close(
            actual[:, sentinel].float(),
            x.float().sum(dim=1),
            atol=2e-1,
            rtol=2e-2,
        )


@pytest.mark.parametrize("batch", [1, 4, 8])
def test_native_marlin_augmented_rows_recover_fused_rmsnorm_reference(
    batch: int,
) -> None:
    """Validate the exact Marlin-layout contract needed by the fused port.

    The eventual CUDA kernel must generate these two rows in its A shared-memory
    tile instead of materializing them globally.  This test intentionally uses
    vLLM's Marlin only as an oracle: it proves that a native-Marlin-packed
    sentinel column returns ``sum(x**2)`` while the adjacent rows compute the
    gamma-scaled projection.
    """
    if not torch.cuda.is_available():
        return
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_awq_marlin_linear,
        marlin_make_empty_g_idx,
        marlin_make_workspace,
    )
    from vllm.scalar_type import scalar_types

    torch.manual_seed(29)
    hidden_dim, out_dim, groups = 128, 200, 1
    values = torch.randint(
        0, 16, (hidden_dim, out_dim), device="cuda", dtype=torch.uint8
    )
    zeros = torch.randint(
        0, 16, (groups, out_dim), device="cuda", dtype=torch.uint8
    )
    scales = torch.rand(groups, out_dim, device="cuda", dtype=torch.float16) / 16
    gamma = torch.rand(hidden_dim, device="cuda", dtype=torch.float16)
    x = torch.randn(batch, hidden_dim, device="cuda", dtype=torch.float16) / 4
    packed = pack_awq_marlin_sentinels(
        _pack_awq(values), _pack_awq(zeros), scales
    )
    # Row 2*i produces x_i * gamma @ W; row 2*i+1 produces sum(x_i^2)
    # through every prefilled all-one sentinel column.
    augmented = torch.empty(batch * 2, hidden_dim, device="cuda", dtype=torch.float16)
    augmented[0::2] = x * gamma
    augmented[1::2] = x.square()
    actual = apply_awq_marlin_linear(
        input=augmented,
        weight=packed.qweight,
        weight_scale=packed.scales,
        weight_zp=packed.zeros,
        g_idx=marlin_make_empty_g_idx(x.device),
        g_idx_sort_indices=marlin_make_empty_g_idx(x.device),
        workspace=marlin_make_workspace(packed.physical_out_dim, x.device),
        quant_type=scalar_types.uint4,
        output_size_per_partition=packed.physical_out_dim,
        input_size_per_partition=hidden_dim,
    )
    dequantized = (values.float() - zeros.float()) * scales.float()
    expected = (x.float() * gamma.float()) @ dequantized
    sum_sq = x.float().square().sum(dim=1)
    for stripe in range(0, packed.physical_out_dim, packed.stripe_n):
        sentinel = stripe + packed.real_columns_per_stripe
        torch.testing.assert_close(
            actual[1::2, sentinel].float(), sum_sq, atol=2e-1, rtol=2e-2
        )
    recovered = torch.empty_like(expected)
    for stripe in range(0, packed.physical_out_dim, packed.stripe_n):
        start = stripe - stripe // packed.stripe_n
        end = min(start + packed.real_columns_per_stripe, out_dim)
        if start < end:
            recovered[:, start:end] = actual[0::2, stripe:stripe + (end - start)].float()
    torch.testing.assert_close(recovered, expected, atol=2e-1, rtol=3e-2)


@pytest.mark.parametrize("batch", [1, 4, 8])
@pytest.mark.parametrize("stripe_n", [32, 64, 128])
def test_single_kernel_wmma_matches_awq_rmsnorm_reference(
    batch: int, stripe_n: int
) -> None:
    if not torch.cuda.is_available():
        return
    torch.manual_seed(13)
    hidden_dim, out_dim, groups = 128, 200, 1
    qweight_values = torch.randint(
        0, 16, (hidden_dim, out_dim), device="cuda", dtype=torch.uint8
    )
    qzero_values = torch.randint(
        0, 16, (groups, out_dim), device="cuda", dtype=torch.uint8
    )
    scales = torch.rand(groups, out_dim, device="cuda", dtype=torch.float16) / 16
    gamma = torch.rand(hidden_dim, device="cuda", dtype=torch.float16)
    bias = torch.rand(out_dim, device="cuda", dtype=torch.float16) / 8
    x = torch.randn(batch, hidden_dim, device="cuda", dtype=torch.float16) / 4
    packed = pack_awq_sentinel_fragment(
        _pack_awq(qweight_values), _pack_awq(qzero_values), scales,
        stripe_n=stripe_n,
    )
    actual = w4a16_marlin_sentinel_wmma(
        x,
        gamma,
        packed.qweight,
        packed.zeros,
        packed.scales,
        out_dim,
        1e-5,
        bias,
        stripe_n=stripe_n,
    )
    torch.cuda.synchronize()
    dequantized = (qweight_values.float() - qzero_values[0].float()) * scales[0].float()
    expected = (x.float() * gamma.float()) @ dequantized
    expected *= torch.rsqrt(x.float().square().mean(dim=1, keepdim=True) + 1e-5)
    expected += bias.float()
    torch.testing.assert_close(actual.float(), expected, atol=2e-1, rtol=3e-2)


def test_decode_layer_keeps_the_single_kernel_contract() -> None:
    if not torch.cuda.is_available():
        return
    torch.manual_seed(17)
    batch, hidden_dim, out_dim = 4, 128, 200
    qweight_values = torch.randint(
        0, 16, (hidden_dim, out_dim), device="cuda", dtype=torch.uint8
    )
    qzero_values = torch.randint(
        0, 16, (1, out_dim), device="cuda", dtype=torch.uint8
    )
    scales = torch.rand(1, out_dim, device="cuda", dtype=torch.float16) / 16
    gamma = torch.rand(hidden_dim, device="cuda", dtype=torch.float16)
    layer = FusedRMSNormLinearW4A16Marlin(
        _pack_awq(qweight_values), _pack_awq(qzero_values), scales, gamma, 1e-5
    )
    x = torch.randn(batch, 1, hidden_dim, device="cuda", dtype=torch.float16) / 4
    actual = layer(x)
    torch.cuda.synchronize()
    expected = (x[:, 0].float() * gamma.float()) @ (
        (qweight_values.float() - qzero_values[0].float()) * scales[0].float()
    )
    expected *= torch.rsqrt(x[:, 0].float().square().mean(dim=1, keepdim=True) + 1e-5)
    torch.testing.assert_close(actual[:, 0].float(), expected, atol=2e-1, rtol=3e-2)


def test_decode_layer_reuses_workspace_and_resets_locks() -> None:
    """Back-to-back decode calls must not inherit a completed split counter."""
    if not torch.cuda.is_available():
        return
    torch.manual_seed(23)
    hidden_dim, out_dim = 128, 200
    values = torch.randint(0, 16, (hidden_dim, out_dim), device="cuda", dtype=torch.uint8)
    zeros = torch.randint(0, 16, (1, out_dim), device="cuda", dtype=torch.uint8)
    scales = torch.rand(1, out_dim, device="cuda", dtype=torch.float16) / 16
    gamma = torch.rand(hidden_dim, device="cuda", dtype=torch.float16)
    layer = FusedRMSNormLinearW4A16Marlin(
        _pack_awq(values), _pack_awq(zeros), scales, gamma, 1e-5
    )
    workspace_ptr = layer.workspace.data_ptr()
    locks_ptr = layer.locks.data_ptr()
    dequantized = (values.float() - zeros.float()) * scales.float()
    for _ in range(2):
        x = torch.randn(4, 1, hidden_dim, device="cuda", dtype=torch.float16)
        actual = layer(x)
        expected = (x[:, 0].float() * gamma.float()) @ dequantized
        expected *= torch.rsqrt(
            x[:, 0].float().square().mean(dim=1, keepdim=True) + 1e-5
        )
        torch.testing.assert_close(
            actual[:, 0].float(), expected, atol=2e-1, rtol=3e-2
        )
    torch.cuda.synchronize()
    assert layer.workspace.data_ptr() == workspace_ptr
    assert layer.locks.data_ptr() == locks_ptr
    assert torch.count_nonzero(layer.locks).item() == 0
