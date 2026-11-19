from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear_w4a16_cutlass import pack_awq_with_sentinels


def _pack_awq(values: torch.Tensor) -> torch.Tensor:
    shifts = torch.tensor(
        [0, 16, 4, 20, 8, 24, 12, 28], dtype=torch.int64
    )
    packed = torch.zeros(values.shape[0], values.shape[1] // 8, dtype=torch.int64)
    chunks = values.reshape(values.shape[0], -1, 8).to(torch.int64)
    for index in range(8):
        packed |= chunks[..., index] << shifts[index]
    return packed.to(torch.int32)


@pytest.mark.parametrize(
    ("layout", "expected_tile_n", "expected_sentinels"),
    [
        ("n_minus_1", 64, [63, 127]),
        ("n_plus_1", 65, [64, 129]),
    ],
)
def test_pack_awq_with_sentinels(
    layout: str,
    expected_tile_n: int,
    expected_sentinels: list[int],
) -> None:
    hidden_dim, out_dim = 128, 72
    logical_weight = torch.randint(0, 16, (hidden_dim, out_dim), dtype=torch.uint8)
    logical_zeros = torch.randint(0, 8, (1, out_dim), dtype=torch.uint8)
    scales = torch.rand(1, out_dim, dtype=torch.float16)
    packed = pack_awq_with_sentinels(
        _pack_awq(logical_weight),
        _pack_awq(logical_zeros),
        scales,
        cta_n=64,
        layout=layout,  # type: ignore[arg-type]
    )

    weight_bytes = packed.physical_out_dim * hidden_dim // 2
    zero_bytes = packed.physical_out_dim
    weights = packed.storage[:weight_bytes].reshape(
        packed.physical_out_dim, hidden_dim // 2
    )
    unpacked = torch.empty(
        packed.physical_out_dim, hidden_dim, dtype=torch.uint8
    )
    unpacked[:, 0::2] = weights & 0xF
    unpacked[:, 1::2] = weights >> 4
    zeros = packed.storage[
        weight_bytes : weight_bytes + zero_bytes
    ]
    stored_scales = packed.storage[weight_bytes + zero_bytes :].view(
        torch.float16
    )

    assert packed.physical_tile_n == expected_tile_n
    assert packed.sentinel_columns.tolist() == expected_sentinels
    torch.testing.assert_close(
        unpacked[packed.real_to_physical.long()].transpose(0, 1),
        logical_weight,
        atol=0,
        rtol=0,
    )
    for sentinel in expected_sentinels:
        assert torch.all(unpacked[sentinel] == 1)
        assert zeros[sentinel].item() == 0
        assert stored_scales[sentinel].item() == 1
