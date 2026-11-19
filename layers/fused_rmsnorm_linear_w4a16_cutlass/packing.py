"""Model-load-time packing for CUTLASS W4A16 sentinel kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


SentinelLayout = Literal["n_minus_1", "n_plus_1"]


@dataclass(frozen=True)
class PackedSentinelWeight:
    """Packed storage and column mapping consumed by the CUDA iterator."""

    storage: torch.Tensor
    real_to_physical: torch.Tensor
    sentinel_columns: torch.Tensor
    logical_out_dim: int
    physical_out_dim: int
    cta_n: int
    physical_tile_n: int
    layout: SentinelLayout


def _unpack_awq_int32(packed: torch.Tensor, out_dim: int) -> torch.Tensor:
    """Unpack AutoAWQ's eight-nibble int32 order into its last dimension."""
    shifts = torch.tensor(
        [0, 16, 4, 20, 8, 24, 12, 28],
        dtype=torch.int32,
        device=packed.device,
    )
    return ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(
        *packed.shape[:-1], out_dim
    ).to(torch.uint8)


def pack_awq_with_sentinels(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    *,
    cta_n: int,
    layout: SentinelLayout,
) -> PackedSentinelWeight:
    """Insert prequantized all-one columns without requantizing real weights.

    ``qweight`` has AutoAWQ shape ``[HiddenDim, OutDim / 8]``. The returned
    byte tensor is ``[packed W4][uint8 zero][FP16 scale]``. Real columns are
    copied bit-exactly after unpack/repack. Sentinel columns use q=1, zero=0,
    and scale=1 for every K group, so their GEMM result is ``sum(z**2)``.
    """
    if qweight.dtype != torch.int32 or qweight.ndim != 2:
        raise TypeError("qweight must be int32 [HiddenDim, OutDim / 8]")
    if qzeros.dtype != torch.int32 or qzeros.ndim != 2:
        raise TypeError("qzeros must be int32 [Groups, OutDim / 8]")
    if scales.dtype != torch.float16 or scales.ndim != 2:
        raise TypeError("scales must be float16 [Groups, OutDim]")
    if not (qweight.device == qzeros.device == scales.device):
        raise ValueError("qweight, qzeros, and scales must share a device")
    hidden_dim = qweight.shape[0]
    out_dim = scales.shape[1]
    if out_dim % 8 or qweight.shape[1] * 8 != out_dim:
        raise ValueError("inconsistent AWQ output dimension")
    if qzeros.shape != (scales.shape[0], out_dim // 8):
        raise ValueError("qzeros shape does not match scales")
    if hidden_dim % scales.shape[0]:
        raise ValueError("invalid AWQ group count")
    if cta_n <= 1 or cta_n % 8:
        raise ValueError("cta_n must be a positive multiple of eight")
    if layout not in ("n_minus_1", "n_plus_1"):
        raise ValueError(f"unsupported sentinel layout: {layout}")

    real_per_tile = cta_n - 1 if layout == "n_minus_1" else cta_n
    physical_tile_n = cta_n if layout == "n_minus_1" else cta_n + 1
    tile_count = (out_dim + real_per_tile - 1) // real_per_tile
    physical_out_dim = tile_count * physical_tile_n

    unpacked_weight = _unpack_awq_int32(qweight, out_dim).transpose(0, 1)
    unpacked_zeros = _unpack_awq_int32(qzeros, out_dim)
    physical_weight = torch.zeros(
        physical_out_dim,
        hidden_dim,
        dtype=torch.uint8,
        device=qweight.device,
    )
    physical_zeros = torch.zeros(
        scales.shape[0],
        physical_out_dim,
        dtype=torch.uint8,
        device=qweight.device,
    )
    physical_scales = torch.zeros(
        scales.shape[0],
        physical_out_dim,
        dtype=torch.float16,
        device=qweight.device,
    )
    real_to_physical = torch.empty(
        out_dim, dtype=torch.int32, device=qweight.device
    )
    sentinel_columns = torch.empty(
        tile_count, dtype=torch.int32, device=qweight.device
    )

    for tile in range(tile_count):
        real_begin = tile * real_per_tile
        real_end = min(real_begin + real_per_tile, out_dim)
        physical_begin = tile * physical_tile_n
        count = real_end - real_begin
        physical_weight[physical_begin : physical_begin + count].copy_(
            unpacked_weight[real_begin:real_end]
        )
        physical_zeros[:, physical_begin : physical_begin + count].copy_(
            unpacked_zeros[:, real_begin:real_end]
        )
        physical_scales[:, physical_begin : physical_begin + count].copy_(
            scales[:, real_begin:real_end]
        )
        real_to_physical[real_begin:real_end] = torch.arange(
            physical_begin,
            physical_begin + count,
            dtype=torch.int32,
            device=qweight.device,
        )
        sentinel = physical_begin + real_per_tile
        physical_weight[sentinel].fill_(1)
        physical_zeros[:, sentinel].zero_()
        physical_scales[:, sentinel].fill_(1)
        sentinel_columns[tile] = sentinel

    packed_values = (
        physical_weight[:, 0::2]
        | (physical_weight[:, 1::2] << 4)
    ).contiguous()
    storage = torch.cat(
        [
            packed_values.view(-1),
            physical_zeros.contiguous().view(-1),
            physical_scales.contiguous().view(torch.uint8).view(-1),
        ]
    )
    return PackedSentinelWeight(
        storage=storage,
        real_to_physical=real_to_physical,
        sentinel_columns=sentinel_columns,
        logical_out_dim=out_dim,
        physical_out_dim=physical_out_dim,
        cta_n=cta_n,
        physical_tile_n=physical_tile_n,
        layout=layout,
    )
