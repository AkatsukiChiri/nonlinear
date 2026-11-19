"""Offline AWQ sentinel insertion followed by vLLM Marlin repacking."""

from __future__ import annotations

from dataclasses import dataclass

import torch


_AWQ_SHIFTS = (0, 16, 4, 20, 8, 24, 12, 28)


@dataclass(frozen=True)
class MarlinSentinelPack:
    """Marlin-ready weights and fixed physical-to-logical stripe metadata."""

    qweight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    logical_out_dim: int
    physical_out_dim: int
    stripe_n: int
    real_columns_per_stripe: int


@dataclass(frozen=True)
class MarlinSentinelWmmaPack:
    """W4 physical stripes in the decode kernel's tile-major source layout.

    The values remain W4 in memory.  The CUDA kernel expands only its current
    ``16 x 128`` K/N tile to shared FP16 immediately before Tensor Core MMA.
    """

    qweight: torch.Tensor
    zeros: torch.Tensor
    scales: torch.Tensor
    logical_out_dim: int
    physical_out_dim: int
    stripe_n: int
    real_columns_per_stripe: int


def _unpack_awq(packed: torch.Tensor, out_dim: int) -> torch.Tensor:
    shifts = torch.tensor(_AWQ_SHIFTS, device=packed.device, dtype=torch.int32)
    return ((packed.unsqueeze(-1) >> shifts) & 15).reshape(
        *packed.shape[:-1], out_dim
    ).to(torch.uint8)


def _pack_awq(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] % 8:
        raise ValueError("AWQ packing requires a multiple of eight columns")
    result = torch.zeros(
        *values.shape[:-1], values.shape[-1] // 8,
        device=values.device,
        dtype=torch.int32,
    )
    for index, shift in enumerate(_AWQ_SHIFTS):
        result |= values[..., index::8].to(torch.int32) << shift
    return result


def quantize_weight_to_awq_groupwise(
    weight: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize an FP16 `[OutDim, HiddenDim]` weight into AWQ-compatible W4.

    The result uses the `[HiddenDim, OutDim / 8]` AWQ packing consumed by both
    vLLM Marlin and the fused decode layer. This is an offline model-load
    utility for FP16 Llama checkpoints, not an inference-path operation.
    """
    if weight.ndim != 2 or weight.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("weight must be rank-two float16 or bfloat16")
    out_dim, hidden_dim = weight.shape
    if hidden_dim % group_size or out_dim % 8:
        raise ValueError("hidden_dim must divide group_size and out_dim must divide 8")
    # AWQ quantizes independently for every input-K group and output channel.
    columns = weight.transpose(0, 1).contiguous().reshape(
        hidden_dim // group_size, group_size, out_dim
    ).float()
    minimum = columns.amin(dim=1)
    maximum = columns.amax(dim=1)
    scales = ((maximum - minimum) / 15.0).clamp_min(1e-6)
    zeros = torch.round(-minimum / scales).clamp_(0, 15).to(torch.uint8)
    quantized = torch.round(columns / scales.unsqueeze(1) + zeros.unsqueeze(1))
    quantized = quantized.clamp_(0, 15).to(torch.uint8).reshape(hidden_dim, out_dim)
    return _pack_awq(quantized), _pack_awq(zeros), scales.to(torch.float16)


def _insert_sentinel_columns(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    stripe_n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand logical AWQ tensors into physical stripes with sentinels.

    This deliberately happens before Marlin's opaque repack operation so the
    sentinel invariant can be unit-tested independently of a GPU extension.
    """
    hidden_dim, packed_n = qweight.shape
    out_dim = scales.shape[1]
    groups = scales.shape[0]
    real_n = stripe_n - 1
    stripes = (out_dim + real_n - 1) // real_n
    physical_n = stripes * stripe_n
    source_w = _unpack_awq(qweight, out_dim)
    source_z = _unpack_awq(qzeros, out_dim)
    physical_w = torch.zeros(
        hidden_dim, physical_n, dtype=torch.uint8, device=qweight.device
    )
    physical_z = torch.zeros(
        groups, physical_n, dtype=torch.uint8, device=qweight.device
    )
    physical_s = torch.zeros(
        groups, physical_n, dtype=torch.float16, device=qweight.device
    )
    for stripe in range(stripes):
        logical_begin = stripe * real_n
        logical_end = min(logical_begin + real_n, out_dim)
        physical_begin = stripe * stripe_n
        count = logical_end - logical_begin
        physical_w[:, physical_begin:physical_begin + count] = source_w[:, logical_begin:logical_end]
        physical_z[:, physical_begin:physical_begin + count] = source_z[:, logical_begin:logical_end]
        physical_s[:, physical_begin:physical_begin + count] = scales[:, logical_begin:logical_end]
        # Exact dequantization: (q - zero) * scale = (1 - 0) * 1 = 1.
        sentinel = physical_begin + real_n
        physical_w[:, sentinel] = 1
        physical_s[:, sentinel] = 1
    return physical_w, physical_z, physical_s


def pack_awq_marlin_sentinels(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    *,
    stripe_n: int = 128,
) -> MarlinSentinelPack:
    """Build a Marlin layout with one exact all-one column per N stripe.

    Every physical stripe contains ``stripe_n - 1`` real channels followed by
    a sentinel.  All transformation happens at model load: the inference
    kernel therefore receives only Marlin-repacked W4 and never writes a
    sentinel or performs an online repack.
    """
    if stripe_n <= 1 or stripe_n % 64:
        raise ValueError("stripe_n must be a multiple of 64 and greater than one")
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise TypeError("AWQ qweight and qzeros must be int32")
    if scales.dtype != torch.float16:
        raise TypeError("AWQ scales must be float16")
    hidden_dim, packed_n = qweight.shape
    out_dim = scales.shape[1]
    groups = scales.shape[0]
    if packed_n * 8 != out_dim or qzeros.shape != (groups, packed_n):
        raise ValueError("inconsistent AWQ tensor shapes")
    if hidden_dim % groups:
        raise ValueError("invalid AWQ group count")

    physical_w, physical_z, physical_s = _insert_sentinel_columns(
        qweight, qzeros, scales, stripe_n
    )
    physical_n = physical_w.shape[1]

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        awq_to_marlin_zero_points,
        marlin_permute_scales,
    )

    packed_w = _pack_awq(physical_w).contiguous()
    packed_z = _pack_awq(physical_z).contiguous()
    return MarlinSentinelPack(
        qweight=ops.awq_marlin_repack(packed_w, hidden_dim, physical_n, 4),
        scales=marlin_permute_scales(
            physical_s, hidden_dim, physical_n, hidden_dim // groups
        ),
        zeros=awq_to_marlin_zero_points(packed_z, groups, physical_n, 4),
        logical_out_dim=out_dim,
        physical_out_dim=physical_n,
        stripe_n=stripe_n,
        real_columns_per_stripe=stripe_n - 1,
    )


def pack_awq_sentinel_wmma(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    *,
    stripe_n: int = 32,
) -> MarlinSentinelWmmaPack:
    """Prepare physical W4 sentinel stripes for the fused Tensor Core path.

    Unlike :func:`pack_awq_marlin_sentinels`, this leaves the tile source in a
    simple K-major nibble format.  It is the bring-up layout for the single
    kernel and is intentionally separate from vLLM's opaque Marlin repack;
    the later cp.async port can change this loader without changing model
    semantics or the sentinel invariant.
    """
    if stripe_n not in (32, 64, 128):
        raise ValueError("the direct-fragment decode kernel supports N=32, 64, or 128")
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise TypeError("AWQ qweight and qzeros must be int32")
    if scales.dtype != torch.float16:
        raise TypeError("AWQ scales must be float16")
    hidden_dim, packed_n = qweight.shape
    out_dim = scales.shape[1]
    groups = scales.shape[0]
    if packed_n * 8 != out_dim or qzeros.shape != (groups, packed_n):
        raise ValueError("inconsistent AWQ tensor shapes")
    physical_w, physical_z, physical_s = _insert_sentinel_columns(
        qweight, qzeros, scales, stripe_n
    )
    packed_w = (
        physical_w[:, 0::2].to(torch.uint8)
        | (physical_w[:, 1::2].to(torch.uint8) << 4)
    ).contiguous()
    return MarlinSentinelWmmaPack(
        qweight=packed_w,
        zeros=physical_z.contiguous(),
        scales=physical_s.contiguous(),
        logical_out_dim=out_dim,
        physical_out_dim=physical_w.shape[1],
        stripe_n=stripe_n,
        real_columns_per_stripe=stripe_n - 1,
    )


def pack_awq_sentinel_fragment(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    *,
    stripe_n: int = 64,
) -> MarlinSentinelWmmaPack:
    """Pack W4 values in the exact per-lane order consumed by decode MMA.

    Storage remains exactly four bits per weight.  The previous K-major layout
    required four byte/nibble loads for every B fragment.  Here each lane's
    four K values are stored as one 16-bit record in the order ``q0,q2,q1,q3``;
    the CUDA path expands that record directly into the Marlin LOP3 format.
    """
    if stripe_n not in (32, 64, 128):
        raise ValueError("the fragment decode kernel supports N=32, 64, or 128")
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise TypeError("AWQ qweight and qzeros must be int32")
    if scales.dtype != torch.float16:
        raise TypeError("AWQ scales must be float16")
    hidden_dim, packed_n = qweight.shape
    out_dim = scales.shape[1]
    groups = scales.shape[0]
    if hidden_dim % 64 or packed_n * 8 != out_dim or qzeros.shape != (groups, packed_n):
        raise ValueError("fragment packing requires K divisible by 64 and consistent AWQ shapes")
    physical_w, physical_z, physical_s = _insert_sentinel_columns(
        qweight, qzeros, scales, stripe_n
    )
    physical_n = physical_w.shape[1]
    stripes = physical_n // stripe_n
    n_tiles = stripe_n // 16
    tiles = physical_w.reshape(hidden_dim // 64, 4, 16, stripes, n_tiles, 16)
    lanes = torch.arange(32, device=qweight.device)
    row0 = (lanes.remainder(4) * 2).long()
    row1 = row0 + 1
    row2 = row0 + 8
    row3 = row1 + 8
    column0 = (lanes // 4).long()
    column1 = column0 + 8

    def pack_fragment(columns: torch.Tensor) -> torch.Tensor:
        # Advanced row/column indexing puts the lane axis first:
        # [lane, K64 block, K16 subtile, stripe, N16 subtile].
        # Normalize it to the CUDA copy order below.
        q0 = tiles[:, :, row0, :, :, columns].permute(1, 3, 2, 4, 0)
        q1 = tiles[:, :, row1, :, :, columns].permute(1, 3, 2, 4, 0)
        q2 = tiles[:, :, row2, :, :, columns].permute(1, 3, 2, 4, 0)
        q3 = tiles[:, :, row3, :, :, columns].permute(1, 3, 2, 4, 0)
        return (q0.to(torch.int16) | (q2.to(torch.int16) << 4) |
                (q1.to(torch.int16) << 8) | (q3.to(torch.int16) << 12))

    first = pack_fragment(column0)
    second = pack_fragment(column1)
    # [K64 block, stripe, K16 subtile, N16 subtile, B half, lane].  Every
    # 16x16 W4 tile is still 128 bytes: 2 * 32 lanes * uint16.
    fragments = torch.stack((first, second), dim=4).contiguous()
    packed_bytes = fragments.view(torch.uint8).reshape(hidden_dim, physical_n // 2)
    return MarlinSentinelWmmaPack(
        qweight=packed_bytes.contiguous(),
        zeros=physical_z.contiguous(),
        scales=physical_s.contiguous(),
        logical_out_dim=out_dim,
        physical_out_dim=physical_n,
        stripe_n=stripe_n,
        real_columns_per_stripe=stripe_n - 1,
    )
