"""Packed symmetric signed-INT4 weight quantization."""

from __future__ import annotations

import torch


def quantize_weight_symmetric_int4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize per output channel and pack adjacent K values into bytes."""
    if weight.ndim != 2 or weight.shape[1] % 2:
        raise ValueError("weight must have even-shaped [OutDim, HiddenDim]")
    absolute_maximum = weight.float().abs().amax(dim=1, keepdim=True)
    scales = absolute_maximum / 7.0
    safe_scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quantized = torch.round(weight.float() / safe_scales).clamp_(-7, 7).to(torch.int8)
    low = quantized[:, 0::2].to(torch.int16) & 0xF
    high = (quantized[:, 1::2].to(torch.int16) & 0xF) << 4
    return (low | high).to(torch.uint8).contiguous(), safe_scales.contiguous()


def unpack_weight_int4(packed_weight: torch.Tensor) -> torch.Tensor:
    """Unpack signed nibbles to an INT8 tensor for eager references."""
    low = (packed_weight & 0xF).to(torch.int8)
    high = ((packed_weight >> 4) & 0xF).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    output = torch.empty(
        packed_weight.shape[0], packed_weight.shape[1] * 2,
        dtype=torch.int8, device=packed_weight.device,
    )
    output[:, 0::2] = low
    output[:, 1::2] = high
    return output
