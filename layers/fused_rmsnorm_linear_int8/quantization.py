"""Group-wise symmetric INT8 quantization helpers."""

from __future__ import annotations

import torch


DEFAULT_GROUP_SIZE = 64


def quantize_weight_symmetric_int8(
    weight: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[OutDim, HiddenDim]`` weights along HiddenDim groups."""
    if weight.ndim != 2:
        raise ValueError("weight must have shape [OutDim, HiddenDim]")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if weight.shape[1] % group_size != 0:
        raise ValueError(
            f"HiddenDim={weight.shape[1]} must be divisible by {group_size}"
        )

    out_dim, hidden_dim = weight.shape
    grouped = weight.float().reshape(out_dim, hidden_dim // group_size, group_size)
    absolute_maximum = grouped.abs().amax(dim=-1)
    scales = absolute_maximum / 127.0
    safe_scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quantized = torch.round(grouped / safe_scales.unsqueeze(-1))
    quantized = quantized.clamp_(-127, 127).to(torch.int8)
    return quantized.reshape_as(weight).contiguous(), safe_scales.contiguous()


def quantize_activation_symmetric_int8(
    x: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize ``[Batch, 1, HiddenDim]`` activations."""
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError("x must have shape [Batch, 1, HiddenDim]")
    quantized, scales = quantize_weight_symmetric_int8(
        x[:, 0, :], group_size
    )
    return quantized, scales
