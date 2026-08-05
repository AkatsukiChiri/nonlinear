"""Eager reference for weight-only INT4 decode."""

from __future__ import annotations

import torch

from .quantization import unpack_weight_int4


def weight_only_int4_dequant_ref(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Model FP16-rounded online weight dequantization and FP32 dot products."""
    batch, _, hidden_dim = x.shape
    out_dim = packed_weight.shape[0]
    weight = unpack_weight_int4(packed_weight).float()
    weight = (weight * weight_scale.float()).to(torch.float16).float()
    projected = x[:, 0, :].float() @ weight.t()
    inverse_rms = torch.rsqrt(x.float().square().mean(dim=-1) + eps)
    output = projected * inverse_rms
    if bias is not None:
        output = output + bias.float()
    return output.to(x.dtype).reshape(batch, 1, out_dim)
