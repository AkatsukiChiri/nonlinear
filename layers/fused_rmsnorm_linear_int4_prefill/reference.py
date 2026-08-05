"""Quantized eager reference for INT4 prefill."""

from __future__ import annotations

import torch

from layers.fused_rmsnorm_linear_weight_only_int4_dequant.quantization import (
    unpack_weight_int4,
)


def fused_rmsnorm_linear_int4_prefill_ref(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Model per-token INT4 quantization and INT32 accumulation."""
    batch, seq_len, hidden_dim = x.shape
    rows = x.reshape(-1, hidden_dim).float()
    activation_scale = rows.abs().amax(dim=1) / 7.0
    safe_scale = torch.where(
        activation_scale > 0, activation_scale, torch.ones_like(activation_scale)
    )
    quantized_x = torch.round(rows / safe_scale[:, None]).clamp(-7, 7)
    quantized_weight = unpack_weight_int4(packed_weight).float()
    accumulator = quantized_x @ quantized_weight.t()
    inverse_rms = torch.rsqrt(rows.square().mean(dim=1) + eps)
    output = accumulator * (safe_scale * inverse_rms)[:, None]
    output = output * weight_scale.float()[None, :]
    if bias is not None:
        output = output + bias.float()
    return output.to(x.dtype).reshape(batch, seq_len, -1)
