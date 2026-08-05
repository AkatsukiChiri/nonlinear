"""Quantized eager reference for INT8 prefill."""

from __future__ import annotations

import torch


def fused_rmsnorm_linear_int8_prefill_ref(
    x: torch.Tensor,
    quantized_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Model per-token activation quantization and INT32 accumulation."""
    batch, seq_len, hidden_dim = x.shape
    rows = x.reshape(-1, hidden_dim).float()
    absolute_maximum = rows.abs().amax(dim=1)
    activation_scale = absolute_maximum / 127.0
    safe_scale = torch.where(
        activation_scale > 0, activation_scale, torch.ones_like(activation_scale)
    )
    quantized_x = torch.round(rows / safe_scale[:, None]).clamp(-127, 127)
    accumulator = quantized_x @ quantized_weight.float().t()
    inverse_rms = torch.rsqrt(rows.square().mean(dim=1) + eps)
    output = accumulator * (safe_scale * inverse_rms)[:, None]
    output = output * weight_scale.float()[None, :]
    if bias is not None:
        output = output + bias.float()
    return output.to(x.dtype).reshape(batch, seq_len, -1)
