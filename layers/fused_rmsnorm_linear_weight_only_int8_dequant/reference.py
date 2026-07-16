"""Reference for FP16 activations and FP16-dequantized INT8 weights."""

from __future__ import annotations

import torch


def weight_only_int8_dequant_ref(
    x: torch.Tensor,
    quantized_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None = None,
    group_size: int = 64,
) -> torch.Tensor:
    """Round each dequantized weight to FP16 before FP32 dot accumulation."""
    batch, _, hidden_dim = x.shape
    out_dim = quantized_weight.shape[0]
    group_count = hidden_dim // group_size
    weight = quantized_weight.reshape(out_dim, group_count, group_size).float()
    weight = (weight * weight_scale.unsqueeze(-1)).to(torch.float16).float()
    projected = torch.matmul(x[:, 0, :].float(), weight.reshape(out_dim, hidden_dim).t())
    inverse_rms = torch.rsqrt(x.float().square().mean(dim=-1) + eps)
    output = projected * inverse_rms
    if bias is not None:
        output = output + bias.float()
    return output.to(x.dtype).reshape(batch, 1, out_dim)
