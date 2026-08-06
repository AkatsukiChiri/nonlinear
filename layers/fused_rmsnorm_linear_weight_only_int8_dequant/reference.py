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
    """Round each dequantized weight to the activation dtype before FP32 math."""
    batch, _, hidden_dim = x.shape
    out_dim = quantized_weight.shape[0]
    if weight_scale.shape[1] == 1:
        group_size = hidden_dim
    group_count = hidden_dim // group_size
    weight = quantized_weight.reshape(out_dim, group_count, group_size).float()
    weight = (weight * weight_scale.unsqueeze(-1)).to(x.dtype).float()
    projected = torch.matmul(x[:, 0, :].float(), weight.reshape(out_dim, hidden_dim).t())
    inverse_rms = torch.rsqrt(x.float().square().mean(dim=-1) + eps)
    output = projected * inverse_rms
    if bias is not None:
        output = output + bias.float()
    return output.to(x.dtype).reshape(batch, 1, out_dim)


def fused_add_weight_only_int8_dequant_ref(
    x: torch.Tensor,
    x_res: torch.Tensor,
    quantized_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None = None,
    group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference residual add followed by weight-only RMSNorm + Linear."""
    residual_sum = x + x_res
    output = weight_only_int8_dequant_ref(
        residual_sum,
        quantized_weight,
        weight_scale,
        eps,
        bias,
        group_size,
    )
    return output, residual_sum
