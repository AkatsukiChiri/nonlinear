"""PyTorch reference for group-wise INT8 fused RMSNorm + Linear."""

from __future__ import annotations

import torch

from .quantization import (
    DEFAULT_GROUP_SIZE,
    quantize_activation_symmetric_int8,
)


def fused_rmsnorm_linear_int8_ref(
    x: torch.Tensor,
    quantized_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> torch.Tensor:
    """Evaluate the same group-wise quantized math as the CUDA kernels."""
    quantized_x, x_scale = quantize_activation_symmetric_int8(x, group_size)
    batch, hidden_dim = quantized_x.shape
    out_dim = quantized_weight.shape[0]
    group_count = hidden_dim // group_size

    x_groups = quantized_x.reshape(batch, group_count, group_size).float()
    weight_groups = quantized_weight.reshape(
        out_dim, group_count, group_size
    ).float()
    integer_dot = torch.einsum("bgk,ogk->bog", x_groups, weight_groups)
    projected = (integer_dot * x_scale[:, None, :] * weight_scale[None, :, :]).sum(
        dim=-1
    )
    inverse_rms = torch.rsqrt(x.float().square().mean(dim=-1) + eps)
    output = projected * inverse_rms
    if bias is not None:
        output = output + bias.float()
    return output.to(dtype=x.dtype).reshape(batch, 1, out_dim)
