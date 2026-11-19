"""Reference for the FP16 Tensor Core sentinel-column decode path."""

from __future__ import annotations

import torch


def tensorcore_rms_linear_decode_ref(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight_fp16: torch.Tensor,
    out_dim: int,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Model the intended fused operation, with FP16 projection operands."""
    residual_sum = x if residual is None else x + residual
    operand = residual_sum.to(torch.float16)
    square_sum = (operand * operand).float().sum(dim=-1, keepdim=True)
    inverse_rms = torch.rsqrt(square_sum / operand.shape[-1] + eps)
    ctas, n_tiles, hidden_dim, _ = weight_fp16.shape
    physical_columns = n_tiles * 16
    packed = weight_fp16.permute(0, 2, 1, 3).reshape(
        ctas, hidden_dim, physical_columns
    )
    unpacked = packed[:, :, :-1].transpose(1, 2).reshape(-1, hidden_dim)
    output = torch.matmul(operand, unpacked[:out_dim].t()) * inverse_rms
    if bias is not None:
        output = output + bias.float()
    output = output.to(x.dtype)
    if residual is None:
        return output
    return output, residual_sum
