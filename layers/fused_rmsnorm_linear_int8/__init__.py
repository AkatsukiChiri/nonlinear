"""Inference-only group-wise INT8 fused RMSNorm and Linear layer."""

from __future__ import annotations

import torch
from torch import nn

from .cuda import (
    fused_rmsnorm_linear_bias_int8,
    fused_rmsnorm_linear_int8,
)
from .quantization import DEFAULT_GROUP_SIZE, quantize_weight_symmetric_int8


def _get_rmsnorm_eps(rmsnorm: nn.Module, dtype: torch.dtype) -> float:
    eps = getattr(rmsnorm, "variance_epsilon", None)
    if eps is None:
        eps = getattr(rmsnorm, "eps", None)
    if eps is None:
        eps = torch.finfo(dtype).eps
    return float(eps)


class FusedRMSNormLinearInt8(nn.Module):
    """Fuse RMSNorm into Linear and prequantize its weight group-wise.

    Weight groups contain 64 consecutive HiddenDim values. Activations are
    dynamically quantized with the same grouping on each forward call. The
    layer snapshots all source tensors and is intended for fixed-weight
    inference.
    """

    quantized_weight: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        rmsnorm: nn.Module,
        linear: nn.Linear,
        group_size: int = DEFAULT_GROUP_SIZE,
    ) -> None:
        super().__init__()
        if group_size != DEFAULT_GROUP_SIZE:
            raise ValueError("the CUDA kernel currently requires group_size=64")

        rmsnorm_weight = getattr(rmsnorm, "weight", None)
        if not isinstance(rmsnorm_weight, torch.Tensor):
            raise TypeError("rmsnorm must expose a tensor weight")
        if rmsnorm_weight.ndim != 1:
            raise ValueError("rmsnorm.weight must have shape [HiddenDim]")
        if linear.in_features != rmsnorm_weight.numel():
            raise ValueError("RMSNorm HiddenDim must equal Linear in_features")
        if linear.in_features % group_size != 0:
            raise ValueError(
                f"HiddenDim={linear.in_features} must be divisible by {group_size}"
            )
        if linear.weight.device != rmsnorm_weight.device:
            raise ValueError("rmsnorm and linear weights must be on the same device")
        if linear.weight.dtype != rmsnorm_weight.dtype:
            raise ValueError("rmsnorm and linear weights must have the same dtype")
        if linear.weight.dtype != torch.float16:
            raise TypeError("the INT8 CUDA kernel requires FP16 source weights")

        self.hidden_dim = linear.in_features
        self.out_dim = linear.out_features
        self.group_size = group_size
        self.eps = _get_rmsnorm_eps(rmsnorm, linear.weight.dtype)

        with torch.no_grad():
            fused_weight = linear.weight * rmsnorm_weight.unsqueeze(0)
            quantized_weight, weight_scale = quantize_weight_symmetric_int8(
                fused_weight, group_size
            )
        self.register_buffer("quantized_weight", quantized_weight)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer(
            "bias",
            None if linear.bias is None else linear.bias.detach().clone().contiguous(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return FP16 output shaped [Batch, 1, OutDim]."""
        if x.ndim != 3 or x.shape[1:] != (1, self.hidden_dim):
            raise ValueError(
                "x must have shape [Batch, 1, HiddenDim], got "
                f"{tuple(x.shape)} with expected HiddenDim={self.hidden_dim}"
            )
        if not 1 <= x.shape[0] <= 8:
            raise ValueError("Batch must be in [1, 8]")
        if x.device != self.quantized_weight.device:
            raise ValueError("x and quantized_weight must be on the same device")
        if x.dtype != torch.float16:
            raise TypeError("x must be float16")
        if not x.is_contiguous():
            raise ValueError("x must be contiguous")

        if self.bias is None:
            return fused_rmsnorm_linear_int8(
                x, self.quantized_weight, self.weight_scale, self.eps
            )
        return fused_rmsnorm_linear_bias_int8(
            x,
            self.quantized_weight,
            self.weight_scale,
            self.bias,
            self.eps,
        )


__all__ = ["FusedRMSNormLinearInt8"]
