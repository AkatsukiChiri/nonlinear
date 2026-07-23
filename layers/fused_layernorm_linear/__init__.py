"""Inference layers for decomposed LayerNorm + Linear."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .cuda import fused_layernorm_linear_fp16
from .reference import FusedLayerNormLinearReference


class FusedLayerNormLinear(nn.Module):
    """Fuse a frozen LayerNorm and Linear projection for FP16 inference.

    The source parameters are snapshotted into three buffers:

    - ``fused_weight = linear.weight * layernorm.weight``
    - ``weight_sum = fused_weight.float().sum(dim=-1)``
    - ``fused_bias = linear.weight @ layernorm.bias + linear.bias``

    Forward accepts contiguous FP16 CUDA inputs shaped
    [Batch, 1, HiddenDim]. The optimized Ampere kernel supports Batch 1--8.
    """

    fused_weight: torch.Tensor
    weight_sum: torch.Tensor
    fused_bias: torch.Tensor

    def __init__(self, layernorm: nn.LayerNorm, linear: nn.Linear) -> None:
        super().__init__()
        if len(layernorm.normalized_shape) != 1:
            raise ValueError(
                "layernorm.normalized_shape must contain only HiddenDim"
            )
        hidden_dim = layernorm.normalized_shape[0]
        if hidden_dim != linear.in_features:
            raise ValueError(
                "LayerNorm HiddenDim must equal Linear in_features, got "
                f"{hidden_dim} and {linear.in_features}"
            )
        if not layernorm.elementwise_affine:
            raise ValueError("layernorm must have elementwise_affine=True")
        if layernorm.weight is None or layernorm.bias is None:
            raise ValueError("layernorm must provide both weight and bias")
        if layernorm.weight.device != linear.weight.device:
            raise ValueError("layernorm and linear must be on the same device")
        if layernorm.weight.dtype != linear.weight.dtype:
            raise TypeError("layernorm and linear must use the same dtype")
        if linear.weight.dtype != torch.float16:
            raise TypeError("the fused CUDA kernel requires float16 weights")
        if linear.bias is not None and (
            linear.bias.device != linear.weight.device
            or linear.bias.dtype != linear.weight.dtype
        ):
            raise ValueError(
                "linear bias must match the weight device and dtype"
            )

        self.hidden_dim = hidden_dim
        self.out_dim = linear.out_features
        self.eps = float(layernorm.eps)

        with torch.no_grad():
            fused_weight = (
                linear.weight.float()
                * layernorm.weight.float().unsqueeze(0)
            ).to(dtype=torch.float16)
            # Sum the exact FP16 weight consumed by the projection kernel.
            weight_sum = fused_weight.float().sum(dim=-1).half()
            linear_bias = (
                None if linear.bias is None else linear.bias.float()
            )
            fused_bias = F.linear(
                layernorm.bias.float(),
                linear.weight.float(),
                linear_bias,
            ).half()

        self.register_buffer(
            "fused_weight",
            fused_weight.contiguous(),
        )
        self.register_buffer(
            "weight_sum",
            weight_sum.contiguous(),
        )
        self.register_buffer(
            "fused_bias",
            fused_bias.contiguous(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fused LayerNorm + Linear to [Batch, 1, HiddenDim]."""
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[2] != self.hidden_dim:
            raise ValueError(
                "x must have shape [Batch, 1, HiddenDim], got "
                f"{tuple(x.shape)} with expected HiddenDim={self.hidden_dim}"
            )
        if x.device != self.fused_weight.device:
            raise ValueError("x and fused buffers must be on the same device")
        if x.dtype != torch.float16:
            raise TypeError("x must be float16")
        if self.fused_weight.dtype != torch.float16:
            raise TypeError("fused_weight must remain float16")
        if (
            self.weight_sum.dtype != torch.float16
            or self.fused_bias.dtype != torch.float16
        ):
            raise TypeError("weight_sum and fused_bias must remain float16")
        if not x.is_contiguous():
            raise ValueError("x must be contiguous")

        return fused_layernorm_linear_fp16(
            x,
            self.fused_weight,
            self.weight_sum,
            self.fused_bias,
            self.eps,
        )


__all__ = [
    "FusedLayerNormLinear",
    "FusedLayerNormLinearReference",
]
