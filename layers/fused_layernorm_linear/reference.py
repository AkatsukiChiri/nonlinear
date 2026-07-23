"""PyTorch reference for the decomposed LayerNorm + Linear computation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FusedLayerNormLinearReference(nn.Module):
    """Inference-only PyTorch implementation of decomposed LayerNorm + Linear.

    Given LayerNorm parameters ``gamma`` and ``beta`` and PyTorch Linear weight
    ``W`` shaped [OutDim, HiddenDim], construction folds the parameters as:

        fused_weight = W * gamma
        fused_bias = W @ beta + linear_bias
        fused_weight_sum = fused_weight.sum(dim=-1)

    For an input shaped [..., HiddenDim], forward computes ``x @
    fused_weight.T``, ``sum(x)``, and ``sum(x.square())`` before recovering the
    result with broadcast element-wise operations. Reductions and the recovery
    epilogue use FP32 for FP16/BF16 inputs.

    Source parameters are copied into buffers, so this module is intended for
    fixed-weight inference rather than training.
    """

    fused_weight: torch.Tensor
    fused_bias: torch.Tensor
    fused_weight_sum: torch.Tensor

    def __init__(
        self,
        fused_weight: torch.Tensor,
        fused_bias: torch.Tensor,
        fused_weight_sum: torch.Tensor,
        eps: float,
    ) -> None:
        super().__init__()
        if fused_weight.ndim != 2:
            raise ValueError(
                "fused_weight must have shape [OutDim, HiddenDim], "
                f"got {tuple(fused_weight.shape)}"
            )
        out_dim, hidden_dim = fused_weight.shape
        if fused_bias.shape != (out_dim,):
            raise ValueError(
                f"fused_bias must have shape [{out_dim}], "
                f"got {tuple(fused_bias.shape)}"
            )
        if fused_weight_sum.shape != (out_dim,):
            raise ValueError(
                f"fused_weight_sum must have shape [{out_dim}], "
                f"got {tuple(fused_weight_sum.shape)}"
            )

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.eps = float(eps)
        self.register_buffer("fused_weight", fused_weight.detach().clone())
        self.register_buffer("fused_bias", fused_bias.detach().clone())
        self.register_buffer(
            "fused_weight_sum",
            fused_weight_sum.detach().clone(),
        )

    @classmethod
    def from_modules(
        cls,
        layernorm: nn.LayerNorm,
        linear: nn.Linear,
    ) -> FusedLayerNormLinearReference:
        """Fold a frozen LayerNorm and Linear into inference parameters."""
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

        with torch.no_grad():
            # PyTorch Linear stores W as [OutDim, HiddenDim], so multiplying
            # columns by gamma is equivalent to diag(gamma) @ W in the
            # row-vector notation used by the derivation document.
            fused_weight = linear.weight * layernorm.weight.unsqueeze(0)
            fused_bias = F.linear(
                layernorm.bias,
                linear.weight,
                linear.bias,
            )
            fused_weight_sum = fused_weight.sum(dim=-1)

        return cls(
            fused_weight=fused_weight,
            fused_bias=fused_bias,
            fused_weight_sum=fused_weight_sum,
            eps=layernorm.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply decomposed LayerNorm + Linear to [..., HiddenDim]."""
        if x.ndim < 1 or x.shape[-1] != self.hidden_dim:
            raise ValueError(
                "x must have shape [..., HiddenDim] with "
                f"HiddenDim={self.hidden_dim}, got {tuple(x.shape)}"
            )
        if x.device != self.fused_weight.device:
            raise ValueError("x and fused parameters must be on the same device")
        if x.dtype != self.fused_weight.dtype:
            raise TypeError("x and fused parameters must use the same dtype")

        output_dtype = x.dtype
        x_float = x.float()
        fused_weight_float = self.fused_weight.float()

        projection = F.linear(x_float, fused_weight_float)
        input_sum = x_float.sum(dim=-1, keepdim=True)
        input_square_sum = x_float.square().sum(dim=-1, keepdim=True)

        mean = input_sum / self.hidden_dim
        variance = input_square_sum / self.hidden_dim - mean.square()
        inverse_std = torch.rsqrt(variance.clamp_min(0.0) + self.eps)

        output = (
            projection - mean * self.fused_weight_sum.float()
        ) * inverse_std + self.fused_bias.float()
        return output.to(dtype=output_dtype)
