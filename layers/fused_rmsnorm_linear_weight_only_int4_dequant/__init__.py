"""Weight-only packed-INT4 fused RMSNorm + Linear decode layer."""

from __future__ import annotations

import torch
from torch import nn

from .cuda import weight_only_int4_dequant_cuda
from .quantization import quantize_weight_symmetric_int4


class FusedRMSNormLinearWeightOnlyInt4Dequant(nn.Module):
    """Keep activations FP16 and dequantize packed INT4 weights online."""

    def __init__(self, rmsnorm: nn.Module, linear: nn.Linear) -> None:
        super().__init__()
        rms_weight = getattr(rmsnorm, "weight", None)
        if not isinstance(rms_weight, torch.Tensor) or rms_weight.ndim != 1:
            raise TypeError("rmsnorm must expose one-dimensional tensor weight")
        if linear.in_features != rms_weight.numel():
            raise ValueError("RMSNorm HiddenDim must equal Linear in_features")
        if linear.in_features % 256:
            raise ValueError("HiddenDim must be divisible by 256")
        if linear.weight.dtype != torch.float16 or rms_weight.dtype != torch.float16:
            raise TypeError("source weights must be float16")
        if linear.weight.device != rms_weight.device:
            raise ValueError("source weights must be on the same device")
        eps = getattr(rmsnorm, "variance_epsilon", getattr(rmsnorm, "eps", None))
        self.eps = float(torch.finfo(torch.float16).eps if eps is None else eps)
        self.hidden_dim = linear.in_features
        self.out_dim = linear.out_features
        with torch.no_grad():
            fused_weight = linear.weight * rms_weight.unsqueeze(0)
            packed_weight, weight_scale = quantize_weight_symmetric_int4(fused_weight)
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer(
            "bias",
            None if linear.bias is None else linear.bias.detach().clone().contiguous(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return output shaped [Batch, 1, OutDim]."""
        if x.ndim != 3 or x.shape[1:] != (1, self.hidden_dim):
            raise ValueError("x must have shape [Batch, 1, HiddenDim]")
        if not 1 <= x.shape[0] <= 8:
            raise ValueError("Batch must be in [1, 8]")
        if x.dtype != torch.float16 or not x.is_contiguous():
            raise TypeError("x must be contiguous float16")
        if x.device != self.packed_weight.device:
            raise ValueError("x and weights must be on the same device")
        return weight_only_int4_dequant_cuda(
            x, self.packed_weight, self.weight_scale, self.bias, self.eps
        )


__all__ = ["FusedRMSNormLinearWeightOnlyInt4Dequant"]
