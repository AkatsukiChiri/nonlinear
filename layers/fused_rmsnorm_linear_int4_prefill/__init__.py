"""Prefill-only INT4 Tensor Core RMSNorm + Linear layer."""

from __future__ import annotations

import torch
from torch import nn

from layers.fused_rmsnorm_linear_weight_only_int4_dequant.quantization import (
    quantize_weight_symmetric_int4,
)

from .cuda import fused_rmsnorm_linear_int4_prefill_cuda


class FusedRMSNormLinearInt4Prefill(nn.Module):
    """Use per-token/per-output INT4 quantization for prefill inference."""

    def __init__(self, rmsnorm: nn.Module, linear: nn.Linear) -> None:
        super().__init__()
        rms_weight = getattr(rmsnorm, "weight", None)
        if not isinstance(rms_weight, torch.Tensor) or rms_weight.ndim != 1:
            raise TypeError("rmsnorm must expose one-dimensional weight")
        if linear.in_features != rms_weight.numel():
            raise ValueError("RMSNorm HiddenDim must match Linear in_features")
        if linear.weight.dtype != torch.float16 or rms_weight.dtype != torch.float16:
            raise TypeError("source weights must be float16")
        if linear.weight.device != rms_weight.device:
            raise ValueError("source weights must be on the same device")
        if linear.in_features % 64 or linear.out_features % 8:
            raise ValueError("HiddenDim must be divisible by 64 and OutDim by 8")
        eps = getattr(rmsnorm, "variance_epsilon", getattr(rmsnorm, "eps", None))
        self.eps = float(torch.finfo(torch.float16).eps if eps is None else eps)
        self.hidden_dim = linear.in_features
        self.out_dim = linear.out_features
        with torch.no_grad():
            fused_weight = linear.weight * rms_weight.unsqueeze(0)
            packed_weight, weight_scale = quantize_weight_symmetric_int4(fused_weight)
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("weight_scale", weight_scale.reshape(-1))
        self.register_buffer(
            "bias",
            None if linear.bias is None else linear.bias.detach().clone().contiguous(),
        )
        self._quantized_x_workspace: torch.Tensor | None = None
        self._row_scale_workspace: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return output shaped [Batch, SeqLen, OutDim]."""
        if x.ndim != 3 or x.shape[2] != self.hidden_dim:
            raise ValueError("x must have shape [Batch, SeqLen, HiddenDim]")
        if x.dtype != torch.float16 or not x.is_contiguous():
            raise TypeError("x must be contiguous float16")
        if x.device != self.packed_weight.device:
            raise ValueError("x and weight must be on the same device")
        rows = x.shape[0] * x.shape[1]
        workspace_shape = (rows, self.hidden_dim // 2)
        if (
            self._quantized_x_workspace is None
            or self._quantized_x_workspace.shape != workspace_shape
            or self._quantized_x_workspace.device != x.device
        ):
            self._quantized_x_workspace = torch.empty(
                workspace_shape, dtype=torch.uint8, device=x.device
            )
            self._row_scale_workspace = torch.empty(
                rows, dtype=torch.float32, device=x.device
            )
        assert self._row_scale_workspace is not None
        return fused_rmsnorm_linear_int4_prefill_cuda(
            x,
            self.packed_weight,
            self.weight_scale,
            self._quantized_x_workspace,
            self._row_scale_workspace,
            self.bias,
            self.eps,
        )


__all__ = ["FusedRMSNormLinearInt4Prefill"]
