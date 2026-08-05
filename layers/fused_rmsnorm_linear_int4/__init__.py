"""Unified decode and prefill INT4 fused RMSNorm + Linear layer."""

from __future__ import annotations

import torch
from torch import nn

from layers.fused_rmsnorm_linear_int4_prefill.cuda import (
    fused_rmsnorm_linear_int4_prefill_cuda,
)
from layers.fused_rmsnorm_linear_weight_only_int4_dequant.cuda import (
    weight_only_int4_dequant_cuda,
)
from layers.fused_rmsnorm_linear_weight_only_int4_dequant.quantization import (
    quantize_weight_symmetric_int4,
)

PREFILL_MIN_TOKENS = 6


class FusedRMSNormLinearInt4(nn.Module):
    """Dispatch shared packed INT4 weights to decode or prefill kernels.

    ``M = Batch * SeqLen < 6`` uses FP16 activation weight-only decode.
    ``M >= 6`` uses online INT4 activation quantization and Tensor Core GEMM.
    """

    def __init__(self, rmsnorm: nn.Module, linear: nn.Linear) -> None:
        super().__init__()
        rms_weight = getattr(rmsnorm, "weight", None)
        if not isinstance(rms_weight, torch.Tensor) or rms_weight.ndim != 1:
            raise TypeError("rmsnorm must expose one-dimensional tensor weight")
        if linear.in_features != rms_weight.numel():
            raise ValueError("RMSNorm HiddenDim must equal Linear in_features")
        if linear.in_features % 256:
            raise ValueError("HiddenDim must be divisible by 256")
        if linear.out_features % 8:
            raise ValueError("OutDim must be divisible by 8")
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
        self._quantized_x_workspace: torch.Tensor | None = None
        self._row_scale_workspace: torch.Tensor | None = None
        self._prefill_weight_scale = self.weight_scale.reshape(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return output shaped [Batch, SeqLen, OutDim]."""
        if x.ndim != 3 or x.shape[2] != self.hidden_dim:
            raise ValueError("x must have shape [Batch, SeqLen, HiddenDim]")
        if x.shape[0] <= 0 or x.shape[1] <= 0:
            raise ValueError("Batch and SeqLen must be positive")
        if x.dtype != torch.float16 or not x.is_contiguous():
            raise TypeError("x must be contiguous float16")
        if x.device != self.packed_weight.device:
            raise ValueError("x and weights must be on the same device")

        batch, seq_len, _ = x.shape
        rows = batch * seq_len
        if rows < PREFILL_MIN_TOKENS:
            decode_input = (
                x if seq_len == 1 else x.reshape(rows, 1, self.hidden_dim)
            )
            output = weight_only_int4_dequant_cuda(
                decode_input,
                self.packed_weight,
                self.weight_scale,
                self.bias,
                self.eps,
            )
            if seq_len == 1:
                return output
            return output.reshape(batch, seq_len, self.out_dim)

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
        if self._prefill_weight_scale.device != x.device:
            self._prefill_weight_scale = self.weight_scale.reshape(-1)
        return fused_rmsnorm_linear_int4_prefill_cuda(
            x,
            self.packed_weight,
            self._prefill_weight_scale,
            self._quantized_x_workspace,
            self._row_scale_workspace,
            self.bias,
            self.eps,
        )


__all__ = ["FusedRMSNormLinearInt4", "PREFILL_MIN_TOKENS"]
