"""Unified decode and prefill INT8 fused RMSNorm + Linear layer."""

from __future__ import annotations

import torch
from torch import nn

from .quantization import DEFAULT_GROUP_SIZE, quantize_weight_symmetric_int8
from layers.fused_rmsnorm_linear_int8_prefill.cuda import (
    fused_rmsnorm_linear_int8_prefill_cuda,
)
from layers.fused_rmsnorm_linear_weight_only_int8_dequant.cuda import (
    weight_only_int8_dequant_cuda,
)

DECODE_MAX_TOKENS = 8


def _get_rmsnorm_eps(rmsnorm: nn.Module, dtype: torch.dtype) -> float:
    eps = getattr(rmsnorm, "variance_epsilon", None)
    if eps is None:
        eps = getattr(rmsnorm, "eps", None)
    if eps is None:
        eps = torch.finfo(dtype).eps
    return float(eps)


class FusedRMSNormLinearInt8(nn.Module):
    """Dispatch one shared per-output INT8 weight to decode or prefill.

    For ``M = Batch * SeqLen <= 8``, the layer uses the weight-only decode
    kernel, which dequantizes weights online and keeps activations in FP16.
    For ``M > 8``, it uses online activation quantization and an INT8 Tensor
    Core GEMM.
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
            raise ValueError("group_size is retained for compatibility and must be 64")

        rmsnorm_weight = getattr(rmsnorm, "weight", None)
        if not isinstance(rmsnorm_weight, torch.Tensor):
            raise TypeError("rmsnorm must expose a tensor weight")
        if rmsnorm_weight.ndim != 1:
            raise ValueError("rmsnorm.weight must have shape [HiddenDim]")
        if linear.in_features != rmsnorm_weight.numel():
            raise ValueError("RMSNorm HiddenDim must equal Linear in_features")
        if linear.in_features % DEFAULT_GROUP_SIZE != 0:
            raise ValueError("HiddenDim must be divisible by 64")
        if linear.out_features % 8 != 0:
            raise ValueError("OutDim must be divisible by 8")
        if linear.weight.device != rmsnorm_weight.device:
            raise ValueError("rmsnorm and linear weights must be on the same device")
        if linear.weight.dtype != rmsnorm_weight.dtype:
            raise ValueError("rmsnorm and linear weights must have the same dtype")
        if linear.weight.dtype != torch.float16:
            raise TypeError("the INT8 CUDA kernels require FP16 source weights")

        self.hidden_dim = linear.in_features
        self.out_dim = linear.out_features
        self.group_size = self.hidden_dim
        self.eps = _get_rmsnorm_eps(rmsnorm, linear.weight.dtype)

        with torch.no_grad():
            fused_weight = linear.weight * rmsnorm_weight.unsqueeze(0)
            quantized_weight, weight_scale = quantize_weight_symmetric_int8(
                fused_weight, self.hidden_dim
            )
        self.register_buffer("quantized_weight", quantized_weight)
        self.register_buffer("weight_scale", weight_scale.reshape(-1))
        self.register_buffer(
            "bias",
            None if linear.bias is None else linear.bias.detach().clone().contiguous(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return output shaped [Batch, SeqLen, OutDim]."""
        if x.ndim != 3 or x.shape[2] != self.hidden_dim:
            raise ValueError("x must have shape [Batch, SeqLen, HiddenDim]")
        if x.shape[0] <= 0 or x.shape[1] <= 0:
            raise ValueError("Batch and SeqLen must be positive")
        if x.device != self.quantized_weight.device:
            raise ValueError("x and quantized_weight must be on the same device")
        if x.dtype != torch.float16 or not x.is_contiguous():
            raise TypeError("x must be contiguous float16")

        batch, seq_len, _ = x.shape
        token_count = batch * seq_len
        if token_count <= DECODE_MAX_TOKENS:
            decode_input = x.reshape(token_count, 1, self.hidden_dim)
            output = weight_only_int8_dequant_cuda(
                decode_input,
                self.quantized_weight,
                self.weight_scale.reshape(self.out_dim, 1),
                self.bias,
                self.eps,
            )
            return output.reshape(batch, seq_len, self.out_dim)

        return fused_rmsnorm_linear_int8_prefill_cuda(
            x,
            self.quantized_weight,
            self.weight_scale,
            self.bias,
            self.eps,
        )


__all__ = ["DECODE_MAX_TOKENS", "FusedRMSNormLinearInt8"]
