"""Weight-only INT8 layer using register-local FP16 dequantization."""

from __future__ import annotations

import torch
from torch import nn

from layers.fused_rmsnorm_linear_int8.quantization import (
    DEFAULT_GROUP_SIZE,
    quantize_weight_symmetric_int8,
)

from .cuda import (
    fused_add_weight_only_int8_dequant_out_cuda,
    fused_add_weight_only_int8_dequant_cuda,
    weight_only_int8_dequant_cuda,
)


class FusedRMSNormLinearInt8Decode(nn.Module):
    """Keep activations in FP16/BF16 and dequantize INT8 weights online.

    By default, fused-add inference reuses persistent output buffers to avoid
    decode-path allocator latency and to provide stable CUDA Graph addresses.
    A later fused-add call with the same batch size overwrites tensors returned
    by the previous call. Set ``reuse_output_buffers=False`` when outputs must
    remain immutable across calls.
    """

    def __init__(
        self,
        rmsnorm: nn.Module,
        linear: nn.Linear,
        reuse_output_buffers: bool = True,
    ) -> None:
        super().__init__()
        rms_weight = getattr(rmsnorm, "weight", None)
        if not isinstance(rms_weight, torch.Tensor) or rms_weight.ndim != 1:
            raise TypeError("rmsnorm must expose a one-dimensional tensor weight")
        if linear.in_features != rms_weight.numel():
            raise ValueError("RMSNorm HiddenDim must equal Linear in_features")
        if linear.in_features % DEFAULT_GROUP_SIZE != 0:
            raise ValueError("HiddenDim must be divisible by 64")
        if linear.weight.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("source weights must be float16 or bfloat16")
        if rms_weight.dtype != linear.weight.dtype:
            raise TypeError("RMSNorm and Linear weights must have the same dtype")
        if linear.weight.device != rms_weight.device:
            raise ValueError("source weights must be on the same device")
        eps = getattr(rmsnorm, "variance_epsilon", getattr(rmsnorm, "eps", None))
        self.eps = float(torch.finfo(linear.weight.dtype).eps if eps is None else eps)
        self.hidden_dim = linear.in_features
        self.out_dim = linear.out_features
        self.compute_dtype = linear.weight.dtype
        self.reuse_output_buffers = reuse_output_buffers
        self._fused_add_buffers: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        with torch.no_grad():
            fused_weight = linear.weight * rms_weight.unsqueeze(0)
            quantized_weight, weight_scale = quantize_weight_symmetric_int8(
                fused_weight, linear.in_features
            )
        self.register_buffer("quantized_weight", quantized_weight)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer(
            "bias",
            None if linear.bias is None else linear.bias.detach().clone().contiguous(),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_res: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return output, plus the updated residual when ``x_res`` is given."""
        if x.ndim != 3 or x.shape[1:] != (1, self.hidden_dim):
            raise ValueError("x must have shape [Batch, 1, HiddenDim]")
        if not 1 <= x.shape[0] <= 8:
            raise ValueError("Batch must be in [1, 8]")
        if x.dtype != self.compute_dtype or not x.is_contiguous():
            raise TypeError(
                f"x must be contiguous {self.compute_dtype}"
            )
        if x.device != self.quantized_weight.device:
            raise ValueError("x and weights must be on the same device")
        if x_res is not None:
            if x_res.shape != x.shape:
                raise ValueError("x_res must have the same shape as x")
            if x_res.dtype != x.dtype or x_res.device != x.device:
                raise ValueError("x_res must have the same dtype and device as x")
            if not x_res.is_contiguous():
                raise TypeError("x_res must be contiguous")
            if not self.reuse_output_buffers:
                return fused_add_weight_only_int8_dequant_cuda(
                    x,
                    x_res,
                    self.quantized_weight,
                    self.weight_scale,
                    self.bias,
                    self.eps,
                )
            buffers = self._fused_add_buffers.get(x.shape[0])
            if buffers is None:
                buffers = (
                    torch.empty(
                        x.shape[0], 1, self.out_dim, device=x.device, dtype=x.dtype
                    ),
                    torch.empty_like(x),
                )
                self._fused_add_buffers[x.shape[0]] = buffers
            return fused_add_weight_only_int8_dequant_out_cuda(
                x,
                x_res,
                self.quantized_weight,
                self.weight_scale,
                self.bias,
                self.eps,
                buffers[0],
                buffers[1],
            )
        return weight_only_int8_dequant_cuda(
            x, self.quantized_weight, self.weight_scale, self.bias, self.eps
        )


__all__ = ["FusedRMSNormLinearInt8Decode"]
