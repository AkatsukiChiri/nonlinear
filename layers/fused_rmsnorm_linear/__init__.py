"""Inference-only fused RMSNorm and Linear layer."""

from __future__ import annotations

import torch
from torch import nn

from .cuda import (
    fused_add_rms_linear_bias_fp16,
    fused_add_rms_linear_fp16,
    fused_rmsnorm_linear_bias_fp16,
    fused_rmsnorm_linear_fp16,
)
from .prefill_cuda import (
    fused_rmsnorm_linear_prefill_bias_fp16,
    fused_rmsnorm_linear_prefill_fp16,
)


DEFAULT_EPS = 1e-6


def _get_rmsnorm_eps(rmsnorm: nn.Module, dtype: torch.dtype) -> float:
    """Read epsilon from PyTorch- or Transformers-style RMSNorm modules."""
    eps = getattr(rmsnorm, "variance_epsilon", None)
    if eps is None:
        eps = getattr(rmsnorm, "eps", None)
    if eps is None:
        # torch.nn.RMSNorm uses the dtype epsilon when eps=None.
        eps = torch.finfo(dtype).eps
    return float(eps)


class FusedRMSNormLinear(nn.Module):
    """Fuse a frozen RMSNorm and Linear projection for inference.

    At construction, the RMSNorm scale is absorbed into the Linear weight and
    stored once::

        fused_weight = linear.weight * rmsnorm.weight

    The forward pass dispatches on the flattened token count ``M=Batch*SeqLen``.
    ``M < 8`` uses the specialized small-M/decode CUDA kernel; ``M >= 8`` uses
    the CUTLASS Tensor Core prefill kernel. If the source Linear has a bias, it
    is added after RMS scaling, matching ``linear(rmsnorm(x))``.

    The source modules are not retained. Their weights are snapshotted into a
    registered buffer, so this layer is intended for fixed-weight inference.
    """

    fused_weight: torch.Tensor
    bias: torch.Tensor | None

    def __init__(self, rmsnorm: nn.Module, linear: nn.Linear) -> None:
        super().__init__()

        rmsnorm_weight = getattr(rmsnorm, "weight", None)
        if not isinstance(rmsnorm_weight, torch.Tensor):
            raise TypeError("rmsnorm must expose a tensor weight")
        if rmsnorm_weight.ndim != 1:
            raise ValueError("rmsnorm.weight must have shape [HiddenDim]")
        if linear.weight.ndim != 2:
            raise ValueError("linear.weight must have shape [OutDim, HiddenDim]")
        if linear.in_features != rmsnorm_weight.numel():
            raise ValueError(
                "RMSNorm HiddenDim must equal Linear in_features, got "
                f"{rmsnorm_weight.numel()} and {linear.in_features}"
            )
        if linear.weight.device != rmsnorm_weight.device:
            raise ValueError("rmsnorm and linear weights must be on the same device")
        if linear.weight.dtype != rmsnorm_weight.dtype:
            raise ValueError("rmsnorm and linear weights must have the same dtype")
        if linear.weight.dtype != torch.float16:
            raise TypeError("the fused CUDA kernel currently requires float16 weights")

        self.hidden_dim = linear.in_features
        self.out_dim = linear.out_features
        self.eps = _get_rmsnorm_eps(rmsnorm, linear.weight.dtype)

        with torch.no_grad():
            projected_weight = linear.weight * rmsnorm_weight.unsqueeze(0)
        self.register_buffer("fused_weight", projected_weight.contiguous())
        self.register_buffer(
            "bias",
            None if linear.bias is None else linear.bias.detach().clone().contiguous(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fused RMSNorm and Linear.

        Args:
            x: Contiguous FP16 CUDA tensor shaped
                [Batch, SeqLen, HiddenDim].

        Returns:
            FP16 tensor shaped [Batch, SeqLen, OutDim].
        """
        if x.ndim != 3 or x.shape[2] != self.hidden_dim:
            raise ValueError(
                "x must have shape [Batch, SeqLen, HiddenDim], got "
                f"{tuple(x.shape)} with expected HiddenDim={self.hidden_dim}"
            )
        if x.shape[0] <= 0 or x.shape[1] <= 0:
            raise ValueError("Batch and SeqLen must be positive")
        if not x.is_contiguous():
            raise ValueError("x must be contiguous")
        if x.device != self.fused_weight.device:
            raise ValueError("x and fused_weight must be on the same device")
        if x.dtype != self.fused_weight.dtype:
            raise TypeError("x and fused_weight must have the same dtype")

        batch, seq_len = x.shape[:2]
        token_count = batch * seq_len
        prefill_compatible = (
            self.hidden_dim % 16 == 0 and self.out_dim % 16 == 0
        )
        if token_count >= 8 and prefill_compatible:
            if self.bias is not None:
                return fused_rmsnorm_linear_prefill_bias_fp16(
                    x,
                    self.fused_weight,
                    self.bias,
                    self.eps,
                )
            return fused_rmsnorm_linear_prefill_fp16(
                x,
                self.fused_weight,
                self.eps,
            )

        if token_count > 8:
            raise ValueError(
                "M >= 8 requires HiddenDim and OutDim divisible by 16 for "
                "the prefill Tensor Core kernel"
            )

        decode_input = x.view(token_count, 1, self.hidden_dim)
        if self.bias is not None:
            output = fused_rmsnorm_linear_bias_fp16(
                decode_input,
                self.fused_weight,
                self.bias,
                self.eps,
            )
        else:
            output = fused_rmsnorm_linear_fp16(
                decode_input, self.fused_weight, self.eps
            )
        return output.view(batch, seq_len, self.out_dim)


class FusedAddRMSNormLinear(FusedRMSNormLinear):
    """Fuse residual add, frozen RMSNorm, and Linear for decode inference.

    The forward pass returns both the Linear projection and the FP16 residual
    sum so the latter can be reused by the next decoder layer without launching
    a separate add kernel.
    """

    def forward(
        self,
        x: torch.Tensor,
        x_res: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply fused residual add, RMSNorm, and Linear.

        Args:
            x: Contiguous FP16 tensor shaped [Batch, 1, HiddenDim].
            x_res: Contiguous FP16 residual with the same shape as ``x``.

        Returns:
            ``(linear_output, residual_sum)`` where the tensors have shapes
            [Batch, 1, OutDim] and [Batch, 1, HiddenDim], respectively.
        """
        if x.shape != x_res.shape:
            raise ValueError(
                "x and x_res must have the same shape, got "
                f"{tuple(x.shape)} and {tuple(x_res.shape)}"
            )
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[2] != self.hidden_dim:
            raise ValueError(
                "x must have shape [Batch, 1, HiddenDim], got "
                f"{tuple(x.shape)} with expected HiddenDim={self.hidden_dim}"
            )
        if x.device != x_res.device:
            raise ValueError("x and x_res must be on the same device")
        if x.dtype != x_res.dtype:
            raise TypeError("x and x_res must have the same dtype")
        if x.device != self.fused_weight.device:
            raise ValueError("inputs and fused_weight must be on the same device")
        if x.dtype != self.fused_weight.dtype:
            raise TypeError("inputs and fused_weight must have the same dtype")

        if self.bias is not None:
            return fused_add_rms_linear_bias_fp16(
                x,
                x_res,
                self.fused_weight,
                self.bias,
                self.eps,
            )
        return fused_add_rms_linear_fp16(
            x,
            x_res,
            self.fused_weight,
            self.eps,
        )
