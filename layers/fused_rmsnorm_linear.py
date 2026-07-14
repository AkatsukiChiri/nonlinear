"""Inference-only fused RMSNorm and bias-free Linear layer."""

from __future__ import annotations

import torch
from torch import nn

from layers.fused_square_linear import fused_square_linear_fp16


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
    """Fuse a frozen RMSNorm and bias-free Linear projection for inference.

    At construction, the RMSNorm scale is absorbed into the Linear weight and
    an all-ones row is appended::

        augmented_weight = cat([linear.weight * rmsnorm.weight, ones], dim=0)

    For an input ``x`` shaped [Batch, 1, HiddenDim], the CUDA kernel computes
    both ``x @ augmented_weight.T`` and
    ``x.square() @ augmented_weight.T``. The former supplies the projected
    numerator, while the final element of the latter is ``sum(x.square())``.

    The source modules are not retained. Their weights are snapshotted into a
    registered buffer, so this layer is intended for fixed-weight inference.
    """

    fused_weight: torch.Tensor

    def __init__(self, rmsnorm: nn.Module, linear: nn.Linear) -> None:
        super().__init__()

        rmsnorm_weight = getattr(rmsnorm, "weight", None)
        if not isinstance(rmsnorm_weight, torch.Tensor):
            raise TypeError("rmsnorm must expose a tensor weight")
        if linear.bias is not None:
            raise ValueError("FusedRMSNormLinear currently requires bias=False")
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
            sum_weight = torch.ones(
                1,
                self.hidden_dim,
                dtype=projected_weight.dtype,
                device=projected_weight.device,
            )
            fused_weight = torch.cat((projected_weight, sum_weight), dim=0)
        self.register_buffer("fused_weight", fused_weight.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fused RMSNorm and Linear.

        Args:
            x: Contiguous FP16 CUDA tensor shaped [Batch, 1, HiddenDim].

        Returns:
            FP16 tensor shaped [Batch, 1, OutDim].
        """
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[2] != self.hidden_dim:
            raise ValueError(
                "x must have shape [Batch, 1, HiddenDim], got "
                f"{tuple(x.shape)} with expected HiddenDim={self.hidden_dim}"
            )
        if x.device != self.fused_weight.device:
            raise ValueError("x and fused_weight must be on the same device")
        if x.dtype != self.fused_weight.dtype:
            raise TypeError("x and fused_weight must have the same dtype")

        # [Batch, 2, OutDim + 1]
        fused_output = fused_square_linear_fp16(x, self.fused_weight)
        numerator = fused_output[:, 0, : self.out_dim]
        square_sum = fused_output[:, 1, self.out_dim].float()

        inverse_rms = torch.rsqrt(square_sum / self.hidden_dim + self.eps)
        output = numerator.float() * inverse_rms.unsqueeze(1)
        return output.to(dtype=x.dtype).unsqueeze(1)
