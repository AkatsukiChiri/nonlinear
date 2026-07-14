"""PyTorch reference for fused square-and-linear projection."""

from __future__ import annotations

import torch


def fused_square_linear_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Compute ``stack((x, x**2), dim=1) @ weight.T``.

    Args:
        x: Input tensor with shape [Batch, 1, HiddenDim].
        weight: Projection weight with shape [OutDim, HiddenDim].

    Returns:
        Projected tensor with shape [Batch, 2, OutDim]. Accumulation is FP32
        and the result is converted back to the input dtype.
    """
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError(f"x must have shape [Batch, 1, HiddenDim], got {x.shape}")
    if weight.ndim != 2 or weight.shape[1] != x.shape[2]:
        raise ValueError(
            "weight must have shape [OutDim, HiddenDim] with matching "
            f"HiddenDim, got x={x.shape}, weight={weight.shape}"
        )
    x_values = x[:, 0, :]
    # Match the fused kernel: the square is an FP16 input transform, while
    # Tensor Core products accumulate into FP32.
    expanded = torch.stack((x_values, x_values.square()), dim=1).float()
    return torch.matmul(expanded, weight.float().t()).to(dtype=x.dtype)
