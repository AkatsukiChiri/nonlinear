"""Experimental prefill-only fused RMSNorm and Linear layer."""

from __future__ import annotations

from torch import nn

from . import FusedRMSNormLinear


class FusedRMSNormLinearPrefill(FusedRMSNormLinear):
    """Backward-compatible name for the unified decode/prefill layer."""

    def __init__(self, rmsnorm: nn.Module, linear: nn.Linear) -> None:
        super().__init__(rmsnorm, linear)
        if self.hidden_dim % 16 != 0 or self.out_dim % 16 != 0:
            raise ValueError("HiddenDim and OutDim must be divisible by 16")
