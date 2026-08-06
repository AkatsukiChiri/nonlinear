"""Explicit decode and prefill INT8 fused RMSNorm + Linear layers."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from layers.fused_rmsnorm_linear_int8_prefill import (
        FusedRMSNormLinearInt8Prefill,
    )
    from layers.fused_rmsnorm_linear_weight_only_int8_dequant import (
        FusedRMSNormLinearInt8Decode,
    )

__all__ = ["FusedRMSNormLinearInt8Decode", "FusedRMSNormLinearInt8Prefill"]


def __getattr__(name: str) -> Any:
    if name == "FusedRMSNormLinearInt8Decode":
        from layers.fused_rmsnorm_linear_weight_only_int8_dequant import (
            FusedRMSNormLinearInt8Decode,
        )

        return FusedRMSNormLinearInt8Decode
    if name == "FusedRMSNormLinearInt8Prefill":
        from layers.fused_rmsnorm_linear_int8_prefill import (
            FusedRMSNormLinearInt8Prefill,
        )

        return FusedRMSNormLinearInt8Prefill
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
