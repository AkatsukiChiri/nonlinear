"""Model integrations for Non-Linear."""

from .fused_llama import (
    ConversionReport,
    DecodeOnlyIdentity,
    convert_llama_to_fused_rmsnorm_linear,
)

__all__ = [
    "ConversionReport",
    "DecodeOnlyIdentity",
    "convert_llama_to_fused_rmsnorm_linear",
]
