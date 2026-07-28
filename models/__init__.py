"""Model integrations for Non-Linear."""

from .fused_llama import (
    ConversionReport,
    DecodeOnlyIdentity,
    convert_llama_to_fused_rmsnorm_linear,
)
from .fused_add_llama import (
    FusedAddConversionReport,
    FusedAddLlamaDecoderLayer,
    PendingResidual,
    convert_llama_to_fused_add_rmsnorm_linear,
)

__all__ = [
    "ConversionReport",
    "DecodeOnlyIdentity",
    "convert_llama_to_fused_rmsnorm_linear",
    "FusedAddConversionReport",
    "FusedAddLlamaDecoderLayer",
    "PendingResidual",
    "convert_llama_to_fused_add_rmsnorm_linear",
]
