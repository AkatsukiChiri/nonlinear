"""vLLM model integrations for Non-Linear."""

from .fused_llama_int8 import (
    FusedInt8LlamaDecoderLayer,
    FusedInt8LlamaForCausalLM,
    register_fused_int8_llama,
)
from .fused_llama_int8_tensorcore import (
    TensorCoreFusedInt8LlamaDecoderLayer,
    TensorCoreFusedInt8LlamaForCausalLM,
    register_tensorcore_fused_int8_llama,
)

__all__ = [
    "FusedInt8LlamaDecoderLayer",
    "FusedInt8LlamaForCausalLM",
    "register_fused_int8_llama",
    "TensorCoreFusedInt8LlamaDecoderLayer",
    "TensorCoreFusedInt8LlamaForCausalLM",
    "register_tensorcore_fused_int8_llama",
]
