"""vLLM Llama integration for the bs=1 sentinel Tensor Core decode path."""

from __future__ import annotations

from torch import nn

from layers.fused_rmsnorm_linear_int8_prefill import (
    FusedRMSNormLinearInt8Prefill,
)
from layers.fused_rmsnorm_linear_tensorcore_decode import (
    FusedRMSNormLinearTensorCoreDecode,
)
from models.vllm.fused_llama_int8 import FusedInt8LlamaDecoderLayer

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.llama import LlamaForCausalLM


class TensorCoreFusedInt8Projection(nn.Module):
    """Use the existing INT8 prefill path and sentinel Tensor Core decode."""

    def __init__(self, rmsnorm: nn.Module, linear: nn.Module) -> None:
        super().__init__()
        self.prefill = FusedRMSNormLinearInt8Prefill.from_quantized_linear(
            rmsnorm, linear, chunk_size=512
        )
        self.decode = FusedRMSNormLinearTensorCoreDecode(
            self.prefill.quantized_weight,
            self.prefill.weight_scale,
            self.prefill.bias,
            self.prefill.eps,
            self.prefill.compute_dtype,
        )

    def forward(self, hidden_states, residual=None):  # type: ignore[no-untyped-def]
        # This experimental kernel deliberately targets exactly one active
        # decode sequence. All prefill and larger batches retain the old path.
        if hidden_states.shape[0] == 1:
            return self.decode(hidden_states, residual)
        return self.prefill(hidden_states, residual)


class TensorCoreFusedInt8LlamaDecoderLayer(FusedInt8LlamaDecoderLayer):
    """Replace the two fused Llama projections with Tensor Core decode."""

    def _initialize_fused_projections(self) -> None:
        if self.fused_qkv is not None:
            return
        self.fused_qkv = TensorCoreFusedInt8Projection(
            self.input_layernorm, self.self_attn.qkv_proj
        )
        self.fused_gate_up = TensorCoreFusedInt8Projection(
            self.post_attention_layernorm, self.mlp.gate_up_proj
        )
        self.self_attn.qkv_proj = nn.Identity()
        self.mlp.gate_up_proj = nn.Identity()
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()


class TensorCoreFusedInt8LlamaForCausalLM(LlamaForCausalLM):
    """Drop-in Llama model retaining existing prefill and new bs=1 decode."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:  # type: ignore[no-untyped-def]
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError("Tensor Core fused Llama supports TP=1")
        if vllm_config.lora_config is not None:
            raise NotImplementedError("Tensor Core fused Llama does not support LoRA")
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=TensorCoreFusedInt8LlamaDecoderLayer,
        )


def register_tensorcore_fused_int8_llama() -> None:
    ModelRegistry.register_model(
        "LlamaForCausalLM",
        "models.vllm.fused_llama_int8_tensorcore:"
        "TensorCoreFusedInt8LlamaForCausalLM",
    )


__all__ = [
    "TensorCoreFusedInt8LlamaDecoderLayer",
    "TensorCoreFusedInt8LlamaForCausalLM",
    "TensorCoreFusedInt8Projection",
    "register_tensorcore_fused_int8_llama",
]
