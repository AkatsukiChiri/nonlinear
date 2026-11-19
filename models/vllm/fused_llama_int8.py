"""vLLM Llama model using fused RMSNorm + merged INT8 projections."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from layers.fused_rmsnorm_linear_int8_prefill import (
    FusedRMSNormLinearInt8Prefill,
)
from layers.fused_rmsnorm_linear_weight_only_int8_dequant import (
    FusedRMSNormLinearInt8Decode,
)

try:
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.llama import (
        LlamaDecoderLayer,
        LlamaForCausalLM,
    )
except ImportError as error:  # pragma: no cover - exercised in vLLM env only.
    raise ImportError("the fused vLLM Llama integration requires vLLM") from error


ExecutionMode = Literal["prefill", "decode"]


def _shared_decode_layer(
    prefill: FusedRMSNormLinearInt8Prefill,
) -> FusedRMSNormLinearInt8Decode:
    """Create a decode executor sharing prefill's folded INT8 storage."""
    decode = FusedRMSNormLinearInt8Decode.__new__(
        FusedRMSNormLinearInt8Decode
    )
    nn.Module.__init__(decode)
    decode.eps = prefill.eps
    decode.hidden_dim = prefill.hidden_dim
    decode.out_dim = prefill.out_dim
    decode.compute_dtype = prefill.compute_dtype
    decode.logical_widths = getattr(prefill, "logical_widths", ())
    decode.reuse_output_buffers = prefill.reuse_output_buffers
    decode._fused_add_buffers = {}
    decode._output_buffers = {}
    decode.register_buffer("quantized_weight", prefill.quantized_weight)
    decode.register_buffer("weight_scale", prefill.weight_scale.reshape(-1, 1))
    decode.register_buffer("bias", prefill.bias)
    return decode


class FusedInt8Projection(nn.Module):
    """Dispatch explicit prefill/decode executors using vLLM metadata."""

    def __init__(
        self,
        rmsnorm: nn.Module,
        linear: nn.Module,
        chunk_size: int = 512,
    ) -> None:
        super().__init__()
        self.prefill = FusedRMSNormLinearInt8Prefill.from_quantized_linear(
            rmsnorm, linear, chunk_size=chunk_size
        )
        self.decode = _shared_decode_layer(self.prefill)
        self.force_mode: ExecutionMode | None = None

    def _execution_mode(self, num_tokens: int) -> ExecutionMode:
        if self.force_mode is not None:
            return self.force_mode
        context = get_forward_context()
        metadata = context.attn_metadata
        max_query_len = int(getattr(metadata, "max_query_len", 0) or 0)
        if max_query_len > 1:
            return "prefill"
        prefill_metadata = getattr(metadata, "prefill_metadata", None)
        num_prefill_tokens = int(
            getattr(metadata, "num_prefill_tokens", 0) or 0
        )
        if prefill_metadata is not None or num_prefill_tokens > 0:
            return "prefill"
        # Decode GEMV is specialized for one token per active sequence.
        return "decode" if num_tokens <= 8 else "prefill"

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        mode = self._execution_mode(hidden_states.shape[0])
        executor = self.prefill if mode == "prefill" else self.decode
        return executor(hidden_states, residual)


class FusedInt8LlamaDecoderLayer(LlamaDecoderLayer):
    """Llama decoder replacing Norm+QKV and Norm+Gate-Up after loading."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.fused_qkv: FusedInt8Projection | None = None
        self.fused_gate_up: FusedInt8Projection | None = None

    @torch.compiler.disable
    def _initialize_fused_projections(self) -> None:
        if self.fused_qkv is not None:
            return
        self.fused_qkv = FusedInt8Projection(
            self.input_layernorm, self.self_attn.qkv_proj
        )
        self.fused_gate_up = FusedInt8Projection(
            self.post_attention_layernorm, self.mlp.gate_up_proj
        )
        # The source parameters are no longer needed after gamma folding.
        self.self_attn.qkv_proj = nn.Identity()
        self.mlp.gate_up_proj = nn.Identity()
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._initialize_fused_projections()
        assert self.fused_qkv is not None
        assert self.fused_gate_up is not None

        if residual is None:
            residual = hidden_states
            qkv = self.fused_qkv(hidden_states)
            assert isinstance(qkv, torch.Tensor)
        else:
            qkv_result = self.fused_qkv(hidden_states, residual)
            assert isinstance(qkv_result, tuple)
            qkv, residual = qkv_result

        q, k, v = qkv.split(
            [self.self_attn.q_size, self.self_attn.kv_size,
             self.self_attn.kv_size],
            dim=-1,
        )
        q, k = self.self_attn.rotary_emb(positions, q, k)
        attention_output = self.self_attn.attn(q, k, v)
        attention_output, _ = self.self_attn.o_proj(attention_output)

        gate_up_result = self.fused_gate_up(attention_output, residual)
        assert isinstance(gate_up_result, tuple)
        gate_up, residual = gate_up_result
        activated = self.mlp.act_fn(gate_up)
        hidden_states, _ = self.mlp.down_proj(activated)
        return hidden_states, residual


class FusedInt8LlamaForCausalLM(LlamaForCausalLM):
    """Drop-in vLLM LlamaForCausalLM with fused decoder layers."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:  # type: ignore[no-untyped-def]
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError(
                "fused INT8 Llama currently supports tensor_parallel_size=1"
            )
        if vllm_config.lora_config is not None:
            raise NotImplementedError("fused INT8 Llama does not support LoRA")
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=FusedInt8LlamaDecoderLayer,
        )
        self._nonlinear_fused_int8_model = True

    def fused_conversion_status(self) -> tuple[int, int]:
        """Return converted and total local decoder-layer counts."""
        layers = list(self.model.layers)
        converted = sum(
            isinstance(layer, FusedInt8LlamaDecoderLayer)
            and layer.fused_qkv is not None
            and layer.fused_gate_up is not None
            for layer in layers
        )
        return converted, len(layers)


def register_fused_int8_llama() -> None:
    """Register the fused model for checkpoints declaring LlamaForCausalLM."""
    ModelRegistry.register_model(
        "LlamaForCausalLM",
        "models.vllm.fused_llama_int8:FusedInt8LlamaForCausalLM",
    )


__all__ = [
    "FusedInt8LlamaDecoderLayer",
    "FusedInt8LlamaForCausalLM",
    "FusedInt8Projection",
    "register_fused_int8_llama",
]
