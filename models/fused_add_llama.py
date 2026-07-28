"""Llama decoder integration that defers residual adds into fused norms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from layers.fused_rmsnorm_linear import (
    FusedAddRMSNormLinear,
    FusedRMSNormLinear,
)
from models.fused_llama import (
    _CONVERSION_MARKER,
    _get_llama_model,
    _validate_model_for_conversion,
)


_FUSED_ADD_MARKER = "_nonlinear_fused_add_rmsnorm_linear"


@dataclass(frozen=True)
class PendingResidual:
    """An update and residual whose element-wise add has been deferred."""

    update: torch.Tensor
    residual: torch.Tensor


@dataclass(frozen=True)
class FusedAddConversionReport:
    """Summary of a fused-add Llama decoder conversion."""

    decoder_layers: int
    fused_add_projections: int
    fused_rms_projections: int
    dtype: torch.dtype
    device: torch.device


class FusedAddLlamaAttention(nn.Module):
    """Llama attention whose input residual add is fused into q projection."""

    q_proj: FusedRMSNormLinear | FusedAddRMSNormLinear

    def __init__(
        self,
        source: nn.Module,
        input_norm: nn.Module,
        *,
        fuse_input_add: bool,
    ) -> None:
        super().__init__()
        self.config = source.config
        self.layer_idx = source.layer_idx
        self.head_dim = source.head_dim
        self.num_key_value_groups = source.num_key_value_groups
        self.scaling = source.scaling
        self.attention_dropout = source.attention_dropout
        self.is_causal = source.is_causal
        self.fuse_input_add = fuse_input_add

        if fuse_input_add:
            self.q_proj = FusedAddRMSNormLinear(input_norm, source.q_proj)
        else:
            self.q_proj = FusedRMSNormLinear(input_norm, source.q_proj)
        self.k_proj = FusedRMSNormLinear(input_norm, source.k_proj)
        self.v_proj = FusedRMSNormLinear(input_norm, source.v_proj)
        self.o_proj = source.o_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Run attention and return its output plus the materialized residual."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.fuse_input_add:
            if residual is None:
                raise TypeError("a deferred residual is required after layer 0")
            query_projection, residual_sum = self.q_proj(
                hidden_states,
                residual,
            )
        else:
            if residual is not None:
                raise TypeError("layer 0 must receive a materialized tensor")
            query_projection = self.q_proj(hidden_states)
            residual_sum = hidden_states

        query_states = query_projection.view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(residual_sum).view(hidden_shape).transpose(1, 2)
        value_states = (
            self.v_proj(residual_sum).view(hidden_shape).transpose(1, 2)
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )
        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        attention_interface = eager_attention_forward
        implementation = self.config._attn_implementation
        output_attentions = kwargs.get("output_attentions", False)
        if implementation != "eager" and not (
            implementation == "sdpa" and output_attentions
        ):
            attention_interface = ALL_ATTENTION_FUNCTIONS[implementation]

        attention_output, attention_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attention_output = attention_output.reshape(
            *input_shape,
            -1,
        ).contiguous()
        attention_output = self.o_proj(attention_output)
        return attention_output, attention_weights, residual_sum


class FusedAddLlamaMLP(nn.Module):
    """Llama MLP whose attention residual add is fused into gate projection."""

    def __init__(self, source: nn.Module, post_attention_norm: nn.Module) -> None:
        super().__init__()
        self.config = source.config
        self.hidden_size = source.hidden_size
        self.intermediate_size = source.intermediate_size
        self.gate_proj = FusedAddRMSNormLinear(
            post_attention_norm,
            source.gate_proj,
        )
        self.up_proj = FusedRMSNormLinear(
            post_attention_norm,
            source.up_proj,
        )
        self.down_proj = source.down_proj
        self.act_fn = source.act_fn

    def forward(
        self,
        attention_output: torch.Tensor,
        attention_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate, residual_sum = self.gate_proj(
            attention_output,
            attention_residual,
        )
        up = self.up_proj(residual_sum)
        output = self.down_proj(self.act_fn(gate) * up)
        return output, residual_sum


class FusedAddLlamaDecoderLayer(nn.Module):
    """Decoder layer that carries the MLP residual add into the next layer."""

    def __init__(
        self,
        source: LlamaDecoderLayer,
        *,
        layer_index: int,
        is_last_layer: bool,
    ) -> None:
        super().__init__()
        self.hidden_size = source.hidden_size
        self.layer_index = layer_index
        self.is_last_layer = is_last_layer
        self.self_attn = FusedAddLlamaAttention(
            source.self_attn,
            source.input_layernorm,
            fuse_input_add=layer_index > 0,
        )
        self.mlp = FusedAddLlamaMLP(
            source.mlp,
            source.post_attention_layernorm,
        )

    def forward(
        self,
        hidden_states: torch.Tensor | PendingResidual,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor | PendingResidual, ...]:
        del position_ids, use_cache
        if position_embeddings is None:
            raise ValueError("position_embeddings must be provided")

        if self.layer_index == 0:
            if not isinstance(hidden_states, torch.Tensor):
                raise TypeError("layer 0 must receive a materialized tensor")
            projection_input = hidden_states
            projection_residual = None
        else:
            if not isinstance(hidden_states, PendingResidual):
                raise TypeError(
                    "layers after layer 0 must receive PendingResidual"
                )
            projection_input = hidden_states.update
            projection_residual = hidden_states.residual

        attention_output, attention_weights, input_residual = self.self_attn(
            hidden_states=projection_input,
            residual=projection_residual,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
        )
        mlp_output, post_attention_sum = self.mlp(
            attention_output,
            input_residual,
        )

        if self.is_last_layer:
            output: torch.Tensor | PendingResidual = (
                post_attention_sum + mlp_output
            )
        else:
            output = PendingResidual(
                update=mlp_output,
                residual=post_attention_sum,
            )

        outputs: tuple[torch.Tensor | PendingResidual, ...] = (output,)
        if output_attentions:
            outputs += (attention_weights,)
        return outputs


def _decode_only_model_hook(
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    if module.training:
        raise RuntimeError("fused-add Llama is inference-only; call model.eval()")
    if torch.is_grad_enabled():
        raise RuntimeError(
            "fused-add Llama requires torch.inference_mode() or torch.no_grad()"
        )
    if kwargs.get(
        "output_hidden_states",
        module.config.output_hidden_states,
    ):
        raise ValueError(
            "fused-add Llama does not support output_hidden_states because "
            "intermediate residual adds are intentionally deferred"
        )

    input_ids = kwargs.get("input_ids")
    inputs_embeds = kwargs.get("inputs_embeds")
    if input_ids is None and inputs_embeds is None and args:
        input_ids = args[0]
    input_tensor = input_ids if input_ids is not None else inputs_embeds
    if input_tensor is not None and input_tensor.shape[1] != 1:
        raise ValueError(
            "fused-add Llama decode requires SeqLen == 1, got "
            f"{input_tensor.shape[1]}"
        )


def convert_llama_to_fused_add_rmsnorm_linear(
    model: LlamaForCausalLM | LlamaModel,
) -> FusedAddConversionReport:
    """Convert a loaded FP16 Llama to deferred-residual decode inference."""
    if getattr(model, _FUSED_ADD_MARKER, False):
        raise RuntimeError("model is already converted to fused-add Llama")
    if getattr(model, _CONVERSION_MARKER, False):
        raise RuntimeError("model is already converted to fused Llama")

    llama_model = _get_llama_model(model)
    device, dtype = _validate_model_for_conversion(model, llama_model)
    layer_count = len(llama_model.layers)
    for layer_index in range(layer_count):
        source_layer = llama_model.layers[layer_index]
        llama_model.layers[layer_index] = FusedAddLlamaDecoderLayer(
            source_layer,
            layer_index=layer_index,
            is_last_layer=layer_index == layer_count - 1,
        )
        del source_layer
    llama_model.register_forward_pre_hook(
        _decode_only_model_hook,
        with_kwargs=True,
    )
    setattr(model, _FUSED_ADD_MARKER, True)
    model.eval()

    fused_add_projections = 2 * layer_count - 1
    return FusedAddConversionReport(
        decoder_layers=layer_count,
        fused_add_projections=fused_add_projections,
        fused_rms_projections=5 * layer_count - fused_add_projections,
        dtype=dtype,
        device=device,
    )


__all__ = [
    "FusedAddConversionReport",
    "FusedAddLlamaDecoderLayer",
    "PendingResidual",
    "convert_llama_to_fused_add_rmsnorm_linear",
]
