"""Decode-only Llama conversion using fused FP16 RMSNorm + Linear layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
)

from layers.fused_rmsnorm_linear import FusedRMSNormLinear


_CONVERSION_MARKER = "_nonlinear_fused_rmsnorm_linear"


@dataclass(frozen=True)
class ConversionReport:
    """Summary of an in-place Llama decoder conversion."""

    decoder_layers: int
    fused_projections: int
    dtype: torch.dtype
    device: torch.device


class DecodeOnlyIdentity(nn.Module):
    """Identity module enforcing the fused Llama decode-only contract."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Validate and forward [Batch, 1, HiddenDim] FP16 CUDA states."""
        if self.training:
            raise RuntimeError("fused Llama is inference-only; call model.eval()")
        if torch.is_grad_enabled():
            raise RuntimeError(
                "fused Llama requires torch.inference_mode() or torch.no_grad()"
            )
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [Batch, SeqLen, HiddenDim]"
            )
        if hidden_states.shape[1] != 1:
            raise ValueError(
                "fused Llama decode requires SeqLen == 1, got "
                f"{hidden_states.shape[1]}"
            )
        if hidden_states.shape[2] != self.hidden_dim:
            raise ValueError(
                f"expected HiddenDim={self.hidden_dim}, got "
                f"{hidden_states.shape[2]}"
            )
        if not hidden_states.is_cuda:
            raise ValueError("fused Llama requires CUDA hidden states")
        if hidden_states.dtype != torch.float16:
            raise TypeError("fused Llama requires float16 hidden states")
        if not hidden_states.is_contiguous():
            raise ValueError("fused Llama requires contiguous hidden states")
        return hidden_states


def _get_llama_model(
    model: LlamaForCausalLM | LlamaModel,
) -> LlamaModel:
    if isinstance(model, LlamaForCausalLM):
        return model.model
    if isinstance(model, LlamaModel):
        return model
    raise TypeError(
        "model must be transformers.LlamaForCausalLM or LlamaModel, got "
        f"{type(model).__name__}"
    )


def _validate_model_for_conversion(
    model: LlamaForCausalLM | LlamaModel,
    llama_model: LlamaModel,
) -> tuple[torch.device, torch.dtype]:
    if getattr(model, _CONVERSION_MARKER, False):
        raise RuntimeError("model is already converted")
    if model.training:
        raise RuntimeError("model must be in eval mode before conversion")
    if not llama_model.layers:
        raise ValueError("Llama model must contain at least one decoder layer")

    first_weight = llama_model.layers[0].self_attn.q_proj.weight
    device = first_weight.device
    dtype = first_weight.dtype
    if device.type != "cuda":
        raise ValueError("fused Llama conversion requires a CUDA model")
    if dtype != torch.float16:
        raise TypeError(
            "fused Llama conversion requires float16 weights, got "
            f"{dtype}"
        )
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 8:
        raise RuntimeError(
            "fused Llama conversion currently requires an Ampere GPU"
        )

    for layer_index, layer in enumerate(llama_model.layers):
        if not isinstance(layer, LlamaDecoderLayer):
            raise TypeError(
                f"layers[{layer_index}] must be LlamaDecoderLayer, got "
                f"{type(layer).__name__}"
            )
        target_modules = (
            layer.input_layernorm,
            layer.post_attention_layernorm,
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
        )
        for module in target_modules:
            tensors = tuple(module.parameters())
            if not tensors:
                raise ValueError(
                    f"layers[{layer_index}] contains a target without parameters"
                )
            for tensor in tensors:
                if tensor.device != device:
                    raise ValueError("all fused targets must be on one CUDA device")
                if tensor.dtype != torch.float16:
                    raise TypeError("all fused targets must use float16 weights")
    return device, dtype


def convert_llama_to_fused_rmsnorm_linear(
    model: LlamaForCausalLM | LlamaModel,
) -> ConversionReport:
    """Replace the five decoder-internal RMSNorm + Linear groups in-place.

    The converted model supports frozen FP16 single-token decode only. Load
    checkpoints and build any prefill KV cache before calling this function.
    Final ``model.norm`` and ``lm_head`` modules remain unchanged.
    """
    llama_model = _get_llama_model(model)
    device, dtype = _validate_model_for_conversion(model, llama_model)

    fused_projections = 0
    for layer in llama_model.layers:
        input_norm = layer.input_layernorm
        layer.self_attn.q_proj = FusedRMSNormLinear(
            input_norm,
            layer.self_attn.q_proj,
        )
        layer.self_attn.k_proj = FusedRMSNormLinear(
            input_norm,
            layer.self_attn.k_proj,
        )
        layer.self_attn.v_proj = FusedRMSNormLinear(
            input_norm,
            layer.self_attn.v_proj,
        )
        layer.input_layernorm = DecodeOnlyIdentity(layer.hidden_size)
        fused_projections += 3

        post_attention_norm = layer.post_attention_layernorm
        layer.mlp.gate_proj = FusedRMSNormLinear(
            post_attention_norm,
            layer.mlp.gate_proj,
        )
        layer.mlp.up_proj = FusedRMSNormLinear(
            post_attention_norm,
            layer.mlp.up_proj,
        )
        layer.post_attention_layernorm = DecodeOnlyIdentity(layer.hidden_size)
        fused_projections += 2

    setattr(model, _CONVERSION_MARKER, True)
    model.eval()
    return ConversionReport(
        decoder_layers=len(llama_model.layers),
        fused_projections=fused_projections,
        dtype=dtype,
        device=device,
    )


__all__ = [
    "ConversionReport",
    "DecodeOnlyIdentity",
    "convert_llama_to_fused_rmsnorm_linear",
]
