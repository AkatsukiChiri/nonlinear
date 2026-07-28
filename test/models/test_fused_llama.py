from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear import FusedRMSNormLinear
from models.fused_llama import (
    DecodeOnlyIdentity,
    convert_llama_to_fused_rmsnorm_linear,
)


def _build_model(*, device: str = "cuda") -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
        rms_norm_eps=1e-5,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return (
        LlamaForCausalLM(config)
        .to(device=device, dtype=torch.float16)
        .eval()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_conversion_replaces_five_projections_per_layer() -> None:
    model = _build_model()

    report = convert_llama_to_fused_rmsnorm_linear(model)

    assert report.decoder_layers == 2
    assert report.fused_projections == 10
    assert report.dtype == torch.float16
    for layer in model.model.layers:
        assert isinstance(layer.input_layernorm, DecodeOnlyIdentity)
        assert isinstance(layer.post_attention_layernorm, DecodeOnlyIdentity)
        assert isinstance(layer.self_attn.q_proj, FusedRMSNormLinear)
        assert isinstance(layer.self_attn.k_proj, FusedRMSNormLinear)
        assert isinstance(layer.self_attn.v_proj, FusedRMSNormLinear)
        assert isinstance(layer.mlp.gate_proj, FusedRMSNormLinear)
        assert isinstance(layer.mlp.up_proj, FusedRMSNormLinear)
        assert isinstance(layer.self_attn.o_proj, torch.nn.Linear)
        assert isinstance(layer.mlp.down_proj, torch.nn.Linear)
    assert not isinstance(model.model.norm, DecodeOnlyIdentity)
    assert isinstance(model.lm_head, torch.nn.Linear)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_llama_logits_match_transformers(batch: int) -> None:
    torch.manual_seed(20260728)
    reference = _build_model()
    fused = deepcopy(reference)
    convert_llama_to_fused_rmsnorm_linear(fused)
    input_ids = torch.randint(
        0,
        reference.config.vocab_size,
        (batch, 1),
        device="cuda",
    )

    with torch.inference_mode():
        expected = reference(input_ids=input_ids, use_cache=False).logits
        actual = fused(input_ids=input_ids, use_cache=False).logits

    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_llama_preserves_cache_interface() -> None:
    torch.manual_seed(20260728)
    reference = _build_model()
    fused = deepcopy(reference)
    convert_llama_to_fused_rmsnorm_linear(fused)
    input_ids = torch.randint(0, 128, (2, 1), device="cuda")

    with torch.inference_mode():
        expected = reference(
            input_ids=input_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
        )
        actual = fused(
            input_ids=input_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
        )

    assert expected.past_key_values is not None
    assert actual.past_key_values is not None
    assert actual.past_key_values.get_seq_length() == 1
    torch.testing.assert_close(
        actual.logits,
        expected.logits,
        atol=3e-3,
        rtol=3e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_decode_only_constraints_are_explicit() -> None:
    model = _build_model()
    convert_llama_to_fused_rmsnorm_linear(model)
    token = torch.randint(0, 128, (1, 1), device="cuda")

    with pytest.raises(RuntimeError, match="inference_mode"):
        model(input_ids=token)

    model.train()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="eval"):
        model(input_ids=token)

    model.eval()
    prompt = torch.randint(0, 128, (1, 2), device="cuda")
    with torch.inference_mode(), pytest.raises(ValueError, match="SeqLen == 1"):
        model(input_ids=prompt)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_repeated_conversion_is_rejected() -> None:
    model = _build_model()
    convert_llama_to_fused_rmsnorm_linear(model)

    with pytest.raises(RuntimeError, match="already converted"):
        convert_llama_to_fused_rmsnorm_linear(model)


def test_conversion_rejects_cpu_model() -> None:
    model = _build_model(device="cpu")

    with pytest.raises(ValueError, match="CUDA"):
        convert_llama_to_fused_rmsnorm_linear(model)
