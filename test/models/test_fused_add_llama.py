from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear import (
    FusedAddRMSNormLinear,
    FusedRMSNormLinear,
)
from models.fused_add_llama import (
    FusedAddLlamaDecoderLayer,
    PendingResidual,
    convert_llama_to_fused_add_rmsnorm_linear,
)


def _build_model(*, layers: int = 3) -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=layers,
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
    return LlamaForCausalLM(config).half().cuda().eval()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_add_conversion_has_expected_projection_layout() -> None:
    model = _build_model(layers=3)

    report = convert_llama_to_fused_add_rmsnorm_linear(model)

    assert report.decoder_layers == 3
    assert report.fused_add_projections == 5
    assert report.fused_rms_projections == 10
    for index, layer in enumerate(model.model.layers):
        assert isinstance(layer, FusedAddLlamaDecoderLayer)
        if index == 0:
            assert isinstance(layer.self_attn.q_proj, FusedRMSNormLinear)
        else:
            assert isinstance(layer.self_attn.q_proj, FusedAddRMSNormLinear)
        assert isinstance(layer.self_attn.k_proj, FusedRMSNormLinear)
        assert isinstance(layer.self_attn.v_proj, FusedRMSNormLinear)
        assert isinstance(layer.mlp.gate_proj, FusedAddRMSNormLinear)
        assert isinstance(layer.mlp.up_proj, FusedRMSNormLinear)
        assert isinstance(layer.self_attn.o_proj, torch.nn.Linear)
        assert isinstance(layer.mlp.down_proj, torch.nn.Linear)
        assert layer.is_last_layer == (index == 2)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_add_llama_logits_match_transformers(batch: int) -> None:
    torch.manual_seed(20260728)
    reference = _build_model()
    fused = deepcopy(reference)
    convert_llama_to_fused_add_rmsnorm_linear(fused)
    input_ids = torch.randint(0, 128, (batch, 1), device="cuda")

    with torch.inference_mode():
        expected = reference(input_ids=input_ids, use_cache=False).logits
        actual = fused(input_ids=input_ids, use_cache=False).logits

    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nonfinal_layer_returns_pending_residual() -> None:
    model = _build_model(layers=2)
    convert_llama_to_fused_add_rmsnorm_linear(model)
    first_layer = model.model.layers[0]
    hidden_states = torch.randn(
        1,
        1,
        model.config.hidden_size,
        device="cuda",
        dtype=torch.float16,
    )
    position_ids = torch.zeros(1, 1, device="cuda", dtype=torch.long)
    position_embeddings = model.model.rotary_emb(
        hidden_states,
        position_ids,
    )

    with torch.inference_mode():
        output = first_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            use_cache=False,
        )[0]

    assert isinstance(output, PendingResidual)
    assert output.update.shape == hidden_states.shape
    assert output.residual.shape == hidden_states.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_add_llama_preserves_dynamic_cache() -> None:
    torch.manual_seed(20260728)
    reference = _build_model()
    fused = deepcopy(reference)
    convert_llama_to_fused_add_rmsnorm_linear(fused)
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

    assert actual.past_key_values is not None
    assert actual.past_key_values.get_seq_length() == 1
    torch.testing.assert_close(
        actual.logits,
        expected.logits,
        atol=5e-3,
        rtol=5e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_add_llama_rejects_hidden_state_output_and_prefill() -> None:
    model = _build_model()
    convert_llama_to_fused_add_rmsnorm_linear(model)
    token = torch.randint(0, 128, (1, 1), device="cuda")

    with torch.inference_mode(), pytest.raises(
        ValueError,
        match="output_hidden_states",
    ):
        model(input_ids=token, output_hidden_states=True)

    prompt = torch.randint(0, 128, (1, 2), device="cuda")
    with torch.inference_mode(), pytest.raises(ValueError, match="SeqLen == 1"):
        model(input_ids=prompt)
