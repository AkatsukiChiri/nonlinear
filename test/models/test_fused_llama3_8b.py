from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.fused_llama import convert_llama_to_fused_rmsnorm_linear
from models.fused_add_llama import (
    convert_llama_to_fused_add_rmsnorm_linear,
)


MODEL_PATH = Path("/home/mashaobo/models/llama3-8b/")
RUN_REAL_MODEL = os.environ.get("NONLINEAR_RUN_LLAMA3_8B") == "1"


@pytest.mark.llama3_8b
@pytest.mark.skipif(
    not RUN_REAL_MODEL,
    reason="set NONLINEAR_RUN_LLAMA3_8B=1 to run the 8B smoke test",
)
def test_real_llama3_8b_single_token_logits() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not MODEL_PATH.is_dir():
        pytest.skip(f"model path is unavailable: {MODEL_PATH}")

    model = LlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    input_ids = torch.tensor([[128000]], device="cuda", dtype=torch.long)

    with torch.inference_mode():
        expected = model(
            input_ids=input_ids,
            use_cache=False,
            logits_to_keep=1,
        ).logits.float()
        report = convert_llama_to_fused_rmsnorm_linear(model)
        actual = model(
            input_ids=input_ids,
            use_cache=False,
            logits_to_keep=1,
        ).logits.float()

    max_absolute_error = (actual - expected).abs().max().item()
    assert report.decoder_layers == 32
    assert report.fused_projections == 160
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))
    assert max_absolute_error < 0.25


@pytest.mark.llama3_8b
@pytest.mark.skipif(
    not RUN_REAL_MODEL,
    reason="set NONLINEAR_RUN_LLAMA3_8B=1 to run the 8B smoke test",
)
def test_real_llama3_8b_fused_add_single_token_logits() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not MODEL_PATH.is_dir():
        pytest.skip(f"model path is unavailable: {MODEL_PATH}")

    model = LlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    input_ids = torch.tensor([[128000]], device="cuda", dtype=torch.long)

    with torch.inference_mode():
        expected = model(
            input_ids=input_ids,
            use_cache=False,
            logits_to_keep=1,
        ).logits.float()
        report = convert_llama_to_fused_add_rmsnorm_linear(model)
        actual = model(
            input_ids=input_ids,
            use_cache=False,
            logits_to_keep=1,
        ).logits.float()

    max_absolute_error = (actual - expected).abs().max().item()
    assert report.decoder_layers == 32
    assert report.fused_add_projections == 63
    assert report.fused_rms_projections == 97
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))
    assert max_absolute_error < 0.25
