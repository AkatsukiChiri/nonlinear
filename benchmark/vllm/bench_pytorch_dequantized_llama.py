"""Benchmark a PyTorch Llama loaded by dequantizing a W8A8 checkpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaRotaryEmbedding,
)


DEFAULT_MODEL = Path(
    "/home/mashaobo/models/Meta-Llama-3.1-8B-Instruct-W8A8"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--profile-nvtx",
        action="store_true",
        help="mark measured prefill/decode components with NVTX ranges",
    )
    return parser.parse_args()


def load_dequantized_model(
    model_path: Path, dtype: torch.dtype
) -> torch.nn.Module:
    """Stream checkpoint tensors into a meta-initialized HF model."""
    config = AutoConfig.from_pretrained(model_path)
    with torch.device("meta"):
        model = LlamaForCausalLM(config).to(dtype=dtype)

    with (model_path / "model.safetensors.index.json").open(
        encoding="utf-8"
    ) as index_file:
        weight_map = json.load(index_file)["weight_map"]
    shards = sorted(set(weight_map.values()))
    loaded_names: set[str] = set()
    for shard_name in shards:
        shard_state: dict[str, torch.Tensor] = {}
        with safe_open(
            model_path / shard_name, framework="pt", device="cpu"
        ) as checkpoint:
            names = set(checkpoint.keys())
            for name in sorted(names):
                if name.endswith((".weight_scale", ".input_scale")):
                    continue
                tensor = checkpoint.get_tensor(name)
                scale_name = name.removesuffix(".weight") + ".weight_scale"
                if tensor.dtype == torch.int8:
                    scale_shard = weight_map.get(scale_name)
                    if scale_shard is None:
                        raise KeyError(f"missing scale for INT8 tensor {name}")
                    if scale_shard == shard_name and scale_name in names:
                        scale = checkpoint.get_tensor(scale_name)
                    else:
                        with safe_open(
                            model_path / scale_shard,
                            framework="pt",
                            device="cpu",
                        ) as scale_checkpoint:
                            scale = scale_checkpoint.get_tensor(scale_name)
                    tensor = tensor.float() * scale.float().reshape(-1, 1)
                shard_state[name] = tensor.to(device="cuda", dtype=dtype)
                loaded_names.add(name)
        model.load_state_dict(shard_state, strict=False, assign=True)
        del shard_state

    if config.tie_word_embeddings:
        model.tie_weights()
    for module in model.modules():
        if isinstance(module, LlamaRotaryEmbedding):
            inv_freq, attention_scaling = module.rope_init_fn(
                module.config, torch.device("cuda")
            )
            module.inv_freq = inv_freq
            module.original_inv_freq = inv_freq
            module.attention_scaling = attention_scaling
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.device.type == "meta"
    ]
    if missing:
        raise RuntimeError(f"unloaded meta parameters: {missing[:8]}")
    missing_buffers = [
        name for name, buffer in model.named_buffers()
        if buffer.device.type == "meta"
    ]
    if missing_buffers:
        raise RuntimeError(f"unloaded meta buffers: {missing_buffers[:8]}")
    model.eval()
    return model


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


class ComponentNvtxRanges:
    """Annotate measured Llama forwards with non-layer component ranges."""

    def __init__(self, model: torch.nn.Module) -> None:
        llama_model = getattr(model, "model", None)
        decoder_layers = getattr(llama_model, "layers", None)
        if llama_model is None or decoder_layers is None:
            raise ValueError("expected a Llama model with model.layers")
        self._phase: str | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for layer in decoder_layers:
            self._register(layer.input_layernorm, "rmsnorm")
            self._register(layer.self_attn, "attention")
            self._register(layer.self_attn.q_proj, "qkv_gemm")
            self._register(layer.self_attn.k_proj, "qkv_gemm")
            self._register(layer.self_attn.v_proj, "qkv_gemm")
            self._register(layer.self_attn.o_proj, "attention_out_gemm")
            self._register(layer.post_attention_layernorm, "rmsnorm")
            self._register(layer.mlp, "mlp")
            self._register(layer.mlp.gate_proj, "mlp_gate_gemm")
            self._register(layer.mlp.up_proj, "mlp_up_gemm")
            self._register(layer.mlp.act_fn, "mlp_activation")
            self._register(layer.mlp.down_proj, "mlp_down_gemm")
        self._register(llama_model.norm, "final_rmsnorm")
        self._register(model.lm_head, "lm_head_gemm")

    def _register(self, module: torch.nn.Module, name: str) -> None:
        self._handles.append(
            module.register_forward_pre_hook(self._pre_hook(name))
        )
        self._handles.append(module.register_forward_hook(self._post_hook))

    def _pre_hook(self, name: str):  # type: ignore[no-untyped-def]
        def hook(*_args: object) -> None:
            if self._phase is not None:
                torch.cuda.nvtx.range_push(f"{self._phase}/{name}")

        return hook

    def _post_hook(self, *_args: object) -> None:
        if self._phase is not None:
            torch.cuda.nvtx.range_pop()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if self._phase is not None:
            raise RuntimeError("nested profiling phases are unsupported")
        self._phase = name
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
            self._phase = None

    def close(self) -> None:
        """Remove hooks after profiling to avoid retaining model references."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def infer_once(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    output_tokens: int,
    nvtx_ranges: ComponentNvtxRanges | None = None,
) -> tuple[float, list[float], list[list[int]]]:
    attention_mask = torch.ones_like(input_ids)
    torch.cuda.synchronize()
    start = time.perf_counter()
    prefill_context = (
        nvtx_ranges.phase("prefill")
        if nvtx_ranges is not None else nullcontext()
    )
    with prefill_context:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    past_key_values = outputs.past_key_values
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - start
    decode_latencies = []
    for _ in range(1, output_tokens):
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=1
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        decode_context = (
            nvtx_ranges.phase("decode")
            if nvtx_ranges is not None else nullcontext()
        )
        with decode_context:
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        past_key_values = outputs.past_key_values
        torch.cuda.synchronize()
        decode_latencies.append(time.perf_counter() - start)
    token_tensor = torch.cat(generated, dim=1).cpu().tolist()
    return prefill_seconds, decode_latencies, token_tensor


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    seed_ids = tokenizer.encode(
        "The following is a deterministic benchmark input for model inference. ",
        add_special_tokens=False,
    )
    prompt = [seed_ids[i % len(seed_ids)] for i in range(args.prompt_tokens)]
    input_ids = torch.tensor(
        [prompt] * args.batch, dtype=torch.long, device="cuda"
    )
    model = load_dequantized_model(args.model, dtype)
    with torch.inference_mode():
        for _ in range(args.warmup):
            infer_once(model, input_ids, args.output_tokens)
        prefill_latencies = []
        decode_latencies = []
        output_ids: list[list[int]] = []
        nvtx_ranges = (
            ComponentNvtxRanges(model) if args.profile_nvtx else None
        )
        try:
            for _ in range(args.iterations):
                prefill, decode, output_ids = infer_once(
                    model, input_ids, args.output_tokens, nvtx_ranges
                )
                prefill_latencies.append(prefill)
                decode_latencies.extend(decode)
        finally:
            if nvtx_ranges is not None:
                nvtx_ranges.close()

    result = {
        "implementation": f"pytorch-{args.dtype}",
        "batch": args.batch,
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "prefill_mean_ms": statistics.mean(prefill_latencies) * 1000,
        "prefill_p50_ms": percentile(prefill_latencies, 0.5) * 1000,
        "prefill_p95_ms": percentile(prefill_latencies, 0.95) * 1000,
        "decode_tpot_ms": (
            statistics.mean(decode_latencies) * 1000
            if decode_latencies else 0.0
        ),
        "last_output_token_ids": output_ids,
    }
    print("RESULT_JSON=" + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
