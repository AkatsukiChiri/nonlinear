#!/usr/bin/env python3
"""Benchmark FP16 Llama 3 and OPT inference with PyTorch Transformers."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FP16 Llama 3 or OPT generation using PyTorch and "
            "Transformers."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face/ModelScope model ID or local model directory.",
    )
    parser.add_argument(
        "--source",
        choices=("huggingface", "modelscope", "local"),
        default="huggingface",
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--input-length", type=int, required=True)
    parser.add_argument("--output-length", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code supplied by the model repository.",
    )
    args = parser.parse_args()
    for name in ("batch_size", "input_length", "output_length", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def resolve_model_path(model: str, source: str) -> str:
    """Resolve a model ID to a path understood by Transformers."""
    if source == "local":
        model_path = Path(model).expanduser().resolve()
        if not model_path.is_dir():
            raise RuntimeError(f"Local model directory does not exist: {model_path}")
        return str(model_path)
    if source == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError as error:
            raise RuntimeError(
                "ModelScope loading requires the 'modelscope' package."
            ) from error
        return str(snapshot_download(model))
    return model


def validate_model_family(config: Any) -> str:
    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type == "llama":
        return "llama3"
    if model_type == "opt":
        return "opt"
    raise RuntimeError(
        f"Unsupported model_type {model_type!r}; expected Llama 3 or OPT."
    )


def make_inputs(
    tokenizer: Any,
    batch_size: int,
    input_length: int,
    device: Any,
) -> tuple[Any, Any]:
    """Build exact-shape input tensors ``[Batch, InputLength]``."""
    import torch

    seed_ids = tokenizer.encode(
        "The following is a benchmark request for model inference. ",
        add_special_tokens=False,
    )
    if not seed_ids:
        raise RuntimeError("Tokenizer produced no IDs for the benchmark prompt.")
    one_prompt: list[int] = []
    if tokenizer.bos_token_id is not None:
        one_prompt.append(int(tokenizer.bos_token_id))
    remaining = input_length - len(one_prompt)
    if remaining > 0:
        repeats = math.ceil(remaining / len(seed_ids))
        one_prompt.extend((seed_ids * repeats)[:remaining])
    one_prompt = one_prompt[:input_length]
    input_ids = torch.tensor(
        [one_prompt] * batch_size, dtype=torch.long, device=device
    )
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def synchronize(device: Any) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def update_progress(completed: int, total: int, phase: str) -> None:
    """Render a dependency-free progress bar outside the timed region."""
    bar_width = 30
    filled_width = round(bar_width * completed / total)
    bar = "=" * filled_width + "-" * (bar_width - filled_width)
    percentage = completed / total * 100
    print(
        f"\r{phase:<10s} [{bar}] {completed}/{total} ({percentage:5.1f}%)",
        end="\n" if completed == total else "",
        file=sys.stderr,
        flush=True,
    )


def infer_once(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    output_length: int,
) -> list[float]:
    """Run greedy autoregressive inference and time every output token.

    The first duration is time to first token (prefill plus first-token
    selection). Remaining durations are decode steps that reuse the KV cache.
    All values are batch-level wall-clock seconds measured after CUDA
    synchronization.
    """
    import torch

    token_latencies: list[float] = []
    synchronize(input_ids.device)
    started_at = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values
    synchronize(input_ids.device)
    token_latencies.append(time.perf_counter() - started_at)

    current_attention_mask = attention_mask
    for _ in range(1, output_length):
        current_attention_mask = torch.cat(
            (
                current_attention_mask,
                torch.ones(
                    (current_attention_mask.shape[0], 1),
                    dtype=current_attention_mask.dtype,
                    device=current_attention_mask.device,
                ),
            ),
            dim=1,
        )
        synchronize(input_ids.device)
        started_at = time.perf_counter()
        outputs = model(
            input_ids=next_token,
            attention_mask=current_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        synchronize(input_ids.device)
        token_latencies.append(time.perf_counter() - started_at)
    return token_latencies


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "PyTorch and Transformers are required; use the nonl environment."
        ) from error

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FP16 inference benchmark requires an available CUDA GPU.")

    model_path = resolve_model_path(args.model, args.source)
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=args.trust_remote_code
    )
    family = validate_model_family(config)
    max_positions = getattr(config, "max_position_embeddings", None)
    total_length = args.input_length + args.output_length
    if max_positions is not None and total_length > int(max_positions):
        raise RuntimeError(
            f"input + output length ({total_length}) exceeds the model limit "
            f"({max_positions})."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    input_ids, attention_mask = make_inputs(
        tokenizer, args.batch_size, args.input_length, device
    )
    print(
        f"Model={args.model} family={family} dtype=float16 device={device} "
        f"batch_size={args.batch_size} input_length={args.input_length} "
        f"output_length={args.output_length}"
    )
    with torch.inference_mode():
        if args.warmup:
            update_progress(0, args.warmup, "Warmup")
        for warmup_index in range(args.warmup):
            infer_once(model, input_ids, attention_mask, args.output_length)
            update_progress(warmup_index + 1, args.warmup, "Warmup")
        synchronize(device)

        per_iteration_latencies: list[list[float]] = []
        update_progress(0, args.iterations, "Iteration")
        for iteration in range(args.iterations):
            per_iteration_latencies.append(
                infer_once(model, input_ids, attention_mask, args.output_length)
            )
            update_progress(iteration + 1, args.iterations, "Iteration")

    latencies = [sum(values) for values in per_iteration_latencies]
    ttfts = [values[0] for values in per_iteration_latencies]
    decode_latencies = [
        latency
        for values in per_iteration_latencies
        for latency in values[1:]
    ]
    total_seconds = sum(latencies)
    generated_tokens = args.batch_size * args.output_length * args.iterations
    requests = args.batch_size * args.iterations
    print("\nResults")
    print(f"  Requests:                  {requests}")
    print(f"  Request throughput:        {requests / total_seconds:.2f} req/s")
    print(
        f"  Output token throughput:   "
        f"{generated_tokens / total_seconds:.2f} tok/s"
    )
    print(
        f"  Batch latency mean/p50/p95: "
        f"{statistics.mean(latencies):.3f} / "
        f"{percentile(latencies, 0.50):.3f} / "
        f"{percentile(latencies, 0.95):.3f} s"
    )
    print(
        f"  TTFT mean/p50/p95:         "
        f"{statistics.mean(ttfts) * 1000:.3f} / "
        f"{percentile(ttfts, 0.50) * 1000:.3f} / "
        f"{percentile(ttfts, 0.95) * 1000:.3f} ms"
    )
    if decode_latencies:
        print(
            f"  TPOT mean/p50/p95:         "
            f"{statistics.mean(decode_latencies) * 1000:.3f} / "
            f"{percentile(decode_latencies, 0.50) * 1000:.3f} / "
            f"{percentile(decode_latencies, 0.95) * 1000:.3f} ms/token"
        )

    print("\nPer-output-token latency (batch-level)")
    for token_index in range(args.output_length):
        values = [latencies[token_index] for latencies in per_iteration_latencies]
        token_label = "TTFT" if token_index == 0 else "decode"
        print(
            f"  Token {token_index + 1:4d} ({token_label:6s}): "
            f"mean={statistics.mean(values) * 1000:.3f} ms, "
            f"p50={percentile(values, 0.50) * 1000:.3f} ms, "
            f"p95={percentile(values, 0.95) * 1000:.3f} ms"
        )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
