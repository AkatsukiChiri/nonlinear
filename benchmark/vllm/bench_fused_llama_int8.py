"""Benchmark native and fused vLLM Llama W8A8 model inference.

Run each implementation in a separate process. This script measures complete
engine inference, including attention, KV cache, logits, and scheduling, while
excluding model loading and fused-weight conversion warmup.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


DEFAULT_MODEL = Path(
    "/home/mashaobo/models/Meta-Llama-3.1-8B-Instruct-W8A8"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation",
        choices=("vllm", "fused", "tensorcore"),
        required=True,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--disable-v1", action="store_true")
    parser.add_argument(
        "--profile-workload",
        choices=("prefill", "decode"),
        help="run only one workload type for an external profiler",
    )
    parser.add_argument(
        "--nsys-capture",
        action="store_true",
        help="bound each measured workload with CUDA profiler start/stop",
    )
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


@contextmanager
def nsys_capture(enabled: bool) -> Iterator[None]:
    """Capture only measured inference when Nsys uses cudaProfilerApi."""
    if enabled:
        torch.cuda.profiler.start()
    try:
        yield
    finally:
        if enabled:
            torch.cuda.profiler.stop()


def main() -> None:
    args = parse_args()
    if args.batch <= 0 or args.prompt_tokens <= 0 or args.output_tokens <= 0:
        raise ValueError("batch, prompt-tokens, and output-tokens must be positive")
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup nonnegative")

    if args.disable_v1:
        os.environ["VLLM_USE_V1"] = "0"
    if args.implementation == "fused":
        from models.vllm import register_fused_int8_llama

        register_fused_int8_llama()
    elif args.implementation == "tensorcore":
        from models.vllm import register_tensorcore_fused_int8_llama

        register_tensorcore_fused_int8_llama()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.model),
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        enable_prefix_caching=False,
    )
    tokenizer = llm.get_tokenizer()
    seed_ids = tokenizer.encode(
        "The following is a deterministic benchmark input for model inference. ",
        add_special_tokens=False,
    )
    if not seed_ids:
        raise RuntimeError("tokenizer produced no benchmark tokens")
    prompt_ids = [
        seed_ids[index % len(seed_ids)] for index in range(args.prompt_tokens)
    ]
    prompts = [{"prompt_token_ids": prompt_ids} for _ in range(args.batch)]
    prefill_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
    )
    decode_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
    )

    if args.profile_workload == "decode":
        if args.batch != 1:
            raise ValueError("profiled decode requires batch=1")
        decode_profile_sampling = SamplingParams(
            temperature=0.0, max_tokens=2, ignore_eos=True
        )
        for _ in range(args.warmup):
            llm.generate(prompts, decode_profile_sampling, use_tqdm=False)
    else:
        for _ in range(args.warmup):
            llm.generate(prompts, prefill_sampling, use_tqdm=False)
            if args.profile_workload is None:
                llm.generate(prompts, decode_sampling, use_tqdm=False)

    if args.profile_workload == "decode":
        engine = llm.llm_engine
        decode_latencies = []
        output_token_ids: list[list[int]] = []
        for iteration in range(args.iterations):
            request_id = f"profiled-decode-{iteration}"
            engine.add_request(
                request_id,
                {"prompt_token_ids": prompt_ids},
                decode_profile_sampling,
            )
            engine.step()  # Build the 512-token KV cache and emit token one.
            start = time.perf_counter()
            with nsys_capture(args.nsys_capture):
                outputs = engine.step()  # One-token cached decode only.
            decode_latencies.append(time.perf_counter() - start)
            output_token_ids = [
                output.outputs[0].token_ids for output in outputs
            ]
        mean_decode_seconds = statistics.mean(decode_latencies)
        result = {
            "implementation": args.implementation,
            "batch": args.batch,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": 1,
            "prefill_mean_ms": 0.0,
            "prefill_p50_ms": 0.0,
            "prefill_p95_ms": 0.0,
            "mean_ms": mean_decode_seconds * 1000,
            "p50_ms": percentile(decode_latencies, 0.50) * 1000,
            "p95_ms": percentile(decode_latencies, 0.95) * 1000,
            "output_tokens_per_second": 1.0 / mean_decode_seconds,
            "decode_tpot_ms": mean_decode_seconds * 1000,
            "last_output_token_ids": output_token_ids,
        }
        print("RESULT_JSON=" + json.dumps(result, separators=(",", ":")))
        return

    prefill_latencies = []
    end_to_end_latencies = []
    generated_token_counts = []
    output_token_ids: list[list[int]] = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        with nsys_capture(args.nsys_capture):
            prefill_outputs = llm.generate(
                prompts, prefill_sampling, use_tqdm=False
            )
        prefill_latencies.append(time.perf_counter() - start)
        if args.profile_workload == "prefill":
            generated = [
                output.outputs[0].token_ids for output in prefill_outputs
            ]
            generated_token_counts.append(sum(len(ids) for ids in generated))
            output_token_ids = generated
            continue
        start = time.perf_counter()
        with nsys_capture(args.nsys_capture):
            outputs = llm.generate(prompts, decode_sampling, use_tqdm=False)
        end_to_end_latencies.append(time.perf_counter() - start)
        generated = [output.outputs[0].token_ids for output in outputs]
        generated_token_counts.append(sum(len(ids) for ids in generated))
        output_token_ids = generated

    mean_prefill_seconds = statistics.mean(prefill_latencies)
    mean_seconds = (
        statistics.mean(end_to_end_latencies)
        if end_to_end_latencies else mean_prefill_seconds
    )
    end_to_end_summary = end_to_end_latencies or prefill_latencies
    mean_generated = statistics.mean(generated_token_counts)
    decode_steps = max(args.output_tokens - 1, 1)
    decode_seconds = max(mean_seconds - mean_prefill_seconds, 0.0)
    result = {
        "implementation": args.implementation,
        "batch": args.batch,
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "prefill_mean_ms": mean_prefill_seconds * 1000,
        "prefill_p50_ms": percentile(prefill_latencies, 0.50) * 1000,
        "prefill_p95_ms": percentile(prefill_latencies, 0.95) * 1000,
        "mean_ms": mean_seconds * 1000,
        "p50_ms": percentile(end_to_end_summary, 0.50) * 1000,
        "p95_ms": percentile(end_to_end_summary, 0.95) * 1000,
        "output_tokens_per_second": mean_generated / mean_seconds,
        "decode_tpot_ms": decode_seconds * 1000 / decode_steps,
        "last_output_token_ids": output_token_ids,
    }
    print("RESULT_JSON=" + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
