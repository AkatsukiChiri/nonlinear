#!/usr/bin/env python3
"""Benchmark Llama 3 or OPT served by vLLM's OpenAI-compatible API.

``batch_size`` means the number of concurrent requests submitted together.
Each request contains exactly ``input_length`` token IDs and asks vLLM to
generate exactly ``output_length`` tokens by setting ``ignore_eos``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequestResult:
    """Timing and token counts for one completion request."""

    latency_seconds: float
    time_to_first_token_seconds: float
    prompt_tokens: int
    completion_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a Llama 3 or OPT model served by vLLM."
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--input-length", type=int, required=True)
    parser.add_argument("--output-length", type=int, required=True)
    parser.add_argument(
        "--model",
        default=None,
        help="Served model name; auto-detected when the server exposes one model.",
    )
    parser.add_argument(
        "--model-family",
        choices=("llama3", "opt"),
        default=None,
        help="Optional model-family assertion; otherwise detected from config.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer name/path; defaults to model root from /models.")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warm-up batches.")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of measured batches.")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    for name in ("batch_size", "input_length", "output_length", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def request_json(url: str, timeout: float, api_key: str | None) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} failed ({error.code}): {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot connect to {url}: {error.reason}. Check that vLLM is running "
            "and that --base-url uses the correct host and port."
        ) from error


def resolve_tokenizer(args: argparse.Namespace) -> str:
    models = request_json(
        f"{args.base_url.rstrip('/')}/models", args.timeout, args.api_key
    ).get("data", [])
    if not models:
        raise RuntimeError(f"No models are available at {args.base_url}/models.")

    if args.model is None:
        if len(models) != 1:
            model_names = ", ".join(str(model.get("id")) for model in models)
            raise RuntimeError(
                f"The server exposes multiple models ({model_names}); select one "
                "with --model."
            )
        selected_model = models[0]
        args.model = str(selected_model["id"])
    else:
        selected_model = next(
            (model for model in models if model.get("id") == args.model), None
        )
        if selected_model is None:
            model_names = ", ".join(str(model.get("id")) for model in models)
            raise RuntimeError(
                f"Model {args.model!r} was not found. Available models: {model_names}"
            )

    if args.tokenizer:
        return args.tokenizer
    return str(selected_model.get("root") or args.model)


def make_prompt_token_ids(tokenizer: Any, input_length: int) -> list[int]:
    """Create an exact-length synthetic prompt without decode/re-tokenize drift."""
    seed_ids = tokenizer.encode(
        "The following is a benchmark request for language model inference. ",
        add_special_tokens=False,
    )
    if not seed_ids:
        raise RuntimeError("Tokenizer produced no token IDs for the benchmark seed.")
    prompt_ids: list[int] = []
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is not None:
        prompt_ids.append(int(bos_token_id))
    remaining = input_length - len(prompt_ids)
    if remaining < 0:
        return prompt_ids[:input_length]
    repeats = math.ceil(remaining / len(seed_ids)) if remaining else 0
    prompt_ids.extend((seed_ids * repeats)[:remaining])
    return prompt_ids


def resolve_model_family(config: Any, requested_family: str | None) -> str:
    """Identify and validate the two model families covered by this benchmark."""
    model_type = str(getattr(config, "model_type", "")).lower()
    detected_family = {"llama": "llama3", "opt": "opt"}.get(model_type)
    if detected_family is None:
        raise RuntimeError(
            f"Unsupported model_type {model_type!r}; this benchmark supports "
            "Llama 3 ('llama') and OPT ('opt')."
        )
    if requested_family and requested_family != detected_family:
        raise RuntimeError(
            f"--model-family={requested_family} does not match model_type "
            f"{model_type!r} ({detected_family})."
        )
    return detected_family


def run_request(
    args: argparse.Namespace,
    prompt_token_ids: list[int],
    tokenizer: Any,
    ready_barrier: threading.Barrier,
    start_event: threading.Event,
) -> RequestResult:
    payload = {
        "model": args.model,
        "prompt": prompt_token_ids,
        "max_tokens": args.output_length,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    ready_barrier.wait()
    start_event.wait()
    started_at = time.perf_counter()
    first_token_at: float | None = None
    output_parts: list[str] = []
    usage: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                text = choices[0].get("text", "") if choices else ""
                if text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    output_parts.append(text)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"POST /completions failed ({error.code}): {detail}"
        ) from error

    finished_at = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("vLLM returned no generated text.")
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = len(
            tokenizer.encode("".join(output_parts), add_special_tokens=False)
        )
    return RequestResult(
        latency_seconds=finished_at - started_at,
        time_to_first_token_seconds=first_token_at - started_at,
        prompt_tokens=int(usage.get("prompt_tokens", len(prompt_token_ids))),
        completion_tokens=int(completion_tokens),
    )


def run_batch(
    args: argparse.Namespace,
    prompt_token_ids: list[int],
    tokenizer: Any,
) -> tuple[float, list[RequestResult]]:
    ready_barrier = threading.Barrier(args.batch_size + 1)
    start_event = threading.Event()
    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = [
            executor.submit(
                run_request,
                args,
                prompt_token_ids,
                tokenizer,
                ready_barrier,
                start_event,
            )
            for _ in range(args.batch_size)
        ]
        ready_barrier.wait()
        batch_started_at = time.perf_counter()
        start_event.set()
        results = [future.result() for future in futures]
        batch_latency = time.perf_counter() - batch_started_at
    return batch_latency, results


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def update_progress(completed: int, total: int, phase: str) -> None:
    """Render one dependency-free progress bar for all benchmark batches."""
    bar_width = 30
    filled_width = round(bar_width * completed / total)
    bar = "=" * filled_width + "-" * (bar_width - filled_width)
    end = "\n" if completed == total else ""
    print(
        f"\rBenchmark [{bar}] {completed}/{total} ({phase})",
        end=end,
        file=sys.stderr,
        flush=True,
    )


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required; run this script in the nonl environment."
        ) from error

    tokenizer_path = resolve_tokenizer(args)
    config = AutoConfig.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=True
    )
    model_family = resolve_model_family(config, args.model_family)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=True
    )
    prompt_token_ids = make_prompt_token_ids(tokenizer, args.input_length)

    print(
        f"Model={args.model} family={model_family} "
        f"batch_size={args.batch_size} "
        f"input_length={len(prompt_token_ids)} output_length={args.output_length}"
    )
    total_batches = args.warmup + args.iterations
    completed_batches = 0
    initial_phase = "warm-up" if args.warmup else "measuring"
    update_progress(completed_batches, total_batches, initial_phase)
    for _ in range(args.warmup):
        run_batch(args, prompt_token_ids, tokenizer)
        completed_batches += 1
        phase = "warm-up" if completed_batches < args.warmup else "measuring"
        update_progress(completed_batches, total_batches, phase)

    batch_latencies: list[float] = []
    all_results: list[RequestResult] = []
    for _ in range(args.iterations):
        latency, results = run_batch(args, prompt_token_ids, tokenizer)
        batch_latencies.append(latency)
        all_results.extend(results)
        completed_batches += 1
        update_progress(completed_batches, total_batches, "measuring")

    total_time = sum(batch_latencies)
    total_output_tokens = sum(result.completion_tokens for result in all_results)
    ttfts = [result.time_to_first_token_seconds for result in all_results]
    latencies = [result.latency_seconds for result in all_results]
    generated_after_first = sum(
        max(0, result.completion_tokens - 1) for result in all_results
    )
    generation_time = sum(
        max(0.0, result.latency_seconds - result.time_to_first_token_seconds)
        for result in all_results
    )

    print("\nResults")
    print(f"  Requests:                  {len(all_results)}")
    print(f"  Request throughput:        {len(all_results) / total_time:.2f} req/s")
    print(f"  Output token throughput:   {total_output_tokens / total_time:.2f} tok/s")
    print(f"  Batch latency mean:        {statistics.mean(batch_latencies):.3f} s")
    print(f"  Request latency mean/p95:  {statistics.mean(latencies):.3f} / "
          f"{percentile(latencies, 0.95):.3f} s")
    print(f"  TTFT mean/p50/p95:         {statistics.mean(ttfts) * 1000:.2f} / "
          f"{percentile(ttfts, 0.50) * 1000:.2f} / "
          f"{percentile(ttfts, 0.95) * 1000:.2f} ms")
    if generation_time > 0 and generated_after_first > 0:
        print(f"  Mean inter-token latency:  "
              f"{generation_time / generated_after_first * 1000:.2f} ms/token")
    if any(result.completion_tokens != args.output_length for result in all_results):
        lengths = sorted({result.completion_tokens for result in all_results})
        print(f"  WARNING: observed output token lengths: {lengths}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
