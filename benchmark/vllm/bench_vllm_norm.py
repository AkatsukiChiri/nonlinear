#!/usr/bin/env python3
"""Measure normalization CUDA-kernel time during vLLM inference.

Start the vLLM server with ``VLLM_TORCH_PROFILER_DIR`` set to the same
directory passed through ``--trace-dir``.  This script warms up the server,
profiles only measured requests, and extracts RMSNorm/LayerNorm CUDA events
from the generated PyTorch profiler traces.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

from bench_vllm_inference import (
    make_prompt_token_ids,
    percentile,
    resolve_tokenizer,
    run_batch,
    update_progress,
)


DEFAULT_NORM_PATTERN = r"(?:rms[_ ]?norm|layer[_ ]?norm)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile normalization CUDA kernels in a running vLLM server."
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--input-length", type=int, required=True)
    parser.add_argument("--output-length", type=int, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        required=True,
        help="Directory used to start the server via VLLM_TORCH_PROFILER_DIR.",
    )
    parser.add_argument(
        "--norm-pattern",
        default=DEFAULT_NORM_PATTERN,
        help="Case-insensitive regex matched against CUDA kernel names.",
    )
    args = parser.parse_args()
    for name in ("batch_size", "input_length", "output_length", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    try:
        re.compile(args.norm_pattern, re.IGNORECASE)
    except re.error as error:
        parser.error(f"invalid --norm-pattern: {error}")
    return args


def profiler_url(base_url: str, action: str) -> str:
    """Return the unversioned vLLM profiler endpoint URL."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return f"{root}/{action}"


def post_empty(url: str, timeout: float, api_key: str | None) -> None:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url, data=b"{}", headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        hint = ""
        if error.code == 404:
            hint = " Start vLLM with VLLM_TORCH_PROFILER_DIR set."
        raise RuntimeError(
            f"POST {url} failed ({error.code}): {detail}.{hint}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Cannot connect to {url}: {error.reason}") from error


TraceSnapshot = dict[Path, tuple[int, int]]


def trace_files(trace_dir: Path) -> TraceSnapshot:
    if not trace_dir.is_dir():
        return {}
    return {
        path.resolve(): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in trace_dir.iterdir()
        if path.is_file()
        and (path.name.endswith(".json") or path.name.endswith(".json.gz"))
    }


def wait_for_traces(
    trace_dir: Path, previous_files: TraceSnapshot, timeout: float
) -> list[Path]:
    deadline = time.monotonic() + min(timeout, 60.0)
    last_candidates: TraceSnapshot = {}
    while time.monotonic() < deadline:
        current_files = trace_files(trace_dir)
        candidates = {
            path: stat
            for path, stat in current_files.items()
            if previous_files.get(path) != stat and stat[1] > 0
        }
        # Require one unchanged poll so a large gzip trace is not parsed while
        # tensorboard_trace_handler is still writing it.
        if candidates and candidates == last_candidates:
            return sorted(candidates)
        last_candidates = candidates
        time.sleep(0.2)
    raise RuntimeError(
        f"No new profiler trace appeared in {trace_dir}. The client and server "
        "must see the same trace directory."
    )


def cuda_kernel_events(path: Path) -> Iterable[tuple[str, float]]:
    """Stream CUDA kernel names/durations without loading a huge trace.

    A trace can decompress to several GB and can contain extremely long JSON
    lines, so loading either the document or complete lines is impractical.
    """
    event_pattern = re.compile(
        rb'"ph"\s*:\s*"X"\s*,\s*"cat"\s*:\s*"kernel"\s*,'
        rb'\s*"name"\s*:\s*("(?:[^"\\]|\\.)*").*?'
        rb'"dur"\s*:\s*([0-9.eE+-]+)',
        re.DOTALL,
    )
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as trace_file:
        buffer = b""
        while chunk := trace_file.read(1024 * 1024):
            buffer += chunk
            last_match_end = 0
            for match in event_pattern.finditer(buffer):
                name = json.loads(match.group(1))
                yield name, float(match.group(2))
                last_match_end = match.end()
            if last_match_end:
                buffer = buffer[last_match_end:]
            elif len(buffer) > 2 * 1024 * 1024:
                buffer = buffer[-1024 * 1024:]


def print_trace_results(
    paths: list[Path], norm_pattern: re.Pattern[str], wall_seconds: float
) -> None:
    norm_durations_us: list[float] = []
    all_kernel_duration_us = 0.0
    by_name: dict[str, list[float]] = {}
    print("\nProfiler traces")
    for path in paths:
        kernel_us = 0.0
        matched_us = 0.0
        matched_count = 0
        for name, duration in cuda_kernel_events(path):
            kernel_us += duration
            if norm_pattern.search(name):
                matched_us += duration
                matched_count += 1
                norm_durations_us.append(duration)
                by_name.setdefault(name, []).append(duration)
        all_kernel_duration_us += kernel_us
        print(
            f"  {path.name}: norm={matched_us / 1000:.3f} ms, "
            f"calls={matched_count}, all CUDA kernels={kernel_us / 1000:.3f} ms"
        )

    if not norm_durations_us:
        raise RuntimeError(
            "No normalization CUDA kernels matched the trace. Inspect the trace "
            "kernel names and adjust --norm-pattern."
        )

    norm_total_us = sum(norm_durations_us)
    print("\nNorm results")
    print(f"  CUDA calls:                 {len(norm_durations_us)}")
    print(f"  Total CUDA kernel time:     {norm_total_us / 1000:.3f} ms")
    mean_norm_us = norm_total_us / len(norm_durations_us)
    print(f"  Mean CUDA kernel time:      {mean_norm_us:.3f} us")
    print(
        f"  CUDA kernel time p50/p95:   "
        f"{percentile(norm_durations_us, 0.50):.3f} / "
        f"{percentile(norm_durations_us, 0.95):.3f} us"
    )
    cuda_share = norm_total_us / all_kernel_duration_us * 100
    wall_share = norm_total_us / 1e6 / wall_seconds * 100
    print(f"  Share of all CUDA kernels:  {cuda_share:.2f}%")
    print(f"  Sum / benchmark wall time:  {wall_share:.2f}%")
    print("\nMatched CUDA kernels")
    for name, durations in sorted(
        by_name.items(), key=lambda item: sum(item[1]), reverse=True
    ):
        print(
            f"  {sum(durations) / 1000:9.3f} ms  {len(durations):7d} calls  {name}"
        )


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required; run this script in the nonl environment."
        ) from error

    tokenizer_path = resolve_tokenizer(args)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=True
    )
    prompt_token_ids = make_prompt_token_ids(tokenizer, args.input_length)
    print(
        f"Model={args.model} batch_size={args.batch_size} "
        f"input_length={len(prompt_token_ids)} output_length={args.output_length}"
    )

    for index in range(args.warmup):
        run_batch(args, prompt_token_ids, tokenizer)
        update_progress(index + 1, args.warmup, "warm-up")

    existing_traces = trace_files(args.trace_dir)
    post_empty(
        profiler_url(args.base_url, "start_profile"), args.timeout, args.api_key
    )
    started_at = time.perf_counter()
    try:
        for index in range(args.iterations):
            run_batch(args, prompt_token_ids, tokenizer)
            update_progress(index + 1, args.iterations, "profiling")
    finally:
        post_empty(
            profiler_url(args.base_url, "stop_profile"), args.timeout, args.api_key
        )
    wall_seconds = time.perf_counter() - started_at

    paths = wait_for_traces(args.trace_dir, existing_traces, args.timeout)
    print(f"\nMeasured wall time: {wall_seconds:.3f} s")
    print_trace_results(
        paths, re.compile(args.norm_pattern, re.IGNORECASE), wall_seconds
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
