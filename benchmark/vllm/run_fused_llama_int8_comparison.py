"""Run native and fused vLLM benchmarks in isolated subprocesses."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


RESULT_PATTERN = re.compile(r"RESULT_JSON=(\{.*\})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--with-pytorch", action="store_true")
    return parser.parse_args()


def run(implementation: str, args: argparse.Namespace) -> dict[str, object]:
    benchmark = Path(__file__).with_name("bench_fused_llama_int8.py")
    command = [
        sys.executable,
        str(benchmark),
        "--implementation",
        implementation,
        "--batch",
        str(args.batch),
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--output-tokens",
        str(args.output_tokens),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    environment = os.environ.copy()
    environment.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    print(completed.stdout, end="")
    match = RESULT_PATTERN.search(completed.stdout)
    if match is None:
        raise RuntimeError(f"{implementation} benchmark returned no result")
    return json.loads(match.group(1))


def run_pytorch(args: argparse.Namespace) -> dict[str, object]:
    benchmark = Path(__file__).with_name("bench_pytorch_dequantized_llama.py")
    command = [
        sys.executable,
        str(benchmark),
        "--batch",
        str(args.batch),
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--output-tokens",
        str(args.output_tokens),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--dtype",
        "bfloat16",
    ]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    print(completed.stdout, end="")
    match = RESULT_PATTERN.search(completed.stdout)
    if match is None:
        raise RuntimeError("PyTorch benchmark returned no result")
    return json.loads(match.group(1))


def token_agreement(
    reference: list[list[int]], candidate: list[list[int]]
) -> tuple[int, int, int]:
    """Return matching positions, compared positions, and common-prefix size."""
    matching = 0
    compared = 0
    common_prefix = 0
    prefix_open = True
    for reference_row, candidate_row in zip(reference, candidate):
        for reference_token, candidate_token in zip(reference_row, candidate_row):
            compared += 1
            is_match = reference_token == candidate_token
            matching += int(is_match)
            if prefix_open and is_match:
                common_prefix += 1
            else:
                prefix_open = False
    return matching, compared, common_prefix


def main() -> None:
    args = parse_args()
    native = run("vllm", args)
    fused = run("fused", args)
    pytorch = run_pytorch(args) if args.with_pytorch else None
    if native["last_output_token_ids"] != fused["last_output_token_ids"]:
        raise AssertionError("native and fused greedy token IDs differ")
    native_ms = float(native["mean_ms"])
    fused_ms = float(fused["mean_ms"])
    native_prefill = float(native["prefill_mean_ms"])
    fused_prefill = float(fused["prefill_mean_ms"])
    native_tpot = float(native["decode_tpot_ms"])
    fused_tpot = float(fused["decode_tpot_ms"])
    print("\nComparison")
    print(
        f"prefill: native={native_prefill:.3f} ms, "
        f"fused={fused_prefill:.3f} ms, "
        f"speedup={native_prefill / fused_prefill:.3f}x"
    )
    print(
        f"decode TPOT: native={native_tpot:.3f} ms, "
        f"fused={fused_tpot:.3f} ms, speedup={native_tpot / fused_tpot:.3f}x"
    )
    print(
        f"end-to-end: native={native_ms:.3f} ms, fused={fused_ms:.3f} ms, "
        f"speedup={native_ms / fused_ms:.3f}x"
    )
    print("greedy token IDs: identical")
    if pytorch is not None:
        matching, compared, common_prefix = token_agreement(
            fused["last_output_token_ids"],
            pytorch["last_output_token_ids"],
        )
        pytorch_prefill = float(pytorch["prefill_mean_ms"])
        pytorch_tpot = float(pytorch["decode_tpot_ms"])
        print(
            f"PyTorch BF16: prefill={pytorch_prefill:.3f} ms, "
            f"decode TPOT={pytorch_tpot:.3f} ms"
        )
        print(
            f"fused speedup vs PyTorch: prefill="
            f"{pytorch_prefill / fused_prefill:.3f}x, "
            f"decode={pytorch_tpot / fused_tpot:.3f}x"
        )
        print(
            "PyTorch BF16 vs W8A8 greedy agreement: "
            f"{matching}/{compared} tokens, common prefix={common_prefix}; "
            "reported for diagnostics only because quantization can change "
            "the greedy path"
        )


if __name__ == "__main__":
    main()
