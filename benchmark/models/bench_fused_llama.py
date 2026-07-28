"""Benchmark decode-only fused RMSNorm + Linear in a local Llama model."""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import transformers
from transformers import LlamaForCausalLM, StaticCache
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.fused_llama import convert_llama_to_fused_rmsnorm_linear
from models.fused_add_llama import (
    convert_llama_to_fused_add_rmsnorm_linear,
)


DEFAULT_MODEL_PATH = "/home/mashaobo/models/llama3-8b/"


@dataclass(frozen=True)
class TimingResult:
    """CUDA timing summary in milliseconds."""

    mean_ms: float
    p50_ms: float
    p95_ms: float


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def _measure(
    function: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
    name: str,
) -> tuple[torch.Tensor, TimingResult]:
    output: torch.Tensor | None = None
    for _ in tqdm(
        range(warmup),
        desc=f"{name} warmup",
        unit="step",
        dynamic_ncols=True,
    ):
        output = function()
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in tqdm(
        range(iterations),
        desc=f"{name} benchmark",
        unit="step",
        dynamic_ncols=True,
    ):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))

    if output is None:
        raise RuntimeError("warmup and iterations cannot both be zero")
    return output, TimingResult(
        mean_ms=statistics.mean(timings),
        p50_ms=_percentile(timings, 0.50),
        p95_ms=_percentile(timings, 0.95),
    )


def _load_model(
    model_path: str,
    device: torch.device,
    attention_implementation: str,
) -> LlamaForCausalLM:
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation=attention_implementation,
    )
    return model.to(device).eval()


def _make_inputs(
    *,
    batch: int,
    prompt_length: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260728)
    prompt = torch.randint(
        0,
        vocab_size,
        (batch, prompt_length),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    next_token = torch.randint(
        0,
        vocab_size,
        (batch, 1),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    return prompt, next_token


def _prefill_static_cache(
    model: LlamaForCausalLM,
    prompt: torch.Tensor,
) -> StaticCache:
    batch, prompt_length = prompt.shape
    cache = StaticCache(
        config=model.config,
        max_batch_size=batch,
        max_cache_len=prompt_length + 1,
        device=prompt.device,
        dtype=torch.float16,
    )
    cache_position = torch.arange(prompt_length, device=prompt.device)
    attention_mask = torch.ones(
        batch,
        prompt_length,
        dtype=torch.long,
        device=prompt.device,
    )
    model(
        input_ids=prompt,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
        logits_to_keep=1,
    )
    return cache


def _build_decode_function(
    model: LlamaForCausalLM,
    cache: StaticCache,
    next_token: torch.Tensor,
    prompt_length: int,
) -> Callable[[], torch.Tensor]:
    batch = next_token.shape[0]
    cache_position = torch.tensor(
        [prompt_length],
        dtype=torch.long,
        device=next_token.device,
    )
    attention_mask = torch.ones(
        batch,
        prompt_length + 1,
        dtype=torch.long,
        device=next_token.device,
    )

    def decode() -> torch.Tensor:
        return model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
            logits_to_keep=1,
        ).logits

    return decode


def _print_timing(name: str, timing: TimingResult, batch: int) -> None:
    tokens_per_second = batch / (timing.mean_ms / 1000.0)
    print(
        f"{name:24s} mean={timing.mean_ms:.4f} ms  "
        f"p50={timing.p50_ms:.4f} ms  "
        f"p95={timing.p95_ms:.4f} ms  "
        f"tokens/s={tokens_per_second:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa"),
        default="sdpa",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 1 <= args.batch <= 8:
        raise ValueError("batch must be in [1, 8]")
    if args.prompt_length <= 0:
        raise ValueError("prompt-length must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"model path does not exist: {model_path}")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}, "
        f"Transformers={transformers.__version__}"
    )
    print(
        f"model={model_path}, batch={args.batch}, "
        f"prompt_length={args.prompt_length}, dtype=float16, "
        f"attention={args.attention_implementation}"
    )

    with torch.inference_mode():
        native_model = _load_model(
            str(model_path),
            device,
            args.attention_implementation,
        )
        prompt, next_token = _make_inputs(
            batch=args.batch,
            prompt_length=args.prompt_length,
            vocab_size=native_model.config.vocab_size,
            device=device,
        )
        native_cache = _prefill_static_cache(native_model, prompt)
        native_decode = _build_decode_function(
            native_model,
            native_cache,
            next_token,
            args.prompt_length,
        )
        torch.cuda.reset_peak_memory_stats(device)
        native_output, native_timing = _measure(
            native_decode,
            args.warmup,
            args.iterations,
            "Transformers Llama",
        )
        reference_logits = native_output.float().cpu()
        native_peak_gib = torch.cuda.max_memory_allocated(device) / 2**30

        del native_decode, native_output, native_cache, native_model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        fused_rms_model = _load_model(
            str(model_path),
            device,
            args.attention_implementation,
        )
        fused_rms_cache = _prefill_static_cache(fused_rms_model, prompt)
        fused_rms_report = convert_llama_to_fused_rmsnorm_linear(
            fused_rms_model
        )
        fused_rms_decode = _build_decode_function(
            fused_rms_model,
            fused_rms_cache,
            next_token,
            args.prompt_length,
        )
        torch.cuda.reset_peak_memory_stats(device)
        fused_rms_output, fused_rms_timing = _measure(
            fused_rms_decode,
            args.warmup,
            args.iterations,
            "Fused RMS Llama",
        )
        fused_rms_peak_gib = (
            torch.cuda.max_memory_allocated(device) / 2**30
        )
        fused_rms_logits = fused_rms_output.float().cpu()

        del (
            fused_rms_decode,
            fused_rms_output,
            fused_rms_cache,
            fused_rms_model,
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        fused_add_model = _load_model(
            str(model_path),
            device,
            args.attention_implementation,
        )
        fused_add_cache = _prefill_static_cache(fused_add_model, prompt)
        fused_add_report = convert_llama_to_fused_add_rmsnorm_linear(
            fused_add_model
        )
        fused_add_decode = _build_decode_function(
            fused_add_model,
            fused_add_cache,
            next_token,
            args.prompt_length,
        )
        torch.cuda.reset_peak_memory_stats(device)
        fused_add_output, fused_add_timing = _measure(
            fused_add_decode,
            args.warmup,
            args.iterations,
            "Fused Add Llama",
        )
        fused_add_peak_gib = (
            torch.cuda.max_memory_allocated(device) / 2**30
        )
        fused_add_logits = fused_add_output.float().cpu()

    fused_rms_error = (fused_rms_logits - reference_logits).abs()
    fused_add_error = (fused_add_logits - reference_logits).abs()
    native_top1 = reference_logits.argmax(dim=-1)
    fused_rms_top1 = fused_rms_logits.argmax(dim=-1)
    fused_add_top1 = fused_add_logits.argmax(dim=-1)

    print(
        f"fused-rms conversion: layers={fused_rms_report.decoder_layers}, "
        f"projections={fused_rms_report.fused_projections}"
    )
    print(
        f"fused-add conversion: layers={fused_add_report.decoder_layers}, "
        f"add projections={fused_add_report.fused_add_projections}, "
        f"rms projections={fused_add_report.fused_rms_projections}"
    )
    _print_timing("Transformers Llama", native_timing, args.batch)
    _print_timing("Fused RMS Llama", fused_rms_timing, args.batch)
    _print_timing("Fused Add Llama", fused_add_timing, args.batch)
    print(
        "fused-rms speedup: "
        f"{native_timing.mean_ms / fused_rms_timing.mean_ms:.3f}x"
    )
    print(
        "fused-add speedup: "
        f"{native_timing.mean_ms / fused_add_timing.mean_ms:.3f}x"
    )
    print(
        "fused-add vs fused-rms: "
        f"{fused_rms_timing.mean_ms / fused_add_timing.mean_ms:.3f}x"
    )
    print(
        f"fused-rms logits max_abs_error={fused_rms_error.max().item():.6g}, "
        f"mean_abs_error={fused_rms_error.mean().item():.6g}, "
        f"top1_match={torch.equal(native_top1, fused_rms_top1)}"
    )
    print(
        f"fused-add logits max_abs_error={fused_add_error.max().item():.6g}, "
        f"mean_abs_error={fused_add_error.mean().item():.6g}, "
        f"top1_match={torch.equal(native_top1, fused_add_top1)}"
    )
    print(
        "peak allocated memory: "
        f"native={native_peak_gib:.3f} GiB, "
        f"fused-rms={fused_rms_peak_gib:.3f} GiB, "
        f"fused-add={fused_add_peak_gib:.3f} GiB"
    )


if __name__ == "__main__":
    main()
