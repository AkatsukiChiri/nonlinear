"""Benchmark one W8A8 Llama layer against PyTorch and vLLM.

Only steady-state forward latency is measured. Checkpoint loading, weight
dequantization, RMSNorm folding, and INT8 requantization happen before timing.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear_int8 import (
    FusedRMSNormLinearInt8Decode,
    FusedRMSNormLinearInt8Prefill,
)


DEFAULT_MODEL_PATH = Path(
    "/home/mashaobo/models/Meta-Llama-3.1-8B-Instruct-W8A8"
)
PROJECTIONS = {
    "qkv": (
        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        "input_layernorm",
        (4096, 1024, 1024),
    ),
    "gate_up": (
        ("mlp.gate_proj", "mlp.up_proj"),
        "post_attention_layernorm",
        (14336, 14336),
    ),
}


class LoadedCompressedLinear(nn.Module):
    """Minimal loaded vLLM compressed Linear interface."""

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_size: int,
        logical_widths: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
        self.input_size = input_size
        self.logical_widths = list(logical_widths)
        self.register_parameter("bias", None)


class VllmKernelLayer(nn.Module):
    """Parameters consumed by vLLM's CutlassScaledMMLinearKernel."""

    def __init__(
        self, weight: torch.Tensor, weight_scale: torch.Tensor
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
        self.register_parameter("input_scale", None)
        self.register_parameter("input_zero_point", None)
        self.register_parameter("azp_adj", None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one real Llama-3.1 W8A8 merged projection."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--projection", choices=PROJECTIONS, default="qkv")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "both"),
        default="both",
    )
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    return parser.parse_args()


def load_projection(
    model_path: Path,
    layer_index: int,
    projection: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    projection_names, norm_name, logical_widths = PROJECTIONS[projection]
    prefix = f"model.layers.{layer_index}"
    names = [f"{prefix}.{name}.weight" for name in projection_names]
    scale_names = [
        f"{prefix}.{name}.weight_scale" for name in projection_names
    ]
    norm_weight_name = f"{prefix}.{norm_name}.weight"

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"checkpoint index not found: {index_path}")
    with index_path.open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file)["weight_map"]
    requested = names + scale_names + [norm_weight_name]
    tensors: dict[str, torch.Tensor] = {}
    by_shard: dict[str, list[str]] = {}
    for name in requested:
        by_shard.setdefault(weight_map[name], []).append(name)
    for shard_name, shard_tensors in by_shard.items():
        with safe_open(
            model_path / shard_name, framework="pt", device="cpu"
        ) as checkpoint:
            tensors.update(
                {name: checkpoint.get_tensor(name) for name in shard_tensors}
            )

    weight = torch.cat([tensors[name] for name in names], dim=0)
    scale = torch.cat(
        [tensors[name].reshape(-1) for name in scale_names], dim=0
    ).float()
    return weight, scale, tensors[norm_weight_name], logical_widths


def measure(
    function: Callable[[], torch.Tensor], warmup: int, iterations: int
) -> tuple[torch.Tensor, list[float]]:
    for _ in range(warmup):
        output = function()
    torch.cuda.synchronize()
    timings = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return output, timings


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def report(name: str, timings: list[float], operations: int) -> float:
    mean_ms = statistics.mean(timings)
    tflops = operations / (mean_ms / 1000.0) / 1e12
    print(
        f"{name:32s} mean={mean_ms:.4f} ms  "
        f"p50={percentile(timings, 0.50):.4f} ms  "
        f"p95={percentile(timings, 0.95):.4f} ms  "
        f"effective_TFLOPS={tflops:.3f}"
    )
    return mean_ms


def benchmark_dtype(
    args: argparse.Namespace,
    checkpoint_weight_cpu: torch.Tensor,
    checkpoint_scale_cpu: torch.Tensor,
    norm_weight_cpu: torch.Tensor,
    logical_widths: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm
    from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
        CutlassScaledMMLinearKernel,
        ScaledMMLinearLayerConfig,
    )

    checkpoint_weight = checkpoint_weight_cpu.cuda()
    checkpoint_scale = checkpoint_scale_cpu.cuda()
    norm_weight = norm_weight_cpu.to(device="cuda", dtype=dtype)
    hidden_dim = checkpoint_weight.shape[1]
    output_dim = checkpoint_weight.shape[0]

    compressed_linear = LoadedCompressedLinear(
        checkpoint_weight.t(), checkpoint_scale, hidden_dim, logical_widths
    ).cuda()
    rmsnorm = nn.RMSNorm(
        hidden_dim, eps=args.eps, device="cuda", dtype=dtype
    )
    with torch.no_grad():
        rmsnorm.weight.copy_(norm_weight)
    fused_type = (
        FusedRMSNormLinearInt8Decode
        if args.mode == "decode"
        else FusedRMSNormLinearInt8Prefill
    )
    fused_layer = fused_type.from_quantized_linear(
        rmsnorm, compressed_linear, chunk_size=args.chunk_size
    ).eval()

    vllm_norm = VllmRMSNorm(hidden_dim, eps=args.eps, dtype=dtype).cuda().eval()
    with torch.no_grad():
        vllm_norm.weight.copy_(norm_weight)
    vllm_linear = VllmKernelLayer(
        checkpoint_weight.t(), checkpoint_scale.reshape(-1, 1)
    ).cuda()
    vllm_kernel = CutlassScaledMMLinearKernel(
        c=ScaledMMLinearLayerConfig(
            is_channelwise=True,
            is_static_input_scheme=False,
            input_symmetric=True,
        ),
        w_q_param_name="weight",
        w_s_param_name="weight_scale",
        i_s_param_name="input_scale",
        i_zp_param_name="input_zero_point",
        azp_adj_param_name="azp_adj",
    )

    pytorch_weight = (
        checkpoint_weight.float() * checkpoint_scale[:, None]
    ).to(dtype)
    # vLLM model execution flattens [Batch, SeqLen, HiddenDim] to
    # [NumTokens, HiddenDim] before entering RMSNorm and Linear layers.
    x_flat = torch.randn(
        args.batch * args.seq_len,
        hidden_dim,
        device="cuda",
        dtype=dtype,
    ) / 4
    x_fused = x_flat.reshape(args.batch, args.seq_len, hidden_dim)

    def fused_forward() -> torch.Tensor:
        return fused_layer(x_fused).reshape(-1, output_dim)

    def vllm_forward() -> torch.Tensor:
        normalized = vllm_norm(x_flat)
        return vllm_kernel.apply_weights(vllm_linear, normalized)

    normalized_input = vllm_norm(x_flat)

    def vllm_norm_forward() -> torch.Tensor:
        return vllm_norm(x_flat)

    def vllm_linear_forward() -> torch.Tensor:
        return vllm_kernel.apply_weights(vllm_linear, normalized_input)

    def pytorch_forward() -> torch.Tensor:
        normalized = F.rms_norm(
            x_flat, (hidden_dim,), norm_weight, args.eps
        )
        return F.linear(normalized, pytorch_weight)

    with torch.inference_mode():
        fused_check = fused_forward().float()
        vllm_check = vllm_forward().float()
        pytorch_check = pytorch_forward().float()
        fused_cosine = F.cosine_similarity(
            fused_check.reshape(-1), pytorch_check.reshape(-1), dim=0
        ).item()
        vllm_cosine = F.cosine_similarity(
            vllm_check.reshape(-1), pytorch_check.reshape(-1), dim=0
        ).item()
        reference_rms = pytorch_check.square().mean().sqrt()
        fused_relative_rmse = (
            (fused_check - pytorch_check).square().mean().sqrt() / reference_rms
        ).item()
        vllm_relative_rmse = (
            (vllm_check - pytorch_check).square().mean().sqrt() / reference_rms
        ).item()
        if (
            min(fused_cosine, vllm_cosine) < 0.99
            or max(fused_relative_rmse, vllm_relative_rmse) > 0.05
        ):
            raise AssertionError(
                "output validation failed: "
                f"fused cosine={fused_cosine:.6f}, "
                f"vLLM cosine={vllm_cosine:.6f}, "
                f"fused relative RMSE={fused_relative_rmse:.6f}, "
                f"vLLM relative RMSE={vllm_relative_rmse:.6f}"
            )
        _, fused_times = measure(
            fused_forward, args.warmup, args.iterations
        )
        _, vllm_times = measure(vllm_forward, args.warmup, args.iterations)
        _, vllm_norm_times = measure(
            vllm_norm_forward, args.warmup, args.iterations
        )
        _, vllm_linear_times = measure(
            vllm_linear_forward, args.warmup, args.iterations
        )
        _, pytorch_times = measure(
            pytorch_forward, args.warmup, args.iterations
        )

    tokens = args.batch * args.seq_len
    operations = 2 * tokens * hidden_dim * output_dim
    dtype_name = str(dtype).removeprefix("torch.")
    print(f"\n[{dtype_name}]")
    fused_mean = report("Our fused W8A8", fused_times, operations)
    vllm_mean = report("vLLM RMSNorm + W8A8", vllm_times, operations)
    vllm_norm_mean = report("  vLLM RMSNorm only", vllm_norm_times, 0)
    vllm_linear_mean = report(
        "  vLLM W8A8 Linear only", vllm_linear_times, operations
    )
    pytorch_mean = report("PyTorch RMSNorm + Linear", pytorch_times, operations)
    print(
        f"Our speedup vs vLLM={vllm_mean / fused_mean:.3f}x, "
        f"vs PyTorch={pytorch_mean / fused_mean:.3f}x"
    )
    print(
        f"vLLM component sum={vllm_norm_mean + vllm_linear_mean:.4f} ms, "
        f"RMSNorm share={vllm_norm_mean / vllm_mean * 100:.1f}%"
    )
    print(
        f"Output cosine vs PyTorch: fused={fused_cosine:.6f}, "
        f"vLLM={vllm_cosine:.6f}"
    )
    print(
        f"Relative RMSE vs PyTorch: fused={fused_relative_rmse:.6f}, "
        f"vLLM={vllm_relative_rmse:.6f}"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch <= 0 or args.seq_len <= 0:
        raise ValueError("batch and seq-len must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations positive")
    if args.mode == "decode" and (args.seq_len != 1 or args.batch > 8):
        raise ValueError("decode requires seq-len=1 and batch <= 8")

    torch.manual_seed(20260806)
    weight, scale, norm_weight, logical_widths = load_projection(
        args.model, args.layer_index, args.projection
    )
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}")
    print(
        f"model={args.model}, layer={args.layer_index}, "
        f"projection={args.projection}, mode={args.mode}"
    )
    print(
        f"B={args.batch}, S={args.seq_len}, M={args.batch * args.seq_len}, "
        f"K={weight.shape[1]}, N={weight.shape[0]}, "
        f"logical_widths={logical_widths}"
    )
    dtypes = (
        (torch.float16, torch.bfloat16)
        if args.dtype == "both"
        else (getattr(torch, args.dtype),)
    )
    for dtype in dtypes:
        benchmark_dtype(
            args, weight, scale, norm_weight, logical_widths, dtype
        )


if __name__ == "__main__":
    main()
