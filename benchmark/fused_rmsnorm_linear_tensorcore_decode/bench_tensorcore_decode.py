"""Compare batched decode RMSNorm projections on real Llama W8A8 weights."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear_tensorcore_decode import (
    FusedRMSNormLinearTensorCoreDecode,
    FusedRMSNormLinearTensorCoreInt8WeightDecode,
)
from layers.fused_rmsnorm_linear_int8_prefill import (
    FusedRMSNormLinearInt8Prefill,
)
from layers.fused_rmsnorm_linear_weight_only_int8_dequant import (
    FusedRMSNormLinearInt8Decode,
)


DEFAULT_MODEL = Path(
    "/home/mashaobo/models/Meta-Llama-3.1-8B-Instruct-W8A8"
)
PROJECTIONS = {
    "q": (("self_attn.q_proj",), "input_layernorm"),
    "qkv": (
        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        "input_layernorm",
    ),
    "gate": (("mlp.gate_proj",), "post_attention_layernorm"),
    "gate_up": (
        ("mlp.gate_proj", "mlp.up_proj"),
        "post_attention_layernorm",
    ),
}


class LoadedCompressedLinear(nn.Module):
    def __init__(
        self, weight: torch.Tensor, scale: torch.Tensor, input_size: int
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = nn.Parameter(scale, requires_grad=False)
        self.input_size = input_size
        self.logical_widths: list[int] = []
        self.register_parameter("bias", None)


class VllmKernelLayer(nn.Module):
    def __init__(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = nn.Parameter(scale, requires_grad=False)
        self.register_parameter("input_scale", None)
        self.register_parameter("input_zero_point", None)
        self.register_parameter("azp_adj", None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--projection", choices=PROJECTIONS, default="qkv")
    parser.add_argument("--batch", type=int, choices=(1, 4, 8, 16), default=4)
    parser.add_argument(
        "--tiles-per-cta", type=int, choices=(1, 2, 4, 8, 16, 32), default=4
    )
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--eps", type=float, default=1e-5)
    return parser.parse_args()


def load_projection(
    model: Path, layer_index: int, projection: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projection_names, norm_name = PROJECTIONS[projection]
    prefix = f"model.layers.{layer_index}"
    weight_names = [f"{prefix}.{name}.weight" for name in projection_names]
    scale_names = [f"{prefix}.{name}.weight_scale" for name in projection_names]
    norm_name = f"{prefix}.{norm_name}.weight"
    with (model / "model.safetensors.index.json").open(encoding="utf-8") as file:
        weight_map = json.load(file)["weight_map"]
    tensors: dict[str, torch.Tensor] = {}
    requested = weight_names + scale_names + [norm_name]
    for shard in sorted({weight_map[name] for name in requested}):
        names = [name for name in requested if weight_map[name] == shard]
        with safe_open(model / shard, framework="pt", device="cpu") as checkpoint:
            tensors.update({name: checkpoint.get_tensor(name) for name in names})
    weight = torch.cat([tensors[name] for name in weight_names], dim=0)
    scale = torch.cat([tensors[name].reshape(-1) for name in scale_names])
    return weight, scale, tensors[norm_name]


def measure(function: Callable[[], object], warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iterations):
        start.record()
        function()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def print_result(name: str, times: list[float]) -> float:
    mean = statistics.mean(times)
    ordered = sorted(times)
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[round((len(ordered) - 1) * 0.95)]
    print(f"{name:30s} mean={mean:.4f} ms p50={p50:.4f} ms p95={p95:.4f} ms")
    return mean


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm
    from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
        CutlassScaledMMLinearKernel,
        ScaledMMLinearLayerConfig,
    )

    checkpoint_weight, checkpoint_scale, norm_weight = load_projection(
        args.model, args.layer_index, args.projection
    )
    checkpoint_weight = checkpoint_weight.cuda()
    checkpoint_scale = checkpoint_scale.cuda().float()
    norm_weight = norm_weight.to(device="cuda", dtype=dtype)
    hidden_dim = checkpoint_weight.shape[1]
    linear = LoadedCompressedLinear(
        checkpoint_weight.t(), checkpoint_scale, hidden_dim
    ).cuda()
    rmsnorm = nn.RMSNorm(hidden_dim, eps=args.eps, device="cuda", dtype=dtype)
    with torch.no_grad():
        rmsnorm.weight.copy_(norm_weight)
    old_fused_class = (
        FusedRMSNormLinearInt8Decode
        if args.batch <= 8
        else FusedRMSNormLinearInt8Prefill
    )
    old_fused = old_fused_class.from_quantized_linear(rmsnorm, linear).eval()
    tensorcore = FusedRMSNormLinearTensorCoreDecode(
        old_fused.quantized_weight,
        old_fused.weight_scale,
        old_fused.bias,
        old_fused.eps,
        dtype,
        args.tiles_per_cta,
    ).eval()
    int8_weight_tensorcore = FusedRMSNormLinearTensorCoreInt8WeightDecode(
        old_fused.quantized_weight,
        old_fused.weight_scale,
        old_fused.bias,
        old_fused.eps,
        dtype,
        args.tiles_per_cta,
    ).eval()
    x = torch.randn(
        args.batch, 1, hidden_dim, device="cuda", dtype=dtype
    ) / 4
    residual = torch.randn_like(x) / 4
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
    vllm_samples = args.warmup + args.iterations + 1
    vllm_x = x.reshape(args.batch, hidden_dim).repeat(vllm_samples, 1)
    vllm_residual_pool = residual.reshape(
        args.batch, hidden_dim
    ).repeat(vllm_samples, 1)
    vllm_index = 0
    ctas, n_tiles, _, _ = tensorcore.weight_fp16.shape
    dense_sentinel_weight = tensorcore.weight_fp16.permute(
        0, 2, 1, 3
    ).reshape(ctas, hidden_dim, n_tiles * 16)
    dense_sentinel_weight = dense_sentinel_weight.permute(
        1, 0, 2
    ).reshape(hidden_dim, -1).contiguous()
    residual_fp16 = (x + residual).to(torch.float16).reshape(
        args.batch, hidden_dim
    )
    augmented_input = torch.cat(
        [residual_fp16, residual_fp16.square()], dim=0
    ).contiguous()

    def old_forward() -> tuple[torch.Tensor, torch.Tensor]:
        result = old_fused(x, residual)
        assert isinstance(result, tuple)
        return result

    def vllm_forward() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal vllm_index
        normalized, updated_residual = vllm_norm(
            vllm_x[
                vllm_index * args.batch : (vllm_index + 1) * args.batch
            ],
            vllm_residual_pool[
                vllm_index * args.batch : (vllm_index + 1) * args.batch
            ],
        )
        vllm_index += 1
        projected = vllm_kernel.apply_weights(vllm_linear, normalized)
        return projected, updated_residual

    def tensorcore_forward() -> tuple[torch.Tensor, torch.Tensor]:
        result = tensorcore(x, residual)
        assert isinstance(result, tuple)
        return result

    def int8_weight_tensorcore_forward() -> tuple[torch.Tensor, torch.Tensor]:
        result = int8_weight_tensorcore(x, residual)
        assert isinstance(result, tuple)
        return result

    def vendor_tensorcore_gemm() -> torch.Tensor:
        return torch.mm(augmented_input, dense_sentinel_weight)

    with torch.inference_mode():
        vllm_output, vllm_residual = vllm_forward()
        old_output, old_residual = old_forward()
        new_output, new_residual = tensorcore_forward()
        int8_weight_output, int8_weight_residual = (
            int8_weight_tensorcore_forward()
        )
        cosine = torch.nn.functional.cosine_similarity(
            old_output.float().reshape(-1), new_output.float().reshape(-1), dim=0
        ).item()
        relative_rmse = (
            (old_output.float() - new_output.float()).square().mean().sqrt()
            / old_output.float().square().mean().sqrt()
        ).item()
        torch.testing.assert_close(new_residual, old_residual, atol=0, rtol=0)
        torch.testing.assert_close(
            int8_weight_residual, old_residual, atol=0, rtol=0
        )
        torch.testing.assert_close(
            int8_weight_output,
            new_output,
            atol=5e-3 if dtype == torch.float16 else 3e-2,
            rtol=5e-3 if dtype == torch.float16 else 3e-2,
        )
        torch.testing.assert_close(
            vllm_residual, old_residual.reshape(args.batch, hidden_dim)
        )
        if cosine < 0.999 or relative_rmse > 0.03:
            raise AssertionError(
                f"output mismatch: cosine={cosine}, relative_rmse={relative_rmse}"
            )
        vllm_times = measure(vllm_forward, args.warmup, args.iterations)
        old_times = measure(old_forward, args.warmup, args.iterations)
        tensorcore_times = measure(
            tensorcore_forward, args.warmup, args.iterations
        )
        int8_weight_tensorcore_times = measure(
            int8_weight_tensorcore_forward, args.warmup, args.iterations
        )
        vendor_times = measure(
            vendor_tensorcore_gemm, args.warmup, args.iterations
        )

    print(
        f"projection={args.projection} dtype={args.dtype} "
        f"shape=[{args.batch},{hidden_dim}]x"
        f"[{hidden_dim},{checkpoint_weight.shape[0]}] "
        f"tiles_per_cta={args.tiles_per_cta}"
    )
    vllm_mean = print_result("vLLM W8A8 baseline", vllm_times)
    old_name = (
        "Previous CUDA-core fused"
        if args.batch <= 8
        else "Previous INT8 prefill fallback"
    )
    old_mean = print_result(old_name, old_times)
    tensorcore_mean = print_result("New Tensor Core fused", tensorcore_times)
    int8_weight_tensorcore_mean = print_result(
        "INT8-weight FP16 TC fused", int8_weight_tensorcore_times
    )
    print_result("Vendor [2B,K] Tensor Core GEMM", vendor_times)
    print(f"Tensor Core speedup={old_mean / tensorcore_mean:.3f}x")
    print(
        "INT8-weight speedup vs FP16-weight="
        f"{tensorcore_mean / int8_weight_tensorcore_mean:.3f}x, "
        "vs previous="
        f"{old_mean / int8_weight_tensorcore_mean:.3f}x, "
        "vs vLLM="
        f"{vllm_mean / int8_weight_tensorcore_mean:.3f}x"
    )
    print(
        f"Previous speedup vs vLLM={vllm_mean / old_mean:.3f}x, "
        f"Tensor Core speedup vs vLLM={vllm_mean / tensorcore_mean:.3f}x"
    )
    print(f"cosine={cosine:.7f} relative_rmse={relative_rmse:.7f}")


if __name__ == "__main__":
    main()
