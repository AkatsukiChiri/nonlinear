"""Compare global-sentinel AWQ-Marlin decode with native vLLM W4A16."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear_w4a16_global_sentinel import (
    FusedRMSNormLinearW4A16GlobalSentinel,
)
from layers.fused_rmsnorm_linear_w4a16_global_sentinel.cuda import (
    finalize_out,
    preprocess_out,
)


DEFAULT_MODEL = Path(
    "/home/mashaobo/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
)
PROJECTIONS = {
    "q": ("self_attn.q_proj", "input_layernorm"),
    "gate": ("mlp.gate_proj", "post_attention_layernorm"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--projection", choices=PROJECTIONS, default="q")
    parser.add_argument("--batch", type=int, choices=(1, 4, 8), default=4)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--tensorcore", action="store_true")
    return parser.parse_args()


def load_tensors(
    model: Path, layer_index: int, projection: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    projection_name, norm_name = PROJECTIONS[projection]
    prefix = f"model.layers.{layer_index}"
    names = [
        f"{prefix}.{projection_name}.qweight",
        f"{prefix}.{projection_name}.qzeros",
        f"{prefix}.{projection_name}.scales",
        f"{prefix}.{norm_name}.weight",
    ]
    with (model / "model.safetensors.index.json").open(encoding="utf-8") as file:
        weight_map = json.load(file)["weight_map"]
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted({weight_map[name] for name in names}):
        with safe_open(model / shard, framework="pt", device="cpu") as checkpoint:
            for name in names:
                if weight_map[name] == shard:
                    tensors[name] = checkpoint.get_tensor(name)
    return tuple(tensors[name] for name in names)  # type: ignore[return-value]


def make_native_marlin(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, ...]:
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        awq_to_marlin_zero_points,
        marlin_make_empty_g_idx,
        marlin_make_workspace,
        marlin_permute_scales,
    )

    hidden_dim = qweight.shape[0]
    out_dim = scales.shape[1]
    marlin_weight = ops.awq_marlin_repack(qweight, hidden_dim, out_dim, 4)
    marlin_scales = marlin_permute_scales(
        scales, hidden_dim, out_dim, group_size
    )
    marlin_zeros = awq_to_marlin_zero_points(
        qzeros, hidden_dim // group_size, out_dim, 4
    )
    empty = marlin_make_empty_g_idx(qweight.device)
    return (
        marlin_weight,
        marlin_scales,
        marlin_zeros,
        empty,
        marlin_make_empty_g_idx(qweight.device),
        marlin_make_workspace(out_dim, qweight.device),
    )


def measure(
    function: Callable[[], object], warmup: int, iterations: int
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iterations):
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def summarize(name: str, samples: list[float]) -> float:
    mean = statistics.mean(samples)
    ordered = sorted(samples)
    print(
        f"{name:34s} mean={mean:.4f} ms "
        f"p50={ordered[len(ordered) // 2]:.4f} ms "
        f"p95={ordered[round((len(ordered) - 1) * 0.95)]:.4f} ms"
    )
    return mean


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    qweight, qzeros, scales, gamma = load_tensors(
        args.model, args.layer_index, args.projection
    )
    qweight = qweight.cuda()
    qzeros = qzeros.cuda()
    scales = scales.cuda()
    gamma = gamma.cuda()
    hidden_dim = qweight.shape[0]
    out_dim = scales.shape[1]
    group_size = hidden_dim // scales.shape[0]
    proposed = FusedRMSNormLinearW4A16GlobalSentinel(
        qweight, qzeros, scales, gamma, None, args.eps, group_size,
        use_tensorcore=args.tensorcore,
    ).eval()
    native = make_native_marlin(qweight, qzeros, scales, group_size)

    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_awq_marlin_linear,
    )
    from vllm.scalar_type import scalar_types

    norm = VllmRMSNorm(
        hidden_dim, eps=args.eps, dtype=torch.float16
    ).cuda().eval()
    with torch.no_grad():
        norm.weight.copy_(gamma)
    x = torch.randn(
        args.batch, hidden_dim, device="cuda", dtype=torch.float16
    ) / 4
    residual = torch.randn_like(x) / 4
    sample_count = args.warmup + args.iterations + 1
    native_x = x.repeat(sample_count, 1)
    native_residual = residual.repeat(sample_count, 1)
    native_index = 0

    def native_forward() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal native_index
        begin = native_index * args.batch
        end = begin + args.batch
        normalized, updated_residual = norm(
            native_x[begin:end], native_residual[begin:end]
        )
        native_index += 1
        output = apply_awq_marlin_linear(
            input=normalized,
            weight=native[0],
            weight_scale=native[1],
            weight_zp=native[2],
            g_idx=native[3],
            g_idx_sort_indices=native[4],
            workspace=native[5],
            quant_type=scalar_types.uint4,
            output_size_per_partition=out_dim,
            input_size_per_partition=hidden_dim,
        )
        return output, updated_residual

    def proposed_forward() -> tuple[torch.Tensor, torch.Tensor]:
        result = proposed(
            x.reshape(args.batch, 1, hidden_dim),
            residual.reshape(args.batch, 1, hidden_dim),
        )
        assert isinstance(result, tuple)
        return result[0].reshape(args.batch, out_dim), result[1].reshape(
            args.batch, hidden_dim
        )

    with torch.inference_mode():
        native_output, native_residual_output = native_forward()
        proposed_output, proposed_residual_output = proposed_forward()
        torch.testing.assert_close(
            proposed_residual_output, native_residual_output, atol=0, rtol=0
        )
        cosine = torch.nn.functional.cosine_similarity(
            proposed_output.float().reshape(-1),
            native_output.float().reshape(-1),
            dim=0,
        ).item()
        relative_rmse = (
            (proposed_output.float() - native_output.float())
            .square()
            .mean()
            .sqrt()
            / native_output.float().square().mean().sqrt()
        ).item()
        if cosine < 0.999 or relative_rmse > 0.04:
            raise AssertionError(
                f"output mismatch: cosine={cosine}, relative_rmse={relative_rmse}"
            )
        native_times = measure(native_forward, args.warmup, args.iterations)
        proposed_times = measure(
            proposed_forward, args.warmup, args.iterations
        )

        # Isolate GPU stages after all persistent buffers and extensions have
        # been initialized. These figures intentionally do not sum exactly to
        # the end-to-end latency because separate measurements pay separate
        # launch/measurement effects.
        augmented, residual_sum, output = proposed._buffers_by_batch[args.batch]

        def preprocess_stage() -> None:
            preprocess_out(x.reshape(args.batch, 1, hidden_dim),
                           residual.reshape(args.batch, 1, hidden_dim),
                           proposed.gamma, augmented, residual_sum)

        component_sample_count = args.warmup + args.iterations
        component_x = x.repeat(component_sample_count, 1)
        component_residual = residual.repeat(component_sample_count, 1)
        component_index = 0

        def native_norm_stage() -> object:
            nonlocal component_index
            begin = component_index * args.batch
            end = begin + args.batch
            component_index += 1
            return norm(component_x[begin:end], component_residual[begin:end])

        def proposed_marlin_stage() -> object:
            return apply_awq_marlin_linear(
                input=augmented,
                weight=proposed.qweight,
                weight_scale=proposed.scales,
                weight_zp=proposed.qzeros,
                g_idx=proposed.g_idx,
                g_idx_sort_indices=proposed.g_idx_sort_indices,
                workspace=proposed.workspace,
                quant_type=scalar_types.uint4,
                output_size_per_partition=proposed.padded_out_dim,
                input_size_per_partition=hidden_dim,
            )

        normalized, _ = norm(x, residual.clone())

        def native_marlin_stage() -> object:
            return apply_awq_marlin_linear(
                input=normalized,
                weight=native[0],
                weight_scale=native[1],
                weight_zp=native[2],
                g_idx=native[3],
                g_idx_sort_indices=native[4],
                workspace=native[5],
                quant_type=scalar_types.uint4,
                output_size_per_partition=out_dim,
                input_size_per_partition=hidden_dim,
            )

        preprocess_stage()
        sentinel_gemm_output = proposed_marlin_stage()

        def finalize_stage() -> None:
            finalize_out(sentinel_gemm_output, proposed.bias, output,
                         out_dim, hidden_dim, proposed.eps)

        component_times = {
            "vLLM RMSNorm": measure(
                native_norm_stage, args.warmup, args.iterations
            ),
            "vLLM Marlin": measure(
                native_marlin_stage, args.warmup, args.iterations
            ),
            "Sentinel preprocess": measure(
                preprocess_stage, args.warmup, args.iterations
            ),
            "Sentinel Marlin": measure(
                proposed_marlin_stage, args.warmup, args.iterations
            ),
            "Sentinel finalize": measure(
                finalize_stage, args.warmup, args.iterations
            ),
        }

    print(
        f"projection={args.projection} shape=[{args.batch},{hidden_dim}]x"
        f"[{hidden_dim},{out_dim}] group_size={group_size} "
        f"padded_out={proposed.padded_out_dim}"
    )
    native_mean = summarize("vLLM AWQ-Marlin W4A16", native_times)
    proposed_mean = summarize("Global-sentinel AWQ-Marlin", proposed_times)
    print(f"speedup={native_mean / proposed_mean:.3f}x")
    print(f"cosine={cosine:.7f} relative_rmse={relative_rmse:.7f}")
    print("component timings (independently measured):")
    for component_name, component_samples in component_times.items():
        summarize(component_name, component_samples)


if __name__ == "__main__":
    main()
