"""Compare CUTLASS single-kernel decode against vLLM and previous fusion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.fused_rmsnorm_linear_w4a16_global_sentinel.bench_global_sentinel import (
    DEFAULT_MODEL,
    PROJECTIONS,
    load_tensors,
    make_native_marlin,
    measure,
    summarize,
)
from layers.fused_rmsnorm_linear_w4a16_cutlass import (
    FusedRMSNormLinearW4A16AsyncDecode,
    FusedRMSNormLinearW4A16Cutlass,
)
from layers.fused_rmsnorm_linear_w4a16_global_sentinel import (
    FusedRMSNormLinearW4A16GlobalSentinel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--projection", choices=PROJECTIONS, default="q")
    parser.add_argument("--batch", type=int, choices=(1, 4, 8), default=1)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--split-k", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    current = FusedRMSNormLinearW4A16Cutlass(
        qweight,
        qzeros,
        scales,
        gamma,
        None,
        args.eps,
        split_k_slices=args.split_k,
    ).eval()
    async_decode = FusedRMSNormLinearW4A16AsyncDecode(
        qweight,
        qzeros,
        scales,
        gamma,
        None,
        args.eps,
        batch_hint=args.batch,
    ).eval()
    previous = FusedRMSNormLinearW4A16GlobalSentinel(
        qweight,
        qzeros,
        scales,
        gamma,
        None,
        args.eps,
        group_size,
        use_tensorcore=args.batch > 1,
    ).eval()
    native = make_native_marlin(qweight, qzeros, scales, group_size)

    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_awq_marlin_linear,
    )
    from vllm.scalar_type import scalar_types

    norm = VllmRMSNorm(hidden_dim, eps=args.eps, dtype=torch.float16).cuda()
    with torch.no_grad():
        norm.weight.copy_(gamma)
    input = torch.randn(
        args.batch, hidden_dim, device="cuda", dtype=torch.float16
    ) / 4

    def native_forward() -> torch.Tensor:
        normalized = norm(input)
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

    def previous_forward() -> torch.Tensor:
        result = previous(input.reshape(args.batch, 1, hidden_dim))
        assert isinstance(result, torch.Tensor)
        return result.reshape(args.batch, out_dim)

    def current_forward() -> torch.Tensor:
        return current(input.reshape(args.batch, 1, hidden_dim)).reshape(
            args.batch, out_dim
        )

    def async_forward() -> torch.Tensor:
        return async_decode(input.reshape(args.batch, 1, hidden_dim)).reshape(
            args.batch, out_dim
        )

    with torch.inference_mode():
        native_output = native_forward()
        previous_output = previous_forward()
        current_output = current_forward()
        async_output = async_forward()
        for name, output in (
            ("previous", previous_output),
            ("current", current_output),
            ("async", async_output),
        ):
            cosine = torch.nn.functional.cosine_similarity(
                output.float().reshape(-1),
                native_output.float().reshape(-1),
                dim=0,
            ).item()
            print(f"{name} cosine={cosine:.7f}")
        native_times = measure(native_forward, args.warmup, args.iterations)
        previous_times = measure(previous_forward, args.warmup, args.iterations)
        current_times = measure(current_forward, args.warmup, args.iterations)
        async_times = measure(async_forward, args.warmup, args.iterations)

    print(
        f"projection={args.projection} batch={args.batch} "
        f"shape={hidden_dim}x{out_dim} split_k={current.split_k_slices}"
    )
    native_mean = summarize("vLLM RMSNorm + AWQ-Marlin", native_times)
    previous_mean = summarize("Previous fused implementation", previous_times)
    current_mean = summarize("CUTLASS single kernel", current_times)
    async_mean = summarize(
        f"Adaptive async single kernel (CTA-N={async_decode.cta_n})",
        async_times,
    )
    print(f"current/vLLM speedup={native_mean / current_mean:.3f}x")
    print(f"current/previous speedup={previous_mean / current_mean:.3f}x")
    print(f"async/vLLM speedup={native_mean / async_mean:.3f}x")
    print(f"async/current speedup={current_mean / async_mean:.3f}x")


if __name__ == "__main__":
    main()
