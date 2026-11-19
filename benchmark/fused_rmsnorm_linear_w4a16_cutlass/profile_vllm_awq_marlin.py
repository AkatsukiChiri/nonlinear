"""Launch steady-state vLLM AWQ-Marlin for Nsight Compute comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.fused_rmsnorm_linear_w4a16_global_sentinel.bench_global_sentinel import (
    DEFAULT_MODEL,
    load_tensors,
    make_native_marlin,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=(1, 4, 8), default=1)
    parser.add_argument("--projection", choices=("q", "gate"), default="gate")
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()

    qweight, qzeros, scales, gamma = load_tensors(
        DEFAULT_MODEL, 0, args.projection
    )
    qweight = qweight.cuda()
    qzeros = qzeros.cuda()
    scales = scales.cuda()
    gamma = gamma.cuda()
    hidden_dim = qweight.shape[0]
    out_dim = scales.shape[1]
    group_size = hidden_dim // scales.shape[0]
    native = make_native_marlin(qweight, qzeros, scales, group_size)

    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_awq_marlin_linear,
    )
    from vllm.scalar_type import scalar_types

    norm = RMSNorm(hidden_dim, eps=args.eps, dtype=torch.float16).cuda()
    with torch.no_grad():
        norm.weight.copy_(gamma)
    input_tensor = torch.randn(
        args.batch, hidden_dim, device="cuda", dtype=torch.float16
    )

    def forward() -> torch.Tensor:
        return apply_awq_marlin_linear(
            input=norm(input_tensor),
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

    with torch.inference_mode():
        for _ in range(10):
            forward()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("vllm_awq_marlin")
        forward()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
