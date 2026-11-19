"""Latency check for the single-kernel Marlin-style W4 decode path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear_w4a16_marlin.packing import (
    _pack_awq,
    pack_awq_marlin_sentinels,
)
from layers.fused_rmsnorm_linear_w4a16_marlin.layer import (
    FusedRMSNormLinearW4A16Marlin,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--out-dim", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--compare-vllm", action="store_true")
    args = parser.parse_args()
    hidden_dim = 4096
    qweight = _pack_awq(torch.randint(0, 16, (hidden_dim, args.out_dim),
                                      device="cuda", dtype=torch.uint8))
    qzeros = _pack_awq(torch.randint(0, 16, (hidden_dim // 128, args.out_dim),
                                     device="cuda", dtype=torch.uint8))
    scales = torch.rand(hidden_dim // 128, args.out_dim, device="cuda",
                        dtype=torch.float16) / 16
    x = torch.randn(args.batch, 1, hidden_dim, device="cuda", dtype=torch.float16)
    gamma = torch.ones(hidden_dim, device="cuda", dtype=torch.float16)
    proposed = FusedRMSNormLinearW4A16Marlin(
        qweight, qzeros, scales, gamma, 1e-5
    )
    for _ in range(args.warmup):
        proposed(x)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(args.iterations):
        proposed(x)
    end.record()
    end.synchronize()
    direct_ms = start.elapsed_time(end) / args.iterations
    print(f"direct_fragment_ms={direct_ms:.4f}")
    if not args.compare_vllm:
        return

    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_awq_marlin_linear,
        marlin_make_empty_g_idx,
        marlin_make_workspace,
    )
    from vllm.scalar_type import scalar_types

    native = pack_awq_marlin_sentinels(qweight, qzeros, scales)
    norm = RMSNorm(hidden_dim, eps=1e-5, dtype=torch.float16).cuda()
    with torch.no_grad():
        norm.weight.copy_(gamma)
    workspace = marlin_make_workspace(native.physical_out_dim, x.device)

    def vllm_forward() -> torch.Tensor:
        return apply_awq_marlin_linear(
            input=norm(x[:, 0]), weight=native.qweight, weight_scale=native.scales,
            weight_zp=native.zeros, g_idx=marlin_make_empty_g_idx(x.device),
            g_idx_sort_indices=marlin_make_empty_g_idx(x.device),
            workspace=workspace, quant_type=scalar_types.uint4,
            output_size_per_partition=native.physical_out_dim,
            input_size_per_partition=hidden_dim,
        )

    for _ in range(args.warmup):
        vllm_forward()
    torch.cuda.synchronize()
    start.record()
    for _ in range(args.iterations):
        vllm_forward()
    end.record()
    end.synchronize()
    vllm_ms = start.elapsed_time(end) / args.iterations
    print(f"vllm_rmsnorm_marlin_ms={vllm_ms:.4f}")
    print(f"speedup_vs_vllm={vllm_ms / direct_ms:.3f}x")


if __name__ == "__main__":
    main()
