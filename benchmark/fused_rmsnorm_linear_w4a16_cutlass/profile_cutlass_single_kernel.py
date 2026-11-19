"""Launch one steady-state CUTLASS fused kernel for Nsight Compute."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.fused_rmsnorm_linear_w4a16_global_sentinel.bench_global_sentinel import (
    DEFAULT_MODEL,
    load_tensors,
)
from layers.fused_rmsnorm_linear_w4a16_cutlass import (
    FusedRMSNormLinearW4A16Cutlass,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=(1, 4, 8), default=1)
    parser.add_argument("--split-k", type=int, default=4)
    parser.add_argument("--projection", choices=("q", "gate"), default="q")
    args = parser.parse_args()
    qweight, qzeros, scales, gamma = load_tensors(
        DEFAULT_MODEL, 0, args.projection
    )
    layer = FusedRMSNormLinearW4A16Cutlass(
        qweight.cuda(), qzeros.cuda(), scales.cuda(), gamma.cuda(),
        split_k_slices=args.split_k,
    ).eval()
    input = torch.randn(
        args.batch, 1, qweight.shape[0], device="cuda", dtype=torch.float16
    )
    with torch.inference_mode():
        for _ in range(10):
            layer(input)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("cutlass_w4a16_fused")
        layer(input)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
