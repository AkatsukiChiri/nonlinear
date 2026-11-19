"""Benchmark a real FP16 Llama projection after offline AWQ-compatible W4 pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.fused_rmsnorm_linear_w4a16_marlin.bench_real_awq_projection import (
    _measure,
)
from benchmark.fused_rmsnorm_linear_w4a16_global_sentinel.bench_global_sentinel import (
    make_native_marlin,
)
from layers.fused_rmsnorm_linear_w4a16_marlin.layer import (
    FusedRMSNormLinearW4A16Marlin,
)
from layers.fused_rmsnorm_linear_w4a16_marlin.packing import (
    quantize_weight_to_awq_groupwise,
)


PROJECTIONS = {
    "q": ("self_attn.q_proj", "input_layernorm"),
    "gate": ("mlp.gate_proj", "post_attention_layernorm"),
}


def _load_tensor(model: Path, name: str) -> torch.Tensor:
    with (model / "model.safetensors.index.json").open(encoding="utf-8") as file:
        shard = json.load(file)["weight_map"][name]
    with safe_open(model / shard, framework="pt", device="cpu") as checkpoint:
        return checkpoint.get_tensor(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--projection", choices=PROJECTIONS, default="q")
    parser.add_argument("--batch", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    projection, norm = PROJECTIONS[args.projection]
    prefix = "model.layers.0"
    weight = _load_tensor(args.model, f"{prefix}.{projection}.weight").cuda()
    gamma = _load_tensor(args.model, f"{prefix}.{norm}.weight").cuda().to(
        torch.float16
    )
    qweight, qzeros, scales = quantize_weight_to_awq_groupwise(weight)
    hidden_dim = qweight.shape[0]
    out_dim = scales.shape[1]
    x = torch.randn(args.batch, 1, hidden_dim, device="cuda", dtype=torch.float16)
    fused = FusedRMSNormLinearW4A16Marlin(
        qweight, qzeros, scales, gamma, 1e-5
    ).eval()
    native = make_native_marlin(qweight, qzeros, scales, 128)

    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_awq_marlin_linear,
    )
    from vllm.scalar_type import scalar_types

    norm_layer = RMSNorm(hidden_dim, eps=1e-5, dtype=torch.float16).cuda().eval()
    with torch.no_grad():
        norm_layer.weight.copy_(gamma)

    def fused_forward() -> torch.Tensor:
        return fused(x)

    def vllm_forward() -> torch.Tensor:
        return apply_awq_marlin_linear(
            input=norm_layer(x[:, 0]), weight=native[0],
            weight_scale=native[1], weight_zp=native[2], g_idx=native[3],
            g_idx_sort_indices=native[4], workspace=native[5],
            quant_type=scalar_types.uint4, output_size_per_partition=out_dim,
            input_size_per_partition=hidden_dim,
        )

    with torch.inference_mode():
        fused_ms = _measure(fused_forward, args.warmup, args.iterations)
        vllm_ms = _measure(vllm_forward, args.warmup, args.iterations)
    print(
        f"model={args.model.name} projection={args.projection} batch={args.batch} "
        f"shape={hidden_dim}x{out_dim} fused_p50_ms={fused_ms:.4f} "
        f"vllm_p50_ms={vllm_ms:.4f} speedup={vllm_ms / fused_ms:.3f}x"
    )


if __name__ == "__main__":
    main()
