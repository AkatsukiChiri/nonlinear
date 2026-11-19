"""Checkpoint and vLLM baseline helpers for the W4A16 benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

DEFAULT_MODEL = Path(
    "/home/mashaobo/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
)
PROJECTIONS = {
    "q": ("self_attn.q_proj", "input_layernorm"),
    "gate": ("mlp.gate_proj", "post_attention_layernorm"),
}


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
