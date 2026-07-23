"""JIT loader for the specialized fused LayerNorm + Linear CUDA extension."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_EXTENSION: ModuleType | None = None


def _load_extension() -> ModuleType:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6+PTX")
    _EXTENSION = load(
        name="nonlinear_fused_layernorm_linear",
        sources=[
            str(
                root
                / "kernels"
                / "fused_layernorm_linear"
                / "fused_layernorm_linear.cpp"
            ),
            str(
                root
                / "kernels"
                / "fused_layernorm_linear"
                / "fused_layernorm_linear.cu"
            ),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "-lineinfo",
        ],
        verbose=os.environ.get("NONLINEAR_VERBOSE_BUILD", "0") == "1",
    )
    return _EXTENSION


def fused_layernorm_linear_fp16(
    x: torch.Tensor,
    fused_weight: torch.Tensor,
    weight_sum: torch.Tensor,
    fused_bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the two-stage small-batch LayerNorm + Linear CUDA kernel."""
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] != 8:
        raise RuntimeError("fused_layernorm_linear_fp16 requires an Ampere GPU")
    return _load_extension().fused_layernorm_linear_fp16(
        x,
        fused_weight,
        weight_sum,
        fused_bias,
        eps,
    )
