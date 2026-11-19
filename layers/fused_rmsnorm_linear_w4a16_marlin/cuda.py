"""CUDA binding for the single-kernel W4A16 sentinel decode bring-up path."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_EXTENSION: ModuleType | None = None


def _load_extension() -> ModuleType:
    global _EXTENSION
    if _EXTENSION is None:
        root = Path(__file__).resolve().parents[2]
        source = root / "kernels" / "fused_rmsnorm_linear_w4a16_marlin"
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6+PTX")
        _EXTENSION = load(
            name="nonlinear_w4a16_marlin_sentinel",
            sources=[str(source / "binding.cpp"), str(source / "kernel.cu")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math", "-lineinfo"],
            verbose=os.environ.get("NONLINEAR_VERBOSE_BUILD", "0") == "1",
        )
    return _EXTENSION


def w4a16_marlin_sentinel_wmma(
    input: torch.Tensor,
    gamma: torch.Tensor,
    qweight: torch.Tensor,
    zeros: torch.Tensor,
    scales: torch.Tensor,
    logical_out_dim: int,
    eps: float,
    bias: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    locks: torch.Tensor | None = None,
    stripe_n: int = 32,
) -> torch.Tensor:
    """Fused RMSNorm + W4 Tensor Core decode for `[Batch, HiddenDim]`."""
    return _load_extension().w4a16_marlin_sentinel_wmma(
        input,
        gamma,
        qweight,
        zeros,
        scales,
        logical_out_dim,
        stripe_n,
        eps,
        bias,
        workspace,
        locks,
    )
