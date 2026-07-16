"""JIT loader for the CUDA Core INT8 RMSNorm + Linear extension."""

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
    source_root = root / "kernels" / "fused_rmsnorm_linear_int8"
    _EXTENSION = load(
        name="nonlinear_fused_rmsnorm_linear_int8",
        sources=[
            str(source_root / "fused_rmsnorm_linear_int8.cpp"),
            str(source_root / "fused_rmsnorm_linear_int8.cu"),
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


def _check_ampere(x: torch.Tensor) -> None:
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] != 8:
        raise RuntimeError("the INT8 fused kernel currently requires an Ampere GPU")


def fused_rmsnorm_linear_int8(
    x: torch.Tensor,
    quantized_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run dynamically quantized activation and DP4A projection kernels."""
    _check_ampere(x)
    return _load_extension().fused_rmsnorm_linear_int8(
        x, quantized_weight, weight_scale, eps
    )


def fused_rmsnorm_linear_bias_int8(
    x: torch.Tensor,
    quantized_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the biased dynamically quantized DP4A projection."""
    _check_ampere(x)
    return _load_extension().fused_rmsnorm_linear_bias_int8(
        x, quantized_weight, weight_scale, bias, eps
    )
