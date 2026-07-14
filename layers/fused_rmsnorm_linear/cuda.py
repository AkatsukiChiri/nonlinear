"""JIT loader for the specialized fused RMSNorm + Linear CUDA extension."""

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
        name="nonlinear_fused_rmsnorm_linear",
        sources=[
            str(
                root
                / "kernels"
                / "fused_rmsnorm_linear"
                / "fused_rmsnorm_linear.cpp"
            ),
            str(
                root
                / "kernels"
                / "fused_rmsnorm_linear"
                / "fused_rmsnorm_linear.cu"
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


def fused_rmsnorm_linear_fp16(
    x: torch.Tensor,
    fused_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the specialized small-batch fused RMSNorm + Linear kernel."""
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] != 8:
        raise RuntimeError("fused_rmsnorm_linear_fp16 requires an Ampere GPU")
    return _load_extension().fused_rmsnorm_linear_fp16(
        x,
        fused_weight,
        eps,
    )


def fused_rmsnorm_linear_bias_fp16(
    x: torch.Tensor,
    fused_weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the biased variant with bias fused into the RMS scale epilogue."""
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] != 8:
        raise RuntimeError("fused_rmsnorm_linear_bias_fp16 requires an Ampere GPU")
    return _load_extension().fused_rmsnorm_linear_bias_fp16(
        x,
        fused_weight,
        bias,
        eps,
    )
