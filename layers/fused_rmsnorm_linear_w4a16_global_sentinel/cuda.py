"""CUDA preprocessing and epilogue for global-sentinel W4A16 decode."""

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
        source = (
            root
            / "kernels"
            / "fused_rmsnorm_linear_w4a16_global_sentinel"
        )
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6+PTX")
        _EXTENSION = load(
            name="nonlinear_w4a16_global_sentinel",
            sources=[str(source / "binding.cpp"), str(source / "kernel.cu")],
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


def preprocess_out(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    gamma: torch.Tensor,
    augmented: torch.Tensor,
    residual_sum: torch.Tensor | None,
) -> None:
    """Construct `[z * gamma; z^2]` in a persistent output buffer."""
    _load_extension().preprocess_out(
        x, residual, gamma, augmented, residual_sum
    )


def finalize_out(
    gemm_output: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
    out_dim: int,
    hidden_dim: int,
    eps: float,
) -> None:
    """Read the one global sentinel and apply RMS scaling plus bias."""
    _load_extension().finalize_out(
        gemm_output, bias, output, out_dim, hidden_dim, eps
    )


def awq_fused_gemv_bs1_out(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    gamma: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
    residual_sum: torch.Tensor | None,
    group_size: int,
    eps: float,
) -> None:
    """Run the direct AutoAWQ-layout fused decode kernel for Batch=1."""
    _load_extension().awq_fused_gemv_bs1_out(
        x, residual, gamma, qweight, qzeros, scales, bias, output,
        residual_sum, group_size, eps
    )


def awq_repacked_gemv_bs1_out(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    gamma: torch.Tensor,
    weight: torch.Tensor,
    zeros: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
    residual_sum: torch.Tensor | None,
    group_size: int,
    eps: float,
) -> None:
    """Run output-major AWQ fused GEMV for Batch=1."""
    _load_extension().awq_repacked_gemv_bs1_out(
        x, residual, gamma, weight, zeros, scales, bias, output,
        residual_sum, group_size, eps
    )


def awq_tensorcore_out(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    gamma: torch.Tensor,
    weight: torch.Tensor,
    zeros: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
    residual_sum: torch.Tensor | None,
    eps: float,
) -> None:
    """Run tiled FP16 Tensor Core projection from packed AWQ INT4 weights."""
    _load_extension().awq_tensorcore_out(
        x, residual, gamma, weight, zeros, scales, bias, output,
        residual_sum, eps
    )
