"""CUDA bindings for the CUTLASS W4A16 fused decode implementation."""

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
        source = root / "kernels" / "fused_rmsnorm_linear_w4a16_cutlass"
        cutlass = Path(
            os.environ.get("CUTLASS_PATH", Path.home() / "third_party" / "cutlass")
        )
        if not (cutlass / "include" / "cutlass" / "cutlass.h").exists():
            raise RuntimeError(f"CUTLASS headers not found under {cutlass}")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6+PTX")
        _EXTENSION = load(
            name="nonlinear_w4a16_cutlass",
            sources=[
                str(source / "binding.cpp"),
                str(source / "kernel.cu"),
                str(source / "async_decode_kernel.cu"),
            ],
            extra_include_paths=[
                str(cutlass / "include"),
                str(cutlass / "tools" / "util" / "include"),
                str(
                    cutlass
                    / "examples"
                    / "37_gemm_layernorm_gemm_fusion"
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


def cutlass_w4a16_probe(
    activation: torch.Tensor,
    packed_weight: torch.Tensor,
    out_dim: int,
) -> torch.Tensor:
    """Run CUTLASS FP16 × groupwise-W4 mixed-input GEMM."""
    return _load_extension().cutlass_w4a16_probe(
        activation, packed_weight, out_dim
    )


def cutlass_w4a16_rms_sentinel(
    input: torch.Tensor,
    gamma: torch.Tensor,
    packed_storage: torch.Tensor,
    logical_out_dim: int,
    physical_out_dim: int,
    hidden_dim: int,
    eps: float,
    bias: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    counters: torch.Tensor | None = None,
    split_k_slices: int = 1,
) -> torch.Tensor:
    """Apply GEMM and RMS recovery in one CUTLASS kernel."""
    tile_count = physical_out_dim // 64
    if workspace is None:
        workspace = torch.empty(
            split_k_slices * tile_count * 8 * 64,
            dtype=torch.float32,
            device=input.device,
        )
    if counters is None:
        counters = torch.zeros(
            tile_count, dtype=torch.int32, device=input.device
        )
    return _load_extension().cutlass_w4a16_rms_sentinel(
        input,
        gamma,
        packed_storage,
        logical_out_dim,
        physical_out_dim,
        hidden_dim,
        eps,
        bias,
        workspace,
        counters,
        split_k_slices,
    )


def w4a16_async_decode(
    input: torch.Tensor,
    gamma: torch.Tensor,
    packed_storage: torch.Tensor,
    logical_out_dim: int,
    physical_out_dim: int,
    hidden_dim: int,
    cta_n: int,
    eps: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute the single-kernel cp.async W4A16 decode path."""
    return _load_extension().w4a16_async_decode(
        input,
        gamma,
        packed_storage,
        logical_out_dim,
        physical_out_dim,
        hidden_dim,
        cta_n,
        eps,
        bias,
    )
