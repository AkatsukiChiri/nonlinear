"""JIT loader for the experimental prefill RMSNorm + Linear extension."""

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
    source_root = root / "kernels" / "fused_rmsnorm_linear"
    cutlass_root = Path(
        os.environ.get("CUTLASS_PATH", Path.home() / "third_party" / "cutlass")
    ).expanduser()
    if not (cutlass_root / "include" / "cutlass" / "cutlass.h").exists():
        raise RuntimeError(
            "CUTLASS headers are required for the prefill kernel; "
            f"tried {cutlass_root}"
        )
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6+PTX")
    _EXTENSION = load(
        name="nonlinear_fused_rmsnorm_linear_prefill",
        sources=[
            str(source_root / "fused_rmsnorm_linear_prefill.cpp"),
            str(source_root / "fused_rmsnorm_linear_prefill.cu"),
        ],
        extra_include_paths=[
            str(cutlass_root / "include"),
            str(cutlass_root / "tools" / "util" / "include"),
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


def fused_rmsnorm_linear_prefill_fp16(
    x: torch.Tensor,
    fused_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the no-bias experimental Tensor Core prefill kernel."""
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] != 8:
        raise RuntimeError("the prefill kernel requires an Ampere GPU")
    return _load_extension().fused_rmsnorm_linear_prefill_fp16(
        x, fused_weight, eps
    )


def fused_rmsnorm_linear_prefill_bias_fp16(
    x: torch.Tensor,
    fused_weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the biased experimental Tensor Core prefill kernel."""
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] != 8:
        raise RuntimeError("the prefill kernel requires an Ampere GPU")
    return _load_extension().fused_rmsnorm_linear_prefill_bias_fp16(
        x, fused_weight, bias, eps
    )
