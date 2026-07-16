"""Loader for the FP32 mixed weight-only INT8 CUDA extension."""

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
        source = root / "kernels" / "fused_rmsnorm_linear_weight_only_int8_mixed"
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6+PTX")
        _EXTENSION = load(
            name="nonlinear_weight_only_int8_mixed",
            sources=[str(source / "binding.cpp"), str(source / "kernel.cu")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math", "-lineinfo"],
            verbose=os.environ.get("NONLINEAR_VERBOSE_BUILD", "0") == "1",
        )
    return _EXTENSION


def weight_only_int8_mixed_cuda(
    x: torch.Tensor,
    quantized_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """Dispatch the FP32 mixed weight-only kernel."""
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] != 8:
        raise RuntimeError("weight-only INT8 currently requires an Ampere GPU")
    return _load_extension().forward(x, quantized_weight, weight_scale, bias, eps)
