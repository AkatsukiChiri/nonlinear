"""JIT loader for the sentinel-column Tensor Core decode kernel."""

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
        source = root / "kernels" / "fused_rmsnorm_linear_tensorcore_decode"
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6+PTX")
        _EXTENSION = load(
            name="nonlinear_tensorcore_rms_decode",
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


def tensorcore_rms_linear_decode_out_cuda(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight_fp16: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    output: torch.Tensor,
    residual_sum: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Execute a batched Tensor Core RMSNorm projection into stable buffers."""
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] != 8:
        raise RuntimeError("Tensor Core decode currently requires an Ampere GPU")
    result = _load_extension().forward_out(
        x,
        residual,
        weight_fp16,
        output.shape[-1],
        bias,
        eps,
        output,
        residual_sum,
    )
    if residual is None:
        return result[0]
    return result[0], result[1]


def tensorcore_rms_linear_int8_weight_decode_out_cuda(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    quantized_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    output: torch.Tensor,
    residual_sum: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run FP16 Tensor Cores with block-local INT8 weight dequantization."""
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] != 8:
        raise RuntimeError("Tensor Core decode currently requires an Ampere GPU")
    result = _load_extension().forward_int8_weight_out(
        x,
        residual,
        quantized_weight,
        weight_scale,
        output.shape[-1],
        bias,
        eps,
        output,
        residual_sum,
    )
    if residual is None:
        return result[0]
    return result[0], result[1]
