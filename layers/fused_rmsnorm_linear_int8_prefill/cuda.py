"""JIT loader for the prefill INT8 extension."""

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
        source = root / "kernels" / "fused_rmsnorm_linear_int8_prefill"
        cutlass = Path(
            os.environ.get("CUTLASS_PATH", Path.home() / "third_party" / "cutlass")
        ).expanduser()
        if not (cutlass / "include" / "cutlass" / "cutlass.h").exists():
            raise RuntimeError(f"CUTLASS headers not found under {cutlass}")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6+PTX")
        _EXTENSION = load(
            name="nonlinear_fused_rmsnorm_linear_int8_prefill",
            sources=[str(source / "binding.cpp"), str(source / "kernel.cu")],
            extra_include_paths=[
                str(cutlass / "include"),
                str(cutlass / "tools" / "util" / "include"),
            ],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3", "-std=c++17", "--use_fast_math", "-lineinfo"
            ],
            verbose=os.environ.get("NONLINEAR_VERBOSE_BUILD", "0") == "1",
        )
    return _EXTENSION


def fused_rmsnorm_linear_int8_prefill_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """Run online quantization followed by INT8 Tensor Core GEMM."""
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] != 8:
        raise RuntimeError("INT8 prefill requires an Ampere GPU")
    return _load_extension().forward(x, weight, weight_scale, bias, eps)


def fused_rmsnorm_linear_int8_prefill_out_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    quantized_x: torch.Tensor,
    row_scale: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Run prefill into persistent quantization and output buffers."""
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] != 8:
        raise RuntimeError("INT8 prefill requires an Ampere GPU")
    return _load_extension().forward_out(
        x,
        weight,
        weight_scale,
        bias,
        eps,
        quantized_x,
        row_scale,
        output,
    )


def fused_add_rmsnorm_linear_int8_prefill_cuda(
    x: torch.Tensor,
    x_res: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused residual add, RMSNorm quantization, and INT8 GEMM."""
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] != 8:
        raise RuntimeError("INT8 fused-add prefill requires an Ampere GPU")
    output, residual_sum = _load_extension().forward_add(
        x, x_res, weight, weight_scale, bias, eps
    )
    return output, residual_sum


def fused_add_rmsnorm_linear_int8_prefill_out_cuda(
    x: torch.Tensor,
    x_res: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    quantized_x: torch.Tensor,
    row_scale: torch.Tensor,
    output: torch.Tensor,
    residual_sum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused-add prefill into persistent outputs and workspace."""
    result, residual = _load_extension().forward_add_out(
        x,
        x_res,
        weight,
        weight_scale,
        bias,
        eps,
        quantized_x,
        row_scale,
        output,
        residual_sum,
    )
    return result, residual
