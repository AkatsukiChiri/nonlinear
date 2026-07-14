"""Python wrapper for the fused square-and-linear CUDA extension."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_EXTENSION: ModuleType | None = None


def _cutlass_path() -> Path:
    configured_path = os.environ.get("CUTLASS_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    return Path.home() / "third_party" / "cutlass"


def _load_extension() -> ModuleType:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    root = Path(__file__).resolve().parents[2]
    cutlass_path = _cutlass_path()
    cute_header = cutlass_path / "include" / "cute" / "tensor.hpp"
    if not cute_header.exists():
        raise RuntimeError(
            "CuTe header not found. Set CUTLASS_PATH to a CUTLASS checkout "
            f"containing include/cute/tensor.hpp. Tried: {cutlass_path}"
        )

    # Both targets use Ampere mma.sync. sm_80 PTX is retained so the driver can
    # JIT for other Ampere variants when no matching cubin is present.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6+PTX")
    _EXTENSION = load(
        name="nonlinear_fused_square_linear",
        sources=[
            str(
                root
                / "kernels"
                / "fused_square_linear"
                / "fused_square_linear.cpp"
            ),
            str(
                root
                / "kernels"
                / "fused_square_linear"
                / "fused_square_linear.cu"
            ),
        ],
        extra_include_paths=[
            str(cutlass_path / "include"),
            str(cutlass_path / "tools" / "util" / "include"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-lineinfo",
        ],
        verbose=os.environ.get("NONLINEAR_VERBOSE_BUILD", "0") == "1",
    )
    return _EXTENSION


def fused_square_linear_fp16(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Apply a fused ``[x, x**2] @ weight.T`` projection on Ampere.

    Args:
        x: Contiguous FP16 CUDA tensor shaped [Batch, 1, HiddenDim].
        weight: Contiguous FP16 CUDA tensor shaped [OutDim, HiddenDim].

    Returns:
        FP16 tensor shaped [Batch, 2, OutDim].
    """
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] != 8:
        raise RuntimeError("fused_square_linear_fp16 requires an Ampere GPU")
    return _load_extension().fused_square_linear_fp16(x, weight)
