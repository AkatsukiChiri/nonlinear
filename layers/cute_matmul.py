"""Python wrapper for the minimal CuTe FP16 matmul extension."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXTENSION = None


def _cutlass_path() -> Path:
    path = os.environ.get("CUTLASS_PATH")
    if path:
        return Path(path).expanduser()
    return Path.home() / "third_party" / "cutlass"


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    cutlass_path = _cutlass_path()
    cute_header = cutlass_path / "include" / "cute" / "tensor.hpp"
    if not cute_header.exists():
        raise RuntimeError(
            "CuTe header not found. Set CUTLASS_PATH to a CUTLASS checkout "
            f"containing include/cute/tensor.hpp. Tried: {cutlass_path}"
        )

    _EXTENSION = load(
        name="nonlinear_cute_matmul",
        sources=[
            str(root / "kernels" / "cute_matmul.cpp"),
            str(root / "kernels" / "cute_matmul.cu"),
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
            "-gencode=arch=compute_80,code=sm_80",
        ],
        verbose=False,
    )
    return _EXTENSION


def cute_matmul_fp16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute C = A @ B for contiguous FP16 CUDA matrices.

    Args:
        a: Left operand with shape [M, K].
        b: Right operand with shape [K, N].

    Returns:
        Matrix product with shape [M, N].
    """
    return _load_extension().cute_matmul_fp16(a, b)
