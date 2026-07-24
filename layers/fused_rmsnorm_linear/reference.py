"""Reference RMSNorm + Linear computation for Llama-style decode inputs."""

from __future__ import annotations

import torch


DEFAULT_EPS = 1e-6
DEFAULT_BATCH = 1
DEFAULT_SEQ_LEN = 1
DEFAULT_HIDDEN_DIM = 4096
DEFAULT_OUT_DIM = 4096


def set_seed(seed: int) -> None:
    """Set PyTorch seeds for reproducible reference inputs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmsnorm_linear_ref(
    x: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Apply RMSNorm followed by a linear projection.

    Args:
        x: Decode hidden states with shape [Batch, SeqLen, HiddenDim].
        rmsnorm_weight: RMSNorm scale with shape [HiddenDim].
        linear_weight: Linear weight with shape [OutDim, HiddenDim].
        eps: Small value added to the RMS variance.

    Returns:
        Projected hidden states with shape [Batch, SeqLen, OutDim].
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape [Batch, SeqLen, HiddenDim], got {x.shape}")
    if x.shape[1] != 1:
        raise ValueError(f"decode reference expects SeqLen == 1, got {x.shape[1]}")

    hidden_dim = x.shape[-1]
    if rmsnorm_weight.shape != (hidden_dim,):
        raise ValueError(
            "rmsnorm_weight must have shape [HiddenDim], "
            f"got {rmsnorm_weight.shape} for HiddenDim={hidden_dim}"
        )
    if linear_weight.ndim != 2:
        raise ValueError(
            "linear_weight must have shape [OutDim, HiddenDim], "
            f"got {linear_weight.shape}"
        )
    if linear_weight.shape[1] != hidden_dim:
        raise ValueError(
            "linear_weight second dimension must equal HiddenDim, "
            f"got {linear_weight.shape[1]} and {hidden_dim}"
        )

    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps).to(dtype=x.dtype)
    x_norm = x_norm * rmsnorm_weight
    return torch.matmul(x_norm, linear_weight.t())


def add_rmsnorm_linear_ref(
    x: torch.Tensor,
    x_res: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor | None = None,
    eps: float = DEFAULT_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply FP16 residual add, RMSNorm, and Linear.

    Args:
        x: Decode hidden states shaped [Batch, 1, HiddenDim].
        x_res: Residual states with the same shape, dtype, and device as ``x``.
        rmsnorm_weight: RMSNorm scale shaped [HiddenDim].
        linear_weight: Linear weight shaped [OutDim, HiddenDim].
        linear_bias: Optional Linear bias shaped [OutDim].
        eps: Small value added to the RMS variance.

    Returns:
        A tuple containing projected states shaped [Batch, 1, OutDim] and the
        FP16 residual sum shaped [Batch, 1, HiddenDim].
    """
    if x.shape != x_res.shape:
        raise ValueError(
            f"x and x_res must have the same shape, got {x.shape} and {x_res.shape}"
        )
    if x.dtype != x_res.dtype or x.device != x_res.device:
        raise ValueError("x and x_res must have the same dtype and device")

    residual_sum = x + x_res
    output = rmsnorm_linear_ref(
        residual_sum,
        rmsnorm_weight,
        linear_weight,
        eps,
    )
    if linear_bias is not None:
        output = output + linear_bias
    return output, residual_sum


def build_llama3_8b_qkv_decode_inputs(
    *,
    batch: int = DEFAULT_BATCH,
    seq_len: int = DEFAULT_SEQ_LEN,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    out_dim: int = DEFAULT_OUT_DIM,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build reproducible Llama3-8B QKV-sized decode reference inputs.

    Shapes:
        x: [Batch, SeqLen, HiddenDim]
        rmsnorm_weight: [HiddenDim]
        linear_weight: [OutDim, HiddenDim]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(batch, seq_len, hidden_dim, device=device, dtype=dtype)
    rmsnorm_weight = torch.randn(hidden_dim, device=device, dtype=dtype)
    linear_weight = torch.randn(out_dim, hidden_dim, device=device, dtype=dtype)
    return x, rmsnorm_weight, linear_weight


def main() -> None:
    """Run a minimal deterministic sanity check."""
    seed = 20260709
    set_seed(seed)
    x, rmsnorm_weight, linear_weight = build_llama3_8b_qkv_decode_inputs()
    out = rmsnorm_linear_ref(x, rmsnorm_weight, linear_weight)

    print(f"seed: {seed}")
    print(f"x shape: {tuple(x.shape)}, dtype: {x.dtype}, device: {x.device}")
    print(f"rmsnorm_weight shape: {tuple(rmsnorm_weight.shape)}")
    print(f"linear_weight shape: {tuple(linear_weight.shape)}")
    print(f"out shape: {tuple(out.shape)}, dtype: {out.dtype}, device: {out.device}")
    print(f"out: {out}")
    print(f"out checksum: {out.float().sum().item():.6f}")


if __name__ == "__main__":
    main()
