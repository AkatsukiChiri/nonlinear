"""PyTorch module for the CUTLASS single-kernel decode path."""

from __future__ import annotations

import torch
from torch import nn

from .cuda import cutlass_w4a16_rms_sentinel, w4a16_async_decode
from .packing import pack_awq_with_sentinels


class FusedRMSNormLinearW4A16Cutlass(nn.Module):
    """Fused RMSNorm and groupwise-W4 projection for decode Batch <= 8."""

    def __init__(
        self,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        gamma: torch.Tensor,
        bias: torch.Tensor | None = None,
        eps: float = 1e-5,
        cta_n: int = 64,
        split_k_slices: int = 4,
    ) -> None:
        super().__init__()
        if gamma.dtype != torch.float16 or gamma.ndim != 1:
            raise TypeError("gamma must be float16 [HiddenDim]")
        if gamma.shape[0] != qweight.shape[0]:
            raise ValueError("gamma and qweight disagree on HiddenDim")
        if bias is not None and (
            bias.dtype != torch.float16
            or bias.ndim != 1
            or bias.shape[0] != scales.shape[1]
        ):
            raise TypeError("bias must be float16 [OutDim]")
        packed = pack_awq_with_sentinels(
            qweight,
            qzeros,
            scales,
            cta_n=cta_n,
            layout="n_minus_1",
        )
        self.register_buffer("packed_storage", packed.storage)
        self.register_buffer("gamma", gamma.detach().contiguous())
        self.register_buffer(
            "bias", None if bias is None else bias.detach().contiguous()
        )
        self.hidden_dim = qweight.shape[0]
        self.out_dim = scales.shape[1]
        self.physical_out_dim = packed.physical_out_dim
        self.cta_n = cta_n
        if split_k_slices < 1:
            raise ValueError("invalid split_k_slices")
        self.split_k_slices = min(
            int(split_k_slices), self.hidden_dim // 64
        )
        tile_count = self.physical_out_dim // self.cta_n
        self.register_buffer(
            "workspace",
            torch.empty(
                self.split_k_slices * tile_count * 8 * self.cta_n,
                dtype=torch.float32,
                device=packed.storage.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "counters",
            torch.zeros(
                tile_count, dtype=torch.int32, device=packed.storage.device
            ),
            persistent=False,
        )
        self.eps = float(eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Project contiguous FP16 input shaped `[Batch, 1, HiddenDim]`."""
        if input.ndim != 3 or input.shape[1:] != (1, self.hidden_dim):
            raise ValueError("input must be [Batch, 1, HiddenDim]")
        if not 1 <= input.shape[0] <= 8:
            raise ValueError("Batch must be in [1, 8]")
        if input.dtype != torch.float16 or not input.is_cuda:
            raise TypeError("input must be CUDA float16")
        if not input.is_contiguous():
            raise ValueError("input must be contiguous")
        output = cutlass_w4a16_rms_sentinel(
            input.reshape(input.shape[0], self.hidden_dim),
            self.gamma,
            self.packed_storage,
            self.out_dim,
            self.physical_out_dim,
            self.hidden_dim,
            self.eps,
            self.bias,
            self.workspace,
            self.counters,
            self.split_k_slices,
        )
        return output.unsqueeze(1)


def select_async_cta_n(hidden_dim: int, out_dim: int, batch: int) -> int:
    """Choose a decode tile from the layer shape instead of fixing CTA-N."""
    if hidden_dim % 64:
        raise ValueError("async decode requires HiddenDim divisible by 64")
    if not 1 <= batch <= 8:
        raise ValueError("Batch must be in [1, 8]")
    narrow_tiles = (out_dim + 62) // 63
    wide_tiles = (out_dim + 126) // 127
    # RTX 3090 has 82 SMs. A wide tile is profitable only if it still gives
    # enough independent CTAs to fill the device.
    return 128 if wide_tiles >= 82 else 64


class FusedRMSNormLinearW4A16AsyncDecode(nn.Module):
    """Single-kernel W4A16 decode with layer-shape-adaptive CTA width."""

    def __init__(
        self,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        gamma: torch.Tensor,
        bias: torch.Tensor | None = None,
        eps: float = 1e-5,
        cta_n: int | None = None,
        batch_hint: int = 1,
    ) -> None:
        super().__init__()
        if gamma.dtype != torch.float16 or gamma.ndim != 1:
            raise TypeError("gamma must be float16 [HiddenDim]")
        hidden_dim = qweight.shape[0]
        out_dim = scales.shape[1]
        if gamma.numel() != hidden_dim:
            raise ValueError("gamma and qweight disagree on HiddenDim")
        selected_cta_n = (
            select_async_cta_n(hidden_dim, out_dim, batch_hint)
            if cta_n is None
            else cta_n
        )
        if selected_cta_n not in (64, 128):
            raise ValueError("cta_n must be 64 or 128")
        packed = pack_awq_with_sentinels(
            qweight,
            qzeros,
            scales,
            cta_n=selected_cta_n,
            layout="n_minus_1",
        )
        self.register_buffer("packed_storage", packed.storage)
        self.register_buffer("gamma", gamma.detach().contiguous())
        self.register_buffer(
            "bias", None if bias is None else bias.detach().contiguous()
        )
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.physical_out_dim = packed.physical_out_dim
        self.cta_n = selected_cta_n
        self.eps = float(eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Project FP16 `[Batch, 1, HiddenDim]` in one CUDA kernel."""
        if input.ndim != 3 or input.shape[1:] != (1, self.hidden_dim):
            raise ValueError("input must be [Batch, 1, HiddenDim]")
        if not 1 <= input.shape[0] <= 8:
            raise ValueError("Batch must be in [1, 8]")
        if input.dtype != torch.float16 or not input.is_cuda:
            raise TypeError("input must be CUDA float16")
        if not input.is_contiguous():
            raise ValueError("input must be contiguous")
        output = w4a16_async_decode(
            input.reshape(input.shape[0], self.hidden_dim),
            self.gamma,
            self.packed_storage,
            self.out_dim,
            self.physical_out_dim,
            self.hidden_dim,
            self.cta_n,
            self.eps,
            self.bias,
        )
        return output.unsqueeze(1)
