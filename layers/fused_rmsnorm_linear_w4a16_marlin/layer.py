"""Decode-only RMSNorm + W4A16 projection using a single CTA-per-stripe kernel."""

from __future__ import annotations

import torch
from torch import nn

from .cuda import w4a16_marlin_sentinel_wmma
from .packing import pack_awq_sentinel_fragment


class FusedRMSNormLinearW4A16Marlin(nn.Module):
    """Single-kernel decode projection for `[Batch, 1, HiddenDim]` inputs.

    The class performs all layout expansion at model load.  Its forward pass
    launches exactly one CUDA kernel. K-split partials and synchronization
    counters are allocated once with the layer and reused for every decode.
    """

    def __init__(
        self,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        bias: torch.Tensor | None = None,
        stripe_n: int = 64,
    ) -> None:
        super().__init__()
        if qweight.ndim != 2 or qzeros.ndim != 2 or scales.ndim != 2:
            raise ValueError("AWQ tensors must be rank two")
        if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
            raise TypeError("AWQ qweight and qzeros must be int32")
        if scales.dtype != torch.float16 or gamma.dtype != torch.float16:
            raise TypeError("scales and gamma must be float16")
        hidden_dim = qweight.shape[0]
        out_dim = scales.shape[1]
        if gamma.shape != (hidden_dim,):
            raise ValueError("gamma must have shape [HiddenDim]")
        if bias is not None and (bias.shape != (out_dim,) or bias.dtype != torch.float16):
            raise ValueError("bias must be float16 [OutDim]")
        if not all(tensor.is_cuda for tensor in (qweight, qzeros, scales, gamma)):
            raise ValueError("weights and gamma must be CUDA tensors")
        packed = pack_awq_sentinel_fragment(
            qweight, qzeros, scales, stripe_n=stripe_n
        )
        self.register_buffer("qweight", packed.qweight)
        self.register_buffer("zeros", packed.zeros)
        self.register_buffer("scales", packed.scales)
        self.register_buffer(
            "workspace",
            torch.empty(
                (8, 16, packed.physical_out_dim),
                dtype=torch.float32,
                device=qweight.device,
            ),
        )
        self.register_buffer(
            "locks",
            torch.zeros(
                (packed.physical_out_dim // packed.stripe_n,),
                dtype=torch.int32,
                device=qweight.device,
            ),
        )
        self.register_buffer("gamma", gamma.detach().contiguous())
        self.register_buffer(
            "bias", None if bias is None else bias.detach().contiguous()
        )
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.stripe_n = packed.stripe_n
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project a contiguous FP16 decode input of shape `[Batch, 1, HiddenDim]`."""
        if x.ndim != 3 or x.shape[1:] != (1, self.hidden_dim):
            raise ValueError("x must have shape [Batch, 1, HiddenDim]")
        if not 1 <= x.shape[0] <= 8:
            raise ValueError("Batch must be in [1, 8]")
        if not x.is_cuda or x.dtype != torch.float16 or not x.is_contiguous():
            raise TypeError("x must be contiguous CUDA float16")
        output = w4a16_marlin_sentinel_wmma(
            x[:, 0],
            self.gamma,
            self.qweight,
            self.zeros,
            self.scales,
            self.out_dim,
            self.eps,
            self.bias,
            self.workspace,
            self.locks,
            self.stripe_n,
        )
        return output.unsqueeze(1)
