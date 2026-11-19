"""Global-sentinel RMSNorm + AWQ-Marlin W4A16 decode prototype."""

from __future__ import annotations

import torch
from torch import nn

from .cuda import (
    awq_repacked_gemv_bs1_out,
    awq_tensorcore_out,
    finalize_out,
    preprocess_out,
)


class FusedRMSNormLinearW4A16GlobalSentinel(nn.Module):
    """Use one layer-wide all-ones column with a mature Marlin W4A16 GEMM."""

    def __init__(
        self,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        gamma: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float,
        group_size: int = 128,
        use_repacked_gemv_bs1: bool = True,
        use_tensorcore: bool = False,
    ) -> None:
        super().__init__()
        if qweight.dtype != torch.int32 or qweight.ndim != 2:
            raise TypeError("qweight must be AWQ int32 [HiddenDim, OutDim / 8]")
        if qzeros.dtype != torch.int32 or qzeros.ndim != 2:
            raise TypeError("qzeros must be AWQ int32 [Groups, OutDim / 8]")
        if scales.dtype != torch.float16 or scales.ndim != 2:
            raise TypeError("scales must be float16 [Groups, OutDim]")
        hidden_dim = qweight.shape[0]
        out_dim = scales.shape[1]
        if qweight.shape[1] * 8 != out_dim:
            raise ValueError("qweight and scales disagree on OutDim")
        if hidden_dim % group_size or scales.shape[0] != hidden_dim // group_size:
            raise ValueError("invalid AWQ group_size or scale shape")
        if qzeros.shape != (scales.shape[0], out_dim // 8):
            raise ValueError("qzeros shape does not match scales")
        if gamma.shape != (hidden_dim,) or gamma.dtype != torch.float16:
            raise TypeError("gamma must be float16 [HiddenDim]")
        if not all(
            tensor.is_cuda for tensor in (qweight, qzeros, scales, gamma)
        ):
            raise ValueError("AWQ weights and gamma must be CUDA tensors")

        # Marlin requires N to be a multiple of 64. The first padded channel
        # is the global all-ones sentinel; all later padding scales are zero.
        padded_out_dim = ((out_dim + 1 + 63) // 64) * 64
        packed_columns = padded_out_dim // 8
        with torch.no_grad():
            shifts = torch.tensor(
                [0, 16, 4, 20, 8, 24, 12, 28],
                dtype=torch.int32,
                device=qweight.device,
            )
            needs_unpacked_awq = use_tensorcore or (
                use_repacked_gemv_bs1 and out_dim <= 4096
            )
            if needs_unpacked_awq:
                unpacked_weight = (
                    (qweight.unsqueeze(-1) >> shifts) & 0xF
                ).reshape(hidden_dim, out_dim)
                unpacked_zeros = (
                    (qzeros.unsqueeze(-1) >> shifts) & 0xF
                ).reshape(scales.shape[0], out_dim).to(torch.uint8)
            else:
                unpacked_weight = torch.empty(
                    0, dtype=torch.int32, device=qweight.device
                )
                unpacked_zeros = torch.empty(
                    0, dtype=torch.uint8, device=qweight.device
                )
            if use_tensorcore:
                tensorcore_values = unpacked_weight.reshape(
                    hidden_dim, out_dim // 64, 4, 16
                ).permute(1, 2, 0, 3).contiguous()
                tensorcore_weight = (
                    tensorcore_values[..., 0::2]
                    | (tensorcore_values[..., 1::2] << 4)
                ).to(torch.uint8)
                tensorcore_zeros = unpacked_zeros.reshape(
                    scales.shape[0], out_dim // 64, 4, 16
                ).permute(1, 2, 0, 3).contiguous()
                tensorcore_scales = scales.reshape(
                    scales.shape[0], out_dim // 64, 4, 16
                ).permute(1, 2, 0, 3).contiguous()
            else:
                tensorcore_weight = torch.empty(
                    0, dtype=torch.uint8, device=qweight.device
                )
                tensorcore_zeros = torch.empty(
                    0, dtype=torch.uint8, device=qweight.device
                )
                tensorcore_scales = torch.empty(
                    0, dtype=torch.float16, device=qweight.device
                )
            if use_repacked_gemv_bs1 and out_dim <= 4096:
                transposed_weight = unpacked_weight.t().contiguous()
                repacked_weight = (
                    transposed_weight[:, 0::2]
                    | (transposed_weight[:, 1::2] << 4)
                ).to(torch.uint8)
                gemv_zeros = unpacked_zeros
            else:
                repacked_weight = torch.empty(
                    0, dtype=torch.uint8, device=qweight.device
                )
                gemv_zeros = torch.empty(
                    0, dtype=torch.uint8, device=qweight.device
                )
            padded_qweight = torch.full(
                (hidden_dim, packed_columns),
                0x11111111,
                dtype=torch.int32,
                device=qweight.device,
            )
            padded_qweight[:, : qweight.shape[1]].copy_(qweight)
            padded_qzeros = torch.zeros(
                scales.shape[0],
                packed_columns,
                dtype=torch.int32,
                device=qzeros.device,
            )
            padded_qzeros[:, : qzeros.shape[1]].copy_(qzeros)
            padded_scales = torch.zeros(
                scales.shape[0],
                padded_out_dim,
                dtype=torch.float16,
                device=scales.device,
            )
            padded_scales[:, :out_dim].copy_(scales)
            padded_scales[:, out_dim].fill_(1.0)

        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            awq_to_marlin_zero_points,
            marlin_make_empty_g_idx,
            marlin_make_workspace,
            marlin_permute_scales,
        )

        marlin_qweight = ops.awq_marlin_repack(
            padded_qweight,
            size_k=hidden_dim,
            size_n=padded_out_dim,
            num_bits=4,
        )
        marlin_scales = marlin_permute_scales(
            padded_scales, hidden_dim, padded_out_dim, group_size
        )
        marlin_qzeros = awq_to_marlin_zero_points(
            padded_qzeros, scales.shape[0], padded_out_dim, 4
        )
        self.register_buffer("awq_qweight", qweight.detach().contiguous())
        self.register_buffer("awq_qzeros", qzeros.detach().contiguous())
        self.register_buffer("awq_scales", scales.detach().contiguous())
        self.register_buffer("gemv_weight", repacked_weight)
        self.register_buffer("gemv_zeros", gemv_zeros)
        self.register_buffer("tensorcore_weight", tensorcore_weight)
        self.register_buffer("tensorcore_zeros", tensorcore_zeros)
        self.register_buffer("tensorcore_scales", tensorcore_scales)
        self.register_buffer("qweight", marlin_qweight)
        self.register_buffer("qzeros", marlin_qzeros)
        self.register_buffer("scales", marlin_scales)
        self.register_buffer("gamma", gamma.detach().clone().contiguous())
        self.register_buffer("bias", bias)
        self.register_buffer(
            "g_idx", marlin_make_empty_g_idx(qweight.device), persistent=False
        )
        self.register_buffer(
            "g_idx_sort_indices",
            marlin_make_empty_g_idx(qweight.device),
            persistent=False,
        )
        self.register_buffer(
            "workspace",
            marlin_make_workspace(padded_out_dim, qweight.device),
            persistent=False,
        )
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.padded_out_dim = padded_out_dim
        self.group_size = group_size
        self.eps = float(eps)
        self.use_repacked_gemv_bs1 = bool(use_repacked_gemv_bs1)
        self.use_tensorcore = bool(use_tensorcore)
        self._buffers_by_batch: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run decode for `[Batch, 1, HiddenDim]`, with Batch in `[1, 8]`."""
        if x.ndim != 3 or x.shape[1:] != (1, self.hidden_dim):
            raise ValueError("x must have shape [Batch, 1, HiddenDim]")
        batch = x.shape[0]
        if not 1 <= batch <= 8:
            raise ValueError("Batch must be in [1, 8]")
        if x.dtype != torch.float16 or not x.is_cuda or not x.is_contiguous():
            raise TypeError("x must be contiguous CUDA float16")
        if residual is not None and (
            residual.shape != x.shape
            or residual.dtype != x.dtype
            or residual.device != x.device
            or not residual.is_contiguous()
        ):
            raise TypeError("residual must match x")
        buffers = self._buffers_by_batch.get(batch)
        if buffers is None:
            buffers = (
                torch.empty(
                    batch * 2,
                    self.hidden_dim,
                    device=x.device,
                    dtype=torch.float16,
                ),
                torch.empty_like(x),
                torch.empty(
                    batch, 1, self.out_dim, device=x.device, dtype=x.dtype
                ),
            )
            self._buffers_by_batch[batch] = buffers
        augmented, residual_sum, output = buffers
        if (
            batch == 1
            and self.use_repacked_gemv_bs1
            and self.hidden_dim == 4096
            and self.gemv_weight.numel() > 0
        ):
            awq_repacked_gemv_bs1_out(
                x,
                residual,
                self.gamma,
                self.gemv_weight,
                self.gemv_zeros,
                self.awq_scales,
                self.bias,
                output,
                residual_sum if residual is not None else None,
                self.group_size,
                self.eps,
            )
            if residual is None:
                return output
            return output, residual_sum
        if self.use_tensorcore and self.tensorcore_weight.numel() > 0:
            awq_tensorcore_out(
                x,
                residual,
                self.gamma,
                self.tensorcore_weight,
                self.tensorcore_zeros,
                self.tensorcore_scales,
                self.bias,
                output,
                residual_sum if residual is not None else None,
                self.eps,
            )
            if residual is None:
                return output
            return output, residual_sum
        preprocess_out(
            x,
            residual,
            self.gamma,
            augmented,
            residual_sum if residual is not None else None,
        )

        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            apply_awq_marlin_linear,
        )
        from vllm.scalar_type import scalar_types

        gemm_output = apply_awq_marlin_linear(
            input=augmented,
            weight=self.qweight,
            weight_scale=self.scales,
            weight_zp=self.qzeros,
            g_idx=self.g_idx,
            g_idx_sort_indices=self.g_idx_sort_indices,
            workspace=self.workspace,
            quant_type=scalar_types.uint4,
            output_size_per_partition=self.padded_out_dim,
            input_size_per_partition=self.hidden_dim,
        )
        finalize_out(
            gemm_output,
            self.bias,
            output,
            self.out_dim,
            self.hidden_dim,
            self.eps,
        )
        if residual is None:
            return output
        return output, residual_sum


__all__ = ["FusedRMSNormLinearW4A16GlobalSentinel"]
