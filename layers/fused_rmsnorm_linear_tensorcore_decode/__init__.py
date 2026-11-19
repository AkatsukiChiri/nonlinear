"""Sentinel-column FP16 Tensor Core RMSNorm + Linear decode layer."""

from __future__ import annotations

import torch
from torch import nn

from .cuda import (
    tensorcore_rms_linear_decode_out_cuda,
    tensorcore_rms_linear_int8_weight_decode_out_cuda,
)


class FusedRMSNormLinearTensorCoreDecode(nn.Module):
    """Compute batched RMSNorm projection with `[x; x^2]` Tensor Core rows."""

    def __init__(
        self,
        quantized_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float,
        compute_dtype: torch.dtype,
        tiles_per_cta: int = 4,
    ) -> None:
        super().__init__()
        if quantized_weight.ndim != 2 or quantized_weight.dtype != torch.int8:
            raise TypeError("quantized_weight must be int8 [OutDim, HiddenDim]")
        if weight_scale.numel() != quantized_weight.shape[0]:
            raise ValueError("weight_scale must contain one value per output")
        if compute_dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("compute_dtype must be float16 or bfloat16")
        if tiles_per_cta not in (1, 2, 4, 8, 16, 32):
            raise ValueError("tiles_per_cta must be 1, 2, 4, 8, 16, or 32")
        with torch.no_grad():
            weight_fp16 = (
                quantized_weight.float()
                * weight_scale.reshape(-1, 1).float()
            ).to(torch.float16)
            physical_columns = tiles_per_cta * 16
            projection_columns = physical_columns - 1
            tile_count = (
                quantized_weight.shape[0] + projection_columns - 1
            ) // projection_columns
            padded = torch.zeros(
                tile_count * projection_columns,
                quantized_weight.shape[1],
                device=quantized_weight.device,
                dtype=torch.float16,
            )
            padded[: quantized_weight.shape[0]].copy_(weight_fp16)
            packed_flat = torch.empty(
                tile_count,
                quantized_weight.shape[1],
                physical_columns,
                device=quantized_weight.device,
                dtype=torch.float16,
            )
            packed_flat[:, :, :projection_columns].copy_(
                padded.reshape(
                    tile_count, projection_columns, quantized_weight.shape[1]
                )
                .transpose(1, 2)
            )
            packed_flat[:, :, projection_columns].fill_(1.0)
            packed = packed_flat.reshape(
                tile_count,
                quantized_weight.shape[1],
                tiles_per_cta,
                16,
            ).permute(0, 2, 1, 3)
        self.register_buffer("weight_fp16", packed.contiguous())
        self.register_buffer("bias", bias)
        self.eps = float(eps)
        self.hidden_dim = quantized_weight.shape[1]
        self.out_dim = quantized_weight.shape[0]
        self.compute_dtype = compute_dtype
        self._io_buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Accept `[Batch, HiddenDim]` or `[Batch, 1, HiddenDim]` input."""
        flat = x.ndim == 2
        if x.ndim not in (2, 3):
            raise ValueError("x must have rank 2 or 3")
        batch = x.shape[0]
        x_3d = x.reshape(batch, 1, self.hidden_dim) if flat else x
        residual_3d = (
            None
            if residual is None
            else residual.reshape(batch, 1, self.hidden_dim)
        )
        if batch not in (1, 4, 8, 16) or x_3d.shape != (
            batch,
            1,
            self.hidden_dim,
        ):
            raise ValueError(
                "Tensor Core decode requires seq=1 and Batch in {1,4,8,16}"
            )
        if x_3d.dtype != self.compute_dtype or not x_3d.is_contiguous():
            raise TypeError(f"x must be contiguous {self.compute_dtype}")
        if residual_3d is not None and (
            residual_3d.shape != x_3d.shape
            or residual_3d.dtype != x_3d.dtype
            or not residual_3d.is_contiguous()
        ):
            raise TypeError("residual must match x shape, dtype, and layout")
        buffers = self._io_buffers.get(batch)
        if buffers is None:
            buffers = (
                torch.empty(
                    batch, 1, self.out_dim, device=x.device, dtype=x.dtype
                ),
                torch.empty_like(x_3d),
            )
            self._io_buffers[batch] = buffers
        result = tensorcore_rms_linear_decode_out_cuda(
            x_3d,
            residual_3d,
            self.weight_fp16,
            self.bias,
            self.eps,
            buffers[0],
            buffers[1] if residual_3d is not None else None,
        )
        if isinstance(result, tuple):
            if flat:
                return result[0].reshape(batch, self.out_dim), result[1].reshape(
                    batch, self.hidden_dim
                )
            return result
        return result.reshape(batch, self.out_dim) if flat else result


class FusedRMSNormLinearTensorCoreInt8WeightDecode(nn.Module):
    """Use INT8 global weights and block-local FP16 dequantization."""

    def __init__(
        self,
        quantized_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float,
        compute_dtype: torch.dtype,
        tiles_per_cta: int = 4,
    ) -> None:
        super().__init__()
        if quantized_weight.ndim != 2 or quantized_weight.dtype != torch.int8:
            raise TypeError("quantized_weight must be int8 [OutDim, HiddenDim]")
        if weight_scale.numel() != quantized_weight.shape[0]:
            raise ValueError("weight_scale must contain one value per output")
        if compute_dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("compute_dtype must be float16 or bfloat16")
        if tiles_per_cta not in (1, 2, 4, 8, 16, 32):
            raise ValueError("tiles_per_cta must be 1, 2, 4, 8, 16, or 32")

        physical_columns = tiles_per_cta * 16
        projection_columns = physical_columns - 1
        tile_count = (
            quantized_weight.shape[0] + projection_columns - 1
        ) // projection_columns
        with torch.no_grad():
            padded_weight = torch.zeros(
                tile_count * projection_columns,
                quantized_weight.shape[1],
                device=quantized_weight.device,
                dtype=torch.int8,
            )
            padded_weight[: quantized_weight.shape[0]].copy_(quantized_weight)
            packed_flat = torch.zeros(
                tile_count,
                quantized_weight.shape[1],
                physical_columns,
                device=quantized_weight.device,
                dtype=torch.int8,
            )
            packed_flat[:, :, :projection_columns].copy_(
                padded_weight.reshape(
                    tile_count, projection_columns, quantized_weight.shape[1]
                ).transpose(1, 2)
            )
            packed_weight = packed_flat.reshape(
                tile_count,
                quantized_weight.shape[1],
                tiles_per_cta,
                16,
            ).permute(0, 2, 1, 3)

            padded_scale = torch.zeros(
                tile_count * projection_columns,
                device=weight_scale.device,
                dtype=torch.float16,
            )
            padded_scale[: quantized_weight.shape[0]].copy_(
                weight_scale.reshape(-1).to(torch.float16)
            )
            packed_scale_flat = torch.zeros(
                tile_count,
                physical_columns,
                device=weight_scale.device,
                dtype=torch.float16,
            )
            packed_scale_flat[:, :projection_columns].copy_(
                padded_scale.reshape(tile_count, projection_columns)
            )
            packed_scale = packed_scale_flat.reshape(
                tile_count, tiles_per_cta, 16
            )

        self.register_buffer("quantized_weight", packed_weight.contiguous())
        self.register_buffer("weight_scale", packed_scale.contiguous())
        self.register_buffer("bias", bias)
        self.eps = float(eps)
        self.hidden_dim = quantized_weight.shape[1]
        self.out_dim = quantized_weight.shape[0]
        self.compute_dtype = compute_dtype
        self._io_buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Accept `[Batch, HiddenDim]` or `[Batch, 1, HiddenDim]` input."""
        flat = x.ndim == 2
        if x.ndim not in (2, 3):
            raise ValueError("x must have rank 2 or 3")
        batch = x.shape[0]
        x_3d = x.reshape(batch, 1, self.hidden_dim) if flat else x
        residual_3d = (
            None
            if residual is None
            else residual.reshape(batch, 1, self.hidden_dim)
        )
        if batch not in (1, 4, 8, 16) or x_3d.shape != (
            batch,
            1,
            self.hidden_dim,
        ):
            raise ValueError(
                "Tensor Core decode requires seq=1 and Batch in {1,4,8,16}"
            )
        if x_3d.dtype != self.compute_dtype or not x_3d.is_contiguous():
            raise TypeError(f"x must be contiguous {self.compute_dtype}")
        if residual_3d is not None and (
            residual_3d.shape != x_3d.shape
            or residual_3d.dtype != x_3d.dtype
            or not residual_3d.is_contiguous()
        ):
            raise TypeError("residual must match x shape, dtype, and layout")
        buffers = self._io_buffers.get(batch)
        if buffers is None:
            buffers = (
                torch.empty(
                    batch, 1, self.out_dim, device=x.device, dtype=x.dtype
                ),
                torch.empty_like(x_3d),
            )
            self._io_buffers[batch] = buffers
        result = tensorcore_rms_linear_int8_weight_decode_out_cuda(
            x_3d,
            residual_3d,
            self.quantized_weight,
            self.weight_scale,
            self.bias,
            self.eps,
            buffers[0],
            buffers[1] if residual_3d is not None else None,
        )
        if isinstance(result, tuple):
            if flat:
                return result[0].reshape(batch, self.out_dim), result[1].reshape(
                    batch, self.hidden_dim
                )
            return result
        return result.reshape(batch, self.out_dim) if flat else result


__all__ = [
    "FusedRMSNormLinearTensorCoreDecode",
    "FusedRMSNormLinearTensorCoreInt8WeightDecode",
]
