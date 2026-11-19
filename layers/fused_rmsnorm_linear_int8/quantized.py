"""Adapters for pre-quantized per-channel INT8 Linear weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


WeightLayout = Literal["out_in", "in_out"]


@dataclass(frozen=True)
class QuantizedLinearWeights:
    """Canonical description of a symmetric per-output INT8 Linear."""

    weight: torch.Tensor
    weight_scale: torch.Tensor
    input_size: int
    output_size: int
    layout: WeightLayout
    bias: torch.Tensor | None = None
    logical_widths: tuple[int, ...] = ()


def adapt_quantized_linear(linear: nn.Module) -> QuantizedLinearWeights:
    """Describe an nn.Module exposing vLLM-style INT8 weight and scale."""
    weight = getattr(linear, "weight", None)
    weight_scale = getattr(linear, "weight_scale", None)
    if not isinstance(weight, torch.Tensor) or weight.dtype != torch.int8:
        raise TypeError("quantized linear must expose an INT8 weight tensor")
    if not isinstance(weight_scale, torch.Tensor):
        raise TypeError("quantized linear must expose a weight_scale tensor")
    if weight.ndim != 2:
        raise ValueError("quantized weight must be two-dimensional")

    input_size = getattr(linear, "input_size", None)
    if input_size is None:
        input_size = getattr(linear, "in_features", None)
    if not isinstance(input_size, int):
        raise TypeError("quantized linear must expose input_size or in_features")

    flattened_scale = weight_scale.reshape(-1)
    output_size = flattened_scale.numel()
    if weight.shape == (output_size, input_size):
        layout: WeightLayout = "out_in"
    elif weight.shape == (input_size, output_size):
        layout = "in_out"
    else:
        raise ValueError(
            "weight shape is incompatible with input_size and weight_scale: "
            f"weight={tuple(weight.shape)}, input_size={input_size}, "
            f"scales={output_size}"
        )

    logical_widths_value = getattr(linear, "logical_widths", ())
    logical_widths = tuple(int(width) for width in logical_widths_value)
    if logical_widths and sum(logical_widths) != output_size:
        raise ValueError("logical_widths must sum to the local output size")
    bias = getattr(linear, "bias", None)
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError("bias must be a tensor or None")
    return QuantizedLinearWeights(
        weight=weight,
        weight_scale=flattened_scale,
        input_size=input_size,
        output_size=output_size,
        layout=layout,
        bias=bias,
        logical_widths=logical_widths,
    )


def fold_rmsnorm_into_quantized_weight(
    rmsnorm_weight: torch.Tensor,
    quantized: QuantizedLinearWeights,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize chunks, absorb RMSNorm gamma, and requantize per output.

    The returned weight always has shape ``[OutDim, HiddenDim]`` and the
    returned FP32 scale has shape ``[OutDim]``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if rmsnorm_weight.ndim != 1:
        raise ValueError("rmsnorm_weight must have shape [HiddenDim]")
    if rmsnorm_weight.numel() != quantized.input_size:
        raise ValueError("RMSNorm HiddenDim must equal Linear input_size")
    if rmsnorm_weight.device != quantized.weight.device:
        raise ValueError("RMSNorm and quantized weights must share a device")
    if quantized.weight_scale.device != quantized.weight.device:
        raise ValueError("quantized weight and scale must share a device")

    output_weight = torch.empty(
        quantized.output_size,
        quantized.input_size,
        dtype=torch.int8,
        device=quantized.weight.device,
    )
    output_scale = torch.empty(
        quantized.output_size,
        dtype=torch.float32,
        device=quantized.weight.device,
    )
    gamma = rmsnorm_weight.float().reshape(1, quantized.input_size)
    source_scale = quantized.weight_scale.float()
    with torch.no_grad():
        for start in range(0, quantized.output_size, chunk_size):
            end = min(start + chunk_size, quantized.output_size)
            if quantized.layout == "out_in":
                quantized_chunk = quantized.weight[start:end]
            else:
                quantized_chunk = quantized.weight[:, start:end].t()
            dequantized = (
                quantized_chunk.float()
                * source_scale[start:end, None]
                * gamma
            )
            maximum = dequantized.abs().amax(dim=1)
            scale = maximum / 127.0
            safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            requantized = torch.round(dequantized / safe_scale[:, None])
            output_weight[start:end] = requantized.clamp(-127, 127).to(torch.int8)
            output_scale[start:end] = safe_scale
    return output_weight.contiguous(), output_scale.contiguous()


__all__ = [
    "QuantizedLinearWeights",
    "adapt_quantized_linear",
    "fold_rmsnorm_into_quantized_weight",
]
