from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_rmsnorm_linear_int8.quantized import (
    QuantizedLinearWeights,
    adapt_quantized_linear,
    fold_rmsnorm_into_quantized_weight,
)
from layers.fused_rmsnorm_linear_int8_prefill import (
    FusedRMSNormLinearInt8Prefill,
)
from layers.fused_rmsnorm_linear_int8_prefill.reference import (
    fused_rmsnorm_linear_int8_prefill_ref,
)


class FakeVllmQuantizedLinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_size: int,
        logical_widths: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
        self.input_size = input_size
        self.logical_widths = list(logical_widths)
        self.register_parameter("bias", None)


MODEL_PATH = Path(
    "/home/mashaobo/models/Meta-Llama-3.1-8B-Instruct-W8A8"
)


def _load_checkpoint_tensors(names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    shard = MODEL_PATH / "model-00001-of-00002.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as checkpoint:
        return {name: checkpoint.get_tensor(name) for name in names}


def _original_quantized_linear_reference(
    x: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    quantized_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    eps: float,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Evaluate the checkpoint's RMSNorm + dequantized Linear in chunks."""
    rows = x.reshape(-1, x.shape[-1]).float()
    variance = rows.square().mean(dim=-1, keepdim=True)
    normalized = rows * torch.rsqrt(variance + eps)
    normalized = normalized * rmsnorm_weight.float()
    output_chunks = []
    for start in range(0, quantized_weight.shape[0], chunk_size):
        end = min(start + chunk_size, quantized_weight.shape[0])
        dequantized = (
            quantized_weight[start:end].float()
            * weight_scale[start:end].float().reshape(-1, 1)
        )
        output_chunks.append(normalized @ dequantized.t())
    output = torch.cat(output_chunks, dim=-1)
    return output.reshape(*x.shape[:-1], quantized_weight.shape[0])


@pytest.mark.parametrize("layout", ["out_in", "in_out"])
def test_fold_quantized_weight_supports_both_layouts(layout: str) -> None:
    torch.manual_seed(20260806)
    out_dim, hidden_dim = 13, 64
    weight = torch.randint(-127, 128, (out_dim, hidden_dim), dtype=torch.int8)
    scale = torch.rand(out_dim, dtype=torch.float32) / 100
    gamma = torch.rand(hidden_dim, dtype=torch.float16) + 0.5
    source = weight if layout == "out_in" else weight.t()
    quantized = QuantizedLinearWeights(
        weight=source,
        weight_scale=scale,
        input_size=hidden_dim,
        output_size=out_dim,
        layout=layout,  # type: ignore[arg-type]
    )

    fused_weight, fused_scale = fold_rmsnorm_into_quantized_weight(
        gamma, quantized, chunk_size=5
    )
    expected = weight.float() * scale[:, None] * gamma.float()[None, :]
    actual = fused_weight.float() * fused_scale[:, None]

    assert fused_weight.shape == (out_dim, hidden_dim)
    assert fused_scale.shape == (out_dim,)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "logical_widths",
    [(64, 16, 16), (80, 80)],
    ids=["qkv", "gate_up"],
)
def test_vllm_adapter_preserves_merged_logical_widths(
    logical_widths: tuple[int, ...],
) -> None:
    hidden_dim = 64
    out_dim = sum(logical_widths)
    checkpoint_weight = torch.arange(
        out_dim, dtype=torch.int8
    )[:, None].expand(out_dim, hidden_dim).contiguous()
    vllm_weight = checkpoint_weight.t()
    scale = torch.ones(out_dim, 1, dtype=torch.float32)
    linear = FakeVllmQuantizedLinear(
        vllm_weight, scale, hidden_dim, logical_widths
    )

    adapted = adapt_quantized_linear(linear)

    assert adapted.layout == "in_out"
    assert adapted.logical_widths == logical_widths
    assert adapted.output_size == out_dim


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_prefill_factory_from_vllm_layout_matches_quantized_reference(
    dtype: torch.dtype,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")
    torch.manual_seed(20260806)
    hidden_dim = 128
    logical_widths = (128, 32, 32)
    out_dim = sum(logical_widths)
    checkpoint_weight = torch.randint(
        -127, 128, (out_dim, hidden_dim), device="cuda", dtype=torch.int8
    )
    scale = torch.rand(out_dim, 1, device="cuda", dtype=torch.float32) / 100
    linear = FakeVllmQuantizedLinear(
        checkpoint_weight.t(), scale, hidden_dim, logical_widths
    ).cuda()
    rmsnorm = nn.RMSNorm(
        hidden_dim, eps=1e-5, device="cuda", dtype=dtype
    )
    with torch.no_grad():
        rmsnorm.weight.uniform_(0.5, 1.5)
    layer = FusedRMSNormLinearInt8Prefill.from_quantized_linear(
        rmsnorm, linear, chunk_size=37
    )
    x = torch.randn(2, 7, hidden_dim, device="cuda", dtype=dtype) / 4

    expected = fused_rmsnorm_linear_int8_prefill_ref(
        x,
        layer.quantized_weight,
        layer.weight_scale,
        layer.eps,
        layer.bias,
    )
    actual = layer(x)

    assert layer.logical_widths == logical_widths
    assert actual.shape == (2, 7, out_dim)
    tolerance = 2e-2 if dtype == torch.float16 else 4e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0)


@pytest.mark.parametrize(
    ("projection_names", "norm_name", "logical_widths"),
    [
        (
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
            "input_layernorm",
            (4096, 1024, 1024),
        ),
        (
            ("mlp.gate_proj", "mlp.up_proj"),
            "post_attention_layernorm",
            (14336, 14336),
        ),
    ],
    ids=["merged_qkv", "merged_gate_up"],
)
def test_real_w8a8_checkpoint_single_layer_output(
    projection_names: tuple[str, ...],
    norm_name: str,
    logical_widths: tuple[int, ...],
) -> None:
    if not MODEL_PATH.exists():
        pytest.skip("local W8A8 checkpoint is unavailable")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("Ampere is required")

    prefix = "model.layers.0"
    weight_names = tuple(
        f"{prefix}.{projection}.weight" for projection in projection_names
    )
    scale_names = tuple(
        f"{prefix}.{projection}.weight_scale"
        for projection in projection_names
    )
    rmsnorm_name = f"{prefix}.{norm_name}.weight"
    tensors = _load_checkpoint_tensors(
        weight_names + scale_names + (rmsnorm_name,)
    )
    checkpoint_weight = torch.cat(
        [tensors[name] for name in weight_names], dim=0
    ).cuda()
    checkpoint_scale = torch.cat(
        [tensors[name].reshape(-1) for name in scale_names], dim=0
    ).cuda()
    rmsnorm_weight = tensors[rmsnorm_name].cuda()

    # CutlassScaledMMLinearKernel transposes loaded [N, K] weights to [K, N].
    compressed_linear = FakeVllmQuantizedLinear(
        checkpoint_weight.t(),
        checkpoint_scale,
        checkpoint_weight.shape[1],
        logical_widths,
    ).cuda()
    rmsnorm = nn.RMSNorm(
        checkpoint_weight.shape[1],
        eps=1e-5,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        rmsnorm.weight.copy_(rmsnorm_weight)
    fused_layer = FusedRMSNormLinearInt8Prefill.from_quantized_linear(
        rmsnorm, compressed_linear, chunk_size=512
    )

    torch.manual_seed(20260806)
    x = torch.randn(
        1,
        4,
        checkpoint_weight.shape[1],
        device="cuda",
        dtype=torch.bfloat16,
    ) / 4
    expected = _original_quantized_linear_reference(
        x,
        rmsnorm_weight,
        checkpoint_weight,
        checkpoint_scale,
        rmsnorm.eps,
    )
    actual = fused_layer(x).float()

    relative_rmse = (actual - expected).square().mean().sqrt()
    relative_rmse /= expected.square().mean().sqrt()
    cosine_similarity = torch.nn.functional.cosine_similarity(
        actual.reshape(-1), expected.reshape(-1), dim=0
    )
    assert fused_layer.logical_widths == logical_widths
    assert actual.shape[-1] == sum(logical_widths)
    assert relative_rmse.item() < 0.05
    assert cosine_similarity.item() > 0.998
