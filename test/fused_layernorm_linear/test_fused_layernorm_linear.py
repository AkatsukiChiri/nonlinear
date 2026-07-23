"""Correctness tests for the PyTorch LayerNorm + Linear decomposition."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.fused_layernorm_linear import (
    FusedLayerNormLinear,
    FusedLayerNormLinearReference,
)


LLM_HIDDEN_DIM = 4096
LLM_OUT_DIM = 4096


def _ampere_cuda_available() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 8
    )


def _print_error_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Print error metrics that remain visible when pytest runs with ``-s``."""
    absolute_error = (actual.float() - expected.float()).abs()
    relative_error = absolute_error / expected.float().abs().clamp_min(1e-6)
    print(
        "\n"
        f"shape={tuple(actual.shape)}, dtype={actual.dtype}, "
        f"max_abs_error={absolute_error.max().item():.8e}, "
        f"mean_abs_error={absolute_error.mean().item():.8e}, "
        f"max_rel_error={relative_error.max().item():.8e}"
    )


@pytest.mark.parametrize("bias", [False, True])
def test_fused_layernorm_linear_matches_eager_fp32(bias: bool) -> None:
    """The decomposed computation should match eager LayerNorm + Linear."""
    torch.manual_seed(20260723)
    batch, seq_len = 2, 4
    x = torch.randn(batch, seq_len, LLM_HIDDEN_DIM)
    layernorm = nn.LayerNorm(LLM_HIDDEN_DIM, eps=1e-5)
    linear = nn.Linear(LLM_HIDDEN_DIM, LLM_OUT_DIM, bias=bias)

    expected = linear(layernorm(x))
    fused = FusedLayerNormLinearReference.from_modules(layernorm, linear)
    actual = fused(x)

    _print_error_metrics(actual, expected)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
def test_fused_layernorm_linear_supports_mixed_precision(
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    """FP16 and BF16 inputs should agree with the corresponding eager path."""
    torch.manual_seed(20260723)
    batch, seq_len = 2, 4
    x = torch.randn(batch, seq_len, LLM_HIDDEN_DIM, dtype=dtype)
    layernorm = nn.LayerNorm(LLM_HIDDEN_DIM, eps=1e-5, dtype=dtype)
    linear = nn.Linear(
        LLM_HIDDEN_DIM,
        LLM_OUT_DIM,
        bias=True,
        dtype=dtype,
    )

    expected = linear(layernorm(x))
    fused = FusedLayerNormLinearReference.from_modules(layernorm, linear)
    actual = fused(x)

    assert actual.dtype == dtype
    _print_error_metrics(actual, expected)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_fused_layernorm_linear_handles_zero_layernorm_scale() -> None:
    """Parameter folding must not divide by the LayerNorm scale."""
    torch.manual_seed(20260723)
    hidden_dim, out_dim = 8, 6
    x = torch.randn(2, 4, hidden_dim)
    layernorm = nn.LayerNorm(hidden_dim)
    linear = nn.Linear(hidden_dim, out_dim)
    with torch.no_grad():
        layernorm.weight[::2] = 0

    expected = linear(layernorm(x))
    fused = FusedLayerNormLinearReference.from_modules(layernorm, linear)
    actual = fused(x)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_fused_layernorm_linear_validates_hidden_dim() -> None:
    """An incompatible input shape should produce a useful error."""
    layernorm = nn.LayerNorm(8)
    linear = nn.Linear(8, 6)
    fused = FusedLayerNormLinearReference.from_modules(layernorm, linear)

    with pytest.raises(ValueError, match="HiddenDim"):
        fused(torch.randn(2, 4, 7))


def test_cuda_layer_precomputes_inference_buffers() -> None:
    """The production layer should snapshot all inference-only constants."""
    torch.manual_seed(20260723)
    hidden_dim, out_dim = 16, 12
    layernorm = nn.LayerNorm(hidden_dim, dtype=torch.float16)
    linear = nn.Linear(hidden_dim, out_dim, bias=True, dtype=torch.float16)
    fused = FusedLayerNormLinear(layernorm, linear)

    expected_weight = (
        linear.weight.float() * layernorm.weight.float().unsqueeze(0)
    ).half()
    expected_weight_sum = expected_weight.float().sum(dim=-1).half()
    expected_bias = torch.nn.functional.linear(
        layernorm.bias.float(),
        linear.weight.float(),
        linear.bias.float(),
    ).half()

    torch.testing.assert_close(fused.fused_weight, expected_weight)
    torch.testing.assert_close(fused.weight_sum, expected_weight_sum)
    torch.testing.assert_close(fused.fused_bias, expected_bias)
    assert fused.fused_weight.dtype == torch.float16
    assert fused.weight_sum.dtype == torch.float16
    assert fused.fused_bias.dtype == torch.float16
    assert dict(fused.named_parameters()) == {}


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("hidden_dim", [63, 64])
@pytest.mark.parametrize("linear_bias", [False, True])
def test_cuda_layer_matches_eager_small_shapes(
    batch: int,
    hidden_dim: int,
    linear_bias: bool,
) -> None:
    """CUDA paths should cover all batches and odd/even hidden dimensions."""
    if not _ampere_cuda_available():
        pytest.skip("an Ampere CUDA GPU is required")

    torch.manual_seed(20260723)
    out_dim = 37
    layernorm = nn.LayerNorm(
        hidden_dim,
        eps=1e-5,
        device="cuda",
        dtype=torch.float16,
    )
    linear = nn.Linear(
        hidden_dim,
        out_dim,
        bias=linear_bias,
        device="cuda",
        dtype=torch.float16,
    )
    with torch.no_grad():
        layernorm.weight.normal_(mean=1.0, std=0.1)
        layernorm.bias.normal_(mean=0.0, std=0.1)
        layernorm.weight[::11] = 0
    x = (
        torch.randn(
            batch,
            1,
            hidden_dim,
            device="cuda",
            dtype=torch.float16,
        )
        / 4
    )

    expected = linear(layernorm(x))
    actual = FusedLayerNormLinear(layernorm, linear)(x)

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_cuda_layer_matches_eager_llm_shape(batch: int) -> None:
    """Validate and report errors for the default 4096 by 4096 decode case."""
    if not _ampere_cuda_available():
        pytest.skip("an Ampere CUDA GPU is required")

    torch.manual_seed(20260723)
    layernorm = nn.LayerNorm(
        LLM_HIDDEN_DIM,
        eps=1e-5,
        device="cuda",
        dtype=torch.float16,
    )
    linear = nn.Linear(
        LLM_HIDDEN_DIM,
        LLM_OUT_DIM,
        bias=True,
        device="cuda",
        dtype=torch.float16,
    )
    with torch.no_grad():
        layernorm.weight.normal_(mean=1.0, std=0.1)
        layernorm.bias.normal_(mean=0.0, std=0.1)
    x = (
        torch.randn(
            batch,
            1,
            LLM_HIDDEN_DIM,
            device="cuda",
            dtype=torch.float16,
        )
        / 4
    )

    expected = linear(layernorm(x))
    actual = FusedLayerNormLinear(layernorm, linear)(x)

    _print_error_metrics(actual, expected)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
