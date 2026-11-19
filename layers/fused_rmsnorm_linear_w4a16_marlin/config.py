"""Per-projection launch-layout selection for fused Marlin decode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class MarlinDecodeConfig:
    """A load-time choice shared by packing and the single-kernel launcher."""

    stripe_n: int
    physical_out_dim: int
    logical_out_dim: int

    @property
    def real_columns_per_stripe(self) -> int:
        return self.stripe_n - 1


def physical_out_dim(logical_out_dim: int, stripe_n: int) -> int:
    """Return N after adding one sentinel per physical stripe."""
    if logical_out_dim <= 0:
        raise ValueError("logical_out_dim must be positive")
    if stripe_n <= 1 or stripe_n % 64:
        raise ValueError("stripe_n must be a multiple of 64 and greater than one")
    real_columns = stripe_n - 1
    return ((logical_out_dim + real_columns - 1) // real_columns) * stripe_n


def choose_decode_config(
    logical_out_dim: int,
    measure_us: Callable[[int], float],
    candidates: Iterable[int] = (64, 128, 256),
) -> MarlinDecodeConfig:
    """Warm up valid stripe widths and retain the fastest configuration."""
    measurements: list[tuple[float, int]] = []
    for stripe_n in candidates:
        latency = float(measure_us(stripe_n))
        if latency <= 0:
            raise ValueError("measure_us must return a positive latency")
        measurements.append((latency, stripe_n))
    if not measurements:
        raise ValueError("at least one stripe candidate is required")
    _, stripe_n = min(measurements)
    return MarlinDecodeConfig(
        stripe_n=stripe_n,
        physical_out_dim=physical_out_dim(logical_out_dim, stripe_n),
        logical_out_dim=logical_out_dim,
    )
