"""Model-load packing for the single-kernel Marlin sentinel decode path."""

from .config import MarlinDecodeConfig, choose_decode_config
from .packing import (
    MarlinSentinelPack,
    MarlinSentinelWmmaPack,
    pack_awq_marlin_sentinels,
    pack_awq_sentinel_fragment,
    pack_awq_sentinel_wmma,
)
from .cuda import w4a16_marlin_sentinel_wmma
from .layer import FusedRMSNormLinearW4A16Marlin

__all__ = [
    "MarlinDecodeConfig",
    "MarlinSentinelPack",
    "MarlinSentinelWmmaPack",
    "FusedRMSNormLinearW4A16Marlin",
    "choose_decode_config",
    "pack_awq_marlin_sentinels",
    "pack_awq_sentinel_fragment",
    "pack_awq_sentinel_wmma",
    "w4a16_marlin_sentinel_wmma",
]
