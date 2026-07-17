"""Benchmark both weight-only INT8 variants against all baseline paths.

The shared benchmark reports mean, p50, p95, effective TFLOPS, and speedups
for the mixed, dequantized, INT8 fused, FP16 fused, and eager FP16 paths.
"""

from __future__ import annotations

import sys
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parents[1] / "fused_rmsnorm_linear_int8"
sys.path.insert(0, str(BENCHMARK_ROOT))

from bench_fused_rmsnorm_linear_int8 import main


if __name__ == "__main__":
    main()
