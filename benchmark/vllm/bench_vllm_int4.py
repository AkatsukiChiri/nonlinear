#!/usr/bin/env python3
"""Backward-compatible entry point for the general vLLM benchmark."""

import sys

from bench_vllm_inference import main


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
