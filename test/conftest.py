"""Shared pytest configuration for project tests."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "llama3_8b: tests requiring the local Llama3-8B checkpoint",
    )
