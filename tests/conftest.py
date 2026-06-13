"""Shared fixtures for Kalman filter tests."""

from __future__ import annotations

import pytest

from kalman_filter import KFParameters


@pytest.fixture
def default_state_format() -> dict[str, list[str]]:
    """Default state vector format used across tests."""
    return {
        "static": ["u", "v", "s", "r"],
        "dynamic": ["u", "v", "s"],
        "acceleration": [],
    }


@pytest.fixture
def default_params(default_state_format: dict[str, list[str]]) -> KFParameters:
    """Validated KFParameters with the default state format."""
    return KFParameters(state_vector_format=default_state_format)
