"""Tests for KFParameters validation and initialization."""

from __future__ import annotations

import numpy as np
import pytest

from kalman_filter import KFParameters


def test_default_parameters_initialize_consistent_shapes(
    default_params: KFParameters,
) -> None:
    """Default parameters produce mutually compatible matrix shapes."""
    dim_x = default_params.x.dim
    dim_z = default_params.z.dim

    assert default_params.R.shape == (dim_z, dim_z)
    assert default_params.P.shape == (dim_x, dim_x)
    assert default_params.Q.shape == (dim_x, dim_x)
    assert default_params.H.shape == (dim_z, dim_x)
    assert default_params.F.shape == (dim_x, dim_x)
    assert default_params.K.shape == (dim_x, dim_z)
    assert default_params.S.shape == (dim_z, dim_z)


def test_r_values_must_be_positive(default_state_format: dict[str, list[str]]) -> None:
    """Non-positive R_values must fail validation."""
    with pytest.raises(ValueError, match="R_values must be greater than 0"):
        KFParameters(
            state_vector_format=default_state_format,
            R_values=[0.0, 1.0, 1.0, 1.0],
        )


def test_r_values_length_must_match_measurement_dimension(
    default_state_format: dict[str, list[str]],
) -> None:
    """R_values length must match the measurement vector dimension."""
    with pytest.raises(ValueError, match="R_values length"):
        KFParameters(
            state_vector_format=default_state_format,
            R_values=[1.0, 1.0],
        )


def test_static_variables_must_be_unique(default_state_format: dict[str, list[str]]) -> None:
    """Duplicate static variable names must fail validation."""
    invalid_format = dict(default_state_format)
    invalid_format["static"] = ["u", "u", "s", "r"]

    with pytest.raises(ValueError, match="Static variables must be unique"):
        KFParameters(state_vector_format=invalid_format)


def test_measurement_matrix_value_from_list(default_state_format: dict[str, list[str]]) -> None:
    """Custom H can be supplied as a flat list of floats."""
    dim_x = 7
    dim_z = 4
    custom_h = [1.0 if row == col else 0.0 for row in range(dim_z) for col in range(dim_x)]

    params = KFParameters(
        state_vector_format=default_state_format,
        measurement_matrix_value=custom_h,
    )

    assert params.H.shape == (dim_z, dim_x)
    assert np.allclose(params.H.value, np.eye(dim_z, dim_x))


def test_clone_creates_independent_parameters(default_params: KFParameters) -> None:
    """clone() returns a fresh parameter set without sharing mutable state."""
    cloned = default_params.clone()
    cloned.x.value[0, 0] = 999.0

    assert default_params.x.value[0, 0] != 999.0
